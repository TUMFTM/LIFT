import demandlib
import importlib.resources as resources
import pandas as pd
import pvlib
from time import time
from typing import TYPE_CHECKING

import numpy as np


from .energy_system import (
    FixedDemand,
    Fleet,
    GridConnection,
    PVSource,
    StationaryStorage,
)

from .interfaces import (
    safe_cache_data,
    Coordinates,
    Inputs,
    PhaseInputLocation,
    PhaseInputEconomics,
    PhaseInputCharger,
    PhaseInputSubfleet,
    Logs,
    SimInputSettings,
    SimInputLocation,
    SimInputSubfleet,
    SimInputCharger,
    SimResults,
    PhaseResults,
    TotalResults,
)

if TYPE_CHECKING:
    pass


@safe_cache_data
def get_log_pv(coordinates: Coordinates,
               settings: SimInputSettings,
               ) -> np.typing.NDArray[np.float64]:
    # ToDo: add default value if internet connection is not available -> debug purposes only
    dti = settings.dti

    data, *_ = pvlib.iotools.get_pvgis_hourly(
        latitude=coordinates.latitude,
        longitude=coordinates.longitude,
        start=int(dti.year.min()),
        end=int(dti.year.max()),
        raddatabase='PVGIS-SARAH3',
        outputformat='json',
        pvcalculation=True,
        peakpower=1,  # convert kWp to Wp
        pvtechchoice='crystSi',
        mountingplace='free',
        loss=0,
        trackingtype=0,  # fixed mount
        optimalangles=True,
        url='https://re.jrc.ec.europa.eu/api/v5_3/',
        map_variables=True,
        timeout=30,  # default value
    )
    data = data['P']
    data.index = data.index.round('h')
    data = data.tz_convert('Europe/Berlin').reindex(dti).ffill().bfill()
    return data.values / 1000


@safe_cache_data
def get_log_dem(slp: str,
                consumption_yrl_wh: float,
                settings: SimInputSettings) -> np.typing.NDArray[np.float64]:
    return pd.concat([demandlib.bdew.ElecSlp(year=year)
                     .get_scaled_profiles({slp: consumption_yrl_wh})  # returns energies
                     .resample(settings.freq_sim).sum()  # sum() as df contains energies -> for hours energy is equal to power
                     .iloc[:, 0]
                      for year in settings.dti.year.unique()]).values / settings.freq_hours  # get first (and only) column as numpy array and convert from energy to power


@safe_cache_data
def get_log_subfleet(vehicle_type: str,
                     settings: SimInputSettings) -> pd.DataFrame:
    with resources.files('lift.data').joinpath(f'log_{vehicle_type}.csv').open('r') as logfile:
        df = pd.read_csv(logfile,
                         header=[0,1])
        df = df.set_index(pd.to_datetime(df.iloc[:, 0], utc=True)).drop(df.columns[0], axis=1).tz_convert('Europe/Berlin')
    return df.loc[settings.dti, :]


@safe_cache_data
def simulate(settings: SimInputSettings,
             logs: Logs,
             capacities: SimInputLocation,
             subfleets: dict[str, SimInputSubfleet],
             chargers: dict[str, SimInputCharger],
             ) -> SimResults:

    dti = settings.dti
    freq_hours = settings.freq_hours

    dem = FixedDemand(log=logs.dem,
                      dti=dti,
                      freq_hours=freq_hours,
                      )
    fleet = Fleet(pwr_lim_w=np.inf,
                  log=logs.fleet,
                  subfleets=subfleets,
                  chargers=chargers,
                  dti=dti,
                  freq_hours=freq_hours,
                  )
    pv = PVSource(pwr_wp=capacities.pv_wp,
                  log_spec=logs.pv_spec,
                  dti=dti,
                  freq_hours=freq_hours,
                  )
    ess = StationaryStorage(capacity_wh=capacities.ess_wh,
                            dti=dti,
                            freq_hours=freq_hours,
                            )
    grid = GridConnection(pwr_max_w=capacities.grid_w,
                          dti=dti,
                          freq_hours=freq_hours,
                          )
    blocks_supply = (pv, ess, grid)

    blocks = [dem, fleet, *fleet.fleet_units.values(), *blocks_supply]

    # Simulate the vehicle fleet over the given datetime index.
    for idx in range(len(settings.dti)):
        # pass time of current timestep to all blocks
        for block in blocks:
            block.idx = idx

        # calculate maximum power supply
        pwr_supply_max_w = sum(block.generation_max_w for block in blocks_supply)
        # get the total demand from the fixed demand block
        pwr_demand_w = dem.demand_w

        # define Fleet charging power limit for dynamic load management
        fleet.pwr_lim_w = pwr_supply_max_w - pwr_demand_w

        # add fleet demand to the total demand
        pwr_demand_w += fleet.demand_w

        # satisfy demand with supply blocks (order represents priority)
        for block in blocks_supply:
            pwr_demand_w = block.satisfy_demand(demand_w=pwr_demand_w)

    return SimResults(energy_pv_pot_wh=pv.energy_pot_wh,
                      energy_pv_curt_wh=grid.energy_curt_wh,
                      energy_grid_buy_wh=grid.energy_buy_wh,
                      energy_grid_sell_wh=grid.energy_sell_wh,
                      pwr_grid_peak_w=grid.pwr_peak_w,
                      energy_dem_site_wh=dem.energy_wh,
                      energy_fleet_site_wh=fleet.energy_site_wh,
                      energy_fleet_route_wh=fleet.energy_route_wh,
                      )


def calc_phase_results(logs: Logs,
                       location: PhaseInputLocation,
                       economics: PhaseInputEconomics,
                       subfleets: dict[str, PhaseInputSubfleet],
                       chargers: dict[str, PhaseInputCharger],
                       ) -> PhaseResults:

    result_sim = simulate(
        settings=economics.get_sim_input(),
        logs=logs,
        capacities=location.get_sim_input(),
        subfleets={sf.name: sf.get_sim_input() for sf in subfleets.values()},
        chargers={chg.name: chg.get_sim_input() for chg in chargers.values()},
    )

    # calculate self-sufficiency and self consumption based on simulation results
    if location.pv.capacity == 0:
        self_sufficiency = 0.0
        self_consumption = 0.0
    else:
        if result_sim.energy_dem_site_wh + result_sim.energy_fleet_site_wh == 0:
            self_sufficiency = 0.0
        else:
            # potential PV energy minus curtailed and sold energy divided by total energy demand (fleet@site + site)
            self_sufficiency = ((result_sim.energy_pv_pot_wh - result_sim.energy_pv_curt_wh -
                                result_sim.energy_grid_sell_wh) /
                                (result_sim.energy_fleet_site_wh + result_sim.energy_dem_site_wh))
        if result_sim.energy_pv_pot_wh == 0:
            self_consumption = 0.0
        else:
            # 1 - (pv energy not used on-site (curtailed or sold) divided by total potential PV energy)
            self_consumption = 1 - ((result_sim.energy_grid_sell_wh + result_sim.energy_pv_curt_wh) / result_sim.energy_pv_pot_wh)

    # calculate share charged energy which was charged at the own site (vs. on route)
    if (result_sim.energy_fleet_site_wh + result_sim.energy_fleet_route_wh) == 0:
        site_charging = 0.0
    else:
        site_charging = (result_sim.energy_fleet_site_wh /
                         (result_sim.energy_fleet_site_wh + result_sim.energy_fleet_route_wh))

    cashflow_capex = np.zeros(economics.period_eco)
    cashflow_opex = np.zeros(economics.period_eco)
    cashflow_capem = np.zeros(economics.period_eco)
    cashflow_opem = np.zeros(economics.period_eco)

    def calc_replacements(ls: float) -> np.typing.NDArray:
        years = np.arange(economics.period_eco)

        # Replacement years: start + 0, lifespan, 2*lifespan, ...
        replacement_years = np.arange(0, economics.period_eco, ls)
        # ToDo: exclude first year if required
        # ToDo: add salvage value if required

        repl = np.isin(years, replacement_years).astype(int)

        return repl

    # fix cost
    cashflow_capex[0] += economics.fix_cost_construction

    # grid
    replacements = calc_replacements(ls=location.grid.ls)
    cashflow_capex += location.grid.capacity * location.grid.capex_spec * replacements
    cashflow_opex += (result_sim.energy_grid_buy_wh * economics.opex_spec_grid_buy -
                      result_sim.energy_grid_sell_wh * economics.opex_spec_grid_sell)
    cashflow_opex += result_sim.pwr_grid_peak_w * economics.opex_spec_grid_peak

    cashflow_capem += location.grid.capacity * location.grid.capem_spec * replacements
    cashflow_opem += result_sim.energy_grid_buy_wh * economics.opem_spec_grid

    # pv
    replacements = calc_replacements(ls=location.pv.ls)
    cashflow_capex += location.pv.capacity * location.pv.capex_spec * replacements
    cashflow_capem += location.pv.capacity * location.pv.capem_spec * replacements

    # ess
    replacements = calc_replacements(ls=location.ess.ls)
    cashflow_capex += location.ess.capacity * location.ess.capex_spec * replacements
    cashflow_capem += location.ess.capacity * location.ess.capem_spec * replacements

    # vehicles
    cashflow_opex += result_sim.energy_fleet_route_wh * economics.opex_spec_route_charging
    cashflow_opem += result_sim.energy_fleet_route_wh * economics.opem_spec_grid

    for sf_in in subfleets.values():
        sf_in = subfleets[sf_in.name]
        replacements = calc_replacements(ls=sf_in.ls)
        num_bev = sf_in.num_bev
        num_icev = sf_in.num_total - sf_in.num_bev

        log = logs.fleet[sf_in.name]
        vehicles = list(log.columns.get_level_values(0).unique())
        bevs = vehicles[0:num_bev]
        icevs = vehicles[num_bev:num_bev + num_icev]

        cashflow_capex += (num_bev * sf_in.capex_bev_eur * replacements +
                           num_icev * sf_in.capex_icev_eur * replacements)
        cashflow_capem += (num_bev * sf_in.capem_bev * replacements +
                           num_icev * sf_in.capem_icev * replacements)

        # bev
        dist_bev = log.loc[:, pd.IndexSlice[bevs, 'dist']].to_numpy().sum() if bevs else 0
        cashflow_opex += (num_bev * economics.insurance_frac * sf_in.capex_bev_eur +  # insurance
                          dist_bev * (
                                  sf_in.mntex_eur_km_bev +  # maintenance
                                  sf_in.toll_eur_per_km_bev * sf_in.toll_frac  # toll
                          ))

        # icev
        dist_icev = log.loc[:, pd.IndexSlice[icevs, 'dist']].to_numpy().sum() if icevs else 0
        cashflow_opex += (num_icev * economics.insurance_frac * sf_in.capex_icev_eur +  # insurance
                          dist_icev * (
                                  sf_in.mntex_eur_km_icev +  # maintenance
                                  sf_in.toll_eur_per_km_icev * sf_in.toll_frac +  # toll
                                  economics.opex_fuel * sf_in.consumption_icev / 100  # fuel
                          ))
        cashflow_opem += dist_icev * sf_in.consumption_icev / 100 * economics.co2_per_liter_diesel_kg

    # chargers
    for chg_in in chargers.values():
        replacements = calc_replacements(ls=chg_in.ls)
        cashflow_capex += chg_in.cost_per_charger_eur * chg_in.num * replacements
        cashflow_capem += chg_in.capem * chg_in.num * replacements

    cashflow = cashflow_capex + cashflow_opex
    co2_flow = cashflow_capem + cashflow_opem

    return PhaseResults(simulation=result_sim,
                        self_sufficiency=self_sufficiency,
                        self_consumption=self_consumption,
                        site_charging=site_charging,
                        cashflow=cashflow,
                        co2_flow=co2_flow,
                        )


def run_backend(inputs: Inputs) -> TotalResults:
    # start time tracking
    start_time = time()

    # ToDo: check timezone consistency of all inputs
    # get log data for the simulation
    logs = Logs(pv_spec=get_log_pv(coordinates=inputs.location.coordinates,
                                   settings=inputs.economics.get_sim_input()),
                dem=get_log_dem(slp=inputs.location.slp.lower(),
                                consumption_yrl_wh=inputs.location.consumption_yrl_wh,
                                settings=inputs.economics.get_sim_input()),
                # ToDo: get input parameters from inputs
                fleet={vehicle_type: get_log_subfleet(vehicle_type=vehicle_type,
                                                      settings=inputs.economics.get_sim_input())
                       for vehicle_type, subfleet in inputs.subfleets.items()},
                )

    results = {phase: calc_phase_results(logs=logs,
                                         location=inputs.location.get_phase_input(phase=phase),
                                         economics=inputs.economics.get_phase_input(phase=phase),
                                         subfleets={sf.name: sf.get_phase_input(phase=phase)
                                                    for sf in inputs.subfleets.values()},
                                         chargers={chg.name: chg.get_phase_input(phase=phase)
                                                   for chg in inputs.chargers.values()},
                                         )
               for phase in ["baseline", "expansion"]}

    # stop time tracking
    print(f'{pd.Timestamp.now().isoformat()} - Backend calculation completed in {time() - start_time:.2f} seconds.')

    return TotalResults(baseline=results['baseline'],
                        expansion=results['expansion'],
                        )


if __name__ == "__main__":
    settings_default = Inputs()
    result = run_backend(inputs=settings_default)
    print(result.baseline.simulation)
    print(result.expansion.simulation)
