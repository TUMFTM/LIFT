import demandlib
import importlib.resources as resources
import pandas as pd
import pvlib
import streamlit as st
from time import time
from typing import TYPE_CHECKING, Literal, Tuple, Dict, Optional

import numpy as np

from definitions import (DTI,
                         TIME_PRJ_YRS,
                         DEF_PV,
                         DEF_ESS,
                         DEF_GRID,
                         DEF_CHARGERS,
                         DEF_SUBFLEETS,
                         CO2_PER_LITER_DIESEL_KG,
                         )

from energy_system import (FixedDemand,
                           Fleet,
                           FleetUnit,
                           GridConnection,
                           PVSource,
                           StationaryStorage)

from interfaces import (Coordinates,
                        Inputs,
                        PhaseInputCharger,
                        PhaseInputSubfleet,
                        PhaseInputEconomic,
                        Logs,
                        Capacities,
                        SimInputSubfleet,
                        SimInputCharger,
                        SimResults,
                        PhaseResults,
                        TotalResults,
                        )

if TYPE_CHECKING:
    pass


@st.cache_data
def get_log_pv(coordinates: Coordinates) -> np.typing.NDArray[np.float64]:
    data, *_ = pvlib.iotools.get_pvgis_hourly(
        latitude=coordinates.latitude,
        longitude=coordinates.longitude,
        start=2023,
        end=2023,
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
    data = data.tz_convert('Europe/Berlin').reindex(DTI).ffill().bfill()
    return data.values / 1000


@st.cache_data
def get_log_dem(slp: str,
                consumption_yrl_wh: float) -> np.typing.NDArray[np.float64]:
    # Example demand data, replace with actual demand data retrieval logic
    e_slp = demandlib.bdew.ElecSlp(year=2023)
    return (e_slp.get_scaled_profiles({slp: consumption_yrl_wh})  # returns energies
            .resample('h').sum()  # sum() as df contains energies -> for hours energy is equal to power
            .iloc[:, 0].values)  # get first (and only) column as numpy array


@st.cache_data
def get_log_subfleet(vehicle_type: str) -> pd.DataFrame:
    with resources.files('lift.data').joinpath(f'log_{vehicle_type}.csv').open('r') as logfile:
        df = pd.read_csv(logfile,
                         header=[0,1])
        df = df.set_index(pd.to_datetime(df.iloc[:, 0], utc=True)).drop(df.columns[0], axis=1).tz_convert('Europe/Berlin')
    return df.loc[DTI, :]


@st.cache_data
def simulate(logs: Logs,
             capacities: Capacities,
             subfleets: dict[str, SimInputSubfleet],
             chargers: dict[str, SimInputCharger],
             ) -> SimResults:

    dem = FixedDemand(log=logs.dem)
    fleet = Fleet(pwr_lim_w=np.inf,
                  log=logs.fleet,
                  subfleets=subfleets,
                  chargers=chargers,)
    pv = PVSource(pwr_wp=capacities.pv_wp,
                  log_spec=logs.pv_spec)
    ess = StationaryStorage(capacity_wh=capacities.ess_wh,)
    grid = GridConnection(pwr_max_w=capacities.grid_w)
    blocks_supply = (pv, ess, grid)

    blocks = [dem, fleet, *fleet.fleet_units.values(), *blocks_supply]

    # Simulate the vehicle fleet over the given datetime index.
    for idx in range(len(DTI)):
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
                      energy_dem_wh=dem.energy_wh,
                      energy_fleet_wh=fleet.energy_wh,
                      )


def calc_phase_results(logs: Logs,
                       capacities: Capacities,
                       economics: PhaseInputEconomic,
                       subfleets: dict[str, PhaseInputSubfleet],
                       chargers: dict[str, PhaseInputCharger],
                       ) -> PhaseResults:

    result_sim = simulate(
        logs=logs,
        capacities=capacities,
        subfleets={sf.name: sf.get_sim_input() for sf in subfleets.values()},
        chargers={chg.name: chg.get_sim_input() for chg in chargers.values()},
    )

    # calculate self-sufficiency and self consumption based on simulation results
    if capacities.pv_wp == 0:
        self_sufficiency = 0.0
        self_consumption = 0.0
    else:
        # potential PV energy minus curtailed and sold energy divided by total energy demand (fleet + site)
        self_sufficiency = (result_sim.energy_pv_pot_wh - result_sim.energy_pv_curt_wh - result_sim.energy_grid_sell_wh) / (result_sim.energy_fleet_wh + result_sim.energy_dem_wh)
        # 1 - (pv energy not used on-site (curtailed or sold) divided by total potential PV energy)
        self_consumption = 1 - ((result_sim.energy_grid_sell_wh + result_sim.energy_pv_curt_wh) / result_sim.energy_pv_pot_wh)

    cashflow_capex = np.zeros(TIME_PRJ_YRS)
    cashflow_opex = np.zeros(TIME_PRJ_YRS)
    cashflow_capem = np.zeros(TIME_PRJ_YRS)
    cashflow_opem = np.zeros(TIME_PRJ_YRS)

    def calc_replacements(ls: float) -> np.typing.NDArray:
        years = np.arange(TIME_PRJ_YRS)

        # Replacement years: start + 0, lifespan, 2*lifespan, ...
        replacement_years = np.arange(0, TIME_PRJ_YRS, ls)
        # ToDo: exclude first year if required
        # ToDo: add salvage value if required

        repl = np.isin(years, replacement_years).astype(int)

        return repl

    # fix cost
    cashflow_capex[0] += economics.fix_cost_construction

    # grid
    replacements = calc_replacements(ls=DEF_GRID['ls'])
    cashflow_capex += capacities.grid_w * DEF_GRID['capex_spec'] * replacements
    cashflow_opex += (result_sim.energy_grid_buy_wh * economics.opex_spec_grid_buy -
                      result_sim.energy_grid_sell_wh * economics.opex_spec_grid_sell)
    cashflow_opex += result_sim.pwr_grid_peak_w * economics.opex_spec_grid_peak

    cashflow_capem += capacities.grid_w * DEF_GRID['capem_spec'] * replacements
    cashflow_opem += result_sim.energy_grid_buy_wh * DEF_GRID['opem_spec']

    # pv
    replacements = calc_replacements(ls=DEF_PV['ls'])
    cashflow_capex += capacities.pv_wp * DEF_PV['capex_spec'] * replacements
    cashflow_capem += capacities.pv_wp * DEF_PV['capem_spec'] * replacements

    # ess
    replacements = calc_replacements(ls=DEF_ESS['ls'])
    cashflow_capex += capacities.ess_wh * DEF_ESS['capex_spec'] * replacements
    cashflow_capem += capacities.ess_wh * DEF_ESS['capem_spec'] * replacements

    # vehicles
    for sf_def in DEF_SUBFLEETS.values():
        sf_in = subfleets[sf_def.name]
        replacements = calc_replacements(ls=sf_def.ls)
        num_bev = sf_in.num_bev
        num_icev = sf_in.num_total - sf_in.num_bev

        log = logs.fleet[sf_in.name]
        vehicles = list(log.columns.get_level_values(0).unique())
        bevs = vehicles[0:num_bev]
        icevs = vehicles[num_bev:num_bev + num_icev]

        cashflow_capex += (num_bev * sf_in.capex_bev_eur * replacements +
                           num_icev * sf_in.capex_icev_eur * replacements)
        cashflow_capem += (num_bev * sf_def.capem_bev * replacements +
                           num_icev * sf_def.capem_icev * replacements)

        # bev
        dist_bev = log.loc[:, pd.IndexSlice[bevs, 'dist']].to_numpy().sum() if bevs else 0
        cashflow_opex += (num_bev * economics.insurance_frac * sf_in.capex_bev_eur +  # insurance
                          dist_bev * (
                                  sf_def.mntex_eur_km_bev +  # maintenance
                                  sf_def.toll_eur_per_km_bev * sf_in.toll_frac  # toll
                          ))

        # icev
        dist_icev = log.loc[:, pd.IndexSlice[icevs, 'dist']].to_numpy().sum() if icevs else 0
        cashflow_opex += (num_icev * economics.insurance_frac * sf_in.capex_icev_eur +  # insurance
                          dist_icev * (
                                  sf_def.mntex_eur_km_icev +  # maintenance
                                  sf_def.toll_eur_per_km_icev * sf_in.toll_frac +  # toll
                                  economics.opex_fuel * sf_def.consumption_icev / 100  # fuel
                          ))
        cashflow_opem += dist_icev * sf_def.consumption_icev / 100 * CO2_PER_LITER_DIESEL_KG

    # chargers
    for chg_def in DEF_CHARGERS.values():
        chg_in = chargers[chg_def.name]
        replacements = calc_replacements(ls=chg_def.ls)
        cashflow_capex += chg_in.cost_per_charger_eur * chg_in.num * replacements
        cashflow_capem += chg_def.capem * chg_in.num * replacements



    cashflow = cashflow_capex + cashflow_opex
    co2_flow = cashflow_capem + cashflow_opem

    return PhaseResults(simulation=result_sim,
                        self_sufficiency=self_sufficiency,
                        self_consumption=self_consumption,
                        cashflow=cashflow,
                        co2_flow=co2_flow,
                        )


def run_backend(inputs: Inputs) -> TotalResults:
    # start time tracking
    start_time = time()

    # get log data for the simulation
    logs = Logs(pv_spec=get_log_pv(coordinates=inputs.location.coordinates),
                dem=get_log_dem(slp=inputs.location.slp.lower(),
                                consumption_yrl_wh=inputs.location.consumption_yrl_wh,
                                ),
                # ToDo: get input parameters from settings
                fleet={vehicle_type: get_log_subfleet(vehicle_type=vehicle_type)
                       for vehicle_type, subfleet in inputs.subfleets.items()},
                )

    results = {phase: calc_phase_results(logs=logs,
                                         capacities=inputs.location.get_capacities(phase),
                                         economics=inputs.economic.get_phase_input(phase=phase),
                                         subfleets={sf.name: sf.get_phase_input(phase=phase)
                                                    for sf in inputs.subfleets.values()},
                                         chargers={chg.name: chg.get_phase_input(phase=phase)
                                                   for chg in inputs.chargers.values()},
                                         )
               for phase in ["baseline", "expansion"]}

    # stop time tracking
    print(f'Backend calculation completed in {time() - start_time:.2f} seconds.')

    return TotalResults(baseline=results['baseline'],
                        expansion=results['expansion'],
                        )


if __name__ == "__main__":
    settings_default = Inputs()
    result = run_backend(inputs=settings_default)
    print(result.baseline.simulation)
    print(result.expansion.simulation)
