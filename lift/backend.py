import demandlib
import importlib.resources as resources
import pandas as pd
import pvlib
import streamlit as st
from time import time
from typing import TYPE_CHECKING

import numpy as np

from definitions import (DTI,
                         CO2_SPEC_KG_PER_WH,
                         OPEX_SPEC_CO2_PER_KG,
                         TIME_PRJ_YRS,
                         )

from energy_system import (FixedDemand,
                           Fleet,
                           FleetUnit,
                           GridConnection,
                           PVSource,
                           StationaryStorage)

from interfaces import (Coordinates,
                        Logs,
                        Capacities,
                        Settings,
                        SubfleetSimSettings,
                        EconomicSettings,
                        SimulationResults,
                        PhaseResults,
                        BackendResults)

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
             subfleets: dict[str, SubfleetSimSettings],
             chargers: dict[str, int],
             ) -> SimulationResults:

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

    return SimulationResults(energy_pv_pot_wh=pv.energy_pot_wh,
                             energy_pv_curt_wh=grid.energy_curt_wh,
                             energy_grid_buy_wh=grid.energy_buy_wh,
                             energy_grid_sell_wh=grid.energy_sell_wh,
                             pwr_grid_peak_w=grid.pwr_peak_w,
                             energy_dem_wh=dem.energy_wh,
                             energy_fleet_wh=fleet.energy_wh,
                             )


@st.cache_data
def calc_phase_results(logs: Logs,
                       capacities: Capacities,
                       economics: EconomicSettings,
                       subfleets: dict[str, SubfleetSimSettings],
                       chargers: dict[str, int],
                       ) -> PhaseResults:

    result_sim = simulate(logs=logs,
                          capacities=capacities,
                          subfleets=subfleets,
                          chargers=chargers,)

    # potential PV energy minus curtailed and sold energy divided by total energy demand (fleet + site)
    if capacities.pv_wp > 0:
        self_sufficiency_pct = (result_sim.energy_pv_pot_wh - result_sim.energy_pv_curt_wh - result_sim.energy_grid_sell_wh) / (result_sim.energy_fleet_wh + result_sim.energy_dem_wh) * 100
        # 1 - (pv energy not used on-site (curtailed or sold) divided by total potential PV energy)
        self_consumption_pct = 100 - ((result_sim.energy_grid_sell_wh + result_sim.energy_pv_curt_wh) / result_sim.energy_pv_pot_wh) * 100
    else:
        self_sufficiency_pct = 0.0
        self_consumption_pct = 0.0

    co2_yrl_kg = result_sim.energy_grid_buy_wh * CO2_SPEC_KG_PER_WH
    co2_yrl_eur = co2_yrl_kg * OPEX_SPEC_CO2_PER_KG

    opex_grid_energy = (result_sim.energy_grid_buy_wh * economics.opex_spec_grid_buy_eur_per_wh -
                        result_sim.energy_grid_sell_wh * economics.opex_spec_grid_sell_eur_per_wh)

    opex_grid_power = result_sim.pwr_grid_peak_w * economics.opex_spec_grid_peak_eur_per_wp

    opex_grid = opex_grid_energy + opex_grid_power

    return PhaseResults(simulation=result_sim,
                        self_sufficiency_pct=self_sufficiency_pct,
                        self_consumption_pct=self_consumption_pct,
                        co2_yrl_kg=co2_yrl_kg,
                        co2_yrl_eur=co2_yrl_eur,
                        capex_eur=0.0,  # ToDo: calculate CAPEX
                        opex_fuel_eur=0.0,  # ToDo: calculate OPEX fuel
                        opex_toll_eur=0.0,  # ToDo: calculate OPEX toll
                        opex_grid_eur=opex_grid,
                        cashflow=np.zeros(TIME_PRJ_YRS, dtype=np.float64),  # ToDo: calculate cashflow
                        )


def run_backend(settings: Settings) -> BackendResults:
    # start time tracking
    start_time = time()

    # get log data for the simulation
    logs = Logs(pv_spec=get_log_pv(coordinates=settings.location.coordinates),
                dem=get_log_dem(slp=settings.location.slp,
                                consumption_yrl_wh=settings.location.consumption_yrl_wh,
                                ),
                # ToDo: get input parameters from settings
                fleet={vehicle_type: get_log_subfleet(vehicle_type=vehicle_type)
                       for vehicle_type, subfleet in settings.subfleets.items()},
                )

    subfleet_sim_settings_baseline = {subfleet.vehicle_type:
                                          subfleet.get_subfleet_sim_settings_baseline(settings.chargers)
                                      for subfleet in settings.subfleets.values()}

    subfleet_sim_settings_expansion = {subfleet.vehicle_type:
                                           subfleet.get_subfleet_sim_settings_expansion(settings.chargers)
                                       for subfleet in settings.subfleets.values()}


    results_baseline = calc_phase_results(logs=logs,
                                          capacities=settings.location.get_capacities('baseline'),
                                          economics=settings.economic,
                                          subfleets=subfleet_sim_settings_baseline,
                                          chargers={charger_type: charger.num_preexisting
                                                    for charger_type, charger in settings.chargers.items()},
                                          )

    results_expansion = calc_phase_results(logs=logs,
                                           capacities=settings.location.get_capacities('expansion'),
                                           economics=settings.economic,
                                           subfleets=subfleet_sim_settings_expansion,
                                           chargers={charger_type: charger.num_preexisting + charger.num_expansion
                                                     for charger_type, charger in settings.chargers.items()},
                                           )



    roi_rel = 0.0  #ToDo: calculate!
    period_payback_rel = 0.0  #ToDo: calculate!
    npc_delta = 0.0  #ToDo: calculate!

    # stop time tracking
    print(f'Backend calculation completed in {time() - start_time:.2f} seconds.')

    return BackendResults(baseline=results_baseline,
                          expansion=results_expansion,
                          roi_rel=roi_rel,
                          period_payback_rel=period_payback_rel,
                          npc_delta=npc_delta,
                          )


if __name__ == "__main__":
    settings_default = Settings()
    result = run_backend(settings=settings_default)
    print(result.baseline.simulation)
    print(result.expansion.simulation)
