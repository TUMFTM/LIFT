import demandlib
import importlib.resources as resources
import pandas as pd
import pvlib
import streamlit as st
from time import time
from typing import TYPE_CHECKING, Literal, Tuple, Dict, Optional
from definitions import FREQ_HOURS, CHARGERS, SUBFLEETS

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
                        Input,
                        InputLocation,
                        SimInputCharger,
                        PhaseInputCharger,
                        InputCharger,
                        SimInputSubfleet,
                        PhaseInputSubfleet,
                        InputSubfleet,
                        InputEconomic,
                        Logs,
                        Capacities,
                        SimInputSubfleet,
                        SimInputCharger,
                        ResultSimulation,
                        PhaseResults,
                        BackendResults,
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
             ) -> ResultSimulation:

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

    return ResultSimulation(energy_pv_pot_wh=pv.energy_pot_wh,
                            energy_pv_curt_wh=grid.energy_curt_wh,
                            energy_grid_buy_wh=grid.energy_buy_wh,
                            energy_grid_sell_wh=grid.energy_sell_wh,
                            pwr_grid_peak_w=grid.pwr_peak_w,
                            energy_dem_wh=dem.energy_wh,
                            energy_fleet_wh=fleet.energy_wh,
                            )


def calc_opex_vehicle(
    logs: "Logs",
    subfleet_settings: Dict[str, "InputSubfleet"],
    economics: "InputEconomic",
    phase: Literal["baseline", "expansion"],
    vehicle_charger: Dict[str, Dict[str, Optional[str]]] = None,
) -> Tuple[
    float, Dict[str, float], float, Dict[str, float], Dict[str, float]
]:

    # Falls nicht mitgegeben: Mapping BEV/ICEV je Fahrzeug erzeugen
    if vehicle_charger is None:
        vehicle_charger = {}
        for vt, sf in subfleet_settings.items():
            total = int(sf.num_total)
            bev_count = int(sf.num_bev.preexisting + (sf.num_bev.expansion if phase == "expansion" else 0))
            vt_map: Dict[str, Optional[str]] = {}
            for i in range(total):
                veh_id = f"{vt}{i}"
                vt_map[veh_id] = (sf.charger.lower() if i < bev_count else None)  # None -> ICEV
            vehicle_charger[vt] = vt_map

    CO2_PER_LITER_DIESEL_KG = 3.08  # kg CO2 / Liter Diesel

    # Kosten-/Tarifparameter (subfleet-unabhängig)
    mntex_bev        = float(economics.mntex_bev_eur_km)
    mntex_icev       = float(economics.mntex_icev_eur_km)
    toll_bev         = float(getattr(economics, "toll_bev_eur_km", 0.0))
    toll_icev        = float(economics.toll_icev_eur_km)
    insurance_rate   = float(economics.insurance_pct) / 100.0
    fuel_price_eur_l = float(economics.fuel_price_eur_liter)

    opex_total_eur: float = 0.0
    opex_by_sf: Dict[str, float] = {}
    co2_total_kg: float = 0.0
    co2_by_sf: Dict[str, float] = {}
    comp_totals = {"maintenance": 0.0, "insurance": 0.0, "fuel_diesel": 0.0, "toll": 0.0}

    for vt, sf in subfleet_settings.items():
        # ---> Verbrauch je Subfleet aus SUBFLEETS holen
        vt_key = vt if vt in SUBFLEETS else vt.lower()
        meta = SUBFLEETS.get(vt_key)

        fuel_l_per_100km = float(meta.consumption_icev)
        fuel_eur_per_km  = fuel_price_eur_l * (fuel_l_per_100km / 100.0)
        co2_kg_per_km    = CO2_PER_LITER_DIESEL_KG * (fuel_l_per_100km / 100.0)

        df = logs.fleet.get(vt)  # MultiIndex-Spalten: (veh_id, metric)

        n_total = int(sf.num_total)
        opex_vt = 0.0
        co2_vt_kg = 0.0
        toll_split = float(sf.toll_share_pct) / 100.0

        opex_maintenance = 0.0
        opex_insurance = 0.0
        opex_fuel = 0.0
        opex_toll = 0.0
        co2_tailpipe_kg = 0.0

        for i in range(n_total):
            veh_id = f"{vt}{i}"
            chg = vehicle_charger[vt][veh_id]
            is_bev = not (chg is None or str(chg).lower() == "none")  # None/"none" → ICEV

            # Distanz (km) – deine Spalte heißt 'dist'
            distance_km = float(df.loc[:, (veh_id, "dist")].astype(float).sum()) if (veh_id, "dist") in df.columns else 0.0

            if is_bev:
                opex_maintenance = mntex_bev * distance_km
                opex_insurance = float(sf.capex_bev_eur) * insurance_rate
                opex_toll = toll_bev * distance_km * toll_split
            else:
                opex_maintenance = mntex_icev * distance_km
                opex_insurance = float(sf.capex_icev_eur) * insurance_rate
                opex_fuel = fuel_eur_per_km * distance_km
                opex_toll = toll_icev * distance_km * toll_split
                co2_tailpipe_kg = co2_kg_per_km * distance_km

            opex_vt   += (opex_maintenance + opex_insurance + opex_fuel + opex_toll)
            co2_vt_kg += co2_tailpipe_kg

        opex_by_sf[vt] = opex_vt
        co2_by_sf[vt]  = co2_vt_kg
        opex_total_eur += opex_vt
        co2_total_kg   += co2_vt_kg

        comp_totals["maintenance"] += opex_maintenance
        comp_totals["insurance"] += opex_insurance
        comp_totals["fuel_diesel"] += opex_fuel
        comp_totals["toll"] += opex_toll

    return float(opex_total_eur), opex_by_sf, float(co2_total_kg), co2_by_sf, comp_totals



def calc_infrastructure_capex(
    location: "InputLocation",
    charger_settings: Dict[str, "InputCharger"],
    economics: "InputEconomic",
    phase: Literal["baseline", "expansion"],
) -> Tuple[float, Dict[str, float], float, Dict[str, float]]:
    """
    Calculate infrastructure CAPEX for the given phase.
    - 'baseline'  -> 0 EUR
    - 'expansion' -> Costs for additional grid/PV/ESS capacity + new chargers

    Uses cost rates from 'economics' (with defaults if not set):
      - capex_spec_grid_eur_per_w
      - capex_spec_pv_eur_per_wp
      - capex_spec_ess_eur_per_wh

    Returns:
      (capex_total_eur,
       {"grid": ..., "pv": ..., "ess": ..., "chargers": ...},
       co2_total_kg,
       {"grid": ..., "pv": ..., "ess": ..., "chargers": ...}
    """

    capex_chargers_ac = 0.0
    capex_chargers_dc = 0.0
    co2_chargers_ac = 0.0
    co2_chargers_dc = 0.0

    if phase == "baseline":
        return (
            0.0,
            {"grid": 0.0, "pv": 0.0, "ess": 0.0, "chargers": 0.0},
            0.0,
            {"grid": 0.0, "pv": 0.0, "ess": 0.0, "chargers": 0.0},
        )

    # Cost rates (same defaults as in your current implementation)
    rate_grid = 200 # EUR / kW
    rate_pv   = 900 # EUR / kWp
    rate_ess  = 450 # EUR / kWh

    # ------- Embodied CO2 factors (kg CO2 per unit) -------
    co2_per_kw_grid = 0.0  #ToDo co2 trafo relevant?
    co2_per_kwp_pv = 798.0  # kg/kWp
    co2_per_kwh_ess = 69.0  # kg/kWh

    # Expansion capacities (W, Wp, Wh) -> nach kW/kWp/kWh umrechnen
    grid_exp_kw = float(location.grid_capacity_w.expansion or 0.0) / 1000.0
    pv_exp_kwp = float(location.pv_capacity_wp.expansion or 0.0) / 1000.0
    ess_exp_kwh = float(location.ess_capacity_wh.expansion or 0.0) / 1000.0

    # CAPEX
    capex_grid     = grid_exp_kw  * rate_grid
    capex_pv       = pv_exp_kwp   * rate_pv
    capex_ess      = ess_exp_kwh  * rate_ess

    # Charger CAPEX: expansion number * cost per charger
    capex_chargers = float(sum(
        float(cs.num.expansion) * float(cs.cost_per_charger_eur)
        for cs in charger_settings.values()
    ))

    co2_grid = grid_exp_kw  * co2_per_kw_grid
    co2_pv   = pv_exp_kwp   * co2_per_kwp_pv
    co2_ess  = ess_exp_kwh  * co2_per_kwh_ess

    for name, cs in charger_settings.items():
        key = str(name).strip().lower()  # "ac" / "dc"
        meta = CHARGERS.get(key)
        capex = float(cs.num.expansion) * float(cs.cost_per_charger_eur)
        co2 = float(cs.num.expansion) * float(meta.settings_CO2_per_unit)
        if key == "ac":
            capex_chargers_ac += capex
            co2_chargers_ac += co2
        elif key == "dc":
            capex_chargers_dc += capex
            co2_chargers_dc += co2

    capex_chargers = capex_chargers_ac + capex_chargers_dc
    co2_chargers = co2_chargers_ac + co2_chargers_dc

    capex_breakdown = {
        "grid": capex_grid,
        "pv": capex_pv,
        "ess": capex_ess,
        "chargers": capex_chargers,
        "chargers_ac": capex_chargers_ac,
        "chargers_dc": capex_chargers_dc,
    }
    co2_breakdown = {
        "grid": co2_grid,
        "pv": co2_pv,
        "ess": co2_ess,
        "chargers": co2_chargers,
        "chargers_ac": co2_chargers_ac,
        "chargers_dc": co2_chargers_dc,
    }
    capex_total = capex_grid + capex_pv + capex_ess + capex_chargers
    co2_total   = co2_grid + co2_pv + co2_ess + co2_chargers
    return float(capex_total), capex_breakdown, float(co2_total), co2_breakdown


def calc_vehicle_capex_split(
    subfleet_settings: dict[str, "InputSubfleet"],
    economics: "InputEconomic",
    phase: Literal["baseline", "expansion"],
) -> Tuple[
    float, float, Dict[str, float], Dict[str, float],   # CAPEX totals & breakdowns
    float, float, Dict[str, float], Dict[str, float],   # CO2 totals  & breakdowns
]:
    """
    CAPEX-Split (BEV/ICEV) + Herstellungs-CO2 (embodied) für die gewählte Phase.
    - baseline:   BEV = num_bev.preexisting; ICEV = num_total - BEV
    - expansion:  BEV = num_bev.preexisting + num_bev.expansion; ICEV = num_total - BEV
                  (Flottengröße bleibt konstant)
    Rückgabe:
      capex_bev_total_eur, capex_icev_total_eur, capex_bev_by_sf, capex_icev_by_sf,
      co2_bev_total_kg,    co2_icev_total_kg,   co2_bev_by_sf,    co2_icev_by_sf
    """
    bev_total_capex = 0.0
    icev_total_capex = 0.0
    capex_bd_bev: Dict[str, float] = {}
    capex_bd_icev: Dict[str, float] = {}

    salv_bev = float(economics.salvage_bev_pct) / 100.0
    salv_ice = float(economics.salvage_icev_pct) / 100.0

    co2_bev_total_kg = 0.0
    co2_icev_total_kg = 0.0
    co2_bd_bev: Dict[str, float] = {}
    co2_bd_icev: Dict[str, float] = {}

    for vt, sf in subfleet_settings.items():
        vt_key = vt if vt in SUBFLEETS else vt.lower()
        meta = SUBFLEETS[vt_key]

        # Fahrzeuganzahl je Phase (alle Fahrzeuge der Flotte)
        if phase == "baseline":
            bev_cnt  = int(sf.num_bev.preexisting)
        else:  # expansion
            bev_cnt  = int(sf.num_bev.preexisting + sf.num_bev.expansion)

        icev_cnt = max(int(sf.num_total) - bev_cnt, 0)

        # --- CAPEX ---
        capex_bev_sf  = float(bev_cnt)  * float(sf.capex_bev_eur) * (1 - salv_bev)
        capex_icev_sf = float(icev_cnt) * float(sf.capex_icev_eur) * (1 - salv_ice)

        capex_bd_bev[vt_key]  = capex_bev_sf
        capex_bd_icev[vt_key] = capex_icev_sf
        bev_total_capex  += capex_bev_sf
        icev_total_capex += capex_icev_sf

        # --- CO2 (Herstellung/embodied) ---
        co2_bev_sf_kg  = float(bev_cnt)  * float(meta.co2_production_bev)
        co2_icev_sf_kg = float(icev_cnt) * float(meta.co2_production_icev)

        co2_bd_bev[vt_key]  = co2_bev_sf_kg
        co2_bd_icev[vt_key] = co2_icev_sf_kg
        co2_bev_total_kg  += co2_bev_sf_kg
        co2_icev_total_kg += co2_icev_sf_kg

    return (
        float(bev_total_capex),
        float(icev_total_capex),
        capex_bd_bev,
        capex_bd_icev,
        float(co2_bev_total_kg),
        float(co2_icev_total_kg),
        co2_bd_bev,
        co2_bd_icev,
    )


def calc_phase_results(logs: Logs,
                       capacities: Capacities,
                       economics: InputEconomic,
                       subfleets: dict[str, PhaseInputSubfleet],
                       chargers: dict[str, PhaseInputCharger],
                       phase: str,
                       location: InputLocation,
                       ) -> PhaseResults:

    cost = dict()
    co2 = dict()

    result_sim = simulate(
        logs=logs,
        capacities=capacities,
        subfleets={sf.name: sf.get_sim_input() for sf in subfleets.values()},
        chargers={chg.name: chg.get_sim_input() for chg in chargers.values()},
    )

    # calculate self-sufficiency and self consumption based on simulation results
    if capacities.pv_wp == 0:
        self_sufficiency_pct = 0.0
        self_consumption_pct = 0.0
    else:
        # potential PV energy minus curtailed and sold energy divided by total energy demand (fleet + site)
        self_sufficiency_pct = (result_sim.energy_pv_pot_wh - result_sim.energy_pv_curt_wh - result_sim.energy_grid_sell_wh) / (result_sim.energy_fleet_wh + result_sim.energy_dem_wh) * 100
        # 1 - (pv energy not used on-site (curtailed or sold) divided by total potential PV energy)
        self_consumption_pct = 100 - ((result_sim.energy_grid_sell_wh + result_sim.energy_pv_curt_wh) / result_sim.energy_pv_pot_wh) * 100

    cost['grid_yrl'] = (result_sim.energy_grid_buy_wh * economics.opex_spec_grid_buy_eur_per_wh -
                     result_sim.energy_grid_sell_wh * economics.opex_spec_grid_sell_eur_per_wh)

    # ToDo: check whether net balance is correct here
    co2['grid_yrl'] = (result_sim.energy_grid_buy_wh - result_sim.energy_grid_sell_wh) * CO2_SPEC_KG_PER_WH

    cashflow = np.zeros(TIME_PRJ_YRS) + np.random.random()  # ToDo: fix
    co2_flow = np.zeros(TIME_PRJ_YRS) + 1

    return PhaseResults(simulation=result_sim,
                        self_sufficiency_pct=self_sufficiency_pct,
                        self_consumption_pct=self_consumption_pct,
                        cashflow=cashflow,
                        co2_flow=co2_flow,
                        )


def run_backend(input: Input) -> BackendResults:
    # start time tracking
    start_time = time()

    # get log data for the simulation
    logs = Logs(pv_spec=get_log_pv(coordinates=input.location.coordinates),
                dem=get_log_dem(slp=input.location.slp.lower(),
                                consumption_yrl_wh=input.location.consumption_yrl_wh,
                                ),
                # ToDo: get input parameters from settings
                fleet={vehicle_type: get_log_subfleet(vehicle_type=vehicle_type)
                       for vehicle_type, subfleet in input.subfleets.items()},
                )

    results_baseline = calc_phase_results(logs=logs,
                                          capacities=input.location.get_capacities('baseline'),
                                          economics=input.economic,
                                          subfleets={sf.name: sf.get_phase_input(phase='baseline')
                                                     for sf in input.subfleets.values()},
                                          chargers={chg.name: chg.get_phase_input(phase='baseline')
                                                     for chg in input.chargers.values()},
                                          phase="baseline",
                                          location=input.location,
                                          )

    results_expansion = calc_phase_results(logs=logs,
                                           capacities=input.location.get_capacities('expansion'),
                                           economics=input.economic,
                                           subfleets={sf.name: sf.get_phase_input(phase='expansion')
                                                      for sf in input.subfleets.values()},
                                           chargers={chg.name: chg.get_phase_input(phase='expansion')
                                                     for chg in input.chargers.values()},
                                           phase="expansion",
                                           location=input.location,
                                           )

    roi_rel = 0.0  #ToDo: calculate!
    period_payback_rel = 0.0  #ToDo: calculate!
    npc_delta = 0.0  #ToDo: calculate!

    # stop time tracking
    print(f'Backend calculation completed in {time() - start_time:.2f} seconds.')

    return BackendResults(baseline=results_baseline,
                          expansion=results_expansion,
                          roi_rel=roi_rel,
                          period_payback=period_payback_rel,
                          )


if __name__ == "__main__":
    settings_default = Input()
    result = run_backend(input=settings_default)
    print(result.baseline.simulation)
    print(result.expansion.simulation)
