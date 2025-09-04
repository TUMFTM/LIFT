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
                        Logs,
                        Capacities,
                        Settings,
                        SubfleetSimSettings,
                        EconomicSettings,
                        SimulationResults,
                        PhaseResults,
                        BackendResults,
                        LocationSettings,
                        ChargerSettings,
                        SubFleetSettings
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


def calc_opex_vehicle(
    logs: "Logs",
    subfleet_settings: Dict[str, "SubFleetSettings"],
    economics: "EconomicSettings",
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
            bev_count = int(sf.num_bev_preexisting + (sf.num_bev_expansion if phase == "expansion" else 0))
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
    location: "LocationSettings",
    charger_settings: Dict[str, "ChargerSettings"],
    economics: "EconomicSettings",
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
        float(cs.num_expansion) * float(cs.cost_per_charger_eur)
        for cs in charger_settings.values()
    ))

    co2_grid = grid_exp_kw  * co2_per_kw_grid
    co2_pv   = pv_exp_kwp   * co2_per_kwp_pv
    co2_ess  = ess_exp_kwh  * co2_per_kwh_ess

    for name, cs in charger_settings.items():
        key = str(name).strip().lower()  # "ac" / "dc"
        meta = CHARGERS.get(key)
        capex = float(cs.num_expansion) * float(cs.cost_per_charger_eur)
        co2 = float(cs.num_expansion) * float(meta.settings_CO2_per_unit)
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
    subfleet_settings: dict[str, "SubFleetSettings"],
    economics: "EconomicSettings",
    phase: Literal["baseline", "expansion"],
) -> Tuple[
    float, float, Dict[str, float], Dict[str, float],   # CAPEX totals & breakdowns
    float, float, Dict[str, float], Dict[str, float],   # CO2 totals  & breakdowns
]:
    """
    CAPEX-Split (BEV/ICEV) + Herstellungs-CO2 (embodied) für die gewählte Phase.
    - baseline:   BEV = num_bev_preexisting; ICEV = num_total - BEV
    - expansion:  BEV = num_bev_preexisting + num_bev_expansion; ICEV = num_total - BEV
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
            bev_cnt  = int(sf.num_bev_preexisting)
        else:  # expansion
            bev_cnt  = int(sf.num_bev_preexisting + sf.num_bev_expansion)

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
                       economics: EconomicSettings,
                       subfleets: dict[str, SubfleetSimSettings],
                       chargers: dict[str, int],
                       *,
                       phase: str,
                       location: LocationSettings,
                       charger_settings: dict[str, ChargerSettings],
                       subfleet_settings: dict[str, SubFleetSettings],
                       ) -> PhaseResults:
    result_sim = simulate(
        logs=logs,
        capacities=capacities,
        subfleets=subfleets,
        chargers=chargers,
    )

    (opex_vehicle_total,
     opex_vehicle_by_sf,
     co2_tailpipe_total_kg,
     co2_tailpipe_by_sf,
     opex_vehicle_comp) = calc_opex_vehicle(
        logs=logs,
        subfleet_settings=subfleet_settings,
        economics=economics,
        phase=phase,
    )

    grid_buy_eur = result_sim.energy_grid_buy_wh * economics.opex_spec_grid_buy_eur_per_wh
    grid_sell_eur = result_sim.energy_grid_sell_wh * economics.opex_spec_grid_sell_eur_per_wh

    opex_breakdown = {
        "Stromeinkauf": float(grid_buy_eur),
        "Stromverkauf": -float(grid_sell_eur),  # als negative Kosten (Erlös)
        "Diesel": float(opex_vehicle_comp["fuel_diesel"]),
        "Maut": float(opex_vehicle_comp["toll"]),
        "Wartung": float(opex_vehicle_comp["maintenance"]),
        "Versicherung": float(opex_vehicle_comp["insurance"]),
    }

    co2_grid_yrl_kg = result_sim.energy_grid_buy_wh * CO2_SPEC_KG_PER_WH
    co2_tailpipe_yrl_kg = co2_tailpipe_total_kg
    co2_yrl_kg = co2_grid_yrl_kg + co2_tailpipe_yrl_kg

    # potential PV energy minus curtailed and sold energy divided by total energy demand (fleet + site)
    if capacities.pv_wp > 0:
        self_sufficiency_pct = (result_sim.energy_pv_pot_wh - result_sim.energy_pv_curt_wh - result_sim.energy_grid_sell_wh) / (result_sim.energy_fleet_wh + result_sim.energy_dem_wh) * 100
        # 1 - (pv energy not used on-site (curtailed or sold) divided by total potential PV energy)
        self_consumption_pct = 100 - ((result_sim.energy_grid_sell_wh + result_sim.energy_pv_curt_wh) / result_sim.energy_pv_pot_wh) * 100
    else:
        self_sufficiency_pct = 0.0
        self_consumption_pct = 0.0

    co2_yrl_kg = result_sim.energy_grid_buy_wh * CO2_SPEC_KG_PER_WH + co2_tailpipe_total_kg
    co2_yrl_eur = co2_yrl_kg * OPEX_SPEC_CO2_PER_KG

    if phase == "baseline":
        capex_infra_eur = 0.0
        capex_infra_bd = {"grid": 0.0, "pv": 0.0, "ess": 0.0, "chargers": 0.0}
        co2_infra_kg = 0.0
        co2_infra_bd = {"grid": 0.0, "pv": 0.0, "ess": 0.0, "chargers": 0.0}
    else:
        (capex_infra_eur,
         capex_infra_bd,
         co2_infra_kg,
         co2_infra_bd) = calc_infrastructure_capex(
            location=location,
            charger_settings=charger_settings,
            economics=economics,
            phase=phase,
        )

    # --- vehicle CAPEX split for each phase---
    (capex_bev_eur, capex_icev_eur,
     capex_bev_by_sf, capex_icev_by_sf,
     co2_prod_bev_kg, co2_prod_icev_kg,
     co2_prod_bev_by_sf, co2_prod_icev_by_sf) = calc_vehicle_capex_split(
        subfleet_settings=subfleet_settings,
        economics=economics,
        phase=phase.lower(),
    )

    capex_vehicles_eur = capex_bev_eur + capex_icev_eur
    co2_vehicles_production_total_kg = co2_prod_bev_kg + co2_prod_icev_kg

    capex_vehicles_by_sf = {
        k: float(capex_bev_by_sf.get(k, 0.0)) + float(capex_icev_by_sf.get(k, 0.0))
        for k in set(capex_bev_by_sf) | set(capex_icev_by_sf)
    }

    # --- total-CAPEX ---
    capex_total_eur = capex_infra_eur + capex_vehicles_eur

    opex_grid_energy = (result_sim.energy_grid_buy_wh * economics.opex_spec_grid_buy_eur_per_wh -
                        result_sim.energy_grid_sell_wh * economics.opex_spec_grid_sell_eur_per_wh)

    opex_grid_power = result_sim.pwr_grid_peak_w * economics.opex_spec_grid_peak_eur_per_wp

    opex_grid = opex_grid_energy + opex_grid_power

    # === CASHFLOW (18 years) according to your rules ===
    cashflow = np.zeros(TIME_PRJ_YRS, dtype=np.float64)

    # Annual OPEX in every year (including grid + vehicle)
    annual_opex = opex_grid + opex_vehicle_total
    for y in range(TIME_PRJ_YRS):
        cashflow[y] += annual_opex

    # Vehicles: full replacement costs in year 1, 6, and 12 (indices 0, 5, 11)
    veh_years_idx = [0, 5, 11]
    for idx in veh_years_idx:
        if idx < TIME_PRJ_YRS:
            cashflow[idx] += capex_vehicles_eur

    # Infrastructure only in expansion scenario
    if phase == "expansion":
        fix_cost = float(getattr(economics, "fix_cost_construction", 0.0))

        # in Breakdown aufnehmen (neuer Key "construction")
        capex_infra_bd["construction"] = capex_infra_bd.get("construction", 0.0) + fix_cost
        capex_infra_eur += fix_cost

        # CO2 für Bau-Fixkosten (meist 0)
        co2_infra_bd["construction"] = co2_infra_bd.get("construction", 0.0) + 0.0
        co2_infra_kg = float(sum(co2_infra_bd.values()))

        infra_year0 = (
                capex_infra_bd.get("grid", 0.0)
                + capex_infra_bd.get("pv", 0.0)
                + capex_infra_bd.get("chargers", 0.0)
                + capex_infra_bd.get("construction", 0.0)
        )
        cashflow[0] += infra_year0

        # ESS in year 1 and year 9 (full cost each time)
        ess_total = capex_infra_bd.get("ess", 0.0)
        cashflow[0] += ess_total
        if 8 < TIME_PRJ_YRS:
            cashflow[8] += ess_total

    # === CO2-FLOW (18 Jahre) analog zum Cashflow ===
    co2_flow = np.zeros(TIME_PRJ_YRS, dtype=np.float64)

    # 1) Betrieb: jährlich gleich (Grid + Tailpipe)
    for y in range(TIME_PRJ_YRS):
        co2_flow[y] += co2_yrl_kg  # = co2_grid_yrl_kg + co2_tailpipe_yrl_kg

    # 2) Fahrzeuge: Herstellung zu den gleichen Zeitpunkten wie Fahrzeug-CAPEX
    veh_cohort_co2 = co2_vehicles_production_total_kg  # BEV+ICEV für die Phase
    for idx in veh_years_idx:
        if idx < TIME_PRJ_YRS:
            co2_flow[idx] += veh_cohort_co2

    # 3) Infrastruktur: nur in Expansion, analog zu CAPEX-Timings
    if phase == "expansion":
        # Grid + PV + Charger in Jahr 1 (Index 0)
        co2_infra_year0 = (
                co2_infra_bd.get("grid", 0.0)
                + co2_infra_bd.get("pv", 0.0)
                + co2_infra_bd.get("chargers", 0.0)
        )
        co2_flow[0] += co2_infra_year0

        # ESS in Jahr 1 und Jahr 9 (Index 0 und 8) – analog zu deinen CAPEX-Regeln
        co2_ess = co2_infra_bd.get("ess", 0.0)
        co2_flow[0] += co2_ess
        if 8 < TIME_PRJ_YRS:
            co2_flow[8] += co2_ess

    return PhaseResults(simulation=result_sim,
                        self_sufficiency_pct=self_sufficiency_pct,
                        self_consumption_pct=self_consumption_pct,
                        co2_yrl_kg=co2_yrl_kg,
                        co2_yrl_eur=co2_yrl_eur,
                        capex_eur=capex_total_eur,
                        capex_vehicles_eur=capex_vehicles_eur,
                        capex_vehicles_bev_eur=capex_bev_eur,
                        capex_vehicles_icev_eur=capex_icev_eur,
                        opex_fuel_eur=0.0,  # ToDo: calculate OPEX fuel
                        opex_toll_eur=0.0,  # ToDo: calculate OPEX toll
                        opex_grid_eur=opex_grid,
                        opex_vehicle_electric_secondary=opex_vehicle_total,
                        opex_breakdown=opex_breakdown,
                        capex_vehicles_by_subfleet=capex_vehicles_by_sf,
                        cashflow=cashflow,  # ToDo: calculate cashflow
                        co2_flow=co2_flow,
                        infra_capex_breakdown=capex_infra_bd,
                        infra_co2_total_kg=co2_infra_kg,
                        infra_co2_breakdown=co2_infra_bd,
                        co2_grid_yrl_kg=co2_grid_yrl_kg,
                        co2_tailpipe_yrl_kg=co2_tailpipe_yrl_kg,
                        co2_tailpipe_by_subfleet_kg=co2_tailpipe_by_sf,
                        vehicles_co2_production_total_kg=co2_vehicles_production_total_kg,
                        vehicles_co2_production_bev_kg=co2_prod_bev_kg,
                        vehicles_co2_production_icev_kg=co2_prod_icev_kg,
                        vehicles_co2_production_breakdown_bev=co2_prod_bev_by_sf,
                        vehicles_co2_production_breakdown_icev=co2_prod_icev_by_sf,
                        )


def run_backend(settings: Settings) -> BackendResults:
    # start time tracking
    start_time = time()

    # get log data for the simulation
    logs = Logs(pv_spec=get_log_pv(coordinates=settings.location.coordinates),
                dem=get_log_dem(slp=settings.location.slp.lower(),
                                consumption_yrl_wh=settings.location.consumption_yrl_wh,
                                ),
                # ToDo: get input parameters from settings
                fleet={vehicle_type: get_log_subfleet(vehicle_type=vehicle_type)
                       for vehicle_type, subfleet in settings.subfleets.items()},
                )

    subfleet_sim_settings_baseline = {subfleet.name:
                                          subfleet.get_subfleet_sim_settings_baseline(settings.chargers)
                                      for subfleet in settings.subfleets.values()}

    subfleet_sim_settings_expansion = {subfleet.name:
                                           subfleet.get_subfleet_sim_settings_expansion(settings.chargers)
                                       for subfleet in settings.subfleets.values()}

    chargers_baseline = {
        str(t).strip().lower(): int(c.num_preexisting)
        for t, c in settings.chargers.items()
    }
    chargers_expansion = {
        str(t).strip().lower(): int(c.num_preexisting + c.num_expansion)
        for t, c in settings.chargers.items()
    }

    results_baseline = calc_phase_results(logs=logs,
                                          capacities=settings.location.get_capacities('baseline'),
                                          economics=settings.economic,
                                          subfleets=subfleet_sim_settings_baseline,
                                          chargers=chargers_baseline,
                                          phase="baseline",
                                          location=settings.location,
                                          charger_settings=settings.chargers,
                                          subfleet_settings=settings.subfleets,
                                          )

    results_expansion = calc_phase_results(logs=logs,
                                           capacities=settings.location.get_capacities('expansion'),
                                           economics=settings.economic,
                                           subfleets=subfleet_sim_settings_expansion,
                                           chargers=chargers_expansion,
                                           phase="expansion",
                                           location=settings.location,
                                           charger_settings=settings.chargers,
                                           subfleet_settings=settings.subfleets,
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
