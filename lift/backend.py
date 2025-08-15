import demandlib
import importlib.resources as resources
import pandas as pd
import pvlib
import streamlit as st
from time import time
from typing import TYPE_CHECKING, Literal, Tuple, Dict, Optional
from definitions import FREQ_HOURS

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

# ---- Helpers: flexible Spaltenerkennung ----
DIST_ALIASES = {
    "distance_km", "distance", "dist_km", "km", "driven_km", "route_km", "mileage_km",
    "distance_m", "dist_m", "meters", "meter", "m",
}

def _find_distance_km(df: pd.DataFrame, veh_id: str) -> Optional[float]:
    """
    Sucht in df (MultiIndex: (veh_id, metric)) eine Distanzspalte und summiert über das Jahr.
    Unterstützt km- und m-Spalten. Gibt km zurück oder None, wenn nichts gefunden.
    """
    if df is None or not isinstance(df.columns, pd.MultiIndex):
        return None

    # Kandidaten dieses Fahrzeugs
    candidates = [c for c in df.columns if c[0] == veh_id]
    if not candidates:
        return None

    def score(metric: str) -> int:
        m = metric.lower()
        if m in DIST_ALIASES:
            return 100
        if m.endswith("_km"):
            return 80
        if "distance" in m:
            return 60
        if "km" in m:
            return 40
        if m.endswith("_m"):
            return 30
        return 0

    best, best_score = None, 0
    for col in candidates:
        s = score(col[1])
        if s > best_score:
            best, best_score = col, s

    if best is None or best_score == 0:
        return None

    s = df.loc[:, best].astype(float)
    metric = best[1].lower()
    if metric.endswith("_m") or metric in {"distance_m", "dist_m", "m", "meter", "meters"}:
        return float(s.sum() / 1000.0)   # m → km
    return float(s.sum())                # km

def _find_driving_hours(df: pd.DataFrame, veh_id: str) -> Optional[float]:
    """
    Nutzt (veh_id, 'atbase') → Fahrzeit = sum((1 - atbase) * Δt).
    Gibt Stunden zurück oder None, wenn 'atbase' fehlt.
    """
    if df is None or not isinstance(df.columns, pd.MultiIndex):
        return None
    col = (veh_id, "atbase")
    if col not in df.columns:
        # häufige Aliasse versuchen
        aliases = [(veh_id, a) for a in ("at_base", "atDepot", "at_depot", "available", "is_at_base")]
        for c in aliases:
            if c in df.columns:
                col = c
                break
        else:
            return None
    atbase = df.loc[:, col].to_numpy(dtype=float)
    return float(((1.0 - atbase) * FREQ_HOURS).sum())

def build_vehicle_charger_map_from_settings(
    subfleet_settings: Dict[str, "SubFleetSettings"],
    phase: Literal["baseline", "expansion"],
) -> Dict[str, Dict[str, Optional[str]]]:
    """
    Erzeugt {vt: {veh_id: charger|None}} aus den Einstellungen:
      - baseline: BEV = num_bev_preexisting
      - expansion: BEV = num_bev_preexisting + num_bev_expansion
      - BEV → 'charger' (z.B. 'ac'/'dc'); ICEV → None
    """
    vehicle_charger: Dict[str, Dict[str, Optional[str]]] = {}
    for vt, sf in subfleet_settings.items():
        total = int(sf.num_total)
        bev_count = int(sf.num_bev_preexisting + (sf.num_bev_expansion if phase == "expansion" else 0))
        vt_map: Dict[str, Optional[str]] = {}
        for i in range(total):
            veh_id = f"{vt}{i}"
            vt_map[veh_id] = (sf.charger.lower() if i < bev_count else None)
        vehicle_charger[vt] = vt_map
    return vehicle_charger

def get_vehicle_usage_from_logs(
    df: Optional[pd.DataFrame],
    veh_id: str,
    avg_speed_kmh: float,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Liefert (distance_km, driving_h) für ein Fahrzeug aus dem Log:
      - distance_km: direkt aus Distanzspalte (inkl. Aliasse), sonst None
      - driving_h:   aus 'atbase' berechnet, sonst (wenn distance vorhanden) via distance/avg_speed_kmh,
                     sonst None
    """
    dist_km = _find_distance_km(df, veh_id) if df is not None else None
    drive_h = _find_driving_hours(df, veh_id) if df is not None else None

    if drive_h is None and dist_km is not None and avg_speed_kmh > 0:
        drive_h = dist_km / avg_speed_kmh

    return dist_km, drive_h


def calc_opex_vehicle(
    logs: "Logs",
    subfleet_settings: Dict[str, "SubFleetSettings"],
    economics: "EconomicSettings",
    phase: Literal["baseline", "expansion"],
    avg_speed_kmh: float = 50.0,
    vehicle_charger: Dict[str, Dict[str, Optional[str]]] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    OPEX-Berechnung auf Basis der Ladeplan-Logs.
    Erwartete Spalten je Fahrzeug (MultiIndex: (veh_id, metric)):
      - dist         : Distanz je Zeitschritt [km]
      - atbase       : True/False (oder 1/0, oder "True"/"False")
      - consumption  : (optional) Leistung [W] für BEV-Fallback -> Distanz via kWh/km
    ICEV benötigen eine Distanz (dist oder distance_km).
    Fahrzeit primär aus atbase, sonst aus Distanz/Ø-Geschwindigkeit.
    """
    # BEV/ICEV-Zuordnung aus Settings ableiten, falls nicht übergeben
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

    driver_wage     = float(economics.driver_wage_eur_h)
    mntex_bev       = float(economics.mntex_bev_eur_km)
    mntex_icev      = float(economics.mntex_icev_eur_km)
    toll_bev        = float(getattr(economics, "toll_bev_eur_km", 0.0))
    toll_icev       = float(economics.toll_icev_eur_km)
    insurance_rate  = float(economics.insurance_pct) / 100.0
    fuel_eur_per_km = float(getattr(economics, "fuel_eur_per_km", 0.0))  # ICEV only

    total_eur = 0.0
    breakdown: Dict[str, float] = {}

    for vt, sf in subfleet_settings.items():
        if vt not in vehicle_charger:
            raise ValueError(f"vehicle_charger missing subfleet '{vt}'.")

        df = logs.fleet.get(vt)  # MultiIndex: (veh_id, metric)
        if df is None or not isinstance(df.columns, pd.MultiIndex):
            raise ValueError(f"No valid logs for subfleet '{vt}' to derive distance/time.")

        n_total = int(sf.num_total)
        opex_vt = 0.0

        for i in range(n_total):
            veh_id = f"{vt}{i}"
            if veh_id not in vehicle_charger[vt]:
                raise ValueError(f"vehicle_charger['{vt}'] missing vehicle '{veh_id}'.")

            chg = vehicle_charger[vt][veh_id]
            is_bev = not (chg is None or str(chg).lower() == "none")  # None/"none" -> ICEV

            # ---------- Distanz (km) ----------
            distance_km = 0.0
            if (veh_id, "dist") in df.columns:
                distance_km = float(df.loc[:, (veh_id, "dist")].astype(float).sum())
            elif (veh_id, "distance_km") in df.columns:
                distance_km = float(df.loc[:, (veh_id, "distance_km")].astype(float).sum())
            else:
                # optionaler BEV-Fallback via consumption + kWh/km
                if is_bev and (veh_id, "consumption") in df.columns:
                    energy_kwh = float(
                        (df.loc[:, (veh_id, "consumption")].to_numpy(dtype=float) * FREQ_HOURS).sum() / 1000.0
                    )
                    kwh_per_km = float(getattr(sf, "kwh_per_km_bev", 0.0) or 0.0)
                    if kwh_per_km > 0:
                        distance_km = energy_kwh / kwh_per_km

            if not is_bev and distance_km <= 0.0:
                raise ValueError(
                    f"ICEV '{veh_id}' hat keine Distanzspalte. "
                    f"Erwarte (veh_id,'dist') oder (veh_id,'distance_km')."
                )

            # ---------- Fahrzeit (h) ----------
            if (veh_id, "atbase") in df.columns:
                atb = df.loc[:, (veh_id, "atbase")]
                # robust cast: bool → float; strings "True"/"False" → 1/0
                if atb.dtype == bool:
                    atbase = atb.astype(float).to_numpy()
                else:
                    try:
                        atbase = atb.astype(float).to_numpy()
                    except Exception:
                        atbase = atb.astype(str).str.lower().map({"true": 1.0, "false": 0.0}).fillna(0.0).to_numpy()
                driving_h = float(((1.0 - atbase) * FREQ_HOURS).sum())
            else:
                # Schätzung aus Distanz
                if distance_km > 0.0 and avg_speed_kmh > 0.0:
                    driving_h = distance_km / avg_speed_kmh
                else:
                    raise ValueError(
                        f"{veh_id}: Fahrzeit nicht bestimmbar (keine 'atbase' und keine Distanz)."
                    )

            # ---------- OPEX ----------
            if is_bev:
                opex_driver      = driver_wage * driving_h
                opex_maintenance = mntex_bev * distance_km
                opex_insurance   = float(sf.capex_bev_eur) * insurance_rate
                opex_fuel        = 0.0
                opex_toll        = toll_bev * distance_km
            else:
                opex_driver      = driver_wage * driving_h
                opex_maintenance = mntex_icev * distance_km
                opex_insurance   = float(sf.capex_icev_eur) * insurance_rate
                opex_fuel        = fuel_eur_per_km * distance_km
                opex_toll        = toll_icev * distance_km

            opex_vt += (opex_driver + opex_maintenance + opex_insurance + opex_fuel + opex_toll)

        breakdown[vt] = opex_vt
        total_eur += opex_vt

    return float(total_eur), breakdown



def calc_vehicle_capex_split(
    subfleet_settings: dict[str, "SubFleetSettings"],
    phase: Literal["baseline", "expansion"],
) -> Tuple[float, float, Dict[str, float], Dict[str, float]]:
    """
    Compute vehicle CAPEX split by powertrain (BEV vs ICEV) for a given phase.

    Phase semantics:
      - baseline:
          BEV count = num_bev_preexisting
          ICEV count = num_total - BEV count
      - expansion:
          BEV count = num_bev_preexisting + num_bev_expansion
          ICEV count = num_total - BEV count
        (Assumption: total fleet size stays constant; adjust if your fleet grows.)

    Returns:
      (bev_total_capex, icev_total_capex, bev_breakdown_by_subfleet, icev_breakdown_by_subfleet)
    """
    bev_total = 0.0
    icev_total = 0.0
    bd_bev: Dict[str, float] = {}
    bd_icev: Dict[str, float] = {}

    for vt, sf in subfleet_settings.items():
        # Determine BEV count in this phase (preexisting in baseline; preexisting+new in expansion)
        bev_phase = (
            int(sf.num_bev_preexisting)
            if phase == "baseline"
            else int(sf.num_bev_preexisting + sf.num_bev_expansion)
        )

        # Derive ICEV count in this phase as residual to total
        # (Clamp at 0 to avoid negative values if inputs are inconsistent)
        icev_phase = max(int(sf.num_total) - bev_phase, 0)

        # Subfleet CAPEX contributions
        capex_bev_sf  = float(bev_phase)  * float(getattr(sf, "capex_bev_eur",  0.0))
        capex_icev_sf = float(icev_phase) * float(getattr(sf, "capex_icev_eur", 0.0))

        # Store breakdown per subfleet key (vt)
        bd_bev[vt]  = capex_bev_sf
        bd_icev[vt] = capex_icev_sf

        # Accumulate totals
        bev_total  += capex_bev_sf
        icev_total += capex_icev_sf

    return float(bev_total), float(icev_total), bd_bev, bd_icev


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

    opex_vehicle_total, opex_vehicle_by_sf = calc_opex_vehicle(
        logs=logs,
        subfleet_settings=subfleet_settings,
        economics=economics,
        phase=phase,
    )

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

    if phase == "baseline":
        capex_infra_eur = 0.0
    else:
        rate_grid = getattr(economics, "capex_spec_grid_eur_per_w", 150)  # ToDo: check value
        rate_pv = getattr(economics, "capex_spec_pv_eur_per_wp", 900)  # ToDo: check value
        rate_ess = getattr(economics, "capex_spec_ess_eur_per_wh", 400)  # ToDo: check value

        grid_exp_kw = float(location.grid_capacity_w.expansion or 0.0)
        pv_exp_kwp = float(location.pv_capacity_wp.expansion or 0.0)
        ess_exp_kwh = float(location.ess_capacity_wh.expansion or 0.0)

        capex_grid = grid_exp_kw * rate_grid
        capex_pv = pv_exp_kwp * rate_pv
        capex_ess = ess_exp_kwh * rate_ess

        capex_chargers = float(sum(
            getattr(cs, "num_expansion", 0) * getattr(cs, "cost_per_charger_eur", 0.0)
            for cs in charger_settings.values()
        ))

        capex_infra_eur = float(capex_grid + capex_pv + capex_ess + capex_chargers)

    # --- Fahrzeug-CAPEX getrennt nach BEV/ICEV für die Phase ---
    capex_bev_eur, capex_icev_eur, _, _ = calc_vehicle_capex_split(
        subfleet_settings=subfleet_settings,
        phase=phase.lower()
    )
    capex_vehicles_eur = capex_bev_eur + capex_icev_eur

    # --- Gesamt-CAPEX ---
    capex_total_eur = capex_infra_eur + capex_vehicles_eur

    opex_grid_energy = (result_sim.energy_grid_buy_wh * economics.opex_spec_grid_buy_eur_per_wh -
                        result_sim.energy_grid_sell_wh * economics.opex_spec_grid_sell_eur_per_wh)

    opex_grid_power = result_sim.pwr_grid_peak_w * economics.opex_spec_grid_peak_eur_per_wp

    opex_grid = opex_grid_energy + opex_grid_power

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

    subfleet_sim_settings_baseline = {subfleet.name:
                                          subfleet.get_subfleet_sim_settings_baseline(settings.chargers)
                                      for subfleet in settings.subfleets.values()}

    subfleet_sim_settings_expansion = {subfleet.name:
                                           subfleet.get_subfleet_sim_settings_expansion(settings.chargers)
                                       for subfleet in settings.subfleets.values()}

    chargers_baseline = {t: c.num_preexisting for t, c in settings.chargers.items()}
    chargers_expansion = {t: c.num_preexisting + c.num_expansion for t, c in settings.chargers.items()}

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
