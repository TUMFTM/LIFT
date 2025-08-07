import numpy as np
from dataclasses import dataclass, field

from definitions import TIME_PRJ_YRS


@dataclass
class LocationSettings:
    latitude: float
    longitude: float
    consumption_annual: float
    grid_capacity_preexisting_kw: float
    grid_capacity_expansion_kw: float
    pv_capacity_preexisting_kwp: float
    pv_capacity_expansion_kwp: float
    ess_capacity_preexisting_kwh: float
    ess_capacity_expansion_kwh: float


@dataclass
class SubFleetSettings:
    vehicle_type: str
    num_total: int
    num_bev_preexisting: int
    num_bev_expansion: int
    battery_capacity_kwh: float
    capex_bev: float
    capex_icev: float
    dist_avg_daily_km: float
    toll_share_pct: float
    # dist_max_daily_km: float
    # depot_time_h: float
    # load_avg_t: float
    # weight_empty_bev_kg: float
    # weight_empty_icev_kg: float


@dataclass
class ChargerSettings:
    num_preexisting: int
    num_expansion: int
    pwr_max_kw: float
    cost_per_charger_eur: float


@dataclass
class EconomicSettings:
    period_holding_yrs: int
    electricity_price_eur_kwh: float
    fuel_price_eur_liter: float
    toll_icev_eur_km: float
    toll_bev_eur_km: float
    driver_wage_eur_h: float
    mntex_bev_eur_km: float
    mntex_icev_eur_km: float
    insurance_pct: float
    salvage_bev_pct: float
    salvage_icev_pct: float
    working_days_yrl: int


@dataclass
class Settings:
    location: LocationSettings
    subfleets: dict[str, SubFleetSettings]
    chargers: dict[str, ChargerSettings]
    economic: EconomicSettings


@dataclass
class SimulationResults:
    energy_pv_pot_wh: float
    energy_pv_curt_wh: float
    energy_grid_buy_wh: float
    energy_grid_sell_wh: float
    pwr_grid_peak_w: float


@dataclass
class PhaseResults:
    energy_pv_pot_yrl_wh: float = 0.0  # pv energy potential of the on-site PV per year in Wh
    energy_pv_curt_yrl_wh: float = 0.0  # pv energy of the on-site PV curtailed per year in Wh
    energy_grid_sell_yrl_wh: float = 0.0  # energy fed into the grid per year in Wh
    energy_grid_buy_yrl_wh: float = 0.0  # energy bought from the grid per year in Wh
    pwr_grid_peak_w: float = 0.0  # peak power of the grid (buying direction) in W
    fleet_yrl_wh: float = 0.0  # energy charged to the Fleet at the site per year in Wh
    self_sufficiency_pct: float = 0.0  # share of energy demand (fleet + site) which is satisfied by the PV (produced - fed in)
    self_consumption_pct: float = 0.0  # share of the energy produced by the on-site PV array which is consumed on-site (1 - feed-in / produced)
    co2_yrl_kg: float = 0.0  # emitted CO2 per year in kg
    co2_yrl_eur: float = 0.0  # cost for emitted co2 per year in Euro
    capex_eur: float = 0.0  # capex over project time in euro
    opex_fuel_eur: float = 0.0  # fuel cost over project time in euro
    opex_toll_eur: float = 0.0  # toll cost over project time in euro
    opex_grid_eur: float = 0.0  # grid cost over project time in euro
    cashflow: np.typing.NDArray[np.floating] = field(init=True,
                                                     default_factory=lambda: np.zeros(TIME_PRJ_YRS))
    # ToDo: think about using a DataFrame or Matrix for cashflows and replace all capex/opex variables by this

    @property
    def opex(self) -> float:
        return self.opex_fuel_eur + self.opex_grid_eur + self.opex_toll_eur


@dataclass
class Results:
    baseline: PhaseResults
    expansion: PhaseResults

    roi_rel: float
    period_payback_rel: float
    npc_delta: float


