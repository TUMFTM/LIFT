from dataclasses import dataclass


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
class SubFleetResults:
    num_total: int = 0
    num_bev: int = 0
    num_bev_additional: int = 0
    num_icev: int = 0
    tco_bev: float = 0.0
    tco_icev: float = 0.0
    energy_bev_daily_kwh: float = 0.0


@dataclass
class SystemResults:
    pv_capacity_kwp: float = 0.0
    pv_energy_yrl_kwh: float = 0.0
    battery_capacity_kwh: float = 0.0
    grid_capacity_kw: float = 0.0
    num_dc_chargers: int = 0
    energy_daily_bevs_kwh: float = 0.0


@dataclass
class Results:
    subfleets: dict[str, SubFleetResults]
    system: SystemResults


