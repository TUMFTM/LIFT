from dataclasses import dataclass


@dataclass
class LocationSettings:
    latitude: float
    longitude: float
    consumption_annual: float
    grid_capacity_kw: float
    pv_capacity_kwp: float
    battery_capacity_kwh: float


@dataclass
class SubFleetSettings:
    vehicle_type: str
    num_total: int
    num_bev: int
    battery_capacity_kwh: float
    dist_max_daily_km: float
    dist_avg_daily_km: float
    toll_share_pct: float
    depot_time_h: float
    load_avg_t: float
    capex_bev: float
    capex_icev: float
    weight_empty_bev_kg: float
    weight_empty_icev_kg: float


@dataclass
class ChargingInfrastructureSettings:
    num: int
    num_per_vehicle: float
    cost_per_charger_eur: float


@dataclass
class PhaseSettings:
    share_electric_total_pct: float
    share_electric_subfleets_pct: dict[str, float]
    package: str


@dataclass
class ElectrificationPhasesSettings:
    num: int
    phases: list[PhaseSettings]


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
    charging_infrastructure: ChargingInfrastructureSettings
    electrification_phases: ElectrificationPhasesSettings
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


