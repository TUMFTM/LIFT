from dataclasses import dataclass


@dataclass
class LocationSettings:
    latitude: float
    longitude: float
    consumption_annual: float
    grid_capacity: float
    pv_capacity: float
    battery_capacity: float


@dataclass
class SubFleetSettings:
    vehicle_type: str
    num_total: int
    num_bev: int
    battery_capacity_kwh: float
    dist_max_km: float
    dist_avg_km: float
    toll_share_pct: float
    depot_time_h: float
    load_avg_t: float
    capex_bev: float
    capex_icev: float
    weight_empty_bev: float
    weight_empty_icev: float


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
    bev_mntex_eur_km: float
    icev_mntex_eur_km: float
    insurance_pct: float
    bev_salvage_pct: float
    icev_salvage_pct: float
    workingdays_per_year: int


@dataclass
class TcoResults:
    dummy: float


@dataclass
class VariousResults:
    dummy: float

