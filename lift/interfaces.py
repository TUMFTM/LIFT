from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from definitions import TIME_PRJ_YRS


class GridPowerExceededError(Exception):
    pass

class SOCError(Exception):
    pass

@dataclass
class Coordinates:
    latitude: float = 48.148
    longitude: float = 11.507

@dataclass
class Size:
    preexisting: float
    expansion: float

    @property
    def total(self) -> float:
        return self.preexisting + self.expansion


@dataclass
class Logs:
    pv_spec: np.typing.NDArray[np.floating]
    dem: np.typing.NDArray[np.floating]
    fleet: pd.DataFrame


@dataclass
class Capacities:
    grid_w: float = 0.0
    pv_wp: float = 0.0
    ess_wh: float = 0.0


@dataclass
class LocationSettings:
    coordinates: Coordinates = field(default_factory=Coordinates)

    slp: str = 'h0'
    consumption_yrl_wh: float = 10000000.0

    grid_capacity_w: Size = field(default_factory=lambda: Size(preexisting=10E3,
                                                               expansion=50E3))

    pv_capacity_wp: Size = field(default_factory=lambda: Size(preexisting=10E3,
                                                              expansion=20E3))

    ess_capacity_wh: Size = field(default_factory=lambda: Size(preexisting=0E3,
                                                               expansion=50E3))

    def get_capacities(self,
                       phase: str) -> Capacities:
        if phase not in ['baseline', 'expansion']:
            raise ValueError(f"Invalid phase: {phase}. Must be 'preexisting' or 'expansion'.")

        attr_name = 'preexisting' if phase == 'baseline' else 'total'

        return Capacities(grid_w=getattr(self.grid_capacity_w, attr_name),
                          pv_wp=getattr(self.pv_capacity_wp, attr_name),
                          ess_wh=getattr(self.ess_capacity_wh, attr_name),
                          )


@dataclass
class SubFleetSettings:
    vehicle_type: str = 'default'
    num_total: int = 5
    num_bev_preexisting: int = 0
    num_bev_expansion: int = 4
    battery_capacity_kwh: float = 80E3
    capex_bev_eur: float = 100E3
    capex_icev_eur: float = 80E3
    dist_avg_daily_km: float = 100.0
    toll_share_pct: float = 30.0
    charger: str = 'ac'
    pwr_max_w: float = 0.0

    # dist_max_daily_km: float
    # depot_time_h: float
    # load_avg_t: float
    # weight_empty_bev_kg: float
    # weight_empty_icev_kg: float


@dataclass
class ChargerSettings:
    num_preexisting: int = 0
    num_expansion: int = 4
    pwr_max_kw: float = 11.0
    cost_per_charger_eur: float = 3000.0


@dataclass
class EconomicSettings:
    electricity_price_eur_wh: float = 30E5
    fuel_price_eur_liter: float = 1.7
    toll_icev_eur_km: float = 0.1
    toll_bev_eur_km: float = 0.0
    driver_wage_eur_h: float = 20.0
    mntex_bev_eur_km: float = 0.05
    mntex_icev_eur_km: float = 0.1
    insurance_pct: float = 20.0
    salvage_bev_pct: float = 40.0
    salvage_icev_pct: float = 40.0
    working_days_yrl: int = 220


@dataclass
class Settings:
    location: LocationSettings = field(default_factory=LocationSettings)
    subfleets: dict[str, SubFleetSettings] = field(default_factory=lambda: dict(bev=SubFleetSettings(),))
    chargers: dict[str, ChargerSettings] = field(default_factory=lambda: dict(bev=ChargerSettings(),))
    economic: EconomicSettings = field(default_factory=EconomicSettings)


@dataclass
class SimulationResults:
    energy_pv_pot_wh: float = 0.0
    energy_pv_curt_wh: float = 0.0
    energy_grid_buy_wh: float = 0.0
    energy_grid_sell_wh: float = 0.0
    pwr_grid_peak_w: float = 0.0
    energy_fleet_wh: float = 0.0
    energy_dem_wh: float = 0.0


@dataclass
class PhaseResults:
    simulation : SimulationResults = field(default_factory=SimulationResults)
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
class BackendResults:
    baseline: PhaseResults
    expansion: PhaseResults

    roi_rel: float = 0.0
    period_payback_rel: float = 0.0
    npc_delta: float = 0.0
