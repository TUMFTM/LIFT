from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from definitions import TIME_PRJ_YRS
from typing import Dict, Tuple, List, Literal, Final


class GridPowerExceededError(Exception):
    pass

class SOCError(Exception):
    pass


@dataclass
class SubfleetSimSettings:
    name: str = 'hlt'
    num: int = 1
    pwr_chg_max_w: float = 11E3
    charger: str = 'ac'
    capacity_wh: float = 80E3


@dataclass
class Coordinates:
    latitude: float = 48.148
    longitude: float = 11.507

    @property
    def as_tuple(self) -> tuple[float, float]:
        return self.latitude, self.longitude

    @staticmethod
    def _decimal_to_dms(decimal_deg: float) -> tuple[int, int, float]:
        degrees = int(abs(decimal_deg))
        minutes_full = (abs(decimal_deg) - degrees) * 60
        minutes = int(minutes_full)
        seconds = (minutes_full - minutes) * 60
        return degrees, minutes, seconds

    @property
    def as_dms_str(self) -> str:
        lat_deg, lat_min, lat_sec = self._decimal_to_dms(self.latitude)
        lon_deg, lon_min, lon_sec = self._decimal_to_dms(self.longitude)

        return (f"{lat_deg}°{lat_min}'{lat_sec:.2f}'' {'N' if self.latitude >= 0 else 'S'}, "
                f"{lon_deg}°{lon_min}'{lon_sec:.2f}'' {'E' if self.longitude >= 0 else 'W'}")


@dataclass
class ExistExpansionValue:
    preexisting: float
    expansion: float

    @property
    def total(self) -> float:
        return self.preexisting + self.expansion


@dataclass
class Capacities:
    grid_w: float = 0.0
    pv_wp: float = 0.0
    ess_wh: float = 0.0


@dataclass
class InputLocation:
    coordinates: Coordinates = field(default_factory=Coordinates)

    slp: str = 'h0'
    consumption_yrl_wh: float = 10000000.0

    grid_capacity_w: ExistExpansionValue = field(default_factory=lambda: ExistExpansionValue(preexisting=10E3,
                                                                                             expansion=50E3))

    pv_capacity_wp: ExistExpansionValue = field(default_factory=lambda: ExistExpansionValue(preexisting=10E3,
                                                                                            expansion=20E3))

    ess_capacity_wh: ExistExpansionValue = field(default_factory=lambda: ExistExpansionValue(preexisting=0E3,
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
class InputSubfleet:
    name: str = 'hlt'
    num_total: int = 5
    num_bev: ExistExpansionValue = field(default_factory=lambda: ExistExpansionValue(preexisting=1,
                                                                                     expansion=4))
    battery_capacity_wh: float = 80E3
    capex_bev_eur: float = 100E3
    capex_icev_eur: float = 80E3
    dist_avg_daily_km: float = 100.0
    toll_share_pct: float = 30.0
    charger: str = 'ac'
    pwr_max_w: float = 11E3

    def get_subfleet_sim_settings_baseline(self,
                                           charger_settings: dict[str, 'InputCharger']) -> SubfleetSimSettings:
        key = str(self.charger).strip().lower()
        return SubfleetSimSettings(name=self.name,
                                   num=int(self.num_bev.preexisting),
                                   pwr_chg_max_w=min(self.pwr_max_w, charger_settings[key].pwr_max_w),
                                   charger=key,
                                   capacity_wh=self.battery_capacity_wh)

    def get_subfleet_sim_settings_expansion(self,
                                            charger_settings: dict[str, 'InputCharger']) -> SubfleetSimSettings:
        key = str(self.charger).strip().lower()
        return SubfleetSimSettings(name=self.name,
                                   num=int(self.num_bev.total),
                                   pwr_chg_max_w=min(self.pwr_max_w, charger_settings[key].pwr_max_w),
                                   charger=key,
                                   capacity_wh=self.battery_capacity_wh)


@dataclass
class InputCharger:
    name: str = 'ac'
    num: ExistExpansionValue = field(default_factory=lambda: ExistExpansionValue(preexisting=0,
                                                                                 expansion=4))
    pwr_max_w: float = 11E3
    cost_per_charger_eur: float = 3000.0


@dataclass
class InputEconomic:
    opex_spec_grid_buy_eur_per_wh: float = 30E-5
    opex_spec_grid_sell_eur_per_wh: float = -6E-5
    opex_spec_grid_peak_eur_per_wp: float = 150E-3
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
    fix_cost_construction: int = 10000

@dataclass
class Input:
    location: InputLocation = field(default_factory=InputLocation)
    subfleets: dict[str, InputSubfleet] = field(default_factory=lambda: dict(hlt=InputSubfleet(name='hlt'),
                                                                             hst=InputSubfleet(name='hst'), ))
    chargers: dict[str, InputCharger] = field(default_factory=lambda: dict(ac=InputCharger(name='ac'),
                                                                           dc=InputCharger(name='dc'), ))
    economic: InputEconomic = field(default_factory=InputEconomic)


@dataclass
class Logs:
    pv_spec: np.typing.NDArray[np.floating]
    dem: np.typing.NDArray[np.floating]
    fleet: dict[str, pd.DataFrame]


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
    co2_grid_yrl_kg: float = 0.0
    co2_tailpipe_yrl_kg: float = 0.0
    co2_tailpipe_by_subfleet_kg: Dict[str, float] = field(default_factory=dict)
    capex_eur: float = 0.0  # capex over project time in euro
    capex_infra_eur: float = 0.0
    capex_vehicles_eur: float = 0.0
    capex_vehicles_bev_eur: float = 0.0
    capex_vehicles_icev_eur: float = 0.0
    vehicles_co2_production_total_kg: float = 0.0
    vehicles_co2_production_bev_kg: float = 0.0
    vehicles_co2_production_icev_kg: float = 0.0
    vehicles_co2_production_breakdown_bev: Dict[str, float] = field(default_factory=dict)
    vehicles_co2_production_breakdown_icev: Dict[str, float] = field(default_factory=dict)
    opex_fuel_eur: float = 0.0  # fuel cost over project time in euro
    opex_toll_eur: float = 0.0  # toll cost over project time in euro
    opex_grid_eur: float = 0.0  # grid cost over project time in euro
    opex_vehicle_electric_secondary: float = 0.0 # maintenance, insurance, driver
    infra_capex_breakdown: Dict[str, float] = field(default_factory=dict)
    infra_co2_total_kg: float = 0.0
    infra_co2_breakdown: Dict[str, float] = field(default_factory=dict)
    cashflow: np.typing.NDArray[np.floating] = field(init=True,
                                                     default_factory=lambda: np.zeros(TIME_PRJ_YRS))
    co2_flow: np.typing.NDArray[np.floating] = field(init=True,
                                                     default_factory=lambda: np.zeros(TIME_PRJ_YRS))
    opex_breakdown: Dict[str, float] = field(default_factory=dict)
    capex_vehicles_by_subfleet: Dict[str, float] = field(default_factory=dict)
    # ToDo: think about using a DataFrame or Matrix for cashflows and replace all capex/opex variables by this

    @property
    def opex_eur(self) -> float:
        return self.opex_fuel_eur + self.opex_grid_eur + self.opex_toll_eur + self.opex_vehicle_electric_secondary


@dataclass
class BackendResults:
    baseline: PhaseResults
    expansion: PhaseResults

    roi_rel: float = 0.0
    period_payback_rel: float = 0.0
    npc_delta: float = 0.0

TIME_PRJ_YRS: Final[int] = 18

@dataclass(frozen=True)
class DefaultLocation:
    consumption_building_yrl_mwh: int = 26 # jährlicher Gebäude-Stromverbrauch [Mh]
    slp: str = 'G0'  # Loadprofile
    grid_connection_kwh: int = 1000
    existing_pv_kwp: int = 100
    existing_ess_kwh: int = 100

@dataclass(frozen=True)
class DefaultEconomics:
    salvage_bev_pct: int = 29  # in %
    salvage_icev_pct: int = 26
    opex_spec_grid_buy_eur_per_wh: float = 0.20
    opex_spec_grid_sell_eur_per_wh: float = 0.10
    opex_spec_grid_peak_eur_per_wp: int = 150
    mntex_bev_eur_km: float = 0.13
    mntex_icev_eur_km: float = 0.18
    toll_bev_eur_km: float = 0.0
    toll_icev_eur_km: float = 0.269
    insurance_pct: float = 2.0
    fuel_price_eur_liter: float = 1.56

@dataclass(frozen=True)
class DefaultChargers:
    # Beispiel: Stückkosten Defaults
    ac_cost_per_unit_eur: float = 1000.0
    dc_cost_per_unit_eur: float = 80000.0

@dataclass(frozen=True)
class DefaultValues:
    location: DefaultLocation = DefaultLocation()
    economics: DefaultEconomics = DefaultEconomics()
    chargers: DefaultChargers = DefaultChargers()

# Eine einzige, zentrale Instanz:
DEFAULTS = DefaultValues()
