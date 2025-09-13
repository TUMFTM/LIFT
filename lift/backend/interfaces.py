from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from typing import Literal, Optional

import streamlit as st


class GridPowerExceededError(Exception):
    pass

class SOCError(Exception):
    pass


class SettingsInput(ABC):
    @abstractmethod
    def get_input(self,
                  label: str,
                  key: Optional[str] = None,
                  domain=st):
        ...


@dataclass
class SettingsNumeric(SettingsInput):
    min_value: float
    max_value: float
    value: float
    format: str = "%d"
    factor: float = 1.0


@dataclass
class SettingsSlider(SettingsNumeric):
    step: float = 1.0

    def get_input(self,
                  label: str,
                  key: Optional[str] = None,
                  domain=st):
        return domain.slider(label=label,
                             key=key,
                             min_value=self.min_value,
                             max_value=self.max_value,
                             value=self.value,
                             format=self.format,
                             step=self.step,
                             ) * self.factor


@dataclass
class SettingsNumberInput(SettingsNumeric):
    def get_input(self,
                  label: str,
                  key: Optional[str] = None,
                  domain=st):
        return domain.number_input(label=label,
                                   key=key,
                                   min_value=self.min_value,
                                   max_value=self.max_value,
                                   value=self.value,
                                   format=self.format,
                                   ) * self.factor


@dataclass
class SettingsSelectBox(SettingsInput):
    options: list[str]
    index: int

    def get_input(self,
                  label: str,
                  key: Optional[str] = None,
                  domain=st):
        return domain.selectbox(label=label,
                                key=key,
                                options=self.options,
                                index=self.index,
                                )


@dataclass
class DefinitionSubfleet:
    label: str
    icon: str
    name: str
    weight_max_str: str
    battery_capactiy_wh: float
    capem_bev: float
    capem_icev: float
    weight_empty_bev: float
    weight_empty_icev: float
    toll_eur_per_km_bev: float
    toll_eur_per_km_icev: float
    mntex_eur_km_bev: float
    mntex_eur_km_icev: float
    consumption_icev: float
    ls: float
    settings_toll_share: SettingsSlider
    settings_capex_bev: SettingsSlider
    settings_capex_icev: SettingsSlider


@dataclass
class DefinitionExpansion:
    name: str
    icon: str
    settings_preexisting: SettingsNumberInput
    settings_expansion: SettingsSlider
    settings_cost_per_unit_eur: SettingsSlider
    capem: float


@dataclass
class DefinitionCharger(DefinitionExpansion):
    settings_pwr_max: SettingsSlider
    ls: float


@dataclass
class DefinitionEnergySystem:
    settings_dem_profile: SettingsSelectBox
    settings_dem_yr: SettingsSlider
    settings_grid_preexisting: SettingsNumberInput
    settings_grid_expansion: SettingsSlider
    settings_pv_preexisting: SettingsNumberInput
    settings_pv_expansion: SettingsSlider
    settings_ess_preexisting: SettingsNumberInput
    settings_ess_expansion: SettingsSlider


@dataclass
class DefinitionEconomics:
    settings_fix_cost_construction: SettingsSlider
    settings_opex_spec_grid_buy: SettingsSlider
    settings_opex_spec_grid_sell: SettingsSlider
    settings_opex_spec_grid_peak: SettingsSlider
    settings_opex_spec_route_charging: SettingsSlider
    settings_opex_fuel: SettingsSlider
    settings_insurance_frac: SettingsSlider
    settings_salvage_bev_frac: SettingsSlider
    settings_salvage_icev_frac: SettingsSlider


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

    def get_value(self,
                  phase: Literal['baseline', 'expansion']) -> float:
        return self.preexisting if phase == 'baseline' else self.total


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
        attr_name = 'preexisting' if phase == 'baseline' else 'total'

        return Capacities(grid_w=getattr(self.grid_capacity_w, attr_name),
                          pv_wp=getattr(self.pv_capacity_wp, attr_name),
                          ess_wh=getattr(self.ess_capacity_wh, attr_name),
                          )


@dataclass
class SimInputSubfleet:
    name: str = 'hlt'
    num: int = 1
    pwr_chg_max_w: float = 11E3
    charger: str = 'ac'
    capacity_wh: float = 80E3


@dataclass
class PhaseInputSubfleet:
    name: str = 'hlt'
    num_total: int = 5
    num_bev: int = 1
    battery_capacity_wh: float = 80E3
    capex_bev_eur: float = 100E3
    capex_icev_eur: float = 80E3
    toll_frac: float = 0.3
    charger: str = 'ac'
    pwr_max_w: float = 11E3

    def get_sim_input(self) -> SimInputSubfleet:
        return SimInputSubfleet(name=self.name,
                                num=self.num_bev,
                                capacity_wh=self.battery_capacity_wh,
                                charger=self.charger,
                                pwr_chg_max_w=self.pwr_max_w,
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
    toll_frac: float = 0.3
    charger: str = 'ac'
    pwr_max_w: float = 11E3

    def get_phase_input(self,
                      phase: Literal['baseline', 'expansion']) -> PhaseInputSubfleet:
        return PhaseInputSubfleet(name=self.name,
                                  num_total=self.num_total,
                                  num_bev=int(self.num_bev.preexisting) if phase == 'baseline' else int(self.num_bev.total),
                                  battery_capacity_wh=self.battery_capacity_wh,
                                  capex_bev_eur=self.capex_bev_eur,
                                  capex_icev_eur=self.capex_icev_eur,
                                  toll_frac=self.toll_frac,
                                  charger=self.charger,
                                  pwr_max_w=self.pwr_max_w,
                                  )


@dataclass
class SimInputCharger:
    name: str = 'ac'
    num: int = 0
    pwr_max_w: float = 11E3


@dataclass
class PhaseInputCharger:
    name: str = 'ac'
    num: int = 0
    pwr_max_w: float = 11E3
    cost_per_charger_eur: float = 3000.0

    def get_sim_input(self) -> SimInputCharger:
        return SimInputCharger(name=self.name,
                               num=self.num,
                               pwr_max_w=self.pwr_max_w)

@dataclass
class InputCharger:
    name: str = 'ac'
    num: ExistExpansionValue = field(default_factory=lambda: ExistExpansionValue(preexisting=0,
                                                                                 expansion=4))
    pwr_max_w: float = 11E3
    cost_per_charger_eur: float = 3000.0

    def get_phase_input(self,
                        phase: Literal['baseline', 'expansion']) -> PhaseInputCharger:
        return PhaseInputCharger(name=self.name,
                                 num=int(self.num.preexisting) if phase == 'baseline' else int(self.num.total),
                                 pwr_max_w=self.pwr_max_w,
                                 cost_per_charger_eur=self.cost_per_charger_eur,
                                 )

@dataclass
class PhaseInputEconomic:
    opex_spec_grid_buy: float = 30E-5
    opex_spec_grid_sell: float = -6E-5
    opex_spec_grid_peak: float = 150E-3
    opex_spec_route_charging: float = 49E-5
    opex_fuel: float = 1.7
    mntex_bev_eur_km: float = 0.05
    mntex_icev_eur_km: float = 0.1
    insurance_frac: float = 0.02
    salvage_bev_frac: float = 40.0
    salvage_icev_frac: float = 40.0
    fix_cost_construction: float = 10000


@dataclass
class InputEconomic:
    opex_spec_grid_buy: float = 30E-5
    opex_spec_grid_sell: float = -6E-5
    opex_spec_grid_peak: float = 150E-3
    opex_spec_route_charging: float = 49E-5
    opex_fuel: float = 1.7
    insurance_frac: float = 0.02
    salvage_bev_frac: float = 40.0
    salvage_icev_frac: float = 40.0
    fix_cost_construction: float = 10000

    def get_phase_input(self,
                        phase: Literal['baseline', 'expansion']) -> 'PhaseInputEconomic':
        return PhaseInputEconomic(
            opex_spec_grid_buy=self.opex_spec_grid_buy,
            opex_spec_grid_sell=self.opex_spec_grid_sell,
            opex_spec_grid_peak=self.opex_spec_grid_peak,
            opex_spec_route_charging=self.opex_spec_route_charging,
            opex_fuel=self.opex_fuel,
            insurance_frac=self.insurance_frac,
            salvage_bev_frac=self.salvage_bev_frac,
            salvage_icev_frac=self.salvage_icev_frac,
            fix_cost_construction=self.fix_cost_construction if phase == 'expansion' else 0,
        )

@dataclass
class Inputs:
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
class SimResults:
    energy_pv_pot_wh: float = 0.0
    energy_pv_curt_wh: float = 0.0
    energy_grid_buy_wh: float = 0.0
    energy_grid_sell_wh: float = 0.0
    pwr_grid_peak_w: float = 0.0
    energy_dem_site_wh: float = 0.0
    energy_fleet_site_wh: float = 0.0
    energy_fleet_route_wh: float = 0.0


@dataclass
class PhaseResults:
    simulation : SimResults = field(default_factory=SimResults)
    self_sufficiency: float = 0.0  # share of energy demand (fleet + site) which is satisfied by the PV (produced - fed in)
    self_consumption: float = 0.0  # share of the energy produced by the on-site PV array which is consumed on-site (1 - feed-in / produced)
    site_charging: float = 0.0  # share of the fleet energy demand which is charged on-site (vs on-route)
    cashflow: np.typing.NDArray[np.floating] = field(init=True,
                                                     default_factory=lambda: np.zeros(18))
    co2_flow: np.typing.NDArray[np.floating] = field(init=True,
                                                     default_factory=lambda: np.zeros(18))


@dataclass
class TotalResults:
    baseline: PhaseResults
    expansion: PhaseResults

    @property
    def npc_delta(self) -> float:
        return self.baseline.cashflow.sum() - self.expansion.cashflow.sum()

    @property
    def payback_period_yrs(self) -> Optional[float]:
        diff = np.cumsum(self.baseline.cashflow) - np.cumsum(self.expansion.cashflow)
        idx = np.flatnonzero(np.diff(np.sign(diff)))

        if idx.size == 0 or diff[0] > 0:
            return None  # No intersection

        i = idx[0]
        y0, y1 = diff[i], diff[i + 1]

        # Linear interpolation to find x where y1 == y2
        return float((i - y0 / (y1 - y0)) + 1)

    @property
    def roi_rel(self) -> float:
        # ToDo: calculate!
        return 0.0
