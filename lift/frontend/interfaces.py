from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Self

import streamlit as st


class SettingsInput(ABC):
    @abstractmethod
    def get_streamlit_element(self, label: str, key: Optional[str] = None, domain=st) -> Any:
        """
        Return a streamlit input element
        :param label:
        :param key:
        :param domain:
        :return:
        """
        ...


@dataclass
class SettingsNumeric(SettingsInput):
    max_value: float
    value: float
    min_value: float = 0.0
    format: str = "%.0f"
    factor: float = 1.0


@dataclass
class SettingsSlider(SettingsNumeric):
    step: float = 1.0

    def get_streamlit_element(self, label: str, key: Optional[str] = None, domain=st) -> Any:
        return (
            domain.slider(
                label=label,
                key=key,
                min_value=self.min_value,
                max_value=self.max_value,
                value=self.value,
                format=self.format,
                step=self.step,
            )
            * self.factor
        )


@dataclass
class SettingsNumberInput(SettingsNumeric):
    def get_streamlit_element(self, label: str, key: Optional[str] = None, domain=st):
        return (
            domain.number_input(
                label=label,
                key=key,
                min_value=self.min_value,
                max_value=self.max_value,
                value=self.value,
                format=self.format,
            )
            * self.factor
        )


@dataclass
class SettingsSelectBox(SettingsInput):
    options: list[str]
    index: int

    def get_streamlit_element(self, label: str, key: Optional[str] = None, domain=st):
        return domain.selectbox(
            label=label,
            key=key,
            options=self.options,
            index=self.index,
        )


class FrontendInterface(ABC):
    @classmethod
    @abstractmethod
    def from_dict(cls, dict_def) -> Self: ...


@dataclass
class FrontendBlockInterface(FrontendInterface, ABC):
    name: str
    label: str
    icon: str
    ls: int


@dataclass
class FrontendEnergyBlockInterface(FrontendBlockInterface, ABC):
    settings_preexisting: SettingsNumberInput
    settings_expansion: SettingsSlider


@dataclass
class FrontendChargerInterface(FrontendEnergyBlockInterface):
    settings_pwr_max: SettingsSlider
    settings_cost_per_unit_eur: SettingsSlider
    capem: float

    @classmethod
    def from_dict(cls, dict_def) -> Self:
        settings = {
            "settings_pwr_max": SettingsSlider(**dict_def["settings_pwr_max"], factor=1e3),
            "settings_preexisting": SettingsNumberInput(**dict_def["settings_preexisting"]),
            "settings_expansion": SettingsSlider(**dict_def["settings_expansion"], step=1.0),
            "settings_cost_per_unit_eur": SettingsSlider(**dict_def["settings_cost_per_unit_eur"]),
        }
        return cls(**{key: dict_def[key] for key in dict_def if key not in settings}, **settings)


@dataclass
class FrontendSubFleetInterface(FrontendBlockInterface):
    weight_max_t: float
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
    settings_toll_share: SettingsSlider
    settings_capex_bev: SettingsSlider
    settings_capex_icev: SettingsSlider

    @classmethod
    def from_dict(cls, dict_def) -> Self:
        settings = {
            "settings_toll_share": SettingsSlider(**dict_def["settings_toll_share"], factor=1e-2),
            "settings_capex_bev": SettingsSlider(**dict_def["settings_capex_bev"], step=1000.0),
            "settings_capex_icev": SettingsSlider(**dict_def["settings_capex_icev"], step=1000.0),
        }
        return cls(**{key: dict_def[key] for key in dict_def if key not in settings}, **settings)


@dataclass
class FrontendSizableBlockInterface(FrontendEnergyBlockInterface):
    """
    Use to model:
    - Grid connection
    - Photovoltaic array (PV)
    - Battery electric stationary storage (BESS)
    """

    capex_spec: float
    capem_spec: float

    @classmethod
    def from_dict(cls, dict_def) -> Self:
        settings = {
            "settings_preexisting": SettingsNumberInput(**dict_def["settings_preexisting"], factor=1e3),
            "settings_expansion": SettingsSlider(**dict_def["settings_expansion"], factor=1e3),
        }
        return cls(**{key: dict_def[key] for key in dict_def if key not in settings}, **settings)

    @property
    def input_dict(self) -> Dict:
        return {
            "capex_spec": self.capex_spec,
            "capem_spec": self.capem_spec,
            "ls": self.ls,
        }


@dataclass
class FrontendDemandInterface:
    settings_dem_profile: SettingsSelectBox
    settings_dem_yr: SettingsSlider

    @classmethod
    def from_parameters(
        cls,
        options: List,
        options_default_index: int,
        max_value: float,
        value: float,
        step: float,
    ) -> Self:
        return cls(
            settings_dem_profile=SettingsSelectBox(options=options, index=options_default_index),
            settings_dem_yr=SettingsSlider(min_value=0.0, max_value=max_value, value=value, step=step, factor=1e6),
        )


@dataclass
class FrontendEconomicsInterface:
    settings_discount_rate: SettingsSlider
    settings_fix_cost_construction: SettingsSlider
    settings_opex_spec_grid_buy: SettingsSlider
    settings_opex_spec_grid_sell: SettingsSlider
    settings_opex_spec_grid_peak: SettingsSlider
    settings_opex_spec_route_charging: SettingsSlider
    settings_opex_fuel: SettingsSlider
    settings_insurance_frac: SettingsSlider
    settings_salvage_bev_frac: SettingsSlider
    settings_salvage_icev_frac: SettingsSlider
