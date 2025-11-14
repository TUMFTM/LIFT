"""Frontend interface definitions for Streamlit inputs.

Purpose:
- Provide reusable abstractions (slider/number/select) with Streamlit bindings and
  domain-specific frontend interfaces for demand, energy blocks, chargers, and subfleets.

Relationships:
- Backed by `frontend/definitions.py`, which instantiates these interfaces from JSON defaults.
- Bridges user selections to backend comparison inputs via `frontend/sidebar.py`.
- Extends backend coordinates to include localized formatting.

Key Logic:
- `Settings*` classes encapsulate Streamlit widgets and apply scaling factors (e.g., kW→W).
- `Frontend*Interface` classes parse JSON definitions (`from_dict`) into rich objects with icons, labels, and widget configs.
- `FrontendCoordinates` inherits simulation coordinates and adds DMS string formatting with localized labels.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal, Self

import streamlit as st

from .utils import get_label

from lift.backend.simulation.interfaces import Coordinates


class StreamlitWrapper:
    def __init__(self, element=None):
        self.element = element
        self.elements = dict()

    def __getattr__(self, name):
        if name in self.elements:
            return self.elements[name]
        else:
            raise AttributeError(f"Attribute {name} is not defined in {self}")

    def __setattr__(self, name, value):
        if name in ("element", "elements"):
            # Internal attributes. Need to be excluded otherwise initialization fails
            super().__setattr__(name, value)
        else:
            self.elements[name] = StreamlitWrapper(element=value)

    def __call__(self, *args, **kwargs):
        return self.element


class SettingsInput(ABC):
    @abstractmethod
    def get_streamlit_element(
        self, label: str, help_msg: str | None = None, key: str | None = None, domain=st
    ) -> Any: ...


@dataclass
class SettingsNumeric(SettingsInput, ABC):
    max_value: float
    value: float
    min_value: float = 0.0
    format: str = "%.0f"
    factor: float = 1.0


@dataclass
class SettingsSlider(SettingsNumeric):
    step: float = 1.0

    def get_streamlit_element(self, label: str, help_msg: str | None = None, key: str | None = None, domain=st) -> Any:
        domain.slider(
            label=label,
            help=help_msg,
            key=key,
            min_value=self.min_value,
            max_value=self.max_value,
            value=st.session_state.get(key, self.value),
            format=self.format,
            step=self.step,
        )


@dataclass
class SettingsNumberInput(SettingsNumeric):
    def get_streamlit_element(self, label: str, help_msg: str | None = None, key: str | None = None, domain=st):
        domain.number_input(
            label=label,
            help=help_msg,
            key=key,
            min_value=self.min_value,
            max_value=self.max_value,
            value=st.session_state.get(key, self.value),
            format=self.format,
        )


@dataclass
class SettingsSelectBox(SettingsInput):
    options: list[str]
    value: str

    def get_streamlit_element(self, label: str, help_msg: str | None = None, key: str | None = None, domain=st):
        return domain.selectbox(
            label=label,
            help=help_msg,
            key=key,
            options=self.options,
            index=self.options.index(st.session_state.get(key, self.value)),
        )


class FrontendInterface(ABC):
    @classmethod
    @abstractmethod
    def from_dict(cls, dict_def) -> Self: ...


@dataclass
class FrontendBlockInterface(FrontendInterface, ABC):
    name: str
    label: str | dict[str, str]  # use dict to support different languages
    icon: str
    ls: int

    def get_label(self, language: str):
        """
        Retrieve the label text for the specified language, with fallbacks.

        If the label is stored as a string, it is returned directly.
        If the label is a dictionary of language codes, the method returns the label for the provided language.
        If this language is not available, the method provides a fallback value (first "en", if not available "de").
        first available translation based on a defined preference order.

        Args:
            language (str): The desired language code for the label.

        Returns:
            str | None: The corresponding label text, or ``None`` if no matching
            translation is found.

        Raises:
            ValueError: If ``self.label`` is neither a string nor a dictionary or an empty dictionary.
        """
        if isinstance(self.label, str):
            return self.label
        elif self.label and isinstance(self.label, dict):
            preferred_languages = (language, "en", "de")
            return next((self.label[k] for k in preferred_languages if k in self.label), None)

        else:
            raise ValueError(
                f"Label of type {type(self.label).__name__} is not supported."
                f"Please provide the label as string or a dict of strings to support different languages."
            )


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

    @property
    def input_dict(self):
        return {
            k: getattr(self, k, None)
            for k in [
                "ls",
                "capem",
            ]
        }


@dataclass
class FrontendSubFleetInterface(FrontendBlockInterface):
    weight_max_t: float
    battery_capacity_wh: float
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

    @property
    def input_dict(self):
        return {
            k: getattr(self, k, None)
            for k in [
                "battery_capacity_wh",
                "ls",
                "capem_bev",
                "capem_icev",
                "mntex_eur_km_bev",
                "mntex_eur_km_icev",
                "consumption_icev",
                "toll_eur_per_km_bev",
                "toll_eur_per_km_icev",
            ]
        }


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
    def input_dict(self) -> dict:
        return {
            k: getattr(self, k, None)
            for k in [
                "capex_spec",
                "capem_spec",
                "ls",
            ]
        }


@dataclass
class FrontendDemandInterface:
    settings_dem_profile: SettingsSelectBox
    settings_dem_yr: SettingsSlider

    @classmethod
    def from_parameters(
        cls,
        options: list,
        options_default: str,
        max_value: float,
        value: float,
        step: float,
    ) -> Self:
        return cls(
            settings_dem_profile=SettingsSelectBox(options=options, value=options_default),
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
    settings_opex_spec_fuel: SettingsSlider


@dataclass
class FrontendCoordinates(Coordinates):
    @property
    def as_dms_str(self) -> str:
        lat_deg, lat_min, lat_sec = self._decimal_to_dms(self.latitude)
        lon_deg, lon_min, lon_sec = self._decimal_to_dms(self.longitude)

        return (
            f"{lat_deg}°{lat_min}'{lat_sec:.2f}'' "
            f"{get_label('sidebar.general.position.north') if self.latitude >= 0 else get_label('sidebar.general.position.south')}, "
            f"{lon_deg}°{lon_min}'{lon_sec:.2f}'' "
            f"{get_label('sidebar.general.position.east') if self.longitude >= 0 else get_label('sidebar.general.position.west')}"
        )
