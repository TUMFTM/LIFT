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
from dataclasses import dataclass, field
from typing import Any, Literal, Self

import pandas as pd
import streamlit as st

from .utils import get_label

from lift.backend.evaluation.blocks import Coordinates


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
    max_value: float = 1.0
    value: float = 0.0
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
    options: list[str] | None = None
    value: str | None = None

    def get_streamlit_element(self, label: str, help_msg: str | None = None, key: str | None = None, domain=st):
        return domain.selectbox(
            label=label,
            help=help_msg,
            key=key,
            options=self.options,
            index=self.options.index(st.session_state.get(key, self.value)),
        )


def create_settings_obj(input_type: Literal["Slider", "Number", "Select"], **kwargs) -> SettingsInput:
    # ToDo: use single dispatchmethod?
    if input_type == "Slider":
        return SettingsSlider(**kwargs)
    elif input_type == "Number":
        return SettingsNumberInput(**kwargs)
    elif input_type == "Select":
        return SettingsSelectBox(**kwargs)
    else:
        raise ValueError(f"Input type '{input_type}' is not supported.")


@dataclass(frozen=True)
class FrontendInterface(ABC):
    name: str
    label: str | dict[str, str]  # use dict to support different languages
    icon: str

    _value_keys: list[str] = field(repr=False)
    _input_keys: list[str] = field(repr=False)

    @classmethod
    def from_dict(cls, dict_def) -> Self:
        values = dict_def.pop("values", {})
        inputs = dict_def.pop("inputs", {})
        return cls(
            **dict_def,
            **values,
            **{k: create_settings_obj(**v) for k, v in inputs.items()},
            _value_keys=list(values.keys()),
            _input_keys=list(inputs.keys()),
        )

    @property
    def values(self) -> dict:
        return {k: getattr(self, k) for k in self._value_keys}

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


@dataclass(frozen=True)
class FrontendBlockInterface(FrontendInterface, ABC):
    ls: int


@dataclass(frozen=True)
class FrontendContinuousBlockInterface(FrontendBlockInterface, ABC):
    capacity_preexisting: SettingsInput
    capacity_expansion: SettingsInput

    capex_spec: float
    capem_spec: float
    opex_spec: float
    opem_spec: float


@dataclass(frozen=True)
class FrontendGridInterface(FrontendContinuousBlockInterface):
    opex_spec: float = field(default=None, init=False, repr=False)
    opex_spec_buy: SettingsInput
    opex_spec_sell: SettingsInput
    opex_spec_peak: SettingsInput


@dataclass(frozen=True)
class FrontendPVInterface(FrontendContinuousBlockInterface):
    pass


@dataclass(frozen=True)
class FrontendESSInterface(FrontendContinuousBlockInterface):
    c_rate_max: float
    soc_init: float


@dataclass(frozen=True)
class FrontendDiscreteBlockInterface(FrontendBlockInterface):
    capex_per_unit: SettingsInput
    capem_per_unit: float

    num_preexisting: SettingsInput
    num_expansion: SettingsInput


@dataclass(frozen=True)
class FrontendChargerTypeInterface(FrontendDiscreteBlockInterface):
    p_max: SettingsInput


@dataclass(frozen=True)
class FrontendSubfleetInterface(FrontendDiscreteBlockInterface):
    # for subfleets these values are distinguished by bev/icev
    capex_per_unit: SettingsInput = field(default=None, init=False, repr=False)
    capem_per_unit: float = field(default=None, init=False, repr=False)
    num_preexisting: SettingsInput = field(default=None, init=False, repr=False)
    num_expansion: SettingsInput = field(default=None, init=False, repr=False)

    weight_max: float  # required for labeling purpose in GUI, no effect on calculation
    capacity: float
    capem_per_unit_bev: float
    capem_per_unit_icev: float
    mntex_spec_bev: float
    mntex_spec_icev: float
    toll_spec_bev: float
    toll_spec_icev: float
    consumption_spec_icev: float
    soc_init: float

    num_total: SettingsInput
    num_bev_preexisting: SettingsInput
    num_bev_expansion: SettingsInput
    charger: SettingsInput
    p_max: SettingsInput
    capex_per_unit_bev: SettingsInput
    capex_per_unit_icev: SettingsInput
    toll_frac: SettingsInput


@dataclass(frozen=True)
class FrontendAggregatorInterface(FrontendInterface, ABC):
    subblocks: dict[str, FrontendBlockInterface]

    @classmethod
    @abstractmethod
    def create_subblock_from_dict(cls, dict_def) -> FrontendBlockInterface: ...

    @classmethod
    def from_dict(cls, dict_def) -> Self:
        values = dict_def.pop("values", {})
        inputs = dict_def.pop("inputs", {})
        subblocks = dict_def.pop("subblocks", {})
        return cls(
            **dict_def,
            **values,
            **{k: create_settings_obj(**v) for k, v in inputs.items()},
            subblocks={k: cls.create_subblock_from_dict(v) for k, v in subblocks.items()},
            _value_keys=list(values.keys()),
            _input_keys=list(inputs.keys()),
        )


@dataclass(frozen=True)
class FrontendChargingInfrastructureInterface(FrontendAggregatorInterface):
    p_lm_max_preexisting: SettingsInput = field(default=None, init=False)
    p_lm_max_expansion: SettingsInput = field(default=None, init=False)

    @classmethod
    def create_subblock_from_dict(cls, dict_def) -> FrontendChargerTypeInterface:
        return FrontendChargerTypeInterface.from_dict(dict_def)


@dataclass(frozen=True)
class FrontendFleetInterface(FrontendAggregatorInterface):
    opem_spec_fuel: float
    opem_spec_onroute_charging: float

    opex_spec_fuel: SettingsInput
    opex_spec_onroute_charging: SettingsInput

    @classmethod
    def create_subblock_from_dict(cls, dict_def) -> FrontendSubfleetInterface:
        return FrontendSubfleetInterface.from_dict(dict_def)


@dataclass(frozen=True)
class FrontendScenarioInterface(FrontendInterface):
    period_eco: int
    sim_start: pd.Timestamp
    sim_duration: pd.Timedelta
    sim_freq: pd.Timedelta
    capem_initial: float

    wacc: SettingsInput
    slp: SettingsInput
    e_yrl: SettingsInput
    capex_initial: SettingsInput

    @classmethod
    def from_dict(cls, dict_def) -> Self:
        values = dict_def.pop("values", {})
        values["sim_start"] = pd.Timestamp(values["sim_start"]).tz_localize(values.pop("timezone"))
        values["sim_duration"] = pd.Timedelta(days=values["sim_duration"])
        values["sim_freq"] = pd.Timedelta(hours=values["sim_freq"])
        inputs = dict_def.pop("inputs", {})
        return cls(
            **dict_def,
            **values,
            **{k: create_settings_obj(**v) for k, v in inputs.items()},
            _value_keys=list(values.keys()),
            _input_keys=list(inputs.keys()),
        )


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
