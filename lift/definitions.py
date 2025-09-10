from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import streamlit as st


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
class SubFleetDefinition:
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
class ExpansionDefinition:
    name: str
    icon: str
    settings_preexisting: SettingsNumberInput
    settings_expansion: SettingsSlider
    settings_cost_per_unit_eur: SettingsSlider
    capem: float


@dataclass
class ChargerDefinition(ExpansionDefinition):
    settings_pwr_max: SettingsSlider
    ls: float


@dataclass
class EnergySystemDefinition:
    settings_dem_profile: SettingsSelectBox
    settings_dem_yr: SettingsSlider
    settings_grid_preexisting: SettingsNumberInput
    settings_grid_expansion: SettingsSlider
    settings_pv_preexisting: SettingsNumberInput
    settings_pv_expansion: SettingsSlider
    settings_ess_preexisting: SettingsNumberInput
    settings_ess_expansion: SettingsSlider


DEF_SUBFLEETS = {
    # Leergewicht BET: 18t, Zuladung: 24t
    "hlt": SubFleetDefinition(
        label="Schwere Lkw",
        icon="🚛",
        name="hlt",
        weight_max_str="42 t zulässiges Zuggesamtgewicht",
        battery_capactiy_wh=480000,
        capem_bev=84600.0,
        capem_icev=54000.0,
        weight_empty_bev=18000,
        weight_empty_icev=16600,
        toll_eur_per_km_bev=0.0,  # ToDo: add correct value
        toll_eur_per_km_icev=1.0,  # ToDo: add correct value
        mntex_eur_km_bev=0.05,  # ToDo: add correct value
        mntex_eur_km_icev=0.1,  # ToDo: add correct value
        consumption_icev=27.0,
        ls=6,
        settings_toll_share=SettingsSlider(min_value=0.0, max_value=100.0, value=80.0, step=1.0, factor=0.01),
        settings_capex_bev=SettingsSlider(min_value=0.0, max_value=250000.0, value=250000.0, step=10000.0),
        settings_capex_icev=SettingsSlider(min_value=0.0, max_value=250000.0, value=150000.0, step=10000.0),
    ),

    # Leergewicht: 10.4t, Zuladung: 16.6t
    "hst":SubFleetDefinition(
        label="Schwerer Verteilerverkehr",
        icon="🚚",
        name="hst",
        weight_max_str="28 t zulässiges Gesamtgewicht",
        battery_capactiy_wh=400000,
        capem_bev=59000.0,
        capem_icev=31200.0,
        weight_empty_bev=10400,
        weight_empty_icev=9000,
        toll_eur_per_km_bev=0.0,  # ToDo: add correct value
        toll_eur_per_km_icev=1.0,  # ToDo: add correct value
        mntex_eur_km_bev=0.05,  # ToDo: add correct value
        mntex_eur_km_icev=0.1,  # ToDo: add correct value
        consumption_icev=23.0,
        ls=6,
        settings_toll_share=SettingsSlider(min_value=0.0, max_value=100.0, value=70.0, step=1.0, factor=0.01),
        settings_capex_bev=SettingsSlider(min_value=0.0, max_value=250000.0, value=250000.0, step=10000.0),
        settings_capex_icev=SettingsSlider(min_value=0.0, max_value=250000.0, value=150000.0, step=10000.0),
    ),

    # Leergewicht: 5.4t, Zuladung: 6.6t
    "ust": SubFleetDefinition(
        label="Urbaner Verteilerverkehr",
        icon="🚚",
        name="ust",
        weight_max_str="12 t zulässiges Gesamtgewicht",
        battery_capactiy_wh=160000,
        capem_bev=26700.0,
        capem_icev=16200.0,
        weight_empty_bev=5400,
        weight_empty_icev=5400,
        toll_eur_per_km_bev=0.0,  # ToDo: add correct value
        toll_eur_per_km_icev=1.0,  # ToDo: add correct value
        mntex_eur_km_bev=0.05,  # ToDo: add correct value
        mntex_eur_km_icev=0.1,  # ToDo: add correct value
        consumption_icev=19.0,
        ls=6,
        settings_toll_share=SettingsSlider(min_value=0.0, max_value=100.0, value=40.0, step=1.0, factor=0.01),
        settings_capex_bev=SettingsSlider(min_value=0.0, max_value=250000.0, value=150000.0, step=10000.0),
        settings_capex_icev=SettingsSlider(min_value=0.0, max_value=250000.0, value=100000.0, step=10000.0),
    ),

    # Leergewicht: 2.5t, Zuladung: 1.0t
    "usv": SubFleetDefinition(
        label="Lieferwagen",
        icon="🚐",
        name="usv",
        weight_max_str="3.5 t zulässiges Gesamtgewicht",
        battery_capactiy_wh=81000,
        capem_bev=13870.0,
        capem_icev=8622.0,
        weight_empty_bev=2500,
        weight_empty_icev=2300,
        toll_eur_per_km_bev=0.0,  # ToDo: add correct value
        toll_eur_per_km_icev=1.0,  # ToDo: add correct value
        mntex_eur_km_bev=0.05,  # ToDo: add correct value
        mntex_eur_km_icev=0.1,  # ToDo: add correct value
        consumption_icev=15.0,
        ls=6,
        settings_toll_share=SettingsSlider(min_value=0.0, max_value=100.0, value=0.0, step=1.0, factor=0.01),
        settings_capex_bev=SettingsSlider(min_value=0.0, max_value=100000.0, value=45000.0, step=1000.0),
        settings_capex_icev=SettingsSlider(min_value=0.0, max_value=100000.0, value=35000.0, step=1000.0),
    ),
}

# ToDo: SettingsSlider, SettingsNumeric, SettingsSelect
# ToDo: add Classes DEF_LOCATION, DEF_ECONOMICS
DEF_CHARGERS = {
    "ac": ChargerDefinition(
        name="AC",
        icon="🔌",
        settings_pwr_max=SettingsSlider(min_value=0, max_value=43, value=11, step=1, factor=1E3),
        settings_preexisting=SettingsNumberInput(min_value=0, max_value=50, value=0),
        settings_expansion=SettingsSlider(min_value=0, max_value=50, value=0, step=1),
        settings_cost_per_unit_eur=SettingsSlider(min_value=0.0, max_value=5000.0, value=800.0, step=50.0),
        capem=65.4,
        ls=6,
    ),
    "dc": ChargerDefinition(
        name="DC",
        icon="⚡️",
        settings_pwr_max=SettingsSlider(min_value=0, max_value=1000, value=150, step=10, factor=1E3),
        settings_preexisting=SettingsNumberInput(min_value=0, max_value=50, value=0),
        settings_expansion=SettingsSlider(min_value=0, max_value=50, value=0, step=1),
        settings_cost_per_unit_eur=SettingsSlider(min_value=0.0, max_value=200000.0, value=80000.0, step=1000.0),
        capem=6520.0,
        ls=6,
    ),
}


DEF_ENERGY_SYSTEM = EnergySystemDefinition(
    settings_dem_profile=SettingsSelectBox(options=['H0', 'H0_dyn',
                                                    'G0', 'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7',
                                                    'L0', 'L1', 'L2'],
                                           index=2),  # 2 corresponds to G0
    settings_dem_yr=SettingsSlider(min_value=0, max_value=1000, value=25, step=1, factor=1E6),
    settings_grid_preexisting=SettingsNumberInput(min_value=0, max_value=10000, value=500, factor=1E3),
    settings_grid_expansion=SettingsSlider(min_value=0, max_value=5000, value=0, step=1, factor=1E3),
    settings_pv_preexisting=SettingsNumberInput(min_value=0, max_value=5000, value=0, factor=1E3),
    settings_pv_expansion=SettingsSlider(min_value=0, max_value=5000, value=0, step=1, factor=1E3),
    settings_ess_preexisting=SettingsNumberInput(min_value=0, max_value=5000, value=0, factor=1E3),
    settings_ess_expansion=SettingsSlider(min_value=0, max_value=5000, value=0, step=1, factor=1E3),
)

DTI = pd.date_range(start='2023-01-01', end='2024-01-01 00:00', freq='h', tz='Europe/Berlin', inclusive='left')
FREQ_HOURS = pd.Timedelta(DTI.freq).total_seconds() / 3600
TIME_PRJ_YRS = 18

CO2_PER_LITER_DIESEL_KG = 3.08  # kg CO2 / Liter Diesel
OPEX_SPEC_CO2_PER_KG = 45E-3  # ToDo: add correct value

DEF_PV = dict(capex_spec=0.9,
              capem_spec=0.798,
              ls=18,
              )
DEF_ESS = dict(capex_spec=0.45,
               capem_spec=0.069,
               ls=9)
DEF_GRID = dict(capex_spec=0.2,
                capem_spec=1,  # ToDo: fix value
                opem_spec=0.0004,  # ToDo: add correct value
                ls=18)
