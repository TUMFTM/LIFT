from dataclasses import dataclass
import pandas as pd


@dataclass
class SliderSettings:
    min_value: float
    max_value: float
    value: float
    step: float
    format: str = "%d"

    @property
    def dict(self):
        return self.__dict__


@dataclass
class SubFleetDefinition:
    label: str
    icon: str
    name: str
    weight_max_str: str
    weight_empty_bev: float
    weight_empty_icev: float
    settings_battery: SliderSettings
    battery_capactiy_wh: float
    settings_toll_share: SliderSettings
    settings_load: SliderSettings
    settings_capex_bev: SliderSettings
    settings_capex_icev: SliderSettings
    co2_production_bev: float
    co2_production_icev: float
    consumption_icev: float


@dataclass
class ExpansionDefinition:
    name: str
    icon: str
    settings_preexisting: SliderSettings
    settings_expansion: SliderSettings
    settings_cost_per_unit_eur: SliderSettings
    settings_CO2_per_unit: float


@dataclass
class ChargerDefinition(ExpansionDefinition):
    settings_pwr_max: SliderSettings


SUBFLEETS = dict(
    # Leergewicht BET: 18t, Zuladung: 24t
    hlt=SubFleetDefinition(
        label="Schwere Lkw",
        icon="🚛",
        name="hlt",
        weight_max_str="42 t zulässiges Zuggesamtgewicht",
        weight_empty_bev=18000,
        weight_empty_icev=16600,
        settings_battery=SliderSettings(min_value=0.0, max_value=1000.0, value=500.0, step=10.0),
        battery_capactiy_wh=480000,
        settings_toll_share=SliderSettings(min_value=0.0, max_value=100.0, value=80.0, step=1.0),
        settings_load=SliderSettings(min_value=0.0, max_value=100.0, value=50.0, step=1.0),
        settings_capex_bev=SliderSettings(min_value=0.0, max_value=250000.0, value=250000.0, step=10000.0),
        settings_capex_icev=SliderSettings(min_value=0.0, max_value=250000.0, value=150000.0, step=10000.0),
        co2_production_bev=84600.0,
        co2_production_icev=54000.0,
        consumption_icev=27.0,
    ),

    # Leergewicht: 10.4t, Zuladung: 16.6t
    hst=SubFleetDefinition(
        label="Schwerer Verteilerverkehr",
        icon="🚚",
        name="hst",
        weight_max_str="28 t zulässiges Gesamtgewicht",
        weight_empty_bev=10400,
        weight_empty_icev=9000,
        settings_battery=SliderSettings(min_value=0.0, max_value=1000.0, value=500.0, step=10.0),
        battery_capactiy_wh=400000,
        settings_toll_share=SliderSettings(min_value=0.0, max_value=100.0, value=70.0, step=1.0),
        settings_load=SliderSettings(min_value=0.0, max_value=100.0, value=50.0, step=1.0),
        settings_capex_bev=SliderSettings(min_value=0.0, max_value=250000.0, value=250000.0, step=10000.0),
        settings_capex_icev=SliderSettings(min_value=0.0, max_value=250000.0, value=150000.0, step=10000.0),
        co2_production_bev=59000.0,
        co2_production_icev=31200.0,
        consumption_icev=23.0,
    ),

    # Leergewicht: 5.4t, Zuladung: 6.6t
    ust=SubFleetDefinition(
        label="Urbaner Verteilerverkehr",
        icon="🚚",
        name="ust",
        weight_max_str="12 t zulässiges Gesamtgewicht",
        weight_empty_bev=5400,
        weight_empty_icev=5400,
        settings_battery=SliderSettings(min_value=0.0, max_value=1000.0, value=300.0, step=10.0),
        battery_capactiy_wh=160000,
        settings_toll_share=SliderSettings(min_value=0.0, max_value=100.0, value=40.0, step=1.0),
        settings_load=SliderSettings(min_value=0.0, max_value=100.0, value=50.0, step=1.0),
        settings_capex_bev=SliderSettings(min_value=0.0, max_value=250000.0, value=150000.0, step=10000.0),
        settings_capex_icev=SliderSettings(min_value=0.0, max_value=250000.0, value=100000.0, step=10000.0),
        co2_production_bev=26700.0,
        co2_production_icev=16200.0,
        consumption_icev=19.0,
    ),

    # Leergewicht: 2.5t, Zuladung: 1.0t
    usv=SubFleetDefinition(
        label="Lieferwagen",
        icon="🚐",
        name="usv",
        weight_max_str="3.5 t zulässiges Gesamtgewicht",
        weight_empty_bev=2500,
        weight_empty_icev=2300,
        settings_battery=SliderSettings(min_value=0.0, max_value=1000.0, value=300.0, step=10.0),
        battery_capactiy_wh=81000,
        settings_toll_share=SliderSettings(min_value=0.0, max_value=100.0, value=0.0, step=1.0),
        settings_load=SliderSettings(min_value=0.0, max_value=100.0, value=50.0, step=1.0),
        settings_capex_bev=SliderSettings(min_value=0.0, max_value=100000.0, value=45000.0, step=1000.0),
        settings_capex_icev=SliderSettings(min_value=0.0, max_value=100000.0, value=35000.0, step=1000.0),
        co2_production_bev=13870.0,
        co2_production_icev=8622.0,
        consumption_icev=15.0,
    ),
)

CHARGERS = dict(
    ac=ChargerDefinition(
        name="AC",
        icon="🔌",
        settings_pwr_max=SliderSettings(min_value=0, max_value=43, value=11, step=1),
        settings_preexisting=SliderSettings(min_value=0, max_value=100, value=10, step=1),
        settings_expansion=SliderSettings(min_value=0, max_value=100, value=2, step=1),
        settings_cost_per_unit_eur=SliderSettings(min_value=0.0, max_value=5000.0, value=800.0, step=50.0),
        settings_CO2_per_unit=65.4,
    ),
    dc=ChargerDefinition(
        name="DC",
        icon="⚡️",
        settings_pwr_max=SliderSettings(min_value=0, max_value=1000, value=150, step=10),
        settings_preexisting=SliderSettings(min_value=0, max_value=100, value=2, step=1),
        settings_expansion=SliderSettings(min_value=0, max_value=100, value=5, step=1),
        settings_cost_per_unit_eur=SliderSettings(min_value=0.0, max_value=200000.0, value=80000.0, step=1000.0),
        settings_CO2_per_unit=6520.0,
    ),
)


DTI = pd.date_range(start='2023-01-01', end='2024-01-01 00:00', freq='h', tz='Europe/Berlin', inclusive='left')
FREQ_HOURS = pd.Timedelta(DTI.freq).total_seconds() / 3600
TIME_PRJ_YRS = 18
LIFESPAN_VEHICLES_YRS = 6
LIFESPAN_STORAGE_YRS = 9
CO2_SPEC_KG_PER_WH = 0.0004  # ToDo: add correct value
OPEX_SPEC_CO2_PER_KG = 45E-3  # ToDo: add correct value
TOLL_EUR_PER_KM = 0.001  # ToDo: is this dependent on the vehicle class?

