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
    id: str
    weight_max_str: str
    weight_empty_bev: float
    weight_empty_icev: float
    settings_battery: SliderSettings
    settings_dist_max: SliderSettings
    settings_dist_avg: SliderSettings
    settings_toll_share: SliderSettings
    settings_depot_time: SliderSettings
    settings_load: SliderSettings
    settings_capex_bev: SliderSettings
    settings_capex_icev: SliderSettings


@dataclass
class ExpansionDefinition:
    id: str
    icon: str
    settings_preexisting: SliderSettings
    settings_expansion: SliderSettings
    settings_cost_per_unit_eur: SliderSettings


@dataclass
class ChargerDefinition(ExpansionDefinition):
    settings_pwr_max: SliderSettings


SUBFLEETS = dict(
    # Leergewicht BET: 18t, Zuladung: 24t
    hlt=SubFleetDefinition(label="Schwere Lkw",
                           icon="🚛",
                           id="hlt",
                           weight_max_str="42 t zulässiges Zuggesamtgewicht",
                           weight_empty_bev=18000,
                           weight_empty_icev=16600,
                           settings_battery=SliderSettings(min_value=0.0,
                                                           max_value=1000.0,
                                                           value=300.0,
                                                           step=10.0),
                           settings_dist_max=SliderSettings(min_value=0.0,
                                                            max_value=1000.0,
                                                            value=500.0,
                                                            step=10.0),
                           settings_dist_avg=SliderSettings(min_value=0.0,
                                                            max_value=1000.0,
                                                            value=200.0,
                                                            step=10.0),
                           settings_toll_share=SliderSettings(min_value=0.0,
                                                              max_value=100.0,
                                                              value=50.0,
                                                              step=1.0),
                           settings_depot_time=SliderSettings(min_value=0.0,
                                                              max_value=24.0,
                                                              value=8.0,
                                                              step=0.5),
                           settings_load=SliderSettings(min_value=0.0,
                                                        max_value=100.0,
                                                        value=50.0,
                                                        step=1.0),
                           settings_capex_bev=SliderSettings(min_value=0.0,
                                                             max_value=100000.0,
                                                             value=50000.0,
                                                             step=1000.0),
                           settings_capex_icev=SliderSettings(min_value=0.0,
                                                              max_value=100000.0,
                                                              value=50000.0,
                                                              step=1000.0)),
    # Leergewicht: 10.4t, Zuladung: 16.6t
    hst=SubFleetDefinition(label="Schwerer Verteilerverkehr",
                           icon="🚚",
                           id="hst",
                           weight_max_str="28 t zulässiges Gesamtgewicht",
                           weight_empty_bev=10400,
                           weight_empty_icev=9000,
                           settings_battery=SliderSettings(min_value=0.0,
                                                           max_value=1000.0,
                                                           value=300.0,
                                                           step=10.0),
                           settings_dist_max=SliderSettings(min_value=0.0,
                                                            max_value=1000.0,
                                                            value=500.0,
                                                            step=10.0),
                           settings_dist_avg=SliderSettings(min_value=0.0,
                                                            max_value=1000.0,
                                                            value=200.0,
                                                            step=10.0),
                           settings_toll_share=SliderSettings(min_value=0.0,
                                                              max_value=100.0,
                                                              value=50.0,
                                                              step=1.0),
                           settings_depot_time=SliderSettings(min_value=0.0,
                                                              max_value=24.0,
                                                              value=8.0,
                                                              step=0.5),
                           settings_load=SliderSettings(min_value=0.0,
                                                        max_value=100.0,
                                                        value=50.0,
                                                        step=1.0),
                           settings_capex_bev=SliderSettings(min_value=0.0,
                                                             max_value=100000.0,
                                                             value=50000.0,
                                                             step=1000.0),
                           settings_capex_icev=SliderSettings(min_value=0.0,
                                                              max_value=100000.0,
                                                              value=50000.0,
                                                              step=1000.0)),
    # Leergewicht: 5.4t, Zuladung: 6.6t
    ust=SubFleetDefinition(label="Urbaner Verteilerverkehr",
                           icon="🚚",
                           id="ust",
                           weight_max_str="12 t zulässiges Gesamtgewicht",
                           weight_empty_bev=5400,
                           weight_empty_icev=5400,
                           settings_battery=SliderSettings(min_value=0.0,
                                                           max_value=1000.0,
                                                           value=300.0,
                                                           step=10.0),
                           settings_dist_max=SliderSettings(min_value=0.0,
                                                            max_value=1000.0,
                                                            value=500.0,
                                                            step=10.0),
                           settings_dist_avg=SliderSettings(min_value=0.0,
                                                            max_value=1000.0,
                                                            value=200.0,
                                                            step=10.0),
                           settings_toll_share=SliderSettings(min_value=0.0,
                                                              max_value=100.0,
                                                              value=50.0,
                                                              step=1.0),
                           settings_depot_time=SliderSettings(min_value=0.0,
                                                              max_value=24.0,
                                                              value=8.0,
                                                              step=0.5),
                           settings_load=SliderSettings(min_value=0.0,
                                                        max_value=100.0,
                                                        value=50.0,
                                                        step=1.0),
                           settings_capex_bev=SliderSettings(min_value=0.0,
                                                             max_value=100000.0,
                                                             value=50000.0,
                                                             step=1000.0),
                           settings_capex_icev=SliderSettings(min_value=0.0,
                                                              max_value=100000.0,
                                                              value=50000.0,
                                                              step=1000.0)),
    # Leergewicht: 2.5t, Zuladung: 1.0t
    usv=SubFleetDefinition(label="Lieferwagen",
                           icon="🚐",
                           id="usv",
                           weight_max_str="3.5 t zulässiges Gesamtgewicht",
                           weight_empty_bev=2500,
                           weight_empty_icev=2300,
                           settings_battery=SliderSettings(min_value=0.0,
                                                           max_value=1000.0,
                                                           value=300.0,
                                                           step=10.0),
                           settings_dist_max=SliderSettings(min_value=0.0,
                                                            max_value=1000.0,
                                                            value=500.0,
                                                            step=10.0),
                           settings_dist_avg=SliderSettings(min_value=0.0,
                                                            max_value=1000.0,
                                                            value=200.0,
                                                            step=10.0),
                           settings_toll_share=SliderSettings(min_value=0.0,
                                                              max_value=100.0,
                                                              value=50.0,
                                                              step=1.0),
                           settings_depot_time=SliderSettings(min_value=0.0,
                                                              max_value=24.0,
                                                              value=8.0,
                                                              step=0.5),
                           settings_load=SliderSettings(min_value=0.0,
                                                        max_value=100.0,
                                                        value=50.0,
                                                        step=1.0),
                           settings_capex_bev=SliderSettings(min_value=0.0,
                                                             max_value=100000.0,
                                                             value=50000.0,
                                                             step=1000.0),
                           settings_capex_icev=SliderSettings(min_value=0.0,
                                                              max_value=100000.0,
                                                              value=50000.0,
                                                              step=1000.0)),
)


CHARGERS = dict(
    ac=ChargerDefinition(id="AC",
                         icon="🔌",
                         settings_pwr_max=SliderSettings(min_value=0,
                                                         max_value=43,
                                                         value=11,
                                                         step=1),
                         settings_preexisting=SliderSettings(min_value=0,
                                                             max_value=100,
                                                             value=10,
                                                             step=1),
                         settings_expansion=SliderSettings(min_value=0,
                                                           max_value=100,
                                                           value=10,
                                                           step=1),
                         settings_cost_per_unit_eur=SliderSettings(min_value=0.0,
                                                                   max_value=5000.0,
                                                                   value=800.0,
                                                                   step=50.0)),
    dc=ChargerDefinition(id="DC",
                         icon="⚡️",
                         settings_pwr_max=SliderSettings(min_value=0,
                                                         max_value=1000,
                                                         value=100,
                                                         step=10),
                         settings_preexisting=SliderSettings(min_value=0,
                                                             max_value=100,
                                                             value=10,
                                                             step=1),
                         settings_expansion=SliderSettings(min_value=0,
                                                           max_value=100,
                                                           value=10,
                                                           step=1),
                         settings_cost_per_unit_eur=SliderSettings(min_value=0.0,
                                                                   max_value=200000.0,
                                                                   value=100000.0,
                                                                   step=500.0)),
)


DTI = pd.date_range(start='2023-01-01', end='2024-01-01 00:00', freq='h', tz='Europe/Berlin', inclusive='left')
FREQ_HOURS = pd.Timedelta(DTI.freq).total_seconds() / 3600
TIME_PRJ_YRS = 18
LIFESPAN_VEHICLES_YRS = 6
LIFESPAN_STORAGE_YRS = 9
CO2_SPEC_KG_PER_WH = 0.0004  # ToDo: add correct value
OPEX_SPEC_GRID_BUY_EUR_PER_WH = 30E-5# ToDo: use value from economic settings
OPEX_SPEC_GRID_SELL_EUR_PER_WH = -6E-5  # ToDo: use value from economic settings
OPEX_SPEC_GRID_PWR_EUR_WP = 150E-3  # ToDo: use value from economic settings
OPEX_SPEC_FUEL_EUR_PER_L = 1.5  # ToDo: use value from economic settings
OPEX_SPEC_CO2_PER_KG = 45E-3  # ToDo: add correct value
TOLL_EUR_PER_KM = 0.001  # ToDo: is this dependent on the vehicle class?

