import importlib.resources as resources
import json
from pathlib import Path
from typing import Dict

import pandas as pd

from .interfaces import (
    FrontendSubFleetInterface,
    FrontendChargerInterface,
    FrontendSizableBlockInterface,
    FrontendDemandInterface,
    FrontendEconomicsInterface,
    SettingsSlider,
)


def read_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, "rb") as f:
        return json.load(f)


DEF_DEMAND = FrontendDemandInterface.from_parameters(
    options=["H0", "H0_dyn", "G0", "G1", "G2", "G3", "G4", "G5", "G6", "G7", "L0", "L1", "L2"],
    options_default_index=2,
    max_value=1000.0,
    value=50.0,
    step=1.0,
)

DEF_GRID = FrontendSizableBlockInterface.from_dict(
    dict(name="grid", label="Netzanschluss", icon="🔌⚡") | read_json(resources.files("lift.data") / "grid.json")
)

DEF_PV = FrontendSizableBlockInterface.from_dict(
    dict(name="pv", label="PV-Anlage", icon="☀️") | read_json(resources.files("lift.data") / "pv.json")
)

DEF_ESS = FrontendSizableBlockInterface.from_dict(
    dict(name="ess", label="Stationärspeicher", icon="🔋") | read_json(resources.files("lift.data") / "ess.json")
)

DEF_ECONOMICS = FrontendEconomicsInterface(
    settings_discount_rate=SettingsSlider(max_value=20.0, value=5.0, step=0.1, factor=0.01, format="%0.1f"),
    settings_fix_cost_construction=SettingsSlider(max_value=1e6, value=10000.0, step=1000.0),
    settings_opex_spec_grid_buy=SettingsSlider(max_value=0.5, value=0.23, step=0.01, factor=1e-3, format="%0.2f"),
    settings_opex_spec_grid_sell=SettingsSlider(max_value=0.5, value=0.06, step=0.01, factor=1e-3, format="%0.2f"),
    settings_opex_spec_grid_peak=SettingsSlider(max_value=300.0, value=150.0, factor=1e-3, step=1.0),
    settings_opex_spec_route_charging=SettingsSlider(max_value=2.0, value=0.49, step=0.01, factor=1e-3, format="%0.2f"),
    settings_opex_fuel=SettingsSlider(max_value=3.0, value=1.56, step=0.01, format="%0.2f"),
    settings_insurance_frac=SettingsSlider(max_value=10.0, value=2.0, step=0.1, factor=0.01, format="%0.1f"),
    settings_salvage_bev_frac=SettingsSlider(max_value=100.0, value=20.0, step=1.0, factor=0.01),
    settings_salvage_icev_frac=SettingsSlider(max_value=100.0, value=20.0, step=1.0, factor=0.01),
)

DEF_CHARGERS = {
    k: FrontendChargerInterface.from_dict(v)
    for k, v in read_json(resources.files("lift.data") / "chargers.json").items()
}

DEF_SUBFLEETS = {
    k: FrontendSubFleetInterface.from_dict(v)
    for k, v in read_json(resources.files("lift.data") / "subfleets.json").items()
}


# ToDo: improve structure here
PERIOD_ECO = 18  # years  # ToDo: to json
PERIOD_SIM = pd.Timedelta(days=365)
START_SIM = pd.to_datetime("2023-01-01 00:00")  # ToDo: to json -> select year
FREQ_SIM = pd.Timedelta(hours=1)  # ToDo: to json?

CO2_PER_LITER_DIESEL_KG = 3.08  # kg CO2 / Liter Diesel
OPEM_SPEC_GRID = 0.0004  # ToDo: add correct value / move to json
