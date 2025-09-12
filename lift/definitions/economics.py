from lift.backend.interfaces import (
    SettingsSlider,
    DefinitionEconomics,
)


DEF_ECONOMICS = DefinitionEconomics(
    settings_fix_cost_construction=SettingsSlider(min_value=0.0, max_value=1E6, value=10000.0, step=1000.0),
    settings_opex_spec_grid_buy=SettingsSlider(min_value=0.0, max_value=0.5, value=0.23, step=0.01, factor=1E-3, format="%0.2f"),
    settings_opex_spec_grid_sell=SettingsSlider(min_value=0.0, max_value=0.5, value=0.06, step=0.01, factor=1E-3, format="%0.2f"),
    settings_opex_spec_grid_peak=SettingsSlider(min_value=0.0, max_value=300.0, value=150.0, factor=1E-3, step=1.0),
    settings_opex_fuel=SettingsSlider(min_value=0.0, max_value=3.0, value=1.56, step=0.01, format="%0.2f"),
    settings_insurance_frac=SettingsSlider(min_value=0.0, max_value=10.0, value=2.0, step=0.1, factor=0.01, format="%0.1f"),
    settings_salvage_bev_frac=SettingsSlider(min_value=0.0, max_value=100.0, value=20.0, step=1.0, factor=0.01),
    settings_salvage_icev_frac=SettingsSlider(min_value=0.0, max_value=100.0, value=20.0, step=1.0, factor=0.01),
)

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
