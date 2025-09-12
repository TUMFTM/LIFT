from lift.backend.interfaces import (
    SettingsNumberInput,
    SettingsSlider,
    SettingsSelectBox,
    DefinitionEnergySystem,
)


DEF_ENERGY_SYSTEM = DefinitionEnergySystem(
    settings_dem_profile=SettingsSelectBox(options=['H0', 'H0_dyn',
                                                    'G0', 'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7',
                                                    'L0', 'L1', 'L2'],
                                           index=2),  # default value: 2 corresponds to G0
    settings_dem_yr=SettingsSlider(min_value=0, max_value=1000, value=50, step=1, factor=1E6),
    settings_grid_preexisting=SettingsNumberInput(min_value=0, max_value=5000, value=200, factor=1E3),
    settings_grid_expansion=SettingsSlider(min_value=0, max_value=5000, value=0, step=1, factor=1E3),
    settings_pv_preexisting=SettingsNumberInput(min_value=0, max_value=1000, value=10, factor=1E3),
    settings_pv_expansion=SettingsSlider(min_value=0, max_value=1000, value=40, step=1, factor=1E3),
    settings_ess_preexisting=SettingsNumberInput(min_value=0, max_value=1000, value=5, factor=1E3),
    settings_ess_expansion=SettingsSlider(min_value=0, max_value=1000, value=15, step=1, factor=1E3),
)
