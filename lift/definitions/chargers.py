from lift.backend.interfaces import (
    SettingsNumberInput,
    SettingsSlider,
    DefinitionCharger,
)


DEF_CHARGERS = {
    "ac": DefinitionCharger(
        name="AC",
        icon="🔌",
        settings_pwr_max=SettingsSlider(min_value=0, max_value=43, value=11, step=1, factor=1e3),
        settings_preexisting=SettingsNumberInput(min_value=0, max_value=50, value=0),
        settings_expansion=SettingsSlider(min_value=0, max_value=50, value=0, step=1),
        settings_cost_per_unit_eur=SettingsSlider(min_value=0.0, max_value=5000.0, value=800.0, step=50.0),
        capem=65.4,
        ls=6,
    ),
    "dc": DefinitionCharger(
        name="DC",
        icon="⚡️",
        settings_pwr_max=SettingsSlider(min_value=0, max_value=1000, value=150, step=10, factor=1e3),
        settings_preexisting=SettingsNumberInput(min_value=0, max_value=50, value=0),
        settings_expansion=SettingsSlider(min_value=0, max_value=50, value=0, step=1),
        settings_cost_per_unit_eur=SettingsSlider(min_value=0.0, max_value=200000.0, value=80000.0, step=1000.0),
        capem=6520.0,
        ls=6,
    ),
}
