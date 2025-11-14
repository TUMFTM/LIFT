"""Streamlit sidebar for LIFT scenario input.

Purpose:
- Provide all UI controls to construct a complete `ComparisonInput`:
  site/location, economics, fleet subfleets, and charging infrastructure.

Relationships:
- Emits `ComparisonInput` consumed by `lift.backend.comparison.run_comparison` via `frontend/app.py`.
- Uses frontend definitions (`definitions.py`) for defaults and labels (`utils.get_label`).

Key Logic:
- Map widget values to typed comparison-layer inputs using factory helpers (e.g., `ExistExpansionValue`).
- Location: interactive map with Folium; energy system (demand/grid/pv/ess) via invest components.
- Fleet: per-subfleet counts (total/BEV baseline+expansion), charger type, power, costs and parameters.
- Chargers: per-charger counts baseline/expansion, unit power, unit cost; overall load management
  limit per phase (static slider or dynamic = infinity).
- Sim settings: auto-refresh / calculate button toggles `st.session_state["run_backend"]`.
"""

from __future__ import annotations
from typing import Literal

import folium
import streamlit as st
from streamlit_folium import st_folium


from .definitions import (
    DEF_LANGUAGE_OPTIONS,
    DEF_DEMAND,
    DEF_GRID,
    DEF_PV,
    DEF_ESS,
    DEF_ECONOMICS,
    DEF_SUBFLEETS,
    DEF_CHARGERS,
)

from .design import LINE_HORIZONTAL
from .interfaces import FrontendSubFleetInterface, FrontendChargerInterface, FrontendCoordinates
from .utils import load_language, get_label

SHARE_COLUMN_INPUT = [3, 7]


def _get_input_location(domain):
    with domain.position():
        if "location" not in st.session_state:
            st.session_state["location"] = FrontendCoordinates(latitude=48.1351, longitude=11.5820)

        try:
            location_start = list(st.session_state["map"]["center"].values())
            zoom_start = st.session_state["map"]["zoom"]
        except KeyError:
            location_start = [48.1351, 11.5820]
            zoom_start = 5

        m = folium.Map(location=location_start, zoom_start=zoom_start)

        folium.Marker(
            st.session_state["location"].as_tuple,
        ).add_to(m)

        def callback():
            if st.session_state["map"]["last_clicked"]:
                st.session_state["location"] = FrontendCoordinates(
                    latitude=st.session_state["map"]["last_clicked"]["lat"],
                    longitude=st.session_state["map"]["last_clicked"]["lng"],
                )

        st_folium(m, height=350, width="5%", key="map", on_change=callback)
        st.markdown(f"Position: {st.session_state['location'].as_dms_str}")

    with domain.energy_system():
        st.markdown(f"**{get_label('sidebar.general.energy_system.demand.title')}**")
        col1, col2 = st.columns(SHARE_COLUMN_INPUT)
        DEF_DEMAND.settings_dem_profile.get_streamlit_element(
            label=get_label("sidebar.general.energy_system.demand.slp.label"),
            help_msg=get_label("sidebar.general.energy_system.demand.slp.help"),
            key="slp",
            domain=col1,
        )

        DEF_DEMAND.settings_dem_yr.get_streamlit_element(
            label=f"{get_label('sidebar.general.energy_system.demand.consumption.label')} (MWh)",
            help_msg=get_label("sidebar.general.energy_system.demand.consumption.help"),
            key="consumption_yrl_wh",
            domain=col2,
        )

        st.markdown(LINE_HORIZONTAL, unsafe_allow_html=True)

        st.markdown(f"**{get_label('sidebar.general.energy_system.grid.title')}**")

        col1, col2 = st.columns(SHARE_COLUMN_INPUT)
        DEF_GRID.settings_preexisting.get_streamlit_element(
            label=f"{get_label('sidebar.general.energy_system.grid.existing.label')} (kW)",
            help_msg=get_label("sidebar.general.energy_system.grid.existing.help"),
            key="grid_preexisting",
            domain=col1,
        )

        DEF_GRID.settings_expansion.get_streamlit_element(
            label=f"{get_label('sidebar.general.energy_system.grid.expansion.label')} (kW)",
            help_msg=get_label("sidebar.general.energy_system.grid.expansion.help"),
            key="grid_expansion",
            domain=col2,
        )

        st.markdown(LINE_HORIZONTAL, unsafe_allow_html=True)

        st.markdown(f"**{get_label('sidebar.general.energy_system.pv.title')}**")
        col1, col2 = st.columns(SHARE_COLUMN_INPUT)
        DEF_PV.settings_preexisting.get_streamlit_element(
            label=f"{get_label('sidebar.general.energy_system.pv.existing.label')} (kWp)",
            help_msg=get_label("sidebar.general.energy_system.pv.existing.help"),
            key="pv_preexisting",
            domain=col1,
        )

        DEF_PV.settings_expansion.get_streamlit_element(
            label=f"{get_label('sidebar.general.energy_system.pv.expansion.label')} (kWp)",
            help_msg=get_label("sidebar.general.energy_system.pv.expansion.help"),
            key="pv_expansion",
            domain=col2,
        )

        st.markdown(LINE_HORIZONTAL, unsafe_allow_html=True)

        st.markdown(f"**{get_label('sidebar.general.energy_system.ess.title')}**")
        col1, col2 = st.columns(SHARE_COLUMN_INPUT)
        DEF_ESS.settings_preexisting.get_streamlit_element(
            label=f"{get_label('sidebar.general.energy_system.ess.existing.label')} (kWh)",
            help_msg=get_label("sidebar.general.energy_system.ess.existing.help"),
            key="ess_preexisting",
            domain=col1,
        )

        DEF_ESS.settings_expansion.get_streamlit_element(
            label=f"{get_label('sidebar.general.energy_system.ess.expansion.label')} (kWh)",
            help_msg=get_label("sidebar.general.energy_system.ess.expansion.help"),
            key="ess_expansion",
            domain=col2,
        )


def _get_input_economic(domain):
    with domain.economics():
        DEF_ECONOMICS.settings_discount_rate.get_streamlit_element(
            label=f"{get_label('sidebar.general.economics.discount.label')} (%)",
            help_msg=get_label("sidebar.general.economics.discount.help"),
            key="eco_discount_rate",
            domain=st,
        )
        DEF_ECONOMICS.settings_fix_cost_construction.get_streamlit_element(
            label=f"{get_label('sidebar.general.economics.fixcost.label')} (EUR)",
            help_msg=get_label("sidebar.general.economics.fixcost.help"),
            key="eco_fix_cost_construction",
            domain=st,
        )
        DEF_ECONOMICS.settings_opex_spec_grid_buy.get_streamlit_element(
            label=f"{get_label('sidebar.general.economics.opexbuy.label')} (EUR/kWh)",
            help_msg=get_label("sidebar.general.economics.opexbuy.help"),
            key="eco_opex_spec_grid_buy",
            domain=st,
        )
        DEF_ECONOMICS.settings_opex_spec_grid_sell.get_streamlit_element(
            label=f"{get_label('sidebar.general.economics.opexfeedin.label')} (EUR/kWh)",
            help_msg=get_label("sidebar.general.economics.opexfeedin.help"),
            key="eco_opex_spec_grid_sell",
            domain=st,
        )
        DEF_ECONOMICS.settings_opex_spec_grid_peak.get_streamlit_element(
            label=f"{get_label('sidebar.general.economics.opexpeakpower.label')} (EUR/kWp)",
            help_msg=get_label("sidebar.general.economics.opexpeakpower.help"),
            key="eco_opex_spec_grid_peak",
            domain=st,
        )
        DEF_ECONOMICS.settings_opex_spec_route_charging.get_streamlit_element(
            label=f"{get_label('sidebar.general.economics.opexonroute.label')} (EUR/kWh)",
            help_msg=get_label("sidebar.general.economics.opexonroute.help"),
            key="eco_opex_spec_route_charging",
            domain=st,
        )
        DEF_ECONOMICS.settings_opex_spec_fuel.get_streamlit_element(
            label=f"{get_label('sidebar.general.economics.opexfuel.label')} (EUR/l)",
            help_msg=get_label("sidebar.general.economics.opexfuel.help"),
            key="eco_opex_spec_fuel",
            domain=st,
        )


def _get_params_subfleet(subfleet: FrontendSubFleetInterface, domain):
    with domain().expander(
        label=f"**{subfleet.get_label(st.session_state['language'])}**  \n"
        f"{subfleet.weight_max_t} t {get_label('sidebar.fleet.subfleet.weight_total.label')}",
        icon=subfleet.icon,
        expanded=False,
    ):
        st.number_input(
            label=get_label("sidebar.fleet.subfleet.num_total.label"),
            key=f"num_{subfleet.name}",
            min_value=0,
            max_value=50,
            value=0,
            step=1,
            help=get_label("sidebar.fleet.subfleet.num_total.help"),
        )

        col1, col2 = st.columns(2)
        col1.number_input(
            label=get_label("sidebar.fleet.subfleet.num_bev_existing.label"),
            key=f"num_bev_preexisting_{subfleet.name}",
            min_value=0,
            max_value=st.session_state[f"num_{subfleet.name}"],
            value=0,
            step=1,
            help=get_label("sidebar.fleet.subfleet.num_bev_existing.help"),
        )

        col2.number_input(
            label=get_label("sidebar.fleet.subfleet.num_bev_expansion.label"),
            key=f"num_bev_expansion_{subfleet.name}",
            min_value=0,
            max_value=st.session_state[f"num_{subfleet.name}"]
            - st.session_state[f"num_bev_preexisting_{subfleet.name}"],
            value=0,
            step=1,
            help=get_label("sidebar.fleet.subfleet.num_bev_expansion.help"),
        )

        col1, col2 = st.columns(SHARE_COLUMN_INPUT)
        col1.selectbox(
            label=get_label("sidebar.fleet.subfleet.charger.label"),
            key=f"charger_{subfleet.name}",
            options=[x.label for x in DEF_CHARGERS.values()],
            help=get_label("sidebar.fleet.subfleet.charger.help"),
        )
        max_value = DEF_CHARGERS[st.session_state[f"charger_{subfleet.name}"].lower()].settings_pwr_max.max_value
        col2.slider(
            label=f"{get_label('sidebar.fleet.subfleet.pwr_max.label')} (kW)",
            key=f"pwr_max_{subfleet.name}",
            min_value=0.0,
            max_value=max_value,
            value=max_value,
            step=1.0,
            format="%.0f",
            help=get_label("sidebar.fleet.subfleet.pwr_max.help"),
        )

        subfleet.settings_capex_bev.get_streamlit_element(
            label=f"{get_label('sidebar.fleet.subfleet.capex.label')} BEV (EUR)",
            key=f"capex_bev_{subfleet.name}",
            help_msg=get_label("sidebar.fleet.subfleet.capex.help"),
        )
        subfleet.settings_capex_icev.get_streamlit_element(
            label=f"{get_label('sidebar.fleet.subfleet.capex.label')} ICEV (EUR)",
            key=f"capex_icev_{subfleet.name}",
            help_msg=get_label("sidebar.fleet.subfleet.capex.help"),
        )
        subfleet.settings_toll_share.get_streamlit_element(
            label=f"{get_label('sidebar.fleet.subfleet.share_toll.label')} (%)",
            key=f"toll_frac_{subfleet.name}",
            help_msg=get_label("sidebar.fleet.subfleet.share_toll.help"),
        )


def _get_load_mngmnt(phase: Literal["baseline", "expansion"]):
    st.markdown(f"**{get_label(f'sidebar.chargers.load_mngmnt.{phase}')}**")
    col1, col2 = st.columns(2)
    col1.radio(
        label=get_label("sidebar.chargers.load_mngmnt.type.label"),
        options=[
            get_label("sidebar.chargers.load_mngmnt.type.options.static"),
            get_label("sidebar.chargers.load_mngmnt.type.options.dynamic"),
        ],
        index=1,
        key=f"load_mngmnt_{phase}",
        help=get_label("sidebar.chargers.load_mngmnt.type.help"),
    )
    if st.session_state[f"load_mngmnt_{phase}"] == get_label("sidebar.chargers.load_mngmnt.type.options.static"):
        pwr_max_grid = (
            st.session_state.grid_preexisting
            if phase == "baseline"
            else st.session_state.grid_preexisting + st.session_state.grid_expansion
        )
        col2.slider(
            label=f"{get_label('sidebar.chargers.load_mngmnt.pwr_max.label')} (kW)",
            min_value=0.0,
            max_value=pwr_max_grid,
            value=(
                pwr_max_grid * 0.5
                if not st.session_state.get(f"load_mngmnt_slider_{phase}", None)
                else st.session_state[f"load_mngmnt_slider_{phase}"]
            ),
            step=1.0,
            format="%.0f",
            key=f"load_mngmnt_slider_{phase}",
            help=get_label("sidebar.chargers.load_mngmnt.pwr_max.help"),
        )


def _get_params_charger(charger: FrontendChargerInterface, domain):
    with domain().expander(
        label=f"**{charger.label}{get_label('sidebar.chargers.charger.title_suffix')}**",
        icon=charger.icon,
        expanded=False,
    ):
        col1, col2 = st.columns(SHARE_COLUMN_INPUT)
        charger.settings_preexisting.get_streamlit_element(
            label=get_label("sidebar.chargers.charger.existing.label"),
            key=f"chg_{charger.name.lower()}_preexisting",
            domain=col1,
            help_msg=get_label("sidebar.chargers.charger.existing.help"),
        )
        charger.settings_expansion.get_streamlit_element(
            label=get_label("sidebar.chargers.charger.expansion.label"),
            key=f"chg_{charger.name.lower()}_expansion",
            domain=col2,
            help_msg=get_label("sidebar.chargers.charger.expansion.help"),
        )

        charger.settings_pwr_max.get_streamlit_element(
            label=f"{get_label('sidebar.chargers.charger.pwr_max.label')} (kW)",
            key=f"chg_{charger.name.lower()}_pwr",
            help_msg=get_label("sidebar.chargers.charger.pwr_max.help"),
        )

        charger.settings_cost_per_unit_eur.get_streamlit_element(
            label=f"{get_label('sidebar.chargers.charger.capex.label')} (EUR)",
            key=f"chg_{charger.name.lower()}_cost",
            help_msg=get_label("sidebar.chargers.charger.capex.help"),
        )


def _get_params_charging_infrastructure(domain):
    with domain().expander(label=f"**{get_label('sidebar.chargers.load_mngmnt.title')}**", icon="⚖️"):
        _get_load_mngmnt(phase="baseline")
        _get_load_mngmnt(phase="expansion")
    chargers = {chg_name: _get_params_charger(chg_def, domain=domain) for chg_name, chg_def in DEF_CHARGERS.items()}


def _get_simsettings(domain):
    col1, col2 = domain().columns([6, 4])
    st.session_state["auto_refresh"] = col1.toggle(
        f"**{get_label('sidebar.autorefresh')}**", value=st.session_state.auto_refresh
    )

    if st.session_state["auto_refresh"] or col2.button(
        f"**{get_label('sidebar.calculate')}**", icon="🚀", key="calc", use_container_width=True
    ):
        st.session_state["run_backend"] = True


def create_sidebar_and_get_input(domain):
    # language selector
    domain.language_selection().selectbox(
        f"**{get_label('sidebar.language_selection')}**",
        options=list(DEF_LANGUAGE_OPTIONS.keys()),
        index=list(DEF_LANGUAGE_OPTIONS.values()).index(st.session_state.language),
        key="language_selection_box",
        label_visibility="collapsed",
        on_change=lambda: load_language(language=DEF_LANGUAGE_OPTIONS[st.session_state.language_selection_box]),
    )

    # get depot parameters
    _get_input_location(domain=domain.general)
    _get_input_economic(domain=domain.general)

    for sf in DEF_SUBFLEETS.values():
        _get_params_subfleet(sf, domain=domain.fleet)

    # get charging infrastructure parameters
    _get_params_charging_infrastructure(domain=domain.chargers)

    # get simulation settings
    _get_simsettings(domain=domain.calculation)
