from __future__ import annotations

import folium
import numpy as np
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
    PERIOD_ECO,
    PERIOD_SIM,
    START_SIM,
    FREQ_SIM,
    CO2_PER_LITER_DIESEL_KG,
    OPEM_SPEC_GRID,
)

from lift.backend.comparison.interfaces import (
    ComparisonInputLocation,
    ComparisonInvestComponent,
    ComparisonInputEconomics,
    ComparisonInputSubfleet,
    ComparisonInputCharger,
    ComparisonInputChargingInfrastructure,
    ComparisonInput,
    ExistExpansionValue,
)

from lift.backend.simulation.interfaces import Coordinates

from .design import LINE_HORIZONTAL
from .interfaces import FrontendSubFleetInterface, FrontendChargerInterface, FrontendCoordinates
from .utils import load_language, get_label

SHARE_COLUMN_INPUT = [3, 7]


# ToDo: combine location and economic parameters in one function
def _get_input_location() -> ComparisonInputLocation:
    with st.sidebar.expander(label=f"**{get_label('sidebar.general.position.title')}**", icon="🗺️"):
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

    with st.sidebar.expander(label=f"**{get_label('sidebar.general.energy_system.title')}**", icon="💡"):
        st.markdown(f"**{get_label('sidebar.general.energy_system.demand.title')}**")
        col1, col2 = st.columns(SHARE_COLUMN_INPUT)
        slp = DEF_DEMAND.settings_dem_profile.get_streamlit_element(
            label=get_label("sidebar.general.energy_system.demand.slp"), key="slp", domain=col1
        ).lower()

        consumption_yrl_wh = DEF_DEMAND.settings_dem_yr.get_streamlit_element(
            label=f"{get_label('sidebar.general.energy_system.demand.consumption')} (MWh)",
            key="consumption_yrl_wh",
            domain=col2,
        )

        st.markdown(LINE_HORIZONTAL, unsafe_allow_html=True)

        st.markdown(f"**{get_label('sidebar.general.energy_system.grid.title')}**")
        # ToDo: distinguish static and dynamic load management
        col1, col2 = st.columns(SHARE_COLUMN_INPUT)
        grid = ComparisonInvestComponent(
            capacity=ExistExpansionValue(
                preexisting=DEF_GRID.settings_preexisting.get_streamlit_element(
                    label=f"{get_label('sidebar.general.energy_system.grid.existing')} (kW)",
                    key="grid_preexisting",
                    domain=col1,
                ),
                expansion=DEF_GRID.settings_expansion.get_streamlit_element(
                    label=f"{get_label('sidebar.general.energy_system.grid.expansion')} (kW)",
                    key="grid_expansion",
                    domain=col2,
                ),
            ),
            **DEF_GRID.input_dict,
        )

        st.markdown(LINE_HORIZONTAL, unsafe_allow_html=True)

        st.markdown(f"**{get_label('sidebar.general.energy_system.pv.title')}**")
        col1, col2 = st.columns(SHARE_COLUMN_INPUT)
        pv = ComparisonInvestComponent(
            capacity=ExistExpansionValue(
                preexisting=DEF_PV.settings_preexisting.get_streamlit_element(
                    label=f"{get_label('sidebar.general.energy_system.pv.existing')} (kWp)",
                    key="pv_preexisting",
                    domain=col1,
                ),
                expansion=DEF_PV.settings_expansion.get_streamlit_element(
                    label=f"{get_label('sidebar.general.energy_system.pv.expansion')} (kWp)",
                    key="pv_expansion",
                    domain=col2,
                ),
            ),
            **DEF_PV.input_dict,
        )

        st.markdown(LINE_HORIZONTAL, unsafe_allow_html=True)

        st.markdown(f"**{get_label('sidebar.general.energy_system.ess.title')}**")
        col1, col2 = st.columns(SHARE_COLUMN_INPUT)
        ess = ComparisonInvestComponent(
            capacity=ExistExpansionValue(
                preexisting=DEF_ESS.settings_preexisting.get_streamlit_element(
                    label=f"{get_label('sidebar.general.energy_system.ess.existing')} (kWh)",
                    key="ess_preexisting",
                    domain=col1,
                ),
                expansion=DEF_ESS.settings_expansion.get_streamlit_element(
                    label=f"{get_label('sidebar.general.energy_system.ess.expansion')} (kWh)",
                    key="ess_expansion",
                    domain=col2,
                ),
            ),
            **DEF_ESS.input_dict,
        )

    return ComparisonInputLocation(
        coordinates=Coordinates.from_frontend_coordinates(st.session_state["location"]),
        slp=slp,
        consumption_yrl_wh=consumption_yrl_wh,
        grid=grid,
        pv=pv,
        ess=ess,
    )


def _get_input_economic() -> ComparisonInputEconomics:
    with st.sidebar.expander(label=f"**{get_label('sidebar.general.economics.title')}**", icon="💶"):
        return ComparisonInputEconomics(
            discount_rate=DEF_ECONOMICS.settings_discount_rate.get_streamlit_element(
                label=f"{get_label('sidebar.general.economics.discount')} (%)",
                key="eco_discount_rate",
                domain=st,
            ),
            fix_cost_construction=DEF_ECONOMICS.settings_fix_cost_construction.get_streamlit_element(
                label=f"{get_label('sidebar.general.economics.fixcost')} (EUR)",
                key="eco_fix_cost_construction",
                domain=st,
            ),
            opex_spec_grid_buy=DEF_ECONOMICS.settings_opex_spec_grid_buy.get_streamlit_element(
                label=f"{get_label('sidebar.general.economics.opexbuy')} (EUR/kWh)",
                key="eco_opex_spec_grid_buy",
                domain=st,
            ),
            opex_spec_grid_sell=DEF_ECONOMICS.settings_opex_spec_grid_sell.get_streamlit_element(
                label=f"{get_label('sidebar.general.economics.opexfeedin')} (EUR/kWh)",
                key="eco_opex_spec_grid_sell",
                domain=st,
            ),
            opex_spec_grid_peak=DEF_ECONOMICS.settings_opex_spec_grid_peak.get_streamlit_element(
                label=f"{get_label('sidebar.general.economics.opexpeakpower')} (EUR/kWp)",
                key="eco_opex_spec_grid_peak",
                domain=st,
            ),
            opex_spec_route_charging=DEF_ECONOMICS.settings_opex_spec_route_charging.get_streamlit_element(
                label=f"{get_label('sidebar.general.economics.opexonroute')} (EUR/kWh)",
                key="eco_opex_spec_route_charging",
                domain=st,
            ),
            opex_fuel=DEF_ECONOMICS.settings_opex_fuel.get_streamlit_element(
                label=f"{get_label('sidebar.general.economics.opexfuel')} (EUR/l)", key="eco_opex_fuel", domain=st
            ),
            insurance_frac=DEF_ECONOMICS.settings_insurance_frac.get_streamlit_element(
                label=f"{get_label('sidebar.general.economics.insurance')}", key="eco_insurance_frac", domain=st
            ),
            salvage_bev_frac=DEF_ECONOMICS.settings_salvage_bev_frac.get_streamlit_element(
                label=f"{get_label('sidebar.general.economics.salvagebev')}",
                key="eco_salvage_bev_frac",
                domain=st,
            ),
            salvage_icev_frac=DEF_ECONOMICS.settings_salvage_icev_frac.get_streamlit_element(
                label=f"{get_label('sidebar.general.economics.salvageicev')}",
                key="eco_salvage_icev_frac",
                domain=st,
            ),
            period_eco=PERIOD_ECO,
            period_sim=PERIOD_SIM,
            start_sim=START_SIM,
            freq_sim=FREQ_SIM,
            co2_per_liter_diesel_kg=CO2_PER_LITER_DIESEL_KG,
            opem_spec_grid=OPEM_SPEC_GRID,
        )


def _get_params_subfleet(subfleet: FrontendSubFleetInterface) -> ComparisonInputSubfleet:
    with st.sidebar.expander(
        label=f"**{subfleet.get_label(st.session_state['language'])}**  \n"
        f"{subfleet.weight_max_t} t {get_label('sidebar.fleet.subfleet.weight_total')}",
        icon=subfleet.icon,
        expanded=False,
    ):
        num_total = st.number_input(
            label=get_label("sidebar.fleet.subfleet.num_total"),
            key=f"num_{subfleet.name}",
            min_value=0,
            max_value=50,
            value=0,
            step=1,
        )

        col1, col2 = st.columns(2)
        preexisting = col1.number_input(
            label=get_label("sidebar.fleet.subfleet.num_bev_existing"),
            key=f"num_bev_preexisting_{subfleet.name}",
            min_value=0,
            max_value=num_total,
            value=0,
            step=1,
        )

        expansion = col2.number_input(
            label=get_label("sidebar.fleet.subfleet.num_bev_expansion"),
            key=f"num_bev_expansion_{subfleet.name}",
            min_value=0,
            max_value=num_total - preexisting,
            value=0,
            step=1,
        )

        col1, col2 = st.columns(SHARE_COLUMN_INPUT)
        charger_type = col1.selectbox(
            label=get_label("sidebar.fleet.subfleet.charger"),
            key=f"charger_{subfleet.label}",
            options=[x.label for x in DEF_CHARGERS.values()],
        ).lower()
        max_value = DEF_CHARGERS[charger_type].settings_pwr_max.max_value
        pwr_max_w = (
            col2.slider(
                label=f"{get_label('sidebar.fleet.subfleet.pwr_max')} (kW)",
                key=f"pwr_max_{subfleet.name}",
                min_value=0.0,
                max_value=max_value,
                value=max_value,
                step=1.0,
                format="%.0f",
            )
            * 1e3
        )

        capex_bev_eur = subfleet.settings_capex_bev.get_streamlit_element(
            label=f"{get_label('sidebar.fleet.subfleet.capex')} BEV (EUR)",
            key=f"capex_bev_{subfleet.name}",
        )
        capex_icev_eur = subfleet.settings_capex_icev.get_streamlit_element(
            label=f"{get_label('sidebar.fleet.subfleet.capex')} ICEV (EUR)",
            key=f"capex_icev_{subfleet.name}",
        )
        toll_frac = subfleet.settings_toll_share.get_streamlit_element(
            label=f"{get_label('sidebar.fleet.subfleet.share_toll')} (%)",
            key=f"toll_frac_{subfleet.name}",
        )

    return ComparisonInputSubfleet(
        name=subfleet.name,
        num_total=num_total,
        num_bev=ExistExpansionValue(preexisting=preexisting, expansion=expansion),
        battery_capacity_wh=subfleet.battery_capactiy_wh,
        capex_bev_eur=capex_bev_eur,
        capex_icev_eur=capex_icev_eur,
        toll_frac=toll_frac,
        charger=charger_type,
        pwr_max_w=pwr_max_w,
        ls=subfleet.ls,
        capem_bev=subfleet.capem_bev,
        capem_icev=subfleet.capem_icev,
        mntex_eur_km_bev=subfleet.mntex_eur_km_bev,
        mntex_eur_km_icev=subfleet.mntex_eur_km_icev,
        consumption_icev=subfleet.consumption_icev,
        toll_eur_per_km_bev=subfleet.toll_eur_per_km_bev,
        toll_eur_per_km_icev=subfleet.toll_eur_per_km_icev,
    )


def _get_params_charger(charger: FrontendChargerInterface) -> ComparisonInputCharger:
    with st.sidebar.expander(
        label=f"**{charger.label}{get_label('sidebar.chargers.charger.title_suffix')}**",
        icon=charger.icon,
        expanded=False,
    ):
        col1, col2 = st.columns(SHARE_COLUMN_INPUT)
        num = ExistExpansionValue(
            preexisting=charger.settings_preexisting.get_streamlit_element(
                label=get_label("sidebar.chargers.charger.existing"),
                key=f"chg_{charger.name.lower()}_preexisting",
                domain=col1,
            ),
            expansion=charger.settings_expansion.get_streamlit_element(
                label=get_label("sidebar.chargers.charger.expansion"),
                key=f"chg_{charger.name.lower()}_expansion",
                domain=col2,
            ),
        )

        pwr_max_w = charger.settings_pwr_max.get_streamlit_element(
            label=f"{get_label('sidebar.chargers.charger.pwr_max')} (kW)",
            key=f"chg_{charger.name.lower()}_pwr",
        )

        cost_per_charger_eur = charger.settings_cost_per_unit_eur.get_streamlit_element(
            label=f"{get_label('sidebar.chargers.charger.capex')} (EUR)",
            key=f"chg_{charger.name.lower()}_cost",
        )

        return ComparisonInputCharger(
            name=charger.name,
            num=num,
            pwr_max_w=pwr_max_w,
            cost_per_charger_eur=cost_per_charger_eur,
            capem=charger.capem,
            ls=charger.ls,
        )


def _get_params_charging_infrastructure():
    # ToDo: integrate in language json file
    with st.sidebar.expander(label="**Lastmanagement**", icon="⚖️"):
        st.markdown("Vorhandenes Lastmanagement")
        col1, col2 = st.columns(2)
        col1.radio(
            label="Lastmanagement",
            label_visibility="collapsed",
            options=["statisch", "dynamisch"],
            key="load_management_baseline",
        )
        if st.session_state["load_management_baseline"] == "statisch":
            # ToDo: get limits from grid connection
            col2.slider(label="Leistung", min_value=0, max_value=100, value=100, key="load_management_slider_baseline")

        st.markdown("Erweitertes Lastmanagement")
        col1, col2 = st.columns(2)
        col1.radio(
            label="Lastmanagement",
            label_visibility="collapsed",
            options=["statisch", "dynamisch"],
            key="load_management_expansion",
        )
        if st.session_state["load_management_expansion"] == "statisch":
            # ToDo: get limits from grid connection
            col2.slider(label="Leistung", min_value=0, max_value=100, value=100, key="load_management_slider_expansion")
    chargers = {chg_name: _get_params_charger(chg_def) for chg_name, chg_def in DEF_CHARGERS.items()}

    return ComparisonInputChargingInfrastructure(
        pwr_max_w_baseline=np.inf
        if st.session_state.load_management_baseline == "dynamisch"
        else st.session_state.load_management_slider_baseline,
        pwr_max_w_expansion=np.inf
        if st.session_state.load_management_expansion == "dynamisch"
        else st.session_state.load_management_slider_expansion,
        chargers=chargers,
    )


def _get_simsettings():
    col1, col2 = st.sidebar.columns([6, 4])
    st.session_state["auto_refresh"] = col1.toggle(
        f"**{get_label('sidebar.autorefresh')}**", value=st.session_state.auto_refresh
    )

    if st.session_state["auto_refresh"] or col2.button(
        f"**{get_label('sidebar.calculate')}**", icon="🚀", key="calc", use_container_width=True
    ):
        st.session_state["run_backend"] = True


def create_sidebar_and_get_input() -> ComparisonInput:
    # language selector
    _, col2 = st.sidebar.columns([7, 3])
    col2.selectbox(
        f"**{get_label('sidebar.language_selection')}**",
        options=DEF_LANGUAGE_OPTIONS,
        index=DEF_LANGUAGE_OPTIONS.index(st.session_state.language),
        key="language_selection_box",
        on_change=lambda: load_language(language=st.session_state.language_selection_box),
    )

    # get depot parameters
    st.sidebar.subheader(get_label("sidebar.general.title"))
    input_location = _get_input_location()
    input_economic = _get_input_economic()

    # get fleet parameters
    st.sidebar.subheader(get_label("sidebar.fleet.title"))
    input_fleet = {sf_name: _get_params_subfleet(sf_def) for sf_name, sf_def in DEF_SUBFLEETS.items()}

    # get charging infrastructure parameters
    st.sidebar.subheader(get_label("sidebar.chargers.title"))
    input_charging_infrastructure = _get_params_charging_infrastructure()

    # get simulation settings
    _get_simsettings()

    return ComparisonInput(
        location=input_location,
        subfleets=input_fleet,
        charging_infrastructure=input_charging_infrastructure,
        economics=input_economic,
    )
