from __future__ import annotations

import folium
import streamlit as st
from streamlit_folium import st_folium


from .definitions import (
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

from lift.backend.interfaces import (
    ComparisonInputLocation,
    ComparisonInvestComponent,
    ComparisonInputEconomics,
    ComparisonInputSubfleet,
    ComparisonInputCharger,
    Inputs,
    ExistExpansionValue,
)
from lift.utils import Coordinates

from .design import LINE_HORIZONTAL
from .interfaces import FrontendSubFleetInterface, FrontendChargerInterface

SHARE_COLUMN_INPUT = [3, 7]


def create_sidebar_and_get_input() -> Inputs:
    # get depot parameters
    st.sidebar.subheader("Allgemeine Parameter")

    # ToDo: combine location and economic parameters in one function
    def _get_input_location() -> ComparisonInputLocation:
        with st.sidebar.expander(label="**Position**", icon="🗺️"):
            if "location" not in st.session_state:
                st.session_state["location"] = Coordinates(latitude=48.1351, longitude=11.5820)

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
                    st.session_state["location"] = Coordinates(
                        latitude=st.session_state["map"]["last_clicked"]["lat"],
                        longitude=st.session_state["map"]["last_clicked"]["lng"],
                    )

            st_folium(m, height=350, width="5%", key="map", on_change=callback)
            st.markdown(f"Position: {st.session_state['location'].as_dms_str}")

        with st.sidebar.expander(label="**Energiesystem**", icon="💡"):
            st.markdown("**Stromverbrauch Standort**")
            col1, col2 = st.columns(SHARE_COLUMN_INPUT)
            slp = DEF_DEMAND.settings_dem_profile.get_streamlit_element(
                label="Lastprofil", key="slp", domain=col1
            ).lower()

            consumption_yrl_wh = DEF_DEMAND.settings_dem_yr.get_streamlit_element(
                label="Jahresstromverbrauch (MWh)", key="consumption_yrl_wh", domain=col2
            )

            st.markdown(LINE_HORIZONTAL, unsafe_allow_html=True)

            st.markdown("**Netzanschluss**")
            # ToDo: distinguish static and dynamic load management
            col1, col2 = st.columns(SHARE_COLUMN_INPUT)
            grid = ComparisonInvestComponent(
                capacity=ExistExpansionValue(
                    preexisting=DEF_GRID.settings_preexisting.get_streamlit_element(
                        label="Vorhanden (kW)", key="grid_preexisting", domain=col1
                    ),
                    expansion=DEF_GRID.settings_expansion.get_streamlit_element(
                        label="Zusätzlich (kW)",
                        key="grid_expansion",
                        domain=col2,
                    ),
                ),
                **DEF_GRID.input_dict,
            )

            st.markdown(LINE_HORIZONTAL, unsafe_allow_html=True)

            st.markdown("**PV-Anlage**")
            col1, col2 = st.columns(SHARE_COLUMN_INPUT)
            pv = ComparisonInvestComponent(
                capacity=ExistExpansionValue(
                    preexisting=DEF_PV.settings_preexisting.get_streamlit_element(
                        label="Vorhanden (kWp)", key="pv_preexisting", domain=col1
                    ),
                    expansion=DEF_PV.settings_expansion.get_streamlit_element(
                        label="Zusätzlich (kWp)",
                        key="pv_expansion",
                        domain=col2,
                    ),
                ),
                **DEF_PV.input_dict,
            )

            st.markdown(LINE_HORIZONTAL, unsafe_allow_html=True)

            st.markdown("**Stationärspeicher**")
            col1, col2 = st.columns(SHARE_COLUMN_INPUT)
            ess = ComparisonInvestComponent(
                capacity=ExistExpansionValue(
                    preexisting=DEF_ESS.settings_preexisting.get_streamlit_element(
                        label="Vorhanden (kWh)", key="ess_preexisting", domain=col1
                    ),
                    expansion=DEF_ESS.settings_expansion.get_streamlit_element(
                        label="Zusätzlich (kWh)",
                        key="ess_expansion",
                        domain=col2,
                    ),
                ),
                **DEF_ESS.input_dict,
            )

        return ComparisonInputLocation(
            coordinates=st.session_state["location"],
            slp=slp,
            consumption_yrl_wh=consumption_yrl_wh,
            grid=grid,
            pv=pv,
            ess=ess,
        )

    input_location = _get_input_location()

    def _get_input_economic() -> ComparisonInputEconomics:
        with st.sidebar.expander(label="**Wirtschaftliche Parameter**", icon="💶"):
            return ComparisonInputEconomics(
                discount_rate=DEF_ECONOMICS.settings_discount_rate.get_streamlit_element(
                    label="Abzinsungsfaktor (%)", key="eco_discount_rate", domain=st
                ),
                fix_cost_construction=DEF_ECONOMICS.settings_fix_cost_construction.get_streamlit_element(
                    label="Fixkosten Standortausbau (EUR)", key="eco_fix_cost_construction", domain=st
                ),
                opex_spec_grid_buy=DEF_ECONOMICS.settings_opex_spec_grid_buy.get_streamlit_element(
                    label="Strombezugskosten (EUR/kWh)", key="eco_opex_spec_grid_buy", domain=st
                ),
                opex_spec_grid_sell=DEF_ECONOMICS.settings_opex_spec_grid_sell.get_streamlit_element(
                    label="Einspeisevergütung (EUR/kWh)", key="eco_opex_spec_grid_sell", domain=st
                ),
                opex_spec_grid_peak=DEF_ECONOMICS.settings_opex_spec_grid_peak.get_streamlit_element(
                    label="Leistungspreis (EUR/kWp)", key="eco_opex_spec_grid_peak", domain=st
                ),
                opex_spec_route_charging=DEF_ECONOMICS.settings_opex_spec_route_charging.get_streamlit_element(
                    label="Energiekosten für On-Route Charging (EUR/kWh)", key="eco_opex_spec_route_charging", domain=st
                ),
                opex_fuel=DEF_ECONOMICS.settings_opex_fuel.get_streamlit_element(
                    label="Dieselkosten (EUR/l)", key="eco_opex_fuel", domain=st
                ),
                insurance_frac=DEF_ECONOMICS.settings_insurance_frac.get_streamlit_element(
                    label="Versicherung (%*Anschaffungspreis)", key="eco_insurance_frac", domain=st
                ),
                salvage_bev_frac=DEF_ECONOMICS.settings_salvage_bev_frac.get_streamlit_element(
                    label="Restwert BET (%) (In Berechnung noch unberücksichtigt)",
                    key="eco_salvage_bev_frac",
                    domain=st,
                ),
                salvage_icev_frac=DEF_ECONOMICS.settings_salvage_icev_frac.get_streamlit_element(
                    label="Restwert ICET (%) (In Berechnung noch unberücksichtigt)",
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

    input_economic = _get_input_economic()

    # get fleet parameters
    st.sidebar.subheader("Flotte")

    def _get_params_subfleet(subfleet: FrontendSubFleetInterface) -> ComparisonInputSubfleet:
        with st.sidebar.expander(
            label=f"**{subfleet.label}**  \n{subfleet.weight_max_t} t zulässiges Gesamtgewicht",
            icon=subfleet.icon,
            expanded=False,
        ):
            num_total = st.number_input(
                label="Fahrzeuge gesamt",
                key=f"num_{subfleet.name}",
                min_value=0,
                max_value=50,
                value=0,
                step=1,
            )

            col1, col2 = st.columns(2)
            preexisting = col1.number_input(
                "Vorhandene E-Fahrzeuge",
                key=f"num_bev_preexisting_{subfleet.name}",
                min_value=0,
                max_value=num_total,
                value=0,
                step=1,
            )

            expansion = col2.number_input(
                "Zusätzliche E-Fahrzeuge",
                key=f"num_bev_expansion_{subfleet.name}",
                min_value=0,
                max_value=num_total - preexisting,
                value=0,
                step=1,
            )

            col1, col2 = st.columns(SHARE_COLUMN_INPUT)
            charger_type = col1.selectbox(
                label="Ladepunkt", key=f"charger_{subfleet.label}", options=[x.label for x in DEF_CHARGERS.values()]
            ).lower()
            max_value = DEF_CHARGERS[charger_type].settings_pwr_max.max_value
            pwr_max_w = (
                col2.slider(
                    label="max. Ladeleistung (kW)",
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
                label="Anschaffungspreis BEV (EUR)",
                key=f"capex_bev_{subfleet.name}",
            )
            capex_icev_eur = subfleet.settings_capex_icev.get_streamlit_element(
                label="Anschaffungspreis ICEV (EUR)",
                key=f"capex_icev_{subfleet.name}",
            )
            toll_frac = subfleet.settings_toll_share.get_streamlit_element(
                label="Anteil mautplichtiger Strecken (%)",
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

    input_fleet = {sf_name: _get_params_subfleet(sf_def) for sf_name, sf_def in DEF_SUBFLEETS.items()}

    # get charging infrastructure parameters
    st.sidebar.subheader("Ladeinfrastruktur")

    def _get_params_charger(charger: FrontendChargerInterface) -> ComparisonInputCharger:
        with st.sidebar.expander(label=f"**{charger.label}-Ladepunkte**", icon=charger.icon, expanded=False):
            col1, col2 = st.columns(SHARE_COLUMN_INPUT)
            num = ExistExpansionValue(
                preexisting=charger.settings_preexisting.get_streamlit_element(
                    label="Vorhandene", key=f"chg_{charger.name.lower()}_preexisting", domain=col1
                ),
                expansion=charger.settings_expansion.get_streamlit_element(
                    label="Zusätzliche",
                    key=f"chg_{charger.name.lower()}_expansion",
                    domain=col2,
                ),
            )

            pwr_max_w = charger.settings_pwr_max.get_streamlit_element(
                label="Maximale Ladeleistung (kW)",
                key=f"chg_{charger.name.lower()}_pwr",
            )

            cost_per_charger_eur = charger.settings_cost_per_unit_eur.get_streamlit_element(
                label="Kosten (EUR pro Ladepunkt)",
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

    input_charger = {chg_name: _get_params_charger(chg_def) for chg_name, chg_def in DEF_CHARGERS.items()}

    # get phase_simulation settings
    def _get_simsettings():
        col1, col2 = st.sidebar.columns([6, 4])
        st.session_state["auto_refresh"] = col1.toggle("**Automatisch aktualisieren**", value=False)

        if st.session_state["auto_refresh"] or col2.button(
            "**Berechnen**", icon="🚀", key="calc", use_container_width=True
        ):
            st.session_state["run_backend"] = True

    _get_simsettings()

    return Inputs(location=input_location, subfleets=input_fleet, chargers=input_charger, economics=input_economic)
