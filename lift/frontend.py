import importlib.resources
import toml
import traceback
from typing import Tuple

import altair as alt
import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

import backend
from definitions import SubFleetDefinition, ChargerDefinition, DEF_SUBFLEETS, DEF_CHARGERS, TIME_PRJ_YRS
from interfaces import (
    GridPowerExceededError,
    SOCError,
    InputLocation,
    InputSubfleet,
    InputCharger,
    InputEconomic,
    Inputs,
    Coordinates,
    ExistExpansionValue,
    DEFAULTS
)

# Load specified project colors from colors.toml
@st.cache_data
def get_colors() -> Tuple[str, str, str, str]:
    # Get custom colors from config.toml
    with importlib.resources.files("lift").joinpath(".streamlit/colors.toml").open("r") as f:
        config = toml.load(f)
    colors = config.get("custom_colors", {})
    return colors['tumblue'], colors['baseline'], colors['expansion'], colors['lightblue']

COLOR_TUMBLUE, COLOR_BL, COLOR_EX, COLOR_LIGHTBLUE = get_colors()


STYLES = f"""
    <style>
        /* Style for fixed footer */
        .footer {{
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            width: 100%;
            background-color: {COLOR_TUMBLUE};
            color: white;
            text-align: right;
            padding: 10px;
            font-size: 14px;
        }}
        
        /* Remove link styling inside the footer */
        .footer a {{
            color: inherit;
            text-decoration: none;
        }}

        .footer a:hover {{
            text-decoration: underline;
        }}
        
        /* Define style for sidebar */
        [data-testid="stSidebar"] {{
            min-width: 450px;
            max-width: 500px;
            width: 450px;
        }}
        [data-testid="stSidebarContent"] {{
            padding-right: 20px;
        }}
        div[data-testid="stSidebar"] button {{
            width: 100% !important;
        }}

        /* remove anchor link icons from headlines */
        /* 1) Markdown-headlines (#, ##, ### ...) */
        [data-testid="stMarkdownContainer"] h1 > a,
        [data-testid="stMarkdownContainer"] h2 > a,
        [data-testid="stMarkdownContainer"] h3 > a,
        [data-testid="stMarkdownContainer"] h4 > a,
        [data-testid="stMarkdownContainer"] h5 > a,
        [data-testid="stMarkdownContainer"] h6 > a,
        [data-testid="stMarkdownContainer"] h1 svg,
        [data-testid="stMarkdownContainer"] h2 svg,
        [data-testid="stMarkdownContainer"] h3 svg,
        [data-testid="stMarkdownContainer"] h4 svg,
        [data-testid="stMarkdownContainer"] h5 svg,
        [data-testid="stMarkdownContainer"] h6 svg {{
            display: none !important;
            visibility: hidden !important;
            pointer-events: none !important;
        }}

        /* 2) component headlines (st.header/subheader/title) */
        [data-testid="stHeading"] a,
        [data-testid="stHeading"] svg,
        [data-testid="stHeadingWithAnchor"] a,
        [data-testid="stHeadingWithAnchor"] svg {{
            display: none !important;
            visibility: hidden !important;
            pointer-events: none !important;
        }}
            
        /* Remove top padding/margin of main page and set top area to be transparent*/
        header.stAppHeader {{
            background-color: transparent;
        }}
        section.stMain .block-container {{
            padding-top: 0rem;
            z-index: 1;
        }}
        
    </style>
    """

SHARE_COLUMN_INPUT = [3, 7]

horizontal_line_style = "<hr style='margin-top: 0.1rem; margin-bottom: 0.5rem;'>"

def create_sidebar_and_get_input() -> Inputs:
    # get simulation settings
    def _get_sim_settings():
        col1, col2 = st.sidebar.columns([6, 4])
        with col1:
            auto_refresh = st.toggle("**Automatisch aktualisieren**",
                                     value=False)
        with col2:
            if auto_refresh:
                st.session_state["run_backend"] = True
            else:
                button_calc_results = st.button("**Berechnen**", icon="🚀")
                if button_calc_results:
                    st.session_state["run_backend"] = True
    _get_sim_settings()

    # get depot parameters
    st.sidebar.subheader("Allgemeine Parameter")
    # ToDo: combine location and economic parameters in one function
    def _get_input_location() -> InputLocation:
        with st.sidebar.expander(label="**Position**", icon="🗺️"):

            if "location" not in st.session_state:
                st.session_state["location"] = Coordinates(latitude=48.1351, longitude=11.5820)

            try:
                location_start = list(st.session_state['map']['center'].values())
                zoom_start = st.session_state['map']['zoom']
            except KeyError:
                location_start = [48.1351, 11.5820]
                zoom_start = 5

            m = folium.Map(location=location_start, zoom_start=zoom_start)

            folium.Marker(
                st.session_state['location'].as_tuple,
            ).add_to(m)

            def callback():
                if st.session_state['map']['last_clicked']:
                    st.session_state['location'] = Coordinates(latitude=st.session_state['map']['last_clicked']['lat'],
                                                               longitude=st.session_state['map']['last_clicked']['lng'])

            st_folium(m, height=350, width='5%', key="map", on_change=callback)
            st.markdown(f"Position: {st.session_state['location'].as_dms_str}")

        with st.sidebar.expander(label="**Energiesystem**", icon="💡"):
            st.markdown("**Stromverbrauch Standort**")
            col1, col2 = st.columns(SHARE_COLUMN_INPUT)
            with col1:
                options = ['H0', 'H0_dyn',
                           'G0', 'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7',
                           'L0', 'L1', 'L2']
                slp = st.selectbox(
                    label="Lastprofil",
                    options=options,
                    index=options.index(DEFAULTS.location.slp),  # 'G0' als Default
                    key="slp",
                )
            with col2:
                consumption_yrl_wh = st.slider(label="Jahresstromverbrauch (MWh)",
                                               key="consumption_yrl_wh",
                                               min_value=10,
                                               max_value=1000,
                                               value=DEFAULTS.location.consumption_building_yrl_mwh,
                                               step=10,
                                               ) * 1E6  # convert to Wh

            st.markdown(horizontal_line_style, unsafe_allow_html=True)
            st.markdown("**Netzanschluss**")
            # ToDo: distinguish static and dynamic load management
            col1, col2 = st.columns(SHARE_COLUMN_INPUT)
            with col1:
                preexisting = st.number_input(label="Vorhanden (kW)",
                                              key="grid_preexisting",
                                              min_value=0,
                                              max_value=10000,
                                              value=DEFAULTS.location.grid_connection_kwh,
                                              )
            with col2:
                expansion = st.slider(label="Zusätzlich (kW)",
                                      key="grid_expansion",
                                      min_value=0,
                                      max_value=10000,
                                      value=0,
                                      step=10,
                                      )
            grid_capacity_w = ExistExpansionValue(preexisting=preexisting * 1E3,
                                                  expansion=expansion * 1E3)
            st.markdown(horizontal_line_style, unsafe_allow_html=True)

            st.markdown("**PV-Anlage**")
            col1, col2 = st.columns(SHARE_COLUMN_INPUT)
            with col1:
                preexisting = st.number_input(label="Vorhanden (kWp)",
                                              key="pv_preexisting",
                                              min_value=0,
                                              max_value=1000,
                                              value=DEFAULTS.location.existing_pv_kwp,
                                              )
            with col2:
                expansion = st.slider(label="Zusätzlich (kWp)",
                                      key="pv_expansion",
                                      min_value=0,
                                      max_value=1000,
                                      value=0,
                                      step=5,
                                      )
            pv_capacity_wp = ExistExpansionValue(preexisting=preexisting * 1E3,
                                                 expansion=expansion * 1E3)
            st.markdown(horizontal_line_style, unsafe_allow_html=True)

            st.markdown("**Stationärspeicher**")
            col1, col2 = st.columns(SHARE_COLUMN_INPUT)
            with col1:
                preexisting = st.number_input(label="Vorhanden (kWh)",
                                              key="ess_preexisting",
                                              min_value=0,
                                              max_value=1000,
                                              value=DEFAULTS.location.existing_ess_kwh,
                                              )
            with col2:
                expansion = st.slider(label="Zusätzlich (kWh)",
                                      key="ess_expansion",
                                      min_value=0,
                                      max_value=1000,
                                      value=0,
                                      step=5,
                                      )
            ess_capacity_wh = ExistExpansionValue(preexisting=preexisting * 1E3,
                                                  expansion=expansion * 1E3)

        return InputLocation(coordinates=st.session_state['location'],
                             slp=slp,
                             consumption_yrl_wh=consumption_yrl_wh,
                             grid_capacity_w=grid_capacity_w,
                             pv_capacity_wp=pv_capacity_wp,
                             ess_capacity_wh=ess_capacity_wh,
                             )
    input_location = _get_input_location()

    def _get_input_economic() -> InputEconomic:
        with st.sidebar.expander(label="**Wirtschaftliche Parameter**",
                                 icon="💶"):
            return InputEconomic(
                fix_cost_construction=st.slider("Fixkosten Standortausbau (EUR)", 0, 1000000, 0, 5000),
                opex_spec_grid_buy_eur_per_wh=st.slider("Strombezugskosten (EUR/kWh)", 0.00, 1.00,
                                                        DEFAULTS.economics.opex_spec_grid_buy_eur_per_wh, 0.01) * 1E-3,
                opex_spec_grid_sell_eur_per_wh=st.slider("Einspeisevergütung (EUR/kWh)", 0.00, 1.00,
                                                         DEFAULTS.economics.opex_spec_grid_sell_eur_per_wh,
                                                         0.01) * 1E-3,
                opex_spec_grid_peak_eur_per_wp=st.slider("Leistungspreis (EUR/kWp)", 0, 300,
                                                         DEFAULTS.economics.opex_spec_grid_peak_eur_per_wp, 1) * 1E-3,
                fuel_price_eur_liter=st.slider("Dieselkosten (EUR/l)", 1.00, 2.00,
                                               DEFAULTS.economics.fuel_price_eur_liter, 0.05),
                toll_icev_eur_km=st.slider("Mautkosten für ICET (EUR/km)", 0.10, 1.00,
                                           DEFAULTS.economics.toll_icev_eur_km, 0.01),
                toll_bev_eur_km=DEFAULTS.economics.toll_bev_eur_km,
                mntex_bev_eur_km=st.slider("Wartung BET (EUR/km)", 0.05, 1.00, DEFAULTS.economics.mntex_bev_eur_km,
                                           0.01),
                mntex_icev_eur_km=st.slider("Wartung ICET (EUR/km)", 0.05, 1.00, DEFAULTS.economics.mntex_icev_eur_km,
                                            0.01),
                insurance_pct=st.slider("Versicherung (%*Anschaffungspreis)", 0.1, 10.0,
                                        DEFAULTS.economics.insurance_pct, 0.1),
                salvage_bev_pct=st.slider("Restwert BET (%)", 10, 80, DEFAULTS.economics.salvage_bev_pct, 1),
                salvage_icev_pct=st.slider("Restwert ICET (%)", 10, 80, DEFAULTS.economics.salvage_icev_pct, 1),
            )
    input_economic = _get_input_economic()

    # get fleet parameters
    st.sidebar.subheader("Flotte")
    def _get_params_subfleet(subfleet: SubFleetDefinition) -> InputSubfleet:
        with st.sidebar.expander(label=f'**{subfleet.label}**  \n{subfleet.weight_max_str}',
                                 icon=subfleet.icon,
                                 expanded=False):
            num_total = st.number_input(label="Fahrzeuge gesamt",
                                        key=f'num_{subfleet.name}',
                                        min_value=0,
                                        max_value=50,
                                        value=0,
                                        step=1,
                                        )

            col1, col2 = st.columns(2)
            with col1:
                preexisting = st.number_input("Vorhandene E-Fahrzeuge",
                                              key=f'num_bev_preexisting_{subfleet.name}',
                                              min_value=0,
                                              max_value=num_total,
                                              value=0,
                                              step=1,
                                              )
            with col2:
                expansion = st.number_input("Zusätzliche E-Fahrzeuge",
                                            key=f'num_bev_expansion_{subfleet.name}',
                                            min_value=0,
                                            max_value=num_total - preexisting,
                                            value=0,
                                            step=1,
                                            )

            col1, col2 = st.columns(SHARE_COLUMN_INPUT)
            with col1:
                charger_type = st.selectbox(label="Ladepunkt",
                                            key=f'charger_{subfleet.name}',
                                            options=[x.name for x in DEF_CHARGERS.values()])
            with col2:
                max_value = DEF_CHARGERS[charger_type.lower()].settings_pwr_max.max_value
                pwr_max_w = st.slider(label="max. Ladeleistung (kW)",
                                      key=f'pwr_max_{subfleet.name}',
                                      min_value=0,
                                      max_value=max_value,
                                      value=max_value,
                                      ) * 1E3

            capex_bev_eur = st.slider(label="Anschaffungspreis BEV (EUR)",
                                      key=f'capex_bev_{subfleet.name}',
                                      **subfleet.settings_capex_bev.dict,
                                      )

            capex_icev_eur = st.slider(label="Anschaffungspreis ICEV (EUR)",
                                       key=f'capex_icev_{subfleet.name}',
                                       **subfleet.settings_capex_icev.dict,
                                       )

            toll_share_pct = st.slider(label="Anteil mautplichtiger Strecken (%)",
                                       key=f'toll_share_pct_{subfleet.name}',
                                       **subfleet.settings_toll_share.dict,
                                       )

        return InputSubfleet(
            name=subfleet.name,
            num_total=num_total,
            num_bev=ExistExpansionValue(preexisting=preexisting,
                                        expansion=expansion),
            battery_capacity_wh=subfleet.battery_capactiy_wh,
            capex_bev_eur=capex_bev_eur,
            capex_icev_eur=capex_icev_eur,
            toll_share_pct=toll_share_pct,
            charger=charger_type,
            pwr_max_w=pwr_max_w,
        )
    input_fleet = {sf_name: _get_params_subfleet(sf_def) for sf_name, sf_def in DEF_SUBFLEETS.items()}

    # get charging infrastructure parameters
    st.sidebar.subheader("Ladeinfrastruktur")
    def _get_params_charger(charger: ChargerDefinition) -> InputCharger:
        with st.sidebar.expander(label=f'**{charger.name}-Ladepunkte**',
                                 icon=charger.icon,
                                 expanded=False):
            col1, col2 = st.columns(SHARE_COLUMN_INPUT)
            with col1:
                preexisting = st.number_input(label="Vorhandene",
                                              key=f'chg_{charger.name.lower()}_preexisting',
                                              **charger.settings_preexisting.dict
                                              )
            with col2:
                expansion = st.slider(label="Zusätzliche",
                                      key=f'chg_{charger.name.lower()}_expansion',
                                      **charger.settings_expansion.dict,
                                      )

            pwr_max_w = st.slider(label="Maximale Ladeleistung (kW)",
                                  key=f'chg_{charger.name.lower()}_pwr',
                                  **charger.settings_pwr_max.dict
                                  ) * 1E3

            cost_per_charger_eur = st.slider(label="Kosten (EUR pro Ladepunkt)",
                                             key=f'chg_{charger.name.lower()}_cost',
                                             **charger.settings_cost_per_unit_eur.dict
                                             )

            return InputCharger(name=charger.name,
                                num=ExistExpansionValue(preexisting=preexisting,
                                                        expansion=expansion),
                                pwr_max_w=pwr_max_w,
                                cost_per_charger_eur=cost_per_charger_eur)
    input_charger = {chg_name: _get_params_charger(chg_def) for chg_name, chg_def in DEF_CHARGERS.items()}

    return Inputs(location=input_location,
                  subfleets=input_fleet,
                  chargers=input_charger,
                  economic=input_economic
                  )

def display_results(results):
    st.success(f"Berechnung erfolgreich!")

    st.markdown(
        f"### <span style='color:{COLOR_BL}'>Baseline</span> vs. "
        f"<span style='color:{COLOR_EX}'>Expansion</span>",
        unsafe_allow_html=True,
    )

    def _show_kpis():

        def _centered_heading(text: str) -> None:
            st.markdown(f"<h5 style='text-align:center; margin:0'>{text}</h5>", unsafe_allow_html=True)

        def _create_comparison_chart(data: pd.DataFrame,
                                  tooltips: list[alt.Tooltip] | None = None):
            def _create_ring(
                    df: pd.DataFrame,
                    radius: float,
                    thickness: float,
                    color: str,
            ) -> alt.Chart:
                background = (alt.Chart(pd.DataFrame({"value": [1]}))
                              .mark_arc(innerRadius=radius, outerRadius=radius + thickness, color=color, opacity=0.4,
                                        tooltip=None)
                              .encode(theta=alt.Theta("value:Q", stack=True))
                              )

                foreground = (
                    # alt.Chart(pd.DataFrame(data_series))
                    alt.Chart(df)
                    .mark_arc(innerRadius=radius, outerRadius=radius + thickness, cornerRadius=2, color=color)
                    .encode(theta=alt.Theta("value:Q", stack=True),
                            # tooltip=[(alt.Tooltip(f"{col}:N", title=col.replace("_", " "))
                            #          if col == "Szenario" else
                            #           alt.Tooltip(f"{col}:Q", title=col.replace("_", " "), format=",.2f"))
                            #          for col in df.columns if col != "value"]
                            tooltip=tooltips
                            )
                )
                return background + foreground

            # Create rings
            ring_baseline = _create_ring(df=data.loc[['baseline']],
                                       radius=40,
                                       thickness=20,
                                       color=COLOR_BL)
            ring_expansion = _create_ring(df=data.loc[['expansion']],
                                        radius=65,
                                        thickness=20,
                                        color=COLOR_EX)

            return ring_baseline, ring_expansion

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            _centered_heading("Gesamtkosten")
            val_baseline = sum(results.baseline.cashflow)
            val_expansion = sum(results.expansion.cashflow)
            val_max = max(val_baseline, val_expansion)
            data = pd.DataFrame(index=['baseline', 'expansion'],
                                data={'value': [val_baseline / val_max,
                                                val_expansion / val_max],
                                      'phase': ['Baseline',
                                                'Expansion'],
                                      'value_display': [val_baseline,
                                                        val_expansion],
                                      })

            tooltips = [alt.Tooltip(shorthand='phase:N', title='Szenario'),
                        alt.Tooltip(shorthand='value_display:Q', title='Gesamtkosten in EUR', format=',.0f'),]
            ring_baseline, ring_expansion = _create_comparison_chart(data=data,
                                                                  tooltips=tooltips)

            # Center text (single-row dataframe, minimal overhead)
            diff = val_expansion - val_baseline
            center_text = alt.Chart(pd.DataFrame(
                {"text": [f"{diff:+.0f} €"]})).mark_text(
                size=18,
                fontWeight="bold",
                color="green" if diff < 0 else "red",
                tooltip=None
            ).encode(
                text="text:N"
            )

            st.altair_chart(
                (ring_baseline + ring_expansion + center_text).properties(width=200, height=200),
                use_container_width=True
            )

        with col2:
            _centered_heading("Gesamt-CO₂")
            val_baseline = sum(results.baseline.co2_flow)
            val_expansion = sum(results.expansion.co2_flow)
            val_max = max(val_baseline, val_expansion)
            data = pd.DataFrame(index=['baseline', 'expansion'],
                                data={'value': [val_baseline / val_max,
                                                val_expansion / val_max],
                                      'phase': ['Baseline',
                                                'Expansion'],
                                      'value_display': [val_baseline,
                                                        val_expansion],
                                      })

            tooltips = [alt.Tooltip(shorthand='phase:N', title='Szenario'),
                        alt.Tooltip(shorthand='value_display:Q', title='CO2-Emissionen in t', format=',.0f'),]
            ring_baseline, ring_expansion = _create_comparison_chart(data=data,
                                                                  tooltips=tooltips)

            # Center text (single-row dataframe, minimal overhead)
            diff = val_expansion - val_baseline
            center_text = alt.Chart(pd.DataFrame(
                {"text": [f"{diff:+.0f} t"]})).mark_text(
                size=18,
                fontWeight="bold",
                color="green" if diff < 0 else "red",
                tooltip=None
            ).encode(
                text="text:N"
            )

            st.altair_chart(
                (ring_baseline + ring_expansion + center_text).properties(width=200, height=200),
                use_container_width=True
            )

        with col3:
            _centered_heading("Eigenverbrauchsquote")
            val_baseline = results.baseline.self_consumption_pct
            val_expansion = results.expansion.self_consumption_pct
            data = pd.DataFrame(index=['baseline', 'expansion'],
                                data={'value': [val_baseline / 100,
                                                val_expansion / 100],
                                      'phase': ['Baseline',
                                                'Expansion'],
                                      'value_display': [val_baseline,
                                                        val_expansion],
                                      })

            tooltips = [alt.Tooltip(shorthand='phase:N', title='Szenario'),
                        alt.Tooltip(shorthand='value_display:Q', title='Eigenverbrauchsquote in %', format=',.2f'),]
            ring_baseline, ring_expansion = _create_comparison_chart(data=data,
                                                                  tooltips=tooltips)

            # Center text (single-row dataframe, minimal overhead)
            diff = val_expansion - val_baseline
            center_text = alt.Chart(pd.DataFrame(
                {"text": [f"{diff:+.0f} %"]})).mark_text(
                size=18,
                fontWeight="bold",
                color="green" if diff > 0 else "red",
                tooltip=None
            ).encode(
                text="text:N"
            )

            st.altair_chart(
                (ring_baseline + ring_expansion + center_text).properties(width=200, height=200),
                use_container_width=True
            )

        with col4:
            _centered_heading("Autarkiegrad")
            val_baseline = results.baseline.self_sufficiency_pct
            val_expansion = results.expansion.self_sufficiency_pct
            data = pd.DataFrame(index=['baseline', 'expansion'],
                                data={'value': [val_baseline / 100,
                                                val_expansion / 100],
                                      'phase': ['Baseline',
                                                'Expansion'],
                                      'value_display': [val_baseline,
                                                        val_expansion],
                                      })

            tooltips = [alt.Tooltip(shorthand='phase:N', title='Szenario'),
                        alt.Tooltip(shorthand='value_display:Q', title='Autarkiegrad in %', format=',.2f'),]
            ring_baseline, ring_expansion = _create_comparison_chart(data=data,
                                                                  tooltips=tooltips)

            # Center text (single-row dataframe, minimal overhead)
            diff = val_expansion - val_baseline
            center_text = alt.Chart(pd.DataFrame(
                {"text": [f"{diff:+.0f} %"]})).mark_text(
                size=18,
                fontWeight="bold",
                color="green" if diff > 0 else "red",
                tooltip=None
            ).encode(
                text="text:N"
            )

            st.altair_chart(
                (ring_baseline + ring_expansion + center_text).properties(width=200, height=200),
                use_container_width=True
            )
    _show_kpis()

    st.markdown("#### Gesamtkosten")
    col1, col2 = st.columns([4, 1])
    with col1:
        plot_flow(flow_baseline=results.baseline.cashflow,
                  flow_expansion=results.expansion.cashflow,
                  y_label="Kumulierte Kosten in EUR",
                  )
    with col2:
        st.markdown("#### Amortisationszeitraum")
        if results.payback_period_yrs is None:
            st.markdown("Investition amortisiert sich nicht.")
        else:
            st.markdown(f"{results.payback_period_yrs:.2f} Jahre")

        st.markdown("#### Kosteneinsparung")
        st.markdown(f"{results.npc_delta:,.0f} EUR nach 18 Jahren")

    with st.expander("#### Verlauf CO₂-Ausstoß"):
        st.markdown("#### Kumulierter CO₂-Ausstoß")
        col1, col2 = st.columns([4, 1])
        with col1:
            plot_flow(results.baseline.co2_flow,
                      results.expansion.co2_flow,
                      y_label="Kumulierte CO₂-Emissionen in t",
                      )
        with col2:
            st.markdown("#### CO₂-Parität")
            # if res is None:
            #     st.markdown("Kein Schnittpunkt")
            # elif res["kind"] == "identical":
            #     st.markdown("Kurven identisch.")
            # else:
            #     yr = res["year_float"]
            #     st.markdown(f"Schnittpunkt bei {yr:.2f} Jahren")
            # st.markdown("#### CO₂-Delta")
            # cost_delta = float(np.cumsum(results.expansion.co2_flow)[-1]) - float(
            #     np.cumsum(results.baseline.co2_flow)[-1])
            # st.markdown(f"{cost_delta:,.0f} kg-CO₂ nach 18 Jahren")


def display_empty_results():
    st.markdown("""
    Willkommen zu deinem datengetriebenen Framework für die Elektrifizierung von Lkw-Flotten 
    und die Optimierung von Speditionsstandorten. Nutze die Seitenleiste, um Parameter 
    einzugeben und klicke anschließend auf den Button, um erste Berechnungen zu starten.
    """)
    st.markdown(horizontal_line_style, unsafe_allow_html=True)
    st.warning("Bitte geben Sie die Parameter in der Seitenleiste ein und klicken Sie auf "
               "**🚀 Berechnen**.")


def plot_flow(
        flow_baseline,
        flow_expansion,
        y_label: str,
):

    years = np.arange(1, TIME_PRJ_YRS + 1, dtype=int)

    y_baseline = np.cumsum(flow_baseline)
    y_expansion  = np.cumsum(flow_expansion)

    df = pd.DataFrame({"year": years, "Baseline": y_baseline, "Expansion": y_expansion})
    df_long = df.melt(id_vars="year", var_name="scenario", value_name="value")

    line = (
        alt.Chart(df_long)
        .mark_line(point={"filled": True, "size": 50}, interpolate="linear")
        .encode(
            x=alt.X(shorthand="year:Q",
                    axis=alt.Axis(title="Jahr", values=years, format=".0f"),
                    scale=alt.Scale(domain=[float(years.min()), float(years.max())], nice=False),
                    ),
            y=alt.Y(shorthand="value:Q",
                    axis=alt.Axis(title=y_label)
                    ),
            color=alt.Color(shorthand="scenario:N",
                            legend=None,
                            scale=alt.Scale(domain=["Baseline", "Expansion"],
                                            range=[COLOR_BL, COLOR_EX]),
                            ),
            tooltip=[alt.Tooltip(shorthand="scenario:N",
                                 title="Szenario"),
                     alt.Tooltip(shorthand="year:Q",
                                 title="Jahr",
                                 format=".0f"),
                     alt.Tooltip(shorthand="value:Q",
                                 title=y_label,
                                 format=",.0f"),
                     ],
        )
        .properties(height=360)
    )

    layers = [line]
    # ToDo: For any annotations use additional layers (intersection, delta values, etc.)

    chart = alt.layer(*layers).configure_axis(
        labelColor="black",
        titleColor="black",
        tickColor="black",
        domainColor="black",
    ).configure_view(
        stroke=None,
        strokeWidth=1,
    )

    st.altair_chart(chart, use_container_width=True)


def run_frontend():
    # define page settings
    st.set_page_config(
        page_title="LIFT - Logistics Infrastructure & Fleet Transformation",
        page_icon="🚚",
        layout="wide"
    )

    # css styles for sidebar
    st.markdown(STYLES, unsafe_allow_html=True)

    # initialize session state for backend run
    if "run_backend" not in st.session_state:
        st.session_state["run_backend"] = False

    # create sidebar and get input parameters from sidebar
    settings = create_sidebar_and_get_input()

    st.title("LIFT - Logistics Infrastructure & Fleet Transformation")

    if st.session_state["run_backend"] is True:
        try:
            results = backend.run_backend(inputs=settings)
            display_results(results)

        except GridPowerExceededError as e:
            st.error(f"""\
            **Fehler in der Simulation**  
            **Der Netzanschluss kann die benötigte Leistung nicht bereitstellen**  
            -> Auftretende Lastspitzen können durch einen größeren Netzanschluss oder 
            mittels PV-Anlage und stationärem Speicher abgedeckt werden.  
              
            Interne Fehlermeldung: {e}
            """)
        except SOCError as e:
            st.error(f"Fehler beim Ladezustand: {e}")
        except Exception as e:
            st.error(f"Fehler bei der Berechnung: {e}")
            st.text(traceback.format_exc())

        st.session_state["run_backend"] = False
    else:
        display_empty_results()

    # Inject footer into the page
    # st.markdown(footer, unsafe_allow_html=True)
    st.markdown('<div class="footer">'
                '<b>'
                '© 2025 Lehrstuhl für Fahrzeugtechnik, Technische Universität München – Alle Rechte vorbehalten'
                '  |  '
                'Demo Version'
                '  |  '
                '<a href="https://gitlab.lrz.de/energysystemmodelling/lift" '
                'target="_blank" '  # open in new tab
                'rel="noopener noreferrer"'  # prevent security and privacy issues with new tab
                '>GitLab</a>'
                '  |  '
                '<a href="https://www.mos.ed.tum.de/ftm/impressum/" '
                'target="_blank" '  # open in new tab
                'rel="noopener noreferrer"'  # prevent security and privacy issues with new tab
                '>Impressum</a>'
                '</b></div>', unsafe_allow_html=True)


if __name__ == "__main__":
    run_frontend()
