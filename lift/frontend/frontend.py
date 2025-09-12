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

from lift.backend import backend

from lift.definitions import (
    DEF_SUBFLEETS,
    DEF_CHARGERS,
    DEF_ENERGY_SYSTEM,
    DEF_ECONOMICS,
    TIME_PRJ_YRS,
)

from lift.backend.interfaces import (
    DefinitionSubfleet,
    DefinitionCharger,
    GridPowerExceededError,
    SOCError,
    InputLocation,
    InputSubfleet,
    InputCharger,
    InputEconomic,
    Inputs,
    Coordinates,
    ExistExpansionValue,
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

PLOT_CONFIG = {"usermeta": {"embedOptions": {"actions": False}}}

@st.cache_data
def get_version() -> str:
    try:
        return f"v{importlib.metadata.version('lift')}"
    except importlib.metadata.PackageNotFoundError:
        return "dev"
VERSION = get_version()


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

        with st.sidebar.expander(label="**Energiesystem**",
                                 icon="💡"):
            st.markdown("**Stromverbrauch Standort**")
            col1, col2 = st.columns(SHARE_COLUMN_INPUT)
            slp = DEF_ENERGY_SYSTEM.settings_dem_profile.get_input(label="Lastprofil",
                                                                   key="slp",
                                                                   domain=col1)

            consumption_yrl_wh = DEF_ENERGY_SYSTEM.settings_dem_yr.get_input(label="Jahresstromverbrauch (MWh)",
                                                                             key="consumption_yrl_wh",
                                                                             domain=col2)

            st.markdown(horizontal_line_style, unsafe_allow_html=True)

            st.markdown("**Netzanschluss**")
            # ToDo: distinguish static and dynamic load management
            col1, col2 = st.columns(SHARE_COLUMN_INPUT)
            grid_capacity_w = ExistExpansionValue(
                preexisting = DEF_ENERGY_SYSTEM.settings_grid_preexisting.get_input(label="Vorhanden (kW)",
                                                                                    key="grid_preexisting",
                                                                                    domain=col1),
                expansion=DEF_ENERGY_SYSTEM.settings_grid_expansion.get_input(label="Zusätzlich (kW)",
                                                                              key="grid_expansion",
                                                                              domain=col2,
                                                                              )
            )

            st.markdown(horizontal_line_style, unsafe_allow_html=True)

            st.markdown("**PV-Anlage**")
            col1, col2 = st.columns(SHARE_COLUMN_INPUT)
            pv_capacity_wp = ExistExpansionValue(
                preexisting=DEF_ENERGY_SYSTEM.settings_pv_preexisting.get_input(label="Vorhanden (kWp)",
                                                                                key="pv_preexisting",
                                                                                domain=col1),
                expansion=DEF_ENERGY_SYSTEM.settings_pv_expansion.get_input(label="Zusätzlich (kWp)",
                                                                            key="pv_expansion",
                                                                            domain=col2,
                                                                            ),
            )

            st.markdown(horizontal_line_style, unsafe_allow_html=True)

            st.markdown("**Stationärspeicher**")
            col1, col2 = st.columns(SHARE_COLUMN_INPUT)
            ess_capacity_wh = ExistExpansionValue(
                preexisting=DEF_ENERGY_SYSTEM.settings_ess_preexisting.get_input(label="Vorhanden (kWh)",
                                                                                 key="ess_preexisting",
                                                                                 domain=col1),
                expansion=DEF_ENERGY_SYSTEM.settings_ess_expansion.get_input(label="Zusätzlich (kWh)",
                                                                             key="ess_expansion",
                                                                             domain=col2,
                                                                             ),
            )

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
                fix_cost_construction=DEF_ECONOMICS.settings_fix_cost_construction.get_input(
                    label="Fixkosten Standortausbau (EUR)",
                    key="eco_fix_cost_construction",
                    domain=st),
                opex_spec_grid_buy=DEF_ECONOMICS.settings_opex_spec_grid_buy.get_input(
                    label="Strombezugskosten (EUR/kWh)",
                    key="eco_opex_spec_grid_buy",
                    domain=st),
                opex_spec_grid_sell=DEF_ECONOMICS.settings_opex_spec_grid_sell.get_input(
                    label="Einspeisevergütung (EUR/kWh)",
                    key="eco_opex_spec_grid_sell",
                    domain=st),
                opex_spec_grid_peak=DEF_ECONOMICS.settings_opex_spec_grid_peak.get_input(
                    label="Leistungspreis (EUR/kWp)",
                    key="eco_opex_spec_grid_peak",
                    domain=st),
                opex_fuel=DEF_ECONOMICS.settings_opex_fuel.get_input(
                    label="Dieselkosten (EUR/l)",
                    key="eco_opex_fuel",
                    domain=st),
                insurance_frac=DEF_ECONOMICS.settings_insurance_frac.get_input(
                    label="Versicherung (%*Anschaffungspreis)",
                    key="eco_insurance_frac",
                    domain=st),
                salvage_bev_frac=DEF_ECONOMICS.settings_salvage_bev_frac.get_input(
                    label="Restwert BET (%) (In Berechnung noch unberücksichtigt)",
                    key="eco_salvage_bev_frac",
                    domain=st),
                salvage_icev_frac=DEF_ECONOMICS.settings_salvage_icev_frac.get_input(
                    label="Restwert ICET (%) (In Berechnung noch unberücksichtigt)",
                    key="eco_salvage_icev_frac",
                    domain=st),
                )
    input_economic = _get_input_economic()

    # get fleet parameters
    st.sidebar.subheader("Flotte")
    def _get_params_subfleet(subfleet: DefinitionSubfleet) -> InputSubfleet:
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
            preexisting = col1.number_input("Vorhandene E-Fahrzeuge",
                                            key=f'num_bev_preexisting_{subfleet.name}',
                                            min_value=0,
                                            max_value=num_total,
                                            value=0,
                                            step=1,
                                            )

            expansion = col2.number_input("Zusätzliche E-Fahrzeuge",
                                          key=f'num_bev_expansion_{subfleet.name}',
                                          min_value=0,
                                          max_value=num_total - preexisting,
                                          value=0,
                                          step=1,
                                          )

            col1, col2 = st.columns(SHARE_COLUMN_INPUT)
            charger_type = col1.selectbox(label="Ladepunkt",
                                          key=f'charger_{subfleet.name}',
                                          options=[x.name for x in DEF_CHARGERS.values()])
            max_value = DEF_CHARGERS[charger_type.lower()].settings_pwr_max.max_value
            pwr_max_w = col2.slider(label="max. Ladeleistung (kW)",
                                    key=f'pwr_max_{subfleet.name}',
                                    min_value=0,
                                    max_value=max_value,
                                    value=max_value,
                                    ) * 1E3

            capex_bev_eur = subfleet.settings_capex_bev.get_input(label="Anschaffungspreis BEV (EUR)",
                                                                  key=f'capex_bev_{subfleet.name}',)
            capex_icev_eur = subfleet.settings_capex_icev.get_input(label="Anschaffungspreis ICEV (EUR)",
                                                                    key=f'capex_icev_{subfleet.name}', )
            toll_frac = subfleet.settings_toll_share.get_input(label="Anteil mautplichtiger Strecken (%)",
                                                               key=f'toll_frac_{subfleet.name}',)


        return InputSubfleet(
            name=subfleet.name,
            num_total=num_total,
            num_bev=ExistExpansionValue(preexisting=preexisting,
                                        expansion=expansion),
            battery_capacity_wh=subfleet.battery_capactiy_wh,
            capex_bev_eur=capex_bev_eur,
            capex_icev_eur=capex_icev_eur,
            toll_frac=toll_frac,
            charger=charger_type,
            pwr_max_w=pwr_max_w,
        )
    input_fleet = {sf_name: _get_params_subfleet(sf_def) for sf_name, sf_def in DEF_SUBFLEETS.items()}

    # get charging infrastructure parameters
    st.sidebar.subheader("Ladeinfrastruktur")
    def _get_params_charger(charger: DefinitionCharger) -> InputCharger:
        with st.sidebar.expander(label=f'**{charger.name}-Ladepunkte**',
                                 icon=charger.icon,
                                 expanded=False):
            col1, col2 = st.columns(SHARE_COLUMN_INPUT)
            num = ExistExpansionValue(
                preexisting=charger.settings_preexisting.get_input(label="Vorhandene",
                                                                   key=f'chg_{charger.name.lower()}_preexisting',
                                                                   domain=col1),
                expansion=charger.settings_expansion.get_input(label="Zusätzliche",
                                                               key=f'chg_{charger.name.lower()}_expansion',
                                                               domain=col2,
                                                               ),
            )

            pwr_max_w = charger.settings_pwr_max.get_input(label="Maximale Ladeleistung (kW)",
                                                           key=f'chg_{charger.name.lower()}_pwr',
                                                           )

            cost_per_charger_eur = charger.settings_cost_per_unit_eur.get_input(label="Kosten (EUR pro Ladepunkt)",
                                                                                key=f'chg_{charger.name.lower()}_cost',
                                                                                )

            return InputCharger(name=charger.name,
                                num=num,
                                pwr_max_w=pwr_max_w,
                                cost_per_charger_eur=cost_per_charger_eur)
    input_charger = {chg_name: _get_params_charger(chg_def) for chg_name, chg_def in DEF_CHARGERS.items()}

    # get simulation settings
    def _get_sim_settings():
        col1, col2 = st.sidebar.columns([6, 4])
        st.session_state["auto_refresh"] = col1.toggle("**Automatisch aktualisieren**",
                                                       value=False)

        if st.session_state["auto_refresh"] or col2.button("**Berechnen**",
                                                           icon="🚀",
                                                           key="calc",
                                                           use_container_width=True):
            st.session_state["run_backend"] = True


    _get_sim_settings()

    return Inputs(location=input_location,
                  subfleets=input_fleet,
                  chargers=input_charger,
                  economic=input_economic
                  )

def display_results(results):
    st.markdown(
        f"### <span style='color:{COLOR_BL}'>Baseline</span> vs. "
        f"<span style='color:{COLOR_EX}'>Expansion</span>",
        unsafe_allow_html=True,
    )

    def _show_kpis():

        def _centered_heading(text: str, domain=st) -> None:
            domain.markdown(f"<h5 style='text-align:center; margin:0'>{text}</h5>",
                            unsafe_allow_html=True)

        def _create_bar_comparison(val_baseline: float,
                                   val_expansion: float,
                                   label: str,
                                   factor_display: float = 1.0,
                                   ) -> alt.VConcatChart:
            data = pd.DataFrame(index=['baseline', 'expansion'],
                                data={'value': [val_baseline,
                                                val_expansion,
                                                ],
                                      'phase': ['Baseline',
                                                'Expansion'],
                                      'value_display': [val_baseline * factor_display,
                                                        val_expansion * factor_display],
                                      })

            text = (
                alt.Chart(pd.DataFrame({"x": [0.5],
                                        "y": [0.5],
                                        "label": [f"{(val_expansion / val_baseline - 1) * 100:+.1f} %"]}))
                .mark_text(fontWeight="bold",
                           size=18,
                           color="green" if val_baseline >= val_expansion else "red",
                           align="center",
                           baseline="middle")
                .encode(
                    x=alt.X("x:Q", axis=None, scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y("y:Q", axis=None, scale=alt.Scale(domain=[0, 1])),
                    text="label:N",
                    tooltip=[],
                )
                .properties(width=200, height=30)
            )

            bars = (
                alt.Chart(data)
                .mark_bar()
                .encode(
                    x=alt.X(shorthand="phase:N",
                            axis=None,
                            sort=["Baseline", "Expansion"],
                            scale=alt.Scale(paddingInner=0.0, paddingOuter=1)
                            ),
                    y=alt.Y(shorthand="value:Q",
                            axis=None,
                            ),
                    color=alt.Color(shorthand="phase:N", legend=None,
                                    scale=alt.Scale(domain=["Baseline", "Expansion"],
                                                    range=[COLOR_BL, COLOR_EX])),
                    tooltip=[alt.Tooltip(shorthand='phase:N', title='Szenario'),
                             alt.Tooltip(shorthand='value_display:Q', title=label, format=',.0f'),
                             ],
                )
            ).properties(width=100, height=130)

            return alt.vconcat(text, bars).configure_view(stroke=None)

        def _create_ring_comparison(val_baseline: float,
                                    val_expansion: float,
                                    label: str,
                                    ) -> alt.LayerChart:
            def _create_ring(
                    df: pd.DataFrame,
                    radius: float,
                    thickness: float,
                    color: str,
                    tooltip_list: list,
            ) -> alt.LayerChart:
                background = (
                    alt.Chart(df)
                    .mark_arc(innerRadius=radius, outerRadius=radius + thickness, color=color, opacity=0.4,
                              tooltip=None)
                    .encode(theta=alt.Theta("value_back:Q", stack=True),
                            tooltip=tooltip_list,
                            )
                )

                foreground = (
                    alt.Chart(df)
                    .mark_arc(innerRadius=radius, outerRadius=radius + thickness, cornerRadius=2, color=color)
                    .encode(theta=alt.Theta("value_front:Q", stack=True),
                            tooltip=tooltip_list,
                            )
                )
                return background + foreground

            data = pd.DataFrame(index=['baseline', 'expansion'],
                                data={'value_front': [val_baseline,
                                                      val_expansion],
                                      'value_back': [1, 1],
                                      'phase': ['Baseline',
                                                'Expansion'],
                                      'value_display': [val_baseline * 100,
                                                        val_expansion * 100],
                                      })

            tooltips = [alt.Tooltip(shorthand='phase:N', title='Szenario'),
                        alt.Tooltip(shorthand='value_display:Q', title=label, format=',.2f'), ]

            # Create rings
            ring_baseline = _create_ring(df=data.loc[['baseline']],
                                         radius=40,
                                         thickness=20,
                                         color=COLOR_BL,
                                         tooltip_list=tooltips,
                                         )
            ring_expansion = _create_ring(df=data.loc[['expansion']],
                                          radius=65,
                                          thickness=20,
                                          color=COLOR_EX,
                                          tooltip_list=tooltips,
                                          )

            # Center text (single-row dataframe, minimal overhead)
            center_text = alt.Chart(pd.DataFrame(
                {"text": [f"{(val_expansion - val_baseline) * 100:+.0f} %"]})).mark_text(
                size=18,
                fontWeight="bold",
                color="green" if val_expansion >= val_baseline else "red",
                tooltip=None
            ).encode(
                text="text:N"
            )

            return (ring_baseline + ring_expansion + center_text).properties(width=200, height=200)

        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

        _centered_heading(text="Kosten", domain=col1)
        col1.altair_chart(_create_bar_comparison(val_baseline=results.baseline.cashflow.sum(),
                                                 val_expansion=results.expansion.cashflow.sum(),
                                                 label="Gesamtkosten in EUR",
                                                 ).properties(**PLOT_CONFIG),
                          use_container_width=True,
                          )

        _centered_heading(text="CO₂-Emissionen", domain=col2)
        col2.altair_chart(_create_bar_comparison(val_baseline=results.baseline.co2_flow.sum(),
                                                 val_expansion=results.expansion.co2_flow.sum(),
                                                 label="CO2-Emissionen in t",
                                                 factor_display=1E-3,  # convert from kg to t
                                                 ).properties(**PLOT_CONFIG),
                          use_container_width=True)

        _centered_heading(text="Eigenverbrauchsquote", domain=col3)
        col3.altair_chart(_create_ring_comparison(val_baseline=results.baseline.self_consumption,
                                                  val_expansion=results.expansion.self_consumption,
                                                  label="Eigenverbrauchsquote in %",
                                                  ).properties(**PLOT_CONFIG),
                          use_container_width=True)

        _centered_heading(text="Autarkiegrad", domain=col4)
        col4.altair_chart(_create_ring_comparison(val_baseline=results.baseline.self_sufficiency,
                                                  val_expansion=results.expansion.self_sufficiency,
                                                  label="Autarkiegrad in %",
                                                  ).properties(**PLOT_CONFIG),
                          use_container_width=True)

    _show_kpis()

    st.markdown("#### Gesamtkosten")
    col1, col2 = st.columns([4, 1])
    with col1:
        plot_flow(flow_baseline=results.baseline.cashflow,
                  flow_expansion=results.expansion.cashflow,
                  y_label="Kumulierte Kosten in EUR",
                  )
    with col2:
        st.markdown("#### Amortisationsdauer")
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

    years = np.arange(TIME_PRJ_YRS, dtype=int) + 1

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
    for key in ["run_backend", "auto_refresh"]:
        if key not in st.session_state:
            st.session_state[key] = False

    # create sidebar and get input parameters from sidebar
    settings = create_sidebar_and_get_input()

    st.title("LIFT - Logistics Infrastructure & Fleet Transformation")

    if st.session_state["run_backend"] is True:
        try:
            results = backend.run_backend(inputs=settings)
            display_results(results)

        except GridPowerExceededError as e:
            st.error(f"""\
            **Netzanschlussfehler**  
            **Der Netzanschluss kann die benötigte Leistung nicht bereitstellen**  
            -> Auftretende Lastspitzen können durch einen größeren Netzanschluss oder 
            mittels PV-Anlage und stationärem Speicher abgedeckt werden.  
              
            Interne Fehlermeldung: {e}
            """)
        except SOCError as e:
            st.error(f"""\
            **Ladezustandsfehler**  
            **Der Ladezustand eines Fahrzeugs reicht nicht für die vorgesehene Fahrt aus**  
            -> Abhilfe kann eine höhere Ladeleistung (Minimum aus Leistung von Fahrzeug und Ladepunkt), eine höhere
            Anzahl an Ladepunkten oder ein größerer Netzanschluss schaffen.  
              
            Interne Fehlermeldung: {e}
            """)
        except Exception as e:
            st.error(f"""\
            **Berechnungsfehler**  
            Wenden Sie sich bitte an den Administrator des Tools. Geben Sie dabei die verwendeten Parameter und die 
            nachfolgend angezeigte Fehlermeldung an.  
              
            Interne Fehlermeldung: {e}
            """)
            st.text(traceback.format_exc())

        st.session_state["run_backend"] = False
    else:
        display_empty_results()

    # Inject footer into the page
    # st.markdown(footer, unsafe_allow_html=True)
    sep = '<span style="margin: 0 20px;"> | </span>'
    st.markdown('<div class="footer">'
                '<b>'
                '© 2025 '
                '<a href="https://www.mos.ed.tum.de/ftm/" '
                'target="_blank" '  # open in new tab
                'rel="noopener noreferrer"'  # prevent security and privacy issues with new tab
                '>Lehrstuhl für Fahrzeugtechnik, Technische Universität München</a>'
                ' – Alle Rechte vorbehalten'
                f'{sep}'
                f'Demo Version {VERSION}'
                f'{sep}'
                '<a href="https://gitlab.lrz.de/energysystemmodelling/lift" '
                'target="_blank" '  # open in new tab
                'rel="noopener noreferrer"'  # prevent security and privacy issues with new tab
                '>GitLab</a>'
                f'{sep}'
                '<a href="https://www.mos.ed.tum.de/ftm/impressum/" '
                'target="_blank" '  # open in new tab
                'rel="noopener noreferrer"'  # prevent security and privacy issues with new tab
                '>Impressum</a>'
                '<span style="margin: 0 10px;"> </span>'
                '</b></div>', unsafe_allow_html=True)


if __name__ == "__main__":
    run_frontend()
