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
from definitions import SubFleetDefinition, ChargerDefinition, SUBFLEETS, CHARGERS, TIME_PRJ_YRS
from interfaces import (
    GridPowerExceededError,
    SOCError,
    LocationSettings,
    SubFleetSettings,
    ChargerSettings,
    EconomicSettings,
    Settings, Coordinates, Size
)


# Load specified project colors from colors.toml
@st.cache_data
def get_colors() -> Tuple[str, str, str]:
    # Get custom colors from config.toml
    with importlib.resources.files("lift").joinpath(".streamlit/colors.toml").open("r") as f:
        config = toml.load(f)
    colors = config.get("custom_colors", {})
    return colors['tumblue'], colors['baseline'], colors['expansion']

COLOR_TUMBLUE, COLOR_BL, COLOR_EX = get_colors()


footer_css = f"""
    <style>
        /* Style for fixed footer */
        .footer {{
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: {COLOR_TUMBLUE};
            color: white;
            text-align: center;
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
    </style>
"""

sidebar_style = """
        <style>
            [data-testid="stSidebar"] {
                min-width: 450px;
                max-width: 500px;
                width: 450px;
            }
            [data-testid="stSidebarContent"] {
                padding-right: 20px;
            }
            div[data-testid="stSidebar"] button {
                width: 100% !important;
            }
        </style>
        """
st.markdown("""
<style>
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
[data-testid="stMarkdownContainer"] h6 svg {
  display: none !important;
  visibility: hidden !important;
  pointer-events: none !important;
}

/* 2) component headlines (st.header/subheader/title) */
[data-testid="stHeading"] a,
[data-testid="stHeading"] svg,
[data-testid="stHeadingWithAnchor"] a,
[data-testid="stHeadingWithAnchor"] svg {
  display: none !important;
  visibility: hidden !important;
  pointer-events: none !important;
}

</style>
""", unsafe_allow_html=True)

horizontal_line_style = "<hr style='margin-top: 0.1rem; margin-bottom: 0.5rem;'>"

VEH_YEARS_IDX = [0, 5, 11]

def _get_params_location() -> LocationSettings:

    col_share = [3, 7]

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
        col1, col2 = st.columns(col_share)
        with col1:
            slp=st.selectbox(label="Lastprofil",
                             key="slp",
                             options=['H0', 'H0_dyn',
                                      'G0', 'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7',
                                      'L0', 'L1', 'L2']).lower()
        with col2:
            consumption_yrl_wh=st.slider(label="Jahresstromverbrauch (MWh)",
                                         key="consumption_yrl_wh",
                                         min_value=10,
                                         max_value=1000,
                                         value=500,
                                         step=10,
                                         ) * 1E6  # convert to Wh

        st.markdown(horizontal_line_style, unsafe_allow_html=True)
        st.markdown("**Netzanschluss**")
        # ToDo: distinguish static and dynamic load management
        col1, col2 = st.columns(col_share)
        with col1:
            preexisting = st.number_input(label="Vorhanden (kW)",
                                          key="grid_preexisting",
                                          min_value=0,
                                          max_value=10000,
                                          value=1000,
                                          )
        with col2:
            expansion = st.slider(label="Zusätzlich (kW)",
                                  key="grid_expansion",
                                  min_value=0,
                                  max_value=10000,
                                  value=0,
                                  step=10,
                                  )
        grid_capacity_w = Size(preexisting=preexisting * 1E3,
                                 expansion=expansion * 1E3)
        st.markdown(horizontal_line_style, unsafe_allow_html=True)

        st.markdown("**PV-Anlage**")
        col1, col2 = st.columns(col_share)
        with col1:
            preexisting = st.number_input(label="Vorhanden (kWp)",
                                          key="pv_preexisting",
                                          min_value=0,
                                          max_value=1000,
                                          value=0,
                                          )
        with col2:
            expansion = st.slider(label="Zusätzlich (kWp)",
                                  key="pv_expansion",
                                  min_value=0,
                                  max_value=1000,
                                  value=0,
                                  step=5,
                                  )
        pv_capacity_wp = Size(preexisting=preexisting * 1E3,
                              expansion=expansion * 1E3)
        st.markdown(horizontal_line_style, unsafe_allow_html=True)

        st.markdown("**Stationärspeicher**")
        col1, col2 = st.columns(col_share)
        with col1:
            preexisting = st.number_input(label="Vorhanden (kWh)",
                                          key="ess_preexisting",
                                          min_value=0,
                                          max_value=1000,
                                          value=0,
                                          )
        with col2:
            expansion = st.slider(label="Zusätzlich (kWh)",
                                  key="ess_expansion",
                                  min_value=0,
                                  max_value=1000,
                                  value=0,
                                  step=5,
                                  )
        ess_capacity_wh = Size(preexisting=preexisting * 1E3,
                               expansion=expansion * 1E3)

    return LocationSettings(coordinates=st.session_state['location'],
                            slp=slp,
                            consumption_yrl_wh=consumption_yrl_wh,
                            grid_capacity_w=grid_capacity_w,
                            pv_capacity_wp=pv_capacity_wp,
                            ess_capacity_wh=ess_capacity_wh,
                            )


def _get_params_economic() -> EconomicSettings:
    with st.sidebar.expander(label="**Wirtschaftliche Parameter**",
                             icon="💶"):
        return EconomicSettings(
            opex_spec_grid_buy_eur_per_wh=st.slider("Strombezugskosten (EUR/kWh)", 0.00, 1.00, 0.20, 0.01) * 1E-3,
            opex_spec_grid_sell_eur_per_wh=st.slider("Einspeisevergütung (EUR/kWh)", 0.00, 1.00, 0.20, 0.01) * 1E-3,
            opex_spec_grid_peak_eur_per_wp=st.slider("Leistungspreis (EUR/kWp)", 0, 300, 50, 1) * 1E-3,
            fuel_price_eur_liter=st.slider("Dieselkosten (EUR/l)", 1.00, 2.00, 1.50, 0.05),
            toll_icev_eur_km=st.slider("Mautkosten für ICET (EUR/km)", 0.10, 1.00, 0.27, 0.01),
            toll_bev_eur_km=0.0,
            mntex_bev_eur_km=st.slider("Wartung BET (EUR/km)", 0.05, 1.00, 0.13, 0.01),
            mntex_icev_eur_km=st.slider("Wartung ICET (EUR/km)", 0.05, 1.00, 0.18, 0.01),
            insurance_pct=st.slider("Versicherung (%*Anschaffungspreis)", 0.1, 10.0, 2.0, 0.1),
            salvage_bev_pct=st.slider("Restwert BET (%)", 10, 80, 25, 1),
            salvage_icev_pct=st.slider("Restwert ICET (%)", 10, 80, 27, 1),
        )


def _get_params_subfleet(subfleet: SubFleetDefinition) -> SubFleetSettings:
    with st.sidebar.expander(label=f'**{subfleet.label}**  \n{subfleet.weight_max_str}',
                             icon=subfleet.icon,
                             expanded=False):
        num_total = st.number_input(label="Fahrzeuge gesamt",
                                    key=f'num_{subfleet.name}',
                                    min_value=0,
                                    max_value=5,
                                    value=0,
                                    step=1,
                                    )

        col1, col2 = st.columns(2)
        with col1:
            num_bev_preexisting = st.number_input("Vorhandene E-Fahrzeuge",
                                                  key=f'num_bev_preexisting_{subfleet.name}',
                                                  min_value=0,
                                                  max_value=num_total,
                                                  value=0,
                                                  step=1,
                                                  )
        with col2:
            num_bev_expansion = st.number_input("Zusätzliche E-Fahrzeuge",
                                                key=f'num_bev_expansion_{subfleet.name}',
                                                min_value=0,
                                                max_value=num_total - num_bev_preexisting,
                                                value=0,
                                                step=1,
                                                )

        col1, col2 = st.columns([3, 7])
        with col1:
            charger = st.selectbox(label="Ladepunkt",
                                   key=f'charger_{subfleet.name}',
                                   options=[x.name for x in CHARGERS.values()])
        with col2:
            max_value = CHARGERS[charger.lower()].settings_pwr_max.max_value
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

    return SubFleetSettings(
        name=subfleet.name,
        num_total=num_total,
        num_bev_preexisting=num_bev_preexisting,
        num_bev_expansion=num_bev_expansion,
        battery_capacity_wh=subfleet.battery_capactiy_wh,
        capex_bev_eur=capex_bev_eur,
        capex_icev_eur=capex_icev_eur,
        toll_share_pct=toll_share_pct,
        charger=charger,
        pwr_max_w=pwr_max_w,

        # dist_max_daily_km=dist_max_km,
        # depot_time_h=depot_avg_h,
        # load_avg_t=load_avg_t,
        # weight_empty_bev_kg=subfleet.weight_empty_bev,
        # weight_empty_icev_kg=subfleet.weight_empty_icev,
    )



def _get_params_charger(charger: ChargerDefinition) -> ChargerSettings:
    with st.sidebar.expander(label=f'**{charger.name}-Ladepunkte**',
                             icon=charger.icon,
                             expanded=False):

        col1, col2 = st.columns([3, 7])
        with col1:
            num_preexisting = st.number_input(label="Vorhandene",
                                              key=f'chg_{charger.name.lower()}_preexisting',
                                              **charger.settings_preexisting.dict
                                              )
        with col2:
            num_expansion = st.slider(label="Zusätzliche",
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

        return ChargerSettings(name=charger.name,
                               num_preexisting=num_preexisting,
                               num_expansion=num_expansion,
                               pwr_max_w=pwr_max_w,
                               cost_per_charger_eur=cost_per_charger_eur)


def get_input_params() -> Settings:
    col1, col2 = st.sidebar.columns([6, 4])
    with col1:
        auto_refresh = st.toggle("**Automatisch aktualisieren**",
                                 value=True)
    with col2:
        if auto_refresh:
            st.session_state["run_backend"] = True
        else:
            button_calc_results = st.button("**Berechnen**", icon="🚀")
            if button_calc_results:
                st.session_state["run_backend"] = True

    # get depot parameters
    st.sidebar.subheader("Allgemeine Parameter")
    location_settings = _get_params_location()

    # get economic parameters
    economic_settings = _get_params_economic()

    # get fleet parameters
    st.sidebar.subheader("Flotte")
    fleet_settings = {}
    for subfleet in SUBFLEETS.values():
        fleet_settings[subfleet.name] = _get_params_subfleet(subfleet)

    # get charging infrastructure parameters
    st.sidebar.subheader("Ladeinfrastruktur")
    charger_settings = {}
    for charger in CHARGERS.values():
        charger_settings[charger.name] = _get_params_charger(charger)

    return Settings(location=location_settings,
                    subfleets=fleet_settings,
                    chargers=charger_settings,
                    economic=economic_settings
                    )

years = TIME_PRJ_YRS


def opex18(pr):
    od = pr.opex_breakdown  # erwartet Dict mit Keys wie unten
    return {
        "Stromeinkauf": float(od.get("Stromeinkauf", 0.0)) * years,
        "Stromverkauf": float(od.get("Stromverkauf", 0.0)) * years,  # negativ = Erlös
        "Diesel": float(od.get("Diesel", 0.0)),  # schon 18y aggregiert
        "Maut": float(od.get("Maut", 0.0)),
        "Wartung": float(od.get("Wartung", 0.0)),
        "Versicherung": float(od.get("Versicherung", 0.0)),
    }


# Ladeinfrastruktur/Netz/PV/Speicher
def infra_row(pr, key):
    return float(pr.infra_capex_breakdown.get(key, 0.0))


def co2_infra_18(pr):
    bd = pr.infra_co2_breakdown or {}
    # ESS wird in deinem Flow in Jahr 0 und Jahr 8 angesetzt → multipliziere bei 18 Jahren mit 2
    ess_cycles = 1 + (1 if (years > 8 and float(bd.get("ess", 0.0)) > 0.0) else 0)

    rows = {}
    # Charger: AC/DC, falls vorhanden – sonst "gesamt"
    ac = float(bd.get("chargers_ac", 0.0))
    dc = float(bd.get("chargers_dc", 0.0))
    ch_total = float(bd.get("chargers", 0.0))
    if ac > 0 or dc > 0:
        rows["Ladeinfrastruktur – AC"] = ac
        rows["Ladeinfrastruktur – DC"] = dc
    elif ch_total > 0:
        rows["Ladeinfrastruktur (gesamt)"] = ch_total

    rows["Netzanschluss"] = float(bd.get("grid", 0.0))
    rows["PV"] = float(bd.get("pv", 0.0))
    rows["Speicher"] = float(bd.get("ess", 0.0)) * ess_cycles
    return rows


def co2_oper_18(pr):
    return {
        "Strom (Betrieb)": float(pr.co2_grid_yrl_kg) * years,
        "Tailpipe (Betrieb)": float(pr.co2_tailpipe_yrl_kg) * years,
    }


def co2_oper_18(pr):
    return {
        "Strom (Betrieb)": float(pr.co2_grid_yrl_kg) * years,
        "Tailpipe (Betrieb)": float(pr.co2_tailpipe_yrl_kg) * years,
    }


def display_results(results):
    st.subheader("Ergebnisse")
    st.success(f"Berechnung erfolgreich!")

    st.subheader("Baseline vs. Expansion")
    col1, col2, col3, col4 = st.columns(4)

    def centered_h4(text: str) -> None:
        st.markdown(f"<h5 style='text-align:center; margin:0'>{text}</h5>", unsafe_allow_html=True)

    with col1:
        centered_h4("Gesamtkosten")
        st.altair_chart(
            make_comparison_chart_discrete_values(
                float(np.cumsum(results.baseline.cashflow)[-1]),
                float(np.cumsum(results.expansion.cashflow)[-1]),
                unit="EUR",
                abs_title="Kosten"
            ),
            use_container_width=True
        )

    with col2:
        centered_h4("Gesamt-CO₂")
        st.altair_chart(
            make_comparison_chart_discrete_values(
                float(np.cumsum(results.baseline.co2_flow)[-1]),
                float(np.cumsum(results.expansion.co2_flow)[-1]),
                unit="kg-CO₂",
                abs_title="Emissionen"
            ),
            use_container_width=True
        )

    with col3:
        centered_h4("Eigenverbrauchsquote")
        st.altair_chart(
            make_comparison_chart(
                results.baseline.self_consumption_pct / 100,
                results.expansion.self_consumption_pct / 100
            ),
            use_container_width=True
        )

    with col4:
        centered_h4("Autarkiegrad")
        st.altair_chart(
            make_comparison_chart(
                results.baseline.self_sufficiency_pct / 100,
                results.expansion.self_sufficiency_pct / 100
            ),
            use_container_width=True
        )

    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown("#### Kumulierte Kosten")
        plot_flow(results, attr="cashflow",
                  y_label="Cumulative cash outflow [EUR]",
                  unit="EUR",
                  cumulative=True,
                  show_table=False)
    with col2:
        st.write("")
        st.write("")
        st.write("")
        res = find_flow_intersection(results, attr="cashflow")
        st.markdown("#### Preisparität")
        if res is None:
            st.markdown("Kein Schnittpunkt")
        elif res["kind"] == "identical":
            st.markdown("Kurven identisch.")
        else:
            yr = res["year_float"]
            st.markdown(f"Schnittpunkt bei {yr:.2f} Jahren")
        st.markdown("#### Kosten-Delta")
        cost_delta = float(np.cumsum(results.expansion.cashflow)[-1]) - float(np.cumsum(results.baseline.cashflow)[-1])
        st.markdown(f"{cost_delta:,.0f} EUR nach 18 Jahren")

    with st.expander("#### Verlauf CO₂-Ausstoß"):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown("#### Kumulierter CO₂-Ausstoß")
            plot_flow(results, attr="co2_flow",
                      y_label="Cumulative CO₂ [EUR]",
                      unit="kg-CO₂",
                      cumulative=True,
                      show_table=False)
        with col2:
            st.write("")
            st.write("")
            st.write("")
            res = find_flow_intersection(results, attr="co2_flow")
            st.markdown("#### CO₂-Parität")
            if res is None:
                st.markdown("Kein Schnittpunkt")
            elif res["kind"] == "identical":
                st.markdown("Kurven identisch.")
            else:
                yr = res["year_float"]
                st.markdown(f"Schnittpunkt bei {yr:.2f} Jahren")
            st.markdown("#### CO₂-Delta")
            cost_delta = float(np.cumsum(results.expansion.co2_flow)[-1]) - float(
                np.cumsum(results.baseline.co2_flow)[-1])
            st.markdown(f"{cost_delta:,.0f} kg-CO₂ nach 18 Jahren")

    with st.expander("Zusammensetzung der Kosten"):
        rb = results.baseline
        re = results.expansion

        # --- KOSTEN ---
        cost_base = build_cost_breakdown_18y(rb, phase="baseline", project_years=TIME_PRJ_YRS)  # dict
        cost_exp = build_cost_breakdown_18y(re, phase="expansion", project_years=TIME_PRJ_YRS)  # dict

        # DataFrame für Tabelle/Chart
        df_cost = pd.DataFrame({"Baseline": cost_base, "Expansion": cost_exp}).T
        st.markdown("### Kosten über 18 Jahre Laufzeit")

        # --- Summen berechnen ---
        total_cost_base = float(sum(map(float, cost_base.values())))
        total_cost_exp = float(sum(map(float, cost_exp.values())))
        delta_cost = total_cost_exp - total_cost_base

        c1, c2 = st.columns(2)
        c1.metric("Gesamtkosten – Baseline (18 Jahre)", f"{total_cost_base:,.0f} €")
        c2.metric("Gesamtkosten – Expansion (18 Jahre)", f"{total_cost_exp:,.0f} €",
                  delta=f"{delta_cost:+,.0f} €")

        # Stacked-Bar
        df_cost_long = df_cost.reset_index(names="Scenario").melt(
            id_vars="Scenario", var_name="Komponente", value_name="EUR"
        )
        chart_cost = alt.Chart(df_cost_long).mark_bar().encode(
            x=alt.X("Scenario:N", title=None),
            y=alt.Y("EUR:Q", title="EUR (18 Jahre)"),
            color=alt.Color("Komponente:N", title=None),
            tooltip=[alt.Tooltip("Komponente:N"), alt.Tooltip("EUR:Q", format=",.0f")]
        ).properties(height=280)
        st.altair_chart(chart_cost, use_container_width=True)

        with st.expander("OPEX-Breakdown (18 Jahre)"):
            rb = results.baseline
            re = results.expansion

            # Summiere OPEX-Breakdown über die Projektlaufzeit (jährliche Anteile * 18)
            # Achtung: Deine vehicle-Komponenten sind bereits 18y-aggregiert (aus calc_opex_vehicle),
            # Strom-Ein-/Verkauf und Peak-Kosten sind jährlich; wir wollen nur Ein-/Verkauf laut Anforderung.
            years = TIME_PRJ_YRS

            opex_base = opex18(rb)
            opex_exp = opex18(re)

            df_opex = pd.DataFrame({"Baseline": opex_base, "Expansion": opex_exp}).T
            st.dataframe(df_opex.style.format("{:,.0f}"))

        with st.expander("CAPEX-Breakdown (18 Jahre)"):
            rb = results.baseline
            re = results.expansion

            # Fahrzeuge je Klasse (BEV+ICEV)
            veh_rows = sorted(set(rb.capex_vehicles_by_subfleet) | set(re.capex_vehicles_by_subfleet))

            def capex_table(pr):
                rows = {}
                # Fahrzeuge je Klasse
                for sf in veh_rows:
                    label = f"Fahrzeuge – {sf.upper()}"
                    rows[label] = float(pr.capex_vehicles_by_subfleet.get(sf, 0.0))
                # Infrastruktur
                rows["Ladeinfrastruktur – AC"] = infra_row(pr, "chargers_ac")
                rows["Ladeinfrastruktur – DC"] = infra_row(pr, "chargers_dc")
                rows["Netzanschluss"] = infra_row(pr, "grid")
                rows["PV"] = infra_row(pr, "pv")
                rows["Speicher"] = infra_row(pr, "ess")
                return rows

            capex_base = capex_table(rb)
            capex_exp = capex_table(re)

            df_capex = pd.DataFrame({"Baseline": capex_base, "Expansion": capex_exp}).T
            st.dataframe(df_capex.style.format("{:,.0f}"))

    with st.expander("Zusammensetzung des CO₂-Ausstoß"):
        # details co2
        rb = results.baseline
        re = results.expansion

        # Dicts mit Komponenten (z.B. Betrieb, Tailpipe, Fahrzeuge, Infrastruktur)
        co2_base = build_co2_breakdown_18y(rb, phase="baseline", project_years=TIME_PRJ_YRS)
        co2_exp = build_co2_breakdown_18y(re, phase="expansion", project_years=TIME_PRJ_YRS)

        df_co2 = pd.DataFrame({"Baseline": co2_base, "Expansion": co2_exp}).T

        st.markdown("### CO₂ über 18 Jahre Laufzeit")

        # Gesamtsummen über alle Komponenten
        total_co2_base = float(sum(map(float, co2_base.values())))
        total_co2_exp = float(sum(map(float, co2_exp.values())))
        delta_co2 = total_co2_exp - total_co2_base

        c1, c2 = st.columns(2)
        c1.metric("Gesamt-CO₂ – Baseline (18 Jahre)", f"{total_co2_base:,.0f} kg CO₂")
        c2.metric("Gesamt-CO₂ – Expansion (18 Jahre)", f"{total_co2_exp:,.0f} kg CO₂",
                  delta=f"{delta_co2:+,.0f} kg CO₂")

        df_co2_long = df_co2.reset_index(names="Scenario").melt(id_vars="Scenario", var_name="Komponente",
                                                                value_name="kg CO₂")
        chart_co2 = alt.Chart(df_co2_long).mark_bar().encode(
            x=alt.X("Scenario:N", title=None),
            y=alt.Y("kg CO₂:Q", title="kg CO₂ (18 Jahre)"),
            color=alt.Color("Komponente:N", title=None),
            tooltip=[alt.Tooltip("Komponente:N"), alt.Tooltip("kg CO₂:Q", format=",.0f")]
        ).properties(height=280)
        st.altair_chart(chart_co2, use_container_width=True)

        def co2_vehicles_18(pr):
            bev = {str(k).lower(): float(v) for k, v in (pr.vehicles_co2_production_breakdown_bev or {}).items()}
            ice = {str(k).lower(): float(v) for k, v in (pr.vehicles_co2_production_breakdown_icev or {}).items()}
            keys = sorted(set(bev) | set(ice))
            return {f"Fahrzeuge – {k.upper()}": (bev.get(k, 0.0) + ice.get(k, 0.0)) * N_VEH_COHORTS for k in keys}

        VEH_REPL_IDX = [0, 5, 11]
        N_VEH_COHORTS = sum(1 for i in VEH_REPL_IDX if i < years)  # -> 3 bei 18 Jahren

        co2_oper_base = co2_oper_18(rb)
        co2_oper_exp = co2_oper_18(re)

        df_co2_oper = pd.DataFrame({"Baseline": co2_oper_base, "Expansion": co2_oper_exp}).T

        with st.expander("CO₂ im Betrieb"):
            st.dataframe(df_co2_oper.style.format("{:,.0f}"))
            co2_veh_base = co2_vehicles_18(rb)
            co2_veh_exp = co2_vehicles_18(re)

        with st.expander("CO₂ durch die Herstellung"):
            df_co2_veh = pd.DataFrame({"Baseline": co2_veh_base, "Expansion": co2_veh_exp}).T
            st.markdown("Fahrzeuge")
            st.dataframe(df_co2_veh.style.format("{:,.0f}"))

            co2_infra_base = co2_infra_18(rb)
            co2_infra_exp = co2_infra_18(re)

            df_co2_infra = pd.DataFrame({"Baseline": co2_infra_base, "Expansion": co2_infra_exp}).T
            st.markdown("Infrastruktur")
            st.dataframe(df_co2_infra.style.format("{:,.0f}"))


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
    results,
    attr: str = "cashflow",         # "cashflow" or "co2_flow"
    y_label: str | None = None,
    unit: str = "EUR",
    cumulative: bool = True,
    show_table: bool = False,
    value_format: str = ",.0f",
    show_intersection: bool = True,  # show intersection marker
):
    # 1) Pull arrays
    base_arr = np.asarray(getattr(results.baseline, attr, np.array([])), dtype=float)
    exp_arr  = np.asarray(getattr(results.expansion, attr, np.array([])), dtype=float)

    if base_arr.size == 0 and exp_arr.size == 0:
        st.warning(f"No array '{attr}' found on results.baseline / results.expansion.")
        return

    # 2) Pad to the same length
    n_years = int(max(len(base_arr), len(exp_arr)))
    pad = lambda a: np.pad(a, (0, n_years - len(a)), mode="constant") if len(a) < n_years else a
    base_arr = pad(base_arr)
    exp_arr  = pad(exp_arr)
    years = np.arange(1, n_years + 1, dtype=float)  # float to support fractional years (interpolation)

    # 3) Cumulative or annual
    y_base = np.cumsum(base_arr) if cumulative else base_arr
    y_exp  = np.cumsum(exp_arr)  if cumulative else exp_arr

    # 4) Build DataFrame
    df = pd.DataFrame({
        "Year": years,
        "Baseline": y_base,
        "Expansion": y_exp,
    })
    df_long = df.melt(id_vars="Year", var_name="Scenario", value_name="Value")

    # 5) Robust axis domain (handles all-zero or negative values)
    y_min = float(min(np.nanmin(y_base), np.nanmin(y_exp), 0.0))
    y_max = float(max(np.nanmax(y_base), np.nanmax(y_exp), 0.0))
    if y_max == y_min:
        y_min, y_max = 0.0, 1.0

    # 6) Defaults for title/labels
    if y_label is None:
        base_lbl = "Cumulative" if cumulative else "Annual"
        y_label = f"{base_lbl} value [{unit}]"

    # 7) Build the base line chart first
    chart = (
        alt.Chart(df_long)
        .mark_line(point=True)
        .encode(
            x=alt.X("Year:Q", axis=alt.Axis(title="Year")),
            y=alt.Y("Value:Q", axis=alt.Axis(title=y_label), scale=alt.Scale(domain=[y_min, y_max])),
            color=alt.Color(
                "Scenario:N",
                legend=alt.Legend(title="Scenario"),
                scale=alt.Scale(domain=["Baseline", "Expansion"],
                                range=[COLOR_BL, COLOR_EX])
            ),
            tooltip=[
                alt.Tooltip("Year:Q", title="Year", format=".2f"),
                alt.Tooltip("Scenario:N", title="Scenario"),
                alt.Tooltip("Value:Q", title=f"Value [{unit}]", format=value_format),
            ],
        )
        .properties(height=360)
    )

    # 8) Optional: add intersection marker
    if show_intersection:
        res = find_flow_intersection(results, attr=attr)
        if isinstance(res, dict) and res.get("value") is not None:
            x_pt = float(res["year_float"])
            y_pt = float(res["value"])
            point_df = pd.DataFrame([{"Year": x_pt, "Value": y_pt}])
            point_layer = (
                alt.Chart(point_df)
                .mark_point(size=120, filled=True)
                .encode(x="Year:Q", y="Value:Q")
            )
            chart = chart + point_layer

    st.altair_chart(chart, use_container_width=True)

    # 9) Optional table
    if show_table:
        st.dataframe(df.style.format({"Baseline": value_format, "Expansion": value_format}))


def find_flow_intersection(results, attr: str = "cashflow"):
    """
    Find the intersection of the cumulative curves (Baseline vs. Expansion)
    for the given array attribute (e.g., 'cashflow' or 'co2_flow').

    Returns:
      - dict describing the intersection:
          kind: 'exact' | 'interp' | 'identical'
          year_float: 1-indexed year as float (can be between two years)
          value: value at the intersection (same for both curves)
        or
      - None if there is no intersection.
    """
    base = np.asarray(getattr(results.baseline, attr, np.array([])), dtype=float)
    exp  = np.asarray(getattr(results.expansion, attr, np.array([])), dtype=float)

    if base.size == 0 or exp.size == 0:
        return None

    # Pad to same length
    n = int(max(len(base), len(exp)))
    pad = lambda a: np.pad(a, (0, n - len(a)), mode="constant") if len(a) < n else a
    base = pad(base)
    exp  = pad(exp)

    base_cum = np.cumsum(base)
    exp_cum  = np.cumsum(exp)
    diff = exp_cum - base_cum

    # Identical curves
    if np.allclose(diff, 0.0, atol=1e-12):
        return {
            "kind": "identical",
            "year_float": None,
            "value": None,
            "message": "Curves are identical across the entire horizon."
        }

    # Exact intersection at a support point
    for i in range(n):
        if np.isclose(diff[i], 0.0, atol=1e-12):
            return {
                "kind": "exact",
                "year_float": i + 1.0,  # 1-indexed
                "value": float(base_cum[i]),
                "index": i
            }

    # Sign change => intersection between two years (linear interpolation)
    for i in range(1, n):
        if diff[i-1] * diff[i] < 0:
            t = (-diff[i-1]) / (diff[i] - diff[i-1])  # 0..1 within the interval
            year_float = (i - 1) + t + 1.0            # 1-indexed
            value = float(base_cum[i-1] + t * (base_cum[i] - base_cum[i-1]))
            return {
                "kind": "interp",
                "year_float": year_float,
                "value": value,
                "between_years": (i, i+1),            # 1-indexed
                "t": float(t)
            }

    # No intersection (one curve stays above/below the other)
    return None

def make_ring(
    phase: str,
    value: float,
    radius: float,
    thickness: float,
    color: str,
    abs_total: float | None = None,
    abs_title: str = "Absolut",
    abs_format: str | None = ",.0f",
    unit: str = "",
    co2_eur_per_t: float | None = None,   # <- NEW: show CO₂ costs if unit is CO2 and this is set
) -> alt.Chart:
    """Draw a ring with percent in the middle; tooltips can include absolute values and optional CO₂ costs."""
    bg = (
        alt.Chart(pd.DataFrame({"value": [1]}))
        .mark_arc(innerRadius=radius, outerRadius=radius + thickness, color=color, opacity=0.4, tooltip=None)
        .encode(theta=alt.Theta("value:Q", stack=True))
    )

    # Build data for tooltip
    data = {"value": [value], "phase": [phase], "percent": [value * 100]}
    if abs_total is not None:
        abs_val = value * abs_total
        data["abs_val"] = [abs_val]
        # If this is CO2, compute cost = kg / 1000 * EUR/t
        if unit and "co2" in unit.lower().replace("-", "").replace(" ", "") and co2_eur_per_t is not None:
            data["co2_cost_eur"] = [abs_val / 1000.0 * float(co2_eur_per_t)]

    # Tooltips
    tooltips = [
        alt.Tooltip("phase:N",   title="Szenario"),
        alt.Tooltip("percent:Q", title="Anteil", format=".1f"),
    ]
    if abs_total is not None:
        title_abs = abs_title + (f" [{unit}]" if unit else "")
        tooltips.append(alt.Tooltip("abs_val:Q", title=title_abs, format=(abs_format or ",.0f")))
        if "co2_cost_eur" in data:
            tooltips.append(alt.Tooltip("co2_cost_eur:Q", title="CO₂-Kosten [EUR]", format=",.0f"))

    fg = (
        alt.Chart(pd.DataFrame(data))
        .mark_arc(innerRadius=radius, outerRadius=radius + thickness, cornerRadius=2, color=color)
        .encode(theta=alt.Theta("value:Q", stack=True), tooltip=tooltips)
    )
    return bg + fg

def make_comparison_chart(val_baseline,
                          val_expansion,
                          ):

    # Create rings
    baseline_ring = make_ring('Baseline', val_baseline, 40, 30, COLOR_BL)
    expansion_ring = make_ring('Expansion', val_expansion, 80, 30, COLOR_EX)

    # Center text (single-row dataframe, minimal overhead)
    center_text = alt.Chart(pd.DataFrame({"text": [f"{(val_expansion - val_baseline) * 100:+.1f} %"]})).mark_text(
        size=20,
        fontWeight="bold",
        color="green" if val_expansion > val_baseline else "red",
        tooltip=None
    ).encode(
        text="text:N"
    )
    # Combine chart
    chart = (baseline_ring + expansion_ring + center_text).properties(width=300, height=300)

    return chart

def make_comparison_chart_discrete_values(
    val_baseline: float,
    val_expansion: float,
    unit: str | None = None,
    abs_title: str = "Absolut",
    abs_format: str | None = None,
) -> alt.Chart:
    max_val = max(float(val_baseline), float(val_expansion), 1e-9)
    base_ratio = float(val_baseline) / max_val
    exp_ratio  = float(val_expansion) / max_val

    if abs_format is None:
        abs_format = ",.0f"

    # Use 55 EUR/t only for CO2 units
    co2_price = 55.0 if (unit and "co2" in unit.lower().replace("-", "").replace(" ", "")) else None

    baseline_ring = make_ring(
        phase="Baseline", value=base_ratio, radius=40, thickness=30, color=COLOR_BL,
        abs_total=max_val, abs_title=abs_title, abs_format=abs_format, unit=(unit or ""), co2_eur_per_t=co2_price
    )
    expansion_ring = make_ring(
        phase="Expansion", value=exp_ratio, radius=80, thickness=30, color=COLOR_EX,
        abs_total=max_val, abs_title=abs_title, abs_format=abs_format, unit=(unit or ""), co2_eur_per_t=co2_price
    )

    # Center text = relative change vs baseline in %
    if val_baseline > 0:
        diff_pct = (val_expansion / val_baseline - 1.0) * 100.0
    else:
        diff_pct = 0.0

    center_text = (
        alt.Chart(pd.DataFrame({"text": [f"{diff_pct:+.1f} %"]}))
        .mark_text(size=20, fontWeight="bold", color="green" if val_expansion < val_baseline else "red", tooltip=None)
        .encode(text="text:N")
    )

    return (baseline_ring + expansion_ring + center_text).properties(width=300, height=300)


def _num_waves(project_years: int) -> int:
    return sum(1 for i in VEH_YEARS_IDX if i < project_years)

def build_cost_breakdown_18y(pr, phase: str, project_years: int) -> dict[str, float]:
    """
    Baut die 18-Jahres-Kosten so auf, wie der cashflow gefüllt wird:
    - Jährliches OPEX: (opex_grid + opex_vehicle) * Jahre
    - CAPEX Fahrzeuge: capex_vehicles_eur je Welle (0/5/11)
    - CAPEX Infra: in Expansion Jahr 0 voll (grid+pv+chargers), ESS in Jahr 0 und 8
    """
    waves = _num_waves(project_years)

    # jährliches OPEX (genau wie im cashflow addiert)
    opex_total_18 = (float(pr.opex_grid_eur) + float(pr.opex_vehicle_electric_secondary)) * project_years

    # Fahrzeuge CAPEX: bereits netto (Salvage schon im capex eingepreist)
    capex_veh_all_waves = float(pr.capex_vehicles_eur) * waves

    # Infrastruktur CAPEX-Timings (nur Expansion)
    if phase.lower() == "expansion":
        capex_infra_y0 = (
            float(pr.infra_capex_breakdown.get("grid", 0.0))
          + float(pr.infra_capex_breakdown.get("pv", 0.0))
          + float(pr.infra_capex_breakdown.get("chargers", 0.0))
        )
        # ESS in Jahr 0 und 8 (sofern im Projektzeitraum)
        ess_once = float(pr.infra_capex_breakdown.get("ess", 0.0))
        ess_count = (1 if project_years > 0 else 0) + (1 if project_years > 8 else 0)
        capex_infra_total = capex_infra_y0 + ess_once * ess_count
    else:
        capex_infra_total = 0.0

    # Für Balken aufschlüsseln (optional)
    return {
        "OPEX (18J)": opex_total_18,
        "CAPEX Fahrzeuge (alle Wellen)": capex_veh_all_waves,
        "CAPEX Infrastruktur (inkl. ESS-Timings)": capex_infra_total,
    }

def build_co2_breakdown_18y(pr, phase: str, project_years: int) -> dict[str, float]:
    """
    Baut die 18-Jahres-CO2-Summen so auf, wie co2_flow gefüllt wird:
    - Betrieb: co2_yrl_kg * Jahre  (Grid + Tailpipe)
    - Fahrzeuge Herstellung: pro Welle (0/5/11)
    - Infrastruktur Herstellung: in Expansion Jahr 0 (grid+pv+chargers), ESS in Jahr 0 und 8
    """
    waves = _num_waves(project_years)

    co2_oper_18 = float(pr.co2_yrl_kg) * project_years
    co2_veh_prod_all_waves = float(pr.vehicles_co2_production_total_kg) * waves

    if phase.lower() == "expansion":
        co2_infra_y0 = (
            float(pr.infra_co2_breakdown.get("grid", 0.0))
          + float(pr.infra_co2_breakdown.get("pv", 0.0))
          + float(pr.infra_co2_breakdown.get("chargers", 0.0))
        )
        ess_once = float(pr.infra_co2_breakdown.get("ess", 0.0))
        ess_count = (1 if project_years > 0 else 0) + (1 if project_years > 8 else 0)
        co2_infra_total = co2_infra_y0 + ess_once * ess_count
    else:
        co2_infra_total = 0.0

    return {
        "Betrieb (18J)": co2_oper_18,
        "Fahrzeuge Herstellung (alle Wellen)": co2_veh_prod_all_waves,
        "Infrastruktur Herstellung (inkl. ESS-Timings)": co2_infra_total,
    }

def run_frontend():
    # define page settings
    st.set_page_config(
        page_title="LIFT - Logistics Infrastructure & Fleet Transformation",
        page_icon="🚚",
        layout="wide"
    )

    # css styles for sidebar
    st.markdown(sidebar_style, unsafe_allow_html=True)
    st.markdown(footer_css, unsafe_allow_html=True)

    # initialize session state for backend run
    if "run_backend" not in st.session_state:
        st.session_state["run_backend"] = False

    # create sidebar and get input parameters from sidebar
    settings = get_input_params()

    st.title("LIFT - Logistics Infrastructure & Fleet Transformation")

    if st.session_state["run_backend"] is True:
        try:
            results = backend.run_backend(settings=settings)
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
                '<b>© 2025 Lehrstuhl für Fahrzeugtechnik, Technische Universität München – Alle Rechte vorbehalten  |  '
                'Demo Version  |  '
                '<a href="https://www.mos.ed.tum.de/ftm/impressum/" '
                'target="_blank" '  # open in new tab
                'rel="noopener noreferrer"'  # prevent security and privacy issues with new tab
                '>Impressum</b></a></div>', unsafe_allow_html=True)


if __name__ == "__main__":
    run_frontend()
