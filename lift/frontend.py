import streamlit as st
from streamlit_folium import st_folium
import folium
import traceback
import numpy as np
import pandas as pd
import altair as alt
from typing import Dict, Tuple, List, Literal
from definitions import TIME_PRJ_YRS


import backend

from definitions import SubFleetDefinition, ChargerDefinition, SUBFLEETS, CHARGERS

from interfaces import (
    GridPowerExceededError,
    SOCError,
    LocationSettings,
    SubFleetSettings,
    ChargerSettings,
    EconomicSettings,
    Settings, Coordinates, Size
)


footer_css = """
    <style>
        /* Style for fixed footer */
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #2c3e50;
            color: white;
            text-align: center;
            padding: 10px;
            font-size: 14px;
        }
        
        /* Remove link styling inside the footer */
        .footer a {
            color: inherit;
            text-decoration: none;
        }

        .footer a:hover {
            text-decoration: underline;
        }
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

horizontal_line_style = "<hr style='margin-top: 0.1rem; margin-bottom: 0.5rem;'>"


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
        st.markdown(horizontal_line_style, unsafe_allow_html=True)

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
            driver_wage_eur_h=st.slider("Fahrerkosten (EUR/h)", 15, 50, 26, 1),
            mntex_bev_eur_km=st.slider("Wartung BET (EUR/km)", 0.05, 1.00, 0.13, 0.01),
            mntex_icev_eur_km=st.slider("Wartung ICET (EUR/km)", 0.05, 1.00, 0.18, 0.01),
            insurance_pct=st.slider("Versicherung (%*Anschaffungspreis)", 0.1, 10.0, 2.0, 0.1),
            salvage_bev_pct=st.slider("Restwert BET (%)", 10, 80, 25, 1),
            salvage_icev_pct=st.slider("Restwert ICET (%)", 10, 80, 27, 1),
            working_days_yrl=st.slider("Arbeitstage pro Jahr", 200, 350, 250, 1)
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

        battery_capacity_wh = st.slider(label="Batteriekapazität (kWh)",
                                        key=f'battery_capacity_wh_{subfleet.name}',
                                        **subfleet.settings_battery.dict,
                                        ) * 1E3

        capex_bev_eur = st.slider(label="Anschaffungspreis BEV (EUR)",
                                  key=f'capex_bev_{subfleet.name}',
                                  **subfleet.settings_capex_bev.dict,
                                  )

        capex_icev_eur = st.slider(label="Anschaffungspreis ICEV (EUR)",
                                   key=f'capex_icev_{subfleet.name}',
                                   **subfleet.settings_capex_icev.dict,
                                   )

        # ToDo: yearly distance instead of daily distance?
        dist_avg_daily_km = st.slider(label="Tägliche Distanz/Fahrzeug (km)",
                                      key=f'dist_avg_km_{subfleet.name}',
                                      **subfleet.settings_dist_avg.dict,
                                      )

        toll_share_pct = st.slider(label="Anteil mautplichtiger Strecken (%)",
                                   key=f'toll_share_pct_{subfleet.name}',
                                   **subfleet.settings_toll_share.dict,
                                   )

        # ToDo: check whether this is required
        #dist_max_km = st.slider(label="Max. Distanz pro Fahrzeug (km)",
         #                       key=f'dist_max_km_{subfleet.name}',
         #                       **subfleet.settings_dist_max.dict,
          #                      )

        # ToDo: check whether this is required
        # depot_avg_h = st.slider(label="Standzeit am Depot (Std.)",
        #                         key=f'depot_avg_h_{subfleet.id}',
        #                         **subfleet.settings_depot_time.dict,
        #                         )

        # ToDo: check whether this is required
        # load_avg_t = st.slider(label="Durchschnittliche Beladung (t)",
        #                        key=f'load_avg_t_{subfleet.id}',
        #                        **subfleet.settings_load.dict,
        #                        )

    return SubFleetSettings(
        name=subfleet.name,
        num_total=num_total,
        num_bev_preexisting=num_bev_preexisting,
        num_bev_expansion=num_bev_expansion,
        battery_capacity_wh=battery_capacity_wh,
        capex_bev_eur=capex_bev_eur,
        capex_icev_eur=capex_icev_eur,
        dist_avg_daily_km=dist_avg_daily_km,
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


def display_results(results):
    st.subheader("Ergebnisse")
    st.success(f"Berechnung erfolgreich!")
    for col, label, attr_name in zip(st.columns(2),
                                     ['Baseline', 'Erweiterung'],
                                     ['baseline', 'expansion']):
        with col:
            st.write(f"**{label}**")
            res = getattr(results, attr_name)
            st.write(f"Autarkiegrad: {res.self_sufficiency_pct:.2f}%")
            st.write(f"Eigenverbrauchsquote: {res.self_consumption_pct:.2f}%")
            st.write(f"CAPEX: {res.capex_eur:.2f} EUR")
            st.write(f"CAPEX Fahrzeuge: {res.capex_vehicles_eur:.2f} EUR")
            st.write(f"OPEX: {res.opex_eur:.2f} EUR")
            st.write(f"OPEX Fahrzeuge (Wartung+Vers.+Fahrer): {res.opex_vehicle_electric_secondary:,.2f} EUR/Jahr")
            st.write(f"CO2-Emissionen: {res.co2_yrl_kg:.2f} kg / Jahr")
            st.write(f"CO2-Kosten: {res.co2_yrl_eur:.2f} EUR / Jahr")
            st.write(f"Infrastruktur-CO₂ (embodied): {res.infra_co2_total_kg:,.0f} kg")
            with st.expander("Details Infrastruktur-CAPEX / CO₂"):
                st.write("CAPEX Breakdown:", res.infra_capex_breakdown)
                st.write("CO₂ Breakdown (kg):", res.infra_co2_breakdown)


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
    title: str | None = None,
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
    if title is None:
        title = f"{'Cumulative' if cumulative else 'Annual'} {attr.replace('_', ' ').title()}"
    if y_label is None:
        base_lbl = "Cumulative" if cumulative else "Annual"
        y_label = f"{base_lbl} value [{unit}]"

    # 7) Build the base line chart first
    chart = (
        alt.Chart(df_long, title=title)
        .mark_line(point=True)
        .encode(
            x=alt.X("Year:Q", axis=alt.Axis(title="Year")),  # quantitative to allow fractional year mark
            y=alt.Y("Value:Q", axis=alt.Axis(title=y_label), scale=alt.Scale(domain=[y_min, y_max])),
            color=alt.Color("Scenario:N", legend=alt.Legend(title="Scenario")),
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

def _radius_to_area(r_px: float) -> float:
    # Altair's circle size uses *area* in pixels; convert a desired radius to area.
    return float(np.pi * r_px * r_px)

def circle_compare(
    title: str,
    base_value: float,
    exp_value: float,
    *,
    unit: str = "",
    max_value: float | None = None,   # shared scale for fair comparison; auto if None
    min_radius: int = 16,             # px (shown even for zero to keep a visible dot)
    max_radius: int = 72,             # px
    base_color: str = "#155e75",      # Baseline color
    exp_color: str = "#ea580c",       # Expansion color
    width: int = 520,
    height: int = 260,
    show_labels_inside: bool = True,
):
    """
    Render two size-encoded circles next to each other: Baseline vs Expansion.
    Circle *area* is proportional to the KPI value. Both use the same max_value.
    """
    # Shared scale across both circles
    if max_value is None:
        m = max(float(base_value), float(exp_value))
        max_value = (m * 1.10) if m > 0 else 1.0
    max_value = float(max_value)

    data = pd.DataFrame({
        "Scenario": ["Baseline", "Expansion"],
        "Value": [float(base_value), float(exp_value)],
        "Color": [base_color, exp_color],
        "Label": [f"{base_value:,.0f} {unit}".strip(),
                  f"{exp_value:,.0f} {unit}".strip()],
    })

    # Size scale (area in px^2)
    size_scale = alt.Scale(
        domain=[0, max_value],
        range=[_radius_to_area(min_radius), _radius_to_area(max_radius)],
        nice=False
    )

    # Baseline chart: two circles on one row
    y_center = height / 2
    chart_circles = (
        alt.Chart(data, title=title)
        .mark_circle(stroke="#0f172a", strokeWidth=1)
        .encode(
            x=alt.X("Scenario:N",
                    axis=alt.Axis(title=None, labelAngle=0),
                    sort=["Baseline", "Expansion"]),
            y=alt.value(y_center),
            size=alt.Size("Value:Q", scale=size_scale, legend=None),
            color=alt.Color("Scenario:N",
                            scale=alt.Scale(
                                domain=["Baseline", "Expansion"],
                                range=[base_color, exp_color]),
                            legend=None),
            tooltip=[
                alt.Tooltip("Scenario:N"),
                alt.Tooltip("Value:Q", title=f"Value [{unit}]", format=",.0f"),
            ],
        )
        .properties(width=width, height=height)
    )

    # Optional numeric labels inside the circles
    if show_labels_inside:
        text = (
            alt.Chart(data)
            .mark_text(baseline="middle", fontSize=14, color="white", stroke=None)
            .encode(
                x=alt.X("Scenario:N", sort=["Baseline", "Expansion"]),
                y=alt.value(y_center),
                text="Label:N",
            )
        )
        chart = chart_circles + text
    else:
        chart = chart_circles

    chart = chart.configure_view(stroke=None)
    st.altair_chart(chart, use_container_width=True)

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

            st.subheader("Baseline vs. Expansion — KPI (circle size = value)")

            # 1) Self-sufficiency (%)
            circle_compare(
                "Autarkiegrad",
                results.baseline.self_sufficiency_pct,
                results.expansion.self_sufficiency_pct,
                unit="%",
                max_value=100.0,  # fix scale to 100% for fair perception
            )

            # 2) Self-consumption (%)
            circle_compare(
                "Eigenverbrauchsquote",
                results.baseline.self_consumption_pct,
                results.expansion.self_consumption_pct,
                unit="%",
                max_value=100.0,
            )

            # 3) CAPEX (total)
            circle_compare(
                "CAPEX (total)",
                getattr(results.baseline, "capex_eur", 0.0),
                getattr(results.expansion, "capex_eur", 0.0),
                unit="EUR",
            )

            # 4) CAPEX Vehicles
            circle_compare(
                "CAPEX Fahrzeuge",
                getattr(results.baseline, "capex_vehicles_eur", 0.0),
                getattr(results.expansion, "capex_vehicles_eur", 0.0),
                unit="EUR",
            )

            # 5) OPEX (per year)
            circle_compare(
                "OPEX (per year)",
                getattr(results.baseline, "opex_eur", 0.0),
                getattr(results.expansion, "opex_eur", 0.0),
                unit="EUR",
            )

            # 6) OPEX Vehicles (maintenance + insurance + driver), per year
            circle_compare(
                "OPEX Fahrzeuge (per year)",
                getattr(results.baseline, "opex_vehicle_electric_secondary", 0.0),
                getattr(results.expansion, "opex_vehicle_electric_secondary", 0.0),
                unit="EUR",
            )

            res = find_flow_intersection(results, attr="cashflow")
            if res is None:
                st.info("Kein Schnittpunkt (eine Kurve liegt stets über/unter der anderen).")
            elif res["kind"] == "identical":
                st.info("Beide Kurven sind identisch.")
            else:
                yr = res["year_float"]
                val = res["value"]
                st.success(f"Schnittpunkt bei Jahr ≈ {yr:.2f}, Wert ≈ {val:,.0f} EUR")
            res_co2 = find_flow_intersection(results, attr="co2_flow")
            if res_co2 and res_co2.get("value") is not None:
                st.success(f"CO₂-Schnittpunkt bei Jahr ≈ {res_co2['year_float']:.2f}, "
                           f"Wert ≈ {res_co2['value']:,.0f} kg")

            plot_flow(results, attr="cashflow",
                      title="Cumulative Cash Outflow",
                      y_label="Cumulative cash outflow [EUR]",
                      unit="EUR",
                      cumulative=True,
                      show_table=False)
            plot_flow(results, attr="co2_flow",
                      title="Cumulative CO₂ Emissions",
                      y_label="Cumulative CO₂ [kg]",
                      unit="kg",
                      cumulative=True,
                      show_table=True,
                      value_format=",.0f")

            rb = results.baseline
            re = results.expansion

            st.subheader("CO₂-Bilanz")

            # Toggle: embodied Emissionen auf Projektlaufzeit umlegen?
            annualize = st.checkbox("Herstellungs-Emissionen (Fahrzeuge & Infrastruktur) auf Projektlaufzeit umlegen",
                                    value=True)
            div = TIME_PRJ_YRS if annualize else 1.0

            # Kennzahlen (Betrieb, jährlich)
            c1, c2 = st.columns(2)
            c1.metric("CO₂ Betrieb (jährlich) – Baseline", f"{rb.co2_yrl_kg:,.0f} kg")
            c2.metric("CO₂ Betrieb (jährlich) – Expansion", f"{re.co2_yrl_kg:,.0f} kg",
                      delta=f"{re.co2_yrl_kg - rb.co2_yrl_kg:,.0f} kg")

            # Tabelle & Chart: Breakdown
            def build_row(pr):
                return {
                    "Strom (jährlich)": pr.co2_grid_yrl_kg,
                    "Tailpipe (jährlich)": pr.co2_tailpipe_yrl_kg,
                    "Fahrzeuge Herstellung": pr.vehicles_co2_production_total_kg / div,
                    "Infrastruktur Herstellung": pr.infra_co2_total_kg / div,
                }

            df_co2 = pd.DataFrame({
                "Baseline": build_row(rb),
                "Expansion": build_row(re),
            }).T

            st.markdown("**Zusammensetzung**")
            st.dataframe(df_co2.style.format("{:,.0f}"))

            st.markdown("**Vergleich (gestapelt)**")
            st.bar_chart(df_co2)

            # Detail: Tailpipe je Subfleet (jährlich)
            with st.expander("Details: Tailpipe-Emissionen je Subfleet (jährlich)"):
                df_tailpipe = pd.DataFrame({
                    "Baseline": rb.co2_tailpipe_by_subfleet_kg,
                    "Expansion": re.co2_tailpipe_by_subfleet_kg,
                })
                df_tailpipe = df_tailpipe.fillna(0).astype(float)
                st.dataframe(df_tailpipe.style.format("{:,.0f}"))

            with st.expander("Details: Fahrzeug-Herstellung je Subfleet"):
                # Keys vereinheitlichen (lowercase), damit Zeilen sauber ausgerichtet werden
                def lc_dict(d):
                    return {str(k).lower(): float(v) for k, v in (d or {}).items()}

                bev_base = lc_dict(rb.vehicles_co2_production_breakdown_bev)
                ice_base = lc_dict(rb.vehicles_co2_production_breakdown_icev)
                bev_expa = lc_dict(re.vehicles_co2_production_breakdown_bev)
                ice_expa = lc_dict(re.vehicles_co2_production_breakdown_icev)

                # Vollständigen Zeilenindex über alle Subfleets bilden
                all_keys = sorted(set().union(bev_base, ice_base, bev_expa, ice_expa))
                df_prod = pd.DataFrame({
                    "BEV (Baseline)": [bev_base.get(k, 0.0) for k in all_keys],
                    "ICEV (Baseline)": [ice_base.get(k, 0.0) for k in all_keys],
                    "BEV (Expansion)": [bev_expa.get(k, 0.0) for k in all_keys],
                    "ICEV (Expansion)": [ice_expa.get(k, 0.0) for k in all_keys],
                }, index=all_keys)

                # Optional: Summenzeile ergänzen
                df_prod.loc["SUMME"] = df_prod.sum(numeric_only=True)

                st.dataframe(df_prod.style.format("{:,.0f}"))


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
