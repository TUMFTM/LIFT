import streamlit as st
from streamlit_folium import st_folium
import folium

import backend

from definitions import SubFleetDefinition, ChargerDefinition, SUBFLEETS, CHARGERS

from interfaces import (
    LocationSettings,
    SubFleetSettings,
    ChargerSettings,
    EconomicSettings,
    Settings,
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
                min-width: 360px;
                max-width: 500px;
                width: 400px;
            }
            [data-testid="stSidebarContent"] {
                padding-right: 20px;
            }
        </style>
        """


def _show_and_get_location() -> LocationSettings:

    # define coordinates of the location
    # ToDo: use geopy (copilot suggestion) to get coordinates from address or use map picker
    # ToDo: This is what ChatGPT suggests: Figure out how to integrate this nicely
    """
    # Initial map center
    start_coords = [51.1657, 10.4515]  # Germany center
    map_obj = folium.Map(location=start_coords, zoom_start=6)

    # Let user click on map
    st.write("📍 Klicken Sie auf die Karte, um einen Standort auszuwählen.")
    click_info = st_folium(map_obj, height=500, returned_objects=["last_clicked"])

    if click_info["last_clicked"]:
        lat = click_info["last_clicked"]["lat"]
        lon = click_info["last_clicked"]["lng"]
        st.success(f"Ausgewählter Standort: {lat:.5f}, {lon:.5f}")
    """

    latitude = st.number_input(label="Breitengrad",
                                       min_value=-90.0,
                                       max_value=90.0,
                                       value=48.137,
                                       step=0.001,
                                       format="%0.3f",
                                       )

    longitude = st.number_input(label="Längengrad",
                                        min_value=-180.0,
                                        max_value=180.0,
                                        value=11.575,
                                        step=0.001,
                                        format="%0.3f",
                                        )

    # define annual energy consumption
    consumption_annual = st.slider(label="Jahresstromverbrauch (MWh)",
                                           min_value=10,
                                           max_value=1000,
                                           value=500,
                                           step=10,
                                           )

    # ToDo: choose corresponding load profile from a dropdown menu


    # define grid capacity
    # ToDo: add numeric field for preexisting size
    # ToDo: distinguish static and dynamic load management
    grid_capacity_preexisting_kw = 0
    grid_capacity_expansion_kw = st.slider(label="Netzanschlussleistung (kW)",
                                           min_value=100,
                                           max_value=10000,
                                           value=1000,
                                           step=100,
                                           )

    # define existing pv capacity
    # ToDo: add numeric field for preexisting size
    pv_capacity_preexisting_kwp = 0
    pv_capacity_expansion_kwp = st.slider(label="Bestehende PV-Leistung (kWp)",
                                          min_value=0,
                                          max_value=1000,
                                          value=0,
                                          step=5,
                                          )

    # define existing battery storage capacity
    # ToDo: add numeric field for preexisting size
    ess_capacity_preexisting_kwh = 0
    ess_capacity_expansion_kwh = st.slider(label="Bestehende Batteriespeicher-Kapazität (kWh)",
                                           min_value=0,
                                           max_value=1000,
                                           value=0,
                                           step=5,
                                           )

    location = LocationSettings(
        latitude=latitude,
        longitude=longitude,
        consumption_annual=consumption_annual,
        grid_capacity_preexisting_kw=grid_capacity_preexisting_kw,
        grid_capacity_expansion_kw=grid_capacity_expansion_kw,
        pv_capacity_preexisting_kwp=pv_capacity_preexisting_kwp,
        pv_capacity_expansion_kwp=pv_capacity_expansion_kwp,
        ess_capacity_preexisting_kwh=ess_capacity_preexisting_kwh,
        ess_capacity_expansion_kwh=ess_capacity_expansion_kwh,
    )

    return location

def _show_and_get_subfleet(subfleet: SubFleetDefinition) -> SubFleetSettings:
    with st.sidebar.expander(label=f'**{subfleet.label}**  \n{subfleet.weight_max_str}',
                             icon=subfleet.icon,
                             expanded=False):
        num_total = st.number_input(label="Anzahl Fahrzeuge gesamt",
                              key=f'num_{subfleet.id}',
                              min_value=0,
                              value=10,
                              step=1,
                              )

        # ToDo: add input for preexisting BEVs
        num_bev_preexisting = 0
        num_bev_expansion = st.number_input("Anzahl Fahrzeuge elektrisch",
                                              key=f'num_bev_{subfleet.id}',
                                              min_value=0,
                                              max_value=num_total,
                                              value=0,
                                              step=1,
                                              )

        battery_capacity_kwh = st.slider(label="Batteriekapazität (kWh)",
                                         key=f'battery_capacity_kwh_{subfleet.id}',
                                         **subfleet.settings_battery.dict,
                                         )

        capex_bev = st.slider(label="Anschaffungspreis BEV (EUR)",
                              key=f'capex_bev_{subfleet.id}',
                              **subfleet.settings_capex_bev.dict,
                              )

        capex_icev = st.slider(label="Anschaffungspreis ICEV (EUR)",
                               key=f'capex_icev_{subfleet.id}',
                               **subfleet.settings_capex_icev.dict,
                               )

        # ToDo: yearly distance instead of daily distance?
        dist_avg_daily_km = st.slider(label="Tägliche Distanz/Fahrzeug (km)",
                                      key=f'dist_avg_km_{subfleet.id}',
                                      **subfleet.settings_dist_avg.dict,
                                      )

        toll_share_pct = st.slider(label="Anteil mautplichtiger Strecken (%)",
                                   key=f'toll_share_pct_{subfleet.id}',
                                   **subfleet.settings_toll_share.dict,
                                   )

        # ToDo: check whether this is required
        dist_max_km = st.slider(label="Max. Distanz pro Fahrzeug (km)",
                                key=f'dist_max_km_{subfleet.id}',
                                **subfleet.settings_dist_max.dict,
                                )

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

    settings = SubFleetSettings(
        vehicle_type=subfleet.id,
        num_total=num_total,
        num_bev_preexisting=num_bev_preexisting,
        num_bev_expansion=num_bev_expansion,
        battery_capacity_kwh=battery_capacity_kwh,
        capex_bev=capex_bev,
        capex_icev=capex_icev,
        dist_avg_daily_km=dist_avg_daily_km,
        toll_share_pct=toll_share_pct,
        # dist_max_daily_km=dist_max_km,
        # depot_time_h=depot_avg_h,
        # load_avg_t=load_avg_t,
        # weight_empty_bev_kg=subfleet.weight_empty_bev,
        # weight_empty_icev_kg=subfleet.weight_empty_icev,
    )
    return settings


def _show_and_get_charger(charger: ChargerDefinition) -> ChargerSettings:
    with st.sidebar.expander(label=f'**{charger.id}-Ladepunkte**  \nmax. {charger.pwr_max_kw:.1f} kW',
                             icon=charger.icon,
                             expanded=False):

        num_charger_preexisting = st.slider(label="aktuell verfügbare Ladepunkte",
                                            key=f'chg_{charger.id.lower()}_preexisting',
                                            **charger.settings_preexisting.dict
                                            )
        num_charger_expansion = st.slider(label="zusätzliche Ladepunkte",
                                          key=f'chg_{charger.id.lower()}_expansion',
                                          **charger.settings_expansion.dict,
                                          )
        cost_per_charger_eur = st.slider(label="Kosten pro Ladepunkt in €",
                                         key=f'chg_{charger.id.lower()}_cost',
                                         **charger.settings_cost_per_unit_eur.dict
                                         )

        return ChargerSettings(num_preexisting=num_charger_preexisting,
                               num_expansion=num_charger_expansion,
                               pwr_max_kw=charger.pwr_max_kw,
                               cost_per_charger_eur=cost_per_charger_eur)


def create_frontend():
    # define page settings
    st.set_page_config(
        page_title="LIFT - Logistics Infrastructure & Fleet Transformation",
        page_icon="🚚",
        layout="wide"
    )

    # css styles for sidebar
    st.markdown(sidebar_style, unsafe_allow_html=True)
    st.markdown(footer_css, unsafe_allow_html=True)
    if "run_backend" not in st.session_state:
        st.session_state["run_backend"] = False

    # region define input in sidebar elements
    col1, col2 = st.sidebar.columns(2)
    # Place buttons in each column
    with col1:
        button_calc_results = st.button("**Ergebnisse berechnen**", icon="🚀")

    with col2:
        button_reset = st.button("**Eingaben zurücksetzen**", icon="🔄")

    if button_calc_results:
        st.session_state["run_backend"] = True

    if button_reset:
        # ToDo: fix this; experimental_rerun() is deprecated
        # st.session_state.clear()
        # st.experimental_rerun()
        pass

    # get depot parameters
    st.sidebar.subheader("Standort")
    with st.sidebar.expander(label="Verfügbare Optionen", icon="⚙️"):
        location_settings = _show_and_get_location()

    # get fleet parameters
    st.sidebar.subheader("Flotte")
    fleet_settings = {}

    for subfleet in SUBFLEETS.values():
        fleet_settings[subfleet.id] = _show_and_get_subfleet(subfleet)

    # get charging infrastructure parameters
    st.sidebar.subheader("Ladeinfrastruktur")
    charger_settings = {}

    for charger in CHARGERS.values():
        charger_settings[charger.id] = _show_and_get_charger(charger)

    # D) Parameter zur Wirtschaftlichkeitsberechnung
    st.sidebar.write("**Wirtschaftlichkeitsberechnung**")
    service_years = st.sidebar.slider("Haltedauer (Jahre)", 1, 12, 6, 1)
    with st.sidebar.expander(label="Erweiterte Einstellungen für Wirtschaftlichkeitsberechnung",
                             icon="⚙️"):
        electricity_cost = st.slider("Stromkosten (EUR/kWh)", 0.05, 1.00, 0.20, 0.05)
        diesel_cost = st.slider("Dieselkosten (EUR/l)", 1.00, 2.00, 1.50, 0.05)
        road_tax = st.slider("Mautkosten für ICET (EUR/km)", 0.10, 1.00, 0.27, 0.01)
        driver_cost = st.slider("Fahrerkosten (EUR/h)", 15, 50, 26, 1)
        maintenance_BET = st.slider("Wartung BET (EUR/km)", 0.05, 1.00, 0.13, 0.01)
        maintenance_ICET = st.slider("Wartung ICET (EUR/km)", 0.05, 1.00, 0.18, 0.01)
        insurance = st.slider("Versicherung (%*Anschaffungspreis)", 0.1, 10.0, 2.0, 0.1)
        residual_value_BET = st.slider("Restwert BET (%)", 10, 80, 25, 1)
        residual_value_ICET = st.slider("Restwert ICET (%)", 10, 80, 27, 1)
        workingdays_year = st.slider("Arbeitstage pro Jahr", 200, 350, 250, 1)

    economic_settings = EconomicSettings(
        period_holding_yrs=service_years,
        electricity_price_eur_kwh=electricity_cost,
        fuel_price_eur_liter=diesel_cost,
        toll_icev_eur_km=road_tax,
        toll_bev_eur_km=0.0,
        driver_wage_eur_h=driver_cost,
        mntex_bev_eur_km=maintenance_BET,
        mntex_icev_eur_km=maintenance_ICET,
        insurance_pct=insurance,
        salvage_bev_pct=residual_value_BET,
        salvage_icev_pct=residual_value_ICET,
        working_days_yrl=workingdays_year
    )

    settings = Settings(location=location_settings,
                        subfleets=fleet_settings,
                        chargers=charger_settings,
                        economic=economic_settings
    )

    # endregion


    st.title("LIFT - Logistics Infrastructure & Fleet Transformation")
    st.markdown("""
    Willkommen zu deinem datengetriebenen Framework für die Elektrifizierung von Lkw-Flotten 
    und die Optimierung von Speditionsstandorten. Nutze die Seitenleiste, um Parameter 
    einzugeben und klicke anschließend auf den Button, um erste Berechnungen zu starten.
    """)

    st.markdown("---")  # Trennlinie

    st.subheader("Ergebnisse")

    if st.session_state["run_backend"] is True:
        results = backend.run_backend(settings=settings)
        print(results)
        st.success(f"Berechnung erfolgreich!")
        st.session_state["run_backend"] = False

        # region TCO
        st.write(f"**TCO E-Fahrzeuge**")
        for col, subfleet_id, subfleet_results in zip(st.columns(len(results.subfleets)),
                                                      results.subfleets.keys(),
                                                      results.subfleets.values()):
            with col:
                st.write(f"{SUBFLEETS[subfleet_id].label}: **{subfleet_results.tco_bev:.3f} EUR/km**")


        st.write(f"**TCO Konventionelle Fahrzeuge**")
        for col, subfleet_id, subfleet_results in zip(st.columns(len(results.subfleets)),
                                                      results.subfleets.keys(),
                                                      results.subfleets.values()):
            with col:
                st.write(f"{SUBFLEETS[subfleet_id].label}: **{subfleet_results.tco_icev:.3f} EUR/km**")
        # endregion


        st.markdown("---")  # Trennlinie

    else:
        st.warning("Bitte geben Sie die Parameter in der Seitenleiste ein und klicken Sie auf "
                   "**🚀 Ergebnisse berechnen**.")

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
    create_frontend()
