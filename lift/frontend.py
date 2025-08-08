import streamlit as st
# from streamlit_folium import st_folium
# import folium
import traceback

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
    col_share = [3, 7]

    with st.sidebar.expander(label="**Energiesystem**", icon="💡"):
        st.markdown("**Position**")
        col1, col2 = st.columns(2)
        with col1:
            latitude = st.number_input(label="Breitengrad",
                                       key="latitude",
                                       min_value=-90.0,
                                       max_value=90.0,
                                       value=48.137,
                                       step=0.001,
                                       format="%0.3f",
                                       ),
        with col2:
            longitude = st.number_input(label="Längengrad",
                                        key="longitude",
                                        min_value=-180.0,
                                        max_value=180.0,
                                        value=11.575,
                                        step=0.001,
                                        format="%0.3f",
                                        )
        coordinates=Coordinates(latitude=latitude, longitude=longitude)
        st.markdown(horizontal_line_style, unsafe_allow_html=True)
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

    return LocationSettings(coordinates=coordinates,
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
                                    key=f'num_{subfleet.vehicle_type}',
                                    min_value=0,
                                    max_value=5,
                                    value=0,
                                    step=1,
                                    )

        col1, col2 = st.columns(2)
        with col1:
            num_bev_preexisting = st.number_input("Vorhandene E-Fahrzeuge",
                                                  key=f'num_bev_preexisting_{subfleet.vehicle_type}',
                                                  min_value=0,
                                                  max_value=num_total,
                                                  value=0,
                                                  step=1,
                                                  )
        with col2:
            num_bev_expansion = st.number_input("Zusätzliche E-Fahrzeuge",
                                                key=f'num_bev_expansion_{subfleet.vehicle_type}',
                                                min_value=0,
                                                max_value=num_total - num_bev_preexisting,
                                                value=0,
                                                step=1,
                                                )

        col1, col2 = st.columns([3, 7])
        with col1:
            charger = st.selectbox(label="Ladepunkt",
                                   key=f'charger_{subfleet.vehicle_type}',
                                   options=[x.name for x in CHARGERS.values()])
        with col2:
            max_value = CHARGERS[charger.lower()].settings_pwr_max.max_value
            pwr_max_w = st.slider(label="max. Ladeleistung (kW)",
                                  key=f'pwr_max_{subfleet.vehicle_type}',
                                  min_value=0,
                                  max_value=max_value,
                                  value=max_value,
                                  ) * 1E3

        battery_capacity_wh = st.slider(label="Batteriekapazität (kWh)",
                                        key=f'battery_capacity_wh_{subfleet.vehicle_type}',
                                        **subfleet.settings_battery.dict,
                                        ) * 1E3

        capex_bev_eur = st.slider(label="Anschaffungspreis BEV (EUR)",
                                  key=f'capex_bev_{subfleet.vehicle_type}',
                                  **subfleet.settings_capex_bev.dict,
                                  )

        capex_icev_eur = st.slider(label="Anschaffungspreis ICEV (EUR)",
                                   key=f'capex_icev_{subfleet.vehicle_type}',
                                   **subfleet.settings_capex_icev.dict,
                                   )

        # ToDo: yearly distance instead of daily distance?
        dist_avg_daily_km = st.slider(label="Tägliche Distanz/Fahrzeug (km)",
                                      key=f'dist_avg_km_{subfleet.vehicle_type}',
                                      **subfleet.settings_dist_avg.dict,
                                      )

        toll_share_pct = st.slider(label="Anteil mautplichtiger Strecken (%)",
                                   key=f'toll_share_pct_{subfleet.vehicle_type}',
                                   **subfleet.settings_toll_share.dict,
                                   )

        # ToDo: check whether this is required
        dist_max_km = st.slider(label="Max. Distanz pro Fahrzeug (km)",
                                key=f'dist_max_km_{subfleet.vehicle_type}',
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

    return SubFleetSettings(
        vehicle_type=subfleet.vehicle_type,
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

        return ChargerSettings(num_preexisting=num_preexisting,
                               num_expansion=num_expansion,
                               pwr_max_w=pwr_max_w,
                               cost_per_charger_eur=cost_per_charger_eur)


def get_input_params() -> Settings:
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

    # get depot parameters
    st.sidebar.subheader("Allgemeine Parameter")
    location_settings = _get_params_location()

    # get economic parameters
    economic_settings = _get_params_economic()

    # get fleet parameters
    st.sidebar.subheader("Flotte")
    fleet_settings = {}
    for subfleet in SUBFLEETS.values():
        fleet_settings[subfleet.vehicle_type] = _get_params_subfleet(subfleet)

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
            st.write(f"OPEX: {res.opex_eur:.2f} EUR")
            st.write(f"CO2-Emissionen: {res.co2_yrl_kg:.2f} kg / Jahr")
            st.write(f"CO2-Kosten: {res.co2_yrl_eur:.2f} EUR / Jahr")


def display_empty_results():
    st.markdown("""
    Willkommen zu deinem datengetriebenen Framework für die Elektrifizierung von Lkw-Flotten 
    und die Optimierung von Speditionsstandorten. Nutze die Seitenleiste, um Parameter 
    einzugeben und klicke anschließend auf den Button, um erste Berechnungen zu starten.
    """)
    st.markdown(horizontal_line_style, unsafe_allow_html=True)
    st.warning("Bitte geben Sie die Parameter in der Seitenleiste ein und klicken Sie auf "
               "**🚀 Berechnen**.")


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
