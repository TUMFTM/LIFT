import streamlit as st
# from calculation_charging import simulate_charging

from backend import run_backend

from lift.interfaces import (
    LocationSettings,
    SubFleetSettings,
    ChargingInfrastructureSettings,
    PhaseSettings,
    ElectrificationPhasesSettings,
    EconomicSettings,
)


# import plots
# import calculations as calc


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

    # define grid capacity
    grid_capacity = st.slider(label="Netzanschlussleistung (kW)",
                                      min_value=100,
                                      max_value=10000,
                                      value=1000,
                                      step=100,
                                      )

    # define existing pv capacity
    pv_capacity = st.slider(label="Bestehende PV-Leistung (kWp)",
                                    min_value=0,
                                    max_value=1000,
                                    value=0,
                                    step=5,
                                    )

    # define existing battery storage capacity
    battery_capacity = st.slider(label="Bestehende Batteriespeicher-Kapazität (kWh)",
                                         min_value=0,
                                         max_value=1000,
                                         value=0,
                                         step=5,
                                         )

    location = LocationSettings(
        latitude=latitude,
        longitude=longitude,
        consumption_annual=consumption_annual,
        grid_capacity=grid_capacity,
        pv_capacity=pv_capacity,
        battery_capacity=battery_capacity,
    )

    return location

def _show_and_get_subfleet(vehicle_type: str,
                           weight_empty_bev: float,
                           weight_empty_icev: float) -> SubFleetSettings:

    num = st.number_input(label="Anzahl Fahrzeuge gesamt",
                          min_value=0,
                          value=10,
                          step=1,
                          key=f'num_{vehicle_type}',
                          )

    num_bev = st.number_input("Anzahl Fahrzeuge elektrisch",
                              min_value=0,
                              max_value=num,
                              value=0,
                              step=1,
                              key=f'num_bev_{vehicle_type}',
                              )

    battery_capacity_kwh = st.number_input(label="Batteriekapazität (kWh)",
                                           min_value=0,
                                           max_value=750,
                                           value=500,
                                           step=5,
                                           key=f'battery_capacity_kwh_{vehicle_type}',
                                           )

    dist_max_km = st.slider(label="Max. Distanz pro Fahrzeug (km)",
                            min_value=100,
                            max_value=1000,
                            value=500,
                            step=20,
                            key=f'dist_max_km_{vehicle_type}',
                            )

    dist_avg_km = st.slider(label="Tägliche Distanz/Fahrzeug (km)",
                            min_value=100,
                            max_value=800,
                            value=400,
                            step=20,
                            key=f'dist_avg_km_{vehicle_type}',
                            )

    toll_share_pct = st.slider(label="Anteil mautplichtiger Strecken (%)",
                               min_value=0,
                               max_value=100,
                               value=90,
                               step=10,
                               key=f'toll_share_pct_{vehicle_type}',
                               )

    depot_avg_h = st.slider(label="Standzeit am Depot (Std.)",
                            min_value=0,
                            max_value=24,
                            value=8,
                            step=1,
                            key=f'depot_avg_h_{vehicle_type}',
                            )

    load_avg_t = st.slider(label="Durchschnittliche Beladung (t)",
                           min_value=0,
                           max_value=24,
                           value=12,
                           step=1,
                           key=f'load_avg_t_{vehicle_type}',
                           )

    capex_bev = st.slider(label="Anschaffungspreis BEV (EUR)",
                          min_value=100000,
                          max_value=350000,
                          value=250000,
                          step=25000,
                          key=f'capex_bev_{vehicle_type}',
                          )

    capex_icev = st.slider(label="Anschaffungspreis ICEV (EUR)",
                           min_value=100000,
                           max_value=350000,
                           value=150000,
                           step=25000,
                           key=f'capex_icev_{vehicle_type}',
                           )

    subfleet = SubFleetSettings(
        vehicle_type=vehicle_type,
        num_total=num,
        num_bev=num_bev,
        battery_capacity_kwh=battery_capacity_kwh,
        dist_max_km=dist_max_km,
        dist_avg_km=dist_avg_km,
        toll_share_pct=toll_share_pct,
        depot_time_h=depot_avg_h,
        load_avg_t=load_avg_t,
        capex_bev=capex_bev,
        capex_icev=capex_icev,
        weight_empty_bev=weight_empty_bev,
        weight_empty_icev=weight_empty_icev,
    )
    return subfleet


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

    # region define sidebar elements
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
    # ToDo: add specific values for min_value / max_value / value for slider parameters
    # ToDo: run as loop
    # Leergewicht BET: 18t, Zuladung: 24t,
    with st.sidebar.expander(label="**Schwere Lkw**  \n(42 t zulässiges Zuggesamtgewicht)", icon="🚛"):
        fleet_settings["hlt"] = _show_and_get_subfleet(vehicle_type="hlt",
                                                       weight_empty_bev=18000,
                                                       weight_empty_icev=16600,
                                                       )

    # Leergewicht: 10.4t, Zuladung: 16.6t
    with st.sidebar.expander(label="**Schwerer Verteilerverkehr**  \n(28 t zulässiges Gesamtgewicht)", icon="🚚"):
        fleet_settings["hst"] = _show_and_get_subfleet(vehicle_type="hst",
                                                       weight_empty_bev=10400,
                                                       weight_empty_icev=9000,
                                                       )

    # Leergewicht: 5.4t, Zuladung: 6.6t
    with st.sidebar.expander(label="**Urbaner Verteilerverkehr**  \n(12 t zulässiges Gesamtgewicht)", icon="🚚"):
        fleet_settings["ust"] = _show_and_get_subfleet(vehicle_type="ust",
                                                       weight_empty_bev=5400,
                                                       weight_empty_icev=5400,
                                                       )

    # Leergewicht: 2.5t, Zuladung: 1.0t
    with st.sidebar.expander(label="**Lieferwagen**  \n(3.5 t zulässiges Gesamtgewicht)", icon="🚐"):
        fleet_settings["usv"] = _show_and_get_subfleet(vehicle_type="usv",
                                                       weight_empty_bev=2500,
                                                       weight_empty_icev=2300,
                                                       )

    # C) Ladeinfrastruktur
    st.sidebar.subheader("Ladeinfrastruktur")
    dc_charger = st.sidebar.slider("aktuell verfügbare DC-Lader (>= 50 kW)", 0, 30, 2, 1)
    #ac_charger = st.sidebar.slider("Anzahl Wallboxen (>= 11 kW)", 0, 30, 2, 1)
    charger_pro_truck = st.sidebar.slider("gewünschte Ladesäulen pro BET", 0.5, 1.5, 1.0, 0.1)
    costs_per_charger = 60000

    charging_infrastructure_settings = ChargingInfrastructureSettings(
        num=dc_charger,
        num_per_vehicle=charger_pro_truck,
        cost_per_charger_eur=costs_per_charger
    )

    # C) Ziele
    st.sidebar.subheader("Elektrifizierungsziele")

    # Anzahl der Phasen auswählen
    num_phases = st.sidebar.number_input("Anzahl Elektrifizierungsphasen", min_value=1, max_value=5, value=2, step=1)

    phase_settings = list()

    # Dynamische Sliders für jede Phase
    for i in range(1, num_phases + 1):
        overall = st.sidebar.slider(
            f"Anteil Elektrifizierung nach Phase {i} (%)",
            min_value=0,
            max_value=100,
            value=min(20 + (i - 1) * 30, 100),
            step=5,
            key=f"phase_{i}"
        )

        with st.sidebar.expander(f"⚙️ Erweiterte Ziele – Phase {i}"):
            goal_hlt = st.slider(f"Schwere Lkw – Phase {i}", 0, 100, overall, 5, key=f"hlt_{i}")
            goal_hst = st.slider(f"Schwerer VV – Phase {i}", 0, 100, overall, 5, key=f"hst_{i}")
            goal_ust = st.slider(f"Urbaner VV – Phase {i}", 0, 100, overall, 5, key=f"ust_{i}")
            goal_usv = st.slider(f"Lieferwagen – Phase {i}", 0, 100, overall, 5, key=f"usv_{i}")

            package = st.radio(
                f"Paketwahl für Phase {i}",
                options=["S", "M", "L"],
                index=1,
                key=f"package_{i}",
                horizontal=True
            )
            st.write(f"S: Minimallösung")
            st.write(f"M: Ausbau Energieinfrastruktur")
            st.write(f"L: zusätzlich Vorbereitung auf nächste Ausbauphase")


        phase_settings.append(PhaseSettings(share_electric_total_pct=overall,
                                            share_electric_subfleets_pct={'hlt': goal_hlt,
                                                                          'hst': goal_hst,
                                                                          'ust': goal_ust,
                                                                          'usv': goal_usv},
                                            package=package))

    electrification_phases = ElectrificationPhasesSettings(num=num_phases,
                                                           phases=phase_settings)


    # D) Parameter zur Wirtschaftlichkeitsberechnung
    st.sidebar.write("**Wirtschaftlichkeitsberechnung**")
    service_years = st.sidebar.slider("Haltedauer (Jahre)", 1, 12, 6, 1)
    with st.sidebar.expander("⚙️ Erweiterte Einstellungen für Wirtschaftlichkeitsberechnung"):
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
        bev_mntex_eur_km=maintenance_BET,
        icev_mntex_eur_km=maintenance_ICET,
        insurance_pct=insurance,
        bev_salvage_pct=residual_value_BET,
        icev_salvage_pct=residual_value_ICET,
        workingdays_per_year=workingdays_year
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
        results = run_backend(location_settings=location_settings,
                              fleet_settings=fleet_settings,
                              charging_infrastructure_settings=charging_infrastructure_settings,
                              electrification_phases_settings=electrification_phases,
                              economic_settings=economic_settings)
        st.success(f"Berechnung erfolgreich!")
        st.session_state["run_backend"] = False

        # region tco bet
        tco_bet1 = round(
            # calc.tco_calculation_bet(avg_daily_distance1, workingdays_year, service_years, purchase_price_BET1,
            #                          costs_per_charger, residual_value_BET, electricity_cost, empty_weight_bet1,
            #                          avg_load_factor1, maintenance_BET, road_tax, toll_share1, insurance,
            #                          driver_cost),
            100,
            2)
        tco_bet2 = round(
            # calc.tco_calculation_bet(avg_daily_distance2, workingdays_year, service_years, purchase_price_BET2,
            #                          costs_per_charger, residual_value_BET, electricity_cost, empty_weight_bet2,
            #                          avg_load_factor2, maintenance_BET, road_tax, toll_share2, insurance,
            #                          driver_cost),
            100,
            2)
        tco_bet3 = round(
            # calc.tco_calculation_bet(avg_daily_distance3, workingdays_year, service_years, purchase_price_BET3,
            #                          costs_per_charger, residual_value_BET, electricity_cost, empty_weight_bet3,
            #                          avg_load_factor3, maintenance_BET, road_tax, toll_share3, insurance,
            #                          driver_cost),
            100,
            2)
        # endregion
        st.write(f"**TCO Elektro-Lkw**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"Schwere Lkw: **{tco_bet1} EUR/km**")
        with col2:
            st.write(f"Schwerer Verteilerverkehr: **{tco_bet2} EUR/km**")
        with col3:
            st.write(f"Urbaner Verteilerverkehr: **{tco_bet3} EUR/km**")

        # region tco icet
        tco_icet1 = round(
            # calc.tco_calculation_icet(avg_daily_distance1, workingdays_year, service_years, purchase_price_ICET1,
            #                           residual_value_ICET, diesel_cost, empty_weight_icet1,
            #                           avg_load_factor1, maintenance_ICET, road_tax, toll_share1, insurance,
            #                           driver_cost),
            100,
            2)
        tco_icet2 = round(
            # calc.tco_calculation_icet(avg_daily_distance2, workingdays_year, service_years, purchase_price_ICET2,
            #                           residual_value_ICET, diesel_cost, empty_weight_icet2,
            #                           avg_load_factor2, maintenance_ICET, road_tax, toll_share2, insurance,
            #                           driver_cost),
            100,
            2)
        tco_icet3 = round(
            # calc.tco_calculation_icet(avg_daily_distance3, workingdays_year, service_years, purchase_price_ICET3,
            #                           residual_value_ICET, diesel_cost, empty_weight_icet3,
            #                           avg_load_factor3, maintenance_ICET, road_tax, toll_share3, insurance,
            #                           driver_cost),
            100,
            2)
        # endregion
        st.write(f"**TCO Diesel-Lkw**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"Schwere Lkw: **{tco_icet1} EUR/km**")
        with col2:
            st.write(f"Schwerer Verteilerverkehr: **{tco_icet2} EUR/km**")
        with col3:
            st.write(f"Urbaner Verteilerverkehr: **{tco_icet3} EUR/km**")

        st.markdown("---")  # Trennlinie

        st.write("### Elektrifizierungs-Roadmap")

        # Tab-Namen dynamisch erstellen
        tab_names = ["Aktueller Stand"] + [f"Phase {i}" for i in range(1, num_phases + 1)]
        tabs = st.tabs(tab_names)

        # Aktueller Stand Tab
        with tabs[0]:
            # Balken einfügen
            bets_0 = 10  # bets_01 + bets_02 + bets_03
            fleet_size = 15 # fleet_size1 + fleet_size2 + fleet_size3
            # svg = plots.plot_goals_svg(round(bets_0/fleet_size*100))
            # st.image(svg, use_container_width=True)
            st.write('plt.svg')
            st.write("**Aktuelle Energieversorgung:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"jährliche PV-Leistung: **{1000 * 10} kWh**")
            with col2:
                st.write(f"Speicherkapazität Batteriespeicher: **{10} kWh**")
            with col3:
                st.write(f"aktueller Netzanschluss: **{1000} kW**")
            st.write("**Aktuelle Ladeinfrastruktur:**")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"DC-Lader (>= 50 kW): **{dc_charger}**")
            with col2:
                st.write(f"Verfügbare Anschlussleistung für Ladestationen: kW")
            st.write(f"**Aktuelle Elektro-Lkw Flotte:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"Schwere Lkw: **{3}**")
            with col2:
                st.write(f"Schwerer Verteilerverkehr: **{3}**")
            with col3:
                st.write(f"Urbaner Verteilerverkehr: **{4}**")

            st.write(f"**Energiebedarf für BETs (kWh)**:")
            daily_energy_need_kwh1 = 150  # calc.calculate_daily_energy_need(bets_01, avg_daily_distance1, empty_weight_bet1, avg_load_factor1)
            daily_energy_need_kwh2 = 120  # calc.calculate_daily_energy_need(bets_02, avg_daily_distance2, empty_weight_bet2, avg_load_factor2)
            daily_energy_need_kwh3 = 100 # calc.calculate_daily_energy_need(bets_03, avg_daily_distance3, empty_weight_bet3, avg_load_factor3)
            daily_energy_need_kwh = daily_energy_need_kwh1 + daily_energy_need_kwh2 + daily_energy_need_kwh3
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Täglich: **{daily_energy_need_kwh} kWh**")
            with col2:
                st.write(f"Jährlich: **{workingdays_year * daily_energy_need_kwh} kWh**")
            col1, col2, col3 = st.columns(3)
            with col1:
                # st.image(
                #     r"C:\Users\fabia\Documents\Semesterarbeit\tool\optische Elemente\battery.svg",
                #     width=100,
                #     use_container_width=False
                # )
                st.write('battery.svg')
            with col2:
                # st.image(
                #     r"C:\Users\fabia\Documents\Semesterarbeit\tool\optische Elemente\charger.svg",
                #     width=100,
                #     use_container_width=False
                # )
                st.write('charger.svg')
                st.write(f"täglich: **{daily_energy_need_kwh} kWh**")
            with col3:
                # st.image(
                #     r"C:\Users\fabia\Documents\Semesterarbeit\tool\optische Elemente\pv.svg",
                #     width=100,
                #     use_container_width=False
                # )
                st.write('pv.svg')
                st.write(f"bis täglich: **{5 * 10} kWh**")

            number_of_vehicles = 10
            battery_capacity = 500.0
            p_max = 100.0
            dt = 1.0
            grid_power = 500.0
            storage_capacity = 1000.0
            storage_soc_initial = 0.5
            p_storage_max_factor = 0.5
            csv_path = "C:/Users/fabia/Documents/Semesterarbeit/tool/truck_and_site_electrification_framework/data/simulation_result.csv"
            bet_battery_capacity = 500.0
            consumption = 1.0
            initial_soc = 1.0
            p_max_charging = 100
            consumption_building = 100
            kw_peak = 500

            # Werte kannst du anpassen!
            df_final = None
            violations = []
            # ToDo: fix
            # df_final, violations = simulate_charging(
            #     number_of_vehicles,
            #     battery_capacity,
            #     p_max,
            #     dt,
            #     grid_power,
            #     storage_capacity,
            #     storage_soc_initial,
            #     p_storage_max_factor,
            #     csv_path,
            #     bet_battery_capacity,
            #     consumption,
            #     initial_soc,
            #     p_max_charging,
            #     consumption_building,
            #     kw_peak
            # )

            if not violations:
                st.write("✅ Ja: Alle Fahrzeuge waren vor Abfahrt vollständig geladen.")
            else:
                st.write("❌ Nein: Es gab Fahrzeuge, die nicht vollständig geladen waren vor Abfahrt.")
                st.write(f"Anzahl Verstöße: {len(violations)}")
                st.write("Beispiele:")
                for v in violations[:5]:
                    st.write(f" - Fahrzeug {v['vehicle']} am {v['time']} mit SOC = {v['soc']:.2f}")

        # Dynamische Phasen-Tabs

        # for i in range(1, num_phases + 1):
        #     with tabs[i]:  # Index i für Phase i (da tabs[0] = "Aktueller Stand")
        #         # Balken einfügen
        #         number_of_new_bets1 = calc.number_of_new_bets(bets_01, fleet_size1, st.session_state[f"heavy_{i}"])
        #         number_of_new_bets2 = calc.number_of_new_bets(bets_02, fleet_size2, st.session_state[f"dist_{i}"])
        #         number_of_new_bets3 = calc.number_of_new_bets(bets_03, fleet_size3, st.session_state[f"urban_{i}"])
        #         number_of_new_bets = number_of_new_bets1 + number_of_new_bets2 + number_of_new_bets3
        #         bets_01 = number_of_new_bets1 + bets_01
        #         bets_02 = number_of_new_bets2 + bets_02
        #         bets_03 = number_of_new_bets3 + bets_03
        #         bets = bets_01 + bets_02 + bets_03
        #         daily_energy_need_kwh1 = calc.calculate_daily_energy_need(bets_01, avg_daily_distance1, empty_weight_bet1,avg_load_factor1)
        #         daily_energy_need_kwh2 = calc.calculate_daily_energy_need(bets_02, avg_daily_distance2, empty_weight_bet2, avg_load_factor2)
        #         daily_energy_need_kwh3 = calc.calculate_daily_energy_need(bets_03, avg_daily_distance3, empty_weight_bet3, avg_load_factor3)
        #         daily_energy_need_kwh = daily_energy_need_kwh1 + daily_energy_need_kwh2 + daily_energy_need_kwh3
        #         svg = plots.plot_goals_svg(round(bets/fleet_size*100))
        #         st.image(svg, use_container_width=True)
        #         st.write("**Fahrzeugflotte:**")
        #         st.write("Neue elektrische Lkw:")
        #         col1, col2, col3 = st.columns(3)
        #         with col1:
        #             st.write(f"Schwere Lkw: **{number_of_new_bets1}**")
        #         with col2:
        #             st.write(f"Schwerer Verteilerverkehr: **{number_of_new_bets2}**")
        #         with col3:
        #             st.write(f"Urbaner Verteilerverkehr **{number_of_new_bets3}**")
        #         st.write("Gesamtflotte elektrischer Lkw:")
        #         col1, col2, col3 = st.columns(3)
        #         with col1:
        #             st.write(f"Schwere Lkw: **{bets_01}**")
        #         with col2:
        #             st.write(f"Schwerer Verteilerverkehr: **{bets_02}**")
        #         with col3:
        #             st.write(f"Urbaner Verteilerverkehr **{bets_03}**")
        #         st.write("**Ladeinfrastruktur:**")
        #         col1, col2, col3 = st.columns(3)
        #         with col1:
        #             st.write(f"neue DC-Ladesäulen: **{bets - dc_charger}**")
        #         with col2:
        #             st.write(f"Gesamtbestand DC-Ladesäulen **{bets}**")
        #         with col3:
        #             st.write(f"Investitionskosten: **{number_of_new_bets * 60000} EUR**")
        #         dc_charger = bets
        #         st.write(f"**Energiebedarf**:")
        #         col1, col2 = st.columns(2)
        #         with col1:
        #             st.write(f"täglich: **{daily_energy_need_kwh} kWh**")
        #         with col2:
        #             st.write(f"jährlich: **{250 * daily_energy_need_kwh} kWh**")
        #         st.write(f"**Energieversorgung:**")
        #         col1, col2, col3 = st.columns(3)
        #         with col1:
        #             st.write(f"Zubau Energiespeicher: **kWh**")
        #         with col2:
        #             st.write(f"zusätzliche Transformatorleistung: **kW**")
        #         with col3:
        #             st.write(f"Investionskosten: EUR")
        #         st.write(f"**Einsparungen:**")
        #         col1, col2, col3, col4 = st.columns(4)
        #         with col1:
        #             st.write(f"CO2-Einsparung:")
        #         with col2:
        #             st.write(f"Einsparungen Energiekosten:")
        #         with col3:
        #             st.write(f"Mautkosten:")
        #         with col4:
        #             st.write(f"Wartungskosten:")

        # >>>>>> Hier etwas, das IMMER sichtbar bleibt <<<<<<
        st.markdown("---")  # Trennlinie
        st.header("Trends")

        # In Streamlit anzeigen
        # Anzahl der Phasen bestimmen (alles was mit "heavy_X" beginnt und eine Zahl enthält)
        phase_indices = sorted([
            int(k.split("_")[1]) for k in st.session_state.keys()
            if k.startswith("heavy_") and k.split("_")[1].isdigit()
        ])
        num_phases = len(phase_indices)

        # Phasennamen (Basis + dynamische Phasen)
        phase_names = ["Basis"] + [f"Phase {i}" for i in phase_indices]

        # Flottengrößen unverändert
        fleetsizes = [6, 6, 8]

        # Elektrifizierungsanteile für Basisphase
        phase_data = [[
            3 / 6,
            3 / 6,
            4 / 8,
        ]]

        # Elektrifizierungsanteile aus Session State dynamisch hinzufügen
        for i in phase_indices:
            phase_data.append([
                st.session_state.get(f"heavy_{i}", 0) / 100,
                st.session_state.get(f"dist_{i}", 0) / 100,
                st.session_state.get(f"urban_{i}", 0) / 100,
            ])


        tco_bet_list = [tco_bet1, tco_bet2, tco_bet3]
        tco_icet_list = [tco_icet1, tco_icet2, tco_icet3]

        """
        svg = plots.phase_plot(phase_data, fleetsizes, tco_bet_list, tco_icet_list, phase_names)
        st.image(svg, use_container_width=True)
        """
        st.write('plot.svg')

    else:
        st.warning("Bitte geben Sie die Parameter in der Seitenleiste ein und klicken Sie auf "
                   "**🚀 Ergebnisse berechnen**.")

    # Inject footer into the page
    # st.markdown(footer, unsafe_allow_html=True)
    st.markdown('<div class="footer">© 2025 Lehrstuhl für Fahrzeugtechnik, Technische Universität München – Alle Rechte vorbehalten | Demo Version</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    create_frontend()
