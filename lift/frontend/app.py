"""Streamlit frontend for LIFT.

Purpose:
- Provides the user interface to configure comparison scenarios and visualize results.
- Manages language/labels, session state, and page styling.

Relationships:
- Calls `lift.backend.comparison.comparison.run_comparison(comparison_input)` to execute the backend.
- Handles domain exceptions from `lift.backend.simulation.interfaces` to present actionable messages.
- Uses UI helpers from `lift.frontend.sidebar`, `lift.frontend.results`, `lift.frontend.design`,
  and label/version helpers from `lift.frontend.utils`.

Key Logic:
- Initializes session state and language, applies styles, and builds `comparison_input` via the sidebar.
- On run, executes the comparison, displays results, and renders a localized footer with version info.
"""

from __future__ import annotations
import os
import traceback

import numpy as np
import streamlit as st

# Flag to use streamlit caching; required to be set before importing lift.backend.utils
# Is automatically deleted after import
os.environ["LIFT_USE_STREAMLIT_CACHE"] = "1"

from lift.backend.comparison.comparison import run_comparison

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

from lift.backend.simulation.interfaces import GridPowerExceededError, SOCError

# relative imports (e.g. from .design) do not work as app.py is not run as part of the package but as standalone script
from lift.frontend.definitions import (
    DEF_GRID,
    DEF_PV,
    DEF_ESS,
    DEF_ECONOMICS,
    DEF_CHARGERS,
    DEF_SUBFLEETS,
    PERIOD_ECO,
    PERIOD_SIM,
    START_SIM,
    FREQ_SIM,
    CO2_PER_LITER_DIESEL_KG,
    OPEM_SPEC_GRID,
)
from lift.frontend.design import STYLES
from lift.frontend.interfaces import StreamlitWrapper
from lift.frontend.sidebar import create_sidebar_and_get_input
from lift.frontend.results import display_results, display_empty_results

from lift.frontend.utils import load_language, get_version, get_label

VERSION = get_version()


def display_footer():
    # Inject footer into the page
    # st.markdown(footer, unsafe_allow_html=True)
    sep = '<span style="margin: 0 20px;"> | </span>'
    st.markdown(
        '<div class="footer">'
        "<b>"
        "© 2025 "
        f"<a href='{get_label('footer.institute_url')}' "
        f"target='_blank' "  # open in new tab
        "rel='noopener noreferrer'"  # prevent security and privacy issues with new tab
        f">{get_label('footer.institute')}</a>"
        f" – {get_label('footer.rights')}"
        f"{sep}"
        f"{get_label('footer.version_prefix')}{VERSION}"
        f"{sep}"
        '<a href="https://gitlab.lrz.de/energysystemmodelling/lift" '
        'target="_blank" '  # open in new tab
        'rel="noopener noreferrer"'  # prevent security and privacy issues with new tab
        ">GitLab</a>"
        f"{sep}"
        '<a href="https://www.mos.ed.tum.de/ftm/impressum/" '
        'target="_blank" '  # open in new tab
        'rel="noopener noreferrer"'  # prevent security and privacy issues with new tab
        f">{get_label('footer.imprint')}</a>"
        '<span style="margin: 0 10px;"> </span>'
        "</b></div>",
        unsafe_allow_html=True,
    )


def run_frontend():
    # initialize session state for backend run
    for key in ["run_backend", "auto_refresh"]:
        if key not in st.session_state:
            st.session_state[key] = False

    # initialize language selection
    if "language" not in st.session_state:
        load_language(language="de")

    # define page settings
    st.set_page_config(page_title=get_label("page.title"), page_icon="🚚", layout="wide")

    # css styles for sidebar
    st.markdown(STYLES, unsafe_allow_html=True)

    app = StreamlitWrapper()
    app.sidebar = st.sidebar

    # Define sidebar structure
    app.sidebar.general = app.sidebar().container()

    # General inputs
    app.sidebar.general.title, app.sidebar.language_selection = app.sidebar.general().columns([8, 2])
    app.sidebar.general.title().subheader(get_label("sidebar.general.title"))
    app.sidebar.general.position = app.sidebar.general().expander(
        label=f"**{get_label('sidebar.general.position.title')}**",
        icon="🗺️",
    )
    app.sidebar.general.energy_system = app.sidebar.general().expander(
        label=f"**{get_label('sidebar.general.energy_system.title')}**",
        icon="💡",
    )
    app.sidebar.general.economics = app.sidebar.general().expander(
        label=f"**{get_label('sidebar.general.economics.title')}**",
        icon="💶",
    )

    # Fleet inputs
    app.sidebar.fleet = app.sidebar().container()
    app.sidebar.fleet().subheader(get_label("sidebar.fleet.title"))

    # Charging Infrastructure inputs
    app.sidebar.chargers = app.sidebar().container()
    app.sidebar.chargers().subheader(get_label("sidebar.chargers.title"))

    # Calculation
    app.sidebar.calculation = app.sidebar().container()

    # Main area
    app.main = st.container()
    app.main().title(get_label("main.title"))

    # Manual box
    app.main.manual = app.main().container()

    # Subtitle with colors as legend
    app.main.subtitle = app.main().container()

    # Space for KPI diagrams
    app.main.kpi_diagrams = app.main().container()

    # Tab structure for cost/emission results
    app.main.time_diagrams = app.main().container()
    app.main.time_diagrams.costs, app.main.time_diagrams.emissions = app.main().tabs(
        [
            get_label("main.time_diagrams.costs.tab"),
            get_label("main.time_diagrams.emissions.tab"),
        ]
    )

    # create sidebar and get input parameters from sidebar
    settings = create_sidebar_and_get_input(domain=app.sidebar)

    def get_inputs():
        return ComparisonInput(
            location=ComparisonInputLocation(
                coordinates=st.session_state.location,
                slp=st.session_state.slp.lower(),
                consumption_yrl_wh=st.session_state.consumption_yrl_wh * 1e6,
                grid=ComparisonInvestComponent(
                    capacity=ExistExpansionValue(
                        preexisting=st.session_state.grid_preexisting * DEF_GRID.settings_preexisting.factor,
                        expansion=st.session_state.grid_expansion * DEF_GRID.settings_expansion.factor,
                    ),
                    **DEF_GRID.input_dict,
                ),
                pv=ComparisonInvestComponent(
                    capacity=ExistExpansionValue(
                        preexisting=st.session_state.pv_preexisting * DEF_PV.settings_preexisting.factor,
                        expansion=st.session_state.pv_expansion * DEF_PV.settings_expansion.factor,
                    ),
                    **DEF_PV.input_dict,
                ),
                ess=ComparisonInvestComponent(
                    capacity=ExistExpansionValue(
                        preexisting=st.session_state.ess_preexisting * DEF_ESS.settings_preexisting.factor,
                        expansion=st.session_state.ess_expansion * DEF_ESS.settings_expansion.factor,
                    ),
                    **DEF_ESS.input_dict,
                ),
            ),
            economics=ComparisonInputEconomics(
                period_sim=PERIOD_SIM,
                start_sim=START_SIM,
                freq_sim=FREQ_SIM,
                discount_rate=st.session_state.eco_discount_rate * DEF_ECONOMICS.settings_discount_rate.factor,
                fix_cost_construction=st.session_state.eco_fix_cost_construction
                * DEF_ECONOMICS.settings_fix_cost_construction.factor,
                opex_spec_grid_buy=st.session_state.eco_opex_spec_grid_buy
                * DEF_ECONOMICS.settings_opex_spec_grid_buy.factor,
                opex_spec_grid_sell=st.session_state.eco_opex_spec_grid_sell
                * DEF_ECONOMICS.settings_opex_spec_grid_sell.factor,
                opex_spec_grid_peak=st.session_state.eco_opex_spec_grid_peak
                * DEF_ECONOMICS.settings_opex_spec_grid_peak.factor,
                opex_spec_route_charging=st.session_state.eco_opex_spec_route_charging
                * DEF_ECONOMICS.settings_opex_spec_route_charging.factor,
                opex_fuel=st.session_state.eco_opex_spec_fuel * DEF_ECONOMICS.settings_opex_spec_fuel.factor,
                period_eco=PERIOD_ECO,
                co2_per_liter_diesel_kg=CO2_PER_LITER_DIESEL_KG,
                opem_spec_grid=OPEM_SPEC_GRID,
            ),
            subfleets={
                k: ComparisonInputSubfleet(
                    name=k,
                    num_bev=ExistExpansionValue(
                        preexisting=st.session_state[f"num_bev_preexisting_{k}"],
                        expansion=st.session_state[f"num_bev_expansion_{k}"],
                    ),
                    num_total=st.session_state[f"num_{k}"],
                    charger=st.session_state[f"charger_{k}"].lower(),
                    pwr_max_w=st.session_state[f"pwr_max_{k}"] * 1e3,
                    capex_bev_eur=st.session_state[f"capex_bev_{k}"] * DEF_SUBFLEETS[k].settings_capex_bev.factor,
                    capex_icev_eur=st.session_state[f"capex_icev_{k}"] * DEF_SUBFLEETS[k].settings_capex_icev.factor,
                    toll_frac=st.session_state[f"toll_frac_{k}"] * DEF_SUBFLEETS[k].settings_toll_share.factor,
                    **DEF_SUBFLEETS[k].input_dict,
                )
                for k in DEF_SUBFLEETS.keys()
            },
            charging_infrastructure=ComparisonInputChargingInfrastructure(
                pwr_max_w_baseline=np.inf
                if st.session_state[f"load_mngmnt_baseline"]
                != get_label("sidebar.chargers.load_mngmnt.type.options.static")
                else st.session_state[f"load_mngmnt_slider_baseline"],
                pwr_max_w_expansion=np.inf
                if st.session_state[f"load_mngmnt_expansion"]
                != get_label("sidebar.chargers.load_mngmnt.type.options.static")
                else st.session_state[f"load_mngmnt_slider_expansion"],
                chargers={
                    k: ComparisonInputCharger(
                        name=k,
                        num=ExistExpansionValue(
                            preexisting=st.session_state[f"chg_{k}_preexisting"],
                            expansion=st.session_state[f"chg_{k}_expansion"],
                        ),
                        pwr_max_w=st.session_state[f"chg_{k}_pwr"] * 1e3,
                        cost_per_charger_eur=st.session_state[f"chg_{k}_cost"],
                        **DEF_CHARGERS[k].input_dict,
                    )
                    for k in DEF_CHARGERS.keys()
                },
            ),
        )

    if st.session_state["run_backend"] is True:
        try:
            results = run_comparison(comparison_input=get_inputs())
            display_results(results, domain=app.main)

        except GridPowerExceededError as e:
            st.error(f"{get_label('main.errors.grid')} {e}")
        except SOCError as e:
            st.error(f"{get_label('main.errors.soc')} {e}")
        except Exception as e:
            st.error(f"{get_label('main.errors.undefined')} {e}")
            st.text(traceback.format_exc())

        st.session_state["run_backend"] = False
    else:
        display_empty_results(domain=app.main.manual)

    display_footer()


if __name__ == "__main__":
    run_frontend()
