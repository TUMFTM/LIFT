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
    ExistExpansionValue,
    ComparisonScenario,
    ComparisonSettings,
    ComparisonFix,
    ComparisonFixedDemand,
    ComparisonGrid,
    ComparisonPV,
    ComparisonESS,
    ComparisonFleet,
    ComparisonSubFleet,
    ComparisonChargingInfrastructure,
    ComparisonChargerType,
)

from lift.backend.evaluation.blocks import GridPowerExceededError, SOCError

# relative imports (e.g. from .design) do not work as app.py is not run as part of the package but as standalone script
from lift.frontend.definitions import DEF_GRID, DEF_PV, DEF_ESS, DEF_FLEET, DEF_CIS, DEF_SCN
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
    create_sidebar_and_get_input(domain=app.sidebar)

    def get_inputs():
        return ComparisonScenario(
            settings=ComparisonSettings(
                latitude=st.session_state.location.latitude,
                longitude=st.session_state.location.longitude,
                wacc=st.session_state.eco_discount_rate * DEF_SCN.wacc.factor,
                period_eco=int(st.session_state.eco_period * DEF_SCN.period_eco.factor),
                sim_start=DEF_SCN.sim_start,
                sim_duration=DEF_SCN.sim_duration,
                sim_freq=DEF_SCN.sim_freq,
            ),
            fix=ComparisonFix(
                capex_initial=ExistExpansionValue(
                    baseline=0.0,
                    expansion=st.session_state.eco_fix_cost_construction * DEF_SCN.capex_initial.factor,
                ),
                capem_initial=ExistExpansionValue(
                    baseline=0.0,
                    expansion=DEF_SCN.capem_initial,
                ),
            ),
            dem=ComparisonFixedDemand(
                slp=st.session_state.slp.lower(),
                e_yrl=st.session_state.consumption_yrl_wh * DEF_SCN.e_yrl.factor,
            ),
            grid=ComparisonGrid(
                capacity=ExistExpansionValue(
                    baseline=st.session_state.grid_preexisting * DEF_GRID.capacity_preexisting.factor,
                    expansion=(st.session_state.grid_preexisting + st.session_state.grid_expansion)
                    * DEF_GRID.capacity_expansion.factor,
                ),
                opex_spec_buy=st.session_state.eco_opex_spec_grid_buy * DEF_GRID.opex_spec_buy.factor,
                opex_spec_sell=st.session_state.eco_opex_spec_grid_sell * DEF_GRID.opex_spec_sell.factor,
                opex_spec_peak=st.session_state.eco_opex_spec_grid_peak * DEF_GRID.opex_spec_peak.factor,
                **DEF_GRID.values,
            ),
            pv=ComparisonPV(
                capacity=ExistExpansionValue(
                    baseline=st.session_state.pv_preexisting * DEF_PV.capacity_preexisting.factor,
                    expansion=(st.session_state.pv_preexisting + st.session_state.pv_expansion)
                    * DEF_PV.capacity_expansion.factor,
                ),
                **DEF_PV.values,
            ),
            ess=ComparisonESS(
                capacity=ExistExpansionValue(
                    baseline=st.session_state.ess_preexisting * DEF_ESS.capacity_preexisting.factor,
                    expansion=(st.session_state.ess_preexisting + st.session_state.ess_expansion)
                    * DEF_ESS.capacity_expansion.factor,
                ),
                **DEF_ESS.values,
            ),
            fleet=ComparisonFleet(
                subblocks={
                    k: ComparisonSubFleet(
                        name=k,
                        num_bev=ExistExpansionValue(
                            baseline=st.session_state[f"num_bev_preexisting_{k}"],
                            expansion=st.session_state[f"num_bev_preexisting_{k}"]
                            + st.session_state[f"num_bev_expansion_{k}"],
                        ),
                        num_icev=ExistExpansionValue(
                            baseline=st.session_state[f"num_{k}"] - st.session_state[f"num_bev_preexisting_{k}"],
                            expansion=st.session_state[f"num_{k}"]
                            - (
                                st.session_state[f"num_bev_preexisting_{k}"]
                                + st.session_state[f"num_bev_expansion_{k}"]
                            ),
                        ),
                        charger=st.session_state[f"charger_{k}"].lower(),
                        p_max=st.session_state[f"pwr_max_{k}"] * 1e3,
                        capex_per_unit_bev=st.session_state[f"capex_bev_{k}"]
                        * DEF_FLEET.subblocks[k].capex_per_unit_bev.factor,
                        capex_per_unit_icev=st.session_state[f"capex_icev_{k}"]
                        * DEF_FLEET.subblocks[k].capex_per_unit_icev.factor,
                        toll_frac=st.session_state[f"toll_frac_{k}"] * DEF_FLEET.subblocks[k].toll_frac.factor,
                        **DEF_FLEET.subblocks[k].values,
                    )
                    for k in DEF_FLEET.subblocks.keys()
                },
                opex_spec_fuel=st.session_state.eco_opex_spec_fuel * DEF_FLEET.opex_spec_fuel.factor,
                opem_spec_fuel=DEF_FLEET.opem_spec_fuel,
                opex_spec_onroute_charging=st.session_state.eco_opex_spec_route_charging
                * DEF_FLEET.opex_spec_onroute_charging.factor,
                opem_spec_onroute_charging=DEF_FLEET.opem_spec_onroute_charging,
            ),
            cis=ComparisonChargingInfrastructure(
                p_lm_max=ExistExpansionValue(
                    baseline=(
                        np.inf
                        if st.session_state[f"load_mngmnt_baseline"]
                        != get_label("sidebar.chargers.load_mngmnt.type.options.static")
                        else st.session_state[f"load_mngmnt_slider_baseline"]
                    ),
                    expansion=(
                        np.inf
                        if st.session_state[f"load_mngmnt_expansion"]
                        != get_label("sidebar.chargers.load_mngmnt.type.options.static")
                        else st.session_state[f"load_mngmnt_slider_expansion"]
                    ),
                ),
                subblocks={
                    k: ComparisonChargerType(
                        name=k,
                        num=ExistExpansionValue(
                            baseline=st.session_state[f"chg_{k}_preexisting"],
                            expansion=st.session_state[f"chg_{k}_preexisting"] + st.session_state[f"chg_{k}_expansion"],
                        ),
                        p_max=st.session_state[f"chg_{k}_pwr"] * 1e3,
                        capex_per_unit=st.session_state[f"chg_{k}_cost"],
                        **DEF_CIS.subblocks[k].values,
                    )
                    for k in DEF_CIS.subblocks.keys()
                },
            ),
        )

    if st.session_state["run_backend"] is True:
        try:
            results = run_comparison(comp_scn=get_inputs())
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
