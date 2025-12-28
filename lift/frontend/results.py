"""Streamlit results and plots for LIFT comparisons.

Purpose:
- Render KPIs and time-profiled charts comparing baseline vs. expansion scenarios.

Relationships:
- Consumes `ComparisonResult` (with `PhaseResult` for baseline/expansion) from the backend.
- Uses frontend styles and labels for consistent presentation.

Key Logic:
- KPI tiles: discounted total costs (TOTEX), emissions totals, self-consumption/-sufficiency, site-charging share.
- Cost and CO₂ time profiles: cumulative lines constructed from periodized CAPEX/OPEX (incl. salvage at end).
- Plotting via Altair with minimal interactivity for tooltips and consistent color scheme.
"""

from __future__ import annotations
import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# Flag to use streamlit caching; required before importing lift.backend.interfaces
# Is automatically deleted after import
os.environ["LIFT_USE_STREAMLIT_CACHE"] = "1"


from .design import COLOR_BL, COLOR_EX
from .plots import BarKpiPlot, RingKpiPlot, TimeseriesPlot
from .utils import get_label


PLOT_CONFIG = {"usermeta": {"embedOptions": {"actions": False}}}


def _heading_with_help(
    label: str, help_msg: str = None, adjust: str = "left", margin_left: int = 0, size: int = 4, domain=st
):
    domain.markdown(
        f"""
        <div style="display: flex; justify-content: {adjust}; align-items: center;">
            <h{size} style="margin: 0; margin-left: {margin_left}px;">{label}</h{size}>
            <span title="{help_msg}" style="margin-left: 0px; cursor: pointer;">
                &#x1F6C8;
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_results(results, domain):
    domain.subtitle().markdown(
        f"""### <span style='color:{COLOR_BL}'>{get_label("main.name_baseline")}</span> vs. <span style='color:{COLOR_EX}'>{get_label("main.name_expansion")}</span>""",
        unsafe_allow_html=True,
    )

    phases = (get_label("main.name_baseline"), get_label("main.name_expansion"))

    def _show_kpis(phases: tuple[str, str], domain):
        col1, col2, col3, col4, col5 = domain().columns([1, 1, 1, 1, 1])

        _heading_with_help(
            label=get_label("main.kpi_diagrams.costs.title"),
            help_msg=get_label("main.kpi_diagrams.costs.help"),
            adjust="center",
            size=5,
            margin_left=23,
            domain=col1,
        )
        col1.altair_chart(
            BarKpiPlot(
                val_baseline=results.baseline.totex_dis.sum(),
                val_expansion=results.expansion.totex_dis.sum(),
                phase_labels=phases,
                label=f"{get_label('main.kpi_diagrams.costs.axis')} in EUR",
                factor_display=1.0,
            ).plot.properties(**PLOT_CONFIG),
            width="stretch",
        )

        _heading_with_help(
            label=get_label("main.kpi_diagrams.emissions.title"),
            help_msg=get_label("main.kpi_diagrams.emissions.help"),
            adjust="center",
            size=5,
            margin_left=23,
            domain=col2,
        )
        col2.altair_chart(
            BarKpiPlot(
                val_baseline=results.baseline.totem.sum(),
                val_expansion=results.expansion.totem.sum(),
                phase_labels=phases,
                label=f"{get_label('main.kpi_diagrams.emissions.axis')} in t CO₂-eq.",
                factor_display=1e-3,  # convert from kg to t
            ).plot.properties(**PLOT_CONFIG),
            width="stretch",
        )

        _heading_with_help(
            label=get_label("main.kpi_diagrams.self_consumption.title"),
            help_msg=get_label("main.kpi_diagrams.self_consumption.help"),
            adjust="center",
            size=5,
            margin_left=23,
            domain=col3,
        )
        col3.altair_chart(
            RingKpiPlot(
                val_baseline=results.baseline.self_consumption,
                val_expansion=results.expansion.self_consumption,
                phase_labels=phases,
                label=f"{get_label('main.kpi_diagrams.self_consumption.axis')} in %",
            ).plot.properties(**PLOT_CONFIG),
            width="stretch",
        )

        _heading_with_help(
            label=get_label("main.kpi_diagrams.self_sufficiency.title"),
            help_msg=get_label("main.kpi_diagrams.self_sufficiency.help"),
            adjust="center",
            size=5,
            margin_left=23,
            domain=col4,
        )
        col4.altair_chart(
            RingKpiPlot(
                val_baseline=results.baseline.self_sufficiency,
                val_expansion=results.expansion.self_sufficiency,
                phase_labels=phases,
                label=f"{get_label('main.kpi_diagrams.self_sufficiency.axis')} in %",
            ).plot.properties(**PLOT_CONFIG),
            width="stretch",
        )

        _heading_with_help(
            label=get_label("main.kpi_diagrams.home_charging.title"),
            help_msg=get_label("main.kpi_diagrams.home_charging.help"),
            adjust="center",
            size=5,
            margin_left=23,
            domain=col5,
        )
        col5.altair_chart(
            RingKpiPlot(
                val_baseline=results.baseline.home_charging_fraction,
                val_expansion=results.expansion.home_charging_fraction,
                phase_labels=phases,
                label=f"{get_label('main.kpi_diagrams.home_charging.axis')} in %",
            ).plot.properties(**PLOT_CONFIG),
            width="stretch",
        )

    _show_kpis(phases=phases, domain=domain.kpi_diagrams)

    with domain.time_diagrams.costs():
        # Show heading
        _heading_with_help(
            label=get_label("main.time_diagrams.costs.title.label"),
            help_msg=get_label("main.time_diagrams.costs.title.help"),
        )

        col1, col2 = st.columns([4, 1])
        with col1:
            plot_costs = TimeseriesPlot(
                baseline_capex=results.baseline.capex_dis,
                baseline_opex=results.baseline.opex_dis,
                expansion_capex=results.expansion.capex_dis,
                expansion_opex=results.expansion.opex_dis,
                x_label=get_label("main.time_diagrams.costs.xaxis"),
                y_label=f"{get_label('main.time_diagrams.costs.yaxis')} in EUR",
                phase_labels=phases,
            )
            st.altair_chart(plot_costs.plot, width="stretch")
        with col2:
            _heading_with_help(
                label=get_label("main.time_diagrams.costs.paybackperiod.title"),
                help_msg=get_label("main.time_diagrams.costs.paybackperiod.help"),
                adjust="left",
                size=5,
            )
            if results.payback_period_yrs is None:
                st.markdown(get_label("main.time_diagrams.costs.paybackperiod.negative_result"))
            else:
                st.markdown(
                    f"{results.payback_period_yrs:.0f} {get_label('main.time_diagrams.costs.paybackperiod.years')}"
                )

            _heading_with_help(
                label=get_label("main.time_diagrams.costs.saving.title"),
                help_msg=get_label("main.time_diagrams.costs.saving.help"),
                adjust="left",
                size=5,
            )
            st.markdown(
                f"{results.npc_delta:,.0f} EUR {get_label('main.time_diagrams.costs.saving.after')} {results.baseline.period_eco} {get_label('main.time_diagrams.costs.saving.years')}"
            )

    with domain.time_diagrams.emissions():
        # Show heading
        _heading_with_help(
            label=get_label("main.time_diagrams.emissions.title.label"),
            help_msg=get_label("main.time_diagrams.emissions.title.help"),
        )
        col1, col2 = st.columns([4, 1])
        with col1:
            plot_emissions = TimeseriesPlot(
                baseline_capex=results.baseline.capem * 1e-3,  # convert from kg to t
                baseline_opex=results.baseline.opem * 1e-3,  # convert from kg to t
                expansion_capex=results.expansion.capem * 1e-3,  # convert from kg to t
                expansion_opex=results.expansion.opem * 1e-3,  # convert from kg to t
                x_label=get_label("main.time_diagrams.emissions.xaxis"),
                y_label=f"{get_label('main.time_diagrams.emissions.yaxis')} in t CO₂-eq.",
                phase_labels=phases,
            )
            st.altair_chart(plot_emissions.plot, width="stretch")

        with col2:
            _heading_with_help(
                label=get_label("main.time_diagrams.emissions.paybackperiod.title"),
                help_msg=get_label("main.time_diagrams.emissions.paybackperiod.help"),
                adjust="left",
                size=5,
            )
            if results.payback_period_co2_yrs is None:
                st.markdown(get_label("main.time_diagrams.emissions.paybackperiod.negative_result"))
            else:
                st.markdown(
                    f"{results.payback_period_co2_yrs:.2f} {get_label('main.time_diagrams.emissions.paybackperiod.years')}"
                )

            _heading_with_help(
                label=get_label("main.time_diagrams.emissions.saving.title"),
                help_msg=get_label("main.time_diagrams.emissions.saving.help"),
                adjust="left",
                size=5,
            )
            st.markdown(
                f"{results.co2_delta * 1e-3:,.0f} t CO₂-eq. {get_label('main.time_diagrams.emissions.saving.after')} {results.baseline.period_eco} {get_label('main.time_diagrams.emissions.saving.years')}"
            )


def display_empty_results(domain):
    domain().warning(
        f"{get_label('main.empty.manual1')}**🚀 {get_label('sidebar.calculate')}**{get_label('main.empty.manual2')}"
    )
