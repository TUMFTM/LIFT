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

import pandas as pd
import streamlit as st

# Flag to use streamlit caching; required before importing lift.backend.interfaces
# Is automatically deleted after import
os.environ["LIFT_USE_STREAMLIT_CACHE"] = "1"


from .design import COLOR_BL, COLOR_EX
from .export import create_report
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

    plots = {
        "kpi_diagrams": {
            "kpi_diagrams.costs": BarKpiPlot(
                val_baseline=results.baseline.totex_dis.sum(),
                val_expansion=results.expansion.totex_dis.sum(),
                phase_labels=phases,
                label=f"{get_label('main.kpi_diagrams.costs.axis')} in EUR",
                factor_display=1.0,
            ),
            "kpi_diagrams.emissions": BarKpiPlot(
                val_baseline=results.baseline.totem.sum(),
                val_expansion=results.expansion.totem.sum(),
                phase_labels=phases,
                label=f"{get_label('main.kpi_diagrams.emissions.axis')} in t CO₂-eq.",
                factor_display=1e-3,  # convert from kg to t
            ),
            "kpi_diagrams.self_consumption": RingKpiPlot(
                val_baseline=results.baseline.self_consumption,
                val_expansion=results.expansion.self_consumption,
                phase_labels=phases,
                label=f"{get_label('main.kpi_diagrams.self_consumption.axis')} in %",
            ),
            "kpi_diagrams.self_sufficiency": RingKpiPlot(
                val_baseline=results.baseline.self_sufficiency,
                val_expansion=results.expansion.self_sufficiency,
                phase_labels=phases,
                label=f"{get_label('main.kpi_diagrams.self_sufficiency.axis')} in %",
            ),
            "kpi_diagrams.home_charging": RingKpiPlot(
                val_baseline=results.baseline.home_charging_fraction,
                val_expansion=results.expansion.home_charging_fraction,
                phase_labels=phases,
                label=f"{get_label('main.kpi_diagrams.home_charging.axis')} in %",
            ),
        },
        "time_diagrams": {
            "time_diagrams.costs": TimeseriesPlot(
                baseline_capex=results.baseline.capex_dis,
                baseline_opex=results.baseline.opex_dis,
                expansion_capex=results.expansion.capex_dis,
                expansion_opex=results.expansion.opex_dis,
                x_label=get_label("main.time_diagrams.costs.xaxis"),
                y_label=f"{get_label('main.time_diagrams.costs.yaxis')} in EUR",
                phase_labels=phases,
            ),
            "time_diagrams.emissions": TimeseriesPlot(
                baseline_capex=results.baseline.capem * 1e-3,  # convert from kg to t
                baseline_opex=results.baseline.opem * 1e-3,  # convert from kg to t
                expansion_capex=results.expansion.capem * 1e-3,  # convert from kg to t
                expansion_opex=results.expansion.opem * 1e-3,  # convert from kg to t
                x_label=get_label("main.time_diagrams.emissions.xaxis"),
                y_label=f"{get_label('main.time_diagrams.emissions.yaxis')} in t CO₂-eq.",
                phase_labels=phases,
            ),
        },
    }

    for col, (plot_key, plot_obj) in zip(domain.kpi_diagrams().columns([1, 1, 1, 1, 1]), plots["kpi_diagrams"].items()):
        _heading_with_help(
            label=get_label(f"main.{plot_key}.title"),
            help_msg=get_label(f"main.{plot_key}.help"),
            adjust="center",
            size=5,
            margin_left=23,
            domain=col,
        )
        col.altair_chart(
            plot_obj.plot.properties(**PLOT_CONFIG),
            width="stretch",
        )

    for tab, key in zip(
        domain().tabs(
            [
                get_label("main.time_diagrams.costs.tab"),
                get_label("main.time_diagrams.emissions.tab"),
            ]
        ),
        ["costs", "emissions"],
    ):
        # Show heading
        _heading_with_help(
            label=get_label(f"main.time_diagrams.{key}.title.label"),
            help_msg=get_label(f"main.time_diagrams.{key}.title.help"),
            domain=tab,
        )

        col1, col2 = tab.columns([4, 1])
        with col1:
            st.altair_chart(plots["time_diagrams"][f"time_diagrams.{key}"].plot, width="stretch")
        with col2:
            _heading_with_help(
                label=get_label(f"main.time_diagrams.{key}.paybackperiod.title"),
                help_msg=get_label(f"main.time_diagrams.{key}.paybackperiod.help"),
                adjust="left",
                size=5,
            )
            if results.payback_period[key] is None:
                st.markdown(get_label(f"main.time_diagrams.{key}.paybackperiod.negative_result"))
            else:
                st.markdown(
                    f"{results.payback_period[key]:.0f} {get_label(f'main.time_diagrams.{key}.paybackperiod.years')}"
                )

            _heading_with_help(
                label=get_label(f"main.time_diagrams.{key}.saving.title"),
                help_msg=get_label(f"main.time_diagrams.{key}.saving.help"),
                adjust="left",
                size=5,
            )
            st.markdown(
                f"{results.delta[key]:,.0f} EUR {get_label(f'main.time_diagrams.{key}.saving.after')} {results.baseline.period_eco} {get_label(f'main.time_diagrams.{key}.saving.years')}"
            )

    word_bytes = create_report(plots=plots, inputs=st.session_state.inputs.df)

    st.download_button(
        "⬇️ Download Word (.docx)",
        data=word_bytes,
        file_name="report.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        on_click="ignore",
    )


def display_empty_results(domain):
    domain().warning(
        f"{get_label('main.empty.manual1')}**🚀 {get_label('sidebar.calculate')}**{get_label('main.empty.manual2')}"
    )
