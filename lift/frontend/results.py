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


from .definitions import DEF_SCN
from .design import COLOR_BL, COLOR_EX
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
        def _create_bar_comparison(
            val_baseline: float,
            val_expansion: float,
            phase_labels: tuple[str, str],
            label: str,
            factor_display: float = 1.0,
        ) -> alt.VConcatChart:
            data = pd.DataFrame(
                index=["baseline", "expansion"],
                data={
                    "value": [
                        val_baseline,
                        val_expansion,
                    ],
                    "phase": phase_labels,
                    "value_display": [val_baseline * factor_display, val_expansion * factor_display],
                },
            )

            text = (
                alt.Chart(
                    pd.DataFrame(
                        {"x": [0.5], "y": [0.5], "label": [f"{(val_expansion / val_baseline - 1) * 100:+.1f} %"]}
                    )
                )
                .mark_text(
                    fontWeight="bold",
                    size=18,
                    color="green" if val_baseline >= val_expansion else "red",
                    align="center",
                    baseline="middle",
                )
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
                    x=alt.X(
                        shorthand="phase:N",
                        axis=None,
                        sort=phase_labels,
                        scale=alt.Scale(paddingInner=0.0, paddingOuter=1),
                    ),
                    y=alt.Y(
                        shorthand="value:Q",
                        axis=None,
                    ),
                    color=alt.Color(
                        shorthand="phase:N",
                        legend=None,
                        scale=alt.Scale(domain=phase_labels, range=[COLOR_BL, COLOR_EX]),
                    ),
                    tooltip=[
                        alt.Tooltip(shorthand="phase:N", title=get_label("main.name_scenario")),
                        alt.Tooltip(shorthand="value_display:Q", title=label, format=",.0f"),
                    ],
                )
            ).properties(width=100, height=130)

            return alt.vconcat(text, bars).configure_view(stroke=None)

        def _create_ring_comparison(
            val_baseline: float,
            val_expansion: float,
            phase_labels: tuple[str, str],
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
                    .mark_arc(
                        innerRadius=radius, outerRadius=radius + thickness, color=color, opacity=0.4, tooltip=None
                    )
                    .encode(
                        theta=alt.Theta("value_back:Q", stack=True),
                        tooltip=tooltip_list,
                    )
                )

                foreground = (
                    alt.Chart(df)
                    .mark_arc(innerRadius=radius, outerRadius=radius + thickness, cornerRadius=2, color=color)
                    .encode(
                        theta=alt.Theta("value_front:Q", stack=True),
                        tooltip=tooltip_list,
                    )
                )
                return background + foreground

            data = pd.DataFrame(
                index=["baseline", "expansion"],
                data={
                    "value_front": [val_baseline, val_expansion],
                    "value_back": [1, 1],
                    "phase": phase_labels,
                    "value_display": [val_baseline * 100, val_expansion * 100],
                },
            )

            tooltips = [
                alt.Tooltip(shorthand="phase:N", title="Szenario"),
                alt.Tooltip(shorthand="value_display:Q", title=label, format=",.2f"),
            ]

            # Create rings
            ring_baseline = _create_ring(
                df=data.loc[["baseline"]],
                radius=40,
                thickness=20,
                color=COLOR_BL,
                tooltip_list=tooltips,
            )
            ring_expansion = _create_ring(
                df=data.loc[["expansion"]],
                radius=65,
                thickness=20,
                color=COLOR_EX,
                tooltip_list=tooltips,
            )

            # Center text (single-row dataframe, minimal overhead)
            center_text = (
                alt.Chart(pd.DataFrame({"text": [f"{(val_expansion - val_baseline) * 100:+.0f} %"]}))
                .mark_text(
                    size=18, fontWeight="bold", color="green" if val_expansion >= val_baseline else "red", tooltip=None
                )
                .encode(text="text:N")
            )

            return (ring_baseline + ring_expansion + center_text).properties(width=200, height=200)

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
            _create_bar_comparison(
                val_baseline=results.baseline.totex_dis.sum(),
                val_expansion=results.expansion.totex_dis.sum(),
                phase_labels=phases,
                label=f"{get_label('main.kpi_diagrams.costs.axis')} in EUR",
            ).properties(**PLOT_CONFIG),
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
            _create_bar_comparison(
                val_baseline=results.baseline.totem.sum(),
                val_expansion=results.expansion.totem.sum(),
                phase_labels=phases,
                label=f"{get_label('main.kpi_diagrams.emissions.axis')} in t CO₂-eq.",
                factor_display=1e-3,  # convert from kg to t
            ).properties(**PLOT_CONFIG),
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
            _create_ring_comparison(
                val_baseline=results.baseline.self_consumption,
                val_expansion=results.expansion.self_consumption,
                phase_labels=phases,
                label=f"{get_label('main.kpi_diagrams.self_consumption.axis')} in %",
            ).properties(**PLOT_CONFIG),
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
            _create_ring_comparison(
                val_baseline=results.baseline.self_sufficiency,
                val_expansion=results.expansion.self_sufficiency,
                phase_labels=phases,
                label=f"{get_label('main.kpi_diagrams.self_sufficiency.axis')} in %",
            ).properties(**PLOT_CONFIG),
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
            _create_ring_comparison(
                val_baseline=results.baseline.home_charging_fraction,
                val_expansion=results.expansion.home_charging_fraction,
                phase_labels=phases,
                label=f"{get_label('main.kpi_diagrams.home_charging.axis')} in %",
            ).properties(**PLOT_CONFIG),
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
            plot_flow(
                baseline_capex=results.baseline.capex_dis,
                baseline_opex=results.baseline.opex_dis,
                expansion_capex=results.expansion.capex_dis,
                expansion_opex=results.expansion.opex_dis,
                x_label=get_label("main.time_diagrams.costs.xaxis"),
                y_label=f"{get_label('main.time_diagrams.costs.yaxis')} in EUR",
                phase_labels=phases,
            )
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
            plot_flow(
                baseline_capex=results.baseline.capem * 1e-3,  # convert from kg to t
                baseline_opex=results.baseline.opem * 1e-3,  # convert from kg to t
                expansion_capex=results.expansion.capem * 1e-3,  # convert from kg to t
                expansion_opex=results.expansion.opem * 1e-3,  # convert from kg to t
                x_label=get_label("main.time_diagrams.emissions.xaxis"),
                y_label=f"{get_label('main.time_diagrams.emissions.yaxis')} in t CO₂-eq.",
                phase_labels=phases,
            )
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


def plot_flow(
    baseline_capex: np.typing.NDArray,
    baseline_opex: np.typing.NDArray,
    expansion_capex: np.typing.NDArray,
    expansion_opex: np.typing.NDArray,
    x_label: str,
    y_label: str,
    phase_labels: tuple[str, str],
):
    n_years = len(baseline_capex)  # DEF_SCN.period_eco + 1
    years = np.arange(n_years, dtype=int)

    # the last entry holds data for the first year after the project duration
    # - capex: this entry holds salvage values, caused by components which are not at the end of their lifespan at
    #          the end of the project period, occurring directly after the project period
    # -opex: this entry always is 0 as there are no opex beyond the project duration

    # Plotting logic:
    # Cost starts at 0, 0 (year, cost). Capex of a year always occur at the beginning of the year, opex at the end.
    # This logic is also implemented in the discounting of capex and opex.
    # Therefore, opex are shifted by one year in this plotting code (end of year n equals beginning of year n + 1)
    # Additionally, a value for the start of the line at 0, 0 has to be added
    df = pd.DataFrame(
        data={
            "value": np.concatenate(
                [[0], baseline_opex[:-1], baseline_capex, [0], expansion_opex[:-1], expansion_capex]
            ),
            "scenario": [phase_labels[0]] * n_years * 2 + [phase_labels[1]] * n_years * 2,
            "type": np.tile(["opex"] * n_years + ["capex"] * n_years, 2),
            "year": np.tile(np.arange(n_years), 4),
        }
    )

    # Ensure type order for correct sorting (opex before capex)
    df["type"] = pd.Categorical(df["type"], categories=["opex", "capex"], ordered=True)

    # Sort by phase, then year, then type
    df = df.sort_values(by=["year", "type"])

    # Compute cumulative value per phase
    df["value"] = df.groupby("scenario")["value"].cumsum()

    line = (
        alt.Chart(df)
        .mark_line(point={"filled": True, "size": 50}, interpolate="linear")
        .encode(
            x=alt.X(
                shorthand="year:Q",
                axis=alt.Axis(title=x_label, values=years, format=".0f"),
                scale=alt.Scale(domain=[float(years.min()), float(years.max())], nice=False),
            ),
            y=alt.Y(shorthand="value:Q", axis=alt.Axis(title=y_label)),
            color=alt.Color(
                shorthand="scenario:N",
                legend=None,
                scale=alt.Scale(domain=phase_labels, range=[COLOR_BL, COLOR_EX]),
            ),
            tooltip=[
                alt.Tooltip(shorthand="scenario:N", title=get_label("main.name_scenario")),
                alt.Tooltip(shorthand="year:Q", title=x_label, format=".0f"),
                alt.Tooltip(shorthand="value:Q", title=y_label, format=",.0f"),
            ],
        )
        .properties(height=360)
    )

    layers = [line]
    # Use additional layers for any annotations (intersection, delta values, etc.)

    chart = (
        alt.layer(*layers)
        .configure_axis(
            labelColor="black",
            titleColor="black",
            tickColor="black",
            domainColor="black",
        )
        .configure_view(
            stroke=None,
            strokeWidth=1,
        )
    )

    st.altair_chart(chart, width="stretch")
