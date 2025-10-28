from __future__ import annotations
import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# Flag to use streamlit caching; required before importing lift.backend.interfaces
# Is automatically deleted after import
os.environ["LIFT_USE_STREAMLIT_CACHE"] = "1"


from .definitions import PERIOD_ECO

from .design import COLOR_BL, COLOR_EX, LINE_HORIZONTAL


PLOT_CONFIG = {"usermeta": {"embedOptions": {"actions": False}}}


def display_results(results):
    st.markdown(
        f"### <span style='color:{COLOR_BL}'>Baseline</span> vs. <span style='color:{COLOR_EX}'>Expansion</span>",
        unsafe_allow_html=True,
    )

    def _show_kpis():
        def _centered_heading(text: str, domain=st) -> None:
            domain.markdown(f"<h5 style='text-align:center; margin:0'>{text}</h5>", unsafe_allow_html=True)

        def _create_bar_comparison(
            val_baseline: float,
            val_expansion: float,
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
                    "phase": ["Baseline", "Expansion"],
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
                        sort=["Baseline", "Expansion"],
                        scale=alt.Scale(paddingInner=0.0, paddingOuter=1),
                    ),
                    y=alt.Y(
                        shorthand="value:Q",
                        axis=None,
                    ),
                    color=alt.Color(
                        shorthand="phase:N",
                        legend=None,
                        scale=alt.Scale(domain=["Baseline", "Expansion"], range=[COLOR_BL, COLOR_EX]),
                    ),
                    tooltip=[
                        alt.Tooltip(shorthand="phase:N", title="Szenario"),
                        alt.Tooltip(shorthand="value_display:Q", title=label, format=",.0f"),
                    ],
                )
            ).properties(width=100, height=130)

            return alt.vconcat(text, bars).configure_view(stroke=None)

        def _create_ring_comparison(
            val_baseline: float,
            val_expansion: float,
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
                    "phase": ["Baseline", "Expansion"],
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

        col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])

        _centered_heading(text="Kosten", domain=col1)
        col1.altair_chart(
            _create_bar_comparison(
                val_baseline=results.baseline.cashflow_dis["totex"].sum(),
                val_expansion=results.expansion.cashflow_dis["totex"].sum(),
                label="Gesamtkosten in EUR",
            ).properties(**PLOT_CONFIG),
            use_container_width=True,
        )

        _centered_heading(text="CO₂-Emissionen", domain=col2)
        col2.altair_chart(
            _create_bar_comparison(
                val_baseline=results.baseline.emissions["totex"].sum(),
                val_expansion=results.expansion.emissions["totex"].sum(),
                label="CO2-Emissionen in t",
                factor_display=1e-3,  # convert from kg to t
            ).properties(**PLOT_CONFIG),
            use_container_width=True,
        )

        _centered_heading(text="Eigenverbrauchsquote", domain=col3)
        col3.altair_chart(
            _create_ring_comparison(
                val_baseline=results.baseline.self_consumption,
                val_expansion=results.expansion.self_consumption,
                label="Eigenverbrauchsquote in %",
            ).properties(**PLOT_CONFIG),
            use_container_width=True,
        )

        _centered_heading(text="Autarkiegrad", domain=col4)
        col4.altair_chart(
            _create_ring_comparison(
                val_baseline=results.baseline.self_sufficiency,
                val_expansion=results.expansion.self_sufficiency,
                label="Autarkiegrad in %",
            ).properties(**PLOT_CONFIG),
            use_container_width=True,
        )

        _centered_heading(text="Heim-Laden", domain=col5)
        col5.altair_chart(
            _create_ring_comparison(
                val_baseline=results.baseline.site_charging,
                val_expansion=results.expansion.site_charging,
                label="Anteil Heimladen (Energie) in %",
            ).properties(**PLOT_CONFIG),
            use_container_width=True,
        )

    _show_kpis()

    st.markdown("#### Gesamtkosten")
    col1, col2 = st.columns([4, 1])
    with col1:
        plot_flow(
            baseline_capex=results.baseline.cashflow_dis["capex"].sum(axis=0),
            baseline_opex=results.baseline.cashflow_dis["opex"].sum(axis=0),
            expansion_capex=results.expansion.cashflow_dis["capex"].sum(axis=0),
            expansion_opex=results.expansion.cashflow_dis["opex"].sum(axis=0),
            y_label="Kumulierte Kosten in EUR",
        )
    with col2:
        st.markdown("#### Amortisationsdauer")
        if results.payback_period_yrs is None:
            st.markdown("Investition amortisiert sich nicht.")
        else:
            st.markdown(f"{results.payback_period_yrs:.2f} Jahre")

        st.markdown("#### Kosteneinsparung")
        st.markdown(f"{results.npc_delta:,.0f} EUR nach 18 Jahren")

    with st.expander("#### Verlauf CO₂-Ausstoß"):
        st.markdown("#### Kumulierter CO₂-Ausstoß")
        col1, col2 = st.columns([4, 1])
        with col1:
            plot_flow(
                baseline_capex=results.baseline.emissions["capex"].sum(axis=0) * 1e-3,  # convert from kg to t
                baseline_opex=results.baseline.emissions["opex"].sum(axis=0) * 1e-3,  # convert from kg to t
                expansion_capex=results.expansion.emissions["capex"].sum(axis=0) * 1e-3,  # convert from kg to t
                expansion_opex=results.expansion.emissions["opex"].sum(axis=0) * 1e-3,  # convert from kg to t
                y_label="Kumulierte CO₂-Emissionen in t",
            )
        with col2:
            st.markdown("#### CO₂-Paritätsdauer")
            if results.payback_period_co2_yrs is None:
                st.markdown("Investition amortisiert sich nicht.")
            else:
                st.markdown(f"{results.payback_period_co2_yrs:.2f} Jahre")

            st.markdown("#### CO₂-Einsparung")
            st.markdown(f"{results.co2_delta * 1e-3:,.0f} t nach 18 Jahren")


def display_empty_results():
    st.markdown("""
    Willkommen zu deinem datengetriebenen Framework für die Elektrifizierung von Lkw-Flotten
    und die Optimierung von Speditionsstandorten. Nutze die Seitenleiste, um Parameter
    einzugeben und klicke anschließend auf den Button, um erste Berechnungen zu starten.
    """)
    st.markdown(LINE_HORIZONTAL, unsafe_allow_html=True)
    st.warning("Bitte geben Sie die Parameter in der Seitenleiste ein und klicken Sie auf **🚀 Berechnen**.")


def plot_flow(
    baseline_capex: np.typing.NDArray,
    baseline_opex: np.typing.NDArray,
    expansion_capex: np.typing.NDArray,
    expansion_opex: np.typing.NDArray,
    y_label: str,
):
    years = np.arange(PERIOD_ECO + 1, dtype=int)
    n_years = PERIOD_ECO + 1

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
            "scenario": ["Baseline"] * n_years * 2 + ["Expansion"] * n_years * 2,
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
                axis=alt.Axis(title="Jahr", values=years, format=".0f"),
                scale=alt.Scale(domain=[float(years.min()), float(years.max())], nice=False),
            ),
            y=alt.Y(shorthand="value:Q", axis=alt.Axis(title=y_label)),
            color=alt.Color(
                shorthand="scenario:N",
                legend=None,
                scale=alt.Scale(domain=["Baseline", "Expansion"], range=[COLOR_BL, COLOR_EX]),
            ),
            tooltip=[
                alt.Tooltip(shorthand="scenario:N", title="Szenario"),
                alt.Tooltip(shorthand="year:Q", title="Jahr", format=".0f"),
                alt.Tooltip(shorthand="value:Q", title=y_label, format=",.0f"),
            ],
        )
        .properties(height=360)
    )

    layers = [line]
    # ToDo: For any annotations use additional layers (intersection, delta values, etc.)

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

    st.altair_chart(chart, use_container_width=True)
