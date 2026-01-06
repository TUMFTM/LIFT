import abc
from functools import cached_property

import altair as alt
import altair_saver
import io
import numpy as np
import pandas as pd

from .design import COLOR_BL, COLOR_EX
from .utils import get_label


def _create_ring(
    df: pd.DataFrame,
    radius: float,
    thickness: float,
    color: str,
    tooltip_list: list,
) -> alt.LayerChart:
    background = (
        alt.Chart(df)
        .mark_arc(innerRadius=radius, outerRadius=radius + thickness, color=color, opacity=0.4, tooltip=None)
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


class ResultPlot(abc.ABC):
    @abc.abstractmethod
    @cached_property
    def plot(self) -> alt.VConcatChart: ...

    @cached_property
    def plot_bytestream(self):
        image_stream = io.BytesIO()
        self.plot.save(image_stream, format="png", ppi=600)
        image_stream.seek(0)  # Rewind to the beginning of the buffer
        return image_stream


class KpiPlot(ResultPlot, abc.ABC):
    def __init__(
        self,
        val_baseline: float,
        val_expansion: float,
        phase_labels: tuple[str, str],
        label: str,
    ):
        self.val_baseline = val_baseline
        self.val_expansion = val_expansion
        self.phase_labels = phase_labels
        self.label = label


class BarKpiPlot(KpiPlot):
    def __init__(
        self,
        val_baseline: float,
        val_expansion: float,
        phase_labels: tuple[str, str],
        label: str,
        factor_display: float,
    ):
        super().__init__(
            val_baseline=val_baseline,
            val_expansion=val_expansion,
            phase_labels=phase_labels,
            label=label,
        )
        self.factor_display = factor_display

    @cached_property
    def plot(self) -> alt.VConcatChart:
        data = pd.DataFrame(
            index=["baseline", "expansion"],
            data={
                "value": [
                    self.val_baseline,
                    self.val_expansion,
                ],
                "phase": self.phase_labels,
                "value_display": [self.val_baseline * self.factor_display, self.val_expansion * self.factor_display],
            },
        )

        text = (
            alt.Chart(
                pd.DataFrame(
                    {"x": [0.5], "y": [0.5], "label": [f"{(self.val_expansion / self.val_baseline - 1) * 100:+.1f} %"]}
                )
            )
            .mark_text(
                fontWeight="bold",
                size=18,
                color="green" if self.val_baseline >= self.val_expansion else "red",
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
                    sort=self.phase_labels,
                    scale=alt.Scale(paddingInner=0.0, paddingOuter=1),
                ),
                y=alt.Y(
                    shorthand="value:Q",
                    axis=None,
                ),
                color=alt.Color(
                    shorthand="phase:N",
                    legend=None,
                    scale=alt.Scale(domain=self.phase_labels, range=[COLOR_BL, COLOR_EX]),
                ),
                tooltip=[
                    alt.Tooltip(shorthand="phase:N", title=get_label("main.name_scenario")),
                    alt.Tooltip(shorthand="value_display:Q", title=self.label, format=",.0f"),
                ],
            )
        ).properties(width=100, height=130)

        return alt.vconcat(text, bars).configure_view(stroke=None)


class RingKpiPlot(KpiPlot):
    @cached_property
    def plot(self) -> alt.VConcatChart:
        data = pd.DataFrame(
            index=["baseline", "expansion"],
            data={
                "value_front": [self.val_baseline, self.val_expansion],
                "value_back": [1, 1],
                "phase": self.phase_labels,
                "value_display": [self.val_baseline * 100, self.val_expansion * 100],
            },
        )

        tooltips = [
            alt.Tooltip(shorthand="phase:N", title="Szenario"),
            alt.Tooltip(shorthand="value_display:Q", title=self.label, format=",.2f"),
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
            alt.Chart(pd.DataFrame({"text": [f"{(self.val_expansion - self.val_baseline) * 100:+.0f} %"]}))
            .mark_text(
                size=18,
                fontWeight="bold",
                color="green" if self.val_expansion >= self.val_baseline else "red",
                tooltip=None,
            )
            .encode(text="text:N")
        )

        return (ring_baseline + ring_expansion + center_text).properties(width=200, height=200)


class TimeseriesPlot(ResultPlot):
    def __init__(
        self,
        baseline_capex: np.typing.NDArray,
        baseline_opex: np.typing.NDArray,
        expansion_capex: np.typing.NDArray,
        expansion_opex: np.typing.NDArray,
        x_label: str,
        y_label: str,
        phase_labels: tuple[str, str],
    ):
        self.baseline_capex = baseline_capex
        self.baseline_opex = baseline_opex
        self.expansion_capex = expansion_capex
        self.expansion_opex = expansion_opex
        self.x_label = x_label
        self.y_label = y_label
        self.phase_labels = phase_labels

    @cached_property
    def plot(self) -> alt.VConcatChart:
        n_years = len(self.baseline_capex)
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
                    [
                        [0],
                        self.baseline_opex[:-1],
                        self.baseline_capex,
                        [0],
                        self.expansion_opex[:-1],
                        self.expansion_capex,
                    ]
                ),
                "scenario": [self.phase_labels[0]] * n_years * 2 + [self.phase_labels[1]] * n_years * 2,
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
                    axis=alt.Axis(title=self.x_label, values=years, format=".0f"),
                    scale=alt.Scale(domain=[float(years.min()), float(years.max())], nice=False),
                ),
                y=alt.Y(shorthand="value:Q", axis=alt.Axis(title=self.y_label)),
                color=alt.Color(
                    shorthand="scenario:N",
                    legend=None,
                    scale=alt.Scale(domain=self.phase_labels, range=[COLOR_BL, COLOR_EX]),
                ),
                tooltip=[
                    alt.Tooltip(shorthand="scenario:N", title=get_label("main.name_scenario")),
                    alt.Tooltip(shorthand="year:Q", title=self.x_label, format=".0f"),
                    alt.Tooltip(shorthand="value:Q", title=self.y_label, format=",.0f"),
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

        return chart
