from __future__ import annotations
from abc import ABC, abstractmethod
import ast
from dataclasses import dataclass, field
import importlib.resources as resources
from pathlib import Path
import re
from typing import Self

import demandlib
import numpy as np
import pandas as pd
import pvlib
import pytz


from lift.backend.comparison.interfaces import ComparisonGrid, ComparisonInvestComponent, ComparisonInputCharger
from lift.backend.simulation.interfaces import Coordinates
from lift.backend.utils import safe_cache_data


@safe_cache_data
def _get_dti(start, duration, freq):
    return pd.date_range(start=start, end=start + duration, freq=freq, inclusive="left")


@safe_cache_data
def _get_dti_extended(start, duration, freq):
    return pd.date_range(start=start, end=start + duration, freq=freq, inclusive="both")


@safe_cache_data
def _get_block_series(series: pd.Series, block_names: str | list) -> pd.Series:
    if isinstance(block_names, str):
        block_names = [block_names]
    filtered = series[series.index.get_level_values("block").isin(block_names)].droplevel("block")
    return filtered


def _apply_timezone_preserve_local_time(
    ts: pd.Series | pd.DataFrame, local_tz: pytz.BaseTzInfo
) -> pd.Series | pd.DataFrame:
    if ts.index.tz:
        print("already TZ aware")
        ts.tz_convert(local_tz)
        return ts
    ts = pd.concat(
        [
            ts.tz_localize(None).tz_localize(tz=local_tz, nonexistent="shift_forward", ambiguous=use_dst)
            for use_dst in [
                False,
                True,
            ]  # make sure to repeat values in the "duplicated" hour caused by shifting from DST
        ],
        axis=0,
    )

    # The previous operation generates duplicates at two steps:
    #  (1) nonexistent="shift_forward" -> when shifting to DST (in european spring) all timesteps 02:00 <= t < 03:00
    #       in previous local time are now indexed as 03:00
    #  (2) use pd.concat() for the two timeseries
    # remove all duplicate index entries and keep only last (relevant for (1), irrelevant for (2)) value
    return ts.loc[~ts.index.duplicated(keep="last")].sort_index()


def _resample_ts(ts: pd.Series, dti: pd.DatetimeIndex, method: str = "numeric_mean"):
    """
    Resample a time series to a specified frequency and forward fill missing values.

    This function resamples the input time series (`ts`) based on the frequency defined in
    `dti` (a pandas `DatetimeIndex`) and applies forward filling to handle any missing values
    during resampling. An additional time step is added to the time series to ensure proper resampling.

    Args:
        ts (pd.Series): A pandas Series containing the time series data, with a DatetimeIndex.
        dti (pd.DatetimeIndex): A pandas DatetimeIndex that defines the desired resampling frequency.
        method (str, optional): The resampling method to use. Supported values are:
            `numeric_mean` (default): Resample by taking the mean over each interval and forward-fill missing values.
            `numeric_sum`: Resample by taking the sum over each interval and forward-fill missing values.
            `numeric_first`: Resample by taking the first non-zero value in each interval; zeros are treated as missing.
            `bool_all`: Resample using the min value of the interval (all elements have to be True).
            `bool_any`: Resample using the max value of the interval (at least one element has to be True).

    Returns:
        pd.Series: A resampled and forward-filled time series, with the same `DatetimeIndex` as `dti`.
    """

    freq_old = ts.index[1] - ts.index[0]
    freq_new = pd.Timedelta(dti.freq)

    if (freq_old > freq_new and freq_old.total_seconds() % freq_new.total_seconds() != 0) or (
        freq_old < freq_new and freq_new.total_seconds() % freq_old.total_seconds() != 0
    ):
        raise ValueError(
            "Cannot resample because the current and target frequencies are not compatible. "
            "Resampling is only supported when one frequency is an integer multiple of the other. "
            f"(current: {freq_old}, target: {freq_new})"
        )

    method_map = {
        "numeric_mean": lambda x: x.resample(freq_new).mean(),
        "numeric_sum": lambda x: x.resample(freq_new).sum(),
        "numeric_first": lambda x: x.mask(ts == 0).resample(freq_new).first().fillna(0.0),
        "bool_all": lambda x: x.resample(freq_new).min(),
        "bool_any": lambda x: x.resample(freq_new).max(),
    }

    if method not in method_map:
        raise NotImplementedError(f"Method {method} is not implemented.")

    # Add one additional entry to the timeseries to allow for proper resampling
    ts = ts.reindex(ts.index.union(ts.index + freq_old))

    return ts.pipe(method_map[method]).reindex(dti).ffill().bfill()


@safe_cache_data
def _get_log_pv(
    coordinates: Coordinates,
    start: pd.Timestamp,
    duration: pd.Timedelta,
    freq: pd.Timedelta,
) -> np.typing.NDArray[np.float64]:
    dti = _get_dti(start, duration, freq)

    data, *_ = pvlib.iotools.get_pvgis_hourly(
        latitude=coordinates.latitude,
        longitude=coordinates.longitude,
        start=int(dti.year.min()),
        end=int(dti.year.max()),
        raddatabase="PVGIS-SARAH3",
        outputformat="json",
        pvcalculation=True,
        peakpower=1,  # convert kWp to Wp
        pvtechchoice="crystSi",
        mountingplace="free",
        loss=0,
        trackingtype=0,  # fixed mount
        optimalangles=True,
        url="https://re.jrc.ec.europa.eu/api/v5_3/",
        map_variables=True,
        timeout=30,  # default value
    )
    data = data["P"] / 1000
    data.index = data.index.round("h")
    data = data.tz_convert("Europe/Berlin")
    data = _resample_ts(ts=data, dti=dti)
    return data.values


@safe_cache_data
def _get_log_dem(
    slp: str,
    consumption_yrl_wh: float,
    start: pd.Timestamp,
    duration: pd.Timedelta,
    freq: pd.Timedelta,
) -> np.typing.NDArray[np.float64]:
    dti = _get_dti(start, duration, freq)

    if slp not in ["h0", "h0_dyn", "g0", "g1", "g2", "g3", "g4", "g5", "g6", "g7", "l0", "l1", "l2"]:
        raise ValueError(f'Specified SLP "{slp}" is not valid. SLP has to be defined as lower case.')

    # get power profile in 15 minute timesteps
    ts = pd.concat(
        [
            (demandlib.bdew.ElecSlp(year=year).get_scaled_power_profiles({slp: consumption_yrl_wh}))
            for year in dti.year.unique()
        ]
    )[slp]

    # Time index ignores DST, but values adapt to DST -> apply new index with TZ information
    ts.index = pd.date_range(
        start=ts.index.min(), end=ts.index.max(), freq=ts.index.freq, inclusive="both", tz="Europe/Berlin"
    )

    return _resample_ts(ts=ts, dti=dti).values


@safe_cache_data
def _get_log_subfleet(
    vehicle_type: str,
    start: pd.Timestamp,
    duration: pd.Timedelta,
    freq: pd.Timedelta,
) -> pd.DataFrame:
    dti = _get_dti(start, duration, freq)

    # Create a dtype map to explicitly assign dtypes to columns
    with resources.files("lift.data.mobility").joinpath(f"log_{vehicle_type}.csv").open("r") as logfile:
        dtype_map = {}
        for col in pd.read_csv(logfile, nrows=0, header=[0, 1]).columns:
            suffix = col[1]
            if suffix in {"atbase", "atac", "atdc"}:
                dtype_map[col] = "boolean"
            elif suffix in {"dist", "consumption", "dsoc"}:
                dtype_map[col] = "float64"

    with resources.files("lift.data.mobility").joinpath(f"log_{vehicle_type}.csv").open("r") as logfile:
        df = pd.read_csv(logfile, header=[0, 1], engine="c", low_memory=False, dtype=dtype_map)
        df = (
            df.set_index(pd.to_datetime(df.iloc[:, 0], utc=True))
            .drop(df.columns[0], axis=1)
            .sort_index(
                axis=1,
                level=0,
                key=lambda x: x.map(
                    lambda s: int(m.group(1))
                    # get the last continuous sequence of digits if possible
                    if (m := re.search(r"(\d+)(?!.*\d)", s))
                    else s
                ),
                sort_remaining=True,
            )
            .tz_convert("Europe/Berlin")
        )

        method_sampling = {
            "atbase": "bool_all",
            "atac": "bool_any",
            "atdc": "bool_any",
            "dsoc": "numeric_first",
            "dist": "numeric_sum",
            "consumption": "numeric_mean",
        }

        # Efficiently apply the resampling function to each column in the DataFrame
        df = df.apply(
            lambda col: _resample_ts(
                ts=col,
                dti=dti,
                method=method_sampling[col.name[1]],
            ),
            axis=0,
        )

    return df.loc[dti, :]


@dataclass
class EcoObject(ABC):
    period_eco: int
    wacc: float

    @classmethod
    def _get_dict_from_settings(
        cls,
        settings: ScenarioSettings,
    ) -> dict:
        return {
            "period_eco": settings.period_eco,
            "wacc": settings.wacc,
        }

    @classmethod
    @abstractmethod
    def from_series_settings(
        cls,
        series: pd.Series,
        settings: ScenarioSettings,
        **kwargs,
    ) -> Self: ...

    @property
    @abstractmethod
    def capex(self) -> np.typing.NDArray: ...

    @property
    @abstractmethod
    def capem(self) -> np.typing.NDArray: ...

    @property
    @abstractmethod
    def opex(self) -> np.typing.NDArray: ...

    @property
    @abstractmethod
    def opem(self) -> np.typing.NDArray: ...

    @property
    def capex_discounted(self) -> np.typing.NDArray:
        # ToDo
        return self.capex

    @property
    def opex_discounted(self) -> np.typing.NDArray:
        # ToDo
        return self.opex

    @property
    def costs(self) -> np.typing.NDArray:
        return self.capex + self.opex

    @property
    def costs_discounted(self) -> np.typing.NDArray:
        return self.capex_discounted + self.opex_discounted

    @property
    def emissions(self) -> np.typing.NDArray:
        return self.capem + self.opem

    @property
    def total_costs(self) -> float:
        return float(np.sum(self.costs))

    @property
    def total_costs_discounted(self) -> float:
        return float(np.sum(self.costs_discounted))

    @property
    def total_emissions(self) -> float:
        return float(np.sum(self.emissions))


@dataclass
class BaseBlock(EcoObject, ABC):
    sim_start: pd.Timestamp
    sim_duration: pd.Timedelta
    sim_freq: pd.Timedelta

    _sim_result: SimResults = field(
        init=False,
        default=None,
    )

    @classmethod
    @abstractmethod
    def from_comparison_obj(
        cls,
        comparison_obj,
        phase: str,
    ) -> Self: ...

    @property
    def sim_result(self) -> SimResults:
        if self._sim_result is None:
            raise ValueError("Attribute sim_result is not set. Set sim_result before calculating results.")
        return self._sim_result

    @sim_result.setter
    def sim_result(self, value: SimResults):
        self._sim_result = value

    @property
    @abstractmethod
    def _opex_yrl(self) -> float: ...

    @property
    @abstractmethod
    def _opem_yrl(self) -> float: ...

    def _get_operational_cashflow_from_year(self, c):
        # allocate result
        opex = np.empty(self.period_eco + 1)

        # fill first N years with the scalar
        opex[: self.period_eco] = c

        # last element is zero
        opex[-1] = 0.0

        return opex

    @property
    @abstractmethod
    def capex(self) -> np.typing.NDArray: ...

    @property
    @abstractmethod
    def capem(self) -> np.typing.NDArray: ...

    @property
    def opex(self) -> np.typing.NDArray:
        return self._get_operational_cashflow_from_year(c=self._opex_yrl)

    @property
    def opem(self) -> np.typing.NDArray:
        return self._get_operational_cashflow_from_year(c=self._opem_yrl)


@dataclass
class InvestBlock(BaseBlock, ABC):
    ls: int

    @classmethod
    def _get_dict_from_settings(
        cls,
        settings: ScenarioSettings,
    ) -> dict:
        return super()._get_dict_from_settings(settings) | {
            "sim_start": settings.sim_start,
            "sim_duration": settings.sim_duration,
            "sim_freq": settings.sim_freq,
        }

    @classmethod
    def from_series_settings(
        cls,
        series: pd.Series,
        settings: ScenarioSettings,
        params: dict = None,
        **kwargs,
    ) -> Self:
        params = params or {}
        return cls(
            **series.to_dict(),
            **cls._get_dict_from_settings(settings),
            **params,
        )

    def _calc_replacements(self) -> np.typing.NDArray:
        years = np.arange(self.period_eco + 1)

        # Replacement years: start + 0, lifespan, 2*lifespan, ...
        replacement_years = np.arange(0, self.period_eco, self.ls)

        repl = np.isin(years, replacement_years).astype(float)

        # residual value
        repl[self.period_eco] = (
            (-1 * (1 - (self.period_eco % self.ls) / self.ls)) if self.period_eco % self.ls != 0 else 0
        )
        return repl

    @property
    @abstractmethod
    def _capex_single(self) -> float: ...

    @property
    @abstractmethod
    def _capem_single(self) -> float: ...

    @property
    def capex(self) -> np.typing.NDArray:
        return self._capex_single * self._calc_replacements()

    @property
    def capem(self) -> np.typing.NDArray:
        return self._capem_single * self._calc_replacements()


@dataclass
class ContinuousInvestBlock(InvestBlock, ABC):
    capacity: float
    capex_spec: float
    capem_spec: float
    opex_spec: float
    opem_spec: float

    @classmethod
    def from_comparison_obj(
        cls,
        comparison_obj: ComparisonInvestComponent,
        phase: str,
    ) -> Self:
        return cls(
            period_eco=comparison_obj.period_eco,
            ls=comparison_obj.ls,
            capacity=comparison_obj.capacity.get_value(phase),
            capex_spec=comparison_obj.capex_spec,
            capem_spec=comparison_obj.capem_spec,
            opex_spec=comparison_obj.opex_spec,
            opem_spec=comparison_obj.opem_spec,
        )

    @property
    def _capex_single(self) -> float:
        return self.capacity * self.capex_spec

    @property
    def _capem_single(self) -> float:
        return self.capacity * self.capem_spec


@dataclass
class Grid(ContinuousInvestBlock):
    opex_spec_buy: float
    opex_spec_sell: float
    opex_spec_peak: float

    opex_spec: float = field(default=None, init=False)

    @classmethod
    def from_comparison_obj(
        cls,
        comparison_obj: ComparisonGrid,
        phase: str,
    ) -> Self:
        return cls(
            period_eco=comparison_obj.period_eco,
            ls=comparison_obj.ls,
            capacity=comparison_obj.capacity.get_value(phase),
            capex_spec=comparison_obj.capex_spec,
            capem_spec=comparison_obj.capem_spec,
            opex_spec_buy=comparison_obj.opex_spec_buy,
            opex_spec_sell=comparison_obj.opex_spec_sell,
            opex_spec_peak=comparison_obj.opex_spec_peak,
            opem_spec=comparison_obj.opem_spec,
        )

    @property
    def _opex_yrl(self) -> float:
        return (
            self.sim_result.energy_grid_buy_wh * self.opex_spec_buy
            + self.sim_result.energy_grid_sell_wh * self.opex_spec_sell
            + self.sim_result.pwr_grid_peak_w * self.opex_spec_peak
        )

    @property
    def _opem_yrl(self) -> float:
        return self.sim_result.energy_grid_buy_wh * self.opem_spec


@dataclass
class PV(ContinuousInvestBlock):
    longitude: float
    latitude: float

    @classmethod
    def _get_dict_from_settings(
        cls,
        settings: ScenarioSettings,
    ) -> dict:
        return super()._get_dict_from_settings(settings) | {
            "longitude": settings.longitude,
            "latitude": settings.latitude,
        }

    @property
    def _opex_yrl(self) -> float:
        return 0

    @property
    def _opem_yrl(self) -> float:
        return 0


@dataclass
class ESS(ContinuousInvestBlock):
    @property
    def _opex_yrl(self) -> float:
        return 0

    @property
    def _opem_yrl(self) -> float:
        return 0


@dataclass
class FixedCosts(InvestBlock):
    capex_initial: float
    capem_initial: float

    @classmethod
    def _get_dict_from_settings(
        cls,
        settings: ScenarioSettings,
    ) -> dict:
        return super()._get_dict_from_settings(settings) | {
            "ls": settings.period_eco,  # one time investment
        }

    @classmethod
    def from_comparison_obj(
        cls,
        comparison_obj: ComparisonInvestComponent,
        phase: str,
    ) -> Self:
        return None

    @property
    def _capex_single(self) -> float:
        return self.capex_initial

    @property
    def _capem_single(self) -> float:
        return self.capem_initial

    @property
    def _opex_yrl(self) -> float:
        return 0

    @property
    def _opem_yrl(self) -> float:
        return 0


@dataclass
class ChargerType(InvestBlock):
    name: str
    num: int
    capex_per_unit: float
    capem_per_unit: float

    @classmethod
    def from_comparison_obj(
        cls,
        comparison_obj: ComparisonInputCharger,
        phase: str,
    ) -> Self:
        return cls(
            period_eco=comparison_obj.period_eco,
            ls=comparison_obj.ls,
            name=comparison_obj.name,
            num=comparison_obj.num.get_value(phase),
            capex_per_unit=comparison_obj.capex_per_unit,
            capem_per_unit=comparison_obj.capem_per_unit,
        )

    @property
    def _capex_single(self) -> float:
        return self.num * self.capex_per_unit

    @property
    def _capem_single(self) -> float:
        return self.num * self.capem_per_unit

    @property
    def _opex_yrl(self) -> float:
        return 0

    @property
    def _opem_yrl(self) -> float:
        return 0


@dataclass
class SubFleet(InvestBlock):
    name: str
    num_bev: int
    num_icev: int
    capex_per_unit_bev: float
    capem_per_unit_bev: float
    capex_per_unit_icev: float
    capem_per_unit_icev: float
    mntex_spec_bev: float
    mntex_spec_icev: float
    toll_frac: float
    toll_spec_bev: float
    toll_spec_icev: float
    consumption_spec_icev: float
    opex_spec_fuel: float
    opem_spec_fuel: float
    opex_spec_onroute_charging: float
    opem_spec_onroute_charging: float

    @classmethod
    def from_comparison_obj(
        cls,
        comparison_obj: ComparisonInputCharger,
        phase: str,
    ) -> Self:
        return cls(
            period_eco=comparison_obj.period_eco,
            ls=comparison_obj.ls,
            name=comparison_obj.name,
            num_bev=comparison_obj.num_bev.get_value(phase),
            num_icev=comparison_obj.num_icev.get_value(phase),
            capex_per_unit_bev=comparison_obj.capex_per_unit_bev,
            capem_per_unit_bev=comparison_obj.capem_per_unit_bev,
            capex_per_unit_icev=comparison_obj.capex_per_unit_icev,
            capem_per_unit_icev=comparison_obj.capem_per_unit_icev,
            mntex_spec_bev=comparison_obj.mntex_spec_bev,
            mntex_spec_icev=comparison_obj.mntex_spec_icev,
            toll_frac=comparison_obj.toll_frac,
            toll_spec_bev=comparison_obj.toll_spec_bev,
            toll_spec_icev=comparison_obj.toll_spec_icev,
            consumption_spec_icev=comparison_obj.consumption_spec_icev,
            opex_spec_fuel=comparison_obj.opex_spec_fuel,
            opem_spec_fuel=comparison_obj.opem_spec_fuel,
            opex_spec_onroute_charging=comparison_obj.opex_spec_onroute_charging,
            opem_spec_onroute_charging=comparison_obj.opem_spec_onroute_charging,
        )

    @property
    def _capex_single(self) -> float:
        return self.num_bev * self.capex_per_unit_bev + self.num_icev * self.capex_per_unit_icev

    @property
    def _capem_single(self) -> float:
        return self.num_bev * self.capem_per_unit_bev + self.num_icev * self.capem_per_unit_icev

    @property
    def _opex_yrl(self) -> float:
        opex_bev = (
            self.sim_result.dist_km[self.name]["bev"] * (self.mntex_spec_bev + self.toll_spec_bev * self.toll_frac)
            + self.sim_result.energy_fleet_route_wh * self.opex_spec_onroute_charging
        )  # ToDo: get energy per subfleet

        opex_icev = self.sim_result.dist_km[self.name]["icev"] * (
            self.mntex_spec_icev
            + self.toll_spec_icev * self.toll_frac
            + self.opex_spec_fuel * self.consumption_spec_icev / 100
        )

        return opex_bev + opex_icev

    @property
    def _opem_yrl(self) -> float:
        opem_icev = self.sim_result.dist_km[self.name]["icev"] * self.consumption_spec_icev / 100 * self.opem_spec_fuel

        opem_bev = (
            self.sim_result.energy_fleet_route_wh * self.opem_spec_onroute_charging
        )  # ToDo: get energy per subfleet

        return opem_bev + opem_icev


@dataclass
class Aggregator(EcoObject):
    subblocks: dict

    @property
    def capex(self) -> np.typing.NDArray:
        return sum([block.capex for block in self.blocks.values()])

    @property
    def capem(self) -> np.typing.NDArray:
        return sum([block.capem_per_unit for block in self.blocks.values()])

    @property
    def opex(self) -> np.typing.NDArray:
        return sum([block.opex for block in self.blocks.values()])

    @property
    def opem(self) -> np.typing.NDArray:
        return sum([block.opem for block in self.blocks.values()])


@dataclass
class Fleet(Aggregator):
    @classmethod
    def from_series_settings(
        cls,
        series: pd.Series,
        settings: ScenarioSettings,
        **kwargs,
    ) -> Self:
        block_dict = _get_block_series(series, "fleet").to_dict()
        block_dict.pop("subblocks")

        # extract global subfleet parameters defined in fleet -> don't store in fleet, but pass to subfleet blocks
        blocks_params = {
            key: block_dict.pop(key)
            for key in ["opex_spec_fuel", "opem_spec_fuel", "opex_spec_onroute_charging", "opem_spec_onroute_charging"]
        }

        return cls(
            **cls._get_dict_from_settings(settings),
            **block_dict,
            subblocks={
                sf_name: SubFleet.from_series_settings(
                    series=_get_block_series(series, sf_name),
                    settings=settings,
                    params=blocks_params,
                )
                for sf_name in ast.literal_eval(series[("fleet", "subblocks")])
            },
        )


@dataclass
class ChargingInfrastructure(Aggregator):
    pwr_max: float

    @classmethod
    def from_series_settings(
        cls,
        series: pd.Series,
        settings: ScenarioSettings,
        **kwargs,
    ) -> Self:
        block_dict = _get_block_series(series, "cis").to_dict()
        block_dict.pop("subblocks")

        return cls(
            **cls._get_dict_from_settings(settings),
            **block_dict,
            subblocks={
                subblock_name: ChargerType.from_series_settings(
                    series=_get_block_series(series, subblock_name),
                    settings=settings,
                )
                for subblock_name in ast.literal_eval(series[("cis", "subblocks")])
            },
        )


@dataclass
class ScenarioSettings:
    period_eco: int
    wacc: float
    sim_start: pd.Timestamp
    sim_duration: pd.Timedelta
    sim_freq: pd.Timedelta

    latitude: float
    longitude: float

    @classmethod
    def from_series_dict(cls, series: pd.Series) -> Self:
        return cls(
            period_eco=series["period_eco"],
            sim_start=pd.to_datetime(series["sim_start"]).tz_localize("Europe/Berlin"),
            sim_duration=pd.Timedelta(days=series["sim_duration"]),
            sim_freq=pd.Timedelta(hours=series["sim_freq"]),
            latitude=series["latitude"],
            longitude=series["longitude"],
            wacc=series["wacc"],
        )


@dataclass
class Scenario(Aggregator):
    subblocks: dict = field(default_factory=dict)

    @classmethod
    def from_series_settings(
        cls,
        series: pd.Series,
        settings: ScenarioSettings,
        **kwargs,
    ) -> Self:
        subblocks = {
            block_name: block_cls.from_series_settings(
                series=_get_block_series(series, block_name),
                settings=settings,
            )
            for block_name, block_cls in [("fix", FixedCosts), ("grid", Grid), ("pv", PV), ("ess", ESS)]
        } | {
            "fleet": Fleet.from_series_settings(
                series=series,
                settings=settings,
            ),
            "cis": ChargingInfrastructure.from_series_settings(
                series=series,
                settings=settings,
            ),
        }

        return cls(**cls._get_dict_from_settings(settings), subblocks=subblocks)

    @classmethod
    def from_series(cls, definition: pd.Series) -> Self:
        settings = ScenarioSettings.from_series_dict(
            definition[definition.index.get_level_values("block") == "scn"].droplevel("block").to_dict()
        )

        return cls.from_series_settings(definition, settings)


if __name__ == "__main__":
    # simple test
    from lift.backend.simulation.interfaces import SimResults
    from time import time

    start1 = time()
    df = pd.read_csv(Path("definition_transposed.csv"), index_col=0, header=[0, 1])
    series = df.loc["sc1", :]
    start2 = time()
    scn = Scenario.from_series(series)
    print(f"Scenario loaded in {time() - start1:.3f} s")
    print(f"Scenario object initialized in {time() - start2:.3f} s")

    pass
