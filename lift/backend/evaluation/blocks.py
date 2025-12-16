from __future__ import annotations

from abc import ABC, abstractmethod
import ast
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
import importlib.resources as resources
from pathlib import Path
import re
from time import time
from typing import Any, Literal, Self
import warnings

import demandlib
import numpy as np
import pandas as pd
import pvlib

from lift.backend.comparison.interfaces import ExistExpansionValue
from lift.backend.utils import safe_cache_data


EPS = 1e-8


@safe_cache_data
def _get_block_series(series: pd.Series, block_names: str | list) -> pd.Series:
    if isinstance(block_names, str):
        block_names = [block_names]
    filtered = series[series.index.get_level_values("block").isin(block_names)].droplevel("block")
    return filtered


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
    settings_sim: SimSettings,
) -> np.typing.NDArray[np.float64]:
    try:
        data, *_ = pvlib.iotools.get_pvgis_hourly(
            latitude=coordinates.latitude,
            longitude=coordinates.longitude,
            start=int(settings_sim.dti.year.min()),
            end=int(settings_sim.dti.year.max()),
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
        data = _resample_ts(ts=data, dti=settings_sim.dti)
        return data.values
    except:
        warnings.warn("Using random values for PV generation")
        return np.random.random(len(settings_sim.dti))


@safe_cache_data
def _get_log_dem(
    slp: str,
    e_yrl: float,
    settings_sim: SimSettings,
) -> np.typing.NDArray[np.float64]:
    if slp not in ["h0", "h0_dyn", "g0", "g1", "g2", "g3", "g4", "g5", "g6", "g7", "l0", "l1", "l2"]:
        raise ValueError(f'Specified SLP "{slp}" is not valid. SLP has to be defined as lower case.')

    # get power profile in 15 minute timesteps
    ts = pd.concat(
        [
            (demandlib.bdew.ElecSlp(year=year).get_scaled_power_profiles({slp: e_yrl}))
            for year in settings_sim.dti.year.unique()
        ]
    )[slp]

    # Time index ignores DST, but values adapt to DST -> apply new index with TZ information
    ts.index = pd.date_range(
        start=ts.index.min(), end=ts.index.max(), freq=ts.index.freq, inclusive="both", tz="Europe/Berlin"
    )

    return _resample_ts(ts=ts, dti=settings_sim.dti).values


@safe_cache_data
def _get_log_subfleet(
    vehicle_type: str,
    settings_sim: SimSettings,
) -> pd.DataFrame:
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
                dti=settings_sim.dti,
                method=method_sampling[col.name[1]],
            ),
            axis=0,
        )

    return df.loc[settings_sim.dti, :]


@safe_cache_data
def _get_soc_min(max_charge_rate, dsoc, atbase):
    # cumulative sums of consumption and possible charging
    cum_dsoc = np.concatenate(([0.0], np.cumsum(dsoc)))
    cum_charge = np.concatenate(([0.0], np.cumsum(max_charge_rate * atbase)))

    # transform space: subtract available charging from required SOC
    t = cum_dsoc - cum_charge

    # reverse max accumulate and reverse back
    m = np.maximum.accumulate(t[::-1])[::-1]

    # translate back to SOC requirement at each timestep
    soc_min = m[1:] - t[:-1]

    # must be at least trip consumption
    soc_min = np.maximum(soc_min, dsoc)

    # no negative SOC
    return np.clip(soc_min, 0.0, None)


@dataclass
class Coordinates:
    latitude: float = 48.148
    longitude: float = 11.507

    @classmethod
    def from_frontend_coordinates(cls, frontend_coordinates: "FrontendCoordinates") -> Self:
        return cls(
            latitude=frontend_coordinates.latitude,
            longitude=frontend_coordinates.longitude,
        )

    @property
    def as_tuple(self) -> tuple[float, float]:
        return self.latitude, self.longitude

    @staticmethod
    def _decimal_to_dms(decimal_deg: float) -> tuple[int, int, float]:
        degrees = int(abs(decimal_deg))
        minutes_full = (abs(decimal_deg) - degrees) * 60
        minutes = int(minutes_full)
        seconds = (minutes_full - minutes) * 60
        return degrees, minutes, seconds


class GridPowerExceededError(Exception):
    pass


class SOCError(Exception):
    pass


@dataclass(frozen=True)
class SimSettings:
    sim_start: pd.Timestamp
    sim_duration: pd.Timedelta
    sim_freq: pd.Timedelta

    coordinates: Coordinates

    @classmethod
    def from_series_dict(cls, series: pd.Series) -> Self:
        return cls(
            sim_start=pd.Timestamp(series["sim_start"]).tz_localize("Europe/Berlin"),
            sim_duration=pd.Timedelta(days=series["sim_duration"]),
            sim_freq=pd.Timedelta(hours=series["sim_freq"]),
            coordinates=Coordinates(
                latitude=series["latitude"],
                longitude=series["longitude"],
            ),
        )

    @classmethod
    def from_comparison_object(cls, comp_obj: Any) -> Self:
        return cls(
            sim_start=comp_obj.sim_start,
            sim_duration=comp_obj.sim_duration,
            sim_freq=comp_obj.sim_freq,
            coordinates=Coordinates(
                latitude=comp_obj.latitude,
                longitude=comp_obj.longitude,
            ),
        )

    def _create_dti(self, inclusive: Literal["both", "left"]) -> pd.DatetimeIndex:
        return pd.date_range(
            start=self.sim_start, end=self.sim_start + self.sim_duration, freq=self.sim_freq, inclusive=inclusive
        )

    @cached_property
    def dti(self) -> pd.DatetimeIndex:
        return self._create_dti("left")

    @cached_property
    def dti_extended(self) -> pd.DatetimeIndex:
        return self._create_dti("both")

    @cached_property
    def sim_freq_h(self) -> float:
        return self.sim_freq.total_seconds() / 3600.0


class Occurrence(Enum):
    BEGINNING = 1
    MIDDLE = 0.5
    END = 0


@dataclass(frozen=True)
class EcoSettings:
    period_eco: int
    wacc: float

    @classmethod
    def from_series_dict(cls, series: pd.Series) -> Self:
        return cls(
            period_eco=series["period_eco"],
            wacc=series["wacc"],
        )

    @classmethod
    def from_comparison_object(cls, comp_obj: Any) -> Self:
        return cls(
            period_eco=comp_obj.period_eco,
            wacc=comp_obj.wacc,
        )

    def __post_init__(self):
        if self.wacc is None or self.wacc < 0:
            raise ValueError("A positive discount rate must be provided if discounting is enabled.")

    def _calc_discount_factors(self, occurs_at: Occurrence) -> np.typing.NDArray[float]:
        periods = np.arange(1, self.period_eco + 2)
        return np.reciprocal(np.power(1 + self.wacc, periods - occurs_at.value))  # np.reciprocal(x) ^= 1/x

    @cached_property
    def _discount_factors_cache(self) -> dict:
        # Cache discount factors for all occurrences
        return {
            Occurrence.BEGINNING: self._calc_discount_factors(Occurrence.BEGINNING),
            Occurrence.MIDDLE: self._calc_discount_factors(Occurrence.MIDDLE),
            Occurrence.END: self._calc_discount_factors(Occurrence.END),
        }

    def get_discount_factors(self, occurs_at: Occurrence) -> np.typing.NDArray[float]:
        # Retrieve the discount factors from the cache based on the occurrence
        return self._discount_factors_cache[occurs_at]


@dataclass
class ScenarioResult:
    self_sufficiency: float
    self_consumption: float
    home_charging_fraction: float
    capex_dis: np.typing.NDArray[np.floating]
    opex_dis: np.typing.NDArray[np.floating]
    totex_dis: np.typing.NDArray[np.floating]
    capem: np.typing.NDArray[np.floating]
    opem: np.typing.NDArray[np.floating]
    totem: np.typing.NDArray[np.floating]


class EcoObject(ABC):
    def __init__(self, settings_eco: EcoSettings):
        self.settings_eco = settings_eco

    @classmethod
    @abstractmethod
    def from_series_settings(
        cls,
        block_name: str,
        series: pd.Series,
        settings_eco: EcoSettings,
        **kwargs,
    ) -> Self: ...

    @classmethod
    @abstractmethod
    def from_comparison_object(
        cls,
        comp_obj: Any,
        settings_eco: EcoSettings,
        phase: str,
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
        return self.settings_eco.get_discount_factors(Occurrence.BEGINNING) * self.capex

    @property
    def opex_discounted(self) -> np.typing.NDArray:
        return self.settings_eco.get_discount_factors(Occurrence.END) * self.opex

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


class Aggregator(EcoObject, ABC):
    def __init__(
        self,
        settings_eco: EcoSettings,
        subblocks: dict[str, EcoObject],
    ):
        super().__init__(settings_eco=settings_eco)

        self.subblocks = subblocks

    def __getattr__(self, item):
        if item in self.subblocks:
            return self.subblocks[item]
        else:
            return super().__getattribute__(item)

    @property
    def capex(self) -> np.typing.NDArray:
        return sum([subblock.capex for subblock in self.subblocks.values()])

    @property
    def capem(self) -> np.typing.NDArray:
        return sum([subblock.capem for subblock in self.subblocks.values()])

    @property
    def opex(self) -> np.typing.NDArray:
        return sum([subblock.opex for subblock in self.subblocks.values()])

    @property
    def opem(self) -> np.typing.NDArray:
        return sum([subblock.opem for subblock in self.subblocks.values()])


class BaseBlock(EcoObject, ABC):
    def __init__(
        self,
        settings_eco: EcoSettings,
        settings_sim: SimSettings,
    ):
        super().__init__(settings_eco=settings_eco)
        self.settings_sim = settings_sim

    @classmethod
    def from_comparison_object(
        cls,
        comp_obj: Any,
        settings_eco: EcoSettings,
        settings_sim: SimSettings,
        phase: str,
        **kwargs,
    ) -> Self:
        return cls(
            **{
                key: (
                    getattr(getattr(comp_obj, key), phase)
                    if isinstance(getattr(comp_obj, key), ExistExpansionValue)
                    else getattr(comp_obj, key)
                )
                for key in comp_obj.to_dict.keys()
            },
            settings_eco=settings_eco,
            settings_sim=settings_sim,
            **kwargs,
        )

    @property
    @abstractmethod
    def _opex_yrl(self) -> float: ...

    @property
    @abstractmethod
    def _opem_yrl(self) -> float: ...

    def _get_operational_cashflow_from_year(self, c):
        # allocate result
        opex = np.empty(self.settings_eco.period_eco + 1)

        # fill first N years with the scalar
        opex[: self.settings_eco.period_eco] = c

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


class FixedDemand(BaseBlock):
    def __init__(
        self,
        settings_eco: EcoSettings,
        settings_sim: SimSettings,
        slp: str,
        e_yrl: float,
    ):
        super().__init__(
            settings_eco=settings_eco,
            settings_sim=settings_sim,
        )

        self.slp = slp
        self.e_yrl = e_yrl

        self.log = _get_log_dem(
            slp=self.slp,
            e_yrl=self.e_yrl,
            settings_sim=self.settings_sim,
        )

    @classmethod
    def from_series_settings(
        cls,
        block_name: str,
        series: pd.Series,
        settings_eco: EcoSettings,
        settings_sim: SimSettings,
        **kwargs,
    ) -> Self:
        return cls(
            settings_eco=settings_eco,
            settings_sim=settings_sim,
            **_get_block_series(series, block_name).to_dict(),
            **kwargs,
        )

    @property
    def _opex_yrl(self) -> float:
        return 0

    @property
    def _opem_yrl(self) -> float:
        return 0

    @property
    def capex(self) -> np.typing.NDArray:
        return np.zeros(self.settings_eco.period_eco + 1)

    @property
    def capem(self) -> np.typing.NDArray:
        return np.zeros(self.settings_eco.period_eco + 1)

    @property
    def e_site(self) -> float:
        return self.log.sum()


class InvestBlock(BaseBlock, ABC):
    def __init__(
        self,
        settings_eco: EcoSettings,
        settings_sim: SimSettings,
        ls: int,
    ):
        super().__init__(
            settings_eco=settings_eco,
            settings_sim=settings_sim,
        )

        self.ls = ls

    @classmethod
    def from_series_settings(
        cls,
        block_name: str,
        series: pd.Series,
        settings_eco: EcoSettings,
        settings_sim: SimSettings,
        **kwargs,
    ) -> Self:
        return cls(
            settings_eco=settings_eco,
            settings_sim=settings_sim,
            **_get_block_series(series, block_name).to_dict(),
            **kwargs,
        )

    def _calc_replacements(self) -> np.typing.NDArray:
        years = np.arange(self.settings_eco.period_eco + 1)

        # Replacement years: start + 0, lifespan, 2*lifespan, ...
        replacement_years = np.arange(0, self.settings_eco.period_eco, self.ls)

        repl = np.isin(years, replacement_years).astype(float)

        # residual value
        repl[self.settings_eco.period_eco] = (
            (-1 * (1 - (self.settings_eco.period_eco % self.ls) / self.ls))
            if self.settings_eco.period_eco % self.ls != 0
            else 0
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
    def __init__(
        self,
        settings_eco: EcoSettings,
        settings_sim: SimSettings,
        ls: int,
        capacity: float,
        capex_spec: float,
        capem_spec: float,
        opex_spec: float,
        opem_spec: float,
    ):
        super().__init__(
            settings_eco=settings_eco,
            settings_sim=settings_sim,
            ls=ls,
        )

        self.capacity = capacity
        self.capex_spec = capex_spec
        self.capem_spec = capem_spec
        self.opex_spec = opex_spec
        self.opem_spec = opem_spec

    @property
    def _capex_single(self) -> float:
        return self.capacity * self.capex_spec

    @property
    def _capem_single(self) -> float:
        return self.capacity * self.capem_spec


@dataclass
class Grid(ContinuousInvestBlock):
    def __init__(
        self,
        settings_eco: EcoSettings,
        settings_sim: SimSettings,
        ls: int,
        capacity: float,
        capex_spec: float,
        capem_spec: float,
        opem_spec: float,
        opex_spec_buy: float,
        opex_spec_sell: float,
        opex_spec_peak: float,
    ):
        super().__init__(
            settings_eco=settings_eco,
            settings_sim=settings_sim,
            ls=ls,
            capacity=capacity,
            capex_spec=capex_spec,
            capem_spec=capem_spec,
            opex_spec=0,
            opem_spec=opem_spec,
        )

        self.opex_spec_buy = opex_spec_buy
        self.opex_spec_sell = opex_spec_sell
        self.opex_spec_peak = opex_spec_peak

        self.e_buy = 0.0
        self.e_sell = 0.0
        self.p_peak = 0.0

    @classmethod
    def _keys_from_comparison_object(cls) -> list[str]:
        return [
            "period_eco",
            "wacc",
            "sim_start",
            "sim_duration",
            "sim_freq",
            "ls",
            "capacity",
            "capex_spec",
            "capem_spec",
            "opem_spec",
            "opex_spec_buy",
            "opex_spec_sell",
            "opex_spec_peak",
        ]

    @property
    def _opex_yrl(self) -> float:
        return self.e_buy * self.opex_spec_buy + self.e_sell * self.opex_spec_sell + self.p_peak * self.opex_spec_peak

    @property
    def _opem_yrl(self) -> float:
        return self.e_buy * self.opem_spec

    def get_p_max(self, idx):
        return self.capacity

    def satisfy_demand(self, demand: float, idx: int) -> float:
        if demand > self.capacity + EPS:
            raise GridPowerExceededError(
                f"Grid demand {demand} W exceeds grid capacity {self.capacity} W in timestep {idx}."
            )

        if demand > 0:
            # buying from grid
            self.e_buy += demand * self.settings_sim.sim_freq_h
            self.p_peak = max(self.p_peak, demand)
            return 0
        else:
            p_feed_in = min(-1 * demand, self.capacity)
            # selling to grid
            self.e_sell += p_feed_in * self.settings_sim.sim_freq_h
            return demand + p_feed_in


class PV(ContinuousInvestBlock):
    def __init__(
        self,
        settings_eco: EcoSettings,
        settings_sim: SimSettings,
        ls: int,
        capacity: float,
        capex_spec: float,
        capem_spec: float,
        opex_spec: float,
        opem_spec: float,
        log: np.typing.NDArray[np.float64] | None = None,
    ):
        super().__init__(
            settings_eco=settings_eco,
            settings_sim=settings_sim,
            ls=ls,
            capacity=capacity,
            capex_spec=capex_spec,
            capem_spec=capem_spec,
            opex_spec=opex_spec,
            opem_spec=opem_spec,
        )

        self.log = (
            log
            if log
            else (
                _get_log_pv(
                    coordinates=self.settings_sim.coordinates,
                    settings_sim=self.settings_sim,
                )
                * self.capacity
            )
        )

        self._p_curt = 0.0

    @property
    def _opex_yrl(self) -> float:
        return 0

    @property
    def _opem_yrl(self) -> float:
        return 0

    def get_p_max(self, idx):
        return self.log[idx]

    def satisfy_demand(self, demand, idx):
        return demand - self.log[idx]

    def curtail_power(self, p: float):
        self._p_curt += p

    @property
    def e_curt(self) -> float:
        return self._p_curt * self.settings_sim.sim_freq_h

    @property
    def e_pot(self) -> float:
        return np.sum(self.log) * self.settings_sim.sim_freq_h


class ESS(ContinuousInvestBlock):
    def __init__(
        self,
        settings_eco: EcoSettings,
        settings_sim: SimSettings,
        ls: int,
        capacity: float,
        capex_spec: float,
        capem_spec: float,
        opex_spec: float,
        opem_spec: float,
        soc_init: float,
        c_rate_max: float,
    ):
        super().__init__(
            settings_eco=settings_eco,
            settings_sim=settings_sim,
            ls=ls,
            capacity=capacity,
            capex_spec=capex_spec,
            capem_spec=capem_spec,
            opex_spec=opex_spec,
            opem_spec=opem_spec,
        )

        self.c_rate_max = c_rate_max
        self.soc = soc_init
        self._soc_track = np.zeros(len(self.settings_sim.dti) + 1)
        self._soc_track[0] = self.soc

    @property
    def _opex_yrl(self) -> float:
        return 0

    @property
    def _opem_yrl(self) -> float:
        return 0

    def get_p_max(self, idx):
        return min(
            self.capacity * self.c_rate_max,  # power limit due to c-rate
            self.capacity * self.soc / self.settings_sim.sim_freq_h,  # power limit due to energy content)
        )

    def satisfy_demand(self, demand: float, idx):
        if self.capacity == 0:
            return demand

        # discharging
        if demand > 0:
            p_ess = min(
                demand,
                self.capacity * self.c_rate_max,  # power limit due to c-rate
                self.capacity * self.soc / self.settings_sim.sim_freq_h,  # power limit due to energy content
            )
        # charging
        else:
            p_ess = max(
                demand,  # max as all values are negative
                -self.capacity * self.c_rate_max,  # power limit due to c-rate
                -self.capacity * (1 - self.soc) / self.settings_sim.sim_freq_h,  # power limit due to energy content
            )
        self.soc -= p_ess * self.settings_sim.sim_freq_h / self.capacity
        self._soc_track[idx] = self.soc
        if self.soc < (0 - EPS) or self.soc > (1 + EPS):
            raise SOCError(
                f"SOC {self.soc} of block ESS out of bounds after applying power {demand} W in timestep {idx}."
            )

        return demand - p_ess


class FixedCosts(InvestBlock):
    def __init__(
        self,
        settings_eco: EcoSettings,
        settings_sim: SimSettings,
        ls: int,
        capex_initial: float,
        capem_initial: float,
    ):
        super().__init__(
            settings_eco=settings_eco,
            settings_sim=settings_sim,
            ls=ls,
        )

        self.capex_initial = capex_initial
        self.capem_initial = capem_initial

    @classmethod
    def from_series_settings(
        cls,
        block_name: str,
        series: pd.Series,
        settings_eco: EcoSettings,
        settings_sim: SimSettings,
        **kwargs,
    ) -> Self:
        return super().from_series_settings(
            block_name=block_name,
            series=series,
            settings_eco=settings_eco,
            settings_sim=settings_sim,
            ls=settings_eco.period_eco,
            **kwargs,
        )

    @classmethod
    def from_comparison_object(
        cls,
        comp_obj: Any,
        settings_eco: EcoSettings,
        settings_sim: SimSettings,
        phase: str,
        **kwargs,
    ) -> Self:
        return super().from_comparison_object(
            comp_obj=comp_obj,
            settings_eco=settings_eco,
            settings_sim=settings_sim,
            phase=phase,
            ls=settings_eco.period_eco,
            **kwargs,
        )

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


class ChargerType(InvestBlock):
    def __init__(
        self,
        settings_eco: EcoSettings,
        settings_sim: SimSettings,
        ls: int,
        name: str,
        num: int,
        p_max: float,
        capex_per_unit: float,
        capem_per_unit: float,
    ):
        super().__init__(
            settings_eco=settings_eco,
            settings_sim=settings_sim,
            ls=ls,
        )

        self.name = name
        self.num = num
        self.p_max = p_max
        self.capex_per_unit = capex_per_unit
        self.capem_per_unit = capem_per_unit

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


class ElectricFleetUnit:
    def __init__(
        self,
        settings_sim: SimSettings,
        name: str,
        p_max: float,
        atbase: np.typing.NDArray[np.float64],
        consumption: np.typing.NDArray[np.float64],
        capacity: float,
        charger: str,
        soc_init: float,
    ):
        self.settings_sim = settings_sim
        self.name = name
        self.p_max = p_max
        self.atbase = atbase
        self.consumption = consumption
        self.capacity = capacity
        self.charger = charger
        self.soc = soc_init

        self.p_max_min = None

        self._soc_track = np.zeros(len(self.atbase) + 1)
        self._soc_track[0] = self.soc

        self._soc_min = _get_soc_min(
            max_charge_rate=self.p_max * self.settings_sim.sim_freq_h / self.capacity,
            dsoc=self.consumption * self.settings_sim.sim_freq_h / self.capacity,
            atbase=self.atbase,
        )

        self._p_site = np.zeros_like(self.atbase)
        self._p_route = np.zeros_like(self.atbase)

    @classmethod
    def from_subfleet(
        cls,
        settings_sim: SimSettings,
        name: str,
        atbase: np.typing.NDArray[np.float64],
        consumption: np.typing.NDArray[np.float64],
        subfleet: SubFleet,
    ):
        params_subfleet = {
            "p_max": subfleet.p_max,
            "capacity": subfleet.capacity,
            "charger": subfleet.charger,
            "soc_init": subfleet.soc_init,
        }

        return cls(settings_sim=settings_sim, name=name, atbase=atbase, consumption=consumption, **params_subfleet)

    def time_flexibility(self, idx) -> float:
        if self.p_max == 0:  # division by 0 causes error
            return np.inf  # no charging possible, so lowest priority
        return float(self.soc - self._soc_min[idx]) * self.capacity / self.p_max

    def charge(
        self,
        p_available: float,
        chargers: dict[str, int],
        idx: int,
    ):
        atbase = self.atbase[idx]

        if atbase == 1 and chargers[self.charger] > 0:
            p_site = min(
                self.p_max_min,  # maximum power based on charger and vehicle limits
                self.capacity * (1 - self.soc) / self.settings_sim.sim_freq_h,  # maximum power based on SOC limit
                p_available,  # available power at site
            )
            if p_site > 0:
                chargers[self.charger] -= 1
        else:
            p_site = 0
        self._p_site[idx] = p_site

        # update SOC based on charging power
        self.soc += (p_site - self.consumption[idx]) * self.settings_sim.sim_freq_h / self.capacity

        # on-route charging
        if self.soc < (-EPS) and not atbase:
            self._p_route[idx] = (
                -1 * self.soc * self.capacity / self.settings_sim.sim_freq_h
            )  # power needed to reach SOC=0
            self.soc = 0.0

        # catch errors
        if not ((-EPS) <= self.soc <= (1 + EPS)):
            raise SOCError(
                f"SOC {self.soc} of block {self.name} out of bounds after charging {p_site} W in timestep {idx}."
            )

        # apply soc tracking
        self._soc_track[idx + 1] = self.soc

        # return the charging power applied to this unit
        return p_site, chargers

    @property
    def e_site(self) -> float:
        return float(np.sum(self._p_site) * self.settings_sim.sim_freq_h)

    @property
    def e_route(self) -> float:
        return float(np.sum(self._p_route) * self.settings_sim.sim_freq_h)


class SubFleet(InvestBlock):
    def __init__(
        self,
        settings_eco: EcoSettings,
        settings_sim: SimSettings,
        ls: int,
        name: str,
        num_bev: int,
        num_icev: int,
        capacity: float,
        charger: str,
        p_max: float,
        capex_per_unit_bev: float,
        capem_per_unit_bev: float,
        capex_per_unit_icev: float,
        capem_per_unit_icev: float,
        mntex_spec_bev: float,
        mntex_spec_icev: float,
        toll_frac: float,
        toll_spec_bev: float,
        toll_spec_icev: float,
        consumption_spec_icev: float,
        soc_init: float,
        # global subfleet attributes
        opex_spec_fuel: float,
        opem_spec_fuel: float,
        opex_spec_onroute_charging: float,
        opem_spec_onroute_charging: float,
    ):
        super().__init__(
            settings_eco=settings_eco,
            settings_sim=settings_sim,
            ls=ls,
        )

        self.name = name
        self.num_bev = num_bev
        self.num_icev = num_icev
        self.capacity = capacity
        self.charger = charger
        self.p_max = p_max
        self.capex_per_unit_bev = capex_per_unit_bev
        self.capem_per_unit_bev = capem_per_unit_bev
        self.capex_per_unit_icev = capex_per_unit_icev
        self.capem_per_unit_icev = capem_per_unit_icev
        self.mntex_spec_bev = mntex_spec_bev
        self.mntex_spec_icev = mntex_spec_icev
        self.toll_frac = toll_frac
        self.toll_spec_bev = toll_spec_bev
        self.toll_spec_icev = toll_spec_icev
        self.consumption_spec_icev = consumption_spec_icev
        self.soc_init = soc_init
        self.opex_spec_fuel = opex_spec_fuel
        self.opem_spec_fuel = opem_spec_fuel
        self.opex_spec_onroute_charging = opex_spec_onroute_charging
        self.opem_spec_onroute_charging = opem_spec_onroute_charging

        self.log = _get_log_subfleet(
            vehicle_type=self.name,
            settings_sim=self.settings_sim,
        )

        self.efus = {
            f"{self.name}{i}": ElectricFleetUnit.from_subfleet(
                settings_sim=self.settings_sim,
                name=f"{self.name}{i}",
                atbase=self.log.loc[:, (f"{self.name}{i}", "atbase")].astype(int).values,
                consumption=self.log.loc[:, (f"{self.name}{i}", "consumption")].astype(int).values,
                subfleet=self,
            )
            for i in range(self.num_bev)
        }

    @property
    def _capex_single(self) -> float:
        return self.num_bev * self.capex_per_unit_bev + self.num_icev * self.capex_per_unit_icev

    @property
    def _capem_single(self) -> float:
        return self.num_bev * self.capem_per_unit_bev + self.num_icev * self.capem_per_unit_icev

    @cached_property
    def _dist_yrl_bev(self) -> float:
        # ToDo: scale to yearly distance
        return sum([self.log[(f"{self.name}{i}", "dist")].sum() for i in range(0, self.num_bev)])

    @cached_property
    def _dist_yrl_icev(self) -> float:
        # ToDo: scale to yearly distance
        return sum([self.log[(f"{self.name}{i}", "dist")].sum() for i in range(self.num_bev, self.num_icev)])

    @property
    def _opex_yrl(self) -> float:
        opex_bev = (
            self._dist_yrl_bev * (self.mntex_spec_bev + self.toll_spec_bev * self.toll_frac)
            + self.e_route * self.opex_spec_onroute_charging
        )

        opex_icev = self._dist_yrl_icev * (
            self.mntex_spec_icev
            + self.toll_spec_icev * self.toll_frac
            + self.opex_spec_fuel * self.consumption_spec_icev / 100
        )

        return opex_bev + opex_icev

    @property
    def _opem_yrl(self) -> float:
        opem_icev = self._dist_yrl_icev * self.consumption_spec_icev / 100 * self.opem_spec_fuel
        opem_bev = self.e_route * self.opem_spec_onroute_charging

        return opem_bev + opem_icev

    @property
    def e_site(self) -> float:
        return sum([efu.e_site for efu in self.efus.values()])

    @property
    def e_route(self) -> float:
        return sum([efu.e_route for efu in self.efus.values()])


class Fleet(Aggregator):
    def __init__(
        self,
        settings_eco: EcoSettings,
        subblocks=dict[str, SubFleet],
    ):
        super().__init__(
            settings_eco=settings_eco,
            subblocks=subblocks,
        )

    @classmethod
    def from_series_settings(
        cls,
        block_name: str,
        series: pd.Series,
        settings_eco: EcoSettings,
        settings_sim: SimSettings,
        **kwargs,
    ) -> Self:
        block_dict = _get_block_series(series, "fleet").to_dict()

        # get subfleet parameters which are defined on fleet level
        params_subblocks = {
            k: block_dict.pop(k)
            for k in ["opex_spec_fuel", "opem_spec_fuel", "opex_spec_onroute_charging", "opem_spec_onroute_charging"]
        }

        # create subblock dict with SubFleet objects
        subblocks = {
            sf_name: SubFleet.from_series_settings(
                block_name=sf_name,
                series=series,
                settings_eco=settings_eco,
                settings_sim=settings_sim,
                **params_subblocks,
            )
            for sf_name in ast.literal_eval(block_dict.pop("subblocks"))
        }

        return cls(
            settings_eco=settings_eco,
            **block_dict,
            subblocks=subblocks,
        )

    @classmethod
    def from_comparison_object(
        cls,
        comp_obj: Any,
        settings_eco: EcoSettings,
        settings_sim: SimSettings,
        phase: str,
        **kwargs,
    ) -> Self:
        dict_in = comp_obj.to_dict

        params_subblocks = {
            k: dict_in.pop(k)
            for k in ["opex_spec_fuel", "opem_spec_fuel", "opex_spec_onroute_charging", "opem_spec_onroute_charging"]
        }

        subblocks = {
            k: SubFleet.from_comparison_object(
                comp_obj=v,
                settings_eco=settings_eco,
                settings_sim=settings_sim,
                phase=phase,
                **params_subblocks,
            )
            for k, v in dict_in.pop("subblocks").items()
        }

        return cls(
            **{
                key: (
                    getattr(getattr(comp_obj, key), phase)
                    if isinstance(getattr(comp_obj, key), ExistExpansionValue)
                    else getattr(comp_obj, key)
                )
                for key in dict_in
            },
            settings_eco=settings_eco,
            subblocks=subblocks,
        )

    @property
    def e_site(self) -> float:
        return sum([sf.e_site for sf in self.subblocks.values()])

    @property
    def e_route(self) -> float:
        return sum([sf.e_route for sf in self.subblocks.values()])


class ChargingInfrastructure(Aggregator):
    def __init__(
        self,
        settings_eco: EcoSettings,
        subblocks: dict[str, ChargerType],
        p_lm_max: float,
    ):
        super().__init__(
            settings_eco=settings_eco,
            subblocks=subblocks,
        )

        self.p_lm_max = p_lm_max

    @classmethod
    def from_series_settings(
        cls,
        block_name: str,
        series: pd.Series,
        settings_eco: EcoSettings,
        settings_sim: SimSettings,
        **kwargs,
    ) -> Self:
        block_dict = _get_block_series(series, "cis").to_dict()

        # get subfleet parameters which are defined on fleet level
        params_subblocks = {k: block_dict.pop(k) for k in []}

        # create subblock dict with SubFleet objects
        subblocks = {
            chg_name: ChargerType.from_series_settings(
                block_name=chg_name,
                series=series,
                settings_eco=settings_eco,
                settings_sim=settings_sim,
                **params_subblocks,
            )
            for chg_name in ast.literal_eval(block_dict.pop("subblocks"))
        }

        return cls(
            settings_eco=settings_eco,
            **block_dict,
            subblocks=subblocks,
        )

    @classmethod
    def from_comparison_object(
        cls,
        comp_obj: Any,
        settings_eco: EcoSettings,
        settings_sim: SimSettings,
        phase: str,
        **kwargs,
    ) -> Self:
        dict_in = comp_obj.to_dict

        subblocks = {
            k: ChargerType.from_comparison_object(
                comp_obj=v,
                settings_eco=settings_eco,
                settings_sim=settings_sim,
                phase=phase,
            )
            for k, v in dict_in.pop("subblocks").items()
        }

        return cls(
            **{
                key: (
                    getattr(getattr(comp_obj, key), phase)
                    if isinstance(getattr(comp_obj, key), ExistExpansionValue)
                    else getattr(comp_obj, key)
                )
                for key in dict_in
            },
            settings_eco=settings_eco,
            subblocks=subblocks,
            **kwargs,
        )


@dataclass
class Scenario(Aggregator):
    def __init__(
        self,
        settings_eco: EcoSettings,
        settings_sim: SimSettings,
        subblocks: dict[str, EcoObject],
    ):
        super().__init__(
            settings_eco=settings_eco,
            subblocks=subblocks,
        )
        self.settings_sim = settings_sim

    @classmethod
    def from_series_settings(
        cls,
        block_name: str,
        series: pd.Series,
        settings_eco: EcoSettings,
        settings_sim: SimSettings,
        **kwargs,
    ) -> Self:
        subblocks = {
            block: block_cls.from_series_settings(
                series=series,
                block_name=block,
                settings_eco=settings_eco,
                settings_sim=settings_sim,
            )
            for block, block_cls in [
                ("fix", FixedCosts),
                ("grid", Grid),
                ("pv", PV),
                ("ess", ESS),
                ("dem", FixedDemand),
            ]
        } | {
            "fleet": Fleet.from_series_settings(
                series=series,
                block_name="fleet",
                settings_eco=settings_eco,
                settings_sim=settings_sim,
            ),
            "cis": ChargingInfrastructure.from_series_settings(
                series=series,
                block_name="cis",
                settings_eco=settings_eco,
                settings_sim=settings_sim,
            ),
        }

        return cls(
            settings_eco=settings_eco,
            settings_sim=settings_sim,
            subblocks=subblocks,
        )

    @classmethod
    def from_series(cls, definition: pd.Series) -> Self:
        series_scn = definition[definition.index.get_level_values("block") == "scn"].droplevel("block").to_dict()
        settings_eco = EcoSettings.from_series_dict(pd.Series(series_scn))
        settings_sim = SimSettings.from_series_dict(pd.Series(series_scn))

        return cls.from_series_settings(
            block_name="scn",
            series=definition,
            settings_eco=settings_eco,
            settings_sim=settings_sim,
        )

    @classmethod
    def from_comparison_object(
        cls,
        comp_obj: Any,
        settings_eco: EcoSettings,
        settings_sim: SimSettings,
        phase: str,
        **kwargs,
    ) -> Self:
        comp_dict = comp_obj.to_dict
        cls_map = {
            "fix": FixedCosts,
            "grid": Grid,
            "pv": PV,
            "ess": ESS,
            "dem": FixedDemand,
            "fleet": Fleet,
            "cis": ChargingInfrastructure,
        }

        subblocks = {
            k: cls_map[k].from_comparison_object(
                comp_obj=v,
                settings_eco=settings_eco,
                settings_sim=settings_sim,
                phase=phase,
            )
            for k, v in comp_dict.items()
        }

        return cls(
            settings_eco=settings_eco,
            settings_sim=settings_sim,
            subblocks=subblocks,
        )

    @classmethod
    def from_comparison(
        cls,
        comp_obj: Any,
        phase: str,
    ) -> Self:
        settings_eco = EcoSettings.from_comparison_object(comp_obj.settings)
        settings_sim = SimSettings.from_comparison_object(comp_obj.settings)

        return cls.from_comparison_object(
            comp_obj=comp_obj,
            settings_eco=settings_eco,
            settings_sim=settings_sim,
            phase=phase,
        )

    def simulate(self):
        start = time()
        efus = [efu for sf in self.fleet.subblocks.values() for efu in sf.efus.values()]

        # get maximum charging power per EFU limited by EFU and specified charger
        for efu in efus:
            efu.p_max_min = min(efu.p_max, self.cis.subblocks[efu.charger].p_max)

        for idx in range(len(self.settings_sim.dti)):
            # get FixedDemand power demand of timestep
            p_demand = self.dem.log[idx]

            if self.cis.p_lm_max == np.inf:
                p_max_fleet = self.grid.get_p_max(idx) + self.pv.get_p_max(idx) + self.ess.get_p_max(idx) - p_demand
            else:
                p_max_fleet = self.cis.p_lm_max

            chargers = {k: v.num for k, v in self.cis.subblocks.items()}

            for efu in sorted(efus, key=lambda x: x.time_flexibility(idx)):
                p_efu, chargers = efu.charge(p_available=p_max_fleet, chargers=chargers, idx=idx)
                p_max_fleet -= p_efu
                p_demand += p_efu

            # apply PV generation on demand
            p_demand = self.pv.satisfy_demand(p_demand, idx)

            # apply ESS to satisfy remaining demand or charge from excess generation
            p_demand = self.ess.satisfy_demand(p_demand, idx)

            # apply grid to satisfy remaining demand or feed excess generation
            p_demand = self.grid.satisfy_demand(
                p_demand, idx
            )  # demand should be negative (excess for curtailment ) or zero here

            self.pv.curtail_power(-1 * p_demand)

        print(f"Scenario simulation time: {time() - start:.4f} s")

        return ScenarioResult(
            self_sufficiency=self.self_sufficiency,
            self_consumption=self.self_consumption,
            home_charging_fraction=self.home_charging_fraction,
            capex_dis=self.capex_discounted,
            opex_dis=self.opex_discounted,
            totex_dis=self.costs_discounted,
            capem=self.capem,
            opem=self.opem,
            totem=self.emissions,
        )

    @property
    def self_consumption(self) -> float:
        if self.pv.e_pot > 0:
            return 1 - ((self.pv.e_curt + self.grid.e_sell) / self.pv.e_pot)
        else:
            return 0.0

    @property
    def self_sufficiency(self) -> float:
        if (e_site := (self.fleet.e_site + self.dem.e_site)) > 0:
            return (self.pv.e_pot - self.pv.e_curt - self.grid.e_sell) / e_site
        else:
            return 0.0

    @property
    def home_charging_fraction(self) -> float:
        try:
            return self.fleet.e_site / (self.fleet.e_site + self.fleet.e_route)
        except ZeroDivisionError:
            return 0.0


if __name__ == "__main__":
    from transpose_csv import transpose_csv

    start1 = time()
    # transpose file -> not directly usable as variable types are only added automatically per column
    transpose_csv("definition.csv", save=True)
    df = pd.read_csv(Path("definition_transposed.csv"), index_col=0, header=[0, 1])

    series_scn = df.loc["sc1", :]
    start2 = time()
    scn = Scenario.from_series(series_scn)
    print(f"Scenario loaded in {time() - start1:.3f} s")
    print(f"Scenario object initialized in {time() - start2:.3f} s")
    start3 = time()
    scn.simulate()
    print(f"Simulation executed in {time() - start3:.3f} s")
    print(f"Self consumption: {scn.self_consumption * 100:.2f} %")
    print(f"Self sufficiency: {scn.self_sufficiency * 100:.2f} %")
    print(f"Depot charging: {scn.home_charging_fraction * 100:.2f} %")
