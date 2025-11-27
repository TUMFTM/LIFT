from __future__ import annotations

from abc import ABC, abstractmethod
import ast
from dataclasses import dataclass, field
import importlib.resources as resources
from pathlib import Path
import re
from typing import Literal, Self
import warnings

import demandlib
import numpy as np
import pandas as pd
import pvlib


from lift.backend.comparison.interfaces import ComparisonGrid, ComparisonInvestComponent, ComparisonInputCharger
from lift.backend.simulation.interfaces import Coordinates
from lift.backend.utils import safe_cache_data


EPS = 1e-8


@safe_cache_data
def _td2h(td: pd.Timedelta) -> float:
    return td.total_seconds() / 3600.0


@safe_cache_data
def _calc_discount_factors(
    periods: int, occurs_at: Literal["beginning", "middle", "end"], discount_rate: float
) -> np.typing.NDArray[float]:
    if discount_rate is None or discount_rate < 0:
        raise ValueError("A positive discount rate must be provided if discounting is enabled.")

    periods = np.arange(0, periods + 1) + 1
    q = 1 + discount_rate

    exp = {"beginning": 1, "middle": 0.5, "end": 0}.get(occurs_at, 0)
    return 1 / (q ** (periods - exp))


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
    try:
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
    except:
        warnings.warn("Using random values for PV generation")
        return np.random.random(len(dti))


@safe_cache_data
def _get_log_dem(
    slp: str,
    e_yrl: float,
    start: pd.Timestamp,
    duration: pd.Timedelta,
    freq: pd.Timedelta,
) -> np.typing.NDArray[np.float64]:
    dti = _get_dti(start, duration, freq)

    if slp not in ["h0", "h0_dyn", "g0", "g1", "g2", "g3", "g4", "g5", "g6", "g7", "l0", "l1", "l2"]:
        raise ValueError(f'Specified SLP "{slp}" is not valid. SLP has to be defined as lower case.')

    # get power profile in 15 minute timesteps
    ts = pd.concat(
        [(demandlib.bdew.ElecSlp(year=year).get_scaled_power_profiles({slp: e_yrl})) for year in dti.year.unique()]
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


class GridPowerExceededError(Exception):
    pass


class SOCError(Exception):
    pass


@dataclass
class EcoObject(ABC):
    period_eco: int
    wacc: float
    parent: EcoObject | None

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
        block_name: str,
        settings: ScenarioSettings,
        parent: EcoObject | None,
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
        return (
            _calc_discount_factors(periods=self.period_eco, occurs_at="beginning", discount_rate=self.wacc) * self.capex
        )

    @property
    def opex_discounted(self) -> np.typing.NDArray:
        return _calc_discount_factors(periods=self.period_eco, occurs_at="end", discount_rate=self.wacc) * self.opex

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
class FixedDemand(BaseBlock):
    slp: str
    e_yrl: float

    log: np.typing.NDArray[np.float64] = field(default=None, init=False)

    def __post_init__(self):
        self.log = _get_log_dem(
            slp=self.slp, e_yrl=self.e_yrl, start=self.sim_start, duration=self.sim_duration, freq=self.sim_freq
        )

    @classmethod
    def from_series_settings(
        cls,
        series: pd.Series,
        block_name: str,
        settings: ScenarioSettings,
        parent: EcoObject | None,
        **kwargs,
    ) -> Self:
        return cls(
            **_get_block_series(series, block_name).to_dict(),
            **cls._get_dict_from_settings(settings),
            parent=parent,
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
        return np.zeros(self.period_eco + 1)

    @property
    def capem(self) -> np.typing.NDArray:
        return np.zeros(self.period_eco + 1)

    @property
    def e_site(self) -> float:
        return self.log.sum()


@dataclass
class InvestBlock(BaseBlock, ABC):
    ls: int

    @classmethod
    def from_series_settings(
        cls,
        series: pd.Series,
        block_name: str,
        settings: ScenarioSettings,
        parent: EcoObject | None,
        **kwargs,
    ) -> Self:
        return cls(
            **_get_block_series(series, block_name).to_dict(),
            **cls._get_dict_from_settings(settings),
            parent=parent,
            **kwargs,
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

    e_buy: float = field(default=0.0, init=False)
    e_sell: float = field(default=0.0, init=False)
    p_peak: float = field(default=0.0, init=False)

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
            self.e_buy += demand * _td2h(self.sim_freq)
            self.p_peak = max(self.p_peak, demand)
            return 0
        else:
            p_feed_in = min(-1 * demand, self.capacity)
            # selling to grid
            self.e_sell += p_feed_in * _td2h(self.sim_freq)
            return demand - p_feed_in


@dataclass
class PV(ContinuousInvestBlock):
    coordinates: Coordinates

    log: np.typing.NDArray[np.float64] = field(default=None, init=False)

    _p_curt: float = field(default=0.0, init=False)

    def __post_init__(self):
        self.log = (
            _get_log_pv(
                coordinates=self.coordinates, start=self.sim_start, duration=self.sim_duration, freq=self.sim_freq
            )
            * self.capacity
        )

    @classmethod
    def _get_dict_from_settings(
        cls,
        settings: ScenarioSettings,
    ) -> dict:
        return super()._get_dict_from_settings(settings) | {
            "coordinates": settings.coordinates,
        }

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
        return self._p_curt * _td2h(self.sim_freq)

    @property
    def e_pot(self) -> float:
        return np.sum(self.log) * _td2h(self.sim_freq)


@dataclass
class ESS(ContinuousInvestBlock):
    c_rate_max: float
    _soc: float = field(default=0.5, init=False)

    @property
    def _opex_yrl(self) -> float:
        return 0

    @property
    def _opem_yrl(self) -> float:
        return 0

    def get_p_max(self, idx):
        return min(
            self.capacity * self.c_rate_max,  # power limit due to c-rate
            self.capacity * self._soc / _td2h(self.sim_freq),  # power limit due to energy content)
        )

    def satisfy_demand(self, demand: float, idx):
        # discharging
        if demand > 0:
            p_ess = min(
                demand,
                self.capacity * self.c_rate_max,  # power limit due to c-rate
                self.capacity * self._soc / _td2h(self.sim_freq),  # power limit due to energy content
            )
        # charging
        else:
            p_ess = max(
                demand,  # max as all values are negative
                -self.capacity * self.c_rate_max,  # power limit due to c-rate
                -self.capacity * (1 - self._soc) / _td2h(self.sim_freq),  # power limit due to energy content
            )
        self._soc -= p_ess * _td2h(self.sim_freq) / self.capacity
        if self._soc < (0 - EPS) or self._soc > (1 + EPS):
            raise SOCError(
                f"SOC {self._soc} of block ESS out of bounds after applying power {demand} W in timestep {idx}."
            )

        return demand - p_ess


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
    p_max: float
    capex_per_unit: float
    capem_per_unit: float

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
class ElectricFleetUnit:
    sim_freq: pd.Timedelta
    name: str
    atbase: np.typing.NDArray[np.float64]
    consumption: np.typing.NDArray[np.float64]
    capacity: float
    charger: str
    p_max: float
    soc: float

    p_max_min: float = field(init=False)

    _soc_min: np.typing.NDArray[np.float64] = field(init=False)
    _soc_track: np.typing.NDArray[np.float64] = field(init=False)
    _p_site: np.typing.NDArray[np.float64] = field(init=False)
    _p_route: np.typing.NDArray[np.float64] = field(init=False)

    def __post_init__(self):
        self._soc_track = np.zeros(len(self.atbase) + 1)
        self._soc_track[0] = self.soc

        self._p_site = np.zeros_like(self.atbase)
        self._p_route = np.zeros_like(self.atbase)

        self._soc_min = _get_soc_min(
            max_charge_rate=self.p_max * _td2h(self.sim_freq) / self.capacity,
            dsoc=self.consumption * _td2h(self.sim_freq) / self.capacity,
            atbase=self.atbase,
        )

    @classmethod
    def from_subfleet(
        cls,
        name: str,
        atbase: np.typing.NDArray[np.float64],
        consumption: np.typing.NDArray[np.float64],
        subfleet: SubFleet,
    ):
        return cls(
            name=name,
            atbase=atbase,
            consumption=consumption,
            capacity=subfleet.capacity,
            charger=subfleet.charger,
            p_max=subfleet.p_max,
            soc=subfleet.soc_init,
            sim_freq=subfleet.sim_freq,
        )

    def time_flexibility(self, idx) -> float:
        if self.p_max == 0:  # division by 0 causes error
            return np.inf  # no charging possible, so lowest priority
        return float(self.soc - self._soc_min[idx]) * self.capacity / self.p_max

    def charge(self, p_available: float, idx: int):
        atbase = self.atbase[idx]

        if atbase == 1:
            p_site = min(
                self.p_max_min,  # maximum power based on charger and vehicle limits
                self.capacity * (1 - self.soc) / _td2h(self.sim_freq),  # maximum power based on SOC limit
                p_available,  # available power at site
            )
        else:
            p_site = 0
        self._p_site[idx] = p_site

        # update SOC based on charging power
        self.soc += (p_site - self.consumption[idx]) * _td2h(self.sim_freq) / self.capacity

        # on-route charging
        if self.soc < (-EPS) and not atbase:
            self._p_route[idx] = -1 * self.soc * self.capacity / _td2h(self.sim_freq)  # power needed to reach SOC=0
            self.soc = 0.0

        # catch errors
        if not ((-EPS) <= self.soc <= (1 + EPS)):
            raise SOCError(
                f"SOC {self.soc} of block {self.name} out of bounds after charging {p_site} W in timestep {idx}."
            )

        # apply soc tracking
        self._soc_track[idx + 1] = self.soc

        # return the charging power applied to this unit
        return p_site

    @property
    def e_site(self) -> float:
        return float(np.sum(self._p_site) * _td2h(self.sim_freq))

    @property
    def e_route(self) -> float:
        return float(np.sum(self._p_route) * _td2h(self.sim_freq))


@dataclass
class SubFleet(InvestBlock):
    # subfleet specific attributes
    name: str
    num_bev: int
    num_icev: int
    capacity: float
    charger: str
    p_max: float
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
    soc_init: float

    # global subfleet attributes
    opex_spec_fuel: float
    opem_spec_fuel: float
    opex_spec_onroute_charging: float
    opem_spec_onroute_charging: float

    def __post_init__(self):
        self.log = _get_log_subfleet(
            vehicle_type=self.name, start=self.sim_start, duration=self.sim_duration, freq=self.sim_freq
        )

        self.efus = {
            f"{self.name}{i}": ElectricFleetUnit.from_subfleet(
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

    @property
    def _opex_yrl(self) -> float:
        opex_bev = (
            self.sim_result.dist_km[self.name]["bev"] * (self.mntex_spec_bev + self.toll_spec_bev * self.toll_frac)
            + self.e_route * self.opex_spec_onroute_charging
        )  # ToDo: get distances

        opex_icev = self.sim_result.dist_km[self.name]["icev"] * (
            self.mntex_spec_icev
            + self.toll_spec_icev * self.toll_frac
            + self.opex_spec_fuel * self.consumption_spec_icev / 100
        )  # ToDo: get distances

        return opex_bev + opex_icev

    @property
    def _opem_yrl(self) -> float:
        # ToDo: get distances
        opem_icev = self.sim_result.dist_km[self.name]["icev"] * self.consumption_spec_icev / 100 * self.opem_spec_fuel

        opem_bev = self.e_route * self.opem_spec_onroute_charging

        return opem_bev + opem_icev

    @property
    def e_site(self) -> float:
        return sum([efu.e_site for efu in self.efus.values()])

    @property
    def e_route(self) -> float:
        return sum([efu.e_route for efu in self.efus.values()])


@dataclass
class Aggregator(EcoObject, ABC):
    subblocks: dict | None
    _series: pd.Series
    _settings: ScenarioSettings

    def __getattr__(self, item):
        if item in self.subblocks:
            return self.subblocks[item]
        else:
            return super().__getattribute__(item)

    @property
    def capex(self) -> np.typing.NDArray:
        return sum([block.capex for block in self.subblocks.values()])

    @property
    def capem(self) -> np.typing.NDArray:
        return sum([block.capem_per_unit for block in self.subblocks.values()])

    @property
    def opex(self) -> np.typing.NDArray:
        return sum([block.opex for block in self.subblocks.values()])

    @property
    def opem(self) -> np.typing.NDArray:
        return sum([block.opem for block in self.blocks.values()])


@dataclass
class Fleet(Aggregator):
    opex_spec_fuel: float
    opem_spec_fuel: float
    opex_spec_onroute_charging: float
    opem_spec_onroute_charging: float

    def __post_init__(self):
        params_subfleet_global = {
            "opex_spec_fuel": self.opex_spec_fuel,
            "opem_spec_fuel": self.opem_spec_fuel,
            "opex_spec_onroute_charging": self.opex_spec_onroute_charging,
            "opem_spec_onroute_charging": self.opem_spec_onroute_charging,
        }

        self.subblocks = {
            sf_name: SubFleet.from_series_settings(
                series=self._series,
                block_name=sf_name,
                settings=self._settings,
                parent=self,
                **params_subfleet_global,
            )
            for sf_name in ast.literal_eval(self._series[("fleet", "subblocks")])
        }

        delattr(self, "_series")
        delattr(self, "_settings")

    @classmethod
    def from_series_settings(
        cls,
        series: pd.Series,
        block_name: str,
        settings: ScenarioSettings,
        parent: EcoObject | None,
        **kwargs,
    ) -> Self:
        block_dict = _get_block_series(series, "fleet").to_dict()
        block_dict.pop("subblocks")

        return cls(
            **cls._get_dict_from_settings(settings),
            **block_dict,
            subblocks=None,
            parent=parent,
            _series=series,
            _settings=settings,
        )

    @property
    def e_site(self) -> float:
        return sum([sf.e_site for sf in self.subblocks.values()])

    @property
    def e_route(self) -> float:
        return sum([sf.e_route for sf in self.subblocks.values()])


@dataclass
class ChargingInfrastructure(Aggregator):
    p_lm_max: float

    def __post_init__(self):
        self.subblocks = {
            subblock_name: ChargerType.from_series_settings(
                series=series,
                block_name=subblock_name,
                settings=self._settings,
                parent=self,
            )
            for subblock_name in ast.literal_eval(series[("cis", "subblocks")])
        }
        delattr(self, "_series")
        delattr(self, "_settings")

    @classmethod
    def from_series_settings(
        cls,
        series: pd.Series,
        block_name: str,
        settings: ScenarioSettings,
        parent: EcoObject | None,
        **kwargs,
    ) -> Self:
        block_dict = _get_block_series(series, "cis").to_dict()
        block_dict.pop("subblocks")

        return cls(
            **cls._get_dict_from_settings(settings),
            **block_dict,
            _series=series,
            _settings=settings,
            subblocks=None,
            parent=parent,
        )


@dataclass
class ScenarioSettings:
    period_eco: int
    wacc: float
    sim_start: pd.Timestamp
    sim_duration: pd.Timedelta
    sim_freq: pd.Timedelta

    coordinates: Coordinates

    @classmethod
    def from_series_dict(cls, series: pd.Series) -> Self:
        return cls(
            period_eco=series["period_eco"],
            sim_start=pd.to_datetime(series["sim_start"]).tz_localize("Europe/Berlin"),
            sim_duration=pd.Timedelta(days=series["sim_duration"]),
            sim_freq=pd.Timedelta(hours=series["sim_freq"]),
            coordinates=Coordinates(latitude=series["latitude"], longitude=series["longitude"]),
            wacc=series["wacc"],
        )


@dataclass
class Scenario(Aggregator):
    sim_start: pd.Timestamp
    sim_duration: pd.Timedelta
    sim_freq: pd.Timedelta

    def __post_init__(self):
        self.subblocks = {
            block_name: block_cls.from_series_settings(
                series=series,
                block_name=block_name,
                settings=self._settings,
                parent=self,
            )
            for block_name, block_cls in [
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
                settings=self._settings,
                parent=self,
            ),
            "cis": ChargingInfrastructure.from_series_settings(
                series=series,
                block_name="cis",
                settings=self._settings,
                parent=self,
            ),
        }

        delattr(self, "_series")
        delattr(self, "_settings")

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
        block_name: str,
        settings: ScenarioSettings,
        parent: EcoObject | None,
        **kwargs,
    ) -> Self:
        return cls(
            **cls._get_dict_from_settings(settings),
            _series=series,
            _settings=settings,
            subblocks=None,
            parent=None,
        )

    @classmethod
    def from_series(cls, definition: pd.Series) -> Self:
        settings = ScenarioSettings.from_series_dict(
            definition[definition.index.get_level_values("block") == "scn"].droplevel("block").to_dict()
        )

        return cls.from_series_settings(series=definition, block_name="scn", settings=settings, parent=None)

    def simulate(self):
        efus = [efu for sf in scn.fleet.subblocks.values() for efu in sf.efus.values()]

        # get maximum charging power per EFU limited by EFU and specified charger
        for efu in efus:
            efu.p_max_min = min(efu.p_max, scn.cis.subblocks[efu.charger].p_max)

        for idx in range(len(_get_dti(self.sim_start, self.sim_duration, self.sim_freq))):
            # get FixedDemand power demand of timestep
            p_demand = self.dem.log[idx]

            if scn.cis.p_lm_max == np.inf:
                p_max_fleet = scn.grid.get_p_max(idx) + scn.pv.get_p_max(idx) + scn.ess.get_p_max(idx) - p_demand
            else:
                p_max_fleet = scn.cis.p_lm_max

            for efu in sorted(efus, key=lambda x: x.time_flexibility(idx)):
                p_efu = efu.charge(p_max_fleet, idx)
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
        pass

    @property
    def self_consumption(self) -> float:
        try:
            return 1 - ((self.pv.e_curt + self.grid.e_sell) / self.pv.e_pot)
        except ZeroDivisionError:
            return 0.0

    @property
    def self_sufficiency(self) -> float:
        try:
            return (self.pv.e_pot - self.pv.e_curt - self.grid.e_sell) / (self.fleet.e_site + self.dem.e_site)
        except ZeroDivisionError:
            return 0.0

    @property
    def home_charging_fraction(self) -> float:
        try:
            return self.fleet.e_site / (self.fleet.e_site + self.fleet.e_route)
        except ZeroDivisionError:
            return 0.0


if __name__ == "__main__":
    from time import time
    from transpose_csv import transpose_csv

    start1 = time()
    # transpose file -> not directly usable as variable types are only added automatically per column
    transpose_csv("definition.csv", save=True)
    df = pd.read_csv(Path("definition_transposed.csv"), index_col=0, header=[0, 1])

    series = df.loc["sc1", :]
    start2 = time()
    scn = Scenario.from_series(series)
    print(f"Scenario loaded in {time() - start1:.3f} s")
    print(f"Scenario object initialized in {time() - start2:.3f} s")
    start3 = time()
    scn.simulate()
    print(f"Simulation executed in {time() - start3:.3f} s")
    print(f"Self consumption: {scn.self_consumption * 100:.2f} %")
    print(f"Self sufficiency: {scn.self_sufficiency * 100:.2f} %")
    print(f"Depot charging: {scn.home_charging_fraction * 100:.2f} %")
