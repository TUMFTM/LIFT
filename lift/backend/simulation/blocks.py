"""Simulation blocks for depot energy system and fleet.

Purpose:
- Provide composable, time-stepped models for site demand, fleet units/aggregation,
  PV generation, stationary storage, and grid connection, including constraint handling.

Relationships:
- Consumes typed inputs from `lift.backend.simulation.interfaces`.
- Used by `lift.backend.simulation.simulation.simulate` to form the system and iterate over time.
- Utilizes `safe_cache_data` for expensive, deterministic data retrieval (e.g., PVGIS, logs).

Key Logic / Formulations:
- Resampling utilities align external time series to the simulation index with strict frequency checks.
- PVSource: uses PVGIS for normalized power (per 1 kWp) then scales to `pwr_wp` and aggregates potential.
- StationaryStorage: symmetric c-rate limit; charge/discharge bounded by SOC window; updates SOC each step.
- GridConnection: caps import/export, records peak demand; raises `GridPowerExceededError` on exceedance.
- FixedDemand: constructs SLP-based site demand time series.
- Fleet/FleetUnit: per-vehicle state (SOC, base presence, consumption); priority charging via time flexibility;
  route charging tracked when SOC would go negative off-base; raises `SOCError` on infeasible SOC.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import importlib.resources as resources
import re
from typing import Self

import demandlib
import numpy as np
import pandas as pd
import pvlib
import pytz

from .interfaces import (
    SimInputSettings,
    SimInputSubfleet,
    SimInputCharger,
    GridPowerExceededError,
    SOCError,
    Coordinates,
)

from lift.backend.utils import safe_cache_data

EPS = 1e-8  # Small epsilon value for numerical stability in calculations


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
    settings: SimInputSettings,
) -> np.typing.NDArray[np.float64]:
    dti = settings.dti

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
def _get_log_dem(slp: str, consumption_yrl_wh: float, settings: SimInputSettings) -> np.typing.NDArray[np.float64]:
    if slp not in ["h0", "h0_dyn", "g0", "g1", "g2", "g3", "g4", "g5", "g6", "g7", "l0", "l1", "l2"]:
        raise ValueError(f'Specified SLP "{slp}" is not valid. SLP has to be defined as lower case.')

    # get power profile in 15 minute timesteps
    ts = pd.concat(
        [
            (demandlib.bdew.ElecSlp(year=year).get_scaled_power_profiles({slp: consumption_yrl_wh}))
            for year in settings.dti.year.unique()
        ]
    )[slp]

    # Time index ignores DST, but values adapt to DST -> apply new index with TZ information
    ts.index = pd.date_range(
        start=ts.index.min(), end=ts.index.max(), freq=ts.index.freq, inclusive="both", tz="Europe/Berlin"
    )

    return _resample_ts(ts=ts, dti=settings.dti).values


@safe_cache_data
def _get_log_subfleet(vehicle_type: str, settings: SimInputSettings) -> pd.DataFrame:
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
                dti=settings.dti,
                method=method_sampling[col.name[1]],
            ),
            axis=0,
        )

    return df.loc[settings.dti, :]


@dataclass
class Block(ABC):
    dti: pd.DatetimeIndex
    freq_hours: float
    _idx: int = field(init=False)

    @property
    def idx(self) -> int:
        return self._idx

    @idx.setter
    def idx(self, value: int):
        self._idx = value


@dataclass
class DemandBlock(Block):
    @property
    @abstractmethod
    def demand_w(self) -> float:
        # Return the demand in W for the current dt.
        ...


@dataclass
class SupplyBlock(Block):
    @abstractmethod
    def generation_max_w(self) -> float:
        """
        Return the maximum generation capacity in W for the current dt.
        """
        ...

    @abstractmethod
    def satisfy_demand(self, demand_w: float) -> float:
        """
        Apply power to the supply block and return remaining demand.
        """
        ...


@dataclass
class PVSource(SupplyBlock):
    pwr_wp: float
    log_spec: np.typing.NDArray[np.float64]
    log: np.typing.NDArray[np.float64]

    @classmethod
    def from_parameters(cls, settings: SimInputSettings, pwr_wp: float, coordinates: Coordinates) -> Self:
        log_spec = _get_log_pv(coordinates=coordinates, settings=settings)
        return cls(
            dti=settings.dti, freq_hours=settings.freq_hours, pwr_wp=pwr_wp, log_spec=log_spec, log=log_spec * pwr_wp
        )

    @property
    def generation_max_w(self) -> float:
        return float(self.log[self.idx])

    @property
    def energy_pot_wh(self) -> float:
        return sum(self.log_spec * self.pwr_wp * self.freq_hours)

    def satisfy_demand(self, demand_w: float) -> float:
        # Return remaining power demand after PV generation (negative if excess generation)
        return demand_w - self.generation_max_w


@dataclass
class StationaryStorage(SupplyBlock):
    capacity_wh: float
    soc: float = field(init=False, default=1.0)
    c_rate_max: float = field(init=False, default=0.5)

    @classmethod
    def from_parameters(cls, settings: SimInputSettings, capacity_wh: float) -> Self:
        return cls(
            dti=settings.dti,
            freq_hours=settings.freq_hours,
            capacity_wh=capacity_wh,
        )

    @property
    def _pwr_max_crate_w(self) -> float:
        return self.capacity_wh * self.c_rate_max

    @property
    def _pwr_max_chg_w(self) -> float:
        return min(
            self._pwr_max_crate_w,  # power limit due to c-rate
            self.capacity_wh * (1 - self.soc) / self.freq_hours,
        )  # power limit due to current SOC

    @property
    def _pwr_max_dis_w(self) -> float:
        return min(
            self._pwr_max_crate_w,  # power limit due to c-rate
            self.capacity_wh * self.soc / self.freq_hours,
        )  # power limit due to current SOC

    @property
    def generation_max_w(self) -> float:
        return self._pwr_max_dis_w

    def satisfy_demand(self, demand_w: float) -> float:
        if self.capacity_wh == 0:
            return demand_w

        # Apply power to the stationary storage, charging or discharging as needed.
        if demand_w > 0:  # discharging
            pwr_ess = min(self._pwr_max_dis_w, demand_w)
        else:  # charging
            pwr_ess = max(-1 * self._pwr_max_chg_w, demand_w)

        dsoc = -1 * pwr_ess * self.freq_hours / self.capacity_wh  # Change in SOC based on power applied
        self.soc += dsoc
        if self.soc < (0 - EPS) or self.soc > (1 + EPS):
            raise ValueError(f"SOC {self.soc} out of bounds after applying power {demand_w} W at {self.dti[self.idx]}.")

        # Return remaining power demand after storage (negative, if PV excess generation cannot be charged into storage)
        return demand_w - pwr_ess


@dataclass
class GridConnection(SupplyBlock):
    pwr_max_w: float

    pwr_peak_w: float = field(init=False, default=0.0)

    _pwr_buy_w: float = field(init=False, default=0.0)

    _pwr_sell_w: float = field(init=False, default=0.0)

    _pwr_curt_w: float = field(init=False, default=0.0)

    @classmethod
    def from_parameters(cls, settings: SimInputSettings, pwr_max_w: float) -> Self:
        return cls(
            dti=settings.dti,
            freq_hours=settings.freq_hours,
            pwr_max_w=pwr_max_w,
        )

    @property
    def generation_max_w(self) -> float:
        # Return the maximum power that can be supplied by the grid connection.
        return self.pwr_max_w

    def satisfy_demand(self, demand_w: float):
        # Apply power to the grid connection, updating peak power and costs/revenue.
        if demand_w > self.pwr_max_w + EPS:
            raise GridPowerExceededError(
                f"Demand {demand_w} W exceeds maximum power {self.pwr_max_w} W at {self.dti[self.idx]}."
            )

        if demand_w > 0:
            self.pwr_peak_w = max(self.pwr_peak_w, demand_w)
            self._pwr_buy_w += demand_w
        else:
            self._pwr_sell_w += min(-demand_w, self.pwr_max_w)
            self._pwr_curt_w += max(-demand_w - self.pwr_max_w, 0)

    @property
    def energy_buy_wh(self) -> float:
        return self._pwr_buy_w * self.freq_hours

    @property
    def energy_sell_wh(self) -> float:
        return self._pwr_sell_w * self.freq_hours

    @property
    def energy_curt_wh(self) -> float:
        return self._pwr_curt_w * self.freq_hours


@dataclass
class FixedDemand(DemandBlock):
    log: np.typing.NDArray[np.float64]

    @classmethod
    def from_parameters(cls, settings: SimInputSettings, consumption_yrl_wh: float, slp: str) -> Self:
        log = _get_log_dem(slp=slp, consumption_yrl_wh=consumption_yrl_wh, settings=settings)
        return cls(dti=settings.dti, freq_hours=settings.freq_hours, log=log)

    @property
    def demand_w(self) -> float:
        # Return the demand in W for the current dt.
        return float(self.log[self.idx])

    @property
    def energy_wh(self) -> float:
        # Return the total energy demand in Wh over the simulation period.
        return sum(self.log) * self.freq_hours


@dataclass
class Fleet(DemandBlock):
    pwr_lim_w: float
    log: dict[str, pd.DataFrame]
    subfleets: dict[str, SimInputSubfleet]
    chargers: dict[str, SimInputCharger]
    fleet_units: dict[str, FleetUnit]

    @classmethod
    def from_parameters(
        cls,
        settings: SimInputSettings,
        pwr_max_w: float,
        subfleets: dict[str, SimInputSubfleet],
        chargers: dict[str, SimInputCharger],
    ):
        log = {
            vehicle_type: _get_log_subfleet(vehicle_type=vehicle_type, settings=settings)
            for vehicle_type, subfleet in subfleets.items()
        }

        fleet_units = {
            f"{subfleet.name}_{i}": FleetUnit(
                name=f"{subfleet.name}_{i}",
                atbase=log[subfleet.name].loc[:, (f"{subfleet.name}{i}", "atbase")].astype(int).values,
                consumption_w=log[subfleet.name].loc[:, (f"{subfleet.name}{i}", "consumption")].values,
                battery_capacity_wh=subfleet.battery_capacity_wh,
                charger=subfleet.charger,
                pwr_max_w=min(subfleet.pwr_max_w, chargers[subfleet.charger].pwr_max_w),
                dti=settings.dti,
                freq_hours=settings.freq_hours,
            )
            for subfleet in subfleets.values()
            for i in range(subfleet.num_bev)
        }

        return cls(
            dti=settings.dti,
            freq_hours=settings.freq_hours,
            pwr_lim_w=pwr_max_w,
            log=log,
            subfleets=subfleets,
            chargers=chargers,
            fleet_units=fleet_units,
        )

    @property
    def demand_w(self) -> float:
        chargers = {chg_name: chg.num for chg_name, chg in self.chargers.items()}
        # get a list of all fleet units and their demand and sort that by priority level
        pwr_available_w = self.pwr_lim_w
        pwr_chg_fleet_w = 0.0
        for fleet_unit in sorted(self.fleet_units.values(), key=lambda x: x.time_flexibility):
            # check for available charger
            pwr_chg_site = fleet_unit.charge(pwr_available_w if chargers[fleet_unit.charger] >= 1 else 0)
            pwr_available_w -= pwr_chg_site
            pwr_chg_fleet_w += pwr_chg_site

            # allocate charger
            if pwr_chg_site > 0:
                chargers[fleet_unit.charger] -= 1
        return pwr_chg_fleet_w

    @property
    def energy_site_wh(self) -> float:
        # Return the total energy charged at the site in Wh over the simulation period.
        return sum(unit.energy_site_wh for unit in self.fleet_units.values())

    @property
    def energy_route_wh(self) -> float:
        # Return the total energy charged on the route in Wh over the simulation period.
        return sum(unit.energy_route_wh for unit in self.fleet_units.values())

    def get_distances(self, scale_factor) -> dict[str, dict[str, float]]:
        distances = {}
        for sf_name, sf in self.subfleets.items():
            log = self.log[sf_name]
            vehicles = log.columns.get_level_values(0).unique()
            splits = {"bev": vehicles[: sf.num_bev], "icev": vehicles[sf.num_bev : sf.num_total]}

            distances[sf_name] = {
                kind: log.loc[:, (cols, "dist")].sum().sum() * scale_factor if len(cols) else 0
                for kind, cols in splits.items()
            }
        return distances


@safe_cache_data
def get_soc_min(max_charge_rate, dsoc, atbase):
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
class FleetUnit(DemandBlock):
    name: str
    atbase: np.typing.NDArray[np.float64]
    consumption_w: np.typing.NDArray[np.float64]
    battery_capacity_wh: float
    charger: str
    pwr_max_w: float

    soc_track: np.typing.NDArray[np.float64] = field(init=False)

    soc: float = field(init=False, default=0.0)

    _pwr_chg_site_w: float = field(init=False, default=0.0)

    _e_chg_route_wh: float = field(init=False, default=0.0)

    def __post_init__(self):
        self.soc_track = np.zeros(len(self.dti) + 1, dtype=np.float64)
        self.soc_track[0] = self.soc

        self.soc_min = get_soc_min(
            max_charge_rate=self.pwr_max_w * self.freq_hours / self.battery_capacity_wh,
            dsoc=self.consumption_w * self.freq_hours / self.battery_capacity_wh,
            atbase=self.atbase,
        )

    def __repr__(self):
        return f"FleetUnit(name={self.name}, soc={self.soc}, charger={self.charger}, pwr_max_w={self.pwr_max_w})"

    @property
    def time_flexibility(self) -> float:
        if self.pwr_max_w == 0:  # division by 0 causes error
            return np.inf  # no charging possible, so lowest priority
        return float(self.soc - self.soc_min[self.idx]) * self.battery_capacity_wh / self.pwr_max_w

    @property
    def demand_w(self) -> float:
        return min(self.pwr_max_w, self.battery_capacity_wh * (1 - self.soc) / self.freq_hours)

    def charge(self, pwr_available_w: float):
        atbase = self.atbase[self.idx]
        # calculate the charging power based on available power and current SOC
        pwr_chg_site = max(
            min(
                self.demand_w * int(atbase),  # only charge if at base
                pwr_available_w,
            ),
            0,
        )
        self._pwr_chg_site_w += pwr_chg_site
        # update SOC based on charging power
        self.soc += (pwr_chg_site - self.consumption_w[self.idx]) * self.freq_hours / self.battery_capacity_wh
        if self.soc < (0 - EPS) and not atbase:
            e_chg_route_wh = abs(self.soc * self.battery_capacity_wh)  # energy needed to reach SOC=0
            self.soc = 0.0  # equivalent to self.soc += e_chg_route_wh / self.capacity_wh
            self._e_chg_route_wh += e_chg_route_wh
        if self.soc > (1 + EPS) or self.soc < (0 - EPS):
            raise SOCError(f"SOC {self.soc} out of bounds after charging {pwr_chg_site} W at {self.dti[self.idx]}.")
        self.soc_track[self.idx + 1] = self.soc
        # return the charging power applied to this unit
        return pwr_chg_site

    @property
    def energy_site_wh(self) -> float:
        # Return the total energy charged at the site in Wh over the simulation period.
        return self._pwr_chg_site_w * self.freq_hours

    @property
    def energy_route_wh(self) -> float:
        # Return the total energy charged on the route in Wh over the simulation period.
        return self._e_chg_route_wh
