from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import importlib.resources as resources
from typing import Self

import demandlib
import numpy as np
import pandas as pd
import pvlib

from .interfaces import SimInputSettings, SimInputSubfleet, SimInputCharger, GridPowerExceededError, SOCError

from lift.utils import safe_cache_data, Coordinates

EPS = 1e-8  # Small epsilon value for numerical stability in calculations


@safe_cache_data
def _get_log_pv(
    coordinates: Coordinates,
    settings: SimInputSettings,
) -> np.typing.NDArray[np.float64]:
    # ToDo: add default value if internet connection is not available -> debug purposes only
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
    data = data["P"]
    data.index = data.index.round("h")
    data = data.tz_convert("Europe/Berlin").reindex(dti).ffill().bfill()
    return data.values / 1000


@safe_cache_data
def _get_log_dem(slp: str, consumption_yrl_wh: float, settings: SimInputSettings) -> np.typing.NDArray[np.float64]:
    if slp not in ["h0", "h0_dyn", "g0", "g1", "g2", "g3", "g4", "g5", "g6", "g7", "l0", "l1", "l2"]:
        raise ValueError(f'Specified SLP "{slp}" is not valid. SLP has to be defined as lower case.')

    return (
        pd.concat(
            [
                demandlib.bdew.ElecSlp(year=year)
                .get_scaled_profiles({slp: consumption_yrl_wh})  # returns energies
                .resample(settings.freq_sim)
                .sum()  # sum() as df contains energies -> for hours energy is equal to power
                .iloc[:, 0]
                for year in settings.dti.year.unique()
            ]
        ).values
        / settings.freq_hours
    )  # get first (and only) column as numpy array and convert from energy to power


@safe_cache_data
def _get_log_subfleet(vehicle_type: str, settings: SimInputSettings) -> pd.DataFrame:
    with resources.files("lift.data.mobility").joinpath(f"log_{vehicle_type}.csv").open("r") as logfile:
        df = pd.read_csv(logfile, header=[0, 1])
        df = (
            df.set_index(pd.to_datetime(df.iloc[:, 0], utc=True))
            .drop(df.columns[0], axis=1)
            .tz_convert("Europe/Berlin")
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
    # ToDo: fix Literal
    def from_parameters(cls, settings: SimInputSettings, consumption_yrl_wh: float, slp: str) -> Self:
        log = _get_log_dem(slp=slp, consumption_yrl_wh=consumption_yrl_wh, settings=settings)
        return cls(dti=settings.dti, freq_hours=settings.freq_hours, log=log)

    @property
    def demand_w(self) -> float:
        # Return the demand in W for the current dt.
        return float(self.log[self.idx])

    @property
    def energy_wh(self) -> float:
        # Return the total energy demand in Wh over the phase_simulation period.
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
        pwr_lim_w: float,
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
                atbase=log[subfleet.name].loc[:, (f"{subfleet.name}{i}", "atbase")].values,
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
            pwr_lim_w=pwr_lim_w,
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

            if pwr_available_w <= 0:
                break
            if sum(chargers.values()) <= 0:
                break
        return pwr_chg_fleet_w

    @property
    def energy_site_wh(self) -> float:
        # Return the total energy charged at the site in Wh over the phase_simulation period.
        return sum(unit.energy_site_wh for unit in self.fleet_units.values())

    @property
    def energy_route_wh(self) -> float:
        # Return the total energy charged on the route in Wh over the phase_simulation period.
        return sum(unit.energy_route_wh for unit in self.fleet_units.values())

    @property
    def distances(self) -> dict[str, dict[str, float]]:
        distances = {}
        for sf_name, sf in self.subfleets.items():
            log = self.log[sf_name]
            vehicles = log.columns.get_level_values(0).unique()
            splits = {"bev": vehicles[: sf.num_bev], "icev": vehicles[sf.num_bev : sf.num_total]}

            distances[sf_name] = {
                kind: log.loc[:, (cols, "dist")].sum().sum() if len(cols) else 0 for kind, cols in splits.items()
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
        self.soc_track = np.zeros(len(self.dti), dtype=np.float64)

        self.soc_min = get_soc_min(
            max_charge_rate=self.pwr_max_w * self.freq_hours / self.battery_capacity_wh,
            dsoc=self.consumption_w * self.freq_hours / self.battery_capacity_wh,
            atbase=self.atbase,
        )

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
        self.soc_track[self.idx] = self.soc
        # return the charging power applied to this unit
        return pwr_chg_site

    @property
    def energy_site_wh(self) -> float:
        # Return the total energy charged at the site in Wh over the phase_simulation period.
        return self._pwr_chg_site_w * self.freq_hours

    @property
    def energy_route_wh(self) -> float:
        # Return the total energy charged on the route in Wh over the phase_simulation period.
        return self._e_chg_route_wh
