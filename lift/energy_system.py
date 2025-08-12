from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import streamlit as st


from definitions import DTI, FREQ_HOURS
from interfaces import GridPowerExceededError, SOCError, SubfleetSimSettings

EPS = 1E-8  # Small epsilon value for numerical stability in calculations


@dataclass
class Block(ABC):
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

    def __post_init__(self):
        self.log = self.log_spec * self.pwr_wp

    @property
    def generation_max_w(self) -> float:
        return float(self.log[self.idx])

    @property
    def energy_pot_wh(self) -> float:
        return sum(self.log_spec * self.pwr_wp * FREQ_HOURS)

    def satisfy_demand(self, demand_w: float) -> float:
        # Return remaining power demand after PV generation (negative if excess generation)
        return demand_w - self.generation_max_w


@dataclass
class StationaryStorage(SupplyBlock):
    capacity_wh: float
    soc: float = field(init=False, default=1.0)
    c_rate_max: float = field(init=False, default=0.5)

    @property
    def _pwr_max_crate_w(self) -> float:
        return self.capacity_wh * self.c_rate_max

    @property
    def _pwr_max_chg_w(self) -> float:
        return min(self._pwr_max_crate_w,  # power limit due to c-rate
                   self.capacity_wh * (1 - self.soc) / FREQ_HOURS)  # power limit due to current SOC

    @property
    def _pwr_max_dis_w(self) -> float:
        return min(self._pwr_max_crate_w,  # power limit due to c-rate
                   self.capacity_wh * self.soc / FREQ_HOURS)  # power limit due to current SOC

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

        dsoc = -1 * pwr_ess * FREQ_HOURS / self.capacity_wh # Change in SOC based on power applied
        self.soc += dsoc
        if self.soc < (0 - EPS) or self.soc > (1 + EPS):
            raise ValueError(f"SOC {self.soc} out of bounds after applying power {demand_w} W at {DTI[self.idx]}.")

        # Return remaining power demand after storage (negative, if PV excess generation cannot be charged into storage)
        return demand_w - pwr_ess


@dataclass
class GridConnection(SupplyBlock):
    pwr_max_w: float

    pwr_peak_w: float = field(init=False,
                              default=0.0)

    _pwr_buy_w: float = field(init=False,
                              default=0.0)

    _pwr_sell_w: float = field(init=False,
                               default=0.0)

    _pwr_curt_w: float = field(init=False,
                               default=0.0)

    @property
    def generation_max_w(self) -> float:
        # Return the maximum power that can be supplied by the grid connection.
        return self.pwr_max_w

    def satisfy_demand(self, demand_w: float):
        # Apply power to the grid connection, updating peak power and costs/revenue.
        if demand_w > self.pwr_max_w + EPS:
            raise GridPowerExceededError(f"Demand {demand_w} W exceeds maximum power {self.pwr_max_w} W at {DTI[self.idx]}.")

        if demand_w > 0:
            self.pwr_peak_w = max(self.pwr_peak_w, demand_w)
            self._pwr_buy_w += demand_w
        else:
            self._pwr_sell_w += min(-demand_w, self.pwr_max_w)
            self._pwr_curt_w += max(-demand_w - self.pwr_max_w, 0)

    @property
    def energy_buy_wh(self) -> float:
        return self._pwr_buy_w * FREQ_HOURS

    @property
    def energy_sell_wh(self) -> float:
        return self._pwr_sell_w * FREQ_HOURS

    @property
    def energy_curt_wh(self) -> float:
        return self._pwr_curt_w * FREQ_HOURS


@dataclass
class FixedDemand(DemandBlock):
    log: np.typing.NDArray[np.float64]

    @property
    def demand_w(self) -> float:
        # Return the demand in W for the current dt.
        return float(self.log[self.idx])

    @property
    def energy_wh(self) -> float:
        # Return the total energy demand in Wh over the simulation period.
        return sum(self.log) * FREQ_HOURS


@dataclass
class Fleet(DemandBlock):
    pwr_lim_w: float
    log: pd.DataFrame
    subfleets: dict[str, SubfleetSimSettings]
    chargers: dict[str, int]

    def __post_init__(self):
        self.fleet_units = {f"{subfleet.vehicle_type}_{i}": FleetUnit(
            name=f"{subfleet.vehicle_type}_{i}",
            atbase=self.log[subfleet.vehicle_type].loc[:, (f'{subfleet.vehicle_type}{i}', 'atbase')].values,
            consumption_w=self.log[subfleet.vehicle_type].loc[:, (f'{subfleet.vehicle_type}{i}', 'consumption')].values,
            capacity_wh=subfleet.capacity_wh,
            charger=subfleet.charger,
            pwr_max_w=subfleet.pwr_chg_max_w,
        ) for subfleet in self.subfleets.values()
            for i in range(subfleet.num)}

    @property
    def demand_w(self) -> float:
        chargers = self.chargers.copy()
        # get a list of all fleet units and their demand and sort that by priority level
        pwr_available_w = self.pwr_lim_w
        pwr_chg_fleet_w = 0.0
        for fleet_unit in sorted(self.fleet_units.values(), key=lambda x: x.time_flexibility):
            # check for available charger
            pwr_chg = fleet_unit.charge(pwr_available_w * min(chargers[fleet_unit.charger], 1))
            pwr_available_w -= pwr_chg
            pwr_chg_fleet_w += pwr_chg

            # allocate charger
            if pwr_chg > 0:
                chargers[fleet_unit.charger] -= 1

            if pwr_available_w <= 0:
                break
            if sum(chargers.values()) <= 0:
                break
        return pwr_chg_fleet_w

    @property
    def energy_wh(self) -> float:
        # Return the total energy demand in Wh over the simulation period.
        return sum(unit.energy_wh for unit in self.fleet_units.values())


@st.cache_data
def get_soc_min(max_charge_rate,
                dsoc,
                atbase):
    # cumulative sums of consumption and possible charging
    cum_dsoc = np.concatenate(([0.0], np.cumsum(dsoc)))
    cum_charge = np.concatenate(([0.0], np.cumsum(max_charge_rate * atbase)))

    # transform space: subtract available charging from required SOC
    T = cum_dsoc - cum_charge

    # reverse max accumulate and reverse back
    M = np.maximum.accumulate(T[::-1])[::-1]

    # translate back to SOC requirement at each timestep
    soc_min = M[1:] - T[:-1]

    # must be at least trip consumption
    soc_min = np.maximum(soc_min, dsoc)

    # no negative SOC
    return np.clip(soc_min, 0.0, None)


@dataclass
class FleetUnit(DemandBlock):
    name: str
    atbase: np.typing.NDArray[np.float64]
    consumption_w: np.typing.NDArray[np.float64]
    capacity_wh: float
    charger: str
    pwr_max_w: float

    soc_track: np.typing.NDArray[np.float64] = field(init=False)

    soc: float = field(init=False,
                       default=0.0)

    _pwr_chg_w: float = field(init=False,
                              default=0.0)

    def __post_init__(self):
        self.soc_track = np.zeros(len(DTI), dtype=np.float64)

        self.soc_min = get_soc_min(max_charge_rate=self.pwr_max_w * FREQ_HOURS / self.capacity_wh,
                                   dsoc=self.consumption_w * FREQ_HOURS / self.capacity_wh,
                                   atbase=self.atbase)

    @property
    def time_flexibility(self) -> float:
        return float(self.soc - self.soc_min[self.idx]) * self.capacity_wh / self.pwr_max_w

    @property
    def availability(self) -> int:
        return int(self.atbase[self.idx])

    @property
    def demand_w(self) -> float:
        return min(self.pwr_max_w, self.capacity_wh * (1 - self.soc) / FREQ_HOURS) * self.availability

    def charge(self, pwr_available_w: float):
        # calculate the charging power based on available power and current SOC
        pwr_chg = max(min(self.demand_w, pwr_available_w), 0)
        self._pwr_chg_w += pwr_chg
        # update SOC based on charging power
        self.soc += (pwr_chg - self.consumption_w[self.idx]) * FREQ_HOURS / self.capacity_wh
        if self.soc < (0 - EPS) or self.soc > (1 + EPS):
            raise SOCError(f"SOC {self.soc} out of bounds after charging {pwr_chg} W at {DTI[self.idx]}.")
        self.soc_track[self.idx] = self.soc
        # return the charging power applied to this unit
        return pwr_chg

    @property
    def energy_wh(self) -> float:
        # Return the total energy demand in Wh over the simulation period.
        return self._pwr_chg_w * FREQ_HOURS
