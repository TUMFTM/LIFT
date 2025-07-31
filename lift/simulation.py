from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from time import time

import numpy as np
import pandas as pd


EPS = 1E-8  # Small epsilon value for numerical stability in calculations


@dataclass
class Block(ABC):
    dti: pd.DatetimeIndex

    _idx: int = field(init=False)

    @property
    def freq_hours(self) -> float:
        return pd.Timedelta(self.dti.freq).total_seconds()/3600

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
    def demand_kw(self) -> float:
        # Return the demand in kW for the current dt.
        ...


@dataclass
class SupplyBlock(Block):

    @abstractmethod
    def generation_max_kw(self) -> float:
        """
        Return the maximum generation capacity in kW for the current dt.
        """
        ...

    @abstractmethod
    def satisfy_demand(self, demand_kw: float) -> float:
        """
        Apply power to the supply block and return remaining demand.
        """
        ...


@dataclass
class PVSource(SupplyBlock):
    pwr_kwp: float
    log_spec: np.ndarray[float] = None

    def __post_init__(self):
        if self.log_spec is None:
            self.log_spec = pd.Series(index=self.dti,
                                      data=np.random.uniform(low=0, high=1, size=len(self.dti))).values

    @property
    def generation_max_kw(self) -> float:
        return self.log_spec[self.idx] * self.pwr_kwp

    def satisfy_demand(self, demand_kw: float) -> float:
        # Return remaining power demand after PV generation (negative if excess generation)
        return demand_kw - self.generation_max_kw


@dataclass
class StationaryStorage(SupplyBlock):
    capacity_kwh: float
    soc: float = field(init=False, default=1.0)
    c_rate_max: float = field(init=False, default=0.5)

    @property
    def _pwr_max_crate_kw(self) -> float:
        return self.capacity_kwh * self.c_rate_max

    @property
    def _pwr_max_chg_kw(self) -> float:
        return min(self._pwr_max_crate_kw,  # power limit due to c-rate
                   self.capacity_kwh * (1 - self.soc) / self.freq_hours)  # power limit due to current SOC

    @property
    def _pwr_max_dis_kw(self) -> float:
        return min(self._pwr_max_crate_kw,  # power limit due to c-rate
                   self.capacity_kwh * self.soc / self.freq_hours)  # power limit due to current SOC

    @property
    def generation_max_kw(self) -> float:
        return self._pwr_max_dis_kw

    def satisfy_demand(self, demand_kw: float) -> float:
        # Apply power to the stationary storage, charging or discharging as needed.
        if demand_kw > 0:  # discharging
            pwr_ess = min(self._pwr_max_dis_kw, demand_kw)
        else:  # charging
            pwr_ess = max(-1 * self._pwr_max_chg_kw, demand_kw)

        dsoc = -1 * pwr_ess * self.freq_hours / self.capacity_kwh  # Change in SOC based on power applied
        self.soc += dsoc
        if self.soc < (0 - EPS) or self.soc > (1 + EPS):
            raise ValueError(f"SOC {self.soc} out of bounds after applying power {demand_kw} kW at {self.dti[self.idx]}.")

        # Return remaining power demand after storage (negative, if PV excess generation cannot be charged into storage)
        return demand_kw - pwr_ess


@dataclass
class GridConnection(SupplyBlock):
    pwr_max_kw: float
    price_buy_eur_kwh: float
    price_sell_eur_kwh: float

    pwr_peak_kw: float = field(init=False,
                               default=0.0)

    cost_eur: float = field(init=False,
                            default=0.0)

    revenue_eur: float = field(init=False,
                               default=0.0)

    @property
    def generation_max_kw(self) -> float:
        # Return the maximum power that can be supplied by the grid connection.
        return self.pwr_max_kw

    def satisfy_demand(self, demand_kw: float):
        # Apply power to the grid connection, updating peak power and costs/revenue.
        if abs(demand_kw) > (self.pwr_max_kw + EPS):
            raise ValueError(f"Demand {demand_kw} kW exceeds maximum power {self.pwr_max_kw} kW at {self.dti[self.idx]}.")

        if demand_kw > 0:
            self.pwr_peak_kw = max(self.pwr_peak_kw, demand_kw)
            self.cost_eur += demand_kw * self.freq_hours * self.price_buy_eur_kwh
        else:
            self.revenue_eur += -demand_kw * self.freq_hours * self.price_sell_eur_kwh


@dataclass
class FixedDemand(DemandBlock):
    log: np.ndarray[float] = None

    def __post_init__(self):
        if self.log is None:
            self.log = pd.Series(index=self.dti, data=np.random.uniform(low=0, high=10, size=len(self.dti))).values

    @property
    def demand_kw(self) -> float:
        # Return the demand in kW for the current dt.
        return self.log[self.idx]


@dataclass
class Fleet(DemandBlock):
    fleet_units: dict[str, FleetUnit] = None

    pwr_lim_kw: float = 22.0

    def __post_init__(self):
        self.fleet_units = {f"unit_{i}": FleetUnit(name=f"unit_{i}",
                                                   dti=self.dti) for i in range(5)}

    @property
    def demand_kw(self) -> float:
        # get a list of all fleet units and their demand and sort that by priority level
        pwr_available_kw = self.pwr_lim_kw
        pwr_chg_fleet_kw = 0.0
        for fleet_unit in sorted(self.fleet_units.values(), key=lambda x: x.priority_lvl):
            pwr_chg = fleet_unit.charge(pwr_available_kw)
            pwr_available_kw -= pwr_chg
            pwr_chg_fleet_kw += pwr_chg
            if pwr_available_kw <= 0:
                break
        return pwr_chg_fleet_kw


@dataclass
class FleetUnit(DemandBlock):
    name: str
    log: pd.DataFrame = field(init=True,
                              default_factory=pd.DataFrame)

    capacity_kwh: float = 100

    soc_track: np.typing.NDArray[np.float64] = field(init=False)

    soc: float = field(init=False,
                       default=0.0)

    pwr_max_kw: float = field(init=False,
                              default=11.0)

    def __post_init__(self):
        self.soc_track = np.zeros(len(self.dti), dtype=np.float64)

    @property
    def priority_lvl(self) -> float:
        # ToDo: Implement logic to calculate priority based required energy for next trip and available charging power and time.
        return np.random.uniform(0, 1)

    @property
    def demand_kw(self) -> float:
        return min(self.pwr_max_kw, self.capacity_kwh * (1 - self.soc) / self.freq_hours)

    def charge(self, pwr_available_kw: float):
        self.soc = np.random.uniform(0, 1)
        # calculate the charging power based on available power and current SOC
        pwr_chg = min(self.demand_kw, pwr_available_kw)
        # update SOC based on charging power
        self.soc += pwr_chg * self.freq_hours / self.capacity_kwh
        self.soc_track[self.idx] = self.soc
        # return the charging power applied to this unit
        return pwr_chg


@dataclass
class Simulation:
    dti: pd.DatetimeIndex
    blocks_demand: dict[str, DemandBlock | Fleet | FixedDemand] = field(init=False)
    blocks_supply: dict[str, SupplyBlock] = field(init=False)

    def __post_init__(self):

        self.blocks_demand = {'dem': FixedDemand(dti=self.dti,
                                                 ),
                              'fleet': Fleet(dti=self.dti),
                              }

        self.blocks_supply = {'grid': GridConnection(dti=self.dti,
                                                     pwr_max_kw=100.0,
                                                     price_buy_eur_kwh=0.2,
                                                     price_sell_eur_kwh=0.1,
                                                     ),
                              'pv': PVSource(dti=self.dti,
                                             pwr_kwp=10.0,
                                             ),
                              'ess': StationaryStorage(dti=self.dti,
                                                       capacity_kwh=50.0,
                                                       ),
                              }

        self.blocks = {**self.blocks_demand, **self.blocks_supply, **self.blocks_demand['fleet'].fleet_units}

    def simulate(self):

        # Improve speed by using the following shortcuts to avoid repeated lookups
        blocks = self.blocks.values()
        blocks_supply = tuple(self.blocks_supply[k] for k in ('pv', 'ess', 'grid'))
        blocks_demand = self.blocks_demand
        fleet = blocks_demand['fleet']
        dem = blocks_demand['dem']

        # Simulate the vehicle fleet over the given datetime index.
        for idx in range(len(self.dti)):
            # pass time of current timestep to all blocks
            for block in blocks:
                block.idx = idx

            # calculate maximum power supply
            pwr_supply_max_kw = sum(block.generation_max_kw for block in blocks_supply)
            # get the total demand from the fixed demand block
            pwr_demand_kw = dem.demand_kw

            # define Fleet charging power limit for dynamic load management
            fleet.pwr_lim_kw = pwr_supply_max_kw - pwr_demand_kw

            # add fleet demand to the total demand
            pwr_demand_kw += fleet.demand_kw

            # satisfy demand with supply blocks (order represents priority)
            for block in blocks_supply:
                pwr_demand_kw = block.satisfy_demand(demand_kw=pwr_demand_kw)

        pass



if __name__ == "__main__":
    # start time tracking
    start_time = time()

    sim = Simulation(dti=pd.date_range(start='2023-01-01', end='2024-01-01 00:00', freq='h', tz='Europe/Berlin'))
    sim.simulate()

    # stop time tracking
    print(f'Simulation completed in {time() - start_time:.2f} seconds.')
