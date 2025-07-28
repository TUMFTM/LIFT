from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from time import time

import numpy as np
import pandas as pd


@dataclass
class Block(ABC):
    dti: pd.DatetimeIndex

    _dt: pd.Timedelta = field(init=False)

    @property
    def freq_hours(self) -> float:
        return pd.Timedelta(self.dti.freq).total_seconds()/3600

    @property
    def dt(self) -> pd.Timedelta:
        return self._dt

    @dt.setter
    def dt(self, value):
        if not value in self.dti:
            raise ValueError("dt must be a valid timedelta within the dti range")
        self._dt = value


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
    def satisfy_demand(self, demand_kw: float) -> float:
        """
        Apply power to the supply block and return remaining demand.
        """
        ...


@dataclass
class PVSource(SupplyBlock):
    pwr_kwp: float
    log_spec: pd.Series = None

    def __post_init__(self):
        if self.log_spec is None:
            self.log_spec = pd.Series(index=self.dti, data=np.random.uniform(low=0, high=1, size=len(self.dti)))

    def satisfy_demand(self, demand_kw: float) -> float:
        # Return remaining power demand after PV generation (negative if excess generation)
        return demand_kw - self.log_spec.at[self.dt] * self.pwr_kwp


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
                   self.capacity_kwh * (1 - 1E-8 - self.soc) / self.freq_hours)  # power limit due to current SOC

    @property
    def _pwr_max_dis_kw(self) -> float:
        return min(self._pwr_max_crate_kw,  # power limit due to c-rate
                   self.capacity_kwh * (self.soc - 1E-8) / self.freq_hours)  # power limit due to current SOC

    def satisfy_demand(self, demand_kw: float) -> float:
        # Apply power to the stationary storage, charging or discharging as needed.
        if demand_kw > 0:  # discharging
            pwr_ess = min(self._pwr_max_dis_kw, demand_kw)
        else:  # charging
            pwr_ess = max(-1 * self._pwr_max_chg_kw, demand_kw)

        dsoc = -1 * pwr_ess * self.freq_hours / self.capacity_kwh  # Change in SOC based on power applied
        self.soc += dsoc
        if self.soc < 0 or self.soc > 1:
            raise ValueError(f"SOC {self.soc} out of bounds after applying power {demand_kw} kW at {self.dt}.")

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

    def satisfy_demand(self, demand_kw: float):
        # Apply power to the grid connection, updating peak power and costs/revenue.
        if abs(demand_kw) > self.pwr_max_kw:
            raise ValueError(f"Demand {demand_kw} kW exceeds maximum power {self.pwr_max_kw} kW at {self.dt}.")

        if demand_kw > 0:
            self.pwr_peak_kw = max(self.pwr_peak_kw, demand_kw)
            self.cost_eur += demand_kw * self.freq_hours * self.price_buy_eur_kwh
        else:
            self.revenue_eur += -demand_kw * self.freq_hours * self.price_sell_eur_kwh


@dataclass
class FixedDemand(DemandBlock):
    log: pd.Series() = None

    def __post_init__(self):
        if self.log is None:
            self.log = pd.Series(index=self.dti, data=np.random.uniform(low=0, high=10, size=len(self.dti)))

    @property
    def demand_kw(self) -> float:
        # Return the demand in kW for the current dt.
        return self.log.at[self.dt]


@dataclass
class Fleet(DemandBlock):
    fleet_units: dict[str, 'FleetUnit'] = None

    def __post_init__(self):
        if self.fleet_units is None:
            self.fleet_units = {f"unit_{i}": FleetUnit(name=f"unit_{i}", dti=self.dti) for i in range(5)}

    @property
    def demand_kw(self) -> float:
        return sum([fleet_unit.demand_kw for fleet_unit in self.fleet_units.values()])


@dataclass
class FleetUnit(DemandBlock):
    name: str
    log: pd.DataFrame = field(init=True,
                              default_factory=pd.DataFrame)
    soc: float = field(init=False,
                       default=1.0)

    pwr_max_kw: float = field(init=False,
                              default=11.0)

    @property
    def demand_kw(self) -> float:
        # ToDo: Replace random value by logic to calculate demand based on subfleet and log.
        return np.random.uniform(low=0, high=self.pwr_max_kw)


@dataclass
class Simulation:
    dti: pd.DatetimeIndex
    blocks_demand: dict[str, DemandBlock] = field(init=False)
    blocks_supply: dict[str, SupplyBlock] = field(init=False)

    def __post_init__(self):
        self.blocks_demand = {'dem': FixedDemand(dti=self.dti,
                                                 ),
                              'fleet': Fleet(dti=self.dti),
                              }
        self.blocks_supply = {'grid': GridConnection(dti=self.dti,
                                                     pwr_max_kw=1000.0,
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

        self.blocks = {**self.blocks_demand, **self.blocks_supply}

    def simulate(self):
        # Simulate the vehicle fleet over the given datetime index.
        for dt in self.dti:
            for block in self.blocks.values():
                block.dt = dt

            pwr_demand_kw = 0
            for block in self.blocks_demand.values():
                pwr_demand_kw += block.demand_kw

            for name in ['pv', 'ess', 'grid']:  # Order represents priority
                pwr_demand_kw = self.blocks_supply[name].satisfy_demand(demand_kw=pwr_demand_kw)


        pass



if __name__ == "__main__":
    # start time tracking
    start_time = time()

    sim = Simulation(dti=pd.date_range(start='2023-01-01', end='2024-01-01 00:00', freq='h', tz='Europe/Berlin'))
    sim.simulate()

    # stop time tracking
    print(f"Simulation completed in {time() - start_time:.2f} seconds.")
