from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from time import time

import demandlib
import numpy as np
import pandas as pd
import pvlib


import lift.data

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

    @property
    def generation_max_w(self) -> float:
        return self.log_spec[self.idx] * self.pwr_wp

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
                   self.capacity_wh * (1 - self.soc) / self.freq_hours)  # power limit due to current SOC

    @property
    def _pwr_max_dis_w(self) -> float:
        return min(self._pwr_max_crate_w,  # power limit due to c-rate
                   self.capacity_wh * self.soc / self.freq_hours)  # power limit due to current SOC

    @property
    def generation_max_w(self) -> float:
        return self._pwr_max_dis_w

    def satisfy_demand(self, demand_w: float) -> float:
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
    price_buy_eur_wh: float
    price_sell_eur_wh: float

    pwr_peak_w: float = field(init=False,
                              default=0.0)

    cost_eur: float = field(init=False,
                            default=0.0)

    revenue_eur: float = field(init=False,
                               default=0.0)

    @property
    def generation_max_w(self) -> float:
        # Return the maximum power that can be supplied by the grid connection.
        return self.pwr_max_w

    def satisfy_demand(self, demand_w: float):
        # Apply power to the grid connection, updating peak power and costs/revenue.
        if abs(demand_w) > (self.pwr_max_w + EPS):
            raise ValueError(f"Demand {demand_w} W exceeds maximum power {self.pwr_max_w} W at {self.dti[self.idx]}.")

        if demand_w > 0:
            self.pwr_peak_w = max(self.pwr_peak_w, demand_w)
            self.cost_eur += demand_w * self.freq_hours * self.price_buy_eur_wh
        else:
            self.revenue_eur += -demand_w * self.freq_hours * self.price_sell_eur_wh


@dataclass
class FixedDemand(DemandBlock):
    log: np.typing.NDArray[np.float64]

    @property
    def demand_w(self) -> float:
        # Return the demand in W for the current dt.
        return self.log[self.idx]


@dataclass
class Fleet(DemandBlock):
    fleet_units: dict[str, FleetUnit]
    pwr_lim_w: float
    log: pd.DataFrame

    def __post_init__(self):
        self.fleet_units = {f"unit_{i}": FleetUnit(name=f"unit_{i}",
                                                   dti=self.dti,
                                                   atbase=self.log.loc[:, (f'icev{i}', 'atbase')].values,
                                                   consumption_w=self.log.loc[:, (f'icev{i}', 'consumption')].values,
                                                   capacity_wh=100E3,
                                                   ) for i in range(5)}

    @property
    def demand_w(self) -> float:
        # get a list of all fleet units and their demand and sort that by priority level
        pwr_available_w = self.pwr_lim_w
        pwr_chg_fleet_w = 0.0
        for fleet_unit in sorted(self.fleet_units.values(), key=lambda x: x.priority_lvl):
            pwr_chg = fleet_unit.charge(pwr_available_w)
            pwr_available_w -= pwr_chg
            pwr_chg_fleet_w += pwr_chg
            if pwr_available_w <= 0:
                break
        return pwr_chg_fleet_w


@dataclass
class FleetUnit(DemandBlock):
    name: str
    atbase: np.typing.NDArray[np.float64]
    consumption_w: np.typing.NDArray[np.float64]
    capacity_wh: float

    soc_track: np.typing.NDArray[np.float64] = field(init=False)

    soc: float = field(init=False,
                       default=0.0)

    pwr_max_w: float = field(init=False,
                             default=11000.0)

    def __post_init__(self):
        self.soc_track = np.zeros(len(self.dti), dtype=np.float64)

    @property
    def priority_lvl(self) -> float:
        # ToDo: Implement logic to calculate priority based required energy for next trip and available charging power and time.
        return np.random.uniform(0, 1)

    @property
    def availability(self) -> float:
        return self.atbase[self.idx]

    @property
    def demand_w(self) -> float:
        return min(self.pwr_max_w, self.capacity_wh * (1 - self.soc) / self.freq_hours) * self.availability

    def charge(self, pwr_available_w: float):
        # calculate the charging power based on available power and current SOC
        pwr_chg = min(self.demand_w, pwr_available_w)
        # update SOC based on charging power
        self.soc += (pwr_chg - self.consumption_w[self.idx]) * self.freq_hours / self.capacity_wh
        self.soc_track[self.idx] = self.soc
        # return the charging power applied to this unit
        return pwr_chg


@dataclass
class Simulation:
    dti: pd.DatetimeIndex
    blocks_demand: dict[str, DemandBlock | Fleet | FixedDemand] = field(init=False)
    blocks_supply: dict[str, SupplyBlock] = field(init=False)

    log_fleet: pd.DataFrame = None
    log_pv: np.typing.NDArray[np.float64] = None
    log_demand: np.typing.NDArray[np.float64] = None

    def __post_init__(self):

        self.blocks_demand = {'dem': FixedDemand(dti=self.dti,
                                                 log=self.log_demand,
                                                 ),
                              'fleet': Fleet(dti=self.dti,
                                             fleet_units=None,
                                             log=self.log_fleet,
                                             pwr_lim_w=np.inf),
                              }

        self.blocks_supply = {'grid': GridConnection(dti=self.dti,
                                                     pwr_max_w=100000.0,
                                                     price_buy_eur_wh=20E-5,
                                                     price_sell_eur_wh=10E-5,
                                                     ),
                              'pv': PVSource(dti=self.dti,
                                             pwr_wp=10E3,
                                             log_spec=self.log_pv,
                                             ),
                              'ess': StationaryStorage(dti=self.dti,
                                                       capacity_wh=50E3,
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
            pwr_supply_max_w = sum(block.generation_max_w for block in blocks_supply)
            # get the total demand from the fixed demand block
            pwr_demand_w = dem.demand_w

            # define Fleet charging power limit for dynamic load management
            fleet.pwr_lim_w = pwr_supply_max_w - pwr_demand_w

            # add fleet demand to the total demand
            pwr_demand_w += fleet.demand_w

            # satisfy demand with supply blocks (order represents priority)
            for block in blocks_supply:
                pwr_demand_w = block.satisfy_demand(demand_w=pwr_demand_w)

            pass

        pass


def get_log_pv(index: pd.DatetimeIndex,
               latitude: float,
               longitude: float) -> np.typing.NDArray[np.float64]:

    data, *_ = pvlib.iotools.get_pvgis_hourly(
        latitude=latitude,
        longitude=11.5756,  # Example longitude
        start=2023,
        end=2023,
        raddatabase='PVGIS-SARAH3',
        outputformat='json',
        pvcalculation=True,
        peakpower=1,
        pvtechchoice='crystSi',
        mountingplace='free',
        loss=0,
        trackingtype=0,  # fixed mount
        optimalangles=True,
        url='https://re.jrc.ec.europa.eu/api/v5_3/',
        map_variables=True,
        timeout=30,  # default value
    )
    data = data['P']
    data.index = data.index.round('h')
    data = data.tz_convert('Europe/Berlin').reindex(index).ffill().bfill()
    return data.values


def get_log_demand(index: pd.DatetimeIndex,
                   slp: str,
                   consumption_yrl_wh: float) -> np.typing.NDArray[np.float64]:
    # Example demand data, replace with actual demand data retrieval logic
    e_slp = demandlib.bdew.ElecSlp(year=2023)
    return (e_slp.get_scaled_profiles({slp: consumption_yrl_wh})  # returns energies
            .resample('h').sum()  # sum() as df contains energies -> for hours energy is equal to power
            .iloc[:, 0].values)  # get first (and only) column as numpy array

def get_log_subfleet(index: pd.DatetimeIndex,
                     ) -> pd.DataFrame:
    return pd.read_csv(Path().cwd() / 'data' / 'log_subfleet.csv',
                       header=[0,1],
                       index_col=0,
                       parse_dates=True)


if __name__ == "__main__":
    # start time tracking
    dti = pd.date_range(start='2023-01-01', end='2024-01-01 00:00', freq='h', tz='Europe/Berlin')
    log_pv = get_log_pv(index=dti,
                        latitude=48.1372,  # Example latitude
                        longitude=11.5756,  # Example longitude
                        )
    log_demand = get_log_demand(index=dti,
                                slp='h0',
                                consumption_yrl_wh=50E6
                                )

    log_fleet = get_log_subfleet(index=dti)

    start_time = time()
    sim = Simulation(dti=dti,
                     log_fleet=log_fleet,
                     log_pv=log_pv,
                     log_demand=log_demand)
    sim.simulate()

    # stop time tracking
    print(f'Simulation completed in {time() - start_time:.2f} seconds.')
