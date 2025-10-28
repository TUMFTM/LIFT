from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Self, TYPE_CHECKING

import pandas as pd

from lift.utils import Coordinates

# ToDo: try to fix this issue without circular imports
if TYPE_CHECKING:
    pass


class BaseInput(ABC):
    @classmethod
    @abstractmethod
    def from_phase_input(cls, phase_input: "PhaseInputBase") -> Self: ...


@dataclass
class SimInputSettings(BaseInput):
    period_sim: pd.Timedelta
    start_sim: pd.Timestamp
    freq_sim: pd.Timedelta
    freq_hours: float
    # dti: pd.DatetimeIndex

    @classmethod
    def from_phase_input(cls, phase_input: "InputEconomics | PhaseInputEconomics") -> Self:
        return cls(
            period_sim=phase_input.period_sim,
            start_sim=phase_input.start_sim,
            freq_sim=phase_input.freq_sim,
            freq_hours=phase_input.freq_sim.total_seconds() / 3600.0,
            # dti=pd.date_range(
            #     start=phase_input.start_sim,
            #     end=phase_input.start_sim + phase_input.period_sim,
            #     freq=phase_input.freq_sim,
            #     tz="Europe/Berlin",
            #     inclusive="left",
            # )
        )

    def __post_init__(self):
        # Define dti here instead of passing it to the constructor: make class hashable for streamlit for caching
        self.dti = pd.date_range(
            start=self.start_sim,
            end=self.start_sim + self.period_sim,
            freq=self.freq_sim,
            tz="Europe/Berlin",
            inclusive="left",
        )


@dataclass
class SimInputLocation(BaseInput):
    coordinates: Coordinates = field(default_factory=Coordinates)
    slp: Literal["H0"] = "h0"  # ToDo: fix this
    consumption_yrl_wh: float = 3e6  # ToDo: fix this
    grid_w: float = 10e3
    pv_wp: float = 10e3
    ess_wh: float = 10e3

    @classmethod
    def from_phase_input(cls, phase_input: "PhaseInputLocation") -> Self:
        return cls(
            coordinates=phase_input.coordinates,
            slp=phase_input.slp,
            consumption_yrl_wh=phase_input.consumption_yrl_wh,
            grid_w=phase_input.grid.capacity,
            pv_wp=phase_input.pv.capacity,
            ess_wh=phase_input.ess.capacity,
        )


@dataclass
class SimInputSubfleet(BaseInput):
    name: str = "hlt"
    num_bev: int = 1
    num_total: int = 1
    battery_capacity_wh: float = 80e3
    pwr_max_w: float = 11e3
    charger: str = "ac"

    @classmethod
    def from_phase_input(cls, phase_input: "PhaseInputSubfleet") -> Self:
        return cls(
            name=phase_input.name,
            num_bev=phase_input.num_bev,
            num_total=phase_input.num_total,
            battery_capacity_wh=phase_input.battery_capacity_wh,
            pwr_max_w=phase_input.pwr_max_w,
            charger=phase_input.charger,
        )


@dataclass
class SimInputCharger(BaseInput):
    name: str = "ac"
    num: int = 0
    pwr_max_w: float = 11e3

    @classmethod
    def from_phase_input(cls, phase_input: "PhaseInputCharger") -> Self:
        return cls(
            name=phase_input.name,
            num=phase_input.num,
            pwr_max_w=phase_input.pwr_max_w,
        )


@dataclass
class SimResults:
    energy_pv_pot_wh: float
    energy_pv_curt_wh: float
    energy_grid_buy_wh: float
    energy_grid_sell_wh: float
    pwr_grid_peak_w: float
    energy_dem_site_wh: float
    energy_fleet_site_wh: float
    energy_fleet_route_wh: float
    dist_km: dict[str, dict[str, float]]  # e.g. {"subfleet": {"bev": 0.0, "icev": 0.0}}


class GridPowerExceededError(Exception):
    pass


class SOCError(Exception):
    pass
