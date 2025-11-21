from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from lift.backend.simulation import SimResults


@dataclass
class BaseBlock(ABC):
    period_eco: int

    _sim_result: SimResults = field(
        init=False,
        default=None,
    )

    def __getattr__(self, name):
        # Make sure that _sim_result is set before accessing it
        if name == "_sim_result":
            if self._sim_result is None:
                raise ValueError(
                    "Attribute _sim_result is not set. Use the method set_sim_result() to set the attribute."
                )
            return self._sim_result

        return super().__getattribute__(name)

    def __setattr__(self, name, value):
        if name == "sim_result":
            self._sim_result = value
        super().__setattr__(name, value)

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
    capex_spec: float = field(default=0.0)
    capem_spec: float = field(default=0.0)
    opex_spec: float = field(default=0.0)
    opem_spec: float = field(default=0.0)

    @property
    def _capex_single(self) -> float:
        return self.capacity * self.capex_spec

    @property
    def _capem_single(self) -> float:
        return self.capacity * self.capem_spec


@dataclass
class Grid(ContinuousInvestBlock):
    opex_spec_buy: float = field(default=0.0)
    opex_spec_sell: float = field(default=0.0)
    opex_spec_peak: float = field(default=0.0)

    @property
    def _opex_yrl(self) -> float:
        return (
            self._sim_result.energy_grid_buy_wh * self.opex_spec_buy
            + self._sim_result.energy_grid_sell_wh * self.opex_spec_sell
            + self._sim_result.pwr_grid_peak_w * self.opex_spec_peak
        )

    @property
    def _opem_yrl(self) -> float:
        return self._sim_result.energy_grid_buy_wh * self.opem_spec


@dataclass
class PV(ContinuousInvestBlock):
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
class Chargers(InvestBlock):
    name: str = field(default="DefaultCharger")
    num: int = field(default=0)
    capex_per_unit: float = field(default=0.0)
    capem_per_unit: float = field(default=0.0)

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
    name: str = field(default="DefaultCharger")
    num_bev: int = 0
    num_icev: int = 0
    capex_bev: float = 0.0
    capex_icev: float = 0.0
    capem_bev: float = 0.0
    capem_icev: float = 0.0
    mntex_eur_km_bev: float = 0.0
    mntex_eur_km_icev: float = 0.0
    toll_frac: float = 0.0
    toll_eur_per_km_bev: float = 0.0
    toll_eur_per_km_icev: float = 0.0
    consumption_icev: float = 27.0
    opex_spec_fuel: float = 0.0
    opem_spec_fuel: float = 0.0
    opex_spec_onroute_charging: float = 0.0
    opem_spec_onroute_charging: float = 0.0

    @property
    def _capex_single(self) -> float:
        return self.num_bev * self.capex_bev + self.num_icev * self.capex_icev

    @property
    def _capem_single(self) -> float:
        return self.num_bev * self.capem_bev + self.num_icev * self.capem_icev

    @property
    def _opex_yrl(self) -> float:
        opex_bev = (
            self._sim_result.dist_km[self.name]["bev"]
            * (self.mntex_eur_km_bev + self.toll_eur_per_km_bev * self.toll_frac)
            + self._sim_result.energy_fleet_route_wh * self.opex_spec_onroute_charging
        )  # ToDo: get energy per subfleet

        opex_icev = self._sim_result.dist_km[self.name]["icev"] * (
            self.mntex_eur_km_icev
            + self.toll_eur_per_km_icev * self.toll_frac
            + self.opex_spec_fuel * self.consumption_icev / 100
        )

        return opex_bev + opex_icev

    @property
    def _opem_yrl(self) -> float:
        opem_icev = self._sim_result.dist_km[self.name]["icev"] * self.consumption_icev / 100 * self.opem_spec_fuel

        opem_bev = (
            self._sim_result.energy_fleet_route_wh * self.opem_spec_onroute_charging
        )  # ToDo: get energy per subfleet

        return opem_bev + opem_icev


@dataclass
class Aggregator:
    subblocks: dict = field(default_factory=dict)

    @property
    def capex(self) -> np.typing.NDArray:
        return sum([block.capex for block in self.subblocks.values()])

    @property
    def capem(self) -> np.typing.NDArray:
        return sum([block.capem for block in self.subblocks.values()])

    @property
    def opex(self) -> np.typing.NDArray:
        return sum([block.opex for block in self.subblocks.values()])

    @property
    def opem(self) -> np.typing.NDArray:
        return sum([block.opem for block in self.subblocks.values()])


if __name__ == "__main__":
    # simple test
    from lift.backend.simulation.interfaces import SimResults

    sim_res = SimResults(
        energy_pv_pot_wh=1000,
        energy_pv_curt_wh=100,
        energy_grid_buy_wh=5000,
        energy_grid_sell_wh=200,
        pwr_grid_peak_w=3000,
        energy_fleet_site_wh=4000,
        energy_fleet_route_wh=1000,
        energy_dem_site_wh=1000,
        dist_km={"subfleet": {"hlt": 10000.0, "hst": 234.0}},
    )

    blocks = {
        "grid": Grid(
            period_eco=10,
            ls=5,
            capacity=10000,
            capex_spec=200,
            capem_spec=10,
            opex_spec_buy=0.2,
            opex_spec_sell=0.1,
            opex_spec_peak=5,
            opem_spec=1.0,
        ),
        "pv": PV(
            period_eco=10,
            ls=20,
            capacity=5000,
            capex_spec=150,
            capem_spec=5,
        ),
        "ess": ESS(
            period_eco=10,
            ls=15,
            capacity=2000,
            capex_spec=300,
            capem_spec=15,
        ),
        "chargers": Aggregator(
            subblocks={
                "ac": Chargers(
                    period_eco=10,
                    ls=10,
                    name="ac",
                    num=10,
                    capex_per_unit=3000,
                    capem_per_unit=100,
                ),
                "dc": Chargers(
                    period_eco=10,
                    ls=10,
                    name="dc",
                    num=5,
                    capex_per_unit=10000,
                    capem_per_unit=300,
                ),
            }
        ),
        "fleets": Aggregator(
            subblocks={
                "hlt": SubFleet(
                    period_eco=10,
                    ls=8,
                    name="hlt",
                    num_bev=3,
                    num_icev=2,
                    capex_bev=40000,
                    capex_icev=30000,
                    capem_bev=2000,
                    capem_icev=1500,
                    mntex_eur_km_bev=0.05,
                    mntex_eur_km_icev=0.07,
                    toll_frac=0.5,
                    toll_eur_per_km_bev=0.02,
                    toll_eur_per_km_icev=0.03,
                    consumption_icev=25.0,
                    opex_spec_fuel=1.5,
                    opem_spec_fuel=0.5,
                    opex_spec_onroute_charging=0.1,
                    opem_spec_onroute_charging=0.05,
                ),
            }
        ),
    }
    for block in blocks.values():
        block.sim_result = sim_res

    pass
