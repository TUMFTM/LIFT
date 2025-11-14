from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd


import lift.backend.evaluation as eval
import lift.backend.simulation as sim


@dataclass
class ExistExpansionValue:
    preexisting: float
    expansion: float

    @property
    def total(self) -> float:
        return self.preexisting + self.expansion

    def get_value(self, phase: Literal["baseline", "expansion"]) -> float:
        return self.preexisting if phase == "baseline" else self.total


@dataclass
class ComparisonInvestComponent:
    capacity: ExistExpansionValue = field(default_factory=lambda: ExistExpansionValue(preexisting=10e3, expansion=50e3))
    capex_spec: float = 1.0
    capem_spec: float = 1.0
    ls: int = 18


@dataclass
class ComparisonInputLocation:
    coordinates: sim.Coordinates = field(default_factory=sim.Coordinates)
    slp: str = "h0"
    consumption_yrl_wh: float = 10000000.0

    grid: ComparisonInvestComponent = field(default_factory=lambda: ComparisonInvestComponent())
    pv: ComparisonInvestComponent = field(default_factory=lambda: ComparisonInvestComponent())
    ess: ComparisonInvestComponent = field(default_factory=lambda: ComparisonInvestComponent())


@dataclass
class ComparisonInputSubfleet:
    name: str = "hlt"
    num_bev: ExistExpansionValue = field(default_factory=lambda: ExistExpansionValue(preexisting=1, expansion=4))
    num_total: int = 5
    battery_capacity_wh: float = 80e3
    charger: str = "ac"
    pwr_max_w: float = 11e3
    capex_bev_eur: float = 100e3
    capex_icev_eur: float = 80e3
    toll_frac: float = 0.3
    ls: float = 6.0
    capem_bev: float = 20000.0
    capem_icev: float = 15000.0
    mntex_eur_km_bev: float = 0.05
    mntex_eur_km_icev: float = 0.1
    consumption_icev: float = 27.0
    toll_eur_per_km_bev: float = 0.0
    toll_eur_per_km_icev: float = 1.0


@dataclass
class ComparisonInputCharger:
    name: str = "ac"
    num: ExistExpansionValue = field(default_factory=lambda: ExistExpansionValue(preexisting=0, expansion=4))
    pwr_max_w: float = 11e3
    cost_per_charger_eur: float = 3000.0
    capem: float = 1.0
    ls: float = 18.0


@dataclass
class ComparisonInputChargingInfrastructure:
    pwr_max_w_baseline: float = np.inf
    pwr_max_w_expansion: float = np.inf
    chargers: dict[str, ComparisonInputCharger] = field(default_factory=lambda: {"ac": ComparisonInputCharger()})


@dataclass
class ComparisonInputEconomics:
    period_sim: pd.Timedelta = field(default_factory=lambda: pd.Timedelta(days=365))
    start_sim: pd.Timestamp = field(default_factory=lambda: pd.Timestamp("2023-01-01 00:00"))
    freq_sim: pd.Timedelta = field(default_factory=lambda: pd.Timedelta(hours=1))
    fix_cost_construction: float = 10000
    opex_spec_grid_buy: float = 30e-5
    opex_spec_grid_sell: float = -6e-5
    opex_spec_grid_peak: float = 150e-3
    opex_spec_route_charging: float = 49e-5
    opex_fuel: float = 1.7
    period_eco: int = 18
    discount_rate: float = 0.05
    co2_per_liter_diesel_kg: float = 3.08
    opem_spec_grid: float = 0.0004


@dataclass
class ComparisonInput:
    location: ComparisonInputLocation = field(default_factory=ComparisonInputLocation)
    economics: ComparisonInputEconomics = field(default_factory=ComparisonInputEconomics)
    subfleets: dict[str, ComparisonInputSubfleet] = field(
        default_factory=lambda: {
            "hlt": ComparisonInputSubfleet(name="hlt"),
            "hst": ComparisonInputSubfleet(name="hst"),
        }
    )
    charging_infrastructure: ComparisonInputChargingInfrastructure = field(
        default_factory=lambda: ComparisonInputChargingInfrastructure()
    )


@dataclass
class ComparisonResult:
    baseline: eval.PhaseResult
    expansion: eval.PhaseResult

    @property
    def npc_delta(self) -> float:
        return self.baseline.cashflow_dis["totex"].sum() - self.expansion.cashflow_dis["totex"].sum()

    @staticmethod
    def get_payback_period_yrs(diff) -> float | None:
        idx = np.flatnonzero(np.diff(np.sign(diff)))

        if idx.size == 0 or diff[0] > 0:
            return None  # No intersection

        i = idx[0]
        y0, y1 = diff[i], diff[i + 1]

        # Linear interpolation to find x where y1 == y2
        return float((i - y0 / (y1 - y0)) + 1)

    @property
    def payback_period_yrs(self) -> float | None:
        diff = np.cumsum(self.baseline.cashflow_dis["totex"].sum(axis=0)) - np.cumsum(
            self.expansion.cashflow_dis["totex"].sum(axis=0)
        )
        return self.get_payback_period_yrs(diff)

    @property
    def co2_delta(self) -> float:
        return self.baseline.emissions["totex"].sum() - self.expansion.emissions["totex"].sum()

    @property
    def payback_period_co2_yrs(self) -> float | None:
        diff = np.cumsum(self.baseline.emissions["totex"].sum(axis=0)) - np.cumsum(
            self.expansion.emissions["totex"].sum(axis=0)
        )
        return self.get_payback_period_yrs(diff)
