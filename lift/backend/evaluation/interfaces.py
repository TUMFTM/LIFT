"""Phase-level interfaces for techno-economic evaluation.

Purpose:
- Represent scenario inputs (location, economics, subfleets, chargers) after applying
  baseline/expansion selection, and hold evaluation results (`PhaseResult`).

Relationships:
- Factories adapt comparison-layer inputs into phase-specific structures consumed by
  `lift.backend.evaluation.evaluate` and converted further to simulation inputs.
- Relies on `lift.backend.simulation` types (`SimResults`, `Coordinates`) for interoperability.

Key Logic:
- `PhaseInputEconomics.from_comparison_input` toggles fixed construction cost only for expansion.
- `PhaseInputInvestComponent` encapsulates shared attributes (capacity, CAPEX/CAPEM, lifespan).
- Location/Subfleet/Charger/Infrastructure classes expose `from_comparison_input` to select per-phase values.
- `PhaseResult` aggregates simulation outputs, KPIs, and arrays for cashflow, discounted cashflow, and emissions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Self

import numpy as np
import pandas as pd

import lift.backend.simulation as sim


class PhaseInputBase(ABC):
    @classmethod
    @abstractmethod
    def from_comparison_input(cls, comparison_input, phase: Literal["baseline", "expansion"]) -> Self: ...


@dataclass
class PhaseInputEconomics(PhaseInputBase):
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

    @classmethod
    def from_comparison_input(
        cls, comparison_input: "ComparisonInputEconomics", phase: Literal["baseline", "expansion"]
    ) -> Self:
        return cls(
            period_sim=comparison_input.period_sim,
            start_sim=comparison_input.start_sim,
            freq_sim=comparison_input.freq_sim,
            fix_cost_construction=comparison_input.fix_cost_construction if phase == "expansion" else 0,
            opex_spec_grid_buy=comparison_input.opex_spec_grid_buy,
            opex_spec_grid_sell=comparison_input.opex_spec_grid_sell,
            opex_spec_grid_peak=comparison_input.opex_spec_grid_peak,
            opex_spec_route_charging=comparison_input.opex_spec_route_charging,
            opex_fuel=comparison_input.opex_fuel,
            period_eco=comparison_input.period_eco,
            discount_rate=comparison_input.discount_rate,
            co2_per_liter_diesel_kg=comparison_input.co2_per_liter_diesel_kg,
            opem_spec_grid=comparison_input.opem_spec_grid,
        )


@dataclass
class PhaseInputInvestComponent(PhaseInputBase):
    capacity: float = 10e3
    capex_spec: float = 1.0
    capem_spec: float = 1.0
    opex_spec: float = 1.0
    opem_spec: float = 1.0
    ls: int = 18

    @classmethod
    def from_comparison_input(
        cls, comparison_input: "ComparisonInvestComponent", phase: Literal["baseline", "expansion"]
    ) -> Self:
        return cls(
            capacity=comparison_input.capacity.get_value(phase),  # select phase
            capex_spec=comparison_input.capex_spec,
            capem_spec=comparison_input.capem_spec,
            opex_spec=comparison_input.opex_spec,
            opem_spec=comparison_input.opem_spec,
            ls=comparison_input.ls,
        )


@dataclass
class PhaseInputLocation(PhaseInputBase):
    coordinates: sim.Coordinates = field(default_factory=sim.Coordinates)
    slp: str = "h0"
    consumption_yrl_wh: float = 3e6
    grid: PhaseInputInvestComponent = field(default_factory=lambda: PhaseInputInvestComponent())
    pv: PhaseInputInvestComponent = field(default_factory=lambda: PhaseInputInvestComponent())
    ess: PhaseInputInvestComponent = field(default_factory=lambda: PhaseInputInvestComponent())

    @classmethod
    def from_comparison_input(
        cls, comparison_input: "ComparisonInputLocation", phase: Literal["baseline", "expansion"]
    ) -> Self:
        return cls(
            coordinates=comparison_input.coordinates,
            slp=comparison_input.slp,
            consumption_yrl_wh=comparison_input.consumption_yrl_wh,
            grid=PhaseInputInvestComponent.from_comparison_input(comparison_input.grid, phase),
            pv=PhaseInputInvestComponent.from_comparison_input(comparison_input.pv, phase),
            ess=PhaseInputInvestComponent.from_comparison_input(comparison_input.ess, phase),
        )


@dataclass
class PhaseInputSubfleet(PhaseInputBase):
    name: str = "hlt"
    num_bev: int = 1
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

    @classmethod
    def from_comparison_input(
        cls, comparison_input: "ComparisonInputSubfleet", phase: Literal["baseline", "expansion"]
    ) -> Self:
        return cls(
            name=comparison_input.name,
            num_bev=comparison_input.num_bev.get_value(phase),
            num_total=comparison_input.num_total,
            battery_capacity_wh=comparison_input.battery_capacity_wh,
            charger=comparison_input.charger,
            pwr_max_w=comparison_input.pwr_max_w,
            capex_bev_eur=comparison_input.capex_bev_eur,
            capex_icev_eur=comparison_input.capex_icev_eur,
            toll_frac=comparison_input.toll_frac,
            ls=comparison_input.ls,
            capem_bev=comparison_input.capem_per_unit_bev,
            capem_icev=comparison_input.capem_per_unit_icev,
            mntex_eur_km_bev=comparison_input.mntex_spec_bev,
            mntex_eur_km_icev=comparison_input.mntex_spec_icev,
            consumption_icev=comparison_input.consumption_spec_icev,
            toll_eur_per_km_bev=comparison_input.toll_spec_bev,
            toll_eur_per_km_icev=comparison_input.toll_spec_icev,
        )


@dataclass
class PhaseInputCharger(PhaseInputBase):
    name: str = "ac"
    num: int = 1
    pwr_max_w: float = 11e3
    cost_per_charger_eur: float = 3000.0
    capem: float = 1.0
    ls: float = 18.0

    @classmethod
    def from_comparison_input(
        cls, comparison_input: "ComparisonInputCharger", phase: Literal["baseline", "expansion"]
    ) -> Self:
        return cls(
            name=comparison_input.name,
            num=comparison_input.num.get_value(phase),
            pwr_max_w=comparison_input.pwr_max_w,
            cost_per_charger_eur=comparison_input.capex_per_unit,
            capem=comparison_input.capem_per_unit,
            ls=comparison_input.ls,
        )


@dataclass
class PhaseInputChargingInfrastructure(PhaseInputBase):
    pwr_max_w: float = np.inf
    chargers: dict[str, PhaseInputCharger] = field(default_factory=lambda: {"ac": PhaseInputCharger()})

    @classmethod
    def from_comparison_input(
        cls, comparison_input: "ComparisonInputChargingInfrastructure", phase: Literal["baseline", "expansion"]
    ):
        return cls(
            pwr_max_w=comparison_input.pwr_max_w_baseline
            if phase == "baseline"
            else comparison_input.pwr_max_w_expansion,
            chargers={
                charger_name: PhaseInputCharger.from_comparison_input(charger, phase=phase)
                for charger_name, charger in comparison_input.chargers.items()
            },
        )


@dataclass
class PhaseResult:
    simulation: sim.SimResults = field(default_factory=sim.SimResults)
    self_sufficiency: float = (
        0.0  # share of energy demand (fleet + site) which is satisfied by the PV (produced - fed in)
    )
    self_consumption: float = (
        0.0  # share of the energy produced by the on-site PV array which is consumed on-site (1 - feed-in / produced)
    )
    site_charging: float = 0.0  # share of the fleet energy demand which is charged on-site (vs on-route)
    cashflow: np.typing.NDArray[np.floating] = field(init=True, default_factory=lambda: np.zeros(18))
    cashflow_dis: np.typing.NDArray[np.floating] = field(init=True, default_factory=lambda: np.zeros(18))
    emissions: np.typing.NDArray[np.floating] = field(init=True, default_factory=lambda: np.zeros(18))
