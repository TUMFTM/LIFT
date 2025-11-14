"""Phase evaluation pipeline: simulation + techno-economic aggregation.

Purpose:
- Execute the simulation for a given phase (baseline/expansion) and translate the
  outputs into financial and emission cash flows, plus KPIs such as self-sufficiency.

Relationships:
- Consumes phase-specific inputs from `.interfaces` and converts them to simulation
  inputs via `lift.backend.simulation` factories.
- Relies on `encapsulated` simulation results to populate economics/emissions arrays
  used downstream in comparisons and frontend visualizations.
- Memoized with `safe_cache_data` to avoid recomputation for identical phase inputs.

Key Logic / Formulations:
- Self-sufficiency = (PV potential − curtailment − export) / (site demand + fleet site demand).
- Self-consumption = 1 − (export + curtailment) / PV potential.
- Site-charging share = fleet site energy / (site + route energy).
- Replacement schedules generated via `calc_replacements` (lifespan multiples, salvages).
- Discounting factors computed by `_calc_discount_factors(period, occurs_at, rate)` for CAPEX/OPEX.
- Aggregates CAPEX, OPEX, TOTEX, and emissions across categories (general, grid, PV, ESS,
  vehicles, chargers) and returns a structured `PhaseResult`.
"""

from typing import Literal

import numpy as np

from lift.backend.utils import safe_cache_data

import lift.backend.simulation as sim

from .interfaces import (
    PhaseInputEconomics,
    PhaseInputLocation,
    PhaseInputSubfleet,
    PhaseResult,
    PhaseInputChargingInfrastructure,
)


CATEGORIES = ["general", "grid", "pv", "ess", "vehicles", "chargers"]
CONV = {name: idx for idx, name in enumerate(CATEGORIES)}


@safe_cache_data
def _calc_discount_factors(
    period_sim: int, occurs_at: Literal["beginning", "middle", "end"], discount_rate: float
) -> np.typing.NDArray[float]:
    if discount_rate is None or discount_rate < 0:
        raise ValueError("A positive discount rate must be provided if discounting is enabled.")

    periods = np.arange(0, period_sim + 1) + 1
    q = 1 + discount_rate

    exp = {"beginning": 1, "middle": 0.5, "end": 0}.get(occurs_at, 0)
    return 1 / (q ** (periods - exp))


def evaluate(
    location: PhaseInputLocation,
    economics: PhaseInputEconomics,
    subfleets: dict[str, PhaseInputSubfleet],
    charging_infrastructure: PhaseInputChargingInfrastructure,
) -> PhaseResult:
    result_sim = sim.simulate(
        settings=sim.SimInputSettings.from_phase_input(economics),
        location=sim.SimInputLocation.from_phase_input(location),
        subfleets={sf.name: sim.SimInputSubfleet.from_phase_input(sf) for sf in subfleets.values()},
        charging_infrastructure=sim.SimInputChargingInfrastructure.from_phase_input(charging_infrastructure),
    )

    period_eco = economics.period_eco

    dtype_flow = np.dtype(
        [
            ("name", "U10"),
            ("capex", "f8", (period_eco + 1,)),
            ("opex", "f8", (period_eco + 1,)),
            ("totex", "f8", (period_eco + 1,)),
        ]
    )

    cashflow = np.array(
        [(category, *tuple(np.zeros(period_eco + 1) for _ in range(3))) for category in CATEGORIES], dtype=dtype_flow
    )

    cashflow_dis = np.array(
        [(category, *tuple(np.zeros(period_eco + 1) for _ in range(3))) for category in CATEGORIES], dtype=dtype_flow
    )

    emissions = np.array(
        [(category, *tuple(np.zeros(period_eco + 1) for _ in range(3))) for category in CATEGORIES], dtype=dtype_flow
    )

    # calculate self-sufficiency and self consumption based on simulation results
    if location.pv.capacity == 0:
        self_sufficiency = 0.0
        self_consumption = 0.0
    else:
        if result_sim.energy_dem_site_wh + result_sim.energy_fleet_site_wh == 0:
            self_sufficiency = 0.0
        else:
            # potential PV energy minus curtailed and sold energy divided by total energy demand (fleet@site + site)
            self_sufficiency = (
                result_sim.energy_pv_pot_wh - result_sim.energy_pv_curt_wh - result_sim.energy_grid_sell_wh
            ) / (result_sim.energy_fleet_site_wh + result_sim.energy_dem_site_wh)
        if result_sim.energy_pv_pot_wh == 0:
            self_consumption = 0.0
        else:
            # 1 - (pv energy not used on-site (curtailed or sold) divided by total potential PV energy)
            self_consumption = 1 - (
                (result_sim.energy_grid_sell_wh + result_sim.energy_pv_curt_wh) / result_sim.energy_pv_pot_wh
            )

    # calculate share charged energy which was charged at the own site (vs. on route)
    if (result_sim.energy_fleet_site_wh + result_sim.energy_fleet_route_wh) == 0:
        site_charging = 0.0
    else:
        site_charging = result_sim.energy_fleet_site_wh / (
            result_sim.energy_fleet_site_wh + result_sim.energy_fleet_route_wh
        )

    def calc_replacements(ls: float) -> np.typing.NDArray:
        years = np.arange(economics.period_eco + 1)

        # Replacement years: start + 0, lifespan, 2*lifespan, ...
        replacement_years = np.arange(0, economics.period_eco, ls)

        repl = np.isin(years, replacement_years).astype(float)

        # salvage value
        repl[economics.period_eco] = (
            (-1 * (1 - (economics.period_eco % ls) / ls)) if economics.period_eco % ls != 0 else 0
        )
        return repl

    # fix cost
    cashflow[CONV["general"]]["capex"][0] += economics.fix_cost_construction

    # grid
    replacements = calc_replacements(ls=location.grid.ls)
    cashflow[CONV["grid"]]["capex"] += location.grid.capacity * location.grid.capex_spec * replacements
    cashflow[CONV["grid"]]["opex"][:period_eco] += (
        result_sim.energy_grid_buy_wh * economics.opex_spec_grid_buy
        + result_sim.energy_grid_sell_wh * economics.opex_spec_grid_sell
    )
    cashflow[CONV["grid"]]["opex"][:period_eco] += result_sim.pwr_grid_peak_w * economics.opex_spec_grid_peak

    emissions[CONV["grid"]]["capex"] += location.grid.capacity * location.grid.capem_spec * replacements
    emissions[CONV["grid"]]["opex"][:period_eco] += result_sim.energy_grid_buy_wh * economics.opem_spec_grid

    # pv
    replacements = calc_replacements(ls=location.pv.ls)
    cashflow[CONV["pv"]]["capex"] += location.pv.capacity * location.pv.capex_spec * replacements
    emissions[CONV["pv"]]["capex"] += location.pv.capacity * location.pv.capem_spec * replacements

    # ess
    replacements = calc_replacements(ls=location.ess.ls)
    cashflow[CONV["ess"]]["capex"] += location.ess.capacity * location.ess.capex_spec * replacements
    emissions[CONV["ess"]]["capex"] += location.ess.capacity * location.ess.capem_spec * replacements

    # vehicles
    cashflow[CONV["vehicles"]]["opex"][:period_eco] += (
        result_sim.energy_fleet_route_wh * economics.opex_spec_route_charging
    )
    emissions[CONV["vehicles"]]["opex"][:period_eco] += result_sim.energy_fleet_route_wh * economics.opem_spec_grid

    for sf_name, sf in subfleets.items():
        replacements = calc_replacements(ls=sf.ls)
        num_bev = sf.num_bev
        num_icev = sf.num_total - sf.num_bev

        cashflow[CONV["vehicles"]]["capex"] += (
            num_bev * sf.capex_bev_eur * replacements + num_icev * sf.capex_icev_eur * replacements
        )
        emissions[CONV["vehicles"]]["capex"] += (
            num_bev * sf.capem_bev * replacements + num_icev * sf.capem_icev * replacements
        )

        cashflow[CONV["vehicles"]]["opex"][:period_eco] += result_sim.dist_km[sf_name]["bev"] * (
            sf.mntex_eur_km_bev  # maintenance
            + sf.toll_eur_per_km_bev * sf.toll_frac  # toll
        )

        # icev
        cashflow[CONV["vehicles"]]["opex"][:period_eco] += result_sim.dist_km[sf_name]["icev"] * (
            sf.mntex_eur_km_icev  # maintenance
            + sf.toll_eur_per_km_icev * sf.toll_frac  # toll
            + economics.opex_fuel * sf.consumption_icev / 100  # fuel
        )
        emissions[CONV["vehicles"]]["opex"][:period_eco] += (
            result_sim.dist_km[sf_name]["icev"] * sf.consumption_icev / 100 * economics.co2_per_liter_diesel_kg
        )

    # chargers
    for chg in charging_infrastructure.chargers.values():
        replacements = calc_replacements(ls=chg.ls)
        cashflow[CONV["chargers"]]["capex"] += chg.cost_per_charger_eur * chg.num * replacements
        emissions[CONV["chargers"]]["capex"] += chg.capem * chg.num * replacements

    # add discounting
    cashflow_dis["capex"] = cashflow["capex"] * _calc_discount_factors(
        period_sim=economics.period_eco, occurs_at="beginning", discount_rate=economics.discount_rate
    )

    cashflow_dis["opex"] = cashflow["opex"] * _calc_discount_factors(
        period_sim=economics.period_eco, occurs_at="end", discount_rate=economics.discount_rate
    )

    # calculate totex
    for flow in [cashflow, cashflow_dis, emissions]:
        flow["totex"] = flow["capex"] + flow["opex"]

    return PhaseResult(
        simulation=result_sim,
        self_sufficiency=self_sufficiency,
        self_consumption=self_consumption,
        site_charging=site_charging,
        cashflow=cashflow,
        cashflow_dis=cashflow_dis,
        emissions=emissions,
    )
