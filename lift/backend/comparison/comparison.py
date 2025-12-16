"""Comparison orchestrator for LIFT.

Purpose:
- Runs a two-phase comparison (baseline vs. expansion) and aggregates results.

Relationships:
- Converts high-level `ComparisonInput` into phase-specific inputs via
  `lift.backend.evaluation` (aliased as `eval`) factories.
- Delegates per-phase evaluation to `eval.evaluate(...)`, which internally calls
  simulation and economics phases.
- Returns a `ComparisonResult` bundling both phases and convenience metrics.

Key Logic:
- Memoized with `safe_cache_data` to avoid recomputation for identical inputs.
- Iterates phases `["baseline", "expansion"]`, builds phase inputs, evaluates,
  collects results, logs timing, and returns a structured result.
"""

from dataclasses import asdict

from time import time

import pandas as pd

from lift.frontend.definitions import DEF_GRID, DEF_PV, DEF_ESS, DEF_FLEET, DEF_CIS, DEF_SCN

from lift.backend.comparison.interfaces import (
    ExistExpansionValue,
    ComparisonScenario,
    ComparisonSettings,
    ComparisonFix,
    ComparisonFixedDemand,
    ComparisonGrid,
    ComparisonPV,
    ComparisonESS,
    ComparisonFleet,
    ComparisonSubFleet,
    ComparisonChargingInfrastructure,
    ComparisonChargerType,
    ComparisonResult,
)

from lift.backend.utils import safe_cache_data

from lift.backend.evaluation.blocks import Scenario


@safe_cache_data
def run_comparison(comp_scn: ComparisonScenario) -> ComparisonResult:
    # start time tracking
    start_time = time()

    scn_dict = asdict(comp_scn)
    scn_dict.pop("settings")

    results = {
        phase: Scenario.from_comparison(comp_obj=comp_scn, phase=phase).simulate()
        for phase in ["baseline", "expansion"]
    }

    # stop time tracking
    print(f"{pd.Timestamp.now().isoformat()} - Backend calculation completed in {time() - start_time:.2f} seconds.")

    return ComparisonResult(**results)


if __name__ == "__main__":
    start = time()

    comp_scn = ComparisonScenario(
        settings=ComparisonSettings(
            latitude=48.1351,
            longitude=11.5820,
            wacc=DEF_SCN.wacc.value * DEF_SCN.wacc.factor,
            period_eco=DEF_SCN.period_eco,
            sim_start=DEF_SCN.sim_start,
            sim_duration=DEF_SCN.sim_duration,
            sim_freq=DEF_SCN.sim_freq,
        ),
        fix=ComparisonFix(
            capex_initial=ExistExpansionValue(
                baseline=0.0,
                expansion=DEF_SCN.capex_initial.value * DEF_SCN.capex_initial.factor,
            ),
            capem_initial=ExistExpansionValue(
                baseline=DEF_SCN.capem_initial,
                expansion=DEF_SCN.capem_initial,
            ),
        ),
        dem=ComparisonFixedDemand(
            slp=DEF_SCN.slp.value.lower(),
            e_yrl=DEF_SCN.e_yrl.value * DEF_SCN.e_yrl.factor,
        ),
        grid=ComparisonGrid(
            capacity=ExistExpansionValue(
                baseline=DEF_GRID.capacity_preexisting.value * DEF_GRID.capacity_preexisting.factor,
                expansion=(DEF_GRID.capacity_preexisting.value + DEF_GRID.capacity_expansion.value)
                * DEF_GRID.capacity_expansion.factor,
            ),
            opex_spec_buy=DEF_GRID.opex_spec_buy.value * DEF_GRID.opex_spec_buy.factor,
            opex_spec_sell=DEF_GRID.opex_spec_sell.value * DEF_GRID.opex_spec_sell.factor,
            opex_spec_peak=DEF_GRID.opex_spec_peak.value * DEF_GRID.opex_spec_peak.factor,
            **DEF_GRID.values,
        ),
        pv=ComparisonPV(
            capacity=ExistExpansionValue(
                baseline=DEF_PV.capacity_preexisting.value * DEF_PV.capacity_preexisting.factor,
                expansion=(DEF_PV.capacity_preexisting.value + DEF_PV.capacity_expansion.value)
                * DEF_PV.capacity_expansion.factor,
            ),
            **DEF_PV.values,
        ),
        ess=ComparisonESS(
            capacity=ExistExpansionValue(
                baseline=DEF_ESS.capacity_preexisting.value * DEF_ESS.capacity_preexisting.factor,
                expansion=(DEF_ESS.capacity_preexisting.value + DEF_ESS.capacity_expansion.value)
                * DEF_ESS.capacity_expansion.factor,
            ),
            **DEF_ESS.values,
        ),
        fleet=ComparisonFleet(
            subblocks={
                k: ComparisonSubFleet(
                    name=k,
                    num_bev=ExistExpansionValue(
                        baseline=0,
                        expansion=0,
                    ),
                    num_icev=ExistExpansionValue(
                        baseline=0,
                        expansion=0,
                    ),
                    charger="ac",
                    p_max=11 * 1e3,
                    capex_per_unit_bev=250000 * DEF_FLEET.subblocks[k].capex_per_unit_bev.factor,
                    capex_per_unit_icev=150000 * DEF_FLEET.subblocks[k].capex_per_unit_icev.factor,
                    toll_frac=80 * DEF_FLEET.subblocks[k].toll_frac.factor,
                    **DEF_FLEET.subblocks[k].values,
                )
                for k in DEF_FLEET.subblocks.keys()
            },
            opex_spec_fuel=1.56 * DEF_FLEET.opex_spec_fuel.factor,
            opex_spec_onroute_charging=49 * DEF_FLEET.opex_spec_onroute_charging.factor,
            **DEF_FLEET.values,
        ),
        cis=ComparisonChargingInfrastructure(
            p_lm_max=ExistExpansionValue(
                baseline=11e3,
                expansion=11e3,
            ),
            subblocks={
                k: ComparisonChargerType(
                    name=k,
                    num=ExistExpansionValue(baseline=0, expansion=0),
                    p_max=11 * 1e3,
                    capex_per_unit=800,
                    **DEF_CIS.subblocks[k].values,
                )
                for k in DEF_CIS.subblocks.keys()
            },
        ),
    )

    scn_dict = {
        "baseline": Scenario.from_comparison(comp_scn, phase="baseline"),
        "expansion": Scenario.from_comparison(comp_scn, phase="expansion"),
    }

    results = {scn_name: scn_obj.simulate() for scn_name, scn_obj in scn_dict.items()}

    print(f"Simulation completed in {time() - start:.2f} seconds.")
