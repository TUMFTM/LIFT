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
    import numpy as np
    import pandas as pd

    def value_factor_product(x):
        return x.value * x.factor

    start = time()

    comp_scn = ComparisonScenario(
        settings=ComparisonSettings(
            latitude=48.1351,
            longitude=11.5820,
            wacc=value_factor_product(DEF_SCN.wacc),
            period_eco=DEF_SCN.period_eco,
            sim_start=DEF_SCN.sim_start,
            sim_duration=DEF_SCN.sim_duration,
            sim_freq=DEF_SCN.sim_freq,
        ),
        fix=ComparisonFix(
            capex_initial=ExistExpansionValue(
                baseline=0.0,
                expansion=value_factor_product(DEF_SCN.capex_initial),
            ),
            capem_initial=ExistExpansionValue(
                baseline=DEF_SCN.capem_initial,
                expansion=DEF_SCN.capem_initial,
            ),
        ),
        dem=ComparisonFixedDemand(
            slp=DEF_SCN.slp.value.lower(),
            e_yrl=value_factor_product(DEF_SCN.e_yrl),
        ),
        grid=ComparisonGrid(
            capacity=ExistExpansionValue(
                baseline=value_factor_product(DEF_GRID.capacity_preexisting),
                expansion=value_factor_product(DEF_GRID.capacity_preexisting)
                + value_factor_product(DEF_GRID.capacity_expansion),
            ),
            opex_spec_buy=value_factor_product(DEF_GRID.opex_spec_buy),
            opex_spec_sell=value_factor_product(DEF_GRID.opex_spec_sell),
            opex_spec_peak=value_factor_product(DEF_GRID.opex_spec_peak),
            **DEF_GRID.values,
        ),
        pv=ComparisonPV(
            capacity=ExistExpansionValue(
                baseline=value_factor_product(DEF_PV.capacity_preexisting),
                expansion=value_factor_product(DEF_PV.capacity_preexisting)
                + value_factor_product(DEF_PV.capacity_expansion),
            ),
            **DEF_PV.values,
        ),
        ess=ComparisonESS(
            capacity=ExistExpansionValue(
                baseline=value_factor_product(DEF_ESS.capacity_preexisting),
                expansion=value_factor_product(DEF_ESS.capacity_preexisting)
                + value_factor_product(DEF_ESS.capacity_expansion),
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
                    capex_per_unit_bev=value_factor_product(DEF_FLEET.subblocks[k].capex_per_unit_bev),
                    capex_per_unit_icev=value_factor_product(DEF_FLEET.subblocks[k].capex_per_unit_icev),
                    toll_frac=value_factor_product(DEF_FLEET.subblocks[k].toll_frac),
                    **DEF_FLEET.subblocks[k].values,
                )
                for k in DEF_FLEET.subblocks.keys()
            },
            opex_spec_fuel=value_factor_product(DEF_FLEET.opex_spec_fuel),
            opex_spec_onroute_charging=value_factor_product(DEF_FLEET.opex_spec_onroute_charging),
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

    comp_result = ComparisonResult(baseline=results["baseline"], expansion=results["expansion"])

    df_cost_comparison = pd.DataFrame({k: np.cumsum(v.totex_dis) for k, v in results.items()})

    print(f"Simulation completed in {time() - start:.2f} seconds.")
