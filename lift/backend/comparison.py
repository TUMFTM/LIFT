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

from lift.backend.utils import safe_cache_data

from lift.backend.scenario import SingleScenario
from lift.backend.interfaces import (
    ComparisonScenario,
    ComparisonResult,
)


@safe_cache_data
def run_comparison(comp_scn: ComparisonScenario) -> ComparisonResult:
    # start time tracking
    start_time = time()

    scn_dict = asdict(comp_scn)
    scn_dict.pop("settings")

    results = {
        phase: SingleScenario.from_comparison(comp_obj=comp_scn, phase=phase).simulate()
        for phase in ["baseline", "expansion"]
    }

    # stop time tracking
    print(f"{pd.Timestamp.now().isoformat()} - Backend calculation completed in {time() - start_time:.2f} seconds.")

    return ComparisonResult(**results)
