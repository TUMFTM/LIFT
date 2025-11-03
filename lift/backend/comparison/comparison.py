from time import time

import pandas as pd

import lift.backend.evaluation as eval

from lift.backend.utils import safe_cache_data

from .interfaces import (
    ComparisonInput,
    ComparisonResult,
)


@safe_cache_data
def run_comparison(comparison_input: ComparisonInput) -> ComparisonResult:
    # start time tracking
    start_time = time()

    results = {
        phase: eval.evaluate(
            location=eval.PhaseInputLocation.from_comparison_input(
                comparison_input=comparison_input.location, phase=phase
            ),
            economics=eval.PhaseInputEconomics.from_comparison_input(
                comparison_input=comparison_input.economics, phase=phase
            ),
            subfleets={
                sf.name: eval.PhaseInputSubfleet.from_comparison_input(comparison_input=sf, phase=phase)
                for sf in comparison_input.subfleets.values()
            },
            charging_infrastructure=eval.PhaseInputChargingInfrastructure.from_comparison_input(
                comparison_input.charging_infrastructure, phase=phase
            ),
        )
        for phase in ["baseline", "expansion"]
    }

    # stop time tracking
    print(f"{pd.Timestamp.now().isoformat()} - Backend calculation completed in {time() - start_time:.2f} seconds.")

    return ComparisonResult(
        baseline=results["baseline"],
        expansion=results["expansion"],
    )
