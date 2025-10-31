from time import time

import pandas as pd

from lift.backend.economics.interfaces import (
    PhaseInputLocation,
    PhaseInputEconomics,
    PhaseInputSubfleet,
    PhaseInputChargingInfrastructure,
)

from lift.backend.economics.economics_phase import calc_phase_results

from lift.utils import safe_cache_data

from lift.backend.comparison.interfaces import (
    ComparisonInput,
    ComparisonResult,
)


@safe_cache_data
def run_comparison(comparison_input: ComparisonInput) -> ComparisonResult:
    # start time tracking
    start_time = time()

    results = {
        phase: calc_phase_results(
            location=PhaseInputLocation.from_comparison_input(comparison_input=comparison_input.location, phase=phase),
            economics=PhaseInputEconomics.from_comparison_input(
                comparison_input=comparison_input.economics, phase=phase
            ),
            subfleets={
                sf.name: PhaseInputSubfleet.from_comparison_input(comparison_input=sf, phase=phase)
                for sf in comparison_input.subfleets.values()
            },
            charging_infrastructure=PhaseInputChargingInfrastructure.from_comparison_input(
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


if __name__ == "__main__":
    settings_default = ComparisonInput()
    result = run_comparison(comparison_input=settings_default)
    print(result.baseline.simulation)
    print(result.expansion.simulation)
