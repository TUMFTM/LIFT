from time import time

import pandas as pd

from lift.backend.phase_economics.interfaces import (
    PhaseInputLocation,
    PhaseInputEconomics,
    PhaseInputSubfleet,
    PhaseInputCharger,
)

from lift.backend.phase_economics.economics_phase import calc_phase_results

from lift.utils import safe_cache_data

from lift.backend.interfaces import (
    Inputs,
    TotalResults,
)


@safe_cache_data
def run_backend(inputs: Inputs) -> TotalResults:
    # start time tracking
    start_time = time()

    results = {
        phase: calc_phase_results(
            location=PhaseInputLocation.from_comparison_input(comparison_input=inputs.location, phase=phase),
            economics=PhaseInputEconomics.from_comparison_input(comparison_input=inputs.economics, phase=phase),
            subfleets={
                sf.name: PhaseInputSubfleet.from_comparison_input(comparison_input=sf, phase=phase)
                for sf in inputs.subfleets.values()
            },
            chargers={
                chg.name: PhaseInputCharger.from_comparison_input(comparison_input=chg, phase=phase)
                for chg in inputs.chargers.values()
            },
        )
        for phase in ["baseline", "expansion"]
    }

    # stop time tracking
    print(f"{pd.Timestamp.now().isoformat()} - Backend calculation completed in {time() - start_time:.2f} seconds.")

    return TotalResults(
        baseline=results["baseline"],
        expansion=results["expansion"],
    )


if __name__ == "__main__":
    settings_default = Inputs()
    result = run_backend(inputs=settings_default)
    print(result.baseline.simulation)
    print(result.expansion.simulation)
