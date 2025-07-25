from lift.interfaces import (
    LocationSettings,
    SubFleetSettings,
    ChargingInfrastructureSettings,
    ElectrificationPhasesSettings,
    EconomicSettings,
    TcoResults,
    VariousResults,
)


def run_backend(location_settings: LocationSettings,
                fleet_settings: dict[str, SubFleetSettings],
                charging_infrastructure_settings: ChargingInfrastructureSettings,
                electrification_phases_settings: ElectrificationPhasesSettings,
                economic_settings: EconomicSettings) -> (TcoResults, VariousResults):
    print('run_backend')

    return TcoResults(dummy=0.0), VariousResults(dummy=0.0)

