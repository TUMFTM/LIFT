from lift.interfaces import (
    LocationSettings,
    SubFleetSettings,
    ChargingInfrastructureSettings,
    ElectrificationPhasesSettings,
    EconomicSettings,
    TcoResults,
    VariousResults,
)


def calc_results(location_settings: LocationSettings,
                 subfleet_settings: SubFleetSettings,
                 charging_infrastructure_settings: ChargingInfrastructureSettings,
                 electrification_phases_settings: ElectrificationPhasesSettings,
                 economic_settings: EconomicSettings) -> (TcoResults, VariousResults):


    return TcoResults(dummy=0.0), VariousResults(dummy=0.0)

