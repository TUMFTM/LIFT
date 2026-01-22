from lift.frontend.interfaces import (
    FrontendChargingInfrastructureInterface,
    FrontendESSInterface,
    FrontendFleetInterface,
    FrontendGridInterface,
    FrontendPVInterface,
    FrontendScenarioInterface,
)
from lift.frontend.utils import get_supported_languages, read_json_from_package_data

DEF_LANGUAGE_OPTIONS = get_supported_languages()

DEF_GRID = FrontendGridInterface.from_dict(read_json_from_package_data("grid.json"))
DEF_PV = FrontendPVInterface.from_dict(read_json_from_package_data("pv.json"))
DEF_ESS = FrontendESSInterface.from_dict(read_json_from_package_data("ess.json"))
DEF_CIS = FrontendChargingInfrastructureInterface.from_dict(read_json_from_package_data("cis.json"))
DEF_FLEET = FrontendFleetInterface.from_dict(read_json_from_package_data("fleet.json"))
DEF_SCN = FrontendScenarioInterface.from_dict(read_json_from_package_data("scn.json"))
