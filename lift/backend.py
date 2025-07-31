from typing import TYPE_CHECKING

import numpy as np

from lift.interfaces import (
    LocationSettings,
    SubFleetSettings,
    ChargerSettings,
    EconomicSettings,
    Settings,
    SubFleetResults,
    SystemResults,
    Results,

)

if TYPE_CHECKING:
    pass


def calc_tco(subfleet_settings: SubFleetSettings,
             economic_settings: EconomicSettings,
             capex_vehicle: float,
             capex_infrastructure: float,
             salvage_value_pct: float,
             mntex_eur_km: float,
             toll_eur_km: float,
             energy_cost: float,
             consumption_km: float,
             ):

    time_total_yrs = economic_settings.period_holding_yrs
    time_total_h = time_total_yrs * 8 * economic_settings.working_days_yrl  # assuming 8 hours per working day
    dist_total = subfleet_settings.dist_avg_daily_km * economic_settings.working_days_yrl * time_total_yrs

    capex = (capex_vehicle + capex_infrastructure) * (1 - (salvage_value_pct / 100))

    opex_dist = (subfleet_settings.toll_share_pct * toll_eur_km +
                 mntex_eur_km +
                 energy_cost * consumption_km) * dist_total

    opex_time = (economic_settings.insurance_pct / 100 * capex_vehicle * time_total_yrs +
                 economic_settings.driver_wage_eur_h * time_total_h)

    return (capex + opex_dist + opex_time) / dist_total


def calc_subfleet_results(subfleet_settings: SubFleetSettings,
                          ci_settings: ChargerSettings,
                          location_settings: LocationSettings,
                          economic_settings: EconomicSettings) -> SubFleetResults:

    # calculate TCO
    tco_bev = calc_tco(subfleet_settings=subfleet_settings,
                       economic_settings=economic_settings,
                       capex_vehicle=subfleet_settings.capex_bev,
                       # ToDo: don't use total number of chargers but only chargers for subfleet
                       capex_infrastructure=ci_settings.cost_per_charger_eur * ci_settings.num,
                       salvage_value_pct=economic_settings.salvage_bev_pct,
                       mntex_eur_km=economic_settings.mntex_bev_eur_km,
                       toll_eur_km=economic_settings.toll_bev_eur_km,
                       energy_cost=economic_settings.electricity_price_eur_kwh,
                       consumption_km=(0.3814 * np.log(subfleet_settings.weight_empty_bev_kg +
                                                       subfleet_settings.load_avg_t) - 2.6735),
                       )

    tco_icev = calc_tco(subfleet_settings=subfleet_settings,
                        economic_settings=economic_settings,
                        capex_vehicle=subfleet_settings.capex_icev,
                        capex_infrastructure=0,
                        salvage_value_pct=economic_settings.salvage_icev_pct,
                        mntex_eur_km=economic_settings.mntex_icev_eur_km,
                        toll_eur_km=economic_settings.toll_icev_eur_km,
                        energy_cost=economic_settings.fuel_price_eur_liter,
                        consumption_km=(0.0903 * np.log(subfleet_settings.weight_empty_icev_kg +
                                                        subfleet_settings.load_avg_t) - 0.6404),
                        )


    return SubFleetResults(num_total=subfleet_settings.num_total,
                           num_bev=subfleet_settings.num_bev,
                           num_bev_additional=0,
                           num_icev=subfleet_settings.num_total - subfleet_settings.num_bev,
                           tco_bev=tco_bev,
                           tco_icev=tco_icev,
                           energy_bev_daily_kwh=0,  # ToDo
                           )



def run_backend(settings: Settings) -> Results:
    print('run_backend')

    subfleet_results = {subfleet_id: calc_subfleet_results(subfleet_settings=subfleet_settings,
                                                           ci_settings=settings.charging_infrastructure,
                                                           economic_settings=settings.economic,
                                                           location_settings=settings.location)
                        for subfleet_id, subfleet_settings in settings.subfleets.items()}

    system_results = SystemResults(
        pv_capacity_kwp = settings.location.pv_capacity_kwp,
        pv_energy_yrl_kwh = 0.0,  # ToDo
        battery_capacity_kwh = settings.location.battery_capacity_kwh,
        grid_capacity_kw = settings.location.grid_capacity_kw,
        num_dc_chargers = settings.charging_infrastructure.num,
        energy_daily_bevs_kwh = 0.0,  # ToDo
    )

    return Results(subfleets=subfleet_results,
                   system=system_results,
                   )

