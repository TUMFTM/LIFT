from lift.backend import backend

from lift.backend.interfaces import (
    InputLocation,
    InputSubfleet,
    InputCharger,
    InputEconomics,
    Inputs,
    Coordinates,
    ExistExpansionValue,
)


if __name__ == "__main__":
    inputs = Inputs(
        location=InputLocation(
            coordinates=Coordinates(longitude=11.576124,
                                    latitude=48.137154),
            slp='g0',
            consumption_yrl_wh=0.0,
            grid_capacity_w=ExistExpansionValue(preexisting=200E3,
                                                expansion=0.0),
            pv_capacity_wp=ExistExpansionValue(preexisting=0.0,
                                               expansion=0.0),
            ess_capacity_wh=ExistExpansionValue(preexisting=0.0,
                                                expansion=0.0),
        ),
        economics=InputEconomics(
            fix_cost_construction=0.0,
            opex_spec_grid_buy=49E-5,
            opex_spec_grid_sell=0.0,
            opex_spec_grid_peak=0.0,
            opex_spec_route_charging=49E-5,
            opex_fuel=0.0,
            insurance_frac=0.0,
            salvage_bev_frac=0.0,
            salvage_icev_frac=0.0,
        ),
        subfleets=dict(
            hlt=InputSubfleet(
                name="hlt",
                battery_capacity_wh=480E3,
                capex_bev_eur=0.0,
                capex_icev_eur=0.0,
                charger='dc',
                num_total=1,
                num_bev=ExistExpansionValue(preexisting=1,
                                            expansion=0),
                pwr_max_w=100E3,
                toll_frac=0.0,
            ),
        ),
        chargers=dict(
            dc=InputCharger(
                name="dc",
                cost_per_charger_eur=0.0,
                pwr_max_w=150E3,
                num=ExistExpansionValue(preexisting=0,
                                        expansion=1),
            ),
        )
    )
    backend.run_backend(inputs)
    pass