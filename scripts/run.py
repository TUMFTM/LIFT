import pandas as pd

from lift.backend.backend import run_backend

from lift.backend.interfaces import (
    SimInputLocation,
    SimInputSubfleet,
    SimInputCharger,
    InputEconomics,
    InputInvestComponent,
    Inputs,
    ExistExpansionValue,
)
from lift.backend.phase_simulation.interfaces import Coordinates

"""
Use this script to directly run the backend without the frontend.
This allows for scalable multi-scenario execution.
"""


def run_scenario():
    inputs = Inputs(
        location=SimInputLocation(
            coordinates=Coordinates(longitude=11.576124, latitude=48.137154),
            slp="g0",
            consumption_yrl_wh=0.0,
            grid=InputInvestComponent(
                capacity=ExistExpansionValue(preexisting=100, expansion=0),
                capex_spec=0.0,
                capem_spec=0.0,
                ls=18,
            ),
            pv=InputInvestComponent(
                capacity=ExistExpansionValue(preexisting=0, expansion=0),
                capex_spec=0.0,
                capem_spec=0.0,
                ls=18,
            ),
            ess=InputInvestComponent(
                capacity=ExistExpansionValue(preexisting=0, expansion=0),
                capex_spec=0.0,
                capem_spec=0.0,
                ls=18,
            ),
        ),
        economics=InputEconomics(
            fix_cost_construction=0,
            opex_spec_grid_buy=49e-5,
            opex_spec_grid_sell=0.0,
            opex_spec_grid_peak=0.0,
            opex_spec_route_charging=49e-5,
            opex_fuel=0.0,
            insurance_frac=0.0,
            salvage_bev_frac=0.0,
            salvage_icev_frac=0.0,
            period_eco=18,
            period_sim=pd.Timedelta(days=365),
            freq_sim=pd.Timedelta(hours=1),
            co2_per_liter_diesel_kg=3.08,
            opem_spec_grid=0.0004,
        ),
        subfleets=dict(
            hlt=SimInputSubfleet(
                name="hlt",
                battery_capacity_wh=480e3,
                capex_bev_eur=0.0,
                capex_icev_eur=0.0,
                charger="dc",
                num_total=1,
                num_bev=ExistExpansionValue(preexisting=1, expansion=0),
                pwr_max_w=100e3,
                toll_frac=0.0,
            ),
        ),
        chargers=dict(
            dc=SimInputCharger(
                name="dc",
                cost_per_charger_eur=0.0,
                pwr_max_w=150e3,
                num=ExistExpansionValue(preexisting=0, expansion=1),
            ),
        ),
    )
    return run_backend(inputs)


if __name__ == "__main__":
    results = run_scenario()
