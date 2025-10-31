import numpy as np
import pandas as pd

from lift.backend.simulation.blocks import (
    FixedDemand,
    Fleet,
    GridConnection,
    PVSource,
    StationaryStorage,
)

from lift.backend.simulation.interfaces import (
    SimInputSettings,
    SimInputLocation,
    SimInputSubfleet,
    SimInputCharger,
    SimInputChargingInfrastructure,
    SimResults,
)

from lift.utils import safe_cache_data


@safe_cache_data
def simulate(
    settings: SimInputSettings,
    location: SimInputLocation,
    subfleets: dict[str, SimInputSubfleet],
    charging_infrastructure: SimInputChargingInfrastructure,
) -> SimResults:
    dem = FixedDemand.from_parameters(
        settings=settings,
        slp=location.slp,
        consumption_yrl_wh=location.consumption_yrl_wh,
    )
    fleet = Fleet.from_parameters(
        settings=settings,
        pwr_max_w=charging_infrastructure.pwr_max_w,
        subfleets=subfleets,
        chargers=charging_infrastructure.chargers,
    )
    pv = PVSource.from_parameters(settings=settings, pwr_wp=location.pv_wp, coordinates=location.coordinates)
    ess = StationaryStorage.from_parameters(
        settings=settings,
        capacity_wh=location.ess_wh,
    )
    grid = GridConnection.from_parameters(
        settings=settings,
        pwr_max_w=location.grid_w,
    )
    blocks_supply = (pv, ess, grid)

    blocks = [dem, fleet, *fleet.fleet_units.values(), *blocks_supply]

    # Simulate the vehicle fleet over the given datetime index.
    for idx in range(len(settings.dti)):
        # pass time of current timestep to all blocks
        for block in blocks:
            block.idx = idx

        # get the total demand from the fixed demand block
        pwr_demand_w = dem.demand_w

        # Update available power for dynamic load management
        if charging_infrastructure.pwr_max_w == np.inf:
            # calculate maximum power supply
            pwr_supply_max_w = sum(block.generation_max_w for block in blocks_supply)
            # define Fleet charging power limit for dynamic load management
            fleet.pwr_lim_w = pwr_supply_max_w - pwr_demand_w

        # add fleet demand to the total demand
        pwr_demand_w += fleet.demand_w

        # satisfy demand with supply blocks (order represents priority)
        for block in blocks_supply:
            pwr_demand_w = block.satisfy_demand(demand_w=pwr_demand_w)

    dist_km = fleet.distances

    return SimResults(
        energy_pv_pot_wh=pv.energy_pot_wh,
        energy_pv_curt_wh=grid.energy_curt_wh,
        energy_grid_buy_wh=grid.energy_buy_wh,
        energy_grid_sell_wh=grid.energy_sell_wh,
        pwr_grid_peak_w=grid.pwr_peak_w,
        energy_dem_site_wh=dem.energy_wh,
        energy_fleet_site_wh=fleet.energy_site_wh,
        energy_fleet_route_wh=fleet.energy_route_wh,
        dist_km=dist_km,
    )


if __name__ == "__main__":
    results = simulate(
        settings=SimInputSettings(
            period_sim=pd.Timedelta(days=365),
            start_sim=pd.Timestamp("2023-01-01 00:00", tz="Europe/Berlin"),
            freq_sim=pd.Timedelta(hours=1),
            freq_hours=1.0,
            dti=pd.date_range(
                start=pd.Timestamp("2023-01-01 00:00", tz="Europe/Berlin"),
                end=pd.Timestamp("2023-01-01 00:00", tz="Europe/Berlin") + pd.Timedelta(days=365),
                freq=pd.Timedelta(hours=1),
                tz="Europe/Berlin",
                inclusive="left",
            ),
        ),
        location=SimInputLocation(),
        subfleets={"hlt": SimInputSubfleet()},
        chargers={"ac": SimInputCharger()},
    )
    pass
