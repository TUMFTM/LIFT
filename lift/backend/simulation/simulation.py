"""Time-series energy and fleet simulation.

Purpose:
- Simulate site demand, fleet charging, PV generation, stationary storage, and grid exchange
  over a discrete time index; optionally scale results to one year.

Relationships:
- Consumes typed inputs from `.interfaces` and composes block models from `.blocks`.
- Used by `lift.backend.evaluation.evaluate` to produce simulation KPIs for economics.
- Decorated with `safe_cache_data` to memoize deterministic runs.

Key Logic / Formulations:
- At each timestep t in settings.dti:
  1) Site demand (FixedDemand) yields P_dem(t).
  2) Fleet sets charging demand P_fleet(t) under a dynamic limit P_lim(t).
     If grid connection is effectively unbounded (pwr_max_w = inf), then
       P_lim(t) = sum(PV_max(t), ESS_max(t), Grid_max(t)) - P_dem(t).
  3) Supply blocks satisfy total demand in priority order (PV → ESS → Grid).
- Aggregates energies (Wh) and grid peak power over the simulation horizon; optionally
  scales integrated quantities by sim2year = 365 days / period_sim.
"""

import numpy as np
import pandas as pd

from .blocks import (
    FixedDemand,
    Fleet,
    GridConnection,
    PVSource,
    StationaryStorage,
)

from .interfaces import (
    SimInputSettings,
    SimInputLocation,
    SimInputSubfleet,
    SimInputChargingInfrastructure,
    SimResults,
)

from lift.backend.utils import safe_cache_data


@safe_cache_data
def simulate(
    settings: SimInputSettings,
    location: SimInputLocation,
    subfleets: dict[str, SimInputSubfleet],
    charging_infrastructure: SimInputChargingInfrastructure,
    scale2year: bool = True,
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
        # Note (design rationale):
        # - Current behavior sets a dynamic fleet power limit ONLY when the charging infrastructure
        #   has no explicit cap (pwr_max_w == inf).
        # - When the infrastructure has a finite cap, we rely on downstream block logic (incl. Grid)
        #   and domain constraints to handle any exceedance, which keeps the fleet logic simpler and
        #   makes violations explicit via domain errors where applicable.
        # Potential improvement: Consider applying a unified limit for all cases, e.g.:
        # - Apply a unified limit for all cases, e.g.:
        #     fleet.pwr_lim_w = min(charging_infrastructure.pwr_max_w,
        #                           sum(block.generation_max_w for block in blocks_supply) - pwr_demand_w)
        if charging_infrastructure.pwr_max_w == np.inf:
            fleet.pwr_lim_w = sum(block.generation_max_w for block in blocks_supply) - pwr_demand_w

        # add fleet demand to the total demand
        pwr_demand_w += fleet.demand_w

        # satisfy demand with supply blocks (order represents priority)
        for block in blocks_supply:
            pwr_demand_w = block.satisfy_demand(demand_w=pwr_demand_w)

    # calculate conversion factor to one year
    sim2year = pd.Timedelta(days=365) / settings.period_sim if scale2year == True else 1

    # scale results to one year
    return SimResults(
        energy_pv_pot_wh=pv.energy_pot_wh * sim2year,
        energy_pv_curt_wh=grid.energy_curt_wh * sim2year,
        energy_grid_buy_wh=grid.energy_buy_wh * sim2year,
        energy_grid_sell_wh=grid.energy_sell_wh * sim2year,
        pwr_grid_peak_w=grid.pwr_peak_w,
        energy_dem_site_wh=dem.energy_wh * sim2year,
        energy_fleet_site_wh=fleet.energy_site_wh * sim2year,
        energy_fleet_route_wh=fleet.energy_route_wh * sim2year,
        dist_km=fleet.get_distances(scale_factor=sim2year),
    )
