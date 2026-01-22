import abc
from dataclasses import asdict, dataclass
from time import time

import numpy as np
import pandas as pd

from lift.backend.scenario import ExistExpansionValue, ScenarioResult, SingleScenario
from lift.utils import safe_cache_data


@dataclass(frozen=True)
class ComparisonInput(abc.ABC):
    @property
    @abc.abstractmethod
    # do not use asdict as this also converts ExistExpansionValue to dicts
    def to_dict(self): ...

    def get_df(self, block_name: str) -> pd.DataFrame:
        block_dict = self.to_dict
        if "subblocks" in block_dict:
            block_dict["subblocks"] = list(block_dict["subblocks"].keys())

        df = pd.concat(
            [
                pd.DataFrame(
                    data=[
                        ([v.baseline, v.expansion] if isinstance(v, ExistExpansionValue) else [v, v])
                        for v in block_dict.values()
                    ],
                    columns=["baseline", "expansion"],
                    index=pd.MultiIndex.from_tuples(
                        tuples=[(block_name, k) for k in block_dict.keys()], names=["block", "parameter"]
                    ),
                )
            ]
            + [v.get_df(k) for k, v in getattr(self, "subblocks", {}).items()]
        )

        return df


@dataclass(frozen=True)
class ComparisonSettings(ComparisonInput):
    latitude: float
    longitude: float
    wacc: float
    period_eco: int
    sim_start: pd.Timestamp
    sim_duration: pd.Timedelta
    sim_freq: pd.Timedelta

    @property
    def to_dict(self):
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "wacc": self.wacc,
            "period_eco": self.period_eco,
            "sim_start": self.sim_start,
            "sim_duration": self.sim_duration,
            "sim_freq": self.sim_freq,
        }


@dataclass(frozen=True)
class ComparisonFix(ComparisonInput):
    capex_initial: ExistExpansionValue
    capem_initial: ExistExpansionValue

    @property
    def to_dict(self):
        return {
            "capex_initial": self.capex_initial,
            "capem_initial": self.capem_initial,
        }


@dataclass(frozen=True)
class ComparisonFixedDemand(ComparisonInput):
    slp: str
    e_yrl: float

    @property
    def to_dict(self):
        return {
            "slp": self.slp,
            "e_yrl": self.e_yrl,
        }


@dataclass(frozen=True)
class ComparisonGrid(ComparisonInput):
    ls: int
    capacity: ExistExpansionValue
    capex_spec: float
    capem_spec: float
    opex_spec_buy: float
    opex_spec_sell: float
    opex_spec_peak: float
    opem_spec: float

    @property
    def to_dict(self):
        return {
            "ls": self.ls,
            "capacity": self.capacity,
            "capex_spec": self.capex_spec,
            "capem_spec": self.capem_spec,
            "opex_spec_buy": self.opex_spec_buy,
            "opex_spec_sell": self.opex_spec_sell,
            "opex_spec_peak": self.opex_spec_peak,
            "opem_spec": self.opem_spec,
        }


@dataclass(frozen=True)
class ComparisonPV(ComparisonInput):
    ls: int
    capacity: ExistExpansionValue
    capex_spec: float
    capem_spec: float
    opex_spec: float
    opem_spec: float

    @property
    def to_dict(self):
        return {
            "ls": self.ls,
            "capacity": self.capacity,
            "capex_spec": self.capex_spec,
            "capem_spec": self.capem_spec,
            "opex_spec": self.opex_spec,
            "opem_spec": self.opem_spec,
        }


@dataclass(frozen=True)
class ComparisonESS(ComparisonInput):
    ls: int
    capacity: ExistExpansionValue
    capex_spec: float
    capem_spec: float
    opex_spec: float
    opem_spec: float
    c_rate_max: float
    soc_init: float

    @property
    def to_dict(self):
        return {
            "ls": self.ls,
            "capacity": self.capacity,
            "capex_spec": self.capex_spec,
            "capem_spec": self.capem_spec,
            "opex_spec": self.opex_spec,
            "opem_spec": self.opem_spec,
            "c_rate_max": self.c_rate_max,
            "soc_init": self.soc_init,
        }


@dataclass(frozen=True)
class ComparisonSubFleet(ComparisonInput):
    name: str
    ls: int
    num_bev: ExistExpansionValue
    num_icev: ExistExpansionValue
    capacity: float
    charger: str
    p_max: float
    capex_per_unit_bev: float
    capex_per_unit_icev: float
    capem_per_unit_bev: float
    capem_per_unit_icev: float
    mntex_spec_bev: float
    mntex_spec_icev: float
    toll_frac: float
    toll_spec_bev: float
    toll_spec_icev: float
    consumption_spec_icev: float
    soc_init: float

    @property
    def to_dict(self):
        return {
            "name": self.name,
            "ls": self.ls,
            "num_bev": self.num_bev,
            "num_icev": self.num_icev,
            "capacity": self.capacity,
            "charger": self.charger,
            "p_max": self.p_max,
            "capex_per_unit_bev": self.capex_per_unit_bev,
            "capex_per_unit_icev": self.capex_per_unit_icev,
            "capem_per_unit_bev": self.capem_per_unit_bev,
            "capem_per_unit_icev": self.capem_per_unit_icev,
            "mntex_spec_bev": self.mntex_spec_bev,
            "mntex_spec_icev": self.mntex_spec_icev,
            "toll_frac": self.toll_frac,
            "toll_spec_bev": self.toll_spec_bev,
            "toll_spec_icev": self.toll_spec_icev,
            "consumption_spec_icev": self.consumption_spec_icev,
            "soc_init": self.soc_init,
        }


@dataclass(frozen=True)
class ComparisonFleet(ComparisonInput):
    subblocks: dict[str, ComparisonSubFleet]
    opex_spec_fuel: float
    opem_spec_fuel: float
    opex_spec_onroute_charging: float
    opem_spec_onroute_charging: float

    @property
    def to_dict(self):
        return {
            "subblocks": self.subblocks,
            "opex_spec_fuel": self.opex_spec_fuel,
            "opem_spec_fuel": self.opem_spec_fuel,
            "opex_spec_onroute_charging": self.opex_spec_onroute_charging,
            "opem_spec_onroute_charging": self.opem_spec_onroute_charging,
        }


@dataclass(frozen=True)
class ComparisonChargerType(ComparisonInput):
    name: str
    num: ExistExpansionValue
    p_max: float
    capex_per_unit: float
    capem_per_unit: float
    ls: int

    @property
    def to_dict(self):
        return {
            "name": self.name,
            "num": self.num,
            "p_max": self.p_max,
            "capex_per_unit": self.capex_per_unit,
            "capem_per_unit": self.capem_per_unit,
            "ls": self.ls,
        }


@dataclass(frozen=True)
class ComparisonChargingInfrastructure(ComparisonInput):
    subblocks: dict[str, ComparisonChargerType]
    p_lm_max: ExistExpansionValue

    @property
    def to_dict(self):
        return {
            "subblocks": self.subblocks,
            "p_lm_max": self.p_lm_max,
        }


@dataclass(frozen=True)
class ComparisonScenario(ComparisonInput):
    settings: ComparisonSettings
    fix: ComparisonFix
    dem: ComparisonFixedDemand
    grid: ComparisonGrid
    pv: ComparisonPV
    ess: ComparisonESS
    fleet: ComparisonFleet
    cis: ComparisonChargingInfrastructure

    @property
    def to_dict(self):
        return {
            "fix": self.fix,
            "dem": self.dem,
            "grid": self.grid,
            "pv": self.pv,
            "ess": self.ess,
            "fleet": self.fleet,
            "cis": self.cis,
        }

    @property
    def df(self) -> pd.DataFrame:
        return pd.concat(
            [self.settings.get_df("scn")] + [v.get_df(k) for k, v in self.to_dict.items()], axis=0, sort=False
        )


@dataclass
class ComparisonResult:
    baseline: "ScenarioResult"
    expansion: "ScenarioResult"

    @property
    def npc_delta(self) -> float:
        return self.baseline.totex_dis.sum() - self.expansion.totex_dis.sum()

    @staticmethod
    def get_payback_period_yrs(diff) -> float | None:
        # np array containing diff indices just before sign change
        idx = np.flatnonzero(np.diff(np.sign(diff)))

        if idx.size == 0 or diff[0] > 0:
            return None  # No intersection
        else:
            # +1 for before switch --> after switch, +1 for index --> year number
            return idx[0] + 2

    @property
    def payback_period_yrs(self) -> float | None:
        diff = np.cumsum(self.baseline.totex_dis) - np.cumsum(self.expansion.totex_dis)
        return self.get_payback_period_yrs(diff)

    @property
    def co2_delta(self) -> float:
        return self.baseline.totem.sum() - self.expansion.totem.sum()

    @property
    def payback_period_co2_yrs(self) -> float | None:
        diff = np.cumsum(self.baseline.totem) - np.cumsum(self.expansion.totem)
        return self.get_payback_period_yrs(diff)

    @property
    def delta(self) -> dict[str, float]:
        return {
            "costs": self.npc_delta,
            "emissions": self.co2_delta,
        }

    @property
    def payback_period(self) -> dict[str, float | None]:
        return {
            "costs": self.payback_period_yrs,
            "emissions": self.payback_period_co2_yrs,
        }


@safe_cache_data
def run_comparison(comp_scn: ComparisonScenario) -> ComparisonResult:
    # start time tracking
    start_time = time()

    scn_dict = asdict(comp_scn)
    scn_dict.pop("settings")

    results = {
        phase: SingleScenario.from_comparison(comp_obj=comp_scn, phase=phase).simulate()
        for phase in ["baseline", "expansion"]
    }

    # stop time tracking
    print(f"{pd.Timestamp.now().isoformat()} - Backend calculation completed in {time() - start_time:.2f} seconds.")

    return ComparisonResult(**results)
