"""Condition registry: the six compared conditions and B5's three ablations.

One place decides what a condition *is* -- its scheduler, its reported estimator, and its
display name -- so that the session runner, the analysis, and the paper's tables cannot drift
apart. Adding a condition here is the only thing needed to put it in every figure.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from ..carryover import CarryoverConfig
from ..estimand import METHOD_CORRECTED, METHOD_POOLED, METHOD_PSYCHOMETRIC, PsychometricConfig
from ..scenes import SceneGrid
from .base import BlockBudget, DeltaModel, History, HistoryRecord, Scheduler, SchedulerDecision, Slot
from .baselines import (
    AlwaysCounterScheduler,
    FixedWashoutScheduler,
    MemorylessScheduler,
    NoCoachScheduler,
    RandomStaticScheduler,
)
from .carryover_aware import CarryoverAwareConfig, CarryoverAwareScheduler, population_washout_slots

CONDITION_NO_COACH = "no_coach"
CONDITION_MEMORYLESS = "memoryless"
CONDITION_FIXED_WASHOUT = "fixed_washout"
CONDITION_RANDOM_STATIC = "random_static"
CONDITION_ALWAYS_COUNTER = "always_counter"
CONDITION_CARRYOVER_AWARE = "carryover_aware"

ABLATION_SCHEDULE_ONLY = "ablation_schedule_only"
ABLATION_ESTIMATOR_ONLY = "ablation_estimator_only"
ABLATION_COUNTER_ONLY = "ablation_counter_only"

#: Conditions that are compared to each other in the primary table, in report order.
COMPARED_CONDITIONS = (
    CONDITION_MEMORYLESS,
    CONDITION_FIXED_WASHOUT,
    CONDITION_RANDOM_STATIC,
    CONDITION_ALWAYS_COUNTER,
    CONDITION_CARRYOVER_AWARE,
)
ABLATIONS = (ABLATION_SCHEDULE_ONLY, ABLATION_ESTIMATOR_ONLY, ABLATION_COUNTER_ONLY)
ALL_CONDITIONS = (CONDITION_NO_COACH,) + COMPARED_CONDITIONS + ABLATIONS

#: Which baseline the pre-specified primary contrast runs against. Fixed before any data.
PRIMARY_COMPARATOR = CONDITION_FIXED_WASHOUT

DISPLAY_NAMES: Dict[str, str] = {
    CONDITION_NO_COACH: "B0 No-coach reference",
    CONDITION_MEMORYLESS: "B1 Memoryless VLA",
    CONDITION_FIXED_WASHOUT: "B2 Fixed washout",
    CONDITION_RANDOM_STATIC: "B3 Random / static",
    CONDITION_ALWAYS_COUNTER: "B4 Always counter-propose",
    CONDITION_CARRYOVER_AWARE: "B5 Carryover-aware VLA",
    ABLATION_SCHEDULE_ONLY: "  ablation: schedule only",
    ABLATION_ESTIMATOR_ONLY: "  ablation: estimator only",
    ABLATION_COUNTER_ONLY: "  ablation: counter only",
}

_ABLATION_SWITCHES: Dict[str, Dict[str, bool]] = {
    ABLATION_SCHEDULE_ONLY: {"adaptive_schedule": True, "estimator_correction": False, "counter_enabled": False},
    ABLATION_ESTIMATOR_ONLY: {"adaptive_schedule": False, "estimator_correction": True, "counter_enabled": False},
    ABLATION_COUNTER_ONLY: {"adaptive_schedule": False, "estimator_correction": False, "counter_enabled": True},
}


def estimator_for(condition: str) -> str:
    """Which ``pi*`` estimator a condition's *reported* estimate uses."""
    if condition == CONDITION_MEMORYLESS:
        return METHOD_POOLED
    if condition == CONDITION_CARRYOVER_AWARE:
        return METHOD_CORRECTED
    if condition in _ABLATION_SWITCHES:
        return METHOD_CORRECTED if _ABLATION_SWITCHES[condition]["estimator_correction"] else METHOD_PSYCHOMETRIC
    return METHOD_PSYCHOMETRIC


def build_scheduler(
    condition: str,
    grid: SceneGrid,
    *,
    carryover_cfg: Optional[CarryoverConfig] = None,
    delta_model: Optional[DeltaModel] = None,
    psych_cfg: Optional[PsychometricConfig] = None,
    aware_cfg: Optional[CarryoverAwareConfig] = None,
    washout_w: Optional[int] = None,
    n_wait: Optional[int] = None,
    seed: int = 0,
    log_prior: Optional[Any] = None,
) -> Scheduler:
    """Instantiate one condition.

    ``washout_w`` defaults to the population washout implied by the *same priors* the adaptive
    policy starts from, so B2 is the strongest fixed rule those priors support rather than a
    weak one chosen to be beaten. ``n_wait`` defaults to matching B2's total waiting, so B3
    differs from B2 only in *placement*.
    """
    cfg = carryover_cfg or CarryoverConfig()
    dm = delta_model or DeltaModel.from_config(cfg)
    w = int(washout_w) if washout_w is not None else population_washout_slots(cfg=cfg, delta_model=dm)

    if condition == CONDITION_NO_COACH:
        return NoCoachScheduler(seed=seed)
    if condition == CONDITION_MEMORYLESS:
        return MemorylessScheduler(seed=seed)
    if condition == CONDITION_FIXED_WASHOUT:
        return FixedWashoutScheduler(w=w, seed=seed)
    if condition == CONDITION_RANDOM_STATIC:
        return RandomStaticScheduler(n_wait=int(n_wait) if n_wait is not None else w, seed=seed)
    if condition == CONDITION_ALWAYS_COUNTER:
        return AlwaysCounterScheduler(seed=seed)
    if condition == CONDITION_CARRYOVER_AWARE:
        if aware_cfg is None and delta_model is not None:
            aware_cfg = CarryoverAwareConfig(
                counter_time_ratio=max(0.0, (dm.counter_s - dm.probe_s) / max(dm.probe_s, 1e-6))
            )
        return CarryoverAwareScheduler(
            grid, cfg=aware_cfg or CarryoverAwareConfig(), carryover_cfg=cfg, delta_model=dm,
            psych_cfg=psych_cfg, seed=seed, log_prior=log_prior,
        )
    if condition in _ABLATION_SWITCHES:
        base = aware_cfg.to_dict() if aware_cfg else CarryoverAwareConfig().to_dict()
        base.update(_ABLATION_SWITCHES[condition])
        return CarryoverAwareScheduler(
            grid, cfg=CarryoverAwareConfig(**base), carryover_cfg=cfg, delta_model=dm,
            psych_cfg=psych_cfg, seed=seed, name=condition, log_prior=log_prior,
        )
    raise KeyError(f"unknown condition {condition!r}; known: {list(ALL_CONDITIONS)}")


__all__ = [
    "CONDITION_NO_COACH",
    "CONDITION_MEMORYLESS",
    "CONDITION_FIXED_WASHOUT",
    "CONDITION_RANDOM_STATIC",
    "CONDITION_ALWAYS_COUNTER",
    "CONDITION_CARRYOVER_AWARE",
    "ABLATION_SCHEDULE_ONLY",
    "ABLATION_ESTIMATOR_ONLY",
    "ABLATION_COUNTER_ONLY",
    "COMPARED_CONDITIONS",
    "ABLATIONS",
    "ALL_CONDITIONS",
    "PRIMARY_COMPARATOR",
    "DISPLAY_NAMES",
    "estimator_for",
    "build_scheduler",
    "population_washout_slots",
    "Scheduler",
    "SchedulerDecision",
    "Slot",
    "BlockBudget",
    "DeltaModel",
    "History",
    "HistoryRecord",
    "CarryoverAwareConfig",
    "CarryoverAwareScheduler",
]
