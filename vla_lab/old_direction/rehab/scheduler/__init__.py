"""W5 — the Phase 0 schedulers: one protocol, the B0–B3 baselines, B4 and its ablations.

The conditions of ``rehab.md`` §1.4, by name:

============================  =========================================================
``no_prompt``                 B0 — reference / retest blocks (zero COACH by definition)
``immediate``                 B1 — maximum contamination
``fixed_washout``             B2 — population washout constant
``random_static``             B3 — same budget split as B2, history-independent placement
``carryover_aware``           B4 — proposed
``ablation_schedule_only``    B4 minus the corrected estimator
``ablation_estimator_only``   B4 minus the adaptive schedule
============================  =========================================================
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from ..carryover import CarryoverConfig
from ..workspace import TargetGrid
from .base import BlockBudget, DeltaModel, Scheduler, SchedulerDecision, Slot
from .baselines import (
    FixedWashoutScheduler,
    ImmediateScheduler,
    NoPromptScheduler,
    RandomStaticScheduler,
)
from .carryover_aware import (
    CarryoverAwareConfig,
    CarryoverAwareScheduler,
    population_washout_slots,
)

CONDITION_NO_PROMPT = "no_prompt"
CONDITION_IMMEDIATE = "immediate"
CONDITION_FIXED_WASHOUT = "fixed_washout"
CONDITION_RANDOM_STATIC = "random_static"
CONDITION_CARRYOVER_AWARE = "carryover_aware"
CONDITION_ABLATION_SCHEDULE = "ablation_schedule_only"
CONDITION_ABLATION_ESTIMATOR = "ablation_estimator_only"

#: Every *compared* condition (matched T and C). B0 is a reference measurement, not a
#: compared condition, so it is not in this tuple (§12.2).
COMPARED_CONDITIONS = (
    CONDITION_IMMEDIATE,
    CONDITION_FIXED_WASHOUT,
    CONDITION_RANDOM_STATIC,
    CONDITION_CARRYOVER_AWARE,
    CONDITION_ABLATION_SCHEDULE,
    CONDITION_ABLATION_ESTIMATOR,
)

ALL_CONDITIONS = (CONDITION_NO_PROMPT,) + COMPARED_CONDITIONS

#: Conditions whose analysis must use the carryover-corrected estimator.
CORRECTED_CONDITIONS = (CONDITION_CARRYOVER_AWARE, CONDITION_ABLATION_ESTIMATOR)


def make_scheduler(
    condition: str,
    *,
    grid: TargetGrid,
    nonpreferred_side: str,
    seed: int = 0,
    fixed_w: int = 2,
    n_wait: int = 0,
    carryover_cfg: Optional[CarryoverConfig] = None,
    delta: Optional[DeltaModel] = None,
    cfg: Optional[CarryoverAwareConfig] = None,
) -> Scheduler:
    """Build the scheduler for a named condition.

    ``fixed_w`` should come from
    :func:`~vla_lab.rehab.scheduler.carryover_aware.population_washout_slots` so that B2 and
    the estimator-only ablation share the derivation the adaptive policy would make from
    population parameters (see :mod:`vla_lab.rehab.protocol`).
    """

    name = str(condition)
    if name == CONDITION_NO_PROMPT:
        return NoPromptScheduler(seed=seed)
    if name == CONDITION_IMMEDIATE:
        return ImmediateScheduler(seed=seed)
    if name == CONDITION_FIXED_WASHOUT:
        return FixedWashoutScheduler(w=int(fixed_w), seed=seed)
    if name == CONDITION_RANDOM_STATIC:
        return RandomStaticScheduler(n_wait=int(n_wait), seed=seed)
    if name in (CONDITION_CARRYOVER_AWARE, CONDITION_ABLATION_SCHEDULE, CONDITION_ABLATION_ESTIMATOR):
        base_cfg = cfg or CarryoverAwareConfig()
        if name == CONDITION_ABLATION_ESTIMATOR:
            base_cfg = CarryoverAwareConfig(
                max_w=base_cfg.max_w,
                top_k=base_cfg.top_k,
                refit_every=base_cfg.refit_every,
                fixed_w=int(fixed_w),
                effort_strength=dict(base_cfg.effort_strength),
            )
        return CarryoverAwareScheduler(
            grid,
            nonpreferred_side,
            adaptive_schedule=(name != CONDITION_ABLATION_ESTIMATOR),
            estimator_correction=(name != CONDITION_ABLATION_SCHEDULE),
            carryover_cfg=carryover_cfg,
            delta=delta,
            cfg=base_cfg,
            seed=seed,
        )
    raise ValueError(f"unknown condition {condition!r}; known: {ALL_CONDITIONS}")


__all__ = [
    "Scheduler",
    "SchedulerDecision",
    "Slot",
    "BlockBudget",
    "DeltaModel",
    "NoPromptScheduler",
    "ImmediateScheduler",
    "FixedWashoutScheduler",
    "RandomStaticScheduler",
    "CarryoverAwareScheduler",
    "CarryoverAwareConfig",
    "population_washout_slots",
    "make_scheduler",
    "CONDITION_NO_PROMPT",
    "CONDITION_IMMEDIATE",
    "CONDITION_FIXED_WASHOUT",
    "CONDITION_RANDOM_STATIC",
    "CONDITION_CARRYOVER_AWARE",
    "CONDITION_ABLATION_SCHEDULE",
    "CONDITION_ABLATION_ESTIMATOR",
    "COMPARED_CONDITIONS",
    "ALL_CONDITIONS",
    "CORRECTED_CONDITIONS",
]
