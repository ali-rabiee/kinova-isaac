"""W5 — the baseline conditions B0–B3 (``rehab.md`` §1.4).

============  ==========================  ==================================================
Condition     Rule                        Why it is in the comparison
============  ==========================  ==================================================
``B0``        Never COACH; all ASSESS     Defines the reference map (§12.2) and doubles as
                                          the cleanest possible baseline.
``B1``        Never WAIT                  Maximum contamination: every probe is as close to
                                          a prompt as the slot layout allows.
``B2``        Fixed washout ``w``         The population bet. ``w`` is *derived*, not
                                          hand-picked: by
                                          :func:`~vla_lab.rehab.scheduler.carryover_aware.population_washout_slots`,
                                          which runs the carryover-aware objective once
                                          against population parameters. That is what makes
                                          B4's schedule-only ablation reduce to B2 exactly
                                          under a point-mass posterior.
``B3``        Random / static placement   The same budget split as B2 placed independently of
                                          history: isolates "waiting at all" from "waiting at
                                          the right time".
============  ==========================  ==================================================

B0 has zero COACH slots **by definition** — it is the reference measurement, not a compared
condition, so the matched-``C`` requirement does not apply to it (§12.2). Every *compared*
condition (B1–B4 and the ablations) consumes an identical ``T`` and ``C``.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Set

from .. import ASSESS, WAIT
from ..trial import History
from .base import BlockBudget, Scheduler, SchedulerDecision, Slot


class NoPromptScheduler(Scheduler):
    """B0 — never COACH, always ASSESS. The reference block's scheduler."""

    name = "b0_no_prompt"
    condition = "no_prompt"

    def decide(self, history: History, slot: Slot) -> SchedulerDecision:
        if bool(slot.is_coach_slot):
            raise ValueError(
                "B0 was handed a COACH slot: the no-prompt reference block must contain zero "
                "COACH events by definition (rehab.md §12.2)"
            )
        return SchedulerDecision(action=ASSESS, target_id=int(slot.target_id), rationale="reference: no-prompt probe")


class ImmediateScheduler(Scheduler):
    """B1 — probe every free slot, including the one right after each COACH."""

    name = "b1_immediate"
    condition = "immediate"

    def decide_free_slot(self, history: History, slot: Slot) -> SchedulerDecision:
        since = history.trials_since_last_coach()
        return SchedulerDecision(
            action=ASSESS,
            target_id=int(slot.target_id),
            rationale=("probe immediately after COACH" if since == 0 else "probe (never waits)"),
            values={"slots_since_coach": float(-1 if since is None else since)},
        )


class FixedWashoutScheduler(Scheduler):
    """B2 — WAIT for a fixed ``w`` slots after each COACH, then ASSESS.

    ``w`` is a **population** constant: that is the bet the condition embodies. Use
    :func:`washout_slots` to derive it from population ``(lambda, beta, g)`` and the same
    contamination threshold the adaptive policy uses.
    """

    name = "b2_fixed_washout"
    condition = "fixed_washout"

    def __init__(self, *, w: int = 2, seed: int = 0) -> None:
        super().__init__(seed=seed)
        self.w = int(w)

    def decide_free_slot(self, history: History, slot: Slot) -> SchedulerDecision:
        since = history.trials_since_last_coach()
        if since is not None and since < self.w:
            return SchedulerDecision(
                action=WAIT,
                target_id=None,
                rationale=f"fixed washout: {since + 1}/{self.w} slots since COACH",
                values={"w": float(self.w), "slots_since_coach": float(since)},
            )
        return SchedulerDecision(
            action=ASSESS,
            target_id=int(slot.target_id),
            rationale="washout complete" if since is not None else "no COACH yet",
            values={"w": float(self.w), "slots_since_coach": float(-1 if since is None else since)},
        )

    def describe(self) -> Dict[str, Any]:
        return {**super().describe(), "w": int(self.w)}


class RandomStaticScheduler(Scheduler):
    """B3 — the same number of WAITs as B2, placed independently of history.

    Drawn once at :meth:`reset` from the seed, so the schedule is *static* (a pre-specified
    pattern) rather than reactive: the control for "does waiting help, or does waiting *at the
    right time* help".
    """

    name = "b3_random_static"
    condition = "random_static"

    def __init__(self, *, n_wait: int = 0, seed: int = 0) -> None:
        super().__init__(seed=seed)
        self.n_wait = int(n_wait)
        self._wait_slots: Set[int] = set()

    def reset(self, budget: BlockBudget) -> None:
        super().reset(budget)
        rng = random.Random(self.seed * 100003 + int(budget.slots_total))
        free = [i for i in range(int(budget.slots_total)) if i not in set(budget.coach_slots)]
        k = int(min(max(0, self.n_wait), len(free)))
        self._wait_slots = set(rng.sample(free, k)) if k else set()

    def decide_free_slot(self, history: History, slot: Slot) -> SchedulerDecision:
        if int(slot.slot_idx) in self._wait_slots:
            return SchedulerDecision(
                action=WAIT, target_id=None, rationale="static schedule: pre-drawn WAIT slot"
            )
        return SchedulerDecision(action=ASSESS, target_id=int(slot.target_id), rationale="static schedule: probe")

    def describe(self) -> Dict[str, Any]:
        return {**super().describe(), "n_wait": int(self.n_wait), "wait_slots": sorted(self._wait_slots)}


__all__ = [
    "NoPromptScheduler",
    "ImmediateScheduler",
    "FixedWashoutScheduler",
    "RandomStaticScheduler",
]
