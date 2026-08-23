"""The transparent comparators. Every one is something a person could execute with a
stopwatch and a rule, which is the point: an adaptive policy has to beat rules a practitioner
would actually use, not a straw man.

All of them consume the identical budget, the identical demonstration schedule, and the
identical scene sequence supplied by the protocol.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from .. import COUNTER, PROBE, WAIT
from ..estimand import METHOD_POOLED, METHOD_PSYCHOMETRIC
from .base import BlockBudget, History, Scheduler, SchedulerDecision, Slot


class NoCoachScheduler(Scheduler):
    """**B0 -- no-coach reference.** Never demonstrates; every slot probes.

    This is simultaneously the cleanest attainable condition and the source of the reference
    map every other condition's error is measured against. It is not a competitor.
    """

    name = "no_coach"
    estimator = METHOD_PSYCHOMETRIC

    def decide(self, history: History, slot: Slot) -> SchedulerDecision:
        return SchedulerDecision(PROBE, slot.scene_id, {"reason": "reference block: always probe"})


class MemorylessScheduler(Scheduler):
    """**B1 -- the Memoryless VLA.** Probes immediately, believes whatever it is told.

    Maximum contamination and a naive per-scene estimator: it neither waits, nor counter-
    proposes, nor models that it has just spent three episodes showing the supervisor a
    strategy. This is the automation-bias failure mode stated as a policy, and it is the
    condition the proposed method has to beat by a wide margin for the problem to be real.
    """

    name = "memoryless"
    estimator = METHOD_POOLED

    def decide_free_slot(self, history: History, slot: Slot) -> SchedulerDecision:
        return SchedulerDecision(PROBE, slot.scene_id, {"reason": "probe immediately; no carryover model"})


class FixedWashoutScheduler(Scheduler):
    """**B2 -- fixed washout.** Wait ``w`` free slots after each demonstration, then probe.

    Current practice, and the strongest baseline we expect. ``w`` is a population constant: the
    whole question is whether a constant can serve people whose decay rates differ.
    """

    name = "fixed_washout"
    estimator = METHOD_PSYCHOMETRIC

    def __init__(self, *, w: int = 2, seed: int = 0) -> None:
        super().__init__(seed=seed)
        self.w = int(w)

    def decide_free_slot(self, history: History, slot: Slot) -> SchedulerDecision:
        waited = history.n_since_last_coach()
        if waited < self.w and slot.free_remaining > 1:
            return SchedulerDecision(WAIT, slot.scene_id, {"reason": f"washout {waited}/{self.w}", "w": self.w})
        return SchedulerDecision(PROBE, slot.scene_id, {"reason": "washout satisfied", "w": self.w, "waited": waited})

    def describe(self) -> Dict[str, Any]:
        d = super().describe()
        d["w"] = self.w
        return d


class RandomStaticScheduler(Scheduler):
    """**B3 -- random / static.** Spend the same number of waits, placed without regard to
    history. Tests whether adaptivity beats an arbitrary but budget-matched allocation -- if a
    policy's advantage survives against B2 but not against B3, the advantage was the *amount*
    of waiting, not its placement.
    """

    name = "random_static"
    estimator = METHOD_PSYCHOMETRIC

    def __init__(self, *, n_wait: int = 0, seed: int = 0) -> None:
        super().__init__(seed=seed)
        self.n_wait = int(n_wait)
        self._wait_slots: set = set()

    def reset(self, budget: BlockBudget) -> None:
        super().reset(budget)
        rng = random.Random(self.seed)
        free = [i for i in range(budget.n_slots) if i not in set(budget.coach_slots)]
        k = min(self.n_wait, max(0, len(free) - 1))
        self._wait_slots = set(rng.sample(free, k)) if k else set()

    def decide_free_slot(self, history: History, slot: Slot) -> SchedulerDecision:
        if slot.index in self._wait_slots:
            return SchedulerDecision(WAIT, slot.scene_id, {"reason": "pre-drawn wait slot"})
        return SchedulerDecision(PROBE, slot.scene_id, {"reason": "pre-drawn probe slot"})

    def describe(self) -> Dict[str, Any]:
        d = super().describe()
        d["n_wait"] = self.n_wait
        return d


class AlwaysCounterScheduler(Scheduler):
    """**B4 -- always counter-propose.** Never waits; every probe names the alternative.

    The upper bound on de-biasing-by-asking and the upper bound on interaction burden. It is
    here because "just always offer the other option" is the obvious thing a practitioner
    would try, and a proposed policy that cannot beat it *on burden* has not earned its
    machinery -- even if it matches it on error.
    """

    name = "always_counter"
    estimator = METHOD_PSYCHOMETRIC

    def decide_free_slot(self, history: History, slot: Slot) -> SchedulerDecision:
        return SchedulerDecision(COUNTER, slot.scene_id, {"reason": "always counter-propose"})


__all__ = [
    "NoCoachScheduler",
    "MemorylessScheduler",
    "FixedWashoutScheduler",
    "RandomStaticScheduler",
    "AlwaysCounterScheduler",
]
