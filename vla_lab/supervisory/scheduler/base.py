"""Slots, decisions, budgets, and the scheduler interface.

The protocol -- not the policy -- fixes what is *manipulated*: how many demonstrations there
are, where in the session they fall, in which direction, and which scene each free slot
presents. Conditions differ **only** in what the policy does with the free slots: probe now,
probe with a counter-proposal, or wait. That restriction is a design commitment rather than a
convenience. A policy allowed to also choose its own demonstration schedule or its own scenes
would be compared against baselines that were not, and any advantage it showed would be
un-attributable.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .. import COACH, COUNTER, PROBE, WAIT
from ..carryover import DECAY_TRIALS, CarryoverConfig


@dataclass
class Slot:
    """One interaction slot as the protocol hands it to the scheduler."""

    index: int
    scene_id: int
    is_coach_slot: bool = False
    coach_direction: int = 1
    coach_strength: float = 1.0
    #: Free slots remaining in this inter-demonstration gap, including this one.
    free_remaining: int = 0
    #: Fraction of the session already spent, for the drift term.
    session_progress: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SchedulerDecision:
    """What the policy chose, and everything needed to reconstruct why.

    ``rationale`` is written to the session log every slot. An adaptive policy whose reasoning
    cannot be recovered from the log is not reviewable, and we treat that as a requirement.
    """

    action: str
    scene_id: int
    rationale: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"action": self.action, "scene_id": int(self.scene_id), "rationale": dict(self.rationale)}


@dataclass
class BlockBudget:
    """The matched budget. Identical across every compared condition, by construction.

    ``coach_slots`` and ``coach_directions`` are drawn once per supervisor and reused by every
    condition, so ``T``, ``C``, the demonstration positions, the demonstration directions, and
    the presented scenes are equal *by construction* rather than by bookkeeping -- which is
    what makes "realized budget" a manipulation check that can only fail if something is
    genuinely broken.
    """

    n_slots: int
    coach_slots: Tuple[int, ...]
    coach_directions: Tuple[int, ...]
    scene_sequence: Tuple[int, ...]
    coach_scene_sequence: Tuple[int, ...] = ()
    coach_strength: float = 1.0

    def __post_init__(self) -> None:
        if len(self.coach_slots) != len(self.coach_directions):
            raise ValueError("coach_slots and coach_directions must have equal length")
        if len(self.scene_sequence) < self.n_slots:
            raise ValueError("scene_sequence shorter than n_slots")

    @property
    def n_coach(self) -> int:
        return len(self.coach_slots)

    @property
    def n_free(self) -> int:
        return int(self.n_slots) - self.n_coach

    def direction_at(self, slot_index: int) -> int:
        for s, d in zip(self.coach_slots, self.coach_directions):
            if s == slot_index:
                return int(d)
        return 1

    def last_coach_direction(self, slot_index: int) -> int:
        """Direction of the most recent demonstration at or before ``slot_index``."""
        best = 1
        for s, d in zip(self.coach_slots, self.coach_directions):
            if s <= slot_index:
                best = int(d)
        return best

    def free_remaining_at(self, slot_index: int) -> int:
        """Free slots from ``slot_index`` up to (not including) the next demonstration."""
        nxt = min([s for s in self.coach_slots if s > slot_index], default=int(self.n_slots))
        return sum(1 for i in range(slot_index, nxt) if i not in self.coach_slots)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_slots": int(self.n_slots),
            "n_coach": self.n_coach,
            "n_free": self.n_free,
            "coach_slots": list(self.coach_slots),
            "coach_directions": list(self.coach_directions),
            "scene_sequence": list(self.scene_sequence),
            "coach_scene_sequence": list(self.coach_scene_sequence),
            "coach_strength": float(self.coach_strength),
        }


@dataclass
class DeltaModel:
    """How much decay each kind of slot buys, in the configured decay units.

    Under **time** decay a slot decays the residue in proportion to how long it took, so a WAIT
    filler is worth exactly its dwell. Under **trials** decay it decays in proportion to how
    much it interfered, and a neutral filler interferes less than a real interaction
    (``wait_interference < 1``) -- which is precisely why the two parameterisations imply
    different WAIT semantics and why the study fits both rather than assuming one.
    """

    decay_mode: str
    time_unit_s: float = 30.0
    probe_s: float = 42.0
    counter_s: float = 58.0
    wait_s: float = 20.0
    coach_s: float = 45.0
    wait_interference: float = 0.55

    def duration_s(self, action: str) -> float:
        return {COACH: self.coach_s, PROBE: self.probe_s, COUNTER: self.counter_s, WAIT: self.wait_s}[str(action)]

    def for_action(self, action: str) -> float:
        if str(self.decay_mode) == DECAY_TRIALS:
            return float(self.wait_interference) if str(action) == WAIT else 1.0
        return float(self.duration_s(action)) / max(1e-9, float(self.time_unit_s))

    @classmethod
    def from_config(cls, cfg: CarryoverConfig, **kw: Any) -> "DeltaModel":
        return cls(decay_mode=str(cfg.decay_mode), time_unit_s=float(cfg.time_unit_s), **kw)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HistoryRecord:
    """One completed slot, as the scheduler remembers it."""

    slot: int
    action: str
    scene_id: int
    delta: float
    coach_direction: int = 1
    coach_strength: float = 1.0
    instructed: Optional[str] = None
    c: float = 0.0
    clutter: int = 0
    duration_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class History(list):
    """The slot record list, with the two lookups every scheduler needs."""

    def since_last_coach(self, *, in_deltas: bool = True) -> float:
        """Decay units accumulated since the last demonstration; ``inf`` if there was none."""
        total = 0.0
        seen = False
        for rec in reversed(self):
            if rec.action == COACH:
                seen = True
                break
            total += float(rec.delta) if in_deltas else 1.0
        return total if seen else float("inf")

    def n_since_last_coach(self) -> int:
        n = 0
        for rec in reversed(self):
            if rec.action == COACH:
                return n
            n += 1
        return n

    def last_coach_direction(self) -> int:
        for rec in reversed(self):
            if rec.action == COACH:
                return int(rec.coach_direction)
        return 1


class Scheduler:
    """Base class. Subclasses override :meth:`decide_free_slot`.

    Demonstration slots are never a choice: when the protocol says a slot is a COACH, every
    condition emits COACH there. That is what keeps the manipulation identical across
    conditions.
    """

    #: Condition id written into the session record.
    name: str = "base"
    #: Which pi* estimator the condition's *reported* estimate uses. Conditions differ in this
    #: as well as in their scheduling, and the ablations exist to separate the two.
    estimator: str = "psychometric"

    def __init__(self, *, seed: int = 0) -> None:
        self.seed = int(seed)
        self.budget: Optional[BlockBudget] = None

    def reset(self, budget: BlockBudget) -> None:
        self.budget = budget

    def observe(self, record: HistoryRecord) -> None:
        """Fold a completed slot into whatever internal belief the policy keeps."""

    def decide(self, history: History, slot: Slot) -> SchedulerDecision:
        if slot.is_coach_slot:
            return SchedulerDecision(
                COACH,
                slot.scene_id,
                {"reason": "protocol-fixed demonstration", "direction": int(slot.coach_direction)},
            )
        return self.decide_free_slot(history, slot)

    def decide_free_slot(self, history: History, slot: Slot) -> SchedulerDecision:  # pragma: no cover
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "estimator": self.estimator, "seed": self.seed}


__all__ = [
    "Slot",
    "SchedulerDecision",
    "BlockBudget",
    "DeltaModel",
    "HistoryRecord",
    "History",
    "Scheduler",
]
