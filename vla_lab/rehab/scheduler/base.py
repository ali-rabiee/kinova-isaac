"""W5 — the scheduler protocol and the budget model every condition shares.

At trial ``t`` the scheduler observes the history ``h_t`` and emits one of
``COACH`` / ``WAIT`` / ``ASSESS`` (``rehab.md`` §1.3). The comparison is only meaningful if
the **interaction budget is matched**, so the split of responsibility is:

- The **protocol** (:mod:`vla_lab.rehab.protocol`) fixes, identically for every compared
  condition: the number of slots ``T``, the target sequence, and *which slots are COACH
  slots* (hence ``C``).
- The **scheduler** decides, on each non-COACH slot, whether to spend it on an ``ASSESS``
  probe or a ``WAIT`` dwell.

That is precisely the contrast §1.3 asks for — "conditions differ only in where the ASSESS
probes are placed relative to the COACH events" — and it makes budget matching exact by
construction rather than by careful bookkeeping. The ASSESS/WAIT *split* does differ between
conditions; that is the tension under study (B1 spends the budget on contaminated probes, B2
spends it on waiting), and it is reported as a manipulation check, not controlled away.

§12.1 is settled here too: the primary contrast varies **only the timing/action**. Target
selection comes from the protocol. A scheduler may only override the target when
``allow_target_choice`` is explicitly enabled, which is the clearly-labelled exploratory arm.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

from .. import ASSESS, COACH, WAIT
from ..carryover import DECAY_TIME, DECAY_TRIALS, CarryoverConfig
from ..trial import History


@runtime_checkable
class Slot(Protocol):
    """Structural type of one protocol-fixed slot (see :mod:`vla_lab.rehab.protocol`)."""

    slot_idx: int
    target_id: int
    is_coach_slot: bool
    effort_level: str


@dataclass
class SchedulerDecision:
    """One scheduling decision, with the belief that produced it.

    ``kappa_prior_mean``/``kappa_prior_sd`` and ``rationale`` are logged into
    ``trials.jsonl`` so an adaptive policy's reasoning is recoverable post hoc — an adaptive
    policy whose decisions cannot be audited is not reviewable (§10).
    """

    action: str
    target_id: Optional[int] = None
    rationale: str = ""
    kappa_prior_mean: float = 0.0
    kappa_prior_sd: float = 0.0
    values: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["values"] = {k: round(float(v), 5) for k, v in (self.values or {}).items()}
        return d


@dataclass
class DeltaModel:
    """How much carryover decay one slot buys, per slot type (§12.3).

    The two parameterizations imply genuinely different WAIT semantics:

    ``trials`` (interference)
        A slot's decay is the interference it causes. A reaching trial is a full interfering
        event (``1.0``); a neutral WAIT filler is a weaker one (``wait_interference``, < 1).
        Setting ``wait_interference = 1.0`` makes waiting *strictly dominated* — same decay,
        one fewer probe — which is a real prediction of a pure interference account and is
        worth reporting if the pilot supports it.
    ``time`` (memory decay)
        A slot's decay is its wall-clock duration in units of ``time_unit_s``. Waiting is
        cheap in trials and expensive in wall-clock, which is where the WAIT cost comes from.
    """

    mode: str = DECAY_TRIALS
    assess: float = 1.0
    coach: float = 1.0
    wait: float = 0.6

    def for_action(self, action: str) -> float:
        return {COACH: float(self.coach), WAIT: float(self.wait)}.get(str(action), float(self.assess))

    @classmethod
    def from_contract(cls, contract: Any, cfg: Optional[CarryoverConfig] = None, *, wait_interference: float = 0.6) -> "DeltaModel":
        cfg = cfg or CarryoverConfig()
        if str(cfg.decay_mode) == DECAY_TIME:
            unit = max(1e-6, float(cfg.time_unit_s))
            trial_s = float(contract.timing.nominal_trial_ms) / 1000.0
            wait_s = float(contract.timing.wait_dwell_ms + contract.timing.inter_trial_ms) / 1000.0
            return cls(mode=DECAY_TIME, assess=trial_s / unit, coach=trial_s / unit, wait=wait_s / unit)
        return cls(mode=DECAY_TRIALS, assess=1.0, coach=1.0, wait=float(wait_interference))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BlockBudget:
    """The matched budget of one block, as the scheduler sees it.

    ``target_sequence`` is included because it is protocol-fixed and written to
    ``protocol.json`` *before* trial 1 — it is public information, identical across
    conditions, and an adaptive policy needs it to look ahead. Handing it over does not give
    B4 an advantage the baselines lack; every condition is presented the same targets in the
    same order (§12.1).
    """

    slots_total: int
    coach_slots: Tuple[int, ...]
    condition: str = ""
    target_sequence: Tuple[int, ...] = ()
    effort_levels: Tuple[str, ...] = ()

    @property
    def n_coach(self) -> int:
        return len(self.coach_slots)

    @property
    def n_free(self) -> int:
        return int(self.slots_total) - self.n_coach

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slots_total": int(self.slots_total),
            "coach_slots": list(self.coach_slots),
            "condition": self.condition,
            "target_sequence": list(self.target_sequence),
            "effort_levels": list(self.effort_levels),
        }


class Scheduler:
    """Base class: emits COACH on protocol-fixed COACH slots, ASSESS otherwise.

    Subclasses override :meth:`decide_free_slot` — the only thing a Phase 0 condition is
    allowed to vary.
    """

    name: str = "base"
    condition: str = "base"
    #: Whether the analysis should apply the carryover-corrected estimator to this condition.
    uses_estimator_correction: bool = False
    #: Exploratory arm only (§12.1): may this scheduler override the protocol's target?
    allow_target_choice: bool = False

    def __init__(self, *, seed: int = 0) -> None:
        self.seed = int(seed)
        self.budget: Optional[BlockBudget] = None

    # -- lifecycle ---------------------------------------------------------
    def reset(self, budget: BlockBudget) -> None:
        self.budget = budget

    def observe(self, record: Any) -> None:
        """Fold a completed trial back in. No-op for history-independent schedulers."""

    # -- decision ----------------------------------------------------------
    def decide(self, history: History, slot: Slot) -> SchedulerDecision:
        if bool(slot.is_coach_slot):
            return SchedulerDecision(
                action=COACH,
                target_id=int(slot.target_id),
                rationale="protocol-fixed COACH slot (budget is matched across conditions)",
            )
        return self.decide_free_slot(history, slot)

    def decide_free_slot(self, history: History, slot: Slot) -> SchedulerDecision:
        return SchedulerDecision(action=ASSESS, target_id=int(slot.target_id), rationale="default: probe")

    # -- reporting ---------------------------------------------------------
    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "condition": self.condition,
            "uses_estimator_correction": bool(self.uses_estimator_correction),
            "allow_target_choice": bool(self.allow_target_choice),
            "seed": int(self.seed),
        }


__all__ = ["Slot", "SchedulerDecision", "DeltaModel", "BlockBudget", "Scheduler"]
