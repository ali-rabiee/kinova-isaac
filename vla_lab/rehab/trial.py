"""W2 — the trial model and the per-trial phase machine.

The Phase 0 unit of analysis is a **trial**, not a fixed-rate tick: one target presentation
plus one arm selection. "Event-locked targets, prompts, selections, outcomes" (deck slide 22)
is a hard requirement rather than logging hygiene, because the carryover model is a function
of *elapsed time since prompt* — timestamp fidelity is a correctness property (``rehab.md``
§6/W2).

Phase sequences, by action:

``ASSESS`` / ``COACH``
    ``PRESENT -> SETTLE -> GO -> REACH -> SELECT -> RETURN -> LOG``
``WAIT``
    ``DWELL -> LOG``

The machine is strict: an out-of-order transition raises rather than being silently
tolerated, because a mis-ordered phase means the timestamps no longer mean what the carryover
model assumes they mean. Per-phase timeouts are enforced against the contract's
:class:`~vla_lab.rehab.contract.TimingConfig`; a timeout is a *recorded outcome*
(``arm="none"``), not an exception.

All timestamps are milliseconds from **one monotonic clock** (:class:`SessionClock`); the
wall-clock offset is recorded once per session so absolute times remain recoverable without
letting NTP steps corrupt inter-event intervals.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import ARM_AMBIGUOUS, ARM_NONE, ARM_NONPREFERRED, ARM_PREFERRED, ASSESS, COACH, WAIT

# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

PHASE_PRESENT = "PRESENT"
PHASE_SETTLE = "SETTLE"
PHASE_GO = "GO"
PHASE_REACH = "REACH"
PHASE_SELECT = "SELECT"
PHASE_RETURN = "RETURN"
PHASE_DWELL = "DWELL"
PHASE_LOG = "LOG"
PHASE_HALTED = "HALTED"

PRESENTED_SEQUENCE: Tuple[str, ...] = (
    PHASE_PRESENT,
    PHASE_SETTLE,
    PHASE_GO,
    PHASE_REACH,
    PHASE_SELECT,
    PHASE_RETURN,
    PHASE_LOG,
)
WAIT_SEQUENCE: Tuple[str, ...] = (PHASE_DWELL, PHASE_LOG)


def phase_sequence(action: str) -> Tuple[str, ...]:
    return WAIT_SEQUENCE if str(action) == WAIT else PRESENTED_SEQUENCE


class PhaseError(RuntimeError):
    """An out-of-order or unknown phase transition. Never swallowed."""


# ---------------------------------------------------------------------------
# Clock
# ---------------------------------------------------------------------------


class SessionClock:
    """One monotonic clock per session, with the wall-clock offset recorded once.

    ``now_ms()`` is milliseconds since the session started. Injectable ``source`` (a
    zero-argument callable returning seconds) so tests drive time deterministically.
    """

    def __init__(self, source: Optional[Any] = None) -> None:
        self._source = source or time.monotonic
        self._t0 = float(self._source())
        self._wall_t0 = time.time()

    def now_ms(self) -> int:
        return int(round((float(self._source()) - self._t0) * 1000.0))

    def offset(self) -> Dict[str, float]:
        """The single recorded mapping from session-ms to wall-clock epoch seconds."""

        return {
            "wall_clock_epoch_s_at_t0": float(self._wall_t0),
            "monotonic_source": getattr(self._source, "__name__", str(self._source)),
        }


class ManualClock:
    """Deterministic clock for tests and synthetic sessions: advance it explicitly."""

    def __init__(self, start_s: float = 0.0) -> None:
        self._t = float(start_s)

    def __call__(self) -> float:  # matches SessionClock's ``source`` contract
        return self._t

    def advance_ms(self, ms: float) -> float:
        self._t += float(ms) / 1000.0
        return self._t


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class Trial:
    """What the scheduler and protocol decided, plus presentation timestamps."""

    trial_idx: int
    block_idx: int
    condition: str
    action: str                      # COACH | WAIT | ASSESS
    target_id: Optional[int] = None  # None for WAIT slots
    target_xy_participant_m: Optional[Tuple[float, float]] = None
    prompt_id: Optional[str] = None
    prompt_hash: str = ""
    effort_level: str = "none"
    slot_idx: int = 0                # position within the block's fixed slot sequence
    is_coach_slot: bool = False      # protocol-fixed COACH slot (identical across conditions)
    t_present_ms: Optional[int] = None
    t_settled_ms: Optional[int] = None
    t_go_ms: Optional[int] = None
    # The scheduler's belief BEFORE this trial, logged so an adaptive policy's decisions
    # are auditable post hoc (§10). An adaptive policy whose reasoning is not recoverable
    # is not reviewable.
    kappa_prior_mean: float = 0.0
    kappa_prior_sd: float = 0.0
    since_last_coach_ms: Optional[int] = None
    coach_count_so_far: int = 0
    scheduler_rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.target_xy_participant_m is not None:
            d["target_xy_participant_m"] = [round(float(v), 4) for v in self.target_xy_participant_m]
        d["kappa_prior_mean"] = round(float(self.kappa_prior_mean), 4)
        d["kappa_prior_sd"] = round(float(self.kappa_prior_sd), 4)
        return d


@dataclass
class TrialResult:
    """What the participant (and the apparatus) actually did."""

    arm: str = ARM_NONE              # preferred | nonpreferred | none | ambiguous
    t_select_ms: Optional[int] = None
    reach_time_ms: Optional[int] = None
    success: bool = False            # reach completed cleanly (touched, no drop, no re-attempt)
    observer: str = ""
    confidence: float = 0.0
    halted: bool = False
    halt_reason: Optional[str] = None
    timed_out: bool = False
    notes: str = ""

    @property
    def is_observation(self) -> bool:
        """True when this trial yields a usable Bernoulli draw for the estimand."""

        return self.arm in (ARM_PREFERRED, ARM_NONPREFERRED) and not self.halted

    @property
    def chose_nonpreferred(self) -> Optional[bool]:
        if not self.is_observation:
            return None
        return self.arm == ARM_NONPREFERRED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selection": {
                "arm": str(self.arm),
                "t_ms": self.t_select_ms,
                "observer": str(self.observer),
                "confidence": round(float(self.confidence), 4),
            },
            "reach_time_ms": self.reach_time_ms,
            "success": bool(self.success),
            "halted": bool(self.halted),
            "halt_reason": self.halt_reason,
            "timed_out": bool(self.timed_out),
            "notes": str(self.notes),
        }


@dataclass
class TrialRecord:
    """One line of ``trials.jsonl``: the decision, the timestamps, and the outcome."""

    trial: Trial
    result: TrialResult
    phases: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {**self.trial.to_dict(), **self.result.to_dict(), "phases": list(self.phases)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrialRecord":
        sel = d.get("selection") or {}
        xy = d.get("target_xy_participant_m")
        trial = Trial(
            trial_idx=int(d.get("trial_idx", 0)),
            block_idx=int(d.get("block_idx", 0)),
            condition=str(d.get("condition", "")),
            action=str(d.get("action", ASSESS)),
            target_id=(int(d["target_id"]) if d.get("target_id") is not None else None),
            target_xy_participant_m=(tuple(float(v) for v in xy) if xy else None),  # type: ignore[arg-type]
            prompt_id=d.get("prompt_id"),
            prompt_hash=str(d.get("prompt_hash", "")),
            effort_level=str(d.get("effort_level", "none")),
            slot_idx=int(d.get("slot_idx", 0)),
            is_coach_slot=bool(d.get("is_coach_slot", False)),
            t_present_ms=d.get("t_present_ms"),
            t_settled_ms=d.get("t_settled_ms"),
            t_go_ms=d.get("t_go_ms"),
            kappa_prior_mean=float(d.get("kappa_prior_mean", 0.0)),
            kappa_prior_sd=float(d.get("kappa_prior_sd", 0.0)),
            since_last_coach_ms=d.get("since_last_coach_ms"),
            coach_count_so_far=int(d.get("coach_count_so_far", 0)),
            scheduler_rationale=str(d.get("scheduler_rationale", "")),
        )
        result = TrialResult(
            arm=str(sel.get("arm", ARM_NONE)),
            t_select_ms=sel.get("t_ms"),
            reach_time_ms=d.get("reach_time_ms"),
            success=bool(d.get("success", False)),
            observer=str(sel.get("observer", "")),
            confidence=float(sel.get("confidence", 0.0) or 0.0),
            halted=bool(d.get("halted", False)),
            halt_reason=d.get("halt_reason"),
            timed_out=bool(d.get("timed_out", False)),
            notes=str(d.get("notes", "")),
        )
        return cls(trial=trial, result=result, phases=list(d.get("phases", [])))


# ---------------------------------------------------------------------------
# Phase machine
# ---------------------------------------------------------------------------


class TrialPhaseMachine:
    """Strict per-trial phase sequencing with per-phase timeouts.

    Usage::

        pm = TrialPhaseMachine(ASSESS, timing)
        pm.enter(PHASE_PRESENT, clock.now_ms())
        ...
        pm.enter(PHASE_SETTLE, clock.now_ms())

    :meth:`enter` raises :class:`PhaseError` on any transition that is not the next phase in
    the action's sequence (``HALTED`` is reachable from anywhere and is terminal).
    :meth:`timed_out` reports whether the phase overran its contract budget.
    """

    def __init__(self, action: str, timing: Any) -> None:
        self.action = str(action)
        self.timing = timing
        self.sequence = phase_sequence(self.action)
        self.index = -1
        self.current: Optional[str] = None
        self.entries: List[Dict[str, Any]] = []

    # -- transitions -------------------------------------------------------
    def enter(self, phase: str, t_ms: int) -> None:
        phase = str(phase)
        if phase == PHASE_HALTED:
            self.current = PHASE_HALTED
            self.entries.append({"phase": phase, "t_ms": int(t_ms)})
            return
        if self.current == PHASE_HALTED:
            raise PhaseError("trial is HALTED; no further phases may be entered")
        if phase not in self.sequence:
            raise PhaseError(f"phase {phase!r} is not part of the {self.action} sequence {self.sequence}")
        expected = self.sequence[self.index + 1] if self.index + 1 < len(self.sequence) else None
        if phase != expected:
            raise PhaseError(
                f"out-of-order phase transition in {self.action} trial: got {phase!r}, expected "
                f"{expected!r} (current={self.current!r})"
            )
        if self.entries and int(t_ms) < int(self.entries[-1]["t_ms"]):
            raise PhaseError(
                f"clock went backwards entering {phase!r}: {t_ms} < {self.entries[-1]['t_ms']}"
            )
        self.index += 1
        self.current = phase
        self.entries.append({"phase": phase, "t_ms": int(t_ms)})

    def halt(self, t_ms: int, reason: str) -> None:
        self.current = PHASE_HALTED
        self.entries.append({"phase": PHASE_HALTED, "t_ms": int(t_ms), "reason": str(reason)})

    # -- queries -----------------------------------------------------------
    @property
    def complete(self) -> bool:
        return self.current == PHASE_LOG

    def t_of(self, phase: str) -> Optional[int]:
        for e in self.entries:
            if e["phase"] == phase:
                return int(e["t_ms"])
        return None

    def budget_ms(self, phase: str) -> Optional[int]:
        t = self.timing
        return {
            PHASE_PRESENT: int(getattr(t, "present_timeout_ms", 0)),
            PHASE_SETTLE: int(getattr(t, "settle_dwell_ms", 0)) * 3,  # settle may need retries
            PHASE_GO: int(getattr(t, "go_window_ms", 0)),
            PHASE_REACH: int(getattr(t, "reach_timeout_ms", 0)),
            PHASE_SELECT: int(getattr(t, "reach_timeout_ms", 0)),
            PHASE_RETURN: int(getattr(t, "return_ms", 0)) * 3,
            PHASE_DWELL: int(getattr(t, "wait_dwell_ms", 0)) * 3,
        }.get(str(phase))

    def timed_out(self, phase: str, t_now_ms: int) -> bool:
        t_entered = self.t_of(phase)
        budget = self.budget_ms(phase)
        if t_entered is None or not budget:
            return False
        return bool(int(t_now_ms) - t_entered > int(budget))

    def to_list(self) -> List[Dict[str, Any]]:
        return list(self.entries)


# ---------------------------------------------------------------------------
# History (what schedulers and estimators consume)
# ---------------------------------------------------------------------------


@dataclass
class History:
    """The trial history ``h_t`` a scheduler observes (§1.3).

    Deliberately a plain container of completed :class:`TrialRecord`s plus the block's fixed
    slot plan, so a scheduler cannot reach for anything the real system would not have.
    """

    records: List[TrialRecord] = field(default_factory=list)
    block_idx: int = 0
    condition: str = ""
    slots_total: int = 0
    coach_slots: Tuple[int, ...] = ()

    def append(self, rec: TrialRecord) -> None:
        self.records.append(rec)

    # -- convenience -------------------------------------------------------
    @property
    def n_done(self) -> int:
        return len(self.records)

    def n_action(self, action: str) -> int:
        return sum(1 for r in self.records if r.trial.action == str(action))

    def last_coach(self) -> Optional[TrialRecord]:
        for r in reversed(self.records):
            if r.trial.action == COACH:
                return r
        return None

    def since_last_coach_ms(self, t_now_ms: int) -> Optional[int]:
        last = self.last_coach()
        if last is None:
            return None
        t = last.trial.t_go_ms if last.trial.t_go_ms is not None else last.trial.t_present_ms
        if t is None:
            return None
        return int(t_now_ms) - int(t)

    def trials_since_last_coach(self) -> Optional[int]:
        for k, r in enumerate(reversed(self.records)):
            if r.trial.action == COACH:
                return k
        return None

    def observations(self) -> List[TrialRecord]:
        """Completed trials that produced a usable arm selection."""

        return [r for r in self.records if r.result.is_observation]

    def budget_spent(self) -> Dict[str, int]:
        """The manipulation check: what the budget was actually spent on."""

        wall = 0
        for r in self.records:
            ts = [e["t_ms"] for e in r.phases] if r.phases else []
            if len(ts) >= 2:
                wall += int(ts[-1]) - int(ts[0])
        return {
            "trials": self.n_done,
            "coach": self.n_action(COACH),
            "assess": self.n_action(ASSESS),
            "wait": self.n_action(WAIT),
            "observations": len(self.observations()),
            "wall_clock_ms": int(wall),
        }


__all__ = [
    "PHASE_PRESENT",
    "PHASE_SETTLE",
    "PHASE_GO",
    "PHASE_REACH",
    "PHASE_SELECT",
    "PHASE_RETURN",
    "PHASE_DWELL",
    "PHASE_LOG",
    "PHASE_HALTED",
    "PRESENTED_SEQUENCE",
    "WAIT_SEQUENCE",
    "PhaseError",
    "SessionClock",
    "ManualClock",
    "Trial",
    "TrialResult",
    "TrialRecord",
    "TrialPhaseMachine",
    "History",
    "phase_sequence",
]
