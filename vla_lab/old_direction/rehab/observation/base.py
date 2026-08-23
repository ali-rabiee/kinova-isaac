"""W8 — the arm-choice observer protocol.

This is the **highest-risk new sensing component** in Phase 0 (``rehab.md`` §6/W8): the
scheduler consumes the detection *online*, so a misdetection propagates into ``kappa_hat``
and into every subsequent decision, and the outcome depends on it too.

Three observers sit behind one protocol:

``keyed``    :mod:`~vla_lab.rehab.observation.keyed` — experimenter keypress. Always running,
             as the real-time gold standard and the fallback.
``vision``   :mod:`~vla_lab.rehab.observation.vision` — online left/right classification of
             the reaching hand.
``coding``   :mod:`~vla_lab.rehab.observation.coding` — offline frame-accurate video coding.
             The gold standard the other two are scored against.

Two rules are structural, not conventional:

1. **Labels are never overwritten.** Every observer's label for every trial is appended to
   ``observers.jsonl``; the online label is what the scheduler acted on, the coded label is
   what the analysis uses, and their disagreement is a *reported quantity*. A trial whose
   online and coded labels disagree is flagged, never silently corrected.
2. **The physical side and the handedness-relative label are both kept.** Observers report a
   physical side (``left``/``right``); :func:`arm_from_side` converts it using the
   participant's nonpreferred side. Storing only the converted label would make the raw
   observation unrecoverable if the handedness inventory were ever re-scored.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Protocol, Sequence, runtime_checkable

from .. import ARM_AMBIGUOUS, ARM_NONE, ARM_NONPREFERRED, ARM_PREFERRED
from ..workspace import SIDE_LEFT, SIDE_RIGHT

SOURCE_ONLINE = "online"
SOURCE_CODED = "coded"


def arm_from_side(physical_side: str, nonpreferred_side: str) -> str:
    """Convert an observed physical side into the handedness-relative label."""

    side = str(physical_side).lower()
    if side in ("", ARM_NONE):
        return ARM_NONE
    if side == ARM_AMBIGUOUS:
        return ARM_AMBIGUOUS
    if side not in (SIDE_LEFT, SIDE_RIGHT):
        raise ValueError(f"physical_side must be 'left'/'right'/'none'/'ambiguous'; got {physical_side!r}")
    return ARM_NONPREFERRED if side == str(nonpreferred_side).lower() else ARM_PREFERRED


def side_from_arm(arm: str, nonpreferred_side: str) -> str:
    """Inverse of :func:`arm_from_side`."""

    a = str(arm)
    if a in (ARM_NONE, ARM_AMBIGUOUS):
        return a
    np_side = str(nonpreferred_side).lower()
    pref = SIDE_RIGHT if np_side == SIDE_LEFT else SIDE_LEFT
    return np_side if a == ARM_NONPREFERRED else pref


@dataclass
class ArmSelection:
    """One observer's verdict on one trial."""

    arm: str = ARM_NONE                # preferred | nonpreferred | none | ambiguous
    physical_side: str = ARM_NONE      # left | right | none | ambiguous
    t_ms: Optional[int] = None         # detection timestamp, session monotonic ms
    confidence: float = 0.0
    observer: str = ""
    source: str = SOURCE_ONLINE
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def resolved(self) -> bool:
        return self.arm in (ARM_PREFERRED, ARM_NONPREFERRED)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["confidence"] = round(float(self.confidence), 4)
        return d


@runtime_checkable
class ArmChoiceObserver(Protocol):
    """One trial's observation lifecycle.

    ``begin_trial`` is called at the GO signal; ``poll`` is called repeatedly while the trial
    is in its REACH/SELECT phases and returns ``None`` until a selection resolves;
    ``end_trial`` closes the trial out (returning the best available verdict, possibly
    ``arm="none"`` on a timeout).
    """

    name: str

    def begin_trial(self, trial_idx: int, t_ms: int) -> None: ...

    def poll(self, t_ms: int) -> Optional[ArmSelection]: ...

    def end_trial(self, t_ms: int) -> ArmSelection: ...


class BaseObserver:
    """Shared bookkeeping: current trial, latch-once semantics, and the timeout verdict."""

    name = "base"

    def __init__(self, nonpreferred_side: str) -> None:
        self.nonpreferred_side = str(nonpreferred_side)
        self.trial_idx: Optional[int] = None
        self.t_go_ms: Optional[int] = None
        self._latched: Optional[ArmSelection] = None

    def begin_trial(self, trial_idx: int, t_ms: int) -> None:
        self.trial_idx = int(trial_idx)
        self.t_go_ms = int(t_ms)
        self._latched = None

    def _latch(self, sel: ArmSelection) -> ArmSelection:
        """First resolved verdict wins. A second detection is a *re-attempt*, not a new choice."""

        if self._latched is None:
            self._latched = sel
        return self._latched

    def poll(self, t_ms: int) -> Optional[ArmSelection]:  # pragma: no cover - overridden
        return self._latched

    def end_trial(self, t_ms: int) -> ArmSelection:
        if self._latched is not None:
            return self._latched
        return ArmSelection(
            arm=ARM_NONE,
            physical_side=ARM_NONE,
            t_ms=int(t_ms),
            confidence=0.0,
            observer=self.name,
            source=SOURCE_ONLINE,
            extra={"reason": "no reach detected within the GO window"},
        )


class CompositeObserver:
    """Run several observers on the same trial; one of them is the *primary*.

    The primary's label is what the scheduler acts on (the contract's ``online_observer``);
    every observer's label is still recorded. This is how the keyed observer stays running in
    every session while vision drives the loop — and how, if the pilot fails the ``kappa``
    gate, the fallback is a one-line change of which observer is primary rather than a
    different code path (W8).
    """

    name = "composite"

    def __init__(self, observers: Sequence[Any], *, primary: int = 0) -> None:
        if not observers:
            raise ValueError("CompositeObserver needs at least one observer")
        self.observers = list(observers)
        self.primary = self.observers[int(primary)]
        self.name = getattr(self.primary, "name", "composite")

    def prepare(self, *args: Any, **kw: Any) -> Any:
        out = None
        for o in self.observers:
            fn = getattr(o, "prepare", None)
            if callable(fn):
                out = fn(*args, **kw)
        return out

    def begin_trial(self, trial_idx: int, t_ms: int) -> None:
        for o in self.observers:
            o.begin_trial(int(trial_idx), int(t_ms))

    def poll(self, t_ms: int) -> Optional[ArmSelection]:
        result: Optional[ArmSelection] = None
        for o in self.observers:
            sel = o.poll(int(t_ms))
            if o is self.primary:
                result = sel
        return result

    def end_trial(self, t_ms: int) -> ArmSelection:
        out: Optional[ArmSelection] = None
        for o in self.observers:
            sel = o.end_trial(int(t_ms))
            if o is self.primary:
                out = sel
        return out if out is not None else self.primary.end_trial(int(t_ms))


__all__ = [
    "SOURCE_ONLINE",
    "SOURCE_CODED",
    "ArmSelection",
    "ArmChoiceObserver",
    "BaseObserver",
    "CompositeObserver",
    "arm_from_side",
    "side_from_arm",
]
