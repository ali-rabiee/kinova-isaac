"""W8 — the experimenter-keypress observer.

**Always running**, in every session, in every condition. Two jobs:

1. the real-time gold standard the vision observer is scored against, and
2. the pre-registered fallback: if the pilot shows vision cannot reach Cohen's
   ``kappa >= 0.9`` against the coded labels, the study runs keyed-only and vision becomes an
   exploratory contribution (``rehab.md`` §6/W8 — that decision is made *at the pilot*, not
   later).

The key source is **injected** (a zero-argument callable returning a pressed key or ``None``),
so the observer is testable without a terminal and can be driven from a GUI, a footswitch, or
a scripted sequence without changing this file.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from .. import ARM_AMBIGUOUS, ARM_NONE
from ..workspace import SIDE_LEFT, SIDE_RIGHT
from .base import SOURCE_ONLINE, ArmSelection, BaseObserver, arm_from_side

#: Default key bindings. Left/right are the *participant's* sides, so the experimenter is
#: instructed to sit facing the participant and press the key matching the hand they see move,
#: from the participant's point of view. The mapping is recorded in ``contract.json``.
DEFAULT_KEYMAP: Dict[str, str] = {
    "z": SIDE_LEFT,
    "m": SIDE_RIGHT,
    "x": ARM_AMBIGUOUS,   # saw a reach, could not call the side
    "n": ARM_NONE,        # no reach happened
}


class KeyedObserver(BaseObserver):
    """Experimenter keypress -> arm selection."""

    name = "keyed"

    def __init__(
        self,
        nonpreferred_side: str,
        key_source: Callable[[], Optional[str]],
        *,
        keymap: Optional[Dict[str, str]] = None,
        confidence: float = 1.0,
    ) -> None:
        super().__init__(nonpreferred_side)
        self.key_source = key_source
        self.keymap = dict(keymap or DEFAULT_KEYMAP)
        self.confidence = float(confidence)
        self.presses: List[Tuple[int, str]] = []

    def poll(self, t_ms: int) -> Optional[ArmSelection]:
        key = self.key_source()
        if key is None:
            return self._latched
        k = str(key).lower()
        if k not in self.keymap:
            return self._latched
        self.presses.append((int(t_ms), k))
        side = self.keymap[k]
        arm = arm_from_side(side, self.nonpreferred_side)
        return self._latch(
            ArmSelection(
                arm=arm,
                physical_side=side,
                t_ms=int(t_ms),
                confidence=self.confidence,
                observer=self.name,
                source=SOURCE_ONLINE,
                extra={"key": k},
            )
        )


class ScriptedKeySource:
    """A deterministic key source for tests and synthetic sessions.

    Feed it ``(trial_idx, key)`` pairs; it emits each trial's key once, on the first poll of
    that trial. ``current_trial`` must be kept in sync by the caller.
    """

    def __init__(self, keys_by_trial: Dict[int, str]) -> None:
        self.keys = dict(keys_by_trial)
        self.current_trial: int = -1
        self._emitted: set = set()

    def __call__(self) -> Optional[str]:
        t = int(self.current_trial)
        if t in self._emitted or t not in self.keys:
            return None
        self._emitted.add(t)
        return self.keys[t]


__all__ = ["DEFAULT_KEYMAP", "KeyedObserver", "ScriptedKeySource"]
