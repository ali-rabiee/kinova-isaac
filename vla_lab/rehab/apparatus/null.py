"""W14 — the no-robot backend.

Runs the *entire* Phase 0 session code path — protocol, scheduler, observers, logging, safety
— with no robot, no participant, and no Isaac. This is what makes ``rehab_pilot.sh`` finish a
full synthetic study in seconds, and it is the backend the test suite uses.

Timings come from the contract, so a null session's timestamps are dimensionally identical to
a real one's: the carryover model's time-based parameterization (§12.3) can be exercised
offline without pretending everything happens instantly. When a :class:`ManualClock` is
supplied the backend advances it, so a synthetic study is deterministic *and* fast.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..contract import TimingConfig
from ..workspace import TargetSpec
from .base import ApparatusState, BaseApparatus, PresentResult


class NullApparatus(BaseApparatus):
    """A perfectly-behaved apparatus that exists only in the log."""

    name = "null"

    def __init__(
        self,
        timing: Optional[TimingConfig] = None,
        *,
        clock: Optional[Any] = None,
        manual_clock: Optional[Any] = None,
        settle_time_ms: int = 900,
        pose_error_m: float = 0.002,
    ) -> None:
        super().__init__(clock=clock)
        self.timing = timing or TimingConfig()
        self.manual_clock = manual_clock
        self.settle_time_ms = int(settle_time_ms)
        self.pose_error_m = float(pose_error_m)
        self._state = ApparatusState(connected=False)

    # -- time --------------------------------------------------------------
    def _advance(self, ms: int) -> None:
        """Advance a manual clock, if one is driving the session; otherwise wall time runs."""

        if self.manual_clock is not None:
            self.manual_clock.advance_ms(int(ms))

    # -- protocol ----------------------------------------------------------
    def connect(self) -> None:
        self._state.connected = True
        self._state.last_heartbeat_ms = self.now_ms()

    def home(self) -> None:
        self._advance(500)
        self._state.homed = True
        self._state.moving = False
        self._state.settled = False
        self._state.ee_xy_participant_m = (0.60, 0.0)

    def present(self, target: TargetSpec) -> PresentResult:
        t0 = self.now_ms()
        self._state.moving = True
        self._state.settled = False
        self._advance(self.settle_time_ms)
        self._state.moving = False
        self._state.settled = True
        self._state.ee_xy_participant_m = (float(target.x_m), float(target.y_m))
        self._advance(self.timing.settle_dwell_ms)
        t1 = self.now_ms()
        self._state.last_heartbeat_ms = t1
        return PresentResult(
            settled=True,
            t_present_ms=t0,
            t_settled_ms=t1,
            pose_error_m=self.pose_error_m,
            settle_time_ms=int(t1 - t0),
        )

    def prompt(self, kind: str, text: str = "") -> int:
        self._advance(600)
        return self.now_ms()

    def go_signal(self) -> int:
        self._advance(20)
        return self.now_ms()

    def configure_effort(self, level: str) -> bool:
        self._state.effort_level = str(level)
        return False  # nothing physical to stage in a null apparatus

    def close(self) -> None:
        self._state.connected = False

    def describe(self) -> Dict[str, Any]:
        return {"apparatus": self.name, "settle_time_ms": self.settle_time_ms}


__all__ = ["NullApparatus"]
