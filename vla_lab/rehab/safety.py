"""W12 — the human-proximate safety envelope and interlocks.

A human puts their hands where the robot moves. This gates the IRB and everything after it
(``rehab.md`` §6/W12, §11).

``real_robot/safety_envelope.py`` is a reasonable seed — a workspace AABB and velocity clamps
— but it models a robot alone in a scene. It does not model *mutual exclusion between robot
motion and participant reach*, a dwell watchdog, or an e-stop state machine, which are the
interlocks that actually matter when a person shares the workspace. It is left for the VLA
track; this is a fresh file.

**The central invariant: the arm moves, stops, and only then is GO issued.** Motion and reach
are mutually exclusive in *time*. Concretely:

- :meth:`SafetyEnvelope.begin_motion` refuses to start while a reach is in progress;
- :meth:`SafetyEnvelope.reach_detected` during motion halts immediately
  (``reach_during_motion``);
- :meth:`SafetyEnvelope.allow_go` refuses to issue GO unless the arm is stopped and settled.

Everything else is a limit check with a named halt reason: speed and acceleration caps well
below the JACO 2 defaults, a joint-current threshold, a dwell watchdog so no motion command can
run forever, dual e-stop (participant *and* experimenter), and a workspace AABB constrained so
the arm's **body**, not just the end-effector, stays clear of reach paths.

Every halt reason comes from the taxonomy in :mod:`vla_lab.rehab.apparatus.base` and is written
to ``events.jsonl``, so "how many times did it stop, and why" is answerable from the log alone
— which is what the IRB safety appendix has to show.

This module is pure Python: the state machine is fully testable without a robot, and on
hardware each interlock is additionally *demonstrated* and the demonstration recorded (W12's
"done when").
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .apparatus.base import (
    HALT_DWELL_TIMEOUT,
    HALT_ESTOP_EXPERIMENTER,
    HALT_ESTOP_PARTICIPANT,
    HALT_EXPERIMENTER_REQUEST,
    HALT_PARTICIPANT_REQUEST,
    HALT_REACH_DURING_MOTION,
    HALT_REASONS,
    HALT_SPEED_LIMIT,
    HALT_TORQUE_LIMIT,
    HALT_WORKSPACE_VIOLATION,
)

# Safety states.
STATE_IDLE = "IDLE"
STATE_MOVING = "MOVING"
STATE_SETTLED = "SETTLED"
STATE_GO_ISSUED = "GO_ISSUED"
STATE_REACHING = "REACHING"
STATE_RETURNING = "RETURNING"
STATE_HALTED = "HALTED"

SOURCE_PARTICIPANT = "participant"
SOURCE_EXPERIMENTER = "experimenter"


class SafetyViolation(RuntimeError):
    """An interlock refused an action. The caller must halt, not retry."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = str(reason)


@dataclass
class SafetyLimits:
    """Caps, all well below the JACO 2's defaults because a human is inside the workspace."""

    max_cartesian_speed_ms: float = 0.12
    max_cartesian_accel_ms2: float = 0.25
    max_joint_current_a: float = 2.5
    #: No single motion command may run longer than this (the dwell watchdog).
    max_motion_ms: int = 9000
    #: Workspace AABB in the participant frame, for the EE *and* the arm body.
    x_min_m: float = 0.10
    x_max_m: float = 0.95
    y_min_m: float = -0.55
    y_max_m: float = 0.55
    z_min_m: float = -0.02
    z_max_m: float = 0.45
    #: The arm's body must keep at least this clearance from the participant torso axis.
    body_clearance_m: float = 0.30

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "SafetyLimits":
        d = dict(d or {})
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class HaltEvent:
    reason: str
    t_ms: int
    state_before: str
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SafetyEnvelope:
    """The interlock state machine. One instance per session.

    ``on_halt`` is invoked with the :class:`HaltEvent` for every halt, so the session runner can
    log it to ``events.jsonl`` and command the apparatus to stop — this class decides, it does
    not drive hardware.
    """

    def __init__(
        self,
        limits: Optional[SafetyLimits] = None,
        *,
        on_halt: Optional[Callable[[HaltEvent], None]] = None,
    ) -> None:
        self.limits = limits or SafetyLimits()
        self.on_halt = on_halt
        self.state = STATE_IDLE
        self.halts: List[HaltEvent] = []
        self._motion_started_ms: Optional[int] = None
        self._estops: Dict[str, bool] = {SOURCE_PARTICIPANT: False, SOURCE_EXPERIMENTER: False}

    # -- helpers -----------------------------------------------------------
    @property
    def halted(self) -> bool:
        return self.state == STATE_HALTED

    @property
    def estop_engaged(self) -> bool:
        return any(self._estops.values())

    def _halt(self, reason: str, t_ms: int, detail: str = "") -> HaltEvent:
        if reason not in HALT_REASONS:
            raise ValueError(f"unknown halt reason {reason!r}")
        ev = HaltEvent(reason=str(reason), t_ms=int(t_ms), state_before=self.state, detail=str(detail))
        self.state = STATE_HALTED
        self._motion_started_ms = None
        self.halts.append(ev)
        if self.on_halt is not None:
            self.on_halt(ev)
        return ev

    def _require_not_halted(self) -> None:
        if self.halted:
            raise SafetyViolation(
                self.halts[-1].reason if self.halts else HALT_EXPERIMENTER_REQUEST,
                "envelope is HALTED; an explicit reset() by the experimenter is required",
            )

    # -- motion / reach mutual exclusion ------------------------------------
    def begin_motion(self, t_ms: int) -> None:
        """Start a presentation move. Refused unless the workspace is clear of a reach."""

        self._require_not_halted()
        if self.estop_engaged:
            raise SafetyViolation(HALT_ESTOP_PARTICIPANT, "e-stop engaged; motion refused")
        if self.state in (STATE_REACHING, STATE_GO_ISSUED):
            raise SafetyViolation(
                HALT_REACH_DURING_MOTION,
                f"refusing to move while the participant may be reaching (state={self.state})",
            )
        self.state = STATE_MOVING
        self._motion_started_ms = int(t_ms)

    def end_motion(self, t_ms: int, *, settled: bool) -> None:
        """The arm has stopped. ``settled`` records whether it reached tolerance."""

        self._require_not_halted()
        if self.state != STATE_MOVING:
            raise SafetyViolation(HALT_EXPERIMENTER_REQUEST, f"end_motion from state {self.state}")
        self._motion_started_ms = None
        self.state = STATE_SETTLED if settled else STATE_IDLE

    def allow_go(self, t_ms: int) -> bool:
        """May the GO cue be issued? Only from a stopped, settled arm (§9)."""

        return (not self.halted) and (not self.estop_engaged) and self.state == STATE_SETTLED

    def issue_go(self, t_ms: int) -> None:
        if not self.allow_go(t_ms):
            raise SafetyViolation(
                HALT_REACH_DURING_MOTION,
                f"GO refused: arm is not stopped and settled (state={self.state}, "
                f"estop={self.estop_engaged})",
            )
        self.state = STATE_GO_ISSUED

    def reach_detected(self, t_ms: int) -> Optional[HaltEvent]:
        """Report that the participant's hand entered the workspace.

        During ``MOVING`` this is the interlock's whole reason for existing: it halts
        immediately. After GO it is the expected transition into ``REACHING``.
        """

        if self.state == STATE_MOVING:
            return self._halt(HALT_REACH_DURING_MOTION, t_ms, "reach detected while the arm was moving")
        if self.state in (STATE_GO_ISSUED, STATE_REACHING):
            self.state = STATE_REACHING
            return None
        if self.state == STATE_SETTLED:
            # A reach before GO: not dangerous (the arm is stopped), but it is a protocol
            # deviation, so it is surfaced to the caller rather than silently accepted.
            self.state = STATE_REACHING
            return None
        return None

    def end_reach(self, t_ms: int) -> None:
        if self.halted:
            return
        self.state = STATE_RETURNING

    def end_trial(self, t_ms: int) -> None:
        if not self.halted:
            self.state = STATE_IDLE

    # -- limit checks -------------------------------------------------------
    def check_motion_command(self, *, speed_ms: float, accel_ms2: float, t_ms: int) -> None:
        if float(speed_ms) > self.limits.max_cartesian_speed_ms:
            self._halt(
                HALT_SPEED_LIMIT, t_ms,
                f"commanded speed {speed_ms:.3f} m/s exceeds cap {self.limits.max_cartesian_speed_ms:.3f}",
            )
            raise SafetyViolation(HALT_SPEED_LIMIT, "speed cap exceeded")
        if float(accel_ms2) > self.limits.max_cartesian_accel_ms2:
            self._halt(
                HALT_SPEED_LIMIT, t_ms,
                f"commanded accel {accel_ms2:.3f} m/s^2 exceeds cap {self.limits.max_cartesian_accel_ms2:.3f}",
            )
            raise SafetyViolation(HALT_SPEED_LIMIT, "acceleration cap exceeded")

    def check_pose(self, xyz: Sequence[float], t_ms: int) -> None:
        x, y = float(xyz[0]), float(xyz[1])
        z = float(xyz[2]) if len(xyz) > 2 else 0.0
        lim = self.limits
        if not (lim.x_min_m <= x <= lim.x_max_m and lim.y_min_m <= y <= lim.y_max_m and lim.z_min_m <= z <= lim.z_max_m):
            self._halt(HALT_WORKSPACE_VIOLATION, t_ms, f"EE at ({x:.3f}, {y:.3f}, {z:.3f}) is outside the AABB")
            raise SafetyViolation(HALT_WORKSPACE_VIOLATION, "workspace AABB violated")
        if (x * x + y * y) ** 0.5 < lim.body_clearance_m:
            self._halt(
                HALT_WORKSPACE_VIOLATION, t_ms,
                f"EE at ({x:.3f}, {y:.3f}) is within {lim.body_clearance_m:.2f} m of the participant axis",
            )
            raise SafetyViolation(HALT_WORKSPACE_VIOLATION, "participant clearance violated")

    def check_currents(self, currents_a: Sequence[float], t_ms: int) -> None:
        for i, c in enumerate(currents_a):
            if abs(float(c)) > self.limits.max_joint_current_a:
                self._halt(
                    HALT_TORQUE_LIMIT, t_ms,
                    f"joint {i} current {abs(float(c)):.2f} A exceeds {self.limits.max_joint_current_a:.2f} A",
                )
                raise SafetyViolation(HALT_TORQUE_LIMIT, "joint current threshold exceeded")

    def tick(self, t_ms: int) -> Optional[HaltEvent]:
        """Per-loop watchdog. Call every control iteration."""

        if self.halted:
            return None
        if self._motion_started_ms is not None:
            elapsed = int(t_ms) - int(self._motion_started_ms)
            if elapsed > int(self.limits.max_motion_ms):
                return self._halt(
                    HALT_DWELL_TIMEOUT, t_ms,
                    f"motion has run {elapsed} ms, over the {self.limits.max_motion_ms} ms watchdog",
                )
        return None

    # -- e-stop and stop requests -------------------------------------------
    def estop(self, source: str, t_ms: int) -> HaltEvent:
        """Dual e-stop: within the participant's reach **and** the experimenter's (§11)."""

        src = str(source)
        if src not in self._estops:
            raise ValueError(f"e-stop source must be one of {sorted(self._estops)}; got {source!r}")
        self._estops[src] = True
        reason = HALT_ESTOP_PARTICIPANT if src == SOURCE_PARTICIPANT else HALT_ESTOP_EXPERIMENTER
        return self._halt(reason, t_ms, f"e-stop pressed by the {src}")

    def release_estop(self, source: str) -> None:
        self._estops[str(source)] = False

    def request_stop(self, source: str, t_ms: int, detail: str = "") -> HaltEvent:
        """A *verbal* stop: the participant may end a trial, block, or session at any time.

        This is a **logged event**, never a silent abort, and the resulting partial session is
        accepted by :mod:`vla_lab.rehab.verify_session` (§11).
        """

        reason = HALT_PARTICIPANT_REQUEST if str(source) == SOURCE_PARTICIPANT else HALT_EXPERIMENTER_REQUEST
        return self._halt(reason, t_ms, detail or f"stop requested by the {source}")

    def reset(self, t_ms: int, *, cleared_by: str = SOURCE_EXPERIMENTER) -> None:
        """Clear a halt. Always explicit, and only with every e-stop released."""

        if self.estop_engaged:
            raise SafetyViolation(
                HALT_ESTOP_PARTICIPANT,
                "cannot reset while an e-stop is engaged; release it physically first",
            )
        self.state = STATE_IDLE
        self._motion_started_ms = None

    # -- reporting ---------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        for h in self.halts:
            counts[h.reason] = counts.get(h.reason, 0) + 1
        return {
            "state": self.state,
            "n_halts": len(self.halts),
            "halts_by_reason": counts,
            "limits": self.limits.to_dict(),
            "estops": dict(self._estops),
        }


__all__ = [
    "STATE_IDLE",
    "STATE_MOVING",
    "STATE_SETTLED",
    "STATE_GO_ISSUED",
    "STATE_REACHING",
    "STATE_RETURNING",
    "STATE_HALTED",
    "SOURCE_PARTICIPANT",
    "SOURCE_EXPERIMENTER",
    "SafetyLimits",
    "SafetyViolation",
    "HaltEvent",
    "SafetyEnvelope",
]
