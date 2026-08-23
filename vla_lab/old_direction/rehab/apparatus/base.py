"""W10/W11 — the apparatus protocol shared by the null, twin, and real backends.

In Phase 0 the robot is an **apparatus and a coach**, not a manipulator (``rehab.md`` §2,
scope table): it *presents* standardized reach targets across a bilateral tabletop workspace
and delivers scripted prompts. The participant reaches with their own arm. The robot never
touches the target.

The protocol is deliberately small and **blocking**, which is the shape Phase 0 needs and the
opposite of the VLA track's ``real_robot/kinova_bridge.py`` (policy-chunk stepping, and
written for a Gen3/Kortex arm besides). One presentation is: move -> **stop** -> verify settle
-> issue GO. The stop is contractual (§9), because a moving arm and a reaching human must
never share the workspace (:mod:`vla_lab.rehab.safety`).

Three backends implement it:

``null``    :mod:`~vla_lab.rehab.apparatus.null` — no robot. Synthetic pilots, CI, analysis
            development. Runs a whole study in seconds.
``isaac``   :mod:`~vla_lab.rehab.apparatus.isaac_apparatus` — the digital twin: geometry,
            reachability, presentation trajectories, and the safety envelope, validated
            before a person is near the arm.
``gen2``    :mod:`~vla_lab.rehab.apparatus.kinova_gen2` — the real JACO 2, over a narrow IPC
            surface to an out-of-process vendor driver.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Protocol, Sequence, Tuple, runtime_checkable

from ..workspace import TargetSpec

# Halt-reason taxonomy. Written verbatim into ``events.jsonl`` so a reviewer can count them.
HALT_ESTOP_PARTICIPANT = "estop_participant"
HALT_ESTOP_EXPERIMENTER = "estop_experimenter"
HALT_REACH_DURING_MOTION = "reach_during_motion"
HALT_TORQUE_LIMIT = "torque_limit"
HALT_SPEED_LIMIT = "speed_limit"
HALT_DWELL_TIMEOUT = "dwell_timeout"
HALT_WORKSPACE_VIOLATION = "workspace_violation"
HALT_DRIVER_FAULT = "driver_fault"
HALT_HEARTBEAT_LOST = "heartbeat_lost"
HALT_PARTICIPANT_REQUEST = "participant_request"
HALT_EXPERIMENTER_REQUEST = "experimenter_request"

HALT_REASONS = (
    HALT_ESTOP_PARTICIPANT,
    HALT_ESTOP_EXPERIMENTER,
    HALT_REACH_DURING_MOTION,
    HALT_TORQUE_LIMIT,
    HALT_SPEED_LIMIT,
    HALT_DWELL_TIMEOUT,
    HALT_WORKSPACE_VIOLATION,
    HALT_DRIVER_FAULT,
    HALT_HEARTBEAT_LOST,
    HALT_PARTICIPANT_REQUEST,
    HALT_EXPERIMENTER_REQUEST,
)


@dataclass
class PresentResult:
    """Outcome of one ``present`` call."""

    settled: bool
    t_present_ms: int = 0
    t_settled_ms: int = 0
    pose_error_m: float = 0.0
    settle_time_ms: int = 0
    fault: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["pose_error_m"] = round(float(self.pose_error_m), 5)
        return d


@dataclass
class ApparatusState:
    """A snapshot of what the arm is doing. Polled by the safety layer every loop."""

    connected: bool = False
    moving: bool = False
    settled: bool = False
    homed: bool = False
    ee_xy_participant_m: Tuple[float, float] = (0.0, 0.0)
    joint_currents_a: Tuple[float, ...] = ()
    fault: Optional[str] = None
    estop_engaged: bool = False
    last_heartbeat_ms: int = 0
    effort_level: str = "none"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["ee_xy_participant_m"] = [round(float(v), 4) for v in self.ee_xy_participant_m]
        d["joint_currents_a"] = [round(float(v), 4) for v in self.joint_currents_a]
        return d


class ApparatusFault(RuntimeError):
    """A driver-level fault. Never swallowed: it ends the trial and is logged with its reason."""


@runtime_checkable
class Apparatus(Protocol):
    """The presentation apparatus contract."""

    name: str

    def connect(self) -> None: ...

    def home(self) -> None: ...

    def present(self, target: TargetSpec) -> PresentResult:
        """Move the EE to ``target``, **stop**, and verify settle. Blocking."""

    def prompt(self, kind: str, text: str = "") -> int:
        """Deliver a scripted utterance. Returns the delivery timestamp (session ms)."""

    def go_signal(self) -> int:
        """Issue the GO cue. Returns its timestamp (session ms)."""

    def configure_effort(self, level: str) -> bool:
        """Apply an effort level. Returns True when the experimenter must act physically."""

    def halt(self, reason: str) -> None: ...

    def state(self) -> ApparatusState: ...

    def close(self) -> None: ...


class BaseApparatus:
    """Bookkeeping shared by every backend: clock, state, halt latch."""

    name = "base"

    def __init__(self, *, clock: Optional[Any] = None) -> None:
        from ..trial import SessionClock

        self.clock = clock or SessionClock()
        self._state = ApparatusState()
        self._halted_reason: Optional[str] = None

    def now_ms(self) -> int:
        return int(self.clock.now_ms())

    def state(self) -> ApparatusState:
        return self._state

    def halt(self, reason: str) -> None:
        if reason not in HALT_REASONS:
            raise ValueError(f"unknown halt reason {reason!r}; taxonomy: {HALT_REASONS}")
        self._halted_reason = str(reason)
        self._state.moving = False
        self._state.settled = False
        self._state.fault = str(reason)

    @property
    def halted(self) -> bool:
        return self._halted_reason is not None

    def clear_halt(self) -> None:
        """Clear a halt after the experimenter has resolved it. Always an explicit act."""

        self._halted_reason = None
        self._state.fault = None

    def configure_effort(self, level: str) -> bool:
        self._state.effort_level = str(level)
        return False

    def close(self) -> None:  # pragma: no cover - default no-op
        pass


__all__ = [
    "Apparatus",
    "BaseApparatus",
    "ApparatusState",
    "ApparatusFault",
    "PresentResult",
    "HALT_REASONS",
    "HALT_ESTOP_PARTICIPANT",
    "HALT_ESTOP_EXPERIMENTER",
    "HALT_REACH_DURING_MOTION",
    "HALT_TORQUE_LIMIT",
    "HALT_SPEED_LIMIT",
    "HALT_DWELL_TIMEOUT",
    "HALT_WORKSPACE_VIOLATION",
    "HALT_DRIVER_FAULT",
    "HALT_HEARTBEAT_LOST",
    "HALT_PARTICIPANT_REQUEST",
    "HALT_EXPERIMENTER_REQUEST",
]
