"""The three seams between the study and whatever is on the other end of it.

Everything the session runner touches goes through one of these, which is what lets exactly the
same session code run against (a) a calibrated surrogate in milliseconds, (b) a closed-loop
Isaac Lab scene with a real policy driving a simulated Kinova, and eventually (c) the physical
arm with a person in the room. If a result changes across those, the change is in the seam and
not in the study logic.

``Apparatus``
    The robot. Sets up a scene, executes a strategy, speaks. Returns what happened.
``SupervisorChannel``
    The human. Receives a query, returns an utterance.
``Grounder``
    The listener. Turns an utterance into a strategy label. Two implementations exist and are
    **compared, not merged**: a lexical reference grounder and the policy's own intent head.
    The label the scheduler acted on and the label the analysis uses stay separately
    recoverable, and their disagreement is a reported quantity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Protocol, runtime_checkable

from ..scenes import SceneSpec


@dataclass
class ExecutionOutcome:
    """What the robot did and how it went."""

    strategy: str
    success: bool
    duration_s: float
    notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SupervisorTurn:
    """What the supervisor said, and how long they took to say it."""

    utterance: str
    latency_s: float
    #: Ground truth, present only when the supervisor is simulated. The session logs it under a
    #: separate key and the estimators never see it -- it exists so the analysis can score how
    #: far a contaminated answer moved from the unprompted one.
    truth: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@runtime_checkable
class Apparatus(Protocol):
    """The robot side."""

    def reset_scene(self, scene: SceneSpec) -> None: ...

    def execute(self, scene: SceneSpec, strategy: str) -> ExecutionOutcome: ...

    def say(self, text: str) -> None: ...

    def describe(self) -> Dict[str, Any]: ...

    def close(self) -> None: ...


@runtime_checkable
class SupervisorChannel(Protocol):
    """The human side."""

    def ask(self, query: str, scene: SceneSpec, *, action: str, session_progress: float) -> SupervisorTurn: ...

    def observe_demonstration(self, scene: SceneSpec, strategy: str, narration: str, *, strength: float) -> None: ...

    def elapse(self, delta: float) -> None: ...

    def describe(self) -> Dict[str, Any]: ...


@runtime_checkable
class Grounder(Protocol):
    """Utterance -> strategy label."""

    name: str

    def ground(self, utterance: str, scene: SceneSpec) -> str: ...


__all__ = ["ExecutionOutcome", "SupervisorTurn", "Apparatus", "SupervisorChannel", "Grounder"]
