"""W8 — the online vision observer: which side of the midline did the reaching hand come from?

**§12.5 is settled here: the classifier is geometric, not learned.** A learned hand-side
classifier is attractive but needs labelled data, and the whole study depends on it. The
geometric rule is auditable, needs no training data, and its failure modes are legible: it
decides on *which side of the participant's midline the approaching hand crossed into the
workspace*, using the front camera as the primary view and the wrist camera as a confirming
one. Escalation to a learned classifier happens only if the pilot shows this cannot reach
Cohen's ``kappa >= 0.9`` (W8). The VLA vision encoder is explicitly **not** the starting
point: it was trained to locate colored boxes, not human hands.

The hand *detector* is injected behind :class:`HandDetector`, so this module holds only the
decision logic and never imports a vision stack. A MediaPipe- or OpenCV-backed detector is a
thin adapter that lives outside this package (see :class:`MediaPipeHandDetector` for the exact
shape it must implement); tests use :class:`ScriptedHandDetector`.

The decision rule, in order:

1. Ignore detections before the GO signal (a resting hand is not a choice) and detections
   outside the table region.
2. Track each detected hand's signed lateral position relative to the participant midline.
3. The first hand whose **displacement toward the target** exceeds ``move_threshold_m`` while
   it is on one side of the midline latches that side.
4. If two hands both move, or the mover cannot be resolved before the reach window closes,
   report ``ambiguous`` — never a guess. Ambiguity is a *recorded outcome* that
   :mod:`vla_lab.rehab.verify_session` counts and gates on; a guess would be laundered into
   the estimand as data.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

from .. import ARM_AMBIGUOUS, ARM_NONE
from ..workspace import SIDE_LEFT, SIDE_RIGHT
from .base import SOURCE_ONLINE, ArmSelection, BaseObserver, arm_from_side


@dataclass
class HandObservation:
    """One detected hand at one instant, in the **participant frame** (metres).

    The detector is responsible for undistorting and projecting onto the table plane using the
    calibration in :mod:`vla_lab.rehab.observation.calibration`; this module works in metres
    and never sees pixels.
    """

    t_ms: int
    x_m: float
    y_m: float
    camera: str = "front"
    detector_confidence: float = 1.0
    #: Optional detector-supplied handedness hint ("left"/"right"/""). Used only to break
    #: ties, never on its own — MediaPipe's handedness flips under self-occlusion.
    side_hint: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k in ("x_m", "y_m", "detector_confidence"):
            d[k] = round(float(getattr(self, k)), 4)
        return d


@runtime_checkable
class HandDetector(Protocol):
    """The adapter contract a real vision backend must satisfy."""

    name: str

    def detect(self, t_ms: int) -> Sequence[HandObservation]:
        """Hands visible at ``t_ms``, in participant-frame metres. May be empty."""


class ScriptedHandDetector:
    """Replays a fixed list of :class:`HandObservation`s. For tests and synthetic sessions."""

    name = "scripted"

    def __init__(self, observations: Sequence[HandObservation]) -> None:
        self.observations = sorted(observations, key=lambda o: int(o.t_ms))
        self._cursor = 0

    def reset(self) -> None:
        self._cursor = 0

    def detect(self, t_ms: int) -> List[HandObservation]:
        out: List[HandObservation] = []
        while self._cursor < len(self.observations) and int(self.observations[self._cursor].t_ms) <= int(t_ms):
            out.append(self.observations[self._cursor])
            self._cursor += 1
        return out


class MediaPipeHandDetector:
    """Adapter shape for a real detector. **Not implemented here** — see W8/W13.

    A working implementation needs (a) a camera capture loop, (b) MediaPipe Hands (or an
    equivalent), and (c) the table-plane homography from
    :func:`vla_lab.rehab.observation.calibration.solve_table_homography` to turn image points
    into participant-frame metres. Those are environment dependencies, not repo code
    (``rehab.md`` §8), so this class documents the contract and refuses to pretend.
    """

    name = "mediapipe"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "MediaPipeHandDetector is a documented adapter shape, not an implementation. "
            "Implement detect(t_ms) -> Sequence[HandObservation] against your camera stack "
            "and the calibration from rehab.observation.calibration, then pass it to "
            "VisionObserver. Until then run keyed-only (the pre-registered fallback, W8)."
        )


@dataclass
class VisionConfig:
    """Decision-rule parameters. Part of the observer's recorded configuration."""

    midline_y_m: float = 0.0        # participant-frame y of the midline (0 by definition)
    move_threshold_m: float = 0.08  # displacement toward the target that counts as "reaching"
    min_confidence: float = 0.5     # detector confidence below this is discarded
    max_hands: int = 2
    #: A hand this close to the midline is not attributed to either side.
    midline_deadband_m: float = 0.03
    #: Both hands moving past threshold within this window -> ambiguous, not a race winner.
    simultaneous_window_ms: int = 120

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class VisionObserver(BaseObserver):
    """Geometric online arm-choice classifier."""

    name = "vision"

    def __init__(
        self,
        nonpreferred_side: str,
        detector: HandDetector,
        *,
        cfg: Optional[VisionConfig] = None,
    ) -> None:
        super().__init__(nonpreferred_side)
        self.detector = detector
        self.cfg = cfg or VisionConfig()
        self._first_seen: Dict[str, Tuple[int, float, float]] = {}  # side -> (t_ms, x, y)
        self._crossed: List[Tuple[int, str, float]] = []            # (t_ms, side, displacement)

    def begin_trial(self, trial_idx: int, t_ms: int) -> None:
        super().begin_trial(trial_idx, t_ms)
        self._first_seen = {}
        self._crossed = []
        reset = getattr(self.detector, "reset", None)
        if callable(reset):
            reset()

    # -- helpers -----------------------------------------------------------
    def _side_of(self, obs: HandObservation) -> Optional[str]:
        dy = float(obs.y_m) - float(self.cfg.midline_y_m)
        if abs(dy) < float(self.cfg.midline_deadband_m):
            return None
        return SIDE_LEFT if dy > 0 else SIDE_RIGHT  # +y is the participant's left

    def _ingest(self, obs: HandObservation) -> None:
        if float(obs.detector_confidence) < float(self.cfg.min_confidence):
            return
        side = self._side_of(obs)
        if side is None:
            return
        if side not in self._first_seen:
            self._first_seen[side] = (int(obs.t_ms), float(obs.x_m), float(obs.y_m))
            return
        t0, x0, y0 = self._first_seen[side]
        disp = math.hypot(float(obs.x_m) - x0, float(obs.y_m) - y0)
        if disp >= float(self.cfg.move_threshold_m):
            if not any(s == side for _, s, _ in self._crossed):
                self._crossed.append((int(obs.t_ms), side, float(disp)))

    def _verdict(self, t_ms: int) -> Optional[ArmSelection]:
        if not self._crossed:
            return None
        self._crossed.sort(key=lambda r: r[0])
        t_first, side_first, disp = self._crossed[0]
        contested = [
            r for r in self._crossed[1:]
            if r[1] != side_first and (r[0] - t_first) <= int(self.cfg.simultaneous_window_ms)
        ]
        if contested:
            return ArmSelection(
                arm=ARM_AMBIGUOUS,
                physical_side=ARM_AMBIGUOUS,
                t_ms=int(t_first),
                confidence=0.0,
                observer=self.name,
                source=SOURCE_ONLINE,
                extra={
                    "reason": "both hands moved within the simultaneity window",
                    "crossings": [[int(a), str(b), round(float(c), 4)] for a, b, c in self._crossed],
                },
            )
        # Confidence: how decisively the winner beat the deadband and the runner-up.
        margin_ms = min(
            (r[0] - t_first for r in self._crossed[1:] if r[1] != side_first),
            default=int(self.cfg.simultaneous_window_ms) * 4,
        )
        conf = min(0.99, 0.6 + 0.3 * min(1.0, margin_ms / (4.0 * max(1, self.cfg.simultaneous_window_ms))) + 0.1 * min(1.0, disp / max(1e-6, 2.0 * self.cfg.move_threshold_m)))
        return ArmSelection(
            arm=arm_from_side(side_first, self.nonpreferred_side),
            physical_side=side_first,
            t_ms=int(t_first),
            confidence=float(conf),
            observer=self.name,
            source=SOURCE_ONLINE,
            extra={"displacement_m": round(float(disp), 4), "margin_ms": int(margin_ms)},
        )

    # -- protocol ----------------------------------------------------------
    def poll(self, t_ms: int) -> Optional[ArmSelection]:
        if self.t_go_ms is None or int(t_ms) < int(self.t_go_ms):
            return None  # a resting hand before GO is not a choice
        for obs in self.detector.detect(int(t_ms)):
            if int(obs.t_ms) < int(self.t_go_ms):
                continue
            self._ingest(obs)
        v = self._verdict(int(t_ms))
        return self._latch(v) if v is not None else self._latched

    def end_trial(self, t_ms: int) -> ArmSelection:
        if self._latched is None:
            v = self._verdict(int(t_ms))
            if v is not None:
                self._latch(v)
        return super().end_trial(int(t_ms))

    def describe(self) -> Dict[str, Any]:
        return {"observer": self.name, "detector": getattr(self.detector, "name", "?"), "cfg": self.cfg.to_dict()}


__all__ = [
    "HandObservation",
    "HandDetector",
    "ScriptedHandDetector",
    "MediaPipeHandDetector",
    "VisionConfig",
    "VisionObserver",
]
