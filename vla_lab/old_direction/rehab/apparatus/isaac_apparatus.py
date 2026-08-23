"""W10 — the Isaac Lab digital twin of the apparatus.

This is the retained value of the Isaac stack after the pivot: validate geometry,
reachability, presentation trajectories, settle behaviour, and the safety envelope **before a
person is anywhere near the arm** (``rehab.md`` §6/W10).

It implements the same :class:`~vla_lab.rehab.apparatus.base.Apparatus` protocol as the real
Gen2 backend and the null backend, over the same target set, so a twin dry-run exercises the
session code path rather than a parallel one.

What the dry-run answers (the W10 "done when"):

1. **Reachability** — is every contract target presentable from the study mounting pose?
2. **Participant clearance** — does any presentation trajectory intersect the seated-participant
   proxy volume?
3. **Wrist framing** — at each target, does the re-aimed wrist camera actually see where the
   participant's hand will arrive? The current VLA mount (offset ``(0, -0.055, -0.11)`` m,
   rpy ``(180, 0, 0)``, FOV 87 deg) points along the *grasp approach* axis, which is not
   obviously the right aim for watching a human hand (W11), and this is where that gets
   checked cheaply.

**Isaac is imported lazily, inside methods.** Importing this module must stay free so the test
suite, ``rehab_pilot.sh``, and the analysis keep running on a machine with no simulator. Every
Isaac symbol is resolved at call time, and a missing simulator raises a clear error rather than
an ``ImportError`` at package import.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..contract import Phase0Contract, TimingConfig
from ..workspace import PlanarTransform, TargetGrid, TargetSpec
from .base import ApparatusFault, ApparatusState, BaseApparatus, PresentResult


@dataclass
class ParticipantProxy:
    """The seated-participant keep-out volume, in the participant frame.

    A cylinder (torso/head) plus the swept box the arms occupy while reaching. Deliberately
    generous: the twin's job is to fail early, not to certify a tight envelope.

    ``torso_center_offset_m`` matters more than it looks. The participant frame's origin is the
    sternum **projection** — the *front* of the chest — so a cylinder centred on the origin
    would model the torso as bulging forward into the workspace and would reject the near
    target row outright. The torso is centred behind the origin by roughly its half-depth.
    """

    torso_radius_m: float = 0.22
    torso_height_m: float = 0.75
    #: Torso centre along ``+x`` (forward). Negative: the sternum is the torso's front face.
    torso_center_offset_m: float = -0.13
    #: Reach envelope the participant's arms sweep, as a box in front of the torso.
    reach_box_forward_m: float = 0.55
    reach_box_halfwidth_m: float = 0.55
    reach_box_height_m: float = 0.35
    #: Clearance the robot must keep from the proxy at every waypoint.
    clearance_m: float = 0.10

    def to_dict(self) -> Dict[str, Any]:
        return {
            "torso_radius_m": self.torso_radius_m,
            "torso_height_m": self.torso_height_m,
            "torso_center_offset_m": self.torso_center_offset_m,
            "reach_box_forward_m": self.reach_box_forward_m,
            "reach_box_halfwidth_m": self.reach_box_halfwidth_m,
            "reach_box_height_m": self.reach_box_height_m,
            "clearance_m": self.clearance_m,
        }

    def torso_center(self) -> Tuple[float, float]:
        return (float(self.torso_center_offset_m), 0.0)

    def intersects(self, xy: Sequence[float]) -> bool:
        """Is a participant-frame table-plane point inside the proxy (plus clearance)?

        Only the torso cylinder is a hard keep-out: the reach box is where the *participant's
        hand* goes, which the arm is allowed to occupy between trials but never while the
        participant is reaching — that exclusion is enforced in time by
        :mod:`vla_lab.rehab.safety`, not in space here.
        """

        cx, cy = self.torso_center()
        dx, dy = float(xy[0]) - cx, float(xy[1]) - cy
        r = (dx * dx + dy * dy) ** 0.5
        return bool(r < (self.torso_radius_m + self.clearance_m))


@dataclass
class TwinReport:
    """The output of a twin dry-run. This is what M2 is gated on."""

    n_targets: int = 0
    reachable: List[int] = field(default_factory=list)
    unreachable: List[int] = field(default_factory=list)
    trajectory_collisions: List[Dict[str, Any]] = field(default_factory=list)
    wrist_views: Dict[str, str] = field(default_factory=dict)  # target_id -> rendered path
    notes: List[str] = field(default_factory=list)

    @property
    def all_reachable(self) -> bool:
        return self.n_targets > 0 and not self.unreachable

    @property
    def passed(self) -> bool:
        return self.all_reachable and not self.trajectory_collisions

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_targets": int(self.n_targets),
            "n_reachable": len(self.reachable),
            "unreachable": list(self.unreachable),
            "n_trajectory_collisions": len(self.trajectory_collisions),
            "trajectory_collisions": self.trajectory_collisions,
            "wrist_views": self.wrist_views,
            "passed": bool(self.passed),
            "notes": list(self.notes),
        }


def _require_isaac() -> None:
    """Fail with an instruction, not an ImportError traceback."""

    try:
        import isaaclab  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - any import failure means "no simulator here"
        raise ApparatusFault(
            "The Isaac twin backend needs Isaac Lab in the active environment "
            "(conda activate riften, then run under the Isaac python). For offline work use "
            "--apparatus null, which runs the same session code path with no simulator."
        ) from exc


def straight_line_waypoints(
    start_xy: Sequence[float], end_xy: Sequence[float], n: int = 24
) -> List[Tuple[float, float]]:
    """The nominal presentation path: a straight table-plane move at constant height."""

    x0, y0 = float(start_xy[0]), float(start_xy[1])
    x1, y1 = float(end_xy[0]), float(end_xy[1])
    return [
        (x0 + (x1 - x0) * i / float(max(1, n - 1)), y0 + (y1 - y0) * i / float(max(1, n - 1)))
        for i in range(int(n))
    ]


def check_trajectories(
    grid: TargetGrid,
    *,
    proxy: Optional[ParticipantProxy] = None,
    home_xy: Tuple[float, float] = (0.60, 0.0),
    n_waypoints: int = 24,
) -> List[Dict[str, Any]]:
    """Geometric collision pre-check of every home->target path against the proxy.

    Pure geometry, so it runs **without** Isaac — the point being that the obvious layout
    mistakes get caught in CI, and the simulator is reserved for what only it can answer
    (joint-space reachability, real settle behaviour, camera framing).
    """

    p = proxy or ParticipantProxy()
    out: List[Dict[str, Any]] = []
    for t in grid:
        for wx, wy in straight_line_waypoints(home_xy, (t.x_m, t.y_m), n_waypoints):
            if p.intersects((wx, wy)):
                out.append(
                    {
                        "target_id": int(t.target_id),
                        "waypoint_xy_m": [round(wx, 4), round(wy, 4)],
                        "reason": "enters the participant torso keep-out volume",
                    }
                )
                break
    return out


class IsaacApparatus(BaseApparatus):
    """Digital-twin backend. Same protocol, same targets, same session code path."""

    name = "isaac_twin"

    def __init__(
        self,
        contract: Phase0Contract,
        *,
        proxy: Optional[ParticipantProxy] = None,
        headless: bool = True,
        clock: Optional[Any] = None,
    ) -> None:
        super().__init__(clock=clock)
        self.contract = contract
        self.timing: TimingConfig = contract.timing
        self.grid = contract.target_grid()
        self.proxy = proxy or ParticipantProxy()
        self.headless = bool(headless)
        self._env: Any = None
        self._state = ApparatusState()

    # -- lifecycle ---------------------------------------------------------
    def connect(self) -> None:
        _require_isaac()
        from environments.bilateral_choice import BilateralChoiceTwin  # lazy: needs Isaac

        self._env = BilateralChoiceTwin(self.contract, proxy=self.proxy, headless=self.headless)
        self._env.setup()
        self._state.connected = True
        self._state.last_heartbeat_ms = self.now_ms()

    def home(self) -> None:
        self._require_env().home()
        self._state.homed = True
        self._state.settled = False
        self._state.moving = False

    def present(self, target: TargetSpec) -> PresentResult:
        env = self._require_env()
        t0 = self.now_ms()
        self._state.moving = True
        settled, pose_err = env.move_to(
            (float(target.x_m), float(target.y_m)),
            tolerance_m=float(self.timing.settle_tolerance_m),
            dwell_ms=int(self.timing.settle_dwell_ms),
            timeout_ms=int(self.timing.present_timeout_ms),
        )
        t1 = self.now_ms()
        self._state.moving = False
        self._state.settled = bool(settled)
        if settled:
            self._state.ee_xy_participant_m = (float(target.x_m), float(target.y_m))
        return PresentResult(
            settled=bool(settled),
            t_present_ms=t0,
            t_settled_ms=t1,
            pose_error_m=float(pose_err),
            settle_time_ms=int(t1 - t0),
            fault=None if settled else "settle_failed",
        )

    def prompt(self, kind: str, text: str = "") -> int:
        # The twin has no speaker; the prompt is logged so twin and real timelines line up.
        return self.now_ms()

    def go_signal(self) -> int:
        return self.now_ms()

    def configure_effort(self, level: str) -> bool:
        self._state.effort_level = str(level)
        return False

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
        self._state.connected = False

    def _require_env(self) -> Any:
        if self._env is None:
            raise ApparatusFault("IsaacApparatus.connect() must be called before use")
        return self._env

    # -- the dry-run -------------------------------------------------------
    def dry_run(self, *, render_wrist: bool = True, out_dir: Optional[str] = None) -> TwinReport:
        """Sweep every contract target: reachability, clearance, and the wrist view.

        The M2 gate: 100% of contract targets reachable, zero trajectory intersections with
        the participant proxy, and one wrist-view render per target.
        """

        env = self._require_env()
        report = TwinReport(n_targets=len(self.grid))
        report.trajectory_collisions = check_trajectories(self.grid, proxy=self.proxy)
        for t in self.grid:
            ok, err = env.check_reachable((float(t.x_m), float(t.y_m)))
            (report.reachable if ok else report.unreachable).append(int(t.target_id))
            if not ok:
                report.notes.append(f"target {t.target_id} unreachable in the twin: {err}")
            if render_wrist and ok:
                path = env.render_wrist_view(int(t.target_id), out_dir=out_dir)
                if path:
                    report.wrist_views[str(t.target_id)] = str(path)
        return report


__all__ = [
    "ParticipantProxy",
    "TwinReport",
    "IsaacApparatus",
    "check_trajectories",
    "straight_line_waypoints",
]
