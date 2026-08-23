"""Tier-2 backend: the closed loop, in Isaac Lab, with the policy actually driving the arm.

Tier 1 samples execution outcomes from measured curves so that thousands of sessions fit in a
coffee break. Tier 2 runs the same session code with the simulator behind the apparatus seam,
so a smaller number of complete sessions are executed end-to-end by the trained policy. The
**fidelity check** between them is reported before any result that depends on Tier 1: if the
surrogate's success and duration distributions do not match what the closed loop produces, the
Tier-1 numbers are describing a system that does not exist.

Isaac and Isaac Lab are imported inside :meth:`IsaacApparatus.open`, never at module import,
so this file can be imported (and its fidelity math tested) on a machine with no simulator.
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .. import STRATEGY_A, STRATEGY_B
from ..scenes import SceneGrid, SceneSpec
from .base import ExecutionOutcome


@dataclass
class IsaacApparatusConfig:
    headless: bool = True
    device: str = "cuda:0"
    #: Physics steps per control tick, and the control rate. 15 Hz matches the data-collection
    #: contract the policies are trained under; changing it silently rescales every action.
    control_hz: float = 15.0
    #: Physics steps allowed per waypoint. The proven collector uses 2400; 240 is roughly one
    #: half-metre of commanded travel at the jog speed, which is less than a transit move needs,
    #: and it presents as "every rollout fails on its first waypoint" rather than as a timeout.
    max_steps_per_waypoint: int = 4000
    settle_steps: int = 30
    #: Steps spent opening or closing the gripper, through the controller's gripper mode.
    gripper_steps: int = 60
    #: Steps held after a gripper-only waypoint, so contact forces settle before the next move.
    #: Lifting on the step the fingers meet the cube catches the grasp mid-slip.
    grasp_settle_steps: int = 40
    #: Height at which the arm crosses the table between phases, in the base frame. ``None``
    #: keeps the default (stay where you are, but no lower than transit height), which for this
    #: scene means crossing at the home pose's 0.44 m.
    rise_height_m: Optional[float] = None
    #: Jog speed and the under-relaxation gain on the Diff-IK velocity command. The soft Jaco
    #: actuators make the deadbeat gain (1.0) under-damped, so the end effector orbits its path.
    #: Jog speed. 1.5 was tried, to compensate for the small fraction of the commanded step the
    #: IK realises near the table, and made things worse: the loop went unstable on the
    #: pregrasp approach and displaced the blocker 0.20 m. The realized descent rate is bought
    #: with step budget (``deep_phase_step_multiplier``), not with speed.
    linear_speed_mps: float = 0.5
    #: Phases whose goal is close to the table descend far more slowly than transits and get a
    #: proportionally larger budget. Measured: a transit converges in ~100 steps, the final
    #: 8 cm of descent takes several thousand.
    deep_phases: tuple = ("descend", "close", "push_start", "push")
    deep_phase_step_multiplier: int = 4
    jog_velocity_gain: float = 1.0
    #: ``pinv`` reached the goal in 444 steps where ``dls`` needed more than 1500: the damping
    #: that makes DLS safe near a singularity is also what stalls it here.
    ik_method: str = "pinv"
    #: See the note in ``SupervisoryFetchScene.open``. Holding the *home* orientation stops the
    #: end effector descending at all in this scene, because the home orientation is inverted;
    #: the hold is armed later, against the aligned pose, by ``hold_after_orient``.
    hold_orientation: bool = False
    #: Passes the wrist alignment may take to converge. Each one closes part of the remaining
    #: angle, so this is a convergence budget, not a fixed schedule.
    orient_max_passes: int = 24
    #: Re-enable the hold once the wrist has been aligned downward. **On**, and the history is
    #: worth recording because the first measurement said the opposite. With the hold re-armed
    #: the descent stalled at z ~ 0.44 m against ~0.066 m with it off, which read as "holding
    #: any orientation over-constrains this arm". That measurement was taken while the
    #: alignment was silently rotating about the wrong axis (see
    #: ``SupervisoryFetchScene._base_rotvec_to_tool``): the pose being held pointed the gripper
    #: at the *ceiling*, so of course holding it made the descent worse. With the frame bug
    #: fixed the alignment reaches a genuinely downward tool axis, and the hold is what keeps it
    #: there -- without it the wrist drifts back to +0.17 downwardness during the very first
    #: translate.
    hold_after_orient: bool = True
    #: Canonical start pose in the robot base frame, matching the reach-to-grasp collection
    #: contract. Every rollout is pre-rolled here so the first move is not a nuisance variable
    #: and the controller's start transient is absorbed before the strategy begins.
    start_ee_pos_b: tuple = (0.454, 0.093, 0.210)
    preroll_steps: int = 900
    preroll_tol_m: float = 0.010
    #: Steps to settle at the configured home joint pose before the controller captures its
    #: orientation hold. Resetting the controller against a still-moving arm anchors the hold
    #: to a transient configuration.
    home_settle_steps: int = 20
    #: Safety workspace box handed to the controller. Permissive by default and deliberately
    #: so: the arm's own home configuration sits outside the collector's tighter box (base
    #: x = 0.05, z = 0.433), and a clamp that is already violated at step zero fights every
    #: command instead of guarding an edge case. The scene's own geometry -- table plane,
    #: object spread -- is the real bound, and it is enforced by the waypoints.
    workspace_min: tuple = (0.05, -0.60, -0.30)
    workspace_max: tuple = (0.80, 0.60, 1.40)
    #: Episode is a success when the target has risen this far above its start height.
    lift_threshold_m: float = 0.06
    capture_frames: bool = False
    #: Render steps before a capture, to clear the renderer's temporal accumulation.
    capture_render_steps: int = 12
    frames_dir: Optional[Path] = None


class IsaacApparatus:
    """Executes a strategy on a real (simulated) Kinova and reports what happened.

    Deliberately thin. It owns scene construction, waypoint execution, and success detection;
    it owns no study logic at all, which is what lets the identical session runner drive it and
    the surrogate.
    """

    def __init__(
        self,
        grid: SceneGrid,
        *,
        cfg: Optional[IsaacApparatusConfig] = None,
        policy: Optional[Any] = None,
        seed: int = 0,
    ) -> None:
        self.grid = grid
        self.cfg = cfg or IsaacApparatusConfig()
        #: When set, the policy drives the arm and the scripted expert is used only as a
        #: fallback for demonstrations. When None, everything is scripted -- which is the mode
        #: the physics sweep runs in.
        self.policy = policy
        self.seed = int(seed)
        self._sim = None
        self._rollouts: List[Dict[str, Any]] = []
        self._scene = None
        self.transcript: List[str] = []

    # -- lifecycle ----------------------------------------------------------
    def open(self) -> None:
        if self._sim is not None:
            return
        from environments.supervisory_fetch.scene import SupervisoryFetchScene  # lazy: needs Isaac

        self._scene = SupervisoryFetchScene(cfg=self.cfg, seed=self.seed)
        self._scene.open()
        self._sim = self._scene.sim

    def close(self) -> None:
        if self._scene is not None:
            self._scene.close()
        self._scene = None
        self._sim = None

    # -- the seam -----------------------------------------------------------
    def reset_scene(self, scene: SceneSpec) -> None:
        self.open()
        self._scene.reset_to(scene)

    def execute(self, scene: SceneSpec, strategy: str) -> ExecutionOutcome:
        self.open()
        t0 = time.time()
        result = self._scene.run_strategy(scene, strategy, policy=self.policy)
        out = ExecutionOutcome(
            strategy=str(strategy),
            success=bool(result["success"]),
            duration_s=float(result.get("sim_time_s", time.time() - t0)),
            notes={k: v for k, v in result.items() if k not in ("success", "sim_time_s")},
        )
        self._rollouts.append(
            {"strategy": str(strategy), "margin_m": float(scene.margin_m), "scene_id": int(scene.scene_id),
             "success": out.success, "duration_s": out.duration_s, **out.notes}
        )
        return out

    def say(self, text: str) -> None:
        self.transcript.append(str(text))

    def describe(self) -> Dict[str, Any]:
        return {
            "backend": "isaac",
            "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in self.cfg.__dict__.items()},
            "policy": None if self.policy is None else getattr(self.policy, "name", type(self.policy).__name__),
            "n_rollouts": len(self._rollouts),
        }

    @property
    def rollouts(self) -> List[Dict[str, Any]]:
        return list(self._rollouts)


# ---------------------------------------------------------------------------
# The fidelity check
# ---------------------------------------------------------------------------
def fidelity_report(
    closed_loop: Sequence[Dict[str, Any]],
    grid: SceneGrid,
    *,
    tol_success: float = 0.10,
    tol_duration_s: float = 5.0,
) -> Dict[str, Any]:
    """Does the surrogate reproduce what the closed loop actually does?

    Reported **before** any Tier-1 result, because every one of them assumes it. Per
    (strategy, scene) the check compares the closed loop's realized success rate and mean
    duration against what the surrogate's physics predicts, and flags any cell outside
    tolerance. A failing cell does not invalidate the study design -- it invalidates the
    *physics*, and the fix is to re-fit it from more rollouts, not to widen the tolerance.
    """
    phys = grid.physics
    by: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for r in closed_loop:
        by.setdefault((str(r["strategy"]), int(r["scene_id"])), []).append(r)

    cells: List[Dict[str, Any]] = []
    for (strategy, sid), rs in sorted(by.items()):
        scene = grid.by_id(sid)
        obs_p = sum(1 for r in rs if r.get("success")) / len(rs)
        pred_p = phys.p_success(strategy, scene.margin_m)
        durs = [float(r["duration_s"]) for r in rs if r.get("duration_s") is not None]
        obs_d = statistics.mean(durs) if durs else float("nan")
        pred_d = phys.duration_s(strategy, scene.margin_m)
        cells.append({
            "strategy": strategy, "scene_id": sid, "margin_m": scene.margin_m, "n": len(rs),
            "p_observed": obs_p, "p_predicted": pred_p, "d_p": obs_p - pred_p,
            "duration_observed": obs_d, "duration_predicted": pred_d,
            "d_duration": (obs_d - pred_d) if durs else float("nan"),
            "within_tolerance": bool(abs(obs_p - pred_p) <= tol_success
                                     and (not durs or abs(obs_d - pred_d) <= tol_duration_s)),
        })
    ok = [c for c in cells if c["within_tolerance"]]
    return {
        "n_cells": len(cells),
        "n_within_tolerance": len(ok),
        "fraction_within_tolerance": len(ok) / max(len(cells), 1),
        "max_abs_success_error": max((abs(c["d_p"]) for c in cells), default=float("nan")),
        "mean_abs_success_error": (sum(abs(c["d_p"]) for c in cells) / len(cells)) if cells else float("nan"),
        "tolerances": {"success": tol_success, "duration_s": tol_duration_s},
        "physics_source": phys.source,
        "cells": cells,
        "verdict": (
            "surrogate faithful" if len(ok) == len(cells) and cells
            else ("no closed-loop rollouts" if not cells else "surrogate NOT faithful -- refit the physics")
        ),
    }


__all__ = ["IsaacApparatusConfig", "IsaacApparatus", "fidelity_report"]
