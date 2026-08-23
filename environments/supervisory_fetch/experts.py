"""Scripted experts for the two plan strategies. Pure geometry -- no simulator, fully testable.

Both strategies end in the same terminal state (the target lifted), which is what makes them
comparable: they differ in *how* they get there and in what they leave behind, not in what they
achieve. That is the property the whole study rests on, and it is enforced here rather than
assumed:

``DIRECT`` (strategy B, the efficient one)
    Approach the target from the side away from the blocker, with a lateral detour above the
    table, come down, close, lift. Fast, leaves the workspace untouched, and its success falls
    off as the gap tightens because the fingers need room on the blocker side.

``CLEAR_FIRST`` (strategy A, the cautious one)
    Push the blocker to the drop-off zone with a closed gripper, return, then take the target
    with an unobstructed top-down grasp. Slower and it rearranges the scene, but its success
    barely depends on the gap.

The waypoint lists are position-plus-gripper commands in the **robot base frame**, in the same
format the existing scripted planner consumes, so they drop into the data-collection driver
without a new controller.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import DEFAULT_SUP_SCENE, SupSceneConfig

GRIPPER_OPEN = 0
GRIPPER_CLOSED = 1

STRATEGY_CLEAR_FIRST = "A"
STRATEGY_DIRECT = "B"


@dataclass
class Waypoint:
    """One commanded end-effector pose plus the gripper command to hold while reaching it."""

    xyz: Tuple[float, float, float]
    gripper: int = GRIPPER_OPEN
    #: Free-text phase name; written into the tick log so a rollout can be segmented offline.
    phase: str = ""
    #: Tolerance (m) for calling this waypoint reached. Transit waypoints are loose, grasp
    #: waypoints tight -- a single global tolerance either stalls on transit or fumbles grasps.
    tol_m: float = 0.02

    def to_dict(self) -> Dict[str, Any]:
        return {"xyz": list(self.xyz), "gripper": int(self.gripper), "phase": self.phase, "tol_m": self.tol_m}


@dataclass
class ExpertConfig:
    """Heights and offsets. Defaults carry over the 2026-06-17 grasp fix that took the
    reach-to-grasp expert from ~7% to ~90% success: the end effector goes to the object's own
    height (no +8 cm offset) and closes 4 cm *below* the object centre.
    """

    transit_height_m: float = 0.22       # above the table top (base frame), to clear objects
    pregrasp_height_m: float = 0.12
    #: Grasp depth relative to the object's **top face**. Measured here rather than inherited:
    #: a sweep over -0.04 .. +0.01 m at a 10 cm gap succeeded everywhere down to 0.00 and failed
    #: at +0.01, and the disturbance the closing fingers impart to the arm fell monotonically as
    #: the grasp got shallower (6.4 cm of displacement at -0.04, 2.0 cm at 0.00) -- the deep
    #: settings close partly against the *table*, which levers the arm rather than gripping.
    #: -0.02 keeps the fingers 2 cm below the cube's top face with the least disturbance among
    #: the settings that grip the cube's body.
    #: Getting the reference surface wrong is not a small error: measured against the cube
    #: *centre* instead, the grasp target lands 1.5 cm **below the table**, which the arm
    #: obviously cannot reach, and the symptom is a descent that stalls at a fixed height and
    #: does not care how much step budget it is given.
    grasp_depth_m: float = -0.02
    lift_height_m: float = 0.18
    #: End-effector height for the sweep, above the table top. **Measured, not derived.** The
    #: hand is *closed* for the push, and a closed three-finger JACO reaches the table with its
    #: fingers well before its tool frame does -- so the grasp's descent height is far too low
    #: here. Commanding 0.015 m drove the hand into the table and stalled the approach outright.
    #: A sweep over 0.015 / 0.030 / 0.045 / 0.060 / 0.075 m at a 2 cm gap gave: 0.015 could not
    #: even complete the transit; 0.030 and 0.045 pushed the blocker (0.13 and 0.17 m) but the
    #: subsequent grasp did not lift; 0.060 was above the cube and moved it 0 m; 0.075 produced
    #: the first end-to-end success in this scene, lifting the target 0.18 m.
    push_height_m: float = 0.075
    push_backoff_m: float = 0.09         # how far behind the blocker to start the push
    push_overshoot_m: float = 0.04       # push past the drop-off so the blocker settles clear
    #: Standard deviation of the error in the object positions **the expert believes**, in
    #: metres, applied per rollout. Not domain randomisation: it is the difference between a
    #: scripted expert reading poses from the simulator and any system that has to perceive
    #: them. With oracle poses the success curves are step functions -- measured, each strategy
    #: goes from 0 to 1 within a single centimetre -- and a step makes the transition width, and
    #: therefore the scene coordinate the whole study is defined over, degenerate.
    #:
    #: The level matters and was chosen by measurement, not by taste. At sigma = 1 cm the noise
    #: swamps the geometry it is meant to smooth: the two strategies' feasibility boundaries are
    #: only 2 cm apart, and at that level the measured curves lost almost all of their gap
    #: dependence -- CLEAR_FIRST ran 60-100 percent at *every* gap, including gaps where the
    #: hand cannot fit. 4 mm spreads each step over roughly a centimetre while leaving the
    #: geometric ordering intact, and is a realistic figure for a robot localising a cube.
    pose_noise_m: float = 0.004
    #: Lateral offset for the DIRECT approach, away from the blocker. This is what threads the
    #: gap: the fingers come down on the far side rather than head-on through the blocker.
    direct_detour_m: float = 0.07

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _unit(dx: float, dy: float) -> Tuple[float, float]:
    n = math.hypot(dx, dy)
    return (dx / n, dy / n) if n > 1e-9 else (1.0, 0.0)


def waypoints_direct(
    layout: Dict[str, Any],
    *,
    cfg: Optional[ExpertConfig] = None,
    scene: Optional[SupSceneConfig] = None,
) -> List[Waypoint]:
    """DIRECT: detour around the blocker, come down on the far side, grasp, lift."""
    cfg = cfg or ExpertConfig()
    scene = scene or DEFAULT_SUP_SCENE
    table = 0.0  # base frame: the arm is mounted on the table, so the surface is z = 0
    (tx, ty), tz = layout["target"]["xy"], layout["target"]["z"]
    z_top = tz + 0.5 * float(scene.cube_size_m)   # grasp geometry references the TOP face
    (bx, by) = layout["blocker"]["xy"]

    # Away-from-blocker unit vector: the side the fingers can actually use.
    ax, ay = _unit(tx - bx, ty - by)
    dx, dy = tx + ax * cfg.direct_detour_m, ty + ay * cfg.direct_detour_m

    return [
        Waypoint((dx, dy, table + cfg.transit_height_m), GRIPPER_OPEN, "transit", 0.03),
        Waypoint((dx, dy, table + cfg.pregrasp_height_m), GRIPPER_OPEN, "detour", 0.02),
        Waypoint((tx, ty, z_top + cfg.pregrasp_height_m), GRIPPER_OPEN, "pregrasp", 0.015),
        Waypoint((tx, ty, z_top + cfg.grasp_depth_m), GRIPPER_OPEN, "descend", 0.015),
        Waypoint((tx, ty, z_top + cfg.grasp_depth_m), GRIPPER_CLOSED, "close", 0.015),
        Waypoint((tx, ty, z_top + cfg.lift_height_m), GRIPPER_CLOSED, "lift", 0.03),
    ]


def waypoints_clear_first(
    layout: Dict[str, Any],
    *,
    cfg: Optional[ExpertConfig] = None,
    scene: Optional[SupSceneConfig] = None,
) -> List[Waypoint]:
    """CLEAR_FIRST: sweep the blocker to the drop-off, then take the target top-down."""
    cfg = cfg or ExpertConfig()
    scene = scene or DEFAULT_SUP_SCENE
    table = 0.0  # base frame: the arm is mounted on the table, so the surface is z = 0
    (tx, ty), tz = layout["target"]["xy"], layout["target"]["z"]
    z_top = tz + 0.5 * float(scene.cube_size_m)   # grasp geometry references the TOP face
    (bx, by) = layout["blocker"]["xy"]
    (cx, cy) = layout["clear_dropoff"]["xy"]

    # Push along blocker -> drop-off, starting from behind the blocker so the sweep is a push
    # and not a swat: approaching from the drop-off side would knock the blocker toward the
    # target, which is the opposite of clearing it.
    px, py = _unit(cx - bx, cy - by)

    # No lateral offset: the hand is **closed** for the sweep, so the pusher is a narrow
    # fingertip cluster rather than a 9 cm paddle. Offsetting the push line by half a hand-width
    # was tried on that mistaken picture and made things worse -- the blocker moved 3--4 cm
    # instead of 11--13 cm, because the contact became a graze. What the gap constrains is the
    # *direction* of the sweep, not the width of the tool; see ``layout_for_margin`` for why the
    # drop-off is perpendicular to the target--blocker axis.
    sx, sy = bx - px * cfg.push_backoff_m, by - py * cfg.push_backoff_m
    ex, ey = cx + px * cfg.push_overshoot_m, cy + py * cfg.push_overshoot_m
    zc = table + cfg.push_height_m

    return [
        Waypoint((sx, sy, table + cfg.transit_height_m), GRIPPER_CLOSED, "transit_to_push", 0.03),
        Waypoint((sx, sy, zc), GRIPPER_CLOSED, "push_start", 0.02),
        Waypoint((ex, ey, zc), GRIPPER_CLOSED, "push", 0.025),
        Waypoint((ex, ey, table + cfg.transit_height_m), GRIPPER_CLOSED, "push_retract", 0.03),
        Waypoint((tx, ty, table + cfg.transit_height_m), GRIPPER_OPEN, "transit_to_target", 0.03),
        Waypoint((tx, ty, z_top + cfg.pregrasp_height_m), GRIPPER_OPEN, "pregrasp", 0.015),
        Waypoint((tx, ty, z_top + cfg.grasp_depth_m), GRIPPER_OPEN, "descend", 0.015),
        Waypoint((tx, ty, z_top + cfg.grasp_depth_m), GRIPPER_CLOSED, "close", 0.015),
        Waypoint((tx, ty, z_top + cfg.lift_height_m), GRIPPER_CLOSED, "lift", 0.03),
    ]


def waypoints_for(
    strategy: str,
    layout: Dict[str, Any],
    *,
    cfg: Optional[ExpertConfig] = None,
    scene: Optional[SupSceneConfig] = None,
) -> List[Waypoint]:
    if str(strategy) == STRATEGY_CLEAR_FIRST:
        return waypoints_clear_first(layout, cfg=cfg, scene=scene)
    if str(strategy) == STRATEGY_DIRECT:
        return waypoints_direct(layout, cfg=cfg, scene=scene)
    raise KeyError(f"unknown strategy {strategy!r} (expected 'A' = CLEAR_FIRST or 'B' = DIRECT)")


def approach_clearance_m(layout: Dict[str, Any], *, cfg: Optional[ExpertConfig] = None,
                         scene: Optional[SupSceneConfig] = None) -> float:
    """Finger clearance the DIRECT approach actually gets, in metres.

    The geometric quantity behind ``p_success_B``: how much room there is between the gripper's
    near finger and the blocker when the hand descends on the far side of the target. Negative
    means the hand would have to occupy the blocker's space, which is why the direct strategy
    fails outright at tight gaps. Exposed so that the *fitted* success curve
    (:meth:`vla_lab.supervisory.scenes.ScenePhysics.fit`) can be sanity-checked against the
    geometry that produced it rather than taken on faith.
    """
    cfg = cfg or ExpertConfig()
    scene = scene or DEFAULT_SUP_SCENE
    (tx, ty) = layout["target"]["xy"]
    (bx, by) = layout["blocker"]["xy"]
    centre_gap = math.hypot(tx - bx, ty - by)
    face_gap = centre_gap - scene.cube_size_m
    return float(face_gap - (scene.gripper_width_m - scene.cube_size_m) / 2.0)


__all__ = [
    "GRIPPER_OPEN",
    "GRIPPER_CLOSED",
    "STRATEGY_CLEAR_FIRST",
    "STRATEGY_DIRECT",
    "Waypoint",
    "ExpertConfig",
    "waypoints_direct",
    "waypoints_clear_first",
    "waypoints_for",
    "approach_clearance_m",
]
