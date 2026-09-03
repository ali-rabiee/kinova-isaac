"""Scene geometry for the supervisory-fetch task.

The whole environment exists to make **one** scalar controllable and everything else fixed: the
clearance gap between the blocking object and the target, which is the raw margin ``m`` that
:mod:`vla_lab.supervisory.scenes` turns into the ambiguity coordinate. Every other degree of
freedom -- object sizes, table height, camera poses, the distractors -- is held constant across
scenes, because a scene set in which the tight cases also happen to be the far ones, or the
cluttered ones, would confound geometry with difficulty and the estimand would stop meaning
what it says.

Cameras are reused from the VLA data-collection contract wherever possible (same top-down pose
and FOV) so that a policy trained on the existing 100-episode reach-to-grasp corpus can be
fine-tuned here without a camera-contract break.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class SupSceneConfig:
    """Table, robot, target, blocker, and distractors."""

    num_origins: int = 1
    table_scale: Tuple[float, float, float] = (1.5, 2.0, 1.0)
    table_translation: Tuple[float, float, float] = (0.0, 0.0, 0.8)
    #: World height of the table top. The Kinova base is mounted **on** the table at this same
    #: height, so in the robot's own base frame -- which is the frame the controller, the
    #: waypoints, and ``get_ee_pos_base_frame`` all speak -- the table top is at z = 0. This
    #: value is therefore used for exactly one thing: converting a base-frame layout into world
    #: coordinates when teleporting an object. Everything else is base frame.
    table_height_m: float = 0.8

    robot_prim_path: str = "/World/Origin1/Robot"
    robot_base_height: float = 0.8

    objects_root: str = "/World/Origin1/Objects"
    #: Cube edge length. The three-finger JACO hand needs roughly this much clearance on the
    #: approach side, which is what makes a gap of a few centimetres genuinely decisive.
    cube_size_m: float = 0.05
    gripper_width_m: float = 0.09

    #: Target sits at a fixed spot in the robot base frame; only the blocker moves.
    #:
    #: The corridor runs along y = -0.10, comfortably inside the measured reachable envelope
    #: (see ``REACHABLE_X_M``, and the correction recorded there about how this offset was
    #: originally chosen). The measured scene physics was fitted with the objects here, so moving
    #: them invalidates it: re-run ``sup_sweep.sh --fit`` and ``sup_frames.sh`` if you do.
    target_xy: Tuple[float, float] = (0.48, -0.10)
    #: Direction (radians in the table plane) from target to blocker. Fixed at "toward the
    #: robot", so a tight gap always blocks the natural approach rather than an arbitrary side.
    blocker_bearing_rad: float = math.pi
    #: How far the blocker is swept, perpendicular to the target--blocker axis. The drop-off
    #: point itself is computed per scene in :func:`layout_for_margin`, which explains why.
    clear_dropoff_distance_m: float = 0.24

    n_distractors: int = 2
    distractor_ring_radius_m: float = 0.20
    distractor_min_sep_m: float = 0.09

    #: Domain randomisation, applied per episode. Deliberately small on anything that would
    #: move the margin: the margin is the independent variable and must not be jittered.
    dr_position_jitter_m: float = 0.008
    dr_yaw_jitter_rad: float = 0.25
    dr_margin_jitter_m: float = 0.0

    target_color: str = "red"
    blocker_color: str = "blue"
    distractor_colors: Tuple[str, ...] = ("green", "yellow", "white")
    #: A visual-only slab laid on the table top with this diffuse colour, to change the
    #: background texture a camera sees without touching the physics. ``None`` = bare table.
    table_overlay_color: Optional[Tuple[float, float, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


#: The **held-out** scene configuration for the distribution-shift evaluation
#: (``run_frames.py --shift``): different object colours and size, an altered table surface,
#: one more distractor, and -- captured alongside -- a second top-down camera pose. Nothing
#: here is trained on. The margin definition and the corridor are unchanged, so every scene id
#: still realises the same clearance gap and the same coordinate ``c``.
SHIFTED_SUP_SCENE_KW: Dict[str, Any] = dict(
    cube_size_m=0.045,
    target_color="magenta",
    blocker_color="teal",
    distractor_colors=("white", "orange", "green"),
    n_distractors=3,
    table_overlay_color=(0.55, 0.42, 0.30),
)


@dataclass
class SupTopDownCameraConfig:
    """Top-down view. Pose and FOV match the VLA data-collection contract exactly, so a policy
    pretrained on the existing reach-to-grasp corpus transfers without a camera-contract break.
    """

    prim_path: str = "/World/Origin1/TopDownCamera"
    position: Tuple[float, float, float] = (0.4, 0.0, 2.2)
    target: Tuple[float, float, float] = (0.4, 0.0, 0.8)
    resolution: Tuple[int, int] = (640, 640)
    fov: float = 65.0


@dataclass
class SupWristCameraConfig:
    """Eye-in-hand view, same mount as the collect_v4 contract."""

    prim_path_suffix: str = "WristCamera"
    ee_link_name: str = "j2n6s300_end_effector"
    offset_pos: Tuple[float, float, float] = (0.0, -0.055, -0.11)
    offset_rpy_deg: Tuple[float, float, float] = (180.0, 0.0, 0.0)
    resolution: Tuple[int, int] = (320, 320)
    fov: float = 87.0


@dataclass
class SupFigureCameraConfig:
    """A three-quarter view used only for figures.

    The top-down contract camera is what the policy sees and must not be moved; it is also a
    poor way to show a person what the task looks like, because a clearance gap read from
    directly above carries no sense of the arm reaching past an obstacle. This camera exists so
    the paper can show the scene the way a supervisor would actually see it, and it is never
    used for training or evaluation.
    """

    prim_path: str = "/World/Origin1/FigureCamera"
    position: Tuple[float, float, float] = (1.85, -1.30, 1.95)
    target: Tuple[float, float, float] = (0.30, 0.02, 0.88)
    resolution: Tuple[int, int] = (960, 640)
    fov: float = 46.0


DEFAULT_SUP_SCENE = SupSceneConfig()
DEFAULT_SUP_TOPDOWN_CAMERA = SupTopDownCameraConfig()
#: A second overhead pose for the distribution-shift atlas: 7 cm of translation and a few
#: degrees of tilt off the collection contract, slightly narrower field of view. Still overhead.
SHIFTED_SUP_TOPDOWN_CAMERA = SupTopDownCameraConfig(
    prim_path="/World/Origin1/TopDownCameraShift",
    position=(0.47, -0.06, 2.05),
    target=(0.42, -0.03, 0.8),
    resolution=(640, 640),
    fov=61.0,
)
DEFAULT_SUP_FIGURE_CAMERA = SupFigureCameraConfig()
DEFAULT_SUP_WRIST_CAMERA = SupWristCameraConfig()


#: The arm's **measured** reachable envelope at grasp height, in the robot base frame, with the
#: tool held pointing down. Produced by ``vla_lab.supervisory.run_reach`` on an empty table and
#: recorded here so that a change to the scene geometry is checked against a measurement rather
#: than against an intuition. Measured over 120 poses at two heights: 55/60 reached at
#: ``z = 0.02`` m and 57/60 at ``z = 0.12`` m, to a median 1.9 cm, across ``x`` in [0.22, 0.66].
#:
#: **A correction worth keeping in view.** An earlier version of this probe returned to the home
#: pose by jogging rather than by resetting the episode, and that decays: the residual home error
#: grew from 1.8 cm to 6.5 cm and then to 14--18 cm as the probe entered awkward regions, after
#: which every measurement was of the arm's recovery rather than its reach. Under that version
#: the row along ``y = 0`` came back marginal along its whole length, and the scene corridor was
#: moved to ``y = -0.10`` on the strength of it. With a hard reset per point the same row is
#: fully reachable. The corridor stays at ``y = -0.10`` because it is comfortably inside the
#: envelope and because the measured scene physics was fitted there --- not because ``y = 0``
#: cannot be reached.
REACHABLE_X_M: Tuple[float, float] = (0.22, 0.62)
REACHABLE_ABS_Y_MIN_M: float = 0.0
REACHABLE_ABS_Y_MAX_M: float = 0.22


def reachable_at_grasp_height(x: float, y: float) -> bool:
    """Is this table position inside the measured envelope? Used by the geometry gate."""
    lo, hi = REACHABLE_X_M
    return bool(lo <= float(x) <= hi and REACHABLE_ABS_Y_MIN_M <= abs(float(y)) <= REACHABLE_ABS_Y_MAX_M)


def layout_for_margin(
    margin_m: float,
    *,
    cfg: Optional[SupSceneConfig] = None,
    n_distractors: Optional[int] = None,
    rng=None,
) -> Dict[str, Any]:
    """Object placement realising a given clearance gap, in the robot base frame.

    ``margin_m`` is the **free gap between the two cube faces**, so the centre-to-centre
    distance is ``margin + cube_size``. That definition is the one the physics model and the
    scene coordinate both use, and it is the one a person looking at the scene can see: a gap of
    2 cm with a 9 cm hand is visibly impossible to reach through, and a gap of 12 cm is visibly
    fine. Getting this definition consistent between the simulator, the value model, and the
    narration is what keeps ``c`` meaning the same thing everywhere.

    **Frame: the robot base frame**, in which the table top is ``z = 0`` because the arm is
    mounted on the table. Returned ``z`` values are therefore half a cube above zero, not
    ``0.8 + half a cube``. Mixing the two frames put every waypoint 0.8 m above the arm's reach
    and produced a sweep in which all 6 rollouts timed out on their first move.
    """
    cfg = cfg or DEFAULT_SUP_SCENE
    import random as _random

    rng = rng or _random.Random(0)
    tx, ty = cfg.target_xy
    d = float(margin_m) + cfg.cube_size_m
    bx = tx + d * math.cos(cfg.blocker_bearing_rad)
    by = ty + d * math.sin(cfg.blocker_bearing_rad)

    n = int(cfg.n_distractors if n_distractors is None else n_distractors)
    distractors: List[Tuple[float, float]] = []
    attempts = 0
    while len(distractors) < n and attempts < 200:
        attempts += 1
        a = rng.uniform(0.0, 2.0 * math.pi)
        r = cfg.distractor_ring_radius_m * rng.uniform(0.8, 1.4)
        px, py = tx + r * math.cos(a), ty + r * math.sin(a)
        ok = all(math.hypot(px - qx, py - qy) >= cfg.distractor_min_sep_m for qx, qy in [(tx, ty), (bx, by)] + distractors)
        # Never let a distractor sit inside the gap: it would change the margin the scene is
        # supposed to realise, which is the one thing this function has to get right.
        if ok and _point_gap_clear(px, py, tx, ty, bx, by, cfg.cube_size_m):
            distractors.append((px, py))

    # **The drop-off is perpendicular to the target--blocker axis, and computed per scene.**
    #
    # A drop-off at a fixed table position makes the sweep direction depend on the gap, and at
    # tight gaps that direction runs the hand along the corridor between the two cubes -- which
    # is precisely the corridor the gap has closed. Measured with a fixed drop-off: the sweep
    # moved the blocker 11--13 cm at wide gaps and **5 mm** at a 2 cm gap, so "clear first" did
    # not clear anything exactly where clearing matters most, and the strategy inherited the
    # direct approach's feasibility boundary. Two strategies with the same boundary leave the
    # study with no ambiguous region at all.
    #
    # Sweeping *sideways* instead -- perpendicular to the line joining the cubes -- keeps the
    # hand at the blocker's own distance from the target for the whole push, so clearing stays
    # feasible at every gap while reaching directly does not. That asymmetry is the scene's
    # entire reason for existing.
    ax, ay = (bx - tx), (by - ty)
    n = math.hypot(ax, ay) or 1.0
    perp_x, perp_y = -ay / n, ax / n            # +90 degrees from target -> blocker
    if perp_y < 0:                              # keep the drop-off on the reachable +y side
        perp_x, perp_y = -perp_x, -perp_y
    dropx, dropy = bx + perp_x * cfg.clear_dropoff_distance_m, by + perp_y * cfg.clear_dropoff_distance_m

    z = cfg.cube_size_m / 2.0  # base frame: the table top is z = 0
    return {
        "margin_m": float(margin_m),
        "target": {"xy": (float(tx), float(ty)), "z": z, "color": cfg.target_color},
        "blocker": {"xy": (float(bx), float(by)), "z": z, "color": cfg.blocker_color},
        "distractors": [{"xy": (float(x), float(y)), "z": z} for x, y in distractors],
        "clear_dropoff": {"xy": (float(dropx), float(dropy)), "z": z},
        "center_distance_m": float(d),
    }


def _point_gap_clear(px: float, py: float, tx: float, ty: float, bx: float, by: float, pad: float) -> bool:
    """True when (px, py) is not inside the corridor between target and blocker."""
    vx, vy = bx - tx, by - ty
    L2 = vx * vx + vy * vy
    if L2 <= 1e-12:
        return True
    t = max(0.0, min(1.0, ((px - tx) * vx + (py - ty) * vy) / L2))
    cx, cy = tx + t * vx, ty + t * vy
    return math.hypot(px - cx, py - cy) > pad


__all__ = [
    "SupSceneConfig",
    "SupTopDownCameraConfig",
    "SupWristCameraConfig",
    "DEFAULT_SUP_SCENE",
    "DEFAULT_SUP_TOPDOWN_CAMERA",
    "DEFAULT_SUP_FIGURE_CAMERA",
    "DEFAULT_SUP_WRIST_CAMERA",
    "SupFigureCameraConfig",
    "layout_for_margin",
]
