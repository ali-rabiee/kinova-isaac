"""Wrist / eye-in-hand camera: rigidly parented to the robot's end-effector.

Unlike top-down/front (raw, unparented USD prims placed at a fixed world
pose), the wrist camera must move with the arm. IsaacLab's ``Camera`` sensor
supports this natively: giving ``CameraCfg`` a ``prim_path`` that is a child
of an existing prim (here, the robot's end-effector link) plus a
``spawn=PinholeCameraCfg(...)`` spawns the camera prim as a rigid child of
that link in one step -- no separate raw-prim-creation function is needed the
way top-down's/front's ``create_*_camera()`` is (confirmed against IsaacLab's
own Franka wrist-camera reference config,
``isaaclab_tasks/manager_based/manipulation/stack/config/franka/stack_ik_rel_visuomotor_env_cfg.py``,
and ``Camera.__init__``, which spawns the prim eagerly when ``spawn`` is not
``None``).

The defaults below are visually confirmed (fingertips at the bottom corners,
clear centre, verified rigid across arm poses). Re-check with
``scripts/debug_cameras.py --cameras wrist`` after any change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.sensors import Camera, CameraCfg


@dataclass
class WristCameraConfig:
    """Eye-in-hand camera, held rigid to the EE link by sync_wrist_camera_to_ee.

    Geometry tuned visually via ``scripts/debug_cameras.py``. Straight view --
    identity rotation, looking down the approach axis -- with one fingertip
    entering each bottom corner and the centre left clear for the object.

    Reading the offset (all in the END-EFFECTOR frame):

    - ``+Z is the approach direction``, NOT height. z = -0.12 sits the camera
      12 cm back along the grasp axis, behind the fingertips (which are at
      z = -0.019). More negative pushes it backwards into the wrist body,
      where the frame goes black.
    - ``y = -0.09`` is the lateral lift clear of the hand housing. Y is the
      right axis for it: the one-vs-two finger split runs along X
      (finger_1 at x=+0.067, the pair at x=-0.062), so offsetting along Y is
      perpendicular to that split and keeps one finger on each side of the
      frame. Offsetting along X instead slides one group into the centre.
    - The sign matters: with convention="ros" (+Y is down in image space) a
      camera at -Y projects the fingers downward, putting them at the BOTTOM
      corners. +Y puts them at the top, which reads upside down.
    - ``offset_rot_wxyz`` is a +12 deg ROLL about the camera's own optical
      axis, which levels the two visible fingertips. It is needed because the
      gripper is mirror-symmetric about the XZ plane (finger_1 at y~0, the
      pair at y=+-0.029), so a camera lifted along Y sits OUT of that symmetry
      plane and sees the two fingers at different heights -- finger_1 projects
      to image y=+0.089 against finger_3's +0.061, a ~12 deg tilt. Rolling by
      that angle cancels it without moving the camera or changing the framing.
    - ``focal_length_mm=10`` against the 20.955 mm aperture is ~93 deg -- wide,
      as eye-in-hand cameras normally are, since the fingers sit centimetres
      away.

    To re-tune: a fingertip at lateral radius r and depth d lands at
    ``r / (d * tan(FOV/2))`` of the half-frame; ~0.85 puts it at the edge.
    """

    parent_body_name: str = "j2n6s300_end_effector"
    prim_leaf: str = "wrist_cam"
    # x=+0.02 centres the gripper horizontally. Only TWO fingertips are ever in
    # frame (the lone one plus one of the pair), and their midpoint -- not the
    # three-finger centroid -- is what reads as the gripper's centre; measured
    # off the render, it sat ~7% right of frame centre at x=0.
    offset_pos: Tuple[float, float, float] = (0.02, -0.09, -0.12)
    # +12 deg roll about the optical axis -- levels the two visible fingertips
    offset_rot_wxyz: Tuple[float, float, float, float] = (0.99452, 0.0, 0.0, 0.10453)
    offset_convention: str = "ros"
    resolution: Tuple[int, int] = (320, 320)
    horizontal_aperture_mm: float = 20.955
    focal_length_mm: float = 9.0
    clipping_range: Tuple[float, float] = (0.01, 2.0)


DEFAULT_WRIST_CAMERA = WristCameraConfig()


def resolve_robot_prim_path(robot: "Articulation") -> str:
    """Same idiom already used elsewhere in this repo (e.g.
    ``scripts/check_gripper_gains.py``) -- read the prim path off the robot's
    own config rather than hardcoding it."""
    prim_path = getattr(getattr(robot, "cfg", None), "prim_path", None)
    if not isinstance(prim_path, str):
        raise RuntimeError("robot.cfg.prim_path is missing; cannot attach wrist camera")
    return prim_path


def find_prim_path_by_name(robot: "Articulation", body_name: str) -> Optional[str]:
    """Recursively search the robot's USD prim tree for a descendant whose
    prim name matches ``body_name`` (e.g. ``"j2n6s300_end_effector"``),
    returning its full prim path, or ``None`` if not found."""
    import omni.usd

    robot_prim_path = resolve_robot_prim_path(robot)
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(robot_prim_path)

    def _walk(prim):
        for child in prim.GetChildren():
            if child.GetName() == body_name:
                return child.GetPath().pathString
            found = _walk(child)
            if found is not None:
                return found
        return None

    return _walk(root)


def list_all_descendant_prim_names(robot: "Articulation") -> list[str]:
    """Debug helper: every descendant prim name under the robot's prim path,
    for diagnosing a failed ``find_prim_path_by_name`` lookup."""
    import omni.usd

    robot_prim_path = resolve_robot_prim_path(robot)
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(robot_prim_path)

    names: list[str] = []

    def _walk(prim):
        for child in prim.GetChildren():
            names.append(child.GetName())
            _walk(child)

    _walk(root)
    return names


def build_wrist_camera_cfg(robot: "Articulation", camera_cfg: "WristCameraConfig") -> "CameraCfg":
    """Build (but do not instantiate) a ``CameraCfg`` that spawns the wrist
    camera as a rigid child of ``camera_cfg.parent_body_name`` under the
    robot's prim. Caller does ``Camera(cfg=build_wrist_camera_cfg(robot, cfg))``
    (or just call ``build_wrist_camera_sensor`` below)."""
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg

    # Deliberately a STANDALONE prim, not a child of the end-effector prim.
    #
    # Parenting it under the EE prim looks right and even spawns in the right
    # place, but it does not track the arm: PhysX keeps link transforms in its
    # own buffers and never writes them back down the USD hierarchy, so the
    # camera's world transform is recomputed forever as (frozen parent x local)
    # -- it sits motionless while the arm moves, and any world pose written to
    # it is immediately overwritten by that recomputation.
    #
    # Unparented, its world pose is ours to set, which sync_wrist_camera_to_ee
    # does every step from the link's live physics pose. offset_pos/offset_rot
    # are applied there instead of here, so they keep their meaning: a rigid
    # mount expressed in the end-effector frame.
    prim_path = f"/World/Origin1/{camera_cfg.prim_leaf}"

    return CameraCfg(
        prim_path=prim_path,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=camera_cfg.focal_length_mm,
            horizontal_aperture=camera_cfg.horizontal_aperture_mm,
            clipping_range=camera_cfg.clipping_range,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention=camera_cfg.offset_convention,
        ),
        data_types=["rgb"],
        width=camera_cfg.resolution[0],
        height=camera_cfg.resolution[1],
    )


def sync_wrist_camera_to_ee(
    robot: "Articulation", camera: "Camera", camera_cfg: "WristCameraConfig"
) -> None:
    """Write the camera's world pose from the parent link's CURRENT physics pose.

    MUST be called every step, before the render that produces the frame.
    Skipping it does not error -- it silently yields a camera sitting still in
    space while the arm moves, producing plausible-looking but wrong frames.

    The camera prim is intentionally NOT parented to the robot (see
    ``build_wrist_camera_cfg``), because a child prim's world transform is
    recomputed from its parent every step and articulation motion never
    reaches the USD hierarchy -- so a parented camera stays frozen and
    ignores any pose written to it. Unparented, this function is what makes
    the mount rigid: it composes the configured EE-frame offset onto the
    link's live physics pose and writes the result.
    """
    import torch
    from isaaclab.utils.math import combine_frame_transforms

    ids, _ = robot.find_bodies([camera_cfg.parent_body_name])
    ee_id = int(ids[0])
    ee_pos = robot.data.body_pose_w[:, ee_id, 0:3]
    ee_quat = robot.data.body_pose_w[:, ee_id, 3:7]

    device = ee_pos.device
    n = ee_pos.shape[0]
    off_pos = torch.tensor(camera_cfg.offset_pos, dtype=ee_pos.dtype, device=device).repeat(n, 1)
    off_quat = torch.tensor(
        camera_cfg.offset_rot_wxyz, dtype=ee_quat.dtype, device=device
    ).repeat(n, 1)

    pos_w, quat_w = combine_frame_transforms(ee_pos, ee_quat, off_pos, off_quat)
    camera.set_world_poses(pos_w, quat_w, convention=camera_cfg.offset_convention)


def build_wrist_camera_sensor(robot: "Articulation", camera_cfg: "WristCameraConfig | None" = None) -> "Camera":
    """Convenience one-call helper mirroring ``topdown.build_topdown_camera_sensor``
    / ``front.build_front_camera_sensor``, for the uniform registry dispatch
    in ``registry.build_camera``."""
    from isaaclab.sensors import Camera

    cfg = camera_cfg or DEFAULT_WRIST_CAMERA
    return Camera(cfg=build_wrist_camera_cfg(robot, cfg))
