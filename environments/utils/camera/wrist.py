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

Placeholder geometry only: ``offset_pos``/``offset_rot_wxyz`` have NOT been
visually verified, and ``parent_body_name`` has only been confirmed as a
PhysX *articulation body* name (via ``robot.find_bodies``) -- not necessarily
the identical USD prim-tree name. ``find_prim_path_by_name`` below resolves
(and, if it can't, reports) the actual prim path live rather than assuming a
fixed relative path. Run ``scripts/debug_cameras.py --cameras wrist`` and
inspect the saved frames (and the printed descendant list if attachment
fails) before trusting these defaults for real data collection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.sensors import Camera, CameraCfg


@dataclass
class WristCameraConfig:
    """Eye-in-hand camera, spawned as a rigid child of the EE link."""

    parent_body_name: str = "j2n6s300_end_effector"
    prim_leaf: str = "wrist_cam"
    offset_pos: Tuple[float, float, float] = (0.0, 0.0, 0.05)
    offset_rot_wxyz: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    offset_convention: str = "ros"
    resolution: Tuple[int, int] = (320, 320)
    horizontal_aperture_mm: float = 20.955
    focal_length_mm: float = 18.0
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

    parent_path = find_prim_path_by_name(robot, camera_cfg.parent_body_name)
    if parent_path is None:
        raise RuntimeError(
            f"could not find a prim named {camera_cfg.parent_body_name!r} under the robot's "
            "USD tree; run scripts/debug_cameras.py to print the full descendant list and "
            "update WristCameraConfig.parent_body_name accordingly"
        )
    prim_path = f"{parent_path}/{camera_cfg.prim_leaf}"

    return CameraCfg(
        prim_path=prim_path,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=camera_cfg.focal_length_mm,
            horizontal_aperture=camera_cfg.horizontal_aperture_mm,
            clipping_range=camera_cfg.clipping_range,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=camera_cfg.offset_pos,
            rot=camera_cfg.offset_rot_wxyz,
            convention=camera_cfg.offset_convention,
        ),
        data_types=["rgb"],
        width=camera_cfg.resolution[0],
        height=camera_cfg.resolution[1],
    )


def build_wrist_camera_sensor(robot: "Articulation", camera_cfg: "WristCameraConfig | None" = None) -> "Camera":
    """Convenience one-call helper mirroring ``topdown.build_topdown_camera_sensor``
    / ``front.build_front_camera_sensor``, for the uniform registry dispatch
    in ``registry.build_camera``."""
    from isaaclab.sensors import Camera

    cfg = camera_cfg or DEFAULT_WRIST_CAMERA
    return Camera(cfg=build_wrist_camera_cfg(robot, cfg))
