"""Scene configuration for the Phase 0 bilateral-choice apparatus twin.

Deliberately **Isaac-free at import time** (unlike
``environments/reach_to_grasp_VLA/config.py``, which imports ``isaaclab`` at module level):
:mod:`vla_lab.rehab.apparatus.isaac_apparatus` names this package in a lazy import, and the
Phase 0 test suite must stay runnable on a machine with no simulator. The USD asset paths are
resolved in :mod:`environments.bilateral_choice.twin`, inside the functions that need them.

**The VLA contract is not disturbed.** ``environments/reach_to_grasp_VLA/config.py`` is
untouched; the Phase 0 wrist camera gets its own class here with the re-aimed mount, because
the VLA mount points along the *grasp approach* axis and Phase 0 needs to watch a human hand
arrive (``rehab.md`` §8, W11).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class TwinSceneConfig:
    """Table, arm mounting pose, and the seated-participant proxy."""

    # Table top height in world coordinates (matches the VLA scene's 0.8 m).
    table_height_m: float = 0.8
    table_scale: Tuple[float, float, float] = (1.5, 2.0, 1.0)
    table_translation: Tuple[float, float, float] = (0.0, 0.0, 0.8)

    robot_prim_path: str = "/World/Phase0/Robot"
    #: The JACO 2 base, in **world** coordinates. The participant frame is placed relative to
    #: this by :class:`~environments.bilateral_choice.twin.BilateralChoiceTwin` using the
    #: contract's ``robot_base_in_participant``.
    robot_base_world: Tuple[float, float, float] = (0.0, 0.0, 0.8)
    robot_default_joint_pos: Optional[Dict[str, float]] = None

    #: Seated-participant proxy: a capsule for the torso plus a head sphere. Rendered so a
    #: reviewer can see what the clearance checks are actually checking.
    participant_prim_path: str = "/World/Phase0/ParticipantProxy"
    participant_torso_radius_m: float = 0.22
    participant_torso_height_m: float = 0.75
    #: Torso centre behind the participant-frame origin: the origin is the sternum *projection*
    #: (the front of the chest), so a torso centred on it would bulge into the workspace.
    participant_torso_center_offset_m: float = -0.13
    participant_head_radius_m: float = 0.12

    #: Target markers, one per contract target, drawn on the table plane.
    target_marker_prim_root: str = "/World/Phase0/Targets"
    target_marker_radius_m: float = 0.03
    target_marker_height_m: float = 0.004


@dataclass
class TwinFrontCameraConfig:
    """The front camera: the **primary** view for arm-choice classification (§12.5).

    Positioned beyond the far edge of the table looking back at the participant, so both hands
    and the midline are in frame. The wrist camera is the *confirming* view, not the primary
    one — a mount that cannot see a hand approaching from the far side is a known risk (§14).
    """

    prim_path: str = "/World/Phase0/FrontCamera"
    position: Tuple[float, float, float] = (1.05, 0.0, 1.55)
    target: Tuple[float, float, float] = (0.30, 0.0, 0.80)
    resolution: Tuple[int, int] = (960, 540)
    fov: float = 78.0


@dataclass
class TwinWristCameraConfig:
    """The Phase 0 wrist mount — **re-aimed** relative to the VLA data-collection mount.

    The VLA mount (``offset_pos=(0, -0.055, -0.11)``, ``rpy=(180, 0, 0)``, FOV 87) points along
    the gripper's approach axis: correct for watching a box being grasped, not obviously
    correct for watching a participant's hand arrive at a presented target. Here the camera is
    tilted back toward the participant. **Whether this framing actually works is an open
    question** that the twin dry-run (W10) and hardware bring-up (W11) answer; the revised
    hand-eye calibration is then recorded in ``contract.json``.
    """

    prim_path: str = "/World/Phase0/Robot/j2n6s300_end_effector/WristCamera"
    parent_link: str = "j2n6s300_end_effector"
    offset_pos: Tuple[float, float, float] = (0.0, -0.04, -0.06)
    #: Tilted ~35 deg back from the approach axis, toward where a hand arrives from.
    offset_rpy_deg: Tuple[float, float, float] = (145.0, 0.0, 0.0)
    resolution: Tuple[int, int] = (640, 480)
    fov: float = 87.0


@dataclass
class TwinConfig:
    scene: TwinSceneConfig = field(default_factory=TwinSceneConfig)
    front_camera: TwinFrontCameraConfig = field(default_factory=TwinFrontCameraConfig)
    wrist_camera: TwinWristCameraConfig = field(default_factory=TwinWristCameraConfig)
    #: EE height above the table when presenting a target (the puck sits on the table).
    present_height_m: float = 0.06
    #: Home pose in the participant frame: parked away from the reach paths.
    home_xy_participant_m: Tuple[float, float] = (0.60, 0.0)


DEFAULT_TWIN = TwinConfig()

__all__ = [
    "TwinSceneConfig",
    "TwinFrontCameraConfig",
    "TwinWristCameraConfig",
    "TwinConfig",
    "DEFAULT_TWIN",
]
