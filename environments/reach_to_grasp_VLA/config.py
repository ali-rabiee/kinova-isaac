from dataclasses import dataclass
from typing import Tuple

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_assets import KINOVA_JACO2_N6S300_CFG
import numpy as np


BASE_ROBOT_CFG = KINOVA_JACO2_N6S300_CFG


@dataclass
class SceneConfig:
    """High-level scene configuration for the JACO2 reach-to-grasp demo."""

    num_origins: int = 1
    origin_spacing: float = 2.0

    table_usd_path: str = (
        f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd"
    )
    table_scale: Tuple[float, float, float] = (1.5, 2.0, 1.0)
    table_translation: Tuple[float, float, float] = (0.0, 0.0, 0.8)

    robot_prim_path: str = "/World/Origin1/Robot"
    robot_base_height: float = 0.8

    # Optional: initial joint configuration (by joint name or regex)
    # Example: {"j2n6s300_joint_[1-6]": 0.0} or per-joint mapping
    robot_default_joint_pos: dict[str, float] | None = None


@dataclass
class CameraConfig:
    """Camera placement for the demo viewport."""

    eye: Tuple[float, float, float] = (3.5, 0.0, 3.2)
    target: Tuple[float, float, float] = (0.0, 0.0, 0.5)


@dataclass
class TopDownCameraConfig:
    """Top-down camera configuration for VLA environment.

    IMPORTANT: this config is shared by data collection (vla_v1/collect_v3) AND
    eval (vla_lab/eval_isaaclab.py). It is part of the trained model's contract:
    do NOT mix sessions recorded with different camera configs in one training
    run, and do not change it between collecting data and evaluating a model.

    To adjust the camera view:
    - position: (x, y, z) - Camera location. Higher z = more overhead view.
    - target: (x, y, z) - Point the camera looks at (informational; the camera
      looks straight down).
    - fov: Field of view in degrees. Higher = wider view, lower = zoomed in.
    """

    prim_path: str = "/World/Origin1/TopDownCamera"
    # 2026-06-11: z lowered 4.0 → 2.2 (~1.4 m above the table top). At z=4.0 the
    # workspace covered only ~17% of the frame and an 8 cm box was ~4 px after the
    # 224×224 training resize — too small for reliable color/identity grounding.
    # At z=2.2 the spawn region + robot fill most of the frame (box ≈ 10 px @224)
    # while domain-rand jitter (z ±0.10 m, yaw ±20°, fov ±5°) still keeps the full
    # workspace in view. Sessions recorded at z=4.0 (e.g. session_20260607_115038)
    # are visually incompatible with this framing.
    position: Tuple[float, float, float] = (0.4, 0.0, 2.2)  # directly above the workspace center
    target: Tuple[float, float, float] = (0.4, 0.0, 0.8)  # workspace center at table surface (table top at z~0.8)
    resolution: Tuple[int, int] = (640, 640)
    fov: float = 65.0  # degrees


@dataclass
class WristCameraConfig:
    """Eye-in-hand (wrist) camera rigidly mounted on the end-effector link.

    Added 2026-07 for the wrist-camera data profile (collect_v4 / profile vla_v4).
    Like TopDownCameraConfig this is part of a trained model's contract: sessions
    recorded with different wrist mounts/FOVs must not be mixed in one training run.

    The camera prim is created as a CHILD of the EE link prim, so it tracks the
    arm automatically. `offset_pos` / `offset_rpy_deg` are the mount calibration
    (hand-eye extrinsics) expressed in the EE-link frame; on the real robot,
    replace them with your measured hand-eye calibration. Both are dumped into
    each episode's `cameras.json` together with the pinhole intrinsics.

    Frame conventions: USD cameras look down their local -Z with +Y up. The EE
    frame `j2n6s300_end_effector` sits at fingertip working level with +Z as the
    tool/approach axis, so `offset_rpy_deg=(180, 0, 0)` points the camera along
    the approach direction (at whatever the gripper is about to grasp) and
    `offset_pos` pulls it back toward the wrist and off the finger axis.
    """

    prim_path: str = "/World/Origin1/Robot/j2n6s300_end_effector/WristCamera"
    parent_link: str = "j2n6s300_end_effector"
    # Mount pose in the EE-link frame (meters / degrees, applied as USD RotateXYZ).
    offset_pos: Tuple[float, float, float] = (0.0, -0.055, -0.11)
    offset_rpy_deg: Tuple[float, float, float] = (180.0, 0.0, 0.0)
    resolution: Tuple[int, int] = (640, 640)
    # Wide FOV, RealSense-D405-like (the common real wrist camera).
    fov: float = 87.0


DEFAULT_SCENE = SceneConfig(
    robot_default_joint_pos={
        "j2n6s300_joint_1": 0*np.pi, # j2n6s300_joint_1: [-inf, inf]
        "j2n6s300_joint_2": np.pi, # j2n6s300_joint_2: [0.820, 5.463]
        "j2n6s300_joint_3": 1.8*np.pi, # j2n6s300_joint_3: [0.332, 5.952]
        "j2n6s300_joint_4": 0*np.pi, # j2n6s300_joint_4: [-inf, inf]
        "j2n6s300_joint_5": 1.75*np.pi, # j2n6s300_joint_5: [-inf, inf]
        "j2n6s300_joint_6": 0.5*np.pi, # j2n6s300_joint_6: [-inf, inf]
        "j2n6s300_joint_finger_1": 0.0, # j2n6s300_joint_finger_1: [0.000, 1.510]
        "j2n6s300_joint_finger_2": 0.0, # j2n6s300_joint_finger_2: [0.000, 1.510]
        "j2n6s300_joint_finger_3": 0.0, # j2n6s300_joint_finger_3: [0.000, 1.510]
        "j2n6s300_joint_finger_tip_1": 0.0, # j2n6s300_joint_finger_tip_1: [0.000, 2.000]
        "j2n6s300_joint_finger_tip_2": 0.0, # j2n6s300_joint_finger_tip_2: [0.000, 2.000]
        "j2n6s300_joint_finger_tip_3": 0.0, # j2n6s300_joint_finger_tip_3: [0.000, 2.000]
        # You can also use regex like "j2n6s300_joint_[1-6]": 0.0
    }
)

DEFAULT_CAMERA = CameraConfig()
DEFAULT_TOP_DOWN_CAMERA = TopDownCameraConfig()
DEFAULT_WRIST_CAMERA = WristCameraConfig()


