"""Observation / action layout aligned with `lerobot/smolvla_base` for fine-tuning.

The base policy expects (see its Hub `config.json`):
  - `observation.state`: 6 floats (MEAN_STD normalized in LeRobot)
  - `observation.images.camera{1,2,3}`: 3×256×256 images (CHW after dataset load)
  - `action`: 6 floats (MEAN_STD)

Our Kinova logs carry 7D `action_from_prev` (Δxyz, Δrotvec×3, gripper) and one
or two RGB cameras. Slot routing: legacy overhead-only exports duplicate the
same image to all three views; `--wrist` exports route the wrist view into
`camera2` (camera1/camera3 stay overhead). Actions export as **6D delta pose**
without the gripper channel. Gripper is **not** controlled by SmolVLA in this bridge; use
TinyVLA or extend the action head if you need learned gripper.

State in the export is **absolute** end-effector pose in the robot base frame:
  - first 3: position (m)
  - last 3: rotation vector (axis–angle, rad) from `ee_quat_wxyz`
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

# LeRobot / policy keys (must match pretrained smolvla_base).
CAMERA_KEYS = (
    "observation.images.camera1",
    "observation.images.camera2",
    "observation.images.camera3",
)
STATE_KEY = "observation.state"
ACTION_KEY = "action"
IMAGE_HWC_SHAPE = (256, 256, 3)  # height, width, channels (dataset convention)
STATE_DIM = 6
ACTION_DIM = 6


def quat_wxyz_to_rotvec(quat: Sequence[float]) -> np.ndarray:
    """Convert unit quaternion (w, x, y, z) to rotation vector (3,)."""

    w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    norm = (w * w + x * x + y * y + z * z) ** 0.5
    if norm > 1e-9:
        w, x, y, z = w / norm, x / norm, y / norm, z / norm
    # Shortest path: ensure w >= 0
    if w < 0:
        w, x, y, z = -w, -x, -y, -z
    t = float(np.sqrt(max(0.0, 1.0 - w * w)))
    if t < 1e-8:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * float(np.arctan2(t, abs(w)))
    s = angle / t
    return np.array([x * s, y * s, z * s], dtype=np.float32)


def pack_state_xyz_quat(
    ee_pos_b: Tuple[float, float, float],
    ee_quat_wxyz: Tuple[float, float, float, float],
) -> np.ndarray:
    """6D state vector for LeRobot parquet rows."""

    p = np.array([ee_pos_b[0], ee_pos_b[1], ee_pos_b[2]], dtype=np.float32)
    r = quat_wxyz_to_rotvec(ee_quat_wxyz)
    return np.concatenate([p, r], axis=0)


def pack_action_six(
    action_from_prev: Tuple[float, float, float, float, float, float, int] | None,
) -> np.ndarray | None:
    """First 6 dims of Kinova `action_from_prev`; returns None if missing."""

    if action_from_prev is None:
        return None
    return np.array([float(action_from_prev[i]) for i in range(6)], dtype=np.float32)


def smolvla_base_dataset_features(*, use_video: bool = False) -> dict:
    """Feature dict for `LeRobotDataset.create` matching `lerobot/smolvla_base` shapes."""

    dtype_img = "video" if use_video else "image"
    h, w, c = IMAGE_HWC_SHAPE
    state_names = ["ee_pos_x", "ee_pos_y", "ee_pos_z", "ee_rotvec_x", "ee_rotvec_y", "ee_rotvec_z"]
    action_names = ["delta_x", "delta_y", "delta_z", "delta_rot_x", "delta_rot_y", "delta_rot_z"]
    feats: dict = {
        STATE_KEY: {"dtype": "float32", "shape": (STATE_DIM,), "names": state_names},
        ACTION_KEY: {"dtype": "float32", "shape": (ACTION_DIM,), "names": action_names},
    }
    for cam in ("camera1", "camera2", "camera3"):
        feats[f"observation.images.{cam}"] = {
            "dtype": dtype_img,
            "shape": (h, w, c),
            "names": ["height", "width", "channels"],
        }
    return feats


def kinematic_contract_readme() -> str:
    return __doc__ or ""
