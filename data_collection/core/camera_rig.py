"""Multi-camera capture + calibration helpers for data-collection profiles.

Used by the wrist-camera path (profile ``vla_v4`` / ``collect_v4.sh``). The
legacy top-down capture code in ``vla_v1`` is intentionally left inline and
untouched — sessions recorded by ``collect_v3.sh`` stay byte-identical.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


def save_rgb_tick_image(
    camera_sensor,
    *,
    images_dir: Path,
    filename: str,
) -> Optional[str]:
    """Grab the sensor's current RGB frame and save it as ``images/<filename>``.

    Returns the tick-relative path (``"images/<filename>"``) or None if no
    frame was available. Mirrors the inline top-down save logic in vla_v1
    (PIL, cv2 fallback, .npy last resort).
    """
    import numpy as np

    try:
        cam_data = camera_sensor.data
        rgb_data = cam_data.output.get("rgb") if cam_data.output is not None else None
        if rgb_data is None:
            return None
        if len(rgb_data.shape) == 4:
            rgb_np = rgb_data[0].cpu().numpy()
        elif len(rgb_data.shape) == 3:
            rgb_np = rgb_data.cpu().numpy()
        else:
            raise ValueError(f"Unexpected RGB data shape: {rgb_data.shape}")

        if rgb_np.max() <= 1.0:
            rgb_np = (rgb_np * 255).astype(np.uint8)
        else:
            rgb_np = rgb_np.astype(np.uint8)
        if rgb_np.ndim == 3 and rgb_np.shape[2] == 4:
            rgb_np = rgb_np[:, :, :3]

        out_path = Path(images_dir) / filename
        try:
            from PIL import Image

            Image.fromarray(rgb_np).save(str(out_path))
        except Exception:
            try:
                import cv2

                if rgb_np.ndim == 3 and rgb_np.shape[2] == 3:
                    cv2.imwrite(str(out_path), cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))
                else:
                    cv2.imwrite(str(out_path), rgb_np)
            except Exception:
                np.save(str(out_path.with_suffix(".npy")), rgb_np)
                out_path = out_path.with_suffix(".npy")
        return f"images/{out_path.name}"
    except Exception:
        return None


def build_cameras_calibration(
    *,
    topdown_cfg=None,
    wrist_cfg=None,
    dr_camera_sample: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the per-episode ``cameras.json`` payload (intrinsics + extrinsics).

    - overhead: world pose. When domain randomization moved the camera this
      episode, ``dr_camera_sample`` (the ``camera`` entry of the DR event) holds
      the ACTUAL post-DR pose/fov and overrides the config baseline.
    - wrist: fixed mount (hand-eye) calibration in the EE-link frame; its world
      pose at any tick is ``ee_pose ∘ offset`` using the tick's ``robot.ee_pose_*``.

    Intrinsics follow the repo-wide 36 mm-aperture pinhole convention (the same
    formula that sets the USD focal length), so they are exact, not estimates.
    """
    from environments.utils.camera import pinhole_intrinsics

    out: Dict[str, Any] = {"version": "cameras_v1", "frame_conventions": {
        "camera_axes": "USD/OpenGL: looks down local -Z, +Y up",
        "wrist_extrinsics_frame": "EE-link frame (hand-eye mount calibration)",
        "overhead_extrinsics_frame": "world",
    }}

    if topdown_cfg is not None:
        fov = float(getattr(topdown_cfg, "fov", 65.0))
        pos = [float(v) for v in getattr(topdown_cfg, "position", (0.0, 0.0, 0.0))]
        rpy = [0.0, 0.0, 0.0]
        dr_applied = False
        if dr_camera_sample:
            try:
                pos = [float(v) for v in dr_camera_sample.get("pos_xyz", pos)]
                rpy = [float(v) for v in dr_camera_sample.get("rpy_deg", rpy)]
                fov = float(dr_camera_sample.get("fov_deg", fov))
                dr_applied = True
            except Exception:
                pass
        out["overhead"] = {
            "prim_path": str(getattr(topdown_cfg, "prim_path", "")),
            "position_w": pos,
            "rpy_deg_w": rpy,
            "domain_randomized": dr_applied,
            "intrinsics": pinhole_intrinsics(fov, getattr(topdown_cfg, "resolution", (640, 640))),
        }

    if wrist_cfg is not None:
        out["wrist"] = {
            "prim_path": str(getattr(wrist_cfg, "prim_path", "")),
            "parent_link": str(getattr(wrist_cfg, "parent_link", "")),
            "offset_pos_parent": [float(v) for v in getattr(wrist_cfg, "offset_pos", (0.0, 0.0, 0.0))],
            "offset_rpy_deg_parent": [float(v) for v in getattr(wrist_cfg, "offset_rpy_deg", (0.0, 0.0, 0.0))],
            "domain_randomized": False,
            "intrinsics": pinhole_intrinsics(
                float(getattr(wrist_cfg, "fov", 87.0)), getattr(wrist_cfg, "resolution", (640, 640))
            ),
        }
    return out
