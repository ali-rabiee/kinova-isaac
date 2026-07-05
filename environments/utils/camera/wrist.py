"""Wrist (eye-in-hand) camera creation utilities.

Mirrors the two-stage pattern used for the top-down camera (`topdown.py`):
1. `create_wrist_camera` makes a raw USD Camera prim — here as a CHILD of the
   end-effector link prim, so the renderer moves it with the arm for free.
2. The caller attaches an IsaacLab `Camera` sensor to that prim with
   `spawn=None` (see data_collection/profiles/vla_v1.py) — attaching to an
   existing prim avoids the silent-black-image failure mode of re-spawning.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from environments.reach_to_grasp_VLA.config import WristCameraConfig


def focal_length_mm_from_fov(fov_deg: float, sensor_size_mm: float = 36.0) -> float:
    """FOV (deg, horizontal) -> focal length (mm) for the repo's 36 mm-aperture convention."""
    import numpy as np

    return float(sensor_size_mm / (2.0 * np.tan(np.radians(float(fov_deg)) / 2.0)))


def pinhole_intrinsics(fov_deg: float, resolution: "tuple[int, int]") -> dict:
    """Pinhole intrinsics implied by the 36 mm-aperture convention.

    fx = W / (2*tan(hfov/2)); square pixels (the vertical aperture follows the
    aspect ratio), principal point at the image center.
    """
    import numpy as np

    w, h = int(resolution[0]), int(resolution[1])
    fx = float(w / (2.0 * np.tan(np.radians(float(fov_deg)) / 2.0)))
    return {
        "model": "pinhole",
        "width_px": w,
        "height_px": h,
        "fx_px": fx,
        "fy_px": fx,
        "cx_px": w / 2.0,
        "cy_px": h / 2.0,
        "fov_deg_horizontal": float(fov_deg),
        "focal_length_mm": focal_length_mm_from_fov(fov_deg),
        "sensor_aperture_mm": 36.0,
    }


def create_wrist_camera(camera_cfg: "WristCameraConfig") -> None:
    """Create a wrist camera prim as a child of the EE link prim.

    The local translate/rotate ops encode the mount (hand-eye) calibration in
    the EE-link frame; the prim inherits the link's world transform, so no
    per-tick pose update is needed.
    """
    import importlib

    prim_utils = importlib.import_module("isaacsim.core.utils.prims")
    from pxr import Gf, UsdGeom
    import omni.usd

    parent_path = str(camera_cfg.prim_path).rsplit("/", 1)[0]
    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath(parent_path).IsValid():
        raise RuntimeError(
            f"Wrist camera parent prim not found: {parent_path}. "
            "Create the robot before the wrist camera."
        )

    prim_utils.create_prim(camera_cfg.prim_path, "Camera")
    camera_prim = stage.GetPrimAtPath(camera_cfg.prim_path)
    xform = UsdGeom.Xformable(camera_prim)
    xform.ClearXformOpOrder()
    translate_op = xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    rotate_op = xform.AddRotateXYZOp(UsdGeom.XformOp.PrecisionFloat)
    translate_op.Set(Gf.Vec3d(*[float(v) for v in camera_cfg.offset_pos]))
    rotate_op.Set(Gf.Vec3f(*[float(v) for v in camera_cfg.offset_rpy_deg]))

    camera = UsdGeom.Camera(camera_prim)
    camera.GetFocalLengthAttr().Set(focal_length_mm_from_fov(camera_cfg.fov))
    # Wrist views have close subjects; widen the near clip guard band.
    try:
        camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 10000.0))
    except Exception:
        pass
