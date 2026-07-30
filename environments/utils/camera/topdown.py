"""Top-down camera: config + raw-prim creation + sensor builder.

Unchanged behavior from before this module owned the config directly --
`create_topdown_camera()`'s body is untouched, it just now also defines
`TopDownCameraConfig` (moved here from `environments.base`, which re-exports
it for backward compatibility) so every camera type is fully self-contained
in its own module under `environments/utils/camera/`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from isaaclab.sensors import Camera


@dataclass
class TopDownCameraConfig:
    """Optional overhead camera prim, used by VLA / data-collection profiles."""

    prim_path: str = "/World/Origin1/TopDownCamera"
    position: Tuple[float, float, float] = (0.4, 0.0, 4.0)
    target: Tuple[float, float, float] = (0.4, 0.0, 0.8)
    resolution: Tuple[int, int] = (640, 640)
    fov: float = 65.0


DEFAULT_TOP_DOWN_CAMERA = TopDownCameraConfig()


def create_topdown_camera(camera_cfg: "TopDownCameraConfig") -> None:
    """Create a top-down camera prim in the scene.

    The camera is positioned above the robot base and oriented to look at the target point,
    providing a clear view of the workspace, gripper, and objects.

    Args:
        camera_cfg: Configuration for the top-down camera including position, target, and properties.

    Camera Configuration Guide:
        - position: (x, y, z) - Camera location in world coordinates
        - target: (x, y, z) - Point the camera looks at
        - fov: Field of view in degrees (typically 50-90 for workspace views)

    To adjust the camera view:
        1. Change 'position' to move the camera (e.g., higher z = more overhead view)
        2. Change 'target' to change what the camera looks at (e.g., workspace center)
        3. Adjust 'fov' to zoom in/out (higher = wider view, lower = zoomed in)
    """
    import importlib
    prim_utils = importlib.import_module("isaacsim.core.utils.prims")
    from pxr import Gf, UsdGeom
    import numpy as np

    # Create camera prim
    prim_utils.create_prim(camera_cfg.prim_path, "Camera")

    # Set position and orientation to look at target
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    camera_prim = stage.GetPrimAtPath(camera_cfg.prim_path)
    xform = UsdGeom.Xformable(camera_prim)

    # Get position and target
    pos = Gf.Vec3d(*camera_cfg.position)
    target = Gf.Vec3d(*camera_cfg.target)

    # For a top-down camera looking straight down
    # USD cameras by default look down -Z in local space
    # In world space, Z is up, so -Z is down
    # If the camera is positioned above and we want it to look down, we may need no rotation
    # or a specific rotation depending on the coordinate system

    # Try: No rotation (camera already looks down -Z which is down in world space)
    # If that doesn't work, try 180° around X
    euler_deg = Gf.Vec3f(0.0, 0.0, 0.0)

    # Clear existing ops and set transform
    xform.ClearXformOpOrder()
    translate_op = xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    rotate_op = xform.AddRotateXYZOp(UsdGeom.XformOp.PrecisionFloat)
    translate_op.Set(pos)
    rotate_op.Set(euler_deg)

    # Set camera properties (FOV)
    camera = UsdGeom.Camera(camera_prim)
    # Convert FOV from degrees to focal length
    # Using standard 35mm sensor size for calculation
    sensor_size_mm = 36.0  # Full frame horizontal sensor size
    focal_length_mm = sensor_size_mm / (2.0 * np.tan(np.radians(camera_cfg.fov) / 2.0))
    camera.GetFocalLengthAttr().Set(focal_length_mm)


def build_topdown_camera_sensor(camera_cfg: "TopDownCameraConfig | None" = None) -> "Camera":
    """Convenience one-call helper: create the raw prim (if not already
    present) and wrap it in an IsaacLab ``Camera`` sensor, ready to capture.

    Does not replace ``create_topdown_camera()``/``BaseSceneEnv.attach_top_down_camera()``
    -- those keep working exactly as before. This is for standalone scripts
    (e.g. ``scripts/debug_cameras.py``) that want a ready-to-capture sensor
    for any of the three camera types via one uniform call (see
    ``environments.utils.camera.registry.build_camera``).
    """
    import omni.usd
    from isaaclab.sensors import Camera, CameraCfg

    cfg = camera_cfg or DEFAULT_TOP_DOWN_CAMERA
    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath(cfg.prim_path).IsValid():
        create_topdown_camera(cfg)

    sensor_cfg = CameraCfg(
        prim_path=cfg.prim_path,
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=None,
        data_types=["rgb"],
        width=cfg.resolution[0],
        height=cfg.resolution[1],
    )
    return Camera(cfg=sensor_cfg)
