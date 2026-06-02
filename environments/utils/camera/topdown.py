"""Top-down camera creation utilities."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from environments.base import TopDownCameraConfig


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

