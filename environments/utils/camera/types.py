"""Shared USD helpers for world-space (unattached) camera prims.

Used by any camera whose config is a fixed world position/target pair (e.g.
``front.py``; ``topdown.py`` keeps its own hand-rolled zero-rotation shortcut
since straight-down happens to need no rotation) rather than one rigidly
parented to a moving robot link (see ``wrist.py`` for that case).
"""

from __future__ import annotations

from typing import Tuple


def look_at_transform(
    position: Tuple[float, float, float],
    target: Tuple[float, float, float],
    up: Tuple[float, float, float] = (0.0, 0.0, 1.0),
):
    """Local-to-world 4x4 transform (``Gf.Matrix4d``) for a camera at
    ``position`` looking at ``target``.

    ``Gf.Matrix4d().SetLookAt(eye, center, up)`` builds the WORLD->CAMERA
    (view) matrix; a camera prim's own placement in the scene is the inverse
    of that (CAMERA->WORLD). This is the standard USD idiom (matches
    ``usdview``/``Gf.Camera``) rather than a hand-derived Euler decomposition,
    but the resulting orientation should still be sanity-checked visually
    (see ``scripts/debug_cameras.py``) before trusting it for a new view.
    """
    from pxr import Gf

    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*position), Gf.Vec3d(*target), Gf.Vec3d(*up))
    return view.GetInverse()


def set_prim_world_transform(prim_path: str, transform) -> None:
    """Clear any existing xform ops on ``prim_path`` and set a single transform op."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(transform)


def set_camera_focal_length_from_fov(camera_prim, fov_deg: float, sensor_size_mm: float = 36.0) -> None:
    """Set a ``UsdGeom.Camera``'s focal length to match a desired horizontal
    FOV (same 35mm-equivalent convention ``topdown.py`` already uses)."""
    import numpy as np
    from pxr import UsdGeom

    camera = UsdGeom.Camera(camera_prim)
    focal_length_mm = sensor_size_mm / (2.0 * np.tan(np.radians(fov_deg) / 2.0))
    camera.GetFocalLengthAttr().Set(focal_length_mm)
