"""Front-angled external camera: a fixed, world-space "third-person" view.

Unlike the wrist camera (rigidly parented to a moving robot link, see
``wrist.py``), this is a raw USD camera prim placed at a fixed point in the
world and pointed at the workspace -- the same category of camera as the
existing top-down view (see ``topdown.py``), just angled from the front
rather than straight down.

Placeholder geometry only: ``position``/``target``/``fov`` below have NOT
been visually verified. Run ``scripts/debug_cameras.py --cameras front`` and
inspect the saved frames before trusting these defaults for real data
collection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from isaaclab.sensors import Camera


@dataclass
class FrontCameraConfig:
    """Fixed front-angled camera looking back at the robot and workspace.

    Geometry tuned visually via ``scripts/debug_cameras.py``. Sits on +X --
    the robot's working side, where boxes spawn (X in [0.30, 0.55], Y in
    [-0.30, 0.30], resting at z ~0.83 on the table) -- so the boxes are in
    the foreground unoccluded and the robot is behind them, facing camera.

    Placed high (z 2.2) and well back (x 2.5), looking down ~31 deg. The
    elevation is what makes that tilt safe: from a camera below ~1.5 m,
    aiming this low clips the top of the arm (which reaches z ~1.6); from
    above it, the whole robot stays in frame and more of the table shows.
    """

    prim_path: str = "/World/Origin1/FrontCamera"
    position: Tuple[float, float, float] = (2.5, 0.0, 2.2)
    target: Tuple[float, float, float] = (0.30, 0.0, 0.90)
    resolution: Tuple[int, int] = (640, 640)
    fov: float = 60.0


DEFAULT_FRONT_CAMERA = FrontCameraConfig()


def create_front_camera(camera_cfg: "FrontCameraConfig") -> None:
    """Create the front camera's raw USD prim, oriented via a real look-at
    transform (unlike top-down's hardcoded zero-rotation shortcut, which only
    works because straight-down happens to need no rotation)."""
    import importlib

    from .types import look_at_transform, set_camera_focal_length_from_fov, set_prim_world_transform

    prim_utils = importlib.import_module("isaacsim.core.utils.prims")
    prim_utils.create_prim(camera_cfg.prim_path, "Camera")

    transform = look_at_transform(camera_cfg.position, camera_cfg.target)
    set_prim_world_transform(camera_cfg.prim_path, transform)

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    camera_prim = stage.GetPrimAtPath(camera_cfg.prim_path)
    set_camera_focal_length_from_fov(camera_prim, camera_cfg.fov)


def build_front_camera_sensor(camera_cfg: "FrontCameraConfig | None" = None) -> "Camera":
    """Convenience one-call helper: create the raw prim (if not already
    present) and wrap it in an IsaacLab ``Camera`` sensor, ready to capture.
    Mirrors ``topdown.build_topdown_camera_sensor``."""
    import omni.usd
    from isaaclab.sensors import Camera, CameraCfg

    cfg = camera_cfg or DEFAULT_FRONT_CAMERA
    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath(cfg.prim_path).IsValid():
        create_front_camera(cfg)

    sensor_cfg = CameraCfg(
        prim_path=cfg.prim_path,
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=None,
        data_types=["rgb"],
        width=cfg.resolution[0],
        height=cfg.resolution[1],
    )
    return Camera(cfg=sensor_cfg)
