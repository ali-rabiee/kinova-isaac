"""Name -> config / builder registry, so environments and scripts can pick
cameras by name instead of importing each module individually.

Extend by adding a new (name, module) pair here; each camera's own config
class and creation/build function stay fully independent (see ``topdown.py``,
``front.py``, ``wrist.py``) -- this module is just a lookup table over them,
not a replacement for direct access to any single camera's config.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from .front import DEFAULT_FRONT_CAMERA, FrontCameraConfig
from .topdown import DEFAULT_TOP_DOWN_CAMERA, TopDownCameraConfig
from .wrist import DEFAULT_WRIST_CAMERA, WristCameraConfig

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.sensors import Camera

CAMERA_CONFIGS: dict[str, object] = {
    "top_down": DEFAULT_TOP_DOWN_CAMERA,
    "front": DEFAULT_FRONT_CAMERA,
    "wrist": DEFAULT_WRIST_CAMERA,
}

CAMERA_CONFIG_TYPES: dict[str, type] = {
    "top_down": TopDownCameraConfig,
    "front": FrontCameraConfig,
    "wrist": WristCameraConfig,
}


def build_camera(name: str, *, robot: Optional["Articulation"] = None, cfg=None) -> "Camera":
    """Uniform entry point: build a ready-to-capture ``Camera`` sensor for
    ``name`` in ``{"top_down", "front", "wrist"}``. ``cfg`` overrides that
    camera's default config; ``robot`` is required only for ``"wrist"``
    (world-space cameras don't need it)."""
    if name == "top_down":
        from .topdown import build_topdown_camera_sensor

        return build_topdown_camera_sensor(cfg)
    if name == "front":
        from .front import build_front_camera_sensor

        return build_front_camera_sensor(cfg)
    if name == "wrist":
        from .wrist import build_wrist_camera_sensor

        if robot is None:
            raise ValueError("wrist camera requires robot=...")
        return build_wrist_camera_sensor(robot, cfg)
    raise ValueError(f"unknown camera name {name!r}; choices: {list(CAMERA_CONFIGS)}")
