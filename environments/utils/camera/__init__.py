"""Modular per-camera configs + creation/build helpers for Isaac Sim
environments.

Each camera type is fully self-contained in its own module:
- ``topdown.py``: ``TopDownCameraConfig``, ``create_topdown_camera``, ``build_topdown_camera_sensor``
- ``front.py``:   ``FrontCameraConfig``, ``create_front_camera``, ``build_front_camera_sensor``
- ``wrist.py``:   ``WristCameraConfig``, ``build_wrist_camera_cfg``, ``build_wrist_camera_sensor``

``registry.py``'s ``CAMERA_CONFIGS``/``build_camera()`` is a thin name ->
config/builder lookup for callers that want to iterate over "whichever
cameras this environment uses" without importing each module by hand; it
does not replace direct access to any single camera's config/functions --
each config is a separate, independently-editable dataclass.
"""

from .front import DEFAULT_FRONT_CAMERA, FrontCameraConfig, build_front_camera_sensor, create_front_camera
from .registry import CAMERA_CONFIG_TYPES, CAMERA_CONFIGS, build_camera
from .topdown import (
    DEFAULT_TOP_DOWN_CAMERA,
    TopDownCameraConfig,
    build_topdown_camera_sensor,
    create_topdown_camera,
)
from .wrist import (
    DEFAULT_WRIST_CAMERA,
    WristCameraConfig,
    build_wrist_camera_cfg,
    build_wrist_camera_sensor,
    find_prim_path_by_name,
    list_all_descendant_prim_names,
    sync_wrist_camera_to_ee,
)

__all__ = [
    "CAMERA_CONFIGS",
    "CAMERA_CONFIG_TYPES",
    "build_camera",
    "TopDownCameraConfig",
    "DEFAULT_TOP_DOWN_CAMERA",
    "create_topdown_camera",
    "build_topdown_camera_sensor",
    "FrontCameraConfig",
    "DEFAULT_FRONT_CAMERA",
    "create_front_camera",
    "build_front_camera_sensor",
    "WristCameraConfig",
    "DEFAULT_WRIST_CAMERA",
    "build_wrist_camera_cfg",
    "build_wrist_camera_sensor",
    "find_prim_path_by_name",
    "list_all_descendant_prim_names",
    "sync_wrist_camera_to_ee",
]
