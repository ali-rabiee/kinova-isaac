"""Reach-to-grasp environment using the YCB Nucleus dataset.

OOP entry point: :class:`YCBReachToGraspEnv`. The legacy module-level
``design_scene``, ``DEFAULT_SCENE``, ``DEFAULT_CAMERA``, and
``DEFAULT_TOP_DOWN_CAMERA`` are also exported for callers that want to drop
into the lower-level scene helpers directly.
"""

from environments.base import (
    CameraConfig,
    SceneConfig,
    TopDownCameraConfig,
    define_origins,
    design_scene,
)
from environments.utils.camera.front import FrontCameraConfig
from environments.utils.camera.wrist import WristCameraConfig

from .config import (
    DEFAULT_CAMERA,
    DEFAULT_FRONT_CAMERA,
    DEFAULT_SCENE,
    DEFAULT_TOP_DOWN_CAMERA,
    DEFAULT_WRIST_CAMERA,
    default_ycb_dir,
)
from .env import YCBReachToGraspEnv

__all__ = [
    "CameraConfig",
    "SceneConfig",
    "TopDownCameraConfig",
    "FrontCameraConfig",
    "WristCameraConfig",
    "DEFAULT_CAMERA",
    "DEFAULT_SCENE",
    "DEFAULT_TOP_DOWN_CAMERA",
    "DEFAULT_FRONT_CAMERA",
    "DEFAULT_WRIST_CAMERA",
    "YCBReachToGraspEnv",
    "default_ycb_dir",
    "define_origins",
    "design_scene",
]
