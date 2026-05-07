"""Cube-stacking environment: JACO2 + colored uniform cubes.

Mirrors the scene used by ``DEMOS/block_stacking/block_stacking_demo.py``:
the same Kinova JACO2 + Thorlabs table setup as
:mod:`environments.ycb_reach_to_grasp`, but with uniformly-colored cuboid
spawns instead of YCB USD assets.

OOP entry point: :class:`CubesEnv`. The legacy module-level
``design_scene`` and the default config singletons are also exported so
existing scripts can drop into the lower-level helpers directly.
"""

from environments.base import (
    CameraConfig,
    SceneConfig,
    TopDownCameraConfig,
    define_origins,
    design_scene,
)

from .config import (
    BOX_COLORS,
    DEFAULT_BOX_SIZE,
    DEFAULT_CAMERA,
    DEFAULT_SCENE,
    DEFAULT_SPAWN_MAX,
    DEFAULT_SPAWN_MIN,
    DEFAULT_TOP_DOWN_CAMERA,
)
from .env import CubesEnv

__all__ = [
    "BOX_COLORS",
    "CameraConfig",
    "CubesEnv",
    "DEFAULT_BOX_SIZE",
    "DEFAULT_CAMERA",
    "DEFAULT_SCENE",
    "DEFAULT_SPAWN_MAX",
    "DEFAULT_SPAWN_MIN",
    "DEFAULT_TOP_DOWN_CAMERA",
    "SceneConfig",
    "TopDownCameraConfig",
    "define_origins",
    "design_scene",
]
