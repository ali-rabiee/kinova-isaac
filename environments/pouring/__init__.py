"""Pouring environment: JACO2 + YCB pitcher + YCB receiver + rigid-sphere pellets.

Mirrors :mod:`environments.cubes` and :mod:`environments.ycb_reach_to_grasp`
in shape, but the "objects" are split into three roles:

- a **pitcher** USD spawned with SDF mesh collision so its hollow interior is
  preserved and pellets can fall out when it is tilted by the robot;
- a **glass / mug / bowl** USD, also SDF-collided, sitting on the table;
- a configurable number of small dynamic-rigid-body **pellets** pre-filled
  inside the pitcher, acting as a discrete proxy for liquid (this matches the
  approach taken by NVIDIA's IsaacLabEvalTasks "nut pouring" benchmark).

The amount poured is queried at any time via :meth:`PouringEnv.count_pellets_in_glass`,
which counts pellets whose world position falls inside an OBB attached to the
receiving container.
"""

from environments.base import (
    CameraConfig,
    SceneConfig,
    TopDownCameraConfig,
    define_origins,
    design_scene,
)

from .config import (
    DEFAULT_CAMERA,
    DEFAULT_GLASS,
    DEFAULT_PELLETS,
    DEFAULT_PITCHER,
    DEFAULT_SCENE,
    DEFAULT_TOP_DOWN_CAMERA,
    DEFAULT_VOLUME_SENSOR,
    GlassConfig,
    PelletConfig,
    PitcherConfig,
    VolumeSensorConfig,
    default_ycb_dir,
)
from .env import PouringEnv

__all__ = [
    "CameraConfig",
    "DEFAULT_CAMERA",
    "DEFAULT_GLASS",
    "DEFAULT_PELLETS",
    "DEFAULT_PITCHER",
    "DEFAULT_SCENE",
    "DEFAULT_TOP_DOWN_CAMERA",
    "DEFAULT_VOLUME_SENSOR",
    "GlassConfig",
    "PelletConfig",
    "PitcherConfig",
    "PouringEnv",
    "SceneConfig",
    "TopDownCameraConfig",
    "VolumeSensorConfig",
    "default_ycb_dir",
    "define_origins",
    "design_scene",
]
