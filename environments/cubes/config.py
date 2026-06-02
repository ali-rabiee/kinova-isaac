"""Default configs for the cubes environment.

The scene/camera dataclasses themselves live in :mod:`environments.base`; this
module just instantiates the defaults that this env package exposes plus the
cube-specific palette / spawn defaults.
"""

from environments.base import (
    CameraConfig,
    SceneConfig,
    TopDownCameraConfig,
    default_jaco2_home_pose,
)


# Palette for colored cubes -- (color_name, RGB in [0,1]).
BOX_COLORS: list[tuple[str, tuple[float, float, float]]] = [
    ("red", (0.9, 0.2, 0.2)),
    ("blue", (0.2, 0.4, 0.9)),
    ("green", (0.2, 0.9, 0.3)),
    ("yellow", (0.9, 0.8, 0.2)),
    ("purple", (0.7, 0.3, 0.8)),
]

# Defaults that match the block-stacking demo's CLI defaults.
DEFAULT_BOX_SIZE: float = 0.08
DEFAULT_SPAWN_MIN: tuple[float, float, float] = (0.30, -0.30, 0.90)
DEFAULT_SPAWN_MAX: tuple[float, float, float] = (0.55, 0.30, 0.95)


DEFAULT_SCENE: SceneConfig = SceneConfig(
    robot_default_joint_pos=default_jaco2_home_pose(),
)
DEFAULT_CAMERA: CameraConfig = CameraConfig()
DEFAULT_TOP_DOWN_CAMERA: TopDownCameraConfig = TopDownCameraConfig()


__all__ = [
    "BOX_COLORS",
    "DEFAULT_BOX_SIZE",
    "DEFAULT_CAMERA",
    "DEFAULT_SCENE",
    "DEFAULT_SPAWN_MAX",
    "DEFAULT_SPAWN_MIN",
    "DEFAULT_TOP_DOWN_CAMERA",
]
