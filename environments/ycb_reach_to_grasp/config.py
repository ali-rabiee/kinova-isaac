"""Default scene + camera configs for the YCB reach-to-grasp environment.

The scene/camera dataclasses themselves live in :mod:`environments.base`; this
module just instantiates the defaults that this env package exposes.
"""

from environments.base import (
    CameraConfig,
    SceneConfig,
    TopDownCameraConfig,
    default_jaco2_home_pose,
)


DEFAULT_SCENE: SceneConfig = SceneConfig(
    robot_default_joint_pos=default_jaco2_home_pose(),
)
DEFAULT_CAMERA: CameraConfig = CameraConfig()
DEFAULT_TOP_DOWN_CAMERA: TopDownCameraConfig = TopDownCameraConfig()


def default_ycb_dir() -> str:
    """Return the YCB dataset directory on Isaac Nucleus, with HTTP fallback."""
    try:
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

        return f"{ISAAC_NUCLEUS_DIR}/Props/YCB"
    except Exception:
        return (
            "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
            "/Assets/Isaac/5.0/Isaac/Props/YCB"
        )


__all__ = [
    "DEFAULT_SCENE",
    "DEFAULT_CAMERA",
    "DEFAULT_TOP_DOWN_CAMERA",
    "default_ycb_dir",
]
