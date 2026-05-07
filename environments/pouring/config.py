"""Default configs for the pouring environment.

Containers are built from primitive cuboids -- a floor plus four thin walls
-- spawned as children of a single rigid body. This sidesteps the long-standing
problem with hollow USD assets (YCB pitcher/mug) on dynamic bodies in PhysX:
convex-hull approximation seals the opening, triangle-mesh approximation is
illegal on dynamic bodies, and the SDF mesh approximation depends on the
specific PhysX/IsaacLab versions shipped with each Isaac Sim release. With
primitive walls the cavity is just the empty volume we deliberately leave
between the children, so it always works at full sim speed.

The visual is a colored opaque box. If you later want a YCB-asset look on top
of the working physics, attach the YCB Mesh as a non-collidable visual child
(``rigid_props=None, collision_props=None``) and keep the wall colliders.
"""

from dataclasses import dataclass, field
from typing import Tuple

from environments.base import (
    CameraConfig,
    SceneConfig,
    TopDownCameraConfig,
    default_jaco2_home_pose,
)


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


@dataclass
class PitcherConfig:
    """The container the robot picks up and tilts to pour.

    Geometry is an open-top hollow cylinder built from primitives: a circular
    floor + ``n_segments`` thin rectangular wall slabs arranged radially. The
    cavity is the empty cylindrical volume of radius ``inner_radius_m`` and
    height ``inner_height_m``. ``spawn_position`` is the world-frame location
    of the container's bottom-center (its lowest collidable point sits at
    ``z = spawn_position[2]``).

    Pitcher defaults: tall + narrow + blue.
    """

    spawn_position: Tuple[float, float, float] = (0.40, 0.18, 0.83)
    inner_radius_m: float = 0.030  # 6 cm interior diameter -- comfortable for the JACO2 fingers
    inner_height_m: float = 0.12   # 12 cm tall

    wall_thickness_m: float = 0.005
    base_thickness_m: float = 0.005
    n_segments: int = 16  # number of polygon wall slabs (more = rounder)

    # Strong blue, opaque (a "fluid container" look).
    color_rgb: Tuple[float, float, float] = (0.20, 0.45, 0.95)

    mass_kg: float = 0.40


@dataclass
class GlassConfig:
    """The receiving container that catches pellets.

    Same shape language as :class:`PitcherConfig` (open-top hollow cylinder).
    Glass defaults: shorter + slightly wider opening + light "glass" tint.
    """

    spawn_position: Tuple[float, float, float] = (0.40, -0.18, 0.83)
    inner_radius_m: float = 0.035  # 7 cm interior diameter -- wider opening to catch pellets
    inner_height_m: float = 0.08   # 8 cm tall

    wall_thickness_m: float = 0.004
    base_thickness_m: float = 0.005
    n_segments: int = 16

    # Light cyan/white, evoking translucent glass.
    color_rgb: Tuple[float, float, float] = (0.80, 0.95, 0.98)

    mass_kg: float = 0.30


@dataclass
class PelletConfig:
    """Discrete-pellet proxy for liquid.

    Pellets are uniform dynamic spheres pre-filled inside the pitcher and
    sampled in *cylindrical* coords (uniform within the pitcher's interior
    radius and the configured z-range), not an AABB. The env's spawn helper
    automatically sizes the safe radius from :class:`PitcherConfig`, so you
    just pick the count and pellet radius.
    """

    count: int = 60
    radius_m: float = 0.006
    # Density of water (kg/m^3). Per-pellet mass = density * (4/3 * pi * r^3).
    density_kg_m3: float = 1000.0

    # Water-blue color.
    color_rgb: Tuple[float, float, float] = (0.10, 0.45, 0.95)

    # Vertical spawn range, in pitcher-local coords (z=0 is the pitcher's
    # interior floor). Default keeps pellets in the lower portion of the
    # cavity so they pile up neatly under gravity.
    spawn_z_min_m: float = 0.010
    spawn_z_max_m: float = 0.090

    # Radial safety margin from the inner wall (clearance kept from the
    # pitcher's inner_radius). Default is 2.5x the pellet radius so spheres
    # spawn fully inside the cavity even after small jitter.
    wall_clearance_m: float = 0.015

    # Per-pellet drive / contact tuning.
    static_friction: float = 1.0
    dynamic_friction: float = 1.0
    restitution: float = 0.05
    contact_offset: float = 0.002
    rest_offset: float = 0.0005


@dataclass
class VolumeSensorConfig:
    """Axis-aligned box (in world frame) that defines the glass's interior.

    A pellet whose world position lies inside this box is considered "in the
    glass". The box is centered above the glass's spawn pose; the defaults
    cover a typical YCB mug interior. Tune via the demo CLI for other glasses.
    """

    half_extents_xyz: Tuple[float, float, float] = (0.040, 0.040, 0.045)
    # Center is glass.spawn_position + (0, 0, z_offset_above_base). The default
    # is half the glass's cavity height so the box hugs the interior tightly.
    z_offset_above_base_m: float = 0.045


# ---------------------------------------------------------------------------
# Singletons exposed as ``DEFAULT_*`` for symmetry with the other env packages.
# ---------------------------------------------------------------------------
DEFAULT_SCENE: SceneConfig = SceneConfig(
    robot_default_joint_pos=default_jaco2_home_pose(),
)
DEFAULT_CAMERA: CameraConfig = CameraConfig()
DEFAULT_TOP_DOWN_CAMERA: TopDownCameraConfig = TopDownCameraConfig()

DEFAULT_PITCHER: PitcherConfig = PitcherConfig()
DEFAULT_GLASS: GlassConfig = GlassConfig()
DEFAULT_PELLETS: PelletConfig = PelletConfig()
DEFAULT_VOLUME_SENSOR: VolumeSensorConfig = VolumeSensorConfig()


__all__ = [
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
    "VolumeSensorConfig",
    "default_ycb_dir",
]
