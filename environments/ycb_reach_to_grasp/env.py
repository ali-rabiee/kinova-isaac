"""``YCBReachToGraspEnv`` -- JACO2 + table + YCB Nucleus objects."""

from __future__ import annotations

from typing import Iterable, Sequence

from environments.base import (
    BaseSceneEnv,
    CameraConfig,
    SceneConfig,
    TopDownCameraConfig,
)
from environments.utils.object_loader import (
    ObjectLoaderConfig,
    SpawnBounds,
)
from environments.utils.physix import (
    PhysicsConfig,
    object_loader_kwargs_from_physix,
)

from .config import (
    DEFAULT_CAMERA,
    DEFAULT_SCENE,
    DEFAULT_TOP_DOWN_CAMERA,
    default_ycb_dir,
)


class YCBReachToGraspEnv(BaseSceneEnv):
    """Reach-to-grasp scene using YCB objects from Isaac Nucleus.

    The shared JACO2 + Thorlabs table + lighting setup comes from
    :class:`environments.base.BaseSceneEnv`. This subclass adds a USD-mode
    object loader factory pointed at the YCB dataset.

    Args:
        ycb_dir: Override the dataset directory. Defaults to ISAAC_NUCLEUS_DIR/
            Props/YCB (with an HTTP fallback if Nucleus isn't available).
        include_labels: Optional whitelist of object labels (e.g. ``"mug"``,
            ``"sugar_box"``) -- only matching USDs are spawned.
        scale_range: Optional uniform scale range applied per spawned object.
    """

    spawn_mode = "usd"

    def __init__(
        self,
        *,
        scene_cfg: SceneConfig | None = None,
        camera_cfg: CameraConfig | None = None,
        top_down_camera_cfg: TopDownCameraConfig | None = None,
        physics_cfg: PhysicsConfig | None = None,
        device: str | None = None,
        ycb_dir: str | None = None,
        include_labels: Iterable[str] | None = None,
        scale_range: tuple[float, float] | None = None,
    ) -> None:
        super().__init__(
            scene_cfg=scene_cfg or DEFAULT_SCENE,
            camera_cfg=camera_cfg or DEFAULT_CAMERA,
            top_down_camera_cfg=top_down_camera_cfg or DEFAULT_TOP_DOWN_CAMERA,
            physics_cfg=physics_cfg,
            device=device,
        )
        self.ycb_dir: str = str(ycb_dir) if ycb_dir is not None else default_ycb_dir()
        self.include_labels: list[str] | None = (
            [str(s) for s in include_labels] if include_labels is not None else None
        )
        self.scale_range: tuple[float, float] | None = (
            (float(scale_range[0]), float(scale_range[1])) if scale_range is not None else None
        )

    # ------------------------------------------------------------------
    # Object loader factory
    # ------------------------------------------------------------------
    def build_object_loader_config(
        self,
        *,
        spawn_min: Sequence[float] | None = None,
        spawn_max: Sequence[float] | None = None,
        min_distance: float = 0.10,
        dataset_dirs: Sequence[str] | None = None,
        include_labels: Iterable[str] | None = None,
        scale_range: tuple[float, float] | None = None,
        **extra,
    ) -> ObjectLoaderConfig:
        """Build a USD-mode :class:`ObjectLoaderConfig` for this env.

        Args:
            spawn_min/spawn_max: AABB for random object placement (m, world).
            min_distance: Min pairwise distance between spawned objects.
            dataset_dirs: Override the dataset list (defaults to ``[self.ycb_dir]``).
            include_labels: Override the label whitelist for this call.
            scale_range: Override the uniform scale range for this call.
            **extra: Forwarded into :class:`ObjectLoaderConfig` (e.g. ``apply_preview_surface``).
        """
        bounds = SpawnBounds(
            min_xyz=tuple(spawn_min) if spawn_min is not None else (0.30, -0.30, 0.85),
            max_xyz=tuple(spawn_max) if spawn_max is not None else (0.55, 0.30, 0.92),
        )
        labels = list(include_labels) if include_labels is not None else self.include_labels
        scale = scale_range if scale_range is not None else self.scale_range

        cfg = ObjectLoaderConfig(
            dataset_dirs=list(dataset_dirs) if dataset_dirs is not None else [self.ycb_dir],
            bounds=bounds,
            min_distance=float(min_distance),
            uniform_scale_range=scale,
            include_labels=labels,
            **object_loader_kwargs_from_physix(self.physics_cfg),
        )
        for key, value in extra.items():
            setattr(cfg, key, value)
        return cfg
