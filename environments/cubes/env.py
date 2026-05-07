"""``CubesEnv`` -- JACO2 + colored uniform cubes for block stacking demos."""

from __future__ import annotations

import math
from dataclasses import replace as dc_replace
from typing import Sequence

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
    BOX_COLORS,
    DEFAULT_BOX_SIZE,
    DEFAULT_CAMERA,
    DEFAULT_SCENE,
    DEFAULT_SPAWN_MAX,
    DEFAULT_SPAWN_MIN,
    DEFAULT_TOP_DOWN_CAMERA,
)


def _coerce_size(
    size: float | Sequence[float] | None,
    default: float,
) -> tuple[float, float, float]:
    """Normalize a scalar / 3-tuple cube size into ``(sx, sy, sz)``."""
    if size is None:
        return (float(default), float(default), float(default))
    if isinstance(size, (int, float)):
        return (float(size), float(size), float(size))
    sx, sy, sz = (float(v) for v in size)
    return (sx, sy, sz)


class CubesEnv(BaseSceneEnv):
    """Block-stacking scene: JACO2 + table + uniform colored cubes.

    The shared JACO2/table/lighting setup comes from
    :class:`environments.base.BaseSceneEnv`. This subclass adds:

    - a box-mode object loader factory (``CuboidCfg`` spawns rather than USDs)
    - a configurable color palette (defaults to :data:`BOX_COLORS`)
    - small USD helpers for reading per-cube yaw / height / color label that
      the stacking demos need

    Note on physics: by default :class:`PhysicsConfig` rotates spawned USD
    assets by ``(-90, 0, 0)`` degrees so YCB meshes (modelled with Y-up) sit
    upright on the table. Cubes spawned via ``CuboidCfg`` are already
    Z-up, so this env clears that tilt automatically.
    """

    spawn_mode = "box"

    def __init__(
        self,
        *,
        scene_cfg: SceneConfig | None = None,
        camera_cfg: CameraConfig | None = None,
        top_down_camera_cfg: TopDownCameraConfig | None = None,
        physics_cfg: PhysicsConfig | None = None,
        device: str | None = None,
        palette: Sequence[tuple[str, tuple[float, float, float]]] | None = None,
        box_size: float | Sequence[float] = DEFAULT_BOX_SIZE,
        box_size_min: Sequence[float] | None = None,
        box_size_max: Sequence[float] | None = None,
    ) -> None:
        if physics_cfg is None:
            physics_cfg = PhysicsConfig()
        # Cubes spawn upright; clear the YCB-oriented tilt so cube faces stay axis-aligned.
        physics_cfg = dc_replace(physics_cfg, orientation_euler_deg=None)

        super().__init__(
            scene_cfg=scene_cfg or DEFAULT_SCENE,
            camera_cfg=camera_cfg or DEFAULT_CAMERA,
            top_down_camera_cfg=top_down_camera_cfg or DEFAULT_TOP_DOWN_CAMERA,
            physics_cfg=physics_cfg,
            device=device,
        )
        self.palette: list[tuple[str, tuple[float, float, float]]] = (
            list(palette) if palette is not None else list(BOX_COLORS)
        )

        # Resolve the cube size range. If explicit min/max are given they win;
        # otherwise we use a uniform ``box_size`` (scalar or per-axis tuple).
        if box_size_min is not None or box_size_max is not None:
            if box_size_min is None or box_size_max is None:
                raise ValueError(
                    "box_size_min and box_size_max must both be provided, or neither."
                )
            self.box_size_min: tuple[float, float, float] = tuple(float(v) for v in box_size_min)  # type: ignore[assignment]
            self.box_size_max: tuple[float, float, float] = tuple(float(v) for v in box_size_max)  # type: ignore[assignment]
        else:
            uniform = _coerce_size(box_size, DEFAULT_BOX_SIZE)
            self.box_size_min = uniform
            self.box_size_max = uniform

    # ------------------------------------------------------------------
    # Object loader factory
    # ------------------------------------------------------------------
    def build_object_loader_config(
        self,
        *,
        spawn_min: Sequence[float] | None = None,
        spawn_max: Sequence[float] | None = None,
        min_distance: float = 0.22,
        box_size: float | Sequence[float] | None = None,
        box_size_min: Sequence[float] | None = None,
        box_size_max: Sequence[float] | None = None,
        palette: Sequence[tuple[str, tuple[float, float, float]]] | None = None,
        **extra,
    ) -> ObjectLoaderConfig:
        """Build a box-mode :class:`ObjectLoaderConfig` for cube spawns."""
        bounds = SpawnBounds(
            min_xyz=tuple(spawn_min) if spawn_min is not None else DEFAULT_SPAWN_MIN,
            max_xyz=tuple(spawn_max) if spawn_max is not None else DEFAULT_SPAWN_MAX,
        )

        if box_size_min is not None or box_size_max is not None or box_size is not None:
            if box_size is not None and (box_size_min is not None or box_size_max is not None):
                raise ValueError("Provide either box_size, or box_size_min/max -- not both.")
            if box_size is not None:
                size_min = _coerce_size(box_size, DEFAULT_BOX_SIZE)
                size_max = size_min
            else:
                if box_size_min is None or box_size_max is None:
                    raise ValueError("box_size_min and box_size_max must both be provided.")
                size_min = tuple(float(v) for v in box_size_min)  # type: ignore[assignment]
                size_max = tuple(float(v) for v in box_size_max)  # type: ignore[assignment]
        else:
            size_min, size_max = self.box_size_min, self.box_size_max

        pal = list(palette) if palette is not None else self.palette

        cfg = ObjectLoaderConfig(
            dataset_dirs=[],  # unused in box mode
            bounds=bounds,
            min_distance=float(min_distance),
            spawn_mode="box",
            box_size_min=tuple(size_min),
            box_size_max=tuple(size_max),
            box_color_palette=[rgb for (_name, rgb) in pal],
            box_color_names=[name for (name, _rgb) in pal],
            **object_loader_kwargs_from_physix(self.physics_cfg),
        )
        for key, value in extra.items():
            setattr(cfg, key, value)
        return cfg

    # ------------------------------------------------------------------
    # Per-cube introspection helpers (used by the stacking demo)
    # ------------------------------------------------------------------
    @staticmethod
    def index_from_prim_path(prim_path: str) -> int | None:
        """Return the 1-based ``Obj_NN`` index encoded in a spawned cube's prim path."""
        leaf = str(prim_path).rstrip("/").split("/")[-1]
        digits = "".join(ch for ch in leaf if ch.isdigit())
        if not digits:
            return None
        try:
            return int(digits)
        except Exception:
            return None

    def color_name_for_index(self, idx: int | None) -> str | None:
        """Return the palette color name for a 1-based cube index (cycles)."""
        if idx is None or len(self.palette) == 0:
            return None
        return self.palette[(int(idx) - 1) % len(self.palette)][0]

    def label_for_prim(self, prim_path: str) -> tuple[str, str | None, int | None]:
        """Return ``(human_label, color_name, box_index)`` for a spawned cube prim.

        Mirrors the block-stacking demo's ``_box_label`` helper.
        """
        idx = self.index_from_prim_path(prim_path)
        if idx is None or len(self.palette) == 0:
            leaf = str(prim_path).rstrip("/").split("/")[-1]
            return f"box {leaf}", None, idx
        color_name = self.color_name_for_index(idx)
        return f"{color_name} box {idx}", color_name, idx

    @staticmethod
    def read_prim_world_yaw_rad(prim_path: str) -> float | None:
        """Yaw (rotation about world +Z, radians) of a prim's world transform.

        Reads the prim's local-to-world matrix and returns ``atan2(M[0][1], M[0][0])``.
        Returns ``None`` if the prim cannot be read. Requires an active stage.
        """
        try:
            import omni.usd  # type: ignore
            from pxr import Usd, UsdGeom  # type: ignore
        except Exception:
            return None
        try:
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(str(prim_path))
            if not prim.IsValid():
                return None
            xformable = UsdGeom.Xformable(prim)
            M = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            return math.atan2(float(M[0][1]), float(M[0][0]))
        except Exception as e:
            print(f"[CubesEnv][WARN] could not read world yaw for {prim_path}: {e}")
            return None

    @staticmethod
    def read_prim_height_m(prim_path: str, default: float = DEFAULT_BOX_SIZE) -> float:
        """World-AABB height (m) for a spawned cube prim. Falls back to ``default``."""
        try:
            import omni.usd  # type: ignore
            from pxr import Usd, UsdGeom  # type: ignore
        except Exception:
            return float(default)
        try:
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(str(prim_path))
            if not prim.IsValid():
                return float(default)
            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
            world_bound = cache.ComputeWorldBound(prim)
            r = world_bound.ComputeAlignedRange()
            return max(1e-4, float(r.GetMax()[2]) - float(r.GetMin()[2]))
        except Exception as e:
            print(f"[CubesEnv][WARN] could not read height for {prim_path}: {e}")
            return float(default)
