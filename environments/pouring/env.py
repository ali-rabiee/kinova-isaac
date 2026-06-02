"""``PouringEnv`` -- pouring task scene built on top of :class:`BaseSceneEnv`.

The env owns three categories of entity beyond the shared JACO2 + table setup:

1. **Pitcher** -- an open-top hollow cylinder authored from primitives: a
   solid disc floor + ``n_segments`` thin tangent wall slabs around the
   circumference. Dynamic rigid body, full speed.
2. **Glass / receiver** -- same shape language as the pitcher, different size
   + color.
3. **Pellets** -- :data:`PelletConfig.count` dynamic spheres pre-filled inside
   the pitcher's cavity. They act as a discrete proxy for liquid.

Volume measurement: :meth:`PouringEnv.count_pellets_in_glass` reads each
pellet's world pose each call and counts those whose position is inside the
:class:`VolumeSensorConfig` box anchored at the glass's pose.

Physics: every container collider is created with the exact same parameters
the YCB reach-to-grasp env uses for graspable objects (high friction, the
configured contact/rest offsets, and a torsional patch radius), and a shared
:class:`RigidBodyMaterialCfg` is bound to each child cuboid so the JACO2
gripper grips and lifts the pitcher / glass without slip -- identical
behaviour to the YCB env.

Why primitive walls instead of YCB pitcher / mug USDs: PhysX rejects
triangle-mesh collision on dynamic bodies and silently falls back to convex
hull, which seals the opening of any hollow vessel. The robust fix is
signed-distance-field (SDF) mesh collision, but the IsaacLab build that ships
with this Isaac Sim release does not wrap ``SDFMeshPropertiesCfg`` and even
authoring it via raw ``pxr.PhysxSchema`` sometimes fails on instanceable YCB
assets. Building the container as a compound of primitive children sidesteps
the whole problem: the cavity is the empty volume between the segments,
every collider is a clean convex shape, and the pellet proxy approach matches
NVIDIA's own IsaacLabEvalTasks "nut pouring" benchmark.
"""

from __future__ import annotations

import math
import random
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


class PouringEnv(BaseSceneEnv):
    """Pouring task scene: pitcher + receiver + rigid-pellet 'liquid'."""

    spawn_mode = "usd"  # also exposes a YCB-style ObjectLoader for clutter

    def __init__(
        self,
        *,
        scene_cfg: SceneConfig | None = None,
        camera_cfg: CameraConfig | None = None,
        top_down_camera_cfg: TopDownCameraConfig | None = None,
        physics_cfg: PhysicsConfig | None = None,
        device: str | None = None,
        pitcher_cfg: PitcherConfig | None = None,
        glass_cfg: GlassConfig | None = None,
        pellet_cfg: PelletConfig | None = None,
        volume_sensor_cfg: VolumeSensorConfig | None = None,
        ycb_dir: str | None = None,
        clutter_include_labels: Iterable[str] | None = None,
        clutter_scale_range: tuple[float, float] | None = None,
    ) -> None:
        super().__init__(
            scene_cfg=scene_cfg or DEFAULT_SCENE,
            camera_cfg=camera_cfg or DEFAULT_CAMERA,
            top_down_camera_cfg=top_down_camera_cfg or DEFAULT_TOP_DOWN_CAMERA,
            physics_cfg=physics_cfg,
            device=device,
        )
        self.pitcher_cfg: PitcherConfig = pitcher_cfg or DEFAULT_PITCHER
        self.glass_cfg: GlassConfig = glass_cfg or DEFAULT_GLASS
        self.pellet_cfg: PelletConfig = pellet_cfg or DEFAULT_PELLETS
        self.volume_sensor_cfg: VolumeSensorConfig = volume_sensor_cfg or DEFAULT_VOLUME_SENSOR

        self.ycb_dir: str = str(ycb_dir) if ycb_dir is not None else default_ycb_dir()
        self.clutter_include_labels: list[str] | None = (
            [str(s) for s in clutter_include_labels] if clutter_include_labels is not None else None
        )
        self.clutter_scale_range: tuple[float, float] | None = (
            (float(clutter_scale_range[0]), float(clutter_scale_range[1]))
            if clutter_scale_range is not None
            else None
        )

        # Filled in by spawn_containers / spawn_pellets.
        self.pitcher_prim_path: str | None = None
        self.glass_prim_path: str | None = None
        self.pellet_prim_paths: list[str] = []

    # ------------------------------------------------------------------
    # Container spawning (primitive walls)
    # ------------------------------------------------------------------
    def spawn_containers(
        self,
        *,
        parent_prim_path: str = "/World/Origin1",
    ) -> tuple[str, str]:
        """Spawn the pitcher + glass under ``parent_prim_path``.

        Each container is one rigid body root with five collidable cuboid
        children (a floor + four walls). Returns
        ``(pitcher_prim_path, glass_prim_path)``.
        """
        import importlib

        prim_utils = importlib.import_module("isaacsim.core.utils.prims")

        import omni.usd  # type: ignore
        stage = omni.usd.get_context().get_stage()
        if not stage.GetPrimAtPath(parent_prim_path).IsValid():
            prim_utils.create_prim(parent_prim_path, "Xform")

        containers_root = f"{parent_prim_path.rstrip('/')}/Containers"
        if not stage.GetPrimAtPath(containers_root).IsValid():
            prim_utils.create_prim(containers_root, "Xform")

        self.pitcher_prim_path = self._spawn_open_top_cylinder(
            prim_path=f"{containers_root}/Pitcher",
            position=self.pitcher_cfg.spawn_position,
            inner_radius=self.pitcher_cfg.inner_radius_m,
            inner_height=self.pitcher_cfg.inner_height_m,
            wall_thickness=self.pitcher_cfg.wall_thickness_m,
            base_thickness=self.pitcher_cfg.base_thickness_m,
            n_segments=self.pitcher_cfg.n_segments,
            color_rgb=self.pitcher_cfg.color_rgb,
            mass_kg=self.pitcher_cfg.mass_kg,
            grip_cfg=self.pitcher_cfg,
        )
        self.glass_prim_path = self._spawn_open_top_cylinder(
            prim_path=f"{containers_root}/Glass",
            position=self.glass_cfg.spawn_position,
            inner_radius=self.glass_cfg.inner_radius_m,
            inner_height=self.glass_cfg.inner_height_m,
            wall_thickness=self.glass_cfg.wall_thickness_m,
            base_thickness=self.glass_cfg.base_thickness_m,
            n_segments=self.glass_cfg.n_segments,
            color_rgb=self.glass_cfg.color_rgb,
            mass_kg=self.glass_cfg.mass_kg,
            grip_cfg=self.glass_cfg,
        )
        return self.pitcher_prim_path, self.glass_prim_path

    @staticmethod
    def _spawn_open_top_cylinder(
        *,
        prim_path: str,
        position: tuple[float, float, float],
        inner_radius: float,
        inner_height: float,
        wall_thickness: float,
        base_thickness: float,
        n_segments: int,
        color_rgb: tuple[float, float, float],
        mass_kg: float,
        grip_cfg: "PitcherConfig | GlassConfig",
    ) -> str:
        """Build one open-top hollow cylinder rigid body.

        ``position`` is the world-frame location of the cylinder's
        bottom-center: the floor's lowest face sits at ``position[2]`` and
        the walls extend upward to ``position[2] + base_thickness + inner_height``.

        Grasp physics (friction, contact/rest offsets, and the torsional patch
        radius) come from ``grip_cfg`` -- the container's own
        :class:`PitcherConfig` / :class:`GlassConfig`. These are deliberately
        grippier than the env's shared object defaults because the vessel is
        round and thin-walled, so the gripper only makes a small curved
        contact patch and would otherwise let the container twist/slip out
        while the arm carries it.
        """
        import importlib

        import isaaclab.sim as sim_utils
        from isaaclab.sim.schemas import (
            define_collision_properties,
            define_mass_properties,
            define_rigid_body_properties,
        )
        from isaaclab.sim.spawners.materials.physics_materials_cfg import (
            RigidBodyMaterialCfg,
        )
        from isaaclab.sim.spawners.materials.visual_materials_cfg import (
            PreviewSurfaceCfg,
        )
        from isaaclab.sim.spawners.shapes.shapes_cfg import CuboidCfg, CylinderCfg

        prim_utils = importlib.import_module("isaacsim.core.utils.prims")

        n = max(6, int(n_segments))
        ir = float(inner_radius)
        ih = float(inner_height)
        wt = float(wall_thickness)
        bt = float(base_thickness)
        outer_r = ir + wt
        r_mid = ir + wt * 0.5

        # 1. Root Xform with rigid body + mass + collision schemas.
        prim_utils.create_prim(prim_path, "Xform", translation=tuple(position))

        define_rigid_body_properties(
            prim_path,
            sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
            ),
        )
        define_mass_properties(prim_path, sim_utils.MassPropertiesCfg(mass=float(mass_kg)))
        define_collision_properties(
            prim_path,
            sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        )

        # 2. Shared visual + physics materials. The friction is intentionally
        #    high (and combine-mode "max") so the round, thin-walled vessel
        #    doesn't slip out of the gripper while the arm carries it.
        visual = PreviewSurfaceCfg(diffuse_color=tuple(color_rgb))
        physics_material = RigidBodyMaterialCfg(
            static_friction=float(grip_cfg.static_friction),
            dynamic_friction=float(grip_cfg.dynamic_friction),
            restitution=float(grip_cfg.restitution),
            friction_combine_mode=str(grip_cfg.friction_combine_mode),
        )
        col = sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=float(grip_cfg.contact_offset),
            rest_offset=float(grip_cfg.rest_offset),
            torsional_patch_radius=float(grip_cfg.torsional_patch_radius),
            min_torsional_patch_radius=float(grip_cfg.min_torsional_patch_radius),
        )

        # 3. Floor: solid cylinder centered at (0, 0, bt/2) in local frame.
        floor_cfg = CylinderCfg(
            radius=outer_r,
            height=bt,
            axis="Z",
            visual_material=visual,
            physics_material=physics_material,
            rigid_props=None,
            mass_props=None,
            collision_props=col,
        )
        floor_cfg.func(
            f"{prim_path}/Floor",
            floor_cfg,
            translation=(0.0, 0.0, bt * 0.5),
        )

        # 4. Wall segments around the cylinder. Each segment is a thin slab
        #    tangent to the cylinder, rotated by its angular position. We
        #    overlap adjacent segments by ~10% so the seams are sealed even
        #    after PhysX adds its contact offset.
        seg_height = ih
        seg_thickness = wt
        # Chord at outer radius, padded for overlap.
        seg_width = 2.0 * outer_r * math.sin(math.pi / n) * 1.10
        z_center = bt + seg_height * 0.5

        for i in range(n):
            angle = 2.0 * math.pi * i / n
            cx = r_mid * math.cos(angle)
            cy = r_mid * math.sin(angle)
            half = angle * 0.5
            quat_wxyz = (math.cos(half), 0.0, 0.0, math.sin(half))
            seg_cfg = CuboidCfg(
                size=(seg_thickness, seg_width, seg_height),
                visual_material=visual,
                physics_material=physics_material,
                rigid_props=None,
                mass_props=None,
                collision_props=col,
            )
            seg_cfg.func(
                f"{prim_path}/Wall_{i:02d}",
                seg_cfg,
                translation=(cx, cy, z_center),
                orientation=quat_wxyz,
            )

        return prim_path

    # ------------------------------------------------------------------
    # Pellets ("liquid" proxy)
    # ------------------------------------------------------------------
    def spawn_pellets(
        self,
        *,
        parent_prim_path: str = "/World/Origin1",
        seed: int | None = None,
    ) -> list[str]:
        """Spawn :data:`PelletConfig.count` dynamic spheres inside the pitcher.

        Pellets are spawned in a small jittered grid centered above the
        pitcher's spawn pose. Returns the list of pellet prim paths and stores
        them on ``self.pellet_prim_paths`` for later volume queries.

        Must be called *after* :meth:`spawn_containers`.
        """
        import importlib

        import isaaclab.sim as sim_utils
        from isaaclab.sim.spawners.materials.physics_materials_cfg import (
            RigidBodyMaterialCfg,
        )
        from isaaclab.sim.spawners.materials.visual_materials_cfg import (
            PreviewSurfaceCfg,
        )
        from isaaclab.sim.spawners.shapes.shapes_cfg import SphereCfg

        prim_utils = importlib.import_module("isaacsim.core.utils.prims")
        import omni.usd  # type: ignore

        stage = omni.usd.get_context().get_stage()
        if not stage.GetPrimAtPath(parent_prim_path).IsValid():
            prim_utils.create_prim(parent_prim_path, "Xform")

        pellets_root = f"{parent_prim_path.rstrip('/')}/Pellets"
        if not stage.GetPrimAtPath(pellets_root).IsValid():
            prim_utils.create_prim(pellets_root, "Xform")

        rng = random.Random(seed)
        cfg = self.pellet_cfg

        # Volume + mass per pellet
        volume_m3 = (4.0 / 3.0) * math.pi * (float(cfg.radius_m) ** 3)
        mass_kg = float(cfg.density_kg_m3) * volume_m3

        # Pellet spawn region: a *cylinder* inside the pitcher's cavity, sized
        # from PitcherConfig (so callers don't have to recompute extents when
        # they change the pitcher dimensions).
        px, py, pz = self.pitcher_cfg.spawn_position
        max_radius = max(
            0.0,
            float(self.pitcher_cfg.inner_radius_m)
            - max(float(cfg.wall_clearance_m), float(cfg.radius_m)),
        )
        # Vertical range is in pitcher-local coords (z=0 is the cylinder's
        # outer base, so spawn between bt+pellet_r and inner_height-pellet_r).
        floor_clearance = float(self.pitcher_cfg.base_thickness_m) + float(cfg.radius_m)
        ceiling_clearance = (
            float(self.pitcher_cfg.base_thickness_m)
            + float(self.pitcher_cfg.inner_height_m)
            - float(cfg.radius_m)
        )
        z_min = max(float(cfg.spawn_z_min_m), floor_clearance)
        z_max = min(float(cfg.spawn_z_max_m), ceiling_clearance)
        if z_max <= z_min:
            z_max = z_min + 1e-3

        # Pellet geometry/physics config -- shared across pellets.
        sphere_cfg = SphereCfg(
            radius=float(cfg.radius_m),
            visual_material=PreviewSurfaceCfg(diffuse_color=tuple(cfg.color_rgb)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=float(mass_kg)),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=float(cfg.contact_offset),
                rest_offset=float(cfg.rest_offset),
            ),
            physics_material=RigidBodyMaterialCfg(
                static_friction=float(cfg.static_friction),
                dynamic_friction=float(cfg.dynamic_friction),
                restitution=float(cfg.restitution),
            ),
        )

        spawned: list[str] = []
        for i in range(int(cfg.count)):
            # Uniform sample in a disc: r = R * sqrt(u), theta = 2*pi*v.
            r = max_radius * math.sqrt(rng.uniform(0.0, 1.0))
            theta = rng.uniform(0.0, 2.0 * math.pi)
            x = px + r * math.cos(theta)
            y = py + r * math.sin(theta)
            z = pz + rng.uniform(z_min, z_max)
            pellet_path = f"{pellets_root}/Pellet_{i + 1:03d}"
            sphere_cfg.func(pellet_path, sphere_cfg, translation=(x, y, z))
            spawned.append(pellet_path)

        self.pellet_prim_paths = spawned
        print(
            f"[PouringEnv] Spawned {len(spawned)} pellets "
            f"(radius={cfg.radius_m * 1000:.1f}mm, mass={mass_kg * 1000:.2f}g each)."
        )
        return spawned

    # ------------------------------------------------------------------
    # Volume sensor: count pellets currently inside the glass
    # ------------------------------------------------------------------
    def count_pellets_in_glass(self) -> int:
        """Count pellets whose world position falls inside the volume sensor box."""
        return len(self._pellets_in_volume())

    def fraction_pellets_in_glass(self) -> float:
        """Fraction (in [0, 1]) of spawned pellets currently inside the glass."""
        if not self.pellet_prim_paths:
            return 0.0
        return float(self.count_pellets_in_glass()) / float(len(self.pellet_prim_paths))

    def amount_poured_grams(self) -> float:
        """Mass (g) of pellets currently inside the glass.

        Convenience wrapper -- assumes uniform pellets, multiplies count by
        per-pellet mass derived from :class:`PelletConfig.radius_m` and
        :class:`PelletConfig.density_kg_m3`.
        """
        cfg = self.pellet_cfg
        volume_m3 = (4.0 / 3.0) * math.pi * (float(cfg.radius_m) ** 3)
        per_pellet_g = float(cfg.density_kg_m3) * volume_m3 * 1000.0
        return per_pellet_g * float(self.count_pellets_in_glass())

    def _pellets_in_volume(self) -> list[str]:
        """Return the prim paths of pellets currently inside the volume sensor."""
        if not self.pellet_prim_paths:
            return []
        try:
            import omni.usd  # type: ignore
            from pxr import Usd, UsdGeom  # type: ignore
        except Exception:
            return []

        stage = omni.usd.get_context().get_stage()
        cx, cy, cz = (
            self.glass_cfg.spawn_position[0],
            self.glass_cfg.spawn_position[1],
            self.glass_cfg.spawn_position[2] + self.volume_sensor_cfg.z_offset_above_base_m,
        )
        hx, hy, hz = self.volume_sensor_cfg.half_extents_xyz

        inside: list[str] = []
        for path in self.pellet_prim_paths:
            prim = stage.GetPrimAtPath(str(path))
            if not prim.IsValid():
                continue
            try:
                xformable = UsdGeom.Xformable(prim)
                M = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                tx = float(M[3][0])
                ty = float(M[3][1])
                tz = float(M[3][2])
            except Exception:
                continue
            if (
                abs(tx - cx) <= hx
                and abs(ty - cy) <= hy
                and abs(tz - cz) <= hz
            ):
                inside.append(path)
        return inside

    # ------------------------------------------------------------------
    # Optional clutter loader (so callers can scatter extra YCB objects)
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
        """Build a USD-mode :class:`ObjectLoaderConfig` for scattering YCB clutter.

        Useful when the task wants distractor objects on the table; the pitcher
        and glass themselves are spawned via :meth:`spawn_containers` and are
        not affected by this config.
        """
        bounds = SpawnBounds(
            min_xyz=tuple(spawn_min) if spawn_min is not None else (0.30, -0.05, 0.85),
            max_xyz=tuple(spawn_max) if spawn_max is not None else (0.55, 0.05, 0.92),
        )
        labels = (
            list(include_labels) if include_labels is not None else self.clutter_include_labels
        )
        scale = scale_range if scale_range is not None else self.clutter_scale_range

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
