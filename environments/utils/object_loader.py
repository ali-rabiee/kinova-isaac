from __future__ import annotations

import random
import math
import importlib
import re
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from isaaclab.sim.spawners.from_files import UsdFileCfg

@dataclass
class SpawnBounds:
    """Axis-aligned bounding box for random spawn (in meters, relative to parent prim)."""
    min_xyz: tuple[float, float, float]
    max_xyz: tuple[float, float, float]


@dataclass
class ObjectLoaderConfig:
    """Configuration for spawning USD objects from dataset folders."""
    
    dataset_dirs: list[str]
    bounds: SpawnBounds
    min_distance: float = 0.08
    # If True, enforce min_distance in XY only (ignores Z). Useful when objects spawn at varying Z
    # (e.g., dropping from above the table) but we still want them spaced out on the tabletop.
    min_distance_xy_only: bool = False
    max_placement_tries_per_object: int = 500
    parent_objects_relpath: str = "Objects"
    
    # Optional uniform scale range applied per object (x=y=z)
    uniform_scale_range: tuple[float, float] | None = None
    # Optional label filtering: if provided, only objects whose derived label
    # (basename without numeric prefix or extension) is in this list are spawned.
    include_labels: list[str] | None = None
    # If True, print the set of available labels at construction time.
    print_available_labels: bool = False
    
    # Physics and placement
    enable_physics: bool = True
    mass_kg: float | None = None
    density: float | None = 500.0
    contact_offset: float = 0.005
    rest_offset: float = 0.001
    # Friction and contact quality
    static_friction: float = 100.0
    dynamic_friction: float = 100.0
    restitution: float = 0.0
    friction_combine_mode: str = "max"
    torsional_patch_radius: float = 0.02
    min_torsional_patch_radius: float = 0.01
    
    # Z placement control
    snap_z_to: float | None = None
    z_clearance: float = 0.005
    z_jitter_max: float = 0.01
    
    # Orientation at spawn: Euler degrees in XYZ order
    orientation_euler_deg: tuple[float, float, float] | None = None
    
    # Visual fallback
    apply_preview_surface: bool = False
    preview_surface_diffuse: tuple[float, float, float] = (0.7, 0.7, 0.7)

    # Collision handling
    # If True, disable authored mesh collisions on imported assets (often triangle-mesh / meshSimplification)
    # and attach a simple box collision proxy under the rigid body root. This avoids PhysX GPU spam:
    # "triangle mesh collision ... cannot be part of a dynamic body".
    # NOTE: Default is False to preserve “solid” asset collisions.
    # Enable proxies only if you explicitly want simplified collisions.
    use_collision_proxies: bool = False
    collision_proxy_padding_m: float = 0.003
    
    # Spawn mode: "usd" (default) or "box"
    spawn_mode: str = "usd"
    # Box dimensions (only used if spawn_mode="box")
    box_size_min: tuple[float, float, float] = (0.04, 0.04, 0.04)
    box_size_max: tuple[float, float, float] = (0.06, 0.06, 0.06)
    box_color: tuple[float, float, float] = (0.2, 0.8, 0.2)
    # Optional palette for per-object coloring in box mode. If provided, color cycles by object index.
    box_color_palette: Sequence[tuple[float, float, float]] | None = None
    # Optional names corresponding to `box_color_palette` (not used by ObjectLoader itself, but useful for logging).
    box_color_names: Sequence[str] | None = None

    # Error reporting / strictness
    # If True, print full tracebacks for exceptions that were previously swallowed.
    report_exceptions: bool = True
    # If True, re-raise exceptions instead of continuing (useful to stop on first failure).
    raise_on_exceptions: bool = False


class ObjectLoader:
    """Loads USD objects from directories and spawns them randomly into the scene."""

    def __init__(self, cfg: ObjectLoaderConfig):
        self.cfg = cfg
        self._usd_file_paths: list[str] = []
        if self.cfg.spawn_mode == "usd":
            self._usd_file_paths = self._collect_usd_files(cfg.dataset_dirs)
            if len(self._usd_file_paths) == 0:
                raise FileNotFoundError(f"No .usd files found in dataset_dirs={cfg.dataset_dirs}.")

        # Pre-compute labels for all USDs using the same logic as spawn()
        self._usd_path_to_label: dict[str, str] = {
            path: self._derive_label_from_usd_path(path) for path in self._usd_file_paths
        }
        # Map of spawned prim path -> human-readable label (derived from USD filename)
        self._last_spawn_label_map: dict[str, str] = {}

        # Optionally print all available labels (unique, sorted)
        if self.cfg.print_available_labels:
            unique_labels = sorted(set(self._usd_path_to_label.values()))
            print("[ObjectLoader] Available labels:")
            for lbl in unique_labels:
                print(f"  - {lbl}")

    def _handle_exception(self, where: str, exc: BaseException) -> None:
        """Centralized exception handler for optionally strict error reporting."""
        if bool(getattr(self.cfg, "report_exceptions", True)):
            print(f"[ObjectLoader][ERROR] {where}: {exc}")
            print(traceback.format_exc())
        if bool(getattr(self.cfg, "raise_on_exceptions", False)):
            raise exc

    def get_last_spawn_labels(self) -> dict[str, str]:
        """Return a copy of the last spawn's mapping from prim path to label.

        Labels are derived from the spawned USD file's basename (without extension).
        Returns an empty dict if nothing has been spawned yet.
        """
        return dict(self._last_spawn_label_map)

    @staticmethod
    def _derive_label_from_usd_path(usd_path: str) -> str:
        """Derive a human-readable label from a USD file path.

        This mirrors the logic used in spawn():
        - Take basename
        - Strip known extensions
        - Strip leading numeric prefixes like "037_" or "005-"
        """
        try:
            base_name = str(usd_path).rstrip("/").split("/")[-1]
            lower_name = base_name.lower()
            raw = base_name
            for ext in (".usd", ".usda", ".usdc", ".usdz"):
                if lower_name.endswith(ext):
                    raw = base_name[: -len(ext)]
                    break
            # Strip leading numeric id prefixes like "037_" or "005-"
            simplified = re.sub(r"^\d+[_-]+", "", raw)
            return simplified
        except Exception:
            return str(usd_path)

    @staticmethod
    def _collect_usd_files(dirs: Iterable[str]) -> list[str]:
        """Collect USD files from local directories and/or Nucleus directories."""
        usd_files: list[str] = []
        omni_client = None
        try:
            omni_client = importlib.import_module("omni.client")
        except Exception:
            omni_client = None

        def is_nucleus_dir(path_str: str) -> bool:
            if path_str.startswith(("omniverse://", "http://", "https://")):
                return True
            if omni_client is None:
                return False
            try:
                res, _ = omni_client.list(path_str)
                return res == getattr(omni_client, "Result").OK
            except Exception:
                return False

        for d in dirs:
            if not d:
                continue
            d_str = str(d)
            if is_nucleus_dir(d_str):
                usd_files.extend(ObjectLoader._list_usd_files_in_nucleus_dir(d_str, omni_client))
                continue
            # Local filesystem
            base = Path(d_str).expanduser().resolve()
            if not base.exists() or not base.is_dir():
                continue
            for path in base.rglob("*.usd"):
                name = path.name
                if name.startswith(".") or name.endswith("~") or ".thumb" in name.lower():
                    continue
                usd_files.append(str(path))
        return sorted(set(usd_files))

    @staticmethod
    def _list_usd_files_in_nucleus_dir(root_url: str, omni_client) -> list[str]:
        """Recursively list USD files from Nucleus directory."""
        if omni_client is None:
            return []
        urls: list[str] = []
        stack: list[str] = [root_url.rstrip("/")]
        seen: set[str] = set()
        
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            
            try:
                res, entries = omni_client.list(current)
                if res != getattr(omni_client, "Result").OK:
                    continue
                    
                for entry in entries:
                    name = getattr(entry, "name", None) or getattr(entry, "relative_path", None)
                    if not name:
                        continue
                    child = f"{current}/{name}" if not current.endswith("/") else f"{current}{name}"
                    
                    if str(name).lower().endswith(".usd"):
                        if ".thumb" not in str(name).lower():
                            urls.append(child)
                        continue
                        
                    # Try treating as directory
                    try:
                        res2, _ = omni_client.list(child)
                        if res2 == getattr(omni_client, "Result").OK:
                            stack.append(child)
                    except Exception:
                        # Skip non-directories
                        continue
            except Exception:
                # Skip failed list attempts; keep walking other branches.
                continue
                
        return urls

    @staticmethod
    def _ensure_parent_prim(parent_prim_path: str, child_relpath: str) -> str:
        """Ensures parent Xform exists and returns full prim path to child root."""
        prim_utils = importlib.import_module("isaacsim.core.utils.prims")
        stage_utils = importlib.import_module("isaacsim.core.utils.stage")
        
        stage = stage_utils.get_current_stage()
        
        # Create parent if needed
        parent_prim = stage.GetPrimAtPath(parent_prim_path)
        if not parent_prim.IsValid():
            prim_utils.create_prim(parent_prim_path, "Xform")
            
        # Create objects root if needed
        objects_root = f"{parent_prim_path.rstrip('/')}/{child_relpath.strip('/')}"
        objects_root_prim = stage.GetPrimAtPath(objects_root)
        if not objects_root_prim.IsValid():
            prim_utils.create_prim(objects_root, "Xform")
            
        return objects_root

    @staticmethod
    def _sample_position(bounds: SpawnBounds) -> tuple[float, float, float]:
        x = random.uniform(bounds.min_xyz[0], bounds.max_xyz[0])
        y = random.uniform(bounds.min_xyz[1], bounds.max_xyz[1])
        z = random.uniform(bounds.min_xyz[2], bounds.max_xyz[2])
        return (x, y, z)

    @staticmethod
    def _distance(a: Sequence[float], b: Sequence[float]) -> float:
        dx, dy, dz = a[0] - b[0], a[1] - b[1], a[2] - b[2]
        return (dx * dx + dy * dy + dz * dz) ** 0.5

    @staticmethod
    def _quat_mul(q1: tuple[float, float, float, float], q2: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        """Quaternion multiplication q = q1 * q2 in (w, x, y, z) format."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )

    @staticmethod
    def _yaw_quat(yaw_rad: float) -> tuple[float, float, float, float]:
        """Quaternion for a pure Z-axis rotation by yaw_rad (radians)."""
        half = 0.5 * yaw_rad
        return (math.cos(half), 0.0, 0.0, math.sin(half))

    def _generate_positions(self, count: int) -> list[tuple[float, float, float]]:
        """Generate non-overlapping positions for objects."""
        positions: list[tuple[float, float, float]] = []
        tries_left = self.cfg.max_placement_tries_per_object * count
        xy_only = bool(getattr(self.cfg, "min_distance_xy_only", False))
        
        while len(positions) < count and tries_left > 0:
            candidate = list(self._sample_position(self.cfg.bounds))
            
            # Override Z if snapping to table
            if self.cfg.snap_z_to is not None:
                z_jitter = random.uniform(0.0, self.cfg.z_jitter_max)
                candidate[2] = self.cfg.snap_z_to + self.cfg.z_clearance + z_jitter
                
            candidate_t = (candidate[0], candidate[1], candidate[2])
            
            if xy_only:
                ok = True
                for p in positions:
                    dx = float(candidate_t[0] - p[0])
                    dy = float(candidate_t[1] - p[1])
                    if math.hypot(dx, dy) < float(self.cfg.min_distance):
                        ok = False
                        break
                if ok:
                    positions.append(candidate_t)
            else:
                if all(self._distance(candidate_t, p) >= self.cfg.min_distance for p in positions):
                    positions.append(candidate_t)
                
            tries_left -= 1
            
        return positions

    def _compute_orientation(self) -> tuple[float, float, float, float] | None:
        """Compute quaternion from Euler degrees config."""
        if self.cfg.orientation_euler_deg is None:
            return None
            
        try:
            import torch
            from isaaclab.utils.math import quat_from_euler_xyz
            
            roll_deg, pitch_deg, yaw_deg = self.cfg.orientation_euler_deg
            r = torch.tensor([math.radians(float(roll_deg))])
            p = torch.tensor([math.radians(float(pitch_deg))])
            y = torch.tensor([math.radians(float(yaw_deg))])
            
            q_tensor = quat_from_euler_xyz(r, p, y)[0]
            qw, qx, qy, qz = [float(v) for v in q_tensor.tolist()]
            return (qw, qx, qy, qz)
            
        except Exception:
            # Fallback for single-axis rotations
            rx, ry, rz = self.cfg.orientation_euler_deg
            if abs(rx) > 0 and abs(ry) == 0 and abs(rz) == 0:
                a = math.radians(rx) * 0.5
                return (math.cos(a), math.sin(a), 0.0, 0.0)
            elif abs(ry) > 0 and abs(rx) == 0 and abs(rz) == 0:
                a = math.radians(ry) * 0.5
                return (math.cos(a), 0.0, math.sin(a), 0.0)
            elif abs(rz) > 0 and abs(rx) == 0 and abs(ry) == 0:
                a = math.radians(rz) * 0.5
                return (math.cos(a), 0.0, 0.0, math.sin(a))
            return None

    def _ensure_object_physics(self, prim_path: str) -> None:
        """Ensure object has proper physics schemas applied."""
        try:
            import isaaclab.sim as sim_utils
            from isaaclab.sim.schemas import (
                define_rigid_body_properties,
                define_collision_properties,
                define_mass_properties,
            )
            from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
            # USD schema access for mesh-collision approximation tweaks (correct API)
            import omni.usd
            from pxr import Usd, UsdGeom, UsdPhysics, Gf
            from isaaclab.sim import schemas as lab_schemas
            from isaaclab.sim.schemas import schemas_cfg as lab_schemas_cfg
            from isaaclab.sim.spawners.shapes.shapes_cfg import CuboidCfg

            def _force_convex_hull_on_meshes(root_prim_path: str) -> None:
                """Force convexHull mesh collision approximation on all Mesh prims under a spawned object.

                IMPORTANT: The correct attribute is driven by `UsdPhysics.MeshCollisionAPI` (not PhysxCollisionAPI).
                IsaacLab exposes helpers that set this via `schemas.modify_mesh_collision_properties`.
                """
                try:
                    stage = omni.usd.get_context().get_stage()
                    root = stage.GetPrimAtPath(root_prim_path)
                    if not root or not root.IsValid():
                        return
                    meshes_seen = 0
                    meshes_set = 0
                    # Traverse including instance proxies so we can detect collision meshes inside instanceable geometry.
                    # NOTE: Instance proxies cannot be authored to directly; we must instead author to the prototype prim.
                    for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
                        if not p.IsValid():
                            continue
                        if not p.IsA(UsdGeom.Mesh):
                            continue
                        meshes_seen += 1
                        try:
                            target_prim_path = p.GetPath().pathString  # default: edit the prim itself
                            # If this is an instance proxy, try to resolve to the prototype prim to author overrides.
                            if hasattr(p, "IsInstanceProxy") and p.IsInstanceProxy():  # type: ignore[attr-defined]
                                # USD API varies; best-effort to find the corresponding prim in prototype.
                                proto_prim = None
                                if hasattr(p, "GetPrimInPrototype"):
                                    try:
                                        proto_prim = p.GetPrimInPrototype()  # type: ignore[attr-defined]
                                    except Exception:
                                        proto_prim = None
                                if proto_prim is not None and hasattr(proto_prim, "GetPath"):
                                    target_prim_path = proto_prim.GetPath().pathString
                            lab_schemas.modify_mesh_collision_properties(
                                prim_path=target_prim_path,
                                cfg=lab_schemas_cfg.ConvexHullPropertiesCfg(hull_vertex_limit=64),
                                stage=stage,
                            )
                            meshes_set += 1
                        except Exception as mesh_err:
                            # Print once so we know why this isn't taking effect.
                            if not hasattr(self, "_warned_mesh_collision_apply_failure"):
                                setattr(self, "_warned_mesh_collision_apply_failure", True)
                                print(
                                    f"[ObjectLoader][WARN] Failed to set MeshCollisionAPI convexHull on '{p.GetPath()}': {mesh_err}"
                                )
                    # Debug once per run (not per object) to confirm this path works.
                    if meshes_seen > 0:
                        if not hasattr(self, "_printed_convex_hull_stats"):
                            setattr(self, "_printed_convex_hull_stats", True)
                            print(f"[ObjectLoader] convexHull collision applied to meshes: {meshes_set}/{meshes_seen} under {root_prim_path}")
                except Exception:
                    return

            def _disable_all_authored_colliders(root_prim_path: str) -> None:
                """Disable any authored colliders (often triangle mesh) under an object.

                We do this because many referenced assets author triangle-mesh colliders inside instanceable geometry,
                which PhysX disallows for dynamic bodies. Disabling them prevents repeated parser errors + UI lag.
                """
                try:
                    stage = omni.usd.get_context().get_stage()
                    root = stage.GetPrimAtPath(root_prim_path)
                    if not root or not root.IsValid():
                        return
                    disabled = 0
                    meshes_seen = 0
                    meshes_fixed = 0
                    for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
                        if not p.IsValid():
                            continue
                        # Resolve a prim we can author to (instance proxies cannot be authored to directly).
                        target_prim = p
                        if hasattr(p, "IsInstanceProxy") and p.IsInstanceProxy():  # type: ignore[attr-defined]
                            if hasattr(p, "GetPrimInPrototype"):
                                try:
                                    target_prim = p.GetPrimInPrototype()  # type: ignore[attr-defined]
                                except Exception:
                                    target_prim = p

                        # Disable collisions wherever collision API is applicable.
                        try:
                            api = UsdPhysics.CollisionAPI.Apply(target_prim)
                            attr = api.GetCollisionEnabledAttr()
                            if not attr or not attr.IsValid():
                                attr = api.CreateCollisionEnabledAttr()
                            attr.Set(False)
                            disabled += 1
                        except Exception:
                            # ignore
                            pass

                        # For meshes: force convexHull approximation using *define* (applies PhysX convex-hull API).
                        if target_prim.IsA(UsdGeom.Mesh):
                            meshes_seen += 1
                            try:
                                lab_schemas.define_mesh_collision_properties(
                                    prim_path=target_prim.GetPath().pathString,
                                    cfg=lab_schemas_cfg.ConvexHullPropertiesCfg(hull_vertex_limit=64),
                                    stage=stage,
                                )
                                meshes_fixed += 1
                            except Exception:
                                pass

                    if (disabled > 0 or meshes_seen > 0) and not hasattr(self, "_printed_disable_collider_stats"):
                        setattr(self, "_printed_disable_collider_stats", True)
                        print(
                            f"[ObjectLoader] Disabled authored colliders under {root_prim_path}: "
                            f"collision_disabled={disabled} meshes_fixed={meshes_fixed}/{meshes_seen}"
                        )
                except Exception:
                    return

            def _attach_box_collision_proxy(root_prim_path: str) -> None:
                """Attach a simple box collider under the rigid body root to replace mesh collision."""
                try:
                    stage = omni.usd.get_context().get_stage()
                    root = stage.GetPrimAtPath(root_prim_path)
                    if not root or not root.IsValid():
                        return

                    # Compute world AABB for the object root
                    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"], useExtentsHint=True)
                    world_bound = bbox_cache.ComputeWorldBound(root)
                    aligned = world_bound.ComputeAlignedBox()
                    bmin = aligned.GetMin()
                    bmax = aligned.GetMax()
                    size_w = (float(bmax[0] - bmin[0]), float(bmax[1] - bmin[1]), float(bmax[2] - bmin[2]))
                    # If bounds are invalid/tiny (often happens if asset geometry isn't loaded yet),
                    # skip proxy creation and fall back to normal collision handling.
                    if any((not math.isfinite(v) or v <= 1e-6) for v in size_w):
                        print(f"[ObjectLoader][WARN] Proxy AABB invalid for '{root_prim_path}' (size_w={size_w}); skipping proxy.")
                        return
                    # Add small padding so contact doesn't miss due to approximation
                    pad = float(getattr(self.cfg, "collision_proxy_padding_m", 0.0))
                    size_w = (max(1e-3, size_w[0] + 2 * pad), max(1e-3, size_w[1] + 2 * pad), max(1e-3, size_w[2] + 2 * pad))
                    center_w = Gf.Vec3d((bmin[0] + bmax[0]) * 0.5, (bmin[1] + bmax[1]) * 0.5, (bmin[2] + bmax[2]) * 0.5)

                    # Convert world center to root-local translation
                    xformable = UsdGeom.Xformable(root)
                    l2w = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    w2l = l2w.GetInverse()
                    center_l = w2l.Transform(center_w)
                    translation_l = (float(center_l[0]), float(center_l[1]), float(center_l[2]))

                    # IMPORTANT: The object root can have a scale transform (common in asset USDs).
                    # The cuboid is authored in *local* units, so we must compensate to keep the proxy
                    # aligned with the visible mesh in world units. Otherwise you can get "invisible"
                    # oversized/undersized colliders that the robot bumps into.
                    try:
                        # TransformDir ignores translation; its length encodes scale along each local axis.
                        sx = float(l2w.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0)).GetLength())
                        sy = float(l2w.TransformDir(Gf.Vec3d(0.0, 1.0, 0.0)).GetLength())
                        sz = float(l2w.TransformDir(Gf.Vec3d(0.0, 0.0, 1.0)).GetLength())
                        sx = sx if sx > 1e-6 else 1.0
                        sy = sy if sy > 1e-6 else 1.0
                        sz = sz if sz > 1e-6 else 1.0
                        size_l = (float(size_w[0]) / sx, float(size_w[1]) / sy, float(size_w[2]) / sz)
                    except Exception:
                        size_l = size_w

                    proxy_path = f"{root_prim_path}/CollisionProxy"
                    # Spawn a cuboid geometry WITHOUT rigid body (must not create nested rigid bodies).
                    proxy_cfg = CuboidCfg(
                        size=size_l,
                        visible=False,
                        rigid_props=None,
                        mass_props=None,
                        collision_props=sim_utils.CollisionPropertiesCfg(
                            collision_enabled=True,
                            contact_offset=self.cfg.contact_offset,
                            rest_offset=self.cfg.rest_offset,
                            torsional_patch_radius=self.cfg.torsional_patch_radius,
                            min_torsional_patch_radius=self.cfg.min_torsional_patch_radius,
                        ),
                    )
                    # Safe: only create if it doesn't already exist
                    try:
                        if stage.GetPrimAtPath(proxy_path).IsValid():
                            return
                    except Exception:
                        pass
                    proxy_cfg.func(proxy_path, proxy_cfg, translation=translation_l, orientation=(1.0, 0.0, 0.0, 0.0))
                    if not hasattr(self, "_printed_proxy_collider"):
                        setattr(self, "_printed_proxy_collider", True)
                        print(f"[ObjectLoader] Using box collision proxies (example created at {proxy_path})")
                except Exception as e:
                    if not hasattr(self, "_warned_proxy_collider_failure"):
                        setattr(self, "_warned_proxy_collider_failure", True)
                        print(f"[ObjectLoader][WARN] Failed to create collision proxy under '{root_prim_path}': {e}")
            
            # Apply rigid body
            rb_cfg = sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
            )
            define_rigid_body_properties(prim_path, rb_cfg)
            
            # Apply collision (including torsional/contact tuning)
            # NOTE: Many YCB assets have authored triangle-mesh collision. PhysX GPU does not allow
            # dynamic triangle-mesh collision and will repeatedly warn + fall back at runtime.
            # Prefer convex hull approximation when the API supports it.
            try:
                col_cfg = sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                    contact_offset=self.cfg.contact_offset,
                    rest_offset=self.cfg.rest_offset,
                    torsional_patch_radius=self.cfg.torsional_patch_radius,
                    min_torsional_patch_radius=self.cfg.min_torsional_patch_radius,
                    collision_approximation="convexHull",
                )
            except Exception:
                col_cfg = sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                    contact_offset=self.cfg.contact_offset,
                    rest_offset=self.cfg.rest_offset,
                    torsional_patch_radius=self.cfg.torsional_patch_radius,
                    min_torsional_patch_radius=self.cfg.min_torsional_patch_radius,
                )
            # Collision strategy:
            # - If use_collision_proxies: disable authored colliders (often triangle mesh) + attach a box collider.
            # - Else: best-effort set convexHull approximation on meshes.
            if getattr(self.cfg, "use_collision_proxies", True):
                # IMPORTANT: don't apply collision properties on the rigid-body root in proxy mode
                # (it can re-enable collisions on authored meshes via nested application).
                _disable_all_authored_colliders(prim_path)
                _attach_box_collision_proxy(prim_path)
            else:
                define_collision_properties(prim_path, col_cfg)
                _force_convex_hull_on_meshes(prim_path)
            
            # Bind a high-friction physics material to the object root
            mat_cfg = RigidBodyMaterialCfg(
                static_friction=self.cfg.static_friction,
                dynamic_friction=self.cfg.dynamic_friction,
                restitution=self.cfg.restitution,
                friction_combine_mode=self.cfg.friction_combine_mode,  # type: ignore[arg-type]
            )
            mat_prim = f"{prim_path}/ObjFrictionMaterial"
            mat_cfg.func(mat_prim, mat_cfg)
            sim_utils.bind_physics_material(prim_path, mat_prim)

            # Apply mass
            mass_val = 0.3  # Default 300g
            if self.cfg.mass_kg is not None:
                mass_val = self.cfg.mass_kg
            elif self.cfg.density is not None:
                mass_val = self.cfg.density * 0.0001  # Rough volume estimate
                
            mass_cfg = sim_utils.MassPropertiesCfg(mass=mass_val)
            define_mass_properties(prim_path, mass_cfg)
            
        except Exception as e:
            # Report and optionally raise (no more silent failures).
            self._handle_exception(f"_ensure_object_physics(prim_path={prim_path})", e)

    def spawn(self, parent_prim_path: str, num_objects: int) -> list[str]:
        """Spawn random objects under the given parent prim."""
        import isaaclab.sim as sim_utils

        # Prepare objects container
        objects_root = self._ensure_parent_prim(parent_prim_path, self.cfg.parent_objects_relpath)

        # Select objects (USD paths or placeholders)
        selection: list[str] = []
        if self.cfg.spawn_mode == "usd":
            # Optionally filter USDs by include_labels if provided
            candidate_paths = self._usd_file_paths
            if self.cfg.include_labels:
                include_set = {lbl.lower() for lbl in self.cfg.include_labels}
                filtered = [
                    p for p in self._usd_file_paths
                    if self._usd_path_to_label.get(p, "").lower() in include_set
                ]
                if len(filtered) == 0:
                    print(f"[ObjectLoader][WARN] No USDs matched include_labels={self.cfg.include_labels}; using all objects.")
                else:
                    candidate_paths = filtered

            if num_objects <= len(candidate_paths):
                selection = random.sample(candidate_paths, k=num_objects)
            else:
                selection = [random.choice(candidate_paths) for _ in range(num_objects)]
        else:
            # Box mode
            selection = ["box"] * num_objects

        positions = self._generate_positions(len(selection))
        base_orientation = self._compute_orientation()
        spawned_prim_paths: list[str] = []

        # Reset label map for this spawn call
        self._last_spawn_label_map = {}

        # Lazy import for box spawning
        CuboidCfg = None
        PreviewSurfaceCfg = None
        RigidBodyMaterialCfg = None
        if self.cfg.spawn_mode == "box":
            try:
                from isaaclab.sim.spawners.shapes.shapes_cfg import CuboidCfg
                from isaaclab.sim.spawners.materials.visual_materials_cfg import PreviewSurfaceCfg
                from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
            except Exception:
                # This is a real setup error in box mode; report it.
                print("[ObjectLoader][ERROR] Failed to import box spawning dependencies from IsaacLab.")
                print(traceback.format_exc())
                if getattr(self.cfg, "raise_on_exceptions", False):
                    raise
            if CuboidCfg is None:
                print(
                    "[ObjectLoader][ERROR] spawn_mode='box' but required IsaacLab shape spawners could not be imported. "
                    "This would cause silent spawn failures. Check your IsaacLab installation/environment."
                )
                return []

        for idx, (item, pos) in enumerate(zip(selection, positions), start=1):
            prim_path = f"{objects_root}/Obj_{idx:02d}"
            label = f"Obj_{idx:02d}"
            
            try:
                # Per-object random yaw around Z to diversify object orientation
                yaw_rad = random.uniform(-math.pi, math.pi)
                yaw_quat = self._yaw_quat(yaw_rad)

                if base_orientation is not None:
                    # Combine configured orientation with random yaw
                    orientation = self._quat_mul(yaw_quat, base_orientation)
                else:
                    orientation = yaw_quat

                if self.cfg.spawn_mode == "box" and CuboidCfg is not None:
                    # Box spawn
                    sx = random.uniform(self.cfg.box_size_min[0], self.cfg.box_size_max[0])
                    sy = random.uniform(self.cfg.box_size_min[1], self.cfg.box_size_max[1])
                    sz = random.uniform(self.cfg.box_size_min[2], self.cfg.box_size_max[2])

                    # Visual: optionally color each box differently (cycles through palette).
                    box_color = tuple(self.cfg.box_color)
                    try:
                        pal = getattr(self.cfg, "box_color_palette", None)
                        if pal is not None and len(pal) > 0:
                            box_color = tuple(float(v) for v in pal[(idx - 1) % len(pal)])
                    except Exception:
                        box_color = tuple(self.cfg.box_color)
                    
                    obj_cfg = CuboidCfg(
                        size=(sx, sy, sz),
                        visual_material=PreviewSurfaceCfg(diffuse_color=box_color),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            rigid_body_enabled=True,
                            kinematic_enabled=False,
                            disable_gravity=False,
                        ),
                        mass_props=sim_utils.MassPropertiesCfg(mass=self.cfg.mass_kg or 0.1),
                        collision_props=sim_utils.CollisionPropertiesCfg(
                            collision_enabled=True,
                            contact_offset=self.cfg.contact_offset,
                            rest_offset=self.cfg.rest_offset,
                            torsional_patch_radius=self.cfg.torsional_patch_radius,
                            min_torsional_patch_radius=self.cfg.min_torsional_patch_radius,
                        ),
                    )
                    obj_cfg.func(prim_path, obj_cfg, translation=pos, orientation=orientation)
                    label = "box"

                    # Bind friction material manually since we don't call _ensure_object_physics
                    if RigidBodyMaterialCfg is not None:
                        mat_cfg = RigidBodyMaterialCfg(
                            static_friction=self.cfg.static_friction,
                            dynamic_friction=self.cfg.dynamic_friction,
                            restitution=self.cfg.restitution,
                            friction_combine_mode=self.cfg.friction_combine_mode,  # type: ignore[arg-type]
                        )
                        mat_prim = f"{prim_path}/ObjFrictionMaterial"
                        mat_cfg.func(mat_prim, mat_cfg)
                        sim_utils.bind_physics_material(prim_path, mat_prim)

                else:
                    # USD spawn
                    usd_path = item
                    scale_arg = None
                    if self.cfg.uniform_scale_range is not None:
                        smin, smax = self.cfg.uniform_scale_range
                        s = random.uniform(smin, smax)
                        scale_arg = (s, s, s)

                    usd_cfg = UsdFileCfg(
                        usd_path=str(usd_path),
                        scale=scale_arg,
                    )

                    # Optional preview material
                    if self.cfg.apply_preview_surface:
                        vm_cfg = importlib.import_module("isaaclab.sim.spawners.materials.visual_materials_cfg")
                        preview = vm_cfg.PreviewSurfaceCfg(diffuse_color=self.cfg.preview_surface_diffuse)
                        usd_cfg.visual_material = preview
                    
                    label = self._usd_path_to_label.get(str(usd_path), f"Obj_{idx:02d}")

                    # Spawn object
                    usd_cfg.func(prim_path, usd_cfg, translation=pos, orientation=orientation)
                    
                    # Ensure physics (report on failure)
                    if self.cfg.enable_physics:
                        self._ensure_object_physics(prim_path)
                
                spawned_prim_paths.append(prim_path)
                self._last_spawn_label_map[prim_path] = label
                
            except Exception as e:
                # Report and optionally raise (no more silent failure).
                self._handle_exception(
                    f"spawn(idx={idx}, prim_path={prim_path}, item={item}, spawn_mode={self.cfg.spawn_mode})",
                    e,
                )
                # If not strict, keep going so you can see multiple failures.
                continue

        return spawned_prim_paths