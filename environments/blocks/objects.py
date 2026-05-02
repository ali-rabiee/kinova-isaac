from __future__ import annotations

from typing import Dict, Optional

from environments.blocks.language import BOX_COLORS, box_desc_from_prim, make_language_command


class BlocksObjectManager:
    """Owns object spawning, labels, target selection, and episode respawns for blocks."""

    def __init__(
        self,
        *,
        args,
        phys,
        sim,
        scene_origins,
        default_scene,
        isaac_nucleus_dir: str,
        parent_prim_path: str = "/World/Origin1",
    ) -> None:
        self.args = args
        self.phys = phys
        self.sim = sim
        self.scene_origins = scene_origins
        self.default_scene = default_scene
        self.isaac_nucleus_dir = str(isaac_nucleus_dir)
        self.parent_prim_path = str(parent_prim_path)
        self.spawned_paths: list[str] = []
        self.id_to_label: Dict[str, str] = {}
        self.loader = None
        self._respawn_rigidprims: dict[str, object] = {}

    def spawn_objects(self) -> tuple[list[str], Dict[str, str]]:
        if getattr(self.args, "no_objects", False):
            self.spawned_paths = []
            self.id_to_label = {}
            return self.spawned_paths, self.id_to_label

        try:
            ycb_dir = f"{self.isaac_nucleus_dir}/Props/YCB"
        except Exception:
            ycb_dir = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Props/YCB"

        scale_range = None
        if getattr(self.args, "scale_min", None) is not None and getattr(self.args, "scale_max", None) is not None:
            scale_range = (float(self.args.scale_min), float(self.args.scale_max))

        if self.loader is None:
            from environments.utils.object_loader import ObjectLoader, ObjectLoaderConfig, SpawnBounds
            from environments.utils.physix import object_loader_kwargs_from_physix

            spawn_mode = str(getattr(self.args, "spawn_mode", "usd"))
            box_size = float(getattr(self.args, "box_size", 0.05))
            phys_loader_kwargs = object_loader_kwargs_from_physix(self.phys)
            min_dist = float(getattr(self.args, "min_distance", 0.1))
            min_dist_xy_only = spawn_mode == "box"
            if min_dist_xy_only:
                min_dist = max(min_dist, 0.20)
            loader_cfg = ObjectLoaderConfig(
                dataset_dirs=[ycb_dir],
                bounds=SpawnBounds(min_xyz=tuple(self.args.spawn_min), max_xyz=tuple(self.args.spawn_max)),
                min_distance=float(min_dist),
                min_distance_xy_only=bool(min_dist_xy_only),
                uniform_scale_range=scale_range,
                spawn_mode=spawn_mode,
                box_size_min=(box_size, box_size, box_size),
                box_size_max=(box_size, box_size, box_size),
                box_color_palette=[rgb for (_name, rgb) in BOX_COLORS],
                box_color_names=[name for (name, _rgb) in BOX_COLORS],
                **phys_loader_kwargs,
            )
            self.loader = ObjectLoader(loader_cfg)

        paths = self.loader.spawn(parent_prim_path=self.parent_prim_path, num_objects=int(getattr(self.args, "num_objects", 4)))
        try:
            prim_to_label = self.loader.get_last_spawn_labels()
            lbl_map = {str(p).split("/")[-1]: str(lbl) for p, lbl in prim_to_label.items()}
        except Exception:
            lbl_map = {}

        try:
            if str(getattr(self.args, "spawn_mode", "usd")) == "box":
                for p in paths:
                    leaf = str(p).split("/")[-1]
                    human, _color, _idx = box_desc_from_prim(str(p))
                    lbl_map[leaf] = human
        except Exception:
            pass

        self.spawned_paths = [str(p) for p in paths]
        self.id_to_label = lbl_map
        return self.spawned_paths, self.id_to_label

    def prim_label(self, prim_path: str) -> str:
        try:
            leaf = str(prim_path).split("/")[-1]
            return str(self.id_to_label.get(leaf, "")) if self.id_to_label is not None else ""
        except Exception:
            return ""

    def select_episode_target_prim(
        self,
        ep_idx: int,
        *,
        object_z_by_leaf: dict[str, float] | None = None,
        prev_target_prim: str | None = None,
    ) -> str | None:
        if not self.spawned_paths:
            return None

        explicit = getattr(self.args, "target_prim", None)
        if explicit:
            return str(explicit)

        candidates_all = sorted([str(p) for p in self.spawned_paths])
        candidates = list(candidates_all)
        if object_z_by_leaf:
            try:
                table_z = float(getattr(self.phys, "snap_z_to", 0.82) if getattr(self.phys, "snap_z_to", None) is not None else 0.82)
                z_min_ok = table_z + 0.01
                filtered: list[str] = []
                for p in candidates_all:
                    leaf = str(p).split("/")[-1]
                    z = object_z_by_leaf.get(leaf, None)
                    if z is None or float(z) >= z_min_ok:
                        filtered.append(p)
                if filtered:
                    candidates = filtered if len(filtered) >= 2 else list(candidates_all)
            except Exception:
                candidates = list(candidates_all)

        idx = getattr(self.args, "target_index", None)
        if idx is not None:
            try:
                idx_i = int(idx)
                if len(candidates) == 0:
                    return None
                return candidates[idx_i % len(candidates)]
            except Exception:
                pass

        label = getattr(self.args, "target_label", None)
        if label:
            try:
                label_l = str(label).lower()
                for p in candidates:
                    if self.prim_label(p).lower() == label_l:
                        return p
                for p in candidates:
                    if label_l in self.prim_label(p).lower():
                        return p
            except Exception:
                pass

        sel = str(getattr(self.args, "target_selection", "first"))
        if sel == "random":
            try:
                import random

                return random.choice(candidates)
            except Exception:
                return candidates[0]
        if sel == "first":
            if len(candidates) == 0:
                return None
            if len(candidates) == 1:
                return candidates[0]
            try:
                if prev_target_prim and str(prev_target_prim) in candidates:
                    i = candidates.index(str(prev_target_prim))
                    return candidates[(i + 1) % len(candidates)]
            except Exception:
                pass
            return candidates[ep_idx % len(candidates)]
        return candidates[0] if candidates else None

    def make_language_command(self, *, ep_idx: int, target_prim: str) -> tuple[str, dict]:
        return make_language_command(
            ep_idx=ep_idx,
            target_prim=target_prim,
            id_to_label=self.id_to_label,
            spawn_mode=str(getattr(self.args, "spawn_mode", "usd")),
        )

    def table_z(self) -> float:
        try:
            t = getattr(self.default_scene, "table_translation", (0.0, 0.0, 0.8))
            return float(t[2])
        except Exception:
            return 0.8

    @staticmethod
    def target_z_from_tracker(tracker, target_prim: str) -> float | None:
        try:
            leaf = str(target_prim).split("/")[-1]
            for o in tracker.snapshot():
                if str(o.id) == leaf:
                    return float(o.pose.position_m[2])
        except Exception:
            return None
        return None

    @staticmethod
    def yaw_quat_wxyz(yaw_rad: float) -> tuple[float, float, float, float]:
        import math

        half = 0.5 * float(yaw_rad)
        return (math.cos(half), 0.0, 0.0, math.sin(half))

    def rerandomize_object_poses(
        self,
        paths: list[str],
        *,
        poses: Optional[list[tuple[tuple[float, float, float], float]]] = None,
    ) -> Optional[list[tuple[tuple[float, float, float], float]]]:
        """Teleport existing objects to poses (or new random poses) without delete/recreate."""
        if not paths:
            return None
        try:
            import importlib
            import math
            import random

            import torch
            from isaacsim.core.simulation_manager import SimulationManager
        except Exception:
            return None

        UsdPhysics = None
        omni_usd = None
        sim_utils = None
        RigidPrim = None
        try:
            UsdPhysics = importlib.import_module("pxr.UsdPhysics")
            omni_usd = importlib.import_module("omni.usd")
            sim_utils = importlib.import_module("isaaclab.sim")
            try:
                RigidPrim = importlib.import_module("isaacsim.core.prims").RigidPrim
            except Exception:
                RigidPrim = None
        except Exception:
            UsdPhysics = None
            omni_usd = None
            sim_utils = None
            RigidPrim = None

        bmin = tuple(float(v) for v in getattr(self.args, "spawn_min", (0.30, -0.20, 0.81)))
        bmax = tuple(float(v) for v in getattr(self.args, "spawn_max", (0.55, 0.20, 0.81)))
        min_dist = float(getattr(self.args, "min_distance", 0.10))
        table_z_guess = float(getattr(self.default_scene, "table_translation", (0.0, 0.0, 0.8))[2])
        z_min_safe = max(float(bmin[2]), float(table_z_guess) + 0.05)

        positions: list[tuple[float, float, float]] = []
        yaws: list[float] = []
        try:
            if poses is not None and len(poses) == len(paths):
                positions = [tuple(map(float, p)) for (p, _yaw) in poses]
                yaws = [float(_yaw) for (_p, _yaw) in poses]
        except Exception:
            positions = []
            yaws = []

        if len(positions) != len(paths) or len(yaws) != len(paths):
            positions = []
            tries = 0
            while len(positions) < len(paths) and tries < 2000:
                tries += 1
                x = random.uniform(bmin[0], bmax[0])
                y = random.uniform(bmin[1], bmax[1])
                z = random.uniform(z_min_safe, float(bmax[2]))
                try:
                    min_robot_dist = float(getattr(self.args, "spawn_min_robot_dist", 0.0))
                    if min_robot_dist > 1e-6 and math.hypot(float(x), float(y)) < min_robot_dist:
                        continue
                except Exception:
                    pass
                cand = (x, y, z)
                ok = True
                for p in positions:
                    dx = cand[0] - p[0]
                    dy = cand[1] - p[1]
                    if math.hypot(dx, dy) < min_dist:
                        ok = False
                        break
                if ok:
                    positions.append(cand)

            if len(positions) != len(paths):
                positions = [
                    (
                        random.uniform(bmin[0], bmax[0]),
                        random.uniform(bmin[1], bmax[1]),
                        random.uniform(z_min_safe, float(bmax[2])),
                    )
                    for _ in paths
                ]
            yaws = [random.uniform(-math.pi, math.pi) for _ in paths]

        sim_view = SimulationManager.get_physics_sim_view()
        origin0 = None
        try:
            origin0 = torch.tensor(self.scene_origins[0], device=self.sim.device).view(-1)
        except Exception:
            origin0 = None

        def _teleport_via_rigidprim(
            *,
            rb_prim_path: str,
            pos_xyz: tuple[float, float, float],
            quat_wxyz: tuple[float, float, float, float],
        ) -> bool:
            if RigidPrim is None:
                return False
            try:
                import numpy as np

                key = str(rb_prim_path)
                rp = self._respawn_rigidprims.get(key)
                if rp is None:
                    rp = RigidPrim(
                        prim_paths_expr=str(rb_prim_path),
                        name=f"respawn_{key.split('/')[-1]}",
                        reset_xform_properties=False,
                    )
                    self._respawn_rigidprims[key] = rp

                try:
                    if hasattr(rp, "initialize"):
                        rp.initialize()
                except Exception:
                    pass

                pos = np.array([[float(pos_xyz[0]), float(pos_xyz[1]), float(pos_xyz[2])]], dtype=np.float32)
                ori = np.array(
                    [[float(quat_wxyz[0]), float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3])]],
                    dtype=np.float32,
                )
                try:
                    rp.set_world_poses(positions=pos, orientations=ori)
                except TypeError:
                    rp.set_world_poses(pos, ori)

                try:
                    rp.set_velocities(np.zeros((1, 6), dtype=np.float32))
                except Exception:
                    try:
                        rp.set_linear_velocities(np.zeros((1, 3), dtype=np.float32))
                        rp.set_angular_velocities(np.zeros((1, 3), dtype=np.float32))
                    except Exception:
                        pass
                return True
            except Exception:
                return False

        for prim_path, pos, yaw in zip(paths, positions, yaws):
            try:
                rb_path = str(prim_path)
                try:
                    if UsdPhysics is not None and omni_usd is not None and sim_utils is not None:
                        stage = omni_usd.get_context().get_stage()
                        root = stage.GetPrimAtPath(str(prim_path))
                        if root.IsValid() and root.HasAPI(UsdPhysics.RigidBodyAPI):
                            rb_path = str(prim_path)
                        else:
                            get_all_matching_child_prims = getattr(sim_utils, "get_all_matching_child_prims", None)
                            if callable(get_all_matching_child_prims):
                                rigid_prims = get_all_matching_child_prims(
                                    str(prim_path),
                                    predicate=lambda p: p.HasAPI(UsdPhysics.RigidBodyAPI),
                                    traverse_instance_prims=True,
                                )
                                if rigid_prims:
                                    try:
                                        rb_path = rigid_prims[0].GetPath().pathString
                                    except Exception:
                                        rb_path = str(prim_path)
                except Exception:
                    rb_path = str(prim_path)

                try:
                    if str(getattr(self.args, "spawn_mode", "usd")) == "box":
                        qw, qx, qy, qz = self.yaw_quat_wxyz(yaw)
                        px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
                        if origin0 is not None and origin0.numel() >= 3:
                            px += float(origin0[0].item())
                            py += float(origin0[1].item())
                            pz += float(origin0[2].item())
                        if _teleport_via_rigidprim(
                            rb_prim_path=str(rb_path),
                            pos_xyz=(float(px), float(py), float(pz)),
                            quat_wxyz=(float(qw), float(qx), float(qy), float(qz)),
                        ):
                            continue
                except Exception:
                    pass

                rb_view = sim_view.create_rigid_body_view(str(rb_path))
                t0 = None
                try:
                    if hasattr(rb_view, "get_transforms"):
                        t0 = rb_view.get_transforms()
                except Exception:
                    t0 = None
                try:
                    if t0 is None or (hasattr(t0, "shape") and int(getattr(t0, "shape")[0]) == 0):
                        rb_view = sim_view.create_rigid_body_view(f"{str(prim_path)}/*")
                        if hasattr(rb_view, "get_transforms"):
                            t0 = rb_view.get_transforms()
                except Exception:
                    t0 = None
                try:
                    if t0 is not None and hasattr(t0, "shape") and int(getattr(t0, "shape")[0]) == 0:
                        continue
                except Exception:
                    pass

                qw, qx, qy, qz = self.yaw_quat_wxyz(yaw)
                px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
                if origin0 is not None and origin0.numel() >= 3:
                    px += float(origin0[0].item())
                    py += float(origin0[1].item())
                    pz += float(origin0[2].item())

                dev = self.sim.device
                n = 1
                try:
                    if t0 is not None and hasattr(t0, "device"):
                        dev = t0.device
                    if t0 is not None and hasattr(t0, "shape"):
                        n = max(1, int(getattr(t0, "shape")[0]))
                except Exception:
                    dev = self.sim.device
                    n = 1
                tf = torch.tensor([[px, py, pz, float(qx), float(qy), float(qz), float(qw)]], device=dev)
                if n != 1:
                    tf = tf.repeat(int(n), 1)
                if hasattr(rb_view, "set_transforms"):
                    rb_view.set_transforms(tf)
                if hasattr(rb_view, "set_linear_velocities"):
                    rb_view.set_linear_velocities(torch.zeros((int(n), 3), device=dev))
                if hasattr(rb_view, "set_angular_velocities"):
                    rb_view.set_angular_velocities(torch.zeros((int(n), 3), device=dev))
            except Exception:
                continue
        return list(zip(positions, yaws))

