"""vla_v2: Scripted pick-and-place data collection (target → transit → bin).

This profile is intentionally a sibling of ``vla_v1`` and shares its on-disk
session layout (per-tick PNGs, ``ticks.jsonl``, ``events.jsonl``,
``instruction.json``) so the same downstream tooling (``vla_lab.dataset``,
``vla_lab.train``) works without modification.

What's different from vla_v1:

* **Scene**
  - One *target* box spawned very close to the robot.
  - Several *obstacle* boxes scattered in the middle of the workspace.
  - Three fixed *bins* at the far end of the workspace.

* **Behavior** (purely scripted, no cuRobo / MotionGen):

  1. Open gripper.
  2. Move above the target box at a high transit Z, then descend to grasp.
  3. Close gripper.
  4. Lift straight up to the transit Z (clears the obstacle boxes).
  5. Translate over the clutter to the chosen bin's XY.
  6. Descend to the drop height, then release the gripper.
  7. Retreat upward.

* **Logging** is identical to vla_v1 in format. Every episode writes:

  - ``instruction.json`` with the natural-language command and metadata
    (target, bin index, bin color, etc.)
  - per-tick ``image_*.png`` (when ``--enable_cameras`` is set)
  - ``ticks.jsonl`` with state/action vectors
  - ``events.jsonl`` with action_start / action_end / drop_result events

This file is self-contained: it does not import cuRobo or any MotionGen
planner. The only "planner" is a hard-coded waypoint state machine.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from data_collection.envs.registry import get_envs
from data_collection.profiles.spec import ProfileSpec


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--env", type=str, default="reach_to_grasp_VLA", choices=sorted(get_envs().keys()))
    parser.add_argument("--logs-root", type=str, default="logs/data_collection")
    parser.add_argument("--log-rate-hz", type=int, default=5)
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--control", type=str, default="planner", choices=["keyboard", "idle", "planner"])
    parser.add_argument("--image-format", type=str, default="png", choices=["png", "jpg"])

    # Domain randomization (mirrors vla_v1; off by default)
    parser.add_argument("--domain-rand", action="store_true", help="Enable domain randomization (camera + lighting).")
    parser.add_argument("--domain-rand-seed", type=int, default=None)
    parser.add_argument("--domain-rand-camera-xy-m", type=float, default=0.02)
    parser.add_argument("--domain-rand-camera-z-m", type=float, default=0.10)
    parser.add_argument("--domain-rand-camera-yaw-deg", type=float, default=20.0)
    parser.add_argument("--domain-rand-camera-pitch-deg", type=float, default=0.0)
    parser.add_argument("--domain-rand-camera-roll-deg", type=float, default=0.0)
    parser.add_argument("--domain-rand-camera-fov-deg", type=float, default=5.0)
    parser.add_argument("--domain-rand-light-intensity-mult-min", type=float, default=0.5)
    parser.add_argument("--domain-rand-light-intensity-mult-max", type=float, default=1.5)
    parser.add_argument("--domain-rand-light-color-jitter", type=float, default=0.15)

    # Workspace safety bounds. Wider than vla_v1 because the bins live near x=0.7m.
    # The Z ceiling stays well below typical singularity altitudes for the Jaco2 Kinova.
    parser.add_argument("--workspace-min-z", type=float, default=0.0)
    parser.add_argument("--workspace-max-z", type=float, default=1.10)
    parser.add_argument("--workspace-min-x", type=float, default=0.10)
    parser.add_argument("--workspace-max-x", type=float, default=0.85)
    parser.add_argument("--workspace-min-y", type=float, default=-0.55)
    parser.add_argument("--workspace-max-y", type=float, default=0.55)

    # Box appearance / sizing
    parser.add_argument("--box-size", type=float, default=0.05, help="Side length for uniform boxes (m).")

    # Scene layout (vla_v2-specific)
    parser.add_argument("--num-obstacle-boxes", type=int, default=6, help="Boxes scattered in the middle of the workspace.")
    parser.add_argument(
        "--target-spawn-min",
        type=float,
        nargs=3,
        default=[0.20, -0.08, 0.83],
        help="Spawn AABB min for the close target box (relative to /World/Origin1).",
    )
    parser.add_argument(
        "--target-spawn-max",
        type=float,
        nargs=3,
        default=[0.28, 0.08, 0.83],
        help="Spawn AABB max for the close target box.",
    )
    parser.add_argument(
        "--obstacle-spawn-min",
        type=float,
        nargs=3,
        default=[0.34, -0.30, 0.83],
        help="Spawn AABB min for obstacle boxes (mid workspace).",
    )
    parser.add_argument(
        "--obstacle-spawn-max",
        type=float,
        nargs=3,
        default=[0.55, 0.30, 0.83],
        help="Spawn AABB max for obstacle boxes (mid workspace).",
    )
    parser.add_argument(
        "--obstacle-min-distance",
        type=float,
        default=0.12,
        help="Minimum tabletop spacing between obstacle boxes (m).",
    )

    # Bins
    parser.add_argument("--num-bins", type=int, default=3, help="Number of bins (fixed at 3 for vla_v2).")
    parser.add_argument(
        "--bin-center-x",
        type=float,
        default=0.70,
        help="Bin center X (m) in /World/Origin1 frame.",
    )
    parser.add_argument(
        "--bin-y-spacing",
        type=float,
        default=0.25,
        help="Distance between bin centers along Y (m).",
    )
    parser.add_argument(
        "--bin-footprint",
        type=float,
        nargs=2,
        default=[0.20, 0.20],
        help="Bin inner footprint (length, width) in m.",
    )
    parser.add_argument(
        "--bin-wall-height",
        type=float,
        default=0.06,
        help="Bin wall height (m).",
    )
    parser.add_argument(
        "--bin-wall-thickness",
        type=float,
        default=0.012,
        help="Bin wall thickness (m).",
    )
    parser.add_argument(
        "--bin-base-thickness",
        type=float,
        default=0.012,
        help="Bin base thickness (m).",
    )
    parser.add_argument(
        "--bin-selection",
        type=str,
        default="random",
        choices=["cycle", "random", "fixed"],
        help="How to choose the destination bin per episode (default: random).",
    )
    parser.add_argument(
        "--bin-index",
        type=int,
        default=None,
        help="Force a specific bin index (0..num_bins-1) when --bin-selection=fixed.",
    )

    # Pick / transit / place tuning
    parser.add_argument("--pregrasp-offset-m", type=float, default=0.08, help="Pregrasp offset above the target box top (m).")
    parser.add_argument("--grasp-depth-m", type=float, default=-0.04, help="Grasp depth relative to the box top (m).")
    parser.add_argument(
        "--approach-clearance-m",
        type=float,
        default=0.18,
        help="Initial pre-pick altitude above the table (m). EE flies to (target_x, target_y, table_z + this) before descending.",
    )
    parser.add_argument(
        "--transit-clearance-m",
        type=float,
        default=0.22,
        help="EE Z above table during the transit-over-clutter phase (m). Must clear all obstacle boxes.",
    )
    parser.add_argument(
        "--drop-clearance-m",
        type=float,
        default=0.20,
        help=(
            "EE Z above the bin floor when releasing (m). "
            "Should be at least ~bin-wall-height + ee-z-offset-m + 0.5 * box-size + a small margin "
            "so the held box clears the bin walls before being released."
        ),
    )
    parser.add_argument(
        "--ee-z-offset-m",
        type=float,
        default=0.08,
        help="Vertical offset added to box-top before computing EE waypoints. Compensates for EE link being above the fingertip TCP.",
    )
    parser.add_argument(
        "--home-pose-b",
        type=float,
        nargs=3,
        default=[0.30, 0.0, 1.00],
        help="Robot home EE position (base frame) used for retreat between episodes.",
    )

    # Episode control
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--max-steps-per-episode", type=int, default=10000)
    parser.add_argument("--render-rate-hz", type=float, default=60.0)
    parser.add_argument("--respawn-each-episode", action="store_true", default=True)
    parser.add_argument("--no-respawn-each-episode", action="store_true")
    parser.add_argument("--settle-steps", type=int, default=180)

    # Waypoint follower / gripper tuning
    parser.add_argument("--planner-speed-mps", type=float, default=0.30, help="EE linear speed during scripted execution (m/s).")
    parser.add_argument(
        "--planner-waypoint-max-seg-m",
        type=float,
        default=0.005,
        help="Max segment length after waypoint densification. Smaller = smoother + slower per step.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.012,
        help="Waypoint convergence tolerance (m). Larger values make the follower pop waypoints sooner = more fluid.",
    )
    parser.add_argument("--stabilize-steps", type=int, default=120)
    parser.add_argument("--gripper-open-steps", type=int, default=24)
    parser.add_argument("--gripper-close-steps", type=int, default=60)
    parser.add_argument("--hold-after-close-steps", type=int, default=20)
    parser.add_argument("--hold-after-release-steps", type=int, default=24)
    parser.add_argument(
        "--wp-max-steps-per-waypoint",
        type=int,
        default=480,
        help=(
            "Per-sub-waypoint watchdog. After N steps the follower drops the current waypoint "
            "even if it didn't fully converge. Smaller = more responsive, less hover-time."
        ),
    )
    parser.add_argument(
        "--phase-timeout-steps",
        type=int,
        default=3600,
        help="Hard cap (physics steps) per top-level motion phase before forcing advancement (~15s at 240Hz).",
    )
    parser.add_argument(
        "--progress-print-stride",
        type=int,
        default=240,
        help="Print a [VLA_V2][EP][PROGRESS] line every N physics steps during a phase. ~1s at 240Hz.",
    )

    # Bin success tuning
    parser.add_argument(
        "--drop-success-margin-m",
        type=float,
        default=0.04,
        help="Tolerance added to the bin footprint when checking drop success.",
    )

    # Override shared CLI defaults so --num-objects matches our intended scene
    # (1 close target + N obstacles).
    try:
        # By default we want 1 close target + 6 obstacles = 7 boxes.
        parser.set_defaults(num_objects=7)
        parser.set_defaults(spawn_min=[0.20, -0.30, 0.83])
        parser.set_defaults(spawn_max=[0.55, 0.30, 0.83])
        parser.set_defaults(min_distance=0.10)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    # sys.path / utils hygiene (same as vla_v1).
    from pathlib import Path as _Path

    ROOT = _Path(__file__).resolve().parents[2]
    root_str = str(ROOT)
    if root_str in sys.path:
        sys.path.remove(root_str)
    sys.path.insert(0, root_str)
    _env_mod = sys.modules.get("environments")
    if _env_mod is not None and not hasattr(_env_mod, "__path__"):
        del sys.modules["environments"]
    _utils_mod = sys.modules.get("utils")
    if _utils_mod is not None:
        _utils_file = str(getattr(_utils_mod, "__file__", "") or "")
        if _utils_file and root_str not in _utils_file:
            for _k in list(sys.modules.keys()):
                if _k == "utils" or _k.startswith("utils."):
                    del sys.modules[_k]

    from isaaclab.app import AppLauncher
    from data_collection.core.input_mux import CommandMuxInputProvider
    from data_collection.core.logger import SessionLogWriter, TickLoggingConfig
    from data_collection.core.objects import ObjectsTracker

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import torch  # noqa: E402

    try:
        import numpy as np  # noqa: E402
    except Exception as e:
        print(f"[VLA_V2] ERROR: numpy is required for image saving but could not be imported: {e}")
        return 2

    import carb  # noqa: E402

    carb_settings = carb.settings.get_settings()
    enable_cameras = bool(getattr(args, "enable_cameras", False))
    carb_settings.set_bool("/isaaclab/cameras_enabled", enable_cameras)
    print(f"[VLA_V2] enable_cameras={enable_cameras}")

    import importlib
    import math
    import random
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import Camera, CameraCfg

    from controllers import CartesianVelocityJogConfig, CartesianVelocityJogController
    from environments.utils.object_loader import ObjectLoader, ObjectLoaderConfig, SpawnBounds
    from environments.utils.physix import PhysicsConfig, apply_to_simulation_cfg, object_loader_kwargs_from_physix
    from controllers.input.waypoint_follower import WaypointFollowerInput
    from utilities import get_ee_pos_base_frame

    env_spec = get_envs()[str(getattr(args, "env", "reach_to_grasp_VLA"))]
    env_cfg_mod = importlib.import_module(f"{env_spec.module_base}.config")
    env_utils_mod = importlib.import_module(f"{env_spec.module_base}.utils")
    DEFAULT_SCENE = getattr(env_cfg_mod, "DEFAULT_SCENE")
    DEFAULT_CAMERA = getattr(env_cfg_mod, "DEFAULT_CAMERA", None)
    DEFAULT_TOP_DOWN_CAMERA = getattr(env_cfg_mod, "DEFAULT_TOP_DOWN_CAMERA", None)
    design_scene = getattr(env_utils_mod, "design_scene")
    create_topdown_camera = getattr(importlib.import_module("environments.utils.camera"), "create_topdown_camera")

    # Sim
    phys = PhysicsConfig(device=str(getattr(args, "device", "cuda:0")))
    sim_cfg = sim_utils.SimulationCfg(device=phys.device)
    apply_to_simulation_cfg(sim_cfg, phys)
    sim = sim_utils.SimulationContext(sim_cfg)
    if (not getattr(args, "headless", False)) and DEFAULT_CAMERA is not None:
        sim.set_camera_view(DEFAULT_CAMERA.eye, DEFAULT_CAMERA.target)

    # Scene + robot
    scene_entities, scene_origins = design_scene(DEFAULT_SCENE)
    robot = scene_entities["kinova_j2n6s300"]

    if enable_cameras and DEFAULT_TOP_DOWN_CAMERA is not None:
        create_topdown_camera(DEFAULT_TOP_DOWN_CAMERA)
        print(f"[VLA_V2] Top-down camera created at: {DEFAULT_TOP_DOWN_CAMERA.prim_path}")

    table_z = float(getattr(DEFAULT_SCENE, "table_translation", (0.0, 0.0, 0.8))[2])

    # -----------------------------------------------------------------------
    # Domain randomization (lighter, mirrors vla_v1)
    # -----------------------------------------------------------------------
    domain_rand_enabled = bool(getattr(args, "domain_rand", False))
    _dr_base_cam_pos = None
    _dr_base_cam_fov = None
    _dr_cam_prim_path = None
    try:
        if DEFAULT_TOP_DOWN_CAMERA is not None:
            _dr_base_cam_pos = tuple(float(v) for v in getattr(DEFAULT_TOP_DOWN_CAMERA, "position", (0.4, 0.0, 4.0)))
            _dr_base_cam_fov = float(getattr(DEFAULT_TOP_DOWN_CAMERA, "fov", 65.0))
            _dr_cam_prim_path = str(getattr(DEFAULT_TOP_DOWN_CAMERA, "prim_path", ""))
    except Exception:
        pass

    _dr_light_prim_path = "/World/Light"
    _dr_seed_base = (
        int(getattr(args, "domain_rand_seed"))
        if getattr(args, "domain_rand_seed", None) is not None
        else int(time.time() * 1000) & 0x7FFFFFFF
    )

    def _apply_domain_randomization(*, ep_idx: int, logger: Optional["SessionLogWriter"] = None) -> Optional[dict]:
        if not domain_rand_enabled:
            return None
        try:
            omni_usd = importlib.import_module("omni.usd")
            UsdGeom = importlib.import_module("pxr.UsdGeom")
            UsdLux = importlib.import_module("pxr.UsdLux")
            Gf = importlib.import_module("pxr.Gf")
            stage = omni_usd.get_context().get_stage()
        except Exception:
            return None

        seed = int((_dr_seed_base or 0) + int(ep_idx))
        rng = random.Random(seed)
        out: dict = {"enabled": True, "seed": int(seed)}

        try:
            prim = stage.GetPrimAtPath(str(_dr_light_prim_path))
            if prim.IsValid():
                dome = UsdLux.DomeLight(prim)
                base_int = float(dome.GetIntensityAttr().Get() or 2000.0)
                c = dome.GetColorAttr().Get()
                base_col = (float(c[0]), float(c[1]), float(c[2])) if c is not None else (0.75, 0.75, 0.75)
                mult = rng.uniform(
                    float(getattr(args, "domain_rand_light_intensity_mult_min", 0.5)),
                    float(getattr(args, "domain_rand_light_intensity_mult_max", 1.5)),
                )
                jitter = float(getattr(args, "domain_rand_light_color_jitter", 0.15))

                def _clamp01(x: float) -> float:
                    return max(0.0, min(1.0, x))

                color = (
                    _clamp01(base_col[0] + rng.uniform(-jitter, jitter)),
                    _clamp01(base_col[1] + rng.uniform(-jitter, jitter)),
                    _clamp01(base_col[2] + rng.uniform(-jitter, jitter)),
                )
                dome.GetIntensityAttr().Set(float(max(0.0, base_int * mult)))
                dome.GetColorAttr().Set(Gf.Vec3f(float(color[0]), float(color[1]), float(color[2])))
                out["light"] = {"intensity_mult": float(mult), "color_rgb": list(color)}
        except Exception:
            pass

        try:
            if enable_cameras and _dr_cam_prim_path and _dr_base_cam_pos is not None:
                cam_prim = stage.GetPrimAtPath(str(_dr_cam_prim_path))
                if cam_prim.IsValid():
                    xy = float(getattr(args, "domain_rand_camera_xy_m", 0.02))
                    z_j = float(getattr(args, "domain_rand_camera_z_m", 0.10))
                    x0, y0, z0 = _dr_base_cam_pos
                    pos = (x0 + rng.uniform(-xy, xy), y0 + rng.uniform(-xy, xy), max(0.5, z0 + rng.uniform(-z_j, z_j)))
                    yaw = rng.uniform(-float(getattr(args, "domain_rand_camera_yaw_deg", 20.0)),
                                      float(getattr(args, "domain_rand_camera_yaw_deg", 20.0)))
                    xform = UsdGeom.Xformable(cam_prim)
                    xform.ClearXformOpOrder()
                    translate_op = xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
                    rotate_op = xform.AddRotateXYZOp(UsdGeom.XformOp.PrecisionFloat)
                    translate_op.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
                    rotate_op.Set(Gf.Vec3f(0.0, 0.0, float(yaw)))
                    out["camera"] = {"pos_xyz": list(pos), "yaw_deg": float(yaw)}
        except Exception:
            pass

        try:
            if logger is not None:
                logger.log_event("domain_randomization", out)
        except Exception:
            pass
        return out

    # -----------------------------------------------------------------------
    # Box / bin spawning
    # -----------------------------------------------------------------------
    BOX_COLORS: List[Tuple[str, Tuple[float, float, float]]] = [
        ("red", (0.85, 0.20, 0.20)),
        ("blue", (0.20, 0.35, 0.90)),
        ("yellow", (0.95, 0.85, 0.20)),
        ("purple", (0.65, 0.25, 0.80)),
        ("orange", (0.95, 0.55, 0.15)),
        ("cyan", (0.15, 0.80, 0.85)),
        ("green", (0.15, 0.75, 0.30)),
    ]

    BIN_COLORS: List[Tuple[str, Tuple[float, float, float]]] = [
        ("red", (0.85, 0.18, 0.18)),
        ("green", (0.20, 0.75, 0.30)),
        ("blue", (0.20, 0.40, 0.90)),
    ]

    BOXES_PARENT = "/World/Origin1/Objects"
    BINS_PARENT = "/World/Origin1/Bins"

    def _yaw_quat_wxyz(yaw_rad: float) -> Tuple[float, float, float, float]:
        half = 0.5 * float(yaw_rad)
        return (math.cos(half), 0.0, 0.0, math.sin(half))

    def _spawn_boxes() -> Tuple[List[str], Dict[str, str], Dict[str, Optional[str]]]:
        """Spawn (1 target + N obstacles) uniform-colored cubes via ObjectLoader.

        Initial layout is unimportant; we explicitly re-randomize each episode.
        Returns (prim_paths, leaf_to_label, leaf_to_color_name).
        """
        n_obstacles = int(getattr(args, "num_obstacle_boxes", 6))
        n_objects = 1 + max(0, n_obstacles)
        box_size = float(getattr(args, "box_size", 0.05))
        phys_loader_kwargs = object_loader_kwargs_from_physix(phys)

        # NOTE: we pass tabletop bounds here only as a fallback; the per-episode
        # re-randomization fully controls placement.
        loader_cfg = ObjectLoaderConfig(
            dataset_dirs=[],
            bounds=SpawnBounds(min_xyz=tuple(args.spawn_min), max_xyz=tuple(args.spawn_max)),
            min_distance=float(getattr(args, "min_distance", 0.10)),
            min_distance_xy_only=True,
            spawn_mode="box",
            box_size_min=(box_size, box_size, box_size),
            box_size_max=(box_size, box_size, box_size),
            box_color_palette=[rgb for (_n, rgb) in BOX_COLORS],
            box_color_names=[name for (name, _rgb) in BOX_COLORS],
            **phys_loader_kwargs,
        )
        loader = ObjectLoader(loader_cfg)
        paths = loader.spawn(parent_prim_path="/World/Origin1", num_objects=int(n_objects))

        leaf_to_label: Dict[str, str] = {}
        leaf_to_color: Dict[str, Optional[str]] = {}
        for p in paths:
            leaf = str(p).split("/")[-1]
            try:
                idx = int(leaf.split("_")[-1])
            except Exception:
                idx = 1
            color_name, _rgb = BOX_COLORS[(idx - 1) % len(BOX_COLORS)]
            leaf_to_color[leaf] = color_name
            # idx=1 is the "close" target; everything else is an obstacle.
            if idx == 1:
                leaf_to_label[leaf] = f"target ({color_name} box)"
            else:
                leaf_to_label[leaf] = f"{color_name} box {idx}"
        return paths, leaf_to_label, leaf_to_color

    def _spawn_bins() -> List[Dict[str, object]]:
        """Spawn three open-top **static** bins along the far end of the workspace.

        Bins are spawned as plain cuboid geometry with collision but **no rigid
        body API** (``rigid_props=None``). This is the same pattern the codebase
        uses for collision proxies in ``environments/utils/object_loader.py``.

        Static-only spawning avoids two problems we saw earlier:

        1. Kinematic rigid bodies could be reset away from their authored pose
           on ``sim.reset()``, making the bins "appear briefly then disappear".
        2. Kinematic bodies briefly intersecting the table on the first physics
           tick can be ejected by the contact solver.

        Each bin = 1 base + 4 thin walls, all static.

        Returns a list of bin descriptors.
        """
        from isaaclab.sim.spawners.shapes.shapes_cfg import CuboidCfg
        from isaaclab.sim.spawners.materials.visual_materials_cfg import PreviewSurfaceCfg

        prim_utils = importlib.import_module("isaacsim.core.utils.prims")
        try:
            prim_utils.create_prim(BINS_PARENT, "Xform")
        except Exception:
            pass

        n_bins = int(getattr(args, "num_bins", 3))
        cx = float(getattr(args, "bin_center_x", 0.70))
        spacing = float(getattr(args, "bin_y_spacing", 0.25))
        inner_lx, inner_ly = (float(v) for v in getattr(args, "bin_footprint", [0.20, 0.20]))
        wall_h = float(getattr(args, "bin_wall_height", 0.06))
        wall_t = float(getattr(args, "bin_wall_thickness", 0.012))
        base_t = float(getattr(args, "bin_base_thickness", 0.012))

        # Add a small clearance so the bin's base bottom never coincides with the
        # table surface (avoids any first-tick contact resolution surprises).
        bin_floor_clearance = 0.005
        base_bottom_z = float(table_z) + bin_floor_clearance
        base_center_z = base_bottom_z + 0.5 * base_t
        bin_floor_z = base_bottom_z + base_t  # interior floor (top surface of base)
        wall_center_z = bin_floor_z + 0.5 * wall_h
        bin_top_z = bin_floor_z + wall_h

        # Center each bin along the y-axis evenly: y in {-spacing, 0, +spacing} for n_bins=3.
        y_centers = [(i - (n_bins - 1) / 2.0) * spacing for i in range(n_bins)]

        def _static_cuboid_cfg(rgb: Tuple[float, float, float], size: Tuple[float, float, float]) -> "CuboidCfg":
            return CuboidCfg(
                size=size,
                visual_material=PreviewSurfaceCfg(diffuse_color=rgb),
                rigid_props=None,  # static (no rigid body API)
                mass_props=None,
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                    contact_offset=phys.contact_offset,
                    rest_offset=phys.rest_offset,
                ),
            )

        out: List[Dict[str, object]] = []
        for i, y in enumerate(y_centers, start=0):
            color_name, rgb = BIN_COLORS[i % len(BIN_COLORS)]
            bin_root = f"{BINS_PARENT}/Bin_{i+1:02d}"
            try:
                prim_utils.create_prim(bin_root, "Xform")
            except Exception:
                pass

            paths_for_bin: List[str] = []

            base_cfg = _static_cuboid_cfg(rgb, (inner_lx + 2 * wall_t, inner_ly + 2 * wall_t, base_t))
            base_path = f"{bin_root}/base"
            base_cfg.func(base_path, base_cfg, translation=(cx, y, base_center_z), orientation=(1.0, 0.0, 0.0, 0.0))
            paths_for_bin.append(base_path)

            walls = [
                ("wall_px", (wall_t, inner_ly + 2 * wall_t, wall_h), (cx + 0.5 * inner_lx + 0.5 * wall_t, y, wall_center_z)),
                ("wall_nx", (wall_t, inner_ly + 2 * wall_t, wall_h), (cx - 0.5 * inner_lx - 0.5 * wall_t, y, wall_center_z)),
                ("wall_py", (inner_lx, wall_t, wall_h), (cx, y + 0.5 * inner_ly + 0.5 * wall_t, wall_center_z)),
                ("wall_ny", (inner_lx, wall_t, wall_h), (cx, y - 0.5 * inner_ly - 0.5 * wall_t, wall_center_z)),
            ]
            for wname, wsize, wpos in walls:
                wcfg = _static_cuboid_cfg(rgb, wsize)
                wpath = f"{bin_root}/{wname}"
                wcfg.func(wpath, wcfg, translation=tuple(wpos), orientation=(1.0, 0.0, 0.0, 0.0))
                paths_for_bin.append(wpath)

            out.append(
                {
                    "idx": int(i),
                    "name": f"bin_{i+1}",
                    "color_name": color_name,
                    "color_rgb": rgb,
                    "center_xy": (float(cx), float(y)),
                    "top_z": float(bin_top_z),
                    "bin_floor_z": float(bin_floor_z),
                    "inner_xy": (float(inner_lx), float(inner_ly)),
                    "prim_paths": paths_for_bin,
                }
            )
        return out

    spawned_paths, id_to_label, id_to_color = _spawn_boxes()
    bins_meta = _spawn_bins()
    print(f"[VLA_V2] Spawned {len(spawned_paths)} boxes and {len(bins_meta)} bins.")
    for _b in bins_meta:
        cx_, cy_ = _b["center_xy"]  # type: ignore[index]
        print(
            f"[VLA_V2]   bin {int(_b['idx']) + 1} ({_b['color_name']}): "
            f"center_xy=({float(cx_):.3f}, {float(cy_):.3f}) "
            f"floor_z={float(_b['bin_floor_z']):.3f} "
            f"top_z={float(_b['top_z']):.3f}"
        )

    target_leaf = "Obj_01"
    target_prim = f"{BOXES_PARENT}/{target_leaf}"
    target_color_name = id_to_color.get(target_leaf, None)

    # -----------------------------------------------------------------------
    # Camera sensor
    # -----------------------------------------------------------------------
    camera_sensor = None
    if enable_cameras and DEFAULT_TOP_DOWN_CAMERA is not None:
        try:
            camera_cfg = CameraCfg(
                prim_path=DEFAULT_TOP_DOWN_CAMERA.prim_path,
                offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
                spawn=None,
                data_types=["rgb"],
                width=DEFAULT_TOP_DOWN_CAMERA.resolution[0],
                height=DEFAULT_TOP_DOWN_CAMERA.resolution[1],
            )
            camera_sensor = Camera(cfg=camera_cfg)
            print(f"[VLA_V2] Camera sensor created: {camera_cfg.width}x{camera_cfg.height}")
        except Exception as create_err:
            print(f"[VLA_V2] ERROR: Failed to create camera: {create_err}")
            camera_sensor = None

    # Reset sim/robot
    def _reset_sim_and_robot() -> None:
        sim.reset()
        origin0 = torch.tensor(scene_origins[0], device=sim.device)
        root_state = robot.data.default_root_state.clone()
        root_state[:, :3] += origin0
        robot.write_root_pose_to_sim(root_state[:, :7])
        robot.write_root_velocity_to_sim(root_state[:, 7:])
        robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
        robot.reset()

    _reset_sim_and_robot()
    if camera_sensor is not None:
        try:
            camera_sensor.reset()
        except Exception:
            camera_sensor = None

    # -----------------------------------------------------------------------
    # Box re-randomizer (vla_v2 specific layout)
    # -----------------------------------------------------------------------
    _respawn_rigidprims: Dict[str, object] = {}

    def _sample_episode_box_poses(
        rng: random.Random,
    ) -> Dict[str, Tuple[Tuple[float, float, float], float]]:
        """Sample a position+yaw for every box.

        - The target box (leaf=Obj_01) is sampled inside the *target* AABB
          (close to the robot).
        - All remaining boxes are sampled inside the *obstacle* AABB
          (mid workspace), with a minimum spacing constraint.

        Z values come straight from the AABB; we don't snap to table_z because
        we want the boxes to drop a couple cm and settle naturally.
        """
        out: Dict[str, Tuple[Tuple[float, float, float], float]] = {}

        tmin = tuple(float(v) for v in getattr(args, "target_spawn_min", [0.20, -0.08, 0.83]))
        tmax = tuple(float(v) for v in getattr(args, "target_spawn_max", [0.28, 0.08, 0.83]))
        tx = rng.uniform(tmin[0], tmax[0])
        ty = rng.uniform(tmin[1], tmax[1])
        tz = rng.uniform(tmin[2], tmax[2])
        out[target_leaf] = ((tx, ty, tz), rng.uniform(-math.pi, math.pi))

        omin = tuple(float(v) for v in getattr(args, "obstacle_spawn_min", [0.34, -0.30, 0.83]))
        omax = tuple(float(v) for v in getattr(args, "obstacle_spawn_max", [0.55, 0.30, 0.83]))
        min_dist = float(getattr(args, "obstacle_min_distance", 0.12))

        existing_xy: List[Tuple[float, float]] = [(tx, ty)]
        for p in spawned_paths:
            leaf = str(p).split("/")[-1]
            if leaf == target_leaf:
                continue
            placed = False
            for _ in range(500):
                ox = rng.uniform(omin[0], omax[0])
                oy = rng.uniform(omin[1], omax[1])
                oz = rng.uniform(omin[2], omax[2])
                if all(math.hypot(ox - x, oy - y) >= min_dist for (x, y) in existing_xy):
                    existing_xy.append((ox, oy))
                    out[leaf] = ((ox, oy, oz), rng.uniform(-math.pi, math.pi))
                    placed = True
                    break
            if not placed:
                # Fallback: place anywhere inside the obstacle AABB.
                ox = rng.uniform(omin[0], omax[0])
                oy = rng.uniform(omin[1], omax[1])
                oz = rng.uniform(omin[2], omax[2])
                out[leaf] = ((ox, oy, oz), rng.uniform(-math.pi, math.pi))
        return out

    def _teleport_box(rb_prim_path: str, pos_xyz: Tuple[float, float, float], yaw_rad: float) -> bool:
        """Teleport a kinematic-or-dynamic rigid prim using isaacsim.core.prims.RigidPrim."""
        try:
            import numpy as np

            RigidPrim = importlib.import_module("isaacsim.core.prims").RigidPrim
            key = str(rb_prim_path)
            rp = _respawn_rigidprims.get(key)
            if rp is None:
                rp = RigidPrim(prim_paths_expr=str(rb_prim_path), name=f"respawn_{key.split('/')[-1]}", reset_xform_properties=False)
                _respawn_rigidprims[key] = rp
            try:
                if hasattr(rp, "initialize"):
                    rp.initialize()
            except Exception:
                pass
            qw, qx, qy, qz = _yaw_quat_wxyz(yaw_rad)
            pos = np.array([[pos_xyz[0], pos_xyz[1], pos_xyz[2]]], dtype=np.float32)
            ori = np.array([[qw, qx, qy, qz]], dtype=np.float32)
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

    def _rerandomize_boxes(seed: int) -> Dict[str, Tuple[Tuple[float, float, float], float]]:
        rng = random.Random(int(seed))
        poses = _sample_episode_box_poses(rng)
        for p in spawned_paths:
            leaf = str(p).split("/")[-1]
            if leaf not in poses:
                continue
            (px, py, pz), yaw = poses[leaf]
            try:
                origin0 = torch.tensor(scene_origins[0], device=sim.device).view(-1)
                px += float(origin0[0].item())
                py += float(origin0[1].item())
                pz += float(origin0[2].item())
            except Exception:
                pass
            _teleport_box(str(p), (px, py, pz), yaw)
        return poses

    # -----------------------------------------------------------------------
    # Controller + waypoint follower
    # -----------------------------------------------------------------------
    linear_speed_mps = float(getattr(args, "planner_speed_mps", 0.4))
    ctrl_cfg = CartesianVelocityJogConfig(
        ee_link_name=str(getattr(args, "ee_link", "j2n6s300_end_effector")),
        device=str(sim.device),
        use_relative_mode=True,
        linear_speed_mps=float(linear_speed_mps),
        workspace_min=(
            float(getattr(args, "workspace_min_x", 0.10)),
            float(getattr(args, "workspace_min_y", -0.55)),
            float(getattr(args, "workspace_min_z", 0.0)),
        ),
        workspace_max=(
            float(getattr(args, "workspace_max_x", 0.90)),
            float(getattr(args, "workspace_max_y", 0.55)),
            float(getattr(args, "workspace_max_z", 1.30)),
        ),
        log_ee_pos=bool(getattr(args, "print_ee", False)),
        log_ee_frame=str(getattr(args, "ee_frame", "world")),
        log_every_n_steps=int(getattr(args, "print_interval", 1)),
    )
    controller = CartesianVelocityJogController(ctrl_cfg, num_envs=1, device=str(sim.device))
    controller.set_mode("translate")
    controller.reset(robot)

    mux_input = CommandMuxInputProvider()
    dt = float(sim.get_physics_dt())
    wp = WaypointFollowerInput(
        step_pos_m=float(ctrl_cfg.linear_speed_mps) * dt,
        tol_m=float(getattr(args, "tolerance", 0.012)),
        max_steps_per_waypoint=int(getattr(args, "wp_max_steps_per_waypoint", 480)),
        stagnation_steps=int(10**9),
        device=str(sim.device),
    )
    mux_input.set_base(wp)
    controller.set_input_provider(mux_input)

    def _densify(pts: List[Tuple[float, float, float]], *, max_seg_m: float) -> List[Tuple[float, float, float]]:
        max_seg_m = float(max(1e-6, max_seg_m))
        if not pts or len(pts) < 2:
            return [tuple(map(float, p)) for p in pts]
        out: List[Tuple[float, float, float]] = [tuple(map(float, pts[0]))]
        for (x0, y0, z0), (x1, y1, z1) in zip(pts, pts[1:]):
            dx, dy, dz = float(x1 - x0), float(y1 - y0), float(z1 - z0)
            d = math.sqrt(dx * dx + dy * dy + dz * dz)
            if d <= max_seg_m:
                out.append((float(x1), float(y1), float(z1)))
                continue
            n = int(math.ceil(d / max_seg_m))
            for i in range(1, n + 1):
                t = float(i) / float(n)
                out.append((float(x0 + t * dx), float(y0 + t * dy), float(z0 + t * dz)))
        return out

    # -----------------------------------------------------------------------
    # Tracker + per-session logger
    # -----------------------------------------------------------------------
    tracker = ObjectsTracker(prim_paths=spawned_paths)
    tick_cfg = TickLoggingConfig(
        log_rate_hz=int(getattr(args, "log_rate_hz", 5)),
        workspace_min=getattr(controller.config.safety_cfg, "workspace_min", None),
        workspace_max=getattr(controller.config.safety_cfg, "workspace_max", None),
        ee_link_name=str(getattr(args, "ee_link", "j2n6s300_end_effector")),
        arm_joint_regex=controller.config.arm_joint_regex,
        log_joint_data=True,
    )

    # -----------------------------------------------------------------------
    # Language commands
    # -----------------------------------------------------------------------
    def _make_language_command(*, ep_idx: int, bin_meta: Dict[str, object]) -> Tuple[str, dict]:
        bin_color = str(bin_meta.get("color_name", "?"))
        bin_idx = int(bin_meta.get("idx", 0)) + 1  # 1-based for humans
        target_color = str(target_color_name or "")
        target_phrase = f"the {target_color} box" if target_color else "the box"
        bin_phrase_color = f"the {bin_color} bin"
        bin_phrase_idx = f"bin {bin_idx}"

        templates = [
            "Pick up {target} and place it in {bin_color}.",
            "Grab {target} and drop it into {bin_color}.",
            "Move {target} into {bin_color}.",
            "Pick up {target} and put it in {bin_idx}.",
            "Take {target} and place it in {bin_idx} ({bin_color}).",
            "Pick up the box closest to the robot and put it in {bin_color}.",
        ]

        rng = random.Random(int(ep_idx) + 4242)
        tmpl = rng.choice(templates)
        cmd = tmpl.format(target=target_phrase, bin_color=bin_phrase_color, bin_idx=bin_phrase_idx)
        meta = {
            "target_leaf": target_leaf,
            "target_color": target_color,
            "bin_idx": int(bin_idx),  # 1-based
            "bin_idx0": int(bin_meta.get("idx", 0)),  # 0-based
            "bin_color": bin_color,
            "bin_center_xy": list(bin_meta.get("center_xy", (0.0, 0.0))),
        }
        return cmd, meta

    # -----------------------------------------------------------------------
    # Helpers shared by the state machine
    # -----------------------------------------------------------------------
    def _read_target_state() -> Optional[Dict[str, object]]:
        try:
            for o in tracker.snapshot():
                if str(o.id) == target_leaf:
                    return {
                        "pos": tuple(float(v) for v in o.pose.position_m),
                        "ori_wxyz": tuple(float(v) for v in o.pose.orientation_wxyz),
                    }
        except Exception:
            return None
        return None

    def _ee_pos_b() -> Optional[Tuple[float, float, float]]:
        try:
            v = get_ee_pos_base_frame(robot, str(getattr(args, "ee_link", "j2n6s300_end_effector")))
            if v is None:
                return None
            return (float(v[0]), float(v[1]), float(v[2]))
        except Exception:
            return None

    def _world_to_base_xy(pos_w: Tuple[float, float, float]) -> Tuple[float, float, float]:
        # Origin1 is the robot's parent; we treat parent-relative coords as base coords.
        try:
            origin0 = scene_origins[0]
            return (float(pos_w[0] - origin0[0]), float(pos_w[1] - origin0[1]), float(pos_w[2] - origin0[2]))
        except Exception:
            return (float(pos_w[0]), float(pos_w[1]), float(pos_w[2]))

    def _select_bin(ep_idx: int, n_bins: int) -> Dict[str, object]:
        sel = str(getattr(args, "bin_selection", "cycle"))
        if sel == "fixed":
            idx = int(getattr(args, "bin_index", 0) or 0)
            return bins_meta[max(0, min(n_bins - 1, idx))]
        if sel == "random":
            rng = random.Random(int(ep_idx) * 9173 + 1)
            return rng.choice(bins_meta)
        # cycle
        return bins_meta[int(ep_idx) % n_bins]

    def _drop_success(target_state: Optional[Dict[str, object]], bin_meta: Dict[str, object]) -> bool:
        if target_state is None:
            return False
        pos = target_state.get("pos", None)
        if pos is None:
            return False
        cx, cy = bin_meta["center_xy"]  # type: ignore[index]
        lx, ly = bin_meta["inner_xy"]  # type: ignore[index]
        margin = float(getattr(args, "drop_success_margin_m", 0.04))
        half_lx = 0.5 * float(lx) + float(margin)
        half_ly = 0.5 * float(ly) + float(margin)
        # XY containment
        if abs(float(pos[0]) - float(cx)) > half_lx:
            return False
        if abs(float(pos[1]) - float(cy)) > half_ly:
            return False
        # Z must be near the bin floor (i.e., the box has actually dropped in).
        bin_floor_z = float(bin_meta.get("bin_floor_z", float(table_z) + 0.02))
        max_floor_dz = float(getattr(args, "bin_wall_height", 0.06)) + 0.05
        return abs(float(pos[2]) - bin_floor_z) <= max_floor_dz

    # -----------------------------------------------------------------------
    # Episode loop
    # -----------------------------------------------------------------------
    logs_root = Path(str(getattr(args, "logs_root", "logs/data_collection")))
    from datetime import datetime as _dt

    session_timestamp = _dt.now().strftime("%Y%m%d_%H%M%S")
    session_folder = logs_root / f"session_{session_timestamp}"
    session_folder.mkdir(parents=True, exist_ok=True)

    num_episodes = int(getattr(args, "num_episodes", 10))
    max_steps_ep = int(getattr(args, "max_steps_per_episode", 10000))
    render_rate_hz = float(getattr(args, "render_rate_hz", 60.0))
    render_stride = max(1, int(round((1.0 / max(1e-9, render_rate_hz)) / max(1e-9, dt))))
    image_format = getattr(args, "image_format", "png")
    period = 1.0 / float(tick_cfg.log_rate_hz)

    # Domain randomization once for non-episode (idle/keyboard) modes is N/A; vla_v2
    # is always planner mode.
    print(f"[VLA_V2] Session directory: {session_folder}")
    print(f"[VLA_V2] Logging rate: {tick_cfg.log_rate_hz} Hz; episodes={num_episodes}; bins={len(bins_meta)}")

    total_ticks = 0
    total_images = 0

    # Phase parameters
    pregrasp_offset = float(getattr(args, "pregrasp_offset_m", 0.10))
    grasp_depth = float(getattr(args, "grasp_depth_m", -0.04))
    transit_clearance = float(getattr(args, "transit_clearance_m", 0.30))
    drop_clearance = float(getattr(args, "drop_clearance_m", 0.18))
    ee_z_offset = float(getattr(args, "ee_z_offset_m", 0.08))
    home_b = tuple(float(v) for v in getattr(args, "home_pose_b", [0.30, 0.0, 1.10]))
    box_size = float(getattr(args, "box_size", 0.05))

    transit_z = float(table_z) + transit_clearance

    for ep in range(num_episodes):
        if not simulation_app.is_running():
            print(f"[VLA_V2][EP] Simulation stopped, ending at episode {ep}/{num_episodes}")
            break

        print(f"[VLA_V2][EP] Starting episode {ep+1}/{num_episodes}")

        try:
            session_logger = SessionLogWriter(root=session_folder, session_name=f"episode_{ep:04d}")
            images_dir = session_logger.root / "images"
            if camera_sensor is not None:
                images_dir.mkdir(exist_ok=True)
            images_captured_episode = 0

            session_logger.write_metadata(
                sim_dt=sim.get_physics_dt(),
                physics_substeps=int(getattr(sim.cfg, "sub_steps", 4)),
                seed=0,
                robot_name="kinova_j2n6s300",
                ee_link=str(getattr(args, "ee_link", "j2n6s300_end_effector")),
                arm_joint_regex=controller.config.arm_joint_regex,
                log_rate_hz=tick_cfg.log_rate_hz,
                window_len_s=2.0,
            )

            _reset_sim_and_robot()
            if camera_sensor is not None:
                try:
                    camera_sensor.reset()
                except Exception:
                    pass

            # Step once so PhysX views populate before respawn.
            try:
                sim.step(render=False)
                robot.update(dt)
            except Exception:
                pass

            do_respawn = bool(getattr(args, "respawn_each_episode", True)) and (
                not bool(getattr(args, "no_respawn_each_episode", False))
            )
            intended_xy = {}
            if do_respawn:
                seed = (int(getattr(args, "domain_rand_seed", 0) or 0) + int(ep)) & 0x7FFFFFFF
                poses = _rerandomize_boxes(seed=seed)
                intended_xy = {leaf: [round(p[0][0], 4), round(p[0][1], 4)] for leaf, p in poses.items()}
                session_logger.log_event(
                    "object_respawn",
                    {"episode_idx": int(ep), "seed": int(seed), "intended_xy": intended_xy},
                )

            # Reset controller + waypoint follower
            try:
                controller.reset(robot)
                controller.set_mode("translate")
            except Exception:
                pass
            try:
                wp.reset()
            except Exception:
                pass
            try:
                ee0 = _ee_pos_b()
                if ee0 is not None:
                    wp.set_current_pose_b(torch.tensor(ee0, dtype=torch.float32, device=sim.device))
            except Exception:
                pass

            # Settle physics
            settle_steps = int(getattr(args, "settle_steps", 180))
            for _ in range(settle_steps):
                if not simulation_app.is_running():
                    break
                try:
                    controller.step(robot, dt)
                except Exception:
                    pass
                sim.step(render=False)
                robot.update(dt)

            try:
                _apply_domain_randomization(ep_idx=int(ep), logger=session_logger)
                if camera_sensor is not None:
                    try:
                        camera_sensor.reset()
                    except Exception:
                        pass
            except Exception:
                pass

            # Recreate tracker per episode (PhysX view safety, mirrors vla_v1).
            try:
                tracker = ObjectsTracker(prim_paths=spawned_paths)
            except Exception:
                pass

            # Pick a destination bin + language command
            bin_meta = _select_bin(int(ep), len(bins_meta))
            lang_cmd, lang_meta = _make_language_command(ep_idx=int(ep), bin_meta=bin_meta)
            try:
                import json as _json

                (session_logger.root / "instruction.json").write_text(
                    _json.dumps(
                        {
                            "episode_idx": int(ep),
                            "target_prim": str(target_prim),
                            "target_leaf": target_leaf,
                            "target_color": target_color_name,
                            "bin_idx": int(lang_meta.get("bin_idx", 0)),  # 1-based
                            "bin_idx0": int(lang_meta.get("bin_idx0", 0)),
                            "bin_color": lang_meta.get("bin_color"),
                            "bin_center_xy": lang_meta.get("bin_center_xy"),
                            "language_command": str(lang_cmd),
                            "created_at": _dt.now().isoformat(timespec="seconds"),
                        },
                        indent=2,
                    )
                )
            except Exception:
                pass

            # Read the target object's current world XY (after settle) so the
            # gripper can lock onto the actual settled position, not the
            # intended pre-settle one.
            target_state = _read_target_state()
            if target_state is None:
                session_logger.log_event(
                    "episode_skipped", {"episode_idx": int(ep), "reason": "no_target_pose"}
                )
                try:
                    session_logger.close()
                except Exception:
                    pass
                continue
            tx_w, ty_w, tz_w = target_state["pos"]  # type: ignore[index]
            tx_b, ty_b, tz_b = _world_to_base_xy((tx_w, ty_w, tz_w))

            # Bin XY in base frame (origin1 == base for our setup).
            bcx, bcy = bin_meta["center_xy"]  # type: ignore[index]
            bin_drop_x_b, bin_drop_y_b, _ = _world_to_base_xy((float(bcx), float(bcy), 0.0))

            box_top_z_b = float(tz_b) + 0.5 * box_size
            grasp_z_b = float(box_top_z_b) + ee_z_offset + grasp_depth
            pregrasp_z_b = float(box_top_z_b) + ee_z_offset + pregrasp_offset
            drop_z_b = float(bin_meta["bin_floor_z"]) - float(scene_origins[0][2]) + drop_clearance  # base z

            # Episode start event
            session_logger.log_event(
                "episode_start",
                {
                    "episode_idx": int(ep),
                    "target_prim": str(target_prim),
                    "target_leaf": target_leaf,
                    "target_color": target_color_name,
                    "target_pos_b": [float(tx_b), float(ty_b), float(tz_b)],
                    "bin_idx": int(lang_meta.get("bin_idx", 0)),
                    "bin_color": lang_meta.get("bin_color"),
                    "bin_center_xy_b": [float(bin_drop_x_b), float(bin_drop_y_b)],
                    "language_command": str(lang_cmd),
                    "language_command_meta": lang_meta,
                    "transit_z_b": float(transit_z),
                    "drop_z_b": float(drop_z_b),
                },
            )

            # ----------------------------------------------------------------
            # Scripted state machine (fluid, grouped waypoints)
            # ----------------------------------------------------------------
            #
            # Top-level phases. Motion phases queue *multiple* sub-waypoints in
            # a single ``wp.set_waypoints_b(...)`` call, which makes the
            # controller flow continuously through them with no pauses
            # in between (the only stops are at gripper / hold phases).
            #
            #   1. OPEN_GRIPPER       — open fingers, no motion.
            #   2. PICK               — fly above target → pregrasp → grasp Z.
            #   3. CLOSE_GRIPPER      — close fingers around the box.
            #   4. POST_CLOSE_HOLD    — short hold so contacts settle.
            #   5. TRANSIT_AND_DROP   — lift → transit XY over clutter → descend
            #                           into bin (one continuous trajectory).
            #   6. RELEASE_GRIPPER    — open fingers (drop).
            #   7. POST_RELEASE_HOLD  — short hold.
            #   8. RETREAT            — retreat up + back to home XY (continuous).
            #   9. DONE.
            #
            # PICK: always go to a moderate "approach" altitude above the target
            # first, then descend through pregrasp to grasp.
            #
            # We deliberately do NOT command the high transit altitude before
            # grasping — that was the original "lift the arm to its ceiling and
            # freeze" failure mode. ``approach_z_b`` is the table top plus
            # ``--approach-clearance-m`` (default 0.18 m → 0.98 m world Z), well
            # above any obstacle box top (~0.85 m world Z) but well below the
            # workspace ceiling so the IK is always solvable.
            approach_z_b = float(table_z) + float(getattr(args, "approach_clearance_m", 0.18))

            pick_points = [
                (float(tx_b), float(ty_b), float(approach_z_b)),
                (float(tx_b), float(ty_b), float(max(pregrasp_z_b, grasp_z_b + 0.015))),
                (float(tx_b), float(ty_b), float(grasp_z_b)),
            ]
            transit_points = [
                (float(tx_b), float(ty_b), float(transit_z)),
                (float(bin_drop_x_b), float(bin_drop_y_b), float(transit_z)),
                (float(bin_drop_x_b), float(bin_drop_y_b), float(drop_z_b)),
            ]
            retreat_points = [
                (float(bin_drop_x_b), float(bin_drop_y_b), float(transit_z)),
                (float(home_b[0]), float(home_b[1]), float(home_b[2])),
            ]

            phases: List[Tuple[str, Dict[str, object]]] = [
                ("open_gripper", {"steps": int(getattr(args, "gripper_open_steps", 24)), "value": +1.0, "name": "OPEN_GRIPPER"}),
                ("waypoint", {"name": "PICK", "points": pick_points,
                              "log": {"target_xy_b": [float(tx_b), float(ty_b)],
                                      "grasp_z_b": float(grasp_z_b)}}),
                ("close_gripper", {"steps": int(getattr(args, "gripper_close_steps", 60)), "value": -1.0, "name": "CLOSE_GRIPPER"}),
                ("hold", {"name": "POST_CLOSE_HOLD", "steps": int(getattr(args, "hold_after_close_steps", 20))}),
                ("waypoint", {"name": "TRANSIT_AND_DROP", "points": transit_points,
                              "log": {"transit_z_b": float(transit_z),
                                      "bin_xy_b": [float(bin_drop_x_b), float(bin_drop_y_b)],
                                      "drop_z_b": float(drop_z_b),
                                      "bin_idx": int(lang_meta.get("bin_idx", 0))}}),
                ("open_gripper", {"steps": int(getattr(args, "gripper_open_steps", 24)), "value": +1.0, "name": "RELEASE_GRIPPER"}),
                ("hold", {"name": "POST_RELEASE_HOLD", "steps": int(getattr(args, "hold_after_release_steps", 24))}),
                ("waypoint", {"name": "RETREAT", "points": retreat_points,
                              "log": {"home_xy_b": [float(home_b[0]), float(home_b[1])],
                                      "home_z_b": float(home_b[2])}}),
                ("done", {}),
            ]

            print(
                f"[VLA_V2][EP {int(ep)+1}/{num_episodes}] target_pos_b=({float(tx_b):.3f}, {float(ty_b):.3f}, {float(tz_b):.3f}) "
                f"-> bin {int(lang_meta.get('bin_idx', 0))} ({lang_meta.get('bin_color')}) @ "
                f"({float(bin_drop_x_b):.3f}, {float(bin_drop_y_b):.3f}) "
                f"| grasp_z={float(grasp_z_b):.3f}m transit_z={float(transit_z):.3f}m drop_z={float(drop_z_b):.3f}m"
            )
            print(f"[VLA_V2][EP {int(ep)+1}/{num_episodes}] instruction: \"{lang_cmd}\"")

            phase_idx = 0
            phase_step_count = 0
            phase_state: Dict[str, object] = {}
            phase_timeout = int(getattr(args, "phase_timeout_steps", 4800))
            progress_stride = max(1, int(getattr(args, "progress_print_stride", 240)))
            stabilize_left = int(getattr(args, "stabilize_steps", 120))

            def _phase_name(idx: int) -> str:
                if idx >= len(phases):
                    return "DONE"
                _kind, _params = phases[idx]
                return str(_params.get("name", _kind)).upper()

            def _print_phase_start(name: str) -> None:
                ee = _ee_pos_b()
                if ee is None:
                    print(f"[VLA_V2][EP {int(ep)+1}/{num_episodes}] PHASE {name} start (EE pose unavailable)")
                else:
                    print(
                        f"[VLA_V2][EP {int(ep)+1}/{num_episodes}] PHASE {name} start "
                        f"EE_b=({ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f})"
                    )

            def _print_progress(name: str, *, goal: Optional[Tuple[float, float, float]] = None,
                                wp_left: Optional[int] = None) -> None:
                ee = _ee_pos_b()
                if ee is None:
                    return
                if goal is not None:
                    dx = ee[0] - goal[0]
                    dy = ee[1] - goal[1]
                    dz = ee[2] - goal[2]
                    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                    extra = f" goal=({goal[0]:.3f},{goal[1]:.3f},{goal[2]:.3f}) dist={dist:.3f}m"
                else:
                    extra = ""
                wp_str = f" wp_left={wp_left}" if wp_left is not None else ""
                print(
                    f"[VLA_V2][EP {int(ep)+1}/{num_episodes}] PHASE {name}{wp_str} "
                    f"step={int(steps)} EE_b=({ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}){extra}"
                )

            def _phase_final_goal(params: Dict[str, object]) -> Optional[Tuple[float, float, float]]:
                pts = list(params.get("points", []))
                if not pts:
                    return None
                p = pts[-1]
                return (float(p[0]), float(p[1]), float(p[2]))

            accum = 0.0
            steps = 0
            episode_done = False

            while simulation_app.is_running() and steps < max_steps_ep and not episode_done:
                steps += 1

                # Update follower with latest EE pose
                try:
                    ee_now = _ee_pos_b()
                    if ee_now is not None:
                        wp.set_current_pose_b(torch.tensor(ee_now, dtype=torch.float32, device=sim.device))
                except Exception:
                    pass

                if stabilize_left > 0:
                    stabilize_left -= 1
                    if stabilize_left == 0:
                        ee = _ee_pos_b()
                        if ee is not None:
                            print(
                                f"[VLA_V2][EP {int(ep)+1}/{num_episodes}] settled. "
                                f"EE_b=({ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f})"
                            )
                else:
                    name, params = phases[phase_idx]
                    log_name = str(params.get("name", name)).upper()

                    if name == "done":
                        episode_done = True

                    elif name in ("open_gripper", "close_gripper"):
                        steps_total = int(params.get("steps", 30))
                        value = float(params.get("value", +1.0))
                        if not phase_state.get("queued", False):
                            _print_phase_start(log_name)
                            session_logger.log_event(
                                "action_start",
                                {"action": log_name, "steps": int(steps_total), "value": float(value),
                                 "episode_idx": int(ep)},
                            )
                            try:
                                controller.set_mode("gripper")
                                wp.queue_gripper(value, steps=int(steps_total))
                            except Exception:
                                pass
                            phase_state["queued"] = True
                            phase_state["wait_left"] = int(steps_total) + 4
                        controller.set_mode("gripper")
                        if int(phase_state.get("wait_left", 0)) > 0:
                            phase_state["wait_left"] = int(phase_state["wait_left"]) - 1
                        else:
                            session_logger.log_event("action_end", {"action": log_name, "episode_idx": int(ep)})
                            phase_idx += 1
                            phase_step_count = 0
                            phase_state = {}
                            controller.set_mode("translate")

                    elif name == "hold":
                        steps_total = int(params.get("steps", 30))
                        if not phase_state.get("started", False):
                            _print_phase_start(log_name)
                            session_logger.log_event(
                                "action_start",
                                {"action": log_name, "steps": int(steps_total), "episode_idx": int(ep)},
                            )
                            phase_state["started"] = True
                            phase_state["left"] = int(steps_total)
                        controller.set_mode("translate")
                        if int(phase_state.get("left", 0)) > 0:
                            phase_state["left"] = int(phase_state["left"]) - 1
                        else:
                            session_logger.log_event("action_end", {"action": log_name, "episode_idx": int(ep)})
                            phase_idx += 1
                            phase_step_count = 0
                            phase_state = {}

                    elif name == "waypoint":
                        if not phase_state.get("queued", False):
                            pts = list(params.get("points", []))
                            dense = _densify(
                                [(float(p[0]), float(p[1]), float(p[2])) for p in pts],
                                max_seg_m=float(getattr(args, "planner_waypoint_max_seg_m", 0.01)),
                            )
                            _print_phase_start(log_name)
                            log_extra = dict(params.get("log", {}) or {})
                            log_extra.update(
                                {
                                    "action": log_name,
                                    "n_subgoals": int(len(pts)),
                                    "n_waypoints_dense": int(len(dense)),
                                    "subgoals": [[float(p[0]), float(p[1]), float(p[2])] for p in pts],
                                    "episode_idx": int(ep),
                                }
                            )
                            session_logger.log_event("action_start", log_extra)
                            controller.set_mode("translate")
                            wp.set_waypoints_b(dense)
                            phase_state["queued"] = True
                            phase_state["start_step"] = int(steps)
                            phase_state["last_progress_step"] = int(steps)
                            phase_state["final_goal"] = _phase_final_goal(params)

                        spent = int(steps) - int(phase_state.get("start_step", steps))
                        wp_left = len(getattr(wp, "_waypoints_b", []))

                        if (int(steps) - int(phase_state.get("last_progress_step", steps))) >= progress_stride:
                            _print_progress(log_name, goal=phase_state.get("final_goal"), wp_left=int(wp_left))
                            phase_state["last_progress_step"] = int(steps)

                        if wp_left == 0 or spent >= phase_timeout:
                            timed_out = bool(spent >= phase_timeout and wp_left > 0)
                            ee = _ee_pos_b()
                            print(
                                f"[VLA_V2][EP {int(ep)+1}/{num_episodes}] PHASE {log_name} done "
                                f"spent={spent}st wp_left={wp_left}{' (TIMED_OUT)' if timed_out else ''} "
                                f"EE_b=({ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f})" if ee is not None
                                else f"[VLA_V2][EP {int(ep)+1}/{num_episodes}] PHASE {log_name} done"
                            )
                            session_logger.log_event(
                                "action_end",
                                {"action": log_name,
                                 "n_remaining_waypoints": int(wp_left),
                                 "spent_steps": int(spent),
                                 "timed_out": bool(timed_out),
                                 "episode_idx": int(ep)},
                            )
                            try:
                                wp.set_waypoints_b([])
                            except Exception:
                                pass
                            phase_idx += 1
                            phase_step_count = 0
                            phase_state = {}

                    else:
                        # Unknown phase: skip
                        phase_idx += 1
                        phase_step_count = 0
                        phase_state = {}

                    phase_step_count += 1

                # Step controller + sim
                try:
                    controller.step(robot, dt)
                except Exception:
                    pass

                do_tick = (accum + dt + 1e-9) >= period
                do_render = bool(do_tick) or (steps % render_stride == 0)
                sim.step(render=bool(do_render))
                robot.update(dt)

                if camera_sensor is not None:
                    try:
                        if hasattr(camera_sensor, "update") and bool(do_render):
                            camera_sensor.update(dt)
                    except Exception:
                        pass

                accum += dt
                if accum + 1e-9 >= period:
                    accum = 0.0

                if do_tick:
                    objs_raw = []
                    try:
                        for o in tracker.snapshot():
                            lbl = id_to_label.get(o.id, o.label)
                            objs_raw.append(
                                {
                                    "id": o.id,
                                    "label": lbl,
                                    "pose": {
                                        "position_m": list(o.pose.position_m),
                                        "orientation_wxyz": list(o.pose.orientation_wxyz),
                                    },
                                    "confidence": o.confidence,
                                }
                            )
                    except Exception:
                        pass

                    image_path = None
                    if camera_sensor is not None:
                        try:
                            cam_data = camera_sensor.data
                            rgb_data = None
                            if cam_data.output is not None:
                                rgb_data = cam_data.output.get("rgb")
                            if rgb_data is not None:
                                if len(rgb_data.shape) == 4:
                                    rgb_np = rgb_data[0].cpu().numpy()
                                elif len(rgb_data.shape) == 3:
                                    rgb_np = rgb_data.cpu().numpy()
                                else:
                                    raise ValueError(f"Unexpected RGB data shape: {rgb_data.shape}")
                                if rgb_np.max() <= 1.0:
                                    rgb_np = (rgb_np * 255).astype(np.uint8)
                                else:
                                    rgb_np = rgb_np.astype(np.uint8)
                                image_filename = f"image_{session_logger.tick_idx:06d}.{image_format}"
                                out_path = images_dir / image_filename
                                try:
                                    from PIL import Image

                                    Image.fromarray(rgb_np).save(str(out_path))
                                except Exception:
                                    try:
                                        import cv2

                                        if len(rgb_np.shape) == 3 and rgb_np.shape[2] == 3:
                                            rgb_np_bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
                                            cv2.imwrite(str(out_path), rgb_np_bgr)
                                        else:
                                            cv2.imwrite(str(out_path), rgb_np)
                                    except Exception:
                                        np.save(str(out_path).replace(f".{image_format}", ".npy"), rgb_np)
                                        out_path = out_path.with_suffix(".npy")
                                image_path = f"images/{image_filename}"
                                images_captured_episode += 1
                        except Exception:
                            image_path = None

                    session_logger.write_tick(
                        robot=robot,
                        controller=controller,
                        objects=objs_raw,
                        last_user_cmd=mux_input.last_cmd,
                        cfg=tick_cfg,
                        image_path=image_path,
                    )

            # Evaluate drop success at the end (or at episode timeout).
            final_state = _read_target_state()
            ok = bool(_drop_success(final_state, bin_meta))
            session_logger.log_event(
                "drop_result",
                {
                    "episode_idx": int(ep),
                    "ok": bool(ok),
                    "bin_idx": int(lang_meta.get("bin_idx", 0)),
                    "bin_color": lang_meta.get("bin_color"),
                    "target_pos": list(final_state["pos"]) if (final_state and "pos" in final_state) else None,
                    "bin_center_xy": list(bin_meta.get("center_xy", (0.0, 0.0))),
                    "bin_inner_xy": list(bin_meta.get("inner_xy", (0.0, 0.0))),
                    "bin_floor_z": float(bin_meta.get("bin_floor_z", 0.0)),
                    "drop_success_margin_m": float(getattr(args, "drop_success_margin_m", 0.04)),
                },
            )
            session_logger.log_event(
                "episode_end",
                {"episode_idx": int(ep), "steps": int(steps), "truncated": not bool(episode_done), "ok": bool(ok)},
            )
            final_pos_str = (
                f"({final_state['pos'][0]:.3f}, {final_state['pos'][1]:.3f}, {final_state['pos'][2]:.3f})"
                if final_state and "pos" in final_state
                else "n/a"
            )
            print(
                f"[VLA_V2][EP {int(ep)+1}/{num_episodes}] end "
                f"ok={ok} steps={steps} ticks={session_logger.tick_idx} images={images_captured_episode} "
                f"final_box_pos={final_pos_str} "
                f"bin {int(lang_meta.get('bin_idx', 0))} ({lang_meta.get('bin_color')})"
            )

            total_ticks += session_logger.tick_idx
            total_images += images_captured_episode
            try:
                session_logger.close()
            except Exception:
                pass

        except Exception as ep_error:
            import traceback

            print(f"[VLA_V2][EP][ERROR] Episode {ep} failed: {ep_error}")
            traceback.print_exc()
            try:
                if "session_logger" in locals():
                    session_logger.close()
            except Exception:
                pass
            continue

    print("\n[VLA_V2] Data collection completed!")
    print(f"[VLA_V2] Session directory: {session_folder}")
    print(f"[VLA_V2] Total episodes: {num_episodes}")
    print(f"[VLA_V2] Total ticks logged: {total_ticks}")
    print(f"[VLA_V2] Total images captured: {total_images}")

    simulation_app.close()
    return 0


PROFILE = ProfileSpec(
    name="vla_v2",
    add_cli_args=add_cli_args,
    run=run,
)
