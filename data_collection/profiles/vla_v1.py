from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Optional

from data_collection.envs.registry import get_envs
from data_collection.profiles.spec import ProfileSpec


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--env", type=str, default="reach_to_grasp_VLA", choices=sorted(get_envs().keys()))
    parser.add_argument("--logs-root", type=str, default="logs/data_collection")
    # For VLA training, a 5Hz policy/control rate is a good default trade-off between
    # responsiveness and dataset size (and matches common OpenVLA-style datasets).
    parser.add_argument("--log-rate-hz", type=int, default=5)
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--control", type=str, default="keyboard", choices=["keyboard", "idle", "planner"])
    parser.add_argument("--image-format", type=str, default="png", choices=["png", "jpg"], help="Image format for saving")
    # Domain randomization (toggleable; off by default). Intended to improve sim2real robustness.
    # Applied once per episode in planner mode (and once at start in non-planner modes).
    parser.add_argument("--domain-rand", action="store_true", help="Enable domain randomization (camera + lighting). Default: off.")
    parser.add_argument("--domain-rand-seed", type=int, default=None, help="Optional base RNG seed for domain randomization.")
    # Camera: jitter around the configured DEFAULT_TOP_DOWN_CAMERA pose (meters/degrees).
    parser.add_argument("--domain-rand-camera-xy-m", type=float, default=0.02, help="Uniform XY jitter (m) for top-down camera.")
    parser.add_argument("--domain-rand-camera-z-m", type=float, default=0.10, help="Uniform Z jitter (m) for top-down camera height.")
    parser.add_argument("--domain-rand-camera-yaw-deg", type=float, default=20.0, help="Uniform yaw jitter (deg) for top-down camera.")
    parser.add_argument("--domain-rand-camera-pitch-deg", type=float, default=0.0, help="Uniform pitch jitter (deg) for top-down camera (tilt).")
    parser.add_argument("--domain-rand-camera-roll-deg", type=float, default=0.0, help="Uniform roll jitter (deg) for top-down camera (tilt).")
    parser.add_argument("--domain-rand-camera-fov-deg", type=float, default=5.0, help="Uniform FOV jitter (deg) around default camera FOV.")
    # Lighting: jitter dome light intensity and color.
    parser.add_argument("--domain-rand-light-intensity-mult-min", type=float, default=0.5, help="Min multiplier for dome light intensity.")
    parser.add_argument("--domain-rand-light-intensity-mult-max", type=float, default=1.5, help="Max multiplier for dome light intensity.")
    parser.add_argument("--domain-rand-light-color-jitter", type=float, default=0.15, help="Per-channel RGB jitter for dome light color (0..1).")
    # vla_v1: nudge default spawn region slightly away from the robot to reduce "too-close" grasps.
    # (These args are registered by scripts/cli.add_demo_cli_args before this profile hook runs.)
    try:
        parser.set_defaults(spawn_min=[0.28, -0.30, 0.90])
    except Exception:
        pass
    # vla_v1: spread objects out more by default to reduce cluttered grasps (especially for box mode).
    try:
        parser.set_defaults(min_distance=0.20)
    except Exception:
        pass
    # Workspace safety bounds (base frame). If Z min is too high, the arm can never descend to the grasp depth
    # and will "hover" forever above the object.
    parser.add_argument("--workspace-min-z", type=float, default=0.0, help="Minimum EE z in base frame (m).")
    # Default was previously 0.35m, which is below the tabletop in this scene (~0.8-0.9m),
    # causing the controller to clamp Z and the arm to jitter/oscillate near the clamp.
    parser.add_argument("--workspace-max-z", type=float, default=1.20, help="Maximum EE z in base frame (m).")
    # Waypoint follower tuning (planner execution)
    parser.add_argument(
        "--wp-max-steps-per-waypoint",
        type=int,
        default=2400,
        help="Max physics steps allowed per waypoint before giving up (higher = less 'hover then replan').",
    )
    parser.add_argument(
        "--wp-stagnation-steps",
        type=int,
        default=10**9,
        help="Disable/raise stagnation watchdog. If distance doesn't improve for this many steps, waypoint is dropped.",
    )
    parser.add_argument(
        "--wp-stagnation-eps-m",
        type=float,
        default=5e-4,
        help="Stagnation improvement threshold in meters (only used if wp-stagnation-steps is finite).",
    )
    parser.add_argument(
        "--replan-cooldown-steps",
        type=int,
        default=120,
        help="Minimum physics steps between replans to avoid plan spam when execution is stalled.",
    )

    # Planner-driven reach-to-grasp (VLA)
    parser.add_argument(
        "--planner",
        type=str,
        default="curobo_v2",
        choices=["curobo_v2", "curobo", "lula", "rmpflow", "scripted"],
        help="Planner backend for --control planner (default: curobo_v2)",
    )
    parser.add_argument("--target-label", type=str, default=None, help="Optional target object label filter")
    parser.add_argument(
        "--target-prim",
        type=str,
        default=None,
        help="Optional explicit USD prim path for the grasp target (e.g. /World/Origin1/Objects/Obj_01).",
    )
    parser.add_argument(
        "--target-index",
        type=int,
        default=None,
        help="Optional index into spawned objects list (sorted). Overrides --target-label if set.",
    )
    parser.add_argument(
        "--target-selection",
        type=str,
        default="first",
        choices=["first", "random"],
        help="How to choose the target object when no --target-prim/--target-index is provided.",
    )
    parser.add_argument(
        "--ee-z-offset-m",
        type=float,
        default=0.08,
        help=(
            "Vertical offset added to the grasp target position before planning/execution (base frame). "
            "Use this to account for the fact that --ee_link is typically at the wrist/palm, not the fingertip TCP. "
            "If the robot 'rests on top' of objects and never reaches the down/grasp waypoint, increase this."
        ),
    )
    parser.add_argument("--pregrasp", type=float, default=0.10, help="Pre-grasp offset above object top (m)")
    parser.add_argument(
        "--grasp-depth",
        type=float,
        default=-0.07,
        help="Grasp depth relative to the estimated object top (m). Negative goes downward.",
    )
    parser.add_argument(
        "--ignore-grasp-orientation",
        action="store_true",
        default=False,
        help="Plan with position-only goals (faster/more robust than constraining EE orientation). Default: false.",
    )
    parser.add_argument("--lift", type=float, default=0.15, help="Lift height after grasp (m)")
    parser.add_argument(
        "--post-lift-hold-steps",
        type=int,
        default=30,
        help="Hold steps after reaching lift waypoint before evaluating lift success (helps object catch up).",
    )
    # IMPORTANT: This should not be larger than --close-if-within-m; otherwise the waypoint follower may
    # declare the final approach "done" before we are allowed to close, leading to hover/requeue loops.
    parser.add_argument("--tolerance", type=float, default=0.005, help="Waypoint tolerance for planner control (m)")
    parser.add_argument("--stabilize-steps", type=int, default=300, help="Hold steps before first plan (physics settle)")
    parser.add_argument("--gripper-open-steps", type=int, default=10, help="Steps to open gripper before approach")
    parser.add_argument("--gripper-close-steps", type=int, default=60, help="Steps to close gripper at grasp")
    parser.add_argument(
        "--hold-after-close-steps",
        type=int,
        default=30,
        help="Hold steps after closing gripper before lifting (lets contacts settle).",
    )
    parser.add_argument(
        "--close-if-within-m",
        type=float,
        default=0.005,
        help="Only start closing gripper if EE is within this distance of the final approach goal.",
    )
    parser.add_argument(
        "--grasp-depth-step",
        type=float,
        default=-0.02,
        help="Per-attempt adjustment applied to --grasp-depth when retrying (negative goes deeper).",
    )
    parser.add_argument(
        "--empty-close-threshold",
        type=float,
        default=0.06,
        help="If mean gripper joint position is within this of close_position, treat as 'empty grasp' and retry.",
    )
    parser.add_argument(
        "--approach-stall-steps-to-close",
        type=int,
        default=900,
        help=(
            "If EE is unable to reduce distance to the final approach goal for this many steps, treat the approach as stalled "
            "(will retry/end attempt depending on --max-grasp-attempts)."
        ),
    )
    parser.add_argument(
        "--approach-stall-eps-m",
        type=float,
        default=2e-4,
        help="Distance improvement threshold (m) for stall detection in approach stage (smaller = less aggressive).",
    )
    parser.add_argument(
        "--approach-stall-grace-steps",
        type=int,
        default=300,
        help="Grace period (physics steps) at the start of approach before stall detection can trigger.",
    )
    parser.add_argument(
        "--force-close-after-approach-steps",
        type=int,
        default=2400,
        help=(
            "Max approach duration in physics steps. If we still can't get close enough by then, count as a failed attempt "
            "(prevents infinite hover)."
        ),
    )
    parser.add_argument(
        "--max-grasp-attempts",
        type=int,
        default=3,
        help="Max grasp attempts within a single episode before giving up (replan + retry).",
    )
    parser.add_argument(
        "--replan-if-far-m",
        type=float,
        default=0.03,
        help="If the EE is farther than this from the final approach waypoint, replan instead of closing.",
    )
    parser.add_argument(
        "--lift-success-min-dz-m",
        type=float,
        default=0.06,
        help="Success threshold: target object must be at least this much above table height after lift.",
    )
    parser.add_argument(
        "--lift-success-confirm-steps",
        type=int,
        default=10,
        help="Require lift success to hold for N consecutive physics steps before ending the episode (filters spikes).",
    )
    parser.add_argument(
        "--lift-success-check-stride",
        type=int,
        default=5,
        help="Check lift success every N physics steps while lifting (smaller = more responsive, bigger = cheaper).",
    )
    parser.add_argument(
        "--spawn-min-robot-dist",
        type=float,
        default=0.30,
        help="Minimum XY distance from robot base for object spawn/respawn sampling (m).",
    )

    # Episode control (NEW in v1.1): end after grasp+lift, then reset and repeat.
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of episodes to run (planner mode)")
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=8000,
        help="Hard cap on physics steps per episode to avoid getting stuck",
    )
    parser.add_argument(
        "--render-rate-hz",
        type=float,
        default=60.0,
        help="Max render rate for interactive runs. Physics still steps at sim dt; rendering is throttled.",
    )
    parser.add_argument(
        "--respawn-each-episode",
        action="store_true",
        help=(
            "Re-randomize object poses each episode. "
            "Note: for PhysX GPU stability we do NOT destroy/recreate rigid bodies; we teleport existing ones."
        ),
    )
    parser.add_argument(
        "--respawn-every-n-episodes",
        type=int,
        default=0,
        help=(
            "Re-randomize object poses every N episodes in planner mode. "
            "If 0 (default), respawn once per full target cycle (i.e., every num_objects episodes). "
            "Use 1 to respawn every episode."
        ),
    )
    parser.add_argument(
        "--no-respawn-each-episode",
        action="store_true",
        help="Disable object pose randomization between episodes (planner mode defaults to respawn once per target cycle).",
    )
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=180,
        help="Physics settle steps after reset/respawn before planning (reduces initial bounce/ragdoll).",
    )

    # NEW in v1: dynamic cuRobo world sync (USD -> cuRobo cuboids)
    # Enable cuRobo world sync by default for stable motion (avoids planning straight through objects/table).
    parser.add_argument(
        "--curobo-world-from-scene",
        action="store_true",
        default=True,
        help="Update cuRobo world collision model from current USD scene prim bounds before planning (default: enabled).",
    )
    parser.add_argument(
        "--no-curobo-world-from-scene",
        action="store_true",
        help="Disable cuRobo world collision model sync (faster, but can cause collisions/jitter).",
    )
    # optional planner speed override (keeps keyboard teleop fast but planner execution smooth)
    parser.add_argument(
        "--planner-speed-mps",
        type=float,
        default=0.4,
        help="Planner execution linear speed (m/s). Used only for --control planner.",
    )
    parser.add_argument(
        "--planner-waypoint-max-seg-m",
        type=float,
        default=0.01,
        help="Max segment length (m) after waypoint densification for smoother execution.",
    )
    parser.add_argument(
        "--curobo-world-include-table",
        action="store_true",
        default=True,
        help="Include /World/Origin1/Table as an obstacle when building cuRobo world (default: true).",
    )

    # Spawn mode: "usd" or "box"
    parser.add_argument(
        "--spawn-mode",
        type=str,
        default="usd",
        choices=["usd", "box"],
        help="Spawn mode: 'usd' for YCB objects, 'box' for uniform cubes.",
    )
    parser.add_argument(
        "--box-size",
        type=float,
        default=0.08,
        help="Side length for uniform boxes (m). Used if spawn-mode=box.",
    )


def run(args: argparse.Namespace) -> int:
    # Ensure kinova-isaac root is first on sys.path (Kit may mutate sys.path).
    from pathlib import Path as _Path

    ROOT = _Path(__file__).resolve().parents[2]
    root_str = str(ROOT)
    if root_str in sys.path:
        sys.path.remove(root_str)
    sys.path.insert(0, root_str)
    _env_mod = sys.modules.get("environments")
    if _env_mod is not None and not hasattr(_env_mod, "__path__"):
        del sys.modules["environments"]
    # Same collision as in collect_data.py: Isaac may preload `cv2.utils` which shadows our `utils/`.
    _utils_mod = sys.modules.get("utils")
    if _utils_mod is not None:
        _utils_file = str(getattr(_utils_mod, "__file__", "") or "")
        if _utils_file and root_str not in _utils_file:
            for _k in list(sys.modules.keys()):
                if _k == "utils" or _k.startswith("utils."):
                    del sys.modules[_k]

    # Heavy imports must happen only after Kit is started.
    from isaaclab.app import AppLauncher
    from data_collection.core.input_mux import CommandMuxInputProvider
    from data_collection.core.logger import SessionLogWriter, TickLoggingConfig
    from data_collection.core.objects import ObjectsTracker

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    if getattr(args, "suppress_spam", False):
        # NOTE: Temporary no-op.
        # Attempts to change Kit's logging/notification settings dynamically have proven unreliable across Kit versions
        # and can corrupt internal dictionaries (leading to crashes like "item omni.physx.plugin is not a string").
        print("[VLA_V1][WARN] --suppress-spam is currently a no-op (kept for compatibility).")

    import torch  # noqa: E402
    try:
        import numpy as np  # noqa: E402
    except Exception as e:
        print(f"[VLA_V1] ERROR: numpy is required for image saving but could not be imported: {e}")
        print("[VLA_V1] Please install numpy in your environment (e.g., `pip install numpy`) and retry.")
        return 2

    import carb  # noqa: E402

    carb_settings = carb.settings.get_settings()
    enable_cameras = bool(getattr(args, "enable_cameras", False))
    carb_settings.set_bool("/isaaclab/cameras_enabled", enable_cameras)
    print(f"[VLA_V1] enable_cameras flag value: {enable_cameras}")
    print(f"[VLA_V1] carb /isaaclab/cameras_enabled={carb_settings.get('/isaaclab/cameras_enabled')}")

    import importlib
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import Camera, CameraCfg
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

    from controllers import CartesianVelocityJogConfig, CartesianVelocityJogController
    from environments.utils.object_loader import ObjectLoader, ObjectLoaderConfig, SpawnBounds
    from environments.utils.physix import PhysicsConfig, apply_to_simulation_cfg, object_loader_kwargs_from_physix

    env_spec = get_envs()[str(getattr(args, "env", "reach_to_grasp_VLA"))]
    env_cfg_mod = importlib.import_module(f"{env_spec.module_base}.config")
    env_utils_mod = importlib.import_module(f"{env_spec.module_base}.utils")
    DEFAULT_SCENE = getattr(env_cfg_mod, "DEFAULT_SCENE")
    DEFAULT_CAMERA = getattr(env_cfg_mod, "DEFAULT_CAMERA", None)
    DEFAULT_TOP_DOWN_CAMERA = getattr(env_cfg_mod, "DEFAULT_TOP_DOWN_CAMERA", None)
    design_scene = getattr(env_utils_mod, "design_scene")
    create_topdown_camera = getattr(importlib.import_module("environments.utils.camera"), "create_topdown_camera")

    # Setup sim
    phys = PhysicsConfig(device=str(getattr(args, "device", "cuda:0")))
    sim_cfg = sim_utils.SimulationCfg(device=phys.device)
    apply_to_simulation_cfg(sim_cfg, phys)
    sim = sim_utils.SimulationContext(sim_cfg)
    if (not getattr(args, "headless", False)) and DEFAULT_CAMERA is not None:
        sim.set_camera_view(DEFAULT_CAMERA.eye, DEFAULT_CAMERA.target)

    # Build scene and robot
    scene_entities, scene_origins = design_scene(DEFAULT_SCENE)
    robot = scene_entities["kinova_j2n6s300"]

    # Create top-down camera prim ONLY if cameras are enabled.
    # IsaacLab will error during sim.reset() if a Camera exists without --enable_cameras.
    if enable_cameras and DEFAULT_TOP_DOWN_CAMERA is not None:
        create_topdown_camera(DEFAULT_TOP_DOWN_CAMERA)
        print(f"[VLA_V1] Top-down camera created at: {DEFAULT_TOP_DOWN_CAMERA.prim_path}")

    # -----------------------------
    # Domain randomization helpers
    # -----------------------------
    domain_rand_enabled = bool(getattr(args, "domain_rand", False))
    # Cache defaults so each episode randomizes around a stable baseline (no drift).
    _dr_base_cam_pos = None
    _dr_base_cam_fov = None
    _dr_cam_prim_path = None
    try:
        if DEFAULT_TOP_DOWN_CAMERA is not None:
            _dr_base_cam_pos = tuple(float(v) for v in getattr(DEFAULT_TOP_DOWN_CAMERA, "position", (0.4, 0.0, 4.0)))
            _dr_base_cam_fov = float(getattr(DEFAULT_TOP_DOWN_CAMERA, "fov", 65.0))
            _dr_cam_prim_path = str(getattr(DEFAULT_TOP_DOWN_CAMERA, "prim_path", ""))
    except Exception:
        _dr_base_cam_pos = None
        _dr_base_cam_fov = None
        _dr_cam_prim_path = None

    _dr_light_prim_path = "/World/Light"
    _dr_seed_base = None
    try:
        # Use user-provided seed when available; otherwise sample once per run.
        if getattr(args, "domain_rand_seed", None) is not None:
            _dr_seed_base = int(getattr(args, "domain_rand_seed"))
        else:
            _dr_seed_base = int(time.time() * 1000) & 0x7FFFFFFF
    except Exception:
        _dr_seed_base = 0

    def _apply_domain_randomization(*, ep_idx: int, logger: Optional["SessionLogWriter"] = None) -> Optional[dict]:
        """Apply domain randomization (camera + lighting) once per episode.

        Off by default; enabled via --domain-rand.
        Returns the sampled parameters dict when applied.
        """
        if not domain_rand_enabled:
            return None
        try:
            import random
            import importlib

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

        # --- Lighting: dome light intensity + color
        try:
            prim = stage.GetPrimAtPath(str(_dr_light_prim_path))
            if prim.IsValid():
                dome = UsdLux.DomeLight(prim)
                # Read current as baseline if possible
                base_int = None
                base_col = None
                try:
                    base_int = float(dome.GetIntensityAttr().Get())
                except Exception:
                    base_int = 2000.0
                try:
                    c = dome.GetColorAttr().Get()
                    base_col = (float(c[0]), float(c[1]), float(c[2]))
                except Exception:
                    base_col = (0.75, 0.75, 0.75)

                mult_min = float(getattr(args, "domain_rand_light_intensity_mult_min", 0.5))
                mult_max = float(getattr(args, "domain_rand_light_intensity_mult_max", 1.5))
                mult = float(rng.uniform(min(mult_min, mult_max), max(mult_min, mult_max)))
                intensity = float(max(0.0, base_int * mult))

                jitter = float(getattr(args, "domain_rand_light_color_jitter", 0.15))
                def _clamp01(x: float) -> float:
                    return float(max(0.0, min(1.0, x)))
                color = (
                    _clamp01(base_col[0] + rng.uniform(-jitter, jitter)),
                    _clamp01(base_col[1] + rng.uniform(-jitter, jitter)),
                    _clamp01(base_col[2] + rng.uniform(-jitter, jitter)),
                )

                dome.GetIntensityAttr().Set(float(intensity))
                dome.GetColorAttr().Set(Gf.Vec3f(float(color[0]), float(color[1]), float(color[2])))
                out["light"] = {
                    "prim_path": str(_dr_light_prim_path),
                    "intensity": float(intensity),
                    "intensity_mult": float(mult),
                    "color_rgb": [float(color[0]), float(color[1]), float(color[2])],
                }
        except Exception:
            pass

        # --- Camera: pose + yaw + (optional) tilt + FOV
        try:
            if enable_cameras and _dr_cam_prim_path and _dr_base_cam_pos is not None:
                cam_prim = stage.GetPrimAtPath(str(_dr_cam_prim_path))
                if cam_prim.IsValid():
                    # Pose jitter
                    xy = float(getattr(args, "domain_rand_camera_xy_m", 0.02))
                    z_j = float(getattr(args, "domain_rand_camera_z_m", 0.10))
                    dx = float(rng.uniform(-xy, xy))
                    dy = float(rng.uniform(-xy, xy))
                    dz = float(rng.uniform(-z_j, z_j))
                    x0, y0, z0 = _dr_base_cam_pos
                    pos = (float(x0 + dx), float(y0 + dy), float(max(0.5, z0 + dz)))

                    # Orientation jitter (degrees)
                    yaw_rng = float(getattr(args, "domain_rand_camera_yaw_deg", 20.0))
                    pitch_rng = float(getattr(args, "domain_rand_camera_pitch_deg", 0.0))
                    roll_rng = float(getattr(args, "domain_rand_camera_roll_deg", 0.0))
                    yaw = float(rng.uniform(-yaw_rng, yaw_rng))
                    pitch = float(rng.uniform(-pitch_rng, pitch_rng)) if pitch_rng > 0 else 0.0
                    roll = float(rng.uniform(-roll_rng, roll_rng)) if roll_rng > 0 else 0.0

                    xform = UsdGeom.Xformable(cam_prim)
                    xform.ClearXformOpOrder()
                    translate_op = xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
                    rotate_op = xform.AddRotateXYZOp(UsdGeom.XformOp.PrecisionFloat)
                    translate_op.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
                    rotate_op.Set(Gf.Vec3f(float(roll), float(pitch), float(yaw)))

                    # FOV jitter via focal length
                    base_fov = float(_dr_base_cam_fov) if _dr_base_cam_fov is not None else 65.0
                    fov_j = float(getattr(args, "domain_rand_camera_fov_deg", 5.0))
                    fov = float(base_fov + rng.uniform(-fov_j, fov_j))
                    fov = float(max(15.0, min(120.0, fov)))
                    try:
                        cam = UsdGeom.Camera(cam_prim)
                        import numpy as np  # noqa: F401
                        # Match environments/utils/camera/topdown.py convention.
                        import math as _math

                        sensor_size_mm = 36.0
                        focal_length_mm = sensor_size_mm / (2.0 * _math.tan(_math.radians(fov) / 2.0))
                        cam.GetFocalLengthAttr().Set(float(focal_length_mm))
                    except Exception:
                        pass

                    out["camera"] = {
                        "prim_path": str(_dr_cam_prim_path),
                        "pos_xyz": [float(pos[0]), float(pos[1]), float(pos[2])],
                        "rpy_deg": [float(roll), float(pitch), float(yaw)],
                        "fov_deg": float(fov),
                    }
        except Exception:
            pass

        # Log sampled randomization parameters for auditing.
        try:
            if logger is not None:
                logger.log_event("domain_randomization", out)
        except Exception:
            pass
        return out

    # Spawn objects (v1 uses helpers so we can respawn per episode)
    spawned_paths: list[str] = []
    id_to_label: Dict[str, str] = {}
    loader: ObjectLoader | None = None

    # Box appearance + language grounding (used only when spawn_mode="box").
    # Keep this stable/deterministic so logs are consistent for VLA training.
    BOX_COLORS: list[tuple[str, tuple[float, float, float]]] = [
        ("red", (0.85, 0.20, 0.20)),
        ("blue", (0.20, 0.35, 0.90)),
        ("yellow", (0.95, 0.85, 0.20)),
        ("purple", (0.65, 0.25, 0.80)),
        ("orange", (0.95, 0.55, 0.15)),
        ("cyan", (0.15, 0.80, 0.85)),
    ]

    def _box_idx_from_leaf(leaf: str) -> Optional[int]:
        try:
            if "_" in leaf:
                return int(str(leaf).split("_")[-1])
            return None
        except Exception:
            return None

    def _box_desc_from_prim(prim_path: str) -> tuple[str, Optional[str], Optional[int]]:
        """Return (human_label, color_name, box_idx) for box mode; otherwise best-effort label."""
        leaf = str(prim_path).split("/")[-1]
        idx = _box_idx_from_leaf(leaf)
        if idx is not None and len(BOX_COLORS) > 0:
            color_name = BOX_COLORS[(idx - 1) % len(BOX_COLORS)][0]
            return f"{color_name} box {idx}", color_name, idx
        # Fallback
        return f"box {leaf}", None, idx

    def _yaw_quat_wxyz(yaw_rad: float) -> tuple[float, float, float, float]:
        import math

        half = 0.5 * float(yaw_rad)
        return (math.cos(half), 0.0, 0.0, math.sin(half))

    def _spawn_objects() -> tuple[list[str], Dict[str, str]]:
        if getattr(args, "no_objects", False):
            return [], {}

        try:
            ycb_dir = f"{ISAAC_NUCLEUS_DIR}/Props/YCB"
        except Exception:
            ycb_dir = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Props/YCB"

        scale_range = None
        if getattr(args, "scale_min", None) is not None and getattr(args, "scale_max", None) is not None:
            scale_range = (float(args.scale_min), float(args.scale_max))

        nonlocal loader
        if loader is None:
            spawn_mode = str(getattr(args, "spawn_mode", "usd"))
            box_size = float(getattr(args, "box_size", 0.05))
            
            phys_loader_kwargs = object_loader_kwargs_from_physix(phys)
            _min_dist = float(getattr(args, "min_distance", 0.1))
            # For box mode we care about tabletop spacing (XY), even if we spawn at different Z.
            _min_dist_xy_only = (spawn_mode == "box")
            # Ensure a reasonable minimum spacing for stable clutter-free grasps.
            if _min_dist_xy_only:
                _min_dist = max(_min_dist, 0.20)
            loader_cfg = ObjectLoaderConfig(
                dataset_dirs=[ycb_dir],
                bounds=SpawnBounds(min_xyz=tuple(args.spawn_min), max_xyz=tuple(args.spawn_max)),
                min_distance=float(_min_dist),
                min_distance_xy_only=bool(_min_dist_xy_only),
                uniform_scale_range=scale_range,
                spawn_mode=spawn_mode,
                box_size_min=(box_size, box_size, box_size),
                box_size_max=(box_size, box_size, box_size),
                # Visual-only: per-box colors in box mode (does not affect physics).
                box_color_palette=[rgb for (_name, rgb) in BOX_COLORS],
                box_color_names=[name for (name, _rgb) in BOX_COLORS],
                **phys_loader_kwargs,
            )
            loader = ObjectLoader(loader_cfg)

        # NOTE: `--num-objects` is a shared CLI arg (see `scripts/cli.py`). Default to 4 if missing.
        paths = loader.spawn(parent_prim_path="/World/Origin1", num_objects=int(getattr(args, "num_objects", 4)))
        try:
            prim_to_label = loader.get_last_spawn_labels()
            lbl_map = {str(p).split("/")[-1]: str(lbl) for p, lbl in prim_to_label.items()}
        except Exception:
            lbl_map = {}

        # For box mode, override labels to include color + number (useful for training + debugging).
        try:
            if str(getattr(args, "spawn_mode", "usd")) == "box":
                for p in paths:
                    leaf = str(p).split("/")[-1]
                    human, _color, _idx = _box_desc_from_prim(str(p))
                    lbl_map[leaf] = human
        except Exception:
            pass
        return paths, lbl_map

    # IMPORTANT: spawn objects once up-front so planner mode has a loader + targets available.
    spawned_paths, id_to_label = _spawn_objects()
    try:
        print(f"[VLA_V1] Spawned {len(spawned_paths)} objects: {spawned_paths}")
    except Exception:
        pass

    def _prim_label(prim_path: str) -> str:
        """Best-effort: map a prim path to a human label using ObjectLoader's id_to_label mapping.

        Note: Object tracker ids typically match the leaf name (e.g. Obj_01).
        """
        try:
            leaf = str(prim_path).split("/")[-1]
            return str(id_to_label.get(leaf, "")) if id_to_label is not None else ""
        except Exception:
            return ""

    def _select_episode_target_prim(
        ep_idx: int,
        *,
        object_z_by_leaf: dict[str, float] | None = None,
        prev_target_prim: str | None = None,
    ) -> str | None:
        """Select exactly ONE target prim for an episode (stable for that episode)."""
        if not spawned_paths:
            return None

        # Prefer explicit prim path if provided
        explicit = getattr(args, "target_prim", None)
        if explicit:
            return str(explicit)

        # Deterministic ordering for index/first selection
        candidates_all = sorted([str(p) for p in spawned_paths])

        # Filter out objects that already fell off the table (common cause of "planner chases ragdoll").
        candidates = list(candidates_all)
        if object_z_by_leaf:
            try:
                table_z = float(getattr(phys, "snap_z_to", 0.82) if getattr(phys, "snap_z_to", None) is not None else 0.82)
                z_min_ok = table_z + 0.01  # 1cm above table plane
                filtered: list[str] = []
                for p in candidates_all:
                    leaf = str(p).split("/")[-1]
                    z = object_z_by_leaf.get(leaf, None)
                    if z is None or float(z) >= z_min_ok:
                        filtered.append(p)
                if filtered:
                    # Only apply filtering if we still have multiple candidates; otherwise we can get
                    # "stuck" selecting the same remaining object forever.
                    candidates = filtered if len(filtered) >= 2 else list(candidates_all)
            except Exception:
                candidates = list(candidates_all)

        idx = getattr(args, "target_index", None)
        if idx is not None:
            try:
                idx_i = int(idx)
                if len(candidates) == 0:
                    return None
                return candidates[idx_i % len(candidates)]
            except Exception:
                pass

        # Label filter (exact/substring) if provided
        label = getattr(args, "target_label", None)
        if label:
            try:
                label_l = str(label).lower()
                # Exact label match first
                for p in candidates:
                    if _prim_label(p).lower() == label_l:
                        return p
                # Substring match
                for p in candidates:
                    if label_l in _prim_label(p).lower():
                        return p
            except Exception:
                pass

        # Fallback selection policy
        sel = str(getattr(args, "target_selection", "first"))
        if sel == "random":
            try:
                import random

                return random.choice(candidates)
            except Exception:
                return candidates[0]
        elif sel == "first":
            # Proper episode-to-episode cycling (non-repeating when possible).
            # If we have a previous target and it's still a valid candidate, pick the *next* one.
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
            # Otherwise fall back to deterministic cycle by episode index.
            return candidates[ep_idx % len(candidates)]
        return candidates[0] if candidates else None

    def _make_language_command(
        *,
        ep_idx: int,
        target_prim: str,
    ) -> tuple[str, dict]:
        """Generate a natural-language instruction for VLA training (vla_v1 only)."""
        leaf = str(target_prim).split("/")[-1]
        human_label = _prim_label(str(target_prim)) or leaf
        # If we're in box mode and have a nice descriptor, use it.
        color_name = None
        box_idx = None
        try:
            if str(getattr(args, "spawn_mode", "usd")) == "box":
                human_label, color_name, box_idx = _box_desc_from_prim(str(target_prim))
        except Exception:
            pass

        templates = [
            "Pick up the {label}.",
            "Reach and pick up the {label}.",
            "Go to the {label} and grasp it.",
            "Grab the {label} and lift it.",
            "Please pick up the {label}.",
            "Move to the {label} and pick it up.",
        ]
        # Add more specific box templates when possible.
        if box_idx is not None:
            templates += [
                "Pick up box number {box_idx}.",
                "Go for box {box_idx} and pick it up.",
                "Grasp box {box_idx} and lift it.",
            ]
        if color_name is not None:
            templates += [
                "Pick up the {color} box.",
                "Reach to the {color} box and pick it up.",
                "Grab the {color} box and lift it.",
            ]
        if (box_idx is not None) and (color_name is not None):
            templates += [
                "Pick up the {color} box (box {box_idx}).",
                "Go to box {box_idx} — the {color} one — and pick it up.",
            ]

        # Deterministic per episode (stable dataset, still diverse across episodes).
        try:
            import random as _random

            rng = _random.Random(int(ep_idx) + 1337)
            tmpl = rng.choice(templates)
        except Exception:
            tmpl = templates[0]

        cmd = tmpl.format(label=human_label, color=str(color_name), box_idx=str(box_idx))
        meta = {
            "target_leaf": leaf,
            "target_label": human_label,
            "box_idx": box_idx,
            "color": color_name,
        }
        return cmd, meta

    def _table_z() -> float:
        try:
            t = getattr(DEFAULT_SCENE, "table_translation", (0.0, 0.0, 0.8))
            return float(t[2])
        except Exception:
            return 0.8

    def _target_z_from_tracker(target_prim: str) -> float | None:
        """Best-effort lookup of target object Z using tracker snapshot."""
        try:
            leaf = str(target_prim).split("/")[-1]
            for o in tracker.snapshot():
                if str(o.id) == leaf:
                    return float(o.pose.position_m[2])
        except Exception:
            return None
        return None

    # Cache for Isaac Sim core prim wrappers used during respawn (box mode).
    # We use `isaacsim.core.prims.RigidPrim` because it is physics-aware and survives repeated `sim.reset()`
    # better than raw tensor views in this script.
    _respawn_rigidprims: dict[str, object] = {}

    def _rerandomize_object_poses(
        paths: list[str],
        *,
        poses: Optional[list[tuple[tuple[float, float, float], float]]] = None,
    ) -> Optional[list[tuple[tuple[float, float, float], float]]]:
        """Teleport existing objects to poses (or new random poses) without delete/recreate.

        This avoids PhysX GPU tensor crashes that can happen if we destroy and recreate
        dynamic rigid bodies while tensor views exist.

        Args:
            paths: Object root prim paths (e.g. /World/Origin1/Objects/Obj_01).
            poses: Optional list of (position_xyz, yaw_rad) to apply. If None, new poses are sampled.

        Returns:
            The poses actually applied as a list of (position_xyz, yaw_rad), or None on failure.
        """
        if not paths:
            return None
        try:
            import random
            import math
            import torch
            import importlib
            from isaacsim.core.simulation_manager import SimulationManager
        except Exception:
            return None

        # Best-effort: locate the *actual* rigid body prim under each object root.
        # Some spawners create a container prim (e.g. Xform) at /Objects/Obj_XX and put the
        # RigidBodyAPI on a child prim. If we try to create a rigid-body view on the container,
        # the view can be empty and the "teleport" becomes a no-op (leading to identical poses
        # across episodes + target selection getting stuck).
        UsdPhysics = None
        omni_usd = None
        sim_utils = None
        RigidPrim = None
        try:
            UsdPhysics = importlib.import_module("pxr.UsdPhysics")
            omni_usd = importlib.import_module("omni.usd")
            sim_utils = importlib.import_module("isaaclab.sim")
            try:
                # Physics-aware rigid prim wrapper (teleport + velocities) backed by SimulationManager tensor views.
                RigidPrim = importlib.import_module("isaacsim.core.prims").RigidPrim
            except Exception:
                RigidPrim = None
        except Exception:
            UsdPhysics = None
            omni_usd = None
            sim_utils = None
            RigidPrim = None

        # Bounds are in meters relative to parent prim; in this env parent is /World/Origin1.
        # IMPORTANT: keep using the configured Z spawn band (often well above the table) so objects
        # fall and settle. Many YCB assets have their origin near the COM; placing them at "table_z"
        # can start them interpenetrating the table and cause a PhysX explosion.
        bmin = tuple(float(v) for v in getattr(args, "spawn_min", (0.30, -0.20, 0.81)))
        bmax = tuple(float(v) for v in getattr(args, "spawn_max", (0.55, 0.20, 0.81)))
        min_dist = float(getattr(args, "min_distance", 0.10))
        table_z_guess = float(getattr(DEFAULT_SCENE, "table_translation", (0.0, 0.0, 0.8))[2]) if "DEFAULT_SCENE" in locals() else 0.8
        z_min_safe = max(float(bmin[2]), float(table_z_guess) + 0.05)  # at least 5cm above table plane

        # If caller provided poses, use them; otherwise sample fresh ones.
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
            # Rejection sample in XYZ within spawn bounds, with a safety floor above the table.
            positions = []
            tries = 0
            while len(positions) < len(paths) and tries < 2000:
                tries += 1
                x = random.uniform(bmin[0], bmax[0])
                y = random.uniform(bmin[1], bmax[1])
                z = random.uniform(z_min_safe, float(bmax[2]))
                # Keep objects slightly away from the robot base (Origin1 frame) to reduce near-singularity grasps.
                try:
                    min_robot_dist = float(getattr(args, "spawn_min_robot_dist", 0.0))
                    if min_robot_dist > 1e-6 and math.hypot(float(x), float(y)) < min_robot_dist:
                        continue
                except Exception:
                    pass
                cand = (x, y, z)
                ok = True
                for p in positions:
                    dx = cand[0] - p[0]
                    dy = cand[1] - p[1]
                    # Enforce spacing on the tabletop (XY). Z is intentionally ignored because we often
                    # spawn above the table and let objects fall/settle.
                    if math.hypot(dx, dy) < min_dist:
                        ok = False
                        break
                if ok:
                    positions.append(cand)

            if len(positions) != len(paths):
                # Fallback: allow overlaps rather than failing.
                positions = [
                    (
                        random.uniform(bmin[0], bmax[0]),
                        random.uniform(bmin[1], bmax[1]),
                        random.uniform(z_min_safe, float(bmax[2])),
                    )
                    for _ in paths
                ]

            # Random yaw per object to diversify orientation.
            yaws = [random.uniform(-math.pi, math.pi) for _ in paths]

        sim_view = SimulationManager.get_physics_sim_view()
        # Convert parent-relative spawn coordinates into world coordinates using the scene origin.
        # (For single-env this is usually zero, but multi-origin scenes offset /World/Origin1.)
        origin0 = None
        try:
            origin0 = torch.tensor(scene_origins[0], device=sim.device).view(-1)  # type: ignore[name-defined]
        except Exception:
            origin0 = None

        def _teleport_via_rigidprim(
            *,
            rb_prim_path: str,
            pos_xyz: tuple[float, float, float],
            quat_wxyz: tuple[float, float, float, float],
        ) -> bool:
            """Teleport a rigid prim using `isaacsim.core.prims.RigidPrim` (box mode only).

            This does NOT call sim.reset() and does not delete/recreate prims.
            """
            if RigidPrim is None:
                return False
            try:
                import numpy as np

                key = str(rb_prim_path)
                rp = _respawn_rigidprims.get(key)
                if rp is None:
                    # reset_xform_properties=False avoids rewriting xformOpOrder on every init.
                    rp = RigidPrim(prim_paths_expr=str(rb_prim_path), name=f"respawn_{key.split('/')[-1]}", reset_xform_properties=False)
                    _respawn_rigidprims[key] = rp

                # Ensure physics handles are valid after sim.reset().
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

                # Teleport pose (RigidPrim uses scalar-first quaternions: wxyz).
                try:
                    rp.set_world_poses(positions=pos, orientations=ori)
                except TypeError:
                    # Some versions accept positional args.
                    rp.set_world_poses(pos, ori)

                # Zero velocities (GPU-safe): [vx,vy,vz, wx,wy,wz]
                try:
                    rp.set_velocities(np.zeros((1, 6), dtype=np.float32))
                except Exception:
                    # Best-effort fallback
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
                        stage = omni_usd.get_context().get_stage()  # type: ignore[union-attr]
                        root = stage.GetPrimAtPath(str(prim_path))
                        if root.IsValid() and root.HasAPI(UsdPhysics.RigidBodyAPI):  # type: ignore[union-attr]
                            rb_path = str(prim_path)
                        else:
                            get_all_matching_child_prims = getattr(sim_utils, "get_all_matching_child_prims", None)
                            if callable(get_all_matching_child_prims):
                                rigid_prims = get_all_matching_child_prims(
                                    str(prim_path),
                                    predicate=lambda p: p.HasAPI(UsdPhysics.RigidBodyAPI),  # type: ignore[union-attr]
                                    # IMPORTANT: box spawners and some assets can create instanceable prims;
                                    # if we don't traverse instance prims, we may fail to find the rigid body
                                    # and the teleport becomes a no-op.
                                    traverse_instance_prims=True,
                                )
                                if rigid_prims:
                                    try:
                                        rb_path = rigid_prims[0].GetPath().pathString
                                    except Exception:
                                        rb_path = str(prim_path)
                except Exception:
                    rb_path = str(prim_path)

                # For box mode, prefer physics-aware RigidPrim teleport (more reliable across sim.reset()).
                try:
                    if str(getattr(args, "spawn_mode", "usd")) == "box":
                        qw, qx, qy, qz = _yaw_quat_wxyz(yaw)
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
                # Resolve a usable rigid-body view:
                # - Some spawners produce a rigid-body on a child prim or instance proxy.
                # - Some view expressions can match 0 or >1 bodies; we must handle both.
                t0 = None
                try:
                    if hasattr(rb_view, "get_transforms"):
                        t0 = rb_view.get_transforms()
                except Exception:
                    t0 = None
                # If empty, try a wildcard match under the root.
                try:
                    if t0 is None or (hasattr(t0, "shape") and int(getattr(t0, "shape")[0]) == 0):
                        rb_view = sim_view.create_rigid_body_view(f"{str(prim_path)}/*")
                        if hasattr(rb_view, "get_transforms"):
                            t0 = rb_view.get_transforms()
                except Exception:
                    # If we still can't read transforms, skip this object.
                    t0 = None
                # If still empty, skip (keeps episode alive).
                try:
                    if t0 is not None and hasattr(t0, "shape") and int(getattr(t0, "shape")[0]) == 0:
                        continue
                except Exception:
                    pass
                qw, qx, qy, qz = _yaw_quat_wxyz(yaw)
                # RigidBodyView transform format: [x,y,z,qx,qy,qz,qw]
                px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
                if origin0 is not None and origin0.numel() >= 3:
                    px += float(origin0[0].item())
                    py += float(origin0[1].item())
                    pz += float(origin0[2].item())
                # Match the tensor device/shape expected by PhysX view.
                dev = sim.device
                n = 1
                try:
                    if t0 is not None and hasattr(t0, "device"):
                        dev = t0.device
                    if t0 is not None and hasattr(t0, "shape"):
                        n = int(getattr(t0, "shape")[0])
                        n = max(1, n)
                except Exception:
                    dev = sim.device
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
                # Best-effort: skip failures (keeps episode alive)
                continue
        return list(zip(positions, yaws))

    # Camera sensor for image capture (same pattern as vla_v0)
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
            print(f"[VLA_V1] Camera sensor created: {camera_cfg.width}x{camera_cfg.height}")
            print(f"[VLA_V1] Attached to existing camera prim: {DEFAULT_TOP_DOWN_CAMERA.prim_path}")
        except Exception as create_err:
            print(f"[VLA_V1] ERROR: Failed to create Camera object: {create_err}")
            camera_sensor = None

    # Reset sim and robot
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
            print("[VLA_V1] Camera sensor reset OK (post sim.reset)")
        except Exception as e:
            print(f"[VLA_V1] ERROR: Camera sensor reset failed post sim.reset: {e}")
            camera_sensor = None

    # Apply domain randomization once for non-episode runs (keyboard/idle). Planner mode applies per episode.
    # This is done after sim.reset so camera/light prims exist and are stable.
    try:
        if domain_rand_enabled:
            _apply_domain_randomization(ep_idx=0, logger=None)
            # If we moved the camera, best-effort reset the camera sensor so render products refresh.
            if camera_sensor is not None:
                try:
                    camera_sensor.reset()
                except Exception:
                    pass
    except Exception:
        pass

    # Controller
    # Smoother default for planner execution (keyboard teleop can remain snappier).
    # We keep user-provided --speed for keyboard mode, but default to --planner-speed-mps for planner mode.
    _control_mode_tmp = str(getattr(args, "control", "keyboard"))
    linear_speed_mps = float(getattr(args, "speed", 0.7))
    if _control_mode_tmp == "planner":
        linear_speed_mps = float(getattr(args, "planner_speed_mps", 0.25))

    ctrl_cfg = CartesianVelocityJogConfig(
        ee_link_name=str(getattr(args, "ee_link", "j2n6s300_end_effector")),
        device=str(sim.device),
        use_relative_mode=True,
        linear_speed_mps=float(linear_speed_mps),
        workspace_min=(0.20, -0.45, float(getattr(args, "workspace_min_z", 0.0))),
        workspace_max=(0.6, 0.45, float(getattr(args, "workspace_max_z", 1.20))),
        log_ee_pos=bool(getattr(args, "print_ee", False)),
        log_ee_frame=str(getattr(args, "ee_frame", "world")),
        log_every_n_steps=int(getattr(args, "print_interval", 1)),
    )
    controller = CartesianVelocityJogController(ctrl_cfg, num_envs=1, device=str(sim.device))
    controller.set_mode("translate")
    controller.reset(robot)

    # Input/control
    mux_input = CommandMuxInputProvider()
    control_mode = str(getattr(args, "control", "keyboard"))
    if (not getattr(args, "headless", False)) and control_mode == "keyboard":
        from controllers.input.keyboard import Se3KeyboardInput

        keyboard = Se3KeyboardInput(
            pos_sensitivity_per_step=ctrl_cfg.linear_speed_mps * sim.get_physics_dt(),
            rot_sensitivity_rad_per_step=float(getattr(args, "rot_speed", 2.0)) * sim.get_physics_dt(),
        )
        mux_input.set_base(keyboard)
    elif control_mode == "planner":
        from pathlib import Path as _Path

        from controllers.input.waypoint_follower import WaypointFollowerInput
        from motion_generation.grasp_estimation.obb import ObbGraspPoseProvider
        from motion_generation.mogen import MotionGenerationAgent
        from motion_generation.planners import PlannerContext, create_planner
        from utilities import get_ee_pos_base_frame

        def _densify_waypoints_b(
            pts: list[tuple[float, float, float]],
            *,
            max_seg_m: float,
        ) -> list[tuple[float, float, float]]:
            """Linear interpolation between waypoints to enforce bounded segment length (smoother motion)."""
            import math

            if not pts or len(pts) < 2:
                return pts
            max_seg_m = float(max(1e-6, max_seg_m))
            out: list[tuple[float, float, float]] = [tuple(map(float, pts[0]))]
            for (x0, y0, z0), (x1, y1, z1) in zip(pts, pts[1:]):
                dx = float(x1) - float(x0)
                dy = float(y1) - float(y0)
                dz = float(z1) - float(z0)
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                if dist <= max_seg_m:
                    out.append((float(x1), float(y1), float(z1)))
                    continue
                n = int(math.ceil(dist / max_seg_m))
                for i in range(1, n + 1):
                    t = float(i) / float(n)
                    out.append((float(x0 + t * dx), float(y0 + t * dy), float(z0 + t * dz)))
            return out

        if loader is None or len(spawned_paths) == 0:
            print("[VLA_V1][PLANNER] ERROR: planner control requires spawned objects to grasp.")
            print("[VLA_V1][PLANNER] Ensure you run WITHOUT --no-objects and WITH --num-objects >= 1.")
            simulation_app.close()
            return 2

        cfg_dir = str((_Path(__file__).resolve().parents[2] / "motion_generation" / "planners" / "planners_config").resolve())
        planner = create_planner(
            str(getattr(args, "planner", "curobo_v2")),
            ctx=PlannerContext(
                base_frame="base_link",
                ee_link_name=str(getattr(args, "ee_link", "j2n6s300_end_effector")),
                urdf_path=str((_Path(cfg_dir) / "cuRobo" / "kinovaJacoJ2N6S300.urdf").resolve()),
                config_dir=cfg_dir,
            ),
        )
        print(f"[VLA_V1][PLANNER] Control enabled. planner={getattr(args, 'planner', 'curobo_v2')} cfg_dir={cfg_dir}")
        grasp_provider = ObbGraspPoseProvider(align_to_min_width=True)

        robot_prim_path: Optional[str] = None
        try:
            robot_prim_path = str(getattr(getattr(robot, "cfg", None), "prim_path", None))
        except Exception:
            robot_prim_path = None

        agent = MotionGenerationAgent(
            sim=sim,
            robot=robot,
            controller=controller,
            planner=planner,
            grasp_provider=grasp_provider,
            loader=loader,  # type: ignore[name-defined]
            robot_prim_path=robot_prim_path,
        )

        dt_local = float(sim.get_physics_dt())
        # Ensure waypoint tolerance is not larger than the close gate; otherwise we can end up
        # "finishing" the last waypoint while still being too far to close, causing a hover loop.
        _close_gate_m = float(getattr(args, "close_if_within_m", 0.005))
        _wp_tol_m = float(getattr(args, "tolerance", 0.005))
        _wp_tol_m = float(min(_wp_tol_m, _close_gate_m))
        wp = WaypointFollowerInput(
            step_pos_m=float(ctrl_cfg.linear_speed_mps) * dt_local,
            tol_m=float(_wp_tol_m),
            max_steps_per_waypoint=int(getattr(args, "wp_max_steps_per_waypoint", 2400)),
            stagnation_steps=int(getattr(args, "wp_stagnation_steps", 10**9)),
            stagnation_eps_m=float(getattr(args, "wp_stagnation_eps_m", 5e-4)),
            device=str(sim.device),
        )
        mux_input.set_base(wp)

        # Cache gripper joint ids for "empty close" detection (did we actually pinch something?)
        _gripper_joint_ids: list[int] = []
        _gripper_close_pos: float = float(getattr(controller.config, "gripper_close_pos", 1.2))
        try:
            gids, _ = robot.find_joints(getattr(controller.config, "gripper_joint_regex", ".*_joint_finger_.*|.*_joint_finger_tip_.*"))
            if hasattr(gids, "view"):
                _gripper_joint_ids = [int(v) for v in gids.view(-1).tolist()]
            else:
                _gripper_joint_ids = [int(v) for v in list(gids)]
        except Exception:
            _gripper_joint_ids = []

        # Cache arm joint ids/names for cuRobo start_state construction.
        _arm_joint_ids = None
        _arm_joint_names: list[str] = []
        try:
            arm_joint_ids, _ = robot.find_joints(controller.config.arm_joint_regex)
            # Normalize tensor/list to python ints
            if hasattr(arm_joint_ids, "view"):
                _arm_joint_ids = [int(v) for v in arm_joint_ids.view(-1).tolist()]
            else:
                _arm_joint_ids = [int(v) for v in list(arm_joint_ids)]
            _arm_joint_names = [str(robot.data.joint_names[i]) for i in _arm_joint_ids]
        except Exception:
            _arm_joint_ids = None
            _arm_joint_names = []

        vla_planner_state = {
            "stage": "init_open",  # init_open -> approach -> close -> lift -> done
            "target_prim": None,
            "lift_pt": None,
            "open_left": int(getattr(args, "gripper_open_steps", 10)),
            "close_left": int(getattr(args, "gripper_close_steps", 60)),
            "stabilize_left": int(getattr(args, "stabilize_steps", 300)),
            "last_plan_step": -10**9,
        }
    controller.set_input_provider(mux_input)

    # Tracker + logger (will be recreated per episode in planner mode)
    tracker = ObjectsTracker(prim_paths=spawned_paths)
    tick_cfg = TickLoggingConfig(
        log_rate_hz=int(getattr(args, "log_rate_hz", 10)),
        workspace_min=getattr(controller.config.safety_cfg, "workspace_min", None),
        workspace_max=getattr(controller.config.safety_cfg, "workspace_max", None),
        ee_link_name=str(getattr(args, "ee_link", "j2n6s300_end_effector")),
        arm_joint_regex=controller.config.arm_joint_regex,
        log_joint_data=True,  # Enable joint positions/velocities for VLA training
    )
    # Create initial logger (for non-planner modes)
    # In planner mode, a new logger will be created for each episode
    session_logger = SessionLogWriter(root=Path(str(getattr(args, "logs_root", "logs/data_collection"))))

    images_dir = session_logger.root / "images"
    # Only create the images directory if cameras are enabled (avoid filesystem churn in non-camera runs).
    if camera_sensor is not None:
        images_dir.mkdir(exist_ok=True)
    image_format = getattr(args, "image_format", "png")

    # Write metadata only for non-planner modes (planner mode writes per episode)
    if control_mode != "planner":
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

    print(f"[VLA_V1] Data collection started!")
    print(f"[VLA_V1] Session directory: {session_logger.root}")
    print(f"[VLA_V1] Images directory: {images_dir}")
    print(f"[VLA_V1] Logging rate: {tick_cfg.log_rate_hz} Hz")
    print(f"[VLA_V1] Duration: {getattr(args, 'duration_s', 30.0)} seconds")
    print(f"[VLA_V1] Control mode: {getattr(args, 'control', 'keyboard')}")
    print(f"[VLA_V1] Camera sensor: {'Active' if camera_sensor is not None else 'Inactive'}")

    # Run (episodes)
    dt = float(sim.get_physics_dt())
    period = 1.0 / float(tick_cfg.log_rate_hz)
    t0 = time.time()
    images_captured = 0
    num_episodes = int(getattr(args, "num_episodes", 10))
    max_steps_ep = int(getattr(args, "max_steps_per_episode", 3000))
    render_rate_hz = float(getattr(args, "render_rate_hz", 60.0))
    render_stride = max(1, int(round((1.0 / max(1e-9, render_rate_hz)) / max(1e-9, dt))))

    # Keyboard/idle mode should behave like a normal interactive demo: no episode resets.
    if control_mode != "planner":
        duration_s = float(getattr(args, "duration_s", 30.0))
        accum = 0.0
        steps = 0
        while simulation_app.is_running() and (time.time() - t0) < duration_s:
            steps += 1

            controller.step(robot, dt)

            # Rendering throttling:
            # - In interactive mode, rendering every physics step (e.g., 240Hz) can tank FPS and cause
            #   lots of GPU/CPU sync with low utilization.
            # - Always render on tick boundaries (camera/logging), otherwise render at render_rate_hz.
            do_tick = (accum + dt + 1e-9) >= period
            do_render = bool(do_tick) or (steps % render_stride == 0)
            sim.step(render=bool(do_render))
            robot.update(dt)

            if camera_sensor is not None:
                try:
                    if hasattr(camera_sensor, "update"):
                        # Only update camera when we render (otherwise the render product may not advance).
                        if bool(do_render):
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
                            images_captured += 1
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
    else:
        # Planner mode runs episode loops (reset + respawn) and ends each episode after lift.
        # Create a session folder with timestamp, episodes will be subfolders inside
        logs_root = Path(str(getattr(args, "logs_root", "logs/data_collection")))
        from datetime import datetime
        session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_folder = logs_root / f"session_{session_timestamp}"
        session_folder.mkdir(parents=True, exist_ok=True)
        
        total_ticks = 0
        total_images = 0
        # Persist target choice across episodes so we can enforce "different target after reset"
        # when multiple objects exist.
        prev_target_prim: str | None = None
        
        # Keep a per-cycle object layout so episodes 0..N-1 share poses, then we reshuffle.
        cycle_object_poses: Optional[list[tuple[tuple[float, float, float], float]]] = None

        for ep in range(num_episodes):
            if not simulation_app.is_running():
                print(f"[VLA_V1][EP] Simulation stopped, ending at episode {ep}/{num_episodes}")
                break

            print(f"[VLA_V1][EP] Starting episode {ep+1}/{num_episodes}")
            
            try:
                # Create a new session logger for each episode (separate log files)
                # Episode folders go inside the session folder
                session_logger = SessionLogWriter(root=session_folder, session_name=f"episode_{ep:04d}")
                images_dir = session_logger.root / "images"
                if camera_sensor is not None:
                    images_dir.mkdir(exist_ok=True)
                image_format = getattr(args, "image_format", "png")
                images_captured_episode = 0
                
                # Write metadata for this episode
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

                # Reset sim/robot for a clean episode start.
                _reset_sim_and_robot()
                if camera_sensor is not None:
                    try:
                        camera_sensor.reset()
                    except Exception:
                        pass

                # IMPORTANT: PhysX views are not reliably populated immediately after `sim.reset()`.
                # If we try to create rigid-body views and teleport *before* the first post-reset step,
                # the view can be empty and the respawn becomes a silent no-op (objects never move).
                # Stepping once here ensures rigid bodies are registered before we respawn/teleport.
                try:
                    sim.step(render=False)
                    robot.update(dt)
                except Exception:
                    pass

                # Reset object poses each episode, but only *generate new random poses* every N episodes.
                do_respawn = bool(getattr(args, "respawn_each_episode", False)) or (
                    control_mode == "planner" and (not bool(getattr(args, "no_respawn_each_episode", False)))
                )
                if do_respawn:
                    # Default behavior in planner mode: respawn once per full target cycle (num objects).
                    # This keeps objects fixed while we attempt each object once, then reshuffles.
                    respawn_every = 1 if bool(getattr(args, "respawn_each_episode", False)) else int(
                        getattr(args, "respawn_every_n_episodes", 0)
                    )
                    if respawn_every <= 0:
                        respawn_every = max(1, int(len(spawned_paths)))
                    regenerate = (cycle_object_poses is None) or ((int(ep) % int(respawn_every)) == 0)
                    if regenerate:
                        print(
                            f"[VLA_V1][RESPAWN] ep={int(ep)} regenerate=True respawn_every={int(respawn_every)} n_objects={len(spawned_paths)}"
                        )
                        new_poses = _rerandomize_object_poses(spawned_paths)
                        if new_poses is not None:
                            cycle_object_poses = new_poses
                            # Log/print intended new positions for easy verification.
                            try:
                                pose_xy = {
                                    str(p).split("/")[-1]: [round(float(pos[0]), 4), round(float(pos[1]), 4)]
                                    for p, (pos, _yaw) in zip(spawned_paths, new_poses)
                                }
                                print(f"[VLA_V1][RESPAWN] intended_xy={pose_xy}")
                                session_logger.log_event(
                                    "object_respawn",
                                    {
                                        "episode_idx": int(ep),
                                        "respawn_every": int(respawn_every),
                                        "intended_xy": pose_xy,
                                    },
                                )
                            except Exception:
                                pass
                            # Best-effort verification: read back current PhysX poses after one step.
                            try:
                                sim.step(render=False)
                                robot.update(dt)
                                _trk = ObjectsTracker(prim_paths=spawned_paths)
                                snap = _trk.snapshot()
                                actual_xy = {
                                    str(o.id): [round(float(o.pose.position_m[0]), 4), round(float(o.pose.position_m[1]), 4)]
                                    for o in snap
                                }
                                print(f"[VLA_V1][RESPAWN] actual_xy={actual_xy}")
                                try:
                                    session_logger.log_event(
                                        "object_respawn_actual",
                                        {
                                            "episode_idx": int(ep),
                                            "respawn_every": int(respawn_every),
                                            "actual_xy": actual_xy,
                                        },
                                    )
                                except Exception:
                                    pass
                            except Exception:
                                pass
                        else:
                            print(
                                "[VLA_V1][RESPAWN][WARN] Respawn requested but teleport failed (no poses applied). "
                                "Objects may remain at their original locations."
                            )
                    else:
                        _rerandomize_object_poses(spawned_paths, poses=cycle_object_poses)

                # HARD reset controller + follower per episode to avoid leftover impulses after sim.reset()
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
                    wp.set_current_pose_b(get_ee_pos_base_frame(robot, str(getattr(args, "ee_link", "j2n6s300_end_effector"))))
                except Exception:
                    pass

                # Let physics settle after reset/respawn before we select target / plan.
                settle_steps = int(getattr(args, "settle_steps", 0))
                if settle_steps > 0:
                    for _ in range(settle_steps):
                        if not simulation_app.is_running():
                            break
                        # Keep the arm stable while the scene settles
                        try:
                            controller.step(robot, dt)
                        except Exception:
                            pass
                        sim.step(render=False)
                        robot.update(dt)

                # Apply domain randomization once per episode (after reset/respawn/settle).
                try:
                    _apply_domain_randomization(ep_idx=int(ep), logger=session_logger)
                    if camera_sensor is not None:
                        try:
                            camera_sensor.reset()
                        except Exception:
                            pass
                except Exception:
                    pass

                # Re-init per-episode follower state to avoid leftover queued waypoints/gripper commands.
                wp.set_waypoints_b([])
                try:
                    setattr(wp, "_gripper_queue", [])  # type: ignore[attr-defined]
                except Exception:
                    pass

                # IMPORTANT: recreate the object tracker per episode.
                # The underlying PhysX rigid-body views can become stale across `sim.reset()`, which
                # breaks lift success detection (target_z stays constant after episode 0).
                try:
                    tracker = ObjectsTracker(prim_paths=spawned_paths)
                except Exception:
                    # Keep the previous tracker as a fallback.
                    pass

                obj_z_by_leaf: dict[str, float] = {}
                try:
                    for o in tracker.snapshot():
                        try:
                            obj_z_by_leaf[str(o.id)] = float(o.pose.position_m[2])
                        except Exception:
                            continue
                except Exception:
                    obj_z_by_leaf = {}
                # Select target once per episode.
                _tgt = _select_episode_target_prim(ep, object_z_by_leaf=obj_z_by_leaf, prev_target_prim=prev_target_prim)
                # Episode-level language command for VLA training (stored once per episode).
                lang_cmd, lang_meta = ("", {})
                try:
                    if _tgt is not None:
                        lang_cmd, lang_meta = _make_language_command(ep_idx=int(ep), target_prim=str(_tgt))
                except Exception:
                    lang_cmd, lang_meta = ("", {})

                # Baseline Z for success detection (more robust than relying on table_z alone).
                z0 = None
                try:
                    if _tgt is not None:
                        leaf = str(_tgt).split("/")[-1]
                        z0 = obj_z_by_leaf.get(str(leaf), None)
                except Exception:
                    z0 = None

                vla_planner_state.update(
                    {
                        "stage": "init_open",
                        "target_prim": _tgt,
                        "target_z0_w": z0,
                        "language_command": lang_cmd,
                        "language_command_meta": lang_meta,
                        "lift_pt": None,
                        "approach_goal_b": None,
                        "grasp_goal_b": None,
                        "attempt_idx": 0,
                        "open_left": int(getattr(args, "gripper_open_steps", 10)),
                        "close_left": int(getattr(args, "gripper_close_steps", 60)),
                        "stabilize_left": int(getattr(args, "stabilize_steps", 300)),
                        # IMPORTANT: `steps` is reset to 0 every episode. If we keep a previous episode's
                        # `last_plan_step`, replanning can be throttled for the entire next episode
                        # (because `steps - last_plan_step` stays negative and thus < cooldown).
                        "last_plan_step": -10**9,
                        "hold_after_close_left": int(getattr(args, "hold_after_close_steps", 30)),
                        "post_lift_hold_left": int(getattr(args, "post_lift_hold_steps", 30)),
                        "lift_success_streak": 0,
                        "open_queued": False,
                        "close_queued": False,
                        "lift_queued": False,
                        "hold_requeue_count": 0,
                        "approach_last_dist_m": None,
                        "approach_stall_steps": 0,
                        "approach_start_step": None,
                    }
                )

                # Persist the episode instruction in the episode folder for easy dataset building.
                try:
                    if lang_cmd:
                        import json as _json
                        from datetime import datetime as _dt

                        (session_logger.root / "instruction.json").write_text(
                            _json.dumps(
                                {
                                    "episode_idx": int(ep),
                                    "target_prim": str(_tgt),
                                    "target_label": _prim_label(str(_tgt or "")),
                                    "language_command": str(lang_cmd),
                                    "language_command_meta": lang_meta,
                                    "created_at": _dt.now().isoformat(timespec="seconds"),
                                },
                                indent=2,
                            )
                        )
                except Exception:
                    pass
                # Update for next episode
                try:
                    prev_target_prim = str(vla_planner_state.get("target_prim", "") or "") or None
                except Exception:
                    prev_target_prim = None
                # Print basic episode header to terminal (helps debug "repeating" confusion).
                try:
                    tgt = vla_planner_state.get("target_prim", None)
                    candidates = sorted([str(p) for p in spawned_paths])
                    print(
                        f"[VLA_V1][EP] start ep={ep} target={tgt} "
                        f"label='{_prim_label(str(tgt or ''))}' n_objects={len(spawned_paths)} "
                        f"candidates={[c.split('/')[-1] for c in candidates]}"
                    )
                except Exception:
                    print(
                        f"[VLA_V1][EP] start ep={ep} target={vla_planner_state.get('target_prim')} "
                        f"label='{_prim_label(str(vla_planner_state.get('target_prim', '')))}'"
                    )
                session_logger.log_event(
                    "episode_start",
                    {
                        "episode_idx": int(ep),
                        "num_objects": int(len(spawned_paths)),
                        "target_prim": str(vla_planner_state.get("target_prim", None)),
                        "target_label": _prim_label(str(vla_planner_state.get("target_prim", ""))),
                        "language_command": str(vla_planner_state.get("language_command", "")),
                        "language_command_meta": vla_planner_state.get("language_command_meta", {}),
                    },
                )

                accum = 0.0
                steps = 0
                while simulation_app.is_running() and steps < max_steps_ep:
                    steps += 1

                    # Planner-driven reach-to-grasp state machine
                    try:
                        wp.set_current_pose_b(get_ee_pos_base_frame(robot, str(getattr(args, "ee_link", "j2n6s300_end_effector"))))
                    except Exception:
                        pass

                    try:
                        stab_left = int(vla_planner_state.get("stabilize_left", 0))
                        if stab_left > 0:
                            vla_planner_state["stabilize_left"] = stab_left - 1
                            # Skip state machine when stabilizing - just hold position
                        else:
                            stage = str(vla_planner_state.get("stage", "idle"))

                            if stage == "init_open":
                                open_left = int(vla_planner_state.get("open_left", 0))
                                if not vla_planner_state.get("open_queued", False):
                                    session_logger.log_event("action_start", {"action": "GRIPPER_OPEN", "steps": int(open_left), "episode_idx": int(ep)})
                                    try:
                                        controller.set_mode("gripper")
                                        wp.queue_gripper(+1.0, steps=open_left)
                                    except Exception:
                                        pass
                                    vla_planner_state["open_queued"] = True
                                controller.set_mode("gripper")
                                if open_left > 0:
                                    vla_planner_state["open_left"] = open_left - 1
                                else:
                                    session_logger.log_event("action_end", {"action": "GRIPPER_OPEN", "episode_idx": int(ep)})
                                    vla_planner_state["stage"] = "idle"
                                    # Ensure mode is set to translate for planning/approach
                                    controller.set_mode("translate")

                            if str(vla_planner_state.get("stage", "idle")) == "idle":
                                if len(spawned_paths) != 0:
                                    # Replan throttle: avoid spamming plan calls if execution is stalled.
                                    should_plan = True
                                    try:
                                        cooldown = int(getattr(args, "replan_cooldown_steps", 120))
                                        last_plan = int(vla_planner_state.get("last_plan_step", -10**9))
                                        if (steps - last_plan) < cooldown:
                                            should_plan = False
                                    except Exception:
                                        # If anything goes wrong, don't block planning.
                                        should_plan = True

                                    if not should_plan:
                                        # Hold position and wait for cooldown.
                                        pass
                                    else:
                                        # IMPORTANT: target is selected ONCE per episode and must remain stable.
                                        # Do not re-select per tick (that causes "Obj_01 always" bugs + inconsistent logs).
                                        target_prim = vla_planner_state.get("target_prim", None)
                                        if not target_prim:
                                            raise RuntimeError("[VLA_V1][PLANNER] No target prim available for this episode.")

                                        # Adaptive depth: if retries happen, go slightly deeper (more negative) per attempt.
                                        attempt_idx = int(vla_planner_state.get("attempt_idx", 0))
                                        base_depth = float(getattr(args, "grasp_depth", -0.04))
                                        depth_step = float(getattr(args, "grasp_depth_step", -0.02))
                                        grasp_depth_eff = base_depth + float(attempt_idx) * depth_step

                                        _pos_w, _quat_wxyz_w, pos_b, quat_b = agent.compute_current_grasp_for_prim(str(target_prim))
                                        # Cache goals for gating and success checks.
                                        # NOTE: pos_b from grasp provider is (x,y,top_z) in base frame.
                                        gx, gy, gz = float(pos_b[0]), float(pos_b[1]), float(pos_b[2])
                                        ee_z_offset = float(getattr(args, "ee_z_offset_m", 0.0))
                                        # Convert a "contact point" on the object into an EE-link target by shifting upward.
                                        # This prevents commanding the wrist/palm inside the object volume.
                                        gz_ee = gz + ee_z_offset
                                        # Clamp grasp depth so the requested grasp Z is reachable under workspace_min_z.
                                        try:
                                            z_floor = float(getattr(ctrl_cfg.safety_cfg, "workspace_min", (None, None, 0.0))[2] or 0.0)
                                            min_grasp_z = z_floor + 0.005  # small clearance above floor
                                            desired_z = gz_ee + float(grasp_depth_eff)
                                            if desired_z < min_grasp_z:
                                                old = float(grasp_depth_eff)
                                                grasp_depth_eff = float(min_grasp_z - gz_ee)
                                                session_logger.log_event(
                                                    "debug",
                                                    {
                                                        "episode_idx": int(ep),
                                                        "event": "CLAMP_GRASP_DEPTH",
                                                        "gz": float(gz),
                                                        "gz_ee": float(gz_ee),
                                                        "desired_grasp_z": float(desired_z),
                                                        "min_grasp_z": float(min_grasp_z),
                                                        "grasp_depth_old": float(old),
                                                        "grasp_depth_new": float(grasp_depth_eff),
                                                    },
                                                )
                                        except Exception:
                                            pass
                                        grasp_goal_b = (
                                            gx,
                                            gy,
                                            gz_ee + float(grasp_depth_eff),
                                        )
                                        vla_planner_state["grasp_goal_b"] = grasp_goal_b

                                        # Provide cuRobo with a proper start_state (required by some MotionGen builds).
                                        try:
                                            if hasattr(planner, "set_start_state") and _arm_joint_ids is not None and len(_arm_joint_ids) > 0:
                                                q = robot.data.joint_pos[0, _arm_joint_ids].detach().to("cpu").tolist()
                                                planner.set_start_state(joint_pos=[float(v) for v in q], joint_names=_arm_joint_names)
                                        except Exception:
                                            pass

                                        # Sync cuRobo world from scene before planning.
                                        ok = False
                                        obstacle_prims = [str(p) for p in list(spawned_paths) if str(p) != str(target_prim)]
                                        use_world = bool(getattr(args, "curobo_world_from_scene", True)) and (not bool(getattr(args, "no_curobo_world_from_scene", False)))
                                        if bool(use_world):
                                            if bool(getattr(args, "curobo_world_include_table", True)):
                                                obstacle_prims = ["/World/Origin1/Table"] + obstacle_prims
                                            try:
                                                if hasattr(planner, "update_world_from_prim_paths"):
                                                    ok = bool(planner.update_world_from_prim_paths(sim=sim, robot=robot, prim_paths=obstacle_prims))
                                            except Exception:
                                                ok = False
                                            session_logger.log_event(
                                                "action_start",
                                                {"action": "CUROBO_UPDATE_WORLD", "ok": bool(ok), "n_prims": int(len(obstacle_prims)), "episode_idx": int(ep)},
                                            )
                                        if ep == 0 and steps < 5:
                                            print(f"[VLA_V1][CUROBO] update_world_from_scene ok={ok} n_prims={len(obstacle_prims)}")

                                        session_logger.log_event(
                                            "action_start",
                                            {
                                                "action": "PLAN_TO_PREGRASP",
                                                "planner": str(getattr(args, "planner", "curobo_v2")),
                                                "target_prim": str(target_prim),
                                                "episode_idx": int(ep),
                                            },
                                        )
                                        waypoints = planner.plan_to_pose_b(
                                            target_pos_b=(gx, gy, gz_ee),
                                            target_quat_b_wxyz=None if bool(getattr(args, "ignore_grasp_orientation", True)) else quat_b,
                                            pregrasp_offset_m=float(getattr(args, "pregrasp", 0.10)),
                                            grasp_depth_m=float(grasp_depth_eff),
                                            lift_height_m=float(getattr(args, "lift", 0.15)),
                                        )
                                        vla_planner_state["last_plan_step"] = int(steps)
                                        session_logger.log_event(
                                            "action_end",
                                            {"action": "PLAN_TO_PREGRASP", "n_waypoints": int(len(waypoints)), "episode_idx": int(ep)},
                                        )

                                        if len(waypoints) >= 2:
                                            lift_pt = waypoints[-1]
                                            approach_pts = waypoints[:-1]
                                        else:
                                            lift_pt = (
                                                float(pos_b[0]),
                                                float(pos_b[1]),
                                                float(pos_b[2] + float(getattr(args, "lift", 0.15))),
                                            )
                                            approach_pts = waypoints

                                        vla_planner_state["lift_pt"] = lift_pt
                                        try:
                                            if len(approach_pts) > 0:
                                                vla_planner_state["approach_goal_b"] = tuple(map(float, approach_pts[-1]))
                                            else:
                                                vla_planner_state["approach_goal_b"] = tuple(map(float, grasp_goal_b))
                                        except Exception:
                                            vla_planner_state["approach_goal_b"] = tuple(map(float, grasp_goal_b))

                                        vla_planner_state["stage"] = "approach"
                                        vla_planner_state["lift_queued"] = False
                                        vla_planner_state["approach_start_step"] = int(steps)

                                        # Densify for smoother execution (smaller per-step corrections => less jitter).
                                        max_seg = float(getattr(args, "planner_waypoint_max_seg_m", 0.02))
                                        approach_pts_dense = _densify_waypoints_b(
                                            [(float(x), float(y), float(z)) for (x, y, z) in approach_pts],
                                            max_seg_m=max_seg,
                                        )

                                        controller.set_mode("translate")
                                        session_logger.log_event(
                                            "action_start",
                                            {"action": "EXECUTE_WAYPOINTS", "n_waypoints": int(len(approach_pts_dense)), "episode_idx": int(ep)},
                                        )
                                        wp.set_waypoints_b(approach_pts_dense)

                            if str(vla_planner_state.get("stage", "")) == "approach":
                                # Absolute time-based escape hatch: prevent infinite hover in approach.
                                # IMPORTANT: do NOT close in mid-air when far from the grasp goal (that causes
                                # "grazing" + empty lifts). Instead, treat it as a failed attempt and retry/restart.
                                try:
                                    start_step = vla_planner_state.get("approach_start_step", None)
                                    if start_step is not None:
                                        force_after = int(getattr(args, "force_close_after_approach_steps", 2400))
                                        if int(steps) - int(start_step) >= force_after:
                                            # Compute distance to the final approach goal (if available). If we don't know
                                            # the distance, we should also avoid closing.
                                            goal = vla_planner_state.get("approach_goal_b", None)
                                            dist_m_timeout = None
                                            try:
                                                if goal is not None:
                                                    ee_pos_b = get_ee_pos_base_frame(
                                                        robot, str(getattr(args, "ee_link", "j2n6s300_end_effector"))
                                                    )
                                                    if ee_pos_b is not None:
                                                        dx = float(ee_pos_b[0]) - float(goal[0])
                                                        dy = float(ee_pos_b[1]) - float(goal[1])
                                                        dz = float(ee_pos_b[2]) - float(goal[2])
                                                        dist_m_timeout = (dx * dx + dy * dy + dz * dz) ** 0.5
                                            except Exception:
                                                dist_m_timeout = None
                                            close_within_m = float(getattr(args, "close_if_within_m", 0.015))
                                            session_logger.log_event(
                                                "action_start",
                                                {
                                                    "action": "APPROACH_TIMEOUT",
                                                    "steps_in_approach": int(steps) - int(start_step),
                                                    "dist_m": dist_m_timeout,
                                                    "close_within_m": float(close_within_m),
                                                    "episode_idx": int(ep),
                                                },
                                            )
                                            try:
                                                wp.set_waypoints_b([])
                                            except Exception:
                                                pass
                                            if dist_m_timeout is not None and dist_m_timeout <= close_within_m:
                                                # We are close enough: proceed to close.
                                                vla_planner_state["stage"] = "close"
                                                vla_planner_state["close_queued"] = False
                                                vla_planner_state["close_left"] = int(getattr(args, "gripper_close_steps", 60))
                                            else:
                                                # Too far (or unknown distance): count as a failed attempt.
                                                attempt = int(vla_planner_state.get("attempt_idx", 0)) + 1
                                                vla_planner_state["attempt_idx"] = attempt
                                                session_logger.log_event(
                                                    "grasp_result",
                                                    {
                                                        "episode_idx": int(ep),
                                                        "ok": False,
                                                        "reason": "approach_timeout",
                                                        "attempt_idx": int(attempt),
                                                        "dist_m": dist_m_timeout,
                                                    },
                                                )
                                                if attempt < int(getattr(args, "max_grasp_attempts", 3)):
                                                    vla_planner_state["stage"] = "init_open"
                                                    vla_planner_state["open_left"] = int(getattr(args, "gripper_open_steps", 10))
                                                    vla_planner_state["open_queued"] = False
                                                    vla_planner_state["close_queued"] = False
                                                    vla_planner_state["lift_queued"] = False
                                                    vla_planner_state["stabilize_left"] = int(getattr(args, "stabilize_steps", 300))
                                                    vla_planner_state["approach_last_dist_m"] = None
                                                    vla_planner_state["approach_stall_steps"] = 0
                                                    vla_planner_state["approach_start_step"] = None
                                                    vla_planner_state["last_plan_step"] = -10**9
                                                    try:
                                                        controller.set_mode("gripper")
                                                    except Exception:
                                                        pass
                                                else:
                                                    vla_planner_state["stage"] = "done"
                                except Exception:
                                    pass

                                # If we switched out of "approach" due to timeout above, skip the remaining
                                # approach handlers for this step (prevents clobbering stage transitions).
                                if str(vla_planner_state.get("stage", "")) == "approach":
                                    # If we're stuck pressing into the object (contact-stall), we may never reach the exact
                                    # grasp waypoint. Detect lack of progress and force a close so we don't "sleep" forever.
                                    try:
                                        goal = vla_planner_state.get("approach_goal_b", None)
                                        ee_pos_b = get_ee_pos_base_frame(robot, str(getattr(args, "ee_link", "j2n6s300_end_effector")))
                                        if goal is not None and ee_pos_b is not None:
                                            dx = float(ee_pos_b[0]) - float(goal[0])
                                            dy = float(ee_pos_b[1]) - float(goal[1])
                                            dz = float(ee_pos_b[2]) - float(goal[2])
                                            dist_m_now = (dx * dx + dy * dy + dz * dz) ** 0.5
                                            # Defaults so downstream checks are always well-defined.
                                            in_grace = False
                                            stall_steps = 0
                                            stall_thresh = 10**9
                                            close_within_m = float(getattr(args, "close_if_within_m", 0.015))
                                            if dist_m_now <= close_within_m:
                                                # Close only when we are genuinely near the grasp goal.
                                                try:
                                                    wp.set_waypoints_b([])
                                                except Exception:
                                                    pass
                                                vla_planner_state["stage"] = "close"
                                                vla_planner_state["close_queued"] = False
                                                vla_planner_state["close_left"] = int(getattr(args, "gripper_close_steps", 60))
                                            else:
                                                # Stall detection (with a grace period) to avoid prematurely resetting
                                                # at the start of an approach (especially at slower execution speeds).
                                                in_grace = False
                                                try:
                                                    grace = int(getattr(args, "approach_stall_grace_steps", 0))
                                                    start_step = vla_planner_state.get("approach_start_step", None)
                                                    if start_step is not None and grace > 0:
                                                        in_grace = (int(steps) - int(start_step)) < grace
                                                except Exception:
                                                    in_grace = False

                                                if in_grace:
                                                    vla_planner_state["approach_stall_steps"] = 0
                                                    vla_planner_state["approach_last_dist_m"] = float(dist_m_now)
                                                else:
                                                    last = vla_planner_state.get("approach_last_dist_m", None)
                                                    eps = float(getattr(args, "approach_stall_eps_m", 2e-4))
                                                    if last is not None and dist_m_now > (float(last) - eps):
                                                        vla_planner_state["approach_stall_steps"] = int(vla_planner_state.get("approach_stall_steps", 0)) + 1
                                                    else:
                                                        vla_planner_state["approach_stall_steps"] = 0
                                                    vla_planner_state["approach_last_dist_m"] = float(dist_m_now)

                                                stall_steps = int(vla_planner_state.get("approach_stall_steps", 0))
                                                stall_thresh = int(getattr(args, "approach_stall_steps_to_close", 900))

                                            if (str(vla_planner_state.get("stage", "")) == "approach") and (not in_grace) and (stall_steps >= stall_thresh):
                                                # If we are not making progress AND we are not close enough to safely close,
                                                # treat this as a failed attempt (instead of closing in mid-air).
                                                session_logger.log_event(
                                                    "action_start",
                                                    {
                                                        "action": "APPROACH_STALL",
                                                        "dist_m": float(dist_m_now),
                                                        "stall_steps": int(stall_steps),
                                                        "stall_thresh": int(stall_thresh),
                                                        "episode_idx": int(ep),
                                                    },
                                                )
                                                attempt = int(vla_planner_state.get("attempt_idx", 0)) + 1
                                                vla_planner_state["attempt_idx"] = attempt
                                                session_logger.log_event(
                                                    "grasp_result",
                                                    {
                                                        "episode_idx": int(ep),
                                                        "ok": False,
                                                        "reason": "approach_stall",
                                                        "attempt_idx": int(attempt),
                                                        "dist_m": float(dist_m_now),
                                                        "stall_steps": int(stall_steps),
                                                    },
                                                )
                                                try:
                                                    wp.set_waypoints_b([])
                                                except Exception:
                                                    pass
                                                if attempt < int(getattr(args, "max_grasp_attempts", 3)):
                                                    vla_planner_state["stage"] = "init_open"
                                                    vla_planner_state["open_left"] = int(getattr(args, "gripper_open_steps", 10))
                                                    vla_planner_state["open_queued"] = False
                                                    vla_planner_state["close_queued"] = False
                                                    vla_planner_state["lift_queued"] = False
                                                    vla_planner_state["stabilize_left"] = int(getattr(args, "stabilize_steps", 300))
                                                    vla_planner_state["approach_last_dist_m"] = None
                                                    vla_planner_state["approach_stall_steps"] = 0
                                                    vla_planner_state["approach_start_step"] = None
                                                    vla_planner_state["last_plan_step"] = -10**9
                                                    try:
                                                        controller.set_mode("gripper")
                                                    except Exception:
                                                        pass
                                                else:
                                                    vla_planner_state["stage"] = "done"
                                    except Exception:
                                        pass

                                if str(vla_planner_state.get("stage", "")) == "approach" and len(getattr(wp, "_waypoints_b", [])) == 0:
                                    session_logger.log_event("action_end", {"action": "EXECUTE_WAYPOINTS", "episode_idx": int(ep)})
                                    # Only close if we are actually near the final approach goal; otherwise replan.
                                    ee_pos_b = None
                                    try:
                                        ee_pos_b = get_ee_pos_base_frame(robot, str(getattr(args, "ee_link", "j2n6s300_end_effector")))
                                    except Exception:
                                        ee_pos_b = None
                                    goal = vla_planner_state.get("approach_goal_b", None)
                                    replan_far_m = float(getattr(args, "replan_if_far_m", 0.03))
                                    close_within_m = float(getattr(args, "close_if_within_m", 0.015))
                                    dist_m = None
                                    try:
                                        if ee_pos_b is not None and goal is not None:
                                            dx = float(ee_pos_b[0]) - float(goal[0])
                                            dy = float(ee_pos_b[1]) - float(goal[1])
                                            dz = float(ee_pos_b[2]) - float(goal[2])
                                            dist_m = (dx * dx + dy * dy + dz * dz) ** 0.5
                                    except Exception:
                                        dist_m = None
                                    if dist_m is None:
                                        # Conservative: never close if we don't know we are close enough.
                                        session_logger.log_event(
                                            "action_start",
                                            {"action": "NO_DISTANCE_TO_GOAL", "episode_idx": int(ep)},
                                        )
                                        try:
                                            if goal is not None:
                                                wp.set_waypoints_b([(float(goal[0]), float(goal[1]), float(goal[2]))])
                                        except Exception:
                                            pass
                                        vla_planner_state["stage"] = "approach"
                                    elif dist_m > replan_far_m:
                                        session_logger.log_event(
                                            "action_start",
                                            {"action": "REPLAN_TOO_FAR", "dist_m": float(dist_m), "thresh_m": float(replan_far_m), "episode_idx": int(ep)},
                                        )
                                        # Previously we replanned here, which causes oscillation/dancing when the EE can't
                                        # satisfy the exact waypoint. Instead, keep trying to converge to the goal.
                                        try:
                                            if goal is not None:
                                                wp.set_waypoints_b([(float(goal[0]), float(goal[1]), float(goal[2]))])
                                        except Exception:
                                            pass
                                        vla_planner_state["stage"] = "approach"
                                    elif dist_m > close_within_m:
                                        session_logger.log_event(
                                            "action_start",
                                            {"action": "HOLD_TOO_FAR_TO_CLOSE", "dist_m": float(dist_m), "thresh_m": float(close_within_m), "episode_idx": int(ep)},
                                        )
                                        # We are close, but not close enough to safely close yet.
                                        # IMPORTANT: if we simply stay in "approach" with an empty waypoint list, the arm
                                        # will just hold position forever (looks like it's "sleeping" on top of the object).
                                        # Re-queue the final approach goal so execution keeps trying to converge.
                                        try:
                                            if goal is not None:
                                                gx, gy, gz = float(goal[0]), float(goal[1]), float(goal[2])
                                                wp.set_waypoints_b([(gx, gy, gz)])
                                                vla_planner_state["hold_requeue_count"] = int(vla_planner_state.get("hold_requeue_count", 0)) + 1
                                        except Exception:
                                            pass
                                        vla_planner_state["stage"] = "approach"
                                    else:
                                        vla_planner_state["stage"] = "close"
                                        vla_planner_state["close_queued"] = False
                                        vla_planner_state["close_left"] = int(getattr(args, "gripper_close_steps", 60))

                            if str(vla_planner_state.get("stage", "")) == "close":
                                close_left = int(vla_planner_state.get("close_left", 0))
                                controller.set_mode("gripper")

                                if not vla_planner_state.get("close_queued", False):
                                    session_logger.log_event(
                                        "action_start",
                                        {"action": "GRIPPER_CLOSE", "steps": int(close_left), "episode_idx": int(ep)},
                                    )
                                    try:
                                        wp.queue_gripper(-1.0, steps=close_left)
                                    except Exception:
                                        pass
                                    vla_planner_state["close_queued"] = True

                                if close_left > 0:
                                    vla_planner_state["close_left"] = close_left - 1
                                else:
                                    session_logger.log_event("action_end", {"action": "GRIPPER_CLOSE", "episode_idx": int(ep)})
                                    # Empty-close heuristic: if fingers reached (almost) fully closed, likely no object was grasped.
                                    empty = False
                                    mean_pos = None
                                    try:
                                        if _gripper_joint_ids:
                                            qg = robot.data.joint_pos[0, _gripper_joint_ids].detach().to("cpu")
                                            mean_pos = float(qg.mean().item())
                                            close_pos = float(_gripper_close_pos)
                                            empty = mean_pos >= (close_pos - float(getattr(args, "empty_close_threshold", 0.06)))
                                    except Exception:
                                        empty = False

                                    if empty:
                                        session_logger.log_event(
                                            "grasp_result",
                                            {"episode_idx": int(ep), "ok": False, "reason": "empty_close", "gripper_mean_pos": mean_pos},
                                        )
                                        attempt = int(vla_planner_state.get("attempt_idx", 0)) + 1
                                        vla_planner_state["attempt_idx"] = attempt
                                        if attempt < int(getattr(args, "max_grasp_attempts", 3)):
                                            wp.set_waypoints_b([])
                                            vla_planner_state["stage"] = "init_open"
                                            vla_planner_state["open_left"] = int(getattr(args, "gripper_open_steps", 10))
                                            vla_planner_state["open_queued"] = False
                                            vla_planner_state["close_queued"] = False
                                            vla_planner_state["lift_queued"] = False
                                            vla_planner_state["stabilize_left"] = int(getattr(args, "stabilize_steps", 300))
                                            # Reset controller mode for retry
                                            controller.set_mode("gripper")
                                            # Skip lift; go retry immediately.
                                            continue

                                    vla_planner_state["stage"] = "post_close_hold"
                                    vla_planner_state["hold_after_close_left"] = int(getattr(args, "hold_after_close_steps", 30))
                                    vla_planner_state["lift_queued"] = False

                            if str(vla_planner_state.get("stage", "")) == "post_close_hold":
                                hold_left = int(vla_planner_state.get("hold_after_close_left", 0))
                                controller.set_mode("translate")
                                if hold_left > 0:
                                    vla_planner_state["hold_after_close_left"] = hold_left - 1
                                else:
                                    vla_planner_state["stage"] = "lift"

                            if str(vla_planner_state.get("stage", "")) == "lift":
                                lift_pt = vla_planner_state.get("lift_pt", None)
                                controller.set_mode("translate")

                                if lift_pt is not None and not vla_planner_state.get("lift_queued", False):
                                    session_logger.log_event(
                                        "action_start",
                                        {"action": "LIFT", "target": [float(lift_pt[0]), float(lift_pt[1]), float(lift_pt[2])], "episode_idx": int(ep)},
                                    )
                                    wp.set_waypoints_b([(float(lift_pt[0]), float(lift_pt[1]), float(lift_pt[2]))])
                                    vla_planner_state["lift_queued"] = True

                                # Early-success: if the target object is already lifted enough while we are
                                # still trying to reach/hold the lift waypoint, end the episode immediately.
                                # This prevents long "hover in air then drop" failures where we only evaluate
                                # success after the waypoint follower times out.
                                try:
                                    if vla_planner_state.get("lift_queued", False):
                                        stride = max(1, int(getattr(args, "lift_success_check_stride", 5)))
                                        if (int(steps) % stride) == 0:
                                            target_prim = str(vla_planner_state.get("target_prim", ""))
                                            z = _target_z_from_tracker(target_prim) if target_prim else None
                                            z0 = vla_planner_state.get("target_z0_w", None)
                                            thresh = float(getattr(args, "lift_success_min_dz_m", 0.06))
                                            dz_from_start = None
                                            dz_table = None
                                            lifted = False
                                            if z is not None:
                                                try:
                                                    dz_table = float(z) - float(_table_z())
                                                except Exception:
                                                    dz_table = None
                                                if z0 is not None:
                                                    try:
                                                        dz_from_start = float(z) - float(z0)
                                                    except Exception:
                                                        dz_from_start = None
                                                # Prefer delta-from-start when available (robust to object origin conventions).
                                                if dz_from_start is not None:
                                                    lifted = bool(dz_from_start >= thresh)
                                                elif dz_table is not None:
                                                    lifted = bool(dz_table >= thresh)

                                            if lifted:
                                                vla_planner_state["lift_success_streak"] = int(vla_planner_state.get("lift_success_streak", 0)) + 1
                                            else:
                                                vla_planner_state["lift_success_streak"] = 0

                                            confirm = max(1, int(getattr(args, "lift_success_confirm_steps", 10)))
                                            if int(vla_planner_state.get("lift_success_streak", 0)) >= confirm:
                                                # Clear motion immediately and end episode.
                                                try:
                                                    wp.set_waypoints_b([])
                                                except Exception:
                                                    pass
                                                session_logger.log_event(
                                                    "grasp_result",
                                                    {
                                                        "episode_idx": int(ep),
                                                        "ok": True,
                                                        "target_z": z,
                                                        "dz_above_table": dz_table,
                                                        "dz_from_start": dz_from_start,
                                                        "lift_success_thresh_m": float(thresh),
                                                        "reason": "early_lift_success",
                                                    },
                                                )
                                                # Also end the LIFT action cleanly for logs.
                                                session_logger.log_event(
                                                    "action_end",
                                                    {"action": "LIFT", "episode_idx": int(ep), "early_success": True},
                                                )
                                                vla_planner_state["stage"] = "done"
                                                # Skip the rest of lift handling this step.
                                                continue
                                except Exception:
                                    pass

                                if (
                                    lift_pt is not None
                                    and vla_planner_state.get("lift_queued", False)
                                    and len(getattr(wp, "_waypoints_b", [])) == 0
                                ):
                                    session_logger.log_event("action_end", {"action": "LIFT", "episode_idx": int(ep)})
                                    # Post-lift hold: let the object catch up before we evaluate success.
                                    vla_planner_state["stage"] = "post_lift_hold"
                                    vla_planner_state["post_lift_hold_left"] = int(getattr(args, "post_lift_hold_steps", 30))

                            if str(vla_planner_state.get("stage", "")) == "post_lift_hold":
                                hold_left = int(vla_planner_state.get("post_lift_hold_left", 0))
                                controller.set_mode("translate")
                                if hold_left > 0:
                                    vla_planner_state["post_lift_hold_left"] = hold_left - 1
                                else:
                                    # Check success (target object lifted above table plane and/or above its initial height)
                                    target_prim = str(vla_planner_state.get("target_prim", ""))
                                    z = _target_z_from_tracker(target_prim) if target_prim else None
                                    z0 = vla_planner_state.get("target_z0_w", None)
                                    dz_table = None
                                    dz_from_start = None
                                    ok = False
                                    thresh = float(getattr(args, "lift_success_min_dz_m", 0.06))
                                    try:
                                        if z is not None:
                                            dz_table = float(z) - float(_table_z())
                                            ok = bool(dz_table >= thresh)
                                            if z0 is not None:
                                                dz_from_start = float(z) - float(z0)
                                                ok = bool(ok or (dz_from_start >= thresh))
                                    except Exception:
                                        ok = False
                                    session_logger.log_event(
                                        "grasp_result",
                                        {
                                            "episode_idx": int(ep),
                                            "ok": bool(ok),
                                            "target_z": z,
                                            "dz_above_table": dz_table,
                                            "dz_from_start": dz_from_start,
                                            "lift_success_thresh_m": float(thresh),
                                        },
                                    )
                                    if ok:
                                        vla_planner_state["stage"] = "done"
                                    else:
                                        # Retry: reopen and replan (within the same episode)
                                        attempt = int(vla_planner_state.get("attempt_idx", 0)) + 1
                                        vla_planner_state["attempt_idx"] = attempt
                                        if attempt < int(getattr(args, "max_grasp_attempts", 3)):
                                            session_logger.log_event(
                                                "action_start",
                                                {"action": "RETRY_GRASP", "attempt_idx": int(attempt), "episode_idx": int(ep)},
                                            )
                                            wp.set_waypoints_b([])
                                            vla_planner_state["stage"] = "init_open"
                                            vla_planner_state["open_left"] = int(getattr(args, "gripper_open_steps", 10))
                                            vla_planner_state["open_queued"] = False
                                            vla_planner_state["close_queued"] = False
                                            vla_planner_state["lift_queued"] = False
                                            # small settle before replan
                                            vla_planner_state["stabilize_left"] = int(getattr(args, "stabilize_steps", 300))
                                            # Reset controller mode for retry
                                            controller.set_mode("gripper")
                                        else:
                                            vla_planner_state["stage"] = "done"

                            if str(vla_planner_state.get("stage", "")) == "done":
                                session_logger.log_event("episode_end", {"episode_idx": int(ep), "steps": int(steps)})
                                print(f"[VLA_V1][EP] end ep={ep} steps={steps} attempts={int(vla_planner_state.get('attempt_idx', 0))} ticks={session_logger.tick_idx} images={images_captured_episode}")
                                # Track totals before closing
                                total_ticks += session_logger.tick_idx
                                total_images += images_captured_episode
                                # Close logger for this episode
                                try:
                                    session_logger.close()
                                except Exception:
                                    pass
                                break
                    except Exception as e:
                        # Log error but continue the episode
                        import traceback
                        if session_logger.tick_idx < 10 or steps % 100 == 0:
                            print(f"[VLA_V1][PLANNER][WARN] Planner control error at ep={ep} step={steps} tick={session_logger.tick_idx}: {e}")
                            if session_logger.tick_idx < 3:
                                traceback.print_exc()

                    controller.step(robot, dt)

                    do_tick = (accum + dt + 1e-9) >= period
                    do_render = bool(do_tick) or (steps % render_stride == 0)
                    sim.step(render=bool(do_render))
                    robot.update(dt)

                    if camera_sensor is not None:
                        try:
                            if hasattr(camera_sensor, "update"):
                                if bool(do_render):
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
                # If we hit the max steps, still end the episode cleanly.
                if str(vla_planner_state.get("stage", "")) != "done":
                    session_logger.log_event("episode_end", {"episode_idx": int(ep), "steps": int(steps), "truncated": True})
                    print(f"[VLA_V1][EP] end ep={ep} (truncated) steps={steps} ticks={session_logger.tick_idx} images={images_captured_episode}")
                    # Track totals before closing
                    total_ticks += session_logger.tick_idx
                    total_images += images_captured_episode
                    # Close logger for this episode
                    try:
                        session_logger.close()
                    except Exception:
                        pass
            except Exception as ep_error:
                # Catch any episode-level errors and continue to next episode
                import traceback
                print(f"[VLA_V1][EP][ERROR] Episode {ep} failed with error: {ep_error}")
                traceback.print_exc()
                # Try to close logger if it was created
                try:
                    if 'session_logger' in locals():
                        session_logger.close()
                except Exception:
                    pass
                # Continue to next episode
                continue

    print(f"\n[VLA_V1] Data collection completed!")
    if control_mode == "planner":
        print(f"[VLA_V1] Session directory: {session_folder}")
        print(f"[VLA_V1] Total episodes: {num_episodes}")
        print(f"[VLA_V1] Total ticks logged: {total_ticks}")
        print(f"[VLA_V1] Total images captured: {total_images}")
    else:
        print(f"[VLA_V1] Total ticks logged: {session_logger.tick_idx}")
        print(f"[VLA_V1] Total images captured: {images_captured}")

    simulation_app.close()
    # Only close logger if not in planner mode (planner mode closes per episode)
    if control_mode != "planner":
        try:
            session_logger.close()
        except Exception:
            pass
    return 0


PROFILE = ProfileSpec(
    name="vla_v1",
    add_cli_args=add_cli_args,
    run=run,
)


