from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from controllers.base import InputProvider
from data_collection.envs.registry import get_envs
from data_collection.profiles.spec import ProfileSpec


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--env", type=str, default="reach_to_grasp_VLA", choices=sorted(get_envs().keys()))
    parser.add_argument("--logs-root", type=str, default="logs/data_collection")
    # For shared autonomy, a higher default makes human teleop usable while still being manageable.
    parser.add_argument("--log-rate-hz", type=int, default=20)
    parser.add_argument("--duration-s", type=float, default=30.0, help="Duration per episode (seconds).")
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--control", type=str, default="keyboard", choices=["keyboard", "idle"])
    parser.add_argument("--image-format", type=str, default="png", choices=["png", "jpg"], help="Image format for saving")

    # Workspace safety bounds (base frame).
    parser.add_argument("--workspace-min-z", type=float, default=0.0, help="Minimum EE z in base frame (m).")
    parser.add_argument("--workspace-max-z", type=float, default=1.20, help="Maximum EE z in base frame (m).")

    # Episode goal selection
    parser.add_argument("--target-label", type=str, default=None, help="Optional target object label filter (substring match).")
    parser.add_argument(
        "--target-selection",
        type=str,
        default="random",
        choices=["first", "random"],
        help="How to choose the target object each episode (after filtering).",
    )
    parser.add_argument(
        "--target-index",
        type=int,
        default=None,
        help="Optional explicit index into the filtered object list (sorted by id). Overrides --target-selection if set.",
    )

    # Expert suggestion (simple reach->grasp->lift state machine in base frame)
    parser.add_argument("--pregrasp", type=float, default=0.10, help="Pre-grasp offset above target position (m)")
    parser.add_argument("--grasp-depth", type=float, default=-0.07, help="Grasp depth relative to target position (m).")
    parser.add_argument("--lift", type=float, default=0.15, help="Lift height above target position (m)")
    parser.add_argument("--tolerance", type=float, default=0.01, help="Waypoint tolerance for stage transitions (m)")
    parser.add_argument("--gripper-open-steps", type=int, default=10, help="Steps to open gripper at episode start")
    parser.add_argument("--gripper-close-steps", type=int, default=60, help="Steps to close gripper during grasp")
    parser.add_argument("--end-on-done", action="store_true", help="End episode early when expert reaches 'done' stage.")

    # Shared autonomy blending (gamma)
    parser.add_argument(
        "--gamma-mode",
        type=str,
        default="stage",
        choices=["fixed", "stage"],
        help="How to choose blending gamma (0=human, 1=expert).",
    )
    parser.add_argument("--gamma", type=float, default=0.5, help="Fixed gamma value if --gamma-mode=fixed.")
    parser.add_argument("--gamma-reach", type=float, default=0.25, help="Stage gamma for reach.")
    parser.add_argument("--gamma-approach", type=float, default=0.50, help="Stage gamma for approach.")
    parser.add_argument("--gamma-grasp", type=float, default=0.80, help="Stage gamma for grasp.")
    parser.add_argument("--gamma-lift", type=float, default=0.60, help="Stage gamma for lift.")


@dataclass
class _EpisodeGoal:
    object_id: str
    object_label: str
    target_pos_b: tuple[float, float, float]


class _BraceExpert(InputProvider):
    """A simple expert suggestion generator for reach->approach->grasp->lift.

    Produces per-step SE(3) delta commands in *base frame*:
      [dx, dy, dz, 0, 0, 0, g]

    Stage machine:
      init_open -> reach -> approach -> grasp -> lift -> done
    """

    def __init__(
        self,
        *,
        step_pos_m: float,
        tol_m: float,
        pregrasp_m: float,
        grasp_depth_m: float,
        lift_m: float,
        open_steps: int,
        close_steps: int,
        device: str,
    ) -> None:
        import torch

        self._torch = torch
        self.device = torch.device(device)
        self.step_pos_m = float(step_pos_m)
        self.tol_m = float(tol_m)
        self.pregrasp_m = float(pregrasp_m)
        self.grasp_depth_m = float(grasp_depth_m)
        self.lift_m = float(lift_m)
        self.open_steps = int(open_steps)
        self.close_steps = int(close_steps)

        self.stage: str = "init_open"
        self._target_b: Optional[torch.Tensor] = None  # (3,)
        self._ee_b: Optional[torch.Tensor] = None  # (3,)
        self._open_left: int = self.open_steps
        self._close_left: int = self.close_steps

    def reset(self) -> None:
        self.stage = "init_open"
        self._open_left = int(self.open_steps)
        self._close_left = int(self.close_steps)
        # Keep target/ee; they are set by the runner.

    def set_target_pos_b(self, pos_b: tuple[float, float, float]) -> None:
        self._target_b = self._torch.tensor(pos_b, dtype=self._torch.float32, device=self.device).view(-1)

    def set_current_ee_pos_b(self, ee_pos_b) -> None:
        t = ee_pos_b
        if hasattr(t, "ndim") and t.ndim == 2:
            t = t.view(-1)
        self._ee_b = t.to(self.device).to(dtype=self._torch.float32).view(-1)

    def _move_toward(self, goal_b) -> "torch.Tensor":
        torch = self._torch
        ee = self._ee_b
        if ee is None:
            return torch.zeros(1, 7, dtype=torch.float32, device=self.device)
        diff = (goal_b - ee).to(self.device)
        dist = float(torch.linalg.norm(diff).item())
        if dist <= float(self.tol_m):
            return torch.zeros(1, 7, dtype=torch.float32, device=self.device)
        direction = diff / (dist + 1e-9)
        step = float(min(float(self.step_pos_m), dist))
        dpos = direction * step
        cmd = torch.zeros(1, 7, dtype=torch.float32, device=self.device)
        cmd[0, 0:3] = dpos
        return cmd

    def advance(self):  # -> torch.Tensor
        torch = self._torch
        if self._target_b is None:
            return torch.zeros(1, 7, dtype=torch.float32, device=self.device)

        # Stage: init_open
        if self.stage == "init_open":
            if self._open_left > 0:
                self._open_left -= 1
                cmd = torch.zeros(1, 7, dtype=torch.float32, device=self.device)
                cmd[0, 6] = +1.0
                return cmd
            self.stage = "reach"

        # Stage: reach (move above target)
        if self.stage == "reach":
            goal = self._target_b + torch.tensor([0.0, 0.0, float(self.pregrasp_m)], device=self.device)
            cmd = self._move_toward(goal)
            # Transition if close enough (re-evaluate with current ee)
            if self._ee_b is not None:
                dist = float(torch.linalg.norm(goal - self._ee_b).item())
                if dist <= float(self.tol_m):
                    self.stage = "approach"
            return cmd

        # Stage: approach (descend to grasp depth)
        if self.stage == "approach":
            goal = self._target_b + torch.tensor([0.0, 0.0, float(self.grasp_depth_m)], device=self.device)
            cmd = self._move_toward(goal)
            if self._ee_b is not None:
                dist = float(torch.linalg.norm(goal - self._ee_b).item())
                if dist <= float(self.tol_m):
                    self.stage = "grasp"
            return cmd

        # Stage: grasp (close gripper)
        if self.stage == "grasp":
            if self._close_left > 0:
                self._close_left -= 1
                cmd = torch.zeros(1, 7, dtype=torch.float32, device=self.device)
                cmd[0, 6] = -1.0
                return cmd
            self.stage = "lift"

        # Stage: lift (move up)
        if self.stage == "lift":
            goal = self._target_b + torch.tensor([0.0, 0.0, float(self.lift_m)], device=self.device)
            cmd = self._move_toward(goal)
            if self._ee_b is not None:
                dist = float(torch.linalg.norm(goal - self._ee_b).item())
                if dist <= float(self.tol_m):
                    self.stage = "done"
            return cmd

        # done / fallback
        return torch.zeros(1, 7, dtype=torch.float32, device=self.device)


def _stage_label(stage_raw: str) -> str:
    s = str(stage_raw or "")
    if s in {"reach", "approach", "grasp", "lift"}:
        return s
    if s == "init_open":
        return "reach"
    if s == "done":
        return "lift"
    return "reach"


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
    from data_collection.core.input_mux import SampleAndHoldInputProvider, SharedAutonomyBlendInputProvider
    from data_collection.core.logger import SessionLogWriter, TickLoggingConfig
    from data_collection.core.objects import ObjectsTracker

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import torch  # noqa: E402
    try:
        import numpy as np  # noqa: E402
    except Exception as e:
        print(f"[BRACE_V1] ERROR: numpy is required for image saving but could not be imported: {e}")
        print("[BRACE_V1] Please install numpy in your environment (e.g., `pip install numpy`) and retry.")
        return 2

    # Isaac Lab's Camera sensor initialization gate checks this carb setting.
    import carb  # noqa: E402

    carb_settings = carb.settings.get_settings()
    enable_cameras = bool(getattr(args, "enable_cameras", False))
    carb_settings.set_bool("/isaaclab/cameras_enabled", enable_cameras)
    print(f"[BRACE_V1] enable_cameras flag value: {enable_cameras}")
    print(f"[BRACE_V1] carb /isaaclab/cameras_enabled={carb_settings.get('/isaaclab/cameras_enabled')}")

    import importlib
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import Camera, CameraCfg
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

    from controllers import CartesianVelocityJogConfig, CartesianVelocityJogController
    from environments.utils.object_loader import ObjectLoader, ObjectLoaderConfig, SpawnBounds
    from environments.utils.physix import PhysicsConfig, apply_to_simulation_cfg, object_loader_kwargs_from_physix
    from utilities import get_ee_pos_base_frame
    from utilities.transforms import world_to_base_pos

    # --- Scene / sim setup ---
    env_spec = get_envs()[str(getattr(args, "env", "reach_to_grasp_VLA"))]
    env_cfg_mod = importlib.import_module(f"{env_spec.module_base}.config")
    env_utils_mod = importlib.import_module(f"{env_spec.module_base}.utils")
    DEFAULT_SCENE = getattr(env_cfg_mod, "DEFAULT_SCENE")
    DEFAULT_CAMERA = getattr(env_cfg_mod, "DEFAULT_CAMERA", None)
    DEFAULT_TOP_DOWN_CAMERA = getattr(env_cfg_mod, "DEFAULT_TOP_DOWN_CAMERA", None)
    design_scene = getattr(env_utils_mod, "design_scene")
    create_topdown_camera = getattr(importlib.import_module("environments.utils.camera"), "create_topdown_camera")

    phys = PhysicsConfig(device=str(getattr(args, "device", "cuda:0")))
    sim_cfg = sim_utils.SimulationCfg(device=phys.device)
    apply_to_simulation_cfg(sim_cfg, phys)
    sim = sim_utils.SimulationContext(sim_cfg)
    if (not getattr(args, "headless", False)) and DEFAULT_CAMERA is not None:
        sim.set_camera_view(DEFAULT_CAMERA.eye, DEFAULT_CAMERA.target)

    scene_entities, scene_origins = design_scene(DEFAULT_SCENE)
    robot = scene_entities["kinova_j2n6s300"]

    # Create top-down camera prim ONLY if cameras are enabled.
    if enable_cameras and DEFAULT_TOP_DOWN_CAMERA is not None:
        create_topdown_camera(DEFAULT_TOP_DOWN_CAMERA)
        print(f"[BRACE_V1] Top-down camera created at: {DEFAULT_TOP_DOWN_CAMERA.prim_path}")

    # Spawn objects
    spawned_paths = []
    id_to_label: Dict[str, str] = {}
    if not getattr(args, "no_objects", False):
        try:
            ycb_dir = f"{ISAAC_NUCLEUS_DIR}/Props/YCB"
        except Exception:
            ycb_dir = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Props/YCB"

        scale_range = None
        if getattr(args, "scale_min", None) is not None and getattr(args, "scale_max", None) is not None:
            scale_range = (float(args.scale_min), float(args.scale_max))

        phys_loader_kwargs = object_loader_kwargs_from_physix(phys)
        loader_cfg = ObjectLoaderConfig(
            dataset_dirs=[ycb_dir],
            bounds=SpawnBounds(min_xyz=tuple(args.spawn_min), max_xyz=tuple(args.spawn_max)),
            min_distance=float(getattr(args, "min_distance", 0.1)),
            uniform_scale_range=scale_range,
            **phys_loader_kwargs,
        )
        loader = ObjectLoader(loader_cfg)
        spawned_paths = loader.spawn(parent_prim_path="/World/Origin1", num_objects=int(getattr(args, "num_objects", 0)))
        try:
            prim_to_label = loader.get_last_spawn_labels()
            id_to_label = {str(p).split("/")[-1]: str(lbl) for p, lbl in prim_to_label.items()}
        except Exception:
            id_to_label = {}

    # Camera sensor (optional)
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
            print(f"[BRACE_V1] Camera sensor created: {camera_cfg.width}x{camera_cfg.height}")
        except Exception as create_err:
            print(f"[BRACE_V1] WARN: Failed to create Camera object: {create_err}")
            camera_sensor = None

    # Reset sim and robot (timeline PLAY)
    sim.reset()
    origin0 = torch.tensor(scene_origins[0], device=sim.device)
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += origin0
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    robot.reset()

    if camera_sensor is not None:
        try:
            camera_sensor.reset()
            print("[BRACE_V1] Camera sensor reset OK (post sim.reset)")
        except Exception as e:
            print(f"[BRACE_V1] WARN: Camera sensor reset failed post sim.reset: {e}")
            camera_sensor = None

    # Controller
    ctrl_cfg = CartesianVelocityJogConfig(
        ee_link_name=str(getattr(args, "ee_link", "j2n6s300_end_effector")),
        device=str(sim.device),
        use_relative_mode=True,
        linear_speed_mps=float(getattr(args, "speed", 0.7)),
        workspace_min=(0.20, -0.45, float(getattr(args, "workspace_min_z", 0.0))),
        workspace_max=(0.60, 0.45, float(getattr(args, "workspace_max_z", 1.20))),
        log_ee_pos=bool(getattr(args, "print_ee", False)),
        log_ee_frame=str(getattr(args, "ee_frame", "world")),
        log_every_n_steps=int(getattr(args, "print_interval", 1)),
    )
    controller = CartesianVelocityJogController(ctrl_cfg, num_envs=1, device=str(sim.device))
    controller.set_mode("translate")
    controller.reset(robot)

    # Shared autonomy providers
    control_mode = str(getattr(args, "control", "keyboard"))
    human_base: Optional[InputProvider] = None
    if (not getattr(args, "headless", False)) and control_mode == "keyboard":
        from controllers.input.keyboard import Se3KeyboardInput

        human_base = Se3KeyboardInput(
            pos_sensitivity_per_step=ctrl_cfg.linear_speed_mps * sim.get_physics_dt(),
            rot_sensitivity_rad_per_step=float(getattr(args, "rot_speed", 2.0)) * sim.get_physics_dt(),
        )

    human_hold = SampleAndHoldInputProvider(human_base, default_dim=7, device=str(sim.device))
    expert = _BraceExpert(
        step_pos_m=float(ctrl_cfg.linear_speed_mps) * float(sim.get_physics_dt()),
        tol_m=float(getattr(args, "tolerance", 0.01)),
        pregrasp_m=float(getattr(args, "pregrasp", 0.10)),
        grasp_depth_m=float(getattr(args, "grasp_depth", -0.07)),
        lift_m=float(getattr(args, "lift", 0.15)),
        open_steps=int(getattr(args, "gripper_open_steps", 10)),
        close_steps=int(getattr(args, "gripper_close_steps", 60)),
        device=str(sim.device),
    )
    auto_hold = SampleAndHoldInputProvider(expert, default_dim=7, device=str(sim.device))

    blend = SharedAutonomyBlendInputProvider(
        human_provider=human_hold,
        auto_provider=auto_hold,
        gamma=float(getattr(args, "gamma", 0.5)),
        device=str(sim.device),
    )
    controller.set_input_provider(blend)

    # Logging config
    tick_cfg = TickLoggingConfig(
        log_rate_hz=int(getattr(args, "log_rate_hz", 20)),
        policy_rate_hz=int(getattr(args, "log_rate_hz", 20)),
        workspace_min=getattr(controller.config.safety_cfg, "workspace_min", None),
        workspace_max=getattr(controller.config.safety_cfg, "workspace_max", None),
        ee_link_name=str(getattr(args, "ee_link", "j2n6s300_end_effector")),
        arm_joint_regex=controller.config.arm_joint_regex,
        log_joint_data=True,
    )

    dt = float(sim.get_physics_dt())
    period = 1.0 / float(max(1, int(tick_cfg.log_rate_hz)))

    logs_root = Path(str(getattr(args, "logs_root", "logs/data_collection")))
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_folder = logs_root / f"session_brace_{session_timestamp}"
    session_folder.mkdir(parents=True, exist_ok=True)

    def _choose_goal(tracker: ObjectsTracker) -> Optional[_EpisodeGoal]:
        try:
            snap = list(tracker.snapshot())
        except Exception:
            snap = []
        if not snap:
            return None

        # Attach best-available labels
        def _lbl(obj_id: str, fallback: str) -> str:
            leaf = str(obj_id).split("/")[-1]
            return str(id_to_label.get(leaf, fallback))

        # Filter by label substring if requested
        filt = []
        q = getattr(args, "target_label", None)
        q_str = str(q).strip().lower() if q is not None else ""
        for o in snap:
            lbl = _lbl(o.id, o.label)
            if q_str and (q_str not in str(lbl).lower()):
                continue
            filt.append((str(o.id), str(lbl), tuple(float(v) for v in o.pose.position_m)))
        if not filt:
            return None
        filt.sort(key=lambda x: x[0])

        idx = getattr(args, "target_index", None)
        if idx is not None:
            i = int(idx)
            i = max(0, min(i, len(filt) - 1))
            obj_id, lbl, pos_w = filt[i]
        else:
            sel = str(getattr(args, "target_selection", "random"))
            if sel == "first":
                obj_id, lbl, pos_w = filt[0]
            else:
                obj_id, lbl, pos_w = random.choice(filt)

        pos_b = world_to_base_pos(sim, robot, pos_w)
        return _EpisodeGoal(object_id=str(obj_id), object_label=str(lbl), target_pos_b=pos_b)

    def _gamma_for_stage(stage: str) -> float:
        mode = str(getattr(args, "gamma_mode", "stage"))
        if mode == "fixed":
            return float(getattr(args, "gamma", 0.5))
        st = _stage_label(stage)
        if st == "reach":
            return float(getattr(args, "gamma_reach", 0.25))
        if st == "approach":
            return float(getattr(args, "gamma_approach", 0.50))
        if st == "grasp":
            return float(getattr(args, "gamma_grasp", 0.80))
        if st == "lift":
            return float(getattr(args, "gamma_lift", 0.60))
        return float(getattr(args, "gamma", 0.5))

    def _cmd7_list(cmd_t: Optional[torch.Tensor]) -> Optional[list[float]]:
        if cmd_t is None:
            return None
        u = cmd_t.detach().view(-1).to("cpu").tolist()
        out = [float(v) for v in (u[:7] if len(u) >= 7 else (u + [0.0] * (7 - len(u))))]
        return out

    total_ticks = 0
    total_images = 0
    num_episodes = int(getattr(args, "num_episodes", 1))

    print(f"[BRACE_V1] Data collection started! session={session_folder}")
    print(f"[BRACE_V1] Episodes={num_episodes} log_rate_hz={tick_cfg.log_rate_hz} duration_s={float(getattr(args, 'duration_s', 30.0))}")

    for ep in range(num_episodes):
        if not simulation_app.is_running():
            break

        # Create logger per episode (separate log files)
        session_logger = SessionLogWriter(root=session_folder, session_name=f"episode_{ep:04d}")
        images_dir = session_logger.root / "images"
        if camera_sensor is not None:
            images_dir.mkdir(exist_ok=True)
        image_format = getattr(args, "image_format", "png")

        session_logger.write_metadata(
            sim_dt=sim.get_physics_dt(),
            physics_substeps=int(getattr(sim.cfg, "sub_steps", 4)),
            seed=0,
            robot_name="kinova_j2n6s300",
            ee_link=str(getattr(args, "ee_link", "j2n6s300_end_effector")),
            arm_joint_regex=controller.config.arm_joint_regex,
            log_rate_hz=tick_cfg.log_rate_hz,
            window_len_s=2.0,
            policy_rate_hz=tick_cfg.policy_rate_hz,
        )

        # Reset sim/robot for a clean episode start
        sim.reset()
        root_state = robot.data.default_root_state.clone()
        root_state[:, :3] += origin0
        robot.write_root_pose_to_sim(root_state[:, :7])
        robot.write_root_velocity_to_sim(root_state[:, 7:])
        robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
        robot.reset()
        try:
            controller.reset(robot)
            controller.set_mode("translate")
        except Exception:
            pass
        if camera_sensor is not None:
            try:
                camera_sensor.reset()
            except Exception:
                pass

        # Recreate tracker per episode (PhysX views can go stale across reset on some builds)
        tracker = ObjectsTracker(prim_paths=spawned_paths)

        goal = _choose_goal(tracker)
        if goal is None:
            try:
                # Ensure we don't carry the previous episode's target.
                setattr(expert, "_target_b", None)  # type: ignore[attr-defined]
            except Exception:
                pass
            session_logger.log_event("episode_goal", {"episode_idx": int(ep), "goal_object_id": None})
            print(f"[BRACE_V1][EP {ep}] WARN: No goal could be selected (no objects?).")
        else:
            session_logger.log_event(
                "episode_goal",
                {
                    "episode_idx": int(ep),
                    "goal_object_id": str(goal.object_id),
                    "goal_object_label": str(goal.object_label),
                    "target_pos_b": [float(v) for v in goal.target_pos_b],
                },
            )
            expert.set_target_pos_b(goal.target_pos_b)

        # Reset shared-autonomy providers and sample the first held commands
        try:
            human_hold.reset()
        except Exception:
            pass
        try:
            auto_hold.reset()
        except Exception:
            pass
        try:
            blend.reset()
        except Exception:
            pass
        try:
            expert.reset()
        except Exception:
            pass
        try:
            expert.set_current_ee_pos_b(get_ee_pos_base_frame(robot, str(getattr(args, "ee_link", "j2n6s300_end_effector"))))
        except Exception:
            pass
        try:
            human_hold.sample()
        except Exception:
            pass
        try:
            auto_hold.sample()
        except Exception:
            pass
        blend.set_gamma(_gamma_for_stage(expert.stage))

        ep_start = time.time()
        accum = 0.0
        steps = 0
        images_captured_ep = 0

        while simulation_app.is_running() and (time.time() - ep_start) < float(getattr(args, "duration_s", 30.0)):
            steps += 1
            controller.step(robot, dt)
            sim.step(render=bool(not getattr(args, "headless", False)))
            robot.update(dt)

            if camera_sensor is not None:
                try:
                    if hasattr(camera_sensor, "update"):
                        camera_sensor.update(dt)
                except Exception:
                    pass

            accum += dt
            if accum + 1e-9 >= period:
                accum = 0.0

                # Snapshot objects
                objs_raw = []
                try:
                    for o in tracker.snapshot():
                        lbl = id_to_label.get(str(o.id).split("/")[-1], o.label)
                        objs_raw.append(
                            {
                                "id": str(o.id),
                                "label": str(lbl),
                                "pose": {
                                    "position_m": list(o.pose.position_m),
                                    "orientation_wxyz": list(o.pose.orientation_wxyz),
                                },
                                "confidence": float(o.confidence),
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
                            images_captured_ep += 1
                    except Exception:
                        image_path = None

                # Shared autonomy signals (aligned to the prev->cur interval, like action_from_prev).
                stage_raw = str(getattr(expert, "stage", ""))
                shared = {
                    "alignment": "from_prev",
                    "u_human_7d": _cmd7_list(getattr(blend, "last_human_cmd", None)),
                    "u_auto_7d": _cmd7_list(getattr(blend, "last_auto_cmd", None)),
                    "u_exec_7d": _cmd7_list(getattr(blend, "last_exec_cmd", None)),
                    "u_exec_7d_post_safety": _cmd7_list(getattr(controller, "last_cmd_post_safety", None)),
                    "gamma_from_prev": float(getattr(blend, "last_gamma", None) or 0.0),
                }
                task = {
                    "goal_object_id": (str(goal.object_id) if goal is not None else None),
                    "goal_object_label": (str(goal.object_label) if goal is not None else None),
                    "candidate_goal_ids": [str(o.get("id", "")) for o in objs_raw],
                    "stage": _stage_label(stage_raw),
                    "stage_raw": stage_raw,
                }

                session_logger.write_tick(
                    robot=robot,
                    controller=controller,
                    objects=objs_raw,
                    last_user_cmd=getattr(human_hold, "last_cmd", None),
                    cfg=tick_cfg,
                    image_path=image_path,
                    shared_autonomy=shared,
                    task=task,
                )

                # Prepare next interval commands: update expert pose, resample, and set gamma for next interval.
                try:
                    expert.set_current_ee_pos_b(get_ee_pos_base_frame(robot, str(getattr(args, "ee_link", "j2n6s300_end_effector"))))
                except Exception:
                    pass
                try:
                    human_hold.sample()
                except Exception:
                    pass
                try:
                    auto_hold.sample()
                except Exception:
                    pass
                blend.set_gamma(_gamma_for_stage(expert.stage))

                if bool(getattr(args, "end_on_done", False)) and str(getattr(expert, "stage", "")) == "done":
                    break

        total_ticks += int(session_logger.tick_idx)
        total_images += int(images_captured_ep)
        try:
            session_logger.log_event(
                "episode_end",
                {"episode_idx": int(ep), "steps": int(steps), "ticks": int(session_logger.tick_idx), "images": int(images_captured_ep)},
            )
        except Exception:
            pass
        try:
            session_logger.close()
        except Exception:
            pass

    print(f"[BRACE_V1] Completed. session={session_folder} total_ticks={total_ticks} total_images={total_images}")
    simulation_app.close()
    return 0


PROFILE = ProfileSpec(
    name="brace_v1",
    add_cli_args=add_cli_args,
    run=run,
)

