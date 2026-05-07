from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict
from data_collection.envs.registry import get_envs
from data_collection.profiles.spec import ProfileSpec


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--env", type=str, default="ycb_reach_to_grasp", choices=sorted(get_envs().keys()))
    parser.add_argument("--logs-root", type=str, default="logs/data_collection")
    parser.add_argument("--log-rate-hz", type=int, default=10)
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--control", type=str, default="keyboard", choices=["keyboard", "idle"])


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

    # Heavy imports (isaaclab / omni) must happen only after Kit is started.
    from isaaclab.app import AppLauncher
    from data_collection.core.input_mux import CommandMuxInputProvider
    from data_collection.core.logger import SessionLogWriter, TickLoggingConfig
    from data_collection.core.objects import ObjectsTracker

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Imports requiring active app
    from controllers import CartesianVelocityJogConfig, CartesianVelocityJogController
    from environments.ycb_reach_to_grasp import YCBReachToGraspEnv

    scale_range = None
    if getattr(args, "scale_min", None) is not None and getattr(args, "scale_max", None) is not None:
        scale_range = (float(args.scale_min), float(args.scale_max))

    env = YCBReachToGraspEnv(
        device=str(getattr(args, "device", "cuda:0")),
        scale_range=scale_range,
    )
    sim = env.build_simulation()
    if not getattr(args, "headless", False):
        env.set_default_camera_view()

    env.design_scene()
    robot = env.robot

    # Spawn objects
    spawned_paths = []
    id_to_label: Dict[str, str] = {}
    if not getattr(args, "no_objects", False):
        loader = env.build_object_loader(
            spawn_min=tuple(args.spawn_min),
            spawn_max=tuple(args.spawn_max),
            min_distance=float(getattr(args, "min_distance", 0.1)),
        )
        spawned_paths = loader.spawn(
            parent_prim_path="/World/Origin1",
            num_objects=int(getattr(args, "num_objects", 0)),
        )
        try:
            prim_to_label = loader.get_last_spawn_labels()
            id_to_label = {str(p).split("/")[-1]: str(lbl) for p, lbl in prim_to_label.items()}
        except Exception:
            id_to_label = {}

    env.reset()

    # Controller
    ctrl_cfg = CartesianVelocityJogConfig(
        ee_link_name=str(getattr(args, "ee_link", "j2n6s300_end_effector")),
        device=str(sim.device),
        use_relative_mode=True,
        linear_speed_mps=float(getattr(args, "speed", 0.7)),
        workspace_min=(0.20, -0.45, 0.01),
        workspace_max=(0.6, 0.45, 0.35),
        log_ee_pos=bool(getattr(args, "print_ee", False)),
        log_ee_frame=str(getattr(args, "ee_frame", "world")),
        log_every_n_steps=int(getattr(args, "print_interval", 1)),
    )
    controller = CartesianVelocityJogController(ctrl_cfg, num_envs=1, device=str(sim.device))
    controller.set_mode("translate")
    # Important: controller must be reset before stepping (initializes internal IK, buffers, etc.)
    controller.reset(robot)

    # Input/control
    mux_input = CommandMuxInputProvider()
    if (not getattr(args, "headless", False)) and str(getattr(args, "control", "keyboard")) == "keyboard":
        from controllers.input.keyboard import Se3KeyboardInput

        keyboard = Se3KeyboardInput(
            pos_sensitivity_per_step=ctrl_cfg.linear_speed_mps * sim.get_physics_dt(),
            rot_sensitivity_rad_per_step=float(getattr(args, "rot_speed", 2.0)) * sim.get_physics_dt(),
        )
        mux_input.set_base(keyboard)
    controller.set_input_provider(mux_input)

    # Tracker + logger
    tracker = ObjectsTracker(prim_paths=spawned_paths)
    tick_cfg = TickLoggingConfig(
        log_rate_hz=int(getattr(args, "log_rate_hz", 10)),
        workspace_min=getattr(controller.config.safety_cfg, "workspace_min", None),
        workspace_max=getattr(controller.config.safety_cfg, "workspace_max", None),
        ee_link_name=str(getattr(args, "ee_link", "j2n6s300_end_effector")),
        arm_joint_regex=controller.config.arm_joint_regex,
    )
    session_logger = SessionLogWriter(root=Path(str(getattr(args, "logs_root", "logs/data_collection"))))
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

    # Run
    dt = float(sim.get_physics_dt())
    period = 1.0 / float(tick_cfg.log_rate_hz)
    accum = 0.0
    t0 = time.time()
    while simulation_app.is_running() and (time.time() - t0) < float(getattr(args, "duration_s", 30.0)):
        controller.step(robot, dt)
        sim.step()
        robot.update(dt)
        accum += dt
        if accum + 1e-9 >= period:
            accum = 0.0
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
            session_logger.write_tick(
                robot=robot,
                controller=controller,
                objects=objs_raw,
                last_user_cmd=mux_input.last_cmd,
                cfg=tick_cfg,
            )

    simulation_app.close()
    try:
        session_logger.close()
    except Exception:
        pass
    return 0


PROFILE = ProfileSpec(
    name="ticks_v0",
    add_cli_args=add_cli_args,
    run=run,
)


