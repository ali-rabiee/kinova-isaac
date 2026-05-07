from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

# Ensure project root on sys.path for modular imports (do this BEFORE any Isaac/Omni imports).
ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str in sys.path:
    sys.path.remove(root_str)
sys.path.insert(0, root_str)

# Isaac Sim sometimes preloads a non-package module named `environments`, which breaks
# `import environments.<...>` later. If that happens, evict it so our local package wins.
_env_mod = sys.modules.get("environments")
if _env_mod is not None and not hasattr(_env_mod, "__path__"):
    del sys.modules["environments"]

from isaaclab.app import AppLauncher

from controllers import (  # noqa: E402
    CartesianVelocityJogConfig,
    CartesianVelocityJogController,
)
from data_collection.core.logger import SessionLogWriter, TickLoggingConfig  # noqa: E402
from data_collection.core.objects import ObjectsTracker  # noqa: E402

from data_collection.config import (  # noqa: E402
    RunConfig,
    EpisodeConfig,
    TaskConfig,
    PlannerConfig,
    ObjectsConfig,
    LoggingConfig,
)


def run_data_collection(args: argparse.Namespace) -> int:
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Import Isaac/scene modules that require an active Omniverse app
    # Re-pin our repo root on sys.path because Kit/Isaac can mutate sys.path during startup.
    root_str = str(ROOT)
    if root_str in sys.path:
        sys.path.remove(root_str)
    sys.path.insert(0, root_str)
    _env_mod = sys.modules.get("environments")
    if _env_mod is not None and not hasattr(_env_mod, "__path__"):
        del sys.modules["environments"]

    from environments.ycb_reach_to_grasp import YCBReachToGraspEnv  # noqa: E402
    from data_collection.engine.episode_runner import EpisodeRunner  # noqa: E402

    scale_range = None
    if getattr(args, "scale_min", None) is not None and getattr(args, "scale_max", None) is not None:
        scale_range = (float(args.scale_min), float(args.scale_max))
        print(f"[DATA] Uniform scale range: {scale_range}")

    env = YCBReachToGraspEnv(device=args.device, scale_range=scale_range)
    sim = env.build_simulation()
    if not args.headless:
        env.set_default_camera_view()

    print("[DATA] App launched; building scene...")
    env.design_scene()
    robot = env.robot

    # Spawn objects (default to Nucleus YCB if not provided)
    id_to_label: Dict[str, str] = {}
    prim_paths: list[str] = []
    dataset_dirs = [str(d) for d in getattr(args, "objects_dataset", [])]
    if len(dataset_dirs) == 0:
        dataset_dirs = [env.ycb_dir]
        print(f"[DATA] Using default YCB dataset: {dataset_dirs[0]}")
    else:
        print(f"[DATA] Using custom object datasets: {dataset_dirs}")

    if (not getattr(args, "no_objects", False)) and int(args.num_objects) > 0:
        loader = env.build_object_loader(
            spawn_min=tuple(args.spawn_min),
            spawn_max=tuple(args.spawn_max),
            min_distance=float(getattr(args, "min_distance", 0.1)),
            dataset_dirs=dataset_dirs,
        )
        print(
            f"[DATA] Spawning {int(args.num_objects)} objects in AABB "
            f"min={tuple(args.spawn_min)} max={tuple(args.spawn_max)}"
        )
        try:
            prim_paths = loader.spawn(
                parent_prim_path="/World/Origin1",
                num_objects=int(args.num_objects),
            )
        except Exception as e:
            print(f"[DATA][WARN] Object spawn failed: {e}")
            prim_paths = []
        try:
            prim_to_label = loader.get_last_spawn_labels()
            id_to_label = {str(p).split("/")[-1]: str(lbl) for p, lbl in prim_to_label.items()}
            print(f"[DATA] Spawned objects: {id_to_label}")
        except Exception:
            id_to_label = {}
            print("[DATA][WARN] Could not build id->label map; labels may default to 'object'.")

    # Reset sim and robot
    env.reset()

    # Controller
    ctrl_cfg = CartesianVelocityJogConfig(
        ee_link_name=str(getattr(args, "ee_link", "j2n6s300_end_effector")),
        device=str(sim.device),
        use_relative_mode=True,
        linear_speed_mps=float(getattr(args, "speed", 0.7)),
        workspace_min=(0.20, -0.45, 0.01),
        workspace_max=(0.6, 0.45, 0.35),
        log_ee_pos=False,
        log_ee_frame="world",
        log_every_n_steps=9999,
    )
    print(f"[DATA] Controller speed={ctrl_cfg.linear_speed_mps} m/s, ee_link={ctrl_cfg.ee_link_name}")
    controller = CartesianVelocityJogController(ctrl_cfg, num_envs=1, device=str(sim.device))
    controller.set_mode("translate")

    # Logging
    tick_cfg = TickLoggingConfig(
        log_rate_hz=10,
        workspace_min=getattr(controller.config.safety_cfg, 'workspace_min', None),
        workspace_max=getattr(controller.config.safety_cfg, 'workspace_max', None),
        ee_link_name=controller.config.ee_link_name,
        arm_joint_regex=controller.config.arm_joint_regex,
    )
    session_logger = SessionLogWriter(root=Path(str(args.logs_root)))
    session_logger.write_metadata(
        sim_dt=sim.get_physics_dt(),
        physics_substeps=int(getattr(sim.cfg, 'sub_steps', 4)),
        seed=0,
        robot_name="kinova_j2n6s300",
        ee_link=controller.config.ee_link_name,
        arm_joint_regex=controller.config.arm_joint_regex,
        log_rate_hz=tick_cfg.log_rate_hz,
        window_len_s=2.0,
    )

    # Tracker
    tracker = ObjectsTracker(prim_paths=prim_paths)

    # Run config (planner type is user-chosen)
    run_cfg = RunConfig(
        episode=EpisodeConfig(num_episodes=int(args.num_episodes)),
        task=TaskConfig(
            target_label=getattr(args, "target_label", None),
            pregrasp_offset_m=float(getattr(args, "pregrasp", 0.10)),
            lift_height_m=float(getattr(args, "lift", 0.15)),
        ),
        planner=PlannerConfig(
            type=str(getattr(args, "planner", "scripted")),
            linear_speed_mps=float(getattr(args, "speed", 0.7)),
            tolerance_m=float(getattr(args, "tolerance", 0.005)),
        ),
        objects=ObjectsConfig(
            dataset_dirs=dataset_dirs,
            num_objects=int(args.num_objects),
            spawn_min_xyz=tuple(args.spawn_min),
            spawn_max_xyz=tuple(args.spawn_max),
        ),
        logging=LoggingConfig(logs_root=str(args.logs_root)),
    )

    print("[DATA] EpisodeRunner initialized. Starting episodes...")
    runner = EpisodeRunner(
        sim=sim,
        robot=robot,
        controller=controller,
        session_logger=session_logger,
        tick_cfg=tick_cfg,
        tracker=tracker,
        id_to_label=id_to_label,
        run_cfg=run_cfg,
    )

    # Episodes
    successes = 0
    for ep in range(int(args.num_episodes)):
        outcome = runner.run_episode(target_label=args.target_label)
        session_logger.log_event(
            "episode_end",
            {
                "episode": ep,
                "success": outcome.success,
                "reason": outcome.reason,
                "target": {"id": outcome.target_id, "label": outcome.target_label},
            },
        )
        successes += int(outcome.success)

    print(f"[DATA] Completed: episodes={args.num_episodes} success={successes}")
    simulation_app.close()
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Data collection using motion_generation planners")
    # Reuse the main demo CLI options from scripts to keep UX similar
    from scripts.cli import add_demo_cli_args  # noqa: E402

    add_demo_cli_args(ap)
    ap.add_argument("--num-episodes", type=int, default=10)
    ap.add_argument("--target-label", type=str, default=None)
    ap.add_argument("--objects-dataset", type=str, nargs="*", default=[])
    ap.add_argument("--pregrasp", type=float, default=0.10)
    ap.add_argument("--lift", type=float, default=0.15)
    ap.add_argument("--tolerance", type=float, default=0.005)
    ap.add_argument("--logs-root", type=str, default="logs/data_collection")
    ap.add_argument("--planner", type=str, default="scripted", choices=["scripted", "rmpflow", "curobo"])
    AppLauncher.add_app_launcher_args(ap)
    args_cli = ap.parse_args()
    raise SystemExit(run_data_collection(args_cli))


