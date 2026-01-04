from __future__ import annotations
import argparse
import sys
from pathlib import Path

import torch
from isaaclab.app import AppLauncher
from controllers import ModeManager

# Ensure project root on sys.path for modular imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from controllers import (
    CartesianVelocityJogConfig,
    CartesianVelocityJogController,
    Se3KeyboardInput,
    ModeManager,
)
from data_collection.core.input_mux import CommandMuxInputProvider
from data_collection.core.objects import ObjectsTracker
from data_collection.core.logger import SessionLogWriter, TickLoggingConfig




def run(sim, robot, controller: CartesianVelocityJogController, simulation_app, *, mux_input: CommandMuxInputProvider | None, obj_tracker: ObjectsTracker | None, session_logger: SessionLogWriter | None, tick_cfg: TickLoggingConfig | None, id_to_label: dict[str, str] | None = None):
    dt = sim.get_physics_dt()
    controller.reset(robot)
    accum = 0.0
    while simulation_app.is_running():
        controller.step(robot, dt)
        sim.step()
        robot.update(dt)
        # Logging-only tick at configured rate
        if session_logger is not None and obj_tracker is not None and tick_cfg is not None and mux_input is not None:
            accum += dt
            period = 1.0 / float(tick_cfg.log_rate_hz)
            if accum + 1e-9 >= period:
                accum = 0.0
                # Build object dict list for logging from tracker snapshot
                objs_raw = []
                try:
                    for o in obj_tracker.snapshot():
                        lbl = o.label
                        if id_to_label is not None:
                            try:
                                lbl = id_to_label.get(o.id, o.label)
                            except Exception:
                                lbl = o.label
                        objs_raw.append({
                            "id": o.id,
                            "label": lbl,
                            "pose": {"position_m": list(o.pose.position_m), "orientation_wxyz": list(o.pose.orientation_wxyz)},
                            "confidence": o.confidence,
                        })
                except Exception as e:
                    print(f"[LOG] Object snapshot failed: {e}")
                session_logger.write_tick(robot=robot, controller=controller, objects=objs_raw, last_user_cmd=mux_input.last_cmd, cfg=tick_cfg)


def main():
    parser = argparse.ArgumentParser(description="Cartesian velocity jog demo with object spawning.")
    from scripts.cli import add_demo_cli_args
    add_demo_cli_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # Import modules that require active Omniverse app
    import isaaclab.sim as sim_utils
    try:
        from environments.reach_to_grasp.config import DEFAULT_SCENE, DEFAULT_CAMERA
        from environments.reach_to_grasp.utils import design_scene
        from environments.utils.object_loader import ObjectLoader, ObjectLoaderConfig, SpawnBounds
        from environments.utils.physix import PhysicsConfig, apply_to_simulation_cfg, object_loader_kwargs_from_physix
    except Exception:
        from environments.reach_to_grasp.config import DEFAULT_SCENE, DEFAULT_CAMERA  # type: ignore
        from environments.reach_to_grasp.utils import design_scene  # type: ignore
        from environments.utils.object_loader import ObjectLoader, ObjectLoaderConfig, SpawnBounds  # type: ignore
        from environments.utils.physix import PhysicsConfig, apply_to_simulation_cfg, object_loader_kwargs_from_physix  # type: ignore

    # Setup simulation with physics config
    phys = PhysicsConfig(device=args_cli.device)
    sim_cfg = sim_utils.SimulationCfg(device=phys.device)
    apply_to_simulation_cfg(sim_cfg, phys)
    sim = sim_utils.SimulationContext(sim_cfg)
    
    if not args_cli.headless:
        sim.set_camera_view(DEFAULT_CAMERA.eye, DEFAULT_CAMERA.target)

    # Build scene and fetch robot
    scene_entities, scene_origins = design_scene(DEFAULT_SCENE)
    robot = scene_entities["kinova_j2n6s300"]

    # Spawn objects from Nucleus YCB
    spawned_paths = []
    id_to_label: dict[str, str] = {}
    if not args_cli.no_objects:
        # Resolve YCB path
        try:
            from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
            ycb_dir = f"{ISAAC_NUCLEUS_DIR}/Props/YCB"
        except Exception:
            ycb_dir = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Props/YCB"

        scale_range = None
        if args_cli.scale_min is not None and args_cli.scale_max is not None:
            scale_range = (float(args_cli.scale_min), float(args_cli.scale_max))

        # Configure object loader
        phys_loader_kwargs = object_loader_kwargs_from_physix(phys)
        loader_cfg = ObjectLoaderConfig(
            dataset_dirs=[ycb_dir],
            bounds=SpawnBounds(min_xyz=tuple(args_cli.spawn_min), max_xyz=tuple(args_cli.spawn_max)),
            min_distance=float(args_cli.min_distance),
            uniform_scale_range=scale_range,
            **phys_loader_kwargs,
        )
        
        loader = ObjectLoader(loader_cfg)
        spawned_paths = loader.spawn(parent_prim_path="/World/Origin1", num_objects=int(args_cli.num_objects))
        # Build id->label mapping from prim path->label (use basename of prim path as id)
        try:
            prim_to_label = loader.get_last_spawn_labels()
            id_to_label = {str(p).split("/")[-1]: str(lbl) for p, lbl in prim_to_label.items()}
        except Exception:
            id_to_label = {}

    # Reset simulation
    sim.reset()
    origin0 = torch.tensor(scene_origins[0], device=sim.device)
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += origin0
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    robot.reset()

    # Setup controller
    ctrl_cfg = CartesianVelocityJogConfig(
        ee_link_name=str(args_cli.ee_link),
        device=str(sim.device),
        use_relative_mode=True,
        linear_speed_mps=float(args_cli.speed),
        workspace_min=(0.20, -0.45, 0.01),
        workspace_max=(0.6, 0.45, 0.35),
        log_ee_pos=bool(args_cli.print_ee),
        log_ee_frame=str(args_cli.ee_frame),
        log_every_n_steps=int(args_cli.print_interval),
    )
    controller = CartesianVelocityJogController(ctrl_cfg, num_envs=1, device=str(sim.device))
    
    # Setup mode management
    mode_manager = ModeManager(initial_mode="translate")
    mode_manager.set_mode_change_callback(lambda mode: controller.set_mode(mode.value))
    controller.set_mode("translate")

    # Logging-only setup
    prim_paths = spawned_paths if not args_cli.no_objects and 'spawned_paths' in locals() else []
    tracker = ObjectsTracker(prim_paths=prim_paths)
    mux_input = CommandMuxInputProvider()
    controller.set_input_provider(mux_input)

    # Session logger
    tick_cfg = TickLoggingConfig(
        log_rate_hz=10,
        workspace_min=getattr(controller.config.safety_cfg, 'workspace_min', None),
        workspace_max=getattr(controller.config.safety_cfg, 'workspace_max', None),
        ee_link_name=str(args_cli.ee_link),
        arm_joint_regex=controller.config.arm_joint_regex,
    )
    session_logger = SessionLogWriter(root=Path("logs/data_collection"))
    # Write metadata.json
    session_logger.write_metadata(
        sim_dt=sim.get_physics_dt(),
        physics_substeps=int(getattr(sim.cfg, 'sub_steps', 4)),
        seed=0,
        robot_name="kinova_j2n6s300",
        ee_link=str(args_cli.ee_link),
        arm_joint_regex=controller.config.arm_joint_regex,
        log_rate_hz=tick_cfg.log_rate_hz,
        window_len_s=2.0,
    )

    if not args_cli.headless:
        keyboard = Se3KeyboardInput(
            pos_sensitivity_per_step=ctrl_cfg.linear_speed_mps * sim.get_physics_dt(),
            rot_sensitivity_rad_per_step=float(args_cli.rot_speed) * sim.get_physics_dt(),
        )
        mux_input.set_base(keyboard)
        translate_fn, rotate_fn, gripper_fn = mode_manager.get_mode_callbacks()
        keyboard.add_mode_callbacks(translate_fn, rotate_fn, gripper_fn)
        # Log mode changes
        def _on_mode_change(m):
            controller.set_mode(m.value)
            try:
                session_logger.log_event("mode_change", {"from": "unknown", "to": str(m)})
            except Exception:
                pass
        mode_manager.set_mode_change_callback(_on_mode_change)

    print("[INFO]: Setup complete... (Mode keys: F/f/1=translate, R/r/2=rotate, G/g/3=gripper)")
    run(sim, robot, controller, simulation_app, mux_input=mux_input, obj_tracker=tracker, session_logger=session_logger, tick_cfg=tick_cfg, id_to_label=id_to_label)
    simulation_app.close()


if __name__ == "__main__":
    main()