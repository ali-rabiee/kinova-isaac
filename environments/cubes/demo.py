"""Smoke-test demo for the cubes environment.

Mirrors :mod:`environments.ycb_reach_to_grasp.demo` (same starting joint pose,
same Cartesian-velocity jog controller, same keyboard input + mode manager)
but spawns colored cubes via :class:`CubesEnv` instead of YCB USD assets.

Run::

    ./IsaacLab/isaaclab.sh -p kinova-isaac/environments/cubes/demo.py --device cuda --num-objects 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
root_str = str(ROOT)
if root_str in sys.path:
    sys.path.remove(root_str)
sys.path.insert(0, root_str)
_env_mod = sys.modules.get("environments")
if _env_mod is not None and not hasattr(_env_mod, "__path__"):
    del sys.modules["environments"]

from controllers import (
    CartesianVelocityJogConfig,
    CartesianVelocityJogController,
    ModeManager,
    Se3KeyboardInput,
)
from data_collection.core.input_mux import CommandMuxInputProvider
from data_collection.core.logger import SessionLogWriter, TickLoggingConfig
from data_collection.core.objects import ObjectsTracker


def _run(sim, robot, controller, simulation_app, *, mux_input, obj_tracker, session_logger,
         tick_cfg, id_to_label):
    dt = sim.get_physics_dt()
    controller.reset(robot)
    accum = 0.0
    while simulation_app.is_running():
        controller.step(robot, dt)
        sim.step()
        robot.update(dt)

        if not (session_logger and obj_tracker and tick_cfg and mux_input):
            continue
        accum += dt
        period = 1.0 / float(tick_cfg.log_rate_hz)
        if accum + 1e-9 < period:
            continue
        accum = 0.0
        objs_raw = []
        try:
            for o in obj_tracker.snapshot():
                lbl = id_to_label.get(o.id, o.label) if id_to_label else o.label
                objs_raw.append({
                    "id": o.id,
                    "label": lbl,
                    "pose": {
                        "position_m": list(o.pose.position_m),
                        "orientation_wxyz": list(o.pose.orientation_wxyz),
                    },
                    "confidence": o.confidence,
                })
        except Exception as e:
            print(f"[LOG] Object snapshot failed: {e}")
        session_logger.write_tick(
            robot=robot,
            controller=controller,
            objects=objs_raw,
            last_user_cmd=mux_input.last_cmd,
            cfg=tick_cfg,
        )


def main():
    parser = argparse.ArgumentParser(description="Cubes Cartesian jog demo.")
    from scripts.cli import add_demo_cli_args

    add_demo_cli_args(parser)
    parser.add_argument(
        "--box-size",
        type=float,
        default=0.08,
        help="Uniform cube side length in meters.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    from environments.cubes import CubesEnv

    env = CubesEnv(device=args_cli.device, box_size=float(args_cli.box_size))
    sim = env.build_simulation()
    if not args_cli.headless:
        env.set_default_camera_view()
    env.design_scene()
    robot = env.robot

    spawned_paths: list[str] = []
    id_to_label: dict[str, str] = {}
    if not args_cli.no_objects:
        loader = env.build_object_loader(
            spawn_min=tuple(args_cli.spawn_min),
            spawn_max=tuple(args_cli.spawn_max),
            min_distance=float(args_cli.min_distance),
        )
        spawned_paths = loader.spawn(
            parent_prim_path="/World/Origin1",
            num_objects=int(args_cli.num_objects),
        )
        for prim in spawned_paths:
            label, color, idx = env.label_for_prim(prim)
            id_to_label[str(prim).split("/")[-1]] = label

    env.reset()

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

    mode_manager = ModeManager(initial_mode="translate")
    mode_manager.set_mode_change_callback(lambda mode: controller.set_mode(mode.value))
    controller.set_mode("translate")

    tracker = ObjectsTracker(prim_paths=spawned_paths)
    mux_input = CommandMuxInputProvider()
    controller.set_input_provider(mux_input)

    tick_cfg = TickLoggingConfig(
        log_rate_hz=10,
        workspace_min=getattr(controller.config.safety_cfg, "workspace_min", None),
        workspace_max=getattr(controller.config.safety_cfg, "workspace_max", None),
        ee_link_name=str(args_cli.ee_link),
        arm_joint_regex=controller.config.arm_joint_regex,
    )
    session_logger = SessionLogWriter(root=Path("logs/data_collection"))
    session_logger.write_metadata(
        sim_dt=sim.get_physics_dt(),
        physics_substeps=int(getattr(sim.cfg, "sub_steps", 4)),
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

        def _on_mode_change(m):
            controller.set_mode(m.value)
            try:
                session_logger.log_event("mode_change", {"from": "unknown", "to": str(m)})
            except Exception:
                pass
        mode_manager.set_mode_change_callback(_on_mode_change)

    print("[INFO]: Setup complete... (Mode keys: F/f/1=translate, R/r/2=rotate, G/g/3=gripper)")
    _run(sim, robot, controller, simulation_app,
         mux_input=mux_input, obj_tracker=tracker, session_logger=session_logger,
         tick_cfg=tick_cfg, id_to_label=id_to_label)
    simulation_app.close()


if __name__ == "__main__":
    main()
