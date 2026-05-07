from __future__ import annotations
import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher
from controllers import ModeManager

# Ensure project root on sys.path for modular imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Import from our modular controller library
from controllers import (
    CartesianVelocityJogConfig,
    CartesianVelocityJogController,
    Se3KeyboardInput,
    ModeManager,
)


def run(sim, robot, controller: CartesianVelocityJogController, simulation_app):
    dt = sim.get_physics_dt()
    controller.reset(robot)
    print("[INFO] Cartesian velocity jog running. In headless mode, commands default to zero.")
    while simulation_app.is_running():
        controller.step(robot, dt)
        sim.step()
        robot.update(dt)


def main():
    parser = argparse.ArgumentParser(description="Cartesian velocity jog demo using modular controller.")
    parser.add_argument("--ee_link", type=str, default="j2n6s300_end_effector", help="End-effector link name")
    parser.add_argument("--speed", type=float, default=0.20, help="Linear speed (m/s)")
    parser.add_argument("--rot-speed", type=float, default=0.5, help="Angular speed (rad/s)")
    # EE logging controls
    parser.add_argument("--print-ee", action="store_true", help="Print EE XYZ each step")
    parser.add_argument("--ee-frame", type=str, default="world", choices=["world", "base"], help="Frame for EE logging")
    parser.add_argument("--print-interval", type=int, default=1, help="Print every N steps")
    # Mode handling
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # Import modules that require an active Omniverse app after AppLauncher
    from environments.ycb_reach_to_grasp import YCBReachToGraspEnv

    env = YCBReachToGraspEnv(device=args_cli.device)
    sim = env.build_simulation()
    if not args_cli.headless:
        env.set_default_camera_view()
    env.design_scene()
    robot = env.robot
    env.reset()

    # Controller setup
    ctrl_cfg = CartesianVelocityJogConfig(
        ee_link_name=str(args_cli.ee_link),
        device=str(sim.device),
        use_relative_mode=True,
        linear_speed_mps=float(args_cli.speed),
        # Per-axis workspace bounds in base frame (meters)
        workspace_min=(0.20, -0.45, 0.01),
        workspace_max=(0.6, 0.45, 0.35),
        log_ee_pos=bool(args_cli.print_ee),
        log_ee_frame=str(args_cli.ee_frame),
        log_every_n_steps=int(args_cli.print_interval),
    )
    controller = CartesianVelocityJogController(ctrl_cfg, num_envs=1, device=str(sim.device))
    
    # Mode management setup
    mode_manager = ModeManager(initial_mode="translate")
    mode_manager.set_mode_change_callback(lambda mode: controller.set_mode(mode.value))
    controller.set_mode("translate")  # Set initial mode

    if not args_cli.headless:
        keyboard = Se3KeyboardInput(
            pos_sensitivity_per_step=ctrl_cfg.linear_speed_mps * sim.get_physics_dt(),
            rot_sensitivity_rad_per_step=float(args_cli.rot_speed) * sim.get_physics_dt(),
        )
        controller.set_input_provider(keyboard)

        # Setup mode switching callbacks
        translate_fn, rotate_fn, gripper_fn = mode_manager.get_mode_callbacks()
        keyboard.add_mode_callbacks(translate_fn, rotate_fn, gripper_fn)

    print("[INFO]: Setup complete... (Mode keys: F/f/1=translate, R/r/2=rotate, G/g/3=gripper). Start mode= translate")
    run(sim, robot, controller, simulation_app)
    simulation_app.close()


if __name__ == "__main__":
    main()


