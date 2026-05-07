"""Smoke-test demo for the pouring environment.

Mirrors :mod:`environments.cubes.demo` and :mod:`environments.ycb_reach_to_grasp.demo`
end-to-end (same starting joint pose, same Cartesian-velocity jog controller,
same keyboard input + mode manager) but spawns a YCB pitcher, a YCB mug, and
a configurable number of rigid-sphere pellets via :class:`PouringEnv`.

Pour amount is printed to stdout once a second so you can see the volume
sensor working as you tilt the pitcher with the robot.

Run::

    ./IsaacLab/isaaclab.sh -p kinova-isaac/environments/pouring/demo.py --device cuda
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


def _run(sim, robot, env, controller, simulation_app, *, mux_input, obj_tracker, session_logger, tick_cfg, id_to_label, pour_log_period_s: float):
    dt = sim.get_physics_dt()
    controller.reset(robot)
    accum_log = 0.0
    accum_pour = 0.0
    while simulation_app.is_running():
        controller.step(robot, dt)
        sim.step()
        robot.update(dt)

        accum_pour += dt
        if accum_pour + 1e-9 >= pour_log_period_s:
            accum_pour = 0.0
            count = env.count_pellets_in_glass()
            grams = env.amount_poured_grams()
            print(f"[POUR] in_glass={count}/{len(env.pellet_prim_paths)} (~{grams:.1f} g)")

        if not (session_logger and obj_tracker and tick_cfg and mux_input):
            continue
        accum_log += dt
        period = 1.0 / float(tick_cfg.log_rate_hz)
        if accum_log + 1e-9 < period:
            continue
        accum_log = 0.0
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
    parser = argparse.ArgumentParser(description="Pouring task Cartesian jog demo.")
    from scripts.cli import add_demo_cli_args

    add_demo_cli_args(parser)
    parser.add_argument("--num-pellets", type=int, default=60, help="Number of pellets pre-filled in the pitcher.")
    parser.add_argument("--pellet-radius-m", type=float, default=0.006, help="Pellet radius in meters.")
    parser.add_argument(
        "--pellet-color",
        type=float,
        nargs=3,
        default=[0.10, 0.45, 0.95],
        metavar=("R", "G", "B"),
        help="Pellet (water) RGB color in [0, 1].",
    )

    parser.add_argument(
        "--pitcher-pos",
        type=float,
        nargs=3,
        default=[0.40, 0.18, 0.83],
        metavar=("X", "Y", "Z"),
        help="Pitcher world spawn position (bottom-center).",
    )
    parser.add_argument(
        "--pitcher-radius-m",
        type=float,
        default=0.030,
        help="Pitcher *interior* radius (m). Outer radius = inner + wall thickness.",
    )
    parser.add_argument(
        "--pitcher-height-m",
        type=float,
        default=0.12,
        help="Pitcher cavity height (m).",
    )
    parser.add_argument(
        "--pitcher-color",
        type=float,
        nargs=3,
        default=[0.20, 0.45, 0.95],
        metavar=("R", "G", "B"),
        help="Pitcher RGB color in [0, 1].",
    )

    parser.add_argument(
        "--glass-pos",
        type=float,
        nargs=3,
        default=[0.40, -0.18, 0.83],
        metavar=("X", "Y", "Z"),
        help="Glass world spawn position (bottom-center).",
    )
    parser.add_argument(
        "--glass-radius-m",
        type=float,
        default=0.035,
        help="Glass *interior* radius (m). Outer radius = inner + wall thickness.",
    )
    parser.add_argument(
        "--glass-height-m",
        type=float,
        default=0.08,
        help="Glass cavity height (m).",
    )
    parser.add_argument(
        "--glass-color",
        type=float,
        nargs=3,
        default=[0.80, 0.95, 0.98],
        metavar=("R", "G", "B"),
        help="Glass RGB color in [0, 1].",
    )

    parser.add_argument(
        "--wall-thickness-m",
        type=float,
        default=0.005,
        help="Wall + base thickness for both containers (m).",
    )
    parser.add_argument(
        "--n-segments",
        type=int,
        default=16,
        help="Number of polygon wall slabs around each cylinder (more = rounder).",
    )
    parser.add_argument(
        "--volume-half-extents",
        type=float,
        nargs=3,
        default=[0.040, 0.040, 0.045],
        metavar=("HX", "HY", "HZ"),
        help="Half-extents of the volume-sensor box anchored on the glass.",
    )
    parser.add_argument("--pour-log-period-s", type=float, default=1.0, help="Print pour count every N seconds.")
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    from environments.pouring import (
        GlassConfig,
        PelletConfig,
        PitcherConfig,
        PouringEnv,
        VolumeSensorConfig,
    )
    from environments.pouring.config import DEFAULT_GLASS, DEFAULT_PITCHER

    pitcher_cfg = PitcherConfig(
        spawn_position=tuple(args_cli.pitcher_pos),
        inner_radius_m=float(args_cli.pitcher_radius_m),
        inner_height_m=float(args_cli.pitcher_height_m),
        wall_thickness_m=float(args_cli.wall_thickness_m),
        base_thickness_m=float(args_cli.wall_thickness_m),
        n_segments=int(args_cli.n_segments),
        color_rgb=tuple(args_cli.pitcher_color),
        mass_kg=DEFAULT_PITCHER.mass_kg,
    )
    glass_cfg = GlassConfig(
        spawn_position=tuple(args_cli.glass_pos),
        inner_radius_m=float(args_cli.glass_radius_m),
        inner_height_m=float(args_cli.glass_height_m),
        wall_thickness_m=float(args_cli.wall_thickness_m),
        base_thickness_m=float(args_cli.wall_thickness_m),
        n_segments=int(args_cli.n_segments),
        color_rgb=tuple(args_cli.glass_color),
        mass_kg=DEFAULT_GLASS.mass_kg,
    )
    pellet_cfg = PelletConfig(
        count=int(args_cli.num_pellets),
        radius_m=float(args_cli.pellet_radius_m),
        color_rgb=tuple(args_cli.pellet_color),
    )
    volume_cfg = VolumeSensorConfig(
        half_extents_xyz=tuple(args_cli.volume_half_extents),
    )

    env = PouringEnv(
        device=args_cli.device,
        pitcher_cfg=pitcher_cfg,
        glass_cfg=glass_cfg,
        pellet_cfg=pellet_cfg,
        volume_sensor_cfg=volume_cfg,
    )
    sim = env.build_simulation()
    if not args_cli.headless:
        env.set_default_camera_view()
    env.design_scene()
    robot = env.robot

    pitcher_path, glass_path = env.spawn_containers()
    pellet_paths = env.spawn_pellets()
    print(f"[POUR] pitcher={pitcher_path} glass={glass_path} pellets={len(pellet_paths)}")

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

    # Track containers (not pellets -- there can be a lot of them and the
    # logger isn't the right channel for the pour count).
    tracker = ObjectsTracker(prim_paths=[pitcher_path, glass_path])
    id_to_label: dict[str, str] = {
        str(pitcher_path).split("/")[-1]: "pitcher",
        str(glass_path).split("/")[-1]: "glass",
    }
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
    _run(
        sim, robot, env, controller, simulation_app,
        mux_input=mux_input,
        obj_tracker=tracker,
        session_logger=session_logger,
        tick_cfg=tick_cfg,
        id_to_label=id_to_label,
        pour_log_period_s=float(args_cli.pour_log_period_s),
    )
    simulation_app.close()


if __name__ == "__main__":
    main()
