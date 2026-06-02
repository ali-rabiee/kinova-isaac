from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Optional

from data_collection.envs.registry import get_envs
from data_collection.profiles.spec import ProfileSpec


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--env", type=str, default="ycb_reach_to_grasp", choices=sorted(get_envs().keys()))
    parser.add_argument("--logs-root", type=str, default="logs/data_collection")
    parser.add_argument("--log-rate-hz", type=int, default=10)
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--control", type=str, default="keyboard", choices=["keyboard", "idle", "planner"])
    parser.add_argument("--image-format", type=str, default="png", choices=["png", "jpg"], help="Image format for saving")
    # Planner-driven reach-to-grasp (VLA)
    parser.add_argument(
        "--planner",
        type=str,
        default="curobo_v2",
        choices=["curobo_v2", "curobo", "lula", "rmpflow", "scripted"],
        help="Planner backend for --control planner (default: curobo_v2)",
    )
    parser.add_argument("--target-label", type=str, default=None, help="Optional target object label filter")
    parser.add_argument("--pregrasp", type=float, default=0.10, help="Pre-grasp offset above object top (m)")
    parser.add_argument("--lift", type=float, default=0.15, help="Lift height after grasp (m)")
    parser.add_argument("--tolerance", type=float, default=0.005, help="Waypoint tolerance for planner control (m)")
    parser.add_argument("--stabilize-steps", type=int, default=300, help="Hold steps before first plan (physics settle)")
    parser.add_argument("--gripper-open-steps", type=int, default=10, help="Steps to open gripper before approach")
    parser.add_argument("--gripper-close-steps", type=int, default=60, help="Steps to close gripper at grasp")
    # Note: --enable_cameras flag should be passed to AppLauncher for camera rendering
    # This is handled automatically by AppLauncher.add_app_launcher_args()


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

    # Heavy imports (torch / isaaclab / omni) must happen only after Kit is started.
    from isaaclab.app import AppLauncher
    from data_collection.core.input_mux import CommandMuxInputProvider
    from data_collection.core.logger import SessionLogWriter, TickLoggingConfig
    from data_collection.core.objects import ObjectsTracker

    # Note: For camera image capture, --enable_cameras flag must be passed to AppLauncher
    # This is handled by AppLauncher.add_app_launcher_args() in collect_data.py
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        import numpy as np  # noqa: E402
    except Exception as e:
        print(f"[VLA] ERROR: numpy is required for image saving but could not be imported: {e}")
        print("[VLA] Please install numpy in your environment (e.g., `pip install numpy`) and retry.")
        return 2

    # Isaac Lab's Camera sensor initialization gate checks this carb setting.
    # Root cause we observed: Camera was being created AFTER the sim started playing,
    # so its PLAY callback never fired => _is_initialized stayed False => Camera.reset() always fails.
    # Additionally, if /isaaclab/cameras_enabled is False, Camera._initialize_impl will raise.
    import carb  # noqa: E402

    carb_settings = carb.settings.get_settings()
    enable_cameras = bool(getattr(args, "enable_cameras", False))
    carb_settings.set_bool("/isaaclab/cameras_enabled", enable_cameras)
    print(f"[VLA] enable_cameras flag value: {enable_cameras}")
    print(f"[VLA] carb /isaaclab/cameras_enabled={carb_settings.get('/isaaclab/cameras_enabled')}")

    # Imports requiring active app
    import random
    from isaaclab.sensors import Camera, CameraCfg

    from controllers import CartesianVelocityJogConfig, CartesianVelocityJogController

    from environments.ycb_reach_to_grasp import (
        DEFAULT_TOP_DOWN_CAMERA,
        YCBReachToGraspEnv,
    )

    scale_range = None
    if getattr(args, "scale_min", None) is not None and getattr(args, "scale_max", None) is not None:
        scale_range = (float(args.scale_min), float(args.scale_max))

    env = YCBReachToGraspEnv(
        device=str(getattr(args, "device", "cuda:0")),
        scale_range=scale_range,
    )
    sim = env.build_simulation()
    phys = env.physics_cfg
    if not getattr(args, "headless", False):
        env.set_default_camera_view()

    env.design_scene()
    robot = env.robot

    env.attach_top_down_camera()
    print(f"[VLA] Top-down camera created at: {DEFAULT_TOP_DOWN_CAMERA.prim_path}")

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

    # Create camera sensor for image capture.
    #
    # CRITICAL (root cause): Sensors initialize lazily on the *timeline PLAY* event (SensorBase callback).
    # If we create the Camera AFTER the sim has already started playing (e.g., after sim.reset()),
    # its PLAY callback will never fire and it will never initialize.
    #
    # Therefore, create the Camera BEFORE calling sim.reset() (which transitions STOP->PLAY).
    camera_sensor = None
    if DEFAULT_TOP_DOWN_CAMERA is not None:
        try:
            camera_cfg = CameraCfg(
                prim_path=DEFAULT_TOP_DOWN_CAMERA.prim_path,
                offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
                spawn=None,  # Don't spawn, attach to existing prim
                data_types=["rgb"],
                width=DEFAULT_TOP_DOWN_CAMERA.resolution[0],
                height=DEFAULT_TOP_DOWN_CAMERA.resolution[1],
            )
            camera_sensor = Camera(cfg=camera_cfg)
            print(f"[VLA] Camera sensor created: {camera_cfg.width}x{camera_cfg.height}")
            print(f"[VLA] Attached to existing camera prim: {DEFAULT_TOP_DOWN_CAMERA.prim_path}")
        except Exception as create_err:
            print(f"[VLA] ERROR: Failed to create Camera object: {create_err}")
            try:
                camera_cfg = CameraCfg(
                    prim_path=DEFAULT_TOP_DOWN_CAMERA.prim_path,
                    offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
                    data_types=["rgb"],
                    width=DEFAULT_TOP_DOWN_CAMERA.resolution[0],
                    height=DEFAULT_TOP_DOWN_CAMERA.resolution[1],
                )
                camera_sensor = Camera(cfg=camera_cfg)
                print(f"[VLA] Camera sensor created with new prim: {camera_cfg.width}x{camera_cfg.height}")
            except Exception as create_err2:
                print(f"[VLA] ERROR: Failed to create Camera object: {create_err2}")
                import traceback
                traceback.print_exc()
                camera_sensor = None

    # Reset sim and robot (this transitions the timeline and triggers SensorBase PLAY callbacks).
    env.reset()

    # Now that the sim has played once, the Camera should be initialized via PLAY callback.
    # Reset the camera internals (timestamps/outdated flags) for clean logging.
    if camera_sensor is not None:
        try:
            camera_sensor.reset()
            print("[VLA] Camera sensor reset OK (post sim.reset)")
        except Exception as e:
            # If this fails, it's a strong indicator /isaaclab/cameras_enabled is false or PLAY callback failed.
            print(f"[VLA] ERROR: Camera sensor reset failed post sim.reset: {e}")
            camera_sensor = None

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
        # Planner-driven reach-to-grasp control.
        # Uses MotionGen-style planners (prefer cuRobo) and executes via WaypointFollowerInput.
        from pathlib import Path as _Path

        from controllers.input.waypoint_follower import WaypointFollowerInput
        from motion_generation.grasp_estimation.obb import ObbGraspPoseProvider
        from motion_generation.mogen import MotionGenerationAgent
        from motion_generation.planners import PlannerContext, create_planner
        from utilities import get_ee_pos_base_frame

        if "loader" not in locals() or loader is None:  # type: ignore[name-defined]
            print("[VLA][PLANNER] ERROR: planner control requires object spawning (loader unavailable).")
            print("[VLA][PLANNER] Run without --no-objects and with --num-objects >= 1.")
            simulation_app.close()
            return 2

        # Planner context config dir: use the repo's bundled planner configs
        cfg_dir = str(
            (_Path(__file__).resolve().parents[2] / "motion_generation" / "planners" / "planners_config").resolve()
        )
        planner = create_planner(
            str(getattr(args, "planner", "curobo_v2")),
            ctx=PlannerContext(
                base_frame="base_link",
                ee_link_name=str(getattr(args, "ee_link", "j2n6s300_end_effector")),
                urdf_path=str((_Path(cfg_dir) / "cuRobo" / "kinovaJacoJ2N6S300.urdf").resolve()),
                config_dir=cfg_dir,
            ),
        )
        print(f"[VLA][PLANNER] Control enabled. planner={getattr(args, 'planner', 'curobo_v2')} cfg_dir={cfg_dir}")
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
        wp = WaypointFollowerInput(
            step_pos_m=float(ctrl_cfg.linear_speed_mps) * dt_local,
            tol_m=float(getattr(args, "tolerance", 0.005)),
            device=str(sim.device),
        )
        mux_input.set_base(wp)

        # Minimal internal state machine for reach->grasp->lift
        vla_planner_state = {
            "stage": "init_open",  # init_open -> approach -> close -> lift -> idle
            "target_prim": None,
            "lift_pt": None,
            "open_left": int(getattr(args, "gripper_open_steps", 10)),
            "close_left": int(getattr(args, "gripper_close_steps", 60)),
            "stabilize_left": int(getattr(args, "stabilize_steps", 300)),
        }
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
    
    # Create images directory for this session
    images_dir = session_logger.root / "images"
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
    )

    print(f"[VLA] Data collection started!")
    print(f"[VLA] Session directory: {session_logger.root}")
    print(f"[VLA] Images directory: {images_dir}")
    print(f"[VLA] Logging rate: {tick_cfg.log_rate_hz} Hz")
    print(f"[VLA] Duration: {getattr(args, 'duration_s', 30.0)} seconds")
    print(f"[VLA] Control mode: {getattr(args, 'control', 'keyboard')}")
    if str(getattr(args, "control", "keyboard")) == "keyboard":
        print(f"[VLA] Use keyboard to control the robot (WASD for translation, QE for rotation)")
    print(f"[VLA] Camera sensor: {'Active' if camera_sensor is not None else 'Inactive'}")
    print(f"[VLA] Starting data collection loop...\n")

    # Run
    dt = float(sim.get_physics_dt())
    period = 1.0 / float(tick_cfg.log_rate_hz)
    accum = 0.0
    t0 = time.time()
    last_progress_print = t0
    images_captured = 0
    while simulation_app.is_running() and (time.time() - t0) < float(getattr(args, "duration_s", 30.0)):
        # Planner-driven reach-to-grasp state machine (optional)
        if control_mode == "planner":
            try:
                # Update EE pose for waypoint tracking
                wp.set_current_pose_b(get_ee_pos_base_frame(robot, str(getattr(args, "ee_link", "j2n6s300_end_effector"))))
            except Exception:
                pass

            try:
                # Optional stabilization window before first plan (lets physics settle)
                stab_left = int(vla_planner_state.get("stabilize_left", 0))
                if stab_left > 0:
                    vla_planner_state["stabilize_left"] = stab_left - 1
                else:
                    stage = str(vla_planner_state.get("stage", "idle"))

                    # Open gripper once at the beginning
                    if stage == "init_open":
                        open_left = int(vla_planner_state.get("open_left", 0))
                        if not vla_planner_state.get("open_queued", False):
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
                            vla_planner_state["stage"] = "idle"

                    # Plan + execute approach/grasp
                    if str(vla_planner_state.get("stage", "idle")) == "idle":
                        if len(spawned_paths) == 0:
                            # No targets; do nothing.
                            pass
                        else:
                            target_label = getattr(args, "target_label", None)
                            target_prim, _pos_w, _quat_wxyz_w, pos_b, quat_b = agent.compute_current_grasp_for_label(
                                label=target_label,
                                prim_paths=spawned_paths,
                            )
                            waypoints = planner.plan_to_pose_b(
                                target_pos_b=pos_b,
                                target_quat_b_wxyz=quat_b,
                                pregrasp_offset_m=float(getattr(args, "pregrasp", 0.10)),
                                grasp_depth_m=0.00,
                                lift_height_m=float(getattr(args, "lift", 0.15)),
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

                            vla_planner_state["target_prim"] = target_prim
                            vla_planner_state["lift_pt"] = lift_pt
                            vla_planner_state["stage"] = "approach"
                            vla_planner_state["lift_queued"] = False

                            controller.set_mode("translate")
                            wp.set_waypoints_b([(float(x), float(y), float(z)) for (x, y, z) in approach_pts])

                    # When approach completes, close gripper
                    if str(vla_planner_state.get("stage", "")) == "approach":
                        if len(getattr(wp, "_waypoints_b", [])) == 0:
                            vla_planner_state["stage"] = "close"
                            vla_planner_state["close_queued"] = False
                            vla_planner_state["close_left"] = int(getattr(args, "gripper_close_steps", 60))

                    if str(vla_planner_state.get("stage", "")) == "close":
                        close_left = int(vla_planner_state.get("close_left", 0))
                        if not vla_planner_state.get("close_queued", False):
                            try:
                                controller.set_mode("gripper")
                                wp.queue_gripper(-1.0, steps=close_left)
                            except Exception:
                                pass
                            vla_planner_state["close_queued"] = True
                        controller.set_mode("gripper")
                        if close_left > 0:
                            vla_planner_state["close_left"] = close_left - 1
                        else:
                            vla_planner_state["stage"] = "lift"
                            vla_planner_state["lift_queued"] = False

                    # Lift after grasp
                    if str(vla_planner_state.get("stage", "")) == "lift":
                        lift_pt = vla_planner_state.get("lift_pt", None)
                        if lift_pt is not None and not vla_planner_state.get("lift_queued", False):
                            controller.set_mode("translate")
                            wp.set_waypoints_b([(float(lift_pt[0]), float(lift_pt[1]), float(lift_pt[2]))])
                            vla_planner_state["lift_queued"] = True
                        # When lift completes, go idle (ready to plan next)
                        if (
                            lift_pt is not None
                            and vla_planner_state.get("lift_queued", False)
                            and len(getattr(wp, "_waypoints_b", [])) == 0
                        ):
                            vla_planner_state["stage"] = "idle"
                            vla_planner_state["target_prim"] = None
                            vla_planner_state["lift_pt"] = None
            except Exception as e:
                # Planner errors should not crash data collection.
                if session_logger.tick_idx < 3:
                    print(f"[VLA][PLANNER][WARN] Planner control error: {e}")

        controller.step(robot, dt)
        sim.step(render=True)
        robot.update(dt)
        
        # Update camera sensor (only if it has update method and is properly initialized)
        if camera_sensor is not None:
            try:
                if hasattr(camera_sensor, 'update'):
                    camera_sensor.update(dt)
            except Exception as e:
                # Silently fail - camera might update automatically via sim.step()
                pass
        
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
            
            # Capture image if camera sensor is available
            image_path = None
            if camera_sensor is not None:
                try:
                    # Isaac Lab Camera returns a CameraData object.
                    # The actual images are stored in CameraData.output dict, keyed by data_types (e.g. "rgb").
                    cam_data = camera_sensor.data
                    rgb_data = None
                    if cam_data.output is not None:
                        rgb_data = cam_data.output.get("rgb")
                    if rgb_data is None:
                        if session_logger.tick_idx == 0:
                            keys = []
                            try:
                                keys = list((cam_data.output or {}).keys())
                            except Exception:
                                keys = []
                            print(f"[VLA] Camera output has no 'rgb' yet. Available keys: {keys}")
                    if rgb_data is not None:
                        # First successful image capture
                        if images_captured == 0:
                            print(f"[VLA] First image captured! RGB shape: {rgb_data.shape}")
                        # Check data shape - could be [num_envs, height, width, channels] or [height, width, channels]
                        if len(rgb_data.shape) == 4:
                            # Shape: [num_envs, height, width, channels]
                            rgb_np = rgb_data[0].cpu().numpy()
                        elif len(rgb_data.shape) == 3:
                            # Shape: [height, width, channels] - single env
                            rgb_np = rgb_data.cpu().numpy()
                        else:
                            raise ValueError(f"Unexpected RGB data shape: {rgb_data.shape}")
                        
                        # Ensure values are in [0, 255] range
                        if rgb_np.max() <= 1.0:
                            rgb_np = (rgb_np * 255).astype(np.uint8)
                        else:
                            rgb_np = rgb_np.astype(np.uint8)
                        
                        # Save image
                        image_filename = f"image_{session_logger.tick_idx:06d}.{image_format}"
                        image_path = images_dir / image_filename
                        
                        # Use PIL or cv2 to save
                        try:
                            from PIL import Image
                            img = Image.fromarray(rgb_np)
                            img.save(str(image_path))
                        except Exception:
                            try:
                                import cv2
                                # OpenCV uses BGR, so convert if needed
                                if len(rgb_np.shape) == 3 and rgb_np.shape[2] == 3:
                                    rgb_np_bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
                                    cv2.imwrite(str(image_path), rgb_np_bgr)
                                else:
                                    cv2.imwrite(str(image_path), rgb_np)
                            except Exception:
                                # Fallback: save as numpy array
                                np.save(str(image_path).replace(f".{image_format}", ".npy"), rgb_np)
                                image_path = image_path.with_suffix(".npy")
                        
                        # Store relative path in tick data
                        image_path = f"images/{image_filename}"
                        images_captured += 1
                    else:
                        # rgb_data is None - camera may not have rendered yet
                        if session_logger.tick_idx < 5:
                            print(f"[VLA] Camera rgb_data is None at tick {session_logger.tick_idx}")
                except (AttributeError, RuntimeError) as e:
                    # Camera data property raises if camera isn't initialized
                    # This is expected if reset() failed or camera isn't ready yet
                    if session_logger.tick_idx == 0:
                        print(f"[VLA] Camera not initialized yet at tick 0: {type(e).__name__}: {e}")
                        print(f"[VLA] Camera will be tried again in subsequent ticks...")
                    image_path = None
                except Exception as e:
                    # Unexpected error
                    if session_logger.tick_idx < 5:
                        print(f"[VLA] Unexpected error capturing image at tick {session_logger.tick_idx}: {e}")
                        import traceback
                        traceback.print_exc()
                    image_path = None
            
            # Write tick with image path
            session_logger.write_tick(
                robot=robot,
                controller=controller,
                objects=objs_raw,
                last_user_cmd=mux_input.last_cmd,
                cfg=tick_cfg,
                image_path=image_path,  # Pass image path to logger
            )
            
            # Print progress every 5 seconds
            elapsed = time.time() - t0
            if time.time() - last_progress_print >= 5.0:
                remaining = float(getattr(args, "duration_s", 30.0)) - elapsed
                print(f"[VLA] Progress: {elapsed:.1f}s / {getattr(args, 'duration_s', 30.0):.1f}s | "
                      f"Ticks: {session_logger.tick_idx} | Images: {images_captured} | "
                      f"Remaining: {remaining:.1f}s")
                last_progress_print = time.time()

    elapsed_total = time.time() - t0
    print(f"\n[VLA] Data collection completed!")
    print(f"[VLA] Total time: {elapsed_total:.1f}s")
    print(f"[VLA] Total ticks logged: {session_logger.tick_idx}")
    print(f"[VLA] Total images captured: {images_captured}")
    print(f"[VLA] Session data saved to: {session_logger.root}")
    print(f"[VLA] - metadata.json")
    print(f"[VLA] - ticks.jsonl ({session_logger.tick_idx} entries)")
    print(f"[VLA] - images/ ({images_captured} images)")
    
    simulation_app.close()
    try:
        session_logger.close()
    except Exception:
        pass
    return 0


PROFILE = ProfileSpec(
    name="vla_v0",
    add_cli_args=add_cli_args,
    run=run,
)

