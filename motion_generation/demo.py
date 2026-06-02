from __future__ import annotations

import argparse
import importlib
import sys
import random
from pathlib import Path
from typing import List, Tuple
from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from controllers import (  
    CartesianVelocityJogConfig,
    CartesianVelocityJogController,
)
from motion_generation.planners import PlannerContext, create_planner  
from controllers.input.waypoint_follower import WaypointFollowerInput  
from motion_generation.grasp_estimation.replicator import ReplicatorGraspProvider  
from motion_generation.mogen import MotionGenerationAgent
from motion_generation.planners.curobo_v2 import CuroboV2Planner
from utilities import (  
    enable_optional_planner_extensions,
    reset_robot_to_origin,
    get_ee_pos_base_frame,
    stabilize_with_hold,
)


def run_grasp_loop_demo(args: argparse.Namespace) -> int:
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Enable optional planner extensions
    enable_optional_planner_extensions()

    # Import modules requiring active app
    import isaaclab.sim as sim_utils  # noqa: F401  (still used for primitives spawning below)
    from environments.ycb_reach_to_grasp import DEFAULT_SCENE, YCBReachToGraspEnv
    from environments.utils.object_loader import ObjectLoader, ObjectLoaderConfig, SpawnBounds
    from environments.utils.physix import object_loader_kwargs_from_physix

    env = YCBReachToGraspEnv(device=args.device)
    phys = env.physics_cfg
    sim = env.build_simulation()
    if not args.headless:
        env.set_default_camera_view()

    print("[MG] App launched; building scene...")
    env.design_scene()
    scene_entities = env.scene_entities
    scene_origins = env.scene_origins
    robot = env.robot
    robot_prim_path = None
    try:
        robot_prim_path = str(getattr(getattr(robot, "cfg", None), "prim_path", None))
    except Exception:
        robot_prim_path = None

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
    print(f"[MG] Controller speed={ctrl_cfg.linear_speed_mps} m/s, ee_link={ctrl_cfg.ee_link_name}")
    controller = CartesianVelocityJogController(ctrl_cfg, num_envs=1, device=str(sim.device))

    # Planner
    # Planner configs live under: motion_generation/planners/planners_config/
    cfg_dir = str((Path(__file__).resolve().parent / "planners" / "planners_config").resolve())
    planner = create_planner(
        str(getattr(args, "planner", "scripted")),
        ctx=PlannerContext(
            base_frame="base_link",
            ee_link_name=str(ctrl_cfg.ee_link_name),
            urdf_path=None,
            config_dir=cfg_dir,
        ),
    )

    # Planner smoke test mode: initialize planner and run a single plan, then exit.
    # This avoids object spawning / Nucleus asset enumeration so you can quickly confirm
    # whether cuRobo is being used or whether the planner falls back to scripted.
    if bool(getattr(args, "planner_check_only", False)):
        try:
            target_b = tuple(getattr(args, "planner_check_target_b", [0.45, 0.0, 0.15]))
            target_pos_b = (float(target_b[0]), float(target_b[1]), float(target_b[2]))
        except Exception:
            target_pos_b = (0.45, 0.0, 0.15)

        print(
            f"[MG][PLANNER_CHECK] planner={getattr(args, 'planner', 'scripted')} "
            f"planner_type={type(planner).__name__} cfg_dir={cfg_dir} target_pos_b={target_pos_b}"
        )
        # Helpful internal state for cuRobo planners
        try:
            if hasattr(planner, "_available"):
                print(f"[MG][PLANNER_CHECK] planner._available={getattr(planner, '_available')}")
            if hasattr(planner, "_last_cfg_path"):
                print(f"[MG][PLANNER_CHECK] planner._last_cfg_path={getattr(planner, '_last_cfg_path')}")
        except Exception:
            pass
        waypoints = []
        try:
            if hasattr(planner, "plan_to_pose_b"):
                waypoints = planner.plan_to_pose_b(
                    target_pos_b=target_pos_b,
                    target_quat_b_wxyz=(1.0, 0.0, 0.0, 0.0),
                    pregrasp_offset_m=float(getattr(args, "pregrasp", 0.10)),
                    grasp_depth_m=0.0,
                    lift_height_m=float(getattr(args, "lift", 0.15)),
                )
            else:
                waypoints = planner.plan_waypoints_b(
                    target_pos_b=target_pos_b,
                    pregrasp_offset_m=float(getattr(args, "pregrasp", 0.10)),
                    grasp_depth_m=0.0,
                    lift_height_m=float(getattr(args, "lift", 0.15)),
                )
        except Exception as e:
            print(f"[MG][PLANNER_CHECK][WARN] Planning call failed: {e}")
            waypoints = []

        n = len(waypoints)
        print(f"[MG][PLANNER_CHECK] waypoint_count={n}")
        # Heuristic:
        # - curobo_v2 plan_to_pose_b usually returns MANY waypoints (downsampled to <=120) + final 3 appended
        # - fallback scripted returns 3 waypoints
        if "curobo" in args.planner.lower() or "vla" in args.planner.lower():
            if n <= 3:
                print("[MG][PLANNER_CHECK] Likely SCRIPTED fallback (<=3 waypoints). Check for [MG][CUROBO_V2][WARN] logs above.")
            else:
                print("[MG][PLANNER_CHECK] Likely cuRobo trajectory (many waypoints). Check for [MG][CUROBO_V2] MotionGen initialized / Planned logs above.")
        elif n <= 3:
            print("[MG][PLANNER_CHECK] Likely SCRIPTED fallback (<=3 waypoints). Check for [MG][CUROBO_V2][WARN] logs above.")
        else:
            print("[MG][PLANNER_CHECK] Likely cuRobo trajectory (many waypoints). Check for [MG][CUROBO_V2] MotionGen initialized / Planned logs above.")

        try:
            simulation_app.close()
        except Exception:
            pass
        return 0

    # Grasp provider selection
    default_rep_yaml = str((Path(cfg_dir) / "gripper_configs" / "j2n6s300_topdown.yaml").resolve())
    grasp_kind = str(getattr(args, "grasp", "obb")).lower()
    rep_yaml = getattr(args, "rep_config_yaml", default_rep_yaml if grasp_kind == "replicator" else None)
    grasp_provider = None
    if grasp_kind == "replicator":
        try:
            grasp_provider = ReplicatorGraspProvider(
                gripper_prim_path=getattr(args, "rep_gripper_prim_path", robot_prim_path),
                config_yaml_path=rep_yaml,
                sampler_config=None,
                max_candidates=int(getattr(args, "rep_max_candidates", 16)),
            )
            print("[MG] Grasp provider: Replicator")
        except Exception as e:
            print(f"[MG][WARN] Replicator unavailable ({e}); falling back to AABB.")
            grasp_kind = "obb"
    if grasp_kind != "replicator" or grasp_provider is None:
        try:
            obb_mod = importlib.import_module("motion_generation.grasp_estimation.obb")
            ProviderCls = getattr(obb_mod, "ObbGraspPoseProvider", None) or getattr(obb_mod, "OBBGraspPoseProvider", None)
            if ProviderCls is None:
                raise AttributeError("ObbGraspPoseProvider/OBBGraspPoseProvider not found")
            grasp_provider = ProviderCls(align_to_min_width=True)
            print("[MG] Grasp provider: OBB (minor-axis aligned)")
        except Exception as e:
            raise ImportError(f"[MG][ERROR] Failed to import OBB grasp provider: {e}")

    # Object loader defaults
    dataset_dirs = [str(d) for d in getattr(args, "objects_dataset", [])]
    use_prims = bool(getattr(args, "spawn_primitives", False))
    if use_prims:
        dataset_dirs = []
        print("[MG] Object spawning: primitives (no Nucleus/network).")
    elif len(dataset_dirs) == 0:
        try:
            from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # type: ignore
            ycb_dir = f"{ISAAC_NUCLEUS_DIR}/Props/YCB"
        except Exception:
            ycb_dir = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Props/YCB"
        dataset_dirs = [ycb_dir]
        print(f"[MG] Using default YCB dataset: {dataset_dirs[0]}")
    else:
        print(f"[MG] Using custom object datasets: {dataset_dirs}")
    
    loader = None
    if not use_prims:
        phys_loader_kwargs = object_loader_kwargs_from_physix(phys)
        loader_cfg = ObjectLoaderConfig(
            dataset_dirs=dataset_dirs,
            bounds=SpawnBounds(min_xyz=tuple(args.spawn_min), max_xyz=tuple(args.spawn_max)),
            min_distance=float(getattr(args, "min_distance", 0.1)),
            uniform_scale_range=(
                (float(args.scale_min), float(args.scale_max))
                if getattr(args, "scale_min", None) is not None and getattr(args, "scale_max", None) is not None
                else None
            ),
            include_labels=[
                "bleach_cleanser",
                "gelatin_box",
                "master_chef_can",
                "mug",
                "mustard_bottle",
                "potted_meat_can",
                "pudding_box",
                "sugar_box",
                "tomato_soup_can",
                "tuna_fish_can",
            ],
            **phys_loader_kwargs,
        )
        loader = ObjectLoader(loader_cfg)

    # High-level motion generation helper for label-based target selection and grasp queries
    if loader is None:
        class _DummyLoader:
            def get_last_spawn_labels(self):
                return {}
        loader_for_agent = _DummyLoader()
    else:
        loader_for_agent = loader
    agent = MotionGenerationAgent(
        sim=sim,
        robot=robot,
        controller=controller,
        planner=planner,
        grasp_provider=grasp_provider,
        loader=loader_for_agent,  # type: ignore[arg-type]
        robot_prim_path=robot_prim_path,
    )

    # Loop
    prev_prim_paths: List[str] = []
    num_episodes = int(getattr(args, "num_episodes", 3))
    pregrasp = float(getattr(args, "pregrasp", 0.10))
    grasp_depth = float(getattr(args, "grasp_depth", -0.04))
    lift_h = float(getattr(args, "lift", 0.15))
    tolerance_m = float(getattr(args, "tolerance", 0.005))
    planner_kind = str(getattr(args, "planner", "scripted")).lower()

    print(f"[MG] Starting grasp loop: episodes={num_episodes} grasp={grasp_kind} planner={getattr(args, 'planner', 'scripted')}")
    successes = 0
    for ep in range(num_episodes):
        print(f"[MG][EP {ep+1}/{num_episodes}] Resetting world and robot...")
        # Remove previously spawned objects
        try:
            import omni.usd  # type: ignore[attr-defined]
            stage = omni.usd.get_context().get_stage()
            for p in prev_prim_paths:
                try:
                    if stage.GetPrimAtPath(p).IsValid():
                        stage.RemovePrim(p)
                except Exception:
                    pass
        except Exception:
            pass
        prev_prim_paths = []

        # Reset sim and robot, then spawn
        sim.reset()
        reset_robot_to_origin(sim, robot, (float(scene_origins[0][0]), float(scene_origins[0][1]), float(scene_origins[0][2])))
        controller.reset(robot)
        try:
            n = int(getattr(args, "num_objects", 1))
        except Exception:
            n = 1
        if bool(getattr(args, "spawn_primitives", False)):
            prev_prim_paths = []
            try:
                import isaaclab.sim as sim_utils
                prim_utils = importlib.import_module("isaacsim.core.utils.prims")
                prim_utils.create_prim("/World/Origin1/Prims", "Xform")
                # Spawn obstacles to match the cuRobo world config in this repo
                # (motion_generation/planners/planners_config/cuRobo/world_demo.yml).
                # Positions are defined in base frame; convert to world by adding robot base height.
                base_h = float(getattr(DEFAULT_SCENE, "robot_base_height", 0.8))
                obstacles_b = [
                    ("Obstacle_00", (0.45, 0.00, 0.10), (0.10, 0.10, 0.20)),
                    ("Obstacle_01", (0.40, -0.15, 0.08), (0.08, 0.20, 0.12)),
                ]
                # Always spawn a separate visible target that is NOT part of the cuRobo world config.
                target_b = ("Target", (0.55, 0.15, 0.08), (0.06, 0.06, 0.10))

                # Obstacles (red)
                num_obs = int(getattr(args, "num_primitives", len(obstacles_b)))
                for name, pos_b, dims in obstacles_b[: max(0, num_obs)]:
                    p = f"/World/Origin1/Prims/{name}"
                    box_cfg = sim_utils.CuboidCfg(
                        size=tuple(dims),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.2, 0.2)),
                        collision_props=sim_utils.CollisionPropertiesCfg(),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    )
                    x, y, z = float(pos_b[0]), float(pos_b[1]), float(pos_b[2] + base_h)
                    box_cfg.func(p, box_cfg, translation=(x, y, z))
                    prev_prim_paths.append(p)

                # Target (green)
                t_name, t_pos_b, t_dims = target_b
                t_path = f"/World/Origin1/Prims/{t_name}"
                t_cfg = sim_utils.CuboidCfg(
                    size=tuple(t_dims),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.9, 0.2)),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                )
                tx, ty, tz = float(t_pos_b[0]), float(t_pos_b[1]), float(t_pos_b[2] + base_h)
                t_cfg.func(t_path, t_cfg, translation=(tx, ty, tz))
                prev_prim_paths.append(t_path)

                print(f"[MG][EP] Spawned primitive scene: obstacles={num_obs}, target={t_path}")
            except Exception as e:
                print(f"[MG][EP][WARN] Primitive spawn failed: {e}")
                prev_prim_paths = []
        else:
            try:
                assert loader is not None
                prev_prim_paths = loader.spawn(parent_prim_path="/World/Origin1", num_objects=n)
                print(f"[MG][EP] Spawned objects: {prev_prim_paths}")
            except Exception as e:
                print(f"[MG][EP][WARN] Object spawn failed: {e}")
                prev_prim_paths = []
        if len(prev_prim_paths) == 0:
            print("[MG][EP] No objects spawned; skipping episode.")
            continue

        # Get physics timestep
        dt = float(sim.get_physics_dt())

        # Allow objects to stabilize while holding robot at start position
        stabilize_steps = int(getattr(args, "stabilize_steps", 500))
        if stabilize_steps > 0:
            stabilize_with_hold(sim, robot, stabilize_steps, dt)

        # Select target
        target_prim = None
        if bool(getattr(args, "spawn_primitives", False)):
            # Prefer the explicitly spawned target prim
            cand = "/World/Origin1/Prims/Target"
            if cand in prev_prim_paths:
                target_prim = cand
        if target_prim is None:
            target_prim = random.choice(prev_prim_paths)
        pos_w, quat_wxyz_w, pos_b, quat_b = agent.compute_current_grasp_for_prim(target_prim)
        print(f"[MG][EP] Target prim: {target_prim}")
        print(f"[MG][EP] Grasp pose (world): pos={pos_w} quat(wxyz)={quat_wxyz_w}")
        print(f"[MG][EP] Grasp pose (base): pos={pos_b} quat_b(wxyz)={quat_b}")

        # Open gripper briefly
        controller.set_mode("gripper")
        inp = WaypointFollowerInput(
            step_pos_m=float(ctrl_cfg.linear_speed_mps) * dt,
            tol_m=tolerance_m,
            device=str(sim.device),
        )
        controller.set_input_provider(inp)
        inp.queue_gripper(+1.0, steps=10)
        for _ in range(10):
            controller.step(robot, dt)
            sim.step()
            robot.update(dt)

        if planner_kind == "scripted":
            # Delegate scripted motion phases (XY align, yaw rotate, descend) to the ScriptedPlanner
            from motion_generation.planners import ScriptedPlanner  # local import to avoid circular deps

            assert isinstance(planner, ScriptedPlanner), "Scripted planner expected when planner_kind='scripted'"
            planner.execute_scripted_motion(
                sim=sim,
                robot=robot,
                controller=controller,
                ctrl_cfg=ctrl_cfg,
                grasp_pos_b=pos_b,
                grasp_quat_b_wxyz=quat_b,
                grasp_quat_wxyz_w=quat_wxyz_w,
                dt=dt,
                tolerance_m=tolerance_m,
                inp=inp,
            )
        else:
            # Planner-driven approach + grasp
            try:
                # For cuRobo planners: update world model and set start state before planning
                if isinstance(planner, CuroboV2Planner):
                    # Update cuRobo's world model with current scene obstacles
                    if hasattr(planner, "update_world_from_prim_paths"):
                        # Exclude target from obstacles (cuRobo should plan around obstacles, not the target)
                        obstacle_paths = [p for p in prev_prim_paths if p != target_prim]
                        if len(obstacle_paths) > 0:
                            planner.update_world_from_prim_paths(sim=sim, robot=robot, prim_paths=obstacle_paths)
                    
                    # Set start state for planning (required by plan_to_pose_b)
                    if hasattr(planner, "set_start_state"):
                        # Get joint names from config
                        cfg_path = str(Path(cfg_dir) / "cuRobo" / "j2n6s300.yaml")
                        import yaml  # type: ignore
                        cfg_dict = yaml.safe_load(Path(cfg_path).read_text()) or {}
                        robot_cfg = (cfg_dict.get("robot_cfg") or {}).get("kinematics") or {}
                        cspace = robot_cfg.get("cspace") or {}
                        joint_names = cspace.get("joint_names") or []
                        if isinstance(joint_names, list) and len(joint_names) > 0:
                            # Build name->index mapping from IsaacLab articulation
                            robot_joint_names = [str(n) for n in getattr(robot.data, "joint_names", [])]
                            name_to_id = {n: i for i, n in enumerate(robot_joint_names)}
                            joint_ids = []
                            for jn in joint_names:
                                if str(jn) in name_to_id:
                                    joint_ids.append(int(name_to_id[str(jn)]))
                            if len(joint_ids) == len(joint_names):
                                # Current robot arm joint positions in cuRobo order
                                start_q = [float(robot.data.joint_pos[0, jid].item()) for jid in joint_ids]
                                planner.set_start_state(joint_pos=start_q, joint_names=[str(j) for j in joint_names])

                # Use plan_to_pose_b which handles cuRobo failures gracefully
                # (it will try to extract EE waypoints even from failed results)
                    if hasattr(planner, "plan_to_pose_b"):
                        waypoints = getattr(planner, "plan_to_pose_b")(
                            target_pos_b=pos_b,
                            target_quat_b_wxyz=quat_b,
                            pregrasp_offset_m=pregrasp,
                            grasp_depth_m=grasp_depth,
                            lift_height_m=lift_h,
                        )
                    else:
                        raise RuntimeError("6D planning interface not available")
            except Exception as e:
                print(f"[MG][EP][WARN] plan_to_pose_b failed ({e}); using position-only waypoints.")
                waypoints = planner.plan_waypoints_b(
                    target_pos_b=pos_b,
                    pregrasp_offset_m=pregrasp,
                    grasp_depth_m=grasp_depth,
                    lift_height_m=lift_h,
                )
            print(f"[MG][EP] Waypoints: {waypoints}")
            controller.set_mode("translate")
            inp.set_waypoints_b(waypoints[:2])
            steps = 0
            while True:
                inp.set_current_pose_b(get_ee_pos_base_frame(robot, ctrl_cfg.ee_link_name))
                controller.step(robot, dt)
                sim.step(); robot.update(dt)
                if len(inp._waypoints_b) == 0 or steps > 2000:
                    break
                steps += 1

        # Close gripper
        controller.set_mode("gripper")
        inp.queue_gripper(-1.0, steps=60)
        for _ in range(60):
            inp.set_current_pose_b(get_ee_pos_base_frame(robot, ctrl_cfg.ee_link_name))
            controller.step(robot, dt)
            sim.step(); robot.update(dt)

        # Lift
        controller.set_mode("translate")
        lift_pt = (pos_b[0], pos_b[1], pos_b[2] + lift_h)
        inp.set_waypoints_b([lift_pt])
        steps = 0
        while True:
            inp.set_current_pose_b(get_ee_pos_base_frame(robot, ctrl_cfg.ee_link_name))
            controller.step(robot, dt)
            sim.step(); robot.update(dt)
            if len(inp._waypoints_b) == 0 or steps > 2000:
                break
            steps += 1

        # Quick success heuristic: lifted z increased
        print(f"[MG][EP] Completed episode {ep+1}.")

    print(f"[MG] Completed: episodes={num_episodes}")
    simulation_app.close()
    return 0


if __name__ == "__main__":
    # Import CLI configuration from motion_generation.cli
    from motion_generation.cli import add_motion_gen_cli_args
    
    ap = argparse.ArgumentParser(description="Motion generation grasp loop demo")
    add_motion_gen_cli_args(ap)
    args_cli = ap.parse_args()
    raise SystemExit(run_grasp_loop_demo(args_cli))


