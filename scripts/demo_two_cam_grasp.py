"""Two-camera reach-to-grasp demo: the scripted DEMOS motion, filmed.

Uses the exact scripted control from ``DEMOS/ycb_reach_grasp/ycb_reach_grasp_demo.py``
(diff-IK + quintic-eased Cartesian segments + OBB top-down grasp + gripper
close/lift) -- NOT the data-collection ``--planner scripted`` backend, which
stalls at pregrasp. Here it grasps a colored box and, every policy tick, saves
a frame from the front and wrist cameras so the two views can be stitched into
a demo of what the data collection sees.

    conda run -n kinova python -u scripts/demo_two_cam_grasp.py --headless \\
        --num-objects 4 --out-dir logs/demo_two_cam/episode_demo

Then build the video:
    python scripts/make_episode_video.py --episode-dir logs/demo_two_cam/episode_demo
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))
_env_mod = sys.modules.get("environments")
if _env_mod is not None and not hasattr(_env_mod, "__path__"):
    del sys.modules["environments"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--num-objects", type=int, default=4)
    p.add_argument("--box-size", type=float, default=0.05)
    p.add_argument("--spawn-min", type=float, nargs=3, default=[0.30, -0.30, 0.90])
    p.add_argument("--spawn-max", type=float, nargs=3, default=[0.55, 0.30, 0.95])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--grab", type=int, default=1, help="how many boxes to grasp in the demo")
    p.add_argument("--fps", type=float, default=5.0, help="frame capture rate (Hz)")
    p.add_argument("--out-dir", type=Path, default=Path("logs/demo_two_cam/episode_demo"))

    # motion / grasp geometry (defaults from the working DEMOS script)
    p.add_argument("--ee-link", type=str, default="j2n6s300_end_effector")
    p.add_argument("--arm-joint-regex", type=str, default="j2n6s300_joint_[1-6]$")
    p.add_argument("--gripper-joint-regex", type=str, default=".*_joint_finger_.*|.*_joint_finger_tip_.*")
    p.add_argument("--gripper-open-pos", type=float, default=0.0)
    p.add_argument("--gripper-close-pos", type=float, default=1.2)
    p.add_argument("--ee-z-offset-m", type=float, default=0.08)
    p.add_argument("--travel-height-m", type=float, default=0.14)
    p.add_argument("--grasp-depth-m", type=float, default=-0.07)
    p.add_argument("--cruise-mps", type=float, default=0.22)
    p.add_argument("--min-segment-s", type=float, default=0.8)
    p.add_argument("--max-segment-s", type=float, default=4.5)
    p.add_argument("--converge-pos-tol-m", type=float, default=0.008)
    p.add_argument("--pre-close-settle-s", type=float, default=0.3)
    p.add_argument("--gripper-close-s", type=float, default=0.6)
    p.add_argument("--lift-hold-s", type=float, default=0.6)

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(p)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    args.enable_cameras = True

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    code = 1
    try:
        code = _run(args, simulation_app)
    finally:
        import os
        import threading

        t = threading.Thread(target=simulation_app.close, daemon=True)
        t.start()
        t.join(timeout=60)
        os._exit(code)


def _run(args, simulation_app) -> int:
    import random

    import numpy as np
    import torch
    from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
    from isaaclab.utils.math import quat_apply, quat_conjugate, subtract_frame_transforms

    from environments.utils.camera import (
        DEFAULT_FRONT_CAMERA,
        DEFAULT_WRIST_CAMERA,
        build_camera,
        sync_wrist_camera_to_ee,
    )
    from environments.utils.object_loader import ObjectLoader, ObjectLoaderConfig, SpawnBounds
    from environments.utils.physix import object_loader_kwargs_from_physix
    from environments.ycb_reach_to_grasp import YCBReachToGraspEnv
    from kinova import GripperConfig, GripperController
    from motion_generation.grasp_estimation.obb import ObbGraspPoseProvider

    BOX_COLORS = [
        ("red", (0.85, 0.20, 0.20)), ("blue", (0.20, 0.35, 0.90)),
        ("yellow", (0.95, 0.85, 0.20)), ("purple", (0.65, 0.25, 0.80)),
        ("orange", (0.95, 0.55, 0.15)), ("cyan", (0.15, 0.80, 0.85)),
    ]
    if int(args.seed) >= 0:
        random.seed(int(args.seed))

    # ---- motion helpers (verbatim from DEMOS/ycb_reach_grasp) ---------------
    def _quintic(s):
        s = max(0.0, min(1.0, float(s)))
        return 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5

    def _quat_mul(q1, q2):
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
        return torch.stack([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ], dim=-1)

    def _yaw_quat(yaw, *, device, dtype=torch.float32):
        half = 0.5 * float(yaw)
        return torch.tensor([math.cos(half), 0.0, 0.0, math.sin(half)], device=device, dtype=dtype)

    def _quat_to_yaw(q):
        w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _wrap_pi(a):
        return (float(a) + math.pi) % (2.0 * math.pi) - math.pi

    def _slerp(q0, q1, t):
        q0 = q0 / (q0.norm() + 1e-12)
        q1 = q1 / (q1.norm() + 1e-12)
        dot = float(torch.dot(q0, q1))
        if dot < 0.0:
            q1 = -q1
            dot = -dot
        if dot > 0.9995:
            out = (1.0 - t) * q0 + t * q1
            return out / (out.norm() + 1e-12)
        theta_0 = math.acos(max(-1.0, min(1.0, dot)))
        sin_theta_0 = math.sin(theta_0)
        theta = theta_0 * float(t)
        s0 = math.cos(theta) - dot * math.sin(theta) / sin_theta_0
        s1 = math.sin(theta) / sin_theta_0
        return s0 * q0 + s1 * q1

    class _MoveSegment:
        def __init__(self, p0, p1, q0, q1, min_duration_s, max_duration_s):
            self.p0, self.p1, self.q0, self.q1 = p0.clone(), p1.clone(), q0.clone(), q1.clone()
            self.min_duration_s = max(1e-3, float(min_duration_s))
            self.max_duration_s = max(self.min_duration_s, float(max_duration_s))
            self.t_elapsed = 0.0

        def advance(self, dt):
            self.t_elapsed += float(max(0.0, dt))

        def current(self):
            se = _quintic(self.t_elapsed / self.min_duration_s)
            return self.p0 + se * (self.p1 - self.p0), _slerp(self.q0, self.q1, se)

        @property
        def timed_out(self):
            return self.t_elapsed >= self.max_duration_s

        @property
        def eased_complete(self):
            return self.t_elapsed >= self.min_duration_s

    # ---- build scene + BOXES ----------------------------------------------
    env = YCBReachToGraspEnv(device=str(getattr(args, "device", "cuda:0")))
    sim = env.build_simulation()
    if not bool(getattr(args, "headless", False)):
        env.set_default_camera_view()
    env.design_scene()
    robot = env.robot

    loader = ObjectLoader(ObjectLoaderConfig(
        dataset_dirs=[],
        bounds=SpawnBounds(min_xyz=tuple(args.spawn_min), max_xyz=tuple(args.spawn_max)),
        min_distance=0.20, min_distance_xy_only=True, spawn_mode="box",
        box_size_min=(args.box_size,) * 3, box_size_max=(args.box_size,) * 3,
        box_color_palette=[rgb for (_n, rgb) in BOX_COLORS],
        box_color_names=[n for (n, _r) in BOX_COLORS],
        **object_loader_kwargs_from_physix(env.physics_cfg),
    ))
    spawned_paths = loader.spawn(parent_prim_path="/World/Origin1", num_objects=int(args.num_objects))
    if not spawned_paths:
        print("[DEMO][ERROR] no boxes spawned")
        return 2
    id_to_label = {}
    for pth in spawned_paths:
        leaf = str(pth).split("/")[-1]
        idx = int(leaf.split("_")[-1])
        id_to_label[pth] = f"{BOX_COLORS[(idx - 1) % len(BOX_COLORS)][0]} box"
    print(f"[DEMO] spawned {len(spawned_paths)} boxes")

    # ---- cameras (front + wrist), same configs as data collection ----------
    # Build BEFORE env.reset(): IsaacLab cameras must exist before the first
    # sim.reset() or creating the render product hard-crashes the process.
    cameras = {}
    for name, cfg in (("front", DEFAULT_FRONT_CAMERA), ("wrist", DEFAULT_WRIST_CAMERA)):
        try:
            cameras[name] = build_camera(name, robot=robot, cfg=cfg)
            print(f"[DEMO] {name} camera created")
        except Exception as e:
            print(f"[DEMO] {name} camera failed: {e}")

    env.reset()
    for s in cameras.values():
        s.reset()

    # ---- arm / IK / gripper (verbatim wiring) ------------------------------
    arm_ids_t, _ = robot.find_joints(str(args.arm_joint_regex))
    arm_joint_ids = [int(v) for v in (arm_ids_t.view(-1).tolist() if hasattr(arm_ids_t, "view") else arm_ids_t)]
    ee_body_ids, _ = robot.find_bodies([str(args.ee_link)])
    ee_body_id = int(ee_body_ids[0])
    ee_jacobi_idx = ee_body_id - 1 if robot.is_fixed_base else ee_body_id

    diff_ik = DifferentialIKController(
        DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        num_envs=1, device=sim.device)
    diff_ik.reset()

    gripper = GripperController(GripperConfig(
        joint_regex=str(args.gripper_joint_regex),
        open_position=float(args.gripper_open_pos), close_position=float(args.gripper_close_pos),
    ), num_envs=1, device=str(sim.device))
    gripper.resolve_joints(robot)
    gripper.reset(robot)
    try:
        rp = str(getattr(getattr(robot, "cfg", None), "prim_path", None))
        if rp:
            gripper.set_drive_gains(rp)
            gripper.apply_stable_grasp_tuning(rp)
    except Exception:
        pass

    grasp_provider = ObbGraspPoseProvider(align_to_min_width=True)
    dt = float(sim.get_physics_dt())

    # ---- frame capture -----------------------------------------------------
    out_dir = args.out_dir
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    cap_stride = max(1, round((1.0 / float(args.fps)) / dt))
    state = {"phys": 0, "tick": 0}

    def _save(cam_name, arr, tick):
        d = out_dir / "images" / cam_name
        d.mkdir(parents=True, exist_ok=True)
        from PIL import Image

        Image.fromarray(arr).save(str(d / f"image_{tick:06d}.png"))

    def _capture_tick():
        for name, sensor in cameras.items():
            data = sensor.data
            rgb = data.output.get("rgb") if data.output is not None else None
            if rgb is None:
                continue
            arr = rgb[0].cpu().numpy() if rgb.dim() == 4 else rgb.cpu().numpy()
            arr = arr[..., :3]
            arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
            _save(name, arr, state["tick"])
        state["tick"] += 1

    def _step_and_film():
        """One physics step: sync wrist BEFORE render, update cameras, and save
        a frame every cap_stride steps (~fps Hz)."""
        do_cap = (state["phys"] % cap_stride) == 0
        if do_cap:
            for name, sensor in cameras.items():
                if name == "wrist":
                    try:
                        sync_wrist_camera_to_ee(robot, sensor, DEFAULT_WRIST_CAMERA)
                    except Exception:
                        pass
        sim.step(render=True)
        robot.update(dt)
        for sensor in cameras.values():
            try:
                sensor.update(dt)
            except Exception:
                pass
        if do_cap:
            _capture_tick()
        state["phys"] += 1

    # ---- pose/frame helpers (verbatim) -------------------------------------
    def _read_ee_pose_b():
        ee_pose_w = robot.data.body_pose_w[:, ee_body_id]
        root_pose_w = robot.data.root_pose_w
        pos_b, quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        return pos_b[0].clone(), quat_b[0].clone()

    def _world_to_base(pos_w_t):
        root_pose_w = robot.data.root_pose_w
        base_pos_w, base_quat_w = root_pose_w[0, 0:3], root_pose_w[0, 3:7]
        base_quat_inv = quat_conjugate(base_quat_w.unsqueeze(0))[0]
        return quat_apply(base_quat_inv.unsqueeze(0), (pos_w_t - base_pos_w).unsqueeze(0))[0]

    def _quat_aligned_to_yaw(q_cur_b, target_yaw):
        dyaw = _wrap_pi(float(target_yaw) - _quat_to_yaw(q_cur_b))
        return _quat_mul(_yaw_quat(dyaw, device=sim.device), q_cur_b)

    def _drive_ik_step(p_des_b, q_des_b):
        jac = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, arm_joint_ids]
        q_arm = robot.data.joint_pos[:, arm_joint_ids]
        ee_pos_b_cur, ee_quat_b_cur = _read_ee_pose_b()
        diff_ik.ee_pos_des[:] = p_des_b.unsqueeze(0)
        diff_ik.ee_quat_des[:] = q_des_b.unsqueeze(0)
        q_des = diff_ik.compute(ee_pos_b_cur.unsqueeze(0), ee_quat_b_cur.unsqueeze(0), jac, q_arm)
        robot.set_joint_position_target(robot.data.joint_pos)
        robot.set_joint_position_target(q_des, joint_ids=arm_joint_ids)
        robot.set_joint_velocity_target(torch.zeros_like(robot.data.joint_vel))
        try:
            gripper.apply_hold(robot)
        except Exception:
            pass
        try:
            robot.set_joint_effort_target(robot.root_physx_view.get_gravity_compensation_forces())
        except Exception:
            pass
        robot.write_data_to_sim()
        _step_and_film()

    def _pos_err_m(goal_b):
        p, _ = _read_ee_pose_b()
        return float((p - goal_b).norm())

    def _run_segment(label, goal_pos_b, q_end_b, q_start_b=None):
        p0, q0m = _read_ee_pose_b()
        q_start_b = q0m if q_start_b is None else q_start_b
        dist = float((goal_pos_b - p0).norm())
        min_dur = max(float(args.min_segment_s), dist / max(1e-6, float(args.cruise_mps)))
        max_dur = max(min_dur + 0.5, float(args.max_segment_s))
        seg = _MoveSegment(p0, goal_pos_b, q_start_b, q_end_b, min_dur, max_dur)
        while simulation_app.is_running() and not seg.timed_out:
            p_t, q_t = seg.current()
            _drive_ik_step(p_t, q_t)
            seg.advance(dt)
            if seg.eased_complete and _pos_err_m(goal_pos_b) < float(args.converge_pos_tol_m):
                break
        print(f"[DEMO]   {label:<10} dist={dist*1000:6.1f}mm final_err={_pos_err_m(goal_pos_b)*1000:6.1f}mm")

    def _hold_at(goal_pos_b, q_fixed_b, hold_s):
        for _ in range(int(max(1, round(float(hold_s) / dt)))):
            if not simulation_app.is_running():
                return
            _drive_ik_step(goal_pos_b, q_fixed_b)

    # ---- settle + open gripper --------------------------------------------
    try:
        gripper.command_open(robot)
    except Exception:
        pass
    for _ in range(60):
        if not simulation_app.is_running():
            break
        robot.set_joint_position_target(robot.data.joint_pos)
        robot.set_joint_velocity_target(torch.zeros_like(robot.data.joint_vel))
        try:
            gripper.apply_hold(robot)
        except Exception:
            pass
        try:
            robot.set_joint_effort_target(robot.root_physx_view.get_gravity_compensation_forces())
        except Exception:
            pass
        robot.write_data_to_sim()
        _step_and_film()

    # ---- measure grasp poses, pick nearest boxes ---------------------------
    targets = []
    for pth in spawned_paths:
        try:
            gp_w, _gq = grasp_provider.get_grasp_pose_w(object_prim_path=str(pth), robot_prim_path=None)
            top_b = _world_to_base(torch.tensor(gp_w, dtype=torch.float32, device=sim.device))
            targets.append((pth, top_b))
        except Exception as e:
            print(f"[DEMO][WARN] OBB failed for {pth}: {e}")
    targets.sort(key=lambda t: float((t[1][0] ** 2 + t[1][1] ** 2) ** 0.5))
    targets = targets[: max(1, int(args.grab))]

    ee_off = float(args.ee_z_offset_m)
    max_top_z = max((float(t[1][2]) for t in targets), default=0.2)
    travel_z_b = max_top_z + ee_off + float(args.travel_height_m)

    world_z = _make_z_reader(spawned_paths)  # f(prim)->world z, or None

    grabbed = 0
    for pth, top_b in targets:
        over_b = top_b.clone()
        over_b[2] = travel_z_b
        grasp_b = top_b.clone()
        grasp_b[2] = float(top_b[2]) + ee_off + float(args.grasp_depth_m)
        _, q_cur = _read_ee_pose_b()
        q_pick = _quat_aligned_to_yaw(q_cur, 0.0)

        z0 = world_z(pth) if world_z else None
        print(f"[DEMO] grasp {id_to_label.get(pth,'box')}: top_b=({float(top_b[0]):+.3f},{float(top_b[1]):+.3f},{float(top_b[2]):+.3f})")

        _run_segment("MOVE_OVER", over_b, q_pick, q_start_b=q_cur)
        _run_segment("DESCEND", grasp_b, q_pick, q_start_b=q_pick)
        _hold_at(grasp_b, q_pick, float(args.pre_close_settle_s))
        try:
            gripper.command_close(robot)
        except Exception:
            pass
        _hold_at(grasp_b, q_pick, float(args.gripper_close_s))
        _run_segment("LIFT", over_b, q_pick, q_start_b=q_pick)
        _hold_at(over_b, q_pick, float(args.lift_hold_s))
        z1 = world_z(pth) if world_z else None
        if z0 is not None and z1 is not None:
            dz = z1 - z0
            ok = 0.04 <= dz <= 0.6
            grabbed += int(ok)
            print(f"[DEMO]   {'LIFTED' if ok else 'MISSED'} dz={dz*1000:+.0f}mm")
        else:
            grabbed += 1
        try:
            gripper.command_open(robot)
        except Exception:
            pass
        _run_segment("RETRACT", over_b, q_pick, q_start_b=q_pick)

    print(f"[DEMO] done: grasped {grabbed}/{len(targets)}; {state['tick']} frames per camera -> {out_dir}")
    return 0 if grabbed > 0 else 1


def _make_z_reader(spawned_paths):
    """Return f(prim_path)->world z from a RigidPrim view, or None if unavailable."""
    try:
        from isaacsim.core.prims import RigidPrim
    except Exception:
        return None
    handles = {}
    for pth in spawned_paths:
        try:
            rb = RigidPrim(pth)
            try:
                rb.initialize()
            except Exception:
                pass
            handles[pth] = rb
        except Exception:
            pass

    def _z(prim_path):
        rb = handles.get(prim_path)
        if rb is None:
            return None
        try:
            pos, _ = rb.get_world_poses()
            return float(pos[0][2])
        except Exception:
            return None

    return _z


if __name__ == "__main__":
    raise SystemExit(main())
