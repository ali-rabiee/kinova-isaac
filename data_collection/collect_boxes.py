"""Solid box reach-to-grasp data collection for the frozen diffusion policy.

Drives the PROVEN scripted diff-IK grasp from ``DEMOS/ycb_reach_grasp`` (the
data-collection ``--planner scripted`` backend stalls at pregrasp), films it
from the front + wrist cameras, and logs each episode in the exact format the
``lsteer`` zarr converter expects:

    <logs-root>/session_<ts>/episode_XXXX/
        metadata.json  instruction.json  ticks.jsonl  events.jsonl
        images/front/image_000000.png  images/wrist/image_000000.png  ...

Per episode: reset arm to home, pick the round-robin target box, grasp+lift it,
log ticks at 5 Hz, and record grasp_result{ok} from a physics lift check. Box
layout is re-randomized once per full target cycle (every --num-objects
episodes) so each layout yields one demo per box -- the multimodal pairing the
frozen-policy work needs.

    conda run -n kinova python -u -m data_collection.collect_boxes \\
        --headless --num-objects 4 --num-episodes 40 \\
        --logs-root logs/boxes_v0 --seed 0
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


BOX_COLORS = [
    ("red", (0.85, 0.20, 0.20)), ("blue", (0.20, 0.35, 0.90)),
    ("yellow", (0.95, 0.85, 0.20)), ("purple", (0.65, 0.25, 0.80)),
    ("orange", (0.95, 0.55, 0.15)), ("cyan", (0.15, 0.80, 0.85)),
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--num-objects", type=int, default=4)
    p.add_argument("--num-episodes", type=int, default=40)
    p.add_argument("--box-size", type=float, default=0.05)
    p.add_argument("--spawn-min", type=float, nargs=3, default=[0.30, -0.30, 0.90])
    p.add_argument("--spawn-max", type=float, nargs=3, default=[0.55, 0.30, 0.95])
    p.add_argument("--min-distance", type=float, default=0.14)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-rate-hz", type=int, default=5)
    p.add_argument("--logs-root", type=str, default="logs/boxes_v0")
    p.add_argument("--respawn-per-cycle", dest="respawn_per_cycle", action="store_true", default=True,
                   help="re-randomize the box layout every num-objects episodes (default on)")
    p.add_argument("--no-respawn", dest="respawn_per_cycle", action="store_false")

    # grasp geometry / timing (defaults from the working DEMOS script)
    p.add_argument("--ee-link", type=str, default="j2n6s300_end_effector")
    p.add_argument("--arm-joint-regex", type=str, default="j2n6s300_joint_[1-6]$")
    p.add_argument("--gripper-joint-regex", type=str, default=".*_joint_finger_.*|.*_joint_finger_tip_.*")
    p.add_argument("--gripper-open-pos", type=float, default=0.0)
    p.add_argument("--gripper-close-pos", type=float, default=1.2)
    p.add_argument("--ee-z-offset-m", type=float, default=0.08)
    p.add_argument("--travel-height-m", type=float, default=0.14)
    p.add_argument("--grasp-depth-m", type=float, default=-0.07)
    p.add_argument("--lift-check-m", type=float, default=0.04)
    p.add_argument("--lift-max-m", type=float, default=0.6)
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
    from datetime import datetime

    import numpy as np
    import torch
    from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
    from isaaclab.utils.math import quat_apply, quat_conjugate, subtract_frame_transforms

    from data_collection.core.logger import SessionLogWriter, TickLoggingConfig
    from data_collection.core.objects import ObjectsTracker
    from environments.utils.camera import (
        DEFAULT_FRONT_CAMERA, DEFAULT_WRIST_CAMERA, build_camera, sync_wrist_camera_to_ee,
    )
    from environments.utils.object_loader import ObjectLoader, ObjectLoaderConfig, SpawnBounds
    from environments.utils.physix import object_loader_kwargs_from_physix
    from environments.ycb_reach_to_grasp import YCBReachToGraspEnv
    from kinova import GripperConfig, GripperController

    rng = random.Random(int(args.seed))
    np_rng = np.random.default_rng(int(args.seed))

    # ---- motion helpers (verbatim from DEMOS/ycb_reach_grasp) --------------
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
        th0 = math.acos(max(-1.0, min(1.0, dot)))
        st0 = math.sin(th0)
        th = th0 * float(t)
        return (math.cos(th) - dot * math.sin(th) / st0) * q0 + (math.sin(th) / st0) * q1

    class _MoveSegment:
        def __init__(self, p0, p1, q0, q1, min_d, max_d):
            self.p0, self.p1, self.q0, self.q1 = p0.clone(), p1.clone(), q0.clone(), q1.clone()
            self.min_duration_s = max(1e-3, float(min_d))
            self.max_duration_s = max(self.min_duration_s, float(max_d))
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

    # ---- build scene + boxes ----------------------------------------------
    env = YCBReachToGraspEnv(device=str(getattr(args, "device", "cuda:0")))
    sim = env.build_simulation()
    if not bool(getattr(args, "headless", False)):
        env.set_default_camera_view()
    env.design_scene()
    robot = env.robot

    loader = ObjectLoader(ObjectLoaderConfig(
        dataset_dirs=[],
        bounds=SpawnBounds(min_xyz=tuple(args.spawn_min), max_xyz=tuple(args.spawn_max)),
        min_distance=float(args.min_distance), min_distance_xy_only=True, spawn_mode="box",
        box_size_min=(args.box_size,) * 3, box_size_max=(args.box_size,) * 3,
        box_color_palette=[rgb for (_n, rgb) in BOX_COLORS],
        box_color_names=[n for (n, _r) in BOX_COLORS],
        **object_loader_kwargs_from_physix(env.physics_cfg),
    ))
    spawned_paths = loader.spawn(parent_prim_path="/World/Origin1", num_objects=int(args.num_objects))
    if not spawned_paths:
        print("[COLLECT][ERROR] no boxes spawned")
        return 2
    n_obj = len(spawned_paths)
    id_to_label, id_to_color = {}, {}
    for pth in spawned_paths:
        leaf = str(pth).split("/")[-1]
        idx = int(leaf.split("_")[-1])
        cname = BOX_COLORS[(idx - 1) % len(BOX_COLORS)][0]
        id_to_label[leaf] = f"{cname} box {idx}"
        id_to_color[leaf] = cname
    print(f"[COLLECT] spawned {n_obj} boxes: {list(id_to_label.values())}")

    # ---- cameras (before first reset) -------------------------------------
    cameras = {}
    for name, cfg in (("front", DEFAULT_FRONT_CAMERA), ("wrist", DEFAULT_WRIST_CAMERA)):
        try:
            cameras[name] = build_camera(name, robot=robot, cfg=cfg)
        except Exception as e:
            print(f"[COLLECT] {name} camera failed: {e}")
    env.reset()
    for s in cameras.values():
        s.reset()

    tracker = ObjectsTracker(prim_paths=list(spawned_paths))

    # ---- arm / IK / gripper -----------------------------------------------
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

    dt = float(sim.get_physics_dt())
    cap_stride = max(1, round((1.0 / float(args.log_rate_hz)) / dt))

    # rigid-body handles for teleport (respawn) + lift verification
    rb_handles = {}
    try:
        from isaacsim.core.prims import RigidPrim
        for pth in spawned_paths:
            try:
                rb = RigidPrim(pth)
                try:
                    rb.initialize()
                except Exception:
                    pass
                rb_handles[pth] = rb
            except Exception:
                pass
    except Exception as e:
        print(f"[COLLECT][WARN] RigidPrim unavailable ({e}); teleport/lift-check limited")

    def _obj_world_pos(pth):
        """Box center in WORLD frame from the physics view. Unlike the OBB /
        USD read, this is always current after a set_world_poses teleport."""
        rb = rb_handles.get(pth)
        if rb is None:
            return None
        try:
            pos, _ = rb.get_world_poses()
            return [float(pos[0][0]), float(pos[0][1]), float(pos[0][2])]
        except Exception:
            return None

    def _obj_world_z(pth):
        p = _obj_world_pos(pth)
        return None if p is None else p[2]

    # ---- per-episode logging state ----------------------------------------
    st = {"logger": None, "logging": False}
    tick_cfg = TickLoggingConfig(
        log_rate_hz=int(args.log_rate_hz), policy_rate_hz=int(args.log_rate_hz),
        ee_link_name=str(args.ee_link), arm_joint_regex=str(args.arm_joint_regex),
        log_joint_data=True,
        workspace_min=(0.20, -0.45, 0.0), workspace_max=(0.6, 0.45, 1.20),
    )

    def _objects_snapshot():
        out = []
        for o in tracker.snapshot():
            out.append({
                "id": o.id,
                "label": id_to_label.get(o.id, o.label),
                "pose": {"position_m": list(o.pose.position_m), "orientation_wxyz": list(o.pose.orientation_wxyz)},
                "confidence": getattr(o, "confidence", 1.0),
            })
        return out

    def _capture_and_log():
        lg = st["logger"]
        image_paths = {}
        for name, sensor in cameras.items():
            data = sensor.data
            rgb = data.output.get("rgb") if data.output is not None else None
            if rgb is None:
                continue
            arr = rgb[0].cpu().numpy() if rgb.dim() == 4 else rgb.cpu().numpy()
            arr = arr[..., :3]
            arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
            d = lg.root / "images" / name
            d.mkdir(parents=True, exist_ok=True)
            from PIL import Image

            Image.fromarray(arr).save(str(d / f"image_{lg.tick_idx:06d}.png"))
            image_paths[name] = f"images/{name}/image_{lg.tick_idx:06d}.png"
        lg.write_tick(robot=robot, controller=diff_ik, objects=_objects_snapshot(),
                      last_user_cmd=None, cfg=tick_cfg, image_paths=image_paths)

    def _step():
        """One physics step. Every cap_stride steps, sync wrist -> render ->
        capture + log a 5 Hz tick (only while an episode is being recorded)."""
        do_cap = st["logging"] and (st["phys"] % cap_stride == 0)
        if do_cap and "wrist" in cameras:
            try:
                sync_wrist_camera_to_ee(robot, cameras["wrist"], DEFAULT_WRIST_CAMERA)
            except Exception:
                pass
        sim.step(render=True)
        robot.update(dt)
        for s in cameras.values():
            try:
                s.update(dt)
            except Exception:
                pass
        if do_cap:
            _capture_and_log()
        st["phys"] += 1

    st["phys"] = 0

    # ---- pose helpers ------------------------------------------------------
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
        _step()

    def _pos_err_m(goal_b):
        p, _ = _read_ee_pose_b()
        return float((p - goal_b).norm())

    def _run_segment(goal_pos_b, q_end_b, q_start_b=None):
        p0, q0m = _read_ee_pose_b()
        q_start_b = q0m if q_start_b is None else q_start_b
        dist = float((goal_pos_b - p0).norm())
        min_dur = max(float(args.min_segment_s), dist / max(1e-6, float(args.cruise_mps)))
        seg = _MoveSegment(p0, goal_pos_b, q_start_b, q_end_b, min_dur, max(min_dur + 0.5, float(args.max_segment_s)))
        while simulation_app.is_running() and not seg.timed_out:
            p_t, q_t = seg.current()
            _drive_ik_step(p_t, q_t)
            seg.advance(dt)
            if seg.eased_complete and _pos_err_m(goal_pos_b) < float(args.converge_pos_tol_m):
                break

    def _hold_at(goal_pos_b, q_fixed_b, hold_s):
        for _ in range(int(max(1, round(float(hold_s) / dt)))):
            if not simulation_app.is_running():
                return
            _drive_ik_step(goal_pos_b, q_fixed_b)

    def _reset_robot_home():
        robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
        robot.reset()
        try:
            gripper.command_open(robot)
        except Exception:
            pass
        for _ in range(40):
            if not simulation_app.is_running():
                return
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
            _step()

    def _sample_layout():
        """Rejection-sample n_obj XY positions within bounds respecting min-distance."""
        lo, hi = args.spawn_min, args.spawn_max
        z = float(lo[2])
        pts = []
        tries = 0
        while len(pts) < n_obj and tries < 2000:
            tries += 1
            x = float(np_rng.uniform(lo[0], hi[0]))
            y = float(np_rng.uniform(lo[1], hi[1]))
            if all(math.hypot(x - px, y - py) >= float(args.min_distance) for px, py, _ in pts):
                pts.append((x, y, z))
        while len(pts) < n_obj:  # fallback: relax distance
            pts.append((float(np_rng.uniform(lo[0], hi[0])), float(np_rng.uniform(lo[1], hi[1])), z))
        return pts

    def _respawn_boxes():
        pts = _sample_layout()
        for pth, (x, y, z) in zip(spawned_paths, pts):
            rb = rb_handles.get(pth)
            if rb is None:
                continue
            try:
                pos = torch.tensor([[x, y, z]], dtype=torch.float32, device=sim.device)
                orn = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=sim.device)
                rb.set_world_poses(pos, orn)
                if hasattr(rb, "set_velocities"):
                    rb.set_velocities(torch.zeros((1, 6), dtype=torch.float32, device=sim.device))
            except Exception:
                pass
        for _ in range(30):
            _step()

    # ---- session + episode loop -------------------------------------------
    session_folder = Path(str(args.logs_root)) / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_folder.mkdir(parents=True, exist_ok=True)
    ee_off = float(args.ee_z_offset_m)

    n_success = 0
    color_hist = {}
    for ep in range(int(args.num_episodes)):
        if not simulation_app.is_running():
            break
        # re-randomize layout at the start of each NEW round-robin cycle
        # (not ep 0 — the initial spawn is already a valid random layout)
        if args.respawn_per_cycle and ep > 0 and (ep % n_obj == 0):
            _respawn_boxes()
        _reset_robot_home()

        target_prim = str(spawned_paths[ep % n_obj])
        target_leaf = target_prim.split("/")[-1]
        color = id_to_color.get(target_leaf, "box")
        label = id_to_label.get(target_leaf, target_leaf)
        lang = f"Pick up the {color} box."

        # per-episode logger
        lg = SessionLogWriter(root=session_folder, session_name=f"episode_{ep:04d}")
        lg.write_metadata(sim_dt=dt, physics_substeps=1, seed=int(args.seed),
                          robot_name="j2n6s300", ee_link=str(args.ee_link),
                          arm_joint_regex=str(args.arm_joint_regex), log_rate_hz=int(args.log_rate_hz),
                          window_len_s=2.0, policy_rate_hz=int(args.log_rate_hz))
        import json as _json
        (lg.root / "instruction.json").write_text(_json.dumps({
            "episode_idx": ep, "target_prim": target_prim, "target_leaf": target_leaf,
            "target_label": label, "language_command": lang,
            "language_command_meta": {"target_leaf": target_leaf, "color": color},
        }, indent=2))

        # grasp pose from the box's CURRENT physics position (robust to the
        # teleport respawn; a uniform box needs no OBB — top-centre + identity
        # yaw is the grasp).
        box_w = _obj_world_pos(target_prim)
        if box_w is None:
            print(f"[COLLECT][ep {ep}] no physics pose for {target_leaf}; skipping")
            lg.log_event("grasp_result", {"episode_idx": ep, "ok": False, "reason": "no_pose"})
            lg.log_event("episode_end", {"episode_idx": ep})
            lg.close()
            continue
        top_w = torch.tensor([box_w[0], box_w[1], box_w[2] + 0.5 * float(args.box_size)],
                             dtype=torch.float32, device=sim.device)
        top_b = _world_to_base(top_w)

        over_b = top_b.clone()
        over_b[2] = float(top_b[2]) + ee_off + float(args.travel_height_m)
        grasp_b = top_b.clone()
        grasp_b[2] = float(top_b[2]) + ee_off + float(args.grasp_depth_m)
        _, q_cur = _read_ee_pose_b()
        q_pick = q_cur.clone()

        lg.log_event("episode_start", {"episode_idx": ep, "target_prim": target_prim,
                                       "target_label": label, "language_command": lang})
        z0 = _obj_world_z(target_prim)

        # ---- record the grasp -----------------------------------------------
        st["logger"] = lg
        st["logging"] = True
        _run_segment(over_b, q_pick, q_start_b=q_cur)      # move over
        _run_segment(grasp_b, q_pick, q_start_b=q_pick)     # descend
        _hold_at(grasp_b, q_pick, float(args.pre_close_settle_s))
        try:
            gripper.command_close(robot)
        except Exception:
            pass
        _hold_at(grasp_b, q_pick, float(args.gripper_close_s))
        _run_segment(over_b, q_pick, q_start_b=q_pick)      # lift
        _hold_at(over_b, q_pick, float(args.lift_hold_s))
        st["logging"] = False

        # ---- verdict --------------------------------------------------------
        z1 = _obj_world_z(target_prim)
        ok = False
        dz = None
        if z0 is not None and z1 is not None:
            dz = z1 - z0
            ok = float(args.lift_check_m) <= dz <= float(args.lift_max_m)
        lg.log_event("grasp_result", {"episode_idx": ep, "ok": bool(ok),
                                      "dz_m": (float(dz) if dz is not None else None)})
        lg.log_event("episode_end", {"episode_idx": ep, "ticks": int(lg.tick_idx)})
        lg.close()

        n_success += int(ok)
        if ok:
            color_hist[color] = color_hist.get(color, 0) + 1
        print(f"[COLLECT] ep {ep:03d} target={label:<14} "
              f"dz={'n/a' if dz is None else f'{dz*1000:+.0f}mm':>8} -> {'OK' if ok else 'MISS'} "
              f"({lg.tick_idx} ticks)")

    print("=" * 60)
    print(f"[COLLECT] DONE: {n_success}/{int(args.num_episodes)} successful "
          f"({100.0*n_success/max(1,int(args.num_episodes)):.0f}%)")
    print(f"[COLLECT] success color histogram: {color_hist}")
    print(f"[COLLECT] session -> {session_folder}")
    return 0 if n_success > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
