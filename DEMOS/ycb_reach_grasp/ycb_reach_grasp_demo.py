"""Scripted reach-to-grasp demo on the YCB scene: pick each object from the top and lift it.

This is a trimmed-down sibling of ``DEMOS/block_stacking/block_stacking_demo.py``
-- same working diff-IK + gripper pipeline (``_MoveSegment`` / ``_run_segment`` /
``_drive_ik_step`` / quintic-eased Cartesian segments, top-down OBB grasp) -- but
instead of stacking it just visits every spawned YCB object in turn:

    YCBReachToGraspEnv -> spawn N YCB objects
      -> for each object:
           measure a top-down grasp pose from its world OBB (top-center + yaw)
           approach from above -> descend around it -> close the gripper
           lift it straight up and hold (the "reach to grasp and lift")
           lower it back, open, and retract
      -> idle

Run with a GUI:
    python DEMOS/ycb_reach_grasp/ycb_reach_grasp_demo.py --device cuda:0 --num-objects 10

Headless smoke test (for logs / CI):
    python DEMOS/ycb_reach_grasp/ycb_reach_grasp_demo.py --headless --device cuda:0 --num-objects 10

Only grasp certain objects:
    python DEMOS/ycb_reach_grasp/ycb_reach_grasp_demo.py --device cuda:0 --include-labels mug banana sugar_box
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path


# --- Path bootstrap (same pattern as block_stacking_demo to survive Kit side effects) ---
ROOT = Path(__file__).resolve().parents[2]
root_str = str(ROOT)
if root_str in sys.path:
    sys.path.remove(root_str)
sys.path.insert(0, root_str)
_env_mod = sys.modules.get("environments")
if _env_mod is not None and not hasattr(_env_mod, "__path__"):
    del sys.modules["environments"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])

    # Scene / objects
    parser.add_argument("--num-objects", type=int, default=7, help="Number of YCB objects to spawn and grasp.")
    parser.add_argument("--spawn-min", type=float, nargs=3, default=[0.30, -0.33, 0.86], metavar=("X", "Y", "Z"))
    parser.add_argument("--spawn-max", type=float, nargs=3, default=[0.66, 0.33, 0.92], metavar=("X", "Y", "Z"))
    parser.add_argument("--min-distance", type=float, default=0.12, help="Min pairwise distance between objects (m). Smaller fits more objects in the spawn area.")
    parser.add_argument("--lift-check-m", type=float, default=0.04, help="Object must rise at least this much (m) for the lift to count as a successful grasp.")
    parser.add_argument("--lift-max-m", type=float, default=0.6, help="A 'lift' larger than this (m) is treated as a physics glitch (object flung), not a real grasp.")
    parser.add_argument("--seed", type=int, default=-1, help="Use a non-negative int for a reproducible layout.")
    parser.add_argument(
        "--include-labels", type=str, nargs="*", default=None,
        help="Optional whitelist of YCB labels to spawn (e.g. mug banana sugar_box). Default: any object.",
    )
    parser.add_argument("--scale-min", type=float, default=None, help="Optional uniform object scale min.")
    parser.add_argument("--scale-max", type=float, default=None, help="Optional uniform object scale max.")

    # ---- Grasp order: which object to go for first ----
    parser.add_argument(
        "--grab-order", type=str, nargs="*", default=None,
        help="Explicit order of objects to grab, by label (substring, case-insensitive), e.g. "
             "--grab-order sugar_box mustard tomato. Matched objects are grabbed in this order; any "
             "objects not listed follow afterwards (sorted by --order-by), unless --only-listed is set.",
    )
    parser.add_argument(
        "--only-listed", action="store_true",
        help="With --grab-order, grab ONLY the listed objects (skip everything else).",
    )
    parser.add_argument(
        "--order-by", type=str, default="spawned",
        choices=["spawned", "nearest", "farthest", "left-to-right", "right-to-left"],
        help="How to order objects when --grab-order is not given (or for the leftover objects). "
             "nearest/farthest = distance from the robot base; left-to-right = by +Y.",
    )

    # EE geometry. The `j2n6s300_end_effector` link is at the wrist/palm, not the fingertips.
    parser.add_argument("--ee-z-offset-m", type=float, default=0.08, help="Wrist-to-fingertip vertical offset for Jaco2 (m).")
    parser.add_argument(
        "--travel-height-m", type=float, default=0.14,
        help="Clearance above the TALLEST object's top at which the arm moves between objects, so it does "
             "not drag through / knock the clutter while traversing.",
    )
    parser.add_argument("--pregrasp-offset-m", type=float, default=0.14, help="Pregrasp palm height above the object top (m).")
    parser.add_argument(
        "--grasp-depth-m", type=float, default=-0.07,
        help="Pickup palm offset relative to object top + ee_z_offset (m). Negative descends around the object.",
    )
    parser.add_argument("--lift-offset-m", type=float, default=0.20, help="Lift palm height above the object top (m).")

    # Yaw alignment to the object's OBB (keeps palm-down, only rotates around base Z).
    parser.add_argument(
        "--align-to-obb", dest="align_to_obb", action="store_true", default=True,
        help="Rotate the gripper around base Z to align the fingers with the object's narrow (OBB) axis.",
    )
    parser.add_argument("--no-align-to-obb", dest="align_to_obb", action="store_false")
    parser.add_argument(
        "--grasp-yaw-offset-deg", type=float, default=0.0,
        help="Extra yaw added to the OBB-aligned grasp (deg). Use 90 if the fingers grab along the wrong axis.",
    )

    # Motion timing
    parser.add_argument("--cruise-mps", type=float, default=0.22, help="Cartesian cruise speed for segment durations (m/s).")
    parser.add_argument("--min-segment-s", type=float, default=0.8, help="Minimum duration for any motion segment (s).")
    parser.add_argument("--max-segment-s", type=float, default=4.5, help="Hard timeout for any motion segment (s).")
    parser.add_argument("--converge-pos-tol-m", type=float, default=0.008, help="Position convergence tolerance (m).")

    # Gripper / settle timing
    parser.add_argument("--pre-close-settle-s", type=float, default=0.3, help="Hold at the grasp pose before closing.")
    parser.add_argument("--gripper-close-s", type=float, default=0.6, help="Hold the close command before lifting.")
    parser.add_argument("--lift-hold-s", type=float, default=0.6, help="Hold at the lifted pose to show the object is grasped.")
    parser.add_argument("--pre-release-settle-s", type=float, default=0.2, help="Hold after lowering before opening.")
    parser.add_argument("--gripper-open-s", type=float, default=0.5, help="Hold the open command after releasing.")
    parser.add_argument("--place-back", dest="place_back", action="store_true", default=True,
                        help="Lower each object back down and release it before moving on (default).")
    parser.add_argument("--no-place-back", dest="place_back", action="store_false",
                        help="Just open the gripper at the lifted height (drop) instead of lowering back.")

    # Robot wiring
    parser.add_argument("--ee-link", type=str, default="j2n6s300_end_effector")
    parser.add_argument("--arm-joint-regex", type=str, default="j2n6s300_joint_[1-6]$")
    parser.add_argument("--gripper-joint-regex", type=str, default=".*_joint_finger_.*|.*_joint_finger_tip_.*")
    parser.add_argument("--gripper-open-pos", type=float, default=0.0)
    parser.add_argument("--gripper-close-pos", type=float, default=1.2)

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


@dataclass
class ObjectInfo:
    prim_path: str
    label: str
    top_b: object              # torch tensor: grasp top-center in robot base frame
    yaw_w_rad: float | None    # world yaw of the THIN axis -- the gripper closes across this
    minor_m: float = 0.0       # thin horizontal extent (m)
    major_m: float = 0.0       # wide horizontal extent (m)


def main() -> int:
    args = _parse_args()

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import random

    import torch
    from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
    from isaaclab.utils.math import quat_apply, quat_conjugate, subtract_frame_transforms

    from environments.ycb_reach_to_grasp import YCBReachToGraspEnv
    from kinova import GripperConfig, GripperController
    from motion_generation.grasp_estimation.obb import ObbGraspPoseProvider

    headless = bool(getattr(args, "headless", False))

    if int(args.seed) >= 0:
        random.seed(int(args.seed))
        print(f"[REACH] RNG seeded with --seed {int(args.seed)}.")
    else:
        print("[REACH] RNG not seeded; pass --seed <int> to reproduce a layout.")

    # ------------------------------------------------------------------
    # Small math helpers (identical conventions to the block-stacking demo).
    # ------------------------------------------------------------------
    def _quintic(s: float) -> float:
        s = max(0.0, min(1.0, float(s)))
        return 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5

    def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
        return torch.stack(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dim=-1,
        )

    def _yaw_quat(yaw_rad: float, *, device, dtype=torch.float32) -> torch.Tensor:
        half = 0.5 * float(yaw_rad)
        return torch.tensor([math.cos(half), 0.0, 0.0, math.sin(half)], device=device, dtype=dtype)

    def _quat_to_yaw(q) -> float:
        w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _wrap_pi(a: float) -> float:
        return (float(a) + math.pi) % (2.0 * math.pi) - math.pi

    def _slerp(q0: torch.Tensor, q1: torch.Tensor, t: float) -> torch.Tensor:
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
            self.p0 = p0.clone()
            self.p1 = p1.clone()
            self.q0 = q0.clone()
            self.q1 = q1.clone()
            self.min_duration_s = max(1e-3, float(min_duration_s))
            self.max_duration_s = max(self.min_duration_s, float(max_duration_s))
            self.t_elapsed = 0.0

        def advance(self, dt: float) -> None:
            self.t_elapsed += float(max(0.0, dt))

        def current(self):
            se = _quintic(self.t_elapsed / self.min_duration_s)
            p_t = self.p0 + se * (self.p1 - self.p0)
            q_t = _slerp(self.q0, self.q1, se)
            return p_t, q_t

        @property
        def timed_out(self) -> bool:
            return self.t_elapsed >= self.max_duration_s

        @property
        def eased_complete(self) -> bool:
            return self.t_elapsed >= self.min_duration_s

    # ------------------------------------------------------------------
    # Build sim + scene + robot via the YCB reach-to-grasp environment.
    # ------------------------------------------------------------------
    scale_range = None
    if args.scale_min is not None and args.scale_max is not None:
        scale_range = (float(args.scale_min), float(args.scale_max))

    env = YCBReachToGraspEnv(
        device=str(getattr(args, "device", "cuda:0")),
        include_labels=args.include_labels,
        scale_range=scale_range,
    )
    sim = env.build_simulation()
    if not headless:
        env.set_default_camera_view()
    env.design_scene()
    robot = env.robot

    loader = env.build_object_loader(
        spawn_min=tuple(args.spawn_min),
        spawn_max=tuple(args.spawn_max),
        min_distance=float(args.min_distance),
    )
    spawned_paths = loader.spawn(parent_prim_path="/World/Origin1", num_objects=int(args.num_objects))
    if len(spawned_paths) == 0:
        print("[REACH][ERROR] No objects spawned. Aborting.")
        simulation_app.close()
        return 2
    labels_map = {}
    try:
        labels_map = loader.get_last_spawn_labels()
    except Exception:
        pass
    print(f"[REACH] Spawned {len(spawned_paths)} objects.")

    env.reset()

    # ------------------------------------------------------------------
    # Arm / IK / gripper wiring (identical to the block-stacking demo).
    # ------------------------------------------------------------------
    arm_joint_ids_t, _ = robot.find_joints(str(args.arm_joint_regex))
    if hasattr(arm_joint_ids_t, "view"):
        arm_joint_ids = [int(v) for v in arm_joint_ids_t.view(-1).tolist()]
    else:
        arm_joint_ids = [int(v) for v in list(arm_joint_ids_t)]
    print(f"[REACH] arm joints: {[str(robot.data.joint_names[i]) for i in arm_joint_ids]}")

    ee_body_ids, _ = robot.find_bodies([str(args.ee_link)])
    ee_body_id = int(ee_body_ids[0])
    ee_jacobi_idx = ee_body_id - 1 if robot.is_fixed_base else ee_body_id

    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik = DifferentialIKController(diff_ik_cfg, num_envs=1, device=sim.device)
    diff_ik.reset()

    gripper_cfg = GripperConfig(
        joint_regex=str(args.gripper_joint_regex),
        open_position=float(args.gripper_open_pos),
        close_position=float(args.gripper_close_pos),
    )
    gripper = GripperController(gripper_cfg, num_envs=1, device=str(sim.device))
    gripper.resolve_joints(robot)
    gripper.reset(robot)
    try:
        robot_prim_path = str(getattr(getattr(robot, "cfg", None), "prim_path", None))
        if robot_prim_path:
            gripper.set_drive_gains(robot_prim_path)
            gripper.apply_stable_grasp_tuning(robot_prim_path)
    except Exception:
        pass

    grasp_provider = ObbGraspPoseProvider(align_to_min_width=True)

    def _thin_axis_world(prim_path: str):
        """World yaw of the object's THIN horizontal axis + (minor, major) extents.

        Uses the oriented world bounding box: of the box's 3 axes we keep the two
        that lie most in the horizontal plane (a top-down grip closes in XY), then
        the one with the smaller horizontal extent is the THIN axis the gripper
        should close across. Returns (yaw_rad, minor_m, major_m) or None.
        """
        try:
            import omni.usd  # type: ignore
            from pxr import Gf, Usd, UsdGeom  # type: ignore
        except Exception:
            return None
        try:
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(str(prim_path))
            if not prim.IsValid():
                return None
            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=True)
            bbox = cache.ComputeWorldBound(prim)
            box = bbox.GetBox()
            mat = bbox.GetMatrix()
            mn, mx = box.GetMin(), box.GetMax()
            half = [0.5 * float(mx[i] - mn[i]) for i in range(3)]
            local_axes = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
            cands = []  # (horizontalness, xy_extent, yaw)
            for i, la in enumerate(local_axes):
                d = mat.TransformDir(Gf.Vec3d(*la))
                n = math.sqrt(float(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)) or 1.0
                dx, dy, dz = float(d[0]) / n, float(d[1]) / n, float(d[2]) / n
                horiz = math.hypot(dx, dy)                 # 1 = horizontal, 0 = vertical
                xy_extent = 2.0 * half[i] * horiz          # this axis' footprint length in XY
                yaw = math.atan2(dy, dx)
                cands.append((horiz, xy_extent, yaw))
            # The two most-horizontal axes define the footprint; close across the thinner one.
            cands.sort(key=lambda c: c[0], reverse=True)
            a, b = cands[0], cands[1]
            thin, wide = (a, b) if a[1] <= b[1] else (b, a)
            return float(thin[2]), float(thin[1]), float(wide[1])
        except Exception:
            return None

    def _read_ee_pose_b():
        ee_pose_w = robot.data.body_pose_w[:, ee_body_id]
        root_pose_w = robot.data.root_pose_w
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        return ee_pos_b[0].clone(), ee_quat_b[0].clone()

    def _world_to_base(pos_w_t: torch.Tensor) -> torch.Tensor:
        root_pose_w = robot.data.root_pose_w
        base_pos_w = root_pose_w[0, 0:3]
        base_quat_w = root_pose_w[0, 3:7]
        base_quat_inv = quat_conjugate(base_quat_w.unsqueeze(0))[0]
        rel_w = (pos_w_t - base_pos_w).unsqueeze(0)
        return quat_apply(base_quat_inv.unsqueeze(0), rel_w)[0]

    def _quat_aligned_to_yaw(q_current_b: torch.Tensor, target_yaw_rad: float) -> torch.Tensor:
        current_yaw = _quat_to_yaw(q_current_b)
        dyaw = _wrap_pi(float(target_yaw_rad) - current_yaw)
        return _quat_mul(_yaw_quat(dyaw, device=sim.device), q_current_b)

    def _drive_ik_step(p_des_b: torch.Tensor, q_des_b: torch.Tensor, dt: float) -> None:
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
            gravity = robot.root_physx_view.get_gravity_compensation_forces()
            robot.set_joint_effort_target(gravity)
        except Exception:
            pass

        robot.write_data_to_sim()
        sim.step(render=True)
        robot.update(dt)

    def _pos_err_m(goal_b: torch.Tensor) -> float:
        ee_pos_b, _ = _read_ee_pose_b()
        return float((ee_pos_b - goal_b).norm())

    def _run_segment(label, goal_pos_b, q_end_b, dt, q_start_b=None):
        p0, q0_measured = _read_ee_pose_b()
        q_start_b = q0_measured if q_start_b is None else q_start_b
        dist = float((goal_pos_b - p0).norm())
        min_dur = max(float(args.min_segment_s), dist / max(1e-6, float(args.cruise_mps)))
        max_dur = max(min_dur + 0.5, float(args.max_segment_s))
        seg = _MoveSegment(p0=p0, p1=goal_pos_b, q0=q_start_b, q1=q_end_b, min_duration_s=min_dur, max_duration_s=max_dur)
        pos_tol = float(args.converge_pos_tol_m)

        converged = False
        while simulation_app.is_running() and not seg.timed_out:
            p_t, q_t = seg.current()
            _drive_ik_step(p_t, q_t, dt)
            seg.advance(dt)
            if seg.eased_complete and _pos_err_m(goal_pos_b) < pos_tol:
                converged = True
                break

        final_err = _pos_err_m(goal_pos_b)
        status = "OK" if converged else ("TIMEOUT" if seg.timed_out else "EXIT")
        print(
            f"[REACH]   {label:<18} dist={dist * 1000:6.1f} mm "
            f"min_dur={min_dur:.2f}s t={seg.t_elapsed:.2f}s final_err={final_err * 1000:6.1f} mm [{status}]"
        )
        return converged, final_err

    def _hold_at(goal_pos_b, q_fixed_b, hold_s, dt):
        steps = int(max(1, round(float(hold_s) / dt)))
        for _ in range(steps):
            if not simulation_app.is_running():
                return
            _drive_ik_step(goal_pos_b, q_fixed_b, dt)

    # ------------------------------------------------------------------
    # Settle + open the gripper.
    # ------------------------------------------------------------------
    dt = float(sim.get_physics_dt())
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
            gravity = robot.root_physx_view.get_gravity_compensation_forces()
            robot.set_joint_effort_target(gravity)
        except Exception:
            pass
        robot.write_data_to_sim()
        sim.step(render=True)
        robot.update(dt)

    ee_pos_b_home, ee_quat_b_home = _read_ee_pose_b()

    # ------------------------------------------------------------------
    # Auto-calibrate the gripper's closing axis. The s300 is a 2+1 finger hand,
    # so it grips like a jaw whose axis runs from the lone "thumb" tip to the
    # midpoint of the other two tips. We read the finger-tip links (open, at home),
    # express them in the EE frame, and measure that axis' angle relative to the EE
    # yaw reference -- then we can point it AT each object's thin axis regardless of
    # the gripper model (no magic --grasp-yaw-offset needed).
    # ------------------------------------------------------------------
    close_axis_offset = 0.0
    try:
        tip_ids_t, tip_names = robot.find_bodies([".*finger_tip.*"])
        tip_ids = [int(v) for v in (tip_ids_t.view(-1).tolist() if hasattr(tip_ids_t, "view") else tip_ids_t)]
        if len(tip_ids) >= 3:
            ee_pose_w = robot.data.body_pose_w[0, ee_body_id]
            ee_pos_w, ee_quat_w = ee_pose_w[0:3], ee_pose_w[3:7]
            ee_quat_inv = quat_conjugate(ee_quat_w.unsqueeze(0))
            pts = []
            for tid in tip_ids[:3]:
                tp_w = robot.data.body_pose_w[0, tid, 0:3]
                rel = quat_apply(ee_quat_inv, (tp_w - ee_pos_w).unsqueeze(0))[0]
                pts.append((float(rel[0]), float(rel[1])))  # EE-frame XY
            d = lambda a, b: math.hypot(pts[a][0] - pts[b][0], pts[a][1] - pts[b][1])
            pairs = [(0, 1, 2), (0, 2, 1), (1, 2, 0)]
            i, j, k = min(pairs, key=lambda p: d(p[0], p[1]))  # i,j = closest pair, k = thumb
            mid = ((pts[i][0] + pts[j][0]) * 0.5, (pts[i][1] + pts[j][1]) * 0.5)
            cx, cy = pts[k][0] - mid[0], pts[k][1] - mid[1]
            close_axis_offset = math.atan2(cy, cx)
            print(f"[REACH] Gripper closing-axis offset (EE frame): {math.degrees(close_axis_offset):+.1f} deg "
                  f"(thumb={tip_names[k] if k < len(tip_names) else k}).")
    except Exception as e:
        print(f"[REACH][WARN] Could not calibrate gripper closing axis ({e}); using --grasp-yaw-offset-deg only.")

    # Effective target = thin_axis_yaw - close_axis_offset + user offset, so the jaw
    # closing axis ends up pointing along the object's thin axis.
    yaw_off = math.radians(float(args.grasp_yaw_offset_deg)) - close_axis_offset

    # ------------------------------------------------------------------
    # Measure a top-down grasp pose for every spawned object (after settle).
    # ------------------------------------------------------------------
    objects: list[ObjectInfo] = []
    for prim_path in spawned_paths:
        try:
            grasp_pos_w, grasp_quat_w = grasp_provider.get_grasp_pose_w(object_prim_path=str(prim_path), robot_prim_path=None)
            top_b = _world_to_base(torch.tensor(grasp_pos_w, dtype=torch.float32, device=sim.device))
        except Exception as e:
            print(f"[REACH][WARN] OBB failed for {prim_path}: {e}. Skipping.")
            continue
        # Robust thin-axis (the gripper closes across this); fall back to the provider yaw.
        thin = _thin_axis_world(str(prim_path))
        if thin is not None:
            yaw_w, minor_m, major_m = thin
        else:
            yaw_w, minor_m, major_m = _quat_to_yaw(grasp_quat_w), 0.0, 0.0
        objects.append(
            ObjectInfo(
                prim_path=str(prim_path),
                label=str(labels_map.get(str(prim_path), str(prim_path).split("/")[-1])),
                top_b=top_b,
                yaw_w_rad=yaw_w,
                minor_m=minor_m,
                major_m=major_m,
            )
        )

    # ------------------------------------------------------------------
    # Decide the order objects are grabbed in (which goes first).
    # ------------------------------------------------------------------
    def _sorted(objs: list[ObjectInfo]) -> list[ObjectInfo]:
        ob = str(args.order_by)
        if ob == "nearest":
            return sorted(objs, key=lambda o: float((o.top_b[0] ** 2 + o.top_b[1] ** 2) ** 0.5))
        if ob == "farthest":
            return sorted(objs, key=lambda o: float((o.top_b[0] ** 2 + o.top_b[1] ** 2) ** 0.5), reverse=True)
        if ob == "left-to-right":
            return sorted(objs, key=lambda o: float(o.top_b[1]), reverse=True)
        if ob == "right-to-left":
            return sorted(objs, key=lambda o: float(o.top_b[1]))
        return list(objs)  # "spawned"

    if args.grab_order:
        wanted = [str(t).lower().strip() for t in args.grab_order if str(t).strip()]
        remaining = list(objects)
        ordered: list[ObjectInfo] = []
        for token in wanted:
            match = next((o for o in remaining if token in o.label.lower()), None)
            if match is None:
                print(f"[REACH][WARN] --grab-order: no spawned object matches '{token}'.")
                continue
            ordered.append(match)
            remaining.remove(match)
        if not bool(args.only_listed):
            ordered.extend(_sorted(remaining))
        elif remaining:
            print(f"[REACH] --only-listed: skipping {len(remaining)} unlisted object(s).")
        objects = ordered
    else:
        objects = _sorted(objects)

    print("[REACH] Grab order: " + ", ".join(f"{i + 1}.{o.label}" for i, o in enumerate(objects)))

    # Physics-view handles so we can verify each lift from the true object pose
    # (USD xform reads can be stale under the headless GPU pipeline).
    obj_rb: dict[str, object] = {}
    try:
        from isaacsim.core.prims import RigidPrim
        for obj in objects:
            try:
                rb = RigidPrim(obj.prim_path)
                try:
                    rb.initialize()
                except Exception:
                    pass
                obj_rb[obj.prim_path] = rb
            except Exception:
                pass
    except Exception as e:
        print(f"[REACH][WARN] RigidPrim unavailable ({e}); lift verification disabled.")

    def _obj_world_z(prim_path: str):
        rb = obj_rb.get(prim_path)
        if rb is None:
            return None
        try:
            pos, _ = rb.get_world_poses()
            return float(pos[0][2])
        except Exception:
            return None

    # Single safe height (base frame) for all between-object travel + lifts, set
    # above the TALLEST object so the arm never drags laterally through the clutter.
    ee_off = float(args.ee_z_offset_m)
    max_top_z = max((float(o.top_b[2]) for o in objects), default=0.2)
    travel_z_b = max_top_z + ee_off + float(args.travel_height_m)

    print(f"[REACH] Reaching for {len(objects)} objects (top-down grasp + lift); travel_z_b={travel_z_b:+.3f}:")
    print("[REACH] " + "=" * 70)

    grabbed = 0
    for idx, obj in enumerate(objects):
        if not simulation_app.is_running():
            break

        top_b = obj.top_b
        # High waypoint directly above the object (lateral moves happen here).
        over_b = top_b.clone()
        over_b[2] = travel_z_b
        grasp_b = top_b.clone()
        grasp_b[2] = float(top_b[2]) + ee_off + float(args.grasp_depth_m)

        ee_pos_b_cur, ee_quat_b_cur = _read_ee_pose_b()
        if bool(args.align_to_obb) and obj.yaw_w_rad is not None:
            # Close the fingers ACROSS the thin axis: aim the gripper's closing
            # direction at the thin-axis world yaw (+ a fixed gripper offset C).
            q_pick = _quat_aligned_to_yaw(ee_quat_b_cur, float(obj.yaw_w_rad) + yaw_off)
            yaw_log = (
                f"thin_axis_yaw={math.degrees(float(obj.yaw_w_rad)):+.1f} deg "
                f"(minor={obj.minor_m * 1000:.0f}mm major={obj.major_m * 1000:.0f}mm) "
                f"ee_yaw={math.degrees(_quat_to_yaw(q_pick)):+.1f} deg"
            )
        else:
            q_pick = ee_quat_b_cur.clone()
            yaw_log = "yaw alignment off"

        print(
            f"[REACH] object {idx + 1}/{len(objects)}: {obj.label}  prim={obj.prim_path}\n"
            f"        top_b=({float(top_b[0]):+.3f},{float(top_b[1]):+.3f},{float(top_b[2]):+.3f}) "
            f"grasp_z={float(grasp_b[2]):+.3f}  {yaw_log}"
        )

        z_before = _obj_world_z(obj.prim_path)

        # Move over the object at travel height, descend, grasp.
        _run_segment("MOVE_OVER", over_b, q_pick, dt, q_start_b=ee_quat_b_cur)
        _run_segment("DESCEND", grasp_b, q_pick, dt, q_start_b=q_pick)
        _hold_at(grasp_b, q_pick, float(args.pre_close_settle_s), dt)
        try:
            gripper.command_close(robot)
        except Exception:
            pass
        _hold_at(grasp_b, q_pick, float(args.gripper_close_s), dt)

        # Lift the object up to travel height and hold.
        _run_segment("LIFT", over_b, q_pick, dt, q_start_b=q_pick)
        _hold_at(over_b, q_pick, float(args.lift_hold_s), dt)

        # Verify from physics that the object actually came up with the gripper.
        z_after = _obj_world_z(obj.prim_path)
        if z_before is None or z_after is None:
            lifted = True  # no physics handle: fall back to "motion completed"
            dz_txt = "no physics handle"
        else:
            dz = z_after - z_before
            if dz > float(args.lift_max_m):
                lifted = False  # absurd jump = physics glitch (object flung), not a grasp
                dz_txt = f"dz={dz * 1000:+.0f} mm (flung/glitch)"
            else:
                lifted = dz >= float(args.lift_check_m)
                dz_txt = f"dz={dz * 1000:+.0f} mm"
        if lifted:
            grabbed += 1
            print(f"[REACH]   GRASPED + LIFTED  {obj.label}  ({dz_txt})")
        else:
            print(f"[REACH]   MISSED            {obj.label}  ({dz_txt})")

        # Put it back (or drop), then retract straight up to travel height.
        if bool(args.place_back):
            _run_segment("LOWER", grasp_b, q_pick, dt, q_start_b=q_pick)
            _hold_at(grasp_b, q_pick, float(args.pre_release_settle_s), dt)
        try:
            gripper.command_open(robot)
        except Exception:
            pass
        _hold_at(over_b if not args.place_back else grasp_b, q_pick, float(args.gripper_open_s), dt)
        _run_segment("RETRACT", over_b, q_pick, dt, q_start_b=q_pick)
        print("[REACH] " + "-" * 70)

    print("[REACH] " + "=" * 70)
    print(f"[REACH] SUMMARY: verified grasped+lifted {grabbed}/{len(objects)} objects "
          f"(rose >= {args.lift_check_m * 1000:.0f} mm).")

    if simulation_app.is_running() and not headless:
        ee_idle, q_idle = _read_ee_pose_b()
        _hold_at(ee_idle, q_idle, 1.0, dt)

    print("[REACH] Done.")
    simulation_app.close()
    return 0 if grabbed > 0 else 1


if __name__ == "__main__":
    import os

    _code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Isaac Sim's Kit threads can keep the headless process alive (pinned to GPU
    # memory) after simulation_app.close(); force a hard exit so repeated runs
    # don't pile up zombie processes that exhaust the GPU.
    os._exit(int(_code) if _code is not None else 0)
