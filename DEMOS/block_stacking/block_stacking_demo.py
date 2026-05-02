"""Scripted block stacking demo using the working diff-IK cube pickup pipeline.

This is intentionally close to ``motion_generation/demo_scripted_approach.py``:

    AppLauncher -> IsaacLab scene -> uniform/rectangular boxes
      -> measure box height + yaw
      -> pick boxes in a height-aware order
      -> rotate/align during pickup
      -> carry each box to a stack pose
      -> rotate to the stack yaw so placed cubes match the bottom cube
      -> place it at the current stack height with the requested placement yaw
      -> release and update the stack height

Run with a GUI:
    python DEMOS/block_stacking/block_stacking_demo.py --device cuda:0 --num-objects 3

Headless smoke test:
    python DEMOS/block_stacking/block_stacking_demo.py --headless --device cuda:0 --num-objects 1
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path


# --- Path bootstrap (same pattern as vla_v0/v1 to survive Kit side effects) ---
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
    parser.add_argument("--num-objects", type=int, default=3, help="Number of boxes to stack.")
    parser.add_argument(
        "--box-size",
        type=float,
        default=0.08,
        help="Uniform cube side length (m). Used unless --box-size-min/--box-size-max are provided.",
    )
    parser.add_argument(
        "--box-size-min",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Optional per-axis minimum box size for rectangular/random-height blocks.",
    )
    parser.add_argument(
        "--box-size-max",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Optional per-axis maximum box size for rectangular/random-height blocks.",
    )
    parser.add_argument("--spawn-min", type=float, nargs=3, default=[0.30, -0.30, 0.90], metavar=("X", "Y", "Z"))
    parser.add_argument("--spawn-max", type=float, nargs=3, default=[0.55, 0.30, 0.95], metavar=("X", "Y", "Z"))
    parser.add_argument("--min-distance", type=float, default=0.22, help="Min pairwise distance between boxes (m).")
    parser.add_argument("--seed", type=int, default=-1, help="Use a non-negative int for a reproducible layout.")

    # Stack target. X/Y are in robot base frame, table Z is a world-frame height.
    parser.add_argument("--stack-x", type=float, default=0.42, help="Stack center X in robot base frame (m).")
    parser.add_argument("--stack-y", type=float, default=0.0, help="Stack center Y in robot base frame (m).")
    parser.add_argument("--table-z-world", type=float, default=0.82, help="Table surface world Z used to initialize stack height.")
    parser.add_argument("--layer-gap-m", type=float, default=0.004, help="Small planned clearance between stacked layers.")
    parser.add_argument(
        "--stack-order",
        type=str,
        default="tallest-first",
        choices=["tallest-first", "shortest-first", "spawned"],
        help="Height-aware stacking order. Tallest-first gives the widest/tallest base by default.",
    )
    parser.add_argument(
        "--placement-rotation",
        type=str,
        default="match-stack",
        choices=["match-stack", "preserve", "fixed", "alternating"],
        help=(
            "How to orient each block when placing: match-stack makes every block match the first/bottom "
            "block yaw, preserve keeps pickup yaw, fixed uses --stack-yaw-deg, alternating adds 90 degrees per layer."
        ),
    )
    parser.add_argument(
        "--stack-yaw-deg",
        type=float,
        default=0.0,
        help="Placement yaw for --placement-rotation fixed/alternating, and fallback yaw for match-stack.",
    )

    # EE geometry. The `j2n6s300_end_effector` link is at the wrist/palm, not the fingertips.
    parser.add_argument("--ee-z-offset-m", type=float, default=0.08, help="Wrist-to-fingertip vertical offset for Jaco2 (m).")

    # Pickup/place offsets in base frame, relative to block top Z and ee_z_offset.
    parser.add_argument("--pregrasp-offset-m", type=float, default=0.12, help="Pregrasp palm offset above block top (m).")
    parser.add_argument(
        "--grasp-depth-m",
        type=float,
        default=-0.07,
        help="Pickup palm offset relative to block top + ee_z_offset (m). Negative descends around the block.",
    )
    parser.add_argument("--lift-offset-m", type=float, default=0.18, help="Lift palm offset above block top (m).")
    parser.add_argument("--preplace-offset-m", type=float, default=0.14, help="Pre-place palm offset above placement pose (m).")
    parser.add_argument(
        "--place-depth-m",
        type=float,
        default=None,
        help="Placement palm offset relative to placed block top + ee_z_offset. Defaults to --grasp-depth-m.",
    )

    # Motion timing
    parser.add_argument("--cruise-mps", type=float, default=0.20, help="Cartesian cruise speed for segment durations (m/s).")
    parser.add_argument("--min-segment-s", type=float, default=1.0, help="Minimum duration for any motion segment (s).")
    parser.add_argument("--max-segment-s", type=float, default=4.5, help="Hard timeout for any motion segment (s).")
    parser.add_argument("--converge-pos-tol-m", type=float, default=0.005, help="Position convergence tolerance (m).")

    # Gripper/settle timing
    parser.add_argument("--pre-close-settle-s", type=float, default=0.5, help="Hold at pickup pose before closing.")
    parser.add_argument("--gripper-close-s", type=float, default=0.8, help="Time to hold close command before lifting.")
    parser.add_argument("--pre-release-settle-s", type=float, default=0.35, help="Hold at stack pose before opening.")
    parser.add_argument("--gripper-open-s", type=float, default=0.8, help="Time to hold open command after placing.")
    parser.add_argument("--post-release-settle-s", type=float, default=0.3, help="Extra hold after release before retracting.")

    # Robot wiring
    parser.add_argument("--ee-link", type=str, default="j2n6s300_end_effector")
    parser.add_argument("--arm-joint-regex", type=str, default="j2n6s300_joint_[1-6]$")
    parser.add_argument("--gripper-joint-regex", type=str, default=".*_joint_finger_.*|.*_joint_finger_tip_.*")
    parser.add_argument("--gripper-open-pos", type=float, default=0.0)
    parser.add_argument("--gripper-close-pos", type=float, default=1.2)

    # Yaw alignment to the source block's OBB (keeps palm-down, only rotates around base Z).
    parser.add_argument(
        "--align-to-obb",
        dest="align_to_obb",
        action="store_true",
        default=True,
        help="Rotate gripper around base Z during pickup approach to align with the source block yaw.",
    )
    parser.add_argument("--no-align-to-obb", dest="align_to_obb", action="store_false")
    parser.add_argument(
        "--pickup-yaw-symmetry",
        type=str,
        default="square",
        choices=["square", "full"],
        help="Use square 90-degree symmetry for cubes, or full shortest-arc yaw alignment for rectangular blocks.",
    )

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


@dataclass
class BlockInfo:
    prim_path: str
    height_m: float
    yaw_w_rad: float | None
    top_b: object
    grasp_quat_w: tuple[float, float, float, float] | None


def main() -> int:
    args = _parse_args()

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import importlib
    import random

    import isaaclab.sim as sim_utils
    import torch
    from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
    from isaaclab.utils.math import quat_apply, quat_conjugate, subtract_frame_transforms

    from environments.utils.object_loader import ObjectLoader, ObjectLoaderConfig, SpawnBounds
    from environments.utils.physix import PhysicsConfig, apply_to_simulation_cfg, object_loader_kwargs_from_physix
    from kinova import GripperConfig, GripperController
    from motion_generation.grasp_estimation.obb import ObbGraspPoseProvider

    env_cfg_mod = importlib.import_module("environments.reach_to_grasp_VLA.config")
    env_utils_mod = importlib.import_module("environments.reach_to_grasp_VLA.utils")
    DEFAULT_SCENE = getattr(env_cfg_mod, "DEFAULT_SCENE")
    DEFAULT_CAMERA = getattr(env_cfg_mod, "DEFAULT_CAMERA", None)
    design_scene = getattr(env_utils_mod, "design_scene")

    if int(args.seed) >= 0:
        random.seed(int(args.seed))
        print(f"[STACK] RNG seeded with --seed {int(args.seed)}.")
    else:
        print("[STACK] RNG not seeded; pass --seed <int> to reproduce a layout.")

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

    def _quat_to_yaw(q: torch.Tensor) -> float:
        w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _wrap_pi(a: float) -> float:
        return (float(a) + math.pi) % (2.0 * math.pi) - math.pi

    def _wrap_quarter_pi(a: float) -> float:
        return (float(a) + math.pi / 4.0) % (math.pi / 2.0) - math.pi / 4.0

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
        def __init__(
            self,
            p0: torch.Tensor,
            p1: torch.Tensor,
            q0: torch.Tensor,
            q1: torch.Tensor,
            min_duration_s: float,
            max_duration_s: float,
        ) -> None:
            self.p0 = p0.clone()
            self.p1 = p1.clone()
            self.q0 = q0.clone()
            self.q1 = q1.clone()
            self.min_duration_s = max(1e-3, float(min_duration_s))
            self.max_duration_s = max(self.min_duration_s, float(max_duration_s))
            self.t_elapsed = 0.0

        def advance(self, dt: float) -> None:
            self.t_elapsed += float(max(0.0, dt))

        def current(self) -> tuple[torch.Tensor, torch.Tensor]:
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
    # Build sim + scene + robot.
    # ------------------------------------------------------------------
    phys = PhysicsConfig(device=str(getattr(args, "device", "cuda:0")))
    sim_cfg = sim_utils.SimulationCfg(device=phys.device)
    apply_to_simulation_cfg(sim_cfg, phys)
    sim = sim_utils.SimulationContext(sim_cfg)
    if (not getattr(args, "headless", False)) and DEFAULT_CAMERA is not None:
        sim.set_camera_view(DEFAULT_CAMERA.eye, DEFAULT_CAMERA.target)

    scene_entities, scene_origins = design_scene(DEFAULT_SCENE)
    robot = scene_entities["kinova_j2n6s300"]

    if args.box_size_min is not None or args.box_size_max is not None:
        if args.box_size_min is None or args.box_size_max is None:
            print("[STACK][ERROR] Provide both --box-size-min and --box-size-max, or neither.")
            simulation_app.close()
            return 2
        box_size_min = tuple(float(v) for v in args.box_size_min)
        box_size_max = tuple(float(v) for v in args.box_size_max)
    else:
        side = float(args.box_size)
        box_size_min = (side, side, side)
        box_size_max = (side, side, side)

    loader_cfg = ObjectLoaderConfig(
        dataset_dirs=[],
        bounds=SpawnBounds(min_xyz=tuple(args.spawn_min), max_xyz=tuple(args.spawn_max)),
        min_distance=float(args.min_distance),
        spawn_mode="box",
        box_size_min=box_size_min,
        box_size_max=box_size_max,
        box_color_palette=[(0.9, 0.2, 0.2), (0.2, 0.4, 0.9), (0.2, 0.9, 0.3), (0.9, 0.8, 0.2), (0.7, 0.3, 0.8)],
        box_color_names=["red", "blue", "green", "yellow", "purple"],
        **object_loader_kwargs_from_physix(phys),
    )
    loader = ObjectLoader(loader_cfg)
    spawned_paths = loader.spawn(parent_prim_path="/World/Origin1", num_objects=int(args.num_objects))
    if len(spawned_paths) == 0:
        print("[STACK][ERROR] No objects spawned. Aborting.")
        simulation_app.close()
        return 2

    sim.reset()
    origin0 = torch.tensor(scene_origins[0], device=sim.device)
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += origin0
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    robot.reset()

    arm_joint_ids_t, _ = robot.find_joints(str(args.arm_joint_regex))
    if hasattr(arm_joint_ids_t, "view"):
        arm_joint_ids = [int(v) for v in arm_joint_ids_t.view(-1).tolist()]
    else:
        arm_joint_ids = [int(v) for v in list(arm_joint_ids_t)]
    arm_joint_names = [str(robot.data.joint_names[i]) for i in arm_joint_ids]
    print(f"[STACK] arm joints: {arm_joint_names}")

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

    grasp_provider = ObbGraspPoseProvider(align_to_min_width=False)

    import importlib as _importlib_for_usd

    _pxr_usd = _importlib_for_usd.import_module("pxr.Usd")
    _pxr_usdgeom = _importlib_for_usd.import_module("pxr.UsdGeom")
    _omni_usd = _importlib_for_usd.import_module("omni.usd")
    _usd_stage = _omni_usd.get_context().get_stage()
    _bbox_cache = _pxr_usdgeom.BBoxCache(_pxr_usd.TimeCode.Default(), [_pxr_usdgeom.Tokens.default_])

    def _read_prim_world_yaw_rad(prim_path: str) -> float | None:
        try:
            prim = _usd_stage.GetPrimAtPath(str(prim_path))
            if not prim.IsValid():
                return None
            xformable = _pxr_usdgeom.Xformable(prim)
            M = xformable.ComputeLocalToWorldTransform(_pxr_usd.TimeCode.Default())
            return math.atan2(float(M[0][1]), float(M[0][0]))
        except Exception as e:
            print(f"[STACK][WARN] could not read world yaw for {prim_path}: {e}")
            return None

    def _read_prim_height_m(prim_path: str) -> float:
        try:
            prim = _usd_stage.GetPrimAtPath(str(prim_path))
            if not prim.IsValid():
                raise RuntimeError("invalid prim")
            world_bound = _bbox_cache.ComputeWorldBound(prim)
            r = world_bound.ComputeAlignedRange()
            return max(1e-4, float(r.GetMax()[2]) - float(r.GetMin()[2]))
        except Exception as e:
            print(f"[STACK][WARN] could not read height for {prim_path}: {e}; falling back to --box-size.")
            return float(args.box_size)

    def _read_ee_pose_b() -> tuple[torch.Tensor, torch.Tensor]:
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

    def _quat_aligned_to_yaw(q_current_b: torch.Tensor, target_yaw_rad: float, *, square_symmetry: bool) -> torch.Tensor:
        current_yaw = _quat_to_yaw(q_current_b)
        dyaw = float(target_yaw_rad) - current_yaw
        dyaw = _wrap_quarter_pi(dyaw) if square_symmetry else _wrap_pi(dyaw)
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

    def _run_segment(
        label: str,
        goal_pos_b: torch.Tensor,
        q_end_b: torch.Tensor,
        dt: float,
        q_start_b: torch.Tensor | None = None,
    ) -> tuple[bool, float]:
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
            f"[STACK]   {label:<22} dist={dist * 1000:6.1f} mm "
            f"min_dur={min_dur:.2f}s t={seg.t_elapsed:.2f}s final_err={final_err * 1000:6.1f} mm [{status}]"
        )
        return converged, final_err

    def _hold_at(goal_pos_b: torch.Tensor, q_fixed_b: torch.Tensor, hold_s: float, dt: float) -> None:
        steps = int(max(1, round(float(hold_s) / dt)))
        for _ in range(steps):
            if not simulation_app.is_running():
                return
            _drive_ik_step(goal_pos_b, q_fixed_b, dt)

    # Initial settle and open gripper.
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
    table_top_b = _world_to_base(torch.tensor([0.0, 0.0, float(args.table_z_world)], dtype=torch.float32, device=sim.device))
    stack_top_z_b = float(table_top_b[2])
    place_depth = float(args.grasp_depth_m if args.place_depth_m is None else args.place_depth_m)

    print(
        f"[STACK] Home EE pose: pos={[round(float(v), 4) for v in ee_pos_b_home.tolist()]} "
        f"quat_wxyz={[round(float(v), 4) for v in ee_quat_b_home.tolist()]}"
    )
    print(
        f"[STACK] Stack center base=({float(args.stack_x):+.3f}, {float(args.stack_y):+.3f}); "
        f"initial table_top_z_b={stack_top_z_b:+.3f}; order={args.stack_order}; placement_rotation={args.placement_rotation}"
    )

    blocks: list[BlockInfo] = []
    for prim_path in spawned_paths:
        try:
            grasp_pos_w, grasp_quat_w = grasp_provider.get_grasp_pose_w(object_prim_path=str(prim_path), robot_prim_path=None)
            top_b = _world_to_base(torch.tensor(grasp_pos_w, dtype=torch.float32, device=sim.device))
        except Exception as e:
            print(f"[STACK][WARN] OBB failed for {prim_path}: {e}. Skipping.")
            continue
        blocks.append(
            BlockInfo(
                prim_path=str(prim_path),
                height_m=_read_prim_height_m(str(prim_path)),
                yaw_w_rad=_read_prim_world_yaw_rad(str(prim_path)),
                top_b=top_b,
                grasp_quat_w=grasp_quat_w,
            )
        )

    if args.stack_order == "tallest-first":
        blocks.sort(key=lambda b: float(b.height_m), reverse=True)
    elif args.stack_order == "shortest-first":
        blocks.sort(key=lambda b: float(b.height_m))

    print("[STACK] Planned block order:")
    for i, b in enumerate(blocks):
        yaw_txt = "None" if b.yaw_w_rad is None else f"{math.degrees(float(b.yaw_w_rad)):+.1f} deg"
        print(f"[STACK]   {i + 1}. {b.prim_path} height={float(b.height_m):.3f} m yaw={yaw_txt}")
    print("[STACK] " + "=" * 74)

    stack_yaw_rad: float | None = None
    for idx, block in enumerate(blocks):
        if not simulation_app.is_running():
            break

        top_b = block.top_b
        ee_off = float(args.ee_z_offset_m)
        pregrasp_b = top_b.clone()
        pregrasp_b[2] = float(top_b[2]) + ee_off + float(args.pregrasp_offset_m)
        grasp_b = top_b.clone()
        grasp_b[2] = float(top_b[2]) + ee_off + float(args.grasp_depth_m)
        source_lift_b = top_b.clone()
        source_lift_b[2] = float(top_b[2]) + ee_off + float(args.lift_offset_m)

        ee_pos_b_cur, ee_quat_b_cur = _read_ee_pose_b()
        if bool(args.align_to_obb) and block.yaw_w_rad is not None:
            q_pick_b = _quat_aligned_to_yaw(
                ee_quat_b_cur,
                float(block.yaw_w_rad),
                square_symmetry=(str(args.pickup_yaw_symmetry) == "square"),
            )
            pickup_yaw_log = (
                f"source_yaw={math.degrees(float(block.yaw_w_rad)):+.2f} deg "
                f"target_ee_yaw={math.degrees(_quat_to_yaw(q_pick_b)):+.2f} deg"
            )
        else:
            q_pick_b = ee_quat_b_cur.clone()
            pickup_yaw_log = "pickup yaw alignment disabled/unavailable"

        placement_rotation = str(args.placement_rotation)
        if placement_rotation == "match-stack":
            if stack_yaw_rad is None:
                stack_yaw_rad = float(block.yaw_w_rad) if block.yaw_w_rad is not None else math.radians(float(args.stack_yaw_deg))
            q_place_b = _quat_aligned_to_yaw(q_pick_b, float(stack_yaw_rad), square_symmetry=False)
            place_yaw_log = (
                f"match stack_yaw={math.degrees(float(stack_yaw_rad)):+.2f} deg "
                f"target_ee_yaw={math.degrees(_quat_to_yaw(q_place_b)):+.2f} deg"
            )
        elif placement_rotation == "preserve" or block.yaw_w_rad is None:
            q_place_b = q_pick_b.clone()
            place_yaw_log = "preserve pickup yaw"
        else:
            base_yaw = math.radians(float(args.stack_yaw_deg))
            if placement_rotation == "alternating":
                base_yaw += (idx % 2) * (math.pi / 2.0)
            q_place_b = _quat_aligned_to_yaw(q_pick_b, base_yaw, square_symmetry=False)
            place_yaw_log = f"place_yaw={math.degrees(base_yaw):+.2f} deg"

        block_top_when_placed_b = stack_top_z_b + float(block.height_m)
        place_b = torch.tensor(
            [
                float(args.stack_x),
                float(args.stack_y),
                block_top_when_placed_b + ee_off + place_depth,
            ],
            dtype=torch.float32,
            device=sim.device,
        )
        preplace_b = place_b.clone()
        preplace_b[2] = float(place_b[2]) + float(args.preplace_offset_m)

        print(
            f"[STACK] block {idx + 1}/{len(blocks)}: prim={block.prim_path}\n"
            f"        height={float(block.height_m):.3f} m stack_bottom_z_b={stack_top_z_b:+.3f} "
            f"stack_top_z_b={block_top_when_placed_b:+.3f}\n"
            f"        pickup: top_b=({float(top_b[0]):+.3f},{float(top_b[1]):+.3f},{float(top_b[2]):+.3f}) "
            f"grasp_z={float(grasp_b[2]):+.3f} {pickup_yaw_log}\n"
            f"        place:  preplace=({float(preplace_b[0]):+.3f},{float(preplace_b[1]):+.3f},{float(preplace_b[2]):+.3f}) "
            f"place=({float(place_b[0]):+.3f},{float(place_b[1]):+.3f},{float(place_b[2]):+.3f}) {place_yaw_log}"
        )

        # Pick.
        _run_segment("APPROACH_PICK", pregrasp_b, q_pick_b, dt, q_start_b=ee_quat_b_cur)
        _run_segment("DESCEND_PICK", grasp_b, q_pick_b, dt, q_start_b=q_pick_b)
        _hold_at(grasp_b, q_pick_b, float(args.pre_close_settle_s), dt)
        try:
            gripper.command_close(robot)
        except Exception:
            pass
        _hold_at(grasp_b, q_pick_b, float(args.gripper_close_s), dt)
        print(f"[STACK]   GRIPPER_CLOSE          held {args.gripper_close_s:.2f}s")
        _run_segment("LIFT_FROM_SOURCE", source_lift_b, q_pick_b, dt, q_start_b=q_pick_b)

        # Place on the stack.
        _run_segment("TRANSFER_TO_STACK", preplace_b, q_place_b, dt, q_start_b=q_pick_b)
        _run_segment("DESCEND_TO_PLACE", place_b, q_place_b, dt, q_start_b=q_place_b)
        _hold_at(place_b, q_place_b, float(args.pre_release_settle_s), dt)
        try:
            gripper.command_open(robot)
        except Exception:
            pass
        _hold_at(place_b, q_place_b, float(args.gripper_open_s), dt)
        _hold_at(place_b, q_place_b, float(args.post_release_settle_s), dt)
        print(f"[STACK]   GRIPPER_OPEN           held {args.gripper_open_s:.2f}s")
        _run_segment("RETRACT_FROM_STACK", preplace_b, q_place_b, dt, q_start_b=q_place_b)

        stack_top_z_b = block_top_when_placed_b + float(args.layer_gap_m)
        print("[STACK] " + "-" * 74)

    if simulation_app.is_running():
        ee_idle, _ = _read_ee_pose_b()
        _hold_at(ee_idle, ee_quat_b_home, 1.5, dt)

    print("[STACK] Done.")
    simulation_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
