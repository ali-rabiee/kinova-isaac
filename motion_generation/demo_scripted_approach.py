"""Phase A visual demo: scripted reach -> descent -> grasp -> lift with diff-IK.

This version intentionally ignores orientation alignment (we freeze the EE
quaternion at the home palm-down pose captured after reset) so we can verify
the XYZ motion pipeline first:

    AppLauncher -> IsaacLab scene -> uniform cubes
      -> (per cube) approach pregrasp -> descent to grasp -> close gripper
         -> lift -> open gripper -> return toward home
      -> quintic-eased absolute pose target per physics step
      -> IsaacLab DifferentialIKController (absolute, DLS) -> joint position targets

No cuRobo, no waypoint chase, no logging. Run with a GUI to watch the motion.

Run:
    python motion_generation/demo_scripted_approach.py --device cuda:0 --num-objects 3
Headless smoke test:
    python motion_generation/demo_scripted_approach.py --headless --device cuda:0 --num-objects 1
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

# --- Path bootstrap (same pattern as vla_v0/v1 to survive Kit side effects) ---
ROOT = Path(__file__).resolve().parents[1]
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
    parser.add_argument("--num-objects", type=int, default=3, help="Number of uniform cubes to visit.")
    parser.add_argument("--box-size", type=float, default=0.08, help="Cube side length (m). Bigger = easier to grasp and see.")
    parser.add_argument("--spawn-min", type=float, nargs=3, default=[0.30, -0.30, 0.90], metavar=("X", "Y", "Z"))
    parser.add_argument("--spawn-max", type=float, nargs=3, default=[0.55, 0.30, 0.95], metavar=("X", "Y", "Z"))
    parser.add_argument("--min-distance", type=float, default=0.22, help="Min pairwise distance between cubes (m).")
    # EE geometry. The `j2n6s300_end_effector` link is at the wrist (palm), not the fingertips.
    # Use ``ee_z_offset_m`` as the vertical offset of the EE link above the fingertips when the
    # gripper is open in its nominal (palm-down) pose. This matches the convention used by
    # ``vla_v1`` so the "working" numbers from that pipeline transfer over unchanged.
    parser.add_argument("--ee-z-offset-m", type=float, default=0.08,
                        help="Wrist-to-fingertip vertical offset for Jaco2 (m). Added to all EE z targets.")
    # Waypoint offsets (base frame, relative to cube top Z and ee_z_offset).
    #   pregrasp_z = top_z + ee_z_offset + pregrasp_offset_m     (palm well above the cube)
    #   grasp_z    = top_z + ee_z_offset + grasp_depth_m         (palm just above cube top; fingers wrap)
    #   lift_z     = top_z + ee_z_offset + lift_offset_m         (palm well above, object dangling)
    parser.add_argument("--pregrasp-offset-m", type=float, default=0.12, help="Pregrasp palm offset above (top + ee_z_offset) (m).")
    parser.add_argument("--grasp-depth-m", type=float, default=-0.07,
                        help="Grasp palm offset relative to (top + ee_z_offset) (m). Default ~+0.01 m above cube top -- fingers wrap around.")
    parser.add_argument("--lift-offset-m", type=float, default=0.18, help="Lift palm offset above (top + ee_z_offset) (m).")
    # Motion timing
    parser.add_argument("--cruise-mps", type=float, default=0.20,
                        help="Cartesian cruise speed used to compute segment durations (m/s).")
    parser.add_argument("--min-segment-s", type=float, default=1.0, help="Minimum duration for any motion segment (s).")
    parser.add_argument("--max-segment-s", type=float, default=4.0, help="Hard timeout for any motion segment (s).")
    parser.add_argument("--converge-pos-tol-m", type=float, default=0.005,
                        help="Per-segment position convergence tolerance (m).")
    # Gripper timing
    parser.add_argument("--pre-close-settle-s", type=float, default=0.5,
                        help="Hold at the grasp target BEFORE closing the gripper, so the arm has finished descending and fingers are stable around the cube.")
    parser.add_argument("--gripper-close-s", type=float, default=0.8, help="Time to hold close command before lifting (s).")
    parser.add_argument("--gripper-open-s", type=float, default=0.8, help="Time to hold open command before moving on (s).")
    # Robot wiring
    parser.add_argument("--ee-link", type=str, default="j2n6s300_end_effector")
    parser.add_argument("--arm-joint-regex", type=str, default="j2n6s300_joint_[1-6]$")
    parser.add_argument("--gripper-joint-regex", type=str,
                        default=".*_joint_finger_.*|.*_joint_finger_tip_.*")
    parser.add_argument("--gripper-open-pos", type=float, default=0.0)
    parser.add_argument("--gripper-close-pos", type=float, default=1.2)
    # Yaw alignment to the cube's OBB (keeps palm-down, only rotates around base Z).
    parser.add_argument("--align-to-obb", dest="align_to_obb", action="store_true", default=True,
                        help="(default) Rotate the gripper around base Z during APPROACH to align with the cube's OBB yaw.")
    parser.add_argument("--no-align-to-obb", dest="align_to_obb", action="store_false",
                        help="Disable OBB yaw alignment. Gripper yaw stays at the home-pose value.")
    parser.add_argument("--seed", type=int, default=-1,
                        help="RNG seed for spawn positions/yaws. Use a non-negative int for a reproducible layout; the default (-1) picks a fresh layout every run.")
    # AppLauncher args (headless, device, enable_cameras, etc.)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    # ------------------------------------------------------------------
    # Launch Isaac Sim / Kit *before* heavy imports.
    # ------------------------------------------------------------------
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import random
    import importlib

    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
    from isaaclab.utils.math import subtract_frame_transforms, quat_conjugate, quat_apply

    from environments.utils.object_loader import ObjectLoader, ObjectLoaderConfig, SpawnBounds
    from environments.utils.physix import PhysicsConfig, apply_to_simulation_cfg, object_loader_kwargs_from_physix
    from motion_generation.grasp_estimation.obb import ObbGraspPoseProvider
    from kinova import GripperConfig, GripperController

    env_cfg_mod = importlib.import_module("environments.reach_to_grasp_VLA.config")
    env_utils_mod = importlib.import_module("environments.reach_to_grasp_VLA.utils")
    DEFAULT_SCENE = getattr(env_cfg_mod, "DEFAULT_SCENE")
    DEFAULT_CAMERA = getattr(env_cfg_mod, "DEFAULT_CAMERA", None)
    design_scene = getattr(env_utils_mod, "design_scene")

    # Only seed when a non-negative value is provided; otherwise let every run differ.
    if int(args.seed) >= 0:
        random.seed(int(args.seed))
        print(f"[DEMO] RNG seeded with --seed {int(args.seed)} (reproducible layout).")
    else:
        print("[DEMO] RNG not seeded (layout is random each run; pass --seed <int> to reproduce).")

    # ------------------------------------------------------------------
    # Easing + quaternion helpers (wxyz convention everywhere).
    # ------------------------------------------------------------------
    def _quintic(s: float) -> float:
        s = max(0.0, min(1.0, float(s)))
        return 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5

    def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Hamilton product of two (..., 4) wxyz quaternions."""
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
        """Unit quaternion for a pure rotation about +Z by ``yaw_rad`` (wxyz)."""
        half = 0.5 * float(yaw_rad)
        return torch.tensor([math.cos(half), 0.0, 0.0, math.sin(half)], device=device, dtype=dtype)

    def _quat_to_yaw(q: torch.Tensor) -> float:
        """Extract yaw (rotation about world/base Z) from a wxyz quaternion."""
        w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _wrap_quarter_pi(a: float) -> float:
        """Wrap angle to [-pi/4, pi/4] using pi/2 symmetry (square cube = 4-fold symmetric).

        Keeps the gripper's rotation to at most 45 deg to align, because for a square
        cube rotations by multiples of pi/2 yield an equivalent grasp footprint."""
        a = (a + math.pi / 4.0) % (math.pi / 2.0) - math.pi / 4.0
        return a

    def _slerp(q0: torch.Tensor, q1: torch.Tensor, t: float) -> torch.Tensor:
        """Spherical linear interpolation between two wxyz unit quaternions."""
        q0 = q0 / (q0.norm() + 1e-12)
        q1 = q1 / (q1.norm() + 1e-12)
        dot = float(torch.dot(q0, q1))
        if dot < 0.0:  # always take the short arc
            q1 = -q1
            dot = -dot
        if dot > 0.9995:  # nearly identical -> linear + renormalise
            out = (1.0 - t) * q0 + t * q1
            return out / (out.norm() + 1e-12)
        theta_0 = math.acos(max(-1.0, min(1.0, dot)))
        sin_theta_0 = math.sin(theta_0)
        theta = theta_0 * t
        s0 = math.cos(theta) - dot * math.sin(theta) / sin_theta_0
        s1 = math.sin(theta) / sin_theta_0
        return s0 * q0 + s1 * q1

    class _MoveSegment:
        """Quintic-eased absolute pose segment.

        Position eases from ``p0`` to ``p1`` and orientation SLERPs from ``q0``
        to ``q1`` using the same quintic factor. If ``q0 == q1`` the SLERP is a
        no-op and the orientation stays constant (used for descend/lift).

        After ``min_duration_s`` the target stays pinned to ``(p1, q1)`` so the
        outer loop can continue driving until convergence.
        """

        def __init__(self, p0: torch.Tensor, p1: torch.Tensor,
                     q0: torch.Tensor, q1: torch.Tensor,
                     min_duration_s: float, max_duration_s: float) -> None:
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
            s = self.t_elapsed / self.min_duration_s
            se = _quintic(s)
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

    # Spawn uniform cubes.
    loader_cfg = ObjectLoaderConfig(
        dataset_dirs=[],  # unused in box mode
        bounds=SpawnBounds(min_xyz=tuple(args.spawn_min), max_xyz=tuple(args.spawn_max)),
        min_distance=float(args.min_distance),
        spawn_mode="box",
        box_size_min=(float(args.box_size), float(args.box_size), float(args.box_size)),
        box_size_max=(float(args.box_size), float(args.box_size), float(args.box_size)),
        box_color_palette=[(0.9, 0.2, 0.2), (0.2, 0.4, 0.9), (0.2, 0.9, 0.3), (0.9, 0.8, 0.2), (0.7, 0.3, 0.8)],
        box_color_names=["red", "blue", "green", "yellow", "purple"],
        **object_loader_kwargs_from_physix(phys),
    )
    loader = ObjectLoader(loader_cfg)
    spawned_paths = loader.spawn(parent_prim_path="/World/Origin1", num_objects=int(args.num_objects))
    if len(spawned_paths) == 0:
        print("[DEMO][ERROR] No objects spawned. Aborting.")
        simulation_app.close()
        return 2

    # Reset sim and robot to default state.
    sim.reset()
    origin0 = torch.tensor(scene_origins[0], device=sim.device)
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += origin0
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    robot.reset()

    # ------------------------------------------------------------------
    # Resolve joint / body ids.
    # ------------------------------------------------------------------
    arm_joint_ids_t, _ = robot.find_joints(str(args.arm_joint_regex))
    if hasattr(arm_joint_ids_t, "view"):
        arm_joint_ids = [int(v) for v in arm_joint_ids_t.view(-1).tolist()]
    else:
        arm_joint_ids = [int(v) for v in list(arm_joint_ids_t)]
    arm_joint_names = [str(robot.data.joint_names[i]) for i in arm_joint_ids]
    print(f"[DEMO] arm joints: {arm_joint_names}")

    ee_body_ids, _ = robot.find_bodies([str(args.ee_link)])
    ee_body_id = int(ee_body_ids[0])
    ee_jacobi_idx = ee_body_id - 1 if robot.is_fixed_base else ee_body_id

    # ------------------------------------------------------------------
    # Differential IK: absolute pose target, damped least-squares.
    # ------------------------------------------------------------------
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik = DifferentialIKController(diff_ik_cfg, num_envs=1, device=sim.device)
    diff_ik.reset()

    # ------------------------------------------------------------------
    # Gripper controller (reuse the project-wide kinova.GripperController).
    # ------------------------------------------------------------------
    gripper_cfg = GripperConfig(
        joint_regex=str(args.gripper_joint_regex),
        open_position=float(args.gripper_open_pos),
        close_position=float(args.gripper_close_pos),
    )
    gripper = GripperController(gripper_cfg, num_envs=1, device=str(sim.device))
    gripper.resolve_joints(robot)
    gripper.reset(robot)
    try:
        prim_path = str(getattr(getattr(robot, "cfg", None), "prim_path", None))
        if prim_path:
            gripper.set_drive_gains(prim_path)
            gripper.apply_stable_grasp_tuning(prim_path)
    except Exception:
        pass

    # ------------------------------------------------------------------
    # OBB grasp provider (we only use its POSITION output in this demo).
    # ------------------------------------------------------------------
    grasp_provider = ObbGraspPoseProvider(align_to_min_width=False)

    # ------------------------------------------------------------------
    # Direct USD yaw reader.
    # The existing ObbGraspPoseProvider uses UsdGeom.BBoxCache.ComputeWorldBound,
    # which for USD shape prims (Cuboid/Sphere/etc.) returns a world-AABB-style
    # GfBBox3d whose matrix is effectively identity. That silently strips the
    # cube's yaw, and the provider returns yaw=0 no matter how the cube was
    # spawned. For this demo we just read the prim's world transform directly
    # and take its yaw about +Z, which is what we actually want to align to.
    # ------------------------------------------------------------------
    import importlib as _importlib_for_yaw
    _pxr_usd = _importlib_for_yaw.import_module("pxr.Usd")
    _pxr_usdgeom = _importlib_for_yaw.import_module("pxr.UsdGeom")
    _omni_usd = _importlib_for_yaw.import_module("omni.usd")
    _usd_stage = _omni_usd.get_context().get_stage()

    def _read_prim_world_yaw_rad(prim_path: str) -> float | None:
        """Yaw (rotation about world +Z) of a prim's world transform, in radians.

        Returns None if the prim cannot be read.
        """
        try:
            prim = _usd_stage.GetPrimAtPath(str(prim_path))
            if not prim.IsValid():
                return None
            xformable = _pxr_usdgeom.Xformable(prim)
            M = xformable.ComputeLocalToWorldTransform(_pxr_usd.TimeCode.Default())
            # USD uses row-vector post-multiplication: v_world = v_local * M
            # -> local +X ends up at world = (M[0][0], M[0][1], M[0][2])
            m00 = float(M[0][0])
            m01 = float(M[0][1])
            return math.atan2(m01, m00)
        except Exception as e:
            print(f"[DEMO][WARN] could not read world yaw for {prim_path}: {e}")
            return None

    # ------------------------------------------------------------------
    # Helpers.
    # ------------------------------------------------------------------
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
        pos_b = quat_apply(base_quat_inv.unsqueeze(0), rel_w)
        return pos_b[0]

    def _drive_ik_step(p_des_b: torch.Tensor, q_des_b: torch.Tensor, dt: float) -> None:
        jac = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, arm_joint_ids]
        q_arm = robot.data.joint_pos[:, arm_joint_ids]
        ee_pos_b_cur, ee_quat_b_cur = _read_ee_pose_b()

        diff_ik.ee_pos_des[:] = p_des_b.unsqueeze(0)
        diff_ik.ee_quat_des[:] = q_des_b.unsqueeze(0)
        q_des = diff_ik.compute(ee_pos_b_cur.unsqueeze(0), ee_quat_b_cur.unsqueeze(0), jac, q_arm)

        # Hold everything at current pose, then override the arm target with the IK solution.
        robot.set_joint_position_target(robot.data.joint_pos)
        robot.set_joint_position_target(q_des, joint_ids=arm_joint_ids)
        robot.set_joint_velocity_target(torch.zeros_like(robot.data.joint_vel))

        # Gripper: honor the controller's current open/close command each step.
        try:
            gripper.apply_hold(robot)
        except Exception:
            pass

        # Gravity compensation.
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

    def _run_segment(label: str, goal_pos_b: torch.Tensor, q_end_b: torch.Tensor,
                     dt: float, q_start_b: torch.Tensor | None = None) -> tuple[bool, float]:
        """Drive the EE to ``goal_pos_b`` with orientation SLERPing from ``q_start_b`` to ``q_end_b``.

        If ``q_start_b`` is None we default to the current EE orientation (so a pose segment
        with only ``q_end_b`` specified behaves like "rotate in place toward q_end_b").
        Pass ``q_start_b == q_end_b`` for pure-XYZ motion with no rotation.

        Uses quintic easing for at least ``min_duration_s`` (distance-proportional),
        then keeps driving until convergence or ``max_duration_s`` timeout.
        Returns (converged, final_err_m).
        """
        p0, q0_measured = _read_ee_pose_b()
        q_start_b = q0_measured if q_start_b is None else q_start_b
        dist = float((goal_pos_b - p0).norm())
        min_dur = max(float(args.min_segment_s), dist / max(1e-6, float(args.cruise_mps)))
        max_dur = max(min_dur + 0.5, float(args.max_segment_s))
        seg = _MoveSegment(p0=p0, p1=goal_pos_b, q0=q_start_b, q1=q_end_b,
                           min_duration_s=min_dur, max_duration_s=max_dur)
        pos_tol = float(args.converge_pos_tol_m)

        converged = False
        while simulation_app.is_running() and not seg.timed_out:
            p_t, q_t = seg.current()
            _drive_ik_step(p_t, q_t, dt)
            seg.advance(dt)
            if seg.eased_complete:
                err = _pos_err_m(goal_pos_b)
                if err < pos_tol:
                    converged = True
                    break

        final_err = _pos_err_m(goal_pos_b)
        status = "OK" if converged else ("TIMEOUT" if seg.timed_out else "EXIT")
        print(f"[DEMO]   {label:<22} dist={dist*1000:6.1f} mm "
              f"min_dur={min_dur:.2f}s t={seg.t_elapsed:.2f}s  final_err={final_err*1000:6.1f} mm  [{status}]")
        return converged, final_err

    def _hold_at(goal_pos_b: torch.Tensor, q_fixed_b: torch.Tensor, hold_s: float, dt: float) -> None:
        """Keep driving the IK toward ``goal_pos_b`` for ``hold_s`` seconds."""
        steps = int(max(1, round(float(hold_s) / dt)))
        for _ in range(steps):
            if not simulation_app.is_running():
                return
            _drive_ik_step(goal_pos_b, q_fixed_b, dt)

    # ------------------------------------------------------------------
    # Initial settle: let the robot stabilize at home, open the gripper,
    # and capture the home palm-down quaternion to reuse for all targets.
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
    q_home_b = ee_quat_b_home.clone()  # <-- frozen orientation for the entire demo
    print(
        f"[DEMO] Home EE pose (base frame): pos={[round(float(v),4) for v in ee_pos_b_home.tolist()]} "
        f"quat_wxyz={[round(float(v),4) for v in q_home_b.tolist()]}"
    )
    if bool(getattr(args, "align_to_obb", True)):
        print("[DEMO] Yaw alignment: ON  (gripper rotates around base Z during APPROACH to match cube OBB yaw,")
        print("[DEMO]                    wrapped to [-pi/4, +pi/4] for 4-fold cube symmetry).")
    else:
        print("[DEMO] Yaw alignment: OFF (--no-align-to-obb).")
    print(f"[DEMO] Spawned {len(spawned_paths)} cubes. Full pick sequence for each.")
    print("[DEMO] " + "=" * 74)

    # ------------------------------------------------------------------
    # Main loop: pick each cube one by one.
    # ------------------------------------------------------------------
    for idx, prim_path in enumerate(spawned_paths):
        if not simulation_app.is_running():
            break

        # 1. Compute cube top center in base frame + OBB yaw (world Z rotation).
        try:
            grasp_pos_w, grasp_quat_w = grasp_provider.get_grasp_pose_w(
                object_prim_path=str(prim_path), robot_prim_path=None
            )
        except Exception as e:
            print(f"[DEMO][WARN] OBB failed for {prim_path}: {e}. Skipping.")
            continue
        grasp_pos_w_t = torch.tensor(grasp_pos_w, dtype=torch.float32, device=sim.device)
        top_b = _world_to_base(grasp_pos_w_t)

        ee_off = float(args.ee_z_offset_m)
        pregrasp_b = top_b.clone()
        pregrasp_b[2] = top_b[2] + ee_off + float(args.pregrasp_offset_m)
        grasp_b = top_b.clone()
        grasp_b[2] = top_b[2] + ee_off + float(args.grasp_depth_m)  # palm ~1cm above cube top by default
        lift_b = top_b.clone()
        lift_b[2] = top_b[2] + ee_off + float(args.lift_offset_m)

        # 1b. Yaw alignment.
        # Robot base is gravity-aligned, so world yaw == base yaw. We wrap the
        # delta from the CURRENT EE yaw to the cube's world yaw into [-pi/4, pi/4]
        # so a square cube only ever needs <=45 deg of rotation to align.
        #
        # IMPORTANT: We read the cube's yaw DIRECTLY from its USD xform, not via
        # ObbGraspPoseProvider, because for USD shape prims the OBB provider's
        # bbox-cache path strips the cube's rotation and always returns 0 deg.
        ee_pos_b_cur, ee_quat_b_cur = _read_ee_pose_b()
        current_yaw = _quat_to_yaw(ee_quat_b_cur)
        cube_yaw_w = _read_prim_world_yaw_rad(str(prim_path))
        if bool(getattr(args, "align_to_obb", True)) and cube_yaw_w is not None:
            delta_yaw = _wrap_quarter_pi(cube_yaw_w - current_yaw)
            q_delta = _yaw_quat(delta_yaw, device=sim.device)
            q_aligned_b = _quat_mul(q_delta, ee_quat_b_cur)
            aligned_yaw = _quat_to_yaw(q_aligned_b)
            yaw_log = (f"  cube_yaw_w={math.degrees(cube_yaw_w):+7.2f} deg,"
                       f" current_yaw={math.degrees(current_yaw):+7.2f} deg,"
                       f" delta={math.degrees(delta_yaw):+6.2f} deg,"
                       f" target_yaw={math.degrees(aligned_yaw):+7.2f} deg")
        else:
            q_aligned_b = ee_quat_b_cur.clone()  # keep current orientation
            if cube_yaw_w is None and bool(getattr(args, "align_to_obb", True)):
                yaw_log = "  (cube yaw unavailable -- keeping current EE orientation)"
            else:
                yaw_log = "  (yaw alignment disabled via --no-align-to-obb)"

        print(
            f"[DEMO] cube {idx+1}/{len(spawned_paths)}: prim={prim_path}\n"
            f"       top_b      =({float(top_b[0]):+.3f},{float(top_b[1]):+.3f},{float(top_b[2]):+.3f}) m\n"
            f"       pregrasp_b =({float(pregrasp_b[0]):+.3f},{float(pregrasp_b[1]):+.3f},{float(pregrasp_b[2]):+.3f}) m  (top + ee_off + {args.pregrasp_offset_m:+.3f})\n"
            f"       grasp_b    =({float(grasp_b[0]):+.3f},{float(grasp_b[1]):+.3f},{float(grasp_b[2]):+.3f}) m  (top + ee_off + {args.grasp_depth_m:+.3f})\n"
            f"       lift_b     =({float(lift_b[0]):+.3f},{float(lift_b[1]):+.3f},{float(lift_b[2]):+.3f}) m  (top + ee_off + {args.lift_offset_m:+.3f})\n"
            f"       yaw: {yaw_log}"
        )

        # 2. Approach pregrasp -- SLERPs yaw from current to aligned while moving XY+Z.
        _run_segment("APPROACH_PREGRASP", pregrasp_b, q_aligned_b, dt, q_start_b=ee_quat_b_cur)

        # 3. Descend to grasp position (yaw FIXED at q_aligned_b, no rotation).
        #    Note: convergence is best-effort. If the fingers contact the cube/table before the EE
        #    target is reached, the segment will time out with a non-zero error. That is OK --
        #    the arm is at the cube. The pre-close settle below then lets the contact stabilize
        #    so the gripper closes ON the cube, not in mid-air.
        _run_segment("DESCEND_TO_GRASP", grasp_b, q_aligned_b, dt, q_start_b=q_aligned_b)

        # 4. Settle at the grasp target so any finger-cube contact stabilizes before we close.
        _hold_at(grasp_b, q_aligned_b, float(args.pre_close_settle_s), dt)
        ee_pos_b_now, ee_quat_b_now = _read_ee_pose_b()
        dx, dy, dz = (
            float(ee_pos_b_now[0] - grasp_b[0]) * 1000.0,
            float(ee_pos_b_now[1] - grasp_b[1]) * 1000.0,
            float(ee_pos_b_now[2] - grasp_b[2]) * 1000.0,
        )
        yaw_err_deg = math.degrees(_quat_to_yaw(ee_quat_b_now) - _quat_to_yaw(q_aligned_b))
        print(
            f"[DEMO]   PRE_CLOSE_SETTLE      held {args.pre_close_settle_s:.2f}s  "
            f"ee-vs-grasp: dx={dx:+.1f} dy={dy:+.1f} dz={dz:+.1f} mm  "
            f"yaw_err={yaw_err_deg:+.2f} deg  -> closing gripper NOW"
        )

        # 5. Close the gripper while holding the grasp target (yaw fixed).
        try:
            gripper.command_close(robot)
        except Exception:
            pass
        _hold_at(grasp_b, q_aligned_b, float(args.gripper_close_s), dt)
        print(f"[DEMO]   GRIPPER_CLOSE          held {args.gripper_close_s:.2f}s")

        # 6. Lift (gripper stays closed, yaw stays at aligned).
        _run_segment("LIFT", lift_b, q_aligned_b, dt, q_start_b=q_aligned_b)

        # 7. Release the gripper above the cube (drops it back roughly in place).
        try:
            gripper.command_open(robot)
        except Exception:
            pass
        _hold_at(lift_b, q_aligned_b, float(args.gripper_open_s), dt)
        print(f"[DEMO]   GRIPPER_OPEN           held {args.gripper_open_s:.2f}s")

        # 8. Short hold at lift height before heading to the next cube.
        #    The NEXT APPROACH_PREGRASP will read the current EE quat as its start and
        #    SLERP toward the next cube's aligned yaw.
        _hold_at(lift_b, q_aligned_b, 0.2, dt)

        print("[DEMO] " + "-" * 74)

    # Gentle idle hold at the end so the viewport doesn't collapse.
    if simulation_app.is_running():
        ee_idle, _ = _read_ee_pose_b()
        _hold_at(ee_idle, q_home_b, 1.5, dt)

    print("[DEMO] Done.")
    simulation_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
