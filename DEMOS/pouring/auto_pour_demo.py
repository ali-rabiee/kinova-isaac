"""Scripted auto-pour demo: side-grasp the pitcher, carry it over the glass, tilt to pour.

This is the pouring analogue of ``DEMOS/block_stacking/block_stacking_demo.py``.
It reuses the exact same diff-IK + gripper pipeline (``_MoveSegment`` /
``_run_segment`` / ``_drive_ik_step`` / quintic-eased Cartesian segments) but
swaps the top-down cube grasp for a *side grasp + wrist-roll pour*:

    PouringEnv -> spawn pitcher + glass + pellets
      -> settle (pellets fall to the bottom of the pitcher)
      -> approach the pitcher from the side (horizontal palm) and grasp its body
      -> lift it off the table
      -> carry it above the glass
      -> roll the wrist about the approach axis to tilt the pitcher and pour
      -> hold while pellets fall into the glass (logging the count each tick)
      -> (optionally) raise the pitcher back upright to stop the pour

Why a side grasp: to pour you must be able to *tilt* the vessel. The home pose
grasps palm-down (good for cubes), so we reorient the gripper to a horizontal
"palm-forward" pose, wrap the round body, and then roll about the approach axis
-- that rolls the held cylinder's mouth down toward the glass.

How the pour amount is measured: :meth:`PouringEnv.count_pellets_in_glass`
counts pellets whose world position is inside the volume-sensor box anchored on
the glass. The count (and an estimated mass in grams) is printed as ``[POUR]``
lines so success is visible even when running ``--headless``.

Run headless smoke test (recommended for CI / logs):

    conda activate kinova
    python DEMOS/pouring/auto_pour_demo.py --headless --device cuda:0 --num-pellets 40

Run with a GUI:

    python DEMOS/pouring/auto_pour_demo.py --device cuda:0
"""

from __future__ import annotations

import argparse
import math
import sys
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

    # ---- Scene: pellets ----
    parser.add_argument("--num-pellets", type=int, default=50, help="Pellets pre-filled in the pitcher.")
    parser.add_argument("--pellet-radius-m", type=float, default=0.006, help="Pellet radius (m).")
    parser.add_argument(
        "--pellet-color", type=float, nargs=3, default=[0.10, 0.45, 0.95],
        metavar=("R", "G", "B"), help="Pellet RGB color in [0, 1].",
    )

    # ---- Scene: pitcher (the vessel we pick up and tilt) ----
    parser.add_argument(
        "--pitcher-pos", type=float, nargs=3, default=[0.55, 0.18, 0.83],
        metavar=("X", "Y", "Z"), help="Pitcher world spawn position (bottom-center). Pushed out in +X so the arm grasps it with a more natural, extended reach.",
    )
    parser.add_argument("--pitcher-radius-m", type=float, default=0.040, help="Pitcher interior radius (m). A wider body is easier for the gripper to clamp.")
    parser.add_argument("--pitcher-height-m", type=float, default=0.14, help="Pitcher cavity height (m).")
    parser.add_argument(
        "--pitcher-color", type=float, nargs=3, default=[0.20, 0.45, 0.95],
        metavar=("R", "G", "B"), help="Pitcher RGB color.",
    )
    parser.add_argument(
        "--pitcher-mass-kg", type=float, default=0.12,
        help="Pitcher mass (kg). A side grasp of a smooth cylinder holds by friction only, "
             "so a lighter vessel is easier to lift without slip.",
    )
    parser.add_argument(
        "--pitcher-friction", type=float, default=6.0,
        help="Static/dynamic friction of the pitcher walls (combine mode max).",
    )

    # ---- Scene: glass (the receiver) ----
    parser.add_argument(
        "--glass-pos", type=float, nargs=3, default=[0.55, -0.18, 0.83],
        metavar=("X", "Y", "Z"), help="Glass world spawn position (bottom-center). Pushed out in +X to match the pitcher.",
    )
    parser.add_argument("--glass-radius-m", type=float, default=0.050, help="Glass interior radius (m).")
    parser.add_argument("--glass-height-m", type=float, default=0.08, help="Glass cavity height (m).")
    parser.add_argument(
        "--glass-color", type=float, nargs=3, default=[0.80, 0.95, 0.98],
        metavar=("R", "G", "B"), help="Glass RGB color.",
    )

    parser.add_argument("--wall-thickness-m", type=float, default=0.005, help="Wall + base thickness for both vessels (m).")
    parser.add_argument("--n-segments", type=int, default=16, help="Polygon wall slabs per cylinder (more = rounder).")
    parser.add_argument(
        "--volume-half-extents", type=float, nargs=3, default=[0.060, 0.060, 0.050],
        metavar=("HX", "HY", "HZ"), help="Half-extents of the volume-sensor box anchored on the glass.",
    )

    # ---- Grasp geometry ----
    parser.add_argument(
        "--ee-z-offset-m", type=float, default=0.08,
        help="Wrist (j2n6s300_end_effector) to fingertip distance along the palm/approach axis (m).",
    )
    parser.add_argument(
        "--grasp-frac", type=float, default=0.5,
        help="Height up the pitcher body (0=floor, 1=rim) where the fingers wrap it.",
    )
    parser.add_argument(
        "--grasp-reach-m", type=float, default=None,
        help=(
            "Distance from the wrist (j2n6s300_end_effector) to the pitcher center axis along the "
            "approach axis (m). If omitted, it is derived as outer_radius + --grasp-gap-m so the palm "
            "sits just in front of the near wall and the fingers wrap the full diameter (the same "
            "palm-near-the-surface geometry the cube grasp uses)."
        ),
    )
    parser.add_argument(
        "--grasp-gap-m", type=float, default=-0.015,
        help="Gap between the palm and the pitcher's near wall when --grasp-reach-m is auto-derived (m). "
             "Negative drives the wrist PAST the near wall so the open fingers close in deeper around the "
             "mug before grasping (go closer before closing). Less negative / positive backs the palm off.",
    )
    parser.add_argument(
        "--grasp-roll-deg", type=float, default=70.0,
        help=(
            "Roll (deg) of the gripper ABOUT ITS OWN WRIST/approach axis -- this is the knob that aligns "
            "the fingers with the pitcher. The bare reorientation from palm-down to palm-forward leaves the "
            "finger-closing axis vertical, so the fingers slide along the upright cylinder instead of "
            "clamping its diameter; ~90 deg turns the closing axis horizontal so the fingers pinch across "
            "the cylinder. For a round pitcher any value that makes the fingers horizontal works; sweep it "
            "if you change the gripper. This roll is applied as the final 'rotate to grasp' motion (see "
            "--rotate-during-grasp), and the pour roll is applied on top of it so the tilt direction is "
            "unaffected."
        ),
    )
    parser.add_argument(
        "--rotate-during-grasp", dest="rotate_during_grasp", action="store_true", default=True,
        help="Stage the approach as forward -> down -> rotate-to-grasp: reach in palm-forward, descend, then "
             "roll the wrist (--grasp-roll-deg) to align with the pitcher while advancing into the grasp.",
    )
    parser.add_argument(
        "--no-rotate-during-grasp", dest="rotate_during_grasp", action="store_false",
        help="Reorient fully (palm-forward + wrist roll) up front and just translate in, like the original approach.",
    )
    parser.add_argument("--pregrasp-back-m", type=float, default=0.12, help="Extra stand-off behind the grasp along the approach axis (m).")
    parser.add_argument("--approach-height-m", type=float, default=0.12, help="Height above the grasp at which the gripper first reaches forward, before descending (m).")
    parser.add_argument("--lift-m", type=float, default=0.20, help="How high to lift the pitcher after grasping (m).")

    # ---- Pour geometry ----
    parser.add_argument("--pour-y-offset-m", type=float, default=0.06, help="Offset of the pitcher body from the glass center, on the +tilt side (m). Aims the stream into the cup.")
    parser.add_argument("--pour-height-m", type=float, default=0.11, help="Height of the pitcher grasp point above the glass rim before tilting (m). Kept high so the tilting pitcher clears the glass rim (a deeper/lower tilt dips the pitcher into the cup and destabilises the contact).")
    parser.add_argument("--pour-angle-deg", type=float, default=110.0, help="Wrist roll about the approach axis to tilt the pitcher (deg). Past horizontal so the pitcher fully empties.")
    parser.add_argument("--pour-sign", type=float, default=1.0, help="Sign of the pour roll. +1 tips toward -Y (toward the default glass), -1 toward +Y.")
    parser.add_argument("--pre-pour-settle-s", type=float, default=0.4, help="Hold over the glass before tilting, so the carry momentum dissipates.")
    parser.add_argument("--pour-tilt-s", type=float, default=4.5, help="Time to ramp from upright to the full pour angle (s). Slow = gentle pour.")
    parser.add_argument("--pour-hold-s", type=float, default=4.0, help="Time to hold at the pour angle while pellets drain (s).")
    parser.add_argument("--return-upright", action="store_true", help="After pouring, roll the pitcher back upright to stop the pour.")

    # ---- Motion timing ----
    parser.add_argument("--cruise-mps", type=float, default=0.28, help="Cartesian cruise speed for segment durations (m/s). Higher = faster moves.")
    parser.add_argument("--min-segment-s", type=float, default=0.7, help="Minimum duration for any motion segment (s).")
    parser.add_argument("--max-segment-s", type=float, default=6.0, help="Hard timeout for any motion segment (s).")
    parser.add_argument("--converge-pos-tol-m", type=float, default=0.008, help="Position convergence tolerance (m).")

    # ---- Gripper / settle timing ----
    parser.add_argument("--settle-steps", type=int, default=60, help="Physics steps to let the pellets settle in the pitcher before grasping.")
    parser.add_argument("--pre-close-settle-s", type=float, default=0.15, help="Hold at grasp pose before closing.")
    parser.add_argument("--gripper-close-s", type=float, default=0.4, help="Hold the close command before lifting. Short is fine: the weld pins the mug and the fingers keep closing during the lift.")
    parser.add_argument("--post-pour-settle-s", type=float, default=1.0, help="Extra hold after the pour to let stragglers fall.")

    # ---- Robot wiring ----
    parser.add_argument("--ee-link", type=str, default="j2n6s300_end_effector")
    parser.add_argument("--arm-joint-regex", type=str, default="j2n6s300_joint_[1-6]$")
    parser.add_argument("--gripper-joint-regex", type=str, default=".*_joint_finger_.*|.*_joint_finger_tip_.*")
    parser.add_argument("--gripper-open-pos", type=float, default=0.0)
    # A side grasp of a smooth cylinder resists gravity only by friction, so we
    # close harder and with stiffer/stronger drives than the cube grasp uses.
    parser.add_argument("--gripper-close-pos", type=float, default=1.5)
    parser.add_argument("--gripper-stiffness", type=float, default=1500.0, help="Gripper joint drive stiffness (higher = harder squeeze).")
    parser.add_argument("--gripper-damping", type=float, default=150.0, help="Gripper joint drive damping.")
    parser.add_argument("--gripper-effort", type=float, default=2000.0, help="Gripper joint drive effort/force limit.")

    parser.add_argument("--pour-log-period-s", type=float, default=0.5, help="Print the pour count every N seconds during the pour.")
    parser.add_argument("--success-frac", type=float, default=0.3, help="Fraction of pellets in the glass that counts as a successful pour (for the final summary).")

    # ---- Grasp realization ----
    # A friction-only side grasp of a smooth, thin-walled cylinder does not develop
    # enough holding force in PhysX to lift it (the contact normals are ~radial and
    # the vessel slides straight down out of the fingers). So once the fingers have
    # closed around the pitcher we kinematically weld it to the end-effector for the
    # lift/carry/tilt; the pellets stay fully dynamic, so the pour itself is real
    # physics. Pass --no-weld to rely on pure friction instead (useful for tuning).
    parser.add_argument(
        "--no-weld", dest="weld", action="store_false", default=True,
        help="Do not weld the pitcher to the gripper on grasp; rely on friction only.",
    )

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import torch
    from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
    from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul, subtract_frame_transforms

    from environments.pouring import (
        GlassConfig,
        PelletConfig,
        PitcherConfig,
        PouringEnv,
        VolumeSensorConfig,
    )
    from environments.pouring.config import DEFAULT_GLASS, DEFAULT_PITCHER
    from kinova import GripperConfig, GripperController

    headless = bool(getattr(args, "headless", False))
    # NOTE: render must stay True even when headless. The volume sensor
    # (PouringEnv.count_pellets_in_glass) reads each pellet's pose from its USD
    # xform, which is only kept in sync with PhysX when the render/fabric flush
    # runs. With render=False the sensor would read stale spawn positions and
    # always report 0. The block-stacking demo likewise steps with render=True
    # in its headless smoke test.
    render = True

    # ------------------------------------------------------------------
    # Small quaternion helpers (wxyz convention, base frame).
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

    def _axis_angle_quat(axis_xyz, angle_rad: float, *, device, dtype=torch.float32) -> torch.Tensor:
        ax = torch.tensor(axis_xyz, device=device, dtype=dtype)
        ax = ax / (ax.norm() + 1e-12)
        half = 0.5 * float(angle_rad)
        s = math.sin(half)
        return torch.tensor(
            [math.cos(half), float(ax[0]) * s, float(ax[1]) * s, float(ax[2]) * s],
            device=device, dtype=dtype,
        )

    def _shortest_arc_quat(u_xyz, v_xyz, *, device, dtype=torch.float32) -> torch.Tensor:
        """Quaternion (base frame) rotating unit vector u onto unit vector v."""
        u = torch.tensor(u_xyz, device=device, dtype=dtype)
        v = torch.tensor(v_xyz, device=device, dtype=dtype)
        u = u / (u.norm() + 1e-12)
        v = v / (v.norm() + 1e-12)
        d = float(torch.dot(u, v))
        if d >= 1.0 - 1e-8:
            return torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype)
        if d <= -1.0 + 1e-8:
            # 180 deg: pick any axis perpendicular to u.
            ref = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
            if abs(float(u[0])) > 0.9:
                ref = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
            axis = torch.cross(u, ref, dim=-1)
            axis = axis / (axis.norm() + 1e-12)
            return _axis_angle_quat(axis.tolist(), math.pi, device=device, dtype=dtype)
        axis = torch.cross(u, v, dim=-1)
        axis = axis / (axis.norm() + 1e-12)
        angle = math.acos(max(-1.0, min(1.0, d)))
        return _axis_angle_quat(axis.tolist(), angle, device=device, dtype=dtype)

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
    # Build sim + scene + robot via the pouring environment.
    # ------------------------------------------------------------------
    pitcher_cfg = PitcherConfig(
        spawn_position=tuple(args.pitcher_pos),
        inner_radius_m=float(args.pitcher_radius_m),
        inner_height_m=float(args.pitcher_height_m),
        wall_thickness_m=float(args.wall_thickness_m),
        base_thickness_m=float(args.wall_thickness_m),
        n_segments=int(args.n_segments),
        color_rgb=tuple(args.pitcher_color),
        mass_kg=float(args.pitcher_mass_kg),
        static_friction=float(args.pitcher_friction),
        dynamic_friction=float(args.pitcher_friction),
    )
    glass_cfg = GlassConfig(
        spawn_position=tuple(args.glass_pos),
        inner_radius_m=float(args.glass_radius_m),
        inner_height_m=float(args.glass_height_m),
        wall_thickness_m=float(args.wall_thickness_m),
        base_thickness_m=float(args.wall_thickness_m),
        n_segments=int(args.n_segments),
        color_rgb=tuple(args.glass_color),
        mass_kg=DEFAULT_GLASS.mass_kg,
    )
    pellet_cfg = PelletConfig(
        count=int(args.num_pellets),
        radius_m=float(args.pellet_radius_m),
        color_rgb=tuple(args.pellet_color),
        restitution=0.0,  # don't bounce back out of the glass
    )
    volume_cfg = VolumeSensorConfig(half_extents_xyz=tuple(args.volume_half_extents))

    env = PouringEnv(
        device=str(getattr(args, "device", "cuda:0")),
        pitcher_cfg=pitcher_cfg,
        glass_cfg=glass_cfg,
        pellet_cfg=pellet_cfg,
        volume_sensor_cfg=volume_cfg,
    )
    sim = env.build_simulation()
    if not headless:
        env.set_default_camera_view()
    env.design_scene()
    robot = env.robot

    pitcher_path, glass_path = env.spawn_containers()
    pellet_paths = env.spawn_pellets(seed=0)
    print(f"[POUR] pitcher={pitcher_path} glass={glass_path} pellets={len(pellet_paths)}")

    env.reset()

    # ------------------------------------------------------------------
    # Arm / IK / gripper wiring (identical to the block-stacking demo).
    # ------------------------------------------------------------------
    arm_joint_ids_t, _ = robot.find_joints(str(args.arm_joint_regex))
    if hasattr(arm_joint_ids_t, "view"):
        arm_joint_ids = [int(v) for v in arm_joint_ids_t.view(-1).tolist()]
    else:
        arm_joint_ids = [int(v) for v in list(arm_joint_ids_t)]
    arm_joint_names = [str(robot.data.joint_names[i]) for i in arm_joint_ids]
    print(f"[POUR] arm joints: {arm_joint_names}")

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
        stiffness=float(args.gripper_stiffness),
        damping=float(args.gripper_damping),
        effort_limit=float(args.gripper_effort),
        # Close the tips fully too (default ratio 0.4 leaves the fingertips open,
        # which barely cages a round body); a stronger tip wrap holds the cylinder.
        tip_ratio_on_close=0.7,
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

    # ------------------------------------------------------------------
    # Kinematic-weld handle for the pitcher (see --no-weld). RigidPrim writes
    # straight to the physics state (works under the GPU pipeline), so we can
    # pin the pitcher's pose to the end-effector each step once it is grasped.
    # ------------------------------------------------------------------
    weld_enabled = bool(args.weld)
    pitcher_rb = None
    if weld_enabled:
        try:
            from isaacsim.core.prims import RigidPrim
            pitcher_rb = RigidPrim(pitcher_path)
            try:
                pitcher_rb.initialize()
            except Exception:
                pass
            # Sanity-check the handle works before we rely on it.
            _ = pitcher_rb.get_world_poses()
            print("[POUR][WELD] Pitcher RigidPrim handle ready; will weld to EE on grasp.")
        except Exception as e:
            print(f"[POUR][WELD] RigidPrim unavailable ({e}); falling back to friction-only grasp.")
            weld_enabled = False
            pitcher_rb = None

    weld_state = {"active": False, "rel_pos": None, "rel_quat": None}

    # ------------------------------------------------------------------
    # Physics-based pellet readout. The volume sensor in PouringEnv reads each
    # pellet's pose from its USD xform, but under the headless GPU pipeline the
    # USD attributes are NOT written back from PhysX (the fabric/renderer holds
    # the live poses), so a USD read returns the authored spawn pose forever and
    # the sensor would always report 0. Reading the pellets through a RigidPrim
    # view goes straight to the physics state, so the pour is measured correctly.
    # ------------------------------------------------------------------
    pellets_rb = None
    if env.pellet_prim_paths:
        try:
            from isaacsim.core.prims import RigidPrim
            pellets_parent = str(env.pellet_prim_paths[0]).rsplit("/", 1)[0]
            pellets_rb = RigidPrim(f"{pellets_parent}/Pellet_.*")
            try:
                pellets_rb.initialize()
            except Exception:
                pass
            _pos_chk, _ = pellets_rb.get_world_poses()
            print(f"[POUR] Pellet physics view ready ({int(_pos_chk.shape[0])} bodies).")
        except Exception as e:
            print(f"[POUR] Pellet physics view unavailable ({e}); using USD volume sensor (may read 0 headless).")
            pellets_rb = None

    # Volume-sensor box in world frame (mirrors PouringEnv._pellets_in_volume).
    _vbox_cx = float(args.glass_pos[0])
    _vbox_cy = float(args.glass_pos[1])
    _vbox_cz = float(args.glass_pos[2]) + float(volume_cfg.z_offset_above_base_m)
    _vbox_hx, _vbox_hy, _vbox_hz = (float(v) for v in volume_cfg.half_extents_xyz)
    _pellet_vol_m3 = (4.0 / 3.0) * math.pi * (float(args.pellet_radius_m) ** 3)
    _pellet_g = float(pellet_cfg.density_kg_m3) * _pellet_vol_m3 * 1000.0

    def _pellet_positions_phys():
        """(N,3) tensor of pellet world positions from physics, or None."""
        if pellets_rb is None:
            return None
        try:
            pos, _ = pellets_rb.get_world_poses()
            return pos
        except Exception:
            return None

    def _count_in_glass():
        """Pellets currently inside the glass volume box (physics if available, else USD sensor)."""
        pos = _pellet_positions_phys()
        if pos is None:
            return int(env.count_pellets_in_glass())
        inside = (
            (torch.abs(pos[:, 0] - _vbox_cx) <= _vbox_hx)
            & (torch.abs(pos[:, 1] - _vbox_cy) <= _vbox_hy)
            & (torch.abs(pos[:, 2] - _vbox_cz) <= _vbox_hz)
        )
        return int(inside.sum().item())

    def _ee_pose_w():
        p = robot.data.body_pose_w[0, ee_body_id]
        return p[0:3].clone(), p[3:7].clone()  # pos, quat (w,x,y,z)

    weld_err = {"printed": False}

    def _engage_weld():
        if not weld_enabled or pitcher_rb is None:
            return
        try:
            epos, equat = _ee_pose_w()
            ppos, pquat = pitcher_rb.get_world_poses()
            ppos = ppos[0].to(epos.device).float()
            pquat = pquat[0].to(epos.device).float()
            equat_inv = quat_conjugate(equat.unsqueeze(0))
            rel_pos = quat_apply(equat_inv, (ppos - epos).unsqueeze(0))[0]
            rel_quat = quat_mul(equat_inv, pquat.unsqueeze(0))[0]
            weld_state.update(active=True, rel_pos=rel_pos, rel_quat=rel_quat)
            # Keep the pitcher a *dynamic* body (so its walls keep colliding and
            # carry the pellets), but turn off gravity and drive it to follow the
            # end-effector by commanding its velocity each step. Velocity control --
            # rather than teleporting the pose -- lets the contact solver push the
            # pellets at the pitcher's true surface speed, so they are carried and
            # poured gently instead of being flung by penetration impulses.
            try:
                pitcher_rb.disable_gravities()
            except Exception as e:
                print(f"[POUR][WELD] Could not disable pitcher gravity ({e!r}).")
            print("[POUR][WELD] Engaged: pitcher velocity-pinned to the end-effector (gravity off).")
        except Exception as e:
            print(f"[POUR][WELD] Failed to engage weld ({e!r}); continuing with friction only.")

    def _apply_weld():
        if not weld_state["active"] or pitcher_rb is None:
            return
        try:
            epos, equat = _ee_pose_w()
            tgt_pos = epos + quat_apply(equat.unsqueeze(0), weld_state["rel_pos"].unsqueeze(0))[0]
            tgt_quat = quat_mul(equat.unsqueeze(0), weld_state["rel_quat"].unsqueeze(0))[0]
            tgt_quat = tgt_quat / (tgt_quat.norm() + 1e-12)

            cur_pos, cur_quat = pitcher_rb.get_world_poses()
            cur_pos = cur_pos[0].to(tgt_pos.device).float()
            cur_quat = cur_quat[0].to(tgt_pos.device).float()

            dt_w = float(sim.get_physics_dt())
            # Linear velocity that closes the position error this step.
            lin = (tgt_pos - cur_pos) / max(dt_w, 1e-5)
            # Angular velocity from the current->target rotation (shortest arc).
            dq = quat_mul(tgt_quat.unsqueeze(0), quat_conjugate(cur_quat.unsqueeze(0)))[0]
            dq = dq / (dq.norm() + 1e-12)
            if float(dq[0]) < 0.0:
                dq = -dq
            w_ = float(max(-1.0, min(1.0, float(dq[0]))))
            s_ = math.sqrt(max(0.0, 1.0 - w_ * w_))
            if s_ < 1e-6:
                ang = torch.zeros(3, device=tgt_pos.device)
            else:
                angle = 2.0 * math.acos(w_)
                ang = (dq[1:4] / s_) * (angle / max(dt_w, 1e-5))
            # Clamp to keep a stuck/contacting step from spiking the solver.
            lin = lin.clamp(-1.5, 1.5)
            ang = ang.clamp(-8.0, 8.0)
            vel = torch.cat([lin, ang]).unsqueeze(0)
            pitcher_rb.set_velocities(vel)
        except Exception as e:
            if not weld_err["printed"]:
                weld_err["printed"] = True
                print(f"[POUR][WELD] _apply_weld error (suppressed after first): {e!r}")

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

    def _read_prim_world_pos(prim_path: str):
        """World translation of a prim's USD xform (None if unreadable)."""
        try:
            import omni.usd  # type: ignore
            from pxr import Usd, UsdGeom  # type: ignore
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(str(prim_path))
            if not prim.IsValid():
                return None
            M = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            return (float(M[3][0]), float(M[3][1]), float(M[3][2]))
        except Exception:
            return None

    def _pellet_stats():
        """(centroid_xyz, min_z) of all pellets in world frame, or (None, None).

        Uses the physics view when available (the USD xforms are stale headless).
        """
        pos = _pellet_positions_phys()
        if pos is not None and pos.shape[0] > 0:
            c = pos.mean(dim=0)
            return (float(c[0]), float(c[1]), float(c[2])), float(pos[:, 2].min())
        xs = ys = zs = 0.0
        zmin = None
        n = 0
        for p in env.pellet_prim_paths:
            pp = _read_prim_world_pos(p)
            if pp is None:
                continue
            xs += pp[0]; ys += pp[1]; zs += pp[2]
            zmin = pp[2] if zmin is None else min(zmin, pp[2])
            n += 1
        if n == 0:
            return None, None
        return (xs / n, ys / n, zs / n), zmin

    grip_joint_ids_t, _ = robot.find_joints(str(args.gripper_joint_regex))
    if hasattr(grip_joint_ids_t, "view"):
        grip_joint_ids = [int(v) for v in grip_joint_ids_t.view(-1).tolist()]
    else:
        grip_joint_ids = [int(v) for v in list(grip_joint_ids_t)]

    def _ee_world_pos():
        p = robot.data.body_pose_w[0, ee_body_id, 0:3]
        return (float(p[0]), float(p[1]), float(p[2]))

    def _quat_axes_base(q):
        """Local EE axes (X,Y,Z) expressed in the base frame, as 3 tuples."""
        w, x, y, z = (float(v) for v in q.tolist())
        col_x = (1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y))
        col_y = (2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x))
        col_z = (2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y))
        return col_x, col_y, col_z

    def _pitcher_phys_pos():
        """Pitcher world position from the physics view (None if unavailable)."""
        if pitcher_rb is None:
            return None
        try:
            pos, _ = pitcher_rb.get_world_poses()
            p = pos[0]
            return (float(p[0]), float(p[1]), float(p[2]))
        except Exception:
            return None

    def _diag(tag: str) -> None:
        pp = _read_prim_world_pos(pitcher_path)
        pp_phys = _pitcher_phys_pos()
        centroid, zmin = _pellet_stats()
        ee_w = _ee_world_pos()
        grip = robot.data.joint_pos[0, grip_joint_ids]
        grip_mean = float(grip.mean()) if grip.numel() else float("nan")
        pp_s = "n/a" if pp is None else f"({pp[0]:+.3f},{pp[1]:+.3f},{pp[2]:+.3f})"
        ppp_s = "n/a" if pp_phys is None else f"({pp_phys[0]:+.3f},{pp_phys[1]:+.3f},{pp_phys[2]:+.3f})"
        c_s = "n/a" if centroid is None else f"({centroid[0]:+.3f},{centroid[1]:+.3f},{centroid[2]:+.3f})"
        zmin_s = "n/a" if zmin is None else f"{zmin:+.3f}"
        print(
            f"[POUR][DIAG] {tag:<12} ee_w=({ee_w[0]:+.3f},{ee_w[1]:+.3f},{ee_w[2]:+.3f}) "
            f"grip~{grip_mean:.2f} pitcher_usd={pp_s} pitcher_phys={ppp_s} "
            f"pellet_centroid_w={c_s} pellet_min_z={zmin_s}"
        )

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
        sim.step(render=render)
        robot.update(dt)
        _apply_weld()

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
            f"[POUR]   {label:<22} dist={dist * 1000:6.1f} mm "
            f"min_dur={min_dur:.2f}s t={seg.t_elapsed:.2f}s final_err={final_err * 1000:6.1f} mm [{status}]"
        )
        return converged, final_err

    def _hold_at(goal_pos_b, q_fixed_b, hold_s, dt):
        steps = int(max(1, round(float(hold_s) / dt)))
        for _ in range(steps):
            if not simulation_app.is_running():
                return
            _drive_ik_step(goal_pos_b, q_fixed_b, dt)

    pour_total = len(env.pellet_prim_paths)
    pour_best = {"count": 0}

    def _log_pour(tag: str, extra: str = "") -> int:
        count = _count_in_glass()
        grams = count * _pellet_g
        pour_best["count"] = max(pour_best["count"], int(count))
        print(f"[POUR]   {tag:<14} in_glass={count}/{pour_total} (~{grams:5.1f} g) {extra}")
        return count

    def _tilt_and_pour(goal_pos_b, q_start_b, q_end_b, tilt_s, hold_s, dt):
        """Roll the wrist from q_start to q_end while holding position, then hold; logging."""
        period = max(1e-3, float(args.pour_log_period_s))
        # Ramp (tilt).
        steps = int(max(1, round(float(tilt_s) / dt)))
        accum = 0.0
        for i in range(steps):
            if not simulation_app.is_running():
                return
            se = _quintic((i + 1) / steps)
            q_t = _slerp(q_start_b, q_end_b, se)
            _drive_ik_step(goal_pos_b, q_t, dt)
            accum += dt
            if accum >= period:
                accum = 0.0
                _log_pour("TILTING", extra=f"tilt={se * float(args.pour_angle_deg):5.1f} deg")
        # Hold at full tilt.
        steps = int(max(1, round(float(hold_s) / dt)))
        accum = 0.0
        for _ in range(steps):
            if not simulation_app.is_running():
                return
            _drive_ik_step(goal_pos_b, q_end_b, dt)
            accum += dt
            if accum >= period:
                accum = 0.0
                _log_pour("HOLD@POUR", extra=f"tilt={float(args.pour_angle_deg):5.1f} deg")

    # ------------------------------------------------------------------
    # Settle the scene: open the gripper and let the pellets fall.
    # ------------------------------------------------------------------
    dt = float(sim.get_physics_dt())
    try:
        gripper.command_open(robot)
    except Exception:
        pass
    for _ in range(int(args.settle_steps)):
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
        sim.step(render=render)
        robot.update(dt)

    ee_pos_b_home, ee_quat_b_home = _read_ee_pose_b()
    print(
        f"[POUR] Home EE pose: pos={[round(float(v), 4) for v in ee_pos_b_home.tolist()]} "
        f"quat_wxyz={[round(float(v), 4) for v in ee_quat_b_home.tolist()]}"
    )
    _log_pour("PRE_GRASP", extra="(pellets settled)")
    _diag("PRE_GRASP")

    # ------------------------------------------------------------------
    # Build the grasp / carry / pour orientations.
    #
    # Home grasp points the palm straight down (approach = base -Z). For a side
    # grasp we rotate that frame so the approach axis points horizontally INTO
    # the pitcher (base +X), then to pour we roll about that same approach axis.
    # ------------------------------------------------------------------
    approach_dir = (1.0, 0.0, 0.0)  # base +X: gripper sits on the robot side, palm faces the pitcher
    q_align = _shortest_arc_quat((0.0, 0.0, -1.0), approach_dir, device=sim.device)
    # Palm-forward but NOT yet rolled: the orientation used while reaching forward and
    # descending. The final "rotate to grasp" rolls from here to q_side.
    q_approach = _quat_mul(q_align, ee_quat_b_home)
    q_approach = q_approach / (q_approach.norm() + 1e-12)
    # Extra roll about the approach axis so the finger-closing axis is horizontal
    # (clamps across the upright cylinder's diameter rather than along its axis).
    q_grasp_roll = _axis_angle_quat(approach_dir, math.radians(float(args.grasp_roll_deg)), device=sim.device)
    q_side = _quat_mul(q_grasp_roll, q_approach)
    q_side = q_side / (q_side.norm() + 1e-12)
    pour_angle_rad = math.radians(float(args.pour_angle_deg)) * float(args.pour_sign)
    q_roll = _axis_angle_quat(approach_dir, pour_angle_rad, device=sim.device)
    q_pour = _quat_mul(q_roll, q_side)
    q_pour = q_pour / (q_pour.norm() + 1e-12)

    hx, hy, hz = _quat_axes_base(ee_quat_b_home)
    sx, sy, sz = _quat_axes_base(q_side)
    print(
        "[POUR][AXES] home EE local axes in base frame:\n"
        f"        X={tuple(round(v, 2) for v in hx)}  Y={tuple(round(v, 2) for v in hy)}  Z={tuple(round(v, 2) for v in hz)}\n"
        "[POUR][AXES] side-grasp EE local axes in base frame:\n"
        f"        X={tuple(round(v, 2) for v in sx)}  Y={tuple(round(v, 2) for v in sy)}  Z={tuple(round(v, 2) for v in sz)}\n"
        "        (whichever home axis ~= (0,0,-1) is the approach/finger direction; after the\n"
        "         side rotation that same axis should be ~= (+1,0,0))"
    )

    # Seat the cylinder in the palm: the wrist stands off from the pitcher center
    # axis by outer_radius + gap, so the palm is ~gap in front of the near wall and
    # the fingers wrap the full diameter (mirrors the cube grasp, where the palm
    # sits ~1 cm from the block's near face and the fingers wrap ~7 cm deep).
    pitcher_outer_r = float(args.pitcher_radius_m) + float(args.wall_thickness_m)
    if args.grasp_reach_m is None:
        grasp_reach = pitcher_outer_r + float(args.grasp_gap_m)
    else:
        grasp_reach = float(args.grasp_reach_m)
    approach_t = torch.tensor(approach_dir, dtype=torch.float32, device=sim.device)

    # Pitcher grasp point (world) -> base.
    px, py, pbz = (float(v) for v in args.pitcher_pos)
    grasp_z_w = pbz + float(args.wall_thickness_m) + float(args.grasp_frac) * float(args.pitcher_height_m)
    grasp_point_b = _world_to_base(torch.tensor([px, py, grasp_z_w], dtype=torch.float32, device=sim.device))
    grasp_wrist_b = grasp_point_b - approach_t * grasp_reach
    pregrasp_b = grasp_wrist_b - approach_t * float(args.pregrasp_back_m)
    # Forward waypoint: stand-off behind the pitcher but raised, so the gripper first
    # reaches forward at height, then descends, then advances into the grasp.
    forward_b = pregrasp_b.clone()
    forward_b[2] = float(pregrasp_b[2]) + float(args.approach_height_m)
    lift_b = grasp_wrist_b.clone()
    lift_b[2] = float(grasp_wrist_b[2]) + float(args.lift_m)

    # Pour point (above the glass) -> base. Offset on the +tilt side so the mouth
    # ends up over the glass center once tilted.
    gx, gy, gbz = (float(v) for v in args.glass_pos)
    glass_top_z_w = gbz + float(args.wall_thickness_m) + float(args.glass_height_m)
    pour_y = gy + float(args.pour_sign) * float(args.pour_y_offset_m)
    pour_point_w = torch.tensor(
        [gx, pour_y, glass_top_z_w + float(args.pour_height_m)], dtype=torch.float32, device=sim.device
    )
    pour_point_b = _world_to_base(pour_point_w)
    pour_wrist_b = pour_point_b - approach_t * grasp_reach
    pour_wrist_b[2] = float(pour_point_b[2])  # keep height at the requested pour height

    print(
        "[POUR] Plan:\n"
        f"        pitcher grasp point (base) = ({float(grasp_point_b[0]):+.3f}, {float(grasp_point_b[1]):+.3f}, {float(grasp_point_b[2]):+.3f})\n"
        f"        grasp wrist  (base)        = ({float(grasp_wrist_b[0]):+.3f}, {float(grasp_wrist_b[1]):+.3f}, {float(grasp_wrist_b[2]):+.3f})\n"
        f"        pour  wrist  (base)        = ({float(pour_wrist_b[0]):+.3f}, {float(pour_wrist_b[1]):+.3f}, {float(pour_wrist_b[2]):+.3f})\n"
        f"        pour angle = {float(args.pour_angle_deg):.0f} deg (sign {float(args.pour_sign):+.0f}); grasp_reach = {grasp_reach:.3f} m"
    )
    print("[POUR] " + "=" * 74)

    # ------------------------------------------------------------------
    # Execute: approach -> grasp -> lift -> carry -> pour -> (return) -> idle.
    # ------------------------------------------------------------------
    if bool(args.rotate_during_grasp):
        # Staged approach: reach FORWARD (palm-forward) above the stand-off, go
        # DOWN to grasp height, then ROTATE the wrist to align with the pitcher
        # while advancing into the grasp.
        _run_segment("REACH_FORWARD", forward_b, q_approach, dt, q_start_b=ee_quat_b_home)
        _run_segment("DESCEND", pregrasp_b, q_approach, dt, q_start_b=q_approach)
        _run_segment("ROTATE_TO_GRASP", grasp_wrist_b, q_side, dt, q_start_b=q_approach)
    else:
        # Original approach: fully reorient up front, then translate straight in.
        _run_segment("APPROACH_SIDE", pregrasp_b, q_side, dt, q_start_b=ee_quat_b_home)
        _run_segment("ADVANCE_TO_GRASP", grasp_wrist_b, q_side, dt, q_start_b=q_side)
    _hold_at(grasp_wrist_b, q_side, float(args.pre_close_settle_s), dt)
    try:
        gripper.command_close(robot)
    except Exception:
        pass
    _hold_at(grasp_wrist_b, q_side, float(args.gripper_close_s), dt)
    print(f"[POUR]   GRIPPER_CLOSE          held {args.gripper_close_s:.2f}s")
    _engage_weld()
    _diag("AT_GRASP")

    _run_segment("LIFT_PITCHER", lift_b, q_side, dt, q_start_b=q_side)
    _log_pour("LIFTED")
    _diag("LIFTED")

    _run_segment("CARRY_TO_GLASS", pour_wrist_b, q_side, dt, q_start_b=q_side)
    _hold_at(pour_wrist_b, q_side, float(args.pre_pour_settle_s), dt)
    _log_pour("OVER_GLASS")
    _diag("OVER_GLASS")

    print("[POUR]   POURING (rolling the wrist to tilt the pitcher) ...")
    _tilt_and_pour(pour_wrist_b, q_side, q_pour, float(args.pour_tilt_s), float(args.pour_hold_s), dt)
    _hold_at(pour_wrist_b, q_pour, float(args.post_pour_settle_s), dt)
    _log_pour("POUR_DONE")
    _diag("POUR_DONE")

    if bool(args.return_upright):
        _run_segment("RETURN_UPRIGHT", pour_wrist_b, q_side, dt, q_start_b=q_pour)
        _log_pour("UPRIGHT")

    # ------------------------------------------------------------------
    # Summary.
    # ------------------------------------------------------------------
    final = _count_in_glass()
    best = int(pour_best["count"])
    frac = (best / pour_total) if pour_total else 0.0
    success = frac >= float(args.success_frac)
    print("[POUR] " + "=" * 74)
    print(
        f"[POUR] SUMMARY: poured {best}/{pour_total} pellets at peak "
        f"({100.0 * frac:.0f}%), {final} currently in glass. "
        f"SUCCESS={success} (threshold {100.0 * float(args.success_frac):.0f}%)."
    )

    if simulation_app.is_running() and not headless:
        ee_idle, q_idle = _read_ee_pose_b()
        _hold_at(ee_idle, q_idle, 1.0, dt)

    print("[POUR] Done.")
    simulation_app.close()
    return 0 if success else 1


if __name__ == "__main__":
    import os

    _code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Isaac Sim's Kit threads can keep the headless process alive (and pinned to
    # GPU memory) after simulation_app.close(); force a hard exit so repeated runs
    # don't pile up zombie processes that exhaust the GPU.
    os._exit(int(_code) if _code is not None else 0)
