"""Isaac Lab driver for the supervisory-fetch task.

Built on the stack this repository has already debugged rather than a fresh one: the table and
Kinova mounting pose come from the reach-to-grasp VLA scene, motion goes through
:class:`~controllers.cartesian_velocity.CartesianVelocityJogController` driven by
:class:`~controllers.input.waypoint_follower.WaypointFollowerInput`, and the gripper is the
same :class:`GripperController` those use. Hand-rolling a differential-IK loop here was tried
and abandoned: it duplicated four things (singularity guarding, workspace clamping, orientation
hold, gripper drive-gain tuning) that the existing controller already gets right, and got two of
them wrong.

Four constraints from this repository's history are respected, each marked at the line that
respects it, because every one previously cost a session:

* object poses are read from **PhysX views**, never from USD -- under the GPU/Fabric pipeline a
  USD read returns the spawn pose, not the live one;
* those views are rebuilt after **every** ``sim.reset()`` -- they go stale across one; this
  driver resets once at setup, so they are built once and stay valid;
* ``set_transforms``/``set_velocities`` are always called **with an ``indices`` argument** --
  omitting it silently no-ops and freezes the layout for a whole session;
* the grasp waypoints carry the 2026-06-17 geometry (end effector at the object's own height,
  ``-4`` cm grasp depth, close on contact) that took the scripted expert from 7% to ~90%.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import (
    DEFAULT_SUP_FIGURE_CAMERA,
    DEFAULT_SUP_SCENE,
    DEFAULT_SUP_TOPDOWN_CAMERA,
    SupSceneConfig,
    layout_for_margin,
)
from .experts import GRIPPER_CLOSED, GRIPPER_OPEN, ExpertConfig, Waypoint, waypoints_for

_COLOR_RGB = {
    "red": (0.85, 0.16, 0.14),
    "blue": (0.14, 0.32, 0.82),
    "green": (0.18, 0.60, 0.26),
    "yellow": (0.92, 0.78, 0.16),
}

#: Names of the object prims, in spawn order. ``target`` and ``blocker`` are the two the study
#: cares about; the rest are visual clutter that never enters the corridor between them.
OBJECT_NAMES = ("target", "blocker", "distractor_0", "distractor_1")


class SupervisoryFetchScene:
    """Owns the simulator, the scene, and one-strategy rollouts. Owns no study logic."""

    def __init__(
        self,
        *,
        cfg: Any = None,
        scene_cfg: Optional[SupSceneConfig] = None,
        expert_cfg: Optional[ExpertConfig] = None,
        seed: int = 0,
        capture_dir: Optional[Path] = None,
    ) -> None:
        self.cfg = cfg
        self.scene_cfg = scene_cfg or DEFAULT_SUP_SCENE
        self.expert_cfg = expert_cfg or ExpertConfig()
        self.seed = int(seed)
        self.capture_dir = Path(capture_dir) if capture_dir else None
        self.sim = None
        self.robot = None
        self.controller = None
        self.follower = None
        self.cameras: Dict[str, Any] = {}
        self._object_paths: List[str] = []
        self._views: Dict[str, Any] = {}
        self._rng = np.random.default_rng(self.seed)
        self._layout: Optional[Dict[str, Any]] = None
        self._grip = GRIPPER_OPEN

    # ------------------------------------------------------------------ setup
    def open(self) -> None:
        if self.sim is not None:
            return
        import isaaclab.sim as sim_utils
        import torch

        from controllers import CartesianVelocityJogConfig, CartesianVelocityJogController
        from controllers.input.waypoint_follower import WaypointFollowerInput
        from environments.reach_to_grasp_VLA.config import DEFAULT_SCENE
        from environments.reach_to_grasp_VLA.utils import design_scene

        sim_cfg = sim_utils.SimulationCfg(device=str(getattr(self.cfg, "device", "cuda:0")))
        self.sim = sim_utils.SimulationContext(sim_cfg)
        entities, _origins = design_scene(DEFAULT_SCENE)
        self.robot = entities["kinova_j2n6s300"]
        self._spawn_objects()
        if getattr(self.cfg, "capture_frames", False):
            self._create_camera_prims()
            self._build_camera_sensors()   # BEFORE sim.reset(); see the docstring there

        self.sim.reset()
        self._rebuild_views()  # PhysX views are valid only after reset, and only until the next one
        self._reset_camera_sensors()      # AFTER sim.reset()

        dt = float(self.sim.get_physics_dt())
        ctrl_cfg = CartesianVelocityJogConfig(
            ee_link_name="j2n6s300_end_effector",
            device=str(self.sim.device),
            use_relative_mode=True,
            linear_speed_mps=float(getattr(self.cfg, "linear_speed_mps", 0.5)),
            jog_velocity_gain=float(getattr(self.cfg, "jog_velocity_gain", 1.0)),
            ik_method=str(getattr(self.cfg, "ik_method", "pinv")),
            # **The orientation hold must be off here**, and this is the single most important
            # line in the file. With it on, a pure descent command from this scene's home
            # configuration produces a ~0.6 mrad joint step and the end effector does not move
            # at all -- measured: -0.027 m of "descent" over 1500 steps, i.e. it drifts upward
            # while retracting toward the base. Releasing the wrist orientation turns the same
            # command into +0.23 m of real descent. The collector this driver borrows from can
            # afford the hold because it aligns the wrist to a grasp quaternion first; this
            # driver has no grasp-pose estimator, so holding an orientation it never chose just
            # over-constrains the IK.
            hold_orientation=bool(getattr(self.cfg, "hold_orientation", False)),
            workspace_min=tuple(getattr(self.cfg, "workspace_min", (0.05, -0.60, -0.30))),
            workspace_max=tuple(getattr(self.cfg, "workspace_max", (0.80, 0.60, 1.40))),
        )
        self.controller = CartesianVelocityJogController(ctrl_cfg, num_envs=1, device=str(self.sim.device))
        self.controller.set_mode("translate")
        self.controller.reset(self.robot)
        self.follower = WaypointFollowerInput(
            step_pos_m=float(ctrl_cfg.linear_speed_mps) * dt,
            tol_m=0.012,
            max_steps_per_waypoint=int(getattr(self.cfg, "max_steps_per_waypoint", 1800)),
            stagnation_steps=10**9,
            device=str(self.sim.device),
        )
        self.controller.set_input_provider(self.follower)

    def close(self) -> None:
        if self.sim is not None:
            try:
                self.sim.stop()
            except Exception:
                pass
        self.sim = None

    # ------------------------------------------------------------- scene build
    def _spawn_objects(self) -> None:
        import isaaclab.sim as sim_utils

        root = self.scene_cfg.objects_root
        s = float(self.scene_cfg.cube_size_m)
        colors = [self.scene_cfg.target_color, self.scene_cfg.blocker_color, "green", "yellow"]
        self._object_paths = []
        for i, (name, color) in enumerate(zip(OBJECT_NAMES, colors)):
            path = f"{root}/{name}"
            cfg = sim_utils.CuboidCfg(
                size=(s, s, s),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False, max_depenetration_velocity=1.0, solver_position_iteration_count=16
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.08),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.004, rest_offset=0.0),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.1, dynamic_friction=1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=_COLOR_RGB.get(color, (0.6, 0.6, 0.6))),
            )
            # Spawn well apart; reset_to() teleports them into position each episode.
            cfg.func(path, cfg, translation=(0.35 + 0.10 * i, 0.30, self.scene_cfg.table_height_m + s))  # world
            self._object_paths.append(path)

    def _create_camera_prims(self) -> None:
        """Create the USD camera prims. The sensors attach to these afterwards."""
        from environments.utils.camera import create_topdown_camera

        create_topdown_camera(DEFAULT_SUP_TOPDOWN_CAMERA)
        self._create_figure_camera_prim()

    def _create_figure_camera_prim(self) -> None:
        """A three-quarter camera aimed at the workspace, for figures only.

        The top-down contract camera is what a policy sees and must not be moved; it is also a
        poor way to show a person what the task is, because a clearance gap viewed from directly
        overhead carries no sense of an arm reaching past an obstacle. This one is never used
        for training or evaluation.
        """
        import importlib
        import math

        import numpy as np
        from pxr import Gf, UsdGeom

        prim_utils = importlib.import_module("isaacsim.core.utils.prims")
        import omni.usd

        cfg = DEFAULT_SUP_FIGURE_CAMERA
        prim_utils.create_prim(cfg.prim_path, "Camera")
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(cfg.prim_path)
        xform = UsdGeom.Xformable(prim)

        # USD cameras look down their local -Z with +Y up. Aim by yaw about world +Z to face the
        # target in plan, then tilt down by the elevation angle.
        # Standard look-at basis. A USD camera looks down its local -Z with +Y up, so build the
        # camera axes explicitly and hand USD the resulting rotation. Composing yaw and pitch by
        # hand was tried and aimed the camera at the sky: every frame came back a uniform 238.
        eye = np.asarray(cfg.position, dtype=float)
        tgt = np.asarray(cfg.target, dtype=float)
        fwd = tgt - eye
        fwd = fwd / max(float(np.linalg.norm(fwd)), 1e-9)
        z_cam = -fwd
        x_cam = np.cross(np.array([0.0, 0.0, 1.0]), z_cam)
        n = float(np.linalg.norm(x_cam))
        x_cam = np.array([1.0, 0.0, 0.0]) if n < 1e-6 else x_cam / n
        y_cam = np.cross(z_cam, x_cam)
        R = np.stack([x_cam, y_cam, z_cam], axis=1)

        # R -> XYZ Euler in degrees, the convention AddRotateXYZOp expects.
        sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        if sy > 1e-6:
            rx, ry, rz = math.atan2(R[2, 1], R[2, 2]), math.atan2(-R[2, 0], sy), math.atan2(R[1, 0], R[0, 0])
        else:
            rx, ry, rz = math.atan2(-R[1, 2], R[1, 1]), math.atan2(-R[2, 0], sy), 0.0

        xform.ClearXformOpOrder()
        xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*cfg.position))
        xform.AddRotateXYZOp(UsdGeom.XformOp.PrecisionFloat).Set(
            Gf.Vec3f(*[float(math.degrees(v)) for v in (rx, ry, rz)])
        )
        cam = UsdGeom.Camera(prim)
        cam.GetFocalLengthAttr().Set(36.0 / (2.0 * math.tan(math.radians(cfg.fov) / 2.0)))

    def _build_camera_sensors(self) -> None:
        """Attach ``Camera`` sensors to the prims. **Before** ``sim.reset()``.

        Two details, both of which fail silently rather than raising. The sensor must carry an
        explicit ``offset`` alongside ``spawn=None`` when attaching to a prim that already
        exists -- without it the render product never binds and every frame comes back black.
        And it must be constructed *before* the simulator resets, with ``reset()`` called
        *after*; built the other way round the constructor half-fails and the only symptom is an
        ``AttributeError`` raised later, from the destructor.
        """
        from isaaclab.sensors import Camera, CameraCfg

        self.cameras = {}
        for name, cfg in (("topdown", DEFAULT_SUP_TOPDOWN_CAMERA), ("figure", DEFAULT_SUP_FIGURE_CAMERA)):
            try:
                cam_cfg = CameraCfg(
                    prim_path=cfg.prim_path,
                    offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
                    spawn=None,                       # attach to the existing prim
                    data_types=["rgb"],
                    width=int(cfg.resolution[0]),
                    height=int(cfg.resolution[1]),
                )
                self.cameras[name] = Camera(cfg=cam_cfg)
                print(f"[sup_scene] camera '{name}' {cfg.resolution[0]}x{cfg.resolution[1]} attached")
            except Exception as exc:
                print(f"[sup_scene] camera '{name}' failed: {exc}")

    def _reset_camera_sensors(self) -> None:
        for name, cam in list(getattr(self, "cameras", {}).items()):
            try:
                cam.reset()
            except Exception as exc:
                print(f"[sup_scene] camera '{name}' reset failed: {exc}")
                self.cameras.pop(name, None)

    # -------------------------------------------------------------- PhysX views
    def _rebuild_views(self) -> None:
        """Rebuild the PhysX rigid-body views. **Every episode.**

        They go stale across ``sim.reset()``; a cached view returns the pose the object had at
        the previous reset, which in this repository once made every second grasp aim 20 cm off.
        This driver resets once, at setup, so this runs once -- but it must run *after* that
        reset, never before.
        """
        from isaacsim.core.simulation_manager import SimulationManager

        backend = SimulationManager.get_physics_sim_view()
        self._views = {}
        for path in self._object_paths:
            try:
                self._views[path] = backend.create_rigid_body_view(path)
            except Exception as exc:
                print(f"[sup_scene] no view for {path}: {exc}")
                self._views[path] = None

    def _read_pose(self, name: str) -> Optional[np.ndarray]:
        """Live pose ``[x y z qx qy qz qw]`` in the **base frame**, from PhysX. Never from USD."""
        view = self._views.get(f"{self.scene_cfg.objects_root}/{name}")
        if view is None:
            return None
        t = view.get_transforms()
        arr = t.cpu().numpy() if hasattr(t, "cpu") else np.asarray(t)
        pose = np.asarray(arr).reshape(-1)[:7].astype(float).copy()
        pose[2] -= float(self.scene_cfg.table_height_m)   # world -> base frame
        return pose

    def _set_pose(self, name: str, xyz: Sequence[float], yaw: float = 0.0) -> None:
        """Teleport one object. ``xyz`` is **base frame**; PhysX wants world.

        The single frame conversion in this file. Layouts, waypoints, and the controller all
        speak the robot base frame, in which the table top is z = 0; PhysX transforms are world,
        in which it is ``table_height_m``. Doing the conversion anywhere else is how every
        waypoint ended up 0.8 m above the arm's reach.

        ``indices`` is mandatory: without it the call silently no-ops.
        """
        import torch

        view = self._views.get(f"{self.scene_cfg.objects_root}/{name}")
        if view is None:
            return
        qw, qx, qy, qz = _yaw_quat(yaw)
        z_world = float(xyz[2]) + float(self.scene_cfg.table_height_m)
        tf = torch.tensor([[float(xyz[0]), float(xyz[1]), z_world, qx, qy, qz, qw]],
                          dtype=torch.float32, device=self.sim.device)
        idx = torch.tensor([0], dtype=torch.int32, device=self.sim.device)
        view.set_transforms(tf, indices=idx)                                    # indices: mandatory
        view.set_velocities(torch.zeros((1, 6), dtype=torch.float32, device=self.sim.device), indices=idx)

    # ------------------------------------------------------------------ episode
    def reset_to(self, scene) -> Dict[str, Any]:
        """Realise this scene's clearance gap, and leave the arm ready to execute.

        **Order matters, and getting it wrong is silent.** The wrist alignment sweeps the
        gripper through a large arc, and the blocker sits between the arm and the target -- so
        aligning *after* placing the objects knocks the blocker out of position on every
        episode. Measured before this was reordered: the target landed at its commanded pose to
        the millimetre every time while the blocker was displaced by 2.6 to 23 cm, which means
        every rollout would have been executed against a clearance gap other than the one the
        trial thought it was measuring. The gap is the study's independent variable; a scene
        that does not realise it produces a success curve fitted against the wrong x-axis.

        So: home the arm, align the wrist on an empty table, *then* place the objects.
        """
        import random as _random
        import torch

        # ``sim.reset()`` every episode, then rebuild the views. Dropping it was tried: writing
        # joint state and teleporting objects into a live simulation injects enormous impulses,
        # and the scene explodes (objects logged a kilometre from the table). The reset is what
        # makes a teleport safe.
        self.sim.reset()
        self._rebuild_views()                                                   # stale across reset

        # Home the arm to the scene's **configured** default joint pose, not to whatever pose
        # happened to be current after the reset. Capturing "home" from a live read picks up
        # however far the arm had sagged, and the controller's orientation hold is then anchored
        # to a sagging configuration. Settle first, reset the controller second.
        self.robot.write_joint_state_to_sim(self.robot.data.default_joint_pos, self.robot.data.default_joint_vel)
        self.robot.write_data_to_sim()
        for _ in range(int(getattr(self.cfg, "home_settle_steps", 20))):
            self.sim.step(render=False)
            self.robot.update(float(self.sim.get_physics_dt()))
        self.controller.reset(self.robot)
        self.follower.reset()
        self._grip = GRIPPER_OPEN
        self._last_goal = None
        self._episode = int(getattr(self, "_episode", 0)) + 1

        # Align the wrist while the table is still clear.
        self._orient = self.orient_tool_down()

        # **Pre-roll to the canonical start pose, still on an empty table.** The configured home
        # *joint* pose puts the end effector at roughly (0.10, 0.07, 0.44): folded in against the
        # base and 44 cm up. Every rollout's first commanded move was therefore a long diagonal
        # across the table from a near-singular configuration, and it showed: the direct
        # strategy's opening alignment to (0.55, -0.10) diverged to y = -0.64 and timed out at
        # 4,000 steps, while the clear-first strategy -- whose first target is 10 cm closer --
        # completed the same phase in 409. That is the start pose acting as a nuisance variable
        # on the very comparison the study is built around.
        #
        # Starting instead from the reach-to-grasp collection contract's pose puts the arm over
        # the table with the tool already down, so the first move of every strategy is short.
        start = getattr(self.cfg, "start_ee_pos_b", None)
        self._preroll_err_m = None
        if start is not None:
            steps = int(getattr(self.cfg, "preroll_steps", 900))
            keep = int(getattr(self.cfg, "max_steps_per_waypoint", 4000))
            self.cfg.max_steps_per_waypoint = steps
            self._follow(Waypoint(tuple(float(v) for v in start), GRIPPER_OPEN, "preroll",
                                  float(getattr(self.cfg, "preroll_tol_m", 0.010))))
            self.cfg.max_steps_per_waypoint = keep
            self._last_goal = None          # the pre-roll is not part of the strategy
            self._preroll_err_m = float(np.linalg.norm(self._ee() - np.asarray(start, dtype=float)))

        # Now place the objects.
        layout = layout_for_margin(
            float(scene.margin_m), cfg=self.scene_cfg, n_distractors=int(getattr(scene, "clutter", 2)),
            rng=_random.Random(1000 + int(scene.scene_id)),
        )
        j = float(self.scene_cfg.dr_position_jitter_m)
        places: List[Tuple[str, Tuple[float, float], float]] = [
            ("target", layout["target"]["xy"], layout["target"]["z"]),
            ("blocker", layout["blocker"]["xy"], layout["blocker"]["z"]),
        ]
        for i, d in enumerate(layout["distractors"][: len(OBJECT_NAMES) - 2]):
            places.append((f"distractor_{i}", d["xy"], d["z"]))
        used = {n for n, _, _ in places}
        for k, n in enumerate(OBJECT_NAMES):
            if n not in used:
                places.append((n, (0.15, 0.45 + 0.08 * k), 0.03))

        for name, (x, y), z in places:
            # Jitter never touches the corridor between target and blocker: the margin is the
            # independent variable and randomising it would randomise the estimand's x-axis.
            fixed = name in ("target", "blocker")
            dx, dy = (0.0, 0.0) if fixed else (
                float(self._rng.normal(0, j)), float(self._rng.normal(0, j))
            )
            # Yaw is exempted for the same reason as position, and the reason is easy to miss.
            # The hand's yaw is whatever the wrist alignment produced; a randomly yawed cube
            # therefore meets the fingers at a random relative angle, and a three-finger hand
            # closing on a cube corner levers the arm rather than gripping it. That is a nuisance
            # variable on grasp success, which is the quantity the sweep exists to measure.
            yaw = 0.0 if fixed else float(self._rng.uniform(-0.3, 0.3))
            self._set_pose(name, (x + dx, y + dy, z), yaw=yaw)

        for _ in range(int(getattr(self.cfg, "settle_steps", 30))):
            self.sim.step(render=False)
            self.robot.update(float(self.sim.get_physics_dt()))

        self._layout = layout
        self._placement_error_m = self.placement_error()
        return layout

    def placement_error(self) -> Optional[float]:
        """How far the realised clearance gap is from the commanded one, in metres.

        Recorded on every episode and carried in the rollout record. A scene that quietly fails
        to realise its own independent variable is the single most damaging thing that can go
        wrong here, and it is invisible in a success rate.
        """
        if self._layout is None:
            return None
        t, b = self._read_pose("target"), self._read_pose("blocker")
        if t is None or b is None:
            return None
        realised = float(np.linalg.norm(t[:2] - b[:2])) - float(self.scene_cfg.cube_size_m)
        return float(realised - float(self._layout["margin_m"]))

    def _rise_waypoints(self, first: Waypoint) -> List[Waypoint]:
        """Prepend a vertical rise before the first commanded move.

        Handed a far-away goal from a freshly-reset pose, the differential-IK loop drives the
        end effector along a diagonal that dips toward the table -- in one trace it swung 38 cm
        off in y, dropped to 8 cm above the surface, and knocked the blocker 21 cm out of place
        before recovering. That is not a controller defect so much as the wrong command: the
        proven scripted planner in this repository rises to a safe height *first*, then aligns
        in xy, then descends. Doing the same here removes the excursion, and it matters for
        correctness rather than tidiness -- a rollout that displaces the blocker on its way to
        the target has changed the very geometry the trial is measuring.
        """
        import torch
        from utilities import get_ee_pos_base_frame

        ee = get_ee_pos_base_frame(self.robot, self.controller.config.ee_link_name)
        ee = np.asarray((ee.detach().cpu() if hasattr(ee, "detach") else ee)).reshape(-1)[:3]
        # Height at which the arm crosses the table. ``rise_height_m`` overrides the default of
        # "wherever the arm already is, but no lower than transit height"; see the note on the
        # config field for why the default is not always safe.
        override = getattr(self.cfg, "rise_height_m", None)
        safe_z = (float(override) if override is not None
                  else max(float(ee[2]), float(self.expert_cfg.transit_height_m)))
        return [
            Waypoint((float(ee[0]), float(ee[1]), safe_z), first.gripper, "rise", 0.035),
            Waypoint((float(first.xyz[0]), float(first.xyz[1]), safe_z), first.gripper, "align_xy", 0.035),
        ]

    def _perceived(self, layout: Dict[str, Any], scene) -> Dict[str, Any]:
        """A copy of the layout with the object positions the expert *believes*.

        Seeded from the scene id and the episode counter so a rollout is reproducible, and
        applied to the target and the blocker only -- the distractors are never manipulated.
        """
        sd = float(getattr(self.expert_cfg, "pose_noise_m", 0.0) or 0.0)
        if sd <= 0.0:
            return layout
        rng = np.random.default_rng(hash((int(getattr(scene, "scene_id", 0)), self._episode)) % (2 ** 32))
        out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in layout.items()}
        for name in ("target", "blocker"):
            x, y = out[name]["xy"]
            out[name] = dict(out[name])
            out[name]["xy"] = (float(x + rng.normal(0.0, sd)), float(y + rng.normal(0.0, sd)))
        self._pose_noise = {n: [round(float(a - b), 4) for a, b in zip(out[n]["xy"], layout[n]["xy"])]
                            for n in ("target", "blocker")}
        return out

    def run_strategy(self, scene, strategy: str, *, policy: Optional[Any] = None) -> Dict[str, Any]:
        """Execute one strategy and report success, duration, and why it failed if it did."""
        layout = self._layout or self.reset_to(scene)
        start = self._read_pose("target")
        z0 = float(start[2]) if start is not None else float(layout["target"]["z"])
        b0 = self._read_pose("blocker")

        # The expert plans against *perceived* poses, not the simulator's. See
        # ``ExpertConfig.pose_noise_m``: with oracle poses every success curve is a step, and a
        # step leaves the transition width -- hence the scene coordinate the study is defined
        # over -- degenerate. The objects themselves are not moved; only the belief about them.
        believed = self._perceived(layout, scene)
        wps = waypoints_for(strategy, believed, cfg=self.expert_cfg, scene=self.scene_cfg)
        wps = self._rise_waypoints(wps[0]) + wps
        t0 = time.time()
        steps = 0
        reached = 0
        fail: Optional[str] = None
        # Per-waypoint trace. Kept in the rollout record rather than behind a debug flag: when a
        # sweep produces a surprising success curve, the first question is always *which phase*
        # is failing, and reconstructing that from an aggregate is guesswork.
        trace: List[Dict[str, Any]] = []
        for wp in wps:
            ok, n = self._follow(wp)
            steps += n
            here = self._ee()
            trace.append({
                "phase": wp.phase,
                "ok": bool(ok),
                "steps": int(n),
                "dist_m": float(np.linalg.norm(here - np.asarray(wp.xyz, dtype=float))),
                "ee": [round(float(v), 4) for v in here],
                "goal": [round(float(v), 4) for v in wp.xyz],
                "gripper": int(wp.gripper),
            })
            if not ok:
                fail = f"waypoint_timeout:{wp.phase}"
                break
            reached += 1

        # Let the grasp settle before scoring: reading the lift height on the step the last
        # waypoint completed catches the object mid-slip and scores it as a success.
        for _ in range(int(getattr(self.cfg, "settle_steps", 30))):
            self.sim.step(render=False)
            self.robot.update(float(self.sim.get_physics_dt()))
            steps += 1

        end = self._read_pose("target")
        dz = float(end[2] - z0) if end is not None else 0.0
        success = bool(dz >= float(getattr(self.cfg, "lift_threshold_m", 0.06)) and fail is None)
        b1 = self._read_pose("blocker")
        blocker_moved = float(np.linalg.norm(b1[:2] - b0[:2])) if (b0 is not None and b1 is not None) else 0.0
        dt = float(self.sim.get_physics_dt())
        return {
            "success": success,
            "sim_time_s": steps * dt,
            "wall_s": time.time() - t0,
            "dz_m": dz,
            "waypoints_reached": reached,
            "waypoints_total": len(wps),
            "blocker_displacement_m": blocker_moved,
            "failure": fail,
            "margin_m": float(scene.margin_m),
            "physics_steps": steps,
            "trace": trace,
            "orient": getattr(self, "_orient", None),
            "placement_error_m": getattr(self, "_placement_error_m", None),
            "placement_error_end_m": self.placement_error(),
            "preroll_error_m": getattr(self, "_preroll_err_m", None),
            "pose_noise_m": getattr(self, "_pose_noise", None),
        }

    # ------------------------------------------------------- wrist orientation
    def tool_axis_b(self) -> np.ndarray:
        """Unit approach axis of the end effector, in the base frame.

        The JACO end-effector frame's ``+z`` is the tool/approach direction, so a top-down
        grasp needs this to point at ``(0, 0, -1)``.
        """
        from isaaclab.utils.math import subtract_frame_transforms

        root = self.robot.data.root_pose_w
        eew = self.robot.data.body_pose_w[:, self._ee_body_id()]
        _pos, quat = subtract_frame_transforms(root[:, 0:3], root[:, 3:7], eew[:, 0:3], eew[:, 3:7])
        w, x, y, z = [float(v) for v in quat[0].detach().cpu().numpy()]
        # Third column of the rotation matrix -- the body +z axis expressed in the base frame.
        return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])

    def _ee_quat_b(self):
        """End-effector orientation in the base frame, as ``(w, x, y, z)``."""
        from isaaclab.utils.math import subtract_frame_transforms

        root = self.robot.data.root_pose_w
        eew = self.robot.data.body_pose_w[:, self._ee_body_id()]
        _pos, quat = subtract_frame_transforms(root[:, 0:3], root[:, 3:7], eew[:, 0:3], eew[:, 3:7])
        return quat

    def _base_rotvec_to_tool(self, rotvec_b: np.ndarray) -> np.ndarray:
        """Express a base-frame rotation vector in the tool frame.

        **The controller's rotate mode takes its command in the tool frame**: it applies
        ``drot = quat_apply(ee_quat_b, drot)`` before using it, i.e. it treats whatever it is
        handed as tool-frame and rotates it into the base frame itself. The alignment below
        derives its axis geometrically in the *base* frame -- it is the axis that carries the
        tool's approach vector onto ``(0, 0, -1)`` -- so handing that axis over raw applies the
        end-effector rotation twice and turns the wrist about an axis that has nothing to do
        with the one intended.

        Measured symptom: the alignment burned 768 steps over its eight passes and left
        downwardness at ``-0.997``, i.e. very slightly *worse* than the ``-0.989`` it started
        from, with the gripper still pointing at the ceiling. Every descent then stalled with
        the arm 10 cm above the table, which reads exactly like a controller-tuning problem and
        is not one.
        """
        import torch
        from isaaclab.utils.math import quat_apply, quat_conjugate

        q = self._ee_quat_b()
        v = torch.tensor([[float(rotvec_b[0]), float(rotvec_b[1]), float(rotvec_b[2])]],
                         dtype=q.dtype, device=q.device)
        out = quat_apply(quat_conjugate(q), v)
        return np.asarray(out[0].detach().cpu().numpy(), dtype=float)

    def _ee_body_id(self) -> int:
        if getattr(self, "_ee_bid", None) is None:
            self._ee_bid = int(self.robot.find_bodies([self.controller.config.ee_link_name])[0][0])
        return int(self._ee_bid)

    def orient_tool_down(self, *, tol: float = 0.85, max_rot_steps: int = 900) -> Dict[str, Any]:
        """Rotate the wrist until the approach axis points at the table.

        **Why this exists.** At this scene's configured home joint pose the gripper points
        *upward*: the measured downwardness of the tool axis is ``-0.81``, i.e. it is aimed
        almost directly away from the table, and it stays negative through transit. An arm in
        that configuration cannot descend to a grasp -- not because the controller is mistuned
        but because the wrist is inverted, and reaching the object would require flipping it.
        That single measurement explains every descent failure this driver had, and no amount of
        solver or gain tuning addresses it.

        The collector this driver borrows from never hits this because it aligns the wrist to a
        grasp pose estimated from the object's bounding box before descending. This is the
        minimal equivalent: for a cube on a table the grasp orientation is known analytically --
        tool axis down -- so the rotation is computed in closed form and applied in the
        controller's rotate mode, after which the orientation hold can safely be re-enabled.
        """
        import torch

        dt = float(self.sim.get_physics_dt())
        # **Disarm the hold first.** It is re-armed at the end of a successful alignment and it
        # lives on the controller config, which outlives the episode -- so without this the
        # second and every subsequent alignment runs with the wrist pinned and fails, while the
        # first one succeeds. The symptom is an alignment that works exactly once per process.
        self.controller.config.hold_orientation = False
        self.controller.reset(self.robot)
        down = np.array([0.0, 0.0, -1.0])
        before = float(-self.tool_axis_b()[2])
        steps = 0
        # Each pass closes part of the remaining angle -- the IK realises a fraction of what it
        # is commanded near the workspace edge -- so the loop runs to convergence rather than a
        # fixed count, and gives up only when a pass stops buying anything. Eight passes were
        # not enough once the axis was computed in the right frame: it converged to 0.83 and
        # stalled one pass short of the 0.85 gate.
        best = float(np.clip(np.dot(self.tool_axis_b(), down), -1.0, 1.0))
        stalled = 0
        for _ in range(int(getattr(self.cfg, "orient_max_passes", 24))):
            tool = self.tool_axis_b()
            dotp = float(np.clip(np.dot(tool, down), -1.0, 1.0))
            if dotp >= tol:
                break
            if dotp <= best + 1e-3:
                stalled += 1
                if stalled >= 4:
                    break
            else:
                stalled = 0
            best = max(best, dotp)
            axis = np.cross(tool, down)
            n = float(np.linalg.norm(axis))
            if n < 1e-6:            # anti-parallel: any perpendicular axis will do
                axis, n = np.array([0.0, 1.0, 0.0]), 1.0
            axis = axis / n
            angle = float(np.arccos(dotp))
            n_steps = max(1, min(int(max_rot_steps), int(abs(angle) / 0.02)))
            self.controller.set_mode("rotate")
            # Base frame -> tool frame: see :meth:`_base_rotvec_to_tool`.
            self.follower.queue_rotate(*self._base_rotvec_to_tool(axis * angle), n_steps)
            for _ in range(n_steps):
                self.controller.step(self.robot, dt)
                self.sim.step(render=False)
                self.robot.update(dt)
                steps += 1
        self.controller.set_mode("translate")
        for _ in range(10):
            self.sim.step(render=False)
            self.robot.update(dt)
        after = float(-self.tool_axis_b()[2])
        ok = after >= tol
        # **Re-arm the orientation hold once the wrist is correct.** The hold was disabled
        # because holding the *home* orientation -- which points the gripper away from the table
        # -- over-constrains the IK so badly that the end effector cannot descend at all. Once
        # the tool axis is pointing down, holding it is no longer a constraint fighting the
        # motion but the thing that keeps the gripper square to the object through the descent,
        # which is what the laboratory's collector gets from its grasp-pose estimator.
        if ok and bool(getattr(self.cfg, "hold_after_orient", True)):
            self.controller.config.hold_orientation = True
            self.controller.reset(self.robot)   # re-captures the hold at the corrected pose
            for _ in range(10):
                self.sim.step(render=False)
                self.robot.update(dt)
        return {"downwardness_before": before, "downwardness_after": after,
                "rot_steps": steps, "ok": ok,
                "hold_rearmed": bool(ok and getattr(self.cfg, "hold_after_orient", True))}

    # ------------------------------------------------------------------ motion
    def _ee(self) -> np.ndarray:
        """End-effector position in the base frame."""
        from utilities import get_ee_pos_base_frame

        e = get_ee_pos_base_frame(self.robot, self.controller.config.ee_link_name)
        return np.asarray((e.detach().cpu() if hasattr(e, "detach") else e)).reshape(-1)[:3].astype(float)

    def _follow(self, wp: Waypoint) -> Tuple[bool, int]:
        """Drive the end effector to one waypoint through the proven controller.

        Gripper commands are queued through the follower rather than written directly, so the
        drive gains and the stable-grasp tuning the controller applies stay in force. Writing
        joint targets around it is how a grasp ends up with the fingers at the right angle and
        no force behind them.
        """
        import torch
        from utilities import get_ee_pos_base_frame

        dt = float(self.sim.get_physics_dt())
        max_steps = int(getattr(self.cfg, "max_steps_per_waypoint", 1800))
        if wp.phase in tuple(getattr(self.cfg, "deep_phases", ())):
            max_steps *= int(getattr(self.cfg, "deep_phase_step_multiplier", 1))
        self.follower.set_tolerance_m(float(wp.tol_m))
        self.follower.set_max_steps_per_waypoint(max_steps)
        self.follower.set_waypoints_b([tuple(float(v) for v in wp.xyz)])

        # Gripper first, so the hand is in the right state as it arrives.
        changed_grip = wp.gripper != self._gripper_state()
        if changed_grip:
            self._set_gripper(wp.gripper, dt)

        # **A waypoint that only changes the gripper is finished when the gripper has moved.**
        # Requiring position convergence as well is what stalled every grasp in the sweep: the
        # fingers close on the cube, contact lifts the hand about a centimetre, and the follower
        # then spends its entire budget trying to jog a *loaded* gripper back down to a target
        # 2.8 cm away -- 16,000 steps in the trace, ending with the cube still on the table and
        # the phase logged as a timeout. The grasp had in fact succeeded.
        prev = getattr(self, "_last_goal", None)
        same_pos = prev is not None and float(np.linalg.norm(np.asarray(wp.xyz, float) - prev)) < 1e-6
        self._last_goal = np.asarray(wp.xyz, dtype=float)
        if same_pos and changed_grip:
            # Settle on **physics alone**. Stepping the controller here was tried and is much
            # worse than doing nothing: the follower still holds the descend goal, so a closed
            # hand with a cube in it gets jogged toward a target it can no longer reach, and the
            # measured result was the end effector wandering 15 cm away from the grasp during
            # the settle -- leaving the cube on the table while every waypoint reported success.
            # Holding the last joint target lets contact forces resolve without commanding
            # anything new, which is all a settle is for.
            settle = int(getattr(self.cfg, "grasp_settle_steps", 40))
            for _ in range(settle):
                self.sim.step(render=False)
                self.robot.update(dt)
            return True, int(getattr(self.cfg, "gripper_steps", 60)) + settle

        for step in range(max_steps):
            ee_b = get_ee_pos_base_frame(self.robot, self.controller.config.ee_link_name)
            self.follower.set_current_pose_b(ee_b if isinstance(ee_b, torch.Tensor) else torch.tensor(ee_b))
            goal = torch.tensor([float(v) for v in wp.xyz], device=self.sim.device)
            here = (ee_b if isinstance(ee_b, torch.Tensor) else torch.tensor(ee_b)).to(self.sim.device).view(-1)[:3]
            if float(torch.linalg.norm(here - goal)) <= float(wp.tol_m):
                return True, step
            self.controller.set_mode("translate")
            self.controller.step(self.robot, dt)
            self.sim.step(render=False)
            self.robot.update(dt)
        return False, max_steps

    def _gripper_state(self) -> int:
        return getattr(self, "_grip", GRIPPER_OPEN)

    def _set_gripper(self, command: int, dt: float) -> None:
        """Open or close, then hold, through the controller's gripper mode."""
        n = int(getattr(self.cfg, "gripper_steps", 60))
        self.controller.set_mode("gripper")
        self.follower.queue_gripper(-1.0 if int(command) == GRIPPER_CLOSED else 1.0, n)
        for _ in range(n):
            self.controller.step(self.robot, dt)
            self.sim.step(render=False)
            self.robot.update(dt)
        self._grip = int(command)
        self.controller.set_mode("translate")

    # ------------------------------------------------------------------ capture
    def capture(self, scene, *, tag: str = "frame", which: Sequence[str] = ("topdown",)) -> Dict[str, Path]:
        """Save one frame per named camera into the atlas directory.

        ``topdown`` frames are what a policy trains on and share the collection contract's pose
        and field of view; ``figure`` frames are for the paper and are never used for training.
        They go to separate subdirectories so that cannot happen by accident.
        """
        out: Dict[str, Path] = {}
        if self.capture_dir is None:
            return out
        dt = float(self.sim.get_physics_dt())
        # The RTX renderer accumulates temporally, so a frame taken immediately after a teleport
        # still carries faint copies of the objects at their previous poses. Two render steps
        # left visible ghosts at the spawn row; this many clears them. It matters beyond
        # tidiness: these frames are the policy's training images, and a ghost is a plausible
        # object in exactly the place the model must learn is empty.
        for _ in range(int(getattr(self.cfg, "capture_render_steps", 12))):
            self.sim.step(render=True)
            self.robot.update(dt)
        for name in which:
            cam = getattr(self, "cameras", {}).get(name)
            if cam is None:
                continue
            try:
                cam.update(dt)
                rgb = cam.data.output["rgb"]
                arr = rgb[0].detach().cpu().numpy() if hasattr(rgb, "detach") else np.asarray(rgb)[0]
                if arr.dtype != np.uint8:
                    arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
                arr = arr[..., :3]
                if float(arr.std()) < 1.0:
                    print(f"[sup_scene] WARNING: '{name}' frame is nearly uniform "
                          f"(std={arr.std():.2f}); the render product may not be bound")
                from PIL import Image

                d = self.capture_dir / name / f"scene_{int(scene.scene_id):03d}"
                d.mkdir(parents=True, exist_ok=True)
                path = d / f"{tag}.png"
                Image.fromarray(arr).save(path)
                out[name] = path
            except Exception as exc:
                print(f"[sup_scene] capture '{name}' failed: {exc}")
        return out


def _yaw_quat(yaw: float) -> Tuple[float, float, float, float]:
    """``(w, x, y, z)`` for a rotation about +z."""
    h = 0.5 * float(yaw)
    return (math.cos(h), 0.0, 0.0, math.sin(h))


__all__ = ["SupervisoryFetchScene", "OBJECT_NAMES"]
