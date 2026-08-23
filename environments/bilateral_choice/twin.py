"""W10 — the Isaac Lab scene and control loop for the Phase 0 apparatus twin.

Every Isaac/Omniverse symbol is imported **inside** a method. Importing this module must stay
free so ``vla_lab.old_direction.rehab``'s test suite, the synthetic pilot, and the analysis keep running on
a machine with no simulator; a missing simulator then fails at :meth:`BilateralChoiceTwin.setup`
with an instruction rather than at import with a traceback.

What the twin is for (``rehab.md`` §6/W10) — validating, before a person is near the arm:

1. every contract target is reachable from the study mounting pose,
2. no presentation trajectory intersects the seated-participant proxy,
3. the re-aimed wrist camera actually frames where a hand will arrive, and
4. the safety envelope's speed and workspace limits are survivable in practice.

The twin uses the **same** :class:`~vla_lab.old_direction.rehab.apparatus.base.Apparatus` protocol, the same
target IDs, and the same contract as the real backend, so a dry-run exercises the session code
path instead of a parallel one.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import DEFAULT_TWIN, TwinConfig


class BilateralChoiceTwin:
    """The Isaac scene plus a blocking ``move_to`` that mirrors the real arm's semantics."""

    def __init__(
        self,
        contract: Any,
        *,
        proxy: Any = None,
        cfg: Optional[TwinConfig] = None,
        headless: bool = True,
        max_speed_ms: float = 0.12,
    ) -> None:
        self.contract = contract
        self.cfg = cfg or DEFAULT_TWIN
        self.proxy = proxy
        self.headless = bool(headless)
        #: Cartesian speed cap, matching :class:`~vla_lab.old_direction.rehab.safety.SafetyLimits`. The twin
        #: moves at the speed the study will actually use, or it has validated nothing.
        self.max_speed_ms = float(max_speed_ms)
        self.grid = contract.target_grid()
        self._app: Any = None
        self._sim: Any = None
        self._robot: Any = None
        self._ctrl: Any = None
        self._waypoints: Any = None
        self._ee_body_id: Optional[int] = None
        self._cameras: Dict[str, Any] = {}
        self._ee_xy = tuple(self.cfg.home_xy_participant_m)

    # -- frames ------------------------------------------------------------
    def participant_to_world(self, xy: Sequence[float]) -> Tuple[float, float, float]:
        """Participant-frame table point -> world coordinates.

        The robot base sits at ``scene.robot_base_world``; the contract says where the
        participant frame is *relative to the robot*, so the chain is
        ``participant -> robot -> world``.
        """

        p2r = self.contract.participant_to_robot()
        rx, ry, _ = p2r.apply((float(xy[0]), float(xy[1]), 0.0))
        bx, by, bz = self.cfg.scene.robot_base_world
        return (bx + rx, by + ry, bz + float(self.cfg.present_height_m))

    # -- lifecycle ---------------------------------------------------------
    def setup(self) -> None:
        """Launch Isaac, build the scene. Raises with an instruction if Isaac is absent."""

        try:
            from isaaclab.app import AppLauncher
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Isaac Lab is not importable. Run the twin under the Isaac python "
                "(conda activate riften), or use --apparatus null for offline work."
            ) from exc

        launcher = AppLauncher({"headless": self.headless})
        self._app = launcher.app
        self._build_scene()

    def _build_scene(self) -> None:
        import isaacsim.core.utils.prims as prim_utils
        from isaaclab.assets import Articulation
        from isaaclab.sim import SimulationCfg, SimulationContext
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
        from isaaclab_assets import KINOVA_JACO2_N6S300_CFG

        sc = self.cfg.scene
        self._sim = SimulationContext(SimulationCfg(dt=1.0 / 120.0))

        prim_utils.create_prim("/World/Phase0", "Xform")
        prim_utils.create_prim(
            "/World/Phase0/Table",
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd",
            translation=sc.table_translation,
            scale=sc.table_scale,
        )
        robot_cfg = KINOVA_JACO2_N6S300_CFG.replace(prim_path=sc.robot_prim_path)
        robot_cfg.init_state.pos = sc.robot_base_world
        if sc.robot_default_joint_pos:
            robot_cfg.init_state.joint_pos = dict(sc.robot_default_joint_pos)
        self._robot = Articulation(robot_cfg)

        self._build_participant_proxy()
        self._build_target_markers()
        self._build_cameras()
        self._sim.reset()

    def _build_participant_proxy(self) -> None:
        """A visible keep-out volume: reviewers should be able to *see* what is being checked."""

        import isaacsim.core.utils.prims as prim_utils
        from pxr import Gf, UsdGeom
        import omni.usd

        sc = self.cfg.scene
        origin = self.participant_to_world((sc.participant_torso_center_offset_m, 0.0))
        stage = omni.usd.get_context().get_stage()
        prim_utils.create_prim(sc.participant_prim_path, "Xform")
        torso = UsdGeom.Cylinder.Define(stage, f"{sc.participant_prim_path}/Torso")
        torso.CreateRadiusAttr(float(sc.participant_torso_radius_m))
        torso.CreateHeightAttr(float(sc.participant_torso_height_m))
        UsdGeom.Xformable(torso).AddTranslateOp().Set(
            Gf.Vec3d(origin[0], origin[1], sc.table_height_m + sc.participant_torso_height_m / 2.0)
        )
        head = UsdGeom.Sphere.Define(stage, f"{sc.participant_prim_path}/Head")
        head.CreateRadiusAttr(float(sc.participant_head_radius_m))
        UsdGeom.Xformable(head).AddTranslateOp().Set(
            Gf.Vec3d(origin[0], origin[1], sc.table_height_m + sc.participant_torso_height_m + 0.12)
        )

    def _build_target_markers(self) -> None:
        import isaacsim.core.utils.prims as prim_utils
        from pxr import Gf, UsdGeom
        import omni.usd

        sc = self.cfg.scene
        stage = omni.usd.get_context().get_stage()
        prim_utils.create_prim(sc.target_marker_prim_root, "Xform")
        for t in self.grid:
            w = self.participant_to_world((t.x_m, t.y_m))
            path = f"{sc.target_marker_prim_root}/T{int(t.target_id):02d}"
            disk = UsdGeom.Cylinder.Define(stage, path)
            disk.CreateRadiusAttr(float(sc.target_marker_radius_m))
            disk.CreateHeightAttr(float(sc.target_marker_height_m))
            UsdGeom.Xformable(disk).AddTranslateOp().Set(Gf.Vec3d(w[0], w[1], sc.table_height_m))

    def _build_cameras(self) -> None:
        from isaaclab.sensors import Camera, CameraCfg
        from isaaclab.utils.math import quat_from_euler_xyz
        import isaacsim.core.utils.prims as prim_utils
        import torch

        fc = self.cfg.front_camera
        prim_utils.create_prim(fc.prim_path, "Camera")
        # spawn=None + an explicit offset attaches the sensor to the prim that already exists.
        # Re-spawning here is the documented cause of silent black images in this repo.
        self._cameras["front"] = Camera(
            CameraCfg(
                prim_path=fc.prim_path,
                spawn=None,
                height=int(fc.resolution[1]),
                width=int(fc.resolution[0]),
                data_types=["rgb"],
            )
        )
        wc = self.cfg.wrist_camera
        prim_utils.create_prim(wc.prim_path, "Camera")
        self._cameras["wrist"] = Camera(
            CameraCfg(
                prim_path=wc.prim_path,
                spawn=None,
                height=int(wc.resolution[1]),
                width=int(wc.resolution[0]),
                data_types=["rgb"],
            )
        )

    def close(self) -> None:
        if self._app is not None:
            try:
                self._app.close()
            finally:
                self._app = None

    # -- apparatus semantics ------------------------------------------------
    def home(self) -> None:
        self.move_to(self.cfg.home_xy_participant_m, tolerance_m=0.02, dwell_ms=200, timeout_ms=8000)

    def _ensure_controller(self) -> None:
        """Build the jog controller + waypoint follower once, on first use.

        Uses the repo's existing ``cartesian_velocity`` controller driven by
        ``input/waypoint_follower.py`` — the twin is supposed to reuse these as-is
        (``rehab.md`` §8), not grow a parallel motion stack.
        """

        from controllers import CartesianVelocityJogConfig, CartesianVelocityJogController
        from controllers.input.waypoint_follower import WaypointFollowerInput

        if self._ctrl is not None:
            return
        dt = self._sim.get_physics_dt()
        # Per-step displacement implied by the safety speed cap. The cap is the study's, not
        # the arm's: a human's hands are in the workspace (W12).
        step_pos_m = float(self.max_speed_ms) * float(dt)
        cfg = CartesianVelocityJogConfig(
            ee_link_name="j2n6s300_end_effector",
            linear_speed_mps=float(self.max_speed_ms),
        )
        self._ctrl = CartesianVelocityJogController(cfg, num_envs=1, device=str(self._sim.device))
        self._waypoints = WaypointFollowerInput(
            step_pos_m=step_pos_m,
            tol_m=float(self.contract.timing.settle_tolerance_m),
            device=str(self._sim.device),
        )
        self._ctrl.set_input_provider(self._waypoints)
        self._ctrl.reset(self._robot)

    def move_to(
        self,
        xy_participant: Sequence[float],
        *,
        tolerance_m: float,
        dwell_ms: int,
        timeout_ms: int,
    ) -> Tuple[bool, float]:
        """Blocking Cartesian move with a verified settle. Returns ``(settled, pose_error_m)``.

        Mirrors the real backend's contract exactly: the arm moves, **stops**, and holds the
        position tolerance for ``dwell_ms`` before the caller is allowed to issue GO. The dwell
        is what makes "settled" mean *stopped*, rather than "passed through tolerance once".
        """

        if self._sim is None or self._robot is None:
            raise RuntimeError("BilateralChoiceTwin.setup() must be called before move_to()")
        self._ensure_controller()

        goal_w = self.participant_to_world(xy_participant)
        goal_b = self._world_to_base(goal_w)
        self._waypoints.reset()
        self._waypoints.set_tolerance_m(float(tolerance_m))
        self._waypoints.set_waypoints_b([goal_b])

        dt = self._sim.get_physics_dt()
        elapsed_ms = 0.0
        held_ms = 0.0
        err = float("inf")
        while elapsed_ms < float(timeout_ms):
            ee_b = self._ee_base()
            self._waypoints.set_current_pose_b(self._as_tensor(ee_b))
            err = math.dist(ee_b[:2], goal_b[:2])
            self._ctrl.step(self._robot, dt)
            self._sim.step()
            self._robot.update(dt)
            elapsed_ms += dt * 1000.0
            held_ms = (held_ms + dt * 1000.0) if err <= float(tolerance_m) else 0.0
            if held_ms >= float(dwell_ms):
                self._ee_xy = (float(xy_participant[0]), float(xy_participant[1]))
                return True, float(err)
        return False, float(err)

    # -- pose helpers ------------------------------------------------------
    def _as_tensor(self, xyz: Sequence[float]):
        import torch

        return torch.tensor([float(xyz[0]), float(xyz[1]), float(xyz[2])], device=str(self._sim.device))

    def _ee_base(self) -> Tuple[float, float, float]:
        """EE position in the robot **base** frame — what the waypoint follower consumes."""

        from isaaclab.utils.math import subtract_frame_transforms

        if self._ee_body_id is None:
            ids, _ = self._robot.find_bodies(["j2n6s300_end_effector"])
            self._ee_body_id = int(ids[0])
        root_pose_w = self._robot.data.root_pose_w
        ee_pose_w = self._robot.data.body_pose_w[:, self._ee_body_id]
        ee_pos_b, _ = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        p = ee_pos_b[0]
        return (float(p[0]), float(p[1]), float(p[2]))

    def _world_to_base(self, xyz_w: Sequence[float]) -> Tuple[float, float, float]:
        bx, by, bz = self.cfg.scene.robot_base_world
        return (float(xyz_w[0]) - bx, float(xyz_w[1]) - by, float(xyz_w[2]) - bz)

    # -- the dry-run checks -------------------------------------------------
    def check_reachable(self, xy_participant: Sequence[float]) -> Tuple[bool, str]:
        """Can the arm actually present this point? Answered by *trying*, in the simulator."""

        settled, err = self.move_to(
            xy_participant,
            tolerance_m=float(self.contract.timing.settle_tolerance_m),
            dwell_ms=int(self.contract.timing.settle_dwell_ms),
            timeout_ms=int(self.contract.timing.present_timeout_ms),
        )
        return (settled, "" if settled else f"did not settle; residual error {1000*err:.0f} mm")

    def render_wrist_view(self, target_id: int, *, out_dir: Optional[str] = None) -> Optional[str]:
        """Render the wrist camera at the current pose. Answers "will the hand be in frame?"."""

        cam = self._cameras.get("wrist")
        if cam is None:
            return None
        try:
            import imageio.v3 as iio
            import numpy as np

            cam.update(dt=0.0)
            rgb = cam.data.output["rgb"][0].cpu().numpy()
            out = Path(out_dir or "vla_lab/results/rehab_phase0/twin_views")
            out.mkdir(parents=True, exist_ok=True)
            path = out / f"wrist_target_{int(target_id):02d}.png"
            iio.imwrite(path, np.asarray(rgb, dtype="uint8"))
            return str(path)
        except Exception:  # noqa: BLE001 - a missing image writer must not fail the sweep
            return None


__all__ = ["BilateralChoiceTwin"]
