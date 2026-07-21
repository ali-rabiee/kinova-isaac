"""Shared scene-environment primitives.

Each concrete environment (``ycb_reach_to_grasp``, ``cubes``, ...) subclasses
``BaseSceneEnv`` and customizes:

- the scene config (table + robot defaults)
- the object-loader factory (USD vs. box mode, dataset paths, palette)

The class encapsulates every step a demo previously copy-pasted: building the
scene, applying physics to the SimulationCfg, attaching an optional top-down
camera, resetting the robot to its default pose, and constructing an
``ObjectLoader``. Demos drop down to ``self.robot``/``self.sim`` for arm/IK
work but never touch scene-construction details.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace as dc_replace
from typing import TYPE_CHECKING, Tuple

import numpy as np

from environments.utils.camera.front import DEFAULT_FRONT_CAMERA, FrontCameraConfig
from environments.utils.camera.topdown import DEFAULT_TOP_DOWN_CAMERA, TopDownCameraConfig
from environments.utils.camera.wrist import DEFAULT_WRIST_CAMERA, WristCameraConfig

if TYPE_CHECKING:  # only for typing -- importing these at module import time
    # would pull in heavy Omni/USD modules before AppLauncher has run.
    from isaaclab.assets import Articulation
    from isaaclab.sensors import Camera
    from isaaclab.sim import SimulationContext

    from environments.utils.object_loader import ObjectLoader, ObjectLoaderConfig
    from environments.utils.physix import PhysicsConfig


# Default joint configuration shared by both reach-to-grasp and cube-stacking
# scenes (everyone uses the same Kinova JACO2 N6S300 robot at the same height).
_DEFAULT_JACO2_HOME_POSE: dict[str, float] = {
    "j2n6s300_joint_1": 0.0 * np.pi,
    "j2n6s300_joint_2": np.pi,
    "j2n6s300_joint_3": 1.8 * np.pi,
    "j2n6s300_joint_4": 0.0 * np.pi,
    "j2n6s300_joint_5": 1.75 * np.pi,
    "j2n6s300_joint_6": 0.5 * np.pi,
    "j2n6s300_joint_finger_1": 0.0,
    "j2n6s300_joint_finger_2": 0.0,
    "j2n6s300_joint_finger_3": 0.0,
    "j2n6s300_joint_finger_tip_1": 0.0,
    "j2n6s300_joint_finger_tip_2": 0.0,
    "j2n6s300_joint_finger_tip_3": 0.0,
}


def default_jaco2_home_pose() -> dict[str, float]:
    """Return a fresh copy of the default Kinova JACO2 home joint pose."""
    return dict(_DEFAULT_JACO2_HOME_POSE)


def _default_table_usd_path() -> str:
    """Resolve the Thorlabs table USD on Isaac Nucleus (lazy import)."""
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

    return f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd"


@dataclass
class SceneConfig:
    """Shared scene configuration: table + robot placement.

    Used by every concrete environment. The robot is always the Kinova JACO2
    N6S300 mounted on top of the table.
    """

    num_origins: int = 1
    origin_spacing: float = 2.0

    # Resolved lazily so importing this module doesn't require Isaac Nucleus.
    table_usd_path: str | None = None
    table_scale: Tuple[float, float, float] = (1.5, 2.0, 1.0)
    table_translation: Tuple[float, float, float] = (0.0, 0.0, 0.8)

    robot_prim_path: str = "/World/Origin1/Robot"
    robot_base_height: float = 0.8

    # Optional initial joint configuration (joint name -> position).
    robot_default_joint_pos: dict[str, float] | None = field(
        default_factory=default_jaco2_home_pose
    )

    def resolved_table_usd_path(self) -> str:
        return self.table_usd_path or _default_table_usd_path()


@dataclass
class CameraConfig:
    """Default viewport camera placement (used by ``sim.set_camera_view``)."""

    eye: Tuple[float, float, float] = (3.5, 0.0, 3.2)
    target: Tuple[float, float, float] = (0.0, 0.0, 0.5)


# TopDownCameraConfig/FrontCameraConfig/WristCameraConfig now live in
# environments.utils.camera.{topdown,front,wrist} (one config per camera,
# fully self-contained); imported above and re-exported here so existing
# ``from environments.base import TopDownCameraConfig`` call sites keep
# working unchanged.

# Default singletons. Concrete env packages re-export these as module-level
# constants for convenience (``environments.ycb_reach_to_grasp.DEFAULT_SCENE``).
DEFAULT_SCENE = SceneConfig()
DEFAULT_CAMERA = CameraConfig()


def define_origins(num_origins: int, spacing: float) -> list[list[float]]:
    """Lay origins out in a square-ish grid, centered at the world origin."""
    import torch

    env_origins = torch.zeros(num_origins, 3)
    num_rows = np.floor(np.sqrt(num_origins))
    num_cols = np.ceil(num_origins / num_rows)
    xx, yy = torch.meshgrid(
        torch.arange(num_rows), torch.arange(num_cols), indexing="xy"
    )
    env_origins[:, 0] = (
        spacing * xx.flatten()[:num_origins] - spacing * (num_rows - 1) / 2
    )
    env_origins[:, 1] = (
        spacing * yy.flatten()[:num_origins] - spacing * (num_cols - 1) / 2
    )
    env_origins[:, 2] = 0.0
    return env_origins.tolist()


def design_scene(scene_cfg: SceneConfig) -> tuple[dict, list[list[float]]]:
    """Build the shared scene (ground/light/origin/table/robot).

    Returns ``(scene_entities, origins)`` where ``scene_entities['kinova_j2n6s300']``
    is the IsaacLab ``Articulation``. Importing IsaacLab/Omni is deferred until
    this is called so the module is safe to import at process start.
    """
    import importlib

    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation
    from isaaclab_assets import KINOVA_JACO2_N6S300_CFG

    # Ground + dome light.
    sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())
    light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light.func("/World/Light", light)

    origins = define_origins(
        num_origins=scene_cfg.num_origins, spacing=scene_cfg.origin_spacing
    )

    prim_utils = importlib.import_module("isaacsim.core.utils.prims")
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])

    # Table.
    table_cfg = sim_utils.UsdFileCfg(
        usd_path=scene_cfg.resolved_table_usd_path(),
        scale=scene_cfg.table_scale,
    )
    table_cfg.func("/World/Origin1/Table", table_cfg, translation=scene_cfg.table_translation)

    # Robot.
    robot_cfg = dc_replace(KINOVA_JACO2_N6S300_CFG, prim_path=scene_cfg.robot_prim_path)
    robot_cfg.init_state.pos = (0.0, 0.0, scene_cfg.robot_base_height)
    if scene_cfg.robot_default_joint_pos:
        robot_cfg.init_state.joint_pos = scene_cfg.robot_default_joint_pos  # type: ignore[arg-type]
    robot = Articulation(cfg=robot_cfg)

    return {"kinova_j2n6s300": robot}, origins


class BaseSceneEnv:
    """OOP wrapper around the JACO2 + table scene.

    Concrete subclasses override :meth:`build_object_loader_config` to control
    object spawning (USD dataset vs. uniform cubes). Everything else (table,
    robot, lights, simulation/physics setup, robot reset, top-down camera) is
    shared.

    Typical usage from a demo / data-collection script::

        env = YCBReachToGraspEnv()                    # or CubesEnv()
        sim = env.build_simulation()                  # SimulationContext
        scene_entities, origins = env.design_scene()  # ground/table/robot
        env.attach_top_down_camera()                  # optional
        loader = env.build_object_loader()
        prim_paths = loader.spawn(parent_prim_path="/World/Origin1", num_objects=N)
        env.reset(sim, scene_entities['kinova_j2n6s300'], origins)
    """

    #: Subclasses can override to change the default loader spawn-mode. ``"usd"``
    #: spawns from a dataset directory; ``"box"`` spawns uniform cubes.
    spawn_mode: str = "usd"

    def __init__(
        self,
        *,
        scene_cfg: SceneConfig | None = None,
        camera_cfg: CameraConfig | None = None,
        top_down_camera_cfg: TopDownCameraConfig | None = None,
        front_camera_cfg: FrontCameraConfig | None = None,
        wrist_camera_cfg: WristCameraConfig | None = None,
        physics_cfg: "PhysicsConfig | None" = None,
        device: str | None = None,
    ) -> None:
        from environments.utils.physix import PhysicsConfig

        self.scene_cfg: SceneConfig = scene_cfg or SceneConfig()
        self.camera_cfg: CameraConfig = camera_cfg or CameraConfig()
        self.top_down_camera_cfg: TopDownCameraConfig = (
            top_down_camera_cfg or TopDownCameraConfig()
        )
        self.front_camera_cfg: FrontCameraConfig = front_camera_cfg or FrontCameraConfig()
        self.wrist_camera_cfg: WristCameraConfig = wrist_camera_cfg or WristCameraConfig()
        self.physics_cfg: PhysicsConfig = (
            physics_cfg if physics_cfg is not None else PhysicsConfig()
        )
        if device is not None:
            self.physics_cfg.device = str(device)

        # Populated by ``design_scene`` / ``build_simulation``.
        self.sim: "SimulationContext | None" = None
        self.scene_entities: dict | None = None
        self.scene_origins: list[list[float]] | None = None

        # Populated on demand by build_*_camera_sensor(); None until called.
        self.top_down_camera_sensor: "Camera | None" = None
        self.front_camera_sensor: "Camera | None" = None
        self.wrist_camera_sensor: "Camera | None" = None

    # ------------------------------------------------------------------
    # Scene + simulation construction
    # ------------------------------------------------------------------
    def build_simulation(self) -> "SimulationContext":
        """Create a ``SimulationContext`` honoring this env's PhysicsConfig."""
        import isaaclab.sim as sim_utils

        from environments.utils.physix import apply_to_simulation_cfg

        sim_cfg = sim_utils.SimulationCfg(device=self.physics_cfg.device)
        apply_to_simulation_cfg(sim_cfg, self.physics_cfg)
        self.sim = sim_utils.SimulationContext(sim_cfg)
        return self.sim

    def design_scene(self) -> tuple[dict, list[list[float]]]:
        """Build ground/light/origin/table/robot and cache the result."""
        self.scene_entities, self.scene_origins = design_scene(self.scene_cfg)
        return self.scene_entities, self.scene_origins

    def set_default_camera_view(self) -> None:
        """Apply the env's default camera_cfg to the live SimulationContext."""
        if self.sim is None:
            raise RuntimeError("Call build_simulation() before set_default_camera_view().")
        self.sim.set_camera_view(self.camera_cfg.eye, self.camera_cfg.target)

    def attach_top_down_camera(self) -> None:
        """Spawn the overhead camera prim defined by ``top_down_camera_cfg``."""
        from environments.utils.camera import create_topdown_camera

        create_topdown_camera(self.top_down_camera_cfg)

    def attach_front_camera(self) -> None:
        """Spawn the front-angled camera prim defined by ``front_camera_cfg``
        (mirrors ``attach_top_down_camera``)."""
        from environments.utils.camera import create_front_camera

        create_front_camera(self.front_camera_cfg)

    def build_top_down_camera_sensor(self) -> "Camera":
        """Build (and cache) a ready-to-capture IsaacLab ``Camera`` sensor for
        the top-down view. Safe to call even if ``attach_top_down_camera()``
        hasn't run yet -- it creates the raw prim on demand."""
        from environments.utils.camera import build_topdown_camera_sensor

        self.top_down_camera_sensor = build_topdown_camera_sensor(self.top_down_camera_cfg)
        return self.top_down_camera_sensor

    def build_front_camera_sensor(self) -> "Camera":
        """Build (and cache) a ready-to-capture IsaacLab ``Camera`` sensor for
        the front view. Safe to call even if ``attach_front_camera()`` hasn't
        run yet -- it creates the raw prim on demand."""
        from environments.utils.camera import build_front_camera_sensor

        self.front_camera_sensor = build_front_camera_sensor(self.front_camera_cfg)
        return self.front_camera_sensor

    def build_wrist_camera_sensor(self) -> "Camera":
        """Build (and cache) a ready-to-capture IsaacLab ``Camera`` sensor for
        the wrist/eye-in-hand view. Requires ``design_scene()`` to have run
        already (needs ``self.robot``)."""
        from environments.utils.camera import build_wrist_camera_sensor

        self.wrist_camera_sensor = build_wrist_camera_sensor(self.robot, self.wrist_camera_cfg)
        return self.wrist_camera_sensor

    @property
    def robot(self) -> "Articulation":
        if self.scene_entities is None:
            raise RuntimeError("Call design_scene() before accessing .robot.")
        return self.scene_entities["kinova_j2n6s300"]

    @property
    def origin0(self) -> list[float]:
        if self.scene_origins is None:
            raise RuntimeError("Call design_scene() before accessing .origin0.")
        return self.scene_origins[0]

    # ------------------------------------------------------------------
    # Robot reset
    # ------------------------------------------------------------------
    def reset(self, sim: "SimulationContext | None" = None, robot=None, origins=None) -> None:
        """Reset the simulation and place the robot at its default pose."""
        import torch

        sim = sim if sim is not None else self.sim
        if sim is None:
            raise RuntimeError("No SimulationContext bound; call build_simulation() first.")
        robot = robot if robot is not None else self.robot
        origins = origins if origins is not None else self.scene_origins
        if origins is None:
            raise RuntimeError("No scene origins available; call design_scene() first.")

        sim.reset()
        origin0 = torch.tensor(origins[0], device=sim.device)
        root_state = robot.data.default_root_state.clone()
        root_state[:, :3] += origin0
        robot.write_root_pose_to_sim(root_state[:, :7])
        robot.write_root_velocity_to_sim(root_state[:, 7:])
        robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
        robot.reset()

    # ------------------------------------------------------------------
    # Object loader factory (subclasses override the cfg builder)
    # ------------------------------------------------------------------
    def build_object_loader_config(self, **overrides) -> "ObjectLoaderConfig":
        """Build an ``ObjectLoaderConfig`` for this env.

        Subclasses must implement this. ``overrides`` lets callers replace any
        field on the resulting config (e.g. spawn bounds, scale range).
        """
        raise NotImplementedError

    def build_object_loader(self, **overrides) -> "ObjectLoader":
        """Instantiate an ``ObjectLoader`` configured for this env."""
        from environments.utils.object_loader import ObjectLoader

        return ObjectLoader(self.build_object_loader_config(**overrides))
