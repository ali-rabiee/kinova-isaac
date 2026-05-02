from dataclasses import replace as dc_replace
import importlib

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: F401 - kept for config compatibility with existing envs.
from isaaclab.assets import Articulation
import numpy as np
import torch

from environments.blocks.config import BASE_ROBOT_CFG, SceneConfig


def define_origins(num_origins: int, spacing: float) -> list[list[float]]:
    """Defines the origins of the scene in a grid layout."""
    env_origins = torch.zeros(num_origins, 3)
    num_rows = np.floor(np.sqrt(num_origins))
    num_cols = np.ceil(num_origins / num_rows)
    xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="xy")
    env_origins[:, 0] = spacing * xx.flatten()[:num_origins] - spacing * (num_rows - 1) / 2
    env_origins[:, 1] = spacing * yy.flatten()[:num_origins] - spacing * (num_cols - 1) / 2
    env_origins[:, 2] = 0.0
    return env_origins.tolist()


def design_scene(scene_cfg: SceneConfig) -> tuple[dict, list[list[float]]]:
    """Design the blocks scene using the provided configuration."""
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    origins = define_origins(num_origins=scene_cfg.num_origins, spacing=scene_cfg.origin_spacing)

    prim_utils = importlib.import_module("isaacsim.core.utils.prims")
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])

    cfg = sim_utils.UsdFileCfg(
        usd_path=scene_cfg.table_usd_path,
        scale=scene_cfg.table_scale,
    )
    cfg.func("/World/Origin1/Table", cfg, translation=scene_cfg.table_translation)

    kinova_j2n6_cfg = dc_replace(BASE_ROBOT_CFG, prim_path=scene_cfg.robot_prim_path)
    kinova_j2n6_cfg.init_state.pos = (0.0, 0.0, scene_cfg.robot_base_height)
    if getattr(scene_cfg, "robot_default_joint_pos", None):
        kinova_j2n6_cfg.init_state.joint_pos = scene_cfg.robot_default_joint_pos  # type: ignore[arg-type]

    kinova_j2n6s300 = Articulation(cfg=kinova_j2n6_cfg)
    return {"kinova_j2n6s300": kinova_j2n6s300}, origins

