"""Translate tool calls into motion commands (planner + waypoint follower)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from controllers.input.waypoint_follower import WaypointFollowerInput
from data_collection.core.input_mux import CommandMuxInputProvider
from motion_generation.planners import PlannerContext, create_planner

from .extractor import world_to_base

YAW_BIN_TO_RAD = {
    "N": 0.0,
    "NE": math.pi / 4.0,
    "E": math.pi / 2.0,
    "SE": 3.0 * math.pi / 4.0,
    "S": math.pi,
    "SW": -3.0 * math.pi / 4.0,
    "W": -math.pi / 2.0,
    "NW": -math.pi / 4.0,
}


@dataclass
class ExecutorConfig:
    pregrasp_offset_m: float
    grasp_depth_m: float
    lift_height_m: float
    align_steps: int


class ActionExecutor:
    """Manage execution of APPROACH/ALIGN_YAW tool calls."""

    def __init__(
        self,
        *,
        cfg: ExecutorConfig,
        controller,
        mux_input: CommandMuxInputProvider,
        teleop_provider,
        planner_kind: str,
        planner_ctx: PlannerContext,
        ee_body_id: int,
        device: str = "cpu",
        step_pos_m: Optional[float] = None,
    ) -> None:
        self.cfg = cfg
        self.controller = controller
        self.mux_input = mux_input
        self.teleop_provider = teleop_provider
        self.ee_body_id = int(ee_body_id)
        self.device = torch.device(device)
        sp = step_pos_m
        if sp is None:
            sp = float(getattr(controller.config, "linear_speed_mps", 0.05)) * float(getattr(controller, "dt", 0.01) or 1.0)
        self.waypoint_input = WaypointFollowerInput(step_pos_m=float(sp), tol_m=0.005, device=str(device))
        self.planner = create_planner(planner_kind, ctx=planner_ctx)
        self.active_action: Optional[str] = None
        self.prev_mode: Optional[str] = None

    def cancel(self) -> None:
        self.waypoint_input.reset()
        self.active_action = None
        self.mux_input.set_base(self.teleop_provider)
        if self.prev_mode:
            try:
                self.controller.set_mode(self.prev_mode)
            except Exception:
                pass
        self.prev_mode = None

    def _current_ee_pos_b(self, robot) -> Tuple[float, float, float]:
        ee_pose_w = robot.data.body_pose_w[:, self.ee_body_id]
        root_pose_w = robot.data.root_pose_w
        ee_pos_w = ee_pose_w[0, 0:3].tolist()
        ee_quat_w = ee_pose_w[0, 3:7].tolist()
        base_pos_w = root_pose_w[0, 0:3].tolist()
        base_quat_w = root_pose_w[0, 3:7].tolist()
        pos_b, _ = world_to_base(ee_pos_w, ee_quat_w, base_pos_w, base_quat_w)
        return pos_b

    def tick(self, robot) -> None:
        """Update follower with current pose and auto-restore teleop on completion."""
        # Always update current pose so the follower can compute deltas.
        try:
            pos_b = self._current_ee_pos_b(robot)
            self.waypoint_input.set_current_pose_b(torch.tensor(pos_b, dtype=torch.float32, device=self.device))
        except Exception:
            pass

        if self.active_action and not self._is_follower_active():
            self.cancel()

    def _is_follower_active(self) -> bool:
        if getattr(self.waypoint_input, "_gripper_steps_left", 0) > 0:
            return True
        if getattr(self.waypoint_input, "_rot_steps_left", 0) > 0:
            return True
        if getattr(self.waypoint_input, "_waypoints_b", []):
            return True
        return False

    def _set_mode(self, mode: str) -> None:
        try:
            if self.prev_mode is None:
                self.prev_mode = getattr(self.controller, "_mode", None)
            self.controller.set_mode(mode)
        except Exception:
            pass

    def execute(self, tool_call: Dict[str, Any], *, objects_b: List[Dict[str, Any]], robot, gripper_yaw_bin: Optional[str] = None) -> bool:
        tool = tool_call.get("tool")
        args = tool_call.get("args") or {}
        if tool not in {"APPROACH", "ALIGN_YAW"}:
            return False
        obj_id = args.get("obj")
        target = next((o for o in objects_b if str(o.get("id")) == str(obj_id)), None)
        if target is None:
            print(f"[EXECUTOR] Target obj {obj_id} not found.")
            return False

        if tool == "APPROACH":
            return self._execute_approach(target, robot)
        return self._execute_align(target, robot, gripper_yaw_bin)

    def _execute_approach(self, target: Dict[str, Any], robot) -> bool:
        pos_b = target.get("_pos_b")
        if pos_b is None:
            print("[EXECUTOR] Target missing base-frame position; cannot approach.")
            return False
        waypoints = self.planner.plan_to_pose_b(
            target_pos_b=tuple(pos_b),
            target_quat_b_wxyz=None,
            pregrasp_offset_m=float(self.cfg.pregrasp_offset_m),
            grasp_depth_m=float(self.cfg.grasp_depth_m),
            lift_height_m=float(self.cfg.lift_height_m),
        )
        if not waypoints:
            print("[EXECUTOR] Planner returned no waypoints.")
            return False
        self.waypoint_input.set_waypoints_b(waypoints)
        self._set_mode("translate")
        self.mux_input.set_base(self.waypoint_input)
        self.active_action = "APPROACH"
        return True

    def _execute_align(self, target: Dict[str, Any], robot, gripper_yaw_bin: Optional[str]) -> bool:
        yaw_bin = target.get("yaw")
        if yaw_bin is None:
            print("[EXECUTOR] Target missing yaw; cannot align.")
            return False
        target_rad = YAW_BIN_TO_RAD.get(yaw_bin, 0.0)
        cur_rad = YAW_BIN_TO_RAD.get(gripper_yaw_bin or "N", 0.0)
        delta = (target_rad - cur_rad + math.pi) % (2 * math.pi) - math.pi
        self.waypoint_input.queue_rotate_z(float(delta), steps=int(self.cfg.align_steps))
        self._set_mode("rotate")
        self.mux_input.set_base(self.waypoint_input)
        self.active_action = "ALIGN_YAW"
        return True
