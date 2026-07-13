"""Cartesian velocity jogging controller implementation."""

from __future__ import annotations

import torch
from typing import Optional, Literal, Tuple

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import subtract_frame_transforms, quat_box_plus, quat_apply

from ..base import ArmControllerConfig, ArmController
from ..safety import ArmSafetyCfg, ArmSafety
from dataclasses import dataclass, field, replace
from kinova import GripperConfig, GripperController, MotionCommandBuilder, MotionPrimitives

@dataclass
class CartesianVelocityJogConfig(ArmControllerConfig):
    """Config for Cartesian velocity jogging using Differential IK and gripper control."""

    linear_speed_mps: float = 0.05
    ik_method: Literal["pinv", "svd", "trans", "dls"] = "dls"
    hold_orientation: bool = True
    # Gripper settings (migrated to kinova.GripperConfig; legacy fields kept for backward compat)
    gripper_joint_regex: str = ".*_joint_finger_.*|.*_joint_finger_tip_.*"
    gripper_open_pos: float = 0.0
    gripper_close_pos: float = 1.2 
    gripper_cfg: GripperConfig | None = None
    # Safety config (legacy thresholds moved here)
    safety_cfg: ArmSafetyCfg = field(default_factory=lambda: ArmSafetyCfg(
        min_sigma_thresh=0.005,
        joint_limit_margin_rad=0.10,
    ))
    # Legacy workspace bounds (now part of safety_cfg, but kept for backward compat)
    workspace_min: Optional[Tuple[Optional[float], Optional[float], Optional[float]]] = field(default=(None, None, None))
    workspace_max: Optional[Tuple[Optional[float], Optional[float], Optional[float]]] = field(default=(None, None, None))

    def __post_init__(self):
        # Migrate legacy bounds to safety_cfg if not set
        if self.safety_cfg.workspace_min is None:
            self.safety_cfg = replace(self.safety_cfg, workspace_min=self.workspace_min)
        if self.safety_cfg.workspace_max is None:
            self.safety_cfg = replace(self.safety_cfg, workspace_max=self.workspace_max)
        # Default gripper config if not provided
        if self.gripper_cfg is None:
            self.gripper_cfg = GripperConfig(
                joint_regex=self.gripper_joint_regex,
                open_position=self.gripper_open_pos,
                close_position=self.gripper_close_pos,
            )


class CartesianVelocityJogController(ArmController):
    """Controller mapping Cartesian deltas to joint velocity via Diff-IK, with a gripper mode.

    Input provider should return up to 7D per step: [dx, dy, dz, rx, ry, rz, g]
    - Translation mode: uses [dx, dy, dz], holds orientation.
    - Rotation mode: uses [rx, ry, rz] (rotation vector, EE frame), holds position.
    - Gripper mode: uses [g] if present (g>0 open, g<0 close) and holds pose.
    - Twist mode: uses all of [dx, dy, dz, rx, ry, rz, g] simultaneously
      (policy rollout). Unlike rotation mode, [rx, ry, rz] is interpreted in
      the BASE frame (matching the logged `ee_delta_rotvec_b` action
      convention); g > 0.5 opens, g < -0.5 closes, otherwise the gripper holds.
    """

    def __init__(self, config: CartesianVelocityJogConfig, num_envs: int = 1, device: Optional[str] = None) -> None:
        super().__init__(config, num_envs=num_envs, device=device)
        self.config: CartesianVelocityJogConfig
        self.safety = ArmSafety(self.config.safety_cfg, num_envs, str(self.device))
        self._diff_ik: Optional[DifferentialIKController] = None
        self._arm_joint_ids = None
        self._ee_body_id = None
        self._ee_jacobi_idx = None
        self._ee_quat_hold_b: Optional[torch.Tensor] = None
        self._ee_quat_last_safe_b: Optional[torch.Tensor] = None
        self._step_count = 0
        self._mode: Literal["translate", "rotate", "gripper", "twist"] = "translate"
        self._refresh_hold_ori_on_translate: bool = False
        # High-level helpers
        _gc = self.config.gripper_cfg or GripperConfig(
            joint_regex=self.config.gripper_joint_regex,
            open_position=self.config.gripper_open_pos,
            close_position=self.config.gripper_close_pos,
        )
        self.gripper = GripperController(_gc, num_envs, str(self.device))
        self._cmd_builder = MotionCommandBuilder(device=str(self.device))
        self.motion = MotionPrimitives(self._cmd_builder, self)

    def set_mode(self, mode: Literal["translate", "rotate", "gripper", "twist"]) -> None:
        if self._mode != mode:
            self._mode = mode
            print(f"[CTRL] Mode set to: {self._mode}")
            if mode == "translate":
                # Refresh the orientation to hold the current orientation when returning to translate
                self._refresh_hold_ori_on_translate = True

    def reset(self, robot) -> None:
        diff_ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=self.config.use_relative_mode,
            ik_method=self.config.ik_method,
        )
        self._diff_ik = DifferentialIKController(diff_ik_cfg, num_envs=self.num_envs, device=self.device)
        self._diff_ik.reset()

        # Resolve indices for EE, arm joints, and gripper joints
        body_ids, _ = robot.find_bodies([self.config.ee_link_name])
        self._ee_body_id = int(body_ids[0])
        self._ee_jacobi_idx = self._ee_body_id - 1 if robot.is_fixed_base else self._ee_body_id
        arm_joint_ids, _ = robot.find_joints(self.config.arm_joint_regex)
        self._arm_joint_ids = arm_joint_ids
        # Prepare gripper mappings and holds
        robot.set_joint_position_target(robot.data.joint_pos)
        self.gripper.resolve_joints(robot)
        self.gripper.reset(robot)
        # Apply stable grasp tuning and drive gains on the robot prim path when available
        prim_path = getattr(getattr(robot, "cfg", None), "prim_path", None)
        if isinstance(prim_path, str):
            self.gripper.set_drive_gains(prim_path)
            self.gripper.apply_stable_grasp_tuning(prim_path)

        # Capture initial EE orientation in base frame to optionally hold during translation
        root_pose_w = robot.data.root_pose_w
        ee_pose_w = robot.data.body_pose_w[:, self._ee_body_id]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        self._ee_quat_hold_b = ee_quat_b.clone()
        self._ee_quat_last_safe_b = ee_quat_b.clone()

    def step(self, robot, dt: float) -> None:
        assert self._diff_ik is not None, "Controller not reset()"
        assert self._arm_joint_ids is not None, "Controller not reset()"
        assert self._ee_body_id is not None, "Controller not reset()"

        # Read input provider and robustly shape it to (N, D)
        if self._input_provider is None:
            cmd = torch.zeros(1, 6, device=self.device)
        else:
            cmd = self._input_provider.advance()
            if cmd.ndim == 1:
                cmd = cmd.view(1, -1)
        if str(cmd.device) != str(torch.device(self.device)):
            cmd = cmd.to(self.device)
        # ensure at least 6, keep extra (g) if present
        if cmd.shape[-1] < 6:
            pad = torch.zeros(cmd.shape[0], 6 - cmd.shape[-1], device=cmd.device, dtype=cmd.dtype)
            cmd = torch.cat([cmd, pad], dim=-1)

        dpos = cmd[..., 0:3]
        drot = cmd[..., 3:6]
        g_val: Optional[torch.Tensor] = None
        if cmd.shape[-1] >= 7:
            g_val = cmd[..., 6]

        # State
        jac = robot.root_physx_view.get_jacobians()[:, self._ee_jacobi_idx, :, self._arm_joint_ids]
        ee_pose_w = robot.data.body_pose_w[:, self._ee_body_id]
        root_pose_w = robot.data.root_pose_w
        q_arm = robot.data.joint_pos[:, self._arm_joint_ids]
        arm_limits = robot.data.soft_joint_pos_limits[:, self._arm_joint_ids, :]  # (N, dof, 2)
        q_lower = arm_limits[..., 0]
        q_upper = arm_limits[..., 1]

        # Relative pose frame
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )

        if self._mode == "rotate":
            drot = quat_apply(ee_quat_b, drot)

        # If returning to translate mode, refresh the held orientation to current orientation
        if self._mode == "translate" and (self._ee_quat_hold_b is None or self._refresh_hold_ori_on_translate):
            self._ee_quat_hold_b = ee_quat_b.clone()
            self._refresh_hold_ori_on_translate = False
            # update last safe orientation as well
            self._ee_quat_last_safe_b = ee_quat_b.clone()

        # Build desired pose based on mode
        # Compute generic 6D twist and project away from low-sigma directions
        twist = torch.cat([dpos, drot], dim=-1)  # (N,6)
        twist_safe = self.safety.project_twist_away_from_low_sigma(jac, twist)
        dpos_safe = twist_safe[..., 0:3]
        drot_safe = twist_safe[..., 3:6]

        if self._mode == "rotate":
            if self.config.safety_cfg.min_sigma_thresh is not None:
                sigma_min = self.safety.smallest_singular_value(jac)
                block_sigma = sigma_min < self.config.safety_cfg.min_sigma_thresh
                target_quat = self._ee_quat_last_safe_b if self._ee_quat_last_safe_b is not None else ee_quat_b
                projected_drot = self.safety.project_rotation_toward_quat(ee_quat_b, target_quat, drot_safe)
                drot_safe = torch.where(block_sigma.unsqueeze(-1), projected_drot, drot_safe)
            pos_des = self.safety.clamp_position(ee_pos_b)
            quat_des = quat_box_plus(ee_quat_b, drot_safe)
            # update last safe orientation for potential external consumers
            self._ee_quat_last_safe_b = quat_des.clone()
        elif self._mode == "twist":
            # Simultaneous translation + rotation: compose the translate and
            # rotate branches, inheriting the same safety machinery.
            if self.config.safety_cfg.min_sigma_thresh is not None:
                sigma_min = self.safety.smallest_singular_value(jac)
                block_sigma = sigma_min < self.config.safety_cfg.min_sigma_thresh
                target_quat = self._ee_quat_last_safe_b if self._ee_quat_last_safe_b is not None else ee_quat_b
                projected_drot = self.safety.project_rotation_toward_quat(ee_quat_b, target_quat, drot_safe)
                drot_safe = torch.where(block_sigma.unsqueeze(-1), projected_drot, drot_safe)
            pos_des = self.safety.clamp_position(ee_pos_b + dpos_safe)
            quat_des = quat_box_plus(ee_quat_b, drot_safe)
            self._ee_quat_last_safe_b = quat_des.clone()
        elif self._mode == "translate":
            pos_des = self.safety.clamp_position(ee_pos_b + dpos_safe)
            quat_des = self.safety.hold_orientation(ee_quat_b, self._ee_quat_hold_b, self.config.hold_orientation)
        else:  # gripper
            pos_des = self.safety.clamp_position(ee_pos_b)
            quat_des = ee_quat_b

        # Feed IK
        self._diff_ik.ee_pos_des[:] = pos_des
        self._diff_ik.ee_quat_des[:] = quat_des
        q_des = self._diff_ik.compute(ee_pos_b, ee_quat_b, jac, q_arm)
        qdot_arm = (q_des - q_arm) / dt
        # Clamp joint velocities that would push further into nearby joint limits
        qdot_arm = self.safety.clamp_qdot_near_limits(qdot_arm, q_arm, q_lower, q_upper)

        # Hold all joints at current position target, zero velocity for non-arm joints
        robot.set_joint_position_target(robot.data.joint_pos)
        robot.set_joint_velocity_target(torch.zeros_like(robot.data.joint_vel))
        robot.set_joint_velocity_target(qdot_arm, joint_ids=self._arm_joint_ids)

        # Hold gripper shape each step
        self.gripper.apply_hold(robot)

        # Gripper control in gripper mode using g_val
        if self._mode == "gripper" and g_val is not None:
            if g_val.mean().item() > 0.0:
                self.gripper.command_open(robot)
            else:
                self.gripper.command_close(robot)
        elif self._mode == "twist" and g_val is not None:
            # deadband so g = 0 means "hold current gripper state"
            g_mean = g_val.mean().item()
            if g_mean > 0.5:
                self.gripper.command_open(robot)
            elif g_mean < -0.5:
                self.gripper.command_close(robot)

        # Gravity compensation
        gravity = robot.root_physx_view.get_gravity_compensation_forces()
        robot.set_joint_effort_target(gravity)

        # Optional real-time EE position logging
        if getattr(self.config, "log_ee_pos", False):
            frame = getattr(self.config, "log_ee_frame", "world")
            every = max(1, int(getattr(self.config, "log_every_n_steps", 1)))
            if (self._step_count % every) == 0:
                if frame == "world":
                    pos = ee_pose_w[:, 0:3]
                else:
                    pos = ee_pos_b
                pos_np = pos.detach().to("cpu").numpy().tolist()
                print(f"[EE] frame={frame} xyz(m)={pos_np} mode={self._mode}")
        self._step_count += 1

        # Write to sim
        robot.write_data_to_sim() 