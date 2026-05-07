from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from controllers.base import InputProvider


class WaypointFollowerInput(InputProvider):
	"""Programmatic input that drives the EE toward a sequence of waypoints (base frame).

	- Produces per-step Cartesian deltas [dx, dy, dz, 0, 0, 0, g].
	- Step magnitudes are bounded by `step_pos_m` per step.
	- Supports queued gripper commands by returning [0..0, g] for a fixed number of steps.
	- The runner must update the current EE pose each step via `set_current_pose_b`.
	"""

	def __init__(
		self,
		*,
		step_pos_m: float,
		tol_m: float = 0.005,
		max_steps_per_waypoint: int = 600,
		stagnation_steps: int = 60,
		stagnation_eps_m: float = 5e-4,
		device: str = "cpu",
	) -> None:
		self.step_pos_m = float(step_pos_m)
		self.tol_m = float(tol_m)
		self.max_steps_per_waypoint = max(1, int(max_steps_per_waypoint))
		self.stagnation_steps = max(1, int(stagnation_steps))
		self.stagnation_eps_m = float(stagnation_eps_m)
		self.device = torch.device(device)
		self._waypoints_b: List[torch.Tensor] = []
		self._current_ee_pos_b: Optional[torch.Tensor] = None
		self._gripper_steps_left: int = 0
		self._gripper_value: float = 0.0
		self._last_cmd: Optional[torch.Tensor] = None
		self._rot_steps_left: int = 0
		self._rot_vec_b: torch.Tensor = torch.zeros(3, dtype=torch.float32, device=self.device)
		# Waypoint progress watchdog (prevents "hover forever" if controller can't converge exactly)
		self._wp_steps_on_current: int = 0
		self._wp_last_dist_m: Optional[float] = None
		self._wp_stagnant_steps: int = 0

	def reset(self) -> None:
		self._waypoints_b = []
		self._current_ee_pos_b = None
		self._gripper_steps_left = 0
		self._gripper_value = 0.0
		self._last_cmd = None
		self._wp_steps_on_current = 0
		self._wp_last_dist_m = None
		self._wp_stagnant_steps = 0

	def set_tolerance_m(self, tol_m: float) -> None:
		"""Change position convergence radius (meters) for subsequent waypoints."""
		self.tol_m = float(tol_m)

	def set_max_steps_per_waypoint(self, n: int) -> None:
		"""Change per-waypoint watchdog step budget."""
		self.max_steps_per_waypoint = max(1, int(n))

	def set_current_pose_b(self, ee_pos_b: torch.Tensor) -> None:
		"""Update the current EE position in base frame: shape (3,) or (1,3)."""
		if ee_pos_b.ndim == 2:
			ee_pos_b = ee_pos_b.view(-1)
		self._current_ee_pos_b = ee_pos_b.to(self.device)

	def set_waypoints_b(self, points_b: List[Tuple[float, float, float]]) -> None:
		self._waypoints_b = [torch.tensor(p, dtype=torch.float32, device=self.device) for p in points_b]
		self._wp_steps_on_current = 0
		self._wp_last_dist_m = None
		self._wp_stagnant_steps = 0

	def queue_gripper(self, g_value: float, steps: int) -> None:
		self._gripper_value = float(g_value)
		self._gripper_steps_left = max(0, int(steps))

	def queue_rotate_z(self, total_angle_rad: float, steps: int) -> None:
		"""Queue a pure yaw rotation around base Z spread across 'steps' increments."""
		steps = max(1, int(steps))
		self._rot_steps_left = steps
		increment = float(total_angle_rad) / float(steps)
		self._rot_vec_b = torch.tensor([0.0, 0.0, increment], dtype=torch.float32, device=self.device)

	def queue_rotate(self, rx: float, ry: float, rz: float, steps: int) -> None:
		"""Queue an arbitrary rotation vector in base frame spread across 'steps' increments."""
		steps = max(1, int(steps))
		self._rot_steps_left = steps
		self._rot_vec_b = torch.tensor(
			[float(rx) / steps, float(ry) / steps, float(rz) / steps],
			dtype=torch.float32,
			device=self.device,
		)

	def advance(self) -> torch.Tensor:
		# Priority: gripper commands
		if self._gripper_steps_left > 0:
			self._gripper_steps_left -= 1
			cmd = torch.zeros(1, 7, dtype=torch.float32, device=self.device)
			cmd[0, 6] = float(self._gripper_value)
			self._last_cmd = cmd
			return cmd

		# Second priority: queued rotation (used with controller in 'rotate' mode)
		if self._rot_steps_left > 0:
			self._rot_steps_left -= 1
			cmd6 = torch.zeros(1, 6, dtype=torch.float32, device=self.device)
			cmd6[0, 3:6] = self._rot_vec_b
			self._last_cmd = cmd6
			return cmd6

		if self._current_ee_pos_b is None or len(self._waypoints_b) == 0:
			cmd = torch.zeros(1, 6, dtype=torch.float32, device=self.device)
			self._last_cmd = cmd
			return cmd

		goal = self._waypoints_b[0]
		diff = goal - self._current_ee_pos_b
		dist = torch.linalg.norm(diff).item()
		self._wp_steps_on_current += 1

		# Watchdog: detect stagnation (distance not improving) to avoid being stuck forever.
		if self._wp_last_dist_m is not None:
			if dist > (self._wp_last_dist_m - float(self.stagnation_eps_m)):
				self._wp_stagnant_steps += 1
			else:
				self._wp_stagnant_steps = 0
		self._wp_last_dist_m = float(dist)

		if dist <= self.tol_m:
			# Reached; pop and output zero
			self._waypoints_b.pop(0)
			self._wp_steps_on_current = 0
			self._wp_last_dist_m = None
			self._wp_stagnant_steps = 0
			cmd = torch.zeros(1, 6, dtype=torch.float32, device=self.device)
			self._last_cmd = cmd
			return cmd

		# If we spend too long on a waypoint or stagnate, give up and pop it.
		if self._wp_steps_on_current >= self.max_steps_per_waypoint or self._wp_stagnant_steps >= self.stagnation_steps:
			self._waypoints_b.pop(0)
			self._wp_steps_on_current = 0
			self._wp_last_dist_m = None
			self._wp_stagnant_steps = 0
			cmd = torch.zeros(1, 6, dtype=torch.float32, device=self.device)
			self._last_cmd = cmd
			return cmd

		step = self.step_pos_m
		d = (diff / (dist + 1e-9)) * min(step, dist)
		cmd6 = torch.zeros(1, 6, dtype=torch.float32, device=self.device)
		cmd6[0, 0:3] = d
		self._last_cmd = cmd6
		return cmd6

	@property
	def last_cmd(self) -> Optional[torch.Tensor]:
		return self._last_cmd


