"""Quantization helpers for mapping continuous poses to the discrete bins expected by grasp-copilot."""

from __future__ import annotations

import math
from typing import Sequence, Tuple

from data_generator import grid as gridlib  # type: ignore
from data_generator import yaw as yawlib  # type: ignore

WorkspaceBounds = Tuple[float, float]


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def pos_to_cell_xy(
    x_b: float,
    y_b: float,
    workspace_min_xy: Sequence[float],
    workspace_max_xy: Sequence[float],
) -> str:
    """Quantize a base-frame (x, y) position into a 3x3 grid cell label (A1..C3).

    - Columns 1..3 span x from workspace_min_xy[0] (near) to workspace_max_xy[0] (far).
    - Rows A..C span y from workspace_min_xy[1] (left) to workspace_max_xy[1] (right).
    Values are clamped to the workspace extents.
    """
    if len(workspace_min_xy) < 2 or len(workspace_max_xy) < 2:
        raise ValueError("workspace_min_xy and workspace_max_xy must have length >=2")
    x_min, y_min = float(workspace_min_xy[0]), float(workspace_min_xy[1])
    x_max, y_max = float(workspace_max_xy[0]), float(workspace_max_xy[1])
    if not (x_max > x_min and y_max > y_min):
        raise ValueError(f"Invalid workspace bounds: min={workspace_min_xy}, max={workspace_max_xy}")

    nx = _clamp01((float(x_b) - x_min) / (x_max - x_min))
    ny = _clamp01((float(y_b) - y_min) / (y_max - y_min))
    c_idx = min(2, int(nx * 3.0))  # 0..2
    r_idx = min(2, int(ny * 3.0))  # 0..2
    return gridlib.Cell(r_idx, c_idx).to_label()


def _wrap_to_pi(rad: float) -> float:
    """Wrap angle to [-pi, pi)."""
    return (rad + math.pi) % (2.0 * math.pi) - math.pi


def quat_to_yaw_bin(quat_wxyz: Sequence[float]) -> str:
    """Quantize quaternion (w, x, y, z) into one of 8 yaw bins defined in yawlib.YAW_BINS.

    Convention:
    - Yaw = 0 means gripper/object faces +X in base frame -> bin "N".
    - Yaw increases counter-clockwise around +Z.
    - Bins are spaced every 45 degrees following yawlib.YAW_BINS ordering: N, NE, E, SE, S, SW, W, NW.
    """
    if len(quat_wxyz) != 4:
        raise ValueError(f"quat_wxyz must have length 4, got {quat_wxyz}")
    w, x, y, z = [float(q) for q in quat_wxyz]
    # Standard yaw extraction from quaternion (Z-up, XYZ convention)
    yaw_rad = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    yaw_rad = _wrap_to_pi(yaw_rad)
    step = math.pi / 4.0  # 45 deg
    idx = int(round(yaw_rad / step)) % len(yawlib.YAW_BINS)
    return yawlib.YAW_BINS[idx]


def z_to_bin(
    z_b: float,
    table_z: float,
    workspace_zmin: float,
    workspace_zmax: float,
) -> str:
    """Quantize base-frame z into LOW/MID/HIGH.

    - The lower boundary is anchored near the table height to make LOW correspond to table-adjacent poses.
    - The upper boundary is placed at 2/3 of the workspace range to reserve HIGH for hovering.
    """
    zmin = float(workspace_zmin)
    zmax = float(workspace_zmax)
    if not zmax > zmin:
        raise ValueError(f"workspace_zmax must be > workspace_zmin (got {zmax} <= {zmin})")
    span = zmax - zmin
    mid_high = zmin + (2.0 * span / 3.0)
    # Anchor LOW/MID split slightly above the table; clamp into workspace.
    low_mid = max(zmin, min(mid_high, float(table_z) + 0.02))

    z_val = float(z_b)
    if z_val <= low_mid:
        return "LOW"
    if z_val <= mid_high:
        return "MID"
    return "HIGH"
