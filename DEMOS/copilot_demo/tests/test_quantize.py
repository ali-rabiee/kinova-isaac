from __future__ import annotations

import math

from copilot_demo.quantize import pos_to_cell_xy, quat_to_yaw_bin, z_to_bin


def test_pos_to_cell_xy_center():
    cell = pos_to_cell_xy(0.5, 0.0, (0.0, -0.5), (1.0, 0.5))
    assert cell == "B2"


def test_pos_to_cell_xy_clamp_edges():
    assert pos_to_cell_xy(-1.0, -1.0, (0.0, 0.0), (1.0, 1.0)) == "A1"
    assert pos_to_cell_xy(2.0, 2.0, (0.0, 0.0), (1.0, 1.0)) == "C3"


def test_quat_to_yaw_bin():
    # Identity -> +X
    assert quat_to_yaw_bin((1.0, 0.0, 0.0, 0.0)) == "N"
    # 90 deg CCW around Z -> +Y (E)
    q = (math.cos(math.pi / 4.0), 0.0, 0.0, math.sin(math.pi / 4.0))
    assert quat_to_yaw_bin(q) == "E"


def test_z_to_bin():
    assert z_to_bin(0.0, 0.0, 0.0, 0.3) == "LOW"
    assert z_to_bin(0.1, 0.0, 0.0, 0.3) == "MID"
    assert z_to_bin(0.26, 0.0, 0.0, 0.3) == "HIGH"

