from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from copilot_demo.extractor import ExtractorConfig, InputExtractor


@dataclass
class DummyPose:
    position_m: Tuple[float, float, float]
    orientation_wxyz: Tuple[float, float, float, float]


@dataclass
class DummyObj:
    id: str
    label: str
    pose: DummyPose
    is_held: bool = False


def test_extractor_builds_blob_and_candidates():
    cfg = ExtractorConfig(
        workspace_min_xyz=(0.0, -0.5, 0.0),
        workspace_max_xyz=(0.6, 0.5, 0.3),
        table_z=0.0,
        candidate_max_dist=2,
    )
    extractor = InputExtractor(cfg)
    obj = DummyObj(
        id="o1",
        label="mug",
        pose=DummyPose(position_m=(0.4, 0.0, 0.02), orientation_wxyz=(1.0, 0.0, 0.0, 0.0)),
    )
    blob = extractor.build_input_blob(
        objects_snapshot=[obj],
        ee_pos_w=(0.3, 0.0, 0.05),
        ee_quat_w=(1.0, 0.0, 0.0, 0.0),
        base_pos_w=(0.0, 0.0, 0.0),
        base_quat_w=(1.0, 0.0, 0.0, 0.0),
    )
    assert blob["objects"][0]["cell"] in {"B2", "B3"}
    assert len(blob["gripper_hist"]) == 6
    assert blob["memory"]["candidates"] == ["o1"]

