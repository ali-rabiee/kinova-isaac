"""Extract grasp-copilot input blobs from IsaacSim state."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Sequence, Tuple

import math

from data_generator import grid as gridlib  # type: ignore

from .quantize import pos_to_cell_xy, quat_to_yaw_bin, z_to_bin


def _default_memory() -> Dict[str, Any]:
    return {
        "n_interactions": 0,
        "past_dialogs": [],
        "candidates": [],
        "last_tool_calls": [],
        "excluded_obj_ids": [],
        "last_action": {},
        "user_state": {"mode": "translation"},
    }


@dataclass
class ExtractorConfig:
    workspace_min_xyz: Tuple[float, float, float]
    workspace_max_xyz: Tuple[float, float, float]
    table_z: float
    candidate_max_dist: int = 2
    max_gripper_hist: int = 6

    def workspace_min_xy(self) -> Tuple[float, float]:
        return (self.workspace_min_xyz[0], self.workspace_min_xyz[1])

    def workspace_max_xy(self) -> Tuple[float, float]:
        return (self.workspace_max_xyz[0], self.workspace_max_xyz[1])


def _quat_multiply(q1: Sequence[float], q2: Sequence[float]) -> Tuple[float, float, float, float]:
    """Hamilton product q = q1 * q2 using (w, x, y, z)."""
    w1, x1, y1, z1 = [float(v) for v in q1]
    w2, x2, y2, z2 = [float(v) for v in q2]
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _quat_conjugate(q: Sequence[float]) -> Tuple[float, float, float, float]:
    w, x, y, z = [float(v) for v in q]
    return (w, -x, -y, -z)


def _quat_apply(q: Sequence[float], v: Sequence[float]) -> Tuple[float, float, float]:
    """Rotate vector v by quaternion q (w, x, y, z)."""
    qv = (0.0, float(v[0]), float(v[1]), float(v[2]))
    qi = _quat_conjugate(q)
    res = _quat_multiply(_quat_multiply(q, qv), qi)
    return (res[1], res[2], res[3])


def world_to_base(
    pos_w: Sequence[float],
    quat_w: Sequence[float],
    base_pos_w: Sequence[float],
    base_quat_w: Sequence[float],
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
    """Convert world pose to base frame using standard quaternion math (wxyz)."""
    # Translate into base origin then rotate by inverse(base_quat_w)
    rel = (
        float(pos_w[0]) - float(base_pos_w[0]),
        float(pos_w[1]) - float(base_pos_w[1]),
        float(pos_w[2]) - float(base_pos_w[2]),
    )
    base_inv = _quat_conjugate(base_quat_w)
    pos_b = _quat_apply(base_inv, rel)
    quat_b = _quat_multiply(base_inv, quat_w)
    return pos_b, quat_b


class InputExtractor:
    """Stateful helper that builds the LLM input blob each simulation tick."""

    def __init__(self, cfg: ExtractorConfig):
        self.cfg = cfg
        self.memory: Dict[str, Any] = _default_memory()
        self.gripper_hist: Deque[Dict[str, str]] = deque(maxlen=int(self.cfg.max_gripper_hist))
        self.last_objects_b: List[Dict[str, Any]] = []

    def reset(self) -> None:
        self.memory = _default_memory()
        self.gripper_hist.clear()

    def _push_gripper_pose(self, pos_b: Sequence[float], quat_b: Sequence[float]) -> None:
        cell = pos_to_cell_xy(pos_b[0], pos_b[1], self.cfg.workspace_min_xy(), self.cfg.workspace_max_xy())
        yaw_bin = quat_to_yaw_bin(quat_b)
        z_bin = z_to_bin(pos_b[2], self.cfg.table_z, self.cfg.workspace_min_xyz[2], self.cfg.workspace_max_xyz[2])
        rec = {"cell": cell, "yaw": yaw_bin, "z": z_bin}
        if not self.gripper_hist:
            # Seed history with the initial pose to reach length max_gripper_hist.
            for _ in range(self.cfg.max_gripper_hist):
                self.gripper_hist.append(rec)
        else:
            self.gripper_hist.append(rec)

    def _convert_objects(
        self,
        objs: Iterable[Any],
        base_pos_w: Sequence[float],
        base_quat_w: Sequence[float],
    ) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for obj in objs:
            try:
                pos_w = getattr(obj.pose, "position_m", None) if hasattr(obj, "pose") else None
                quat_w = getattr(obj.pose, "orientation_wxyz", None) if hasattr(obj, "pose") else None
                pos_w = pos_w if pos_w is not None else getattr(obj, "position_m", None)
                quat_w = quat_w if quat_w is not None else getattr(obj, "orientation_wxyz", None)
                if pos_w is None or quat_w is None:
                    continue
                pos_b, quat_b = world_to_base(pos_w, quat_w, base_pos_w, base_quat_w)
                cell = pos_to_cell_xy(pos_b[0], pos_b[1], self.cfg.workspace_min_xy(), self.cfg.workspace_max_xy())
                yaw_bin = quat_to_yaw_bin(quat_b)
                records.append(
                    {
                        "id": getattr(obj, "id", None) or getattr(obj, "name", "obj"),
                        "label": getattr(obj, "label", "object"),
                        "cell": cell,
                        "yaw": yaw_bin,
                        "is_held": bool(getattr(obj, "is_held", False)),
                        "_pos_b": pos_b,
                        "_quat_b": quat_b,
                    }
                )
            except Exception:
                continue
        return records

    def _update_candidates(self, objects: List[Dict[str, Any]]) -> List[str]:
        if not self.gripper_hist:
            return []
        cur_cell = self.gripper_hist[-1]["cell"]
        excluded = set(self.memory.get("excluded_obj_ids") or [])
        cands: List[str] = []
        for o in objects:
            if o.get("is_held"):
                continue
            oid = str(o.get("id"))
            if oid in excluded:
                continue
            if gridlib.manhattan(cur_cell, o["cell"]) <= int(self.cfg.candidate_max_dist):
                cands.append(oid)
        self.memory["candidates"] = cands
        return cands

    def build_input_blob(
        self,
        *,
        objects_snapshot: Iterable[Any],
        ee_pos_w: Sequence[float],
        ee_quat_w: Sequence[float],
        base_pos_w: Sequence[float],
        base_quat_w: Sequence[float],
        user_state: Dict[str, str] | None = None,
    ) -> Dict[str, Any]:
        """Convert raw world poses into the JSON-serializable blob expected by the policy."""
        pos_b, quat_b = world_to_base(ee_pos_w, ee_quat_w, base_pos_w, base_quat_w)
        self._push_gripper_pose(pos_b, quat_b)
        objects_b = self._convert_objects(objects_snapshot, base_pos_w, base_quat_w)
        self.last_objects_b = objects_b
        self._update_candidates(objects_b)

        if user_state is not None:
            self.memory["user_state"] = {"mode": str(user_state.get("mode", "translation"))}
        else:
            self.memory.setdefault("user_state", {"mode": "translation"})

        # Strip helper fields before returning.
        objs_public = [
            {k: v for k, v in o.items() if not k.startswith("_")}
            for o in objects_b
        ]

        return {
            "objects": objs_public,
            "gripper_hist": list(self.gripper_hist),
            "memory": self.memory,
            "user_state": self.memory.get("user_state", {"mode": "translation"}),
        }

    # Convenience helpers to work directly with the robot object (torch tensors).
    def build_from_robot(
        self,
        robot,
        ee_body_id: int,
        objects_snapshot: Iterable[Any],
        user_state: Dict[str, str] | None = None,
    ) -> Dict[str, Any]:
        """Read robot root + EE pose from the IsaacLab robot and delegate to build_input_blob."""
        import torch

        root_pose = robot.data.root_pose_w
        base_pos_w = root_pose[0, 0:3].tolist() if isinstance(root_pose, torch.Tensor) else list(root_pose[0][0:3])
        base_quat_w = root_pose[0, 3:7].tolist() if isinstance(root_pose, torch.Tensor) else list(root_pose[0][3:7])

        ee_pose_w = robot.data.body_pose_w[:, ee_body_id]
        ee_pos_w = ee_pose_w[0, 0:3].tolist() if isinstance(ee_pose_w, torch.Tensor) else list(ee_pose_w[0][0:3])
        ee_quat_w = ee_pose_w[0, 3:7].tolist() if isinstance(ee_pose_w, torch.Tensor) else list(ee_pose_w[0][3:7])

        return self.build_input_blob(
            objects_snapshot=objects_snapshot,
            ee_pos_w=ee_pos_w,
            ee_quat_w=ee_quat_w,
            base_pos_w=base_pos_w,
            base_quat_w=base_quat_w,
            user_state=user_state,
        )
