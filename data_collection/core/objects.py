from __future__ import annotations

import importlib
from typing import Dict, List, Optional, Tuple

from .schemas import DetectedObject, Pose


class ObjectsTracker:
    """Lightweight object tracker over spawned prims.

    For v0, we read world transforms from USD stage at each snapshot.
    If USD access fails, returns an empty list.
    """

    def __init__(self, prim_paths: Optional[List[str]] = None) -> None:
        self.prim_paths = prim_paths or []
        # Cache: root path -> (rigid_body_root_path, physx_view)
        self._rigid_root_map: Dict[str, Tuple[str, object]] = {}

    def _ensure_rigid_view(self, root_path: str) -> Optional[Tuple[str, object]]:
        if root_path in self._rigid_root_map:
            return self._rigid_root_map[root_path]
        # Find a rigid body prim under the root using USD Physics API
        UsdPhysics = importlib.import_module("pxr.UsdPhysics")
        omni_usd = importlib.import_module("omni.usd")
        sim_utils = importlib.import_module("isaaclab.sim")
        stage = omni_usd.get_context().get_stage()  # type: ignore[attr-defined]
        root_prim = stage.GetPrimAtPath(root_path)
        if not root_prim.IsValid():
            return None
        # Common case (especially for simple shape spawners): the rigid body is the root prim itself.
        # If we don't handle this, we fall back to USD XformCache which may not reflect dynamic motion reliably.
        try:
            if root_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rigid_path = str(root_path)
                SimulationManager = importlib.import_module("isaacsim.core.simulation_manager").SimulationManager
                physx_view = SimulationManager.get_physics_sim_view().create_rigid_body_view(rigid_path)
                self._rigid_root_map[root_path] = (rigid_path, physx_view)
                return self._rigid_root_map[root_path]
        except Exception:
            pass
        # Traverse children and find first prim with RigidBodyAPI applied
        get_all_matching_child_prims = getattr(sim_utils, "get_all_matching_child_prims")
        rigid_prims = get_all_matching_child_prims(
            root_path,
            predicate=lambda p: p.HasAPI(UsdPhysics.RigidBodyAPI),
            traverse_instance_prims=False,
        )
        if not rigid_prims:
            return None
        rigid_root = rigid_prims[0]
        rigid_path = rigid_root.GetPath().pathString
        # Create a PhysX rigid body view for this prim path expression
        SimulationManager = importlib.import_module("isaacsim.core.simulation_manager").SimulationManager
        physx_view = SimulationManager.get_physics_sim_view().create_rigid_body_view(rigid_path)
        self._rigid_root_map[root_path] = (rigid_path, physx_view)
        return self._rigid_root_map[root_path]

    def _read_prim_pose(
        self, prim_path: str
    ) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]]:
        # Prefer PhysX transforms for dynamic objects
        try:
            rv = self._ensure_rigid_view(prim_path)
            if rv is not None:
                _, physx_view = rv
                # get_transforms returns [x,y,z,qx,qy,qz,qw]
                pose = getattr(physx_view, "get_transforms")().clone()[0]
                px, py, pz = float(pose[0].item()), float(pose[1].item()), float(pose[2].item())
                qx, qy, qz, qw = (
                    float(pose[3].item()),
                    float(pose[4].item()),
                    float(pose[5].item()),
                    float(pose[6].item()),
                )
                return (px, py, pz), (qw, qx, qy, qz)
        except Exception:
            pass
        # Fallback to USD XformCache (may not reflect dynamic motion)
        UsdGeom = importlib.import_module("pxr.UsdGeom")
        omni_usd = importlib.import_module("omni.usd")
        stage = omni_usd.get_context().get_stage()  # type: ignore[attr-defined]
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return None
        xf_cache = UsdGeom.XformCache()
        mat = xf_cache.GetLocalToWorldTransform(prim)
        trans = mat.ExtractTranslation()
        rotation = mat.ExtractRotation()
        quat = rotation.GetQuat()
        px, py, pz = float(trans[0]), float(trans[1]), float(trans[2])
        w = float(quat.GetReal())
        x, y, z = quat.GetImaginary()
        return (px, py, pz), (w, float(x), float(y), float(z))

    def snapshot(self) -> List[DetectedObject]:
        objects: List[DetectedObject] = []
        for path in self.prim_paths:
            pose = self._read_prim_pose(path)
            if pose is None:
                continue
            pos, ori = pose
            obj = DetectedObject(
                id=path.split("/")[-1],
                label="object",
                color=None,
                pose=Pose(position_m=pos, orientation_wxyz=ori),
                bbox_xywh=None,
                confidence=1.0,
            )
            objects.append(obj)
        return objects


