from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def _ensure_repo_on_syspath() -> Path:
    """
    This file lives at <repo-root>/grasp-vla/rollout_xvla_isaac.py.
    Make sure <repo-root> is on sys.path so we can import controllers/environments/etc.
    """
    root = Path(__file__).resolve().parents[1]
    root_str = str(root)
    if root_str in sys.path:
        sys.path.remove(root_str)
    sys.path.insert(0, root_str)

    # Isaac/Omni may preload a non-package module named `environments`, which breaks our imports.
    _env_mod = sys.modules.get("environments")
    if _env_mod is not None and not hasattr(_env_mod, "__path__"):
        del sys.modules["environments"]

    # Isaac may preload `cv2.utils` which can shadow this repo's `utils/` package.
    _utils_mod = sys.modules.get("utils")
    if _utils_mod is not None:
        _utils_file = str(getattr(_utils_mod, "__file__", "") or "")
        if _utils_file and root_str not in _utils_file:
            for _k in list(sys.modules.keys()):
                if _k == "utils" or _k.startswith("utils."):
                    del sys.modules[_k]

    return root


def _ensure_lerobot_on_syspath(repo_root: Path, lerobot_src: str | None = None) -> Path | None:
    """
    Isaac/Kit can mutate sys.path; ensure LeRobot is importable even inside the Kit runtime.

    Priority:
    - CLI arg `--lerobot-src`
    - env var `LEROBOT_SRC`
    - common local checkout locations relative to this repo
    """
    candidates: list[Path] = []

    if lerobot_src:
        candidates.append(Path(lerobot_src).expanduser().resolve())
    env_src = os.environ.get("LEROBOT_SRC")
    if env_src:
        candidates.append(Path(env_src).expanduser().resolve())

    # Common layouts:
    # - <repo-root>/grasp-vla/lerobot/src
    # - <repo-root>/../Grasp-VLA/lerobot/src   (your current setup)
    candidates.extend(
        [
            (repo_root / "grasp-vla" / "lerobot" / "src").resolve(),
            (repo_root.parent / "Grasp-VLA" / "lerobot" / "src").resolve(),
            (repo_root.parent / "grasp-vla" / "lerobot" / "src").resolve(),
            (repo_root.parent / "lerobot" / "src").resolve(),
        ]
    )

    def looks_like_lerobot_src(p: Path) -> bool:
        return (p / "lerobot" / "__init__.py").exists()

    for p in candidates:
        if looks_like_lerobot_src(p):
            p_str = str(p)
            if p_str not in sys.path:
                sys.path.insert(0, p_str)
                print(f"[xVLA] Added LeRobot to sys.path: {p_str}", flush=True)
            return p

    return None


def _stub_pkg(fullname: str, pkg_dir: Path) -> None:
    """Insert a lightweight package stub into sys.modules to avoid executing __init__.py side effects."""
    if fullname in sys.modules:
        return
    if not pkg_dir.is_dir():
        return

    import types

    m = types.ModuleType(fullname)
    m.__path__ = [str(pkg_dir)]
    m.__package__ = fullname
    sys.modules[fullname] = m

    parent_name, _, attr = fullname.rpartition(".")
    if parent_name and parent_name in sys.modules:
        try:
            setattr(sys.modules[parent_name], attr, m)
        except Exception:
            pass


def _stub_module(fullname: str, *, attrs: dict[str, object] | None = None) -> None:
    """Insert a lightweight module stub into sys.modules (used to bypass optional training-time deps)."""
    if fullname in sys.modules:
        return

    import types

    m = types.ModuleType(fullname)
    if attrs:
        for k, v in attrs.items():
            setattr(m, k, v)
    sys.modules[fullname] = m

    parent_name, _, attr = fullname.rpartition(".")
    if parent_name and parent_name in sys.modules:
        try:
            setattr(sys.modules[parent_name], attr, m)
        except Exception:
            pass


def _bypass_lerobot_policy_inits(lerobot_src: Path, *, verbose: bool = False) -> None:
    """
    LeRobot's `lerobot.policies` and `lerobot.policies.xvla` package `__init__.py` files pull in a lot of
    optional training-time dependencies (e.g. `diffusers`, `pyserial`) that are unnecessary for this rollout.

    To keep the IsaacSim runtime environment light and deterministic, we stub those packages as namespace
    packages so we can import *only* the specific xvla modules we need:
      - `lerobot.policies.xvla.configuration_xvla`
      - `lerobot.policies.xvla.modeling_xvla`
    """
    policies_dir = (lerobot_src / "lerobot" / "policies").resolve()
    xvla_dir = (policies_dir / "xvla").resolve()
    processor_dir = (lerobot_src / "lerobot" / "processor").resolve()

    # Import the lightweight parent package so we can attach the stub as an attribute (helpful for tooling).
    try:
        import lerobot  # noqa: F401
    except Exception:
        return

    _stub_pkg("lerobot.policies", policies_dir)
    _stub_pkg("lerobot.policies.xvla", xvla_dir)
    _stub_pkg("lerobot.processor", processor_dir)

    # The inference stack doesn't need training configs or dataset objects, but they are imported
    # transitively in upstream LeRobot. Stub them to avoid optional deps like PyAV.
    _stub_module("lerobot.configs.train", attrs={"TrainPipelineConfig": type("TrainPipelineConfig", (), {})})
    _stub_module("lerobot.datasets.lerobot_dataset", attrs={"LeRobotDataset": type("LeRobotDataset", (), {})})

    # Make `from lerobot.processor import PolicyAction, RobotAction, RobotObservation` work even with the
    # package stub (used by lerobot.policies.utils).
    try:
        import importlib

        core = importlib.import_module("lerobot.processor.core")
        proc_pkg = sys.modules.get("lerobot.processor")
        if proc_pkg is not None:
            for name in ("PolicyAction", "RobotAction", "RobotObservation"):
                if hasattr(core, name):
                    setattr(proc_pkg, name, getattr(core, name))
    except Exception:
        pass

    if verbose:
        print("[xVLA] Stubbed LeRobot packages to avoid optional-heavy imports (diffusers/av/training stack).")
        sys.stdout.flush()


def _register_minimal_xvla_processor_steps(*, verbose: bool = False) -> None:
    """
    The saved `policy_preprocessor.json` references xvla-specific processor steps by registry name:
      - xvla_add_domain_id
      - xvla_image_to_float
      - xvla_imagenet_normalize

    The upstream `lerobot.policies.xvla` package registers them by importing `processor_xvla.py`, but that
    file pulls in training stack dependencies (e.g. robot drivers). For this rollout we only need the simple
    image + domain-id steps, so we re-register tiny equivalents here.
    """
    from lerobot.processor.core import EnvTransition, TransitionKey
    from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry

    existing = set(ProcessorStepRegistry.list())
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    if "xvla_add_domain_id" not in existing:

        @dataclass
        @ProcessorStepRegistry.register(name="xvla_add_domain_id")
        class _XVLAAddDomainIdProcessorStep(ProcessorStep):
            domain_id: int = 0

            def __call__(self, transition: EnvTransition) -> EnvTransition:
                new_transition = transition.copy()
                comp = new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
                comp = {} if comp is None else comp.copy()

                obs = new_transition.get(TransitionKey.OBSERVATION, {}) or {}
                batch_size = 1
                for v in obs.values():
                    if isinstance(v, torch.Tensor):
                        batch_size = int(v.shape[0])
                        break

                comp["domain_id"] = torch.tensor([int(self.domain_id)] * batch_size, dtype=torch.long)
                new_transition[TransitionKey.COMPLEMENTARY_DATA] = comp
                return new_transition

            def transform_features(self, features):  # noqa: ANN001
                return features

    if "xvla_image_to_float" not in existing:

        @dataclass
        @ProcessorStepRegistry.register(name="xvla_image_to_float")
        class _XVLAImageToFloatProcessorStep(ProcessorStep):
            image_keys: list[str] | None = None
            validate_range: bool = True

            def __call__(self, transition: EnvTransition) -> EnvTransition:
                new_transition = transition.copy()
                obs = new_transition.get(TransitionKey.OBSERVATION, {})
                if obs is None:
                    return new_transition

                obs = obs.copy()
                keys_to_convert = self.image_keys
                if keys_to_convert is None:
                    keys_to_convert = [k for k in obs if k.startswith("observation.images.")]

                for key in keys_to_convert:
                    if key in obs and isinstance(obs[key], torch.Tensor):
                        tensor = obs[key]
                        min_val = float(tensor.min().item())
                        max_val = float(tensor.max().item())

                        # Already in [0, 1] range: keep scale, just ensure float dtype.
                        if max_val <= 1.0:
                            obs[key] = tensor.float()
                            continue

                        if self.validate_range and (min_val < 0.0 or max_val > 255.0):
                            raise ValueError(
                                f"Image '{key}' has values outside [0, 255] range: "
                                f"min={min_val:.4f}, max={max_val:.4f}."
                            )
                        obs[key] = tensor.float() / 255.0

                new_transition[TransitionKey.OBSERVATION] = obs
                return new_transition

            def transform_features(self, features):  # noqa: ANN001
                return features

    if "xvla_imagenet_normalize" not in existing:

        @dataclass
        @ProcessorStepRegistry.register(name="xvla_imagenet_normalize")
        class _XVLAImageNetNormalizeProcessorStep(ProcessorStep):
            image_keys: list[str] | None = None

            def __call__(self, transition: EnvTransition) -> EnvTransition:
                new_transition = transition.copy()
                obs = new_transition.get(TransitionKey.OBSERVATION, {})
                if obs is None:
                    return new_transition

                obs = obs.copy()
                keys_to_norm = self.image_keys
                if keys_to_norm is None:
                    keys_to_norm = [k for k in obs if k.startswith("observation.images.")]

                def _infer_channel_axis(t: torch.Tensor) -> int:
                    # Common cases:
                    # - BCHW: (B, 3, H, W)   -> c_axis = 1
                    # - BHWC: (B, H, W, 3)   -> c_axis = -1
                    # - CHW:  (3, H, W)      -> c_axis = 0
                    # - HWC:  (H, W, 3)      -> c_axis = -1
                    if t.ndim == 4:
                        if t.shape[1] == 3:
                            return 1
                        if t.shape[-1] == 3:
                            return -1
                    if t.ndim == 3:
                        if t.shape[0] == 3 and t.shape[-1] != 3:
                            return 0
                        if t.shape[-1] == 3:
                            return -1
                    # Fallback: assume channels-first if plausible, else channels-last.
                    if t.ndim >= 2 and t.shape[0] == 3:
                        return 0
                    return -1

                for key in keys_to_norm:
                    if key in obs and isinstance(obs[key], torch.Tensor):
                        tensor = obs[key]
                        min_val = float(tensor.min().item())
                        max_val = float(tensor.max().item())
                        if min_val < 0.0 or max_val > 1.0:
                            raise ValueError(
                                f"Image '{key}' has values outside [0, 1] range: "
                                f"min={min_val:.4f}, max={max_val:.4f}."
                            )

                        c_axis = _infer_channel_axis(tensor)
                        mean = torch.tensor(imagenet_mean, device=tensor.device, dtype=tensor.dtype)
                        std = torch.tensor(imagenet_std, device=tensor.device, dtype=tensor.dtype)
                        shape = [1] * tensor.ndim
                        shape[c_axis] = 3
                        mean = mean.view(*shape)
                        std = std.view(*shape)
                        obs[key] = (tensor - mean) / std

                new_transition[TransitionKey.OBSERVATION] = obs
                return new_transition

            def transform_features(self, features):  # noqa: ANN001
                return features

    if verbose:
        print("[xVLA] Registered minimal xvla processor steps for PolicyProcessorPipeline.")
        sys.stdout.flush()


class _ConstantCmdInput:
    """Simple controller input provider: returns the last command tensor you set."""

    def __init__(self, device: str):
        self.device = torch.device(device)
        # Keep 7D always: [dx, dy, dz, rx, ry, rz, g]
        self._cmd = torch.zeros(1, 7, dtype=torch.float32, device=self.device)

    def reset(self) -> None:
        self._cmd = torch.zeros(1, 7, dtype=torch.float32, device=self.device)

    def set(self, cmd: torch.Tensor) -> None:
        if cmd.ndim == 1:
            cmd = cmd.view(1, -1)
        self._cmd = cmd.to(self.device, dtype=torch.float32)

    def advance(self) -> torch.Tensor:
        return self._cmd


def _spawn_colored_boxes(
    *,
    parent_prim_path: str,
    num_boxes: int,
    spawn_min: Tuple[float, float, float],
    spawn_max: Tuple[float, float, float],
    min_dist: float,
    size_m: float,
    device: str,
    fixed_poses_w: Optional[List[Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]]] = None,
    id_to_label_override: Optional[Dict[str, str]] = None,
) -> Tuple[List[str], Dict[str, str]]:
    """
    Spawn boxes under parent_prim_path with deterministic IDs Obj_01..Obj_N
    and labels like "red box 1", matching the log dataset style.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
    from isaaclab.sim.spawners.materials.visual_materials_cfg import PreviewSurfaceCfg
    from isaaclab.sim.spawners.shapes.shapes_cfg import CuboidCfg

    import importlib

    prim_utils = importlib.import_module("isaacsim.core.utils.prims")
    prim_utils.create_prim(parent_prim_path, "Xform")

    # Match the dataset convention (see raw_data instruction.json meta + ticks.jsonl labels):
    #   Obj_01 -> red box 1
    #   Obj_02 -> blue box 2
    #   Obj_03 -> yellow box 3
    #   Obj_04 -> purple box 4
    colors = [
        ("red", (0.85, 0.15, 0.15)),
        ("blue", (0.20, 0.35, 0.95)),
        ("yellow", (0.90, 0.85, 0.20)),
        ("purple", (0.65, 0.25, 0.85)),
    ]

    def sample_positions(n: int) -> List[Tuple[float, float, float]]:
        out: List[Tuple[float, float, float]] = []
        tries = 0
        while len(out) < n and tries < 4000:
            tries += 1
            x = float(np.random.uniform(spawn_min[0], spawn_max[0]))
            y = float(np.random.uniform(spawn_min[1], spawn_max[1]))
            z = float(np.random.uniform(spawn_min[2], spawn_max[2]))
            cand = (x, y, z)
            ok = True
            for p in out:
                dx, dy, dz = cand[0] - p[0], cand[1] - p[1], cand[2] - p[2]
                if (dx * dx + dy * dy + dz * dz) ** 0.5 < float(min_dist):
                    ok = False
                    break
            if ok:
                out.append(cand)
        if len(out) < n:
            # Fallback: allow overlaps rather than failing.
            out = [
                (
                    float(np.random.uniform(spawn_min[0], spawn_max[0])),
                    float(np.random.uniform(spawn_min[1], spawn_max[1])),
                    float(np.random.uniform(spawn_min[2], spawn_max[2])),
                )
                for _ in range(n)
            ]
        return out

    if fixed_poses_w is not None:
        if len(fixed_poses_w) < int(num_boxes):
            raise ValueError(f"fixed_poses_w must have >= num_boxes entries, got {len(fixed_poses_w)}")
        positions = [tuple(float(x) for x in fixed_poses_w[i][0]) for i in range(int(num_boxes))]
        orientations = [tuple(float(x) for x in fixed_poses_w[i][1]) for i in range(int(num_boxes))]
    else:
        positions = sample_positions(int(num_boxes))
        orientations = []
    prim_paths: List[str] = []
    id_to_label: Dict[str, str] = {} if id_to_label_override is None else dict(id_to_label_override)

    for i in range(1, int(num_boxes) + 1):
        leaf = f"Obj_{i:02d}"
        prim = f"{parent_prim_path.rstrip('/')}/{leaf}"
        color_name, rgb = colors[(i - 1) % len(colors)]
        label = f"{color_name} box {i}"

        if fixed_poses_w is not None:
            quat_wxyz = orientations[i - 1]
        else:
            # Random yaw for variety
            yaw = float(np.random.uniform(-np.pi, np.pi))
            half = 0.5 * yaw
            quat_wxyz = (float(np.cos(half)), 0.0, 0.0, float(np.sin(half)))

        cfg = CuboidCfg(
            size=(float(size_m), float(size_m), float(size_m)),
            visual_material=PreviewSurfaceCfg(diffuse_color=rgb),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.10),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.005,
                rest_offset=0.0015,
            ),
        )
        cfg.func(prim, cfg, translation=positions[i - 1], orientation=quat_wxyz)

        # Friction material
        mat_cfg = RigidBodyMaterialCfg(
            static_friction=3.0,
            dynamic_friction=3.0,
            restitution=0.0,
            friction_combine_mode="max",
        )
        mat_prim = f"{prim}/ObjFrictionMaterial"
        mat_cfg.func(mat_prim, mat_cfg)
        sim_utils.bind_physics_material(prim, mat_prim)

        prim_paths.append(prim)
        if leaf not in id_to_label:
            id_to_label[leaf] = label

    return prim_paths, id_to_label


def _load_fixed_layout_from_raw_episode(
    raw_episode: Path, *, num_boxes: int
) -> Tuple[List[Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]], Dict[str, str]]:
    """
    Load a fixed box layout from a raw_data episode folder (or a direct ticks.jsonl file path).

    We read the FIRST tick of ticks.jsonl and use each object's `pose_w` (world-frame pose).
    This is the easiest way to evaluate a trained policy on an in-distribution layout from your dataset.
    """
    raw_episode = raw_episode.expanduser().resolve()
    ticks_path = raw_episode if raw_episode.is_file() else (raw_episode / "ticks.jsonl")
    if not ticks_path.exists():
        raise FileNotFoundError(f"Could not find ticks.jsonl at: {ticks_path}")

    with ticks_path.open("r", encoding="utf-8") as f:
        line = f.readline()
    tick0 = json.loads(line)
    objs = tick0.get("objects") or []
    if not isinstance(objs, list) or not objs:
        raise RuntimeError(f"No objects found in first tick of: {ticks_path}")

    obj_by_id: Dict[str, dict] = {}
    for o in objs:
        if not isinstance(o, dict):
            continue
        oid = o.get("id")
        if isinstance(oid, str) and oid:
            obj_by_id[oid] = o

    fixed: List[Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]] = []
    id_to_label: Dict[str, str] = {}

    for i in range(1, int(num_boxes) + 1):
        leaf = f"Obj_{i:02d}"
        o = obj_by_id.get(leaf)
        if o is None:
            raise RuntimeError(
                f"Missing object '{leaf}' in first tick of {ticks_path}. Found IDs: {sorted(obj_by_id.keys())}"
            )

        pose_w = o.get("pose_w") or {}
        pos = pose_w.get("position_m")
        quat = pose_w.get("orientation_wxyz")

        if not (isinstance(pos, (list, tuple)) and len(pos) == 3):
            raise RuntimeError(f"Object '{leaf}' missing pose_w.position_m in {ticks_path}")
        if not (isinstance(quat, (list, tuple)) and len(quat) == 4):
            # Default orientation if missing
            quat = (1.0, 0.0, 0.0, 0.0)

        pos_f = (float(pos[0]), float(pos[1]), float(pos[2]))
        quat_f = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
        fixed.append((pos_f, quat_f))

        lab = o.get("label")
        if isinstance(lab, str) and lab.strip():
            id_to_label[leaf] = lab.strip()

    return fixed, id_to_label


def _load_init_joints_from_raw_episode(raw_episode: Path) -> Optional[List[float]]:
    """
    Best-effort helper: read tick0 robot joint positions (first 6) from a raw_data episode.
    Returns None if missing/invalid.
    """
    raw_episode = raw_episode.expanduser().resolve()
    ticks_path = raw_episode if raw_episode.is_file() else (raw_episode / "ticks.jsonl")
    if not ticks_path.exists():
        return None
    try:
        with ticks_path.open("r", encoding="utf-8") as f:
            line = f.readline()
        tick0 = json.loads(line)
        robot = tick0.get("robot") or {}
        joints = (robot.get("joints") or {}).get("positions") or []
        jf: List[float] = []
        for v in joints:
            try:
                jf.append(float(v))
            except Exception:
                continue
        if len(jf) < 6:
            return None
        return [float(x) for x in jf[:6]]
    except Exception:
        return None


def _load_domain_randomization_from_raw_episode(raw_episode: Path) -> Optional[dict]:
    """
    Best-effort helper: read the first `domain_randomization` event from a raw_data episode's events.jsonl.

    Returns the event `data` dict (with optional 'light' and 'camera' fields) or None if missing.
    """
    raw_episode = raw_episode.expanduser().resolve()
    ep_dir = raw_episode if raw_episode.is_dir() else raw_episode.parent
    events_path = ep_dir / "events.jsonl"
    if not events_path.exists():
        return None
    try:
        with events_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    evt = json.loads(line)
                except Exception:
                    continue
                if not isinstance(evt, dict):
                    continue
                if str(evt.get("type") or "").strip() == "domain_randomization":
                    data = evt.get("data")
                    return data if isinstance(data, dict) else None
    except Exception:
        return None
    return None


def _infer_box_size_from_raw_episode(raw_episode: Path, *, table_z_w: float) -> Optional[float]:
    """
    Best-effort helper: infer a cube side length from a raw episode by assuming objects rest on the table.

    For each object in tick0, if its world-frame center z is `z_center`, and the table top is at `table_z_w`,
    then for an axis-aligned cube resting on the table, side ~= 2*(z_center - table_z_w).
    We take the median over objects and clamp to a reasonable range.
    """
    raw_episode = raw_episode.expanduser().resolve()
    ticks_path = raw_episode if raw_episode.is_file() else (raw_episode / "ticks.jsonl")
    if not ticks_path.exists():
        return None
    try:
        with ticks_path.open("r", encoding="utf-8") as f:
            line = f.readline()
        tick0 = json.loads(line)
        objs = tick0.get("objects") or []
        if not isinstance(objs, list) or not objs:
            return None
        sizes: list[float] = []
        for o in objs:
            if not isinstance(o, dict):
                continue
            pose_w = o.get("pose_w") or {}
            pos = pose_w.get("position_m")
            if not (isinstance(pos, (list, tuple)) and len(pos) == 3):
                continue
            try:
                zc = float(pos[2])
            except Exception:
                continue
            dz = zc - float(table_z_w)
            if dz <= 0.0:
                continue
            s = 2.0 * dz
            if 0.01 <= s <= 0.20:
                sizes.append(float(s))
        if not sizes:
            return None
        s_med = float(np.median(np.asarray(sizes, dtype=np.float32)))
        # Clamp to sane bounds in case of small penetration/jitter.
        return float(np.clip(s_med, 0.02, 0.12))
    except Exception:
        return None


def _apply_domain_randomization_dict(dr: dict, *, enable_cameras: bool, debug: bool = False) -> None:
    """Apply a vla_v1-style `domain_randomization` dict to the current USD stage (light + camera)."""
    try:
        import importlib

        omni_usd = importlib.import_module("omni.usd")
        UsdGeom = importlib.import_module("pxr.UsdGeom")
        UsdLux = importlib.import_module("pxr.UsdLux")
        Gf = importlib.import_module("pxr.Gf")
        stage = omni_usd.get_context().get_stage()
    except Exception as e:
        if debug:
            print(f"[xVLA][WARN] DR apply: failed to import USD modules: {e}")
        return

    # --- Lighting (UsdLux.DomeLight)
    try:
        light = dr.get("light") if isinstance(dr, dict) else None
        if isinstance(light, dict):
            prim_path = str(light.get("prim_path") or "/World/Light")
            prim = stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                dome = UsdLux.DomeLight(prim)
                if "intensity" in light:
                    dome.GetIntensityAttr().Set(float(light["intensity"]))
                c = light.get("color_rgb")
                if isinstance(c, (list, tuple)) and len(c) == 3:
                    dome.GetColorAttr().Set(Gf.Vec3f(float(c[0]), float(c[1]), float(c[2])))
                if debug:
                    try:
                        _i = float(dome.GetIntensityAttr().Get())
                        _c = dome.GetColorAttr().Get()
                        print(
                            f"[xVLA] Applied DR light: path={prim_path} intensity={_i:.4f} "
                            f"color_rgb={[float(_c[0]), float(_c[1]), float(_c[2])]}"
                        )
                    except Exception:
                        print(f"[xVLA] Applied DR light: path={prim_path}")
    except Exception as e:
        if debug:
            print(f"[xVLA][WARN] DR apply: light failed: {e}")

    # --- Camera pose + FOV
    if not enable_cameras:
        return
    try:
        cam = dr.get("camera") if isinstance(dr, dict) else None
        if not isinstance(cam, dict):
            return
        cam_prim_path = cam.get("prim_path")
        if not cam_prim_path:
            return
        cam_prim_path = str(cam_prim_path)
        cam_prim = stage.GetPrimAtPath(cam_prim_path)
        if not cam_prim.IsValid():
            return

        pos = cam.get("pos_xyz")
        rpy = cam.get("rpy_deg")
        if isinstance(pos, (list, tuple)) and len(pos) == 3 and isinstance(rpy, (list, tuple)) and len(rpy) == 3:
            xform = UsdGeom.Xformable(cam_prim)
            xform.ClearXformOpOrder()
            translate_op = xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
            rotate_op = xform.AddRotateXYZOp(UsdGeom.XformOp.PrecisionFloat)
            translate_op.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
            rotate_op.Set(Gf.Vec3f(float(rpy[0]), float(rpy[1]), float(rpy[2])))

        fov = cam.get("fov_deg")
        if fov is not None:
            try:
                cam_geom = UsdGeom.Camera(cam_prim)
                import math as _math

                # Match data_collection/profiles/vla_v1.py convention.
                sensor_size_mm = 36.0
                focal_length_mm = sensor_size_mm / (2.0 * _math.tan(_math.radians(float(fov)) / 2.0))
                cam_geom.GetFocalLengthAttr().Set(float(focal_length_mm))
            except Exception:
                pass

        if debug:
            print(
                f"[xVLA] Applied DR camera: path={cam_prim_path} "
                f"pos_xyz={pos if isinstance(pos, (list, tuple)) else 'NA'} "
                f"rpy_deg={rpy if isinstance(rpy, (list, tuple)) else 'NA'} "
                f"fov_deg={fov if fov is not None else 'NA'}"
            )
    except Exception as e:
        if debug:
            print(f"[xVLA][WARN] DR apply: camera failed: {e}")


def _read_ee_pose_b(robot, ee_link_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (pos_b (3,), quat_b_wxyz (4,)) in base frame."""
    from isaaclab.utils.math import subtract_frame_transforms

    body_ids, _ = robot.find_bodies([ee_link_name])
    ee_id = int(body_ids[0])
    ee_pose_w = robot.data.body_pose_w[:, ee_id]
    root_pose_w = robot.data.root_pose_w
    pos_b, quat_b = subtract_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )
    return pos_b[0], quat_b[0]


def _read_gripper_open_fraction(robot, *, joint_regex: str, open_pos: float, close_pos: float) -> float:
    """
    Estimate gripper openness in [0,1] from current gripper joint positions.
    Best-effort (depends on robot asset).
    """
    try:
        ids, _ = robot.find_joints(joint_regex)
        if hasattr(ids, "view"):
            ids_list = [int(v) for v in ids.view(-1).tolist()]
        else:
            ids_list = [int(v) for v in list(ids)]
        if not ids_list:
            return 0.0
        q = robot.data.joint_pos[0, ids_list]
        mean_q = float(q.mean().item())
        denom = float(close_pos) - float(open_pos)
        if abs(denom) < 1e-6:
            return 0.0
        frac = (mean_q - float(open_pos)) / denom
        return float(np.clip(frac, 0.0, 1.0))
    except Exception:
        return 0.0


def _resize_chw_uint8(img_chw: torch.Tensor, *, size_hw: Tuple[int, int]) -> torch.Tensor:
    """Resize a (3,H,W) uint8 tensor to (3,H2,W2) using bilinear (convert to float, back to uint8)."""
    if img_chw.ndim != 3:
        raise ValueError(f"Expected CHW image tensor, got shape={tuple(img_chw.shape)}")
    h2, w2 = int(size_hw[0]), int(size_hw[1])
    x = img_chw.to(dtype=torch.float32).unsqueeze(0)  # (1,3,H,W)
    x = torch.nn.functional.interpolate(x, size=(h2, w2), mode="bilinear", align_corners=False)
    x = torch.clamp(x, 0.0, 255.0).to(dtype=torch.uint8).squeeze(0)
    return x


def main() -> int:
    repo_root = _ensure_repo_on_syspath()

    # Make prints show up promptly even when stdout is not a TTY (e.g. headless/background runs).
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Run a LeRobot xVLA policy inside Isaac Sim (Kinova Jaco2).")
    ap.add_argument(
        "--model-dir",
        type=str,
        default=str((Path(__file__).resolve().parent / "models" / "xvla" / "stage2")),
        help="Path to your stage2 folder (contains config.json + model.safetensors + policy_{pre,post}processor.json).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for box spawning (and random box yaw when using random layout).",
    )
    ap.add_argument(
        "--layout-raw-episode",
        type=str,
        default=None,
        help=(
            "If set, spawn boxes using poses from a raw_data episode (ticks.jsonl first line, pose_w). "
            "Provide an episode dir like .../raw_data/session_*/episode_0000 or a direct ticks.jsonl path."
        ),
    )
    ap.add_argument(
        "--init-joints-from-layout",
        action="store_true",
        default=True,
        help=(
            "When using --layout-raw-episode, also apply tick0 joint positions (first 6) as --init-joints. "
            "Disable with --no-init-joints-from-layout."
        ),
    )
    ap.add_argument(
        "--no-init-joints-from-layout",
        action="store_true",
        help="Disable --init-joints-from-layout behavior.",
    )
    ap.add_argument(
        "--domain-rand-from-layout",
        action="store_true",
        default=True,
        help=(
            "When using --layout-raw-episode, also apply the episode's logged domain randomization "
            "(camera + light) from events.jsonl. Disable with --no-domain-rand-from-layout."
        ),
    )
    ap.add_argument(
        "--no-domain-rand-from-layout",
        action="store_true",
        help="Disable --domain-rand-from-layout behavior.",
    )
    ap.add_argument("--policy-hz", type=float, default=5.0, help="How often to query the policy (Hz).")
    ap.add_argument("--max-seconds", type=float, default=30.0)
    ap.add_argument(
        "--settle-steps",
        type=int,
        default=180,
        help="Physics settle steps after sim.reset/robot reset before starting policy control (reduces initial jitter).",
    )
    ap.add_argument("--num-objects", type=int, default=4)
    ap.add_argument(
        "--box-size",
        type=float,
        default=None,
        help=(
            "Side length for spawned cubes (meters). If omitted, defaults to ~0.05.\n"
            "When using --layout-raw-episode and box size is omitted, we will auto-infer it from tick0 object z."
        ),
    )
    ap.add_argument("--spawn-min", type=float, nargs=3, default=[0.25, -0.25, 0.95])
    ap.add_argument("--spawn-max", type=float, nargs=3, default=[0.60, 0.25, 0.95])
    ap.add_argument("--min-distance", type=float, default=0.10)
    ap.add_argument("--target-index", type=int, default=1, help="1-based target index (Obj_01=1).")
    ap.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="Override the language command. If omitted, uses 'Pick up the <color> box <idx>.'",
    )
    ap.add_argument("--rot-speed", type=float, default=2.0, help="Rotation speed (rad/s) used for action clamping.")
    ap.add_argument(
        "--speed",
        type=float,
        default=0.7,
        help=(
            "Max translational speed for clamping (m/s). "
            "This controls max per-physics-step delta via (speed * dt). "
            "If the robot is jerky, try 0.1–0.3."
        ),
    )
    ap.add_argument(
        "--action-scale",
        type=float,
        default=1.0,
        help=(
            "Multiply the unnormalized policy action before converting to per-physics-step deltas. "
            "If the robot is too aggressive, try 0.1–0.5."
        ),
    )
    ap.add_argument(
        "--action-ema",
        type=float,
        default=0.0,
        help=(
            "EMA smoothing for the 6D motion command (0=no smoothing, 0.9=strong). "
            "Helps reduce twitching."
        ),
    )
    ap.add_argument("--print-actions", action="store_true", help="Print policy actions periodically.")
    ap.add_argument(
        "--print-actions-every",
        type=int,
        default=10,
        help="When --print-actions is enabled, print once every N policy calls (default: 10).",
    )
    ap.add_argument(
        "--trace-policy",
        action="store_true",
        help=(
            "Print a debug line every policy call: image stats + state + raw/scaled action. "
            "Useful to diagnose 'policy always goes one direction'."
        ),
    )
    ap.add_argument(
        "--trace-policy-every",
        type=int,
        default=1,
        help="When --trace-policy is enabled, only emit the trace once every N policy calls (default: 1).",
    )
    ap.add_argument(
        "--state-mode",
        type=str,
        default="joints8",
        choices=["joints8", "ee_pose8", "joints7"],
        help=(
            "Which proprio state to feed the policy.\n"
            "  - joints8: [j1..j6, gripper_open_frac, 0.0] (matches your stage2 normalizer stats)\n"
            "  - ee_pose8: [ee_pos_b(3), ee_quat_b_wxyz(4), gripper_open_frac] (legacy/debug)\n"
            "  - joints7: [j1..j6, 0.0] (legacy; will be zero-padded/truncated to match model state dim)\n"
        ),
    )
    ap.add_argument(
        "--debug",
        action="store_true",
        help="Enable extra debug prints (policy load timing, camera frame stats, action norms).",
    )
    ap.add_argument(
        "--print-every-s",
        type=float,
        default=2.0,
        help="Print a small heartbeat line every N seconds while running.",
    )
    ap.add_argument(
        "--lerobot-src",
        type=str,
        default=None,
        help=(
            "Optional path to LeRobot source '.../lerobot/src'. "
            "If not provided, we try $LEROBOT_SRC and common local checkout locations."
        ),
    )
    ap.add_argument(
        "--init-joints",
        type=float,
        nargs=6,
        default=None,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help=(
            "Optional: override the initial arm joint positions (radians) for joints 1..6.\n"
            "Tip: Use values from a vla_v1 log tick0 to match the training start distribution."
        ),
    )

    # IsaacLab / AppLauncher args (headless, enable_cameras, device, etc.)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(ap)
    args = ap.parse_args()
    _box_size_provided = getattr(args, "box_size", None) is not None
    # Stable default close to the dataset distribution (vla_v1 uses ~0.04–0.06).
    if getattr(args, "box_size", None) is None:
        args.box_size = 0.05  # type: ignore[attr-defined]
    # Keep a private marker so we can decide whether to override from layout.
    setattr(args, "_box_size_provided", bool(_box_size_provided))
    if getattr(args, "seed", None) is not None:
        np.random.seed(int(args.seed))

    # ---------------------------------------------------------------------
    # Load LeRobot policy BEFORE starting Kit.
    #
    # Reason: Kit/Isaac can mutate sys.path and (in your runs) causes LeRobot imports
    # to appear "stuck" or never complete. Loading the policy first avoids that.
    # ---------------------------------------------------------------------
    lerobot_src_path = _ensure_lerobot_on_syspath(repo_root, lerobot_src=getattr(args, "lerobot_src", None))
    if lerobot_src_path is None:
        raise RuntimeError(
            "Failed to locate LeRobot source. Expected a path like '/abs/path/to/lerobot/src'.\n\n"
            "Fix options:\n"
            "  - Pass:   --lerobot-src /abs/path/to/lerobot/src\n"
            "  - Or set: export LEROBOT_SRC=/abs/path/to/lerobot/src\n"
            "  - Or install into this python env: pip install -e /path/to/lerobot\n"
        )

    # Avoid importing optional-heavy policy packages (diffusers/pyserial) just to load xvla.
    _bypass_lerobot_policy_inits(lerobot_src_path, verbose=bool(getattr(args, "debug", False)))

    policy_device_cli = str(getattr(args, "device", "cuda:0"))
    model_dir = Path(str(args.model_dir)).expanduser().resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f"--model-dir not found: {model_dir}")

    try:
        print("[xVLA] (pre-Kit) Importing lerobot.configs.policies.PreTrainedConfig ...")
        sys.stdout.flush()
        from lerobot.configs.policies import PreTrainedConfig
        print("[xVLA] (pre-Kit) Importing lerobot.policies.xvla.configuration_xvla (register config) ...")
        sys.stdout.flush()
        import lerobot.policies.xvla.configuration_xvla  # noqa: F401
        print("[xVLA] (pre-Kit) Importing lerobot.policies.xvla.modeling_xvla.XVLAPolicy ...")
        sys.stdout.flush()
        from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
        print("[xVLA] (pre-Kit) Importing lerobot.processor pipelines ...")
        sys.stdout.flush()
        from lerobot.processor.pipeline import PolicyProcessorPipeline
        from lerobot.processor.converters import (
            batch_to_transition,
            policy_action_to_transition,
            transition_to_batch,
            transition_to_policy_action,
        )

        # Ensure built-in step registry entries referenced by the saved processor JSON are available.
        import lerobot.processor.batch_processor  # noqa: F401
        import lerobot.processor.device_processor  # noqa: F401
        import lerobot.processor.normalize_processor  # noqa: F401
        import lerobot.processor.rename_processor  # noqa: F401
        import lerobot.processor.tokenizer_processor  # noqa: F401

        # The on-disk pipeline references xvla-specific steps; register light equivalents here
        # without importing the full xvla policy package (which can drag training-time deps).
        _register_minimal_xvla_processor_steps(verbose=bool(getattr(args, "debug", False)))
    except Exception as e:
        raise RuntimeError(
            "Failed to import LeRobot in the current Python environment.\n\n"
            "Fix options:\n"
            "  - Point to your LeRobot checkout:\n"
            "      --lerobot-src /abs/path/to/lerobot/src\n"
            "      or export LEROBOT_SRC=/abs/path/to/lerobot/src\n"
            "  - Install core deps into THIS python env (the one launching Isaac):\n"
            "      python -m pip install 'draccus==0.10.0' 'datasets>=4.0.0,<4.2.0' \\\n"
            "        'accelerate>=1.10.0,<2.0.0' 'transformers>=4.57.1,<5.0.0'\n"
            "  - Or install LeRobot in editable mode:\n"
            "      python -m pip install -e /abs/path/to/lerobot\n"
        ) from e

    # Load config + policy + processors (still pre-Kit)
    try:
        t_load0 = time.time()
        print(f"[xVLA] (pre-Kit) Loading policy config from: {model_dir}")
        sys.stdout.flush()
        cfg = PreTrainedConfig.from_pretrained(str(model_dir), cli_overrides=[f"--device={policy_device_cli}"])
        print(f"[xVLA] (pre-Kit) Policy config loaded: type={cfg.type} device={cfg.device}")
        sys.stdout.flush()
    except Exception as e:
        raise RuntimeError(f"Failed to load policy config from {model_dir}: {e}") from e

    # Model input/output dims (useful for debugging mismatches).
    model_state_dim = 8
    try:
        model_state_dim = int(((cfg.input_features or {}).get("observation.state") or {}).get("shape", [8])[0])
    except Exception:
        model_state_dim = 8
    model_action_dim = 20
    try:
        model_action_dim = int(((cfg.output_features or {}).get("action") or {}).get("shape", [20])[0])
    except Exception:
        model_action_dim = 20

    try:
        print("[xVLA] (pre-Kit) Loading policy weights (this can take a while the first time)...")
        sys.stdout.flush()
        policy = XVLAPolicy.from_pretrained(str(model_dir), config=cfg)
        policy.reset()
        t_load1 = time.time()
        print(f"[xVLA] (pre-Kit) Policy loaded in {t_load1 - t_load0:.1f}s")
        sys.stdout.flush()
    except Exception as e:
        raise RuntimeError(f"Failed to load policy weights from {model_dir}: {e}") from e

    try:
        print("[xVLA] (pre-Kit) Loading policy preprocessor pipeline...")
        sys.stdout.flush()
        preproc = PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=str(model_dir),
            config_filename="policy_preprocessor.json",
            to_transition=batch_to_transition,
            to_output=transition_to_batch,
        )
        print("[xVLA] (pre-Kit) Loading policy postprocessor pipeline...")
        sys.stdout.flush()
        postproc = PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=str(model_dir),
            config_filename="policy_postprocessor.json",
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )
        print("[xVLA] (pre-Kit) Processors loaded.")
        sys.stdout.flush()
    except Exception as e:
        raise RuntimeError(f"Failed to load policy processors from {model_dir}: {e}") from e

    # ---------------------------------------------------------------------
    # Start Kit / Isaac Sim (after policy is ready)
    # ---------------------------------------------------------------------
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Kit may mutate sys.path; re-pin our repo root (for environments/controllers imports).
    repo_root = _ensure_repo_on_syspath()
    # (LeRobot already loaded; but keep the src path available for any lazy imports.)
    lerobot_src_path2 = _ensure_lerobot_on_syspath(repo_root, lerobot_src=getattr(args, "lerobot_src", None))
    if lerobot_src_path2 is not None:
        _bypass_lerobot_policy_inits(lerobot_src_path2, verbose=False)

    # Heavy imports must happen after Kit is started.
    import carb
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import Camera, CameraCfg

    # Enable cameras (if requested). IsaacLab sensors read this setting at init.
    enable_cameras = bool(getattr(args, "enable_cameras", False))
    carb.settings.get_settings().set_bool("/isaaclab/cameras_enabled", enable_cameras)

    # Scene + robot
    from environments.reach_to_grasp_VLA.config import DEFAULT_SCENE, DEFAULT_TOP_DOWN_CAMERA
    from environments.reach_to_grasp_VLA.utils import design_scene
    from environments.utils.camera import create_topdown_camera
    from environments.utils.physix import PhysicsConfig, apply_to_simulation_cfg
    from data_collection.core.objects import ObjectsTracker
    from controllers import CartesianVelocityJogConfig, CartesianVelocityJogController

    phys = PhysicsConfig(device=str(getattr(args, "device", "cuda:0")))
    sim_cfg = sim_utils.SimulationCfg(device=phys.device)
    apply_to_simulation_cfg(sim_cfg, phys)
    sim = sim_utils.SimulationContext(sim_cfg)

    # Optional viewport camera for GUI runs
    if not getattr(args, "headless", False):
        try:
            from environments.reach_to_grasp.config import DEFAULT_CAMERA

            sim.set_camera_view(DEFAULT_CAMERA.eye, DEFAULT_CAMERA.target)
        except Exception:
            pass

    scene_entities, scene_origins = design_scene(DEFAULT_SCENE)
    robot = scene_entities["kinova_j2n6s300"]

    # Create top-down camera prim and optionally attach a sensor.
    create_topdown_camera(DEFAULT_TOP_DOWN_CAMERA)
    camera_sensor = None
    if enable_cameras:
        # Match data-collection camera sensor resolution (typically 640x640) and downsample to model size later.
        camera_cfg = CameraCfg(
            prim_path=DEFAULT_TOP_DOWN_CAMERA.prim_path,
            offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
            spawn=None,
            data_types=["rgb"],
            width=int(getattr(DEFAULT_TOP_DOWN_CAMERA, "resolution", (640, 640))[0]),
            height=int(getattr(DEFAULT_TOP_DOWN_CAMERA, "resolution", (640, 640))[1]),
        )
        camera_sensor = Camera(cfg=camera_cfg)

    # Spawn colored boxes under /World/Origin1/Objects (matching log dataset naming).
    obj_root = "/World/Origin1/Objects"
    fixed_poses_w = None
    label_override = None
    dr_from_layout_data: Optional[dict] = None
    if getattr(args, "layout_raw_episode", None):
        raw_ep = Path(str(args.layout_raw_episode))
        fixed_poses_w, label_override = _load_fixed_layout_from_raw_episode(raw_ep, num_boxes=int(args.num_objects))
        print(f"[xVLA] Using fixed object layout from raw episode: {args.layout_raw_episode}")

        # If box size was not explicitly provided, infer it from tick0 object center heights.
        if not bool(getattr(args, "_box_size_provided", False)):
            try:
                table_z = float(getattr(DEFAULT_SCENE, "table_translation", (0.0, 0.0, 0.8))[2])
            except Exception:
                table_z = 0.8
            est = _infer_box_size_from_raw_episode(raw_ep, table_z_w=table_z)
            if est is not None:
                args.box_size = float(est)  # type: ignore[attr-defined]
                print(f"[xVLA] Loaded --box-size from raw episode tick0 (median): {float(est):.4f} m")

        # Optionally also match the training start distribution by applying tick0 joint positions.
        init_from_layout = bool(getattr(args, "init_joints_from_layout", True)) and not bool(
            getattr(args, "no_init_joints_from_layout", False)
        )
        if init_from_layout and getattr(args, "init_joints", None) is None:
            init6 = _load_init_joints_from_raw_episode(raw_ep)
            if init6 is not None:
                args.init_joints = init6  # type: ignore[attr-defined]
                print(f"[xVLA] Loaded --init-joints from raw episode tick0: {init6}")

        # Optionally also match the domain-randomization distribution by applying the logged DR parameters.
        dr_from_layout = bool(getattr(args, "domain_rand_from_layout", True)) and not bool(
            getattr(args, "no_domain_rand_from_layout", False)
        )
        if dr_from_layout:
            dr_from_layout_data = _load_domain_randomization_from_raw_episode(raw_ep)
            if dr_from_layout_data is not None and bool(dr_from_layout_data.get("enabled", True)):
                print("[xVLA] Loaded domain randomization from raw episode events.jsonl (will apply).")
    prim_paths, id_to_label = _spawn_colored_boxes(
        parent_prim_path=obj_root,
        num_boxes=int(args.num_objects),
        spawn_min=tuple(float(v) for v in args.spawn_min),
        spawn_max=tuple(float(v) for v in args.spawn_max),
        min_dist=float(args.min_distance),
        size_m=float(args.box_size),
        device=str(sim.device),
        fixed_poses_w=fixed_poses_w,
        id_to_label_override=label_override,
    )
    tracker = ObjectsTracker(prim_paths=prim_paths)

    # Reset sim/robot (this transitions the timeline and initializes sensors).
    sim.reset()
    # Match the data-collection demos: explicitly reset robot state to default pose/vel to avoid "physics explosion".
    try:
        origin0 = torch.tensor(scene_origins[0], device=sim.device)
        root_state = robot.data.default_root_state.clone()
        root_state[:, :3] += origin0
        robot.write_root_pose_to_sim(root_state[:, :7])
        robot.write_root_velocity_to_sim(root_state[:, 7:])
        robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
        robot.reset()
    except Exception as e:
        print(f"[xVLA][WARN] Robot reset-to-default failed (continuing): {e}")
    if camera_sensor is not None:
        try:
            camera_sensor.reset()
            if bool(getattr(args, "debug", False)):
                print("[xVLA] Camera sensor reset OK (post sim.reset)")
        except Exception as e:
            print(f"[xVLA][WARN] Camera reset failed: {e}. Disabling camera.")
            camera_sensor = None

    # If available, apply domain randomization from the raw episode after sim.reset so the prims exist/stabilize.
    if dr_from_layout_data is not None and bool(dr_from_layout_data.get("enabled", True)):
        try:
            _apply_domain_randomization_dict(
                dr_from_layout_data,
                enable_cameras=bool(enable_cameras),
                debug=bool(getattr(args, "debug", False)),
            )
            # Reset camera so render products refresh after moving the camera/changing FOV.
            if camera_sensor is not None:
                try:
                    camera_sensor.reset()
                except Exception:
                    pass
        except Exception as e:
            print(f"[xVLA][WARN] Failed to apply domain randomization from layout (continuing): {e}")

    # Controller
    ctrl_cfg = CartesianVelocityJogConfig(
        ee_link_name="j2n6s300_end_effector",
        device=str(sim.device),
        use_relative_mode=True,
        linear_speed_mps=float(getattr(args, "speed", 0.7)),
        workspace_min=(0.20, -0.45, -0.02),
        workspace_max=(0.6, 0.45, 0.35),
    )
    controller = CartesianVelocityJogController(ctrl_cfg, num_envs=1, device=str(sim.device))
    controller.reset(robot)
    controller.set_mode("translate")

    cmd_input = _ConstantCmdInput(device=str(sim.device))
    controller.set_input_provider(cmd_input)  # type: ignore[arg-type]

    # Cache arm joint ids for joints7 state mode (avoid find_joints in the loop).
    arm_joint_ids: list[int] = []
    try:
        _ids, _names = robot.find_joints(ctrl_cfg.arm_joint_regex)
        if hasattr(_ids, "view"):
            arm_joint_ids = [int(v) for v in _ids.view(-1).tolist()]
        else:
            arm_joint_ids = [int(v) for v in list(_ids)]
    except Exception:
        arm_joint_ids = []

    # Optional: reset robot to provided initial joint positions (helps match training distribution).
    if getattr(args, "init_joints", None) is not None:
        try:
            init = [float(x) for x in list(getattr(args, "init_joints"))]
            if len(init) >= 6 and arm_joint_ids:
                q = robot.data.joint_pos.clone()
                qd = torch.zeros_like(robot.data.joint_vel)
                q[0, arm_joint_ids[:6]] = torch.tensor(init[:6], device=sim.device, dtype=q.dtype)
                robot.write_joint_state_to_sim(q, qd)
                robot.reset()
                controller.reset(robot)
                controller.set_mode("translate")
                print(f"[xVLA] Applied --init-joints to arm joints: {init[:6]}")
        except Exception as e:
            print(f"[xVLA][WARN] Failed to apply --init-joints (continuing): {e}")

    # --- Policy already loaded pre-Kit. Keep `policy`, `preproc`, `postproc` from above. ---

    # Build instruction
    tidx = max(1, int(args.target_index))
    target_leaf = f"Obj_{tidx:02d}"
    default_instruction = f"Pick up the {id_to_label.get(target_leaf, f'box {tidx}')}."
    instruction = str(args.instruction) if args.instruction else default_instruction
    print(f"[xVLA] model_dir={model_dir}")
    print(f"[xVLA] target={target_leaf} label='{id_to_label.get(target_leaf, '')}'")
    print(f"[xVLA] instruction: {instruction}")
    if camera_sensor is None:
        print("[xVLA][WARN] Camera disabled; policy cannot run without images. Run with --enable_cameras.")

    dt = float(sim.get_physics_dt())
    # Policy update stride in physics steps
    policy_stride = max(1, int(round((1.0 / max(1e-6, float(args.policy_hz))) / max(1e-9, dt))))
    render_stride = max(1, int(round((1.0 / 60.0) / max(1e-9, dt))))  # ~60 Hz render cap

    # Clamp per-physics-step deltas to keep the robot stable.
    max_dpos = float(ctrl_cfg.linear_speed_mps) * dt
    max_drot = float(args.rot_speed) * dt

    # Optional settle window before starting control (lets physics + contacts stabilize).
    settle_steps = max(0, int(getattr(args, "settle_steps", 0)))
    if settle_steps > 0:
        zero_cmd = torch.zeros(1, 7, dtype=torch.float32, device=sim.device)
        for _ in range(settle_steps):
            cmd_input.set(zero_cmd)
            controller.step(robot, dt)
            sim.step(render=bool(enable_cameras))
            robot.update(dt)
            if camera_sensor is not None:
                try:
                    if hasattr(camera_sensor, "update") and bool(enable_cameras):
                        camera_sensor.update(dt)
                except Exception:
                    pass
        if bool(getattr(args, "debug", False)):
            print(f"[xVLA] Settled for {settle_steps} physics steps.")

    t0 = time.time()
    steps = 0
    last_action = torch.zeros(1, 7, dtype=torch.float32, device=sim.device)
    last_print_t = 0.0
    policy_calls = 0
    skipped_no_rgb = 0
    warned_state_dim = False
    prev_rgb_u8 = None
    prev_a7_raw = None
    prev_dist_to_target = None

    while simulation_app.is_running() and (time.time() - t0) < float(args.max_seconds):
        steps += 1

        # Apply last action every physics step
        cmd_input.set(last_action)
        controller.step(robot, dt)

        do_policy = (steps % policy_stride) == 0
        do_render = bool(do_policy) or (steps % render_stride) == 0

        sim.step(render=bool(do_render))
        robot.update(dt)

        if camera_sensor is not None:
            try:
                if hasattr(camera_sensor, "update") and bool(do_render):
                    camera_sensor.update(dt)
            except Exception:
                pass

        if do_policy and camera_sensor is not None:
            # Read camera RGB -> CHW uint8
            rgb = None
            try:
                cam_out = getattr(camera_sensor.data, "output", None)
                if cam_out is not None:
                    rgb = cam_out.get("rgb")
            except Exception:
                rgb = None
            if rgb is None:
                skipped_no_rgb += 1
                continue

            if rgb.ndim == 4:
                rgb = rgb[0]
            rgb_np = rgb.detach().cpu().numpy()
            # Drop alpha channel if present (RGBA -> RGB)
            if rgb_np.ndim == 3 and rgb_np.shape[-1] >= 4:
                rgb_np = rgb_np[..., :3]
            # If not square, center-crop to square (matches dataset conversion).
            if rgb_np.ndim == 3 and rgb_np.shape[0] != rgb_np.shape[1]:
                h, w = int(rgb_np.shape[0]), int(rgb_np.shape[1])
                side = min(h, w)
                top = (h - side) // 2
                left = (w - side) // 2
                rgb_np = rgb_np[top : top + side, left : left + side, :]
            if rgb_np.max() <= 1.0:
                rgb_np = (rgb_np * 255).astype(np.uint8)
            else:
                rgb_np = rgb_np.astype(np.uint8)
            img = torch.from_numpy(rgb_np).permute(2, 0, 1).contiguous()  # (3,H,W) uint8
            # Downsample to model input size exactly like dataset conversion (bilinear resize).
            if img.shape[-2:] != (256, 256):
                img = _resize_chw_uint8(img, size_hw=(256, 256))

            # Model expects:
            #  - observation.images.image (3,256,256)
            #  - observation.images.image2 (3,256,256)  (duplicate top-down view)
            #  - observation.images.empty_camera_0 (3,224,224)
            empty_224 = torch.zeros(3, 224, 224, dtype=torch.uint8)

            # Proprio state must match model config (your Stage-2 config expects 8D).
            state_mode = str(getattr(args, "state_mode", "ee_pose8"))
            if state_mode == "ee_pose8":
                pos_b, quat_b = _read_ee_pose_b(robot, ctrl_cfg.ee_link_name)
                g_frac = _read_gripper_open_fraction(
                    robot,
                    joint_regex=str(
                        getattr(ctrl_cfg, "gripper_joint_regex", ".*_joint_finger_.*|.*_joint_finger_tip_.*")
                    ),
                    open_pos=float(getattr(ctrl_cfg, "gripper_open_pos", 0.0)),
                    close_pos=float(getattr(ctrl_cfg, "gripper_close_pos", 1.2)),
                )
                state = torch.tensor(
                    [
                        float(pos_b[0]),
                        float(pos_b[1]),
                        float(pos_b[2]),
                        float(quat_b[0]),
                        float(quat_b[1]),
                        float(quat_b[2]),
                        float(quat_b[3]),
                        float(g_frac),
                    ],
                    dtype=torch.float32,
                )
            elif state_mode == "joints8":
                try:
                    if arm_joint_ids:
                        q = robot.data.joint_pos[0, arm_joint_ids].detach().to("cpu", dtype=torch.float32).view(-1)
                    else:
                        q = robot.data.joint_pos[0].detach().to("cpu", dtype=torch.float32).view(-1)
                    q6 = q[:6] if q.numel() >= 6 else torch.nn.functional.pad(q, (0, 6 - int(q.numel())))
                except Exception:
                    q6 = torch.zeros(6, dtype=torch.float32)
                g_frac = _read_gripper_open_fraction(
                    robot,
                    joint_regex=str(
                        getattr(ctrl_cfg, "gripper_joint_regex", ".*_joint_finger_.*|.*_joint_finger_tip_.*")
                    ),
                    open_pos=float(getattr(ctrl_cfg, "gripper_open_pos", 0.0)),
                    close_pos=float(getattr(ctrl_cfg, "gripper_close_pos", 1.2)),
                )
                state = torch.cat(
                    [
                        q6,
                        torch.tensor([float(g_frac), 0.0], dtype=torch.float32),
                    ],
                    dim=0,
                )  # (8,)
            else:
                # Legacy: joints7. Keep it for compatibility, but ensure we match the model dim via pad/truncate.
                try:
                    if arm_joint_ids:
                        q = robot.data.joint_pos[0, arm_joint_ids].detach().to("cpu", dtype=torch.float32).view(-1)
                    else:
                        q = robot.data.joint_pos[0].detach().to("cpu", dtype=torch.float32).view(-1)
                    q6 = q[:6] if q.numel() >= 6 else torch.nn.functional.pad(q, (0, 6 - int(q.numel())))
                except Exception:
                    q6 = torch.zeros(6, dtype=torch.float32)
                state = torch.cat([q6, torch.zeros(1, dtype=torch.float32)], dim=0)  # (7,)

            # Match state vector length expected by the model config.
            state = state.view(-1).to(dtype=torch.float32, device="cpu")
            if int(state.numel()) != int(model_state_dim):
                if not warned_state_dim:
                    print(
                        f"[xVLA][WARN] observation.state dim={int(state.numel())} but model expects {int(model_state_dim)}. "
                        "Padding/truncating."
                    )
                    warned_state_dim = True
                if int(state.numel()) < int(model_state_dim):
                    state = torch.nn.functional.pad(state, (0, int(model_state_dim) - int(state.numel())))
                else:
                    state = state[: int(model_state_dim)]

            raw_batch = {
                "task": instruction,
                "observation.images.image": img,
                "observation.images.image2": img,
                "observation.images.empty_camera_0": empty_224,
                "observation.state": state,
            }

            try:
                t_inf0 = time.time()
                batch = preproc(raw_batch)
                act = policy.select_action(batch)
                act = postproc(act)  # -> cpu (may still be bfloat16)
                # Numpy cannot represent bfloat16; cast before converting.
                act_f32 = act.to(dtype=torch.float32)
                a_model = act_f32.view(-1).detach().cpu().numpy()
                policy_calls += 1
                if bool(getattr(args, "debug", False)) and policy_calls <= 3:
                    # Early sanity: show shapes + basic stats
                    print(
                        f"[xVLA][dbg] infer#{policy_calls} rgb_shape={tuple(rgb_np.shape)} "
                        f"state={state.tolist()} a_shape={tuple(a_model.shape)} "
                        f"dt={time.time() - t_inf0:.3f}s"
                    )
            except Exception as e:
                print(f"[xVLA][WARN] Policy step failed: {e}")
                continue

            # Controller uses 7D: [dx, dy, dz, rx, ry, rz, g]. Model outputs may be padded to 20D.
            a7_raw = np.array(a_model[:7], dtype=np.float32, copy=True)
            if a7_raw.shape[0] < 7:
                a7_raw = np.pad(a7_raw, (0, 7 - a7_raw.shape[0])).astype(np.float32, copy=False)
            a7_scaled = a7_raw.copy()
            # Optional global scaling (helps speed up / tame outputs without retraining).
            a7_scaled[0:6] = a7_scaled[0:6] * float(getattr(args, "action_scale", 1.0))

            # Convert policy action (per policy tick) into per-physics-step deltas.
            # Your dataset stats show action magnitudes on the order of mm–cm per tick (policy_hz),
            # so we distribute the delta across `policy_stride` physics steps.
            stride_f = float(policy_stride)
            a7_step = a7_scaled.copy()
            a7_step[0:6] = a7_step[0:6] / max(1.0, stride_f)

            # Clamp translation/rotation per physics step for stability.
            a7_step[0:3] = np.clip(a7_step[0:3], -max_dpos, max_dpos)
            a7_step[3:6] = np.clip(a7_step[3:6], -max_drot, max_drot)

            # Choose controller mode
            g_val = float(a7_step[6]) if a7_step.shape[0] >= 7 else 0.0
            drot_norm = float(np.linalg.norm(a7_step[3:6]))
            ema = float(getattr(args, "action_ema", 0.0))
            ema = ema if (0.0 <= ema < 1.0) else 0.0
            chosen_mode = "translate"
            if abs(g_val) > 0.2:
                controller.set_mode("gripper")
                chosen_mode = "gripper"
                last_action = torch.zeros(1, 7, dtype=torch.float32, device=sim.device)
                last_action[0, 6] = 1.0 if g_val > 0.0 else -1.0
            elif drot_norm > (0.25 * max_drot):
                controller.set_mode("rotate")
                chosen_mode = "rotate"
                new_cmd = torch.zeros(1, 7, dtype=torch.float32, device=sim.device)
                new_cmd[0, 0:6] = torch.tensor(a7_step[:6], dtype=torch.float32, device=sim.device).view(1, 6)
                if ema > 0.0:
                    new_cmd[0, 0:6] = ema * last_action[0, 0:6] + (1.0 - ema) * new_cmd[0, 0:6]
                last_action = new_cmd
            else:
                controller.set_mode("translate")
                chosen_mode = "translate"
                new_cmd = torch.zeros(1, 7, dtype=torch.float32, device=sim.device)
                new_cmd[0, 0:6] = torch.tensor(a7_step[:6], dtype=torch.float32, device=sim.device).view(1, 6)
                if ema > 0.0:
                    new_cmd[0, 0:6] = ema * last_action[0, 0:6] + (1.0 - ema) * new_cmd[0, 0:6]
                last_action = new_cmd

            # Trace/debug prints
            if bool(getattr(args, "trace_policy", False)):
                every = max(1, int(getattr(args, "trace_policy_every", 1)))
                if (policy_calls % every) == 0:
                    # Image stats + frame delta (helps detect a stuck camera feed).
                    rgb_min = int(rgb_np.min())
                    rgb_max = int(rgb_np.max())
                    rgb_mean = float(rgb_np.mean())
                    rgb_diff = None
                    if prev_rgb_u8 is not None:
                        try:
                            rgb_diff = float(
                                np.mean(np.abs(rgb_np.astype(np.int16) - prev_rgb_u8.astype(np.int16)))
                            )
                        except Exception:
                            rgb_diff = None
                    prev_rgb_u8 = rgb_np

                    a_delta = None
                    if prev_a7_raw is not None:
                        try:
                            a_delta = float(np.linalg.norm(a7_raw - prev_a7_raw))
                        except Exception:
                            a_delta = None
                    prev_a7_raw = a7_raw.copy()

                    # Target-relative geometry (helps diagnose frame mismatches / safety clamping).
                    target_leaf = f"Obj_{max(1, int(getattr(args, 'target_index', 1))):02d}"
                    target_pos_b_list = None
                    dist_to_target = None
                    cos_to_target = None
                    pinned = []
                    try:
                        ee_pos_b_dbg, _ee_quat_b_dbg = _read_ee_pose_b(robot, ctrl_cfg.ee_link_name)
                        ee_pos_b_cpu = ee_pos_b_dbg.detach().to("cpu", dtype=torch.float32).view(-1).numpy()

                        target_obj = None
                        for o in tracker.snapshot():
                            if str(getattr(o, "id", "")) == target_leaf:
                                target_obj = o
                                break
                        if target_obj is not None:
                            from isaaclab.utils.math import subtract_frame_transforms

                            root_pose_w = robot.data.root_pose_w
                            pos_w = torch.tensor(
                                list(getattr(getattr(target_obj, "pose", None), "position_m", (0.0, 0.0, 0.0))),
                                device=sim.device,
                                dtype=torch.float32,
                            ).view(1, 3)
                            quat_w = torch.tensor(
                                list(getattr(getattr(target_obj, "pose", None), "orientation_wxyz", (1.0, 0.0, 0.0, 0.0))),
                                device=sim.device,
                                dtype=torch.float32,
                            ).view(1, 4)
                            pos_b, _quat_b = subtract_frame_transforms(
                                root_pose_w[:, 0:3], root_pose_w[:, 3:7], pos_w, quat_w
                            )
                            target_pos_b = pos_b[0].detach().to("cpu", dtype=torch.float32).view(-1).numpy()
                            target_pos_b_list = [float(x) for x in target_pos_b.tolist()]

                            rel = target_pos_b - ee_pos_b_cpu
                            dist_to_target = float(np.linalg.norm(rel))
                            if prev_dist_to_target is not None:
                                d_dist = float(dist_to_target - prev_dist_to_target)
                            else:
                                d_dist = None
                            prev_dist_to_target = float(dist_to_target)

                            dpos = a7_scaled[0:3].astype(np.float32, copy=False)
                            denom = float(np.linalg.norm(dpos) * np.linalg.norm(rel))
                            if denom > 1e-9:
                                cos_to_target = float(np.dot(dpos, rel) / denom)
                            else:
                                cos_to_target = None

                            # Detect likely workspace pinning (command pushing out of bounds).
                            ws_min = getattr(ctrl_cfg, "workspace_min", (None, None, None))
                            ws_max = getattr(ctrl_cfg, "workspace_max", (None, None, None))
                            eps = 1e-4
                            if ws_min and ws_min[0] is not None and (ee_pos_b_cpu[0] <= float(ws_min[0]) + eps) and dpos[0] < 0:
                                pinned.append("x_min")
                            if ws_max and ws_max[0] is not None and (ee_pos_b_cpu[0] >= float(ws_max[0]) - eps) and dpos[0] > 0:
                                pinned.append("x_max")
                            if ws_min and ws_min[1] is not None and (ee_pos_b_cpu[1] <= float(ws_min[1]) + eps) and dpos[1] < 0:
                                pinned.append("y_min")
                            if ws_max and ws_max[1] is not None and (ee_pos_b_cpu[1] >= float(ws_max[1]) - eps) and dpos[1] > 0:
                                pinned.append("y_max")
                            if ws_min and ws_min[2] is not None and (ee_pos_b_cpu[2] <= float(ws_min[2]) + eps) and dpos[2] < 0:
                                pinned.append("z_min")
                            if ws_max and ws_max[2] is not None and (ee_pos_b_cpu[2] >= float(ws_max[2]) - eps) and dpos[2] > 0:
                                pinned.append("z_max")

                            # Append delta-distance info to target line (positive means moving away).
                            if d_dist is not None:
                                dist_to_target = float(dist_to_target)
                                # Keep d_dist in a local for the print below.
                                _d_dist = d_dist
                            else:
                                _d_dist = None
                        else:
                            _d_dist = None
                    except Exception:
                        target_pos_b_list = None
                        dist_to_target = None
                        cos_to_target = None
                        pinned = []
                        _d_dist = None

                    print(
                        f"[xVLA][trace] policy#{policy_calls} "
                        f"rgb(min/mean/max)={rgb_min}/{rgb_mean:.1f}/{rgb_max} "
                        f"rgb_diff={rgb_diff if rgb_diff is not None else 'NA'} "
                        f"state={state.tolist()} "
                        f"a7_raw={a7_raw.tolist()} "
                        f"a7_scaled={a7_scaled.tolist()} "
                        f"a7_step={a7_step.tolist()} "
                        f"a_raw_delta={a_delta if a_delta is not None else 'NA'} "
                        f"mode={chosen_mode} "
                        f"model_action_dim={int(a_model.shape[0])}/{int(model_action_dim)} "
                        f"target={target_leaf} "
                        f"target_pos_b={target_pos_b_list if target_pos_b_list is not None else 'NA'} "
                        f"dist_to_target={dist_to_target if dist_to_target is not None else 'NA'} "
                        f"d_dist={_d_dist if '_d_dist' in locals() and _d_dist is not None else 'NA'} "
                        f"cos_to_target={cos_to_target if cos_to_target is not None else 'NA'} "
                        f"pinned={','.join(pinned) if pinned else 'none'}"
                    )

            if bool(getattr(args, "print_actions", False)):
                every = max(1, int(getattr(args, "print_actions_every", 10)))
                if (policy_calls % every) == 0:
                    print(
                        f"[xVLA][act] policy#{policy_calls} "
                        f"a7_raw={a7_raw.tolist()} a7_scaled={a7_scaled.tolist()} a7_step={a7_step.tolist()} "
                        f"mode={chosen_mode} model_action_dim={int(a_model.shape[0])}"
                    )

        # Heartbeat (helps confirm the loop is actually running)
        now = time.time() - t0
        if float(getattr(args, "print_every_s", 0.0)) > 0.0 and (now - last_print_t) >= float(args.print_every_s):
            last_print_t = now
            la = last_action.detach().view(-1).to("cpu").numpy() if isinstance(last_action, torch.Tensor) else None
            sim_t = float(steps) * float(dt)
            rtf = sim_t / max(1e-6, float(now))
            dpos_m = float(np.linalg.norm(la[0:3])) if la is not None and la.size >= 3 else 0.0
            drot_rad = float(np.linalg.norm(la[3:6])) if la is not None and la.size >= 6 else 0.0
            v_mps = dpos_m / max(1e-9, float(dt))
            w_rps = drot_rad / max(1e-9, float(dt))
            print(
                f"[xVLA] wall={now:6.1f}s sim={sim_t:6.2f}s rtf={rtf:4.2f} "
                f"steps={steps} policy_calls={policy_calls} no_rgb={skipped_no_rgb} "
                f"dpos_step={dpos_m*1e3:6.3f}mm v~{v_mps:6.3f}m/s "
                f"drot_step={drot_rad*1e3:6.3f}mrad w~{w_rps:6.3f}rad/s"
            )

    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


