"""IsaacLab evaluation entrypoint for TinyVLA or SmolVLA + (optional) TTC.

Policy backends:
  - `tiny` (default): checkpoint from `python -m vla_lab.train` (`.pt`).
  - `smolvla`: LeRobot policy (`lerobot/smolvla_base` or a fine-tuned output dir).
    Requires `--lerobot-dataset-root` pointing at the **exported** dataset
    (same path as `convert_kinova_to_lerobot --out-dir`) so `meta/stats.json`
    matches training normalization.

Usage (from repo root):
    ./IsaacLab/isaaclab.sh -p vla_lab/eval_isaaclab.py \\
        --config vla_lab/configs/eval_isaac.yaml \\
        --enable_cameras \\
        --device cuda:0

SmolVLA example:
    ./IsaacLab/isaaclab.sh -p vla_lab/eval_isaaclab.py \\
        --config vla_lab/configs/eval_isaac.yaml \\
        --policy-backend smolvla \\
        --ckpt lerobot/smolvla_base \\
        --lerobot-dataset-root vla_lab/datasets/lerobot_kinova_v0 \\
        --enable_cameras --device cuda:0

For headless eval add `--headless`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Path setup BEFORE any other repo imports.
from vla_lab._path import setup_repo_imports  # noqa: E402

ROOT = setup_repo_imports()


# ---------------------------------------------------------------------------
# Argument parsing (Isaac AppLauncher args are added after we have YAML config)
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VLA-TTC IsaacLab evaluator")
    parser.add_argument("--config", type=str, default="vla_lab/configs/eval_isaac.yaml")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--num-objects", type=int, default=None)
    parser.add_argument("--spawn-mode", type=str, default=None, choices=[None, "box", "usd"])
    parser.add_argument("--language", type=str, default=None, help="Override episode language with a fixed string")
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument(
        "--policy-backend",
        type=str,
        default=None,
        help="tiny (default) or smolvla",
    )
    parser.add_argument(
        "--lerobot-dataset-root",
        type=str,
        default=None,
        help="LeRobot dataset directory (meta/stats.json) used for SmolVLA normalization.",
    )
    parser.add_argument(
        "--occlusion-mode",
        type=str,
        default=None,
        help="Partial-observation ablation: none|bottom_strip|top_strip|center_box|random_patch",
    )
    parser.add_argument(
        "--occlusion-fraction",
        type=float,
        default=None,
        help="Fraction of image rows/area occluded (see vla_lab.partial_obs).",
    )
    parser.add_argument(
        "--observability-mode",
        type=str,
        default=None,
        help="Label for logging only, e.g. external_only | occluded (see eval YAML).",
    )
    parser.add_argument(
        "--occlusion-patch-seed",
        type=int,
        default=None,
        help="If set, random_patch occlusion uses a deterministic seed (paired episodes / sweeps).",
    )
    parser.add_argument(
        "--gating-threshold",
        type=float,
        default=None,
        help="Override yaml ttc.gating_threshold (τ sweeps for gated TTC).",
    )
    parser.add_argument(
        "--gating-noise-std",
        type=float,
        default=None,
        help="Override yaml ttc.gating_noise_std (ε sensitivity).",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default=None,
        help="act/compute/query controller: none|autonomy|fixed_compute|compute_gated|scale|"
        "knowno|insight|allocator (default none = legacy TTC path).",
    )
    parser.add_argument(
        "--allocator-fit",
        type=str,
        default=None,
        help="Path to allocator_fit.json from `python -m vla_lab.fit_allocator` (unlocks the query branch).",
    )
    parser.add_argument(
        "--emit-calibration",
        type=str,
        default=None,
        help="Write per-step calibration records (JSONL) for vla_lab.calibration.analyze.",
    )

    # Defer adding Isaac AppLauncher args to after we know we want to launch.
    from isaaclab.app import AppLauncher  # type: ignore

    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError("vla_lab.eval_isaaclab requires PyYAML.") from exc
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _make_language_command(*, ep_idx: int, target_label: str, color: Optional[str], box_idx: Optional[int], rng_seed: int) -> str:
    """Reproduces the `vla_v1` template selection rules so eval prompts match training."""

    templates: List[str] = [
        "Pick up the {label}.",
        "Reach and pick up the {label}.",
        "Go to the {label} and grasp it.",
        "Grab the {label} and lift it.",
        "Please pick up the {label}.",
        "Move to the {label} and pick it up.",
    ]
    if box_idx is not None:
        templates += [
            "Pick up box number {box_idx}.",
            "Go for box {box_idx} and pick it up.",
            "Grasp box {box_idx} and lift it.",
        ]
    if color is not None:
        templates += [
            "Pick up the {color} box.",
            "Reach to the {color} box and pick it up.",
            "Grab the {color} box and lift it.",
        ]
    if (box_idx is not None) and (color is not None):
        templates += [
            "Pick up the {color} box (box {box_idx}).",
            "Go to box {box_idx} \u2014 the {color} one \u2014 and pick it up.",
        ]

    rng = random.Random(int(ep_idx) + int(rng_seed))
    tmpl = rng.choice(templates)
    return tmpl.format(label=target_label, color=str(color), box_idx=str(box_idx))


def _try_git_commit(repo_root: Path) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return None


def _aggregate_ttc_latencies(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-query TTC stats for reviewer-style reporting (K̄, gate expansion rate)."""

    if not rows:
        return {}
    k_effs = [int(r["k_eff"]) for r in rows if r.get("k_eff") is not None]
    expanded = [bool(r.get("gate_expanded")) for r in rows if r.get("gate_expanded") is not None]
    unc = [float(r["uncertainty"]) for r in rows if r.get("uncertainty") is not None]
    if not k_effs:
        return {}
    ks = sorted(k_effs)
    p90_idx = max(0, min(len(ks) - 1, int(math.ceil(0.9 * len(ks)) - 1)))
    out: Dict[str, Any] = {
        "num_policy_queries": int(len(k_effs)),
        "k_mean": float(statistics.mean(k_effs)),
        "k_median": float(statistics.median(k_effs)),
        "k_p90": float(ks[p90_idx]),
    }
    if expanded:
        out["frac_gate_expanded"] = float(sum(1 for x in expanded if x) / max(1, len(expanded)))
    if unc:
        out["uncertainty_median"] = float(statistics.median(unc))
    return out


def _box_desc_from_prim(prim_path: str, box_colors: List[str]) -> Tuple[str, Optional[str], Optional[int]]:
    leaf = str(prim_path).split("/")[-1]
    try:
        if "_" in leaf:
            idx = int(leaf.split("_")[-1])
            color = box_colors[(idx - 1) % len(box_colors)]
            return f"{color} box {idx}", color, idx
    except Exception:
        pass
    return f"box {leaf}", None, None


# ---------------------------------------------------------------------------
# Policy adapter that bridges trained model -> CartesianVelocityJogController
# ---------------------------------------------------------------------------


class PolicyInputProvider:
    """A `controllers.base.InputProvider` that streams per-physics-step
    velocity commands from a trained TinyVLA via the TTC pipeline.

    The model produces an action chunk at the *policy rate* (e.g. 5 Hz),
    where each action is a delta-EE pose + a discrete gripper signal over
    one policy interval. This provider distributes that delta across the
    physics steps that fall inside one policy interval, then advances to
    the next chunk action when the interval is consumed.

    On every physics step it returns a 7D tensor `[dx, dy, dz, rx, ry, rz, g]`.
    The eval loop is responsible for switching the controller's mode to
    "gripper" on physics steps where `g != 0`, and back to "translate" on
    other steps. We expose the desired mode via `current_mode_hint`.
    """

    def __init__(
        self,
        *,
        physics_dt: float,
        policy_rate_hz: int,
        device: str,
        chunk_consume: int = 1,
    ) -> None:
        self.physics_dt = float(physics_dt)
        self.policy_rate_hz = int(policy_rate_hz)
        self.device = device
        self.chunk_consume = max(1, int(chunk_consume))

        # number of physics steps per policy action interval
        self._n_phys_per_action = max(1, int(round(1.0 / (self.policy_rate_hz * self.physics_dt))))

        self._chunk: Optional[Any] = None  # tensor (T, A) on device
        self._action_idx: int = 0
        self._phys_in_action: int = 0
        self._gripper_flush_steps: int = 0
        self._gripper_dir: float = 0.0  # +1 open, -1 close, 0 none
        self.current_mode_hint: str = "translate"
        self._last_cmd: Optional[Any] = None

    def reset(self) -> None:
        self._chunk = None
        self._action_idx = 0
        self._phys_in_action = 0
        self._gripper_flush_steps = 0
        self._gripper_dir = 0.0
        self.current_mode_hint = "translate"
        self._last_cmd = None

    def needs_new_chunk(self) -> bool:
        return self._chunk is None or self._action_idx >= min(self.chunk_consume, self._chunk.size(0))

    def set_chunk(self, chunk_actions: "torch.Tensor") -> None:
        """Provide a new (T, A) action chunk in *physical* (denormalised) units."""

        self._chunk = chunk_actions
        self._action_idx = 0
        self._phys_in_action = 0
        self._gripper_flush_steps = 0
        self._gripper_dir = 0.0

    def _advance_to_next_action(self) -> None:
        self._action_idx += 1
        self._phys_in_action = 0
        self._gripper_flush_steps = 0
        self._gripper_dir = 0.0

    def advance(self) -> "torch.Tensor":
        import torch as _torch

        if self._chunk is None or self._action_idx >= self._chunk.size(0):
            cmd = _torch.zeros(1, 7, device=self.device)
            self.current_mode_hint = "translate"
            self._last_cmd = cmd
            return cmd

        action = self._chunk[self._action_idx]  # (A,)
        gripper_val = float(action[6].item()) if action.numel() >= 7 else 0.0

        # If a gripper trigger is queued, flush it for a few physics steps.
        if self._gripper_flush_steps > 0:
            cmd = _torch.zeros(1, 7, device=self.device)
            cmd[0, 6] = float(self._gripper_dir)
            self.current_mode_hint = "gripper"
            self._gripper_flush_steps -= 1
            if self._gripper_flush_steps == 0:
                # Done with gripper; move on to next chunk action.
                self._advance_to_next_action()
            self._last_cmd = cmd
            return cmd

        # Detect a discrete gripper transition for this action.
        if abs(gripper_val) > 0.4 and self._phys_in_action == 0:
            self._gripper_dir = 1.0 if gripper_val > 0 else -1.0
            # Hold the gripper command for a fixed number of physics steps so
            # the controller has time to actually open/close.
            self._gripper_flush_steps = max(15, int(0.5 * self._n_phys_per_action))
            cmd = _torch.zeros(1, 7, device=self.device)
            cmd[0, 6] = float(self._gripper_dir)
            self.current_mode_hint = "gripper"
            self._gripper_flush_steps -= 1  # consume one step now
            self._last_cmd = cmd
            return cmd

        # Otherwise produce a per-physics-step delta-pose command.
        per_step = action[:6] / float(self._n_phys_per_action)
        cmd = _torch.zeros(1, 7, device=self.device)
        cmd[0, :6] = per_step
        self.current_mode_hint = "translate"
        self._phys_in_action += 1
        if self._phys_in_action >= self._n_phys_per_action:
            self._advance_to_next_action()
        self._last_cmd = cmd
        return cmd

    @property
    def last_cmd(self) -> Optional[Any]:
        return self._last_cmd


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = _parse_args()
    cfg = _load_yaml(Path(args.config))
    eval_cfg = cfg.get("eval", {})
    ttc_cfg_dict = dict(cfg.get("ttc") or {})
    occlusion_cfg = cfg.get("occlusion", {}) or {}
    if args.gating_threshold is not None:
        ttc_cfg_dict["gating_threshold"] = float(args.gating_threshold)
    if args.gating_noise_std is not None:
        ttc_cfg_dict["gating_noise_std"] = float(args.gating_noise_std)
    observability_mode = str(
        args.observability_mode if args.observability_mode is not None else cfg.get("observability_mode", "external_only")
    )
    ckpt_path = Path(args.ckpt or cfg.get("ckpt", "vla_lab/checkpoints/tiny_v0/last.pt"))

    # Act/compute/query controller (HRI pivot). controller="none" keeps the legacy TTC path.
    alloc_cfg = dict(cfg.get("allocation") or {})
    controller_kind = str(args.controller if args.controller is not None else alloc_cfg.get("controller", "none")).lower().strip()
    use_controller = controller_kind not in ("", "none")
    allocator_fit_path = args.allocator_fit if args.allocator_fit is not None else alloc_cfg.get("fit_path")
    emit_calibration = args.emit_calibration if args.emit_calibration is not None else alloc_cfg.get("emit_calibration")

    policy_backend = str(args.policy_backend or cfg.get("policy_backend", "tiny")).lower().strip()
    if policy_backend not in ("tiny", "smolvla"):
        print(f"[eval] ERROR: unknown policy_backend {policy_backend!r} (use tiny or smolvla)")
        return 2

    lerobot_root_cfg = args.lerobot_dataset_root or cfg.get("lerobot_dataset_root")
    lerobot_dataset_root: Optional[Path] = Path(lerobot_root_cfg).resolve() if lerobot_root_cfg else None

    if policy_backend == "tiny":
        if not ckpt_path.exists():
            print(f"[eval] ERROR: checkpoint not found: {ckpt_path}")
            return 2
    else:
        if lerobot_dataset_root is None or not lerobot_dataset_root.is_dir():
            print(
                "[eval] ERROR: smolvla backend requires --lerobot-dataset-root or "
                "`lerobot_dataset_root` in YAML (exported LeRobot dataset with meta/stats.json)."
            )
            return 2

    if args.num_episodes is not None:
        eval_cfg["num_episodes"] = int(args.num_episodes)
    if args.max_steps is not None:
        eval_cfg["max_steps_per_episode"] = int(args.max_steps)
    if args.out_dir is not None:
        eval_cfg["out_dir"] = args.out_dir
    if args.num_objects is not None:
        eval_cfg["num_objects"] = int(args.num_objects)
    if args.spawn_mode is not None:
        eval_cfg["spawn_mode"] = args.spawn_mode
    if args.seed is not None:
        eval_cfg["language_seed"] = int(args.seed)

    occ_patch_base: Optional[int] = args.occlusion_patch_seed
    if occ_patch_base is None and eval_cfg.get("occlusion_patch_seed") is not None:
        try:
            occ_patch_base = int(eval_cfg["occlusion_patch_seed"])
        except (TypeError, ValueError):
            occ_patch_base = None

    occ_mode = str(args.occlusion_mode) if args.occlusion_mode is not None else str(occlusion_cfg.get("mode", "none"))
    occ_frac = float(args.occlusion_fraction) if args.occlusion_fraction is not None else float(occlusion_cfg.get("fraction", 0.35))

    out_dir = Path(eval_cfg.get("out_dir", "vla_lab/eval_results/tiny_v0"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Launch Isaac Lab
    # ------------------------------------------------------------------
    from isaaclab.app import AppLauncher  # type: ignore

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Heavy imports must happen AFTER the app is started.
    import numpy as np
    import torch
    import carb  # type: ignore

    enable_cameras = bool(getattr(args, "enable_cameras", False))
    if not enable_cameras:
        print("[eval] WARNING: --enable_cameras was not set; the policy will receive blank images.")
    carb.settings.get_settings().set_bool("/isaaclab/cameras_enabled", enable_cameras)

    import isaaclab.sim as sim_utils  # type: ignore
    from isaaclab.sensors import Camera, CameraCfg  # type: ignore
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # type: ignore

    from controllers import CartesianVelocityJogConfig, CartesianVelocityJogController
    from data_collection.core.objects import ObjectsTracker
    from environments.reach_to_grasp.config import DEFAULT_CAMERA, DEFAULT_SCENE
    from environments.reach_to_grasp_VLA.config import DEFAULT_TOP_DOWN_CAMERA
    from environments.reach_to_grasp_VLA.utils import design_scene
    from environments.utils.camera import create_topdown_camera
    from environments.utils.object_loader import ObjectLoader, ObjectLoaderConfig, SpawnBounds
    from environments.utils.physix import PhysicsConfig, apply_to_simulation_cfg, object_loader_kwargs_from_physix

    # vla_lab imports (deferred so the path manipulation in _path takes effect first).
    from vla_lab.dataset import TinyTokenizer
    from vla_lab.partial_obs import apply_occlusion_rgb_chw
    from vla_lab.smolvla_bridge.action_obs_contract import pack_state_xyz_quat
    from vla_lab.calibration.records import CalibrationRecord, write_jsonl

    # ------------------------------------------------------------------
    # Load policy (TinyVLA ckpt or SmolVLA Hub / directory)
    # ------------------------------------------------------------------
    device = torch.device(str(eval_cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")))

    tiny_model_cfg_obj = None
    action_stats = None
    tokenizer = None
    pipeline = None
    smol_policy = None

    tiny_ttc_latencies: List[Dict[str, Any]] = []

    if policy_backend == "tiny":
        from vla_lab.dataset import ActionStats
        from vla_lab.models import TinyVLA, TinyVLAConfig
        from vla_lab.ttc import TTCConfig, TTCPipeline

        ckpt = torch.load(str(ckpt_path), map_location=device)
        tiny_model_cfg_obj = TinyVLAConfig.from_dict(ckpt["model_config"])
        model = TinyVLA(tiny_model_cfg_obj).to(device)
        model.load_state_dict(ckpt["model_state"], strict=True)
        model.eval()
        tokenizer = TinyTokenizer.from_dict(ckpt["tokenizer"])
        if ckpt.get("action_stats") is not None:
            action_stats = ActionStats.from_dict(ckpt["action_stats"])

        ttc_live = TTCConfig.from_dict(ttc_cfg_dict)
        ttc_live.latency_buffer = tiny_ttc_latencies
        pipeline = TTCPipeline(model, ttc_live)
        k_ttc = int(ttc_cfg_dict.get("k_action_samples", 1))
        print(
            f"[eval] loaded TinyVLA {ckpt_path}  params={model.num_parameters():,}  "
            f"vocab={tokenizer.vocab_size}  K_max={k_ttc}  "
            f"selection={ttc_cfg_dict.get('selection', ttc_cfg_dict.get('scoring', 'naive_consensus'))}  "
            f"gating={ttc_cfg_dict.get('gating', 'none')}"
        )
    else:
        from vla_lab.smolvla_bridge.policy_wrapper import SmolVLAIsaacPolicy

        smol_policy = SmolVLAIsaacPolicy(
            policy_path=str(ckpt_path),
            dataset_root=lerobot_dataset_root,
            device=device,
        )
        k_ttc = int(ttc_cfg_dict.get("k_action_samples", 1))
        print(
            f"[eval] loaded SmolVLA policy={ckpt_path!r}  dataset={lerobot_dataset_root}  "
            f"K_max={k_ttc}  selection={ttc_cfg_dict.get('selection', ttc_cfg_dict.get('scoring', 'naive_consensus'))}  "
            f"gating={ttc_cfg_dict.get('gating', 'none')}"
        )

    # Build the act/compute/query controller harness over the loaded policy (if requested).
    harness = None
    if use_controller:
        from vla_lab.allocation.isaac_glue import build_eval_harness

        tiny_objs = None
        if policy_backend == "tiny":
            tiny_objs = {"model": model, "tokenizer": tokenizer, "action_stats": action_stats}
        harness = build_eval_harness(
            controller_kind=controller_kind,
            policy_backend=policy_backend,
            tiny=tiny_objs,
            smol=smol_policy,
            device=device,
            ttc_cfg=ttc_cfg_dict,
            alloc_cfg=alloc_cfg,
            fit_path=str(allocator_fit_path) if allocator_fit_path else None,
        )
        print(f"[eval] controller={controller_kind!r}  fit={allocator_fit_path}  emit_calibration={emit_calibration}")

    calib_records: List[Any] = []
    run_id = f"{policy_backend}_{controller_kind}_{int(time.time())}"

    infer_image_size = int(tiny_model_cfg_obj.image_size) if policy_backend == "tiny" else 224

    # ------------------------------------------------------------------
    # Build sim + scene
    # ------------------------------------------------------------------
    phys = PhysicsConfig(device=str(device))
    sim_cfg = sim_utils.SimulationCfg(device=phys.device)
    apply_to_simulation_cfg(sim_cfg, phys)
    sim = sim_utils.SimulationContext(sim_cfg)
    if (not getattr(args, "headless", False)) and DEFAULT_CAMERA is not None:
        sim.set_camera_view(DEFAULT_CAMERA.eye, DEFAULT_CAMERA.target)

    scene_entities, _ = design_scene(DEFAULT_SCENE)
    robot = scene_entities["kinova_j2n6s300"]
    if enable_cameras and DEFAULT_TOP_DOWN_CAMERA is not None:
        create_topdown_camera(DEFAULT_TOP_DOWN_CAMERA)

    # Object loader (mirrors `vla_v1.py` but stripped down for evaluation).
    BOX_COLORS = ["red", "blue", "yellow", "purple", "orange", "cyan"]
    BOX_COLOR_RGB = [
        (0.85, 0.20, 0.20),
        (0.20, 0.35, 0.90),
        (0.95, 0.85, 0.20),
        (0.65, 0.25, 0.80),
        (0.95, 0.55, 0.15),
        (0.15, 0.80, 0.85),
    ]
    spawn_mode = str(eval_cfg.get("spawn_mode", "box"))
    box_size = float(eval_cfg.get("box_size", 0.08))
    spawn_min = tuple(eval_cfg.get("spawn_min", (0.28, -0.30, 0.90)))
    spawn_max = tuple(eval_cfg.get("spawn_max", (0.55, 0.30, 1.05)))
    min_distance = float(eval_cfg.get("min_distance", 0.20))

    try:
        ycb_dir = f"{ISAAC_NUCLEUS_DIR}/Props/YCB"
    except Exception:
        ycb_dir = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Props/YCB"

    phys_loader_kwargs = object_loader_kwargs_from_physix(phys)
    loader_cfg = ObjectLoaderConfig(
        dataset_dirs=[ycb_dir],
        bounds=SpawnBounds(min_xyz=spawn_min, max_xyz=spawn_max),
        min_distance=min_distance,
        min_distance_xy_only=(spawn_mode == "box"),
        spawn_mode=spawn_mode,
        box_size_min=(box_size, box_size, box_size),
        box_size_max=(box_size, box_size, box_size),
        box_color_palette=BOX_COLOR_RGB,
        box_color_names=BOX_COLORS,
        **phys_loader_kwargs,
    )
    loader = ObjectLoader(loader_cfg)
    spawned_paths = loader.spawn(parent_prim_path="/World/Origin1", num_objects=int(eval_cfg.get("num_objects", 3)))
    print(f"[eval] spawned {len(spawned_paths)} objects: {spawned_paths}")

    # Camera sensor (only if enabled).
    camera_sensor: Optional[Camera] = None
    if enable_cameras and DEFAULT_TOP_DOWN_CAMERA is not None:
        try:
            cam_cfg = CameraCfg(
                prim_path=DEFAULT_TOP_DOWN_CAMERA.prim_path,
                width=DEFAULT_TOP_DOWN_CAMERA.resolution[0],
                height=DEFAULT_TOP_DOWN_CAMERA.resolution[1],
                update_period=0.0,
                data_types=["rgb"],
            )
            camera_sensor = Camera(cfg=cam_cfg)
        except Exception as e:
            print(f"[eval] could not create Camera sensor: {e}; continuing without images.")
            camera_sensor = None

    # Reset sim + robot
    sim.reset()
    if camera_sensor is not None:
        try:
            camera_sensor.reset()
        except Exception:
            pass
    root_state = robot.data.default_root_state.clone()
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    robot.reset()

    ctrl_cfg = CartesianVelocityJogConfig(
        ee_link_name=str(eval_cfg.get("ee_link", "j2n6s300_end_effector")),
        device=str(device),
        use_relative_mode=True,
        linear_speed_mps=0.4,
        workspace_min=(0.20, -0.45, float(eval_cfg.get("workspace_min_z", 0.0))),
        workspace_max=(0.65, 0.45, float(eval_cfg.get("workspace_max_z", 1.20))),
    )
    controller = CartesianVelocityJogController(ctrl_cfg, num_envs=1, device=str(device))
    controller.reset(robot)
    controller.set_mode("translate")

    physics_dt = float(sim.get_physics_dt())
    policy_rate_hz = int(eval_cfg.get("policy_rate_hz", 5))
    policy = PolicyInputProvider(
        physics_dt=physics_dt,
        policy_rate_hz=policy_rate_hz,
        device=str(device),
        chunk_consume=int(eval_cfg.get("chunk_consume", 1)),
    )
    controller.set_input_provider(policy)

    # Tracker for object positions
    tracker = ObjectsTracker(prim_paths=spawned_paths)

    # ------------------------------------------------------------------
    # Helpers for getting current EE state in base frame
    # ------------------------------------------------------------------
    from isaaclab.utils.math import subtract_frame_transforms  # type: ignore

    body_ids, _ = robot.find_bodies([ctrl_cfg.ee_link_name])
    ee_id = int(body_ids[0])

    def _ee_pose_b() -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
        ee_pose_w = robot.data.body_pose_w[:, ee_id]
        root_pose_w = robot.data.root_pose_w
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        return (
            (float(ee_pos_b[0, 0]), float(ee_pos_b[0, 1]), float(ee_pos_b[0, 2])),
            (
                float(ee_quat_b[0, 0]),
                float(ee_quat_b[0, 1]),
                float(ee_quat_b[0, 2]),
                float(ee_quat_b[0, 3]),
            ),
        )

    def _gripper_state() -> str:
        try:
            joint_pos = robot.data.joint_pos[0]
            names = robot.data.joint_names
            idxs = [i for i, n in enumerate(names) if "finger" in str(n)]
            if idxs:
                vals = torch.tensor([joint_pos[i] for i in idxs], dtype=torch.float32)
                return "close" if vals.mean().item() > 0.5 else "open"
        except Exception:
            pass
        return "open"

    # ------------------------------------------------------------------
    # Episode loop
    # ------------------------------------------------------------------
    num_episodes = int(eval_cfg.get("num_episodes", 10))
    max_steps_ep = int(eval_cfg.get("max_steps_per_episode", 4000))
    lift_thresh = float(eval_cfg.get("lift_success_min_dz_m", 0.06))

    n_phys_per_policy = max(1, int(round(1.0 / (policy_rate_hz * physics_dt))))
    print(
        f"[eval] physics_dt={physics_dt:.4f}s  policy_rate_hz={policy_rate_hz}  "
        f"phys_steps_per_policy_tick={n_phys_per_policy}"
    )

    from vla_lab.ttc import _normalize_gating_label as _ttc_gating_mode

    _gate_on = _ttc_gating_mode(str(ttc_cfg_dict.get("gating", "none"))) != "none"
    use_smol_ttc = policy_backend == "smolvla" and (
        int(ttc_cfg_dict.get("k_action_samples", 1)) > 1 or _gate_on
    )
    print(
        f"[eval] observability_mode={observability_mode!r}  occlusion={occ_mode!r} (frac={occ_frac:.3f})  "
        f"smol_ttc={use_smol_ttc}"
    )

    smol_latencies_all: List[Dict[str, Any]] = []
    ep_results: List[Dict[str, Any]] = []
    for ep in range(num_episodes):
        if not simulation_app.is_running():
            break
        # Cycle target through spawned objects deterministically.
        if not spawned_paths:
            print("[eval] no spawned objects; aborting.")
            break
        target_prim = sorted([str(p) for p in spawned_paths])[ep % len(spawned_paths)]
        leaf = target_prim.split("/")[-1]

        if args.language is not None:
            instruction = str(args.language)
            human_label = leaf
            color: Optional[str] = None
        else:
            human_label, color, box_idx = _box_desc_from_prim(target_prim, BOX_COLORS)
            instruction = _make_language_command(
                ep_idx=ep,
                target_label=human_label,
                color=color,
                box_idx=box_idx,
                rng_seed=int(eval_cfg.get("language_seed", 1337)),
            )
        print(f"[eval][EP {ep}] target={leaf!r}  instruction={instruction!r}")

        # Per-episode allocator state: query options (object descriptions) + the oracle target.
        ep_decisions: List[Tuple[int, Any]] = []
        policy_tick_idx = 0
        ep_options: List[str] = []
        ep_oracle: Optional[str] = None
        if harness is not None:
            try:
                for sp in sorted([str(p) for p in spawned_paths]):
                    desc, c, _bidx = _box_desc_from_prim(sp, BOX_COLORS)
                    ep_options.append(c or desc)
                tgt_desc, tgt_color, _ = _box_desc_from_prim(target_prim, BOX_COLORS)
                ep_oracle = tgt_color or tgt_desc
            except Exception:
                ep_options, ep_oracle = [], None
            harness.set_oracle(ep_oracle)

        ids = attn = None
        if policy_backend == "tiny":
            assert tokenizer is not None
            ids, attn = tokenizer.encode(instruction)
            ids = ids.to(device)
            attn = attn.to(device)

        # Settle a few sim steps before evaluation.
        for _ in range(60):
            sim.step(render=False)
            robot.update(physics_dt)
        if camera_sensor is not None:
            try:
                camera_sensor.update(physics_dt)
            except Exception:
                pass

        # Capture initial target z for success check.
        snap = tracker.snapshot()
        z0_target = None
        for o in snap:
            if str(o.id) == leaf:
                z0_target = float(o.pose.position_m[2])
                break

        steps = 0
        ep_t0 = time.time()
        smol_latencies: List[Dict[str, Any]] = []
        while simulation_app.is_running() and steps < max_steps_ep:
            steps += 1

            # Refresh policy chunk at the policy rate.
            if policy.needs_new_chunk():
                # Build observation tensors.
                if camera_sensor is not None:
                    try:
                        cam_data = camera_sensor.data
                        rgb = cam_data.output.get("rgb") if cam_data.output is not None else None
                    except Exception:
                        rgb = None
                else:
                    rgb = None
                if rgb is not None:
                    if rgb.dim() == 4:
                        rgb_t = rgb[0]
                    else:
                        rgb_t = rgb
                    rgb_t = rgb_t.float()
                    if rgb_t.max() > 1.5:
                        rgb_t = rgb_t / 255.0
                    if rgb_t.shape[-1] == 3 and rgb_t.dim() == 3:
                        rgb_t = rgb_t.permute(2, 0, 1).contiguous()
                    rgb_t = torch.nn.functional.interpolate(
                        rgb_t.unsqueeze(0).to(device),
                        size=(infer_image_size, infer_image_size),
                        mode="bilinear",
                        align_corners=False,
                    )[0]
                else:
                    rgb_t = torch.zeros(3, infer_image_size, infer_image_size, device=device)

                occ_patch_seed = None
                if occ_patch_base is not None and str(occ_mode).lower() == "random_patch":
                    occ_patch_seed = int(occ_patch_base) + int(ep)
                rgb_used = apply_occlusion_rgb_chw(
                    rgb_t,
                    occ_mode,
                    fraction=occ_frac,
                    patch_seed=occ_patch_seed,
                )

                pos_b, quat_b = _ee_pose_b()
                gx = 1.0 if _gripper_state().startswith("o") else 0.0

                with torch.no_grad():
                    if harness is not None:
                        # Act/compute/query controller path (HRI pivot).
                        if policy_backend == "tiny":
                            assert tiny_model_cfg_obj is not None
                            state_t = torch.tensor(
                                list(pos_b) + [gx] + [0.0] * max(0, tiny_model_cfg_obj.state_dim - 4),
                                dtype=torch.float32,
                                device=device,
                            )[: tiny_model_cfg_obj.state_dim]
                            obs_h = {"image": rgb_used, "state": state_t, "occlusion_fraction": float(occ_frac)}
                        else:
                            st6 = torch.from_numpy(pack_state_xyz_quat(pos_b, quat_b)).to(device)
                            obs_h = {"image": rgb_used, "state": st6, "occlusion_fraction": float(occ_frac)}
                        decision = harness.decide(
                            obs_h, instruction=instruction, options=ep_options or None, oracle_answer=ep_oracle
                        )
                        chunk_phys = decision.chunk
                        ep_decisions.append((policy_tick_idx, decision))
                    elif policy_backend == "tiny":
                        assert pipeline is not None and tiny_model_cfg_obj is not None and ids is not None
                        state_t = torch.tensor(
                            list(pos_b) + [gx] + [0.0] * max(0, tiny_model_cfg_obj.state_dim - 4),
                            dtype=torch.float32,
                            device=device,
                        )[: tiny_model_cfg_obj.state_dim]
                        chunk = pipeline.predict_action_chunk(rgb_used, state_t, ids, attn)
                        chunk_phys = chunk
                        if action_stats is not None:
                            chunk_phys = action_stats.denormalize(chunk.detach().cpu()).to(device)
                    else:
                        assert smol_policy is not None
                        st6 = torch.from_numpy(pack_state_xyz_quat(pos_b, quat_b)).to(device)
                        if use_smol_ttc:
                            one_call_lat: List[Dict[str, Any]] = []
                            chunk_phys = smol_policy.predict_chunk_phys_ttc(
                                rgb_chw_float=rgb_used,
                                state6=st6,
                                instruction=instruction,
                                ttc_cfg_dict=ttc_cfg_dict,
                                latency_log=one_call_lat,
                            )
                            smol_latencies.extend(one_call_lat)
                        else:
                            chunk_phys = smol_policy.predict_chunk_phys(
                                rgb_chw_float=rgb_used,
                                state6=st6,
                                instruction=instruction,
                            )
                policy.set_chunk(chunk_phys)
                policy_tick_idx += 1

            # Mode hint (gripper vs translate) drives controller mode.
            try:
                if policy.current_mode_hint != getattr(controller, "_mode", "translate"):
                    controller.set_mode(policy.current_mode_hint)
            except Exception:
                pass

            controller.step(robot, physics_dt)
            sim.step(render=True)
            robot.update(physics_dt)
            if camera_sensor is not None:
                try:
                    camera_sensor.update(physics_dt)
                except Exception:
                    pass

        # Success: target z increased over baseline by >= lift_thresh.
        success = False
        z_after: Optional[float] = None
        try:
            for o in tracker.snapshot():
                if str(o.id) == leaf:
                    z_after = float(o.pose.position_m[2])
                    break
        except Exception:
            z_after = None
        if z0_target is not None and z_after is not None:
            success = (z_after - z0_target) >= lift_thresh

        elapsed = time.time() - ep_t0
        lat_totals = [float(r["total_ms"]) for r in smol_latencies if r.get("total_ms") is not None]
        result = {
            "episode_idx": int(ep),
            "target_prim": target_prim,
            "target_leaf": leaf,
            "instruction": instruction,
            "z0_target": float(z0_target) if z0_target is not None else None,
            "z_after": float(z_after) if z_after is not None else None,
            "success": bool(success),
            "steps": int(steps),
            "elapsed_s": float(elapsed),
            "observability_mode": observability_mode,
            "occlusion_mode": occ_mode,
            "occlusion_fraction": float(occ_frac),
            "policy_infer_calls": int(len(smol_latencies)),
            "policy_latency_median_ms": float(np.median(lat_totals)) if lat_totals else None,
            "policy_latency_p95_ms": float(np.quantile(np.asarray(lat_totals, dtype=np.float64), 0.95))
            if len(lat_totals) > 0
            else None,
        }
        ep_results.append(result)
        print(
            f"[eval][EP {ep}] success={success}  steps={steps}  z0={z0_target}  z_after={z_after}  elapsed={elapsed:.1f}s"
        )
        smol_latencies_all.extend(smol_latencies)

        # Backfill per-step calibration records with the episode outcome (sim correctness proxy).
        if harness is not None and ep_decisions:
            for st_idx, dec in ep_decisions:
                calib_records.append(
                    CalibrationRecord.from_decision(
                        dec,
                        run_id=run_id,
                        episode_idx=ep,
                        step_idx=st_idx,
                        occlusion_mode=occ_mode,
                        occlusion_fraction=float(occ_frac),
                        correct=bool(success),
                        success=bool(success),
                        target_leaf=leaf,
                        instruction=instruction,
                    )
                )

        # Reset robot for next episode.
        try:
            policy.reset()
            controller.reset(robot)
            controller.set_mode("translate")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    ttc_rows_all = tiny_ttc_latencies + smol_latencies_all

    summary = {
        "schema": "vla_lab_eval/v3",
        "task": "reach_to_grasp_lift",
        "eval_timestamp_unix": float(time.time()),
        "num_episodes": len(ep_results),
        "num_success": int(sum(1 for r in ep_results if r["success"])),
        "success_rate": float(sum(1 for r in ep_results if r["success"]) / max(1, len(ep_results))),
        "ckpt": str(ckpt_path),
        "policy_backend": policy_backend,
        "lerobot_dataset_root": str(lerobot_dataset_root) if lerobot_dataset_root else None,
        "observability_mode": observability_mode,
        "occlusion": {"mode": occ_mode, "fraction": float(occ_frac)},
        "ttc": ttc_cfg_dict,
        "ttc_aggregate": _aggregate_ttc_latencies(ttc_rows_all),
        "controller": controller_kind if use_controller else None,
        "controller_aggregate": harness.aggregate() if harness is not None else {},
        "allocator_fit": str(allocator_fit_path) if allocator_fit_path else None,
        "emit_calibration": str(emit_calibration) if emit_calibration else None,
        "provenance": {
            "git_commit": _try_git_commit(ROOT),
            "repo_root": str(ROOT.resolve()),
        },
        "episodes": ep_results,
        "results": ep_results,
        "config": cfg,
    }
    out_file = out_dir / f"results_{int(time.time())}.json"
    out_file.write_text(json.dumps(summary, indent=2))
    print(f"[eval] success rate = {summary['success_rate']:.3f}  ({summary['num_success']}/{summary['num_episodes']})")
    agg = summary.get("ttc_aggregate") or {}
    if agg.get("k_mean") is not None:
        print(
            f"[eval] TTC aggregate: k_mean={agg['k_mean']:.3f}  k_median={agg.get('k_median')}  "
            f"frac_gate_expanded={agg.get('frac_gate_expanded')}"
        )
    if summary.get("provenance", {}).get("git_commit"):
        print(f"[eval] git_commit={summary['provenance']['git_commit']}")
    print(f"[eval] wrote {out_file}")

    if harness is not None:
        agg = harness.aggregate()
        print(
            f"[eval] controller={controller_kind}  act={agg.get('frac_act')}  compute={agg.get('frac_compute')}  "
            f"query={agg.get('frac_query')}  query_rate={agg.get('query_rate')}  k_mean={agg.get('k_mean')}"
        )
    if emit_calibration and calib_records:
        cpath = Path(emit_calibration)
        write_jsonl(calib_records, cpath)
        print(f"[eval] wrote {len(calib_records)} calibration records to {cpath}")

    if pipeline is not None:
        pipeline.close()
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
