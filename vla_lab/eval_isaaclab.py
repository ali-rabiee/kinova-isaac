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
import random
import statistics
import subprocess
import time
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
        "knowno|insight|allocator|intent_allocator (default none = legacy TTC path).",
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
    parser.add_argument(
        "--policy-rate-hz",
        type=int,
        default=None,
        help="Override yaml eval.policy_rate_hz (should match the collection log rate).",
    )
    parser.add_argument(
        "--debug-actions",
        action="store_true",
        help="Log per-policy-tick action magnitudes and EE positions (JSONL in out_dir + console).",
    )
    parser.add_argument(
        "--replay-episode",
        type=str,
        default=None,
        help="Open-loop replay: path to a collected episode_XXXX dir; feeds its recorded "
        "action_from_prev sequence through the controller, bypassing the policy.",
    )

    # --- camera-set ablations (2026-07 wrist camera) ---------------------------
    parser.add_argument(
        "--cameras",
        type=str,
        default=None,
        help="Camera set fed to the policy: overhead|wrist|both (default: the checkpoint's "
        "camera_set). A strict subset of a multi-camera checkpoint runs the missing-view "
        "ablation via the camera_present mask.",
    )
    parser.add_argument(
        "--occlusion-camera",
        type=str,
        default=None,
        help="Which view the occlusion mask applies to: overhead|wrist|both (default overhead).",
    )
    # --- human-intent uncertainty + real-time feedback -------------------------
    parser.add_argument(
        "--instruction-ambiguity",
        type=str,
        default=None,
        help="none|half|full — replace episode instructions with colorless templates "
        "('Pick up the box.') never / ~50%% of episodes / always. Drives intent uncertainty.",
    )
    parser.add_argument(
        "--intent-every-n-ticks",
        type=int,
        default=None,
        help="Compute the intent/perception decomposition every N policy ticks "
        "(0 = off; default 1 when a controller or feedback channel is active, tiny backend).",
    )
    parser.add_argument(
        "--feedback-channel",
        type=str,
        default=None,
        help="none|scripted|console — real-time human interjections (vla_lab.feedback). "
        "'scripted' simulates a human with the quality knobs below; 'console' reads your keyboard.",
    )
    parser.add_argument("--feedback-accuracy", type=float, default=None, help="Simulated human: P(correct reference).")
    parser.add_argument("--feedback-latency-ticks", type=int, default=None, help="Simulated human: delivery delay (policy ticks).")
    parser.add_argument("--feedback-specificity", type=str, default=None, help="Simulated human: color|directional|vague.")
    parser.add_argument("--feedback-noise-rate", type=float, default=None, help="Simulated human: P(chatter instead of a correction).")
    parser.add_argument("--feedback-seed", type=int, default=None, help="Simulated human RNG seed.")
    parser.add_argument(
        "--no-fusion-verify",
        action="store_true",
        help="Trust-all ablation: apply corrections directly instead of verifying low-reliability ones.",
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
        action_scale: float = 1.0,
        max_action_dpos_m: Optional[float] = None,
        max_action_drot_rad: Optional[float] = None,
    ) -> None:
        self.physics_dt = float(physics_dt)
        self.policy_rate_hz = int(policy_rate_hz)
        self.device = device
        self.chunk_consume = max(1, int(chunk_consume))
        # Rescale for train/eval action-rate mismatch (train_rate / policy_rate),
        # applied to the 6 continuous dims so executed EE speed matches the demos.
        self.action_scale = float(action_scale)
        # Per-action safety clamps (per policy interval). Generous bounds that a
        # sane policy never hits; they exist to turn "seizure" into a logged event.
        self.max_action_dpos_m = max_action_dpos_m
        self.max_action_drot_rad = max_action_drot_rad
        self.clamp_count = 0

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
        """Provide a new (T, A) action chunk in *physical* (denormalised) units.

        Applies the train/eval rate rescale and the per-action safety clamps
        before the chunk is executed.
        """

        chunk = chunk_actions.clone()
        if abs(self.action_scale - 1.0) > 1e-9:
            chunk[:, :6] = chunk[:, :6] * self.action_scale
        if self.max_action_dpos_m is not None:
            norms = chunk[:, :3].norm(dim=-1, keepdim=True)
            factor = (float(self.max_action_dpos_m) / norms.clamp_min(1e-9)).clamp(max=1.0)
            self.clamp_count += int((factor < 1.0).sum().item())
            chunk[:, :3] = chunk[:, :3] * factor
        if self.max_action_drot_rad is not None:
            norms = chunk[:, 3:6].norm(dim=-1, keepdim=True)
            factor = (float(self.max_action_drot_rad) / norms.clamp_min(1e-9)).clamp(max=1.0)
            self.clamp_count += int((factor < 1.0).sum().item())
            chunk[:, 3:6] = chunk[:, 3:6] * factor
        self._chunk = chunk
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


class _FixedCmdProvider:
    """Input provider that returns a fixed per-step command (scripted pre-roll)."""

    def __init__(self, device: str) -> None:
        self.device = device
        self.cmd: Optional[Any] = None

    def advance(self) -> "torch.Tensor":
        import torch as _torch

        if self.cmd is None:
            return _torch.zeros(1, 7, device=self.device)
        return self.cmd


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

    # Open-loop replay mode: bypass the policy and execute a recorded episode's
    # action sequence (the gold-standard execution-path diagnostic).
    replay_dir: Optional[Path] = Path(args.replay_episode).resolve() if args.replay_episode else None
    replay_meta_rate: Optional[int] = None
    if replay_dir is not None:
        if not (replay_dir / "ticks.jsonl").exists():
            print(f"[eval] ERROR: --replay-episode {replay_dir} has no ticks.jsonl")
            return 2
        meta_path = replay_dir / "metadata.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                replay_meta_rate = int(meta.get("config", {}).get("log_rate_hz"))
            except Exception:
                replay_meta_rate = None
        eval_cfg["num_episodes"] = 1
        print(f"[eval][REPLAY] open-loop replay of {replay_dir} (policy is bypassed)")

    if replay_dir is not None:
        pass  # replay needs no policy/checkpoint
    elif policy_backend == "tiny":
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
    occ_camera = str(args.occlusion_camera if args.occlusion_camera is not None else occlusion_cfg.get("camera", "overhead")).lower()
    if occ_camera not in ("overhead", "wrist", "both"):
        print(f"[eval] ERROR: --occlusion-camera must be overhead|wrist|both, got {occ_camera!r}")
        return 2

    def _parse_camera_set(raw: Any) -> Optional[Tuple[str, ...]]:
        if raw is None:
            return None
        s = str(raw).lower().strip().replace(" ", "")
        if s in ("both", "overhead+wrist", "overhead,wrist", "wrist,overhead"):
            return ("overhead", "wrist")
        if s in ("overhead", "topdown", "external"):
            return ("overhead",)
        if s in ("wrist", "eye_in_hand"):
            return ("wrist",)
        raise ValueError(f"unknown --cameras value {raw!r} (use overhead|wrist|both)")

    eval_cameras_req = _parse_camera_set(args.cameras if args.cameras is not None else eval_cfg.get("cameras"))

    # Human-intent + feedback config (YAML `feedback:` / `intent:` blocks + CLI overrides).
    fb_cfg = dict(cfg.get("feedback") or {})
    if args.feedback_channel is not None:
        fb_cfg["channel"] = args.feedback_channel
    if args.feedback_accuracy is not None:
        fb_cfg["accuracy"] = float(args.feedback_accuracy)
    if args.feedback_latency_ticks is not None:
        fb_cfg["latency_ticks"] = int(args.feedback_latency_ticks)
    if args.feedback_specificity is not None:
        fb_cfg["specificity"] = str(args.feedback_specificity)
    if args.feedback_noise_rate is not None:
        fb_cfg["noise_rate"] = float(args.feedback_noise_rate)
    if args.feedback_seed is not None:
        fb_cfg["seed"] = int(args.feedback_seed)
    if args.no_fusion_verify:
        fb_cfg["fusion_verify"] = False
    fb_channel_kind = str(fb_cfg.get("channel", "none")).lower()

    intent_cfg = dict(cfg.get("intent") or {})
    if args.intent_every_n_ticks is not None:
        intent_cfg["every_n_ticks"] = int(args.intent_every_n_ticks)
    ambig_mode = str(
        args.instruction_ambiguity if args.instruction_ambiguity is not None else eval_cfg.get("instruction_ambiguity", "none")
    ).lower()
    if ambig_mode not in ("none", "half", "full"):
        print(f"[eval] ERROR: --instruction-ambiguity must be none|half|full, got {ambig_mode!r}")
        return 2

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
    model = None
    model_cameras: Tuple[str, ...] = ("overhead",)
    ckpt_action_rate: Optional[int] = None

    tiny_ttc_latencies: List[Dict[str, Any]] = []

    if replay_dir is not None:
        print("[eval][REPLAY] skipping policy load")
    elif policy_backend == "tiny":
        from vla_lab.dataset import ActionStats
        from vla_lab.models import TinyVLA, TinyVLAConfig
        from vla_lab.ttc import TTCConfig, TTCPipeline

        from vla_lab.checkpoint_utils import torch_load_checkpoint

        ckpt = torch_load_checkpoint(ckpt_path, map_location=device)
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
        model_cameras = tuple(ckpt.get("camera_set") or tiny_model_cfg_obj.cameras)
        ckpt_action_rate = ckpt.get("action_rate_hz")
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

    # ------------------------------------------------------------------
    # Camera-set resolution: which streams the MODEL expects (checkpoint)
    # vs which views EVAL feeds it. A strict subset runs the missing-view
    # ablation (the dropped stream is masked via camera_present).
    # ------------------------------------------------------------------
    if policy_backend == "smolvla" and replay_dir is None:
        model_cameras = _parse_camera_set(cfg.get("smolvla_cameras", "overhead")) or ("overhead",)
    eval_cameras: Tuple[str, ...] = eval_cameras_req or model_cameras
    bad = [c for c in eval_cameras if c not in model_cameras]
    if bad and replay_dir is None:
        print(
            f"[eval] ERROR: --cameras {list(eval_cameras)} is not a subset of the model's "
            f"camera set {list(model_cameras)} (a single-camera checkpoint cannot consume the other view)."
        )
        simulation_app.close()
        return 2
    if replay_dir is None:
        print(f"[eval] cameras: model={list(model_cameras)}  eval={list(eval_cameras)}  occlusion_camera={occ_camera}")

    # ------------------------------------------------------------------
    # Real-time feedback channel + intent estimator (2026-07 human-intent axis)
    # ------------------------------------------------------------------
    from vla_lab.feedback import (
        ConsoleFeedbackChannel,
        FusionConfig,
        ReliabilityTracker,
        ScriptedFeedbackChannel,
        SimulatedHuman,
        SimulatedHumanConfig,
        fuse,
        parse_utterance,
    )
    from vla_lab.intent import IntentEstimator, cross_view_disagreement

    feedback_on = fb_channel_kind not in ("", "none", "off") and replay_dir is None
    sim_human_cfg = SimulatedHumanConfig(
        accuracy=float(fb_cfg.get("accuracy", 0.95)),
        latency_ticks=int(fb_cfg.get("latency_ticks", 3)),
        specificity=str(fb_cfg.get("specificity", "color")),
        noise_rate=float(fb_cfg.get("noise_rate", 0.0)),
        react_cos=float(fb_cfg.get("react_cos", 0.85)),
        react_dist_m=float(fb_cfg.get("react_dist_m", 0.30)),
        min_gap_ticks=int(fb_cfg.get("min_gap_ticks", 20)),
        query_latency_s=float(fb_cfg.get("query_latency_s", alloc_cfg.get("query_latency_s", 3.0))),
        seed=int(fb_cfg.get("seed", 0)),
    )
    fusion_cfg = FusionConfig(
        trust_threshold=float(fb_cfg.get("trust_threshold", 0.55)),
        ignore_threshold=float(fb_cfg.get("ignore_threshold", 0.30)),
        verify_enabled=bool(fb_cfg.get("fusion_verify", True)),
    )
    reliability = ReliabilityTracker(
        alpha0=float(fb_cfg.get("reliability_alpha0", 2.0)),
        beta0=float(fb_cfg.get("reliability_beta0", 1.0)),
    )
    console_channel = ConsoleFeedbackChannel() if (feedback_on and fb_channel_kind in ("console", "stdin", "keyboard")) else None
    fb_channel = console_channel  # scripted channels are rebuilt per episode (they need the true target)

    class _LiveHumanQueryInterface:
        """Routes robot-initiated queries to the CURRENT feedback channel's human."""

        name = "feedback_channel"

        def __init__(self) -> None:
            self.channel = None

        def ask(self, question, options=None):
            if self.channel is not None and getattr(self.channel, "supports_ask", False):
                return self.channel.ask(question, options)
            raise RuntimeError("feedback channel cannot answer queries")

    live_qi = _LiveHumanQueryInterface() if feedback_on else None
    nudge_step_m = float(fb_cfg.get("nudge_step_m", 0.02))
    nudge_horizon_ticks = int(fb_cfg.get("nudge_horizon_ticks", 5))
    auto_resume_ticks = int(fb_cfg.get("auto_resume_ticks", 45))
    if feedback_on:
        print(f"[eval] feedback channel={fb_channel_kind!r}  human={sim_human_cfg.to_dict() if fb_channel_kind == 'scripted' else 'live'}")
        print(f"[eval] fusion={fusion_cfg.to_dict()}")

    intent_every = int(intent_cfg.get("every_n_ticks", 1 if (use_controller or feedback_on) else 0))
    intent_estimator = None
    if intent_every > 0 and policy_backend == "tiny" and replay_dir is None and tokenizer is not None:
        intent_estimator = IntentEstimator(
            model,
            tokenizer,
            device=device,
            temperature=float(intent_cfg.get("temperature", 0.02)),
            lexical_prior_weight=float(intent_cfg.get("lexical_prior_weight", 0.75)),
            geometric_weight=float(intent_cfg.get("geometric_weight", 0.5)),
        )
        print(f"[eval] intent estimator ON (every {intent_every} tick(s))  ambiguity={ambig_mode}")

    # Build the act/compute/query controller harness over the loaded policy (if requested).
    harness = None
    if use_controller and replay_dir is not None:
        print("[eval][REPLAY] ignoring allocation controller during replay")
        use_controller = False
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
            query_interface=live_qi,  # None unless a feedback channel is active
        )
        print(f"[eval] controller={controller_kind!r}  fit={allocator_fit_path}  emit_calibration={emit_calibration}")

    calib_records: List[Any] = []
    run_id = f"{policy_backend}_{controller_kind}_{int(time.time())}"

    infer_image_size = int(tiny_model_cfg_obj.image_size) if tiny_model_cfg_obj is not None else 224

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
    # Wrist camera prim (2026-07): only when the eval camera set consumes it.
    use_wrist = enable_cameras and ("wrist" in eval_cameras) and replay_dir is None
    if use_wrist:
        from environments.reach_to_grasp_VLA.config import DEFAULT_WRIST_CAMERA
        from environments.utils.camera import create_wrist_camera

        create_wrist_camera(DEFAULT_WRIST_CAMERA)
        print(f"[eval] wrist camera created at {DEFAULT_WRIST_CAMERA.prim_path}")

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

    # Camera sensor (only if enabled). `spawn=None` attaches to the prim that
    # `create_topdown_camera` already made (same pattern as vla_v1) — without
    # it CameraCfg validation fails and eval silently runs on black images.
    camera_sensor: Optional[Camera] = None
    if enable_cameras and DEFAULT_TOP_DOWN_CAMERA is not None:
        try:
            cam_cfg = CameraCfg(
                prim_path=DEFAULT_TOP_DOWN_CAMERA.prim_path,
                offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
                spawn=None,
                width=DEFAULT_TOP_DOWN_CAMERA.resolution[0],
                height=DEFAULT_TOP_DOWN_CAMERA.resolution[1],
                update_period=0.0,
                data_types=["rgb"],
            )
            camera_sensor = Camera(cfg=cam_cfg)
            print(f"[eval] camera sensor attached: {cam_cfg.width}x{cam_cfg.height} at {DEFAULT_TOP_DOWN_CAMERA.prim_path}")
        except Exception as e:
            print(f"[eval] ERROR: could not create Camera sensor: {e}; policy will see BLACK images.")
            camera_sensor = None

    wrist_sensor: Optional[Camera] = None
    if use_wrist:
        try:
            wrist_cam_cfg = CameraCfg(
                prim_path=DEFAULT_WRIST_CAMERA.prim_path,
                offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
                spawn=None,
                width=DEFAULT_WRIST_CAMERA.resolution[0],
                height=DEFAULT_WRIST_CAMERA.resolution[1],
                update_period=0.0,
                data_types=["rgb"],
            )
            wrist_sensor = Camera(cfg=wrist_cam_cfg)
            print(f"[eval] wrist sensor attached: {wrist_cam_cfg.width}x{wrist_cam_cfg.height}")
        except Exception as e:
            print(f"[eval] ERROR: could not create wrist Camera sensor: {e}; wrist view will be BLACK.")
            wrist_sensor = None

    # Reset sim + robot
    sim.reset()
    if camera_sensor is not None:
        try:
            camera_sensor.reset()
        except Exception:
            pass
    if wrist_sensor is not None:
        try:
            wrist_sensor.reset()
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
    if args.policy_rate_hz is not None:
        policy_rate_hz = int(args.policy_rate_hz)
    # New contract field (2026-07): checkpoints trained after the camera-set
    # change carry the collection log rate; warn on mismatch (the YAML
    # train_action_rate_hz rescale below still applies either way).
    if ckpt_action_rate is not None and int(ckpt_action_rate) != int(policy_rate_hz) and replay_dir is None:
        print(
            f"[eval] WARNING: checkpoint action_rate_hz={ckpt_action_rate} != eval policy_rate_hz={policy_rate_hz}; "
            "set eval.policy_rate_hz to the training sessions' log rate."
        )

    # Actions are EE deltas per 1/log_rate_hz of the *collection* session. If
    # eval runs at a different rate, rescale the deltas so EE speed matches the
    # demos (prefer matching the rates outright).
    action_scale = 1.0
    if replay_meta_rate is not None:
        policy_rate_hz = int(replay_meta_rate)
        print(f"[eval][REPLAY] policy_rate_hz={policy_rate_hz} (from session metadata)")
    else:
        train_rate = eval_cfg.get("train_action_rate_hz")
        if train_rate:
            action_scale = float(train_rate) / float(policy_rate_hz)
            if abs(action_scale - 1.0) > 1e-6:
                print(
                    f"[eval] WARNING: train_action_rate_hz={train_rate} != policy_rate_hz={policy_rate_hz}; "
                    f"scaling action deltas by {action_scale:.3f} to preserve demo EE speed. "
                    "Prefer setting policy_rate_hz to the collection log rate."
                )

    policy = PolicyInputProvider(
        physics_dt=physics_dt,
        policy_rate_hz=policy_rate_hz,
        device=str(device),
        chunk_consume=int(eval_cfg.get("chunk_consume", 1)),
        action_scale=action_scale,
        max_action_dpos_m=(
            float(eval_cfg["max_action_dpos_m"]) if eval_cfg.get("max_action_dpos_m") is not None else None
        ),
        max_action_drot_rad=(
            float(eval_cfg["max_action_drot_rad"]) if eval_cfg.get("max_action_drot_rad") is not None else None
        ),
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
        # Proximal finger joints only — tips stay near 0 and fingers stall well
        # below the commanded close position when wrapped around an object
        # (same fix as data_collection/core/logger.py, keeps train/eval aligned).
        try:
            joint_pos = robot.data.joint_pos[0]
            names = robot.data.joint_names
            idxs = [i for i, n in enumerate(names) if ("finger" in str(n) and "finger_tip" not in str(n))]
            if idxs:
                vals = torch.tensor([joint_pos[i] for i in idxs], dtype=torch.float32)
                return "close" if vals.mean().item() > 0.2 else "open"
        except Exception:
            pass
        return "open"

    def _grab_rgb_resized(sensor) -> Optional[torch.Tensor]:
        """Sensor RGB -> (3, S, S) float [0,1] on device; None when unavailable."""
        if sensor is None:
            return None
        try:
            data = sensor.data
            rgb = data.output.get("rgb") if data.output is not None else None
        except Exception:
            return None
        if rgb is None:
            return None
        t = rgb[0] if rgb.dim() == 4 else rgb
        t = t.float()
        if t.max() > 1.5:
            t = t / 255.0
        if t.dim() == 3 and t.shape[-1] == 4:
            t = t[..., :3]
        if t.dim() == 3 and t.shape[-1] == 3:
            t = t.permute(2, 0, 1).contiguous()
        return torch.nn.functional.interpolate(
            t.unsqueeze(0).to(device), size=(infer_image_size, infer_image_size), mode="bilinear", align_corners=False
        )[0]

    def _object_pos_b_by_color(color_by_leaf: Dict[str, str]) -> Dict[str, Tuple[float, float, float]]:
        """Live object positions (PhysX tracker) in the robot base frame, keyed by color."""
        out: Dict[str, Tuple[float, float, float]] = {}
        try:
            root = robot.data.root_pose_w
            for o in tracker.snapshot():
                cname = color_by_leaf.get(str(o.id))
                if cname is None:
                    continue
                pw = torch.tensor(
                    [[float(o.pose.position_m[0]), float(o.pose.position_m[1]), float(o.pose.position_m[2])]],
                    device=root.device,
                )
                qw = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=root.device)
                pb, _ = subtract_frame_transforms(root[:, 0:3], root[:, 3:7], pw, qw)
                out[cname] = (float(pb[0, 0]), float(pb[0, 1]), float(pb[0, 2]))
        except Exception:
            pass
        return out

    # ------------------------------------------------------------------
    # Replay actions + per-tick debug logging
    # ------------------------------------------------------------------
    replay_actions: Optional[List[Tuple[float, ...]]] = None
    replay_targets: Optional[List[Tuple[float, float, float]]] = None
    replay_instruction: Optional[str] = None
    if replay_dir is not None:
        from vla_lab.dataset import _parse_instruction, _parse_ticks

        _rticks = _parse_ticks(replay_dir / "ticks.jsonl")
        replay_actions = [t.action_from_prev for t in _rticks if t.action_from_prev is not None]
        replay_targets = [t.ee_pos_b for t in _rticks if t.action_from_prev is not None]
        replay_instruction, _ = _parse_instruction(replay_dir)
        if not replay_actions:
            print(f"[eval] ERROR: no action_from_prev records in {replay_dir}/ticks.jsonl")
            simulation_app.close()
            return 2
        n_grip = sum(1 for a in replay_actions if abs(float(a[6])) > 0.5)
        print(f"[eval][REPLAY] {len(replay_actions)} recorded actions ({n_grip} gripper)  instruction={replay_instruction!r}")

    debug_fp = None
    if bool(args.debug_actions):
        debug_path = out_dir / f"debug_actions_{int(time.time())}.jsonl"
        debug_fp = open(debug_path, "a", buffering=1)
        print(f"[eval] --debug-actions: writing per-tick logs to {debug_path}")

    def _debug_log_tick(*, ep: int, tick: int, chunk_phys: Any, source: str, chunk_norm: Any = None) -> None:
        if debug_fp is None:
            return
        try:
            pos_now, _ = _ee_pose_b()
            ch = chunk_phys.detach().to("cpu")
            dp = ch[:, :3].norm(dim=-1)
            dr = ch[:, 3:6].norm(dim=-1)
            rec: Dict[str, Any] = {
                "ep": int(ep),
                "tick": int(tick),
                "source": source,
                "ee_pos_b": [round(float(v), 4) for v in pos_now],
                "dpos_mm": {"mean": round(float(dp.mean()) * 1000, 2), "max": round(float(dp.max()) * 1000, 2)},
                "drot_rad_max": round(float(dr.max()), 4),
                "gripper": {"min": round(float(ch[:, 6].min()), 3), "max": round(float(ch[:, 6].max()), 3)},
                "clamp_count": int(policy.clamp_count),
            }
            if chunk_norm is not None:
                rec["norm_absmax"] = round(float(chunk_norm.detach().abs().max()), 3)
            debug_fp.write(json.dumps(rec) + "\n")
            if tick < 10 or tick % 25 == 0:
                print(
                    f"[dbg][EP {ep} t{tick}] ee=({pos_now[0]:.3f},{pos_now[1]:.3f},{pos_now[2]:.3f})  "
                    f"|dpos|max={float(dp.max()) * 1000:.1f}mm  |drot|max={float(dr.max()):.4f}rad  "
                    f"g=[{float(ch[:, 6].min()):.2f},{float(ch[:, 6].max()):.2f}]  clamps={policy.clamp_count}"
                )
        except Exception:
            pass

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
        # Instruction-ambiguity axis (2026-07 intent uncertainty): with prob. per
        # `ambig_mode`, replace the instruction with a colorless template. The
        # human still WANTS the scheduled target — only the instruction
        # underspecifies it, so resolving it requires the human (query/feedback).
        instr_ambiguous_flag = False
        if replay_dir is None and args.language is None and ambig_mode != "none":
            _arng = random.Random(int(eval_cfg.get("language_seed", 1337)) * 7919 + ep)
            if ambig_mode == "full" or (_arng.random() < 0.5):
                instruction = _arng.choice(["Pick up the box.", "Grab a box and lift it.", "Pick it up."])
                instr_ambiguous_flag = True

        if replay_dir is not None:
            instruction = replay_instruction or instruction
        instruction_initial = str(instruction)  # feedback may refine `instruction` mid-episode
        print(f"[eval][EP {ep}] target={leaf!r}  instruction={instruction!r}"
              + ("  [AMBIGUOUS]" if instr_ambiguous_flag else ""))

        # Scene candidates by color (used by the allocator's query options, the
        # intent estimator, and the feedback channel).
        scene_colors: List[str] = []
        color_by_leaf: Dict[str, str] = {}
        for sp in sorted([str(p) for p in spawned_paths]):
            desc, c, _bidx = _box_desc_from_prim(sp, BOX_COLORS)
            cname = c or desc
            scene_colors.append(cname)
            color_by_leaf[sp.split("/")[-1]] = cname
        true_color = color_by_leaf.get(leaf, leaf)

        # Per-episode allocator state: query options (object descriptions) + the oracle target.
        ep_decisions: List[Tuple[int, Any]] = []
        policy_tick_idx = 0
        ep_options: List[str] = list(scene_colors)
        ep_oracle: Optional[str] = true_color
        if harness is not None:
            harness.set_oracle(ep_oracle)

        ids = attn = None
        if policy_backend == "tiny" and tokenizer is not None:
            ids, attn = tokenizer.encode(instruction)
            ids = ids.to(device)
            attn = attn.to(device)

        def _set_instruction(new_instr: str) -> None:
            """Apply a mid-episode instruction refinement (re-encodes tokens for tiny)."""
            nonlocal instruction, ids, attn
            instruction = str(new_instr)
            if policy_backend == "tiny" and tokenizer is not None:
                _i, _a = tokenizer.encode(instruction)
                ids, attn = _i.to(device), _a.to(device)

        # Per-episode feedback state (reliability persists ACROSS episodes).
        ep_feedback_events: List[Dict[str, Any]] = []
        ep_intent_entropies: List[float] = []
        if feedback_on and fb_channel_kind == "scripted":
            ep_sim_human = SimulatedHuman(
                sim_human_cfg,
                true_target_label=true_color,
                candidate_labels=scene_colors,
                episode_idx=ep,
            )
            fb_channel = ScriptedFeedbackChannel(ep_sim_human)
        if live_qi is not None:
            live_qi.channel = fb_channel
        paused_ticks_left = 0
        pending_nudge: Optional[Tuple[Tuple[float, float, float], float]] = None
        nudge_ticks_left = 0
        last_intent_est = None
        last_xview: Optional[float] = None

        # Reset the robot to its default state, then settle with the controller
        # actively holding the arm (mirrors vla_v1's settle loop). A bare
        # `sim.step()` loop leaves the arm without fresh drive targets or
        # gravity compensation, so it sags/swings during the settle and the
        # first policy step then fights a large stale-orientation IK error —
        # which looks like violent flailing the moment the policy loop starts.
        root_state = robot.data.default_root_state.clone()
        robot.write_root_pose_to_sim(root_state[:, :7])
        robot.write_root_velocity_to_sim(root_state[:, 7:])
        robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
        robot.reset()
        policy.reset()
        # Step once so robot.data (body poses, jacobians) reflects the written
        # state before the controller captures its hold orientation — resetting
        # the controller on stale buffers makes the settle drag the arm toward a
        # bogus pre-reset pose (vla_v1 does the same single step).
        sim.step(render=False)
        robot.update(physics_dt)
        controller.reset(robot)
        controller.set_mode("translate")
        for settle_i in range(60):
            if not simulation_app.is_running():
                break
            controller.step(robot, physics_dt)
            sim.step(render=False)
            robot.update(physics_dt)
        # Re-capture IK state + hold orientation from the *settled* pose.
        controller.reset(robot)
        controller.set_mode("translate")

        # Scripted pre-roll: drive the EE to the demos' start pose before the
        # policy (or replay) takes over. The collection pipeline's setup leaves
        # the arm at a different pose than a plain settle, so without this the
        # first observation is out of distribution and open-loop replay carries
        # a constant offset.
        start_pos_cfg = eval_cfg.get("start_ee_pos_b")
        if start_pos_cfg is not None:
            tgt = [float(v) for v in start_pos_cfg]
            preroll = _FixedCmdProvider(device=str(device))
            controller.set_input_provider(preroll)
            pre_steps = 0
            for _ in range(900):  # <= ~3.75 s sim at 240 Hz
                if not simulation_app.is_running():
                    break
                cur, _ = _ee_pose_b()
                err = [tgt[k] - cur[k] for k in range(3)]
                dist = math.sqrt(sum(e * e for e in err))
                if dist < 0.005:
                    break
                step_m = min(0.002, dist)  # <= ~0.5 m/s
                cmd = torch.zeros(1, 7, device=device)
                for k in range(3):
                    cmd[0, k] = step_m * err[k] / dist
                preroll.cmd = cmd
                controller.step(robot, physics_dt)
                sim.step(render=False)
                robot.update(physics_dt)
                pre_steps += 1
            controller.set_input_provider(policy)
            controller.reset(robot)
            controller.set_mode("translate")
            if debug_fp is not None:
                print(f"[dbg][EP {ep}] pre-roll {pre_steps} steps to start_ee_pos_b={tgt}")

        # A few rendered hold steps so the camera has fresh frames at tick 0.
        for _ in range(4):
            if not simulation_app.is_running():
                break
            controller.step(robot, physics_dt)
            sim.step(render=True)
            robot.update(physics_dt)
            if camera_sensor is not None:
                try:
                    camera_sensor.update(physics_dt)
                except Exception:
                    pass
            if wrist_sensor is not None:
                try:
                    wrist_sensor.update(physics_dt)
                except Exception:
                    pass
        if debug_fp is not None:
            _pos_dbg, _ = _ee_pose_b()
            print(f"[dbg][EP {ep}] settled EE pos_b=({_pos_dbg[0]:.3f}, {_pos_dbg[1]:.3f}, {_pos_dbg[2]:.3f})")

        # Capture initial target z for success check.
        snap = tracker.snapshot()
        z0_target = None
        for o in snap:
            if str(o.id) == leaf:
                z0_target = float(o.pose.position_m[2])
                break

        steps = 0
        ep_t0 = time.time()
        ep_clamp_start = int(policy.clamp_count)  # provider total is cumulative across episodes
        smol_latencies: List[Dict[str, Any]] = []
        replay_idx = 0
        replay_errs: List[float] = []
        while simulation_app.is_running() and steps < max_steps_ep:
            steps += 1

            # Refresh policy chunk at the policy rate.
            if replay_actions is not None and policy.needs_new_chunk():
                # Open-loop replay: feed the next recorded action, tracking how
                # far execution drifts from the recorded EE trajectory.
                if replay_idx >= len(replay_actions):
                    break
                if replay_idx > 0 and replay_targets is not None:
                    _cur, _ = _ee_pose_b()
                    replay_errs.append(math.dist(_cur, replay_targets[replay_idx - 1]))
                chunk_phys = torch.tensor(replay_actions[replay_idx], dtype=torch.float32, device=device).view(1, 7)
                replay_idx += 1
                policy.set_chunk(chunk_phys)
                policy_tick_idx += 1
                _debug_log_tick(ep=ep, tick=policy_tick_idx - 1, chunk_phys=chunk_phys, source="replay")
            elif policy.needs_new_chunk():
                # Build observation tensors.
                rgb_t = _grab_rgb_resized(camera_sensor)
                if rgb_t is None:
                    rgb_t = torch.zeros(3, infer_image_size, infer_image_size, device=device)

                occ_patch_seed = None
                if occ_patch_base is not None and str(occ_mode).lower() == "random_patch":
                    occ_patch_seed = int(occ_patch_base) + int(ep)
                rgb_used = apply_occlusion_rgb_chw(
                    rgb_t,
                    occ_mode if occ_camera in ("overhead", "both") else "none",
                    fraction=occ_frac,
                    patch_seed=occ_patch_seed,
                )

                # Wrist view (2026-07): grabbed only when the eval camera set uses it.
                wrist_used: Optional[torch.Tensor] = None
                if wrist_sensor is not None:
                    wrist_t = _grab_rgb_resized(wrist_sensor)
                    if wrist_t is None:
                        wrist_t = torch.zeros(3, infer_image_size, infer_image_size, device=device)
                    wrist_used = apply_occlusion_rgb_chw(
                        wrist_t,
                        occ_mode if occ_camera in ("wrist", "both") else "none",
                        fraction=occ_frac,
                        patch_seed=occ_patch_seed,
                    )
                camera_present_t: Optional[torch.Tensor] = None
                if policy_backend == "tiny" and len(model_cameras) > 1:
                    camera_present_t = torch.tensor(
                        [1.0 if c in eval_cameras else 0.0 for c in model_cameras], device=device
                    )

                pos_b, quat_b = _ee_pose_b()
                gx = 1.0 if _gripper_state().startswith("o") else 0.0
                state_t: Optional[torch.Tensor] = None
                if policy_backend == "tiny" and tiny_model_cfg_obj is not None:
                    state_t = torch.tensor(
                        list(pos_b) + [gx] + [0.0] * max(0, tiny_model_cfg_obj.state_dim - 4),
                        dtype=torch.float32,
                        device=device,
                    )[: tiny_model_cfg_obj.state_dim]

                obj_pos_by_color: Dict[str, Tuple[float, float, float]] = {}
                if feedback_on or intent_estimator is not None:
                    obj_pos_by_color = _object_pos_b_by_color(color_by_leaf)

                # Intent/perception decomposition (tiny backend).
                if (
                    intent_estimator is not None
                    and state_t is not None
                    and policy_tick_idx % max(1, intent_every) == 0
                ):
                    last_intent_est = intent_estimator.estimate(
                        instruction=instruction,
                        candidate_labels=scene_colors,
                        image=rgb_used if "overhead" in model_cameras else None,
                        image_wrist=wrist_used if "wrist" in model_cameras else None,
                        state=state_t,
                        camera_present=camera_present_t,
                        object_pos_b=obj_pos_by_color,
                        ee_pos_b=pos_b,
                    )
                    ep_intent_entropies.append(float(last_intent_est.entropy))
                    if (
                        "overhead" in model_cameras
                        and "wrist" in model_cameras
                        and wrist_used is not None
                        and ids is not None
                    ):
                        last_xview = cross_view_disagreement(
                            model,
                            image=rgb_used,
                            image_wrist=wrist_used,
                            state=state_t,
                            lang_ids=ids,
                            lang_mask=attn,
                        )

                # Two-way feedback: deliver interjections, parse, fuse by reliability.
                if feedback_on and fb_channel is not None:
                    fb_channel.observe(
                        tick_idx=policy_tick_idx, ee_pos_b=pos_b, object_pos_b=obj_pos_by_color
                    )
                    for inter in fb_channel.poll():
                        ref = parse_utterance(inter.text, scene_colors)
                        fd = fuse(
                            ref,
                            reliability=reliability.mean,
                            intent_posterior=(last_intent_est.posterior if last_intent_est else None),
                            cfg=fusion_cfg,
                        )
                        ev: Dict[str, Any] = {
                            "tick": int(policy_tick_idx),
                            "text": inter.text,
                            "kind": ref.kind,
                            "fusion": fd.action,
                            "reason": fd.reason,
                            "intended_correct": inter.intended_correct,
                            "reliability_before": round(reliability.mean, 4),
                        }
                        act_now = fd.action
                        # Verification: re-ask WHICH target through the same channel
                        # (counts as a query; covers low-reliability corrections).
                        if (
                            act_now == "verify"
                            and ref.kind in ("target", "avoid")
                            and getattr(fb_channel, "supports_ask", False)
                        ):
                            va = fb_channel.ask(
                                "I want to double-check — which object should I pick up?", scene_colors
                            )
                            ev["verify_answer"] = va.text
                            ev["verify_latency_s"] = float(va.latency_s)
                            if str(va.text).lower() == str(ref.target_label or "").lower():
                                act_now = "apply"
                            elif str(va.text) in scene_colors:
                                ref = type(ref)(kind="target", target_label=str(va.text), raw_text=inter.text)
                                act_now = "apply"
                            else:
                                act_now = "ignore"
                        applied_target: Optional[str] = None
                        if act_now == "apply":
                            if ref.kind == "target" and ref.target_label:
                                _set_instruction(f"Pick up the {ref.target_label} box.")
                                applied_target = ref.target_label
                            elif ref.kind == "avoid" and ref.target_label:
                                if getattr(fb_channel, "supports_ask", False):
                                    va = fb_channel.ask(
                                        "Understood, not that one — which object should I pick up?",
                                        [c for c in scene_colors if c != ref.target_label],
                                    )
                                    ev["verify_answer"] = va.text
                                    if str(va.text) in scene_colors:
                                        _set_instruction(f"Pick up the {va.text} box.")
                                        applied_target = str(va.text)
                                else:
                                    _set_instruction("Pick up the box.")  # forces intent uncertainty
                            elif ref.kind == "nudge" and ref.direction is not None:
                                pending_nudge = (ref.direction, nudge_step_m * float(ref.magnitude) * float(fd.scale))
                                nudge_ticks_left = nudge_horizon_ticks
                            elif ref.kind == "stop":
                                paused_ticks_left = auto_resume_ticks
                            elif ref.kind == "resume":
                                paused_ticks_left = 0
                        ev["applied_target"] = applied_target
                        # Update reliability AFTER the decision (sim grading; on the
                        # real robot this update comes from verified outcomes).
                        if inter.intended_correct is not None:
                            reliability.update(bool(inter.intended_correct))
                        ev["reliability_after"] = round(reliability.mean, 4)
                        ep_feedback_events.append(ev)
                        print(
                            f"[eval][EP {ep} t{policy_tick_idx}] feedback {inter.text!r} -> {ref.kind}"
                            f" ({ev['fusion']}/{act_now}); reliability={reliability.mean:.2f}"
                        )

                chunk_norm_dbg = None
                if paused_ticks_left > 0:
                    paused_ticks_left -= 1
                    chunk_phys = torch.zeros(1, 7, device=device)
                else:
                    with torch.no_grad():
                        if harness is not None:
                            # Act/compute/query controller path (HRI pivot).
                            if policy_backend == "tiny":
                                assert state_t is not None
                                obs_h = {"image": rgb_used, "state": state_t, "occlusion_fraction": float(occ_frac)}
                            else:
                                st6 = torch.from_numpy(pack_state_xyz_quat(pos_b, quat_b)).to(device)
                                obs_h = {"image": rgb_used, "state": st6, "occlusion_fraction": float(occ_frac)}
                            if wrist_used is not None:
                                obs_h["image_wrist"] = wrist_used
                            if camera_present_t is not None:
                                obs_h["camera_present"] = camera_present_t
                            if last_intent_est is not None:
                                obs_h.update(last_intent_est.features())
                            if last_xview is not None:
                                obs_h["cross_view_disagreement"] = float(last_xview)
                            decision = harness.decide(
                                obs_h, instruction=instruction, options=ep_options or None, oracle_answer=ep_oracle
                            )
                            chunk_phys = decision.chunk
                            ep_decisions.append((policy_tick_idx, decision))
                        elif policy_backend == "tiny":
                            assert pipeline is not None and state_t is not None and ids is not None
                            chunk = pipeline.predict_action_chunk(
                                rgb_used, state_t, ids, attn,
                                image_wrist=wrist_used, camera_present=camera_present_t,
                            )
                            chunk_norm_dbg = chunk
                            chunk_phys = chunk
                            if action_stats is not None:
                                chunk_phys = action_stats.denormalize(chunk.detach().cpu()).to(device)
                        else:
                            assert smol_policy is not None
                            st6 = torch.from_numpy(pack_state_xyz_quat(pos_b, quat_b)).to(device)
                            smol_wrist = wrist_used if "wrist" in eval_cameras else None
                            if use_smol_ttc:
                                one_call_lat: List[Dict[str, Any]] = []
                                chunk_phys = smol_policy.predict_chunk_phys_ttc(
                                    rgb_chw_float=rgb_used,
                                    state6=st6,
                                    instruction=instruction,
                                    ttc_cfg_dict=ttc_cfg_dict,
                                    latency_log=one_call_lat,
                                    wrist_rgb_chw_float=smol_wrist,
                                )
                                smol_latencies.extend(one_call_lat)
                            else:
                                chunk_phys = smol_policy.predict_chunk_phys(
                                    rgb_chw_float=rgb_used,
                                    state6=st6,
                                    instruction=instruction,
                                    wrist_rgb_chw_float=smol_wrist,
                                )
                    # Directional nudge: reliability-scaled bias on upcoming deltas.
                    if nudge_ticks_left > 0 and pending_nudge is not None:
                        vec, gain = pending_nudge
                        nvec = torch.tensor(vec, device=chunk_phys.device, dtype=chunk_phys.dtype)
                        chunk_phys = chunk_phys.clone()
                        chunk_phys[:, :3] = chunk_phys[:, :3] + float(gain) * nvec
                        nudge_ticks_left -= 1
                policy.set_chunk(chunk_phys)
                policy_tick_idx += 1
                _debug_log_tick(
                    ep=ep,
                    tick=policy_tick_idx - 1,
                    chunk_phys=chunk_phys,
                    source=policy_backend if paused_ticks_left == 0 else "paused",
                    chunk_norm=chunk_norm_dbg,
                )

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
            if wrist_sensor is not None:
                try:
                    wrist_sensor.update(physics_dt)
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
            "instruction_ambiguous": bool(instr_ambiguous_flag),
            "instruction_initial": instruction_initial,
            "cameras_eval": list(eval_cameras),
        }
        if ep_intent_entropies:
            result["intent"] = {
                "n_estimates": len(ep_intent_entropies),
                "mean_entropy": float(statistics.mean(ep_intent_entropies)),
                "max_entropy": float(max(ep_intent_entropies)),
            }
        if feedback_on:
            result["feedback"] = {
                "events": ep_feedback_events,
                "n_interjections": len(ep_feedback_events),
                "n_applied": sum(1 for e in ep_feedback_events if e.get("applied_target") or e.get("fusion") == "apply"),
                "n_verified": sum(1 for e in ep_feedback_events if "verify_answer" in e),
                "n_ignored": sum(1 for e in ep_feedback_events if e.get("fusion") == "ignore"),
                "reliability_after_episode": float(reliability.mean),
            }
        if replay_actions is not None:
            result["replay_episode"] = str(replay_dir)
            result["replay_actions_executed"] = int(replay_idx)
            if replay_errs:
                result["replay_track_err_mean_m"] = float(statistics.mean(replay_errs))
                result["replay_track_err_max_m"] = float(max(replay_errs))
                print(
                    f"[eval][REPLAY] executed {replay_idx}/{len(replay_actions)} actions  "
                    f"EE tracking error mean={statistics.mean(replay_errs) * 1000:.1f}mm  "
                    f"max={max(replay_errs) * 1000:.1f}mm"
                )
        ep_clamps = int(policy.clamp_count) - ep_clamp_start
        if ep_clamps:
            result["action_clamp_count"] = ep_clamps
            print(f"[eval][EP {ep}] WARNING: safety clamps triggered {ep_clamps}x this episode (see max_action_dpos_m/max_action_drot_rad)")
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

        # Robot/controller are reset at the start of the next episode (see the
        # powered-settle block above), so nothing to do here.

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
        "policy_backend": policy_backend if replay_dir is None else "replay",
        "replay_episode": str(replay_dir) if replay_dir else None,
        "policy_rate_hz": int(policy_rate_hz),
        "action_scale": float(action_scale),
        "action_clamp_count": int(policy.clamp_count),
        "lerobot_dataset_root": str(lerobot_dataset_root) if lerobot_dataset_root else None,
        "observability_mode": observability_mode,
        "occlusion": {"mode": occ_mode, "fraction": float(occ_frac), "camera": occ_camera},
        "camera_set_model": list(model_cameras),
        "camera_set_eval": list(eval_cameras),
        "instruction_ambiguity": ambig_mode,
        "intent": {
            "enabled": intent_estimator is not None,
            "every_n_ticks": int(intent_every),
        },
        "feedback": (
            {
                "channel": fb_channel_kind,
                "sim_human": sim_human_cfg.to_dict() if fb_channel_kind == "scripted" else None,
                "fusion": fusion_cfg.to_dict(),
                "reliability_final": float(reliability.mean),
                "n_interjections_total": int(sum(len(r.get("feedback", {}).get("events", [])) for r in ep_results)),
            }
            if feedback_on
            else None
        ),
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

    if debug_fp is not None:
        try:
            debug_fp.close()
        except Exception:
            pass
    if pipeline is not None:
        pipeline.close()
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
