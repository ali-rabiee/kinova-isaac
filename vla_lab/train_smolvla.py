"""Launch `lerobot-train` for SmolVLA using a small YAML shim (see `configs/train_smolvla.example.yaml`).

Example:
  python -m vla_lab.train_smolvla --config vla_lab/configs/train_smolvla.example.yaml

`--dry-run` prints the resolved command without executing.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError("vla_lab.train_smolvla requires PyYAML (`pip install pyyaml`).") from exc


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _yaml_bool(v: Any, *, default: bool) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off", ""):
        return False
    return default


def main() -> int:
    parser = argparse.ArgumentParser(description="SmolVLA fine-tune via lerobot-train")
    parser.add_argument("--config", type=str, default="vla_lab/configs/train_smolvla.example.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        print(f"[train_smolvla] ERROR: config not found: {cfg_path}", file=sys.stderr)
        return 2

    cfg = _load_yaml(cfg_path)
    policy_path = str(cfg.get("policy_path", "lerobot/smolvla_base"))
    push_to_hub = _yaml_bool(cfg.get("push_to_hub"), default=False)
    ds = cfg.get("dataset", {}) or {}
    dataset_root = str(ds.get("root", "vla_lab/datasets/lerobot_kinova_v0"))
    repo_id = str(ds.get("repo_id", "kinova_isaac_vla"))
    tr = cfg.get("training", {}) or {}
    steps = int(tr.get("steps", 9824))
    batch_size = int(tr.get("batch_size", 32))
    device = str(tr.get("device", "cuda"))
    out_dir = str(tr.get("output_dir", "")) or ""
    save_checkpoint = _yaml_bool(tr.get("save_checkpoint"), default=True)
    save_freq = int(tr.get("save_freq", 2000))
    log_freq = int(tr.get("log_freq", 50))
    eval_freq = tr.get("eval_freq")
    train_seed = tr.get("seed")
    resume = _yaml_bool(tr.get("resume"), default=False)

    if not shutil.which("lerobot-train"):
        print(
            "[train_smolvla] ERROR: `lerobot-train` not on PATH. Install:\n"
            "  pip install -r vla_lab/requirements-smolvla.txt",
            file=sys.stderr,
        )
        return 2

    run_name = str(tr.get("job_name", "")) or f"smolvla_ft_{int(time.time())}"
    cmd: List[str] = [
        "lerobot-train",
        f"--policy.path={policy_path}",
        f"--policy.push_to_hub={str(push_to_hub).lower()}",
        f"--dataset.repo_id={repo_id}",
        f"--dataset.root={dataset_root}",
        f"--policy.device={device}",
        f"--batch_size={batch_size}",
        f"--steps={steps}",
        f"--job_name={run_name}",
        f"--save_checkpoint={str(save_checkpoint).lower()}",
        f"--save_freq={save_freq}",
        f"--log_freq={log_freq}",
    ]
    if eval_freq is not None:
        cmd.append(f"--eval_freq={int(eval_freq)}")
    if train_seed is not None:
        cmd.append(f"--seed={int(train_seed)}")
    if out_dir:
        cmd.append(f"--output_dir={out_dir}")
    if resume:
        cmd.append("--resume=true")
    if extra:
        cmd.extend(extra)

    capture = os.environ.get("VLA_LAB_CAPTURE_RESULTS", "1") not in ("0", "false", "False", "no")
    run_cmd: List[str] = list(cmd)
    if capture:
        os.environ["VLA_LAB_RUN_NAME"] = run_name
        run_cmd = [sys.executable, "-m", "vla_lab.lerobot_train_capture", "--"] + cmd[1:]

    manifest = {
        "created_unix": float(time.time()),
        "config": str(cfg_path.resolve()),
        "lerobot_cmd": cmd,
        "executed_cmd": run_cmd,
        "capture_results": capture,
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0

    print("[train_smolvla] " + " ".join(run_cmd))
    subprocess.check_call(run_cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
