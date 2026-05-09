"""Launch `lerobot-train` for SmolVLA using a small YAML shim (see `configs/train_smolvla.example.yaml`).

Example:
  python -m vla_lab.train_smolvla --config vla_lab/configs/train_smolvla.example.yaml

`--dry-run` prints the resolved command without executing.
"""

from __future__ import annotations

import argparse
import json
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
    ds = cfg.get("dataset", {}) or {}
    dataset_root = str(ds.get("root", "vla_lab/datasets/lerobot_kinova_v0"))
    repo_id = str(ds.get("repo_id", "kinova_isaac_vla"))
    tr = cfg.get("training", {}) or {}
    steps = int(tr.get("steps", 20000))
    batch_size = int(tr.get("batch_size", 32))
    device = str(tr.get("device", "cuda"))
    out_dir = str(tr.get("output_dir", "")) or ""

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
        f"--dataset.repo_id={repo_id}",
        f"--dataset.root={dataset_root}",
        f"--policy.device={device}",
        f"--batch_size={batch_size}",
        f"--steps={steps}",
        f"--job_name={run_name}",
    ]
    if out_dir:
        cmd.append(f"--output_dir={out_dir}")
    if extra:
        cmd.extend(extra)

    manifest = {
        "created_unix": float(time.time()),
        "config": str(cfg_path.resolve()),
        "cmd": cmd,
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0

    print("[train_smolvla] " + " ".join(cmd))
    subprocess.check_call(cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
