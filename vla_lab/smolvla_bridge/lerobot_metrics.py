"""Helpers to align LeRobot training logs with `vla_lab/plot_metrics.py` consumers.

LeRobot's primary training UI is W&B / TensorBoard. For a minimal path:
  - Use `RUN_DIR=... lerobot-train ...` and point W&B at the same run name as in
    `vla_lab/scripts/train_smolvla.sh`, **or**
  - Parse your logs and append lines to `<checkpoint_dir>/metrics.jsonl` using the
    schema documented in `vla_lab/train.py` (`type: train_log` / `epoch_end`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def append_train_log_line(metrics_path: Path, payload: Dict[str, Any]) -> None:
    """Append one JSON object to `metrics_path` (TinyVLA-compatible jsonl)."""

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=str) + "\n")


def template_train_log(*, step: int, loss: float, lr: float, **extra: Any) -> Dict[str, Any]:
    """Build a `type: train_log` row like `vla_lab.train` writes."""

    row: Dict[str, Any] = {
        "type": "train_log",
        "step": int(step),
        "loss": float(loss),
        "lr": float(lr),
        "act_loss": float(extra.get("act_loss", loss)),
        "fa_loss": float(extra.get("fa_loss", 0.0)),
    }
    row.update({k: v for k, v in extra.items() if k not in row})
    return row
