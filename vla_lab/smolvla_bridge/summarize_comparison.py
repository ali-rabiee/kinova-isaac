"""Compare two `vla_lab/eval_isaaclab.py` result JSON files (CLI)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from ..stats_utils import wilson_ci


def _summarize(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    n = int(data.get("num_episodes", 0))
    s = int(data.get("num_success", 0))
    lo, hi = wilson_ci(s, n)
    return {
        "path": str(path),
        "policy_backend": data.get("policy_backend"),
        "ckpt": data.get("ckpt"),
        "n": n,
        "success_rate": float(data.get("success_rate", s / max(1, n))),
        "wilson_95": (lo, hi),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results_a", type=str)
    p.add_argument("results_b", type=str)
    args = p.parse_args()
    a = _summarize(Path(args.results_a))
    b = _summarize(Path(args.results_b))
    print(json.dumps({"a": a, "b": b}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
