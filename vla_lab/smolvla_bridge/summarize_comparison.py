"""Compare two `vla_lab/eval_isaaclab.py` result JSON files (CLI)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Tuple


def _wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1.0 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _summarize(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    n = int(data.get("num_episodes", 0))
    s = int(data.get("num_success", 0))
    lo, hi = _wilson_ci(s, n)
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
