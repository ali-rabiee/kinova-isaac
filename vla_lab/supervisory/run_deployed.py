"""Closed-loop evaluation: the trained policy *is* the robot's ear.

The architecture sweep measures a checkpoint on held-out dialogues. This measures what happens
when the checkpoint is dropped into the running system and the schedulers, the belief module and
the estimand all consume *its* readings instead of the lexical reference channel's. It is the
evaluation that corresponds to a deployment, and it can fail in ways the offline metric cannot
see -- a model that grounds well on average but abstains on exactly the crossover-band scenes
would score well offline and starve the estimand here.

Three channels are compared under an otherwise identical study:

``lexical``
    The conservative keyword grounder. The reference.
``policy[said]``
    The model reads the utterance. Isolates grounding quality.
``policy[unprompted]``
    The model reports what it believes the supervisor would have said uncoached, and the robot
    acts on that. This is the de-biasing head in charge, and it is reported separately because a
    system that substitutes its own guess for a person's instruction is a different thing from
    one that reads it accurately.

    python -m vla_lab.supervisory.run_deployed --checkpoint vla_lab/results/models_isaac/tiny__film/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .apparatus import LexicalGrounder
from .contract import Contract
from .narration import grounding_agreement
from .run_study import _fit_reported, aggregate, run_one_supervisor
from .scheduler import CONDITION_CARRYOVER_AWARE, PRIMARY_COMPARATOR


def _resolve(p: Path) -> Path:
    """Accept a run directory or a checkpoint file; prefer ``best.pt``."""
    p = Path(p)
    if p.is_dir():
        for name in ("best.pt", "last.pt"):
            if (p / name).exists():
                return p / name
        raise FileNotFoundError(f"no best.pt or last.pt in {p}")
    return p


def _tag(p: Path) -> str:
    """A short label for a checkpoint: the run directory's name."""
    return Path(p).parent.name or Path(p).stem


def build_grounder(kind: str, *, checkpoint: Optional[Path], contract: Contract,
                   frames: Optional[Path], min_conf: float, device: Optional[str] = None):
    if kind == "lexical":
        return LexicalGrounder(contract.axis)
    from ..policy.grounder import PolicyGrounder
    from ..training.scene_atlas import SceneAtlas

    atlas = SceneAtlas(contract.grid, frames_dir=frames)
    read = "said" if kind.endswith("said") else "unprompted"
    return PolicyGrounder(Path(checkpoint), axis=contract.axis, read=read, atlas=atlas,
                          min_confidence=min_conf, device=device)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, nargs="+", required=True,
                    help="one or more run directories or .pt files; each is evaluated separately")
    ap.add_argument("--supervisors", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20260823)
    ap.add_argument("--frames", type=Path, default=Path("vla_lab/results/physics/frames/topdown"))
    ap.add_argument("--min-confidence", type=float, default=0.60)
    ap.add_argument("--conditions", nargs="+", default=[CONDITION_CARRYOVER_AWARE, PRIMARY_COMPARATOR])
    ap.add_argument("--channels", nargs="+", default=["lexical", "policy_said", "policy_unprompted"])
    ap.add_argument("--out", type=Path, default=Path("vla_lab/results/deployed"))
    ap.add_argument("--device", default=None, help="torch device for the policy (default: auto)")
    args = ap.parse_args(argv)

    contract = Contract()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {}

    jobs: List[Tuple[str, Optional[Path]]] = []
    for chan in args.channels:
        if chan == "lexical":
            jobs.append((chan, None))
            continue
        for ck in args.checkpoint:
            jobs.append((chan, _resolve(Path(ck))))

    for chan, ck in jobs:
        label = chan if ck is None else f"{chan}@{_tag(ck)}"
        t0 = time.time()
        grounder = build_grounder(chan, checkpoint=ck, contract=contract,
                                  frames=args.frames, min_conf=args.min_confidence,
                                  device=args.device)
        rows: List[Dict[str, Any]] = []
        for i in range(int(args.supervisors)):
            rows.append(run_one_supervisor(index=i, contract=contract, conditions=list(args.conditions),
                                           seed=int(args.seed), grounder=grounder))
            print(f"  [{label}] supervisor {i + 1}/{args.supervisors}   ", end="\r", file=sys.stderr)
        for r in rows:
            r.pop("_identification_posterior", None)
        summary = aggregate(rows, list(args.conditions), seed=int(args.seed))
        summary["channel"] = chan
        summary["checkpoint"] = str(ck) if ck else None
        summary["elapsed_s"] = time.time() - t0
        if hasattr(grounder, "report"):
            summary["grounder"] = grounder.report()
        results[label] = summary
        print(f"[{label}] done in {summary['elapsed_s']:.0f}s", file=sys.stderr)

    (out / "summary.json").write_text(json.dumps(results, indent=2, default=float) + "\n")
    print(render(results, list(args.conditions)))
    (out / "table.txt").write_text(render(results, list(args.conditions)) + "\n")
    return 0


def render(results: Dict[str, Any], conditions: Sequence[str]) -> str:
    head = (f"{'grounding channel':32s}{'condition':22s}{'MAE_x':>9}{'regret':>9}"
            f"{'align':>8}{'ungrounded':>12}{'abstain':>9}{'in band':>9}{'agree':>8}")
    lines = ["", "Closed-loop evaluation: the policy in the session, not on held-out data.",
             head, "-" * len(head)]
    for chan, s in results.items():
        g = s.get("grounder", {})
        for c in conditions:
            cell = s["conditions"].get(c, {})
            lines.append(
                f"{chan:32s}{c:22s}"
                f"{cell.get('mae_crossover', {}).get('mean', float('nan')):>9.4f}"
                f"{cell.get('deployment_regret', {}).get('mean', float('nan')):>9.4f}"
                f"{cell.get('alignment', {}).get('mean', float('nan')):>8.3f}"
                f"{cell.get('n_ungrounded', {}).get('mean', float('nan')):>12.1f}"
                f"{g.get('abstain_rate', float('nan')):>9.3f}"
                f"{g.get('abstain_rate_band', float('nan')):>9.3f}"
                f"{g.get('agreement_with_lexical', float('nan')):>8.3f}"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
