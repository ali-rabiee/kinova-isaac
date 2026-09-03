"""Objective ablations: which term in the training loss is doing the de-biasing?

The architecture sweep asks *where the context enters*. This asks a different question --- what
the model is being asked to optimise --- and it is the one that decides whether the de-biasing
claim is about a mechanism or about a loss function. Four cells, all on the same backbone,
context mode, data, seed and split:

``full``
    Every term.
``no-forward``
    Drop the self-supervised consistency term, which requires the model's de-biased belief,
    pushed through the contamination model, to reproduce the observed utterance. This is the
    term the paper argues makes de-biasing *identifiable without a reference block*, so if the
    gain survives its removal that argument is wrong.
``no-anti-copy``
    Drop the compliance penalty, which charges the model for agreeing with an instruction in
    proportion to how contaminated it is.
``no-reference``
    Drop supervision against an uncontaminated reference draw. A deployed system will not have
    one, so a method that collapses without it does not deploy.

    python -m vla_lab.training.sweep_ablations --model tiny --context film
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ABLATIONS: Dict[str, List[str]] = {
    "full": [],
    "no-forward": ["--no-forward"],
    "no-anti-copy": ["--no-anti-copy"],
    "no-reference": ["--no-reference"],
}

DESCRIPTIONS = {
    "full": "every term",
    "no-forward": "without forward consistency",
    "no-anti-copy": "without the compliance penalty",
    "no-reference": "without reference supervision",
}


def render_table(rows: Sequence[Dict[str, Any]]) -> str:
    head = (f"{'objective':30s}{'seed':>5}{'gain(Brier)':>13}{'d vs full':>11}{'gap':>7}"
            f"{'gap~kappa':>11}{'acc_said':>10}{'min':>7}")
    lines = [head, "-" * len(head)]
    for r in rows:
        if r.get("skipped"):
            lines.append(f"{DESCRIPTIONS.get(r['ablation'], r['ablation']):30s}{str(r.get('seed', '')):>5}{'  n/a':>13}   ({r['skipped']})")
            continue
        base = next((x["debias_gain_brier"] for x in rows if x["ablation"] == "full"
                     and x.get("seed") == r.get("seed") and not x.get("skipped")), None)
        d = (r["debias_gain_brier"] - base) if base is not None else float("nan")
        lines.append(
            f"{DESCRIPTIONS.get(r['ablation'], r['ablation']):30s}{str(r.get('seed', '')):>5}"
            f"{r['debias_gain_brier']:>+13.4f}{d:>+11.4f}{r['debias_gap']:>7.2f}"
            f"{r['debias_kappa_corr']:>+11.3f}{r['acc_said']:>10.3f}{r['minutes']:>7.1f}"
        )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="tiny")
    ap.add_argument("--context", default="film")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--supervisors", type=int, default=80)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1, help="single seed (legacy); prefer --seeds")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="run every ablation under each seed; report mean +- SD and the seed floor")
    ap.add_argument("--frames", type=Path, default=None)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--only", nargs="+", default=None, choices=list(ABLATIONS))
    ap.add_argument("--out", type=Path, default=Path("vla_lab/results/ablations"))
    args = ap.parse_args(argv)

    from .seeds import render_seeds, write_seed_tables
    from .sweep_models import train_cell

    seeds = list(args.seeds) if args.seeds else [int(args.seed)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = list(args.only) if args.only else list(ABLATIONS)
    rows: List[Dict[str, Any]] = []

    for name in names:
        for seed in seeds:
            run_dir = out / (f"{args.model}__{args.context}__{name}" if len(seeds) == 1 and args.seeds is None
                             else f"{args.model}__{args.context}__{name}__s{seed}")
            base = {"ablation": name, "model": args.model, "context": args.context}
            row = train_cell(run_dir, base=base, seed=int(seed), args=args,
                             train_args=["--model", args.model, "--context", args.context],
                             label=f"ablation {name} ({DESCRIPTIONS[name]}) / seed {seed}",
                             ablation_args=ABLATIONS[name])
            rows.append(row)
            (out / "table.json").write_text(json.dumps(rows, indent=2, default=float) + "\n")
            write_seed_tables(rows, out, group_by=("model", "context", "ablation"))

    table = render_table(rows)
    (out / "table.txt").write_text(table + "\n")
    (out / "table.json").write_text(json.dumps(rows, indent=2, default=float) + "\n")
    agg = write_seed_tables(rows, out, group_by=("model", "context", "ablation"))
    print("\n" + table)
    print("\n" + render_seeds(agg))
    print(f"\nwrote {out}/table.txt, table_seeds.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
