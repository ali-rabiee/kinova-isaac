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
    head = (f"{'objective':30s}{'gain(Brier)':>13}{'d vs full':>11}{'gap':>7}"
            f"{'gap~kappa':>11}{'acc_said':>10}{'min':>7}")
    lines = [head, "-" * len(head)]
    base = next((r["debias_gain_brier"] for r in rows if r["ablation"] == "full"), None)
    for r in rows:
        if r.get("skipped"):
            lines.append(f"{DESCRIPTIONS.get(r['ablation'], r['ablation']):30s}{'  n/a':>13}   ({r['skipped']})")
            continue
        d = (r["debias_gain_brier"] - base) if base is not None else float("nan")
        lines.append(
            f"{DESCRIPTIONS.get(r['ablation'], r['ablation']):30s}"
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
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--frames", type=Path, default=None)
    ap.add_argument("--only", nargs="+", default=None, choices=list(ABLATIONS))
    ap.add_argument("--out", type=Path, default=Path("vla_lab/results/ablations"))
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = list(args.only) if args.only else list(ABLATIONS)
    rows: List[Dict[str, Any]] = []

    for name in names:
        run_dir = out / f"{args.model}__{args.context}__{name}"
        cmd = [sys.executable, "-m", "vla_lab.training.train",
               "--model", args.model, "--context", args.context,
               "--epochs", str(args.epochs), "--supervisors", str(args.supervisors),
               "--batch", str(args.batch), "--accum", str(args.accum),
               "--seed", str(args.seed), "--workers", "0", "--out", str(run_dir)] + ABLATIONS[name]
        if args.frames:
            cmd += ["--frames", str(args.frames)]
        print(f"\n[ablation] {name}: {DESCRIPTIONS[name]}", file=sys.stderr)
        t0 = time.time()
        rc = subprocess.run(cmd).returncode
        summary_path = run_dir / "summary.json"
        if rc != 0 or not (summary_path.exists() and summary_path.stat().st_mtime >= t0):
            rows.append({"ablation": name, "skipped": f"training failed (exit {rc})"})
            continue
        summary = json.loads(summary_path.read_text())
        final = summary.get("final", {})
        rows.append({
            "ablation": name,
            "model": args.model,
            "context": args.context,
            "image_source": summary["manifest"]["data"]["image_source"],
            "debias_gain_brier": float(summary.get("best_debias_gain", float("nan"))),
            "debias_gap": float(final.get("mean_abs_debias_gap", float("nan"))),
            "debias_kappa_corr": float(final.get("debias_kappa_corr", float("nan"))),
            "acc_said": float(final.get("acc_said", float("nan"))),
            "brier_unprompted_vs_pi": float(final.get("brier_unprompted_vs_pi", float("nan"))),
            "minutes": (time.time() - t0) / 60.0,
            "run_dir": str(run_dir),
        })
        (out / "table.json").write_text(json.dumps(rows, indent=2, default=float) + "\n")

    table = render_table(rows)
    (out / "table.txt").write_text(table + "\n")
    (out / "table.json").write_text(json.dumps(rows, indent=2, default=float) + "\n")
    print("\n" + table)
    print(f"\nwrote {out}/table.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
