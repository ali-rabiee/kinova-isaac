"""The architecture comparison: every (backbone x context mode) cell, in one table.

    python -m vla_lab.training.sweep_models --models tiny --contexts none token film
    python -m vla_lab.training.sweep_models --models tiny smolvla --contexts none text token film

The independent variable is **where the carryover context enters the network**; the backbone is
the blocking factor. Cells the registry declares impossible -- a verbalised context on a
backbone with no language model -- are skipped and reported as skipped, not as failures.

Every cell is trained by the same script, on the same generated data, with the same seed and
split, so a difference between two rows is a difference between two architectures and not
between two training recipes. Each cell writes its own manifest; the table below is assembled
from those, so it can be regenerated without retraining.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..policy.registry import MODEL_CARDS, available_models


def cells(models: Sequence[str], contexts: Sequence[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in models:
        card = MODEL_CARDS[m]
        for c in contexts:
            out.append({"model": m, "context": c, "supported": c in card.context_modes})
    return out


def render_table(rows: Sequence[Dict[str, Any]]) -> str:
    head = (f"{'model':26s}{'pretrain':>9}{'lang':>6}{'context':>9}{'trainable':>12}"
            f"{'gain(Brier)':>13}{'gap':>7}{'gap~kappa':>11}{'ask rho':>9}{'min':>7}")
    lines = [head, "-" * len(head)]
    for r in rows:
        if r.get("skipped"):
            lines.append(f"{r['display']:26s}{r['pretrained']:>9}{'yes' if r['language'] else 'no':>6}"
                         f"{r['context']:>9}{'--':>12}{'  n/a':>13}{'--':>7}{'--':>11}{'--':>9}{'--':>7}"
                         f"   ({r['skipped']})")
            continue
        lines.append(
            f"{r['display']:26s}{r['pretrained']:>9}{'yes' if r['language'] else 'no':>6}"
            f"{r['context']:>9}{r['params_trainable']:>12,}"
            f"{r['debias_gain_brier']:>+13.4f}{r['debias_gap']:>7.2f}"
            f"{r['debias_kappa_corr']:>+11.3f}{r['ask_rank_corr']:>+9.3f}{r['minutes']:>7.1f}"
        )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["tiny"], choices=available_models())
    ap.add_argument("--contexts", nargs="+", default=["none", "token", "film"])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--supervisors", type=int, default=80)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--frames", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("vla_lab/results/models"))
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="passed through to the trainer")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for cell in cells(args.models, args.contexts):
        card = MODEL_CARDS[cell["model"]]
        base = {"model": cell["model"], "context": cell["context"], "display": card.display,
                "pretrained": card.pretrained, "language": card.language,
                "action_head": card.action_head, "params": card.params}
        if not cell["supported"]:
            reason = ("no language model to inject text into" if cell["context"] == "text"
                      else "unsupported by this backbone")
            rows.append({**base, "skipped": reason})
            print(f"[skip] {card.display} / {cell['context']}: {reason}", file=sys.stderr)
            continue

        run_dir = out / f"{cell['model']}__{cell['context']}"
        cmd = [sys.executable, "-m", "vla_lab.training.train",
               "--model", cell["model"], "--context", cell["context"],
               "--epochs", str(args.epochs), "--supervisors", str(args.supervisors),
               "--batch", str(args.batch), "--accum", str(args.accum),
               "--seed", str(args.seed), "--workers", "0", "--out", str(run_dir)]
        if args.lr is not None:
            cmd += ["--lr", str(args.lr)]
        if args.frames:
            cmd += ["--frames", str(args.frames)]
        cmd += list(args.extra)
        print(f"\n[cell] {card.display} / context={cell['context']}", file=sys.stderr)
        t0 = time.time()
        rc = subprocess.run(cmd).returncode
        summary_path = run_dir / "summary.json"
        # Belt and braces with the trainer's own cleanup: only accept a summary written *after*
        # this cell started. A result table assembled from whatever files happen to be on disk
        # is how a failed configuration gets credited with a working one's numbers.
        fresh = summary_path.exists() and summary_path.stat().st_mtime >= t0
        if rc != 0 or not fresh:
            why = f"training failed (exit {rc})" if rc != 0 else "no fresh summary written"
            rows.append({**base, "skipped": why})
            print(f"[fail] {card.display} / {cell['context']}: {why}", file=sys.stderr)
            continue
        summary = json.loads(summary_path.read_text())
        final = summary.get("final", {})
        rows.append({
            **base,
            "params_trainable": int(summary["manifest"]["model"]["params_trainable"]),
            "params_total": int(summary["manifest"]["model"]["params_total"]),
            "adapt": summary["manifest"]["model"]["adapt"]["applied"],
            "image_source": summary["manifest"]["data"]["image_source"],
            "debias_gain_brier": float(summary.get("best_debias_gain", float("nan"))),
            "debias_gap": float(final.get("mean_abs_debias_gap", float("nan"))),
            "debias_kappa_corr": float(final.get("debias_kappa_corr", float("nan"))),
            "ask_rank_corr": float(final.get("ask_rank_corr", float("nan"))),
            "acc_said": float(final.get("acc_said", float("nan"))),
            "brier_unprompted_vs_pi": float(final.get("brier_unprompted_vs_pi", float("nan"))),
            "brier_said_vs_pi": float(final.get("brier_said_vs_pi", float("nan"))),
            "minutes": (time.time() - t0) / 60.0,
            "run_dir": str(run_dir),
        })
        (out / "table.json").write_text(json.dumps(rows, indent=2, default=float) + "\n")

    table = render_table(rows)
    (out / "table.txt").write_text(table + "\n")
    (out / "table.json").write_text(json.dumps(rows, indent=2, default=float) + "\n")
    print("\n" + table)
    print(f"\nwrote {out}/table.txt and table.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
