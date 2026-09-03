"""Does residue tracking survive distribution shift? Score every checkpoint on a held-out atlas.

The architecture table is computed on 19 scenes from one camera, one table and one pair of
cubes. The context-blind cells on the pretrained backbones track the residue anyway, which could
mean the backbones read it off dataset regularities -- scene image and wording correlates that
a different scene set would not supply. This decides whether Table 4 measures an ability or an
artefact: every trained checkpoint is re-scored on its **own validation supervisors** (the same
dialogues, rebuilt from the run's seed) with the images swapped for the held-out atlas
(``run_frames.py --shift``: different colours and cube size, an altered table surface, an added
distractor, and a second overhead camera pose).

Three image distributions per checkpoint:

``matched``        the contract atlas the model trained on (should reproduce the table);
``shift``          the held-out scene set from the contract camera pose;
``shift+camera``   the held-out scene set from the second camera pose.

Reported per cell: Brier gain and ``gap~kappa`` under each, and the **degradation**
(matched minus shifted). The question the paper asks is whether context-blind cells collapse
while context-injected cells hold.

    python -m vla_lab.training.eval_shift vla_lab/results/models_isaac --out vla_lab/results/shift
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def evaluate_checkpoint(run_dir: Path, atlases: Dict[str, Optional[Path]], *, device: Optional[str] = None,
                        batch: int = 32) -> Dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader, Subset

    from ..policy.grounder import PolicyGrounder
    from ..supervisory.contract import Contract
    from .data import SupervisoryDialogueDataset, collate, generate_dialogues
    from .losses import CarryoverLossConfig
    from .scene_atlas import SceneAtlas
    from .train import evaluate

    run_dir = Path(run_dir)
    ckpt = run_dir / "best.pt" if (run_dir / "best.pt").exists() else run_dir / "last.pt"
    manifest = json.loads((run_dir / "manifest.json").read_text())
    args = manifest.get("args", {})
    seed = int(args.get("seed", manifest.get("data", {}).get("seed", 0)))
    g = PolicyGrounder(ckpt, read="said", device=device)          # reuses the checkpoint loader
    model, tokenizer = g.model, g.tokenizer
    contract = Contract()
    # The dialogues are a deterministic function of (contract, n_supervisors, seed, doses), so the
    # validation set here is the one the run was scored on.
    samples = generate_dialogues(contract=contract, n_supervisors=int(args.get("supervisors", 60)), seed=seed,
                                 doses=list(args.get("train_doses", ["weak", "moderate", "strong"])))
    val_ids = set(manifest["data"]["val_supervisors"])
    va_idx = [i for i, s in enumerate(samples) if s.supervisor_id in val_ids]
    lcfg = CarryoverLossConfig(**{k: v for k, v in (manifest.get("loss") or {}).items()
                                  if k in CarryoverLossConfig.__dataclass_fields__})
    out: Dict[str, Any] = {"run_dir": str(run_dir), "model": manifest.get("card", {}).get("key"),
                           "display": manifest.get("card", {}).get("display"),
                           "context": manifest.get("context_mode"), "seed": seed,
                           "n_val_samples": len(va_idx), "scores": {}}
    for name, frames in atlases.items():
        atlas = SceneAtlas(contract.grid, frames_dir=frames)
        cov = atlas.coverage()
        ds = SupervisoryDialogueDataset(samples, contract=contract, atlas=atlas,
                                        context_mode=str(manifest.get("context_mode")),
                                        context_style=str(manifest.get("context_style", "compact")),
                                        tokenizer=tokenizer, seed=seed)
        dl = DataLoader(Subset(ds, va_idx), batch_size=int(batch), shuffle=False, num_workers=0, collate_fn=collate)
        ev = evaluate(model, dl, g.device, lcfg)
        out["scores"][name] = {"image_source": cov["source"], "frames": cov["frames"],
                               "scenes_with_frames": cov["scenes_with_frames"],
                               **{k: ev[k] for k in ("debias_gain_brier", "debias_kappa_corr", "acc_said",
                                                     "mean_abs_debias_gap", "ask_rank_corr", "n_val")}}
    m = out["scores"].get("matched")
    if m:
        for name, sc in out["scores"].items():
            if name == "matched":
                continue
            sc["degradation_gain"] = float(m["debias_gain_brier"] - sc["debias_gain_brier"])
            sc["degradation_kappa_corr"] = float(m["debias_kappa_corr"] - sc["debias_kappa_corr"])
    return out


def render(rows: Sequence[Dict[str, Any]], names: Sequence[str]) -> str:
    head = f"{'cell':40s}{'seed':>5}" + "".join(f"{n + ' gain':>16}{n + ' gap~k':>16}" for n in names)
    lines = [head, "-" * len(head)]
    for r in rows:
        cells = ""
        for n in names:
            sc = r["scores"].get(n)
            cells += (f"{sc['debias_gain_brier']:>+16.3f}{sc['debias_kappa_corr']:>+16.3f}" if sc
                      else f"{'--':>16}{'--':>16}")
        lines.append(f"{(r.get('display') or r.get('model')) + ' / ' + str(r.get('context')):40s}{r.get('seed', ''):>5}{cells}")
    return "\n".join(lines)


def aggregate(rows: Sequence[Dict[str, Any]], names: Sequence[str]) -> Dict[str, Any]:
    """Per-(model, context) means over seeds of each score and of the degradation."""
    cells: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in rows:
        cells.setdefault((str(r.get("model")), str(r.get("context"))), []).append(r)
    out: List[Dict[str, Any]] = []
    for (model, ctx), rs in cells.items():
        cell: Dict[str, Any] = {"model": model, "context": ctx, "display": rs[0].get("display"),
                                "n_seeds": len(rs)}
        for n in names:
            for key in ("debias_gain_brier", "debias_kappa_corr", "degradation_gain", "degradation_kappa_corr"):
                vals = [r["scores"][n][key] for r in rs if n in r["scores"] and key in r["scores"][n]
                        and np.isfinite(r["scores"][n][key])]
                if vals:
                    cell[f"{n}__{key}"] = {"mean": float(np.mean(vals)),
                                           "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None}
        out.append(cell)
    return {"cells": out, "atlases": list(names)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", type=Path, nargs="+", help="sweep directories containing run dirs")
    ap.add_argument("--matched", type=Path, default=Path("vla_lab/results/physics/frames/topdown"))
    ap.add_argument("--shift", type=Path, default=Path("vla_lab/results/physics/frames_shift/topdown"))
    ap.add_argument("--shift-camera", type=Path, default=Path("vla_lab/results/physics/frames_shift/topdown_shift"))
    ap.add_argument("--out", type=Path, default=Path("vla_lab/results/shift"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args(argv)

    atlases = {"matched": args.matched, "shift": args.shift, "shift+camera": args.shift_camera}
    for name, p in atlases.items():
        if not Path(p).exists():
            print(f"[FAIL] atlas '{name}' missing at {p}; render it first (run_frames.py{' --shift' if name != 'matched' else ''})",
                  file=sys.stderr)
            return 2
    run_dirs = [d for root in args.roots for d in sorted(Path(root).iterdir())
                if d.is_dir() and (d / "manifest.json").exists() and ((d / "best.pt").exists() or (d / "last.pt").exists())]
    rows: List[Dict[str, Any]] = []
    args.out.mkdir(parents=True, exist_ok=True)
    for d in run_dirs:
        print(f"[shift] {d}", file=sys.stderr)
        try:
            rows.append(evaluate_checkpoint(d, atlases, device=args.device, batch=args.batch))
        except Exception as exc:                                   # a broken checkpoint must not stop the sweep
            print(f"[shift] {d}: {type(exc).__name__}: {exc}", file=sys.stderr)
            rows.append({"run_dir": str(d), "error": f"{type(exc).__name__}: {exc}", "scores": {}})
        (args.out / "table.json").write_text(json.dumps(rows, indent=2, default=float) + "\n")
    names = list(atlases)
    text = render([r for r in rows if r.get("scores")], names)
    (args.out / "table.txt").write_text(text + "\n")
    (args.out / "table_cells.json").write_text(json.dumps(aggregate([r for r in rows if r.get("scores")], names),
                                                          indent=2, default=float) + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
