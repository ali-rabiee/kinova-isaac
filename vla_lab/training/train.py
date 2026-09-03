"""Trainer for the Carryover-Aware VLA. Built to finish, not to be fast.

    python -m vla_lab.training.train --model tiny --context token --epochs 20
    python -m vla_lab.training.train --model smolvla --context text --epochs 4
    python -m vla_lab.training.train --model qwen25vl-3b --context text --adapt lora --batch 1 --accum 16

Four things this script insists on, each because the alternative fails quietly:

**Splitting by supervisor, never by sample.** Two answers from the same person at the same
scene are not independent, and a random sample split would let the model memorise a supervisor's
preference map and report it as de-biasing. The split is by ``supervisor_id`` and the manifest
records both id lists.

**A VRAM preflight.** Before the first optimiser step the trainer runs one forward/backward at
the configured batch size and reports peak memory. On a 16 GB card the difference between "this
will finish overnight" and "this dies at step 40 when a longer prompt arrives" is a few hundred
megabytes of headroom, and finding out at step 40 wastes the night.

**The headline validation metric is the de-biasing gain, not the loss.** A model can drive the
training objective down by grounding utterances well and never de-biasing anything. The metric
reported every epoch is how much better the ``unprompted`` head predicts the supervisor's cold
answer than simply believing what they said -- which is zero for a Memoryless VLA by
construction, and is the number the paper's model table carries.

**Every run writes a manifest.** Model card, context mode, what was actually trainable (LoRA or
frozen -- these are different experiments), the data seed, the image source, and the split. A
results table assembled from runs that cannot say which of those they were is not a table.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ..calibration.metrics import expected_calibration_error
from ..policy import build_model
from ..policy.registry import MODEL_CARDS
from ..supervisory.contract import Contract
from .data import SupervisoryDialogueDataset, collate, generate_dialogues
from .losses import CarryoverLossConfig, carryover_loss
from .scene_atlas import SceneAtlas


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Rank correlation. Used for the ask gate because its target is a *probability*.

    Thresholding a soft target to compute AUROC throws away the gradation the gate exists to
    learn, and returns ``nan`` outright whenever every sample happens to fall on one side of the
    threshold -- which is the normal case here, since a counter-proposal usually would *not*
    have changed the answer.
    """
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if x.size < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    rx, ry = _rank(x), _rank(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = float(np.sqrt((rx**2).sum() * (ry**2).sum()))
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    r = np.empty_like(order, dtype=float)
    r[order] = np.arange(1, len(x) + 1)
    return r


@torch.no_grad()
def evaluate(model, loader, device: torch.device, cfg: CarryoverLossConfig) -> Dict[str, Any]:
    model.eval()
    said_hit = un_hit = copy_hit = n = 0
    gaps: List[float] = []
    kappas: List[float] = []
    asks: List[float] = []
    ask_labels: List[float] = []
    probs: List[float] = []
    outs: List[bool] = []
    losses: List[float] = []
    # Scored against pi* -- the supervisor's true unprompted preference probability -- rather
    # than against one Bernoulli draw of it. The draw is the label a real reference block gives
    # you and it is what the model trains on; but as a *metric* its noise floor is so high that
    # it hides the effect. Both are reported.
    p_un: List[float] = []
    p_said: List[float] = []
    pi_true: List[float] = []
    for b in loader:
        b = _to(b, device)
        out = model(b, context=b["context"], kappa=b["kappa"], beta=b.get("beta"), rho=b.get("rho"))
        losses.append(float(carryover_loss(out, b, cfg).total))
        said_pred = out.said.argmax(-1)
        un_pred = out.unprompted.argmax(-1)
        said_hit += int((said_pred == b["said"]).sum())
        un_hit += int((un_pred == b["unprompted"]).sum())
        # The baseline the de-biasing gain is measured against: believe what you were told.
        copy_hit += int((b["said"] == b["unprompted"]).sum())
        n += int(b["said"].numel())
        gaps.extend(out.debias_gap().detach().cpu().tolist())
        kappas.extend(b["kappa"].detach().cpu().tolist())
        asks.extend(torch.sigmoid(out.ask).detach().cpu().tolist())
        ask_labels.extend(b["ask_label"].detach().cpu().tolist())
        p = torch.softmax(out.unprompted, -1)[:, 0]
        probs.extend(p.detach().cpu().tolist())
        outs.extend((b["unprompted"] == 0).detach().cpu().tolist())
        if "pi_star" in b:
            p_un.extend(p.detach().cpu().tolist())
            p_said.extend(torch.softmax(out.said, -1)[:, 0].detach().cpu().tolist())
            pi_true.extend(b["pi_star"].detach().cpu().tolist())
    model.train()
    acc_un = un_hit / max(n, 1)
    acc_copy = copy_hit / max(n, 1)
    brier_un = float(np.mean((np.asarray(p_un) - np.asarray(pi_true)) ** 2)) if pi_true else float("nan")
    brier_said = float(np.mean((np.asarray(p_said) - np.asarray(pi_true)) ** 2)) if pi_true else float("nan")
    return {
        # The headline. Positive means the de-biased head tracks the supervisor's true
        # unprompted preference more closely than simply believing the instruction does.
        "brier_unprompted_vs_pi": brier_un,
        "brier_said_vs_pi": brier_said,
        "debias_gain_brier": (brier_said - brier_un) if pi_true else float("nan"),
        "val_loss": float(np.mean(losses)) if losses else float("nan"),
        "acc_said": said_hit / max(n, 1),
        "acc_unprompted": acc_un,
        "acc_copy_baseline": acc_copy,
        #: The headline. Zero for a model that just repeats the instruction.
        "debias_gain": acc_un - acc_copy,
        "mean_abs_debias_gap": float(np.mean(np.abs(gaps))) if gaps else 0.0,
        # **The architectural metric.** A model whose encoder cannot see the carryover context
        # can still learn a *constant* correction, because the training objective supplies
        # kappa through the forward-consistency term -- and a constant shrink toward chance
        # already improves the Brier score against pi*, since contaminated instructions
        # overshoot on average. What only a context-conditioned model can do is correct
        # **more when there is more to correct**. This correlation is that ability, and it is
        # what the architecture table is actually about.
        "debias_kappa_corr": _spearman([-g for g in gaps], kappas) if gaps else float("nan"),
        "ask_rank_corr": _spearman(asks, ask_labels),
        "unprompted_ece": float(expected_calibration_error(probs, outs, n_bins=10)) if probs else float("nan"),
        "n_val": n,
    }


def _to(b: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
def vram_preflight(model, batch: Dict[str, Any], device: torch.device, cfg: CarryoverLossConfig) -> Dict[str, Any]:
    if device.type != "cuda":
        return {"device": str(device), "peak_gb": None, "note": "CPU run; no VRAM preflight"}
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    b = _to(batch, device)
    out = model(b, context=b["context"], kappa=b["kappa"], beta=b.get("beta"), rho=b.get("rho"))
    carryover_loss(out, b, cfg).total.backward()
    peak = torch.cuda.max_memory_allocated() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    return {
        "device": torch.cuda.get_device_name(0),
        "peak_gb": round(float(peak), 2),
        "total_gb": round(float(total), 1),
        "headroom_gb": round(float(total - peak), 2),
        "ok": bool(peak < 0.88 * total),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="tiny", choices=list(MODEL_CARDS))
    ap.add_argument("--context", default="token", help="none | text | token | film")
    ap.add_argument("--adapt", default=None, choices=["full", "lora", "frozen"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--supervisors", type=int, default=60, help="synthetic supervisors to draw dialogues from")
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frames", type=Path, default=None, help="Isaac frame directory for the scene atlas")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--amp", default="auto", choices=["auto", "bf16", "fp16", "off"])
    ap.add_argument("--max-steps", type=int, default=0, help="stop early; 0 = run all epochs")
    ap.add_argument("--no-anti-copy", action="store_true", help="ablate the compliance penalty")
    ap.add_argument("--no-forward", action="store_true", help="ablate the forward-consistency term")
    ap.add_argument("--no-reference", action="store_true", help="ablate the reference supervision")
    ap.add_argument("--context-style", default=None, choices=["rich", "compact"],
                    help="verbalisation of the carryover context; defaults to what the backbone's "
                         "instruction budget can hold without truncating")
    ap.add_argument("--train-doses", nargs="+", default=["weak", "moderate", "strong"],
                    help="dose ladder to mix in the TRAINING dialogues; the evaluation population "
                         "is unchanged")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    contract = Contract()
    t0 = time.time()

    print(f"[data] generating dialogues from {args.supervisors} synthetic supervisors...", file=sys.stderr)
    samples = generate_dialogues(contract=contract, n_supervisors=int(args.supervisors), seed=int(args.seed),
                                 doses=list(args.train_doses))
    atlas = SceneAtlas(contract.grid, frames_dir=args.frames)
    card = MODEL_CARDS[args.model]
    style = args.context_style or card.context_style
    ds = SupervisoryDialogueDataset(samples, contract=contract, atlas=atlas, context_mode=args.context,
                                    context_style=style, seed=int(args.seed))

    # Split by supervisor. Never by sample.
    ids = sorted({s.supervisor_id for s in samples})
    rng = random.Random(int(args.seed))
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * float(args.val_frac))))
    val_ids, train_ids = set(ids[:n_val]), set(ids[n_val:])
    tr_idx = [i for i, s in enumerate(samples) if s.supervisor_id in train_ids]
    va_idx = [i for i, s in enumerate(samples) if s.supervisor_id in val_ids]

    dl_tr = DataLoader(Subset(ds, tr_idx), batch_size=int(args.batch), shuffle=True,
                       num_workers=int(args.workers), collate_fn=collate, drop_last=False)
    dl_va = DataLoader(Subset(ds, va_idx), batch_size=int(args.batch), shuffle=False,
                       num_workers=int(args.workers), collate_fn=collate)

    extra: Dict[str, Any] = {}
    if args.model == "tiny":
        # The from-scratch backbone owns its embedding table, so it needs the vocabulary this
        # corpus actually produced -- including the context words in `text` mode.
        extra = {"vocab_size": int(ds.tokenizer.vocab_size), "max_lang_len": int(ds.max_lang_len),
                 "chunk_len": int(ds.chunk_len), "action_dim": int(ds.action_dim)}
    model = build_model(args.model, context_mode=args.context, adapt=args.adapt, **extra)
    model.to(device)
    report = model.report()

    lcfg = CarryoverLossConfig(
        w_anti_copy=0.0 if args.no_anti_copy else CarryoverLossConfig().w_anti_copy,
        w_forward=0.0 if args.no_forward else CarryoverLossConfig().w_forward,
        w_unprompted=0.0 if args.no_reference else CarryoverLossConfig().w_unprompted,
    )

    # --- prompt-budget audit, before the first step -------------------------------------
    # A prompt that overflows the backbone's instruction budget does not fail; it trains on a
    # truncated context and reads afterwards as "this architecture did not help". This check is
    # what turns that into a startup warning.
    prompt_audit: Dict[str, Any] = {"checked": False}
    if args.context == "text":
        try:
            bb_tok = getattr(model.backbone, "_tokenizer", None)
            tokenizer = bb_tok() if callable(bb_tok) else None
            if tokenizer is not None:
                prompt_audit = ds.prompt_length_report(tokenizer, budget=int(card.instruction_budget))
                prompt_audit["checked"] = True
                if not prompt_audit["ok"]:
                    print(
                        f"[prompt] WARNING: {prompt_audit['frac_truncated']*100:.0f}% of prompts exceed "
                        f"the {card.instruction_budget}-token budget (mean {prompt_audit['mean_tokens']:.0f}, "
                        f"max {prompt_audit['max_tokens']}). The carryover context will be truncated away and "
                        f"this run will understate what verbalised context can do. Use --context-style compact.",
                        file=sys.stderr,
                    )
                else:
                    print(f"[prompt] ok: max {prompt_audit['max_tokens']} tokens within the "
                          f"{card.instruction_budget}-token budget ({style} style)", file=sys.stderr)
        except Exception as exc:  # pragma: no cover
            prompt_audit = {"checked": False, "error": str(exc)}

    pre = vram_preflight(model, next(iter(dl_tr)), device, lcfg)
    print(f"[preflight] {pre}", file=sys.stderr)
    if pre.get("ok") is False:
        print("[preflight] peak memory is within 12% of the card; reduce --batch or raise --accum.", file=sys.stderr)

    amp = args.amp
    if amp == "auto":
        amp = "bf16" if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else "off"
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(amp)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp == "fp16"))

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(args.lr), weight_decay=float(args.weight_decay))
    total_steps = max(1, len(dl_tr) * int(args.epochs) // max(1, int(args.accum)))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=float(args.lr), total_steps=total_steps, pct_start=0.15)

    out_dir = Path(args.out) if args.out else Path(f"vla_lab/checkpoints/{args.model}_{args.context}_{int(t0)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Clear any completed-run markers left in this directory by an earlier attempt. Without
    # this, a cell that crashes part-way leaves the *previous* run's summary in place, and the
    # sweep reads it as this run's result -- so a failed configuration silently inherits a
    # working one's numbers. The manifest is rewritten below either way; these are the files
    # that assert "this run finished".
    for stale in ("summary.json", "best.pt", "last.pt", "metrics.jsonl"):
        (out_dir / stale).unlink(missing_ok=True)

    manifest = {
        "model": report,
        "card": MODEL_CARDS[args.model].to_dict(),
        "context_mode": args.context,
        "loss": lcfg.to_dict(),
        "ablations": {"no_anti_copy": args.no_anti_copy, "no_forward": args.no_forward,
                      "no_reference": args.no_reference},
        "data": {
            "n_samples": len(samples), "n_train": len(tr_idx), "n_val": len(va_idx),
            "train_supervisors": sorted(train_ids), "val_supervisors": sorted(val_ids),
            "image_source": atlas.source, "atlas_coverage": atlas.coverage(),
            "vocab_size": int(ds.tokenizer.vocab_size), "seed": int(args.seed),
        },
        "context_style": style,
        "prompt_audit": prompt_audit,
        "contract_hash": contract.hash(),
        "physics": {"source": contract.grid.physics.source, "fit_method": contract.grid.physics.fit_method,
                    "quantile": contract.grid.physics.quantile,
                    "crossover_margin_m": contract.grid.physics.crossover_margin(),
                    "transition_width_m": contract.grid.physics.transition_width_m()},
        "git_sha": _git_sha(),
        "optim": {"lr": args.lr, "batch": args.batch, "accum": args.accum, "epochs": args.epochs, "amp": amp},
        "preflight": pre,
        "env": {"python": platform.python_version(), "torch": torch.__version__},
        "args": vars(args) | {"out": str(out_dir), "frames": str(args.frames) if args.frames else None},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(f"[model] {MODEL_CARDS[args.model].display} | context={args.context} | "
          f"trainable {report['params_trainable']:,}/{report['params_total']:,} "
          f"({100 * report['trainable_fraction']:.1f}%) | adapt={report['adapt']}", file=sys.stderr)
    if atlas.source == "schematic":
        print("[data] WARNING: scene images are SCHEMATIC (no Isaac frames found). Results from this "
              "run are for pipeline validation only and must not enter the headline table.", file=sys.stderr)

    metrics_path = out_dir / "metrics.jsonl"
    best = -math.inf
    step = 0
    with metrics_path.open("w") as mf:
        for epoch in range(int(args.epochs)):
            run: List[float] = []
            opt.zero_grad(set_to_none=True)
            for i, b in enumerate(dl_tr):
                b = _to(b, device)
                ctxm = torch.autocast(device.type, dtype=amp_dtype) if amp_dtype else _null()
                with ctxm:
                    out = model(b, context=b["context"], kappa=b["kappa"], beta=b.get("beta"), rho=b.get("rho"))
                    L = carryover_loss(out, b, lcfg)
                loss = L.total / max(1, int(args.accum))
                scaler.scale(loss).backward() if scaler.is_enabled() else loss.backward()
                run.append(float(L.total.detach()))
                if (i + 1) % int(args.accum) == 0:
                    if scaler.is_enabled():
                        scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    if scaler.is_enabled():
                        scaler.step(opt)
                        scaler.update()
                    else:
                        opt.step()
                    opt.zero_grad(set_to_none=True)
                    if sched.last_epoch < total_steps - 1:
                        sched.step()
                    step += 1
                    if args.max_steps and step >= int(args.max_steps):
                        break
            ev = evaluate(model, dl_va, device, lcfg)
            row = {"epoch": epoch, "step": step, "train_loss": float(np.mean(run)) if run else None,
                   "lr": float(sched.get_last_lr()[0]), "elapsed_s": time.time() - t0, **ev}
            mf.write(json.dumps(row, default=float) + "\n")
            mf.flush()
            print(f"  epoch {epoch:3d}  train {row['train_loss']:.4f}  val {ev['val_loss']:.4f}  "
                  f"acc_said {ev['acc_said']:.3f}  acc_unprompted {ev['acc_unprompted']:.3f}  "
                  f"GAIN(brier) {ev['debias_gain_brier']:+.4f}  gain(acc) {ev['debias_gain']:+.3f}  "
                  f"gap {ev['mean_abs_debias_gap']:.3f}  gap~kappa {ev['debias_kappa_corr']:+.3f}  "
                  f"ask rho {ev['ask_rank_corr']:+.3f}", file=sys.stderr)
            score = ev["debias_gain_brier"] if np.isfinite(ev["debias_gain_brier"]) else ev["debias_gain"]
            if score > best:
                best = score
                torch.save({"model": model.state_dict(), "config": model.cfg.to_dict(),
                            "tokenizer": asdict(ds.tokenizer), "manifest": manifest, "metrics": row},
                           out_dir / "best.pt")
            torch.save({"model": model.state_dict(), "config": model.cfg.to_dict(),
                        "tokenizer": asdict(ds.tokenizer), "manifest": manifest, "metrics": row},
                       out_dir / "last.pt")
            if args.max_steps and step >= int(args.max_steps):
                break

    (out_dir / "summary.json").write_text(json.dumps(
        {"best_debias_gain": best, "manifest": manifest,
         "final": row}, indent=2, default=str) + "\n")
    print(f"\nbest de-biasing gain {best:+.3f}   ->  {out_dir}", file=sys.stderr)
    return 0


def _git_sha() -> Optional[str]:
    """The commit this run was produced from, for the manifest. ``None`` outside a checkout."""
    import subprocess

    try:
        out = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True, timeout=5)
        sha = out.stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5).stdout.strip()
        return (sha + ("-dirty" if dirty else "")) if sha else None
    except Exception:                                           # pragma: no cover
        return None


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    raise SystemExit(main())
