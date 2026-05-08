"""Single-GPU training entrypoint for TinyVLA on `vla_v1` collect_data logs.

Usage:
    python -m vla_lab.train --config vla_lab/configs/train_tiny.yaml
    python -m vla_lab.train --config vla_lab/configs/train_tiny.yaml --auto-resume
    python -m vla_lab.train --config ... --resume vla_lab/checkpoints/run/last.pt
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "vla_lab.train requires PyYAML. Install with `pip install pyyaml`."
    ) from exc

from .dataset import (
    ActionStats,
    DatasetConfig,
    KinovaSessionDataset,
    TinyTokenizer,
    compute_action_stats,
    discover_episodes,
    split_episodes,
)
from .losses import FeatureAlignmentConfig, FeatureAlignmentLoss, masked_action_loss
from .models import TinyVLA, TinyVLAConfig


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _format_minutes(seconds: float) -> str:
    return f"{seconds / 60:.1f}m"


def _build_lr_scheduler(opt: torch.optim.Optimizer, total_steps: int, warmup: int) -> torch.optim.lr_scheduler.LambdaLR:
    def fn(step: int) -> float:
        if step < warmup:
            return float(step) / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=fn)


def _torch_load_checkpoint(path: Path, map_location: torch.device) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"map_location": map_location}
    if "weights_only" in inspect.signature(torch.load).parameters:  # PyTorch >= 2.0
        kwargs["weights_only"] = False
    return torch.load(str(path), **kwargs)


def _rng_state_payload() -> Dict[str, Any]:
    cuda_states = None
    if torch.cuda.is_available():
        try:
            cuda_states = [s.cpu() for s in torch.cuda.get_rng_state_all()]
        except Exception:  # pragma: no cover
            cuda_states = None
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": cuda_states,
    }


def _load_rng_state(payload: Optional[Dict[str, Any]]) -> None:
    if not payload:
        return
    try:
        if "python" in payload:
            random.setstate(payload["python"])
        if "numpy" in payload:
            np.random.set_state(payload["numpy"])
        if "torch" in payload:
            torch.set_rng_state(payload["torch"])
        if payload.get("torch_cuda") is not None and torch.cuda.is_available():
            states = payload["torch_cuda"]
            torch.cuda.set_rng_state_all([s.cuda() for s in states])
    except Exception:  # pragma: no cover
        pass


def _append_metrics_line(metrics_path: Path, obj: Dict[str, Any]) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, default=str) + "\n")


def _maybe_plot_at_end(run_dir: Path, plot_eval_json: List[str]) -> None:
    try:
        from .plot_metrics import plot_run_dir
    except ImportError:
        return
    try:
        plot_run_dir(run_dir, eval_json_paths=[Path(p) for p in plot_eval_json if p], fig_format="pdf")
    except Exception as exc:  # pragma: no cover
        print(f"[train] plot_metrics skipped: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description="TinyVLA trainer")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--data-roots",
        nargs="+",
        type=str,
        default=None,
        help="Override config 'data.data_roots'.",
    )
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-feature-alignment", action="store_true")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint (.pt): restores model, optimizer, scheduler, step, and dataloader position when possible.",
    )
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Resume from out_dir/last.pt if it exists.",
    )
    parser.add_argument(
        "--no-plot-at-end",
        action="store_true",
        help="Skip automatic plot_metrics after training (requires matplotlib).",
    )
    parser.add_argument(
        "--plot-eval-json",
        action="append",
        default=[],
        help="Optional eval result JSON path(s) for figures (repeatable).",
    )
    args = parser.parse_args()

    cfg = _load_yaml(Path(args.config))
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})

    if args.data_roots is not None:
        data_cfg["data_roots"] = list(args.data_roots)
    if args.out_dir is not None:
        train_cfg["out_dir"] = args.out_dir
    if args.device is not None:
        train_cfg["device"] = args.device
    if args.epochs is not None:
        train_cfg["num_epochs"] = int(args.epochs)
    if args.batch_size is not None:
        train_cfg["batch_size"] = int(args.batch_size)
    if args.no_feature_alignment:
        train_cfg.setdefault("feature_alignment", {})["enabled"] = False

    seed = int(train_cfg.get("seed", 42))
    _seed_everything(seed)

    out_dir = Path(train_cfg.get("out_dir", "vla_lab/checkpoints/tiny_v0"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))

    resume_path: Optional[Path] = None
    if args.resume:
        resume_path = Path(args.resume)
    elif args.auto_resume:
        candidate = out_dir / "last.pt"
        if candidate.is_file():
            resume_path = candidate

    device = torch.device(str(train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")))
    if device.type == "cuda" and bool(train_cfg.get("cudnn_benchmark", True)):
        torch.backends.cudnn.benchmark = True

    roots: List[Path] = [Path(p) for p in data_cfg.get("data_roots", ["logs/data_collection"])]
    print(f"[train] discovering episodes under: {[str(r) for r in roots]}")
    episodes = discover_episodes(roots)
    if not episodes:
        print("[train] ERROR: no episodes found. Run data collection first (see vla_lab/README.md).")
        return 2
    print(f"[train] found {len(episodes)} episodes, total ticks: {sum(len(e.ticks) for e in episodes)}")

    train_eps, val_eps = split_episodes(
        episodes,
        val_fraction=float(data_cfg.get("val_fraction", 0.1)),
        seed=int(data_cfg.get("split_seed", 0)),
    )
    print(f"[train] split: train={len(train_eps)}  val={len(val_eps)}")

    sentences = [ep.instruction for ep in episodes if ep.instruction]
    if not sentences:
        sentences = ["pick up the object."]
    tokenizer = TinyTokenizer.build_from_corpus(
        sentences,
        max_len=int(model_cfg.get("max_lang_len", 24)),
    )
    print(f"[train] tokenizer vocab size: {tokenizer.vocab_size}")

    action_stats: Optional[ActionStats] = None
    if bool(data_cfg.get("normalize_actions", True)):
        action_stats = compute_action_stats(train_eps)
        print(f"[train] action stats: mean={action_stats.mean.tolist()}, std={action_stats.std.tolist()}")

    ds_cfg = DatasetConfig(
        image_size=int(data_cfg.get("image_size", 224)),
        chunk_len=int(data_cfg.get("chunk_len", model_cfg.get("chunk_len", 8))),
        state_dim=int(data_cfg.get("state_dim", model_cfg.get("state_dim", 4))),
        action_dim=int(data_cfg.get("action_dim", model_cfg.get("action_dim", 7))),
        drop_no_image=bool(data_cfg.get("drop_no_image", True)),
        fast_image_io=bool(data_cfg.get("fast_image_io", True)),
    )

    train_ds = KinovaSessionDataset(train_eps, tokenizer, ds_cfg, action_stats=action_stats, train=True)
    val_ds: Optional[KinovaSessionDataset] = None
    if val_eps:
        try:
            val_ds = KinovaSessionDataset(val_eps, tokenizer, ds_cfg, action_stats=action_stats, train=False)
        except RuntimeError:
            val_ds = None
    print(f"[train] frames: train={len(train_ds)}  val={len(val_ds) if val_ds else 0}")

    batch_size = int(train_cfg.get("batch_size", 64))
    nw_tr = int(train_cfg.get("num_workers", 8))
    pf = train_cfg.get("prefetch_factor")
    prefetch = int(pf) if pf is not None else (4 if nw_tr > 0 else 2)
    pers = bool(train_cfg.get("persistent_workers", True)) and nw_tr > 0
    train_ld: Dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": True,
        "drop_last": True,
        "pin_memory": device.type == "cuda",
    }
    if nw_tr > 0:
        train_ld["num_workers"] = nw_tr
        train_ld["prefetch_factor"] = prefetch
        if pers:
            train_ld["persistent_workers"] = True
    else:
        train_ld["num_workers"] = 0

    train_loader = DataLoader(train_ds, **train_ld)

    val_nw = train_cfg.get("val_num_workers")
    if val_nw is None:
        val_nw = max(1, nw_tr // 2) if nw_tr > 0 else 0
    else:
        val_nw = int(val_nw)
    val_prefetch = int(train_cfg.get("val_prefetch_factor", prefetch)) if val_nw > 0 else None
    val_ld: Dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "drop_last": False,
        "pin_memory": device.type == "cuda",
    }
    if val_nw > 0:
        val_ld["num_workers"] = val_nw
        val_ld["prefetch_factor"] = val_prefetch or 2
        if bool(train_cfg.get("persistent_workers", True)):
            val_ld["persistent_workers"] = True
    else:
        val_ld["num_workers"] = 0

    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(val_ds, **val_ld)

    model_cfg.setdefault("vocab_size", tokenizer.vocab_size)
    model_cfg.setdefault("pad_id", tokenizer.pad_id)
    model_cfg.setdefault("max_lang_len", tokenizer.max_len)
    model_cfg["chunk_len"] = ds_cfg.chunk_len
    model_cfg["action_dim"] = ds_cfg.action_dim
    model_cfg["state_dim"] = ds_cfg.state_dim

    cfg_obj = TinyVLAConfig.from_dict(model_cfg)
    model = TinyVLA(cfg_obj).to(device)
    print(f"[train] TinyVLA params: {model.num_parameters():,}")

    fa_cfg_dict = train_cfg.get("feature_alignment", {})
    fa_cfg = FeatureAlignmentConfig(
        teacher_name=str(fa_cfg_dict.get("teacher_name", "facebook/dinov2-base")),
        teacher_dim=int(fa_cfg_dict.get("teacher_dim", 768)),
        image_size=int(data_cfg.get("image_size", 224)),
        pool_to_tokens=int(fa_cfg_dict.get("pool_to_tokens", 64)),
        enabled=bool(fa_cfg_dict.get("enabled", False)),
    )
    fa_weight = float(fa_cfg_dict.get("weight", 0.5))
    fa_loss: Optional[FeatureAlignmentLoss] = None
    if fa_cfg.enabled:
        fa_loss = FeatureAlignmentLoss(fa_cfg, student_dim=cfg_obj.embed_dim).to(device)
        print(f"[train] feature alignment ENABLED (teacher={fa_cfg.teacher_name}, w={fa_weight})")
    else:
        print("[train] feature alignment disabled")

    params = [{"params": [p for p in model.parameters() if p.requires_grad]}]
    if fa_loss is not None:
        params.append({"params": [p for p in fa_loss.parameters() if p.requires_grad]})
    optim = torch.optim.AdamW(
        params,
        lr=float(train_cfg.get("lr", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )

    num_epochs = int(train_cfg.get("num_epochs", 30))
    total_steps = max(1, num_epochs * max(1, len(train_loader)))
    warmup_steps = int(train_cfg.get("warmup_steps", 200))
    sched = _build_lr_scheduler(optim, total_steps, warmup_steps)
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    log_every = int(train_cfg.get("log_every", 50))
    gw = float(train_cfg.get("gripper_weight", 1.0))

    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    ads = str(train_cfg.get("amp_dtype", "auto")).lower()
    if ads == "auto":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif ads in ("bf16", "bfloat16"):
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler: Optional[torch.amp.GradScaler] = None
    if device.type == "cuda" and use_scaler:
        scaler = torch.amp.GradScaler("cuda")

    ckpt_every_epochs = int(train_cfg.get("ckpt_every_epochs", 5))
    save_every_steps = int(train_cfg.get("save_every_steps", 0))
    max_step_ckpts = int(train_cfg.get("max_step_checkpoints", 3))
    metrics_path = out_dir / str(train_cfg.get("metrics_file", "metrics.jsonl"))

    t_start = time.time()
    global_step = 0
    best_val: Optional[float] = None
    start_epoch = 0
    skip_batches_resume = 0

    def _build_ckpt_dict(
        *,
        next_epoch: int,
        skip_batches: int,
        step: int,
        val_loss: Optional[float],
    ) -> Dict[str, Any]:
        return {
            "model_state": model.state_dict(),
            "model_config": cfg_obj.to_dict(),
            "tokenizer": tokenizer.to_dict(),
            "action_stats": action_stats.to_dict() if action_stats else None,
            "feature_alignment_state": fa_loss.state_dict() if fa_loss is not None else None,
            "feature_alignment_config": {**fa_cfg_dict, "enabled": bool(fa_cfg.enabled)},
            "optimizer_state": optim.state_dict(),
            "scheduler_state": sched.state_dict(),
            # Progress: next_epoch = epoch index to start/continue; skip_batches = batches already done in that epoch
            "next_epoch": int(next_epoch),
            "skip_batches": int(skip_batches),
            "step": int(step),
            "val_loss": float(val_loss) if val_loss is not None else None,
            "best_val_loss": float(best_val) if best_val is not None else None,
            "config": cfg,
            "train_meta": {
                "total_steps": total_steps,
                "warmup_steps": warmup_steps,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "seed": seed,
                "use_amp": use_amp,
                "amp_dtype": str(amp_dtype),
                "use_scaler": use_scaler,
            },
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "rng_state": _rng_state_payload(),
        }

    def _save_ckpt(tag: str, next_epoch: int, skip_batches: int, step: int, val_loss: Optional[float]) -> Path:
        path = out_dir / f"{tag}.pt"
        ckpt = _build_ckpt_dict(next_epoch=next_epoch, skip_batches=skip_batches, step=step, val_loss=val_loss)
        torch.save(ckpt, str(path))
        return path

    def _prune_step_ckpts() -> None:
        if max_step_ckpts <= 0:
            return
        step_files = sorted(out_dir.glob("step_*.pt"))
        while len(step_files) > max_step_ckpts:
            try:
                step_files[0].unlink()
            except OSError:
                pass
            step_files = step_files[1:]

    if resume_path is not None:
        if not resume_path.is_file():
            print(f"[train] ERROR: resume path not found: {resume_path}")
            return 2
        print(f"[train] loading checkpoint: {resume_path}")
        ckpt = _torch_load_checkpoint(resume_path, map_location=device)
        meta = ckpt.get("train_meta") or {}
        mismatch = False
        if meta.get("total_steps") is not None and int(meta["total_steps"]) != total_steps:
            print(
                f"[train] WARNING: total_steps mismatch (ckpt={meta.get('total_steps')} vs now={total_steps}). "
                "Change epochs/batch size? Optimizer scheduler may be wrong; prefer fresh run or matching config."
            )
            mismatch = True

        model.load_state_dict(ckpt["model_state"], strict=True)
        if fa_loss is not None and ckpt.get("feature_alignment_state"):
            fa_loss.load_state_dict(ckpt["feature_alignment_state"], strict=True)
        if ckpt.get("action_stats") and action_stats is not None:
            loaded = ActionStats.from_dict(ckpt["action_stats"])
            if not torch.allclose(loaded.mean, action_stats.mean, rtol=1e-3) or not torch.allclose(
                loaded.std, action_stats.std, rtol=1e-3
            ):
                print("[train] WARNING: action_stats in checkpoint differ from current train split.")

        if not mismatch:
            if ckpt.get("optimizer_state"):
                try:
                    optim.load_state_dict(ckpt["optimizer_state"])
                except Exception as exc:
                    print(f"[train] WARNING: optimizer load failed ({exc}); continuing with new optimizer state.")
            if ckpt.get("scheduler_state"):
                try:
                    sched.load_state_dict(ckpt["scheduler_state"])
                except Exception as exc:
                    print(f"[train] WARNING: scheduler load failed ({exc}).")
            if scaler is not None and ckpt.get("scaler_state"):
                try:
                    scaler.load_state_dict(ckpt["scaler_state"])
                except Exception as exc:
                    print(f"[train] WARNING: GradScaler load failed ({exc}).")
        else:
            print("[train] Not loading optimizer/scheduler state due to train shape mismatch.")

        global_step = int(ckpt.get("step", 0))
        if "next_epoch" in ckpt:
            start_epoch = int(ckpt["next_epoch"])
            skip_batches_resume = int(ckpt.get("skip_batches", 0))
        else:
            # Legacy checkpoint
            start_epoch = int(ckpt.get("epoch", -1)) + 1
            skip_batches_resume = 0
        bv = ckpt.get("best_val_loss")
        if bv is not None:
            best_val = float(bv)
        elif ckpt.get("val_loss") is not None:
            best_val = float(ckpt["val_loss"])
        _load_rng_state(ckpt.get("rng_state"))
        print(
            f"[train] resumed: start_epoch={start_epoch} skip_batches={skip_batches_resume} "
            f"global_step={global_step} best_val={best_val}"
        )
        _append_metrics_line(
            metrics_path,
            {
                "type": "resume",
                "from": str(resume_path),
                "start_epoch": start_epoch,
                "skip_batches": skip_batches_resume,
                "step": global_step,
                "wall_time_s": round(time.time() - t_start, 3),
            },
        )

    print(
        f"[train] perf: batch={batch_size} train_workers={nw_tr} prefetch={prefetch} "
        f"pin_mem={device.type == 'cuda'} cudnn_benchmark={device.type == 'cuda' and bool(train_cfg.get('cudnn_benchmark', True))}"
    )
    print(f"[train] amp={'on' if use_amp else 'off'} dtype={amp_dtype} grad_scaler={scaler is not None}")
    print(
        f"[train] starting: epochs={num_epochs} start_epoch={start_epoch} "
        f"steps/epoch={len(train_loader)} total_steps={total_steps}"
    )

    def _forward_ctx():
        if device.type == "cuda" and use_amp:
            return torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
        return nullcontext()

    skip_batches_once = skip_batches_resume
    resume_epoch_apply_skip = start_epoch

    for epoch in range(start_epoch, num_epochs):
        model.train()
        if fa_loss is not None:
            fa_loss.train()
        ep_losses: List[float] = []
        ep_act_losses: List[float] = []
        ep_fa_losses: List[float] = []

        loader_iter = iter(train_loader)
        batches_done_in_epoch = 0
        if epoch == resume_epoch_apply_skip and skip_batches_once > 0:
            print(f"[train] skipping first {skip_batches_once} batches of epoch {epoch} (resume)")
            try:
                for _ in range(skip_batches_once):
                    next(loader_iter)
                    batches_done_in_epoch += 1
            except StopIteration:
                print("[train] WARNING: skip_batches exceeded loader length; continuing next epoch.")
            skip_batches_once = 0

        for batch in loader_iter:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with _forward_ctx():
                out = model(
                    image=batch["image"],
                    state=batch["state"],
                    lang_ids=batch["lang_ids"],
                    lang_mask=batch["lang_mask"],
                )

                act_loss = masked_action_loss(
                    out.actions, batch["actions"], batch["action_mask"], gripper_weight=gw
                )
                fa_l = torch.zeros((), device=device)
                if fa_loss is not None and fa_cfg.enabled:
                    fa_l = fa_loss(out.features, batch["image"])

                loss = act_loss + fa_weight * fa_l

            optim.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optim.step()
            sched.step()

            ep_losses.append(float(loss.item()))
            ep_act_losses.append(float(act_loss.item()))
            ep_fa_losses.append(float(fa_l.item()) if fa_loss is not None else 0.0)
            global_step += 1
            batches_done_in_epoch += 1

            if global_step % log_every == 0:
                lr_now = optim.param_groups[0]["lr"]
                elapsed = time.time() - t_start
                print(
                    f"[train] ep={epoch:02d} step={global_step:6d} "
                    f"loss={float(np.mean(ep_losses[-log_every:])):.4f} "
                    f"act={float(np.mean(ep_act_losses[-log_every:])):.4f} "
                    f"fa={float(np.mean(ep_fa_losses[-log_every:])):.4f} "
                    f"lr={lr_now:.2e}  elapsed={_format_minutes(elapsed)}"
                )
                _append_metrics_line(
                    metrics_path,
                    {
                        "type": "train_log",
                        "epoch": epoch,
                        "step": global_step,
                        "loss": float(np.mean(ep_losses[-log_every:])),
                        "act_loss": float(np.mean(ep_act_losses[-log_every:])),
                        "fa_loss": float(np.mean(ep_fa_losses[-log_every:])),
                        "lr": float(lr_now),
                        "wall_time_s": round(elapsed, 3),
                    },
                )

            if save_every_steps > 0 and global_step % save_every_steps == 0:
                p = _save_ckpt(f"step_{global_step:07d}", epoch, batches_done_in_epoch, global_step, None)
                print(f"[train] step checkpoint {p}")
                _prune_step_ckpts()

        val_loss: Optional[float] = None
        if val_loader is not None:
            model.eval()
            if fa_loss is not None:
                fa_loss.eval()
            with torch.no_grad():
                losses = []
                for batch in val_loader:
                    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                    with _forward_ctx():
                        out = model(
                            image=batch["image"],
                            state=batch["state"],
                            lang_ids=batch["lang_ids"],
                            lang_mask=batch["lang_mask"],
                        )
                        losses.append(
                            float(
                                masked_action_loss(
                                    out.actions, batch["actions"], batch["action_mask"], gripper_weight=gw
                                ).item()
                            )
                        )
                val_loss = float(np.mean(losses)) if losses else None

        train_loss = float(np.mean(ep_losses)) if ep_losses else float("nan")
        msg = f"[train] EP {epoch:02d} done: train_loss={train_loss:.4f}"
        if val_loss is not None:
            msg += f"  val_loss={val_loss:.4f}"
        print(msg)

        _append_metrics_line(
            metrics_path,
            {
                "type": "epoch_end",
                "epoch": epoch,
                "step": global_step,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "wall_time_s": round(time.time() - t_start, 3),
            },
        )

        next_epoch_after = epoch + 1
        if (epoch + 1) % max(1, ckpt_every_epochs) == 0 or (epoch + 1) == num_epochs:
            p = _save_ckpt("last", next_epoch_after, 0, global_step, val_loss)
            print(f"[train] saved checkpoint {p}")
        if val_loss is not None and (best_val is None or val_loss < best_val):
            best_val = val_loss
            p = _save_ckpt("best", next_epoch_after, 0, global_step, val_loss)
            print(f"[train] new best (val_loss={val_loss:.4f}) -> {p}")

    print(f"[train] done in {_format_minutes(time.time() - t_start)}")
    abs_out = out_dir.resolve()
    print(f"[train] run directory: {abs_out}")
    print(f"[train] last checkpoint: {(abs_out / 'last.pt')}")
    print(f"[train] metrics log: {(abs_out / str(train_cfg.get('metrics_file', 'metrics.jsonl')))}")
    print(f"[train] plot: RUN_DIR={abs_out} ./vla_lab/scripts/plot.sh")

    if not args.no_plot_at_end:
        _maybe_plot_at_end(out_dir, list(args.plot_eval_json))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
