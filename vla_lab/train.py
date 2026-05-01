"""Single-GPU training entrypoint for TinyVLA on `vla_v1` collect_data logs.

Usage:
    python -m vla_lab.train --config vla_lab/configs/train_tiny.yaml

This entrypoint has no Isaac Lab dependency and can be run anywhere with
PyTorch + Pillow + PyYAML installed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
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
    args = parser.parse_args()

    cfg = _load_yaml(Path(args.config))
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})

    # CLI overrides
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

    device = torch.device(str(train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")))

    # ------------------------------------------------------------------
    # Data discovery
    # ------------------------------------------------------------------
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

    # Tokenizer + action stats
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
    )

    train_ds = KinovaSessionDataset(train_eps, tokenizer, ds_cfg, action_stats=action_stats, train=True)
    val_ds: Optional[KinovaSessionDataset] = None
    if val_eps:
        try:
            val_ds = KinovaSessionDataset(val_eps, tokenizer, ds_cfg, action_stats=action_stats, train=False)
        except RuntimeError:
            val_ds = None
    print(f"[train] frames: train={len(train_ds)}  val={len(val_ds) if val_ds else 0}")

    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg.get("batch_size", 32)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 4)),
        drop_last=True,
        pin_memory=device.type == "cuda",
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=int(train_cfg.get("batch_size", 32)),
            shuffle=False,
            num_workers=int(train_cfg.get("num_workers", 2)),
            drop_last=False,
            pin_memory=device.type == "cuda",
        )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model_cfg.setdefault("vocab_size", tokenizer.vocab_size)
    model_cfg.setdefault("pad_id", tokenizer.pad_id)
    model_cfg.setdefault("max_lang_len", tokenizer.max_len)
    model_cfg["chunk_len"] = ds_cfg.chunk_len
    model_cfg["action_dim"] = ds_cfg.action_dim
    model_cfg["state_dim"] = ds_cfg.state_dim

    cfg_obj = TinyVLAConfig.from_dict(model_cfg)
    model = TinyVLA(cfg_obj).to(device)
    print(f"[train] TinyVLA params: {model.num_parameters():,}")

    # Feature alignment (optional)
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
        print(f"[train] feature alignment disabled")

    # Optimizer
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
    sched = _build_lr_scheduler(optim, total_steps, int(train_cfg.get("warmup_steps", 200)))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    log_every = int(train_cfg.get("log_every", 50))
    gw = float(train_cfg.get("gripper_weight", 1.0))

    def _save_ckpt(tag: str, epoch: int, step: int, val_loss: Optional[float]) -> Path:
        path = out_dir / f"{tag}.pt"
        ckpt = {
            "model_state": model.state_dict(),
            "model_config": cfg_obj.to_dict(),
            "tokenizer": tokenizer.to_dict(),
            "action_stats": action_stats.to_dict() if action_stats else None,
            "feature_alignment_state": fa_loss.state_dict() if fa_loss is not None else None,
            "feature_alignment_config": {
                **fa_cfg_dict,
                "enabled": bool(fa_cfg.enabled),
            },
            "epoch": int(epoch),
            "step": int(step),
            "val_loss": float(val_loss) if val_loss is not None else None,
            "config": cfg,
        }
        torch.save(ckpt, str(path))
        return path

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print(f"[train] starting: epochs={num_epochs} steps_per_epoch={len(train_loader)} total_steps={total_steps}")
    global_step = 0
    best_val: Optional[float] = None
    t_start = time.time()
    for epoch in range(num_epochs):
        model.train()
        if fa_loss is not None:
            fa_loss.train()
        ep_losses: List[float] = []
        ep_act_losses: List[float] = []
        ep_fa_losses: List[float] = []

        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
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
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()
            sched.step()

            ep_losses.append(float(loss.item()))
            ep_act_losses.append(float(act_loss.item()))
            ep_fa_losses.append(float(fa_l.item()) if fa_loss is not None else 0.0)
            global_step += 1

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

        # ------- Validation -------
        val_loss: Optional[float] = None
        if val_loader is not None:
            model.eval()
            if fa_loss is not None:
                fa_loss.eval()
            with torch.no_grad():
                losses = []
                for batch in val_loader:
                    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
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

        ckpt_every = int(train_cfg.get("ckpt_every_epochs", 5))
        if (epoch + 1) % max(1, ckpt_every) == 0 or (epoch + 1) == num_epochs:
            p = _save_ckpt("last", epoch, global_step, val_loss)
            print(f"[train] saved checkpoint {p}")
        if val_loss is not None and (best_val is None or val_loss < best_val):
            best_val = val_loss
            p = _save_ckpt("best", epoch, global_step, val_loss)
            print(f"[train] new best (val_loss={val_loss:.4f}) -> {p}")

    print(f"[train] done in {_format_minutes(time.time() - t_start)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
