"""VRAM and latency dry-fit for the inference pipeline.

This is the spec's Day-3 sanity check: load the model on the deployment GPU
and run repeated forward passes to confirm the pipeline fits the budget
*before* you spend time training.

Usage:
    python -m vla_lab.dryrun --device cuda:0 --k 4 --iters 100

Notes:
- This entrypoint has no Isaac Lab dependency.
- It loads either an untrained TinyVLA built from `--config` (default) or a
  trained checkpoint (`--ckpt`) so you can profile the exact deployment model.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .models import TinyVLA, TinyVLAConfig
from .ttc import TTCConfig, TTCPipeline


def _parse_yaml_or_default(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text()) or {}


def main() -> int:
    parser = argparse.ArgumentParser(description="VLA-TTC dry-fit (VRAM + latency)")
    parser.add_argument("--config", type=str, default="vla_lab/configs/train_tiny.yaml")
    parser.add_argument("--ckpt", type=str, default=None, help="Optional trained checkpoint")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--k", type=int, default=4, help="K candidates for sample-and-verify")
    parser.add_argument("--noise-std", type=float, default=0.1)
    args = parser.parse_args()

    device = torch.device(args.device)

    # Build model.
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location=device)
        cfg = TinyVLAConfig.from_dict(ckpt["model_config"])
        model = TinyVLA(cfg).to(device)
        model.load_state_dict(ckpt["model_state"], strict=True)
    else:
        cfg_dict = _parse_yaml_or_default(args.config).get("model", {})
        cfg = TinyVLAConfig.from_dict(cfg_dict)
        model = TinyVLA(cfg).to(device)
    model.eval()

    pipe = TTCPipeline(
        model,
        TTCConfig(
            k_action_samples=int(args.k),
            noise_std=float(args.noise_std),
            scoring="consensus" if int(args.k) > 1 else "first",
        ),
    )

    # Synthetic inputs.
    image = torch.rand(1, 3, cfg.image_size, cfg.image_size, device=device)
    state = torch.zeros(1, cfg.state_dim, device=device)
    lang_ids = torch.zeros(1, cfg.max_lang_len, dtype=torch.long, device=device)
    lang_mask = torch.zeros(1, cfg.max_lang_len, dtype=torch.long, device=device)
    lang_mask[:, 0] = 1

    print(f"[dryrun] device={device}  K={args.k}  iters={args.iters}  warmup={args.warmup}")
    print(f"[dryrun] TinyVLA params: {model.num_parameters():,}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Warmup.
    with torch.no_grad():
        for _ in range(int(args.warmup)):
            _ = pipe.predict_action_chunk(image[0], state[0], lang_ids[0], lang_mask[0])

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    # Timed runs.
    durations_ms = []
    with torch.no_grad():
        for _ in range(int(args.iters)):
            t0 = time.time()
            _ = pipe.predict_action_chunk(image[0], state[0], lang_ids[0], lang_mask[0])
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            durations_ms.append((time.time() - t0) * 1000.0)

    p50 = statistics.median(durations_ms)
    p95 = sorted(durations_ms)[int(0.95 * len(durations_ms)) - 1]
    p99 = sorted(durations_ms)[int(0.99 * len(durations_ms)) - 1]
    mean_ms = sum(durations_ms) / len(durations_ms)

    print(f"[dryrun] latency  mean={mean_ms:.1f}ms  p50={p50:.1f}ms  p95={p95:.1f}ms  p99={p99:.1f}ms")

    if device.type == "cuda":
        peak_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
        print(f"[dryrun] CUDA memory  peak_alloc={peak_alloc:.2f}GiB  peak_reserved={peak_reserved:.2f}GiB")

    pipe.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
