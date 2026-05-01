"""Minimal Test-Time Computation (TTC) inference pipeline.

This is the Phase-1 fallback described in the engineering spec
(`vla_lab/vla_ttc_engineering_spec.md`, sections 4.11/4.12 + the
"Brutally honest things to know" note that the verifier may need to fall
back to a cheaper scorer).

What is implemented here:
- K-noise sampling: run the model K times with independent Gaussian
  noise injected at the bottleneck (delegated to
  `models.TinyVLA.sample_actions`).
- Consensus scoring: pick the candidate whose action chunk is closest to
  the K-sample median (Phase-1 fallback for the verifier). This gives a
  cheap, model-agnostic robustness boost over k=1 single sample.
- Per-step JSON-lines logging of inference decisions for offline analysis.

Components left as TODO for later phases:
- Learned verifier (Section 4.9): replace `_consensus_score` with a
  trained scorer that takes (image_features, language, candidate_chunk).
- OOD detector (RND, Section 4.11): swap the always-slow path for a
  triggered fast/slow path based on a calibrated novelty score.
- Scene parser + inpainter (Section 4.10): pre-process image with
  Qwen2.5-VL + Grounding-DINO + SAM2 before model inference.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .models import TinyVLA, ModelOutput


@dataclass
class TTCConfig:
    k_action_samples: int = 4
    noise_std: float = 0.1
    scoring: str = "consensus"  # one of {"consensus", "first"}
    log_jsonl: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "k_action_samples": int(self.k_action_samples),
            "noise_std": float(self.noise_std),
            "scoring": str(self.scoring),
            "log_jsonl": str(self.log_jsonl) if self.log_jsonl else None,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TTCConfig":
        return cls(
            k_action_samples=int(d.get("k_action_samples", 4)),
            noise_std=float(d.get("noise_std", 0.1)),
            scoring=str(d.get("scoring", "consensus")),
            log_jsonl=(str(d["log_jsonl"]) if d.get("log_jsonl") else None),
        )


class TTCPipeline:
    """Wraps a TinyVLA and exposes a simple `predict_action_chunk` method.

    The pipeline runs everything on a single device (the model's device).
    For a single observation it returns the *best* (T, A) action chunk
    according to the configured scoring rule.
    """

    def __init__(self, model: TinyVLA, cfg: TTCConfig) -> None:
        self.model = model
        self.cfg = cfg
        self._log_fp = None
        if cfg.log_jsonl:
            Path(cfg.log_jsonl).parent.mkdir(parents=True, exist_ok=True)
            self._log_fp = open(cfg.log_jsonl, "a", buffering=1)

    def close(self) -> None:
        if self._log_fp is not None:
            try:
                self._log_fp.close()
            finally:
                self._log_fp = None

    @staticmethod
    def _consensus_score(candidates: torch.Tensor) -> torch.Tensor:
        """Score by negative L2 distance to the median candidate.

        - candidates: (K, T, A)
        - returns:    (K,)  - higher is better
        """

        median = candidates.median(dim=0).values  # (T, A)
        diffs = (candidates - median).pow(2).flatten(start_dim=1).mean(dim=-1)  # (K,)
        return -diffs

    @torch.no_grad()
    def predict_action_chunk(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        lang_ids: torch.Tensor,
        lang_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return the best action chunk of shape (T, A) for a single sample.

        Inputs may be unbatched (no leading B dim); we add and remove the
        batch dim internally.
        """

        was_unbatched = image.dim() == 3
        if was_unbatched:
            image = image.unsqueeze(0)
            state = state.unsqueeze(0)
            lang_ids = lang_ids.unsqueeze(0)
            lang_mask = lang_mask.unsqueeze(0)

        t0 = time.time()
        candidates = self.model.sample_actions(
            image=image,
            state=state,
            lang_ids=lang_ids,
            lang_mask=lang_mask,
            k=int(self.cfg.k_action_samples),
            noise_std=float(self.cfg.noise_std),
        )  # (K, B, T, A)
        latency_ms = (time.time() - t0) * 1000.0

        # Squeeze batch dim out for the single-sample case.
        if candidates.size(1) != 1:
            raise ValueError("TTCPipeline.predict_action_chunk supports B=1 only.")
        candidates = candidates.squeeze(1)  # (K, T, A)

        if candidates.size(0) == 1 or self.cfg.scoring == "first":
            best = candidates[0]
            scores = torch.zeros(candidates.size(0), device=candidates.device)
        else:
            scores = self._consensus_score(candidates)
            best_idx = int(scores.argmax().item())
            best = candidates[best_idx]

        if self._log_fp is not None:
            try:
                self._log_fp.write(
                    json.dumps(
                        {
                            "ts": time.time(),
                            "k": int(candidates.size(0)),
                            "scoring": str(self.cfg.scoring),
                            "scores": [float(s) for s in scores.tolist()],
                            "latency_ms": float(latency_ms),
                        }
                    )
                    + "\n"
                )
            except Exception:
                pass

        return best
