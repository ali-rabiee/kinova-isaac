"""Test-time computation (TTC) inference with multiple selection rules and gating.

This module targets the **partial observability** paper angle: cheap probes decide when
extra samples are worth the compute, and selection goes beyond plain median consensus.

Selection methods (``selection`` / legacy ``scoring``):
- ``naive_consensus`` — negative L2 distance to the per-timestep median (legacy ``consensus``).
- ``mg_select_kl`` — leave-one-out histogram KL over discretized actions (see ``baselines/mg_select``).
- ``flow_variance`` — negative trajectory variance (useful when candidates are diverse seeds).
- ``first`` — always take the first sample.

Gating (``gating``):
- ``none`` — draw ``k_action_samples`` candidates whenever K>1.
- ``twin_uncertainty`` — two low-noise forwards; if they disagree beyond ``gating_threshold``,
  draw up to ``k_action_samples``; otherwise use a single deterministic forward.

Latency records: pass ``latency_log`` for a JSONL path; each line includes forward/score/total ms.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from .baselines import mg_select
from .models import TinyVLA
from .ttc_methods.entropy_gate import twin_sample_uncertainty, uncertainty_gate_effective_k
from .ttc_methods.flow_variance import flow_variance_score


@dataclass
class TTCConfig:
    k_action_samples: int = 4
    noise_std: float = 0.1
    # Preferred key (new). Legacy key `scoring` is still accepted in `from_dict`.
    selection: str = "naive_consensus"
    # Back-compat with older configs that used `scoring` only.
    scoring: str = "naive_consensus"
    log_jsonl: Optional[str] = None
    latency_log: Optional[str] = None

    gating: str = "none"  # none | twin_uncertainty
    gating_noise_std: float = 0.05
    gating_threshold: float = 0.02

    def to_dict(self) -> Dict[str, Any]:
        return {
            "k_action_samples": int(self.k_action_samples),
            "noise_std": float(self.noise_std),
            "selection": str(self.selection),
            "scoring": str(self.scoring),
            "log_jsonl": str(self.log_jsonl) if self.log_jsonl else None,
            "latency_log": str(self.latency_log) if self.latency_log else None,
            "gating": str(self.gating),
            "gating_noise_std": float(self.gating_noise_std),
            "gating_threshold": float(self.gating_threshold),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TTCConfig":
        # Accept `selection` or fall back to `scoring`, mapping legacy labels.
        raw_sel = d.get("selection", d.get("scoring", "naive_consensus"))
        sel = _normalize_selection_label(str(raw_sel))
        gate = str(d.get("gating", "none")).lower().strip()
        if gate not in ("none", "twin_uncertainty"):
            gate = "none"
        return cls(
            k_action_samples=int(d.get("k_action_samples", 4)),
            noise_std=float(d.get("noise_std", 0.1)),
            selection=sel,
            scoring=sel,
            log_jsonl=(str(d["log_jsonl"]) if d.get("log_jsonl") else None),
            latency_log=(str(d["latency_log"]) if d.get("latency_log") else None),
            gating=gate,
            gating_noise_std=float(d.get("gating_noise_std", 0.05)),
            gating_threshold=float(d.get("gating_threshold", 0.02)),
        )


def _normalize_selection_label(name: str) -> str:
    n = name.lower().strip()
    if n in ("consensus", "naive_consensus", "median"):
        return "naive_consensus"
    if n in ("first", "greedy"):
        return "first"
    if n in ("mg_select", "mg_select_kl", "mgselect"):
        return "mg_select_kl"
    if n in ("flow_var", "flow_variance", "variance"):
        return "flow_variance"
    if n in ("entropy_gated",):
        # Historical alias: actual gating is controlled via `gating`; keep selection sensible.
        return "naive_consensus"
    return n if n in ("naive_consensus", "mg_select_kl", "flow_variance", "first") else "naive_consensus"


def _candidate_scores(candidates: torch.Tensor, selection: str) -> torch.Tensor:
    if candidates.dim() != 3:
        raise ValueError(f"candidates must be (K,T,A); got {tuple(candidates.shape)}")
    sel = _normalize_selection_label(selection)
    if sel == "first":
        return torch.zeros(candidates.size(0), device=candidates.device, dtype=candidates.dtype)
    if sel == "naive_consensus":
        median = candidates.median(dim=0).values
        diffs = (candidates - median).pow(2).flatten(start_dim=1).mean(dim=-1)
        return -diffs
    if sel == "mg_select_kl":
        return mg_select.mg_select_kl_scores(candidates)
    if sel == "flow_variance":
        return flow_variance_score(candidates)
    median = candidates.median(dim=0).values
    diffs = (candidates - median).pow(2).flatten(start_dim=1).mean(dim=-1)
    return -diffs


def select_from_candidates(candidates: torch.Tensor, selection: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (best_chunk[T,A], scores[K])."""

    scores = _candidate_scores(candidates, selection)
    if candidates.size(0) == 1 or _normalize_selection_label(selection) == "first":
        return candidates[0], scores
    best_idx = int(scores.argmax().item())
    return candidates[best_idx], scores


class TTCPipeline:
    """TinyVLA + TTC selection / gating (single observation, B=1)."""

    def __init__(self, model: TinyVLA, cfg: TTCConfig) -> None:
        self.model = model
        self.cfg = cfg
        self._log_fp = None
        self._lat_fp = None
        if cfg.log_jsonl:
            Path(cfg.log_jsonl).parent.mkdir(parents=True, exist_ok=True)
            self._log_fp = open(cfg.log_jsonl, "a", buffering=1)
        if cfg.latency_log:
            Path(cfg.latency_log).parent.mkdir(parents=True, exist_ok=True)
            self._lat_fp = open(cfg.latency_log, "a", buffering=1)

    def close(self) -> None:
        for fp in (self._log_fp, self._lat_fp):
            if fp is not None:
                try:
                    fp.close()
                finally:
                    pass
        self._log_fp = None
        self._lat_fp = None

    @torch.no_grad()
    def predict_action_chunk(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        lang_ids: torch.Tensor,
        lang_mask: torch.Tensor,
    ) -> torch.Tensor:
        was_unbatched = image.dim() == 3
        if was_unbatched:
            image = image.unsqueeze(0)
            state = state.unsqueeze(0)
            lang_ids = lang_ids.unsqueeze(0)
            lang_mask = lang_mask.unsqueeze(0)

        t_total0 = time.time()
        forward_ms = 0.0
        score_ms = 0.0
        uncertainty = None
        gated = False

        sel = _normalize_selection_label(self.cfg.selection)
        k_max = max(1, int(self.cfg.k_action_samples))
        k_eff = k_max

        # Twin-probe gating (partial observability: spend K only when uncertain).
        if k_max > 1 and self.cfg.gating == "twin_uncertainty":
            t0 = time.time()
            out0 = self.model.forward(image, state, lang_ids, lang_mask, noise_std=0.0)
            out1 = self.model.forward(
                image, state, lang_ids, lang_mask, noise_std=float(self.cfg.gating_noise_std)
            )
            forward_ms += (time.time() - t0) * 1000.0
            a0 = out0.actions.squeeze(0)
            a1 = out1.actions.squeeze(0)
            uncertainty = float(twin_sample_uncertainty(a0, a1).item())
            k_eff = uncertainty_gate_effective_k(
                uncertainty,
                threshold=float(self.cfg.gating_threshold),
                k_when_uncertain=k_max,
                k_when_certain=1,
            )
            gated = True
            if k_eff == 1:
                best = a0
                scores = torch.zeros(1, device=best.device)
                t1 = time.time()
                score_ms = (t1 - t_total0) * 1000.0 - forward_ms
                total_ms = (t1 - t_total0) * 1000.0
                self._write_logs(
                    k_eff=k_eff,
                    k_max=k_max,
                    scores=scores,
                    selection=sel,
                    forward_ms=forward_ms,
                    score_ms=max(0.0, score_ms),
                    total_ms=total_ms,
                    uncertainty=uncertainty,
                    gated=gated,
                )
                return best
        else:
            k_eff = k_max

        t0 = time.time()
        candidates = self.model.sample_actions(
            image=image,
            state=state,
            lang_ids=lang_ids,
            lang_mask=lang_mask,
            k=int(k_eff),
            noise_std=float(self.cfg.noise_std),
        )
        forward_ms += (time.time() - t0) * 1000.0

        if candidates.size(1) != 1:
            raise ValueError("TTCPipeline.predict_action_chunk supports B=1 only.")
        candidates = candidates.squeeze(1)

        t_s0 = time.time()
        best, scores = select_from_candidates(candidates, sel)
        score_ms += (time.time() - t_s0) * 1000.0
        total_ms = (time.time() - t_total0) * 1000.0

        self._write_logs(
            k_eff=int(candidates.size(0)),
            k_max=k_max,
            scores=scores,
            selection=sel,
            forward_ms=forward_ms,
            score_ms=score_ms,
            total_ms=total_ms,
            uncertainty=uncertainty,
            gated=gated,
        )
        return best

    def _write_logs(
        self,
        *,
        k_eff: int,
        k_max: int,
        scores: torch.Tensor,
        selection: str,
        forward_ms: float,
        score_ms: float,
        total_ms: float,
        uncertainty: Optional[float],
        gated: bool,
    ) -> None:
        if self._log_fp is not None:
            try:
                self._log_fp.write(
                    json.dumps(
                        {
                            "ts": time.time(),
                            "k_eff": int(k_eff),
                            "k_max": int(k_max),
                            "selection": selection,
                            "gated": bool(gated),
                            "uncertainty": uncertainty,
                            "scores": [float(s) for s in scores.tolist()],
                            "latency_ms": float(total_ms),
                        }
                    )
                    + "\n"
                )
            except Exception:
                pass
        if self._lat_fp is not None:
            try:
                self._lat_fp.write(
                    json.dumps(
                        {
                            "ts": time.time(),
                            "forward_ms": float(forward_ms),
                            "score_ms": float(score_ms),
                            "total_ms": float(total_ms),
                            "k_eff": int(k_eff),
                            "k_max": int(k_max),
                            "selection": selection,
                            "gated": bool(gated),
                            "uncertainty": uncertainty,
                        }
                    )
                    + "\n"
                )
            except Exception:
                pass
