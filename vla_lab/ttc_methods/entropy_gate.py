"""Uncertainty gating: spend extra samples only when cheap probes disagree.

Full SmolVLA action-token entropy requires model internals; this module implements a
**lightweight proxy** suitable for both TinyVLA and SmolVLA wrappers: twin stochastic
forward passes and an L2 (or max) norm on the resulting action chunks.
"""

from __future__ import annotations

from typing import Tuple

import torch


def twin_sample_uncertainty(a0: torch.Tensor, a1: torch.Tensor) -> torch.Tensor:
    """Scalar disagreement between two (T, A) chunks (mean L2 over timesteps)."""

    if a0.shape != a1.shape:
        raise ValueError(f"Shape mismatch {tuple(a0.shape)} vs {tuple(a1.shape)}")
    return (a0 - a1).reshape(a0.shape[0], -1).pow(2).mean().sqrt()


def uncertainty_gate_effective_k(
    uncertainty: float,
    *,
    threshold: float,
    k_when_uncertain: int,
    k_when_certain: int = 1,
) -> int:
    """Return how many candidates to draw (at least 1)."""

    ku = max(1, int(k_when_uncertain))
    kc = max(1, int(k_when_certain))
    if float(uncertainty) > float(threshold):
        return ku
    return kc


def shannon_entropy_from_probs(probs: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """H(p) = -sum p log p. `probs` should be non-negative and sum to 1 along `dim`."""

    p = probs.clamp_min(eps)
    return -(p * p.log()).sum(dim=dim)


def token_entropy_gate(token_logits: torch.Tensor, threshold_nat: float) -> Tuple[bool, float]:
    """If first-step marginal entropy exceeds threshold, signal to use k>1.

    `token_logits`: (V,) unnormalized logits for a single categorical (e.g. first action token).
    Returns (should_expand, entropy_nats).
    """

    if token_logits.dim() != 1:
        raise ValueError("token_logits must be shape (V,)")
    logp = torch.log_softmax(token_logits, dim=0)
    p = logp.exp()
    ent = float((-(p * logp)).sum().item())
    return ent > float(threshold_nat), ent
