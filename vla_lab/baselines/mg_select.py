"""MG-Select–style selection over continuous action chunks (practical reimplementation).

The published method operates on token distributions from a masked reference policy.
Here we apply the same *leave-one-out reference + KL* idea to **discretized action
chunks**: each candidate yields a histogram over shared bins; the reference for
candidate *i* pools the other K−1 histograms. The score is negative
KL(P_i || P_ref) so higher is better.

This matches the codebase need to score K sampled trajectories without a separate
verifier network. For very small K, use `mg_select_loo_l2_scores` as a stable fallback.
"""

from __future__ import annotations

import torch


def _kl_div(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """KL(p || q) for distributions p, q of shape (..., D)."""

    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return (p * (p.log() - q.log())).sum(dim=-1)


def mg_select_kl_scores(candidates: torch.Tensor, *, num_bins: int = 32, eps: float = 1e-6) -> torch.Tensor:
    """Score each candidate chunk; return shape (K,) higher is better.

    - candidates: (K, T, A) action chunks (normalized or physical, consistent is enough).
    """

    if candidates.dim() != 3:
        raise ValueError(f"Expected candidates (K,T,A); got {tuple(candidates.shape)}")
    k = int(candidates.shape[0])
    if k <= 1:
        return torch.zeros(k, device=candidates.device, dtype=candidates.dtype)

    flat = candidates.reshape(k, -1)
    lo = flat.min(dim=0).values
    hi = flat.max(dim=0).values
    span = (hi - lo).clamp_min(1e-6)
    z = (flat - lo) / span
    idx = (z * float(num_bins - 1)).long().clamp(0, num_bins - 1)  # (K, D)
    d = idx.shape[1]
    vocab = num_bins
    # Histogram per candidate: product space would be huge; use mean of per-dim categoricals.
    scores = torch.zeros(k, device=candidates.device, dtype=candidates.dtype)
    for dim in range(d):
        hist = torch.zeros(k, vocab, device=candidates.device, dtype=candidates.dtype)
        hist.scatter_add_(1, idx[:, dim : dim + 1], torch.ones_like(idx[:, dim : dim + 1], dtype=candidates.dtype))
        hist = hist / hist.sum(dim=1, keepdim=True).clamp_min(eps)
        ref = (hist.sum(dim=0, keepdim=True) - hist) / float(max(1, k - 1))
        ref = ref / ref.sum(dim=1, keepdim=True).clamp_min(eps)
        scores = scores - _kl_div(hist, ref.expand_as(hist), eps=eps)
    return scores / float(max(1, d))


def mg_select_loo_l2_scores(candidates: torch.Tensor) -> torch.Tensor:
    """Leave-one-out consistency: score_i = -||c_i - mean_{j!=i}(c_j)||^2."""

    if candidates.dim() != 3:
        raise ValueError(f"Expected candidates (K,T,A); got {tuple(candidates.shape)}")
    k = int(candidates.shape[0])
    if k <= 1:
        return torch.zeros(k, device=candidates.device, dtype=candidates.dtype)
    flat = candidates.reshape(k, -1)
    total = flat.sum(dim=0)
    loo_mean = (total.unsqueeze(0) - flat) / float(k - 1)
    diff = flat - loo_mean
    return -(diff.pow(2).mean(dim=-1))
