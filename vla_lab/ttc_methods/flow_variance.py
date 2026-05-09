"""Flow-matching / multi-seed trajectory-variance scoring (optional extension).

SmolVLA’s action expert is flow-based; when multiple noise seeds are available,
terminal variance across trajectories is a natural uncertainty signal. This module
scores candidates by **negative variance** (higher = more consensus).
"""

from __future__ import annotations

import torch


def flow_variance_score(candidates: torch.Tensor) -> torch.Tensor:
    """Per-candidate scores (K,) — higher is better (lower leave-one-out variance).

    candidates: (K, T, A)
    """

    if candidates.dim() != 3:
        raise ValueError(f"Expected (K,T,A); got {tuple(candidates.shape)}")
    k = int(candidates.shape[0])
    if k <= 1:
        return torch.zeros(k, device=candidates.device, dtype=candidates.dtype)
    mean = candidates.mean(dim=0, keepdim=True)
    var = (candidates - mean).pow(2).mean(dim=(1, 2))
    return -var
