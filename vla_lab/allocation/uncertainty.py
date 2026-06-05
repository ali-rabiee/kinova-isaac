"""Self-generated uncertainty signals over action chunks (NumPy; torch-optional).

These are the *dispersion* signals — how much the policy disagrees with itself when
re-sampled. Per the pivot memo (§4.2) dispersion measures the spread of ``π(·|õ)``;
it is **not** the same as calibration (the gap between confidence and correctness).
Under masking, dispersion can be low while error is high ("confidently wrong"). The
allocator therefore treats dispersion as the *reducible-ambiguity* signal only, and
relies on a separately-fitted irreducibility detector for the rest.

This module deliberately avoids importing torch: tensors are accepted by duck-typing
(``.detach().cpu().numpy()``) so the allocator core stays importable anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


def _to_numpy(x: Any) -> np.ndarray:
    """Best-effort convert a torch tensor / array-like to a float64 ndarray."""

    if x is None:
        raise ValueError("expected an array-like, got None")
    if hasattr(x, "detach"):  # torch.Tensor without importing torch
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def twin_dispersion(a0: Any, a1: Any) -> float:
    """Disagreement between two ``(T, A)`` chunks: RMS over all entries.

    Mirrors :func:`vla_lab.ttc_methods.entropy_gate.twin_sample_uncertainty` (mean
    squared error then sqrt) so the allocator and the legacy gated-TTC path produce
    comparable scalars.
    """

    x0 = _to_numpy(a0)
    x1 = _to_numpy(a1)
    if x0.shape != x1.shape:
        raise ValueError(f"shape mismatch {x0.shape} vs {x1.shape}")
    return float(np.sqrt(np.mean((x0 - x1) ** 2)))


def chunk_dispersion(candidates: Any) -> float:
    """Mean pairwise dispersion of ``(K, T, A)`` candidates (RMS about the mean).

    K<=1 returns 0.0. This is the multi-sample generalization of the twin probe.
    """

    c = _to_numpy(candidates)
    if c.ndim != 3:
        raise ValueError(f"expected (K,T,A); got {c.shape}")
    k = c.shape[0]
    if k <= 1:
        return 0.0
    mean = c.mean(axis=0, keepdims=True)
    var = np.mean((c - mean) ** 2)
    return float(np.sqrt(max(0.0, var)))


def normalized_dispersion(dispersion: float, scale: float) -> float:
    """Map a raw dispersion onto ``[0, 1)`` via ``1 - exp(-d/scale)`` (scale>0)."""

    s = max(1e-9, float(scale))
    return float(1.0 - np.exp(-max(0.0, float(dispersion)) / s))


@dataclass
class ProbeResult:
    """Output of the cheap two-forward probe used to decide act vs. {compute, query}.

    Attributes
    ----------
    dispersion:
        Scalar self-disagreement between the two probe chunks (twin or noisy-pair).
    act_chunk:
        The chunk to execute if we decide to ``act`` (typically the deterministic /
        first probe). Opaque to the allocator core.
    features:
        Optional feature dict consumed by the irreducibility detector (e.g.
        ``{"dispersion": .., "max_score": .., "occlusion_fraction": ..}``).
    forward_ms:
        Wall-clock of the probe forward passes (for latency accounting).
    """

    dispersion: float
    act_chunk: Any
    features: Dict[str, float] = field(default_factory=dict)
    forward_ms: float = 0.0


@dataclass
class ComputeResult:
    """Output of the full best-of-K compute branch."""

    chunk: Any
    dispersion: float
    k_used: int
    candidates: Optional[Any] = None
    scores: Optional[Any] = None
    forward_ms: float = 0.0
    score_ms: float = 0.0


def feature_vector(features: Dict[str, float], keys) -> np.ndarray:
    """Stable ordered feature vector for a fitted detector (missing keys -> 0.0)."""

    return np.asarray([float(features.get(str(k), 0.0)) for k in keys], dtype=np.float64)
