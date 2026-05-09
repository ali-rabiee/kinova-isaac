"""Pluggable test-time scaling helpers (gating, optional flow / VLM scoring)."""

from __future__ import annotations

from .entropy_gate import twin_sample_uncertainty, uncertainty_gate_effective_k
from .flow_variance import flow_variance_score
from .vlm_verifier import VLMVerifierStub, score_trajectory_stub

__all__ = [
    "twin_sample_uncertainty",
    "uncertainty_gate_effective_k",
    "flow_variance_score",
    "VLMVerifierStub",
    "score_trajectory_stub",
]
