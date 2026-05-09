"""Baseline methods for test-time scaling comparisons (e.g. MG-Select-style selection)."""

from __future__ import annotations

from .mg_select import mg_select_kl_scores, mg_select_loo_l2_scores

__all__ = ["mg_select_kl_scores", "mg_select_loo_l2_scores"]
