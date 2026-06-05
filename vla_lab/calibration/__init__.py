"""Robot-only calibration experiments — the empirical heart of the pivot (memo §4.2).

This package produces **Result 2**: the demonstration that self-agreement decouples from
correctness as the camera is occluded, which no humans are needed to produce and which
de-risks the schedule. It supplies

- ``records``  — the per-step :class:`CalibrationRecord` schema + JSONL IO, including a
  ``from_decision`` adapter so the act/compute/query allocator's traces feed straight in.
- ``metrics``  — ECE / reliability diagrams, the agreement→correctness correlation (and its
  inversion under masking), AUROC of agreement as an error detector, and conformal coverage
  vs. occlusion (the §4.3 degradation curve).
- ``analyze``  — a CLI that turns a calibration log into the Result-2 figures + a summary.

All metrics are pure NumPy; figures use a lazily-imported matplotlib.
"""

from __future__ import annotations

from .metrics import (
    auroc,
    confidence_from_dispersion,
    coverage_vs_occlusion,
    decoupling_table,
    ece_vs_bucket,
    expected_calibration_error,
    point_biserial,
    reliability_bins,
)
from .records import CalibrationRecord, read_jsonl, write_jsonl

__all__ = [
    "CalibrationRecord",
    "read_jsonl",
    "write_jsonl",
    "confidence_from_dispersion",
    "reliability_bins",
    "expected_calibration_error",
    "ece_vs_bucket",
    "point_biserial",
    "auroc",
    "decoupling_table",
    "coverage_vs_occlusion",
]
