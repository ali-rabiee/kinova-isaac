"""Tests for ``vla_lab.calibration`` — Result-2 metrics (ECE, decoupling, coverage)."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np

from vla_lab.calibration.metrics import (
    auroc,
    confidence_from_dispersion,
    coverage_vs_occlusion,
    decoupling_table,
    ece_vs_bucket,
    expected_calibration_error,
    point_biserial,
    reliability_bins,
    summary,
)
from vla_lab.calibration.records import CalibrationRecord, read_jsonl, write_jsonl
from vla_lab.tests.mocks import synthetic_records


def _bucket_mid(label: str) -> float:
    inner = label.strip()[1:-1]
    lo, hi = inner.split(",")
    return (float(lo) + (0.6 if hi.startswith("inf") else float(hi))) / 2.0


# --------------------------------------------------------------------------- records IO


def test_records_jsonl_roundtrip():
    recs = [
        CalibrationRecord(run_id="r", episode_idx=i, occlusion_fraction=0.1 * i, dispersion=0.01 * i,
                          correct=bool(i % 2), extra={"custom": i})
        for i in range(5)
    ]
    with tempfile.TemporaryDirectory() as td:
        p = write_jsonl(recs, Path(td) / "r.jsonl")
        back = read_jsonl(p)
    assert len(back) == 5
    assert back[3].correct == bool(3 % 2)
    assert back[3].extra["custom"] == 3  # unknown/extra fields preserved


def test_record_from_decision():
    from vla_lab.allocation.allocator import Decision

    dec = Decision(action="query", dispersion=0.4, irreducible_score=0.8, k_used=1, queried=True,
                   query_answer_correct=True, conformal_anomaly=True, controller="allocator")
    rec = CalibrationRecord.from_decision(dec, run_id="r", episode_idx=2, step_idx=3,
                                          occlusion_mode="bottom_strip", occlusion_fraction=0.5, correct=False)
    assert rec.action == "query" and rec.queried is True
    assert rec.controller == "allocator" and rec.correct is False
    assert rec.dispersion == 0.4 and rec.irreducible_score == 0.8


# --------------------------------------------------------------------------- calibration / ECE


def test_confidence_is_monotone_decreasing_in_dispersion():
    c = confidence_from_dispersion([0.0, 0.1, 1.0], scale=0.1)
    assert c[0] > c[1] > c[2]
    assert abs(c[0] - 1.0) < 1e-9  # zero dispersion -> full confidence


def test_ece_zero_when_calibrated_positive_when_not():
    # Perfectly calibrated: in each occupied bin, accuracy == mean confidence.
    conf = np.array([0.95] * 100 + [0.05] * 100)
    corr = np.array([1.0] * 95 + [0.0] * 5 + [1.0] * 5 + [0.0] * 95)  # 95% and 5% accuracy
    assert expected_calibration_error(conf, corr, n_bins=10) < 1e-6
    # Over-confident: high confidence but zero accuracy -> ECE near the confidence level.
    bad_conf = np.array([0.95] * 10)
    bad_corr = np.array([0.0] * 10)
    assert expected_calibration_error(bad_conf, bad_corr, n_bins=10) > 0.9


def test_reliability_bins_partition_all_samples():
    rng = np.random.default_rng(0)
    conf = rng.uniform(size=200)
    corr = (rng.uniform(size=200) < conf).astype(float)
    b = reliability_bins(conf, corr, n_bins=10)
    assert sum(b["bin_count"]) == 200


def test_ece_degrades_with_occlusion():
    """Prediction 1: ECE grows as the occlusion bucket gets harder (confidently-wrong regime)."""
    recs = synthetic_records(800, seed=1, confidently_wrong=True)
    table = ece_vs_bucket(recs, scale=0.05)
    items = sorted(table.items(), key=lambda kv: _bucket_mid(kv[0]))
    eces = [v["ece"] for _, v in items if v["n"] > 0]
    assert eces[-1] > eces[0] + 0.1, eces  # worst (most occluded) bucket much worse than clean


# --------------------------------------------------------------------------- AUROC / correlation


def test_auroc_known_orderings():
    assert auroc([0.1, 0.2, 0.9, 0.95], [0, 0, 1, 1]) == 1.0
    assert auroc([0.9, 0.95, 0.1, 0.2], [0, 0, 1, 1]) == 0.0
    assert math.isnan(auroc([0.1, 0.2], [1, 1]))  # one class -> undefined


def test_point_biserial_sign():
    x = [0.0, 1.0, 2.0, 3.0]
    assert point_biserial(x, [0, 0, 1, 1]) > 0.5
    assert point_biserial(x, [1, 1, 0, 0]) < -0.5


# --------------------------------------------------------------------------- decoupling (Prediction 2/3)


def test_agreement_decouples_under_occlusion():
    """Confidently-wrong regime: self-agreement carries ~no correctness signal (AUROC≈0.5)."""
    recs = synthetic_records(1500, seed=2, confidently_wrong=True)
    s = summary(recs, scale=0.05)
    assert 0.42 <= s["overall_auroc_agreement_correct"] <= 0.58, s["overall_auroc_agreement_correct"]
    # accuracy still falls with occlusion even though agreement does not flag it.
    dt = decoupling_table(recs, scale=0.05)
    items = sorted(dt.items(), key=lambda kv: _bucket_mid(kv[0]))
    accs = [v["accuracy"] for _, v in items if v["n"] > 0]
    assert accs[0] > accs[-1] + 0.3, accs


def test_agreement_couples_when_dispersion_tracks_error():
    """Contrast: when dispersion grows with occlusion, agreement *does* predict correctness."""
    recs = synthetic_records(1500, seed=3, confidently_wrong=False)
    s = summary(recs, scale=0.05)
    assert s["overall_auroc_agreement_correct"] > 0.6, s["overall_auroc_agreement_correct"]


# --------------------------------------------------------------------------- conformal coverage (§4.3)


def test_selective_accuracy_degrades_under_occlusion():
    recs = synthetic_records(1500, seed=4, confidently_wrong=True)
    cov = coverage_vs_occlusion(recs, alpha=0.1)
    sel = cov["selective_accuracy"]
    items = sorted(sel.items(), key=lambda kv: _bucket_mid(kv[0]))
    vals = [v for _, v in items if not math.isnan(v)]
    # The "confident" decisions are reliable on clean data but not under heavy occlusion.
    assert vals[0] > vals[-1] + 0.2, sel
    assert cov["calib_bucket"] is not None


def test_summary_has_expected_keys():
    recs = synthetic_records(200, seed=5)
    s = summary(recs)
    for key in ["overall_ece", "overall_accuracy", "overall_auroc_agreement_correct",
                "ece_vs_occlusion", "decoupling", "coverage_vs_occlusion", "n_labeled"]:
        assert key in s, key


if __name__ == "__main__":
    import sys

    from vla_lab.tests import run_namespace

    sys.exit(1 if run_namespace(dict(globals()), label="test_calibration") else 0)
