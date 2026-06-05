"""Tests for ``vla_lab.fit_allocator`` — fitting the allocator from calibration logs."""

from __future__ import annotations

import tempfile
from pathlib import Path

from vla_lab.allocation.allocator import ActComputeQueryAllocator, AllocatorFit
from vla_lab.allocation.conformal import MondrianConformal, SplitConformal
from vla_lab.allocation.query import ScriptedQueryInterface
from vla_lab.calibration.records import CalibrationRecord
from vla_lab.fit_allocator import fit_from_records
from vla_lab.tests.mocks import MockComputeBranch, synthetic_records


def test_fit_learns_irreducibility_detector():
    recs = synthetic_records(600, seed=1, confidently_wrong=True)
    fit, report = fit_from_records(recs, features=["occlusion_fraction"], conformal_kind="mondrian", estimate_tau=False)
    assert report["label_source"] == "identifiable"      # auto-picks the oracle label
    assert report["model_fitted"] is True
    assert report["train_auroc_irreducible"] > 0.95       # occlusion separates identifiability
    assert fit.conformal_kind == "mondrian"
    assert isinstance(fit.conformal, MondrianConformal)
    assert isinstance(fit.knowno_conformal, SplitConformal)


def test_fit_save_load_roundtrip():
    recs = synthetic_records(400, seed=2)
    fit, _ = fit_from_records(recs, features=["occlusion_fraction"], conformal_kind="split")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "allocator_fit.json"
        fit.save(p)
        loaded = AllocatorFit.load(p)
    assert loaded.conformal_kind == "split"
    assert loaded.irreducibility.fitted == fit.irreducibility.fitted
    assert loaded.knowno_conformal is not None


def test_fitted_allocator_queries_under_occlusion_not_when_clean():
    recs = synthetic_records(600, seed=3, confidently_wrong=True)
    fit, _ = fit_from_records(recs, features=["occlusion_fraction"], conformal_kind="mondrian", estimate_tau=False)
    alloc = ActComputeQueryAllocator(MockComputeBranch(), ScriptedQueryInterface(oracle_answer="red"), fit=fit)
    d_clean = alloc.decide({"occlusion_fraction": 0.0}, instruction="pick", options=["red", "blue"], oracle_answer="red")
    d_heavy = alloc.decide({"occlusion_fraction": 0.5}, instruction="pick", options=["red", "blue"], oracle_answer="red")
    assert d_clean.action != "query"
    assert d_heavy.action == "query"
    assert d_heavy.query_answer_correct is True


def test_label_source_falls_back_to_confidently_wrong():
    # Only correctness labels (no `identifiable` oracle) -> confidently-wrong proxy.
    recs = [
        CalibrationRecord(run_id="r", episode_idx=i, occlusion_fraction=0.1 * (i % 5),
                          dispersion=0.01, correct=bool(i % 3), identifiable=None)
        for i in range(60)
    ]
    _, report = fit_from_records(recs, features=["dispersion", "occlusion_fraction"], conformal_kind="none")
    assert report["label_source"] == "confidently_wrong"


def test_label_source_falls_back_to_occlusion():
    # Neither identifiable nor correctness -> occlusion proxy.
    recs = [
        CalibrationRecord(run_id="r", episode_idx=i, occlusion_fraction=0.1 * (i % 6), dispersion=0.01)
        for i in range(60)
    ]
    _, report = fit_from_records(recs, features=["occlusion_fraction"], conformal_kind="none")
    assert report["label_source"] == "occlusion"


if __name__ == "__main__":
    import sys

    from vla_lab.tests import run_namespace

    sys.exit(1 if run_namespace(dict(globals()), label="test_fit_allocator") else 0)
