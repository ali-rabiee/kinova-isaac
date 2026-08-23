"""The run audit: does it catch the provenance failures that have actually happened here?"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict

from vla_lab.training.audit import audit_run, find_runs, read_run, render


def _run(**over) -> Dict[str, Any]:
    man = {
        "card": {"display": "TinyVLA-2M"},
        "context_mode": "token",
        "context_style": "compact",
        "model": {"model_key": "tiny", "adapt": {"requested": "full", "applied": "full"},
                  "params_trainable": 2_000_000, "params_total": 2_000_000},
        "data": {"image_source": "isaac", "train_supervisors": ["S0", "S1"], "val_supervisors": ["S2"]},
        "preflight": {"peak_gb": 0.4},
        "prompt_audit": {"checked": True, "ok": True, "budget": 48, "max_tokens": 30,
                         "frac_truncated": 0.0},
    }
    for k, v in over.get("manifest", {}).items():
        man[k] = v
    hist = over.get("history", [{"train_loss": 1.0, "val_loss": 1.0, "acc_said": 1.0,
                                 "debias_gain_brier": 0.10, "mean_abs_debias_gap": 2.0,
                                 "debias_kappa_corr": 0.1, "ask_rank_corr": 0.0}])
    return {"dir": "d", "manifest": man, "history": hist, "summary": over.get("summary", {}),
            "finished": over.get("finished", True)}


def _flags(run) -> str:
    return " | ".join(audit_run(run)["flags"])


def test_a_clean_run_raises_no_flags():
    assert audit_run(_run())["flags"] == []


def test_schematic_images_are_always_flagged():
    r = _run()
    r["manifest"]["data"]["image_source"] = "schematic"
    assert "schematic" in _flags(r)


def test_a_silent_lora_fallback_is_flagged():
    # "LoRA" and "frozen backbone" are different experiments; the manifest is where that
    # difference survives, so the audit has to surface it.
    r = _run()
    r["manifest"]["model"]["adapt"] = {"requested": "lora", "applied": "frozen",
                                       "reason": "peft not installed"}
    assert "fell back" in _flags(r)


def test_prompt_truncation_is_flagged():
    r = _run()
    r["manifest"]["prompt_audit"] = {"checked": True, "ok": False, "budget": 48,
                                     "max_tokens": 81, "frac_truncated": 1.0}
    assert "truncated" in _flags(r)


def test_a_leaking_split_is_flagged():
    r = _run()
    r["manifest"]["data"]["val_supervisors"] = ["S1", "S2"]
    assert "leaks" in _flags(r)


def test_an_unfinished_run_is_flagged():
    assert "did not finish" in _flags(_run(finished=False, summary=None))


def test_a_model_that_never_de_biased_is_flagged():
    r = _run(history=[{"train_loss": 1.0, "val_loss": 1.0, "acc_said": 1.0,
                       "debias_gain_brier": 0.001, "mean_abs_debias_gap": 0.0,
                       "debias_kappa_corr": 0.0, "ask_rank_corr": 0.0}])
    assert "did not learn to de-bias" in _flags(r)


def test_a_model_that_cannot_read_the_instruction_is_flagged():
    # Any de-biasing number from a model that is not grounding the utterance is moot, so this
    # has to be checked before the de-biasing metric is believed.
    r = _run(history=[{"train_loss": 1.0, "val_loss": 1.0, "acc_said": 0.5,
                       "debias_gain_brier": 0.10, "mean_abs_debias_gap": 2.0,
                       "debias_kappa_corr": 0.1, "ask_rank_corr": 0.0}])
    assert "grounding accuracy" in _flags(r)


def test_overfitting_is_flagged():
    hist = [{"train_loss": 1.0, "val_loss": v, "acc_said": 1.0, "debias_gain_brier": 0.1,
             "mean_abs_debias_gap": 2.0, "debias_kappa_corr": 0.1, "ask_rank_corr": 0.0}
            for v in (1.0, 0.6, 0.5, 0.9)]
    assert "overfitting" in _flags(_run(history=hist))


def test_reading_a_directory_without_a_manifest_returns_nothing():
    with TemporaryDirectory() as td:
        assert read_run(Path(td)) is None
        assert find_runs(Path(td)) == []


def test_the_rendered_report_names_every_run_and_counts_the_flags():
    a = [audit_run(_run()), audit_run(_run(manifest={"data": {"image_source": "schematic",
                                                              "train_supervisors": ["S0"],
                                                              "val_supervisors": ["S1"]}}))]
    text = render(a)
    assert "TinyVLA-2M" in text and "1 carry at least one flag" in text


# --------------------------------------------------------------------------- study-level audit
from vla_lab.supervisory.analyze import audit_study, render_audit


def _summary(**over):
    conds = ["memoryless", "fixed_washout", "carryover_aware"]
    base_cell = {"n_probe": {"mean": 50.0}, "n_counter": {"mean": 0.0}, "n_wait": {"mean": 0.0},
                 "n_ungrounded": {"mean": 1.0}, "mae_crossover": {"mean": 0.10}}
    s = {
        "n_supervisors": 40,
        "prior": "loo",
        "conditions_run": conds,
        "conditions": {c: dict(base_cell) for c in conds},
        "contract": {"grid": {"physics": {"source": "measured", "n_measured": 400}}},
        "test_retest": {"mae_crossover": {"mean": 0.02}},
        "population": {"n_noncompliers": 5},
    }
    s["conditions"]["memoryless"]["mae_crossover"] = {"mean": 0.20}
    s["conditions"]["carryover_aware"]["mae_crossover"] = {"mean": 0.06}
    for k, v in over.items():
        s[k] = v
    return s


def _rows(identified_lambda: bool, n: int = 6):
    return [{"conditions": {"carryover_aware": {"joint_carryover": {"identifiability": {
        "lambda": {"identified": identified_lambda}, "beta_g": {"identified": True}}}}}}
        for _ in range(n)]


def test_a_clean_study_raises_no_flags():
    a = audit_study(_summary(), _rows(True))
    assert a["flags"] == [], a["flags"]


def test_prior_physics_is_always_flagged():
    s = _summary()
    s["contract"]["grid"]["physics"] = {"source": "prior", "n_measured": 0}
    assert any("PRIOR" in f for f in audit_study(s, _rows(True))["flags"])


def test_an_unmatched_budget_is_flagged():
    s = _summary()
    s["conditions"]["carryover_aware"]["n_probe"] = {"mean": 30.0}
    assert any("not matched" in f for f in audit_study(s, _rows(True))["flags"])


def test_an_unidentified_decay_rate_is_flagged():
    # The flag has to say what it means for the reading: a null on the scheduling contrast is
    # then a statement about the design, not about the mechanism.
    flags = audit_study(_summary(), _rows(False))["flags"]
    assert any("running on its prior" in f for f in flags)


def test_differences_inside_the_test_retest_floor_are_flagged():
    s = _summary()
    s["test_retest"] = {"mae_crossover": {"mean": 0.30}}
    assert any("inside" in f and "measurement noise" in f for f in audit_study(s, _rows(True))["flags"])


def test_a_low_grounding_rate_is_flagged():
    s = _summary()
    for c in s["conditions"].values():
        c["n_ungrounded"] = {"mean": 20.0}
    assert any("could not be grounded" in f for f in audit_study(s, _rows(True))["flags"])


def test_the_rendered_study_audit_names_the_provenance():
    text = render_audit(audit_study(_summary(), _rows(True)))
    assert "scene physics" in text and "test-retest floor" in text
