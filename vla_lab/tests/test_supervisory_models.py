"""The Carryover-Aware VLA architecture and its training objective.

Torch-dependent; every test skips cleanly when torch is absent so the rest of the gate still
runs on a machine without it.
"""

from __future__ import annotations

try:
    import torch

    HAVE_TORCH = True
except ImportError:  # pragma: no cover
    HAVE_TORCH = False

from vla_lab.policy.context import CONTEXT_DIM, CONTEXT_MODES, CarryoverContext
from vla_lab.policy.registry import MODEL_CARDS, available_models, describe_models
from vla_lab.tests import approx, assert_raises


def _skip() -> bool:
    if not HAVE_TORCH:
        print("      (torch unavailable; skipped)")
        return True
    return False


def _batch(n=4, vocab=64, chunk=8, adim=7):
    return {
        "image": torch.randn(n, 3, 224, 224),
        "state": torch.randn(n, 4),
        "lang_ids": torch.randint(1, vocab - 1, (n, 24)),
        "lang_mask": torch.ones(n, 24, dtype=torch.long),
        "action": torch.randn(n, chunk, adim),
        "action_mask": torch.ones(n),
        "said": torch.randint(0, 2, (n,)),
        "said_mask": torch.ones(n),
        "unprompted": torch.randint(0, 2, (n,)),
        "unprompted_mask": torch.ones(n),
        "ask_label": torch.rand(n),
        "ask_mask": torch.ones(n),
        "kappa": torch.tensor([1.2, -0.9, 0.05, 0.7][:n]),
    }


# --- context ---------------------------------------------------------------
def test_the_context_feature_vector_has_the_declared_width():
    c = CarryoverContext(kappa=0.8, recent=(("A", 2, True),))
    assert len(c.features()) == CONTEXT_DIM


def test_an_empty_context_says_the_answer_is_unprompted():
    text = CarryoverContext.empty().to_text()
    assert "not demonstrated" in text and "unprompted" in text


def test_the_verbalised_context_names_the_strategy_and_the_risk():
    c = CarryoverContext(kappa=1.1, kappa_sd=0.2, lambda_hat=0.7, slots_since_coach=1,
                         recent=(("A", 1, True), ("A", 7, True)))
    t = c.to_text()
    assert "CLEAR_FIRST" in t and "high" in t and "+1.1" in t


def test_the_context_round_trips_through_a_dict():
    c = CarryoverContext(kappa=0.4, recent=(("B", 3, False),), scene_c=-0.2)
    assert CarryoverContext.from_dict(c.to_dict()).to_dict() == c.to_dict()


def test_the_sign_of_kappa_tracks_which_strategy_was_demonstrated():
    a = CarryoverContext(kappa=0.9, recent=(("A", 1, True),)).features()
    b = CarryoverContext(kappa=-0.9, recent=(("B", 1, True),)).features()
    assert a[0] > 0 > b[0]
    assert a[5] > 0 > b[5]


# --- registry --------------------------------------------------------------
def test_the_model_roster_spans_the_architectural_axes_the_paper_compares():
    cards = [MODEL_CARDS[k] for k in available_models()]
    assert {c.pretrained for c in cards} == {"none", "vla", "vlm"}
    assert any(c.language for c in cards) and any(not c.language for c in cards)
    assert {c.action_head for c in cards} >= {"regress", "flow"}


def test_a_backbone_without_a_language_model_cannot_be_given_a_verbalised_context():
    if _skip():
        return
    from vla_lab.policy import build_model

    assert "text" not in MODEL_CARDS["tiny"].context_modes
    assert_raises(ValueError, lambda: build_model("tiny", context_mode="text"))


def test_describe_models_renders_every_column_the_table_needs():
    t = describe_models()
    for col in ("params", "pretrain", "lang", "action", "adapt", "VRAM"):
        assert col in t


# --- the wrapper -----------------------------------------------------------
def test_every_context_mode_the_backbone_supports_produces_the_full_output():
    if _skip():
        return
    from vla_lab.policy import build_model

    for mode in MODEL_CARDS["tiny"].context_modes:
        m = build_model("tiny", context_mode=mode, vocab_size=64)
        b = _batch()
        out = m(b, context=torch.randn(4, CONTEXT_DIM), kappa=b["kappa"])
        assert out.said.shape == (4, 2)
        assert out.unprompted.shape == (4, 2)
        assert out.said_from_unprompted.shape == (4, 2)
        assert out.ask.shape == (4,)
        assert out.actions.shape[0] == 4


def test_an_untrained_model_is_exactly_the_memoryless_baseline():
    # The de-biased head is a zero-initialised correction on the grounded one, so a model that
    # has learned nothing believes exactly what it was told -- which is the right null.
    if _skip():
        return
    from vla_lab.policy import build_model

    m = build_model("tiny", context_mode="token", vocab_size=64)
    b = _batch()
    out = m(b, context=torch.randn(4, CONTEXT_DIM), kappa=b["kappa"])
    assert float(out.debias_gap().abs().max()) < 1e-6


def test_the_forward_contamination_model_shifts_in_the_direction_of_the_residue():
    if _skip():
        return
    from vla_lab.policy.heads import ForwardContamination

    fwd = ForwardContamination(learn_beta=True, init_beta=1.0)
    logits = torch.zeros(3, 2)
    out = fwd(logits, torch.tensor([1.0, 0.0, -1.0]))
    d = out[:, 0] - out[:, 1]
    assert float(d[0]) > 0 > float(d[2])
    assert approx(float(d[1]), 0.0, tol=1e-6)


def test_the_context_mode_changes_the_parameter_count_only_where_it_should():
    if _skip():
        return
    from vla_lab.policy import build_model

    n = {mode: build_model("tiny", context_mode=mode, vocab_size=64).report()["params_total"]
         for mode in ("none", "token", "film")}
    assert n["token"] > n["none"] and n["film"] > n["none"]


# --- the objective ---------------------------------------------------------
def test_every_loss_term_is_present_and_gradients_reach_the_backbone():
    if _skip():
        return
    from vla_lab.policy import build_model
    from vla_lab.training.losses import CarryoverLossConfig, carryover_loss

    m = build_model("tiny", context_mode="token", vocab_size=64)
    b = _batch()
    out = m(b, context=torch.randn(4, CONTEXT_DIM), kappa=b["kappa"])
    L = carryover_loss(out, b, CarryoverLossConfig())
    assert set(L.parts) == {"action", "said", "forward", "unprompted", "anti_copy", "ask"}
    L.total.backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in m.backbone.parameters())


def test_each_loss_term_can_be_ablated_to_exactly_zero():
    if _skip():
        return
    from vla_lab.policy import build_model
    from vla_lab.training.losses import CarryoverLossConfig, carryover_loss

    m = build_model("tiny", context_mode="token", vocab_size=64)
    b = _batch()
    out = m(b, context=torch.randn(4, CONTEXT_DIM), kappa=b["kappa"])
    full = float(carryover_loss(out, b, CarryoverLossConfig()).total)
    off = float(carryover_loss(out, b, CarryoverLossConfig(w_anti_copy=0.0, w_forward=0.0)).total)
    assert off < full


def test_the_compliance_penalty_only_fires_when_the_instruction_echoes_the_coaching():
    if _skip():
        return
    from vla_lab.policy import build_model
    from vla_lab.training.losses import CarryoverLossConfig, carryover_loss

    m = build_model("tiny", context_mode="token", vocab_size=64)
    b = _batch(n=2)
    # kappa > 0 pushes toward strategy A (class 0). Sample 0 echoes it; sample 1 opposes it.
    b["kappa"] = torch.tensor([1.5, 1.5])
    b["said"] = torch.tensor([0, 1])
    b["said_mask"] = torch.tensor([1.0, 0.0])
    echo = float(carryover_loss(m(b, context=torch.zeros(2, CONTEXT_DIM), kappa=b["kappa"]),
                                b, CarryoverLossConfig()).parts["anti_copy"])
    b["said_mask"] = torch.tensor([0.0, 1.0])
    oppose = float(carryover_loss(m(b, context=torch.zeros(2, CONTEXT_DIM), kappa=b["kappa"]),
                                  b, CarryoverLossConfig()).parts["anti_copy"])
    assert echo > 0.0 and approx(oppose, 0.0, tol=1e-9), (echo, oppose)


def test_the_compliance_penalty_is_silent_when_there_was_no_coaching():
    if _skip():
        return
    from vla_lab.policy import build_model
    from vla_lab.training.losses import CarryoverLossConfig, carryover_loss

    m = build_model("tiny", context_mode="token", vocab_size=64)
    b = _batch()
    b["kappa"] = torch.zeros(4)
    out = m(b, context=torch.zeros(4, CONTEXT_DIM), kappa=b["kappa"])
    assert approx(float(carryover_loss(out, b, CarryoverLossConfig()).parts["anti_copy"]), 0.0, tol=1e-9)


def test_every_backbone_declares_what_the_batch_must_carry():
    """A backbone that silently expects keys the collator never produces fails at step one.

    This is the shape of the bug that killed every Qwen cell for a whole sweep: the wrapper's
    docstring said tokenisation happened in the collator, the collator was backbone-agnostic and
    produced no ``input_ids``, and the model died deep inside an embedding lookup on ``None``
    rather than anywhere that named the missing key. The collator emits exactly the keys below;
    anything a backbone needs beyond them it has to build itself.
    """
    from vla_lab.training.data import collate

    produced = {"prompt", "image", "state", "lang_ids", "lang_mask", "context", "kappa", "rho",
                "beta", "said", "said_mask", "unprompted", "unprompted_mask"}
    sample = {k: (["x"] if k == "prompt" else 0.0) for k in produced}
    if HAVE_TORCH:
        sample = {k: (["x"][0] if k == "prompt" else torch.zeros(1)) for k in produced}
    out = collate([sample, sample])
    assert produced.issubset(set(out)), f"collator dropped {produced - set(out)}"
    assert "input_ids" not in out, (
        "the collator does not tokenise for any particular backbone; a backbone needing "
        "input_ids must build them in its own forward"
    )


def test_qwen_backbone_builds_its_own_inputs():
    """The Qwen wrapper must not assume the collator handed it token ids."""
    import inspect

    from vla_lab.policy.backbones import qwen

    src = inspect.getsource(qwen.QwenVLBackbone.forward)
    assert "_encode" in src, "Qwen forward must fall back to building its own inputs"
    enc = inspect.getsource(qwen.QwenVLBackbone._encode)
    for needed in ("apply_chat_template", "prompt", "image"):
        assert needed in enc, f"Qwen._encode should use {needed!r}"


def test_backbone_options_that_do_not_apply_are_dropped_and_named():
    """A checkpoint's manifest carries options that are not the same set for every backbone.

    Reloading a pretrained checkpoint through the generic path used to hand it the from-scratch
    model's ``vocab_size`` and die on a TypeError. Dropping the surplus is right; dropping it
    silently is not, because then a misspelled option has no effect and no symptom.
    """
    from vla_lab.policy.backbones import _accepted

    class Narrow:
        def __init__(self, alpha: int = 1, beta: int = 2) -> None:
            pass

    class Permissive:
        def __init__(self, alpha: int = 1, **kw) -> None:
            pass

    keep, dropped = _accepted(Narrow, {"alpha": 5, "gamma": 9})
    assert keep == {"alpha": 5}, keep
    assert dropped == ("gamma",), dropped

    # A backbone that takes **kw genuinely accepts anything, so nothing is reported as ignored.
    keep, dropped = _accepted(Permissive, {"alpha": 5, "gamma": 9})
    assert keep == {"alpha": 5, "gamma": 9}, keep
    assert dropped == (), dropped


def test_smolvla_does_not_receive_the_from_scratch_vocabulary_size():
    """The exact reload failure, checked without downloading half a gigabyte of weights."""
    import inspect

    from vla_lab.policy.backbones.smolvla import SmolVLABackbone

    params = inspect.signature(SmolVLABackbone.__init__).parameters
    assert "vocab_size" not in params, "test premise changed"
    assert not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()), (
        "if SmolVLABackbone starts taking **kw this guard stops working"
    )


# ---------------------------------------------------------------------------
# 2026-08-23: seed dispersion and the seed floor (P0-1)
# ---------------------------------------------------------------------------
def test_seed_floor_is_the_mean_pairwise_difference_within_cells():
    from vla_lab.training.seeds import aggregate_seeds, seed_floor

    rows = []
    for seed, g in ((1, 0.10), (2, 0.12), (3, 0.11)):
        rows.append({"model": "tiny", "context": "film", "seed": seed, "debias_gain_brier": g,
                     "debias_kappa_corr": 0.1 + 0.01 * seed, "ask_rank_corr": 0.0})
    for seed, g in ((1, 0.09), (2, 0.09), (3, 0.09)):
        rows.append({"model": "tiny", "context": "none", "seed": seed, "debias_gain_brier": g,
                     "debias_kappa_corr": 0.0, "ask_rank_corr": 0.0})
    fl = seed_floor(rows, "debias_gain_brier", group_by=("model", "context"))
    # film pairs: |0.10-0.12|, |0.10-0.11|, |0.12-0.11| = 0.02, 0.01, 0.01; none pairs: 0, 0, 0
    assert abs(fl["floor"] - (0.04 / 6)) < 1e-12 and fl["n_pairs"] == 6
    agg = aggregate_seeds(rows)
    film = next(c for c in agg["cells"] if c["context"] == "film")
    assert film["n_seeds"] == 3 and abs(film["debias_gain_brier"]["mean"] - 0.11) < 1e-12
    con = next(c for c in agg["contrasts"] if c["metric"] == "debias_gain_brier")
    assert con["n_seeds"] == 3 and con["same_sign_every_seed"]
    assert abs(abs(con["delta_b_minus_a"]) - 0.02) < 1e-12
    assert con["clears_floor"], "a 0.02 difference clears a 0.0067 floor"


def test_an_ordering_inside_the_seed_floor_is_not_allowed_to_clear_it():
    from vla_lab.training.seeds import aggregate_seeds

    rows = []
    for seed, ga, gb in ((1, 0.10, 0.11), (2, 0.13, 0.10), (3, 0.11, 0.12)):
        rows.append({"model": "m", "context": "a", "seed": seed, "debias_gain_brier": ga, "debias_kappa_corr": 0.0})
        rows.append({"model": "m", "context": "b", "seed": seed, "debias_gain_brier": gb, "debias_kappa_corr": 0.0})
    agg = aggregate_seeds(rows)
    con = next(c for c in agg["contrasts"] if c["metric"] == "debias_gain_brier")
    assert not con["clears_floor"] and not con["same_sign_every_seed"]


def test_single_seed_rows_report_no_dispersion_rather_than_a_fake_one():
    from vla_lab.training.seeds import aggregate_seeds

    rows = [{"model": "m", "context": "a", "seed": 1, "debias_gain_brier": 0.1, "debias_kappa_corr": 0.2}]
    agg = aggregate_seeds(rows)
    cell = agg["cells"][0]
    assert cell["n_seeds"] == 1 and cell["debias_gain_brier"]["sd"] is None
    assert agg["seed_floor"]["debias_gain_brier"]["floor"] is None
