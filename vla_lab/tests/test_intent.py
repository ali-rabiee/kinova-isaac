"""Offline tests for intent estimation + the intent-routed controller.

The lexical and controller tests are torch-free; estimator tests need torch
(a tiny randomly-initialized TinyVLA) and are skipped cleanly when torch is
unavailable, keeping the suite runnable anywhere.
"""

from __future__ import annotations

from vla_lab.intent.estimator import lexical_target_candidates

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False

COLORS = ["red", "blue", "yellow"]


def test_lexical_candidates_specific_and_ambiguous() -> None:
    assert lexical_target_candidates("Pick up the red box.", COLORS) == ["red"]
    assert lexical_target_candidates("Pick up the box.", COLORS) == []
    assert set(lexical_target_candidates("the red or the blue one?", COLORS)) == {"red", "blue"}


def _make_model_and_tok(cameras=("overhead",)):
    from vla_lab.dataset import TinyTokenizer
    from vla_lab.models import TinyVLA, TinyVLAConfig

    tok = TinyTokenizer.build_from_corpus(
        [f"Pick up the {c} box." for c in COLORS] + ["Pick up the box."]
    )
    cfg = TinyVLAConfig(
        image_size=64, embed_dim=32, vis_channels=(8, 16, 16, 32), lang_heads=2,
        decoder_heads=2, vocab_size=tok.vocab_size, max_lang_len=tok.max_len,
        cameras=cameras,
    )
    torch.manual_seed(0)
    model = TinyVLA(cfg)
    model.eval()
    return model, tok


def test_intent_estimator_specific_vs_ambiguous_entropy() -> None:
    if not _HAS_TORCH:
        print("SKIP (torch unavailable)")
        return
    from vla_lab.intent import IntentEstimator

    model, tok = _make_model_and_tok()
    est = IntentEstimator(model, tok, device="cpu", template="Pick up the {label} box.")
    img = torch.rand(3, 64, 64)
    st = torch.rand(4)
    e_specific = est.estimate(
        instruction="Pick up the red box.", candidate_labels=COLORS, image=img, state=st
    )
    e_ambig = est.estimate(
        instruction="Pick up the box.", candidate_labels=COLORS, image=img, state=st
    )
    assert not e_specific.instruction_ambiguous and e_ambig.instruction_ambiguous
    assert abs(sum(e_specific.posterior.values()) - 1.0) < 1e-5
    assert e_specific.top1 == "red"
    assert e_specific.entropy < e_ambig.entropy, (e_specific.entropy, e_ambig.entropy)
    for e in (e_specific, e_ambig):
        f = e.features()
        assert set(f) == {"intent_entropy", "intent_top2_margin", "instruction_ambiguous"}


def test_intent_estimator_geometric_grounding_prefers_pointed_object() -> None:
    if not _HAS_TORCH:
        print("SKIP (torch unavailable)")
        return
    from vla_lab.intent import IntentEstimator

    model, tok = _make_model_and_tok()
    # Behavioral logits are ~equal for a random net + ambiguous instruction, so a
    # strong geometric term should dominate the posterior ordering.
    est = IntentEstimator(model, tok, device="cpu", geometric_weight=5.0)
    img = torch.rand(3, 64, 64)
    st = torch.zeros(4)
    obj = {"red": (1.0, 0.0, 0.0), "blue": (-1.0, 0.0, 0.0), "yellow": (0.0, -1.0, 0.0)}

    class _Directed:
        """Wrap the model so the predicted chunk moves straight +x."""

        def __init__(self, m):
            self._m = m
            self.cfg = m.cfg

        def forward(self, *a, **k):
            out = self._m.forward(*a, **k)
            acts = torch.zeros_like(out.actions)
            acts[..., 0] = 0.01  # +x every step
            return type(out)(actions=acts, features=out.features, bottleneck=out.bottleneck)

    est.model = _Directed(model)
    e = est.estimate(
        instruction="Pick up the box.", candidate_labels=COLORS, image=img, state=st,
        object_pos_b=obj, ee_pos_b=(0.0, 0.0, 0.0),
    )
    assert e.top1 == "red", e.posterior  # +x points at red


def test_cross_view_disagreement_defined_only_for_two_cam_models() -> None:
    if not _HAS_TORCH:
        print("SKIP (torch unavailable)")
        return
    from vla_lab.intent import cross_view_disagreement

    model1, tok = _make_model_and_tok(cameras=("overhead",))
    ids, attn = tok.encode("pick up the red box.")
    img = torch.rand(3, 64, 64)
    st = torch.rand(4)
    assert cross_view_disagreement(model1, image=img, image_wrist=img, state=st, lang_ids=ids, lang_mask=attn) is None

    model2, tok2 = _make_model_and_tok(cameras=("overhead", "wrist"))
    ids2, attn2 = tok2.encode("pick up the red box.")
    v = cross_view_disagreement(
        model2, image=torch.rand(3, 64, 64), image_wrist=torch.rand(3, 64, 64),
        state=st, lang_ids=ids2, lang_mask=attn2,
    )
    assert v is not None and v >= 0.0


def test_intent_routed_controller_routes_by_source() -> None:
    """intent high -> query; perception high -> compute; both low -> act (torch-free)."""

    from vla_lab.allocation.baselines import build_controller
    from vla_lab.allocation.query import ScriptedQueryInterface
    from vla_lab.tests.mocks import MockComputeBranch

    def run(features, dispersion):
        cb = MockComputeBranch(dispersion=dispersion)
        qi = ScriptedQueryInterface(oracle_answer="red", accuracy=1.0, sim_latency_s=0.0)
        ctrl = build_controller(
            "intent_allocator", compute_branch=cb, query_interface=qi,
            tau_intent=0.5, tau_disp=0.05, tau_view=0.08, k=4,
        )
        obs = dict(features)  # intent features flow through obs -> probe.features
        return ctrl.decide(obs, instruction="Pick up the box.", options=["red", "blue"], oracle_answer="red")

    d_query = run({"intent_entropy": 0.9}, dispersion=0.01)
    assert d_query.action == "query" and d_query.queried and d_query.uncertainty_type == "intent"
    assert d_query.query_answer_correct is True

    d_compute = run({"intent_entropy": 0.1, "cross_view_disagreement": 0.5}, dispersion=0.01)
    assert d_compute.action == "compute" and d_compute.uncertainty_type == "perception"

    d_compute2 = run({"intent_entropy": 0.1}, dispersion=0.5)
    assert d_compute2.action == "compute"

    d_act = run({"intent_entropy": 0.1}, dispersion=0.01)
    assert d_act.action == "act"


if __name__ == "__main__":
    from vla_lab.tests import run_namespace

    raise SystemExit(1 if run_namespace(dict(globals()), label="test_intent") else 0)
