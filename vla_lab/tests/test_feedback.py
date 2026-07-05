"""Offline tests for the real-time feedback stack (parser / sim human / fusion).

Pure Python — no torch/Isaac, same contract as the rest of the suite.
"""

from __future__ import annotations

from vla_lab.feedback import (
    FusionConfig,
    Interjection,
    ReliabilityTracker,
    ScriptedFeedbackChannel,
    SimulatedHuman,
    SimulatedHumanConfig,
    fuse,
    parse_utterance,
)

COLORS = ["red", "blue", "yellow", "purple", "orange", "cyan"]


# ------------------------------------------------------------------- parser
def test_parser_target_correction() -> None:
    r = parse_utterance("no, the red one", COLORS)
    assert r.kind == "target" and r.target_label == "red", r


def test_parser_negated_target_is_avoid() -> None:
    r = parse_utterance("not the red one", COLORS)
    assert r.kind == "avoid" and r.target_label == "red", r


def test_parser_directional_nudge() -> None:
    r = parse_utterance("a bit to the left", COLORS)
    assert r.kind == "nudge" and r.direction is not None
    assert r.direction[1] > 0.9 and abs(r.magnitude - 0.5) < 1e-9, r  # left = +y, "a bit" = 0.5


def test_parser_combined_direction() -> None:
    r = parse_utterance("up and left", COLORS)
    assert r.kind == "nudge" and r.direction is not None
    assert r.direction[1] > 0 and r.direction[2] > 0, r


def test_parser_right_is_direction_not_confirmation() -> None:
    r = parse_utterance("more to the right", COLORS)
    assert r.kind == "nudge" and r.direction[1] < 0, r


def test_parser_stop_beats_everything() -> None:
    assert parse_utterance("stop stop the red one", COLORS).kind == "stop"


def test_parser_confirm_and_chatter() -> None:
    assert parse_utterance("yes", COLORS).kind == "confirm_yes"
    assert parse_utterance("not that one", COLORS).kind == "confirm_no"
    assert parse_utterance("the weather is nice today", COLORS).kind == "unknown"


# ---------------------------------------------------------------- sim human
def _run_human(accuracy: float, seed: int = 0):
    cfg = SimulatedHumanConfig(accuracy=accuracy, latency_ticks=0, min_gap_ticks=0, seed=seed)
    h = SimulatedHuman(cfg, true_target_label="red", candidate_labels=COLORS, episode_idx=0)
    # EE walks straight toward the WRONG (blue) object -> must trigger.
    objects = {"red": (0.5, 0.3, 0.0), "blue": (0.5, -0.3, 0.0)}
    out = []
    for t in range(12):
        ee = (0.5, -0.05 - 0.02 * t, 0.0)
        out.extend(h.observe(tick_idx=t, ee_pos_b=ee, object_pos_b=objects))
    return h, out


def test_sim_human_reacts_to_wrong_object() -> None:
    _, inters = _run_human(accuracy=1.0)
    assert inters, "human should interject when robot heads for the wrong object"
    assert all(i.intended_correct for i in inters), inters
    assert any("red" in i.text for i in inters), inters


def test_sim_human_accuracy_zero_gives_wrong_reference() -> None:
    _, inters = _run_human(accuracy=0.0)
    assert inters and all(i.intended_correct is False for i in inters), inters


def test_sim_human_deterministic_given_seed() -> None:
    _, a = _run_human(accuracy=0.5, seed=7)
    _, b = _run_human(accuracy=0.5, seed=7)
    assert [i.text for i in a] == [i.text for i in b]


def test_sim_human_latency_delays_delivery() -> None:
    cfg = SimulatedHumanConfig(accuracy=1.0, latency_ticks=5, min_gap_ticks=0, seed=0)
    h = SimulatedHuman(cfg, true_target_label="red", candidate_labels=COLORS, episode_idx=0)
    objects = {"red": (0.5, 0.3, 0.0), "blue": (0.5, -0.3, 0.0)}
    delivered_at = None
    for t in range(20):
        ee = (0.5, -0.05 - 0.02 * t, 0.0)
        if h.observe(tick_idx=t, ee_pos_b=ee, object_pos_b=objects):
            delivered_at = t
            break
    assert delivered_at is not None and delivered_at >= 5 + 1, delivered_at


def test_sim_human_query_answers_share_accuracy_model() -> None:
    cfg = SimulatedHumanConfig(accuracy=1.0, seed=0)
    h = SimulatedHuman(cfg, true_target_label="red", candidate_labels=COLORS)
    ans = h.ask("Which one?", COLORS)
    assert ans.text == "red" and ans.correct is True and ans.option_idx == 0


def test_scripted_channel_poll_and_ask() -> None:
    cfg = SimulatedHumanConfig(accuracy=1.0, latency_ticks=0, min_gap_ticks=0, seed=0)
    h = SimulatedHuman(cfg, true_target_label="red", candidate_labels=COLORS)
    ch = ScriptedFeedbackChannel(h)
    objects = {"red": (0.5, 0.3, 0.0), "blue": (0.5, -0.3, 0.0)}
    for t in range(12):
        ch.observe(tick_idx=t, ee_pos_b=(0.5, -0.05 - 0.02 * t, 0.0), object_pos_b=objects)
    got = ch.poll()
    assert got and ch.poll() == []  # poll drains
    assert ch.supports_ask and ch.ask("Which?", COLORS).text == "red"


# ------------------------------------------------------------------- fusion
def test_reliability_tracker_moves_with_evidence() -> None:
    tr = ReliabilityTracker()
    m0 = tr.mean
    for _ in range(10):
        tr.update(False)
    assert tr.mean < 0.25 < m0
    for _ in range(30):
        tr.update(True)
    assert tr.mean > 0.6


def test_fusion_trusted_human_applies() -> None:
    r = parse_utterance("no, the red one", COLORS)
    d = fuse(r, reliability=0.9, intent_posterior={"red": 0.3, "blue": 0.7})
    assert d.action == "apply", d


def test_fusion_unreliable_human_ignored() -> None:
    r = parse_utterance("no, the red one", COLORS)
    d = fuse(r, reliability=0.1, intent_posterior=None)
    assert d.action == "ignore", d


def test_fusion_midband_verifies_and_trustall_ablation() -> None:
    r = parse_utterance("no, the red one", COLORS)
    d = fuse(r, reliability=0.45, intent_posterior=None)
    assert d.action == "verify", d
    d2 = fuse(r, reliability=0.45, intent_posterior=None, cfg=FusionConfig(verify_enabled=False))
    assert d2.action == "apply", d2


def test_fusion_conflict_with_confident_visual_intent_verifies() -> None:
    r = parse_utterance("no, the red one", COLORS)
    post = {"red": 0.02, "blue": 0.9, "yellow": 0.08}
    d = fuse(r, reliability=0.9, intent_posterior=post)
    assert d.action == "verify", d


def test_fusion_stop_always_applies() -> None:
    r = parse_utterance("stop", COLORS)
    assert fuse(r, reliability=0.0, intent_posterior=None).action == "apply"


def test_fusion_nudge_scaled_by_reliability() -> None:
    r = parse_utterance("a bit left", COLORS)
    d = fuse(r, reliability=0.8, intent_posterior=None)
    assert d.action == "apply" and abs(d.scale - 0.8) < 1e-9
    d_low = fuse(r, reliability=0.1, intent_posterior=None)
    assert d_low.action == "ignore"


def test_interjection_dataclass_roundtrip() -> None:
    i = Interjection(text="no, the red one", tick_idx=4, source="sim", intended_correct=True)
    assert i.text and i.tick_idx == 4 and i.intended_correct


if __name__ == "__main__":
    from vla_lab.tests import run_namespace

    raise SystemExit(1 if run_namespace(dict(globals()), label="test_feedback") else 0)
