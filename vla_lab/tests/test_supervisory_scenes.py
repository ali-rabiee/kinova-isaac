"""Scenes, strategies, narration, grounding: the vocabulary the rest of the study is written in."""

from __future__ import annotations

import math

from vla_lab.supervisory import STRATEGY_A, STRATEGY_B, STRATEGY_UNRESOLVED
from vla_lab.supervisory.narration import (
    DEFAULT_DOSES,
    coach_narration,
    content_hash,
    counter_query,
    dose_by_name,
    ground,
    grounding_agreement,
    probe_query,
)
from vla_lab.supervisory.scenes import ScenePhysics, build_scene_grid
from vla_lab.supervisory.strategies import AXES, GRASP_AXIS, PLAN_AXIS, get_axis, other
from vla_lab.tests import approx, assert_raises


def test_every_scripted_phrase_grounds_to_the_strategy_that_produced_it():
    # A phrase the simulator can emit but the grounder cannot resolve is a silent
    # observation-loss bug: the answer is given, and the estimand never sees it.
    for axis in AXES.values():
        for phrase in axis.phrases_a:
            assert ground(phrase, axis) == STRATEGY_A, (axis.name, phrase)
        for phrase in axis.phrases_b:
            assert ground(phrase, axis) == STRATEGY_B, (axis.name, phrase)


def test_the_grounder_refuses_to_guess():
    assert ground("hmm, either way I guess", PLAN_AXIS) == STRATEGY_UNRESOLVED
    assert ground("", PLAN_AXIS) == STRATEGY_UNRESOLVED
    # Hitting both sides is ambiguous, not a coin flip.
    assert ground("clear it first, or just go direct", PLAN_AXIS) == STRATEGY_UNRESOLVED


def test_the_counter_proposal_names_the_option_that_was_not_demonstrated():
    after_a = counter_query(PLAN_AXIS, STRATEGY_A)
    after_b = counter_query(PLAN_AXIS, STRATEGY_B)
    assert ground(after_a, PLAN_AXIS) == STRATEGY_B
    assert ground(after_b, PLAN_AXIS) == STRATEGY_A


def test_the_neutral_probe_names_no_option():
    assert ground(probe_query(), PLAN_AXIS) == STRATEGY_UNRESOLVED
    assert ground(probe_query(), GRASP_AXIS) == STRATEGY_UNRESOLVED


def test_narration_hash_is_stable_and_axis_specific():
    assert content_hash("plan") == content_hash("plan")
    assert content_hash("plan") != content_hash("grasp")


def test_the_dose_ladder_is_monotone():
    doses = [dose_by_name(n) for n in ("weak", "moderate", "strong")]
    assert [d.carryover_scale for d in doses] == sorted(d.carryover_scale for d in doses)
    assert [d.delta_logit for d in doses] == sorted(d.delta_logit for d in doses)


def test_the_coordinate_is_zero_exactly_at_the_value_crossover():
    p = ScenePhysics()
    m_star = p.crossover_margin()
    assert abs(p.coordinate(m_star)) < 1e-6
    assert approx(p.value("A", m_star), p.value("B", m_star), tol=1e-4)


def test_the_coordinate_signs_agree_with_which_strategy_is_actually_better():
    p = ScenePhysics()
    for m in (0.0, 0.02, 0.05, 0.08, 0.12, 0.16):
        c = p.coordinate(m)
        best = p.optimal_strategy(m)
        assert (c > 0) == (best == STRATEGY_A), (m, c, best)


def test_the_coordinate_is_monotone_decreasing_in_the_gap():
    p = ScenePhysics()
    cs = [p.coordinate(m) for m in [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.14]]
    assert all(a > b for a, b in zip(cs, cs[1:])), cs


def test_the_grid_is_densest_in_the_crossover_band():
    g = build_scene_grid()
    band = [s for s in g.probe_scenes() if g.in_crossover_band(s)]
    assert len(band) >= 6
    w = g.band_weights(crossover_weighted=True)
    in_band = sum(v for k, v in w.items() if g.in_crossover_band(g.by_id(k)))
    assert in_band > 0.5, in_band


def test_demonstration_scenes_sit_outside_the_band_on_both_sides():
    # A demonstration on an ambiguous scene is an implicit answer to the question being asked.
    g = build_scene_grid()
    coach = g.coach_scenes()
    assert coach
    assert all(not g.in_crossover_band(s) for s in coach)
    assert any(s.c > 0 for s in coach) and any(s.c < 0 for s in coach)


def test_every_probe_scene_is_reachable_from_the_flanks_to_the_band():
    g = build_scene_grid()
    cs = sorted(s.c for s in g.probe_scenes())
    assert cs[0] < -1.0 and cs[-1] > 1.0, cs
    assert min(abs(c) for c in cs) < 0.2


def test_physics_fit_recovers_a_known_success_curve():
    import random

    truth = ScenePhysics(b_slope=50.0, b_mid=0.045, p_b_asym=0.95)
    rng = random.Random(0)
    rows = []
    for m in [i / 100.0 for i in range(0, 17, 2)]:
        for s in (STRATEGY_A, STRATEGY_B):
            for _ in range(40):
                rows.append({"strategy": s, "margin_m": m,
                             "success": rng.random() < truth.p_success(s, m),
                             "duration_s": truth.duration_s(s, m)})
    fit = ScenePhysics.fit(rows)
    assert fit.source == "measured"
    assert fit.n_measured == len(rows)
    assert abs(fit.crossover_margin() - truth.crossover_margin()) < 0.03, (
        fit.crossover_margin(), truth.crossover_margin()
    )


def test_degenerate_physics_is_reported_not_hidden():
    # If one strategy always wins there is no ambiguous scene and the study has nothing to
    # measure. That must be detectable rather than silently producing a lopsided grid.
    p = ScenePhysics(p_b_asym=0.02, b_slope=1.0)
    assert p.is_degenerate()


def test_grounding_agreement_is_perfect_when_channels_agree():
    pairs = [(STRATEGY_A, STRATEGY_A), (STRATEGY_B, STRATEGY_B), (STRATEGY_A, STRATEGY_A), (STRATEGY_B, STRATEGY_B)]
    r = grounding_agreement(pairs)
    assert approx(r["agreement"], 1.0)
    assert approx(r["kappa"], 1.0)


def test_other_flips_the_axis_member():
    assert other(STRATEGY_A) == STRATEGY_B
    assert other(STRATEGY_B) == STRATEGY_A


def test_unknown_axis_is_an_error_not_a_default():
    assert_raises(KeyError, lambda: get_axis("no_such_axis"))


def test_every_manipulated_position_is_inside_the_measured_envelope():
    """The scene may only put objects where the arm has been *measured* to reach.

    This is the gate on the mistake that cost this project a whole sweep: the objects sat at
    y = +0.02, and with the tool held down the arm cannot descend to grasp height anywhere near
    y = 0 -- so every rollout timed out on its descent and it read as a controller problem for
    days. The envelope in ``environments.supervisory_fetch.config`` comes from an empty-table
    reachability probe, and any change to the layout has to answer to it.
    """
    from environments.supervisory_fetch import layout_for_margin
    from environments.supervisory_fetch.config import reachable_at_grasp_height

    from vla_lab.supervisory.scenes import build_scene_grid

    grid = build_scene_grid()
    margins = sorted({float(s.margin_m) for s in grid.scenes})
    assert margins, "the scene grid is empty"
    for m in margins:
        L = layout_for_margin(m)
        for name in ("target", "blocker", "clear_dropoff"):
            x, y = L[name]["xy"]
            assert reachable_at_grasp_height(x, y), (
                f"{name} at ({x:.3f}, {y:.3f}) for a {m * 100:.1f} cm gap is outside the measured "
                f"reachable envelope; the arm cannot grasp there"
            )


def test_physics_figure_reports_a_degenerate_fit_instead_of_drawing_a_crossover():
    """A value model with no crossover must be labelled, not quietly plotted.

    This is the check that would have caught the scene whose two strategies had the same
    feasibility boundary: the parameters looked ordinary, and only the value curves lying on top
    of one another showed that the study had no ambiguous region to measure.
    """
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("      (matplotlib unavailable; skipped)")
        return

    import tempfile
    from pathlib import Path

    from vla_lab.supervisory import STRATEGY_A, STRATEGY_B
    from vla_lab.supervisory.physics_figure import figure
    from vla_lab.supervisory.scenes import ScenePhysics

    # Both strategies identical -> the value gap is a constant, so no crossover exists.
    phys = ScenePhysics()
    rollouts = [{"margin_m": m, "strategy": st, "success": True, "duration_s": 30.0}
                for m in (0.0, 0.05, 0.10, 0.15) for st in (STRATEGY_A, STRATEGY_B)]
    with tempfile.TemporaryDirectory() as td:
        info = figure(rollouts, phys, Path(td) / "fig.pdf")
        assert (Path(td) / "fig.pdf").exists(), "no figure written"
        assert set(info) >= {"crossover_margin_m", "transition_width_m", "degenerate", "n_rollouts"}
        assert info["n_rollouts"] == len(rollouts)
        assert isinstance(info["degenerate"], bool)


def test_measured_physics_propagates_and_falls_back_cleanly():
    """One measurement must reach every consumer, and a missing one must not break anything.

    The earlier arrangement passed the fitted physics around as a command-line flag, which is a
    trap: the sweep writes a measurement, one command is given the flag and the others are not,
    and the study runs half under measured physics and half under the prior with nothing in the
    output to say so.
    """
    import json
    import tempfile
    from pathlib import Path

    from vla_lab.supervisory.scenes import ScenePhysics, build_scene_grid, default_physics, save_physics

    with tempfile.TemporaryDirectory() as td:
        missing = Path(td) / "nope.json"
        assert default_physics(missing).source == "prior", "a missing file must fall back to the prior"

        corrupt = Path(td) / "corrupt.json"
        corrupt.write_text("{not json at all")
        assert default_physics(corrupt).source == "prior", "a corrupt file must fall back, not raise"

        good = Path(td) / "physics.json"
        phys = ScenePhysics()
        save_physics(phys, good)
        loaded = default_physics(good)
        assert approx(loaded.crossover_margin(), phys.crossover_margin(), tol=1e-9)

    # An explicitly supplied physics still wins over whatever is on disk.
    grid = build_scene_grid(physics=ScenePhysics())
    assert len(grid.scenes) > 0
    assert all(0.0 <= s.margin_m <= 0.20 for s in grid.scenes)
