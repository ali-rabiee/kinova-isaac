"""W1 — tests for ``vla_lab.rehab.workspace`` and ``vla_lab.rehab.contract``.

"Done when" (``rehab.md`` §6/W1): frames round-trip; every emitted target is robot-reachable
and human-reachable; crossover densification is monotone in the parameter; target IDs are
stable across runs given the same contract hash.
"""

from __future__ import annotations

import math

from vla_lab.rehab.contract import Phase0Contract, TimingConfig
from vla_lab.rehab.workspace import (
    SIDE_LEFT,
    SIDE_RIGHT,
    PlanarTransform,
    TargetGrid,
    WorkspaceConfig,
    effort_asymmetry,
    nonpreferred_lateral,
    reach_distances,
)


# --------------------------------------------------------------------------- frames


def test_planar_transform_round_trips():
    t = PlanarTransform(tx=0.31, ty=-0.22, tz=0.05, yaw_rad=0.73)
    for p in [(0.0, 0.0, 0.0), (0.4, 0.1, 0.0), (-0.2, 0.35, 0.02)]:
        back = t.inverse().apply(t.apply(p))
        assert max(abs(a - b) for a, b in zip(p, back)) < 1e-9


def test_planar_transform_compose_matches_sequential_apply():
    a = PlanarTransform(0.2, 0.1, 0.0, 0.4)
    b = PlanarTransform(-0.3, 0.25, 0.0, -0.9)
    p = (0.33, -0.12, 0.0)
    direct = a.apply(b.apply(p))
    composed = a.compose(b).apply(p)
    assert max(abs(x - y) for x, y in zip(direct, composed)) < 1e-9


def test_participant_to_robot_puts_targets_in_front_of_the_arm():
    c = Phase0Contract()
    p2r = c.participant_to_robot()
    grid = c.target_grid()
    for t in grid:
        x, y, _ = p2r.apply((t.x_m, t.y_m, 0.0))
        r = math.hypot(x, y)
        assert c.workspace.robot_reach_min_m <= r <= c.workspace.robot_reach_max_m, (t.target_id, r)


# --------------------------------------------------------------------------- handedness


def test_nonpreferred_lateral_flips_with_handedness():
    assert nonpreferred_lateral(0.2, SIDE_LEFT) == 0.2
    assert nonpreferred_lateral(0.2, SIDE_RIGHT) == -0.2


def test_effort_asymmetry_is_near_zero_at_the_midline_and_signed_outward():
    grid = TargetGrid()
    mid = min(grid, key=lambda t: abs(t.y_m))
    assert abs(effort_asymmetry(mid, grid.cfg, SIDE_LEFT)) < 0.05
    far_left = max(grid, key=lambda t: t.y_m)
    # The nonpreferred (left) arm has the cheaper reach to a far-left target.
    assert effort_asymmetry(far_left, grid.cfg, SIDE_LEFT) > 0.1


# --------------------------------------------------------------------------- the grid


def test_default_grid_is_reachable_by_both_arms_and_the_robot():
    c = Phase0Contract()
    assert c.validate() == [], c.validate()


def test_every_target_is_human_reachable_by_both_arms():
    grid = TargetGrid()
    for t in grid:
        d = reach_distances(t, grid.cfg)
        assert grid.human_reachable(t), (t.target_id, d)
        assert d[SIDE_LEFT] <= grid.cfg.human_max_reach_m
        assert d[SIDE_RIGHT] <= grid.cfg.human_max_reach_m


def test_crossover_densification_is_monotone():
    counts = []
    for d in (1.0, 1.5, 2.0, 3.0, 4.0):
        g = TargetGrid(WorkspaceConfig(densification=d))
        counts.append(len(g.crossover_targets()))
    assert counts == sorted(counts), counts
    assert counts[-1] > counts[0]


def test_target_ids_are_stable_across_runs():
    a = TargetGrid(WorkspaceConfig())
    b = TargetGrid(WorkspaceConfig())
    assert [t.to_dict() for t in a] == [t.to_dict() for t in b]
    assert a.ids() == list(range(len(a)))


def test_validate_flags_an_over_densified_grid():
    # Squeezing more targets into the band eventually violates the minimum spacing; the
    # validator must say so instead of silently emitting indistinguishable targets.
    g = TargetGrid(WorkspaceConfig(densification=6.0, min_spacing_m=0.05))
    problems = g.validate()
    assert any("spacing" in p for p in problems), problems


def test_crossover_weights_favour_the_band():
    g = TargetGrid()
    w = g.crossover_weights()
    band = {t.target_id for t in g.crossover_targets()}
    assert all(w[i] == 1.0 for i in band)
    assert all(w[i] < 1.0 for i in g.ids() if i not in band)


# --------------------------------------------------------------------------- the contract


def test_contract_hash_is_stable_and_sensitive():
    a, b = Phase0Contract(), Phase0Contract()
    assert a.contract_hash() == b.contract_hash()
    c = Phase0Contract(workspace=WorkspaceConfig(n_lateral=7))
    assert c.contract_hash() != a.contract_hash()


def test_contract_round_trips_through_dict_without_changing_its_hash():
    a = Phase0Contract()
    b = Phase0Contract.from_dict(a.to_dict())
    assert b.contract_hash() == a.contract_hash()


def test_provenance_is_not_hashed():
    """A different backend or git commit must not un-pool an otherwise identical session."""

    a = Phase0Contract()
    b = a.stamped(apparatus_backend="kinova_gen2", git_commit="deadbeef", driver_version="1.2.3")
    assert b.contract_hash() == a.contract_hash()
    assert b.provenance.apparatus_backend == "kinova_gen2"


def test_contract_rejects_an_impossible_budget():
    from vla_lab.rehab.contract import BudgetConfig

    c = Phase0Contract(budget=BudgetConfig(trials_per_block=8, coach_per_block=8))
    assert any("coach_per_block" in p for p in c.validate())


def test_timing_nominal_trial_is_the_sum_of_its_phases():
    t = TimingConfig()
    assert t.nominal_trial_ms == t.settle_dwell_ms + t.go_window_ms + t.return_ms + t.inter_trial_ms


if __name__ == "__main__":
    import sys

    from vla_lab.tests import run_namespace

    sys.exit(1 if run_namespace(dict(globals()), label="test_rehab_workspace") else 0)
