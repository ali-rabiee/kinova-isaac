"""W8 — tests for ``vla_lab.rehab.observation`` (+ W13 calibration).

``rehab.md`` §4 lists eight ``test_rehab_*`` modules and this is not one of them, but W8's
"done when" has an offline half — *"agreement machinery validated on fixtures"* — and W13's is
*"re-running calibration on a fixed rig reproduces target positions within a stated
tolerance"*. Both are testable here; the online halves (pilot video, real cameras) are not.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np

from vla_lab.rehab import ARM_AMBIGUOUS, ARM_NONE, ARM_NONPREFERRED, ARM_PREFERRED
from vla_lab.rehab.observation import (
    KAPPA_ACCEPTANCE,
    CompositeObserver,
    KeyedObserver,
    ScriptedHandDetector,
    ScriptedKeySource,
    VisionConfig,
    VisionObserver,
    agreement_report,
    apply_homography,
    arm_from_side,
    cohens_kappa,
    homography_residuals,
    pinhole_intrinsics,
    session_agreement,
    side_from_arm,
    solve_participant_frame,
    solve_table_homography,
)
from vla_lab.rehab.observation.vision import HandObservation

SIDE = "left"  # the participant's nonpreferred side


# --------------------------------------------------------------------------- labels


def test_arm_labels_convert_both_ways():
    assert arm_from_side("left", SIDE) == ARM_NONPREFERRED
    assert arm_from_side("right", SIDE) == ARM_PREFERRED
    assert arm_from_side("left", "right") == ARM_PREFERRED
    assert side_from_arm(ARM_NONPREFERRED, SIDE) == "left"
    assert side_from_arm(ARM_PREFERRED, SIDE) == "right"
    for special in (ARM_NONE, ARM_AMBIGUOUS):
        assert arm_from_side(special, SIDE) == special


def test_an_unknown_side_is_rejected_rather_than_guessed():
    try:
        arm_from_side("middle", SIDE)
        raise AssertionError("an unknown side must raise")
    except ValueError:
        pass


# --------------------------------------------------------------------------- keyed observer


def test_keyed_observer_latches_the_first_press():
    src = ScriptedKeySource({0: "z"})
    obs = KeyedObserver(SIDE, src)
    src.current_trial = 0
    obs.begin_trial(0, 1000)
    sel = obs.poll(1100)
    assert sel is not None and sel.arm == ARM_NONPREFERRED and sel.physical_side == "left"
    assert obs.poll(1200) is sel  # a later press is a re-attempt, not a new choice


def test_keyed_observer_reports_no_reach_on_a_timeout():
    obs = KeyedObserver(SIDE, lambda: None)
    obs.begin_trial(0, 0)
    sel = obs.end_trial(9000)
    assert sel.arm == ARM_NONE and "no reach" in sel.extra["reason"]


def test_keyed_observer_can_report_ambiguity():
    src = ScriptedKeySource({0: "x"})
    obs = KeyedObserver(SIDE, src)
    src.current_trial = 0
    obs.begin_trial(0, 0)
    assert obs.poll(10).arm == ARM_AMBIGUOUS


# --------------------------------------------------------------------------- vision observer


def _hand_track(side_y: float, *, t0: int, n: int = 6, dx: float = 0.03):
    """A hand entering from one side of the midline and moving toward the target."""

    return [
        HandObservation(t_ms=t0 + i * 50, x_m=0.20 + i * dx, y_m=side_y, camera="front")
        for i in range(n)
    ]


def test_vision_observer_calls_the_side_the_moving_hand_came_from():
    det = ScriptedHandDetector(_hand_track(0.18, t0=1000))
    obs = VisionObserver(SIDE, det, cfg=VisionConfig(move_threshold_m=0.08))
    obs.begin_trial(0, 1000)
    sel = None
    for t in range(1000, 1500, 25):
        sel = obs.poll(t)
        if sel:
            break
    assert sel is not None and sel.arm == ARM_NONPREFERRED  # +y is the participant's left
    assert sel.confidence > 0.5


def test_vision_observer_ignores_motion_before_the_go_signal():
    det = ScriptedHandDetector(_hand_track(0.18, t0=0))
    obs = VisionObserver(SIDE, det, cfg=VisionConfig(move_threshold_m=0.08))
    obs.begin_trial(0, 5000)  # GO is well after every observation
    assert obs.poll(5100) is None
    assert obs.end_trial(9000).arm == ARM_NONE


def test_vision_observer_reports_ambiguous_when_both_hands_move():
    both = sorted(
        _hand_track(0.18, t0=1000) + _hand_track(-0.18, t0=1000),
        key=lambda o: o.t_ms,
    )
    obs = VisionObserver(SIDE, ScriptedHandDetector(both), cfg=VisionConfig(move_threshold_m=0.08))
    obs.begin_trial(0, 1000)
    sel = None
    for t in range(1000, 1600, 25):
        sel = obs.poll(t)
        if sel:
            break
    assert sel is not None and sel.arm == ARM_AMBIGUOUS
    # Ambiguity is a recorded outcome, never a guess.
    assert sel.confidence == 0.0


def test_a_hand_hovering_on_the_midline_is_not_attributed_to_a_side():
    det = ScriptedHandDetector(_hand_track(0.005, t0=1000))
    obs = VisionObserver(SIDE, det, cfg=VisionConfig(midline_deadband_m=0.03))
    obs.begin_trial(0, 1000)
    for t in range(1000, 1500, 25):
        obs.poll(t)
    assert obs.end_trial(9000).arm == ARM_NONE


def test_low_confidence_detections_are_discarded():
    track = [
        HandObservation(t_ms=1000 + i * 50, x_m=0.20 + i * 0.03, y_m=0.18, detector_confidence=0.2)
        for i in range(6)
    ]
    obs = VisionObserver(SIDE, ScriptedHandDetector(track), cfg=VisionConfig(min_confidence=0.5))
    obs.begin_trial(0, 1000)
    for t in range(1000, 1500, 25):
        obs.poll(t)
    assert obs.end_trial(9000).arm == ARM_NONE


def test_mediapipe_adapter_refuses_to_pretend():
    from vla_lab.rehab.observation.vision import MediaPipeHandDetector

    try:
        MediaPipeHandDetector()
        raise AssertionError("the adapter shape must not masquerade as an implementation")
    except NotImplementedError as exc:
        assert "keyed-only" in str(exc)


# --------------------------------------------------------------------------- composite


def test_the_composite_runs_every_observer_and_reports_the_primary():
    src = ScriptedKeySource({0: "m"})
    keyed = KeyedObserver(SIDE, src)
    det = ScriptedHandDetector(_hand_track(0.18, t0=1000))
    vision = VisionObserver(SIDE, det, cfg=VisionConfig(move_threshold_m=0.08))
    comp = CompositeObserver([vision, keyed], primary=0)
    src.current_trial = 0
    comp.begin_trial(0, 1000)
    sel = None
    for t in range(1000, 1600, 25):
        sel = comp.poll(t)
        if sel:
            break
    assert sel is not None and sel.observer == "vision"
    # ...and the keyed observer disagreed, which must remain recoverable.
    assert keyed.end_trial(2000).arm == ARM_PREFERRED


# --------------------------------------------------------------------------- agreement


def test_cohens_kappa_endpoints():
    a = [ARM_PREFERRED, ARM_NONPREFERRED] * 10
    assert abs(cohens_kappa(a, a) - 1.0) < 1e-12
    b = [ARM_NONPREFERRED, ARM_PREFERRED] * 10
    assert cohens_kappa(a, b) < 0.0  # systematically opposite: worse than chance
    # One category on both sides: kappa is undefined, not 1.0.
    assert math.isnan(cohens_kappa([ARM_PREFERRED] * 5, [ARM_PREFERRED] * 5))


def test_agreement_report_lists_disagreements_for_review():
    online = {0: ARM_PREFERRED, 1: ARM_NONPREFERRED, 2: ARM_PREFERRED, 3: ARM_NONPREFERRED}
    coded = {0: ARM_PREFERRED, 1: ARM_NONPREFERRED, 2: ARM_NONPREFERRED, 3: ARM_NONPREFERRED}
    rep = agreement_report(online, coded, source_a="vision", source_b="coded").to_dict()
    assert rep["n_resolved"] == 4
    assert rep["n_disagreements"] == 1
    assert rep["disagreements"][0]["trial_idx"] == 2
    assert 0.0 < rep["kappa"] < 1.0


def test_unresolved_labels_are_excluded_from_kappa_by_default():
    a = {0: ARM_NONE, 1: ARM_PREFERRED, 2: ARM_NONPREFERRED}
    b = {0: ARM_NONE, 1: ARM_PREFERRED, 2: ARM_NONPREFERRED}
    rep = agreement_report(a, b).to_dict()
    assert rep["n"] == 3 and rep["n_resolved"] == 2  # the shared abstention does not inflate it


def test_session_agreement_covers_every_pair():
    rows = [
        {"trial_idx": 0, "observer": "vision", "source": "online", "arm": ARM_PREFERRED},
        {"trial_idx": 0, "observer": "keyed", "source": "online", "arm": ARM_PREFERRED},
        {"trial_idx": 0, "observer": "coder_a", "source": "coded", "arm": ARM_PREFERRED},
        {"trial_idx": 1, "observer": "vision", "source": "online", "arm": ARM_NONPREFERRED},
        {"trial_idx": 1, "observer": "keyed", "source": "online", "arm": ARM_NONPREFERRED},
        {"trial_idx": 1, "observer": "coder_a", "source": "coded", "arm": ARM_PREFERRED},
    ]
    reports = session_agreement(rows)
    assert any(":coded" in k for k in reports)
    assert len(reports) == 3  # 3 sources -> 3 pairs


def test_the_acceptance_threshold_is_the_preregistered_one():
    assert KAPPA_ACCEPTANCE == 0.9


def test_ingesting_coded_labels_appends_and_never_rewrites():
    from vla_lab.rehab.logging import OBSERVERS_FILE, SessionWriter
    from vla_lab.rehab.observation.coding import ingest_coded_labels

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "s"
        w = SessionWriter(d)
        w.write_protocol({"blocks": []})
        w.log_observation(trial_idx=0, observer="vision", arm=ARM_PREFERRED, t_ms=10)
        w.close()
        coded = Path(tmp) / "coded.jsonl"
        coded.write_text(json.dumps({"trial_idx": 0, "physical_side": "left", "coder": "coder_a"}) + "\n")
        n = ingest_coded_labels(d, coded, nonpreferred_side=SIDE)
        assert n == 1
        rows = [json.loads(l) for l in (d / OBSERVERS_FILE).read_text().splitlines() if l.strip()]
        assert len(rows) == 2
        assert {r["source"] for r in rows} == {"online", "coded"}


# --------------------------------------------------------------------------- calibration (W13)


def test_homography_round_trips_a_known_planar_map():
    table = [(0.0, 0.0), (0.4, 0.0), (0.4, 0.3), (0.0, 0.3), (0.2, 0.15)]
    # A synthetic camera: scale + translate + a little perspective.
    def project(x, y):
        w = 1.0 + 0.15 * x
        return ((600.0 * x + 100.0) / w, (600.0 * y + 50.0) / w)

    image = [project(x, y) for x, y in table]
    h = solve_table_homography(image, table)
    for ip, tp in zip(image, table):
        got = apply_homography(h, ip)
        assert math.dist(got, tp) < 1e-6
    res = homography_residuals(h, image, table)
    assert res["max_m"] < 1e-6


def test_homography_needs_enough_correspondences():
    try:
        solve_table_homography([(0, 0), (1, 0), (0, 1)], [(0, 0), (1, 0), (0, 1)])
        raise AssertionError("three points cannot determine a homography")
    except ValueError:
        pass


def test_participant_frame_solve_recovers_the_midline_and_reports_its_uncertainty():
    # Shoulders 0.38 m apart, participant facing +x in the table frame, sternum at (0.9, 0.1).
    rng = np.random.default_rng(0)
    left = [(0.9 + rng.normal(0, 0.002), 0.29 + rng.normal(0, 0.002)) for _ in range(20)]
    right = [(0.9 + rng.normal(0, 0.002), -0.09 + rng.normal(0, 0.002)) for _ in range(20)]
    pf = solve_participant_frame(left, right)
    assert abs(pf.shoulder_halfwidth_m - 0.19) < 0.01
    assert abs(pf.participant_in_table.tx - 0.9) < 0.01
    assert abs(pf.participant_in_table.ty - 0.1) < 0.01
    assert 0.0 < pf.midline_sd_m < 0.01  # reported, not assumed away


def test_the_calibration_gate_refuses_a_sloppy_midline():
    from vla_lab.rehab.observation.calibration import CalibrationBundle, ParticipantFrame

    b = CalibrationBundle(participant_frame=ParticipantFrame(midline_sd_m=0.05))
    problems = b.check(max_midline_sd_m=0.02)
    assert any("midline" in p for p in problems)
    assert CalibrationBundle(participant_frame=ParticipantFrame(midline_sd_m=0.005)).check() == []


def test_intrinsics_follow_the_repo_convention():
    k = pinhole_intrinsics(87.0, (640, 480))
    assert k["cx_px"] == 320.0 and k["cy_px"] == 240.0
    assert abs(k["fx_px"] - 640 / (2 * math.tan(math.radians(87.0) / 2))) < 1e-9


if __name__ == "__main__":
    import sys

    from vla_lab.tests import run_namespace

    sys.exit(1 if run_namespace(dict(globals()), label="test_rehab_observation") else 0)
