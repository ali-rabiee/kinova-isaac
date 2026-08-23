"""W12 — tests for ``vla_lab.old_direction.rehab.safety`` and the apparatus backends.

"Done when" (``rehab.md`` §6/W12): the state machine passes; on hardware every interlock is
additionally *demonstrated* and the demonstration recorded for the IRB packet — that part
cannot be tested here, and this file is not a substitute for it.

The central invariant under test: **motion and reach are mutually exclusive in time.** The arm
moves, stops, and only then is GO issued.
"""

from __future__ import annotations

from vla_lab.old_direction.rehab.apparatus import FakeGen2Driver, KinovaGen2Apparatus, LoopbackTransport, NullApparatus
from vla_lab.old_direction.rehab.apparatus.base import (
    HALT_DRIVER_FAULT,
    HALT_DWELL_TIMEOUT,
    HALT_ESTOP_PARTICIPANT,
    HALT_REACH_DURING_MOTION,
    HALT_REASONS,
    HALT_SPEED_LIMIT,
    HALT_TORQUE_LIMIT,
    HALT_WORKSPACE_VIOLATION,
    ApparatusFault,
)
from vla_lab.old_direction.rehab.apparatus.isaac_apparatus import ParticipantProxy, check_trajectories, straight_line_waypoints
from vla_lab.old_direction.rehab.contract import Phase0Contract
from vla_lab.old_direction.rehab.safety import (
    SOURCE_EXPERIMENTER,
    SOURCE_PARTICIPANT,
    STATE_GO_ISSUED,
    STATE_HALTED,
    STATE_MOVING,
    STATE_SETTLED,
    SafetyEnvelope,
    SafetyLimits,
    SafetyViolation,
)
from vla_lab.old_direction.rehab.trial import ManualClock, SessionClock

CONTRACT = Phase0Contract()


def _env(**kw):
    return SafetyEnvelope(SafetyLimits(**kw))


# --------------------------------------------------------------------------- mutual exclusion


def test_go_is_refused_unless_the_arm_is_stopped_and_settled():
    e = _env()
    assert not e.allow_go(0)
    e.begin_motion(0)
    assert not e.allow_go(10)
    try:
        e.issue_go(10)
        raise AssertionError("GO must be refused while the arm is moving")
    except SafetyViolation as exc:
        assert exc.reason == HALT_REACH_DURING_MOTION
    e.end_motion(20, settled=True)
    assert e.allow_go(20)
    e.issue_go(20)
    assert e.state == STATE_GO_ISSUED


def test_a_reach_during_motion_halts_immediately():
    e = _env()
    e.begin_motion(0)
    ev = e.reach_detected(50)
    assert ev is not None and ev.reason == HALT_REACH_DURING_MOTION
    assert e.state == STATE_HALTED
    assert e.halts[-1].state_before == STATE_MOVING


def test_motion_is_refused_while_the_participant_may_be_reaching():
    e = _env()
    e.begin_motion(0)
    e.end_motion(10, settled=True)
    e.issue_go(10)
    e.reach_detected(20)
    try:
        e.begin_motion(30)
        raise AssertionError("must refuse to move while a reach is in progress")
    except SafetyViolation as exc:
        assert exc.reason == HALT_REACH_DURING_MOTION


def test_a_full_clean_trial_returns_to_idle():
    e = _env()
    e.begin_motion(0)
    e.end_motion(10, settled=True)
    e.issue_go(10)
    e.reach_detected(20)
    e.end_reach(40)
    e.end_trial(60)
    assert not e.halted
    assert e.summary()["n_halts"] == 0


# --------------------------------------------------------------------------- limits


def test_speed_and_acceleration_caps_halt():
    e = _env(max_cartesian_speed_ms=0.12)
    try:
        e.check_motion_command(speed_ms=0.5, accel_ms2=0.1, t_ms=0)
        raise AssertionError("speed cap must halt")
    except SafetyViolation as exc:
        assert exc.reason == HALT_SPEED_LIMIT
    assert e.state == STATE_HALTED


def test_workspace_aabb_and_participant_clearance():
    e = _env()
    e.check_pose((0.4, 0.0, 0.1), 0)          # fine
    try:
        e.check_pose((0.05, 0.0, 0.1), 1)     # inside the participant clearance radius
        raise AssertionError("must halt inside the participant clearance")
    except SafetyViolation as exc:
        assert exc.reason == HALT_WORKSPACE_VIOLATION


def test_joint_current_threshold_halts():
    e = _env(max_joint_current_a=2.0)
    e.check_currents([0.5, 0.9, 1.2], 0)
    try:
        e.check_currents([0.5, 3.0], 1)
        raise AssertionError("current threshold must halt")
    except SafetyViolation as exc:
        assert exc.reason == HALT_TORQUE_LIMIT


def test_dwell_watchdog_halts_a_motion_that_never_ends():
    e = _env(max_motion_ms=1000)
    e.begin_motion(0)
    assert e.tick(500) is None
    ev = e.tick(2000)
    assert ev is not None and ev.reason == HALT_DWELL_TIMEOUT


# --------------------------------------------------------------------------- e-stop


def test_dual_estop_from_either_side():
    for src, reason in ((SOURCE_PARTICIPANT, "estop_participant"), (SOURCE_EXPERIMENTER, "estop_experimenter")):
        e = _env()
        ev = e.estop(src, 100)
        assert ev.reason == reason
        assert e.estop_engaged and e.halted


def test_reset_is_refused_while_an_estop_is_engaged():
    e = _env()
    e.estop(SOURCE_PARTICIPANT, 0)
    try:
        e.reset(10)
        raise AssertionError("reset must be refused while an e-stop is engaged")
    except SafetyViolation:
        pass
    e.release_estop(SOURCE_PARTICIPANT)
    e.reset(20)
    assert not e.halted


def test_participant_stop_request_is_a_logged_event_not_a_crash():
    e = _env()
    ev = e.request_stop(SOURCE_PARTICIPANT, 500, "asked to stop")
    assert ev.reason == "participant_request"
    assert e.halted
    assert e.summary()["halts_by_reason"]["participant_request"] == 1


def test_every_halt_reason_is_in_the_taxonomy():
    e = _env()
    e.estop(SOURCE_PARTICIPANT, 0)
    assert all(h.reason in HALT_REASONS for h in e.halts)
    try:
        e._halt("made_up_reason", 0)
        raise AssertionError("halt reasons outside the taxonomy must be rejected")
    except ValueError:
        pass


# --------------------------------------------------------------------------- apparatus backends


def test_null_apparatus_presents_and_settles():
    clock = ManualClock()
    app = NullApparatus(CONTRACT.timing, clock=SessionClock(source=clock), manual_clock=clock)
    app.connect()
    app.home()
    t = CONTRACT.target_grid().targets[0]
    res = app.present(t)
    assert res.settled and res.t_settled_ms > res.t_present_ms
    assert app.state().ee_xy_participant_m == (t.x_m, t.y_m)


def test_gen2_backend_speaks_the_bridge_protocol_over_a_fake_driver():
    driver = FakeGen2Driver()
    app = KinovaGen2Apparatus(LoopbackTransport(driver), timing=CONTRACT.timing)
    app.connect()
    assert app.driver_version == driver.driver_version
    app.home()
    t = CONTRACT.target_grid().targets[0]
    res = app.present(t)
    assert res.settled and res.pose_error_m == driver.pose_error_m
    assert app.heartbeat()
    st = app.state()
    assert st.connected and not st.estop_engaged
    app.close()
    assert driver.closed


def test_gen2_backend_retries_a_failed_settle_then_reports_failure():
    driver = FakeGen2Driver(fail_settle_on=(1, 2, 3))
    app = KinovaGen2Apparatus(LoopbackTransport(driver), timing=CONTRACT.timing)
    app.connect()
    res = app.present(CONTRACT.target_grid().targets[0])
    assert not res.settled and res.fault == "settle_failed"
    assert driver.n_present == 2  # one retry, then it reports rather than pretending


def test_gen2_backend_surfaces_a_driver_fault_as_a_halt():
    driver = FakeGen2Driver(fault_on={1: "E0301_joint_overheat"})
    app = KinovaGen2Apparatus(LoopbackTransport(driver), timing=CONTRACT.timing)
    app.connect()
    try:
        app.present(CONTRACT.target_grid().targets[0])
        raise AssertionError("a driver fault must not be swallowed")
    except ApparatusFault as exc:
        assert "E0301_joint_overheat" in str(exc)
    assert app.halted and app.state().fault


def test_gen2_backend_halts_on_a_lost_heartbeat():
    driver = FakeGen2Driver(timeout_on=(1,))
    app = KinovaGen2Apparatus(LoopbackTransport(driver), timing=CONTRACT.timing)
    app.connect()
    try:
        app.present(CONTRACT.target_grid().targets[0])
        raise AssertionError("a bridge timeout must not be swallowed")
    except ApparatusFault:
        pass
    assert app.halted


def test_effort_manipulation_can_require_the_experimenter():
    driver = FakeGen2Driver()
    app = KinovaGen2Apparatus(LoopbackTransport(driver), timing=CONTRACT.timing)
    app.connect()
    assert app.configure_effort("moderate") is True   # a weighted puck must be staged by hand
    assert NullApparatus(CONTRACT.timing).configure_effort("moderate") is False


# --------------------------------------------------------------------------- twin geometry (no Isaac)


def test_no_presentation_trajectory_crosses_the_participant_proxy():
    grid = CONTRACT.target_grid()
    collisions = check_trajectories(grid, proxy=ParticipantProxy())
    assert collisions == [], collisions[:3]


def test_the_clearance_check_actually_catches_a_bad_layout():
    grid = CONTRACT.target_grid()
    # A home pose behind the participant would sweep the arm straight through them.
    collisions = check_trajectories(grid, proxy=ParticipantProxy(), home_xy=(-0.40, 0.0))
    assert collisions, "a trajectory through the participant must be caught"


def test_waypoints_span_the_whole_path():
    wp = straight_line_waypoints((0.6, 0.0), (0.28, -0.29), n=10)
    assert len(wp) == 10
    assert wp[0] == (0.6, 0.0)
    assert abs(wp[-1][0] - 0.28) < 1e-9 and abs(wp[-1][1] + 0.29) < 1e-9


if __name__ == "__main__":
    import sys

    from vla_lab.tests import run_namespace

    sys.exit(1 if run_namespace(dict(globals()), label="test_rehab_safety") else 0)
