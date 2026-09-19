"""Driving modes (plan phase D5): classification ladder + per-mode policy."""

from __future__ import annotations

import json

import pytest

from beamng_autopilot.control.drive_mode import (
    MODE_ALIGN,
    MODE_CRUISE,
    MODE_CURVE,
    MODE_OBSTACLE,
    MODE_RECOVERY,
    MODE_STARTING,
    MODE_STOP,
    MODES,
    DriveModeClassifier,
)


def _classify(c: DriveModeClassifier, **kw):
    base = dict(now=100.0, speed_mps=6.0, target_speed=6.0,
                stop_commanded=False, risk_braking=False, radius_m=None,
                lateral_error_m=None, perception_ok=True, elapsed_s=60.0)
    base.update(kw)
    return c.classify(**base)


def test_stop_commanded_wins_over_everything() -> None:
    c = DriveModeClassifier()
    p = _classify(c, stop_commanded=True, speed_mps=6.0, risk_braking=True)
    assert p.mode == MODE_STOP
    assert p.max_speed_mps == 0.0
    assert p.allow_throttle is False


def test_zero_target_is_a_stop_command() -> None:
    p = _classify(DriveModeClassifier(), target_speed=0.0)
    assert p.mode == MODE_STOP


def test_obstacle_braking_forbids_throttle() -> None:
    p = _classify(DriveModeClassifier(), risk_braking=True, speed_mps=6.0)
    assert p.mode == MODE_OBSTACLE
    assert p.allow_throttle is False
    # the speed bound stays with the C3 risk cap, not this mode
    assert p.max_speed_mps == float("inf")


def test_launch_reduces_steering_authority() -> None:
    p = _classify(DriveModeClassifier(), speed_mps=1.0, elapsed_s=1.0)
    assert p.mode == MODE_STARTING
    assert p.steer_gain < 1.0
    assert p.allow_throttle is True


def test_low_speed_alignment_needs_a_real_offset() -> None:
    c = DriveModeClassifier()
    on_centre = _classify(c, speed_mps=2.0, lateral_error_m=0.05,
                          elapsed_s=60.0)
    assert on_centre.mode == MODE_CRUISE
    c2 = DriveModeClassifier()
    off = _classify(c2, speed_mps=2.0, lateral_error_m=0.8, elapsed_s=60.0)
    assert off.mode == MODE_ALIGN
    assert off.max_speed_mps <= 3.0
    assert off.steer_gain < 1.0


def test_bend_ahead_is_curve_entry() -> None:
    p = _classify(DriveModeClassifier(), radius_m=15.0, speed_mps=5.0)
    assert p.mode == MODE_CURVE


def test_cruising_is_the_default() -> None:
    p = _classify(DriveModeClassifier(), radius_m=200.0)
    assert p.mode == MODE_CRUISE
    assert p.steer_gain == 1.0
    assert p.allow_throttle is True


def test_stopped_after_a_command_becomes_recovery_and_creeps() -> None:
    c = DriveModeClassifier()
    assert _classify(c, stop_commanded=True, speed_mps=0.0).mode == MODE_STOP
    # commanded stop released, car still stopped: creep back, not cruise
    p = _classify(c, speed_mps=0.1, now=101.0)
    assert p.mode == MODE_RECOVERY
    assert p.max_speed_mps <= 1.5
    # ...and once it is moving again it is no longer "recovering"
    p2 = _classify(c, speed_mps=2.0, now=104.0)
    assert p2.mode == MODE_CRUISE


def test_recovery_requires_usable_perception() -> None:
    """Stopped with perception unavailable is a stop, not a creep."""
    c = DriveModeClassifier()
    _classify(c, stop_commanded=True, speed_mps=0.0)
    p = _classify(c, speed_mps=0.0, now=101.0, perception_ok=False)
    assert p.mode == MODE_STOP
    assert p.allow_throttle is False


def test_escalation_is_immediate_and_de_escalation_waits() -> None:
    c = DriveModeClassifier(dwell_s=0.5)
    _classify(c, speed_mps=6.0, now=0.0)                 # cruise
    stop = _classify(c, stop_commanded=True, speed_mps=6.0, now=0.1)
    assert stop.mode == MODE_STOP and stop.changed is True
    # releasing the stop does NOT immediately claim cruising
    held = _classify(c, speed_mps=6.0, now=0.2)
    assert held.mode == MODE_STOP
    assert "dwell" in held.reason
    after = _classify(c, speed_mps=6.0, now=1.0)
    assert after.mode == MODE_CRUISE


def test_obstacle_escalates_without_waiting() -> None:
    c = DriveModeClassifier(dwell_s=5.0)
    _classify(c, speed_mps=6.0, now=0.0)
    p = _classify(c, risk_braking=True, speed_mps=6.0, now=0.1)
    assert p.mode == MODE_OBSTACLE


def test_every_mode_has_a_policy_and_a_digest() -> None:
    assert len(set(MODES)) == len(MODES)
    # a FRESH classifier per scenario: reusing one would (correctly) let
    # the state - the stop memory and the de-escalation dwell - carry over
    for kw, expected in (
            (dict(stop_commanded=True), MODE_STOP),
            (dict(risk_braking=True), MODE_OBSTACLE),
            (dict(speed_mps=1.0, elapsed_s=0.5), MODE_STARTING),
            (dict(speed_mps=2.0, lateral_error_m=1.0), MODE_ALIGN),
            (dict(radius_m=10.0), MODE_CURVE),
            (dict(), MODE_CRUISE)):
        p = _classify(DriveModeClassifier(), **kw)
        assert p.mode == expected, (kw, p.mode)
        json.dumps(p.digest())


def test_reset_clears_state() -> None:
    c = DriveModeClassifier()
    _classify(c, stop_commanded=True, speed_mps=5.0)
    c.reset()
    assert c.mode == MODE_CRUISE and c.saw_stop is False
    assert c.switches == 0 and c.since is None


def test_drive_modes_are_opt_in_and_env_wired() -> None:
    """The loop must keep its exact behaviour unless the A/B turns D5 on."""
    import os
    import beamng_autopilot.fsd_drive as fd
    assert fd.DRIVE_MODES_ENABLED == (
        os.environ.get("BEAMNG_DRIVE_MODES", "0") == "1")
    if "BEAMNG_DRIVE_MODES" not in os.environ:
        assert fd.DRIVE_MODES_ENABLED is False


def test_policy_suppression_can_only_remove_throttle() -> None:
    """The stop modes must never ask for throttle; the others never cap it."""
    c = DriveModeClassifier()
    stopping = _classify(c, stop_commanded=True, speed_mps=4.0)
    assert stopping.allow_throttle is False
    c2 = DriveModeClassifier()
    cruising = _classify(c2, speed_mps=8.0)
    assert cruising.allow_throttle is True
    assert cruising.max_speed_mps == float("inf")
