"""Offline tests for the reverse guard (FSD-mode never drives backward)."""

from __future__ import annotations

import pytest

from beamng_autopilot.control.reverse_guard import ReverseGuard


def test_reverse_detected_brakes() -> None:
    g = ReverseGuard()
    brake, rev = g.decide(-0.8)
    assert rev and brake == 1.0


def test_forward_no_brake() -> None:
    g = ReverseGuard()
    brake, rev = g.decide(2.0)
    assert not rev and brake == 0.0


def test_hysteresis_holds_through_near_zero() -> None:
    """Once reversing, the guard keeps braking until the car is clearly
    forward again (a bounce that settles at 0 must not release)."""
    g = ReverseGuard()
    g.decide(-1.0)       # reversing
    brake, rev = g.decide(-0.05)   # near zero, not yet forward
    assert rev and brake > 0.0
    brake, rev = g.decide(0.5)     # clearly forward again
    assert not rev and brake == 0.0


def test_detects_then_releases() -> None:
    g = ReverseGuard()
    seq = [(-0.6, True), (0.0, True), (0.8, False)]
    for spd, expected_rev in seq:
        _, rev = g.decide(spd)
        assert rev is expected_rev, f"speed {spd} -> {rev}"


def test_reset_clears_state() -> None:
    g = ReverseGuard()
    g.decide(-1.0)
    assert g.braking
    g.reset()
    assert not g.braking



def test_standstill_releases_after_settle() -> None:
    """A wall impact that stops the car releases after it has sat still
    long enough, otherwise the brakes lock forever at 0 m/s."""
    g = ReverseGuard()
    g.decide(-1.0)                      # reverse detected -> braking
    _, rev = g.decide(0.0, dt=0.3)      # stopping, still below clear
    assert rev
    _, rev = g.decide(0.0, dt=0.3)      # 0.6 s at standstill -> release
    assert not rev


def test_standstill_not_released_early() -> None:
    """Below the settle duration the guard keeps braking."""
    g = ReverseGuard()
    g.decide(-1.0)
    for _ in range(3):
        _, rev = g.decide(0.0, dt=0.1)  # only 0.3 s
        assert rev
