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