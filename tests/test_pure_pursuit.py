"""Offline tests for PurePursuit adaptive lookahead (speed-aware)."""

from __future__ import annotations

import numpy as np
import pytest

from beamng_autopilot.control.pure_pursuit import PurePursuit


def _straight_path(n=80, step=0.5):
    x = np.arange(n, dtype=float) * step
    return np.stack([x, np.zeros(n)], axis=1)


def test_find_target_speed_extends_lookahead() -> None:
    pp = PurePursuit(lookahead=6.0)
    path = _straight_path()
    # straight ahead on the +x axis
    t0, _, _ = pp.find_target(np.array([0.0, 0.0]), path, 0)
    t1, _, _ = pp.find_target(np.array([0.0, 0.0]), path, 0, speed=10.0)
    assert t0[0] == pytest.approx(6.0, abs=0.6)
    assert t1[0] == pytest.approx(pp.adaptive_lookahead(10.0), abs=0.6)
    assert t1[0] > t0[0]


def test_steering_adaptive_softer_at_speed() -> None:
    pp = PurePursuit(lookahead=6.0)
    path = _straight_path()
    # heading slightly off +x -> a correction steer; adaptive lookahead at
    # speed reduces the required angle magnitude (farther target).
    s0, _, _ = pp.steering(np.array([0.0, 0.0]), 0.3, path, 0)
    s1, _, _ = pp.steering(np.array([0.0, 0.0]), 0.3, path, 0, speed=12.0)
    assert abs(s1) < abs(s0)


def test_backward_compatible_no_speed() -> None:
    pp = PurePursuit(lookahead=6.0)
    path = _straight_path()
    t_a, _, _ = pp.find_target(np.array([0.0, 0.0]), path, 0)
    t_b, _, _ = pp.find_target(np.array([0.0, 0.0]), path, 0, speed=None)
    assert np.allclose(t_a, t_b)
