"""Unit tests for the steady painted-line lateral corrector.

``PaintedLineLateralCorrector`` nudges the near part of the chosen path
laterally toward the PERCEIVED own-lane centre (painted line right side
+ lane half width) at a bounded rate - the FSD-real way to keep the car
in its own lane when the map-prior lane rides the centre line.  The CV
back-projection is not exercised here; these tests cover the pure shift
math, rate limiting, hold/decay and horizon blending.
"""

from __future__ import annotations

import numpy as np

from beamng_autopilot.vision.lanes import PaintedLineLateralCorrector


def _rightward_target(pos, right_m: float, heading: float = 0.0):
    """World point ``right_m`` to the RIGHT of ``pos`` (heading 0 = +x)."""
    import math
    fwd = np.array([math.cos(heading), math.sin(heading)])
    right = np.array([fwd[1], -fwd[0]])
    p = np.asarray(pos[:2], dtype=float)
    return (p + right * right_m).tolist()


def test_desired_shift_sign_and_deadband():
    corr = PaintedLineLateralCorrector(max_shift_m=1.0)
    # Target 0.6 m right of the ego -> positive shift (move right).
    d = corr.desired_shift(_rightward_target((10.0, 20.0), 0.6),
                           (10.0, 20.0, 1.5), 0.0)
    assert abs(d - 0.6) < 1e-9
    # Target left of the ego -> negative shift.
    d = corr.desired_shift(_rightward_target((10.0, 20.0), -0.4),
                           (10.0, 20.0, 1.5), 0.0)
    assert abs(d + 0.4) < 1e-9
    # A few cm of jitter around the centre is "already centred".
    assert corr.desired_shift(_rightward_target((10.0, 20.0), 0.01),
                              (10.0, 20.0, 1.5), 0.0) == 0.0


def test_desired_shift_clamps_at_max():
    corr = PaintedLineLateralCorrector(max_shift_m=1.0)
    d = corr.desired_shift(_rightward_target((10.0, 20.0), 2.5),
                           (10.0, 20.0, 1.5), 0.0)
    assert d == 1.0
    d = corr.desired_shift(_rightward_target((10.0, 20.0), -2.5),
                           (10.0, 20.0, 1.5), 0.0)
    assert d == -1.0


def test_update_rate_limits_toward_desired():
    corr = PaintedLineLateralCorrector(max_shift_m=1.0, rate_m_s=1.2)
    s1 = corr.update(1.0, dt=0.5, speed=6.0)   # step = 0.6
    assert abs(s1 - 0.6) < 1e-9
    s2 = corr.update(1.0, dt=0.5, speed=6.0)   # 0.6 -> 1.0
    assert abs(s2 - 1.0) < 1e-9
    # A later frame with the line back at the centre ramps DOWN, not snap.
    s3 = corr.update(0.0, dt=0.25, speed=6.0)  # step = 0.3
    assert abs(s3 - 0.7) < 1e-9


def test_update_clamps_state_to_max():
    corr = PaintedLineLateralCorrector(max_shift_m=0.8, rate_m_s=5.0)
    s = corr.update(3.0, dt=1.0, speed=6.0)
    assert abs(s - 0.8) < 1e-9


def test_update_freezes_while_parked():
    corr = PaintedLineLateralCorrector(min_speed_mps=0.5)
    corr.update(0.8, dt=0.5, speed=6.0)
    parked = corr.update(0.8, dt=0.5, speed=0.1)
    assert parked == corr.shift_m  # no integration at standstill


def test_update_holds_then_decays_on_line_dropout():
    corr = PaintedLineLateralCorrector(max_shift_m=1.0, rate_m_s=1.2,
                                       hold_s=2.0)
    s1 = corr.update(0.8, dt=0.5, speed=6.0, now=100.0)
    assert abs(s1 - 0.6) < 1e-9
    # Line drops but the window is fresh -> the desired shift is held.
    s2 = corr.update(None, dt=0.5, speed=6.0, now=100.5)
    assert s2 > s1  # still integrating toward the last perceived centre
    # After the hold window the shift decays toward zero.
    s3 = corr.update(None, dt=0.5, speed=6.0, now=103.0)
    assert s3 < s2


def test_apply_blends_shift_to_zero_at_horizon():
    corr = PaintedLineLateralCorrector(horizon_m=12.0)
    path = np.column_stack([np.linspace(0.0, 20.0, 11),
                            np.zeros(11)])
    corr.shift_m = 0.6  # right of heading 0 (fwd +x, right = -y)
    out = corr.apply(path, (0.0, 0.0, 1.5), 0.0)
    # At the ego the full shift is applied.
    assert abs(out[0, 1] + 0.6) < 1e-9
    # Halfway to the horizon (lon = 6 m) half the shift.
    assert abs(out[3, 1] + 0.3) < 1e-9
    # At / beyond the horizon the path is untouched.
    assert abs(out[10, 1]) < 1e-9


def test_apply_preserves_extra_columns():
    corr = PaintedLineLateralCorrector(horizon_m=12.0)
    path = np.column_stack([np.linspace(0.0, 20.0, 11),
                            np.zeros(11),
                            np.linspace(1.0, 3.0, 11)])
    corr.shift_m = 0.5
    out = corr.apply(path, (0.0, 0.0, 1.5), 0.0)
    assert out.shape == path.shape
    np.testing.assert_allclose(out[:, 2], path[:, 2])


def test_apply_noop_without_shift():
    corr = PaintedLineLateralCorrector()
    path = np.column_stack([np.linspace(0.0, 20.0, 11), np.zeros(11)])
    corr.shift_m = 0.0
    assert corr.apply(path, (0.0, 0.0, 1.5), 0.0) is path
    assert corr.apply(None, (0.0, 0.0, 1.5), 0.0) is None
