"""Speed-weighted steering blend (plan phase D2): opt-in, reversible."""

from __future__ import annotations

import json

import numpy as np
import pytest

from beamng_autopilot.control.blend import (
    SteeringBlendWeights,
    _schedule,
    blend_steering,
)


def test_disabled_blend_is_pure_pursuit_plus_feedforward() -> None:
    """The default must not change today's behaviour by one bit."""
    b = blend_steering(0.3, -0.1, lateral_error_m=1.5,
                       heading_error_rad=0.4, speed_mps=6.0)
    assert b.enabled is False
    assert b.steer == pytest.approx(0.2)
    assert b.lat_term == 0.0 and b.head_term == 0.0
    assert b.lat_gain == 0.0 and b.head_gain == 0.0


def test_enabled_blend_adds_the_error_terms() -> None:
    w = SteeringBlendWeights(enabled=True, lat_gain_low=0.1,
                             lat_gain_high=0.1, head_gain_low=0.5,
                             head_gain_high=0.5)
    b = blend_steering(0.2, 0.0, lateral_error_m=1.0,
                       heading_error_rad=0.2, speed_mps=6.0, weights=w)
    assert b.enabled is True
    assert b.lat_term == pytest.approx(0.1)
    assert b.head_term == pytest.approx(0.1)
    assert b.steer == pytest.approx(0.4)
    assert 0.0 < b.rate < 1.0


def test_weights_are_scheduled_by_speed() -> None:
    w = SteeringBlendWeights(enabled=True, lat_gain_low=1.0,
                             lat_gain_high=0.0, head_gain_low=0.0,
                             head_gain_high=1.0)
    # both errors are 1.0 so the scheduled GAINS show through directly
    slow = blend_steering(0.0, 0.0, 1.0, 1.0, speed_mps=1.0, weights=w)
    fast = blend_steering(0.0, 0.0, 1.0, 1.0, speed_mps=20.0, weights=w)
    assert slow.lat_term == pytest.approx(1.0)      # lateral dominates low
    assert slow.head_term == pytest.approx(0.0)
    assert fast.lat_term == pytest.approx(0.0)      # heading dominates high
    assert fast.head_term == pytest.approx(1.0)
    mid = blend_steering(0.0, 0.0, 1.0, 1.0, speed_mps=7.0, weights=w)
    assert 0.0 < mid.lat_term < 1.0
    assert 0.0 < mid.head_term < 1.0


def test_schedule_endpoints_clamp() -> None:
    assert _schedule(1.0, 0.0, -5.0, 2.0, 12.0) == pytest.approx(1.0)
    assert _schedule(1.0, 0.0, 99.0, 2.0, 12.0) == pytest.approx(0.0)
    # a degenerate schedule must not divide by zero
    assert _schedule(1.0, 0.5, 5.0, 4.0, 4.0) == pytest.approx(0.5)


def test_output_is_clamped_and_reported() -> None:
    w = SteeringBlendWeights(enabled=True, lat_gain_low=5.0,
                             lat_gain_high=5.0)
    b = blend_steering(0.9, 0.0, lateral_error_m=1.0,
                       heading_error_rad=0.0, speed_mps=2.0, weights=w)
    assert b.steer == pytest.approx(1.0)
    assert b.clamped is True


def test_zero_demand_has_no_share_and_no_nan() -> None:
    w = SteeringBlendWeights(enabled=True)
    b = blend_steering(0.0, 0.0, 0.0, 0.0, speed_mps=0.0, weights=w)
    assert b.steer == 0.0
    assert np.isfinite(b.rate)
    assert b.rate == 0.0


def test_curvature_feedforward_gain_is_respected() -> None:
    b = blend_steering(0.0, 0.2, 0.0, 0.0, speed_mps=6.0,
                       weights=SteeringBlendWeights(ff_gain=0.5))
    assert b.steer == pytest.approx(0.1)


def test_digest_is_json_safe() -> None:
    w = SteeringBlendWeights(enabled=True, lat_gain_low=0.2,
                             lat_gain_high=0.2)
    b = blend_steering(0.1, 0.0, 0.5, 0.1, speed_mps=5.0, weights=w)
    text = json.dumps(b.digest())
    assert "lat_w" in text
    assert "nan" not in text.lower()
