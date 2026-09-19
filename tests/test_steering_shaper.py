"""Steering shaper (plan phase D3): rate, jerk and reversal limits."""

from __future__ import annotations

import numpy as np
import pytest

from beamng_autopilot.control.steering import (
    STEER_MAX_JERK_PER_S2,
    STEER_MAX_RATE_PER_S,
    STEER_TRIM_BAND,
    SteeringShaper,
)


def _run(shaper: SteeringShaper, requests, dt: float = 0.05):
    out = []
    t = 0.0
    for r in requests:
        out.append(shaper.update(r, dt, now=t))
        t += dt
    return out


def test_first_step_respects_the_rate_limit() -> None:
    """The first step is bounded by BOTH limits.

    From rest the jerk bound is the binding one (the rate has to ramp up
    from zero), so the step is smaller than rate*dt - but never larger.
    """
    s = SteeringShaper()
    v = s.update(1.0, 0.1, now=0.0)
    assert 0.0 < v <= STEER_MAX_RATE_PER_S * 0.1 + 1e-9
    assert s.rate <= STEER_MAX_RATE_PER_S + 1e-9


def test_rate_ramps_under_the_jerk_bound() -> None:
    """A step request must not snap the rate to its ceiling in one step."""
    s = SteeringShaper()
    rates = []
    for _ in range(20):
        s.update(1.0, 0.05, now=0.0)
        rates.append(s.rate)
    drates = np.abs(np.diff([0.0] + rates))
    assert np.all(drates <= STEER_MAX_JERK_PER_S2 * 0.05 + 1e-9)
    # the ramp is visible: it climbs to the ceiling rather than starting there
    assert rates[0] == pytest.approx(STEER_MAX_JERK_PER_S2 * 0.05, abs=1e-9)
    assert max(rates) == pytest.approx(STEER_MAX_RATE_PER_S, abs=1e-6)
    # once the wheel saturates the request is satisfied and the rate bleeds
    # back down - no permanent full-rate demand against the stop
    assert rates[-1] < max(rates)


def test_jerk_bound_holds_on_a_reversal() -> None:
    """Reversing the request must not flip the rate instantly either."""
    s = SteeringShaper()
    _run(s, [1.0] * 40)
    rates = [s.rate]
    for _ in range(20):
        s.update(-1.0, 0.05)
        rates.append(s.rate)
    drates = np.abs(np.diff(rates))
    assert np.all(drates <= STEER_MAX_JERK_PER_S2 * 0.05 + 1e-9)
    # ...and the reversal does take full effect (the wheel was saturated,
    # so the rate had already bled to zero before the request flipped)
    assert min(rates) == pytest.approx(-STEER_MAX_RATE_PER_S, abs=1e-6)


def test_small_amplitude_wiggle_is_suppressed() -> None:
    """The failure this exists for: flickering trim commands."""
    s = SteeringShaper(max_reversals=3)
    seq = [0.03, -0.03] * 20
    values = _run(s, seq, dt=0.05)
    assert s.suppressed > 0, "the reversal guard never engaged"
    tail = values[-10:]
    assert max(abs(v) for v in tail) <= STEER_TRIM_BAND + 1e-9, tail


def test_real_corrections_are_never_suppressed() -> None:
    """A large alternating command is a real manoeuvre, not a wiggle."""
    s = SteeringShaper(max_reversals=1)
    values = _run(s, [0.5, -0.5] * 6, dt=0.05)
    assert s.suppressed == 0
    # the wheel keeps following: later values are far outside the trim band
    assert max(abs(v) for v in values[-4:]) > STEER_TRIM_BAND


def test_reversal_window_resets() -> None:
    s = SteeringShaper(max_reversals=2, reversal_window_s=0.2)
    for i in range(6):
        _run(s, [0.03], dt=0.05)
        s.update(-0.03, 0.05, now=i * 1.0)     # a fresh window each time
    assert s.suppressed == 0


def test_force_bypasses_every_limit() -> None:
    """A safety action must never be delayed by the shaper."""
    s = SteeringShaper()
    _run(s, [0.02] * 5)
    v = s.update(0.9, 0.05, force=True)
    assert v == pytest.approx(0.9)
    assert s.value == pytest.approx(0.9)


def test_clamped_to_the_normalized_input_range() -> None:
    s = SteeringShaper()
    assert s.update(5.0, 0.1) <= 1.0
    s2 = SteeringShaper()
    assert s2.update(-5.0, 0.1) >= -1.0


def test_degenerate_dt_is_safe() -> None:
    s = SteeringShaper()
    for dt in (0.0, -1.0, 10.0, float("nan")):
        v = s.update(0.5, dt)
        assert np.isfinite(v)
        assert -1.0 <= v <= 1.0


def test_digest_is_json_safe() -> None:
    import json
    s = SteeringShaper()
    _run(s, [0.1, -0.1, 0.05])
    text = json.dumps(s.digest())
    assert "rate" in text


def test_reset_clears_the_state() -> None:
    s = SteeringShaper()
    _run(s, [0.4] * 10)
    s.reset()
    assert s.value == 0.0 and s.rate == 0.0 and s.suppressed == 0
    assert s.update(0.4, 0.05) < 0.4
