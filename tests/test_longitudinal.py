"""Longitudinal composition + jerk limiting (plan phase D4)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from beamng_autopilot.obstacle_risk import ttc_speed_cap
from beamng_autopilot.planning.longitudinal import (
    LONG_CONFIDENCE_FLOOR,
    LONG_MAX_ACCEL_MPS2,
    LONG_MAX_DECEL_MPS2,
    LONG_MAX_JERK_MPS3,
    LongitudinalPlanner,
)


# ---------------------------------------------------------------------------
# composition
# ---------------------------------------------------------------------------

def test_curvature_cap_from_radius() -> None:
    p = LongitudinalPlanner(comfort_lat_mps2=2.0)
    t = p.compose(plan_speed=20.0, radius_m=16.0)
    assert t.curvature == pytest.approx(math.sqrt(2.0 * 16.0))
    assert t.target == pytest.approx(math.sqrt(32.0))


def test_missing_radius_does_not_cap() -> None:
    t = LongitudinalPlanner().compose(plan_speed=8.0)
    assert math.isinf(t.curvature)
    assert t.target == pytest.approx(8.0)


def test_obstacle_cap_matches_the_risk_model_formula() -> None:
    t = LongitudinalPlanner().compose(plan_speed=20.0, gap_m=10.0, ttc_s=2.0)
    assert t.obstacle == pytest.approx(ttc_speed_cap(10.0, 5.0))


def test_obstacle_inside_the_margin_targets_zero() -> None:
    t = LongitudinalPlanner().compose(plan_speed=6.0, gap_m=1.0)
    assert t.target == 0.0


def test_confidence_scales_down_but_never_up() -> None:
    p = LongitudinalPlanner()
    weak = p.compose(plan_speed=10.0, road_conf=0.0, line_conf=0.0)
    assert weak.confidence == pytest.approx(LONG_CONFIDENCE_FLOOR)
    assert weak.target == pytest.approx(10.0 * LONG_CONFIDENCE_FLOOR)
    # the WEAKEST confidence governs: 0.9 does not pass as 1.0
    strong = p.compose(plan_speed=10.0, road_conf=1.0, line_conf=0.9)
    assert strong.confidence == pytest.approx(
        LONG_CONFIDENCE_FLOOR + (1.0 - LONG_CONFIDENCE_FLOOR) * 0.9)
    assert strong.target == pytest.approx(10.0 * strong.confidence)
    full = p.compose(plan_speed=10.0, road_conf=1.0, line_conf=1.0)
    assert full.confidence == pytest.approx(1.0)
    assert full.target == pytest.approx(10.0)            # never above plan
    assert p.compose(plan_speed=10.0).confidence == 1.0  # unknown -> no scale


def test_tightest_cap_wins() -> None:
    t = LongitudinalPlanner().compose(
        plan_speed=30.0, radius_m=100.0, gap_m=12.0, ttc_s=6.0,
        road_conf=0.5, line_conf=0.5)
    assert t.target < min(t.curvature, t.obstacle)
    assert t.target == pytest.approx(
        min(t.curvature, t.obstacle) * t.confidence, rel=1e-6)


# ---------------------------------------------------------------------------
# shaping
# ---------------------------------------------------------------------------

def _ramp(p: LongitudinalPlanner, target_plan: float, steps: int = 60,
          v0: float = 0.0, dt: float = 0.05):
    v = v0
    out = []
    for _ in range(steps):
        t = p.update(speed_mps=v, dt=dt, plan_speed=target_plan)
        out.append(t)
        v = t.reference
    return out


def test_reference_respects_the_acceleration_and_jerk_bounds() -> None:
    p = LongitudinalPlanner()
    terms = _ramp(p, 12.0)
    accels = [t.accel for t in terms]
    jerks = [abs(t.jerk) for t in terms]
    assert max(accels) <= LONG_MAX_ACCEL_MPS2 + 1e-9
    assert max(jerks) <= LONG_MAX_JERK_MPS3 + 1e-9
    # the ramp is visible: it does not jump to full accel in one step
    assert accels[0] < LONG_MAX_ACCEL_MPS2
    assert terms[-1].reference > terms[0].reference


def test_braking_is_jerk_limited_too() -> None:
    p = LongitudinalPlanner()
    _ramp(p, 12.0, steps=80)
    v = 12.0
    decels = []
    # 3 m/s^2 with a 2.5 m/s^3 jerk ramp needs ~1.5 s just to reach full
    # braking, so stopping from 12 m/s takes ~6 s - that IS the comfort
    # bound; the emergency paths bypass this module entirely.
    for _ in range(200):
        t = p.update(speed_mps=v, dt=0.05, plan_speed=0.0)
        decels.append(t.accel)
        v = t.reference
    assert min(decels) >= -LONG_MAX_DECEL_MPS2 - 1e-9
    jerks = np.abs(np.diff(np.asarray(decels, dtype=float))) / 0.05
    assert float(jerks.max()) <= LONG_MAX_JERK_MPS3 + 1e-6
    assert v == pytest.approx(0.0, abs=0.05)          # it does stop


def test_no_overshoot_toward_the_target() -> None:
    p = LongitudinalPlanner()
    v = 0.0
    for _ in range(120):
        t = p.update(speed_mps=v, dt=0.05, plan_speed=5.0)
        assert t.reference <= 5.0 + 1e-9
        v = t.reference
    assert v == pytest.approx(5.0, abs=0.05)


def test_stop_is_not_overshot_from_above() -> None:
    p = LongitudinalPlanner()
    v = 8.0
    for _ in range(120):
        t = p.update(speed_mps=v, dt=0.05, plan_speed=0.0)
        assert t.reference >= -1e-9
        v = t.reference
    assert v == pytest.approx(0.0, abs=0.05)


def test_degenerate_dt_is_safe() -> None:
    p = LongitudinalPlanner()
    for dt in (0.0, -1.0, float("nan"), 99.0):
        t = p.update(speed_mps=3.0, dt=dt, plan_speed=6.0)
        assert np.isfinite(t.reference)
        assert np.isfinite(t.accel) and np.isfinite(t.jerk)


def test_reset_clears_the_accel_state() -> None:
    p = LongitudinalPlanner()
    _ramp(p, 10.0, steps=20)
    p.reset()
    assert p.accel == 0.0 and p.started is False
    t = p.update(speed_mps=0.0, dt=0.05, plan_speed=10.0)
    assert t.jerk <= LONG_MAX_JERK_MPS3 + 1e-9


def test_digest_is_json_safe_with_infinities() -> None:
    t = LongitudinalPlanner().compose(plan_speed=7.0)
    text = json.dumps(t.digest())
    assert "Infinity" not in text and "NaN" not in text
    assert t.digest()["curve"] is None


# ---------------------------------------------------------------------------
# _path_radius_m (fsd_drive helper feeding the curvature cap)
# ---------------------------------------------------------------------------

def test_path_radius_none_for_straight_and_short_paths() -> None:
    from beamng_autopilot.fsd_drive import _path_radius_m
    assert _path_radius_m(None, (0.0, 0.0, 0.0), 0.0) is None
    straight = np.column_stack([np.linspace(0.0, 30.0, 31), np.zeros(31)])
    assert _path_radius_m(straight, (0.0, 0.0, 0.0), 0.0) is None
    assert _path_radius_m(straight[:2], (0.0, 0.0, 0.0), 0.0) is None


def test_path_radius_matches_a_known_arc() -> None:
    """A 20 m-radius arc must measure as ~20 m (step-independent)."""
    from beamng_autopilot.fsd_drive import _path_radius_m
    r = 20.0
    ang = np.linspace(0.0, math.radians(60.0), 40)
    arc = np.column_stack([r * np.sin(ang), r * (1.0 - np.cos(ang))])
    got = _path_radius_m(arc, (0.0, 0.0, 0.0), 0.0)
    assert got is not None
    assert got == pytest.approx(r, rel=0.05), got
