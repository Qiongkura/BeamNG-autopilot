"""Offline tests for the motion-prediction layer.

The stack tracks objects with a smoothed velocity that nothing consumed, so a
moving obstacle was planned against as if it stood still.  These pin the
prediction contract: constant-velocity geometry, the optional turn-rate arc,
the bounds that stop tracker jitter from becoming a phantom moving object, and
the corridor query a planner would use.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from beamng_autopilot.prediction import (
    PREDICT_HORIZON_S,
    PREDICT_MAX_HORIZON_S,
    PREDICT_MAX_SPEED_MPS,
    predict_track,
    predict_tracks,
    predicted_points_at,
    nearest_predicted_gap_m,
    prediction_digest,
)
from beamng_autopilot.temporal import TrackedObject


def test_constant_velocity_straight_line() -> None:
    tr = TrackedObject(track_id=1, x=0.0, y=0.0, vx=2.0, vy=0.0)
    p = predict_track(tr, horizon_s=3.0, dt=1.0)
    assert p is not None and p.moving
    assert p.points.shape == (4, 2)
    np.testing.assert_allclose(p.t, [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(p.points[:, 0], [0.0, 2.0, 4.0, 6.0])
    np.testing.assert_allclose(p.points[:, 1], 0.0)
    assert p.final == pytest.approx((6.0, 0.0))
    assert p.speed_mps == pytest.approx(2.0)


def test_stationary_track_stays_put() -> None:
    """Sub-threshold motion is stationary, not a slow phantom vehicle."""
    tr = TrackedObject(track_id=2, x=5.0, y=-1.0, vx=0.05, vy=0.01)
    p = predict_track(tr, horizon_s=3.0, dt=1.0)
    assert p is not None and not p.moving
    np.testing.assert_allclose(p.points, np.tile([5.0, -1.0], (4, 1)))


def test_speed_is_clamped() -> None:
    tr = TrackedObject(track_id=3, x=0.0, y=0.0, vx=500.0, vy=0.0)
    p = predict_track(tr, horizon_s=1.0, dt=1.0)
    assert p is not None
    assert p.speed_mps == pytest.approx(PREDICT_MAX_SPEED_MPS)
    assert p.final[0] == pytest.approx(PREDICT_MAX_SPEED_MPS)


def test_horizon_is_capped() -> None:
    tr = TrackedObject(track_id=4, x=0.0, y=0.0, vx=1.0, vy=0.0)
    p = predict_track(tr, horizon_s=1e6, dt=1.0)
    assert p is not None
    assert p.t[-1] <= PREDICT_MAX_HORIZON_S + 1e-9


def test_stale_track_is_not_predicted() -> None:
    tr = TrackedObject(track_id=5, x=0.0, y=0.0, vx=3.0, vy=0.0, lost=9)
    assert predict_track(tr) is None
    assert predict_tracks([tr]) == []


def test_turn_rate_arc_keeps_radius_and_origin() -> None:
    """Constant turn rate describes a circle of r = v / |w| about a centre."""
    v, w = 4.0, 0.5
    tr = TrackedObject(track_id=6, x=0.0, y=0.0, vx=v, vy=0.0)
    p = predict_track(tr, horizon_s=6.0, dt=0.5, yaw_rate_rad_s=w)
    assert p is not None and p.moving
    r = v / abs(w)
    start = np.array([0.0, 0.0])
    # every predicted point sits on the circle through the start point
    d = np.linalg.norm(p.points - start, axis=1)
    assert d[-1] > 1.0                      # it actually travelled
    # equal-time samples are equally spaced along the arc, and consecutive
    # samples are separated by the CHORD (not the arc length):
    #   chord = 2 r sin(w dt / 2)
    steps = np.linalg.norm(np.diff(p.points, axis=0), axis=1)
    np.testing.assert_allclose(steps, steps[0], rtol=1e-6)
    assert steps[0] == pytest.approx(2.0 * r * math.sin(w * 0.5 / 2.0),
                                     rel=1e-6)
    assert math.isfinite(r)


def test_predicted_points_at_samples_the_horizon() -> None:
    tr = TrackedObject(track_id=7, x=0.0, y=0.0, vx=1.0, vy=0.0)
    preds = predict_tracks([tr], horizon_s=4.0, dt=1.0)
    at2 = predicted_points_at(preds, 2.0)
    np.testing.assert_allclose(at2, [[2.0, 0.0]])
    assert predicted_points_at([], 1.0).shape == (0, 2)


def test_corridor_gap_sees_a_crossing_vehicle_before_it_arrives() -> None:
    """The whole point: an object that is outside the corridor NOW but will
    enter it must be visible to the corridor query."""
    # car crossing left-to-right 10 m ahead of the ego, currently 6 m left
    ego = np.array([0.0, 0.0])
    tr = TrackedObject(track_id=8, x=10.0, y=6.0, vx=0.0, vy=-2.0)
    preds = predict_tracks([tr], horizon_s=4.0, dt=0.5)
    # now: 6 m left -> outside a 1.6 m corridor
    now = predicted_points_at(preds, 0.0)[0]
    assert abs(now[1]) > 1.6
    # predicted: enters the corridor within the horizon
    gap = nearest_predicted_gap_m(preds, ego, 0.0, corridor_half_m=1.6,
                                 horizon_s=4.0)
    assert gap is not None and gap == pytest.approx(10.0, abs=0.6)


def test_corridor_gap_none_when_nothing_enters() -> None:
    tr = TrackedObject(track_id=9, x=20.0, y=30.0, vx=0.0, vy=0.0)
    preds = predict_tracks([tr])
    assert nearest_predicted_gap_m(preds, np.zeros(2), 0.0,
                                   corridor_half_m=1.6) is None
    assert nearest_predicted_gap_m([], np.zeros(2), 0.0,
                                   corridor_half_m=1.6) is None


def test_digest_is_json_safe_and_counts_motion() -> None:
    still = TrackedObject(track_id=10, x=0.0, y=0.0)
    mov = TrackedObject(track_id=11, x=0.0, y=0.0, vx=3.0, vy=0.0)
    d = prediction_digest(predict_tracks([still, mov], horizon_s=2.0,
                                         dt=1.0))
    import json
    json.dumps(d)                       # must serialise
    assert d["n"] == 2 and d["n_moving"] == 1
    assert d["max_speed_mps"] == pytest.approx(3.0)
    assert d["max_travel_m"] == pytest.approx(6.0, abs=0.5)


def test_default_horizon_is_bounded_and_sampled() -> None:
    tr = TrackedObject(track_id=12, x=0.0, y=0.0, vx=1.0, vy=0.0)
    p = predict_track(tr)
    assert p is not None
    assert p.t[-1] == pytest.approx(PREDICT_HORIZON_S)
    assert len(p.t) == int(PREDICT_HORIZON_S / 0.5) + 1
