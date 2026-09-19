"""Obstacle risk model (plan phase C3): classification, TTC, speed caps."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from beamng_autopilot.obstacle_risk import (
    RISK_BRAKING,
    RISK_HARD_COLLISION,
    RISK_ROADSIDE,
    RISK_UNKNOWN,
    RISK_CONTACT_BAND_M,
    assess_obstacles,
    stop_distance_m,
    ttc_speed_cap,
)


def _track(x, y, vx=0.0, vy=0.0, matches: int = 4, lost: int = 0,
           category: str = "object", track_id: int = 1):
    return SimpleNamespace(x=float(x), y=float(y), vx=float(vx),
                           vy=float(vy), matches=int(matches),
                           lost=int(lost), category=category,
                           track_id=int(track_id))


def _straight(y: float = 0.0, n: int = 31):
    return np.column_stack([np.linspace(0.0, 30.0, n),
                            np.full(n, float(y))])


def test_no_tracks_is_neutral() -> None:
    risk = assess_obstacles([], (0.0, 0.0), 0.0, 6.0)
    assert risk.kind == RISK_UNKNOWN
    assert risk.stop is False
    assert math.isinf(risk.target_speed_cap)
    assert risk.closest_m == float("inf")


def test_roadside_object_is_not_a_brake_demand() -> None:
    """A tree/curb beside the lane is a lane bound, not a brake demand."""
    risk = assess_obstacles([_track(6.0, 3.0)], (0.0, 0.0), 0.0, 6.0)
    assert risk.kind == RISK_ROADSIDE
    assert risk.n_roadside == 1
    assert math.isinf(risk.target_speed_cap)
    assert risk.stop is False


def test_unconfirmed_corridor_return_caps_nothing() -> None:
    """A single-frame speck must not brake the car (plan C4).

    The car can still stop for an object 20 m out, so the unconfirmed
    return carries no speed cap at all.
    """
    risk = assess_obstacles(
        [_track(20.0, 0.0, matches=1)], (0.0, 0.0), 0.0, 6.0)
    assert risk.kind == RISK_UNKNOWN
    assert risk.n_unknown == 1
    assert math.isinf(risk.target_speed_cap)


def test_unconfirmed_return_inside_stopping_distance_may_brake() -> None:
    """...unless the car can no longer stop in the remaining distance.

    "车辆已经无法在剩余距离内刹停" is one of the plan's explicit
    no-wait cases: at 6 m and 6 m/s the stopping distance (7.2 m) already
    exceeds the gap, so it is too late to insist on confirmation.
    """
    risk = assess_obstacles(
        [_track(6.0, 0.0, matches=1)], (0.0, 0.0), 0.0, 6.0)
    assert risk.kind == RISK_BRAKING
    assert risk.target_speed_cap < 6.0


def test_unconfirmed_return_inside_contact_band_still_stops() -> None:
    """...but inside the contact band there is NO confirmation wait."""
    risk = assess_obstacles(
        [_track(RISK_CONTACT_BAND_M - 0.5, 0.0, matches=1)],
        (0.0, 0.0), 0.0, 6.0)
    assert risk.kind == RISK_HARD_COLLISION
    assert risk.stop is True
    assert risk.target_speed_cap == 0.0


def test_closing_vehicle_brakes_on_ttc() -> None:
    """A vehicle closing head-on caps the speed BELOW the current one."""
    # 12 m away, closing at 12 m/s (6 ego + 6 oncoming) -> TTC 1.0 s
    risk = assess_obstacles([_track(12.0, 0.0, vx=-6.0)],
                            (0.0, 0.0), 0.0, 6.0)
    assert risk.kind == RISK_BRAKING
    assert risk.min_ttc_s == pytest.approx(1.0, abs=0.05)
    # cap must force a real slowdown, and equal the stopping bound
    assert risk.target_speed_cap < 6.0
    assert risk.target_speed_cap == pytest.approx(
        ttc_speed_cap(12.0, 12.0), abs=1e-6)
    assert risk.stop is False


def test_receding_vehicle_does_not_cap() -> None:
    """A vehicle driving away faster than the ego is not closing."""
    risk = assess_obstacles([_track(10.0, 0.0, vx=8.0)],
                            (0.0, 0.0), 0.0, 6.0)
    assert risk.kind == RISK_BRAKING          # confirmed, in corridor
    assert risk.min_ttc_s is None
    # the stopping-distance bound at 10 m does not bind at 6 m/s
    assert risk.target_speed_cap > 6.0


def test_crossing_vehicle_outside_corridor_but_intruding_brakes() -> None:
    """Roadside-positioned but predicted to enter the corridor -> brake."""
    # 5 m right of the corridor centre, moving left at 3 m/s
    risk = assess_obstacles([_track(10.0, -5.0, vy=3.0)],
                            (0.0, 0.0), 0.0, 6.0)
    assert risk.items[0].predicted_intrusion is True
    assert risk.items[0].in_corridor is False
    assert risk.kind == RISK_BRAKING


def test_static_obstacle_ttc_comes_from_the_ego_motion() -> None:
    """A parked obstacle has a TTC - against the EGO's own approach.

    The object velocity is zero, so its closing speed equals the ego's
    along-path speed; nothing may be invented from the object itself.
    The swept-envelope / occupancy gates stay the authority for the
    static hard-collision case.
    """
    risk = assess_obstacles([_track(8.0, 0.0)], (0.0, 0.0), 0.0, 8.0)
    assert risk.items[0].ttc_s == pytest.approx(1.0, abs=0.05)
    assert risk.items[0].closing_speed_mps == pytest.approx(8.0, abs=1e-6)
    cap = risk.target_speed_cap
    assert cap == pytest.approx(ttc_speed_cap(8.0, 8.0), abs=1e-6)
    assert cap < 8.0


def test_stop_distance_is_the_familiar_v_squared_bound() -> None:
    assert stop_distance_m(6.0) == pytest.approx(7.2, abs=1e-6)
    assert stop_distance_m(0.0) == 0.0


def test_cap_is_monotonic_in_gap() -> None:
    near = assess_obstacles([_track(6.0, 0.0)], (0.0, 0.0), 0.0, 6.0)
    far = assess_obstacles([_track(20.0, 0.0)], (0.0, 0.0), 0.0, 6.0)
    assert near.target_speed_cap < far.target_speed_cap


def test_obstacle_behind_the_ego_is_ignored() -> None:
    risk = assess_obstacles([_track(-4.0, 0.0)], (0.0, 0.0), 0.0, 6.0)
    assert risk.items == []
    assert math.isinf(risk.target_speed_cap)


def test_path_corridor_follows_the_driven_trajectory() -> None:
    """Distance/lateral are measured against the path, not the heading.

    A lane-shifted path puts the corridor where the car will actually
    drive: an object beside the NEW path is roadside even though it sits
    inside the straight-ahead heading corridor.
    """
    path = _straight(y=-3.0)
    risk = assess_obstacles([_track(10.0, 0.0)],
                            (0.0, 0.0), 0.0, 6.0, path=path)
    assert risk.items[0].in_corridor is False
    assert risk.kind == RISK_ROADSIDE
    # the same object IS in corridor when the car drives straight at it
    risk2 = assess_obstacles([_track(10.0, 0.0)], (0.0, 0.0), 0.0, 6.0)
    assert risk2.items[0].in_corridor is True


def test_digest_is_json_safe() -> None:
    risk = assess_obstacles([_track(12.0, 0.0, vx=-6.0)],
                            (0.0, 0.0), 0.0, 6.0)
    import json
    text = json.dumps(risk.digest())
    assert "braking_obstacle" in text
