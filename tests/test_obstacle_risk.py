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
    RISK_STOP_MARGIN_M,
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


def test_contact_band_stop_needs_corroboration_when_occupancy_says_free() -> None:
    """Two sources disagreeing at contact distance must not park the car.

    Measured 2026-09-21 (town, strict): 72-82 LiDAR + 17-20 raycast terrain
    returns, one unconfirmed speck inside the contact band, the fused
    occupancy showing 13.5 m of clear path - 37 of 55 frames stopped on it
    and the car could never move, so the speck never left the band.
    """
    from beamng_autopilot.occupancy import OccupancyGrid

    grid = OccupancyGrid(60, 60, 0.5)
    grid.observed[:] = 1
    # The grid must carry real evidence elsewhere (roadside vegetation, the
    # road edge) for its "free here" to mean anything - an all-zero grid is
    # no evidence at all and must not contradict a contact return.
    grid.mark_obstacle_region(6.0, 5.0, 2.0, 1.0)
    risk = assess_obstacles(
        [_track(RISK_CONTACT_BAND_M - 0.5, 0.0, matches=1)],
        (0.0, 0.0), 0.0, 6.0, grid=grid)
    assert risk.stop is False
    assert risk.kind != RISK_HARD_COLLISION
    # Creep toward it, bounded by the stop margin - never a full stop, and
    # never a speed that could reach it.
    assert 0.0 < risk.target_speed_cap <= 2.0


def test_contact_band_stop_survives_when_the_grid_agrees() -> None:
    from beamng_autopilot.occupancy import OccupancyGrid

    grid = OccupancyGrid(60, 60, 0.5)
    grid.observed[:] = 1
    grid.obstacle[:, 30] = 1                   # occupied where the track is
    grid.mark_obstacle_region(2.5, 0.0, 0.25, 0.25)
    risk = assess_obstacles(
        [_track(RISK_CONTACT_BAND_M - 0.5, 0.0, matches=1)],
        (0.0, 0.0), 0.0, 6.0, grid=grid)
    assert risk.kind == RISK_HARD_COLLISION
    assert risk.stop is True


def test_a_confirmed_static_object_is_still_stopped_when_the_grid_agrees() -> None:
    from beamng_autopilot.occupancy import OccupancyGrid

    grid = OccupancyGrid(60, 60, 0.5)
    grid.observed[:] = 1
    grid.mark_obstacle_region(6.0, 5.0, 2.0, 1.0)      # grid carries evidence
    grid.mark_obstacle_region(2.5, 0.0, 0.4, 0.4)       # ...and agrees here
    risk = assess_obstacles(
        [_track(RISK_CONTACT_BAND_M - 0.5, 0.0, matches=4)],
        (0.0, 0.0), 0.0, 6.0, grid=grid)
    assert risk.kind == RISK_HARD_COLLISION
    assert risk.stop is True


def test_a_moving_object_in_the_contact_band_is_never_contradicted() -> None:
    """The track layer's unique value over a raster is motion: the raster
    lags a vehicle, so a moving object stops the car even where the fused
    occupancy is still empty."""
    from beamng_autopilot.occupancy import OccupancyGrid

    grid = OccupancyGrid(60, 60, 0.5)
    grid.observed[:] = 1
    grid.mark_obstacle_region(6.0, 5.0, 2.0, 1.0)
    risk = assess_obstacles(
        [_track(RISK_CONTACT_BAND_M - 0.5, 0.0, vx=-4.0, matches=2)],
        (0.0, 0.0), 0.0, 6.0, grid=grid)
    assert risk.kind == RISK_HARD_COLLISION
    assert risk.stop is True


def test_a_static_object_the_ego_approaches_is_not_excused_by_closing() -> None:
    """A static wall closes at the EGO speed; that must not count as the
    object moving, or no static obstacle could ever be graded."""
    from beamng_autopilot.occupancy import OccupancyGrid

    grid = OccupancyGrid(60, 60, 0.5)
    grid.observed[:] = 1
    grid.mark_obstacle_region(6.0, 5.0, 2.0, 1.0)
    grid.mark_obstacle_region(2.5, 0.0, 0.4, 0.4)     # raster agrees here
    risk = assess_obstacles(
        [_track(RISK_CONTACT_BAND_M - 0.5, 0.0, matches=4)],
        (0.0, 0.0), 0.0, 6.0, grid=grid)
    assert risk.kind == RISK_HARD_COLLISION
    assert risk.stop is True


def test_a_static_object_the_grid_contradicts_creeps_instead_of_stopping() -> None:
    """A persistent return the fused raster calls free is terrain, not a
    wall: the two sources disagree and the raster is the filtered one.  The
    creep cap still keeps the stop margin, so a real object the raster
    missed halts the car ~2 m short instead of hitting it."""
    from beamng_autopilot.occupancy import OccupancyGrid

    grid = OccupancyGrid(60, 60, 0.5)
    grid.observed[:] = 1
    grid.mark_obstacle_region(6.0, 5.0, 2.0, 1.0)
    risk = assess_obstacles(
        [_track(RISK_CONTACT_BAND_M - 0.5, 0.0, matches=4)],
        (0.0, 0.0), 0.0, 6.0, grid=grid)
    assert risk.stop is False
    assert 0.0 < risk.target_speed_cap <= 2.0
    # The cap falls with the gap and reaches zero at the margin: the car
    # halts short of it even if it is real and the raster missed it.
    close = assess_obstacles(
        [_track(RISK_STOP_MARGIN_M + 0.2, 0.0, matches=4)],
        (0.0, 0.0), 0.0, 6.0, grid=grid)
    assert close.target_speed_cap == pytest.approx(
        ttc_speed_cap(RISK_STOP_MARGIN_M + 0.2, 0.0), abs=0.05)
    at_margin = assess_obstacles(
        [_track(RISK_STOP_MARGIN_M - 0.1, 0.0, matches=4)],
        (0.0, 0.0), 0.0, 6.0, grid=grid)
    assert at_margin.target_speed_cap == pytest.approx(0.0)


def test_a_detection_inside_the_ego_footprint_is_not_an_obstacle() -> None:
    """Own body / ground under the car: 0.02 m "collisions" are not real."""
    risk = assess_obstacles([_track(0.05, 0.1, matches=6)], (0.0, 0.0), 0.0, 0.0)
    assert risk.stop is False
    assert risk.kind == RISK_UNKNOWN
    assert risk.items == []


def test_a_real_object_just_ahead_is_still_graded() -> None:
    risk = assess_obstacles([_track(2.0, 0.0, matches=6)], (0.0, 0.0), 0.0, 0.0)
    assert risk.kind == RISK_HARD_COLLISION
    assert risk.stop is True


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


def test_roadside_object_in_the_contact_band_does_not_stop_the_car() -> None:
    """The channel-wall failure: distance alone is not danger (live 2026-09-20).

    A town street has trees, kerbs and parked cars within 3 m of the car
    all the time.  Ordering the contact band BEFORE the corridor gate made
    every one of them an immediate collision: the baseline stopped on
    144/144 frames with the lane paired and 16 m of clear road ahead.
    """
    beside = _track(2.0, 4.0)          # 2 m ahead, 4 m to the side
    risk = assess_obstacles([beside], (0.0, 0.0), 0.0, 3.0)
    assert risk.kind == RISK_ROADSIDE
    assert risk.stop is False
    assert math.isinf(risk.target_speed_cap)


def test_in_corridor_object_in_the_contact_band_still_stops() -> None:
    """...while something genuinely in the driven corridor still does."""
    ahead = _track(2.0, 0.4)           # 2 m ahead, inside the 1.6 m corridor
    risk = assess_obstacles([ahead], (0.0, 0.0), 0.0, 3.0)
    assert risk.kind == RISK_HARD_COLLISION
    assert risk.stop is True
    assert risk.target_speed_cap == 0.0


def test_roadside_object_predicted_to_intrude_still_stops() -> None:
    """A crossing vehicle is not "roadside", however far out it starts."""
    crossing = _track(2.5, -3.0, vy=6.0)   # 3 m right, moving left fast
    risk = assess_obstacles([crossing], (0.0, 0.0), 0.0, 3.0)
    assert risk.stop is True
