"""Offline tests for the trajectory speed-profile planner."""

from __future__ import annotations

import numpy as np
import pytest

from beamng_autopilot.occupancy import OccupancyGrid
from beamng_autopilot.planning import (
    Scene,
    select_trajectory,
    speed_profile_for_path,
)
from beamng_autopilot.planning.trajectory import CandidateSet


def _scene(with_obstacle=False):
    grid = OccupancyGrid(60, 60, 0.5)
    grid.origin = (0.0, 0.0)
    grid.heading = 0.0
    if with_obstacle:
        grid.mark_obstacle_region(8.0, 0.0, 1.2, 1.2)
    xs = np.linspace(0, 30, 31)
    route = np.column_stack([xs, np.zeros_like(xs)])
    return Scene(pos=np.array([0.0, 0.0]), heading=0.0, grid=grid,
                 route=route, lane_ref=route, target_speed=12.0)


def test_straight_path_cruises_target() -> None:
    sc = _scene()
    path = np.column_stack([np.linspace(0, 20, 40), np.zeros(40)])
    sp = speed_profile_for_path(path, sc, target_speed=12.0)
    assert sp.shape == (40,)
    # a straight path should cruise near the target throughout
    assert float(sp.max()) > 10.0
    assert float(np.median(sp)) > 9.0


def test_curved_path_slows_in_bend() -> None:
    sc = _scene()
    # a tight S-curve (large curvature)
    t = np.linspace(0, 4 * np.pi, 60)
    path = np.column_stack([t, np.sin(t) * 2.0])
    sp = speed_profile_for_path(path, sc, target_speed=12.0,
                                comfort_lat=2.0)
    # the bend region must be slower than the cruise target
    assert float(sp.min()) < 8.0


def test_roadside_wall_does_not_pin_speed() -> None:
    """A wall lining the road but OUTSIDE the driven corridor is not an
    obstacle ahead: the profile must cruise, not brake to 1 m/s (town
    runs 2026-08-22: every path in a wall-lined street got a 1.0 m/s
    profile because the closest occupied cell was a roadside wall)."""
    sc = _scene()
    # walls 3.5 m either side of the straight path (corridor half-width 2.0)
    for y in (-3.5, 3.5):
        sc.grid.mark_obstacle_region(6.0, y, 1.2, 1.2)
    path = np.column_stack([np.linspace(0, 20, 40), np.zeros(40)])
    sp = speed_profile_for_path(path, sc, target_speed=12.0,
                                obstacle_brake_m=25.0)
    assert float(np.median(sp)) > 9.0, sp


def test_obstacle_inside_corridor_still_slows() -> None:
    """An obstacle that intrudes INTO the path corridor must still trigger
    the brake band (regression guard for the corridor filter)."""
    sc = _scene()
    sc.grid.mark_obstacle_region(8.0, 1.0, 1.2, 1.2)  # inside corridor
    path = np.column_stack([np.linspace(0, 20, 40), np.zeros(40)])
    sp = speed_profile_for_path(path, sc, target_speed=12.0,
                                obstacle_brake_m=25.0)
    assert float(sp.min()) < 8.0, sp


def test_obstacle_brake_band_slows() -> None:
    sc = _scene(with_obstacle=True)
    path = np.column_stack([np.linspace(0, 20, 40), np.zeros(40)])
    sp = speed_profile_for_path(path, sc, target_speed=12.0,
                                obstacle_brake_m=25.0)
    # near the obstacle (at x=8) the profile is much slower than the
    # cruise target; far behind the obstacle it recovers toward target
    assert float(sp.min()) < 6.0                       # brake before box
    assert float(sp[-1]) > float(sp.min())             # recovery after


def test_selector_attaches_speed_profile() -> None:
    sc = _scene()
    from beamng_autopilot.planning import sample_arc
    cs = sample_arc([0.0, 0.0], 0.0, speed=8.0, max_steer=0.3, n_curv=5)
    from beamng_autopilot.planning import Constraints
    cons = Constraints(w_collision=5.0, w_curvature=0.5, w_lane_align=1.0)
    best, meta = select_trajectory(sc, cs, cons)
    assert best is not None
    assert "speed_profile" in meta
    sp = meta["speed_profile"]
    assert sp.shape == (best.shape[0],)


def test_speed_at_idx_clamps() -> None:
    from beamng_autopilot.planning.trajectory import Candidate
    c = Candidate(path=np.zeros((5, 2)), speed_profile=np.array([3.0]))
    assert c.speed_at_idx(0) == pytest.approx(3.0)
    assert c.speed_at_idx(99) == pytest.approx(3.0)  # clamped
    empty = Candidate(path=np.zeros((5, 2)))
    assert empty.speed_at_idx(0) == 0.0

def test_obstacle_min_speed_floor_when_open() -> None:
    """With the forward corridor verified open, dense junction/end-zone
    LiDAR inside the brake band must not pin the plan to the 1 m/s
    MIN_SPEED crawl (fsd opt22: plan snapped 6.0 <-> 1.0 and the car
    brake->stalled at the junction).  A caller with corridor knowledge
    passes obstacle_min_speed = 0.4*target; the profile then eases to
    that floor instead of MIN_SPEED."""
    sc = _scene()
    sc.grid.mark_obstacle_region(8.0, 0.0, 1.2, 1.2)  # inside corridor
    path = np.column_stack([np.linspace(0, 20, 40), np.zeros(40)])
    sp_low = speed_profile_for_path(path, sc, target_speed=12.0,
                                    obstacle_brake_m=25.0)
    sp_open = speed_profile_for_path(path, sc, target_speed=12.0,
                                     obstacle_brake_m=25.0,
                                     obstacle_min_speed=4.8)
    assert float(sp_low.min()) <= 1.2, sp_low
    assert float(sp_open.min()) >= 4.0, sp_open
