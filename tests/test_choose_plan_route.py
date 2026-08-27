"""Additional tests for choose_plan_route (route vs sensor lane)."""
from __future__ import annotations

import numpy as np
import pytest

from beamng_autopilot.occupancy import OccupancyGrid
from beamng_autopilot.planning import choose_plan_route


def _grid(obstacle_world):
    g = OccupancyGrid(60, 60, 0.5, origin=(0.0, 0.0), heading=0.0)
    for x, y in obstacle_world:
        g.add_obstacle_point(float(x), float(y), weight=1.0)
    g.obstacle[:] = (g.occupancy >= 0.4).astype(np.uint8)
    return g


def test_route_kept_when_free() -> None:
    route = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
    lane = np.array([[0.0, 0.0], [5.0, 2.0], [10.0, 4.0]])
    g = _grid([(30.0, 30.0)])   # far away, not in window
    out = choose_plan_route(route, lane, np.array([0.0, 0.0]), 0.0, g)
    np.testing.assert_allclose(out, route)


def test_sensor_lane_wins_when_route_blocked() -> None:
    # route goes straight into a wall; the ego lane turns right (own lane
    # on the RIGHT of the road centreline, right-hand traffic) and clear
    route = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
    lane = np.array([[0.0, 0.0], [5.0, -3.0], [10.0, -6.0]])
    # wall along x=0..8, y=-1..1 so route points (5,0),(10,0) blocked
    wall = [(x, 0.0) for x in np.linspace(3.0, 8.0, 12)]
    wall += [(x, 0.5) for x in np.linspace(3.0, 8.0, 6)]
    wall += [(x, -0.5) for x in np.linspace(3.0, 8.0, 6)]
    g = _grid(wall)
    out = choose_plan_route(route, lane, np.array([0.0, 0.0]), 0.0, g)
    # lane goes -y (the ego side, right-hand traffic) so its samples are
    # clear of the wall; expect lane chosen
    np.testing.assert_allclose(out, lane)


def test_sensor_lane_on_oncoming_side_rejected() -> None:
    """A sensor lane LEFT of the road centreline is the oncoming lane:
    never follow it, even when the route corridor is blocked (the car
    must never cross into oncoming traffic - town runs 2026-08-22)."""
    route = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
    lane = np.array([[0.0, 0.0], [5.0, 3.0], [10.0, 6.0]])  # LEFT side
    wall = [(x, 0.0) for x in np.linspace(3.0, 8.0, 12)]
    wall += [(x, 0.5) for x in np.linspace(3.0, 8.0, 6)]
    wall += [(x, -0.5) for x in np.linspace(3.0, 8.0, 6)]
    g = _grid(wall)
    out = choose_plan_route(route, lane, np.array([0.0, 0.0]), 0.0, g)
    np.testing.assert_allclose(out, route)


def test_route_kept_when_lane_also_blocked() -> None:
    route = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
    lane = np.array([[0.0, 0.0], [5.0, -3.0], [10.0, -6.0]])
    wall = [(x, 0.0) for x in np.linspace(3.0, 8.0, 12)]
    wall += [(x, -3.0) for x in np.linspace(3.0, 8.0, 12)]
    wall += [(x, -6.0) for x in np.linspace(3.0, 8.0, 12)]
    g = _grid(wall)
    out = choose_plan_route(route, lane, np.array([0.0, 0.0]), 0.0, g)
    np.testing.assert_allclose(out, route)


def test_no_lane_returns_route() -> None:
    route = np.array([[0.0, 0.0], [5.0, 0.0]])
    g = _grid([])
    out = choose_plan_route(route, None, np.array([0.0, 0.0]), 0.0, g)
    np.testing.assert_allclose(out, route)


def test_no_route_returns_lane() -> None:
    lane = np.array([[0.0, 0.0], [5.0, 2.0]])
    g = _grid([])
    out = choose_plan_route(None, lane, np.array([0.0, 0.0]), 0.0, g)
    np.testing.assert_allclose(out, lane)
