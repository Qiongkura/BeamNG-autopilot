# -*- coding: utf-8 -*-
"""Offline regression for path clearance helpers (raw-hit + grid)."""
from __future__ import annotations

import numpy as np

from beamng_autopilot.occupancy import OccupancyGrid
from beamng_autopilot.planner import (
    path_forward_clearance_m,
    path_grid_clearance_m,
)


def _grid_with_obstacle(ox: float, oy: float) -> OccupancyGrid:
    """60x60 grid, res 0.5, ego at world (30,30) heading 0."""
    g = OccupancyGrid(60, 60, 0.5, origin=(30.0, 30.0), heading=0.0)
    # occupy a cell at world (ox, oy): world (x,y) -> ego (x-30, y-30)
    # grid row 0 is forward (+x), col 0 is left (+y); row = (extent-ex)/res
    # but simplest: mark via world_to_cell
    cell = g.world_to_cell(ox, oy)
    assert cell is not None
    g.obstacle[cell] = 1
    g.occupancy[cell] = 1.0
    return g


class TestPathGridClearance:
    def test_empty_grid_infinite(self):
        g = OccupancyGrid(60, 60, 0.5, origin=(30.0, 30.0), heading=0.0)
        path = np.array([[30, 30], [40, 30]], dtype=float)
        assert np.isinf(path_grid_clearance_m(path, g))

    def test_wall_ahead_returns_distance(self):
        # obstacle at world (37, 30), path along +x from (30,30)
        g = _grid_with_obstacle(37.0, 30.0)
        path = np.array([[30, 30], [45, 30]], dtype=float)
        c = path_grid_clearance_m(path, g)
        assert 6.0 < c < 8.0, c

    def test_wall_off_path_ignored(self):
        g = _grid_with_obstacle(37.0, 34.0)  # 4 m left of straight path
        path = np.array([[30, 30], [45, 30]], dtype=float)
        c = path_grid_clearance_m(path, g)
        assert np.isinf(c)

    def test_none_grid_infinite(self):
        path = np.array([[0, 0], [5, 0]], dtype=float)
        assert np.isinf(path_grid_clearance_m(path, None))

    def test_none_path_infinite(self):
        g = OccupancyGrid(60, 60, 0.5, origin=(0.0, 0.0), heading=0.0)
        assert np.isinf(path_grid_clearance_m(None, g))

    def test_path_through_empty_corridor_of_occupied_sides(self):
        # occupied cells 1.5 m either side of the straight path
        g = OccupancyGrid(60, 60, 0.5, origin=(30.0, 30.0), heading=0.0)
        for y in (32.2, 27.8):
            cell = g.world_to_cell(35.0, y)
            g.obstacle[cell] = 1
            g.occupancy[cell] = 1.0
        path = np.array([[30, 30], [40, 30]], dtype=float)
        c = path_grid_clearance_m(path, g)
        # corridor (half_width 1.6) only samples the path centreline, so
        # side cells 2.2 m away do not block
        assert np.isinf(c)


class TestPathRawClearance:
    def test_wall_ahead_returns_distance(self):
        path = np.array([[0, 0], [10, 0]], dtype=float)
        hits = [(5.0, 0.0), (6.0, 0.3)]
        c = path_forward_clearance_m(path, hits, half_width=1.5)
        assert np.isclose(c, 5.0)

    def test_turn_away_from_nose_wall(self):
        # path turns right immediately; a wall straight ahead is outside
        # the swept corridor of the path
        path = np.array([[0, 0], [2, 3], [5, 5]], dtype=float)
        hits = [(4.0, 0.0)]
        assert np.isinf(path_forward_clearance_m(path, hits, half_width=1.5))

    def test_hit_behind_ignored(self):
        path = np.array([[0, 0], [10, 0]], dtype=float)
        assert np.isinf(path_forward_clearance_m(path, [(-3.0, 0.0)],
                                                 half_width=1.5))
