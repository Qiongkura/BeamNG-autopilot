"""Offline tests for the drivable-surface gate (never drive on grass)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from beamng_autopilot.planning import Scene
from beamng_autopilot.planning.constraints import (
    Constraints,
    _path_off_drivable,
    lane_cross_dist_m,
)


class _Grid:
    """Minimal ego grid: rows = ahead, cols = right (+)."""

    n_rows = n_cols = 60
    res = 0.5
    extent = 15.0

    def __init__(self, pos, heading, road_band=(-4.0, 4.0)):
        self.pos = np.asarray(pos, dtype=float)
        self.heading = float(heading)
        self.occupancy = np.zeros((60, 60), np.float32)
        self.obstacle = np.zeros((60, 60), np.uint8)
        self.drivable = np.zeros((60, 60), np.float32)
        for r in range(60):
            for c in range(60):
                wx, wy = self.cell_to_world(r, c)
                # road along +x through the ego, lane band around y=0
                if road_band[0] <= wy <= road_band[1] and 0.0 <= wx <= 30.0:
                    self.drivable[r, c] = 1.0

    def cell_to_world(self, r, c):
        ch, sh = math.cos(self.heading), math.sin(self.heading)
        ex = self.extent - (r + 0.5) * self.res
        ey = self.extent - (c + 0.5) * self.res
        return (self.pos[0] + ex * ch - ey * sh,
                self.pos[1] + ex * sh + ey * ch)

    def world_to_cell(self, x, y):
        ch, sh = math.cos(self.heading), math.sin(self.heading)
        ex = (x - self.pos[0]) * ch + (y - self.pos[1]) * sh
        ey = -(x - self.pos[0]) * sh + (y - self.pos[1]) * ch
        r = int((self.extent - ex) / self.res)
        c = int((self.extent - ey) / self.res)
        if 0 <= r < 60 and 0 <= c < 60:
            return (r, c)
        return None


class _Cand:
    def __init__(self, path, kind="arc"):
        self.path = np.asarray(path, dtype=float)
        self.meta = {"kind": kind, "offset": 0.0}
        self.speed_profile = None


def _scene(pos=(0.0, 0.0), heading=0.0, grid=None, route=None):
    return Scene(pos=pos, heading=heading, grid=grid, route=route,
                 lane_ref=None, lane_left=None, lane_right=None,
                 lane_width=3.5, target_speed=6.0)


def test_off_drivable_counts_grass() -> None:
    g = _Grid((0.0, 0.0), 0.0)
    path = np.array([[0.0, 0.0], [4.0, 0.0], [5.0, 6.0], [9.0, 6.0]])
    bad, tot, nb, nt = _path_off_drivable(_scene(grid=g), path)
    assert tot > 0
    assert bad > 0
    assert nb > 0


def test_path_through_grass_rejected() -> None:
    g = _Grid((0.0, 0.0), 0.0)
    cons = Constraints()
    path = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0],
                     [12.0, 5.0], [16.0, 8.0]])
    cost, ok = cons.score(_scene(grid=g, route=path), _Cand(path))
    assert not ok


def test_path_on_road_kept() -> None:
    g = _Grid((0.0, 0.0), 0.0)
    cons = Constraints()
    path = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0], [15.0, 0.0]])
    cost, ok = cons.score(_scene(grid=g, route=path), _Cand(path))
    assert ok


def test_gate_inactive_without_drivable_evidence() -> None:
    """A missing road mask is 'unknown', not grass - the gate must not
    park the car on a sensor miss."""
    g = _Grid((0.0, 0.0), 0.0)
    g.drivable[:] = 0.0
    cons = Constraints()
    path = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0], [15.0, 8.0]])
    cost, ok = cons.score(_scene(grid=g, route=path), _Cand(path))
    assert ok


def test_near_zone_grass_rejected() -> None:
    """Even a path that leaves drivable only for a few metres near the
    car is rejected - no brief off-road excursion."""
    g = _Grid((0.0, 0.0), 0.0)
    cons = Constraints()
    path = np.array([[0.0, 0.0], [3.0, 0.0], [5.0, 6.0], [9.0, 6.0],
                     [14.0, 0.0], [18.0, 0.0]])
    cost, ok = cons.score(_scene(grid=g, route=path), _Cand(path))
    assert not ok
