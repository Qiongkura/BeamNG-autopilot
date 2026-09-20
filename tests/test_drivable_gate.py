"""Offline tests for the drivable-surface gate (never drive on grass)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from beamng_autopilot.planning import Scene
from beamng_autopilot.planning.constraints import (
    Constraints,
    _path_off_drivable,
    ego_drivable_coverage,
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


class TestEgoDrivableCoverage:
    """The ego FOOTPRINT vs the road surface, with no lane line needed.

    This is the boundary-independent "is the car on the pavement?"
    question the 2026-09-20 town collision exposed: the car finished 2.3 m
    off the pavement while both detected boundaries were gone, so the
    off-road metric read 0.0 and recovery could never arm.
    """

    def test_fully_on_road(self) -> None:
        g = _Grid((0.0, 0.0), 0.0)
        # car at (5, 0): all nine footprint samples land in the road band
        assert ego_drivable_coverage(_scene(pos=(5.0, 0.0), grid=g)) == 1.0

    def test_fully_off_road_with_the_road_alongside(self) -> None:
        # car straddling the kerb line with the road immediately beside it
        g = _Grid((0.0, 0.0), 0.0)
        assert ego_drivable_coverage(_scene(pos=(5.0, 5.0), grid=g)) == 0.0

    def test_road_far_from_the_ego_is_unknown_not_off_road(self) -> None:
        """The layer marks road 10 m away and nothing near the car.

        The 2026-09-20 town runs look like this: the drivable layer holds a
        median of 12 cells (~3 m^2) whose nearest cell is ALWAYS at least
        3.75 m ahead of the ego - even on the frame the car was teleported
        onto the road - so it never overlaps the footprint (0/197 frames).
        Reading that as "the car is off the road" would fire on every
        frame.  A layer that says nothing about the car's own
        neighbourhood is UNKNOWN.
        """
        g = _Grid((0.0, 0.0), 0.0, road_band=(10.0, 13.0))
        assert ego_drivable_coverage(_scene(pos=(5.0, 0.0), grid=g)) is None

    def test_nearby_road_patch_still_reads_off_road(self) -> None:
        """A road just beside a car whose footprint misses it is a real
        off-road reading, not an unknown one."""
        g = _Grid((0.0, 0.0), 0.0, road_band=(1.0, 4.0))
        frac = ego_drivable_coverage(_scene(pos=(5.0, 0.0), grid=g))
        assert frac == 0.0

    def test_local_radius_can_be_widened(self) -> None:
        g = _Grid((0.0, 0.0), 0.0, road_band=(10.0, 13.0))
        scene = _scene(pos=(5.0, 0.0), grid=g)
        assert ego_drivable_coverage(scene) is None
        assert ego_drivable_coverage(scene, local_radius_m=12.0) == 0.0

    def test_half_on_the_kerb(self) -> None:
        # car centred on the road edge (the band is |y| <= 4): the inner
        # samples are on the road, the outer ones are not
        g = _Grid((0.0, 0.0), 0.0)
        frac = ego_drivable_coverage(_scene(pos=(5.0, 4.0), grid=g))
        assert frac is not None and 0.4 < frac < 0.8

    def test_no_grid_is_unknown_not_off_road(self) -> None:
        assert ego_drivable_coverage(_scene(grid=None)) is None

    def test_missing_drivable_layer_is_unknown_not_off_road(self) -> None:
        """A sensor miss must never read as 'the car is on grass' - the
        mountain run 2026-08-27 never started because a 10% camera
        footprint made every position read off-road."""
        g = _Grid((0.0, 0.0), 0.0)
        g.drivable[:] = 0.0
        assert ego_drivable_coverage(_scene(pos=(0.0, 12.0), grid=g)) is None

    def test_unobserved_cells_are_dropped_not_counted_as_grass(self) -> None:
        g = _Grid((0.0, 0.0), 0.0)
        g.observed = np.zeros((60, 60), np.uint8)
        g.observed[25:35, 25:35] = 1        # a small observed patch
        # car far from the observed patch: no evidence about its footprint
        assert ego_drivable_coverage(_scene(pos=(0.0, 12.0), grid=g)) is None

    def test_observed_off_road_cells_do_count(self) -> None:
        g = _Grid((0.0, 0.0), 0.0)
        g.observed = np.ones((60, 60), np.uint8)
        assert ego_drivable_coverage(_scene(pos=(5.0, 5.0), grid=g)) == 0.0

    def test_fraction_is_over_the_observed_footprint_only(self) -> None:
        """Only the front half of the footprint observed: the fraction is
        computed over what was actually seen."""
        g = _Grid((0.0, 0.0), 0.0)
        g.observed = np.zeros((60, 60), np.uint8)
        # observe the road band ahead of the car only (world y >= 4.5)
        for r in range(60):
            for c in range(60):
                wx, wy = g.cell_to_world(r, c)
                if wy >= 4.5:
                    g.observed[r, c] = 1
        frac = ego_drivable_coverage(_scene(pos=(0.0, 4.0), grid=g))
        assert frac is not None and frac < 0.6

    def test_degenerate_pos_returns_unknown(self) -> None:
        g = _Grid((0.0, 0.0), 0.0)
        scene = _scene(pos=(float("nan"), 0.0), grid=g)
        assert ego_drivable_coverage(scene) is None
