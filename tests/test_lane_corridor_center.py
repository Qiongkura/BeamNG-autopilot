"""Corridor lane candidate: geometry of a REFUTED idea, pinned by tests.

**Status: refuted live** (2026-09-12, town ``town_1789142315``): the free
corridor's right edge is the ROAD's right edge — on town it spills ~1.9 m
past the painted lane line, so the candidate pointed 1.24 m (p50) off the
lane centre on 52.7% of frames.  A whole-corridor width gate cannot
distinguish "two lanes" from "two lanes + shoulder", so the assumption is
unverifiable at runtime and the candidate must not be enabled.

These tests stay because they pin (a) the width-gate contract that any
future redesign must keep, (b) the strict-mode fail-closed behaviour
around ``corridor_fallback`` (default OFF), and (c) the source plumbing.
"""

from __future__ import annotations

import numpy as np
import pytest

from beamng_autopilot.fsd_realism import SRC_CORRIDOR, SRC_UNAVAILABLE
from beamng_autopilot.lane.reference import (
    CORRIDOR_LANE_MAX_WIDTH_M,
    bev_corridor_lane_center,
    select_lane_reference,
)
from beamng_autopilot.occupancy import OccupancyGrid

RES = 0.5
N = 60
POS = np.array([0.0, 0.0, 0.0])
HEADING = 0.0


def _grid_with_road(right_m: float, left_m: float) -> OccupancyGrid:
    """Grid whose drivable band spans [right_m, left_m] laterally."""
    g = OccupancyGrid(N, N, RES)
    for r in range(N):
        for c in range(N):
            ey = 0.5 * N * RES - (c + 0.5) * RES
            if right_m <= ey <= left_m:
                g.drivable[r, c] = 1.0
    return g


def _poly_lat(poly) -> list[float]:
    """Ego-frame lateral of every polyline point (left positive)."""
    return [float(p[1]) for p in np.asarray(poly, dtype=float)]


def test_two_lane_road_right_edge_plus_half_is_lane_centre():
    # 7 m road with its right edge at -1.75 m: the candidate must sit on
    # the lane centre (0 m) and run ahead of the car.
    g = _grid_with_road(right_m=-1.75, left_m=5.25)
    poly = bev_corridor_lane_center(g, POS, HEADING)
    assert poly is not None and len(poly) >= 3
    lats = _poly_lat(poly)
    assert abs(float(np.median(lats))) < 0.3
    # near -> far, and not starting metres ahead of the car (the anchor
    # rule shared with bev_drivable_center: prepend ego only if > 2 m)
    fwd = np.asarray(poly, dtype=float)[:, 0]
    assert np.all(np.diff(fwd) > -1e-6)
    assert fwd[0] <= 2.0 + 1e-6


@pytest.mark.parametrize("right,left", [(-7.0, 7.0), (-6.0, 8.0)])
def test_wide_road_abstains(right, left):
    g = _grid_with_road(right_m=right, left_m=left)
    assert bev_corridor_lane_center(g, POS, HEADING) is None


def test_no_drivable_abstains():
    g = OccupancyGrid(N, N, RES)
    assert bev_corridor_lane_center(g, POS, HEADING) is None


def test_obstacle_on_right_edge_is_not_read_as_boundary():
    # A wall fusing onto the rightmost drivable column must not become
    # the lane's right boundary: the free corridor excludes it.
    g = _grid_with_road(right_m=-1.75, left_m=5.25)
    for r in range(N):
        c = int((0.5 * N * RES - (-1.75)) / RES - 0.5)
        g.obstacle[r, c] = 1
    poly = bev_corridor_lane_center(g, POS, HEADING)
    assert poly is not None
    lats = _poly_lat(poly)
    # right edge moved in by one cell (0.5 m) -> centre near +0.5, never
    # on/beyond the walled column
    assert 0.0 < float(np.median(lats)) <= 1.0


def test_strict_flag_off_keeps_fail_closed():
    g = _grid_with_road(right_m=-1.75, left_m=5.25)
    ref = select_lane_reference(
        lane_frame=None, pos=POS, heading=HEADING, grid=g,
        lane_mode="sensor", strict_sensor=True)
    assert ref.center is None
    assert ref.src == SRC_UNAVAILABLE
    assert ref.meta["lane_src"] == "perception-unavailable"


def test_strict_flag_on_uses_corridor_on_two_lane_road():
    g = _grid_with_road(right_m=-1.75, left_m=5.25)
    ref = select_lane_reference(
        lane_frame=None, pos=POS, heading=HEADING, grid=g,
        lane_mode="sensor", strict_sensor=True, corridor_fallback=True)
    assert ref.center is not None
    assert ref.src == SRC_CORRIDOR
    assert ref.meta["lane_src"] == "corridor"
    # the reference is the lane centre, not the road centre
    lats = _poly_lat(ref.center)
    assert abs(float(np.median(lats))) < 0.3


def test_strict_flag_on_still_fails_closed_on_wide_road():
    g = _grid_with_road(right_m=-6.0, left_m=8.0)
    ref = select_lane_reference(
        lane_frame=None, pos=POS, heading=HEADING, grid=g,
        lane_mode="sensor", strict_sensor=True, corridor_fallback=True)
    assert ref.center is None
    assert ref.src == SRC_UNAVAILABLE


def test_non_strict_mode_is_untouched_by_the_flag():
    g = _grid_with_road(right_m=-1.75, left_m=5.25)
    ref = select_lane_reference(
        lane_frame=None, pos=POS, heading=HEADING, grid=g,
        lane_mode="map", strict_sensor=False, corridor_fallback=True)
    # legacy mode keeps its own BEV/route fallback semantics; the corridor
    # candidate is a strict-mode-only branch and must not fire here.
    assert ref.src != SRC_CORRIDOR


def test_default_gate_matches_measured_buckets():
    assert CORRIDOR_LANE_MAX_WIDTH_M == 10.0
