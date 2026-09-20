"""Offline tests for the perception-only lateral road guard (no map)."""

from __future__ import annotations

import numpy as np
import pytest

from beamng_autopilot.lane import (
    ROAD_SURFACE_OFF,
    ROAD_SURFACE_ON,
    ROAD_SURFACE_UNKNOWN,
    perceived_road_state,
    perception_curve_speed,
    perception_lateral_guard,
    perception_road_bands,
)
from beamng_autopilot.occupancy import OccupancyGrid


def _grid_with_road(y_lo: float, y_hi: float,
                    x_lo: float = 2.0, x_hi: float = 9.0) -> OccupancyGrid:
    """Ego grid whose drivable mask is a road slab between y_lo..y_hi."""
    g = OccupancyGrid(60, 60, 0.5, origin=(0.0, 0.0), heading=0.0)
    ext = g.extent
    r0 = int((ext - x_hi) / g.res)
    r1 = int((ext - x_lo) / g.res)
    c0 = int((ext - y_hi) / g.res)
    c1 = int((ext - y_lo) / g.res)
    g.drivable[r0:r1 + 1, c0:c1 + 1] = 1.0
    return g


def test_no_road_returns_zero() -> None:
    g = OccupancyGrid(60, 60, 0.5, origin=(0.0, 0.0), heading=0.0)
    assert perception_lateral_guard(g) == 0.0


def test_road_centred_on_ego_no_correction() -> None:
    g = _grid_with_road(-3.0, 3.0)
    assert abs(perception_lateral_guard(g)) < 1e-6


def test_road_centre_right_steers_right() -> None:
    g = _grid_with_road(-5.0, -1.0)   # perceived centre ~ -3 m (right)
    corr = perception_lateral_guard(g, gate_m=1.5)
    assert corr > 0.1                  # steer right (+)


def test_road_centre_left_steers_left() -> None:
    g = _grid_with_road(1.0, 5.0)      # perceived centre ~ +3 m (left)
    corr = perception_lateral_guard(g, gate_m=1.5)
    assert corr < -0.1                 # steer left (-)


def test_edge_guard_pulls_away_from_close_edge() -> None:
    # road starts just 0.5 m to the right of the ego: the right edge is
    # dangerously close, so the guard must steer left (negative).
    g = _grid_with_road(-0.5, 6.0)
    corr = perception_lateral_guard(g, gate_m=1.5, edge_margin_m=1.2)
    assert corr < 0.0


def test_perceived_corner_disables_guard() -> None:
    """A road that curves ahead must not fight the turn (guard off)."""
    g = OccupancyGrid(60, 60, 0.5, origin=(0.0, 0.0), heading=0.0)
    ext = g.extent
    # near band (2-4 m) centred on ego; far band (6-9 m) shifted 3 m left
    r0n = int((ext - 4.0) / g.res); r1n = int((ext - 2.0) / g.res)
    r0f = int((ext - 9.0) / g.res); r1f = int((ext - 6.0) / g.res)
    def _cols(y_lo, y_hi):
        return int((ext - y_hi) / g.res), int((ext - y_lo) / g.res)
    c0, c1 = _cols(-3.0, 3.0)
    g.drivable[r0n:r1n + 1, c0:c1 + 1] = 1.0
    c0, c1 = _cols(0.0, 6.0)   # far road shifted left
    g.drivable[r0f:r1f + 1, c0:c1 + 1] = 1.0
    assert perception_lateral_guard(g, gate_m=1.0) == 0.0


def test_curve_speed_keeps_cruise_on_straight_road() -> None:
    g = _grid_with_road(-3.0, 3.0)   # straight slab
    assert perception_curve_speed(g, cruise=6.0) == 6.0


def test_curve_speed_caps_on_perceived_bend() -> None:
    g = OccupancyGrid(60, 60, 0.5, origin=(0.0, 0.0), heading=0.0)
    ext = g.extent
    r0n = int((ext - 4.0) / g.res); r1n = int((ext - 2.0) / g.res)
    r0f = int((ext - 9.0) / g.res); r1f = int((ext - 6.0) / g.res)
    def _cols(y_lo, y_hi):
        return int((ext - y_hi) / g.res), int((ext - y_lo) / g.res)
    c0, c1 = _cols(-3.0, 3.0)          # near: centred
    g.drivable[r0n:r1n + 1, c0:c1 + 1] = 1.0
    c0, c1 = _cols(0.0, 6.0)           # far: shifted 3 m left
    g.drivable[r0f:r1f + 1, c0:c1 + 1] = 1.0
    cap = perception_curve_speed(g, cruise=6.0)
    assert cap < 6.0 and cap >= 1.5


def test_curve_speed_no_road_keeps_cruise() -> None:
    g = OccupancyGrid(60, 60, 0.5, origin=(0.0, 0.0), heading=0.0)
    assert perception_curve_speed(g, cruise=6.0) == 6.0


def test_sparse_road_returns_zero() -> None:
    g = _grid_with_road(-0.5, 0.5, x_lo=4.0, x_hi=4.5)  # tiny patch
    assert perception_lateral_guard(g) == 0.0


# ---------------------------------------------------------------------
# perceived_road_state: "did I see the road at all" (2026-09-20)
# ---------------------------------------------------------------------
HALF_W = 0.95          # planning.constraints.HALF_WIDTH_M


def test_state_on_road_when_band_overlaps_ego() -> None:
    state, bands = perceived_road_state(_grid_with_road(-3.0, 3.0), HALF_W)
    assert state == ROAD_SURFACE_ON
    assert bands is not None and bands["y_left"] > 0.0 > bands["y_right"]


def test_state_off_road_when_band_clears_the_ego_left() -> None:
    # perceived road starts 2.5 m to the LEFT (+y) of the ego: the whole
    # band is outside the car, so the car is off it on the right.
    state, bands = perceived_road_state(_grid_with_road(2.5, 8.0), HALF_W)
    assert state == ROAD_SURFACE_OFF
    assert bands is not None


def test_state_off_road_when_band_clears_the_ego_right() -> None:
    state, _ = perceived_road_state(_grid_with_road(-8.0, -2.5), HALF_W)
    assert state == ROAD_SURFACE_OFF


def test_margin_absorbs_a_band_edge_just_beside_the_ego() -> None:
    """A band edge 0.3 m outside the car is a bend, not an excursion.

    The band is read 2-12 m AHEAD, so on a curve it legitimately slides
    sideways relative to the car.  With the default margin the car is
    still ON; with ``margin_m=0`` the same frame reads OFF, which pins
    that the margin (not the band) is what decides.
    """
    g = _grid_with_road(1.2, 6.0)      # perceived y_right ~ +1.25 m
    state, _ = perceived_road_state(g, HALF_W)
    assert state == ROAD_SURFACE_ON
    state0, _ = perceived_road_state(g, HALF_W, margin_m=0.0)
    assert state0 == ROAD_SURFACE_OFF


def test_state_unknown_when_no_road_evidence() -> None:
    g = OccupancyGrid(60, 60, 0.5, origin=(0.0, 0.0), heading=0.0)
    state, bands = perceived_road_state(g, HALF_W)
    assert state == ROAD_SURFACE_UNKNOWN
    assert bands is None


def test_state_unknown_when_the_road_is_too_sparse() -> None:
    """Silence is UNKNOWN - never "on the road" and never "off the road".

    This is the regression the 2026-09-20 runs exposed: a road mask that
    carries too little surface cannot answer the question, and the old
    off-road metric read exactly that silence as 0.0 m ("perfectly on the
    road") while the car finished 6.96 m past the pavement edge.
    """
    g = _grid_with_road(-0.5, 0.5, x_lo=4.0, x_hi=4.5)   # ~2 cells
    state, bands = perceived_road_state(g, HALF_W)
    assert state == ROAD_SURFACE_UNKNOWN
    assert bands is None


def test_state_unknown_when_the_road_is_beyond_the_band() -> None:
    # 13-14.5 m ahead: outside the 2-12 m read, so the band does not
    # exist even though the road does.
    g = _grid_with_road(-3.0, 3.0, x_lo=13.0, x_hi=14.5)
    state, bands = perceived_road_state(g, HALF_W)
    assert state == ROAD_SURFACE_UNKNOWN
    assert bands is None


def test_state_matches_the_band_reader_used_by_the_lateral_guard() -> None:
    """The state must come from the SAME read the guard and the corner
    governor use, or the layers can disagree about where the road is."""
    g = _grid_with_road(-3.0, 3.0)
    state, bands = perceived_road_state(g, HALF_W)
    assert state == ROAD_SURFACE_ON
    assert bands == perception_road_bands(g)
