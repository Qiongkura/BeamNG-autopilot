"""Offline tests for the perception-only lateral road guard (no map)."""

from __future__ import annotations

import numpy as np
import pytest

from beamng_autopilot.lane import perception_curve_speed, perception_lateral_guard
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
