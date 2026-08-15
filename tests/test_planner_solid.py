"""cross_solid overtake gate + road-width clamp regression (pure logic)."""
from __future__ import annotations

import numpy as np
import pytest

from beamng_autopilot.perception import Obstacle
from beamng_autopilot.planner import (
    LocalPlanner,
    _MapLaneBoundary,
    _clamp_path_lateral,
)


def _route(n=13, step=5.0):
    return np.array([[i * step, 0.0] for i in range(n)], dtype=float)


def _veh(x, y=0.0):
    return Obstacle(x=x, y=y, half_w=1.0, half_h=2.2, category="vehicle")


class TestClampPathLateral:
    def test_path_within_corridor_untouched(self):
        route = _route()
        assert np.allclose(_clamp_path_lateral(route.copy(), route, 3.0),
                           route)

    def test_overshoot_pulled_back(self):
        route = _route()
        out = route.copy()
        out[:, 1] = 5.0  # 5 m left of the centre
        assert np.allclose(_clamp_path_lateral(out, route, 3.0)[:, 1], 3.0)

    def test_mixed_path_only_overshoot_clamped(self):
        route = _route()
        out = route.copy()
        out[3:7, 1] = 4.0
        r = _clamp_path_lateral(out, route, 2.0)
        assert np.allclose(r[3:7, 1], 2.0)
        assert np.allclose(r[:3, 1], 0.0)
        assert np.allclose(r[7:, 1], 0.0)

    def test_garbage_inputs(self):
        assert len(_clamp_path_lateral(np.empty((0, 2)), _route(), 3.0)) == 0


class TestCrossSolid:
    def _plan(self, cross_solid):
        planner = LocalPlanner()
        route = _route()
        # centre line along y=0; legal lanes lie to the right (+1, -Y).
        # The stopped car sits in the legal lane (right side), so passing
        # it means crossing the centre line - exactly the real scenario.
        centre = _MapLaneBoundary(
            np.array([[0.0, 0.0], [60.0, 0.0]]), allowed_side=1.0)
        drive, blocked = planner.plan(
            route, [_veh(15.0, -1.75)], (0.0, 0.0), 0.0, 0,
            solid_lines=[centre], cross_solid=cross_solid)
        return drive, blocked

    def test_solid_blocks_detour_by_default(self):
        _, blocked = self._plan(False)
        assert blocked is True

    def test_cross_solid_allows_centre_crossing(self):
        drive, blocked = self._plan(True)
        assert blocked is False
        assert len(drive) >= 2
        # the detour must actually cross the centre line (left of y=0)
        assert float(np.max(drive[:, 1])) > 0.5

    def test_default_param_keeps_old_behaviour(self):
        planner = LocalPlanner()
        route = _route()
        centre = _MapLaneBoundary(
            np.array([[0.0, 0.0], [60.0, 0.0]]), allowed_side=1.0)
        _, blocked = planner.plan(
            route, [_veh(15.0, -1.75)], (0.0, 0.0), 0.0, 0,
            solid_lines=[centre])
        assert blocked is True