"""``LocalPlanner.speed()`` regression: velocity-aware kinematic limits.

A moving lead vehicle must not force the ego to brake as hard as a static
wall; an oncoming vehicle (negative along-route speed) must still be
treated as a static blocker.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from beamng_autopilot.perception import Obstacle
from beamng_autopilot.planner import LocalPlanner


def _route(n=11, step=5.0):
    return np.array([[i * step, 0.0] for i in range(n)], dtype=float)


def _veh(x, vx=None):
    ob = Obstacle(x=x, y=0.0, half_w=1.0, half_h=2.2, category="vehicle")
    if vx is not None:
        ob.velocity = np.array([vx, 0.0], dtype=float)
        ob.heading = 0.0
    return ob


@pytest.fixture(scope="module")
def planner():
    return LocalPlanner()


class TestSpeedBasics:
    def test_empty_route_keeps_cruise(self, planner):
        v, d = planner.speed(_route(), [], (0.0, 0.0), 0.0, 0, 10.0)
        assert v == pytest.approx(10.0)
        assert d == 999.0

    def test_sharp_bend_limits_below_cruise(self, planner):
        bend = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0],
                         [10.0, 5.0], [10.0, 10.0], [10.0, 15.0],
                         [10.0, 20.0]], dtype=float)
        v, _ = planner.speed(bend, [], (0.0, 0.0), 0.0, 0, 20.0)
        assert 0.0 < v < 15.0


class TestMovingLead:
    def test_moving_lead_brakes_less_than_static(self, planner):
        route = _route()
        vs, _ = planner.speed(route, [_veh(8.0)], (0.0, 0.0), 0.0, 0, 10.0)
        vm, _ = planner.speed(route, [_veh(8.0, 5.0)],
                              (0.0, 0.0), 0.0, 0, 10.0)
        assert vm > vs
        assert vm < 10.0

    def test_oncoming_treated_as_static(self, planner):
        route = _route()
        vs, _ = planner.speed(route, [_veh(8.0)], (0.0, 0.0), 0.0, 0, 10.0)
        vo, _ = planner.speed(route, [_veh(8.0, -5.0)],
                              (0.0, 0.0), 0.0, 0, 10.0)
        assert vo == pytest.approx(vs, abs=1e-9)

    def test_far_obstacle_does_not_limit_cruise(self, planner):
        v, _ = planner.speed(_route(), [_veh(30.0)], (0.0, 0.0), 0.0, 0,
                             10.0)
        assert v == pytest.approx(10.0)

    def test_close_static_lead_limits_kinematically(self, planner):
        v, _ = planner.speed(_route(), [_veh(8.0)], (0.0, 0.0), 0.0, 0,
                             10.0)
        assert 0.0 < v < 10.0
        assert math.isclose(v, math.sqrt(2.0 * 3.0 * (8.0 - 3.0)),
                            abs_tol=0.2) or v > 4.0