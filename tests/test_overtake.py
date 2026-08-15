"""Overtake intent manager regression (``traffic.py``, pure logic).

Covers the three-state machine (none -> requested -> active with
hysteresis) and the two hazard helpers that gate it: oncoming traffic and
a solid left line.
"""
from __future__ import annotations

import numpy as np
import pytest

from beamng_autopilot.perception import Obstacle
from beamng_autopilot.traffic import (
    OvertakeStateMachine,
    oncoming_vehicle_ahead,
    solid_marking_left,
)

STRAIGHT = np.array([[0.0, 0.0], [50.0, 0.0], [100.0, 0.0]])


def _veh(x, y=0.0, vx=0.0, vy=0.0):
    ob = Obstacle(x=x, y=y, half_w=1.0, half_h=2.2, category="vehicle")
    ob.velocity = np.array([vx, vy], dtype=float)
    ob.heading = 0.0
    return ob


class TestOncomingVehicleAhead:
    def test_oncoming_in_lane_detected(self):
        # driving toward the ego at 15 m/s, 40 m ahead
        assert oncoming_vehicle_ahead(
            [_veh(40.0, 0.0, vx=-15.0)], STRAIGHT, (0.0, 0.0), 0.0) is True

    def test_same_direction_is_not_oncoming(self):
        assert oncoming_vehicle_ahead(
            [_veh(40.0, 0.0, vx=15.0)], STRAIGHT, (0.0, 0.0), 0.0) is False

    def test_static_vehicle_is_not_oncoming(self):
        assert oncoming_vehicle_ahead(
            [_veh(40.0, 0.0)], STRAIGHT, (0.0, 0.0), 0.0) is False

    def test_behind_ego_ignored(self):
        assert oncoming_vehicle_ahead(
            [_veh(-10.0, 0.0, vx=-15.0)], STRAIGHT, (0.0, 0.0), 0.0) is False

    def test_no_route_uses_heading(self):
        assert oncoming_vehicle_ahead(
            [_veh(30.0, 0.0, vx=-10.0)], None, (0.0, 0.0), heading=0.0) \
            is True


def _marking(kind, x, y, heading=0.0):
    """One short lane marking running along ``heading`` at (x, y)."""
    fwd = np.array([np.cos(heading), np.sin(heading)])
    pts = np.array([np.array([x, y]) + fwd * s for s in (-6.0, 6.0)])
    return type("Mk", (), {"kind": kind, "world": pts})()


class TestSolidMarkingLeft:
    def test_solid_line_left_blocks(self):
        assert solid_marking_left(
            [_marking("solid", 0.0, 2.5)], (0.0, 0.0), 0.0) is True

    def test_dashed_line_left_does_not(self):
        assert solid_marking_left(
            [_marking("dashed", 0.0, 2.5)], (0.0, 0.0), 0.0) is False

    def test_solid_line_right_does_not(self):
        assert solid_marking_left(
            [_marking("solid", 0.0, -2.5)], (0.0, 0.0), 0.0) is False

    def test_far_solid_line_outside_window(self):
        # solid line 40 m ahead (not beside the car): no block
        assert solid_marking_left(
            [_marking("solid", 40.0, 2.5)], (0.0, 0.0), 0.0) is False


class TestOvertakeStateMachine:
    def _sm(self, **kw):
        return OvertakeStateMachine(request_hold_s=1.0, confirm_s=0.5, **kw)

    def _wants(self):
        return dict(has_lead=True, lead_speed=3.0, lead_dist=20.0,
                    cruise=10.0, ego_speed=8.0)

    def test_transient_slow_lead_never_activates(self):
        sm = self._sm()
        for i in range(10):
            assert sm.update(100.0 + i * 0.1, **self._wants()) == "none"

    def test_sustained_request_goes_active(self):
        sm = self._sm()
        t = 100.0
        # 1.0 s of sustained slow lead (6 x 0.2 s) -> requested
        for i in range(6):
            sm.update(t + i * 0.2, **self._wants())
        assert sm.state == "requested"
        # confirm window (0.5 s) starts on the next update
        sm.update(t + 6 * 0.2 + 0.1, **self._wants())
        assert sm.state == "requested"
        sm.update(t + 6 * 0.2 + 0.6, **self._wants())
        assert sm.state == "active"
        # a transient "not wants" resets the request
        sm2 = self._sm()
        for i in range(6):
            sm2.update(t + i * 0.2, **self._wants())
        assert sm2.state == "requested"
        w = self._wants()
        w["oncoming"] = True
        sm2.update(t + 6.0, **w)
        assert sm2.state == "none"

    def test_lead_speeding_up_aborts_active(self):
        sm = self._sm()
        t = 100.0
        for i in range(12):
            sm.update(t + i * 0.2, **self._wants())
        st = sm.update(t + 12 * 0.2 + 0.6, **self._wants())
        assert st == "active"
        w = self._wants()
        w["lead_speed"] = 9.8  # >= cruise * 0.95
        assert sm.update(t + 13.0, **w) == "none"

    def test_lost_lead_aborts_active(self):
        sm = self._sm()
        t = 100.0
        for i in range(12):
            sm.update(t + i * 0.2, **self._wants())
        sm.update(t + 12 * 0.2 + 0.6, **self._wants())
        w = self._wants()
        w["has_lead"] = False
        assert sm.update(t + 13.0, **w) == "none"

    def test_oncoming_blocks_request(self):
        sm = self._sm()
        w = self._wants()
        w["oncoming"] = True
        t = 100.0
        for i in range(12):
            assert sm.update(t + i * 0.2, **w) == "none"

    def test_oncoming_aborts_active(self):
        sm = self._sm()
        t = 100.0
        for i in range(12):
            sm.update(t + i * 0.2, **self._wants())
        sm.update(t + 12 * 0.2 + 0.6, **self._wants())
        w = self._wants()
        w["oncoming"] = True
        assert sm.update(t + 13.0, **w) == "none"

    def test_solid_left_blocks_request(self):
        sm = self._sm()
        w = self._wants()
        w["solid_left"] = True
        t = 100.0
        for i in range(12):
            assert sm.update(t + i * 0.2, **w) == "none"

    def test_min_lead_dist_gate(self):
        sm = self._sm()
        w = self._wants()
        w["lead_dist"] = 3.0  # bumper range: no overtake
        for i in range(12):
            assert sm.update(100.0 + i * 0.2, **w) == "none"

    def test_standstill_gate(self):
        sm = self._sm()
        w = self._wants()
        w["ego_speed"] = 0.0
        for i in range(12):
            assert sm.update(100.0 + i * 0.2, **w) == "none"