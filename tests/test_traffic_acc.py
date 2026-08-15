"""ACC / car-following helpers from ``traffic.py`` (dynamic traffic).

Pure-Python decision logic: no game instance needed, works on both
runtimes.  Covers the helpers added for ACC / overtaking: lead-vehicle
projection, time-gap following, overtake gating and along-route speed.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from beamng_autopilot.perception import Obstacle
from beamng_autopilot.traffic import (
    find_lead_vehicle,
    follow_speed,
    should_overtake,
    vehicle_along_speed,
)


def _veh(x, y=0.0, vx=None, vy=None):
    ob = Obstacle(x=x, y=y, half_w=1.0, half_h=2.2, category="vehicle")
    if vx is not None:
        ob.velocity = np.array([vx, vy or 0.0], dtype=float)
        ob.heading = 0.0
        ob.vehicle_id = "test-veh"
    return ob


STRAIGHT = np.array([[0.0, 0.0], [20.0, 0.0], [40.0, 0.0], [60.0, 0.0],
                     [80.0, 0.0], [100.0, 0.0]])


class TestFindLeadVehicle:
    def test_lead_straight_ahead(self):
        lead, lon, lat = find_lead_vehicle(
            [_veh(25.0, 0.0)], STRAIGHT, (0.0, 0.0), 0.0)
        assert lead is not None
        assert lon == pytest.approx(25.0, abs=0.5)
        assert lat == pytest.approx(0.0, abs=0.1)

    def test_non_vehicle_is_not_a_lead(self):
        wall = Obstacle(x=20.0, y=0.0, half_w=1.0, half_h=2.0,
                        category="wall")
        lead, lon, _ = find_lead_vehicle([wall], STRAIGHT, (0.0, 0.0), 0.0)
        assert lead is None
        assert lon == math.inf

    def test_lead_beside_lane_is_not_a_lead(self):
        lead, _, _ = find_lead_vehicle(
            [_veh(25.0, 3.5)], STRAIGHT, (0.0, 0.0), 0.0)
        assert lead is None

    def test_lead_behind_ego_is_not_a_lead(self):
        lead, _, _ = find_lead_vehicle(
            [_veh(-5.0, 0.0)], STRAIGHT, (0.0, 0.0), 0.0)
        assert lead is None

    def test_nearest_of_two_leads(self):
        lead, lon, _ = find_lead_vehicle(
            [_veh(40.0), _veh(15.0)], STRAIGHT, (0.0, 0.0), 0.0)
        assert lead.x == pytest.approx(15.0)
        assert lon == pytest.approx(15.0, abs=0.5)

    def test_no_route_uses_heading_corridor(self):
        lead, lon, _ = find_lead_vehicle(
            [_veh(12.0, 0.0)], None, (0.0, 0.0), heading=0.0)
        assert lead is not None
        assert lon == pytest.approx(12.0, abs=0.5)

    def test_beyond_max_dist_ignored(self):
        lead, _, _ = find_lead_vehicle(
            [_veh(80.0)], STRAIGHT, (0.0, 0.0), 0.0, max_dist=60.0)
        assert lead is None


class TestFollowSpeed:
    def test_no_lead_returns_cruise(self):
        assert follow_speed(10.0, math.inf, 5.0, 10.0) == 10.0
        assert follow_speed(10.0, None, 5.0, 10.0) == 10.0

    def test_far_fast_lead_keeps_cruise(self):
        assert follow_speed(10.0, 40.0, 9.0, 10.0) == 10.0

    def test_close_lead_brakes(self):
        v = follow_speed(10.0, 8.0, 5.0, 10.0)
        assert 0.0 <= v < 5.0

    def test_lead_within_min_gap_stops(self):
        assert follow_speed(10.0, 1.0, 5.0, 10.0) == 0.0

    def test_never_exceeds_cruise(self):
        assert follow_speed(6.0, 60.0, 20.0, 6.0) == 6.0

    def test_never_negative(self):
        assert follow_speed(10.0, 5.0, 0.0, 10.0) >= 0.0


class TestShouldOvertake:
    def test_slow_lead_triggers(self):
        assert should_overtake(2.0, 10.0) is True

    def test_near_cruise_lead_does_not(self):
        assert should_overtake(9.0, 10.0) is False

    def test_low_cruise_never_overtakes(self):
        # cruise below the min overtake speed: no overtake
        assert should_overtake(1.0, 2.0) is False


class TestVehicleAlongSpeed:
    def test_moving_lead_positive(self):
        ob = _veh(20.0, 0.0, vx=5.0)
        assert vehicle_along_speed(ob, STRAIGHT, (0.0, 0.0), 0.0) \
            == pytest.approx(5.0)

    def test_oncoming_clamped_to_zero(self):
        ob = _veh(20.0, 0.0, vx=-5.0)
        assert vehicle_along_speed(ob, STRAIGHT, (0.0, 0.0), 0.0) == 0.0

    def test_no_velocity_is_zero(self):
        assert vehicle_along_speed(_veh(20.0), STRAIGHT,
                                   (0.0, 0.0), 0.0) == 0.0