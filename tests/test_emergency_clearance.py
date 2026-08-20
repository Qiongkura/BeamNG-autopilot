"""Raw-sensor emergency-stop clearance regression (pure logic, no game).

``forward_clearance_m`` is the last line of defence against wall hits: it
reads raw LiDAR/raycast points directly instead of relying on obstacle
classification, so a wall misread as roadside furniture can never defeat
the speed limit / stop decision.
"""
from __future__ import annotations

import numpy as np

from beamng_autopilot.planner import (
    emergency_speed_limit_mps,
    emergency_stop_clearance_m,
    forward_clearance_m,
)


class TestForwardClearance:
    def test_empty_hits_infinite(self):
        assert np.isinf(forward_clearance_m([], (0.0, 0.0), (1.0, 0.0)))

    def test_wall_ahead_returns_distance(self):
        hits = [(5.0, 0.0), (6.0, 0.3), (8.0, -0.2)]
        assert forward_clearance_m(
            hits, (0.0, 0.0), (1.0, 0.0)) == 5.0

    def test_side_hit_outside_corridor_ignored(self):
        # 2.5 m to the side > half_width 1.5 -> ignored
        hits = [(5.0, 2.5)]
        assert np.isinf(forward_clearance_m(
            hits, (0.0, 0.0), (1.0, 0.0), half_width=1.5))

    def test_hit_behind_car_ignored(self):
        hits = [(-3.0, 0.0)]
        assert np.isinf(forward_clearance_m(
            hits, (0.0, 0.0), (1.0, 0.0)))

    def test_forward_heading_respected(self):
        # heading north: a hit at (5, 0) is to the west -> ignored
        hits = [(5.0, 0.0)]
        assert np.isinf(forward_clearance_m(
            hits, (0.0, 0.0), (0.0, 1.0)))

    def test_oriented_fwd_uses_unit_dir(self):
        hits = [(2.0, 0.0)]
        assert forward_clearance_m(
            hits, (0.0, 0.0), (2.0, 0.0)) == 2.0


class TestEmergencyStopClearance:
    def test_zero_speed_small_margin(self):
        assert np.isclose(emergency_stop_clearance_m(0.0), 1.0)

    def test_braking_distance_grows(self):
        assert (emergency_stop_clearance_m(20.0)
                > emergency_stop_clearance_m(5.0))

    def test_longer_margin_more_room(self):
        assert (emergency_stop_clearance_m(10.0, margin=2.0)
                > emergency_stop_clearance_m(10.0, margin=1.0))


class TestEmergencySpeedLimit:
    def test_inside_reserve_forces_stop(self):
        stop, cap = emergency_speed_limit_mps(2.0, need=6.0)
        assert stop is True and cap == 0.0

    def test_at_reserve_caps_to_zero(self):
        # At exactly the reserve the cap is 0 (still stops, but the flag
        # only fires when the clearance is *inside* the reserve).
        stop, cap = emergency_speed_limit_mps(6.0, need=6.0)
        assert stop is False and cap == 0.0

    def test_outside_reserve_returns_smooth_approach(self):
        stop, cap = emergency_speed_limit_mps(10.0, need=6.0, gain=2.5)
        assert stop is False and np.isclose(cap, 10.0)

    def test_far_clearance_does_not_cap(self):
        stop, cap = emergency_speed_limit_mps(30.0, need=6.0, gain=2.5)
        assert stop is False and np.isclose(cap, 60.0)

    def test_cap_never_exceeds_raw_spare(self):
        # The cap is proportionally bounded; near the reserve it shrinks.
        a = emergency_speed_limit_mps(7.0, need=6.0, gain=2.5)[1]
        b = emergency_speed_limit_mps(9.0, need=6.0, gain=2.5)[1]
        assert 0.0 <= a < b
