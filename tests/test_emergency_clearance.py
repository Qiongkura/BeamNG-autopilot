"""Raw-sensor emergency-stop clearance regression (pure logic, no game).

``forward_clearance_m`` is the last line of defence against wall hits: it
reads raw LiDAR/raycast points directly instead of relying on obstacle
classification, so a wall misread as roadside furniture can never defeat
the speed limit / stop decision.
"""
from __future__ import annotations

import numpy as np

from beamng_autopilot.planner import (
    contact_envelope_speed_mps,
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


class TestContactEnvelopeSpeed:
    """The largest speed whose stop still fits, reaction included."""

    def test_no_reaction_is_the_exact_inverse_of_the_reserve(self):
        for speed in (1.0, 2.46, 3.30, 8.0, 15.0):
            room = emergency_stop_clearance_m(speed) - 1.0
            back = contact_envelope_speed_mps(room + 1.0, reaction_s=0.0)
            assert np.isclose(back, speed, atol=1e-9)

    def test_reaction_always_lowers_the_allowed_speed(self):
        assert (contact_envelope_speed_mps(6.0, reaction_s=0.8)
                < contact_envelope_speed_mps(6.0, reaction_s=0.0))

    def test_more_clearance_allows_more_speed(self):
        a = contact_envelope_speed_mps(2.0, reaction_s=0.8)
        b = contact_envelope_speed_mps(8.0, reaction_s=0.8)
        assert 0.0 <= a < b

    def test_clearance_inside_the_margin_allows_nothing(self):
        assert contact_envelope_speed_mps(0.5, reaction_s=0.8) == 0.0
        assert contact_envelope_speed_mps(1.0, reaction_s=0.8) == 0.0

    def test_allowed_speed_really_stops_within_the_clearance(self):
        """The contract: reserve(allowed) + travel(allowed) <= room."""
        for clearance, tick in ((2.0, 0.8), (3.789, 0.8), (7.5, 0.6),
                                (12.0, 1.0)):
            s = contact_envelope_speed_mps(clearance, reaction_s=tick)
            used = emergency_stop_clearance_m(s) - 1.0 + s * tick
            assert used <= clearance - 1.0 + 1e-9

    def test_the_2026_09_20_collision_frame_refuses_the_hatch(self):
        """The frame the escape hatch raised the target on.

        ``town_1789886413`` t=26.772 s: 3.789 m to the obstacle, car at
        2.461 m/s, and the hatch raised the target to 3.30 m/s
        (= max_speed 6.0 * corridor_open_floor 0.55).  The reserve for
        3.30 m/s is 1.91 m, which fits inside 3.789 m, so a reserve-only
        test lets it through; with the control period it does not.
        """
        allowed = contact_envelope_speed_mps(3.789, reaction_s=0.8)
        assert np.isclose(allowed, 2.72, atol=0.01)
        assert allowed < 3.30                     # the hatch is refused
        # ... and the reserve-only form is exactly what let it through.
        assert 3.789 > emergency_stop_clearance_m(3.30) - 1.0

    def test_the_next_frame_is_past_helping(self):
        """t=27.572 s: 1.206 m left.  Nothing legal can be allowed here.

        The target was already 0.0 by then, which is why the reserve-only
        gate was a no-op on this chain - it only fired once the obstacle
        risk layer had already acted.
        """
        assert contact_envelope_speed_mps(1.206, reaction_s=0.8) < 0.3
