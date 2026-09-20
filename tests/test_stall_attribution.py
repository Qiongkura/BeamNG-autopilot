"""P6: classify a stop before trying to remove it.

The dangerous move is raising a speed floor because "it stalls too
much".  That removes real obstacles' protection along with the bad
stops, so the classification has to be conservative: a stop is
unjustified only when the evidence says nothing required it.
"""

from scripts.m5_stall_attribution import (
    CONTROL_TIMEOUT_S,
    MIN_STOP_FRAMES,
    attribute,
    classify_stop,
    classify_stops,
    find_stops,
    print_attribute,
    stop_segment,
)


def _f(**kw):
    base = {"t": 0.0, "speed": 0.0, "reason": "", "level": "safe",
            "closest_obs_m": 999.0, "road_surface": "on_road",
            "road_checked": True, "cmd_gap_s": 0.1}
    base.update(kw)
    return base


def _stop(**kw):
    return dict({"reasons": [], "levels": [], "min_closest_obs_m": None,
                 "road_states": [], "stale_any": False,
                 "max_cmd_gap_s": None, "goal_remaining_m": None,
                 "has_route": True, "frames": 5, "duration_s": 3.0}, **kw)


class TestFindStops:
    def test_a_run_of_stationary_frames_is_a_stop(self):
        assert len(find_stops([_f()] * 5)) == 1

    def test_a_brief_dip_is_not_a_stop(self):
        # One frame at 0 is a sample, not a stall.
        frames = [_f(speed=5.0), _f(speed=0.0), _f(speed=5.0)]
        assert find_stops(frames) == []

    def test_moving_frames_are_not_a_stop(self):
        assert find_stops([_f(speed=5.0)] * 10) == []

    def test_a_missing_speed_reads_as_moving_not_stopped(self):
        """No measurement is not evidence of a stop - this is the
        default-value-as-healthy error pointed the other way."""
        frames = [_f() for _ in range(5)]
        for f in frames:
            del f["speed"]
        assert find_stops(frames) == []

    def test_two_stops_are_two_stops(self):
        frames = ([_f()] * 4 + [_f(speed=5.0)] * 3 + [_f()] * 4)
        assert len(find_stops(frames)) == 2

    def test_the_minimum_length_is_respected(self):
        assert MIN_STOP_FRAMES == 3
        assert find_stops([_f()] * 2) == []


class TestClassify:
    def test_no_drivable_path(self):
        assert classify_stop(_stop(reasons=["no drivable path"])) == \
            "no_executable_path"

    def test_a_path_hold_is_not_obstacle_stop(self):
        assert classify_stop(_stop(reasons=["path hold (grace)"])) == \
            "no_executable_path"

    def test_stale_sensor(self):
        assert classify_stop(_stop(stale_any=True)) == "stale_sensor"

    def test_an_obstacle_was_there(self):
        assert classify_stop(_stop(min_closest_obs_m=2.0)) == \
            "obstacle_or_boundary"

    def test_the_999_sentinel_is_not_an_obstacle(self):
        # stop_segment filters it; a raw 999 must not classify as one.
        seg = stop_segment([_f(closest_obs_m=999.0)])
        assert seg["min_closest_obs_m"] is None

    def test_an_unknown_road_is_a_reason_to_stop(self):
        assert classify_stop(_stop(road_states=["unknown"])) == \
            "obstacle_or_boundary"

    def test_control_timeout(self):
        assert classify_stop(_stop(max_cmd_gap_s=CONTROL_TIMEOUT_S + 1)) == \
            "control_timeout"

    def test_arriving_is_not_a_stall(self):
        assert classify_stop(_stop(goal_remaining_m=1.0)) == "near_goal"

    def test_no_route(self):
        assert classify_stop(_stop(has_route=False,
                                   route_column_seen=True)) == "no_route_config"

    def test_a_level_with_no_reason_is_unknown_not_unjustified(self):
        """The monitor asked for a stop but the frame says nothing about
        why.  Absence of a reason is not proof of a bad stop."""
        assert classify_stop(_stop(levels=["degraded"])) == "unknown"

    def test_nothing_at_all_is_unknown(self):
        assert classify_stop(_stop()) == "unknown"

    def test_a_missing_route_column_is_unknown_not_configuration(self):
        assert classify_stop(_stop(has_route=False,
                                   route_column_seen=False)) == "unknown"

    def test_unjustified_needs_a_clean_frame(self):
        assert classify_stop(_stop(reasons=["safe"],
                                   levels=["safe"])) == "unjustified"

    def test_an_obstacle_beats_the_missing_route(self):
        """Order matters: a stop with an obstacle is never counted as
        unjustified just because the route was also missing."""
        assert classify_stop(_stop(min_closest_obs_m=1.0,
                                   has_route=False)) == \
            "obstacle_or_boundary"


class TestAttribute:
    def test_shares_are_of_stopped_frames(self):
        frames = [_f(min_closest_obs_m=1.0, closest_obs_m=1.0)] * 5
        a = attribute(frames)
        assert a["n_stops"] == 1
        assert a["by_class"]["obstacle_or_boundary"]["frames"] == 5

    def test_a_run_with_no_stops(self):
        a = attribute([_f(speed=5.0)] * 10)
        assert a["n_stops"] == 0
        assert a["stopped_share"] == 0.0

    def test_the_longest_stop_is_reported(self):
        frames = [_f(t=float(i), closest_obs_m=1.0,
                     min_closest_obs_m=1.0) for i in range(6)]
        a = attribute(frames)
        assert a["by_class"]["obstacle_or_boundary"]["longest_s"] == 5.0

    def test_print_mentions_the_legal_path_caveat(self, capsys):
        frames = [_f(t=float(i), reason="safe", level="safe",
                     closest_obs_m=999.0, route_dist=10.0) for i in range(4)]
        a = attribute(frames)
        print_attribute("x", a)
        out = capsys.readouterr().out
        assert "candidates for optimisation" in out
        assert "legal" in out

    def test_stops_are_listed_with_their_class(self):
        segs = classify_stops([_f(speed=5.0)] * 2
                              + [_f(closest_obs_m=1.0,
                                    min_closest_obs_m=1.0)] * 4)
        assert len(segs) == 1
        assert segs[0]["class"] == "obstacle_or_boundary"


class TestRouteColumnVersusRouteValue:
    """Only 35 of 131 town runs carry route_dist.  A missing COLUMN is a
    measurement gap, not a configuration finding."""

    def _seg(self, **kw):
        return dict({"reasons": ["safe"], "levels": ["safe"],
                     "min_closest_obs_m": None, "road_states": [],
                     "stale_any": False, "max_cmd_gap_s": None,
                     "goal_remaining_m": None, "has_route": False,
                     "route_column_seen": True}, **kw)

    def test_a_route_column_that_says_none_is_no_route(self):
        assert classify_stop(self._seg()) == "no_route_config"

    def test_a_missing_route_column_is_unknown(self):
        assert classify_stop(self._seg(route_column_seen=False)) == "unknown"

    def test_the_segment_records_which_it_was(self):
        seg = stop_segment([{"t": 0.0, "speed": 0.0, "route_dist": 12.0}])
        assert seg["route_column_seen"] is True
        assert seg["has_route"] is True
        seg2 = stop_segment([{"t": 0.0, "speed": 0.0}])
        assert seg2["route_column_seen"] is False
        assert seg2["has_route"] is False
