"""Offline tests for the fsd_drive module's pure helper functions."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from beamng_autopilot import fsd_drive


def test_fs_drive_session_keeps_args_and_compat_wrapper():
    args = SimpleNamespace(runtime="tech", attach=True)
    session = fsd_drive.FSDriveSession(args)
    assert session.args is args
    assert callable(fsd_drive.run)


def test_fs_drive_session_build_route_without_goal_is_safe():
    class _Conn:
        class _State:
            pos = np.array([0.0, 0.0, 0.0])

        def get_state(self):
            return self._State()

        def read_navigation_route(self):
            return None

    args = SimpleNamespace(goal=None, traffic=0)
    result = fsd_drive.FSDriveSession(args)._build_route(_Conn())
    assert result == (None, None, None, None)




# --- _trim_backtrack ---------------------------------------------------
def test_trim_backtrack_drops_reversed_tail() -> None:
    fwd = np.column_stack([np.linspace(0, 20, 11), np.zeros(11)])
    back = np.column_stack([np.linspace(18, 14, 3), np.full(3, 0.3)])
    route = np.vstack([fwd, back])
    trimmed = fsd_drive._trim_backtrack(route)
    assert len(trimmed) < len(route)
    # the trimmed route never doubles back
    seg = np.diff(trimmed[:, 0])
    assert (seg >= 0).all()


def test_trim_backtrack_keeps_normal_route() -> None:
    route = np.column_stack([np.linspace(0, 30, 16),
                             np.zeros(16)])
    assert fsd_drive._trim_backtrack(route) is route or len(
        fsd_drive._trim_backtrack(route)) == len(route)


def test_trim_backtrack_short_route_untouched() -> None:
    route = np.array([[0.0, 0.0], [5.0, 0.0], [1.0, 0.0]])
    out = fsd_drive._trim_backtrack(route)
    assert len(out) == len(route)


# --- _ref_bearing ------------------------------------------------------
def test_ref_bearing_straight_east() -> None:
    ref = np.column_stack([np.linspace(0, 40, 41), np.zeros(41)])
    assert fsd_drive._ref_bearing(ref, np.array([0.0, 0.0])) == 0.0


def test_ref_bearing_window_and_none() -> None:
    ref = np.column_stack([np.linspace(0, 40, 41), np.zeros(41)])
    # only samples 5..15 m are measured
    b = fsd_drive._ref_bearing(ref, np.array([0.0, 0.0]),
                               min_m=5.0, max_m=15.0)
    assert b == 0.0
    assert fsd_drive._ref_bearing(None, np.array([0.0, 0.0])) is None
    # fewer than 2 samples beyond min_m: no measurable forward extent
    short = np.array([[0.0, 0.0], [1.0, 0.0]])
    assert fsd_drive._ref_bearing(short, np.array([0.0, 0.0])) is None


def test_ref_bearing_north() -> None:
    ref = np.column_stack([np.zeros(41), np.linspace(0, 40, 41)])
    assert fsd_drive._ref_bearing(ref, np.array([0.0, 0.0])) == 90.0


# --- _perception_off_road_m --------------------------------------------
def _lane_out(left=None, right=None):
    """Perception stub carrying ONLY detected lane boundaries."""
    return SimpleNamespace(lane_left=left, lane_right=right)


def _line(x0, y0, x1, y1, n=13):
    """Straight boundary polyline in travel direction."""
    return np.column_stack([np.linspace(x0, x1, n),
                            np.linspace(y0, y1, n)])


def test_perception_off_road_unknown_without_boundaries() -> None:
    # A boundary the sensors did not publish this tick is "unknown" ->
    # 0.0, never a map / nav fallback (project lateral rule).
    assert fsd_drive._perception_off_road_m(
        _lane_out(), np.array([10.0, 0.0]), 0.0) == 0.0
    assert fsd_drive._perception_off_road_m(
        None, np.array([10.0, 0.0]), 0.0) == 0.0
    # Degenerate boundary (a single vertex, no segment) is unusable too.
    assert fsd_drive._perception_off_road_m(
        _lane_out(left=np.array([[0.0, 3.5]])),
        np.array([10.0, 0.0]), 0.0) == 0.0


def test_perception_off_road_in_lane_is_zero() -> None:
    out = _lane_out(left=_line(0, 3.5, 60, 3.5),
                    right=_line(0, -3.5, 60, -3.5))
    assert fsd_drive._perception_off_road_m(
        out, np.array([10.0, 0.0]), 0.0) == 0.0


def test_perception_off_road_right_corner_overshoot() -> None:
    # Centre 3.0 m right of the lane centre: the right corners sit 0.4 m
    # past the detected right boundary (half width 0.9 m, line at -3.5 m).
    out = _lane_out(right=_line(0, -3.5, 60, -3.5))
    off = fsd_drive._perception_off_road_m(
        out, np.array([10.0, -3.0]), 0.0)
    assert off == pytest.approx(0.4, abs=1e-6)


def test_perception_off_road_left_corner_overshoot() -> None:
    # Mirrored: a corner past the left line is oncoming-traffic territory.
    out = _lane_out(left=_line(0, 3.5, 60, 3.5))
    off = fsd_drive._perception_off_road_m(
        out, np.array([10.0, 4.2]), 0.0)
    assert off == pytest.approx(1.6, abs=1e-6)


def test_perception_off_road_reports_worst_corner() -> None:
    # Worst of the four corners wins (not the centre point, not a sum).
    out = _lane_out(left=_line(0, 3.5, 60, 3.5),
                   right=_line(0, -3.5, 60, -3.5))
    off = fsd_drive._perception_off_road_m(
       out, np.array([10.0, 3.0]), 0.0)
    assert off == pytest.approx(0.4, abs=1e-6)


# --- _endzone_align_yaw_dev --------------------------------------------
def test_endzone_align_uses_perceived_lane_direction() -> None:
    # Straight ahead, lane bearing 0 -> nothing to straighten.
    assert fsd_drive._endzone_align_yaw_dev(
        0.0, np.array([1.0, 0.0]), "painted") == 0.0
    # Nose 0.2 rad left of the perceived lane direction -> +0.2 (steer
    # right pulls the heading back down).
    dev = fsd_drive._endzone_align_yaw_dev(
        0.2, np.array([1.0, 0.0]), "sensor_lane")
    assert dev == pytest.approx(0.2, abs=1e-9)


def test_endzone_align_wraps_at_pi_boundary() -> None:
    # 350 deg heading vs a 10 deg lane bearing is -20 deg, not +340.
    bear = math.radians(10.0)
    dev = fsd_drive._endzone_align_yaw_dev(
        math.radians(350.0), np.array([math.cos(bear), math.sin(bear)]),
        "painted")
    assert dev == pytest.approx(math.radians(-20.0), abs=1e-9)


def test_endzone_align_refuses_map_and_degenerate_directions() -> None:
    # A nav-route / heading fallback must NOT become a steering
    # reference: no perceived lane direction -> None -> hold the brake.
    d = np.array([1.0, 0.0])
    for src in ("route", "none", "", None):
        assert fsd_drive._endzone_align_yaw_dev(0.3, d, src) is None, src
    for bad in (None, np.array([0.0, 0.0]), np.array([np.nan, 0.0]),
                np.array([1.0])):
        assert fsd_drive._endzone_align_yaw_dev(0.3, bad, "painted") is None


# --- _endzone_travel_direction -----------------------------------------
def test_endzone_direction_prefers_perception_over_route() -> None:
    d, src = fsd_drive._endzone_travel_direction(
        0.0, painted=(1.0, 0.0), sensor=(0.0, 1.0),
        route_tangent=(0.0, -1.0))
    assert src == "painted"
    assert d == pytest.approx([1.0, 0.0])
    d, src = fsd_drive._endzone_travel_direction(
        0.0, sensor=(0.0, 1.0), route_tangent=(0.0, -1.0))
    assert src == "sensor_lane"
    assert d == pytest.approx([0.0, 1.0])


def test_endzone_direction_strict_never_uses_route_tangent() -> None:
    # Route tangent 90 deg off the ego heading must NOT steer the stop
    # ray: strict FSD holds the current heading instead (the legal
    # no-perception degradation).
    d, src = fsd_drive._endzone_travel_direction(
        0.0, route_tangent=(0.0, -1.0), strict=True)
    assert src == "none"
    assert d == pytest.approx([1.0, 0.0])


def test_endzone_direction_legacy_route_fallback_kept() -> None:
    d, src = fsd_drive._endzone_travel_direction(
        0.0, route_tangent=(0.0, 2.0), strict=False)
    assert src == "route"
    assert d == pytest.approx([0.0, 1.0])


def test_endzone_direction_normalises_and_rejects_degenerate() -> None:
    d, src = fsd_drive._endzone_travel_direction(0.0, painted=(3.0, 0.0))
    assert src == "painted"
    assert float(np.hypot(float(d[0]), float(d[1]))) == pytest.approx(1.0)
    for bad in (None, (0.0, 0.0), (float("nan"), 0.0), (1.0,)):
        _, src = fsd_drive._endzone_travel_direction(
            0.5, painted=bad, strict=True)
        assert src == "none"
# --- _path_curvature_ff ------------------------------------------------
def test_ff_zero_on_straight_path() -> None:
    path = np.column_stack([np.linspace(0, 30, 61), np.zeros(61)])
    ff = fsd_drive._path_curvature_ff(path, np.array([0.0, 0.0, 0.0]), 0.0)
    assert ff == 0.0


def test_ff_left_curve_gives_negative_input() -> None:
    # left-bending arc: heading rotates from 0 toward +90 deg
    t = np.linspace(0.0, 1.6, 80)
    r = 10.0
    path = np.column_stack([r * np.sin(t), r - r * np.cos(t)])
    ff = fsd_drive._path_curvature_ff(
        path, np.array([0.0, 0.0, 0.0]), 0.0)
    assert ff < 0.0                      # left curve -> negative input
    assert abs(ff) <= 0.40 + 1e-9        # clamped at max_ff


def test_ff_short_path_zero() -> None:
    path = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.1]])
    assert fsd_drive._path_curvature_ff(
        path, np.array([0.0, 0.0, 0.0]), 0.0) == 0.0


def test_snapshot_age_uses_oldest_head_and_range():
    out = SimpleNamespace(
        frame=np.zeros((4, 4, 3), dtype=np.uint8),
        head_outputs={"semantic": object()},
        meta={"head_age_s": {"semantic": 0.2, "object": 0.7},
              "range_age_s": 0.5})
    assert fsd_drive._sensor_snapshot_age(out) == pytest.approx(0.7)


def test_snapshot_age_no_sensor_is_infinite():
    out = SimpleNamespace(frame=None, head_outputs={}, meta={})
    assert math.isinf(fsd_drive._sensor_snapshot_age(out))


# --- _painted_line_lat -------------------------------------------------
def _tick_with_marks(world):
    marks = [SimpleNamespace(world=np.asarray(world, dtype=float))]
    sem = SimpleNamespace(masks={"line": np.zeros((4, 4), dtype=bool)})
    return SimpleNamespace(
        frame=np.zeros((4, 4, 3), dtype=np.uint8),
        cam=object(),
        head_outputs={"semantic": sem}), marks


def test_painted_line_lat_left_positive() -> None:
    out, marks = _tick_with_marks([[0.0, 2.0, 0.0], [20.0, 2.0, 0.0]])
    lat = fsd_drive._painted_line_lat(
        out, np.array([0.0, 0.0, 0.0]), 0.0, marks=marks)
    assert lat == pytest.approx(2.0, abs=1e-6)   # left = +


def test_painted_line_lat_right_negative() -> None:
    out, marks = _tick_with_marks([[0.0, -1.75, 0.0], [20.0, -1.75, 0.0]])
    lat = fsd_drive._painted_line_lat(
        out, np.array([0.0, 0.0, 0.0]), 0.0, marks=marks)
    assert lat == pytest.approx(-1.75, abs=1e-6)


def test_painted_line_lat_near_window_only() -> None:
    # a far marking segment 40 m left must not drag the mean
    out, marks = _tick_with_marks([
        [0.0, 2.0, 0.0], [20.0, 2.0, 0.0],
        [39.0, 9.0, 0.0], [60.0, 9.0, 0.0]])
    lat = fsd_drive._painted_line_lat(
        out, np.array([0.0, 0.0, 0.0]), 0.0, marks=marks)
    assert lat == pytest.approx(2.0, abs=1e-6)


def test_painted_line_lat_none_cases() -> None:
    assert fsd_drive._painted_line_lat(None, None, 0.0) is None
    out = SimpleNamespace(frame=None, cam=None, head_outputs={})
    assert fsd_drive._painted_line_lat(
        out, np.array([0.0, 0.0, 0.0]), 0.0, marks=[]) is None


# --- _snap_heading ------------------------------------------------------
def test_snap_heading_rejects_graph_diagonal_start() -> None:
    # route: first interpolated step points 250 deg (graph zigzag),
    # then the road runs ~172 deg - the snap must face the ROAD way
    road = np.column_stack([np.linspace(0, 60, 41), np.zeros(41)])
    route = np.vstack([[[0.0, 0.0]],
                       [[-0.8, -1.9]],          # diagonal lead-in (250 deg)
                       road + np.array([1.7, 0.6])])
    h = fsd_drive._snap_heading(route, 0.0, 0.0, math.radians(250.0))
    assert abs(math.degrees(h)) < 30.0          # along the road, not 250


def test_snap_heading_keeps_agreeing_segment() -> None:
    # first segment already runs along the road: keep it
    route = np.column_stack([np.linspace(0, 40, 41), np.zeros(41)])
    h = fsd_drive._snap_heading(route, 0.0, 0.0, 0.0)
    assert abs(math.degrees(h)) < 1e-6


def test_snap_heading_falls_back_without_bearing() -> None:
    # a degenerate route with no measurable forward extent keeps h_seg
    route = np.array([[0.0, 0.0], [0.5, 0.0]])
    assert fsd_drive._snap_heading(route, 0.0, 0.0, 1.23) == 1.23
