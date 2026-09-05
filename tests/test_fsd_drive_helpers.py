"""Offline tests for the fsd_drive module's pure helper functions."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from beamng_autopilot import fsd_drive


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


# --- _route_lateral_off_m ----------------------------------------------
def test_route_lateral_right_offset() -> None:
    n = 40
    route = np.column_stack([np.linspace(0, 60, n), np.zeros(n)])
    left = np.column_stack([np.linspace(0, 60, n), np.full(n, 3.5)])
    right = np.column_stack([np.linspace(0, 60, n), np.full(n, -3.5)])
    lat, beyond, hw = fsd_drive._route_lateral_off_m(
        np.array([10.0, -2.0]), route, left, right)
    assert lat == pytest.approx(2.0, abs=1e-9)   # positive = RIGHT
    assert hw == pytest.approx(3.5, abs=1e-9)
    assert beyond == pytest.approx(0.0, abs=1e-9)


def test_route_lateral_beyond_edge() -> None:
    n = 40
    route = np.column_stack([np.linspace(0, 60, n), np.zeros(n)])
    left = np.column_stack([np.linspace(0, 60, n), np.full(n, 3.5)])
    right = np.column_stack([np.linspace(0, 60, n), np.full(n, -3.5)])
    lat, beyond, hw = fsd_drive._route_lateral_off_m(
        np.array([10.0, -5.0]), route, left, right)
    assert lat == pytest.approx(5.0, abs=1e-9)
    assert beyond == pytest.approx(1.5, abs=1e-9)


def test_route_lateral_degenerate_inputs() -> None:
    lat, beyond, hw = fsd_drive._route_lateral_off_m(
        np.array([0.0, 0.0]), np.array([[0.0, 0.0]]), None, None)
    assert (lat, beyond, hw) == (0.0, 0.0, 7.0)


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
