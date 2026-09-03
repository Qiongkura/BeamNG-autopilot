"""Unit tests for perception-only painted-line lane placement.

``painted_line_lane_center`` measures the painted line's lateral offset
from the semantic mask (left = +) and returns the world position that
puts the car 1.5 m to the RIGHT of that line - no map-centre / offset
constant.  The CV back-projection (``_mask_to_markings``) is faked here
so the tests exercise the selection / confidence / shift logic directly.
"""

from __future__ import annotations

import math

import numpy as np

from beamng_autopilot.vision import lanes
from beamng_autopilot.vision.lanes import (
    LaneMarking,
    painted_line_direction,
    painted_line_lane_center,
    polyline_dir_at,
)


class _Sem:
    def __init__(self, masks):
        self.masks = masks


def _fake_masks_to_markings(markings):
    def _inner(*_a, **_k):
        return markings
    return _inner


def _line(lat, pos=(10.0, 20.0), kind="solid", n=20):
    """One LaneMarking: a straight painted line at lateral ``lat`` (left=+)."""
    px, py = pos
    lon = np.linspace(2.0, 14.0, n)
    world = np.column_stack([px + lon, py + np.full(n, lat)])
    return LaneMarking(world=world, pixels=world.copy(), color="white",
                       kind=kind, confidence=0.9)


def test_moves_car_into_own_lane_from_line():
    """Line 3 m left -> target 1.5 m right of the line (lane centre)."""
    lanes._mask_to_markings = _fake_masks_to_markings([_line(3.0)])
    tgt = painted_line_lane_center(_Sem({"line": np.zeros((8, 8), np.uint8)}),
                                   cam_model=None,
                                   pos=(10.0, 20.0, 1.5), heading=0.0)
    assert tgt is not None
    assert abs(tgt[0] - 10.0) < 1e-6
    assert abs(tgt[1] - 21.5) < 1e-6  # line 3 m left -> move right 1.5 m


def test_no_line_mask_returns_none():
    lanes._mask_to_markings = _fake_masks_to_markings([])
    assert painted_line_lane_center(_Sem({"road": np.zeros((8, 8), np.uint8)}),
                                    cam_model=None,
                                    pos=(10.0, 20.0, 1.5), heading=0.0) is None


def test_ambiguous_edges_on_both_sides_returns_none():
    """Edge lines on both sides cancel to a bogus centre; must bail out."""
    lanes._mask_to_markings = _fake_masks_to_markings(
        [_line(-2.0), _line(2.0)])
    assert painted_line_lane_center(_Sem({"line": np.zeros((8, 8), np.uint8)}),
                                    cam_model=None,
                                    pos=(10.0, 20.0, 1.5), heading=0.0) is None


def test_already_in_lane_no_teleport():
    """Line 1.6 m left -> shift 0.1 m < deadband; keep the pose."""
    lanes._mask_to_markings = _fake_masks_to_markings([_line(1.6)])
    assert painted_line_lane_center(_Sem({"line": np.zeros((8, 8), np.uint8)}),
                                    cam_model=None,
                                    pos=(10.0, 20.0, 1.5), heading=0.0) is None


def test_clamps_lateral_shift():
    lanes._mask_to_markings = _fake_masks_to_markings([_line(1.0)])
    tgt = painted_line_lane_center(
        _Sem({"line": np.zeros((8, 8), np.uint8)}),
        cam_model=None, pos=(10.0, 20.0, 1.5), heading=0.0,
        max_shift_m=0.5)
    assert tgt is not None
    assert abs(tgt[1] - 19.5) < 1e-6  # clamped to 0.5 m right


def test_ignores_non_line_kinds_and_far_lats():
    lanes._mask_to_markings = _fake_masks_to_markings([_line(1.0, kind="unknown")])
    assert painted_line_lane_center(_Sem({"line": np.zeros((8, 8), np.uint8)}),
                                    cam_model=None,
                                    pos=(10.0, 20.0, 1.5), heading=0.0) is None
    lanes._mask_to_markings = _fake_masks_to_markings([_line(4.5)])
    assert painted_line_lane_center(_Sem({"line": np.zeros((8, 8), np.uint8)}),
                                    cam_model=None,
                                    pos=(10.0, 20.0, 1.5), heading=0.0) is None


def _heading_line(angle_deg, lat, pos=(10.0, 20.0), n=24,
                  lon0=2.0, lon1=14.0):
    """A straight painted line at lateral ``lat`` rotated by ``angle_deg``."""
    a = math.radians(float(angle_deg))
    u = np.array([math.cos(a), math.sin(a)])
    lon = np.linspace(lon0, lon1, n)
    base = np.asarray(pos[:2], dtype=float) + np.array([0.0, lat])
    world = base[None, :] + lon[:, None] * u[None, :]
    return LaneMarking(world=world, pixels=world.copy(), color="white",
                       kind="solid", confidence=0.9)


def test_painted_line_direction_straight_ahead():
    """Line running straight ahead -> the stop ray points along travel."""
    lanes._mask_to_markings = _fake_masks_to_markings(
        [_heading_line(0.0, 1.0)])
    d = painted_line_direction(_Sem({"line": np.zeros((8, 8), np.uint8)}),
                               cam_model=None,
                               pos=(10.0, 20.0, 1.5), heading=0.0)
    assert d is not None
    assert abs(d[0] - 1.0) < 1e-6
    assert abs(d[1]) < 1e-6


def test_painted_line_direction_follows_bend():
    """Line bending 30 deg right -> direction carries that yaw."""
    lanes._mask_to_markings = _fake_masks_to_markings(
        [_heading_line(30.0, 1.0)])
    d = painted_line_direction(_Sem({"line": np.zeros((8, 8), np.uint8)}),
                               cam_model=None,
                               pos=(10.0, 20.0, 1.5), heading=0.0)
    assert d is not None
    assert abs(d[1] / max(1e-9, abs(d[0])) - math.tan(math.radians(30.0))) \
        < 0.05
    assert d[0] > 0.0


def test_painted_line_direction_no_line_returns_none():
    lanes._mask_to_markings = _fake_masks_to_markings([])
    assert painted_line_direction(_Sem({"road": np.zeros((8, 8), np.uint8)}),
                                  cam_model=None,
                                  pos=(10.0, 20.0, 1.5), heading=0.0) is None


def test_painted_line_direction_scattered_returns_none():
    """Dis-agreeing line points (no clear lane heading) must bail out."""
    lanes._mask_to_markings = _fake_masks_to_markings([_heading_line(
        90.0, 0.0, lon0=-4.0, lon1=4.0)])
    assert painted_line_direction(_Sem({"line": np.zeros((8, 8), np.uint8)}),
                                  cam_model=None,
                                  pos=(10.0, 20.0, 1.5), heading=0.0) is None


def test_polyline_dir_at_local_heading():
    """Nearest-segment heading of a lane centreline at the ego."""
    r = np.column_stack([
        np.linspace(0.0, 20.0, 21),
        np.full(21, 20.0),
    ])
    d = polyline_dir_at(r, (10.0, 20.0))
    assert d is not None
    assert abs(d[0] - 1.0) < 1e-9
    assert abs(d[1]) < 1e-9
    assert polyline_dir_at(r[:2], (10.0, 20.0)) is None
