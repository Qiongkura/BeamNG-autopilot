"""Unit tests for perception-only painted-line lane placement.

``painted_line_lane_center`` measures the painted line's lateral offset
from the semantic mask (left = +) and returns the world position that
puts the car 1.5 m to the RIGHT of that line - no map-centre / offset
constant.  The CV back-projection (``_mask_to_markings``) is faked here
so the tests exercise the selection / confidence / shift logic directly.
"""

from __future__ import annotations

import numpy as np

from beamng_autopilot.vision import lanes
from beamng_autopilot.vision.lanes import (
    LaneMarking,
    painted_line_lane_center,
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
