"""Pairing regressions for US single-edge lane recovery."""

from __future__ import annotations

import numpy as np
import pytest

from beamng_autopilot.lane import pair_lane_markings
from beamng_autopilot.vision.lanes import LaneMarking
from beamng_autopilot.lane.pairing import _pair_perspective_valid


def _line(y: float, x0: float = 4.0, x1: float = 9.0,
          kind: str = "solid", color: str = "white") -> LaneMarking:
    world = np.column_stack([
        np.linspace(x0, x1, 8), np.full(8, y)])
    return LaneMarking(world=world, pixels=world.copy(),
                       color=color, kind=kind, confidence=0.75)


def test_near_left_painted_edge_is_trusted_in_strict_mode():
    debug = {}
    frame = pair_lane_markings([_line(2.3)], np.zeros(3), 0.0,
                               debug=debug)
    assert frame is not None
    assert frame.paired is False
    assert frame.confidence >= 0.50
    assert debug["mode"] == "mirror_left"


def test_far_yellow_center_line_builds_own_lane():
    world = np.column_stack([
        np.linspace(20.0, 32.0, 12), np.full(12, 1.8)])
    frame = pair_lane_markings([
        LaneMarking(world=world, pixels=world.copy(),
                    color="yellow", kind="solid", confidence=0.9)],
        np.zeros(3), 0.0)
    assert frame is not None
    assert frame.paired is True
    assert frame.span_m >= 6.0


def test_short_explicit_center_paint_fragment_builds_own_lane():
    """A real 2.1 m yellow dashed fragment is enough to keep strict
    control alive; generic white fragments keep the old 4 m floor."""
    world = np.column_stack([
        np.linspace(4.0, 6.1, 8), np.full(8, 0.05)])
    frame = pair_lane_markings([
        LaneMarking(world=world, pixels=world.copy(),
                    color="yellow", kind="dashed", confidence=0.42)],
        np.zeros(3), 0.0)
    assert frame is not None
    assert frame.paired is True
    assert frame.span_m == pytest.approx(2.1)


def test_short_white_fragment_does_not_become_center_paint():
    world = np.column_stack([
        np.linspace(4.0, 6.1, 8), np.full(8, 0.05)])
    frame = pair_lane_markings([
        LaneMarking(world=world, pixels=world.copy(),
                    color="white", kind="solid", confidence=0.9)],
        np.zeros(3), 0.0)
    assert frame is None or frame.span_m >= 4.0


def test_pair_perspective_accepts_far_vanishing_point():
    s = np.linspace(2.0, 20.0, 10)
    left = np.column_stack([s, 2.0 + 0.02 * s])
    right = np.column_stack([s, -1.5 + 0.01 * s])
    ok, reason = _pair_perspective_valid(left, right)
    assert ok
    assert reason in {"ok", "parallel_or_diverging"}


def test_pair_perspective_rejects_near_crossing():
    s = np.linspace(2.0, 20.0, 10)
    left = np.column_stack([s, 2.0 - 0.20 * s])
    right = np.column_stack([s, -1.5 + 0.20 * s])
    ok, reason = _pair_perspective_valid(left, right)
    assert not ok
    assert reason == "near_perspective_crossing"


def test_pair_perspective_rejects_unrelated_angles():
    s = np.linspace(2.0, 20.0, 10)
    left = np.column_stack([s, 1.5 + 0.65 * s])
    right = np.column_stack([s, -1.5 - 0.05 * s])
    ok, reason = _pair_perspective_valid(left, right)
    assert not ok
    assert reason == "perspective_angle"

    debug = {}
    frame = pair_lane_markings([_line(4.0)], np.zeros(3), 0.0,
                               debug=debug)
    assert frame is not None
    assert frame.confidence < 0.50


def test_centre_dash_under_the_car_wins_over_mirror_right():
    """Live 2026-09-19 frame_6: the car sat ON the centre dash (med ~0.0);
    read as white it fell through to a right-edge mirror that "centred"
    the car across the paint.  An explicit dashed candidate in the centre
    band must take the centre-line path instead."""
    right_edge = _line(-1.88, x0=4.0, x1=14.0, kind="solid")
    centre = _line(0.07, x0=4.0, x1=14.0, kind="dashed")
    debug = {}
    frame = pair_lane_markings([right_edge, centre], np.zeros(3), 0.0,
                               debug=debug)
    assert frame is not None
    assert debug["mode"] == "centre_line_own_lane"
    assert frame.paired is True


def test_pair_with_centre_paint_and_right_edge_is_the_own_lane():
    """Paint at +0.6 and right edge at -1.88 pair into the correct own
    lane (midpoint -0.64) - the pair path beats both the centre fallback
    and any mirror."""
    debug = {}
    frame = pair_lane_markings(
        [_line(-1.88), _line(0.6, kind="dashed")], np.zeros(3), 0.0,
        debug=debug)
    assert frame is not None
    assert debug["mode"] == "pair"
    assert frame.center[0, 1] == pytest.approx(-0.64, abs=0.05)
    assert frame.left is not None and frame.right is not None


def test_mirror_right_refused_when_it_crosses_the_centre_paint():
    """Belt and braces: when pairing fails (edge+paint too close to pair,
    too far to merge), a mirror whose centre lands on the oncoming side
    of an observed centre paint must be refused (fail closed)."""
    # right edge -1.88, yellow paint -0.9: 0.98 apart -> pair rejected by
    # width/geometry, not merged (gap > axis threshold? 0.98 < 1.2 merges
    # into an axis...) - use a paint far enough to avoid both: -0.2.
    right_edge = _line(-1.88, x0=4.0, x1=14.0, kind="solid")
    centre = _line(-0.2, x0=4.0, x1=14.0, kind="dashed", color="yellow")
    debug = {}
    frame = pair_lane_markings([right_edge, centre], np.zeros(3), 0.0,
                               debug=debug)
    if debug["mode"] == "mirror_right":
        # the mirror was taken: it must NOT have contradicted the paint
        assert frame is not None
        assert frame.center[0, 1] <= centre.world[0, 1]
    else:
        # centre path or refusal - both are safe outcomes
        assert "centre_contradiction" in debug.get("mirror_reject", "")             or debug["mode"] in ("centre_line_own_lane", "none", "pair")
