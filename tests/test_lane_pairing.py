"""Pairing regressions for US single-edge lane recovery."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from beamng_autopilot.lane import pair_lane_markings
from beamng_autopilot.vision.lanes import LaneMarking
from beamng_autopilot.lane.pairing import (
    _pair_perspective_valid,
    _pair_world_geometry,
)


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


def _world_pair_geometry(left, right, pos=None, heading=0.0):
    pos = np.zeros(2) if pos is None else np.asarray(pos, dtype=float)
    fwd = np.array([np.cos(heading), np.sin(heading)])
    geometry = _pair_world_geometry(
        SimpleNamespace(world=left), SimpleNamespace(world=right),
        pos, fwd, 1.5, 18)
    assert geometry is not None
    return geometry


def _nearest_segment_oracle(points, polyline):
    """Scalar Euclidean projection, independent of the batched matcher."""
    projected = []
    covered = []
    lengths = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    for point in points:
        best_dist = float("inf")
        best_point = None
        best_station = 0.0
        station = 0.0
        for a, b, length in zip(polyline[:-1], polyline[1:], lengths):
            delta = b - a
            squared = float(delta @ delta)
            t = (float((point - a) @ delta) / squared
                 if squared > 0.0 else 0.0)
            t = min(1.0, max(0.0, t))
            q = a + t * delta
            distance = float(np.linalg.norm(point - q))
            if distance < best_dist:
                best_dist = distance
                best_point = q
                best_station = station + t * length
            station += length
        projected.append(best_point)
        covered.append(0.5 <= best_station <= float(lengths.sum()) - 0.5)
    return np.asarray(projected), np.asarray(covered)


@pytest.mark.parametrize("right_x", [
    [2.0, 18.0],
    [4.0, 15.0],
    [2.0, 2.0, 9.0, 18.0],
])
@pytest.mark.parametrize("heading_deg", [0.0, 37.0, 90.0, 180.0])
def test_world_pair_projection_matches_scalar_oracle(right_x, heading_deg):
    x = np.linspace(2.0, 18.0, 17)
    left = np.column_stack([x, np.full_like(x, 1.75)])
    right = np.column_stack([right_x, np.full(len(right_x), -1.75)])
    heading = np.radians(heading_deg)
    rotation = np.array([[np.cos(heading), -np.sin(heading)],
                         [np.sin(heading), np.cos(heading)]])
    origin = np.array([31.0, -7.0])
    left = left @ rotation.T + origin
    right = right @ rotation.T + origin
    _, center, matched_left, matched_right, valid = _world_pair_geometry(
        left, right, origin, heading)
    expected, covered = _nearest_segment_oracle(matched_left, right)
    np.testing.assert_allclose(matched_right, expected, atol=1e-10)
    np.testing.assert_array_equal(valid, covered)
    np.testing.assert_allclose(center, 0.5 * (matched_left + expected),
                               atol=1e-10)
    assert valid.any()
    assert not valid.all(), "clamped endpoints are not a two-sided read"


@pytest.mark.parametrize("heading_deg", [12.0, 45.0, 90.0, 180.0])
def test_world_pair_is_rotation_and_translation_equivariant(heading_deg):
    x = np.linspace(2.0, 18.0, 17)
    left = np.column_stack([x, np.full_like(x, 1.75)])
    right = np.column_stack([x, np.full_like(x, -1.75)])
    base = _world_pair_geometry(left, right)
    heading = np.radians(heading_deg)
    rotation = np.array([[np.cos(heading), -np.sin(heading)],
                         [np.sin(heading), np.cos(heading)]])
    origin = np.array([31.0, -7.0])
    moved = _world_pair_geometry(left @ rotation.T + origin,
                                 right @ rotation.T + origin,
                                 origin, heading)
    np.testing.assert_allclose(moved[0], base[0], atol=1e-10)
    np.testing.assert_array_equal(moved[4], base[4])
    for before, after in zip(base[1:4], moved[1:4]):
        np.testing.assert_allclose(after, before @ rotation.T + origin,
                                   atol=1e-10)
    valid = moved[4]
    np.testing.assert_allclose(
        np.linalg.norm(moved[2][valid] - moved[3][valid], axis=1), 3.5,
        atol=1e-10)


def test_world_pair_uses_true_curved_boundary_nearest_points():
    theta = np.linspace(0.08, 0.75, 17)
    radius = 20.0
    left = np.column_stack([
        (radius - 1.75) * np.sin(theta),
        radius - (radius - 1.75) * np.cos(theta)])
    right = np.column_stack([
        (radius + 1.75) * np.sin(theta),
        radius - (radius + 1.75) * np.cos(theta)])
    _, _, matched_left, matched_right, valid = _world_pair_geometry(left, right)
    expected, covered = _nearest_segment_oracle(matched_left, right)
    np.testing.assert_allclose(matched_right, expected, atol=1e-10)
    np.testing.assert_array_equal(valid, covered)
    assert valid.any()
