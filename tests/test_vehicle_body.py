"""Ego footprint geometry: one authoritative body for every safety gate."""

from __future__ import annotations

import math

import numpy as np
import pytest

from beamng_autopilot import config, vehicle_body
from beamng_autopilot.planning import constraints


def test_footprint_follows_config():
    assert vehicle_body.HALF_LENGTH_M == pytest.approx(config.EGO_HALF_LENGTH_M)
    assert vehicle_body.HALF_WIDTH_M == pytest.approx(config.EGO_HALF_WIDTH_M)


def test_footprint_corners_axis_aligned():
    corners = vehicle_body.footprint_corners((1.0, 2.0), 0.0)
    hl, hw = config.EGO_HALF_LENGTH_M, config.EGO_HALF_WIDTH_M
    assert corners.shape == (4, 2)
    for got, want in zip(corners, [(1.0 + hl, 2.0 + hw),
                                   (1.0 + hl, 2.0 - hw),
                                   (1.0 - hl, 2.0 + hw),
                                   (1.0 - hl, 2.0 - hw)]):
        assert got == pytest.approx(want)


def test_footprint_corners_rotate_with_heading():
    # 90 deg: the car's length now runs along +y, its width along -x.
    corners = vehicle_body.footprint_corners((0.0, 0.0), math.pi / 2.0)
    hl, hw = config.EGO_HALF_LENGTH_M, config.EGO_HALF_WIDTH_M
    assert corners[0] == pytest.approx((-hw, hl))
    assert corners[1] == pytest.approx((hw, hl))
    assert corners[2] == pytest.approx((-hw, -hl))
    assert corners[3] == pytest.approx((hw, -hl))


def test_body_crosses_boundary_now_uses_full_body():
    # Lane edge (left boundary) parallel to travel, 1.0 m left of a
    # centred ego: a 0.9 m half width fits, a 0.4 m offset does not.
    left = np.array([[0.0, 1.0], [30.0, 1.0]])
    assert not vehicle_body.body_crosses_boundary_now(
        (5.0, 0.0), 0.0, left, None)
    assert vehicle_body.body_crosses_boundary_now(
        (5.0, 0.4), 0.0, left, None)
    # Yaw alone is enough: a 30 deg yawed nose corner reaches past a
    # boundary the centre point never touches.
    assert vehicle_body.body_crosses_boundary_now(
        (5.0, 0.0), math.radians(30.0), left, None)


def test_body_crossing_ignores_line_end():
    # A boundary that stops short of the ego (paint ends at an
    # intersection) must not be reported as a crossing.
    left = np.array([[10.0, 0.6], [30.0, 0.6]])
    assert not vehicle_body.body_crosses_boundary_now(
        (2.0, 0.0), 0.0, left, None)


def test_swept_check_covers_the_gap_between_waypoints():
    """A poke between two waypoints is caught by the sweep.

    The path is a straight 4 m-sampled line; the boundary poke sits where
    only an interpolated body pose reaches it.  Coarse (waypoint-only)
    sampling misses it - which is exactly the old behaviour.
    """
    path = np.array([[0.0, 0.0], [4.0, 0.0], [8.0, 0.0], [12.0, 0.0]])
    poke = np.array([[9.0, 0.8], [9.4, 0.8]])
    coarse = vehicle_body.first_boundary_crossing_m(
        (0.0, 0.0), path, poke, None, step_m=8.0)
    assert coarse == 0.0
    swept = vehicle_body.first_boundary_crossing_m(
        (0.0, 0.0), path, poke, None)
    assert swept > 0.0
    # The crossing sits between the second and the last waypoint.
    assert 4.0 < swept < 12.0


def test_swept_check_stops_at_the_first_crossing():
    left = np.array([[0.0, 0.5], [30.0, 0.5]])
    path = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
    dist = vehicle_body.first_boundary_crossing_m(
        (0.0, 0.0), path, left, None)
    assert dist > 0.0
    # Later samples are far past the line; only the first is reported.
    assert dist < 5.0


def test_swept_check_starts_at_the_front_bumper():
    """No near-field hole just past the bumper.

    ``body_crosses_boundary_now`` owns the ground the car already
    covers: its rectangle reaches ``HALF_LENGTH_M`` ahead of the ego
    centre.  The sweep therefore starts at ``HALF_LENGTH_M``: with a
    boundary 0.5 m to the side, the first reported crossing is the first
    path sample past the bumper (2.25 m), where the old 2.5 m start
    reported the next one (2.5 m) instead.
    """
    hl = vehicle_body.HALF_LENGTH_M
    path = np.column_stack([np.linspace(0.0, 10.0, 41), np.zeros(41)])
    left = np.array([[0.0, 0.5], [6.0, 0.5]])
    dist = vehicle_body.first_boundary_crossing_m(
        (0.0, 0.0), path, left, None)
    assert hl <= dist < 2.5, dist
    # ...and the sweep window itself now starts at the bumper.
    assert vehicle_body.SWEEP_NEAR_M == pytest.approx(hl)


def test_no_boundary_never_crosses():
    path = np.array([[0.0, 0.0], [5.0, 0.0]])
    assert vehicle_body.first_boundary_crossing_m(
        (0.0, 0.0), path, None, None) == 0.0
    assert not vehicle_body.body_crosses_boundary_now(
        (0.0, 0.0), 0.0, None, None)


class _Scene:
    def __init__(self, left=None, right=None, pos=(0.0, 0.0), heading=0.0,
                 envelope=None):
        self.lane_left = left
        self.lane_right = right
        self.pos = np.asarray(pos, dtype=float)
        self.heading = heading
        self.lane_envelope = envelope


class _Envelope:
    def __init__(self, left, right):
        self.left = left
        self.right = right


def test_constraint_gates_share_the_configured_footprint():
    """Constraints must use the shared body, not their own constants."""
    assert constraints.body_pose_crosses_lane.__defaults__[0] == \
        pytest.approx(config.EGO_HALF_LENGTH_M)
    assert constraints.body_pose_crosses_lane.__defaults__[1] == \
        pytest.approx(config.EGO_HALF_WIDTH_M)
    assert constraints.body_lane_cross_dist_m.__defaults__[0] == \
        pytest.approx(config.EGO_HALF_LENGTH_M)
    assert constraints.body_lane_cross_dist_m.__defaults__[1] == \
        pytest.approx(config.EGO_HALF_WIDTH_M)


def test_constraints_read_lane_envelope_when_lines_are_absent():
    left = np.array([[0.0, 0.40], [30.0, 0.40]])
    scene = _Scene(envelope=_Envelope(left, None))
    assert constraints.body_pose_crosses_lane(scene, (5.0, 0.0), 0.0)
    path = np.array([[0.0, 0.0], [6.0, 0.0]])
    assert constraints.body_lane_cross_dist_m(scene, path) > 0.0


def test_constraints_body_gate_matches_vertex_only_on_straight_lane():
    left = np.array([[0.0, 1.2], [30.0, 1.2]])
    scene = _Scene(left=left)
    path = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
    assert constraints.body_pose_crosses_lane(scene, (0.0, 0.0), 0.0) is False
    assert constraints.body_lane_cross_dist_m(scene, path) == 0.0
