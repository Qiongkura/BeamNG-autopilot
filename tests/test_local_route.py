"""Offline tests for the nav-route local window (planning/local_route.py).

Regression (town runs 2026-08-22, g8/g10): the old window walked the
route BACKWARDS at a sharp corner vertex, truncated the local route to a
few metres, the planner no longer saw the turn and the car drove
straight through the intersection.
"""
from __future__ import annotations

import math

import numpy as np

from beamng_autopilot.planning.local_route import (
    local_route,
    map_lane_local,
)


def _l_route():
    """An L-shaped route: along +x to (10,0), then north along x=10."""
    a = np.column_stack([np.linspace(0, 10, 21), np.zeros(21)])
    b = np.column_stack([np.full(21, 10.0), np.linspace(0.5, 20, 21)])
    return np.vstack([a, b])


def _heading_bend(seg, k0=0, k1=2, k2=8):
    if len(seg) <= k2:
        return 0.0
    a = seg[k1] - seg[k0]
    b = seg[k2] - seg[k1]
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return math.degrees(math.acos(float(np.clip(np.dot(a, b) / (na * nb), -1, 1))))


def test_local_route_straight_keeps_forward_window() -> None:
    r = np.column_stack([np.linspace(0, 60, 61), np.zeros(61)])
    seg = local_route(np.array([20.0, 0.0]), 0.0, r, ahead_m=32.0)
    assert len(seg) >= 15
    # all points ahead of (or at) the ego along +x
    assert float(seg[-1, 0]) > 45.0
    assert float(seg[0, 0]) >= 18.0


def test_local_route_at_corner_keeps_the_turn() -> None:
    """At the corner vertex the local window must still contain the
    post-turn points (the old backwards walk truncated to the entry)."""
    r = _l_route()
    pos = np.array([9.6, 0.0])           # just before the corner
    seg = local_route(pos, 0.0, r, ahead_m=40.0)
    assert len(seg) >= 15, f"corner window collapsed: {len(seg)} pts"
    # the window must reach well past the corner into the vertical leg
    assert float(np.max(seg[:, 1])) > 8.0
    # and the route must still bend inside the window
    assert _heading_bend(seg) > 30.0


def test_local_route_no_reverse_walk() -> None:
    """The ego may sit exactly on the corner vertex with the exit
    segment initially 'behind' the ego heading; the window must never
    return a route that walks backward along the entry leg."""
    r = _l_route()
    seg = local_route(np.array([10.0, 0.0]), 0.0, r, ahead_m=40.0)
    assert len(seg) >= 10
    # no point of the window should lie far behind the ego along -x
    assert float(np.min(seg[:, 0])) >= 8.0


def test_map_lane_local_own_lane_right_of_centreline() -> None:
    r = _l_route()
    out = map_lane_local(r, np.array([5.0, 0.0]), 0.0)
    assert out is not None
    center, left, right = out
    assert len(center) >= 4 and len(left) >= 4 and len(right) >= 4
    # own lane: the reference starts AT the ego (car-anchored blend - the
    # car sits on the centreline at the start) and converges to half a
    # lane width right of the road centreline over CENTER_BLEND_M; check
    # the FAR part of the window converges.
    assert float(np.linalg.norm(center[0] - np.array([5.0, 0.0]))) < 0.5
    offs = []
    for p in center[-8:-2]:
        d = float(np.min(np.linalg.norm(left[:, :2] - p, axis=1)))
        offs.append(d)
    cl = float(np.median(offs))
    assert 1.5 <= cl <= 2.0, cl
    # road width: the right edge is one lane width right of the centreline
    rl = float(np.median([float(np.min(np.linalg.norm(left[:, :2] - p, axis=1))) for p in right[:, :2]]))
    assert 3.0 <= rl <= 4.2, rl
    # centreline (left boundary) stays on the road axis; the own lane is
    # on the RIGHT of the (possibly corner-rounded) centreline - with the
    # full-route fillet the road already curves before the corner, so the
    # own lane is measured relative to the rounded centreline, not to the
    # raw +x axis.
    assert abs(float(left[1, 1])) < 0.5
    assert float(right[1, 1]) < -2.5


def test_map_lane_local_none_without_route() -> None:
    assert map_lane_local(None, np.array([0.0, 0.0]), 0.0) is None


def test_map_lane_local_ego_anchor_does_not_flip_boundaries() -> None:
    """Regression (mountain stall 2026-08-22 at (750.5,742.5)): when the
    local route prepends the EGO vertex (car >~2 m off the centreline),
    that vertex must not participate in the boundary-normals smoothing.
    Otherwise the first smoothed normal is a cross-field connector to
    the car, the left/right boundaries flip onto the wrong side of the
    road, and every in-lane lane_shift candidate is rejected as a
    boundary crossing (the planner then only has off-road arcs and the
    car stops at the corner).
    """
    r = _l_route()
    # Car sits 2.5 m RIGHT of the +x centreline - the OWN lane side for
    # right-hand traffic (travel +x -> right = -y).  That is far enough
    # off the centreline for local_route() to prepend the ego anchor.
    pos = np.array([5.0, -2.5])
    seg = local_route(pos, 0.0, r, ahead_m=40.0)
    assert float(np.linalg.norm(seg[0] - pos)) < 0.01   # ego prepended
    out = map_lane_local(seg, pos, 0.0)
    assert out is not None
    center, left, right = out
    # Boundaries must be built from ROAD vertices only: the first left
    # vertex is the actual centreline (y ~ 0), not the ego anchor, and
    # the right edge stays on the -y side of the road.
    assert abs(float(left[0, 1])) < 0.5
    assert abs(float(right[0, 1]) - (-3.5)) < 0.5
    # The ego must sit BETWEEN the (unflipped) boundaries: right of the
    # centreline (lat_l < 0) and left of the road edge (lat_r > 0).
    from beamng_autopilot.planning.constraints import _boundary_lateral
    fwd = np.array([1.0, 0.0])
    lat_l, cov_l = _boundary_lateral(*pos, left, fwd)
    lat_r, cov_r = _boundary_lateral(*pos, right, fwd)
    assert cov_l and cov_r
    assert lat_l <= 0.0 <= lat_r, (lat_l, lat_r)
    # lane centre starts at the car (drivable planner start) and the
    # own lane stays between the two boundaries
    assert float(np.linalg.norm(center[0] - pos)) < 2.0
    assert float(np.linalg.norm(center[3] - left[3])) > 1.4
    assert float(np.linalg.norm(right[3] - left[3])) > 2.8
