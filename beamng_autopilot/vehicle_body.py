"""Ego body geometry - one authoritative footprint for the whole stack.

The car's rectangle used to be spelled out in several places (the
candidate constraints, the safety monitor and the live drive loop each
had their own ``2.2 x 0.9``), so a change to the vehicle could leave one
gate checking a different body than the others.  Everything now comes
from :mod:`beamng_autopilot.config` and is projected here, once:
candidate feasibility, the safety monitor, the planner and the live loop
all test the SAME swept rectangle.

Pure geometry only.  Callers pass the detected lane boundaries (a
perception output) and get a crossing answer back - nothing in this
module reads a map route, a lane centre line or a lateral offset, per
the project rule that lateral reference may only come from perception.
"""

from __future__ import annotations

import math

import numpy as np

from beamng_autopilot import config

# Authoritative body halves (see config.EGO_HALF_*_M).
HALF_LENGTH_M = float(config.EGO_HALF_LENGTH_M)
HALF_WIDTH_M = float(config.EGO_HALF_WIDTH_M)

# Pose spacing of the swept-body check.  Consecutive path samples closer
# than this are checked as-is; wider gaps get interpolated poses so a
# yawed corner cannot slip through between two waypoints (the old check
# stamped the rectangle on the waypoints only).
SWEPT_STEP_M = HALF_LENGTH_M / 3.0

# Window (distance from the ego CENTRE) where the planned-path sweep
# looks.  The current pose is checked separately by
# :func:`body_crosses_boundary_now`, whose rectangle reaches exactly
# ``HALF_LENGTH_M`` ahead of the centre - so the sweep starts there and
# the union of the two checks covers the near field with no hole.  (It
# used to start at 2.5 m, which left the band just past the bumper,
# 2.2-2.5 m, unchecked until the car rolled into it.)
SWEEP_NEAR_M = HALF_LENGTH_M
SWEEP_FAR_M = 15.0

# Lateral detection margin added to the body half-width whenever a
# module asks "does this cell intrude into the corridor the car is
# about to drive through?".  The corridor is always a *superset* of the
# body rectangle and is derived here, so no module can invent a second
# size (the live loop used to spell 1.5 / 1.6 / 2.0 out by hand, each
# from a different idea of how wide the car is).
CORRIDOR_MARGIN_M = 0.7
CORRIDOR_HALF_WIDTH_M = HALF_WIDTH_M + CORRIDOR_MARGIN_M


def footprint_corners(centre, heading: float,
                      half_len: float = HALF_LENGTH_M,
                      half_width: float = HALF_WIDTH_M) -> np.ndarray:
    """Four world-space corners of the ego rectangle.

    Row order is front-left, front-right, rear-left, rear-right, relative
    to ``heading`` (the car's forward axis).
    """
    p = np.asarray(centre, dtype=float).ravel()[:2]
    fwd = np.array([math.cos(float(heading)), math.sin(float(heading))])
    lft = np.array([-fwd[1], fwd[0]])
    return np.asarray([
        p + half_len * fwd + half_width * lft,
        p + half_len * fwd - half_width * lft,
        p - half_len * fwd + half_width * lft,
        p - half_len * fwd - half_width * lft,
    ])


def boundary_lateral(wx, wy, ref, fwd):
    """Signed lateral offset of a world point from a boundary polyline.

    Returns ``(lat, covered)`` - ``covered`` is False when the nearest
    polyline point is an endpoint (the boundary simply does not extend
    to that location: a painted line ends at an intersection / a lane
    change, so a path turning there must not be punished as a
    crossing).  Positive lat = left of travel.
    """
    pts = np.asarray(ref[:, :2], dtype=float)
    best = float("inf")
    sign = 0.0
    covered = False
    best_k = None
    best_t = 0.0
    for k in range(len(pts) - 1):
        ax, ay = pts[k]
        bx, by = pts[k + 1]
        abx, aby = bx - ax, by - ay
        l2 = abx * abx + aby * aby
        if l2 < 1e-12:
            continue
        t = float(((wx - ax) * abx + (wy - ay) * aby) / l2)
        tc = min(1.0, max(0.0, t))
        cx, cy = ax + tc * abx, ay + tc * aby
        d = math.hypot(wx - cx, wy - cy)
        # cross product of ref tangent and point offset
        s = float((abx * (wy - ay) - aby * (wx - ax)) / math.sqrt(l2))
        if d < best:
            best = d
            sign = s
            covered = 0.02 < t < 0.98
            best_k = k
            best_t = t
    if best_k is not None and not covered:
        # The nearest point lies at an endpoint of the best segment.
        # Only the FIRST/LAST vertex of the whole polyline is a true
        # line end (paint stops at an intersection / lane change); an
        # *interior* vertex is just a bend of the same boundary, and a
        # crossing exactly at that bend must be caught too - otherwise
        # a path that cuts the line at a corner is not punished
        # (cross_right vertex repro 2026-08-22).
        if best_t <= 0.02 and best_k > 0:
            covered = True
        elif best_t >= 0.98 and best_k < len(pts) - 2:
            covered = True
    # No fwd-based sign flip: the boundary polylines are stored in
    # their own travel direction (map prior near->far, sensor lanes
    # along the ego's forward at capture time).  A lane crossing is a
    # WORLD constraint - never into the oncoming lane, never off the
    # road edge - independent of which way the car happens to point at
    # a bend (hairpin repro 2026-08-22: the flip inverted every
    # in-lane candidate into a "crossing" at the apex, so the planner
    # only had cross-lot arcs left and drove off the road).
    return sign, covered


def corners_cross_boundaries(corners, left, right,
                             max_cross_m: float = 0.05) -> bool:
    """Whether any body corner lies beyond a detected boundary.

    ``left`` is the boundary the car must stay left-of (its lateral
    offset must stay positive), ``right`` the one it must stay right-of.
    """
    if left is not None:
        for c in corners:
            lat, covered = boundary_lateral(
                float(c[0]), float(c[1]), left, None)
            if covered and lat > max_cross_m:
                return True
    if right is not None:
        for c in corners:
            lat, covered = boundary_lateral(
                float(c[0]), float(c[1]), right, None)
            if covered and lat < -max_cross_m:
                return True
    return False


def body_crosses_boundary_now(pos, heading: float, left, right,
                              half_len: float = HALF_LENGTH_M,
                              half_width: float = HALF_WIDTH_M,
                              max_cross_m: float = 0.05) -> bool:
    """Whether the CURRENT body rectangle crosses a detected boundary."""
    if left is None and right is None:
        return False
    p = np.asarray(pos, dtype=float).ravel()
    if p.size < 2 or not np.isfinite(p[:2]).all():
        return False
    corners = footprint_corners(p[:2], float(heading),
                                half_len, half_width)
    return corners_cross_boundaries(corners, left, right, max_cross_m)


def _vertex_tangents(pts: np.ndarray) -> np.ndarray:
    """Per-vertex path tangents (central difference, forward/back at ends)."""
    tv = np.empty_like(pts)
    tv[0] = pts[1] - pts[0]
    tv[-1] = pts[-1] - pts[-2]
    if len(pts) > 2:
        tv[1:-1] = pts[2:] - pts[:-2]
    return tv


def first_boundary_crossing_m(pos, path, left=None, right=None,
                              half_len: float = HALF_LENGTH_M,
                              half_width: float = HALF_WIDTH_M,
                              max_cross_m: float = 0.05,
                              near_m: float = SWEEP_NEAR_M,
                              far_m: float = SWEEP_FAR_M,
                              step_m: float = SWEPT_STEP_M) -> float:
    """First along-path distance where the swept body crosses a boundary.

    The centreline-only gate is insufficient: a path can keep its centre
    inside the lane while a yawed car's front/rear corner crosses the
    line.  Every path pose carries the same body rectangle as the safety
    monitor and the candidate feasibility gate, aligned to the local path
    tangent; poses are interpolated every ``step_m`` so a corner cannot
    hide between two waypoints.  Poses within ``near_m`` of the ego are
    skipped (the current body is checked by
    :func:`body_crosses_boundary_now`) and poses beyond ``far_m`` are
    outside the sensor lane horizon.

    Returns 0.0 when no corner crosses or no boundaries exist.
    """
    if (left is None and right is None) or path is None:
        return 0.0
    pth = np.asarray(path, dtype=float)[:, :2]
    if len(pth) < 2:
        return 0.0
    origin = np.asarray(pos, dtype=float).ravel()[:2]
    tangents = _vertex_tangents(pth)
    step = float(step_m) if step_m and step_m > 0.0 else SWEPT_STEP_M
    cum = 0.0
    for i in range(len(pth) - 1):
        a, b = pth[i], pth[i + 1]
        seg = float(np.linalg.norm(b - a))
        if not np.isfinite(seg) or seg <= 1e-12:
            continue
        n = max(1, int(math.ceil(seg / step)))
        for k in range(1, n + 1):
            t = k / float(n)
            p = a + (b - a) * t
            d0 = float(np.linalg.norm(p - origin))
            if d0 < near_m or d0 > far_m:
                continue
            tv = (1.0 - t) * tangents[i] + t * tangents[i + 1]
            if float(np.linalg.norm(tv)) < 1e-9:
                tv = b - a
            heading = math.atan2(float(tv[1]), float(tv[0]))
            corners = footprint_corners(p, heading, half_len, half_width)
            if corners_cross_boundaries(corners, left, right, max_cross_m):
                return max(cum + t * seg, 0.1)
        cum += seg
    return 0.0
