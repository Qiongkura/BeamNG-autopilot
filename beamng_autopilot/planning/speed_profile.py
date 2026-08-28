"""Trajectory speed profiling - the longitudinal half of FSD planning.

A trajectory is not just a path: FSD plans a matching speed profile
along it (slow into a bend, cruise on a straight, brake before an
obstacle).  This module adds that to the planning package:

* ``speed_profile_for_path`` - given a path and the scene (occupancy
  grid + lane curv+target), returns a per-point maximum speed in m/s.
  Curvature-limited (comfort lateral accel), obstacle-limited (brake
  band ahead of the closest obstacle) and capped by the target.

The existing speed controllers stay; this is the *planning-side* speed
that a candidate trajectory carries, so the whole pipeline
(selector -> safety monitor -> control) can act on one coherent
"trajectory + speed" plan - exactly the FSD data structure.
"""

from __future__ import annotations

import math

import numpy as np

# Comfort lateral acceleration limit for curvature speed (m/s^2).
COMFORT_LAT = 2.0
# How far ahead we start braking for an obstacle (m).
OBSTACLE_BRAKE_M = 25.0
# Half-width of the path corridor that can actually limit speed.  Roadside
# walls/buildings sit OUTSIDE the driven corridor: slowing to 1 m/s for a
# wall 2 m beside the lane is not FSD behaviour and it parks the car in
# town (2026-08-22 runs: every path in a wall-lined street got a 1.0 m/s
# profile because the closest occupied cell was a roadside wall).  Only
# occupancy that intrudes into this corridor is a real obstacle ahead.
CORRIDOR_HALF_WIDTH_M = 2.0
MIN_SPEED = 1.0
MAX_SPEED = 40.0


def speed_profile_for_path(path, scene, target_speed: float = 12.0,
                           comfort_lat: float = COMFORT_LAT,
                           obstacle_brake_m: float = OBSTACLE_BRAKE_M,
                           corridor_half_width_m: float = CORRIDOR_HALF_WIDTH_M,
                           obstacle_min_speed: float = MIN_SPEED,
                           ) -> np.ndarray:
    """Per-point max speed (m/s) along ``path``.

    ``path`` is (N, 2) world polyline.  Three factors, all coherent with
    the fused scene:

    * curvature: lat = sqrt(a_lat * r) so a tight bend is slow;
    * obstacle: a brake band ahead of the first obstacle that actually
      intrudes into the path corridor (within ``corridor_half_width_m``
      laterally), NOT any occupied cell within 25 m - roadside walls and
      buildings must not pin the profile to 1 m/s in town.  The band
      eases to ``obstacle_min_speed`` (default ``MIN_SPEED``): callers
      with corridor knowledge (the safety monitor already verified a
      free band exists) raise this floor so dense junction LiDAR does
      not pin the plan to a 1 m/s crawl;
    * target: caps the cruise speed.

    Returns an array of length N.
    """
    path = np.asarray(path, dtype=float)[:, :2]
    n = len(path)
    if n < 4:
        return np.full(n, float(target_speed), dtype=np.float32)
    target = min(float(target_speed), MAX_SPEED)
    v = np.full(n, target, dtype=np.float32)

    # --- curvature-limited speed ---------------------------------------
    A = float(comfort_lat)
    for i in range(1, n - 1):
        a = path[i - 1]
        b = path[i]
        c = path[i + 1]
        v1 = b - a
        v2 = c - b
        cr = (v1[0] * v2[1] - v1[1] * v2[0])
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        # Polyline curvature at a vertex (1/m): 2*|cross|/(|v1||v2|(|v1|+
        # |v2|)) equals sin(dtheta)/s for equal-length segments s, i.e.
        # the true circle curvature dtheta/s (the older ``|cross|/(n1*n2)``
        # was dimensionless sin(dtheta) and made the speed limit depend on
        # the sampling step - on the 0.6 m-rounded first hairpin it read
        # ~6.3 m/s for an 8 m radius instead of ~4.9).
        curvature = 2.0 * abs(cr) / (n1 * n2 * (n1 + n2))
        if curvature > 1e-6:
            v[i] = min(v[i], math.sqrt(A / curvature))
    v[0] = min(v[0], v[1])
    v[-1] = min(v[-1], v[-2])

    # --- obstacle-limited speed (brake band) ---------------------------
    obstacles = _scene_obstacle_points(scene, path, corridor_half_width_m)
    if obstacles is not None and len(obstacles) and len(obstacles[0]):
        occ = obstacles[0]
        d = np.linalg.norm(path[:, None, :] - occ[None, :, :], axis=2)
        for i in range(n):
            near = float(d[i].min()) if d.shape[1] else float("inf")
            if near < obstacle_brake_m:
                f = max(0.0, near / obstacle_brake_m)
                v[i] = min(v[i], max(obstacle_min_speed, target * f))

    # --- look-ahead braking -------------------------------------------
    # The per-point limits above apply AT the bend/obstacle; the plan
    # must brake BEFORE them, or the car arrives at the corner at cruise
    # speed (mountain hairpin 2026-08-22: ``best_speed = v[0]`` stayed
    # 8 m/s while the apex 6 m ahead was limited to ~1.7 m/s, so the
    # throttle kept accelerating into a 5-8 m radius hairpin, the car ran
    # wide off the road and parked itself beside it).  Propagate each
    # point's limit back over the next ``lookahead_m`` metres so the
    # entry speed of a bend is already the bend speed.
    lookahead_m = 12.0
    arc = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    for i in range(n):
        hi = int(np.searchsorted(arc, arc[i] + lookahead_m))
        v[i] = float(np.min(v[i: hi + 1]))
    return v


def _scene_obstacle_points(scene, path, corridor_half_width_m: float):
    """World-space occupied cells that actually intrude into the lane
    corridor (from the scene grid).

    An occupied cell only counts when its centre lies within
    ``corridor_half_width_m`` laterally of the path polyline - the walls
    that line a town road are obstacles the path must not cross, but they
    are NOT in the lane and must not trigger the speed brake band.  This
    mirrors ``path_grid_clearance_m`` (the safety layer measures the same
    corridor); the profile then brakes on the along-path distance.

    Vectorized: the old per-cell Python loop over thousands of LiDAR
    occupied cells cost ~1 s per frame on the full-route profile (the
    dominant "rest" stall of the FSD drive loop).  World transform and
    polyline lateral offsets are computed with numpy.
    """
    grid = getattr(scene, "grid", None)
    if grid is None:
        return None
    rr, cc = np.nonzero(getattr(grid, "obstacle", np.zeros((0, 0), dtype=np.uint8)))
    if len(rr) == 0:
        return None
    path = np.asarray(path, dtype=float)[:, :2]
    if len(path) < 2:
        return None
    ch = math.cos(grid.heading)
    sh = math.sin(grid.heading)
    ex = grid.max_x - (rr + 0.5) * grid.res
    ey = grid.max_y - (cc + 0.5) * grid.res
    wx = grid.origin[0] + ex * ch - ey * sh
    wy = grid.origin[1] + ex * sh + ey * ch
    pts = np.column_stack((wx, wy))
    lat = _points_path_lat(pts, path)
    sel = np.abs(lat) <= corridor_half_width_m
    if not np.any(sel):
        return None
    return [np.asarray(pts[sel], dtype=float)]


def _points_path_lat(points: np.ndarray, path: np.ndarray) -> np.ndarray:
    """Lateral offset of N points from a polyline, vectorized.

    Equivalent to ``_point_path_lat`` per point but O(segments) Python
    iterations over N-element numpy arrays instead of an O(N * segments)
    Python double loop.  Returns an N-array (inf never occurs for a
    valid polyline).
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim == 1:
        pts = pts[None, :]
    seg = path[1:] - path[:-1]
    l2 = np.einsum("ij,ij->i", seg, seg)
    lat = np.full(len(pts), np.inf, dtype=float)
    for k in range(len(seg)):
        s2 = float(l2[k])
        if s2 < 1e-12:
            continue
        ax = path[k]
        rel = pts - ax
        t = (rel[:, 0] * seg[k, 0] + rel[:, 1] * seg[k, 1]) / s2
        tc = np.clip(t, 0.0, 1.0)
        cx = ax[0] + tc * seg[k, 0]
        cy = ax[1] + tc * seg[k, 1]
        d = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        lat = np.minimum(lat, d)
    return lat


def _point_path_lat(wx: float, wy: float, path) -> float | None:
    """Lateral offset of a point from a polyline (None when far from every
    segment - then it cannot be inside the corridor)."""
    best = float("inf")
    for k in range(len(path) - 1):
        ax, ay = path[k]
        bx, by = path[k + 1]
        abx, aby = bx - ax, by - ay
        l2 = abx * abx + aby * aby
        if l2 < 1e-12:
            continue
        t = float(((wx - ax) * abx + (wy - ay) * aby) / l2)
        tc = min(1.0, max(0.0, t))
        cx, cy = ax + tc * abx, ay + tc * aby
        d = math.hypot(wx - cx, wy - cy)
        if d < best:
            best = d
        if best < 1e-9:
            break
    return best if best < float("inf") else None
