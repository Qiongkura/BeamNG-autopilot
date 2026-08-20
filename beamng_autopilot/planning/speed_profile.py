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
COMFORT_LAT = 3.0
# How far ahead we start braking for an obstacle (m).
OBSTACLE_BRAKE_M = 25.0
MIN_SPEED = 1.0
MAX_SPEED = 40.0


def speed_profile_for_path(path, scene, target_speed: float = 12.0,
                           comfort_lat: float = COMFORT_LAT,
                           obstacle_brake_m: float = OBSTACLE_BRAKE_M
                           ) -> np.ndarray:
    """Per-point max speed (m/s) along ``path``.

    ``path`` is (N, 2) world polyline.  Three factors, all coherent with
    the fused scene:

    * curvature: v = sqrt(a_lat * r) so a tight bend is slow;
    * obstacle: brake band ahead of the closest occupied cell;
    * target: never exceed the planner's cruise speed.

    Returns an array of length N.
    """
    path = np.asarray(path, dtype=float)[:, :2]
    n = len(path)
    if n < 4:
        return np.full(n, float(target_speed), dtype=np.float32)
    target = min(float(target_speed), MAX_SPEED)
    v = np.full(n, target, dtype=np.float32)

    # --- curvature-limited speed --------------------------------------
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
        curvature = abs(cr) / (n1 * n2)
        if curvature > 1e-6:
            v[i] = min(v[i], math.sqrt(A / curvature))
    v[0] = min(v[0], v[1])
    v[-1] = min(v[-1], v[-2])

    # --- obstacle-limited speed (brake band) --------------------------
    obstacles = _scene_obstacle_points(scene, path)
    if obstacles is not None and len(obstacles) and len(obstacles[0]):
        occ = obstacles[0]
        # along-path distance from each sample to the nearest obstacle
        d = np.linalg.norm(path[:, None, :] - occ[None, :, :], axis=2)
        for i in range(n):
            near = float(d[i].min()) if d.shape[1] else float("inf")
            # ease below cruise once inside the brake band
            if near < obstacle_brake_m:
                f = max(0.0, near / obstacle_brake_m)
                v[i] = min(v[i], max(MIN_SPEED, target * f))
    return v


def _scene_obstacle_points(scene, path):
    """World-space occupied cells near the path (from the scene grid)."""
    grid = getattr(scene, "grid", None)
    if grid is None:
        return None
    rr, cc = np.nonzero(getattr(grid, "obstacle", np.zeros((0, 0), dtype=np.uint8)))
    if len(rr) == 0:
        return None
    pts = []
    for r, c in zip(rr, cc):
        ex = grid.max_x - (r + 0.5) * grid.res
        ey = grid.max_y - (c + 0.5) * grid.res
        ch = math.cos(grid.heading)
        sh = math.sin(grid.heading)
        wx = grid.origin[0] + ex * ch - ey * sh
        wy = grid.origin[1] + ex * sh + ey * ch
        pts.append((wx, wy))
    return [np.asarray(pts, dtype=float)]