"""Constraint / cost evaluation for candidate trajectories.

Each cost function rates one ``Candidate`` against the ``Scene`` and
returns a non-negative scalar; the selector sums weighted costs and
applies hard feasibility (collision / off-road) so infeasible
candidates are dropped before ranking.  All functions are pure and
game-free - they read only the Scene (occupancy grid, lane reference,
ego pose).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Constraints:
    """Weights and feasibility thresholds for trajectory scoring."""

    w_collision: float = 5.0
    w_curvature: float = 1.0
    w_lane_align: float = 2.0
    w_progress: float = 0.01
    # Paths that CROSS a detected lane boundary (solid paint / map
    # boundary) are never drivable - a real stack never crosses the
    # centre line (wrong-way / 逆行 is not allowed, even for one
    # second).  Only paths that stay on the legal side of both
    # detected boundaries are feasible.
    lane_cross_max_m: float = 0.35
    # Paths whose sampled cells are > this fraction occupied are "blocked".
    collision_fraction_max: float = 0.15
    # A candidate is NEVER drivable when more than this fraction of its
    # samples lie inside occupied cells, even if a lateral corridor slice
    # is open.  The corridor-free-band relaxation only stops a scattered
    # roadside cluster from declaring "no drivable path"; it must not let
    # the planner pick a trajectory that is almost entirely inside
    # obstacles (town runs 2026-08-21: the planner chose a path with 100%
    # of its samples in trees and the car crawled at 1 m/s through them).
    collision_fraction_stop: float = 0.9
    # Paths that leave the nav route farther than this are rejected.
    lane_dev_max_m: float = 4.0
    # Spatial-connectivity gate: a candidate is only fully infeasible when
    # the forward corridor keeps no free lateral band OF THIS WIDTH across
    # THIS MANY longitudinal bands.  Scattered roadside clusters must not
    # declare "no drivable path".
    corridor_clear_m: float = 3.0
    corridor_bands: int = 8
    # A trajectory that does not actually move the vehicle ahead is not a
    # drivable path - a mis-anchored map reference sitting behind/ beside
    # the ego used to stay feasible and push the car into a wall (town
    # runs 2026-08-21).  Require at least this much forward net progress
    # (metres of displacement along the ego heading).
    progress_min_m: float = 1.0
    # Candidate starts farther than this from the ego are considered
    # mis-anchored map priors (not drivable from where the car is).
    starts_near_m: float = 3.0

    def _has_forward_progress(self, scene: Scene, path) -> bool:
        path = np.asarray(path, dtype=float)[:, :2]
        pos = np.asarray(scene.pos[:2], dtype=float)
        ch = math.cos(float(scene.heading))
        sh = math.sin(float(scene.heading))
        d0 = math.hypot(path[0, 0] - pos[0], path[0, 1] - pos[1])
        if d0 > self.starts_near_m:
            return False
        # net forward displacement from the path start
        dx = path[-1, 0] - path[0, 0]
        dy = path[-1, 1] - path[0, 1]
        return bool(dx * ch + dy * sh >= self.progress_min_m)

    def score(self, scene: Scene, candidate) -> tuple[float, bool]:
        """Return (cost, feasible)."""
        path = candidate.path
        if path is None or len(path) < 2:
            return 1e9, False
        if not self._has_forward_progress(scene, path):
            return 1e9, False
        if lane_cross_dist_m(scene, path, max_cross_m=self.lane_cross_max_m) > 0.0:
            return 1e9, False
        feasible = True
        cost = 0.0
        col = cost_collision(scene, path, self.collision_fraction_max)
        # A path whose samples are mostly inside occupied cells is a
        # collision and is never drivable.  The corridor-free-band
        # relaxation below only protects a scattered roadside cluster from
        # declaring "no drivable path" - it must not let a candidate that
        # runs almost entirely through obstacles stay feasible.
        _bad, _tot = _path_infractions(scene, path)
        _frac = (_bad / _tot) if _tot else 0.0
        if _frac > self.collision_fraction_stop:
            feasible = False
        elif col >= 1.0 and not corridor_free_band(
                scene, min_clear_m=self.corridor_clear_m,
                bands=self.corridor_bands):
            feasible = False
        cost += self.w_collision * col
        cost += self.w_curvature * cost_curvature(path)
        if scene.route is not None and len(scene.route) >= 2:
            align = cost_lane_align(scene, path)
            if align > self.lane_dev_max_m:
                feasible = False
            cost += self.w_lane_align * align
        cost += self.w_progress * (self._length(path))
        return cost, feasible

    @staticmethod
    def _length(path) -> float:
        d = np.diff(np.asarray(path, dtype=float), axis=0)
        return float(np.sum(np.linalg.norm(d, axis=1))) if len(d) else 0.0


def _path_infractions(scene: Scene, path, span: float = 2.0) -> list:
    """Sample the path (skip the car's own footprint) and return the
    fraction of samples inside an occupied grid cell."""
    path = np.asarray(path, dtype=float)[:, :2]
    if scene.grid is None or len(path) < 2:
        return []
    pos = np.asarray(scene.pos[:2], dtype=float)
    extent = float(getattr(scene.grid, "extent", 0.0) or 0.0)
    bad = 0
    total = 0
    for x, y in path:
        d = math.hypot(x - pos[0], y - pos[1])
        if d < 2.5:
            continue
        # Only cells inside the sensor grid carry collision evidence;
        # beyond the FOV horizon is unknown, not a wall.
        if extent > 0.0 and d > extent:
            continue
        total += 1
        cell = scene.grid.world_to_cell(x, y)
        if cell is None:
            continue
        if scene.grid.obstacle[cell] > 0:
            bad += 1
    return [bad, total]


def _boundary_lateral(wx, wy, ref, fwd):
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
    # fwd vs ref tangent sign flip: positive lateral in the ref frame
    # stays positive only when the ref runs the same direction as fwd
    v0 = pts[min(1, len(pts) - 1)] - pts[0]
    n0 = float(np.linalg.norm(v0))
    if n0 > 1e-9:
        if float(v0[0] * fwd[0] + v0[1] * fwd[1]) < 0.0:
            sign = -sign
    return sign * best, covered



def lane_cross_dist_m(scene: Scene, path, max_cross_m: float = 0.35) -> float:
    """Distance along the path at which it first crosses a lane boundary.

    Returns > 0 when the path crosses a detected left/right boundary
    (``scene.lane_left`` / ``scene.lane_right``) at a lateral intrusion
    deeper than ``max_cross_m`` beyond the boundary - a hard no-cross
    rule.  Returns 0 when no boundary is detected or no crossing exists.
    Only samples within 2.5-15 m of the ego are checked (beyond the car
    footprint and within the sensor lane horizon); the boundary coverage
    flag (``_boundary_lateral``) ensures a path turning at a line ending
    at an intersection is not falsely rejected.
    """
    left = getattr(scene, "lane_left", None)
    right = getattr(scene, "lane_right", None)
    if left is None and right is None:
        return 0.0
    path = np.asarray(path, dtype=float)[:, :2]
    if len(path) < 2:
        return 0.0
    pos = np.asarray(scene.pos[:2], dtype=float)
    fwd = np.array([math.cos(float(scene.heading)),
                    math.sin(float(scene.heading))])
    cum = 0.0
    for i in range(len(path) - 1):
        ax, ay = path[i]
        bx, by = path[i + 1]
        seg = math.hypot(bx - ax, by - ay)
        if seg < 1e-9:
            continue
        n = max(2, int(seg / 0.5) + 1)
        for k in range(n):
            t = k / (n - 1) if n > 1 else 0.0
            px, py = ax + t * (bx - ax), ay + t * (by - ay)
            d0 = math.hypot(px - pos[0], py - pos[1])
            if d0 < 2.5 or d0 > 15.0:
                continue
            lat_l = cov_l = 0.0
            lat_r = cov_r = 0.0
            if left is not None:
                lat_l, cov_l = _boundary_lateral(px, py, left, fwd)
            if right is not None:
                lat_r, cov_r = _boundary_lateral(px, py, right, fwd)
            # Left boundary: legal lane lies right of it (lat <= 0);
            # crossing left into oncoming traffic is forbidden.  Right
            # boundary: legal lane lies left of it (lat >= 0); crossing
            # right off the road edge is forbidden.  Only a boundary that
            # actually extends to this sample (``covered``) can be
            # violated - a line ending at an intersection is not a wall
            # the turn must stop for.
            if (left is not None and cov_l and lat_l > max_cross_m) or \
               (right is not None and cov_r and lat_r < -max_cross_m):
                return cum + t * seg
        cum += seg
    return 0.0


def cost_collision(scene: Scene, path, max_frac: float = 0.15) -> float:
    """Collision cost: 0..1 scaled by how much of the path is inside an
    occupied cell (out-of-grid samples count as occupied = unknown)."""
    bad, total = _path_infractions(scene, path)
    if total == 0:
        return 0.0
    frac = bad / total
    if frac > max_frac:
        return 1.0
    return frac / max_frac


def corridor_free_band(scene: Scene, min_clear_m: float = 3.0,
                       bands: int = 8, max_blocked_bands: int = 3
                       ) -> bool:
    """True when the forward corridor keeps at least one laterally free
    band before the horizon.

    FSD plans over a drivable-space corridor, not just "does this one
    path hit a cell".  A horizontally scattered cluster of roadside
    obstacles still leaves an open lane between them; only a band whose
    occupied cells span the whole lateral width (or nearly all of it)
    actually blocks driving.  This is the spatial-connectivity gate that
    stops a transient LiDAR cluster from declaring "no drivable path":
    the corridor is only blocked when *several* consecutive longitudinal
    bands are fully closed.
    """
    grid = getattr(scene, "grid", None)
    if grid is None:
        return True
    occ = getattr(grid, "obstacle", None)
    if occ is None or occ.size == 0:
        return True
    n_rows = int(grid.n_rows)
    n_cols = int(grid.n_cols)
    if n_cols < 4:
        return True
    # longitudinal bands ahead of the ego (top rows of the grid = ahead)
    step = max(1, int(n_rows / bands))
    min_clear_cells = max(1, int(min_clear_m / max(1e-9, grid.res)))
    blocked = 0
    for r in range(0, min(n_rows, n_rows - 1), step):
        if r <= int(n_rows * 0.12):       # skip the ego's own band
            continue
        row = occ[r]
        # lateral free cells in this band
        free = int((row == 0).sum())
        if free < min_clear_cells:
            blocked += 1
    return blocked < max_blocked_bands


def cost_curvature(path, jerk_w: float = 0.0) -> float:
    """Mean curvature of the path (rad/m), a comfort cost."""
    path = np.asarray(path, dtype=float)[:, :2]
    if len(path) < 4:
        return 0.0
    angles = []
    for i in range(1, len(path) - 1):
        a = path[i - 1]
        b = path[i]
        c = path[i + 1]
        v1 = b - a
        v2 = c - b
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        angles.append(abs(math.acos(float(cos_a))) / max(n1, 1e-6))
    return float(np.mean(angles)) if angles else 0.0


def cost_lane_align(scene: Scene, path) -> float:
    """Median lateral distance of the path's near segment from the lane reference.

    Uses ``scene.lane_ref`` when available (the sensor lane centre), else
    ``scene.route`` (the nav route / map road centre).  The sensor lane
    centre is the correct centre of the ego lane (vision/LiDAR pairing),
    not the road centreline - a real FSD never aligns to the centre line
    of a two-way road.
    """
    route = getattr(scene, "lane_ref", None)
    if route is None or len(route) < 2:
        route = getattr(scene, "route", None)
    if route is None or len(route) < 2:
        return 0.0
    path = np.asarray(path, dtype=float)[:, :2]
    if len(route) < 2 or len(path) < 2:
        return 0.0
    pos = np.asarray(scene.pos[:2], dtype=float)
    # only evaluate the near part of the path (0..25 m ahead)
    d0 = np.linalg.norm(path - pos, axis=1)
    near = path[d0 <= 25.0]
    if len(near) < 2:
        near = path[: min(4, len(path))]
    # median distance to the nearest route segment
    offs = []
    for px, py in near:
        best = float("inf")
        for k in range(len(route) - 1):
            ax, ay = route[k]
            bx, by = route[k + 1]
            abx, aby = bx - ax, by - ay
            l2 = abx * abx + aby * aby
            if l2 < 1e-12:
                d = math.hypot(px - ax, py - ay)
            else:
                t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby)
                                 / l2))
                cx, cy = ax + t * abx, ay + t * aby
                d = math.hypot(px - cx, py - cy)
            if d < best:
                best = d
        offs.append(best)
    return float(np.median(offs)) if offs else 0.0