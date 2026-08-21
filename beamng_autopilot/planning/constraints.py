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
    # Paths whose sampled cells are > this fraction occupied are infeasible.
    collision_fraction_max: float = 0.15
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
        feasible = True
        cost = 0.0
        col = cost_collision(scene, path, self.collision_fraction_max)
        # A path whose samples are mostly inside occupied cells is a
        # collision.  BUT a scattered cluster of roadside obstacles that
        # leaves the corridor open must not kill every candidate: the car
        # can pick another arc.  Only when the forward corridor is truly
        # closed (a full lateral band occupied across several bands) is
        # the path set infeasible.
        if col >= 1.0 and not corridor_free_band(
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
    """Median lateral distance of the path's near segment from the route.

    Positive mean = the path runs right of the nav route in the ego frame;
    the route is the lane centre, so large values mean the candidate left
    the lane.
    """
    route = np.asarray(scene.route[:, :2], dtype=float)
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