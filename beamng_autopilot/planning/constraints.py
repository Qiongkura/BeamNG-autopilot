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
    # Paths that leave the nav route farther than this are rejected.
    lane_dev_max_m: float = 4.0
    # --- drivable-surface gate -----------------------------------------
    # A real FSD never leaves the drivable road surface.  Grass / terrain
    # is NOT an obstacle cell (the collision layer alone cannot see it),
    # so the camera/LiDAR drivable layer is the road-surface authority:
    # candidates whose samples mostly fall outside drivable cells are
    # infeasible, and one that leaves drivable within the next few metres
    # is rejected outright (town runs 2026-08-26: the car drove onto the
    # grass / terrain at the first hairpin because only obstacles were
    # enforced).  The gate only activates when the drivable layer has
    # real evidence (a missing road mask is "unknown", not "grass").
    off_drivable_fraction_max: float = 0.15
    # Reference candidates (lane centre / route / lane shift) are only
    # drivable as TRACKING paths when the lane they track points
    # forward of the ego.  The measured direction is the lane TANGENT
    # (first real lane segment), not the ego-anchor diagonal: at a
    # hairpin the lane legitimately turns far more than 50 deg right in
    # front of the car, and a car sitting at the road edge must be able
    # to converge back along its own lane.  Only a reference whose near
    # lane points backward (~180 deg) is rejected; the lane-boundary and
    # drivable-surface gates still stop a bad map-lane from cutting
    # across the road (mountain run 2026-08-27 run_fix29: the old
    # first-segment 50 deg gate rejected the lane centre at the hairpin
    # apex, the planner fell back to arcs, over-rotated onto the wrong
    # side and stalled at the right edge).
    ref_start_yaw_max_deg: float = 120.0
    off_drivable_near_fraction_max: float = 0.25
    off_drivable_near_m: float = 8.0
    off_drivable_min_evidence: float = 0.03
    # A candidate that runs mostly through UNOBSERVED cells is not a
    # drivable path: unknown space carries no collision/off-road
    # evidence, so a looping arc that leaves the sensor footprint
    # passes every other gate and is picked with full throttle -
    # that is the in-place spin (mountain runs 2026-08-27 run_fix22:
    # at (741.0,745.8) the only 'feasible' arc pointed 57 deg off the
    # nose into cells the sensors never saw and the car spun on the
    # grass).  A real vector-space stack treats unknown as 'do not
    # drive here' unless the path follows the known route/lane.
    known_min_frac: float = 0.35
    known_near_m: float = 10.0
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
    # Ego-anchored reference candidates (own-lane centre, plan route,
    # lane shifts) get a lower bar.  At a hairpin apex the car can sit at
    # a large angle to its own lane: the first metres of the lane centre
    # still point sideways while the path curves back onto the road, and
    # a 1 m forward requirement rejects the correct rejoin path, leaving
    # only looping arcs (mountain run 2026-08-26 run_fix11: at
    # (725.3,755.2) heading -128 deg, lane_center/reference/lane_shift
    # were all prog-0.00 and the car circled off-road instead of turning
    # left into the hairpin).  0.35 m still rejects a reference that
    # points backward or sideways-away, which is what the gate exists for.
    progress_min_m_ref: float = 0.35
    # Candidate starts farther than this from the ego are considered
    # mis-anchored map priors (not drivable from where the car is).
    starts_near_m: float = 3.0
    # Candidate kinds that are derived from the ego-anchored lane/route
    # reference (not raw kinematic arcs) - they may use the relaxed
    # forward-progress bar above.
    ref_kinds: tuple = ("lane_center", "reference", "lane_shift")

    def _has_forward_progress(self, scene: Scene, path,
                              min_m: float | None = None) -> bool:
        path = np.asarray(path, dtype=float)[:, :2]
        pos = np.asarray(scene.pos[:2], dtype=float)
        ch = math.cos(float(scene.heading))
        sh = math.sin(float(scene.heading))
        d0 = math.hypot(path[0, 0] - pos[0], path[0, 1] - pos[1])
        if d0 > self.starts_near_m:
            return False
        # Maximum forward displacement over the FIRST HALF of the path.
        # A hairpin lane-shift / map-lane reference ENDS "behind" the ego
        # heading after the 90-degree bend, but it still drives forward
        # along the road; the old endpoint gate killed those candidates and
        # the planner was left with a single over-aggressive arc that cut
        # the corner and drove off-route (mountain run 2026-08-23,
        # run_fix7: t=5.7 n_eval=1 -> src=none stall at (720.2,743.8)).
        # Only the path's start is evidence that it is drivable from here.
        n = len(path)
        k = max(2, int(n * 0.5))
        rel = path[:k] - path[0]
        fwd = rel[:, 0] * ch + rel[:, 1] * sh
        return bool(float(fwd.max()) >= float(self.progress_min_m
                                              if min_m is None else min_m))

    def score(self, scene: Scene, candidate) -> tuple[float, bool]:
        """Return (cost, feasible)."""
        path = candidate.path
        if path is None or len(path) < 2:
            return 1e9, False
        kind = str(candidate.meta.get("kind", ""))
        min_m = self.progress_min_m_ref if kind in self.ref_kinds \
            else self.progress_min_m
        if not self._has_forward_progress(scene, path, min_m):
            return 1e9, False
        if kind in self.ref_kinds and len(path) >= 2:
            # Direction of the LANE just ahead of the ego, not the
            # ego-anchor diagonal.  References are re-anchored at the car
            # (path[0] == pos), so path[0]->path[1] is a cross-field line
            # to the nearest lane vertex - at a hairpin apex or with the
            # car sitting at the road edge that diagonal is 80-130 deg off
            # the nose even when the lane is correct.  Measuring it
            # rejected the lane centre exactly where the car had to turn,
            # leaving only arcs that over-rotated and drove off the road
            # (mountain run 2026-08-27 run_fix29: hairpin exit swung +30
            # deg wrong-way, then stalled at (741.2,746.0) on the right
            # edge).  The lane tangent (first real lane segment) is what
            # the reference actually tracks; a reference whose near lane
            # points backward still fails the same gate (~180 deg).
            if len(path) >= 3:
                _v = np.asarray(path[2], dtype=float)[:2] \
                    - np.asarray(path[1], dtype=float)[:2]
            else:
                _v = np.asarray(path[1], dtype=float)[:2] \
                    - np.asarray(path[0], dtype=float)[:2]
            _L = float(np.linalg.norm(_v))
            if _L > 1e-9:
                _a = math.degrees(math.atan2(_v[1], _v[0]))
                _d = (_a - math.degrees(float(scene.heading)) + 180.0) \
                    % 360.0 - 180.0
                if abs(_d) > self.ref_start_yaw_max_deg:
                    return 1e9, False
        if lane_cross_dist_m(scene, path, max_cross_m=self.lane_cross_max_m) > 0.0:
            return 1e9, False
        # Hard drivable-surface gate: never leave the road (grass/terrain
        # is not an obstacle cell, so the collision layer cannot catch it).
        _bdrv, _tdrv, _nbdrv, _ntdrv = _path_off_drivable(
            scene, path, near_m=self.off_drivable_near_m,
            min_evidence=self.off_drivable_min_evidence)
        if _tdrv:
            if (_bdrv / _tdrv) > self.off_drivable_fraction_max:
                return 1e9, False
            if _ntdrv >= 4 and \
                    (_nbdrv / _ntdrv) > self.off_drivable_near_fraction_max:
                return 1e9, False
        # Driving blind is not allowed.  A path that mostly crosses cells
        # the sensors never saw is not drivable - it is the loop-arc /
        # grass-escape failure mode, not a road.  Map-prior candidates
        # (lane centre / route reference / lane shift) are KNOWN geometry
        # from the nav route, not blind guesses - only the kinematic arc
        # fan must prove it stays inside the sensor footprint (mountain
        # runs 2026-08-27: the lane-following candidate was rejected with
        # blind0/12 while the sparse camera footprint saw only ~10% of
        # the BEV, leaving a single off-road loop arc as the only
        # "feasible" path and the car swung onto the grass at the
        # switchback).  The gate is skipped when the observed layer is
        # missing/empty (offline unit scenes).
        _kn, _kt = _path_known(scene, path, near_m=self.known_near_m)
        if kind not in self.ref_kinds and _kt >= 4 \
                and (_kn / _kt) < self.known_min_frac:
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

def _path_off_drivable(scene: Scene, path, near_m: float = 8.0,
                       min_evidence: float = 0.03) -> list:
    """Sample the path and count cells OUTSIDE the drivable road layer.

    Returns ``(bad, total, near_bad, near_total)``.  The drivable layer
    marks the road surface seen by the camera (semantic road mask lifted
    into BEV) / LiDAR corridor; a 0 cell is grass / terrain / unknown.
    The gate only activates when the layer actually saw a road
    (``drivable.sum() >= min_evidence * size``): a missing road mask
    means "unknown", not "grass", and must not park the car.
    """
    path = np.asarray(path, dtype=float)[:, :2]
    if scene.grid is None or len(path) < 2:
        return [0, 0, 0, 0]
    grid = scene.grid
    drv = getattr(grid, "drivable", None)
    if drv is None or drv.size == 0:
        return [0, 0, 0, 0]
    obs = getattr(grid, "observed", None)
    # Activation: the sensor must have SEEN enough of the world to tell
    # road from grass.  Unknown space (no observed evidence) is not
    # grass - a sparse camera footprint must not reject every candidate
    # (mountain run 2026-08-27 run_fix15: the drivable layer covered
    # only ~10% of the grid, so every straight-ahead path read "off
    # road" and the car never started).
    evid = obs if (obs is not None and obs.size == drv.size) else drv
    if float(np.asarray(evid).sum()) < float(min_evidence) * evid.size:
        return [0, 0, 0, 0]
    pos = np.asarray(scene.pos[:2], dtype=float)
    extent = float(getattr(grid, "extent", 0.0) or 0.0)
    fwd = np.array([math.cos(float(scene.heading)),
                    math.sin(float(scene.heading))])
    left = getattr(scene, "lane_left", None)
    right = getattr(scene, "lane_right", None)
    bad = near_bad = total = near_total = 0
    for x, y in path:
        d = math.hypot(x - pos[0], y - pos[1])
        if d < 2.5 or (extent > 0.0 and d > extent):
            continue
        total += 1
        if d <= near_m:
            near_total += 1
        cell = grid.world_to_cell(x, y)
        if cell is None:
            continue
        if drv[cell] <= 0:
            # The camera road mask is partial/noisy; a cell the map-prior
            # lane corridor covers (between lane_left / lane_right) is
            # ROAD even when this frame's mask missed it - only a cell
            # OUTSIDE the corridor that the sensor saw as non-road counts
            # as grass/terrain (mountain corner runs 2026-08-27: a 2%
            # partial mask at (752.7,741.9) rejected every road-aligned
            # arc as "grass" and left only off-road looping arcs).
            if left is not None or right is not None:
                in_corr = True
                if left is not None:
                    ll, cl = _boundary_lateral(x, y, left, fwd)
                    if cl and ll > 0.35:
                        in_corr = False
                if right is not None:
                    lr, cr = _boundary_lateral(x, y, right, fwd)
                    if cr and lr < -0.35:
                        in_corr = False
                if in_corr:
                    continue
            # Only a cell the sensor actually SAW as non-road counts as
            # grass/terrain; unobserved cells are unknown, not a wall.
            if obs is not None and obs.size == drv.size and obs[cell] <= 0:
                continue
            bad += 1
            if d <= near_m:
                near_bad += 1
    return [bad, total, near_bad, near_total]


def _path_known(scene: Scene, path, near_m: float = 10.0) -> list:
    """Count near-window path samples inside sensor-observed cells.

    Returns ``[known, total]``.  ``total`` is the number of samples in the
    2.5..``near_m`` window; ``known`` are those that fall inside an
    ``observed`` (or drivable) cell.  A path that mostly runs through
    unobserved cells is driving blind - infeasible for a real stack.
    """
    path = np.asarray(path, dtype=float)[:, :2]
    if scene.grid is None or len(path) < 2:
        return [0, 0]
    grid = scene.grid
    obs = getattr(grid, "observed", None)
    drv = getattr(grid, "drivable", None)
    evid = None
    if obs is not None and obs.size > 0:
        evid = obs
    elif drv is not None and drv.size > 0:
        evid = drv
    if evid is None:
        return [1, 1]      # no evidence layer: gate not enforced
    if float(np.asarray(evid).sum()) < 0.03 * evid.size:
        return [1, 1]      # sparse footprint: unknown, gate not enforced
    pos = np.asarray(scene.pos[:2], dtype=float)
    known = total = 0
    for x, y in path:
        d = math.hypot(x - pos[0], y - pos[1])
        if d < 2.5 or d > near_m:
            continue
        total += 1
        cell = grid.world_to_cell(x, y)
        if cell is not None and (obs is None or obs.size == 0 or obs[cell] > 0):
            known += 1
    return [known, total]



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
    # No fwd-based sign flip: the boundary polylines are stored in
    # their own travel direction (map prior near->far, sensor lanes
    # along the ego's forward at capture time).  A lane crossing is a
    # WORLD constraint - never into the oncoming lane, never off the
    # road edge - independent of which way the car happens to point at
    # a bend (hairpin repro 2026-08-22: the flip inverted every
    # in-lane candidate into a "crossing" at the apex, so the planner
    # only had cross-lot arcs left and drove off the road).
    return sign, covered



def lane_cross_dist_m(scene: Scene, path, max_cross_m: float = 0.35) -> float:
    """Distance along the path at which it first violates a lane boundary.

    Returns > 0 when the path violates a detected left/right boundary
    (``scene.lane_left`` / ``scene.lane_right``) - a hard no-cross rule.
    Returns 0 when no boundary is detected or no violation exists.
    Only samples within 2.5-15 m of the ego are checked (beyond the car
    footprint and within the sensor lane horizon); the boundary coverage
    flag (``_boundary_lateral``) ensures a path turning at a line ending
    at an intersection is not falsely rejected.

    Recovery: when the EGO already sits beyond a boundary (drifted onto
    the shoulder / past the centre line), a path that CONVERGES back into
    the lane is legal - a real stack eases back, it does not stand still.
    Only a path that keeps diverging farther outside, or that never
    returns inside the corridor, is a violation (mountain runs 2026-08-27
    run_fix22: the car started 3 m outside the right boundary and every
    road-aligned arc was rejected with cross=3.00 because its near
    samples were still outside, leaving only a looping arc that spun it
    on the grass).
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

    def _ego_lat(bnd):
        if bnd is None:
            return None
        # Raw signed lateral of the ego, regardless of coverage: the
        # ego often sits beside the START of a windowed boundary (the
        # window cuts the line ~1 m behind the nearest road vertex), so
        # the nearest point is an endpoint and ``covered`` is False even
        # though the car is clearly outside the lane.  The convergence
        # rule below only requires re-entry when the boundary actually
        # covers the path samples, so a line that ends at an
        # intersection is never held against the car.
        return _boundary_lateral(float(pos[0]), float(pos[1]), bnd, fwd)[0]

    e_l = _ego_lat(left)
    e_r = _ego_lat(right)
    l_beyond = e_l is not None and e_l > max_cross_m
    r_beyond = e_r is not None and e_r < -max_cross_m
    l_seen = l_converged = False
    r_seen = r_converged = False
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
            if left is not None and cov_l:
                l_seen = True
                if l_beyond:
                    # Ego already past the line: diverging farther is a
                    # violation, converging back is recovery.
                    if lat_l > e_l + 0.3:
                        return cum + t * seg
                    if lat_l <= max_cross_m:
                        l_converged = True
                elif lat_l > max_cross_m:
                    return cum + t * seg
            if right is not None and cov_r:
                r_seen = True
                if r_beyond:
                    if lat_r < e_r - 0.3:
                        return cum + t * seg
                    if lat_r >= -max_cross_m:
                        r_converged = True
                elif lat_r < -max_cross_m:
                    return cum + t * seg
        cum += seg
    # An off-corridor start must actually re-enter the lane within the
    # window; a path that drives parallel outside forever is a violation
    # (the drivable gate also rejects it when the camera sees the grass).
    if l_beyond and l_seen and not l_converged:
        return max(cum, 0.1)
    if r_beyond and r_seen and not r_converged:
        return max(cum, 0.1)
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
