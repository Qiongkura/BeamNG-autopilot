"""Local obstacle-aware path and speed planning.

The global route (road-graph A* or the in-game navigation route) is only a
rough corridor / direction.  On top of it this module plans a locally
drivable path:

* no obstacle near the path      -> follow the global route as-is
* obstacle offset from the path  -> elastic-band deformation (small nudge)
* obstacle blocking the corridor -> local occupancy-grid A* detour, i.e.
  the car actually re-plans a driveable way around the thing instead of
  blindly charging into it; when no detour exists the path is truncated
  just before the obstacle so the speed controller brings the car to a
  stop in front of it.

The cruise speed is then reduced for sharp curvature and for obstacles
that sit near the planned path.  Pure pursuit follows the local path.
"""

from __future__ import annotations

import math
import time

import numpy as np

from beamng_autopilot.lane import (
    LANE_MIN_CONF,
    LANE_WIDTH_MAX_M,
    LANE_WIDTH_DEFAULT_M,
    lane_frame_usable,
)
from beamng_autopilot.traffic import RoadRuleView, legal_lane_view
from beamng_autopilot.vision.lanes import marking_is_zigzag

CAR_HALF_WIDTH = 1.0      # lateral half width of the ego car
SAFETY_MARGIN = 1.7       # extra clearance kept around obstacles (m)
MAX_LATERAL_DEV = 8.0     # how far the path may leave the nav corridor
PLAN_HORIZON_M = 48.0     # how far ahead we re-plan the path
CORRIDOR_HALF_W = 1.6     # pass-by slower than this gap from an obstacle
                          # footprint edge is treated as a tight squeeze
STOP_MARGIN_M = 2.5       # keep this much room when stopping for an obstacle
DECEL_MPS2 = 4.0          # assumed decel when computing obstacle speed limit
PASS_BY_MIN_MPS = 2.0     # slowest speed while passing a box that sits
                          # beside the path; a pass-by never stops the car
SPECK_PASS_BY_MIN_MPS = 6.0  # sparse raycast artefacts keep a gentle slow-
                             # down but must not turn a roadside grove into
                             # a 2 m/s crawl every frame
SOLID_LINE_MARGIN = CAR_HALF_WIDTH + 0.3
SOLID_LINE_MAX_M = 8.0
SOLID_MIN_CONF = 0.55
SOLID_MIN_LEN_M = 8.0
SOLID_MAX_CORRIDOR_DEV_M = 5.0
SOLID_MIN_ALIGNMENT = 0.7
SOLID_ANCHOR_NEAR_M = 0.5   # car closer than this to a line is ambiguous:
                            # pick the side from the nearby path instead
                            # of one noisy anchor coordinate
SOLID_BLOCK_MIN_M = 3.0   # solid-line crossings closer than this are treated
                          # as noise under the car, not a legal stop point
SOLID_BLOCK_MAX_M = 30.0  # crossings farther than this are treated as a
                          # distant rule, not a reason to stand still now
SOLID_BLOCK_LANE_CONF = 0.55  # only a confident lane frame may turn a
                              # detected line into a full stop; shaky
                              # vision only nudges the path away
RIGHT_OFFSET_M = 1.5
RIGHT_RAMP_M = 12.0
SHARP_ANGLE_DEG = 45.0
SHARP_CORNER_KPH = 40.0

GRID_RES = 0.5            # occupancy-grid cell size (m)
GRID_AHEAD = 55.0         # grid extent behind the car (m)
GRID_BEHIND = 10.0        # grid extent behind the car (m)
GRID_HALF_W = 20.0        # grid half width (m)
DEV_PENALTY = 0.15        # A* cost added per metre of lateral deviation
GRID_ANTICIPATE = 12.0    # extend blocking boxes toward the car this far (m)
                          # so the A* detour starts steering early and
                          # gently instead of swerving at the last moment
GRID_RIGHT_BIAS = 0.02    # A* cost per metre on the left side of the car:
                          # when both sides are drivable, prefer the right
ROADSIDE_WALL_MIN_LEN_M = 5.0
ROADSIDE_WALL_MAX_THICK_M = 3.5
ROADSIDE_WALL_MIN_EDGE_M = 0.5
SOLID_MAX_LAT_SPAN_FACTOR = 0.5
SOLID_MAX_LAT_SPAN_M = 3.5
SOLID_MAX_PERP_SPAN_M = 0.6
SOLID_MAX_PERP_SPAN_FRAC = 0.02
SPECK_RAYCAST_MAX_M = 1.2
LANE_EDGE_WALL_MIN_ALIGN = 0.70
LANE_EDGE_WALL_MAX_THICK_M = 1.5
LANE_EDGE_WALL_MAX_INTRUDE_M = 0.35
LANE_EDGE_WALL_EDGE_TOL_M = 0.6
# Legacy clamp kept for import compatibility; a paired sensor lane is now
# the drive path itself and is not clamped against the nav route.
LANE_CORRECTION_MAX_M = 2.0
# Lane corrections are ramped in with frame confidence: a frame just above
# the usable threshold is too shaky to steer the car, only confident frames
# get the full capped correction.
LANE_FULL_CONF = 0.6
# A one-sided mirror fallback assumes the lane width instead of seeing it,
# so it may only nudge a nav-route path away from a boundary that is too
# close.  A two-sided lane frame is the lane centre and replaces the route.
LANE_MIRROR_CORRECTION_MAX_M = 0.4
# The LiDAR single-edge frame mirrors an assumed lane width from a wall /
# curb, which is the same one-sided assumption but with a physical edge.
# A single edge cannot prove where the other side of the lane is: a far
# guardrail says nothing about the lane centre, so the fallback is only a
# small nudge away from the wall.  The real centre comes from pairing
# that LiDAR edge with a vision lane boundary.
LANE_LIDAR_CORRECTION_MAX_M = 0.35
# A single painted line (vision mirror) is a real lane edge, but it proves
# nothing about where the opposite side is.  The nav route stays primary;
# the edge only pushes the path away when the route is already too close
# to the paint.  A single LiDAR wall / guardrail is treated the same way.
LANE_BOUNDARY_CORRECTION_MAX_M = LANE_MIRROR_CORRECTION_MAX_M
LANE_LIDAR_EDGE_CORRECTION_MAX_M = LANE_LIDAR_CORRECTION_MAX_M
# The in-game nav route follows the road/link centre, not the legal lane
# centre.  When no paired sensor lane is present, the map's legal-lane
# offset is the lateral reference; only clamp it to the widest lane the
# sensor chain may report.
MAP_LANE_OFFSET_MAX_M = LANE_WIDTH_MAX_M
# Keep this much of the car footprint clear of a single detected edge.
LANE_BOUNDARY_CLEAR_M = CAR_HALF_WIDTH + 0.4
# Boundaries farther than this are usually a different road / far lane and
# must not drag the car toward them.  ``LANE_WIDTH_MAX_M`` is the widest
# single lane the vision/LiDAR chain may report, so anything beyond it is
# not the near edge of the current lane.
LANE_BOUNDARY_MAX_M = LANE_WIDTH_MAX_M


class _MapLaneBoundary:
    """Authoritative no-cross lane boundary derived from the map link.

    ``world`` is a short straight polyline along the current link's lane
    boundary and ``allowed_side`` is +1 when the legal lanes lie to the
    right of the boundary, -1 when they lie to the left.  The class looks
    enough like ``LaneMarking`` for ``_clamp_to_solid_lines`` to consume
    it without running CV-only heuristics.
    """

    def __init__(self, world, allowed_side: float):
        self.world = np.asarray(world, dtype=float)
        self.allowed_side = 1.0 if allowed_side >= 0.0 else -1.0
        self.kind = "solid"
        self.confidence = 1.0
        self.is_map_boundary = True


def corner_angle_deg(points, nearest, back_idx=10, ahead_idx=24):
    """Total heading change (deg) over the curvature window."""
    pts = np.asarray(points[:, :2], dtype=float)
    n = len(pts)
    if n < 4:
        return 0.0
    i0 = max(0, nearest - back_idx)
    i1 = min(n - 1, nearest + ahead_idx)
    sub = pts[i0:i1 + 1]
    if len(sub) < 3:
        return 0.0
    d = np.diff(sub, axis=0)
    seg = np.linalg.norm(d, axis=1)
    if float(seg.sum()) <= 1e-9:
        return 0.0
    ang = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    return abs(float(np.degrees(ang[-1] - ang[0])))


def corner_speed(points, nearest, base_speed, a_lat=6.5,
                 back_idx=10, ahead_idx=24,
                 sharp_angle_deg=45.0, sharp_speed_mps=40.0 / 3.6):
    """Open-path curvature speed limit (m/s) around the nearest index.

    A corner whose total heading change exceeds ``sharp_angle_deg`` is
    additionally capped at ``sharp_speed_mps`` (default 40 km/h).
    """
    pts = np.asarray(points[:, :2], dtype=float)
    n = len(pts)
    if n < 4:
        return base_speed
    i0 = max(0, nearest - back_idx)
    i1 = min(n - 1, nearest + ahead_idx)
    sub = pts[i0:i1 + 1]
    if len(sub) < 3:
        return base_speed
    d = np.diff(sub, axis=0)
    seg = np.linalg.norm(d, axis=1)
    total_len = float(seg.sum())
    total_da = math.radians(corner_angle_deg(pts, nearest, back_idx, ahead_idx))
    if total_da < 1e-6:
        return base_speed
    radius = total_len / total_da
    v = min(base_speed, float(np.sqrt(a_lat * radius)))
    if total_da >= math.radians(sharp_angle_deg):
        v = min(v, float(sharp_speed_mps))
    return v


def _inflated_boxes(obstacles, margin: float):
    boxes = []
    for ob in obstacles:
        hw = ob.half_w + margin
        hh = ob.half_h + margin
        if hw < 0.15 or hh < 0.15:
            continue
        boxes.append((ob.x, ob.y, hw, hh))
    return boxes


def _seg_hits_box(ax, ay, bx, by, cx, cy, hw, hh) -> bool:
    """True when the segment (a-b) intersects the axis-aligned box."""
    dx = bx - ax
    dy = by - ay
    t0, t1 = 0.0, 1.0
    for p, q in (
        (-dx, ax - (cx - hw)),
        (dx, (cx + hw) - ax),
        (-dy, ay - (cy - hh)),
        (dy, (cy + hh) - ay),
    ):
        if abs(p) < 1e-12:
            if q < 0.0:
                return False
        else:
            r = q / p
            if p < 0.0:
                if r > t1:
                    return False
                t0 = max(t0, r)
            else:
                if r < t0:
                    return False
                t1 = min(t1, r)
    return True


def _obstacle_oriented(ob) -> bool:
    """True when the obstacle carries a real oriented footprint."""
    return (ob.axis is not None and ob.half_len > 0.0
            and bool(getattr(ob, "half_len", 0.0)))


def _seg_hits_obstacle(ax, ay, bx, by, ob, half_w: float) -> bool:
    """True when a segment intersects an obstacle's actual footprint.

    Raycast walls keep their oriented footprint (axis + extents) so a
    diagonal roadside wall does not turn into a world-aligned square that
    falsely blocks the lane.  Other obstacle sources stay axis-aligned.
    """
    if _obstacle_oriented(ob):
        ux, uy = float(ob.axis[0]), float(ob.axis[1])
        vx, vy = -uy, ux

        def tr(px, py):
            dx, dy = px - ob.x, py - ob.y
            return dx * ux + dy * uy, dx * vx + dy * vy

        ax1, ay1 = tr(ax, ay)
        bx1, by1 = tr(bx, by)
        return _seg_hits_box(
            ax1, ay1, bx1, by1, 0.0, 0.0,
            ob.half_len + half_w, max(0.0, ob.half_thick) + half_w)
    return _seg_hits_box(ax, ay, bx, by, ob.x, ob.y,
                         ob.half_w + half_w, ob.half_h + half_w)


def _obstacle_corners(ob):
    """World-space corners of an obstacle's actual footprint."""
    if _obstacle_oriented(ob):
        ux, uy = float(ob.axis[0]), float(ob.axis[1])
        vx, vy = -uy, ux
        hu = ob.half_len
        hv = max(0.0, ob.half_thick)
        return (
            (ob.x + ux * hu + vx * hv, ob.y + uy * hu + vy * hv),
            (ob.x + ux * hu - vx * hv, ob.y + uy * hu - vy * hv),
            (ob.x - ux * hu + vx * hv, ob.y - uy * hu + vy * hv),
            (ob.x - ux * hu - vx * hv, ob.y - uy * hu - vy * hv),
        )
    return (
        (ob.x - ob.half_w, ob.y - ob.half_h),
        (ob.x + ob.half_w, ob.y - ob.half_h),
        (ob.x + ob.half_w, ob.y + ob.half_h),
        (ob.x - ob.half_w, ob.y + ob.half_h),
    )


def _seg_seg_dist(ax, ay, bx, by, cx, cy, dx, dy) -> float:
    """Distance between two 2D segments (a-b) and (c-d)."""
    d1x, d1y = bx - ax, by - ay
    d2x, d2y = dx - cx, dy - cy
    rx, ry = ax - cx, ay - cy
    a = d1x * d1x + d1y * d1y
    e = d2x * d2x + d2y * d2y
    f = d2x * rx + d2y * ry
    eps = 1e-9
    if a <= eps and e <= eps:
        return math.hypot(rx, ry)
    if a <= eps:
        t = max(0.0, min(1.0, f / e)) if e > eps else 0.0
        return math.hypot(ax - (cx + d2x * t), ay - (cy + d2y * t))
    c = d1x * rx + d1y * ry
    if e <= eps:
        s = max(0.0, min(1.0, -c / a))
        return math.hypot((ax + d1x * s) - cx, (ay + d1y * s) - cy)
    b = d1x * d2x + d1y * d2y
    denom = a * e - b * b
    s = max(0.0, min(1.0, (b * f - c * e) / denom)) if denom > eps else 0.0
    t = (b * s + f) / e
    if t < 0.0:
        t = 0.0
        s = max(0.0, min(1.0, -c / a))
    elif t > 1.0:
        t = 1.0
        s = max(0.0, min(1.0, (b - c) / a))
    px = ax + d1x * s
    py = ay + d1y * s
    qx = cx + d2x * t
    qy = cy + d2y * t
    return math.hypot(px - qx, py - qy)


def _box_seg_dist(cx, cy, hw, hh, ax, ay, bx, by) -> float:
    """Closest distance from an AABB to a segment (0 when intersecting)."""
    if _seg_hits_box(ax, ay, bx, by, cx, cy, hw, hh):
        return 0.0
    best = float("inf")
    # Distance from each segment endpoint to the box faces (handles a car
    # approaching the box head-on: the closest point is on the near face,
    # not a corner).
    for px, py in ((ax, ay), (bx, by)):
        ddx = max(0.0, max((cx - hw) - px, px - (cx + hw)))
        ddy = max(0.0, max((cy - hh) - py, py - (cy + hh)))
        best = min(best, math.hypot(ddx, ddy))
    # Distance from the segment to each box edge (handles a segment passing
    # alongside the box, closest point on the side face).
    corners = ((cx - hw, cy - hh), (cx + hw, cy - hh),
               (cx + hw, cy + hh), (cx - hw, cy + hh))
    for i in range(4):
        x1, y1 = corners[i]
        x2, y2 = corners[(i + 1) % 4]
        best = min(best, _seg_seg_dist(ax, ay, bx, by, x1, y1, x2, y2))
    return best


def _obstacle_half_extents(ob, fwd, lat):
    """Half extents of an obstacle projected onto (fwd, lat) axes."""
    if _obstacle_oriented(ob):
        ux, uy = float(ob.axis[0]), float(ob.axis[1])
        vx, vy = -uy, ux
        thick = max(0.0, ob.half_thick)
        half_lon = (ob.half_len * abs(fwd[0] * ux + fwd[1] * uy)
                    + thick * abs(fwd[0] * vx + fwd[1] * vy))
        half_lat = (ob.half_len * abs(lat[0] * ux + lat[1] * uy)
                    + thick * abs(lat[0] * vx + lat[1] * vy))
        return half_lon, half_lat
    return (ob.half_w * abs(fwd[0]) + ob.half_h * abs(fwd[1]),
            ob.half_w * abs(lat[0]) + ob.half_h * abs(lat[1]))


def _obstacle_seg_dist(ob, ax, ay, bx, by) -> float:
    """Closest distance from a segment to an obstacle's actual footprint."""
    if _obstacle_oriented(ob):
        ux, uy = float(ob.axis[0]), float(ob.axis[1])
        vx, vy = -uy, ux

        def tr(px, py):
            dx, dy = px - ob.x, py - ob.y
            return dx * ux + dy * uy, dx * vx + dy * vy

        ax1, ay1 = tr(ax, ay)
        bx1, by1 = tr(bx, by)
        return _box_seg_dist(0.0, 0.0, ob.half_len,
                             max(0.0, ob.half_thick),
                             ax1, ay1, bx1, by1)
    return _box_seg_dist(ob.x, ob.y, ob.half_w, ob.half_h,
                         ax, ay, bx, by)


def _point_lat_offset(px, py, pts) -> float:
    """Signed lateral offset of a point from a polyline (nearest segment)."""
    pts = np.asarray(pts[:, :2], dtype=float)
    best = None
    best_d = float("inf")
    for k in range(len(pts) - 1):
        ax, ay = pts[k]
        bx, by = pts[k + 1]
        abx, aby = bx - ax, by - ay
        l2 = abx * abx + aby * aby
        if l2 < 1e-12:
            continue
        t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / l2))
        cx, cy = ax + t * abx, ay + t * aby
        dd = math.hypot(px - cx, py - cy)
        if dd < best_d:
            best_d = dd
            best = ((px - cx) * aby - (py - cy) * abx) / math.sqrt(l2)
    return best if best is not None else 0.0


def _point_route_pos(px, py, pts) -> tuple[float, float]:
    """(along-route arclength, signed lateral offset) of a point.

    The arclength runs from the start of the polyline to the nearest
    point on it; the lateral offset is signed like ``_point_lat_offset``.
    Used by the speed planner to tell obstacles ahead of the car from
    roadside furniture behind / beside it.
    """
    pts = np.asarray(pts[:, :2], dtype=float)
    best_arc = 0.0
    best_lat = 0.0
    best_d = float("inf")
    arc = 0.0
    for k in range(len(pts) - 1):
        ax, ay = pts[k]
        bx, by = pts[k + 1]
        abx, aby = bx - ax, by - ay
        l2 = abx * abx + aby * aby
        if l2 < 1e-12:
            continue
        seg_len = math.sqrt(l2)
        t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / l2))
        qx, qy = ax + t * abx, ay + t * aby
        dd = math.hypot(px - qx, py - qy)
        lat = ((px - qx) * aby - (py - qy) * abx) / seg_len
        if dd < best_d:
            best_d = dd
            best_arc = arc + t * seg_len
            best_lat = lat
        arc += seg_len
    return best_arc, best_lat


def _point_route_pos_np(px, py, pts) -> tuple[float, float]:
    """Vectorized version of ``_point_route_pos`` for full routes."""
    pts = np.asarray(pts[:, :2], dtype=float)
    if len(pts) < 2:
        return 0.0, 0.0
    seg = pts[1:] - pts[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    l2 = seg_len * seg_len
    valid = l2 > 1e-12
    if not np.any(valid):
        return 0.0, 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(valid, (
            (px - pts[:-1, 0]) * seg[:, 0]
            + (py - pts[:-1, 1]) * seg[:, 1]) / l2, 0.0)
    t = np.clip(t, 0.0, 1.0)
    qx = pts[:-1, 0] + t * seg[:, 0]
    qy = pts[:-1, 1] + t * seg[:, 1]
    d2 = (px - qx) ** 2 + (py - qy) ** 2
    k = int(np.argmin(d2))
    lat = ((px - qx[k]) * seg[k, 1] - (py - qy[k]) * seg[k, 0]) / seg_len[k]
    arc = float(np.sum(seg_len[:k])) + t[k] * seg_len[k]
    return arc, float(lat)


def _point_seg_dist(px, py, ax, ay, bx, by) -> float:
    """Shortest distance from a point to a 2D segment."""
    abx, aby = bx - ax, by - ay
    l2 = abx * abx + aby * aby
    if l2 < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = min(1.0, max(0.0, ((px - ax) * abx + (py - ay) * aby) / l2))
    qx = ax + t * abx
    qy = ay + t * aby
    return math.hypot(px - qx, py - qy)


def _pts_to_segments(points, seg):
    """Nearest-segment projection of ``points`` onto a segment array.

    ``seg`` is the ``(M, 2)`` list of segment endpoints (each row is the
    start of a segment; the end is the next row).  Returns arrays of the
    closest point, squared distance, signed lateral offset and the unit
    normal at that point, all with shape ``(len(points),)`` except the
    closest point which is ``(len(points), 2)``.
    """
    p = np.asarray(points[:, :2], dtype=float)
    s = np.asarray(seg[:, :2], dtype=float)
    n = len(p)
    m = len(s) - 1
    if n == 0 or m <= 0:
        return (np.empty((0, 2)), np.empty(0), np.empty(0),
                np.empty((0, 2)))
    d = s[1:] - s[:-1]
    l2 = np.einsum("ij,ij->i", d, d)
    valid = l2 > 1e-12
    inv = np.where(valid, 1.0 / np.where(l2 > 0.0, l2, 1.0), 0.0)
    # t has shape (n, m): projection parameter of every point onto every
    # segment.  Invalid segments get t = 0 and an infinite distance.
    t = ((p[:, None, 0] - s[:-1, 0][None, :])
         * d[None, :, 0] + (p[:, None, 1] - s[:-1, 1][None, :])
         * d[None, :, 1]) * inv[None, :]
    t = np.clip(t, 0.0, 1.0)
    qx = s[:-1, 0][None, :] + t * d[None, :, 0]
    qy = s[:-1, 1][None, :] + t * d[None, :, 1]
    dx = p[:, None, 0] - qx
    dy = p[:, None, 1] - qy
    dist2 = dx * dx + dy * dy
    dist2[:, ~valid] = np.inf
    k = np.argmin(dist2, axis=1)
    rows = np.arange(n)
    q = np.stack([qx[rows, k], qy[rows, k]], axis=1)
    dd = dist2[rows, k]
    seg_len = np.sqrt(np.where(l2 > 0.0, l2, 1.0))
    lat = (dx[rows, k] * d[k, 1] - dy[rows, k] * d[k, 0]) / seg_len[k]
    normal = np.stack([d[k, 1] / seg_len[k],
                       -d[k, 0] / seg_len[k]], axis=1)
    return q, dd, lat, normal


def _points_to_polyline_lat(points, poly):
    """Signed lateral offset of every point from its nearest polyline seg."""
    _, _, lat, _ = _pts_to_segments(points, poly)
    return lat


def is_sparse_raycast_speck(ob) -> bool:
    """True when a raycast cluster is too sparse to act as a path blocker.

    A real wall or a dense trunk cluster comes back as an elongated box
    (labelled "wall") or a compact box several metres across.  A single
    hit point becomes a 0.9 x 0.9 m artefact box, and an unlabelled fat
    raycast blob is usually a few points from two surfaces fused into one
    box.  These are kept for a gentle speed limit but must not pin the
    path to blocked, otherwise the car parks in an open lane.
    """
    if getattr(ob, "category", "") != "raycast":
        return False
    if getattr(ob, "label", "") == "wall":
        return False
    if _obstacle_oriented(ob):
        length = 2.0 * float(getattr(ob, "half_len", 0.0))
        thick = 2.0 * float(getattr(ob, "half_thick", 0.0))
        if length > 4.5 and thick > 2.5:
            return True
        return length < SPECK_RAYCAST_MAX_M \
            and thick < SPECK_RAYCAST_MAX_M
    # Single-hit raycasts have no oriented spread; the 0.9 m box is the
    # min_size floor, not a measured footprint.
    return (2.0 * ob.half_w <= 2.1
            and 2.0 * ob.half_h <= 2.1)


def is_small_lidar_clutter(ob) -> bool:
    """True when a LiDAR cluster is small enough to be roadside clutter.

    Dense town scenes return dozens of small lidar boxes (poles, trunks,
    mailboxes, wall corners) that inflate the A* grid until no detour
    exists.  Like ``is_sparse_raycast_speck`` these are kept for a gentle
    speed limit but must not pin the path to blocked.  A real vehicle or
    pedestrian is still covered by the Lua vehicle/scenario scans and the
    vision channel, so dropping the small lidar boxes does not remove a
    safety layer - it removes grid noise.
    """
    if getattr(ob, "category", "") != "lidar":
        return False
    if getattr(ob, "label", "") == "wall":
        return False
    if _obstacle_oriented(ob):
        return (2.0 * float(getattr(ob, "half_len", 0.0)) <= 2.1
                and 2.0 * float(getattr(ob, "half_thick", 0.0)) <= 2.1)
    return (2.0 * ob.half_w <= 2.1
            and 2.0 * ob.half_h <= 2.1)


def _lane_tangent_at(lane_center, x: float, y: float) -> np.ndarray:
    """Local travel direction of a lane polyline near a world point."""
    pts = np.asarray(lane_center[:, :2], dtype=float)
    if len(pts) < 2:
        return np.array([1.0, 0.0])
    d2 = (pts[:, 0] - x) ** 2 + (pts[:, 1] - y) ** 2
    k = int(np.argmin(d2))
    i0 = max(0, k - 2)
    i1 = min(len(pts) - 1, k + 2)
    v = pts[i1] - pts[i0]
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.array([1.0, 0.0])
    return v / n


def is_lane_edge_wall(ob, lane_center, lane_width: float,
                      lane_edge=None, edge_side: float = 0.0) -> bool:
    """True when a thin raycast wall lines the outside of a detected lane.

    The vision/LiDAR lane frame already puts the driving path in the
    middle of the free lane.  A short wall segment that lies entirely at
    (or just outside) the detected lane edge is the road boundary, not an
    object to detour around or stop for.  Treating it as a blocker made
    the car park beside a roadside wall whose axis-aligned box poked into
    the lane, because the fixed CAR_HALF_WIDTH + 0.8 clearance reaches
    almost all the way to the far edge of a 3.5 m lane.
    """
    if ob.category != "raycast" or not _obstacle_oriented(ob):
        return False
    lane = np.asarray(lane_center[:, :2], dtype=float)
    if len(lane) < 2 or lane_width <= 0.0:
        return False
    axis = np.asarray(ob.axis[:2], dtype=float)
    tangent = _lane_tangent_at(lane, ob.x, ob.y)
    if abs(float(axis @ tangent)) < LANE_EDGE_WALL_MIN_ALIGN:
        return False
    # A single-edge LiDAR frame was built from the wall itself: its right
    # (or left) boundary polyline is the wall surface.  When the wall lies
    # at/outside that detected edge it is the road boundary, so a slightly
    # inflated or thick cluster must not close the corridor.
    if lane_edge is not None and len(lane_edge) >= 2:
        edge_pts = np.asarray(lane_edge[:, :2], dtype=float)
        if len(edge_pts) >= 2:
            edge_lats = np.asarray([
                _point_lat_offset(c[0], c[1], edge_pts)
                for c in _obstacle_corners(ob)], dtype=float)
            if len(edge_lats) >= 2:
                side = float(edge_side)
                if side == 0.0:
                    # The lane centre sits on the road side of the edge;
                    # the boundary itself is on the opposite side.
                    c_lats = np.asarray([
                        _point_lat_offset(c[0], c[1], edge_pts)
                        for c in lane[::max(1, len(lane) // 6)]],
                        dtype=float)
                    # ``_point_lat_offset`` is positive to the right, so a
                    # right edge has the lane centre on its negative side.
                    side = -1.0 if float(np.median(c_lats)) < 0.0 \
                        else 1.0
                outside = -side * edge_lats
                if float(np.min(outside)) >= -LANE_EDGE_WALL_EDGE_TOL_M \
                        and float(np.median(outside)) >= 0.05:
                    return True
    if 2.0 * max(0.0, ob.half_thick) > LANE_EDGE_WALL_MAX_THICK_M:
        return False
    half_lane = 0.5 * lane_width
    edge = half_lane - LANE_EDGE_WALL_MAX_INTRUDE_M
    if edge <= 0.0:
        return False
    lats = [_point_lat_offset(c[0], c[1], lane)
            for c in _obstacle_corners(ob)]
    if len(lats) < 2:
        return False
    pos = sum(1 for v in lats if v > edge)
    neg = sum(1 for v in lats if v < -edge)
    return pos >= len(lats) - 1 or neg >= len(lats) - 1


def _polyline_crosses(w, poly, threshold: float = 0.1) -> bool:
    """True when a route polyline has points on both sides of a marking."""
    pos = neg = 0
    for px, py in np.asarray(poly[:, :2], dtype=float):
        lat = _point_lat_offset(float(px), float(py), w)
        if lat > threshold:
            pos += 1
        elif lat < -threshold:
            neg += 1
    return pos >= 2 and neg >= 2


def _marking_is_road_boundary(mk, path, corridor=None) -> bool:
    """Only treat a marking as a no-cross boundary when it looks like a real
    road line: solid, confident, long, and aligned with the driving corridor.

    Classic CV also picks up short paint blobs, kerbs and random pavement
    edges near the camera; those must never pin the car to a standstill.
    """
    if getattr(mk, "is_map_boundary", False):
        world = np.asarray(getattr(mk, "world", None), dtype=float)
        return world.ndim == 2 and world.shape[1] >= 2 and len(world) >= 2
    try:
        world = np.asarray(getattr(mk, "world", None), dtype=float)
        conf = float(getattr(mk, "confidence", 0.0))
    except (TypeError, ValueError):
        return False
    if world.ndim != 2 or world.shape[1] < 2 or len(world) < 2:
        return False
    if getattr(mk, "kind", "solid") != "solid":
        return False
    if conf < SOLID_MIN_CONF:
        return False
    w = world[:, :2].astype(float)
    p = np.asarray(path[:, :2], dtype=float)
    if len(p) < 2:
        return False
    # Dominant direction of the marking must roughly follow the corridor
    # direction, not a crosswalk stripe or a kerb cutting across the road.
    center = w.mean(axis=0)
    d = w - center
    cov = d.T @ d
    evals, evecs = np.linalg.eigh(cov)
    mdir = evecs[:, int(np.argmax(evals))]
    pdir = p[-1] - p[0]
    pn = float(np.linalg.norm(pdir))
    if pn < 1e-9:
        return False
    # Net extent along the marking's own axis is what makes it a road
    # line.  A back-projected blob can zig-zag and sum to a long polyline
    # while covering almost no ground; such a marking is noise.
    proj = w @ mdir
    span = float(np.max(proj) - np.min(proj))
    world_len = float(np.sum(np.linalg.norm(np.diff(w, axis=0), axis=1)))
    if span < SOLID_MIN_LEN_M or marking_is_zigzag(span, world_len):
        return False
    align = abs(float(mdir @ pdir)) / pn
    if align < SOLID_MIN_ALIGNMENT:
        return False
    if corridor is not None and len(corridor) >= 2:
        try:
            cor = np.asarray(corridor[:, :2], dtype=float)
        except (TypeError, ValueError):
            cor = None
        if cor is not None and len(cor) >= 2 \
                and _polyline_crosses(w, cor):
            # The nav route itself crosses the marking, so it is a kerb,
            # track paint or a projection artefact, not a lane boundary
            # this route is supposed to stay beside.
            return False
    # The line must run near the corridor; a distant kerb or roadside edge
    # is not the boundary the car is driving against.
    _, v_dist2, _, _ = _pts_to_segments(w, p)
    if len(v_dist2) == 0 or float(np.sqrt(np.min(v_dist2))) \
            > SOLID_MAX_CORRIDOR_DEV_M:
        return False
    # A real boundary stays roughly parallel to the driving corridor.
    # Pavement texture, a roadside white block or a kerb crossing the
    # field of view can be long and confident while sweeping 10+ m
    # laterally; that is not a lane edge.
    lats = _points_to_polyline_lat(w, p)
    lat_span = float(np.max(lats) - np.min(lats)) if len(lats) else 0.0
    if lat_span > max(SOLID_MAX_LAT_SPAN_M,
                      SOLID_MAX_LAT_SPAN_FACTOR * span):
        return False
    # A genuine painted line is a thin ribbon in world space.  A dark
    # patch / repair scar may back-project as a long, confident "solid"
    # blob; its cross-track width gives it away and it must never block.
    minor = evecs[:, int(np.argmin(evals))]
    perp = w @ minor
    perp_span = float(np.max(perp) - np.min(perp))
    if perp_span > max(SOLID_MAX_PERP_SPAN_M,
                       SOLID_MAX_PERP_SPAN_FRAC * span):
        return False
    return True


def _smoothstep(x: float) -> float:
    x = min(1.0, max(0.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def _lane_correction_gain(conf: float, min_conf: float = LANE_MIN_CONF,
                          full_conf: float = LANE_FULL_CONF) -> float:
    """Scale a lane-centre correction by frame confidence."""
    if conf <= min_conf:
        return 0.0
    if conf >= full_conf:
        return 1.0
    return (conf - min_conf) / (full_conf - min_conf)


def _clamp_to_solid_lines(path, solid_lines, anchor,
                          margin: float = SOLID_LINE_MARGIN,
                          corridor=None, allow_block: bool = True,
                          block_near_cross: bool = False,
                          map_nudge: bool = True):
    """Push a path off detected solid markings; block when it crosses one
    well ahead of the car.

    The allowed side of each marking is the side the route anchor (the
    car / route centre) is already on.  A point that would cross to the
    forbidden side makes the lane change illegal, so the caller stops
    before the line; a point that is merely too close is nudged back to
    ``margin`` so the car footprint never presses the paint.  A crossing
    within ``SOLID_BLOCK_MIN_M`` of the anchor is a noisy/stale marking
    under the car: it is nudged like a close point instead of parking on
    the spot.  ``block_near_cross`` lifts that guard for deliberate lane
    changes: a detour that starts by crossing a validated solid boundary
    is a rule violation even when the crossing is only metres ahead.
    ``map_nudge`` controls whether a map boundary may push a legal-but-
    close path back to the map side; it still blocks a crossing either way.
    """
    out = np.asarray(path, dtype=float).copy()
    if out.ndim != 2 or len(out) < 2 or not solid_lines:
        return out, False, 0.0
    anchor = np.asarray(anchor, dtype=float)[:2]
    for mk in solid_lines:
        if not _marking_is_road_boundary(mk, out, corridor=corridor):
            continue
        world = np.asarray(mk.world, dtype=float)
        if world.ndim != 2 or world.shape[1] < 2 or len(world) < 2:
            continue
        w = world[:, :2].astype(float)
        is_map = bool(getattr(mk, "is_map_boundary", False))
        if is_map:
            # The map says which side is legal; do not infer it from a car
            # that may already be on the wrong side of the centre line.
            side = float(mk.allowed_side)
        else:
            anchor_lat = _point_lat_offset(anchor[0], anchor[1], w)
            side = 1.0 if anchor_lat >= 0.0 else -1.0
            if abs(anchor_lat) < SOLID_ANCHOR_NEAR_M:
                # The car is sitting on/near the paint; a 2 cm pose error
                # can flip the "allowed side" and turn a legal road edge
                # into a phantom blocker.  Let the nearby path majority
                # decide.
                near_lats = [_point_lat_offset(
                    float(p[0]), float(p[1]), w)
                    for p in out[: min(len(out), 8), :2]]
                if near_lats:
                    side = 1.0 if float(np.median(near_lats)) >= 0.0 \
                        else -1.0
        axis = None
        span = 0.0
        if is_map:
            d = w[-1] - w[0]
            span = float(np.linalg.norm(d))
            if span > 1e-9:
                axis = d / span
        # Crossing to the forbidden side is a rule violation: truncate the
        # path before the crossing so the speed controller stops the car.
        # Remember the closest crossing and its distance along the path so
        # a line sitting under the car does not turn into a 0 m "blocked".
        seg_len = np.linalg.norm(np.diff(out[:, :2], axis=0), axis=1)
        path_cum = np.concatenate([[0.0], np.cumsum(seg_len)])
        samples = np.concatenate([
            out[i][:2][None, :]
            + np.linspace(0.0, 1.0, 5)[:, None] * (out[i + 1][:2] - out[i][:2])
            for i in range(len(out) - 1)], axis=0)
        _, _, s_lat, _ = _pts_to_segments(samples, w)
        s_lat = s_lat.reshape(len(out) - 1, 5)
        bad = (s_lat * side) < -0.05
        if axis is not None:
            along = (samples - w[0]) @ axis
            in_span = (along >= -0.5) & (along <= span + 0.5)
            bad &= in_span.reshape(len(out) - 1, 5)
        first_bad = np.argmax(bad, axis=1)
        has_bad = np.any(bad, axis=1)
        cross_i = -1
        cross_first = 0
        cross_dist = float("inf")
        for i in range(len(out) - 1):
            if not has_bad[i]:
                continue
            t = float(np.linspace(0.0, 1.0, 5)[first_bad[i]])
            d = float(path_cum[i] + t * seg_len[i])
            if d < cross_dist:
                cross_i = i
                cross_first = int(first_bad[i])
                cross_dist = d
        if allow_block and cross_i >= 0 \
                and cross_dist <= SOLID_BLOCK_MAX_M \
                and (block_near_cross
                     or cross_dist >= SOLID_BLOCK_MIN_M):
            # Stop before the crossing.  A coarse detour vertex can already
            # be far across the paint, so keep the complete vertices before
            # the crossing segment plus the last sample still on the
            # allowed side instead of cutting at the segment end.
            head = out[: cross_i + 1]
            if cross_first > 1:
                last_good = samples[cross_i * 5 + cross_first - 1]
                cut = np.vstack([head, last_good])
            else:
                cut = head
            if len(cut) < 2:
                cut = np.vstack([cut, cut[0]])
            return cut, True, cross_dist
        # Legal but too close: keep the car footprint off the paint.  A
        # map boundary only defines which side is legal: when a sensor
        # lane is already driving the path it must not turn that boundary
        # into a keep-right push toward a wall.
        q, _, lat, normal = _pts_to_segments(out, w)
        near = (lat * side) < margin
        if is_map and not map_nudge:
            near[:] = False
        if axis is not None:
            along = (out[:, :2] - w[0]) @ axis
            near &= (along >= -0.5) & (along <= span + 0.5)
        if np.any(near):
            target = side * np.maximum(margin, side * lat[near])
            out[near, 0] = q[near, 0] + normal[near, 0] * target
            out[near, 1] = q[near, 1] + normal[near, 1] * target
    return out, False, 0.0


def _obstacle_aabb(ob, half_w: float):
    """Axis-aligned bounding box of an obstacle inflated by ``half_w``."""
    if _obstacle_oriented(ob):
        ux, uy = float(ob.axis[0]), float(ob.axis[1])
        vx, vy = -uy, ux
        hu = ob.half_len + half_w
        hv = max(0.0, ob.half_thick) + half_w
        pts = (
            (ob.x + ux * hu + vx * hv, ob.y + uy * hu + vy * hv),
            (ob.x + ux * hu - vx * hv, ob.y + uy * hu - vy * hv),
            (ob.x - ux * hu + vx * hv, ob.y - uy * hu + vy * hv),
            (ob.x - ux * hu - vx * hv, ob.y - uy * hu - vy * hv),
        )
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (min(xs), min(ys), max(xs), max(ys))
    return (ob.x - ob.half_w - half_w, ob.y - ob.half_h - half_w,
            ob.x + ob.half_w + half_w, ob.y + ob.half_h + half_w)


def _path_hit_index(pts, i0: int, i1: int, obstacles, half_w: float) -> int:
    """Index of the first path vertex whose next segment is blocked.

    ``pts`` is the full route; only the ``[i0, i1]`` window is inspected.
    Returns -1 when no segment in the window is blocked.
    """
    n = len(pts)
    boxes = [_obstacle_aabb(ob, half_w) for ob in obstacles]
    for i in range(i0, min(i1, n - 1)):
        ax, ay = pts[i]
        bx, by = pts[i + 1]
        seg_min_x = min(ax, bx)
        seg_max_x = max(ax, bx)
        seg_min_y = min(ay, by)
        seg_max_y = max(ay, by)
        for k, ob in enumerate(obstacles):
            x0, y0, x1, y1 = boxes[k]
            if (seg_max_x < x0 or seg_min_x > x1
                    or seg_max_y < y0 or seg_min_y > y1):
                continue
            if _seg_hits_obstacle(ax, ay, bx, by, ob, half_w):
                return i
    return -1


def _find_blocker(pts, i0: int, i1: int, obstacles, half_w: float):
    """First obstacle whose footprint intrudes into the planning window."""
    n = len(pts)
    boxes = [_obstacle_aabb(ob, half_w) for ob in obstacles]
    for i in range(i0, min(i1, n - 1)):
        ax, ay = pts[i]
        bx, by = pts[i + 1]
        seg_min_x = min(ax, bx)
        seg_max_x = max(ax, bx)
        seg_min_y = min(ay, by)
        seg_max_y = max(ay, by)
        for k, ob in enumerate(obstacles):
            x0, y0, x1, y1 = boxes[k]
            if (seg_max_x < x0 or seg_min_x > x1
                    or seg_max_y < y0 or seg_min_y > y1):
                continue
            if _seg_hits_obstacle(ax, ay, bx, by, ob, half_w):
                return ob
    return None


def _vehicle_speed_along(ob, seg_pts, seg_k: int) -> float:
    """Signed speed of a dynamic vehicle along the local route segment.

    Used by speed planning so a moving lead vehicle does not force the ego
    to brake as hard as for a static wall.
    """
    if ob is None or ob.velocity is None or seg_pts is None:
        return 0.0
    if seg_k < 0 or seg_k >= len(seg_pts) - 1:
        return 0.0
    ax, ay = seg_pts[seg_k]
    bx, by = seg_pts[seg_k + 1]
    dx, dy = bx - ax, by - ay
    n = math.hypot(dx, dy)
    if n < 1e-9:
        return 0.0
    return max(0.0, float((ob.velocity[0] * dx + ob.velocity[1] * dy) / n))


class LocalPlanner:
    """Plans a locally drivable path around obstacles and a safe speed."""

    def __init__(
        self,
        horizon_m: float = PLAN_HORIZON_M,
        max_dev: float = MAX_LATERAL_DEV,
        corridor_half_w: float = CORRIDOR_HALF_W,
        margin: float = SAFETY_MARGIN,
        relax_iters: int = 15,
        smooth_passes: int = 3,
        push_gain: float = 0.45,
        lateral_clear: float = 0.6,
        anticipate: float = 5.0,
        right_offset: float = RIGHT_OFFSET_M,
        right_ramp_m: float = RIGHT_RAMP_M,
        sharp_angle_deg: float = SHARP_ANGLE_DEG,
        sharp_corner_kph: float = SHARP_CORNER_KPH,
        grid_right_bias: float = GRID_RIGHT_BIAS,
    ):
        self.horizon_m = horizon_m
        self.max_dev = max_dev
        self.corridor_half_w = corridor_half_w
        self.margin = margin
        self.relax_iters = relax_iters
        self.smooth_passes = smooth_passes
        self.push_gain = push_gain
        # Extra clearance kept beyond the car half width when the elastic
        # band nudges the path around a nearby obstacle.
        self.lateral_clear = lateral_clear
        self.anticipate = anticipate
        self.right_offset = right_offset
        self.right_ramp_m = right_ramp_m
        self.sharp_angle_deg = sharp_angle_deg
        self.sharp_corner_kph = sharp_corner_kph
        self.grid_right_bias = grid_right_bias
        # Obstacles whose center sits this close (laterally) to the original
        # navigation route are treated as "blockers in the lane": the car
        # eases off as it approaches them.  Roadside poles/curbs 3 m off the
        # route do not qualify.
        self.blocker_lat = 2.6
        # Last planning outcome for HUD/telemetry: "follow" (nav route as-is),
        # "deform" (elastic band nudge), "detour" (A* around a blocker) or
        # "blocked" (no drivable way; stop in front).
        self.last_mode = "follow"
        # Last lane-reference source used by plan(): "nav" (the route
        # centre, no right bias by default) or the first entry of the
        # LaneFrame ``sources`` tuple ("vision" / "lidar").
        self.last_lane_mode = "nav"
        # Median lateral path offset applied from the nav route when a
        # single lane boundary (right line / wall / guardrail) defined the
        # drive path.  Diagnostics only.
        self.last_lane_offset = 0.0
        # When ``last_mode`` is "blocked": (label, distance from the car) of
        # the obstacle that left no drivable way, for HUD diagnostics.
        self.last_blocker: tuple[str, float] | None = None
        self.last_route: np.ndarray | None = None
        # Per-frame planning timing (ms) for slow-frame diagnostics.
        self.last_plan_stages: dict[str, float] = {}
        # Speed-planning diagnostics (read by telemetry / GUI):
        # ``last_corner`` is the cruise limited by path curvature only and
        # ``last_obs_lim`` the kinematic speed limit of the closest obstacle
        # that slowed the car (None when no obstacle did).
        self.last_corner: float | None = None
        self.last_obs_lim: float | None = None
        self.last_sharp: bool = False

    # ---- path planning -------------------------------------------------

    def plan(self, route: np.ndarray | None, obstacles, pos, heading: float,
             nearest: int, solid_lines=None,
             sensor_lane=None,
             road_rule: RoadRuleView | None = None
             ) -> tuple[np.ndarray, bool]:
        """Return (drive_path, blocked).

        ``drive_path`` is a dense 2D polyline the car should follow next
        (global route, elastic-band deformation, or an A* detour around a
        blocking obstacle).  ``blocked`` is True when no drivable way was
        found and the car should stop in front of the first obstacle.
        ``solid_lines`` is an optional list of detected solid lane markings
        used as no-cross boundaries.  ``sensor_lane`` is an optional
        ``LaneFrame`` from the camera / LiDAR lane modules: when it is
        confident, a two-sided frame is the lane centre itself and the
        drive path follows it directly.  ``route`` may be None when the
        sensor lane is present: no navigation route is required for
        lane-level driving.  ``road_rule`` is an optional map snapshot
        used to keep the car on the legal side of the road and to block
        paths that would cross the map's opposing-lane boundary.
        """
        self.last_blocker = None
        self.last_plan_stages = {}
        _t0 = time.perf_counter()

        def _stage(name: str) -> None:
            self.last_plan_stages[name] = (
                time.perf_counter() - _t0) * 1000.0
        # Remember the original reference for diagnostics.  A paired sensor
        # lane replaces the nav route as the lane centre when available.
        self.last_route = (None if route is None
                           else np.asarray(route, dtype=float))
        lane_mode = None
        lane_edge = None
        lane_edge_side = 0.0
        lane_center_hint = False
        lane_center = None
        lane_primary = False
        if lane_frame_usable(sensor_lane):
            lane_center = np.asarray(sensor_lane.center, dtype=float)
            if (lane_center.ndim == 2 and lane_center.shape[1] >= 2
                    and len(lane_center) >= 2):
                src = (sensor_lane.sources[0]
                       if sensor_lane.sources else "sensor")
                if sensor_lane.paired:
                    # A real two-sided lane (painted pair or vision + LiDAR
                    # fusion) defines the current lane: its centre is the
                    # drive path, so a nav route must not pull the car out
                    # of the detected lane.
                    lane_mode = src
                    lane_primary = True
                elif src.startswith("vision"):
                    # A single-edge mirror assumes the lane width from one
                    # painted line, so it cannot prove where the lane is
                    # once a nav route exists.  The nav route stays the
                    # primary centre and the mirror only pushes the path
                    # away from a boundary that is too close.  Without a
                    # nav route the mirror is the only lane reference and
                    # may drive its inferred centre.
                    if route is None:
                        lane_mode = src
                        lane_primary = True
                    elif getattr(sensor_lane, "right", None) is not None:
                        lane_mode = f"{src}_right"
                        lane_edge = sensor_lane.right
                        lane_edge_side = -1.0
                    elif getattr(sensor_lane, "left", None) is not None:
                        lane_mode = f"{src}_left"
                        lane_edge = sensor_lane.left
                        lane_edge_side = 1.0
                    else:
                        lane_mode = src
                        lane_center_hint = True
                elif src.startswith("lidar"):
                    # A single LiDAR edge is a low-trust boundary: it may
                    # nudge the path, but the centre line stays primary.
                    lane_mode = src
                    lane_center_hint = True
                    if getattr(sensor_lane, "right", None) is not None:
                        lane_mode = f"{src}_right"
                        lane_edge = sensor_lane.right
                        lane_edge_side = -1.0
                    elif getattr(sensor_lane, "left", None) is not None:
                        lane_mode = f"{src}_left"
                        lane_edge = sensor_lane.left
                        lane_edge_side = 1.0
                elif getattr(sensor_lane, "right", None) is not None:
                    # Painted right line first: it is the primary boundary
                    # under right-hand traffic and outranks any wall.
                    lane_mode = f"{src}_right"
                    lane_edge = sensor_lane.right
                    lane_edge_side = -1.0
                elif getattr(sensor_lane, "left", None) is not None:
                    lane_mode = f"{src}_left"
                    lane_edge = sensor_lane.left
                    lane_edge_side = 1.0
        self.last_lane_mode = (lane_mode
                               or ("nav" if route is not None else "sensor"))
        self.last_lane_offset = 0.0
        map_lane, map_boundaries = self._map_legal_lane(road_rule)
        if map_lane is not None and not map_lane.legal:
            # The map link has no forward lane in the ego's direction:
            # proceeding would drive the wrong way on a one-way road.
            self.last_mode = "blocked"
            self.last_blocker = ("wrong-way road", 0.0)
            return np.empty((0, 2), dtype=float), True
        # Map legal-lane data is a no-cross / wrong-way safety layer, not a
        # lateral driving target.  The nav route is route-level direction;
        # the position inside the lane comes from the sensor lane.  Keeping
        # ``map_offset`` at ``None`` means no code path may shift the drive
        # path by ``preferred_offset_m``.
        map_offset = None
        raw_path = None
        if lane_primary:
            # The frame centre is the midpoint of the detected markings
            # (or of a marking plus its mirrored/opposite-side boundary),
            # so it is the drive path itself.  The nav route may still
            # exist as a long-range direction, but it must not bias the
            # car right of the lane.
            nav_pts, nav_i0, nav_i1 = self._sensor_window(lane_center, pos)
            pts = nav_pts[nav_i0:nav_i1 + 1].copy()
            raw_path = pts.copy()
            i0, i1 = 0, len(pts) - 1
        elif lane_mode is not None and route is not None:
            nav_pts, nav_i0, nav_i1 = self._window(route, nearest)
            pts = nav_pts[nav_i0:nav_i1 + 1].copy()
            raw_path = pts.copy()
            i0, i1 = 0, len(pts) - 1
            if src.startswith("lidar") and lane_edge is None:
                lane_gain = _lane_correction_gain(sensor_lane.confidence)
                corr_max = LANE_LIDAR_CORRECTION_MAX_M
                base = pts.copy()
                n = len(pts)
                d = np.linalg.norm(np.diff(base, axis=0), axis=1)
                cum = np.concatenate([[0.0], np.cumsum(d)])
                for i in range(n):
                    f = _smoothstep(cum[i] / max(1e-9, self.right_ramp_m))
                    a = base[max(0, i - 2)]
                    b = base[min(n - 1, i + 2)]
                    tv = b - a
                    tn = float(np.linalg.norm(tv))
                    if tn < 1e-9:
                        left = np.array([-math.sin(heading),
                                         math.cos(heading)])
                    else:
                        left = np.array([-tv[1] / tn, tv[0] / tn])
                    off = _point_lat_offset(
                        base[i, 0], base[i, 1], lane_center)
                    off = max(-corr_max, min(corr_max, off))
                    pts[i] = base[i] + left * (f * off * lane_gain)
                self.last_lane_offset = 0.0
            elif lane_edge is not None:
                # Single-boundary protection: the nav route is the primary
                # lane centre, so a right paint / wall / guardrail only
                # pushes the path away when the route point is already too
                # close to the edge.  It never actively pulls the car
                # toward a half-lane position.
                lane_gain = _lane_correction_gain(sensor_lane.confidence)
                corr_max = (
                    LANE_BOUNDARY_CORRECTION_MAX_M
                    if src.startswith("vision")
                    else LANE_LIDAR_EDGE_CORRECTION_MAX_M)
                edge_pts = np.asarray(lane_edge[:, :2], dtype=float)
                lane_w = float(getattr(sensor_lane, "width",
                                       LANE_WIDTH_DEFAULT_M))
                base = pts.copy()
                offsets = []
                n = len(pts)
                d = np.linalg.norm(np.diff(base, axis=0), axis=1)
                cum = np.concatenate([[0.0], np.cumsum(d)])
                for i in range(n):
                    off = self._boundary_path_offset(
                        base[i, 0], base[i, 1], edge_pts,
                        side=lane_edge_side, lane_width=lane_w,
                        corr_max=corr_max)
                    if off is None:
                        off = 0.0
                    offsets.append(off)
                    f = _smoothstep(
                        cum[i] / max(1e-9, self.right_ramp_m))
                    a = base[max(0, i - 2)]
                    b = base[min(n - 1, i + 2)]
                    tv = b - a
                    tn = float(np.linalg.norm(tv))
                    if tn < 1e-9:
                        right = np.array([math.sin(heading),
                                          -math.cos(heading)])
                    else:
                        right = np.array([tv[1] / tn, -tv[0] / tn])
                    pts[i] = base[i] + right * (
                        f * off * lane_gain)
                self.last_lane_offset = float(np.median(offsets))
        elif route is not None:
            pts, i0, i1 = self._window(route, nearest)
        elif lane_mode is not None:
            # No nav route and only a low-trust single-edge frame: use the
            # sensor-derived mirror centre, but still classify it as a
            # single-side fallback for telemetry.
            nav_pts, nav_i0, nav_i1 = self._sensor_window(lane_center, pos)
            pts = nav_pts[nav_i0:nav_i1 + 1].copy()
            raw_path = pts.copy()
            i0, i1 = 0, len(pts) - 1
        else:
            self.last_mode = "no-lane"
            return np.empty((0, 2), dtype=float), False
        if len(pts) < 2:
            self.last_mode = "follow"
            return pts, False
        raw_pts = pts.copy()
        # Sparse raycast artefacts are kept for the speed planner (they
        # still ease off the throttle) but are not allowed to close the
        # whole corridor: a 0.9 m single-hit box or an unlabelled fused
        # blob must not park the car in an empty lane.
        obstacles = [ob for ob in obstacles
                     if not is_sparse_raycast_speck(ob)
                     and not is_small_lidar_clutter(ob)]
        if lane_mode is not None:
            # The lane centre already keeps the car inside the detected
            # lane; a thin wall at the lane edge is the boundary itself,
            # so it must not close the whole corridor from the side.
            obstacles = [ob for ob in obstacles
                         if not is_lane_edge_wall(
                             ob, raw_path if raw_path is not None else pts,
                             sensor_lane.width,
                             lane_edge=lane_edge,
                             edge_side=lane_edge_side)]
        if route is not None and (
                lane_mode is None or not (
                    sensor_lane is not None and lane_primary)):
            # A single-edge camera/LiDAR read cannot prove where the lane
            # centre is: it only nudges the path away from a boundary that
            # is too close.  The legal-lane offset from the nav route is
            # the driving reference until a real two-sided lane exists.
            # The offset is never pressed into a wall or a lane blocker; on
            # a wall-lined road the path simply eases back instead of
            # trying to squeeze past the wall.
            safe_off = self._safe_right_offset(
                raw_pts, i0, i1, heading, obstacles,
                edge_pts=lane_edge, edge_side=lane_edge_side)
            self.last_lane_offset = float(safe_off) if abs(safe_off) > 1e-9 \
                else 0.0
            _stage("safe_offset")
            pts = self._right_offset_path(raw_pts, i0, heading,
                                          offset=safe_off)
            _stage("offset")

        hit = -1

        def finish(out, mode: str):
            """Apply the solid-line boundary rule and record the mode."""
            boundaries = list(solid_lines or [])
            # Map lane boundaries are long-range direction / wrong-way
            # safety, not lane-level geometry.  Once the camera or LiDAR
            # has produced a sensor lane, the car must steer by that local
            # lane only; a map centre-line must not turn a legal local
            # path into a phantom "solid line" stop.
            if map_boundaries and lane_mode is None:
                boundaries.extend(map_boundaries)
            if boundaries:
                _t1 = time.perf_counter()
                # A detour deliberately leaves the current lane: if it
                # crosses a detected solid line, reject the lane change and
                # stop in front of the obstacle instead.  For ordinary
                # follow/deform paths a solid-line stop still needs a real
                # two-sided lane read; a single-edge mirror/fallback knows
                # one boundary, not the lane geometry, so it may nudge away
                # from the paint but must not turn a CV line into a full
                # stop.
                allow_block = (mode == "detour"
                               or sensor_lane is None
                               or (sensor_lane.paired
                                   and sensor_lane.confidence
                                   >= SOLID_BLOCK_LANE_CONF))
                allow_block = allow_block or bool(
                    map_boundaries and lane_mode is None)
                out, crossed, cross_dist = _clamp_to_solid_lines(
                    out, boundaries, pos, SOLID_LINE_MARGIN,
                    corridor=raw_pts, allow_block=allow_block,
                    block_near_cross=(mode == "detour"),
                    map_nudge=(lane_mode is None))
                self.last_plan_stages["solid"] = (
                    time.perf_counter() - _t1) * 1000.0
                if crossed:
                    self.last_blocker = ("solid line",
                                         round(cross_dist, 1))
                    self.last_mode = "blocked"
                    if mode == "detour" and hit >= 0:
                        # Refuse the lane change: stop on the original
                        # lane path in front of the obstacle instead of
                        # following a path that crosses the boundary.
                        return pts[: max(2, hit + 2)], True
                    return out, True
            self.last_mode = mode
            return out, mode == "blocked"

        # Obstacles' own footprints (no safety inflation): only a footprint
        # that really intrudes into the lane matters.  Roadside poles/curbs
        # 2.5-3 m off the route must NOT make the car swerve or brake.
        if not obstacles:
            return finish(pts, "follow")
        hit = _path_hit_index(pts, i0, i1, obstacles,
                              CAR_HALF_WIDTH + 0.8)
        _stage("hit")
        if hit < 0:
            out = self.deform(pts, obstacles, 0 if lane_mode else nearest)
            _stage("deform")
            if np.allclose(out, pts, atol=0.05):
                return finish(pts, "follow")
            return finish(out, "deform")
        blocker = _find_blocker(pts, i0, i1, obstacles,
                                CAR_HALF_WIDTH + 0.8)
        _stage("blocker")
        if blocker is not None:
            bd = math.hypot(blocker.x - float(pos[0]),
                            blocker.y - float(pos[1]))
            self.last_blocker = (blocker.label or blocker.category, bd)
        # A long wall that only lines the road side is a boundary, not a
        # detour target.  If the lane itself is too narrow for the car and
        # the wall, stop in front of it instead of steering toward/through
        # it.
        if blocker is not None and self._is_roadside_wall(
                blocker, self._obstacle_route_profile(
                    blocker, pts, i0, i1), pts=pts):
            return finish(pts[: hit + 2], "blocked")
        detour, reached = self._grid_path(
            pts, obstacles, pos, heading, i0, i1, margin=self.margin)
        _stage("grid")
        if detour is not None and len(detour) >= 2 and reached:
            # A detour that actually reaches the route horizon is drivable.
            return finish(detour, "detour")
        # A* could not reach the horizon (wide inflation, a wall of boxes):
        # try a smooth lateral bypass around the first compact blocker
        # before declaring the corridor blocked, so a parked car in the
        # lane becomes a lane change instead of an emergency stop.
        bypass = self._lateral_bypass(pts, obstacles, i0, i1)
        _stage("bypass")
        if bypass is not None and len(bypass) >= 2:
            return finish(bypass, "detour")
        if detour is not None and len(detour) >= 2:
            # Truncated "best reachable cell" path: drive up to the
            # obstacle, then stop in front of it instead of creeping on.
            return finish(detour, "blocked")
        # No drivable way at all: stop in front of the first blocker.
        return finish(pts[: hit + 2], "blocked")

    def _map_legal_lane(self, road_rule: RoadRuleView | None):
        """Return (LegalLaneView|None, map no-cross boundary list)."""
        view = legal_lane_view(road_rule)
        boundaries: list[_MapLaneBoundary] = []
        if view is None or not view.legal or road_rule is None:
            return view, boundaries
        if not (road_rule.in_pos and road_rule.out_pos and road_rule.right_vec):
            return view, boundaries
        p1 = np.asarray(road_rule.in_pos[:2], dtype=float)
        p2 = np.asarray(road_rule.out_pos[:2], dtype=float)
        right = np.asarray(road_rule.right_vec[:2], dtype=float)
        rn = float(np.linalg.norm(right))
        if rn < 1e-9:
            return view, boundaries
        right = right / rn
        for offset_m, allowed_side in view.boundaries:
            a = p1 + right * offset_m
            b = p2 + right * offset_m
            boundaries.append(_MapLaneBoundary(
                np.vstack([a, b]), allowed_side))
        return view, boundaries

    def _boundary_path_offset(self, x: float, y: float, edge_pts,
                              side: float, lane_width: float,
                              corr_max: float
                              = LANE_BOUNDARY_CORRECTION_MAX_M
                              ) -> float | None:
        """Signed lateral shift that keeps a path point clear of an edge.

        ``side`` is -1.0 for a right boundary (painted line / wall /
        guardrail) and +1.0 for a left boundary.  The returned offset is
        positive to the right of the path tangent, matching
        ``_point_lat_offset``; ``None`` means the boundary is too far or
        not clearly on the expected side, so the caller keeps the nav
        route point.  The nav route is the lane centre: a far edge adds
        no correction at all.
        """
        edge_pts = np.asarray(edge_pts[:, :2], dtype=float)
        if len(edge_pts) < 2:
            return None
        lane_width = float(lane_width)
        if not math.isfinite(lane_width) or lane_width <= 0.0:
            lane_width = LANE_WIDTH_DEFAULT_M
        boundary_lat = _point_lat_offset(float(x), float(y), edge_pts)
        if not math.isfinite(boundary_lat):
            return None
        if abs(boundary_lat) > LANE_BOUNDARY_MAX_M:
            return None
        # ``_point_lat_offset`` is positive to the right of the boundary.
        # A right edge has the road on its negative side, a left edge on
        # its positive side.  Only when the nav point is closer than the
        # car's clearance do we shift away from the edge; a route point
        # that already clears it stays untouched.
        clear = LANE_BOUNDARY_CLEAR_M
        if side < 0.0:
            if boundary_lat <= -clear:
                return 0.0
            off = -boundary_lat - clear
        else:
            if boundary_lat >= clear:
                return 0.0
            off = clear - boundary_lat
        corr_max = max(0.0, float(corr_max))
        return max(-corr_max, min(corr_max, off))

    def _right_offset_path(self, pts, i0: int, heading: float,
                           offset: float | None = None):
        """Shift the planning window toward the right-hand side if asked.

        The default is no lateral offset (route centre).  When an offset is
        configured it ramps in over ``right_ramp_m`` so the car leaves the
        nav corridor smoothly instead of jumping sideways at the start.
        Each point is shifted along that point's local route right vector
        (perpendicular to the route tangent), so a bend keeps the offset
        on the same side of the road instead of pushing the path toward
        the outside of the corner.
        """
        target = self.right_offset if offset is None else float(offset)
        if abs(target) < 1e-9:
            return np.asarray(pts, dtype=float)
        src = np.asarray(pts, dtype=float)
        out = src.copy()
        d = np.linalg.norm(np.diff(out, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(d)])
        base = cum[i0]
        n = len(out)
        for i in range(i0, n):
            f = _smoothstep((cum[i] - base)
                            / max(1e-9, self.right_ramp_m))
            a = src[max(0, i - 2)]
            b = src[min(n - 1, i + 2)]
            tv = b - a
            tn = float(np.linalg.norm(tv))
            if tn < 1e-9:
                right = np.array([math.sin(heading), -math.cos(heading)])
            else:
                right = np.array([tv[1] / tn, -tv[0] / tn])
            out[i] = out[i] + right * (f * target)
        return out

    def _obstacle_route_profile(self, ob, pts, i0: int, i1: int):
        """Route-local (lon, lat) extent of an obstacle footprint.

        The window is approximated by one forward direction from ``i0`` to
        ``i1``; lat is positive to the left of travel.  Returns
        ``(lon0, lon1, lat0, lat1)``.
        """
        fwd = pts[i1] - pts[i0]
        fn = float(np.linalg.norm(fwd))
        if fn < 1e-9:
            fwd = np.array([1.0, 0.0])
        else:
            fwd = fwd / fn
        lat = np.array([-fwd[1], fwd[0]])
        p0 = pts[i0]
        corners = _obstacle_corners(ob)
        lons = [float((c[0] - p0[0]) * fwd[0] + (c[1] - p0[1]) * fwd[1])
                for c in corners]
        lats = [float((c[0] - p0[0]) * lat[0] + (c[1] - p0[1]) * lat[1])
                for c in corners]
        return min(lons), max(lons), min(lats), max(lats)

    def _is_roadside_wall(self, ob, profile, pts=None) -> bool:
        """True when the obstacle is a long thin wall beside the route.

        Such a wall is a no-cross road boundary: the planner may drive
        beside it (with clearance) but must never treat it as a compact
        object to swerve around.
        """
        if ob.label != "wall" and ob.category not in ("wall", "raycast"):
            return False
        if _obstacle_oriented(ob):
            length = 2.0 * ob.half_len
            thick = 2.0 * max(0.0, ob.half_thick)
        else:
            length = max(2.0 * ob.half_w, 2.0 * ob.half_h)
            thick = min(2.0 * ob.half_w, 2.0 * ob.half_h)
        if length < ROADSIDE_WALL_MIN_LEN_M:
            return False
        if thick > ROADSIDE_WALL_MAX_THICK_M:
            return False
        # A roadside wall's footprint must stay entirely on one side of the
        # driving polyline.  Using the nearest route segment (instead of the
        # window chord) keeps this correct through bends, where a straight
        # i0->i1 axis makes a diagonal wall look like it crosses the road.
        if pts is not None and len(pts) >= 2:
            lats = [_point_lat_offset(c[0], c[1], pts)
                    for c in _obstacle_corners(ob)]
            pos = sum(1 for v in lats if v > ROADSIDE_WALL_MIN_EDGE_M)
            neg = sum(1 for v in lats if v < -ROADSIDE_WALL_MIN_EDGE_M)
            return pos >= 3 or neg >= 3
        lon0, lon1, lat0, lat1 = profile
        lon_span = lon1 - lon0
        lat_span = lat1 - lat0
        if lon_span < 2.0 * lat_span:
            return False
        # Fallback without a polyline: clearly on one side of the window.
        return lat0 > ROADSIDE_WALL_MIN_EDGE_M \
            or lat1 < -ROADSIDE_WALL_MIN_EDGE_M

    def _side_has_roadside_wall(self, pts, i0: int, i1: int,
                                obstacles, side: float) -> bool:
        """True when a roadside wall lines the chosen bypass side."""
        for ob in obstacles:
            profile = self._obstacle_route_profile(ob, pts, i0, i1)
            if not self._is_roadside_wall(ob, profile, pts=pts):
                continue
            lat0, lat1 = profile[2], profile[3]
            if side < 0 and lat1 < -ROADSIDE_WALL_MIN_EDGE_M:
                return True
            if side > 0 and lat0 > ROADSIDE_WALL_MIN_EDGE_M:
                return True
        return False

    def _edge_clear(self, cand, edge_pts, edge_side: float,
                    clearance: float = LANE_BOUNDARY_CLEAR_M) -> bool:
        """True when a candidate path keeps the car clear of one edge."""
        if edge_pts is None or edge_side == 0.0:
            return True
        edge = np.asarray(edge_pts[:, :2], dtype=float)
        if len(edge) < 2:
            return True
        d = edge[-1] - edge[0]
        dn = float(np.linalg.norm(d))
        if dn < 1e-9:
            return True
        d = d / dn
        lon = (edge - edge[0]) @ d
        lon_max = float(lon.max())
        for p in cand:
            p = np.asarray(p, dtype=float)[:2]
            if float((p - edge[0]) @ d) > lon_max + 2.0:
                continue
            lat = _point_lat_offset(float(p[0]), float(p[1]), edge)
            if not math.isfinite(lat):
                continue
            if edge_side < 0.0 and lat > -clearance:
                return False
            if edge_side > 0.0 and lat < clearance:
                return False
        return True

    def _safe_right_offset(self, pts, i0: int, i1: int, heading: float,
                           obstacles, clearance: float = 0.8,
                           edge_pts=None,
                           edge_side: float = 0.0) -> float:
        """Largest keep-right offset that clears every obstacle footprint.

        The full right offset is preferred on open roads; when a wall or a
        parked object sits on the right, the offset shrinks in 0.1 m steps
        so the path never presses the car into the obstacle.

        One extra rule keeps the car from being dragged back onto the
        centre line by sensor boxes that hug the nav corridor itself: when
        even the zero-offset path is already "hit" (a ghost box, a wall
        AABB that overlaps the route, or an obstacle that sits in the
        lane), shrinking the keep-right offset cannot fix the collision,
        so a small right bias is retained instead of returning 0.  A wall
        that only blocks the right-hand side still shrinks the offset to 0
        (``off=0`` is clear), which is the old safe behaviour.
        """
        if abs(self.right_offset) < 1e-9 or (
                not obstacles and (edge_pts is None or edge_side == 0.0)):
            return self.right_offset
        step = 0.1
        n_steps = int(round(self.right_offset / step)) + 1
        # The lateral shift per point scales linearly with the offset, so
        # compute the unit offset direction once and reuse it for every
        # candidate instead of rebuilding the full shifted path.
        src = np.asarray(pts, dtype=float)
        unit = self._right_offset_path(src, i0, heading, offset=1.0)
        off_dir = unit - src
        for k in range(n_steps):
            off = max(0.0, self.right_offset - k * step)
            cand = src + off_dir * off
            if _path_hit_index(cand, i0, i1, obstacles,
                               CAR_HALF_WIDTH + clearance) < 0 \
                    and self._edge_clear(
                        cand, edge_pts, edge_side):
                return off
        if _path_hit_index(src, i0, i1, obstacles,
                           CAR_HALF_WIDTH + clearance) >= 0:
            # The corridor itself is blocked, not the right-hand lane:
            # keep a modest right bias so a noisy/merged wall box cannot
            # steer the car onto the centre line while approaching it.
            return max(0.5, min(0.8, self.right_offset))
        return 0.0

    def _safe_lateral_offset(self, pts, i0: int, i1: int, heading: float,
                             obstacles, target: float,
                             clearance: float = 0.8) -> float:
        """Largest legal-lane offset that clears every obstacle footprint.

        Works for both LHD (positive-right map offset) and RHD (negative)
        without changing ``RIGHT_OFFSET_M`` default semantics.  Unlike the
        keep-right fallback it never flips to the opposite side: when the
        target side is blocked it eases back to the route centre and lets
        the map boundary clamp / detour logic decide what happens next.
        """
        target = float(target)
        if abs(target) < 1e-9 or not obstacles:
            return target
        src = np.asarray(pts, dtype=float)
        unit = self._right_offset_path(src, i0, heading, offset=1.0)
        off_dir = unit - src
        step = 0.1
        direction = 1.0 if target >= 0.0 else -1.0
        n_steps = int(math.ceil(abs(target) / step))
        for k in range(n_steps + 1):
            off = direction * max(0.0, abs(target) - k * step)
            cand = src + off_dir * off
            if _path_hit_index(cand, i0, i1, obstacles,
                               CAR_HALF_WIDTH + clearance) < 0:
                return off
        return 0.0

    def _lateral_bypass(self, pts, obstacles, i0: int, i1: int):
        """Smooth lane-shift around the first compact blocker.

        Used when the occupancy-grid A* cannot reach the route horizon
        (wide inflated boxes or a wall of obstacles).  The corridor is
        offset laterally across the blocker - like a lane change - and
        blended back into the navigation route after it, so the car
        actually drives around the thing instead of stopping.  A genuine
        wall that fills both sides yields ``None`` and the caller stops.

        Returns a world-frame polyline or None.
        """
        pairs = [ob for ob in obstacles
                 if (ob.half_w >= 0.15 and ob.half_h >= 0.15
                     or _obstacle_oriented(ob))]
        if not pairs:
            return None
        # First obstacle that actually intrudes into the corridor.
        hit_i, hit_ob = -1, None
        for i in range(i0, min(i1, len(pts) - 1)):
            for ob in pairs:
                if _seg_hits_obstacle(
                        pts[i, 0], pts[i, 1],
                        pts[i + 1, 0], pts[i + 1, 1],
                        ob, CAR_HALF_WIDTH + 0.6):
                    hit_i, hit_ob = i, ob
                    break
            if hit_i >= 0:
                break
        if hit_i < 0 or hit_ob is None:
            return None
        # Long roadside walls are boundaries, not obstacles to lane-change
        # around; crossing them (or steering toward them) is what caused the
        # violent left swerve into the wall.
        if self._is_roadside_wall(
                hit_ob, self._obstacle_route_profile(
                    hit_ob, pts, i0, i1), pts=pts):
            return None
        bx, by = hit_ob.x, hit_ob.y
        # Local route frame: forward along the planning window, lateral to
        # the left of the travel direction.
        fwd = pts[i1] - pts[i0]
        fn = float(np.linalg.norm(fwd))
        if fn < 1e-9:
            return None
        fwd = fwd / fn
        lat = np.array([-fwd[1], fwd[0]])
        p0 = pts[i0]
        lon_c = (bx - p0[0]) * fwd[0] + (by - p0[1]) * fwd[1]
        half_lon, half_lat = _obstacle_half_extents(hit_ob, fwd, lat)
        # Lateral path offset needed for the car to clear the box.
        need = half_lat + CAR_HALF_WIDTH + 0.9
        # Clearance zone around the box (footprint + car half width +
        # buffer): the corridor check guarantees this zone stays clear at
        # the chosen offset, so the lane-shift ramp must be *finished*
        # before the path enters it, not taper inside it.
        infl = CAR_HALF_WIDTH + 0.8
        zc0 = lon_c - half_lon - infl
        zc1 = lon_c + half_lon + infl
        hold0 = zc0 - 1.0
        hold1 = zc1 + 1.0
        taper = max(6.0, (zc1 - zc0) / 3.0)
        lon0 = hold0 - taper
        lon1 = hold1 + taper

        def corridor_free(side: float, off: float) -> bool:
            """True when the offset corridor clears every obstacle (the
            blocker itself included) and stays within ``max_dev``."""
            if off > self.max_dev + 1e-9:
                return False
            a = p0 + fwd * lon0 + lat * (off * side)
            b = p0 + fwd * lon1 + lat * (off * side)
            for ob in pairs:
                if _seg_hits_obstacle(
                        a[0], a[1], b[0], b[1], ob,
                        CAR_HALF_WIDTH + 0.8):
                    return False
            return True

        # Prefer the right-hand side (keep-right rule), then the left.
        # A side lined by a roadside wall is skipped entirely: the car must
        # not swerve toward a wall even when the box geometry nominally
        # clears it.
        offset, side = None, 0.0
        for s in (-1.0, 1.0):
            if self._side_has_roadside_wall(pts, i0, i1, obstacles, s):
                continue
            off = need
            while off <= self.max_dev + 1e-9:
                if corridor_free(s, off):
                    offset, side = off, s
                    break
                off += 0.4
            if offset is not None:
                break
        if offset is None:
            return None

        # Longitudinal distance of each window point from i0.
        d = np.linalg.norm(np.diff(pts[i0:i1 + 1], axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(d)])

        def _st(x: float) -> float:
            x = min(1.0, max(0.0, x))
            return x * x * (3.0 - 2.0 * x)

        out = pts[i0:i1 + 1].copy()
        for k in range(len(out)):
            s = float(cum[k])
            if s <= lon0 or s >= lon1:
                f = 0.0
            elif hold1 > hold0 and hold0 <= s <= hold1:
                f = 1.0
            elif s < hold0:
                f = _st((s - lon0) / max(1e-6, hold0 - lon0))
            else:
                f = 1.0 - _st((s - hold1) / max(1e-6, lon1 - hold1))
            out[k] = out[k] + lat * (offset * side * f)
        # One smoothing pass, endpoints stay fixed on the nav corridor.
        nxt = out.copy()
        for i in range(1, len(out) - 1):
            nxt[i] = 0.25 * out[i - 1] + 0.5 * out[i] + 0.25 * out[i + 1]
        return nxt

    def _grid_path(self, pts, obstacles, pos, heading: float,
                   i0: int, i1: int, margin: float = SAFETY_MARGIN):
        """Local occupancy-grid A* detour around blocking obstacles.

        Builds a grid in the car's local frame (x = heading, y = left),
        marks inflated obstacles, and plans a path from the car to the
        global route horizon that hugs the route when possible.  Returns
        a smoothed world-frame polyline (or None when no detour exists) and
        a bool telling whether the path reaches the route horizon.
        """
        import heapq

        hx = math.cos(heading)
        hy = math.sin(heading)
        lx = -hy
        ly = hx
        ox, oy = float(pos[0]), float(pos[1])
        res = GRID_RES
        rows = int(math.ceil((GRID_AHEAD + GRID_BEHIND) / res))
        cols = int(math.ceil(2.0 * GRID_HALF_W / res))
        x0 = -GRID_BEHIND
        y0 = -GRID_HALF_W

        def to_local(wx: float, wy: float):
            dx = wx - ox
            dy = wy - oy
            return dx * hx + dy * hy, dx * lx + dy * ly

        # Car position in grid coordinates (needed to extend obstacles only
        # when they sit clearly ahead of the car).
        def cell_index(wx: float, wy: float):
            f, s = to_local(wx, wy)
            return (int(round((f - x0) / res)), int(round((s - y0) / res)))

        r_start, c_start = cell_index(ox, oy)
        r_start = int(np.clip(r_start, 0, rows - 1))
        c_start = int(np.clip(c_start, 0, cols - 1))
        start = (r_start, c_start)

        # Mark inflated obstacle boxes as occupied (with a 1-cell pad).
        # Each blocking box is extended toward the car by GRID_ANTICIPATE so
        # the detour starts steering well before the obstacle instead of
        # swerving at the last moment.  Only boxes clearly ahead get the
        # extension (a box beside/behind the car must not reach in front of
        # it).
        occ = np.zeros((rows, cols), dtype=bool)
        pad = int(math.ceil(CAR_HALF_WIDTH / res)) + 1
        anticipate_cells = int(math.ceil(GRID_ANTICIPATE / res))
        for ob in obstacles:
            if _obstacle_oriented(ob):
                ux, uy = float(ob.axis[0]), float(ob.axis[1])
                vx, vy = -uy, ux
                hu = ob.half_len + margin
                hv = max(0.0, ob.half_thick) + margin
                corners = [
                    to_local(ob.x + ux * hu + vx * hv,
                             ob.y + uy * hu + vy * hv),
                    to_local(ob.x + ux * hu - vx * hv,
                             ob.y + uy * hu - vy * hv),
                    to_local(ob.x - ux * hu + vx * hv,
                             ob.y - uy * hu + vy * hv),
                    to_local(ob.x - ux * hu - vx * hv,
                             ob.y - uy * hu - vy * hv),
                ]
            else:
                hw = ob.half_w + margin
                hh = ob.half_h + margin
                corners = [
                    to_local(ob.x - hw, ob.y - hh),
                    to_local(ob.x + hw, ob.y - hh),
                    to_local(ob.x + hw, ob.y + hh),
                    to_local(ob.x - hw, ob.y + hh),
                ]
            rxs = [c[0] for c in corners]
            rys = [c[1] for c in corners]
            near_r = int(math.floor((min(rxs) - x0) / res)) - pad
            if near_r > r_start + 12:
                # Obstacle is well ahead: pull its near edge toward the car.
                near_r -= anticipate_cells
            r0 = max(0, near_r)
            r1 = min(rows - 1, int(math.floor((max(rxs) - x0) / res)) + pad)
            c0 = max(0, int(math.floor((min(rys) - y0) / res)) - pad)
            c1 = min(cols - 1, int(math.floor((max(rys) - y0) / res)) + pad)
            if r1 >= r0 and c1 >= c0:
                occ[r0:r1 + 1, c0:c1 + 1] = True

        # Lateral-deviation cost per cell: prefer hugging the global route,
        # but allow real detours when the corridor is blocked.
        dev = np.zeros((rows, cols), dtype=float)
        path_p = pts[max(0, i0 - 5): i1 + 1]
        if len(path_p):
            f_vals = x0 + (np.arange(rows) + 0.5) * res
            s_vals = y0 + (np.arange(cols) + 0.5) * res
            wx = ox + f_vals[:, None] * hx + s_vals[None, :] * lx
            wy = oy + f_vals[:, None] * hy + s_vals[None, :] * ly
            cell_xy = np.stack([wx, wy], axis=-1)
            d2 = ((cell_xy[:, :, None, :]
                   - path_p[None, None, :, :]) ** 2).sum(-1)
            dev = np.sqrt(d2.min(axis=2))
        # Small keep-right preference in the A* cost (local y is left of
        # the car): when both sides of a blocker are reachable the planner
        # chooses the right-hand side instead of cutting left.
        s_vals = y0 + (np.arange(cols) + 0.5) * res
        left_cost = np.maximum(0.0, s_vals[None, :])
        cost = (np.ones((rows, cols), dtype=float)
                + dev * DEV_PENALTY
                + self.grid_right_bias * left_cost)
        cost[occ] = np.inf

        if occ[start]:
            nudge = None
            for rr in range(max(0, r_start - 5), min(rows, r_start + 6)):
                for cc in range(max(0, c_start - 5), min(cols, c_start + 6)):
                    if not occ[rr, cc]:
                        nudge = (rr, cc)
                        break
                if nudge:
                    break
            if nudge is None:
                return None, False
            start = nudge

        # Goal: the route horizon point (i1) projected into the grid.  When
        # that cell is occupied or out of bounds the corridor is effectively
        # blocked: A* will not reach it and the best-reachable-cell fallback
        # below stops the car in front of the obstacle.
        f_h, s_h = to_local(float(pts[i1, 0]), float(pts[i1, 1]))
        r_goal = int(np.clip(round((f_h - x0) / res), 0, rows - 1))
        c_goal = int(np.clip(round((s_h - y0) / res), 0, cols - 1))
        goal = (r_goal, c_goal)
        if goal == start:
            return None, False

        open_heap: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
        g_cost = {start: 0.0}
        came: dict[tuple[int, int], tuple[int, int]] = {}
        closed: set[tuple[int, int]] = set()
        found = False
        while open_heap:
            _, cur = heapq.heappop(open_heap)
            if cur == goal:
                found = True
                break
            if cur in closed:
                continue
            closed.add(cur)
            r, c = cur
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < rows and 0 <= nc < cols):
                        continue
                    if occ[nr, nc]:
                        continue
                    if dr != 0 and dc != 0:
                        # no corner cutting between diagonally-touching walls
                        if occ[r + dr, c] or occ[r, c + dc]:
                            continue
                    step = math.hypot(dr, dc)
                    ncost = g_cost[cur] + cost[nr, nc] * step
                    if ncost < g_cost.get((nr, nc), float("inf")):
                        g_cost[(nr, nc)] = ncost
                        came[(nr, nc)] = cur
                        h = math.hypot((nr - r_goal) * res,
                                       (nc - c_goal) * res)
                        heapq.heappush(open_heap, (ncost + h, (nr, nc)))

        if not found:
            # No route to the horizon: keep the best reachable cell (lowest
            # cost + distance to the goal) so we can at least inch forward.
            best = None
            best_score = float("inf")
            for cell, gc in g_cost.items():
                if occ[cell]:
                    continue
                h = math.hypot((cell[0] - r_goal) * res,
                               (cell[1] - c_goal) * res)
                score = gc + h
                if score < best_score:
                    best_score = score
                    best = cell
            if best is None or best == start:
                return None, False
            goal = best
            reached_goal = False
        else:
            reached_goal = True

        path_cells = []
        cur = goal
        while True:
            path_cells.append(cur)
            if cur == start:
                break
            nxt = came.get(cur)
            if nxt is None:
                return None, False
            cur = nxt
        path_cells.reverse()

        world = []
        for r, c in path_cells:
            f = x0 + (r + 0.5) * res
            s = y0 + (c + 0.5) * res
            world.append((ox + f * hx + s * lx, oy + f * hy + s * ly))
        out = np.asarray(world, dtype=float)
        if len(out) < 3:
            return out, reached_goal
        # Smooth the A* zig-zag while keeping both endpoints fixed.
        for _ in range(2):
            nxt = out.copy()
            for i in range(1, len(out) - 1):
                nxt[i] = 0.25 * out[i - 1] + 0.5 * out[i] + 0.25 * out[i + 1]
            out = nxt
        return out, reached_goal

    def _window(self, route: np.ndarray, nearest: int):
        """Local planning window: (pts, i_start, i_end)."""
        pts = np.asarray(route[:, :2], dtype=float)
        n = len(pts)
        end = n - 1
        if n < 2 or nearest >= n - 1:
            return pts, min(nearest, n - 1), end
        d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(d)])
        base = cum[nearest]
        for i in range(nearest + 1, len(cum)):
            if cum[i] - base > self.horizon_m:
                end = i
                break
        return pts, int(nearest), int(end)

    def _sensor_window(self, lane_center, pos):
        """Local planning window on a sensor lane centre near the car."""
        pts = np.asarray(lane_center[:, :2], dtype=float)
        if len(pts) < 2:
            return pts, 0, max(0, len(pts) - 1)
        pos = np.asarray(pos, dtype=float)[:2]
        nearest = int(np.argmin(np.linalg.norm(pts - pos, axis=1)))
        return self._window(pts, nearest)

    def deform(self, route: np.ndarray, obstacles, nearest: int) -> np.ndarray:
        """Return a route deformed around obstacles (same length, Nx2)."""
        pts, i0, i1 = self._window(route, nearest)
        if not obstacles or i1 <= i0:
            return pts

        work = pts.copy()
        orig = pts.copy()
        # Local travel direction of the planning window.
        fwd = pts[i1] - pts[i0]
        fn = float(np.linalg.norm(fwd))
        if fn < 1e-9:
            fwd = np.array([0.0, 1.0])
        else:
            fwd = fwd / fn
        lat = np.array([-fwd[1], fwd[0]])  # left of the travel direction
        # The obstacle half extents along (fwd, lat) are constant for the
        # whole elastic-band solve; precompute them once instead of per
        # route point per iteration.
        centers = np.empty((len(obstacles), 2), dtype=float)
        half_fwd = np.empty(len(obstacles), dtype=float)
        half_lat = np.empty(len(obstacles), dtype=float)
        wall_side = np.zeros(len(obstacles), dtype=float)
        for k, ob in enumerate(obstacles):
            centers[k, 0] = ob.x
            centers[k, 1] = ob.y
            hf, hl = _obstacle_half_extents(ob, fwd, lat)
            half_fwd[k] = hf
            half_lat[k] = hl
            if self._is_roadside_wall(
                    ob, self._obstacle_route_profile(ob, pts, i0, i1),
                    pts=pts):
                # A long wall beside the road is a one-sided boundary:
                # it may push the path toward the road but never further
                # onto the shoulder.
                wall_lats = [_point_lat_offset(c[0], c[1], pts)
                             for c in _obstacle_corners(ob)]
                if wall_lats:
                    wall_side[k] = (1.0
                                    if float(np.median(wall_lats)) < 0.0
                                    else -1.0)
        need = half_lat + CAR_HALF_WIDTH + self.lateral_clear
        reach = half_fwd + self.anticipate
        denom = np.maximum(reach - half_fwd, 1e-6)
        active = np.arange(i0 + 1, i1 + 1)
        base = orig[active]
        lat_x = float(lat[0])
        lat_y = float(lat[1])
        for _ in range(self.relax_iters):
            cur = work[active]
            diff = cur[:, None, :] - centers[None, :, :]
            fwd_proj = diff[..., 0] * fwd[0] + diff[..., 1] * fwd[1]
            lat_proj = diff[..., 0] * lat[0] + diff[..., 1] * lat[1]
            mask = ((np.abs(fwd_proj) <= reach[None, :])
                    & (np.abs(lat_proj) < need[None, :]))
            # Taper the push before/after the box so the route blends back
            # into the nav corridor instead of kinking.
            taper = 1.0 - np.maximum(
                0.0, np.abs(fwd_proj) - half_fwd[None, :]) / denom[None, :]
            # Constant repulsion inside the cleared band: the elastic band
            # settles at the band edge (need) instead of decaying to a
            # point still inside the inflated box.
            force = self.push_gain * need[None, :] * taper
            # Roadside walls only repel toward the road.  A right-side
            # wall (wall_side=-1) pushes the path left whenever the path
            # is on the wall's side of the wall centre; a left-side wall
            # pushes right.  Ordinary obstacles still repel from both
            # sides so the path can pass either way.
            wall_mask = wall_side[None, :] != 0.0
            wall_dir = np.where(wall_mask, -wall_side[None, :], 0.0)
            wall_allow = np.where(
                wall_mask,
                lat_proj * wall_side[None, :] < need[None, :],
                True)
            side = np.where(
                wall_mask, wall_dir,
                np.where(lat_proj >= 0.0, 1.0, -1.0))
            use_mask = mask & wall_allow
            fx = np.where(use_mask, lat_x * side * force, 0.0).sum(axis=1)
            fy = np.where(use_mask, lat_y * side * force, 0.0).sum(axis=1)
            moved = cur + np.stack([fx, fy], axis=1)
            # Pull back toward the original nav corridor.
            moved += 0.30 * (base - moved)
            dev = moved - base
            nd = np.linalg.norm(dev, axis=1)
            over = nd > self.max_dev
            if np.any(over):
                scale = self.max_dev / nd[over]
                moved[over] = base[over] + dev[over] * scale[:, None]
            work[active] = moved

        # Smooth the deformed section so steering stays gentle.
        for _ in range(self.smooth_passes):
            nxt = work.copy()
            for i in range(i0 + 1, i1):
                nxt[i] = 0.25 * work[i - 1] + 0.5 * work[i] + 0.25 * work[i + 1]
            work = nxt
        return work

    # ---- speed planning ------------------------------------------------

    def speed(
        self,
        route: np.ndarray,
        obstacles,
        pos,
        heading: float,
        nearest: int,
        cruise: float,
    ) -> tuple[float, float]:
        """Return (target_speed, nearest_obstacle_distance).

        The speed is the cruise speed limited by path curvature and by
        obstacles ahead.  Two distinct cases slow the car:

        * a real blocker - an obstacle whose centre sits inside the car's
          own track on the drive path (a parked car in the lane, a wall) -
          is approached with a kinematic braking curve, so the car eases
          off smoothly instead of charging at it and stops in front of it;
        * a tight pass-by - the (already deformed) path runs within
          ``corridor_half_w`` of the obstacle's own footprint (not inflated
          by the car width) - limits the speed so the car does not zip past
          something nearly scraping its mirrors.  A pass-by never pins the
          speed below ``PASS_BY_MIN_MPS`` (sparse raycast specks use the
          higher ``SPECK_PASS_BY_MIN_MPS``): the box is beside the path,
          not blocking it, so stopping every frame (stop-creep twitch) is
          wrong even when the box projects onto the route start.

        Roadside furniture 3 m or more off the route triggers neither, which
        keeps open roads fast.
        """
        self.last_corner = float(max(cruise, 0.0))
        self.last_obs_lim = None
        self.last_sharp = False
        pts = np.asarray(route[:, :2], dtype=float)
        n = len(pts)
        if n < 2:
            return max(cruise, 0.0), 999.0
        self.last_sharp = corner_angle_deg(pts, nearest) >= self.sharp_angle_deg
        v = corner_speed(
            pts, nearest, cruise,
            sharp_angle_deg=self.sharp_angle_deg,
            sharp_speed_mps=self.sharp_corner_kph / 3.6)
        self.last_corner = float(v)
        nearest_obs = 999.0

        # Only inspect obstacles up to ~40 m along the path.
        d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(d)])
        i_end = n - 1
        base = cum[min(nearest, n - 1)]
        for i in range(nearest + 1, len(cum)):
            if cum[i] - base > 40.0:
                i_end = i
                break
        window = slice(min(nearest, n - 1), i_end + 1)
        seg_pts = pts[window]
        if len(seg_pts) < 2:
            return max(v, 0.0), nearest_obs
        win_x0 = float(seg_pts[:, 0].min()) - self.corridor_half_w
        win_x1 = float(seg_pts[:, 0].max()) + self.corridor_half_w
        win_y0 = float(seg_pts[:, 1].min()) - self.corridor_half_w
        win_y1 = float(seg_pts[:, 1].max()) + self.corridor_half_w

        for ob in obstacles:
            bx0, by0, bx1, by1 = _obstacle_aabb(
                ob, self.corridor_half_w)
            if (bx1 < win_x0 or bx0 > win_x1
                    or by1 < win_y0 or by0 > win_y1):
                continue
            # Signed along-route position of the box centre relative to
            # the car.  The polyline projection clamps to the route start
            # (arc_c can never go negative), so also measure along the
            # first segment direction: an obstacle whose centre sits
            # clearly behind the car (roadside furniture we have already
            # passed) must not drag the speed to zero.
            arc_c, lat_c = _point_route_pos_np(ob.x, ob.y, pts)
            u0x = pts[1, 0] - pts[0, 0]
            u0y = pts[1, 1] - pts[0, 1]
            n0 = math.hypot(u0x, u0y)
            rel0 = 0.0 if n0 < 1e-9 else (
                (ob.x - pts[0, 0]) * u0x + (ob.y - pts[0, 1]) * u0y) / n0
            if rel0 < -2.0 and arc_c - base <= 0.0:
                continue
            if arc_c - base >= 40.0:
                continue
            closest = 999.0
            lon = 0.0
            seg_k = 0
            for k in range(len(seg_pts) - 1):
                dd = _obstacle_seg_dist(
                    ob, seg_pts[k, 0], seg_pts[k, 1],
                    seg_pts[k + 1, 0], seg_pts[k + 1, 1])
                if dd < closest:
                    closest = dd
                    lon = cum[min(nearest, n - 1) + k] - base
                    seg_k = k
                if dd < 1e-6:
                    break
            if closest >= 999.0:
                continue
            nearest_obs = min(nearest_obs, closest)
            # The closest point can sit exactly on the window start (the
            # car) even when the box is really a few metres ahead (its
            # footprint reaches back to the car) or off to the side; a
            # bare "0 m" would slam the speed to zero for a box that is
            # still metres away.  Fall back to the box-centre projection
            # in that case.
            if lon <= 0.0:
                lon = max(0.0, arc_c - base)
            if lon >= 40.0:
                continue
            # A box whose centre sits inside the car's own track on the
            # drive path is a real lane blocker: it cannot be driven past,
            # so the kinematic braking curve (possibly down to a stop)
            # applies.  Anything further off the path is a pass-by: only
            # the tight-corridor limit applies and it never pins the speed
            # below a crawl - the car is beside (or will pass beside) the
            # box, so a zero-speed demand every frame is what caused the
            # stop/creep twitch when a roadside grove projected onto the
            # trimmed route start (lon = 0) while the detour ran around it.
            in_lane = abs(lat_c) < CAR_HALF_WIDTH + 0.3
            if in_lane or closest < self.corridor_half_w:
                v_max = math.sqrt(
                    _vehicle_speed_along(ob, seg_pts, seg_k) ** 2
                    + 2.0 * DECEL_MPS2 * max(0.0, lon - STOP_MARGIN_M))
                if not in_lane:
                    floor = (SPECK_PASS_BY_MIN_MPS
                             if is_sparse_raycast_speck(ob)
                             else PASS_BY_MIN_MPS)
                    v_max = max(v_max, floor)
                # Only the obstacle that actually pins the speed may set
                # last_obs_lim; a far-away, higher limit must not mask the
                # creep trigger after a close obstacle already stopped us.
                if v_max < v:
                    v = v_max
                    self.last_obs_lim = float(v_max)
        return max(v, 0.0), nearest_obs


def creep_speed(blocked: bool, obs_lim, desired_speed: float,
                speed: float, since, now: float,
                creep_mps: float = 1.5, hold: float = 1.5):
    """ACC-style creep decision.

    When a kinematic obstacle limit pins the target at 0 but a drivable
    path still exists (``blocked`` is False), inch forward slowly instead
    of parking forever.  Returns ``(target_speed, creep_active, since)``;
    ``since`` is the frame time the pinned state started, or None when the
    car is not in the pinned state.
    """
    if (not blocked and obs_lim is not None
            and obs_lim <= 0.01 and desired_speed <= 0.01
            and speed < 1.0):
        since = since if since is not None else now
        if now - since > hold:
            return float(creep_mps), 1, since
        return float(desired_speed), 0, since
    return float(desired_speed), 0, None
