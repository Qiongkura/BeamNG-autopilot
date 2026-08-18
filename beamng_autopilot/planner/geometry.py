"""Pure geometry and math utilities for path planning."""

from __future__ import annotations

import math

import numpy as np

from .constants import LANE_MIN_CONF, LANE_FULL_CONF


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


def corner_angle_max_deg(points, nearest, back_idx=10, ahead_idx=24,
                         window_m: float = 40.0):
    """Largest heading change over any sub-window of the curvature window.

    An S-bend has alternating direction: the total heading change over
    the whole lookahead cancels out (run 37: 16-25 deg total while the
    road is really two 40+ deg curves), so a total-angle test misses it
    and the car charges into the first curve.  Real drivers judge the
    sharpest sub-curve, so this returns the max heading change over any
    ``window_m``-long sub-window inside the lookahead.
    """
    pts = np.asarray(points[:, :2], dtype=float)
    n = len(pts)
    if n < 4:
        return 0.0
    i0 = max(0, nearest - back_idx)
    i1 = min(n - 1, nearest + ahead_idx)
    sub = pts[i0:i1 + 1]
    if len(sub) < 4:
        return 0.0
    d = np.diff(sub, axis=0)
    seg = np.linalg.norm(d, axis=1)
    angs = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    # Number of points per ~window_m metres; clamp to the available span
    # so a short route still measures its full corner.
    step = max(1, int(round(window_m / max(float(np.median(seg)), 0.5))))
    step = min(step, max(1, len(angs) - 1))
    best = 0.0
    for j in range(0, len(angs) - step):
        da = abs(float(np.degrees(angs[j + step] - angs[j])))
        if da > best:
            best = da
    return best


def corner_angle_max_deg_arc(points, nearest, back_idx=10, ahead_idx=24,
                             window_m: float = 40.0):
    """(angle_deg, arc_m) of the sharpest sub-window of the lookahead.

    The sharpest bend's own arc length, not the whole lookahead span,
    defines its radius: a hairpin followed by a straight has a huge
    total arc over a moderate total angle, which dilutes the radius and
    lets the speed planner under-limit the corner (run 46: 156 deg
    hairpin capped at 11 km/h while the local radius needs ~6 km/h).
    """
    pts = np.asarray(points[:, :2], dtype=float)
    n = len(pts)
    if n < 4:
        return 0.0, 0.0
    i0 = max(0, nearest - back_idx)
    i1 = min(n - 1, nearest + ahead_idx)
    sub = pts[i0:i1 + 1]
    if len(sub) < 4:
        return 0.0, 0.0
    d = np.diff(sub, axis=0)
    seg = np.linalg.norm(d, axis=1)
    angs = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    step = max(1, int(round(window_m / max(float(np.median(seg)), 0.5))))
    step = min(step, max(1, len(angs) - 1))
    best = 0.0
    best_len = 0.0
    for j in range(0, len(angs) - step):
        da = abs(float(np.degrees(angs[j + step] - angs[j])))
        if da > best:
            best = da
            best_len = float(np.sum(seg[j:j + step]))
    return best, best_len


def corner_speed(points, nearest, base_speed, a_lat=4.0,
                 back_idx=10, ahead_idx=24,
                 sharp_angle_deg=45.0, sharp_speed_mps=30.0 / 3.6):
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
    # Use the sharpest sub-curve (S-bends cancel out in a total-angle
    # test and would be driven at cruise speed into the first curve).
    # Its own arc length defines the radius: a hairpin followed by a
    # straight dilutes the radius when measured over the whole lookahead
    # and the car enters the bend too fast (run 46 wall hit).  A single
    # 40 m window still dilutes a tight hairpin (real radius 5-8 m), so
    # measure at two scales and keep the tighter radius.
    total_da, bend_len = corner_angle_max_deg_arc(
        pts, nearest, back_idx, ahead_idx)
    if total_da < 1e-6:
        return base_speed
    radius = max(bend_len, 1e-6) / math.radians(total_da)
    # Tight local scale: the hairpin itself, not the 40 m window.
    if ahead_idx > 6:
        da_loc, len_loc = corner_angle_max_deg_arc(
            pts, nearest, back_idx, ahead_idx, window_m=12.0)
        if da_loc > 1e-6:
            radius = min(radius, max(len_loc, 1e-6)
                         / math.radians(da_loc))
    v = min(base_speed, float(np.sqrt(a_lat * radius)))
    if total_da >= sharp_angle_deg:
        # A sharp bend (>= 45 deg over the lookahead) is not driven at the
        # constant-acceleration limit of the whole arc: the local radius
        # at the bend is what matters, and the car must also hold its lane
        # through it.  Cap by the tighter of the fixed sharp limit and a
        # radius-based limit that keeps the car inside a half-lane offset
        # (a_lat already encodes lateral comfort; halving it models the
        # lane-keeping margin).
        v = min(v, float(sharp_speed_mps))
        v = min(v, float(np.sqrt(a_lat * 0.7 * radius)))
    return v


def _route_spacing_m_impl(pts) -> float:
    """Median segment length (m) of a 2D polyline."""
    p = np.asarray(pts[:, :2], dtype=float)
    if len(p) < 2:
        return 2.0
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    return float(np.median(seg)) if len(seg) else 2.0


def adaptive_lookahead_idx(speed_mps: float,
                           spacing_m: float = 2.0,
                           min_m: float = 60.0,
                           max_m: float = 260.0) -> int:
    """Route points to look ahead for curvature, scaled by speed.

    A fixed 48 m window misses bends that start further out: the highway
    run entered an 80 deg corner 120 m past a straight and charged into
    it at cruise speed (the guardrail hit).  Real ACC looks further ahead
    as speed rises - enough distance to brake from the current speed -
    so the lookahead grows with speed up to a cap.
    """
    want = max(min_m, min(max_m, speed_mps * 14.0))
    return int(math.ceil(want / max(spacing_m, 0.5)))


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


def _clamp_path_lateral(out, center, half_w: float):
    """Pull path points back inside a lateral corridor around ``center``.

    Detour / deform paths around an obstacle can drift onto the shoulder or
    the roadside when the obstacle field is dense; real drivers stay on the
    road surface.  Every point whose lateral offset from the reference
    polyline exceeds ``half_w`` is pulled back along its segment normal so
    the path stays inside the road (the corridor itself can still cross the
    centre line when overtaking).
    """
    out = np.asarray(out, dtype=float)
    if out.ndim != 2 or out.shape[1] < 2 or len(out) == 0:
        return out
    center = np.asarray(center, dtype=float)
    if center.ndim != 2 or center.shape[1] < 2 or len(center) < 2:
        return out
    q, _, lat, normal = _pts_to_segments(out, center)
    over = np.abs(lat) > half_w
    if not np.any(over):
        return out
    clipped = np.clip(lat, -half_w, half_w)
    out2 = out.copy()
    out2[over] = q[over] + normal[over] * clipped[over][:, None]
    return out2


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