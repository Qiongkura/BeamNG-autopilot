"""Nav-route local window: ego-anchored forward part of a route polyline.

The in-game nav route is a map prior; planning and lane-keep must measure
against the *forward* part of the route only.  The old windowing picked
the nearest route vertex and walked the route BACKWARDS whenever that
vertex was a sharp corner (the exit segment starts "opposite" while the
ego still points along the entry segment).  That truncated the local
route to a few metres at the corner vertex, the planner no longer saw
the turn, and the car drove straight through the intersection (town
repro 2026-08-22, runs g8/g10).  This implementation windows the route
by ARC LENGTH around the nearest vertex and never reverses the polyline,
so the bend itself stays inside the local route and the planner keeps
seeing the turn while crossing it.
"""

from __future__ import annotations

import math

import numpy as np

AHEAD_M = 40.0       # default local-route horizon (m)
BACK_M = 2.0         # keep ~2 m of route behind the nearest vertex: at a
                     # hairpin the nearest vertex sits ON the bend, and a
                     # 1 m back-window excluded the entry vertex, so the
                     # local window started after the corner and the
                     # planner never saw the turn (mountain run 2026-08-23).
RESAMPLE_M = 0.8     # resample the route at ~0.8 m before windowing
DUP_MIN_M = 0.35     # drop near-duplicate vertices (a corner stub vertex)


def local_route(pos, heading, nav_route, ahead_m: float = AHEAD_M,
                back_m: float = BACK_M) -> np.ndarray:
    """Return the route in front of the ego, re-anchored at the car.

    ``nav_route`` is the dense world polyline (N, 2).  The returned
    reference starts at (or within ~2 m of) the ego and extends roughly
    ``ahead_m`` along the route.  Falls back to a straight line ahead
    without a usable route.
    """
    pos = np.asarray(pos[:2], dtype=float)
    fwd = np.array([float(np.cos(heading)), float(np.sin(heading))])
    r = np.asarray(nav_route[:, :2], dtype=float) \
        if nav_route is not None and len(nav_route) >= 2 else None
    if r is not None:
        keep = np.ones(len(r), dtype=bool)
        d = np.linalg.norm(np.diff(r, axis=0), axis=1)
        keep[1:] = d > DUP_MIN_M
        r = r[keep]
        # Smooth the polyline through the road nodes: the raw road-graph
        # vertices sit 1-2 m apart with the hairpin turn collapsed into a
        # single corner, so a window anchored at the bend starts ON the
        # hard 90-degree corner - the three-point curvature of the local
        # window then reads nearly straight (speed profile keeps cruise
        # speed into the bend) and the smoothed normals in map_lane_local
        # cross the road (planner sees only arcs and cuts the corner off
        # the road; mountain run 2026-08-23).  Catmull-Rom interpolation
        # rounds the corner so the windowed route / lane geometry / speed
        # profile all see the real curve.
        r = _resample(r)
    if r is not None and len(r) >= 2:
        arc = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(r, axis=0), axis=1))])
        i0 = int(np.argmin(np.linalg.norm(r - pos, axis=1)))
        lo = max(0.0, float(arc[i0]) - back_m)
        hi = min(float(arc[-1]), float(arc[i0]) + ahead_m)
        m = (arc >= lo) & (arc <= hi)
        seg = r[m]
        if len(seg) >= 2:
            # Cut leading vertices BEHIND the car along the route.  The
            # window is anchored at the nearest VERTEX; when the car sits
            # between two route vertices the nearest one is AHEAD, so the
            # back-window includes the previous vertex - which lies
            # BEHIND the car and makes the local route (and the derived
            # map-lane centre) start by pointing BACKWARD along the road
            # (mountain run 2026-08-26 run_fix10: route_local[0] was
            # 1.65 m behind the nose at (728.3, 760.8), the lane centre
            # candidate pointed 121 deg into the oncoming direction and
            # PurePursuit drove the car off-road).  Project the ego onto
            # the polyline and drop everything with smaller arc.
            car_arc = _project_arc(r, pos)
            seg_arc = np.concatenate(
                [[0.0], np.cumsum(np.linalg.norm(np.diff(seg, axis=0),
                                                 axis=1))])
            cut = seg_arc >= (float(car_arc) - float(lo) - 0.3)
            seg = seg[cut]
            if len(seg) >= 2:
                if float(np.linalg.norm(seg[0] - pos)) > 0.6:
                    seg = np.vstack([pos, seg])
                ahead = np.dot(seg - pos, fwd)
                if len(seg) >= 4 or int((ahead > 1.0).sum()) >= 2:
                    return seg
    xs = np.linspace(0, ahead_m, 25)
    return np.column_stack([pos[0] + xs * np.cos(heading),
                            pos[1] + xs * np.sin(heading)])


LANE_WIDTH_DEFAULT_M = 3.5   # own-lane width used for the map prior
LANE_WIDTH_MIN_M = 2.2
# Map-prior own-lane HALF width (m): the lane centre sits this far right
# of the road centreline.  Derived from the real edge spacing (a quarter
# of the road width = half of the two-way own lane) and clamped so an
# oversized junction area never pushes the centre onto a normal road's
# edge.
MAP_LANE_HALF_MIN_M = 1.5
MAP_LANE_HALF_MAX_M = 2.0

# Corner rounding for map-prior lane geometry: the road graph collapses
# a hairpin into ONE sharp vertex, so a parallel offset of the raw
# centreline makes the lane centre a ~1.5 m arc around that vertex - the
# car follows it and cuts the corner off the road (mountain run
# 2026-08-27 run_fix31: the first switchback centre ran through the
# inside of the bend and the car left the road at t=5-9).  Rounding the
# centreline with tangent circular arcs BEFORE offsetting keeps the lane
# centre on the road's real curve (radius ~R + lane half), so the
# reference stays drivable through the hairpin.
CORNER_RADIUS_M = 8.0        # desired hairpin arc radius (m)
CORNER_MIN_ANGLE_DEG = 15.0  # only round sharper-than-this corners
CORNER_STEP_M = 0.35         # arc sampling step (m)
CORNER_RESAMPLE_M = 0.6      # resample rounded centreline to this step
# The lane-centre reference blends from the road centreline at the ego
# into the own-lane offset over this many metres: at a hairpin entry the
# car sits ON the route centreline (the offset lane is a metre to its
# side), and an offset-only centre first points BACKWARD / sideways and
# the car misses the bend (fix37-41 first-bend runs).  With the blend the
# reference follows the road curve through the corner, then settles into
# the right lane on the straight after it.  The blend is kept short so
# the car converges into its own lane before the next curve instead of
# riding the road centreline for the whole approach.
CENTER_BLEND_M = 8.0



def _project_arc(r: np.ndarray, pos: np.ndarray) -> float:
    """Arc length along ``r`` at the closest point to ``pos``."""
    arc = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(r, axis=0), axis=1))])
    best = (1e18, 0.0)
    for i in range(len(r) - 1):
        ax, ay = r[i]
        bx, by = r[i + 1]
        tx, ty = bx - ax, by - ay
        l2 = tx * tx + ty * ty
        if l2 < 1e-12:
            continue
        t = max(0.0, min(1.0, ((pos[0] - ax) * tx + (pos[1] - ay) * ty)
                         / l2))
        cx, cy = ax + t * tx, ay + t * ty
        d = math.hypot(pos[0] - cx, pos[1] - cy)
        if d < best[0]:
            best = (d, float(arc[i]) + t * math.sqrt(l2))
    return best[1]


def _right_normals(r: np.ndarray) -> np.ndarray:
    """Unit right-of-travel normal per route vertex (smoothed tangent)."""
    norm = np.zeros_like(r)
    for i in range(len(r)):
        i0 = max(0, i - 2)
        i1 = min(len(r) - 1, i + 2)
        tv = r[i1] - r[i0]
        L = float(np.linalg.norm(tv))
        if L < 1e-9:
            continue
        norm[i] = np.array([tv[1] / L, -tv[0] / L])
    return norm


def _resample(r: np.ndarray, step: float = RESAMPLE_M) -> np.ndarray:
    """Catmull-Rom resample of a road polyline at ~``step`` metres.

    The road-graph route is a polyline of centreline nodes (1-2 m
    spacing); at a hairpin the turn is one sharp 90-degree corner.  A
    speed/curvature profile computed on the raw polyline sees a kink
    only when the three-point window straddles it - the look-ahead
    braking then misses the bend whenever the window starts after the
    corner.  Resampling with Catmull-Rom interpolates a smooth arc
    through the nodes, so the window, lane geometry, speed profile and
    steering all see a real (rounded) corner instead of a point.
    """
    r = np.asarray(r, dtype=float)
    if r is None or len(r) < 3:
        return r
    d = np.linalg.norm(np.diff(r, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(d)])
    total = float(arc[-1])
    if total <= 0.0:
        return r
    # already dense enough: keep the original vertices
    if float(np.median(d)) <= step * 0.75:
        return r
    n = int(math.ceil(total / step)) + 1
    out = np.zeros((n, 2), dtype=float)
    ts = np.linspace(0.0, total, n)
    i = 0
    for k, s in enumerate(ts):
        while i < len(arc) - 2 and arc[i + 1] < s:
            i += 1
        i1 = i
        i0 = max(0, i1 - 1)
        i2 = min(len(r) - 1, i1 + 1)
        i3 = min(len(r) - 1, i1 + 2)
        seg = max(1e-9, float(arc[i1 + 1] - arc[i1]))
        t = (float(s) - float(arc[i1])) / seg
        t2 = t * t
        t3 = t2 * t
        p0 = r[i0]
        p1 = r[i1]
        p2 = r[i2]
        p3 = r[i3]
        out[k] = 0.5 * (
            (2.0 * p1)
            + (-p0 + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
    return out


def _dedup(r: np.ndarray) -> np.ndarray:
    keep = np.ones(len(r), dtype=bool)
    d = np.linalg.norm(np.diff(r, axis=0), axis=1)
    keep[1:] = d > DUP_MIN_M
    return r[keep]


def _window(r: np.ndarray, pos: np.ndarray, ahead_m: float,
            back_m: float) -> np.ndarray:
    """Arc-length window of ``r`` around the vertex nearest to ``pos``."""
    arc = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(r, axis=0), axis=1))])
    i0 = int(np.argmin(np.linalg.norm(r - pos, axis=1)))
    lo = max(0.0, float(arc[i0]) - back_m)
    hi = min(float(arc[-1]), float(arc[i0]) + ahead_m)
    m = (arc >= lo) & (arc <= hi)
    return r[m]

def _extend_back(r: np.ndarray, m: float = 14.0, step: float = 0.6) -> np.ndarray:
    """Extend a route polyline backward along its entry heading.

    ``_round_corners`` caps a fillet's tangent distance at 45% of the
    remaining polyline arc on each side, so a hairpin only ~8 m from the
    route START gets rounded to a ~3.9 m radius instead of the intended
    8 m - the lane-centre bend collapses into a 1 m kink and the car
    under-steers straight past the first corner (fix37-41 runs: the
    -110 -> -24 deg bend was missed every time).  Prepending a synthetic
    straight extension gives the corner real entry length so the fillet
    keeps its full radius; the extension only feeds the corner rounding
    (the returned polyline is windowed around the ego afterwards).
    """
    r = np.asarray(r, dtype=float)[:, :2]
    if len(r) < 2:
        return r
    d = r[0] - r[1]
    L = float(np.linalg.norm(d))
    if L < 1e-9:
        return r
    u = d / L
    n = max(1, int(math.ceil(float(m) / max(1e-3, step))))
    ext = np.array([r[0] + u * step * (k + 1) for k in range(n)])
    return np.vstack([ext[::-1], r])


def _offset_rounded(r: np.ndarray, d: float, arc_step_m: float = 0.4) -> np.ndarray:
    """Rounded parallel offset of a polyline at distance ``d`` on the
    right-of-travel side.

    Offsetting each vertex along its own normal and CONNECTING the
    offset points with straight segments shortcuts every corner: at a
    switchback the right-edge / lane-centre offset cuts across the
    inside of the turn, so a road-following arc or the lane centre
    itself is read as *outside the lane* and rejected by the no-cross
    gate (mountain run 2026-08-27: after the first switchback the only
    feasible candidate was an off-road loop and the car spun on the
    grass).  A real offset curve rounds the corner with a circular arc
    of radius ``d`` around the corner vertex; this samples that arc
    between the incoming and outgoing smoothed normals so the offset
    polyline stays a true parallel curve (width preserved on straights
    and corners alike).  Degenerate polyline returns ``None``.
    """
    r = np.asarray(r, dtype=float)[:, :2]
    n = len(r)
    if n < 2:
        return None
    norm = _right_normals(r)
    out = [r[0] + d * norm[0]]
    for i in range(1, n - 1):
        a0 = math.atan2(float(norm[i - 1][1]), float(norm[i - 1][0]))
        a1 = math.atan2(float(norm[i][1]), float(norm[i][0]))
        da = (a1 - a0 + math.pi) % (2.0 * math.pi) - math.pi
        if abs(da) > 0.2:
            steps = max(1, int(math.ceil(abs(da) * d / arc_step_m)))
            for k in range(1, steps + 1):
                a = a0 + da * k / steps
                out.append(r[i] + d * np.array([math.cos(a), math.sin(a)]))
        else:
            out.append(r[i] + d * norm[i])
    out.append(r[-1] + d * norm[-1])
    return np.asarray(out)



def map_lane_local(nav_route, pos, heading, lane_width: float =
                   LANE_WIDTH_DEFAULT_M, ahead_m: float = AHEAD_M,
                   back_m: float = BACK_M):
    """Own-lane geometry from the map prior when sensors see no lane.

    Returns ``(center, left, right)`` or ``None``: ``center`` is the
    centre of the ego's own lane (half a lane width RIGHT of the road
    centreline for right-hand traffic - a real stack keeps to its lane,
    never the road centre line), ``left`` is the road centre line (the
    boundary that must never be crossed into oncoming traffic) and
    ``right`` is the road's right edge.  All three are local windows
    around the ego, ordered near->far.
    """
    r = _dedup(np.asarray(nav_route[:, :2], dtype=float)) \
        if nav_route is not None and len(nav_route) >= 2 else None
    if r is None or len(r) < 3:
        return None
    pos = np.asarray(pos[:2], dtype=float)
    lane_m = float(lane_width)
    if lane_m <= 0.0:
        lane_m = LANE_WIDTH_DEFAULT_M
    # ONE shared arc-length window on the ROAD CENTRE polyline; the lane
    # centre and right edge are offsets of the SAME vertices.  The old
    # code windowed each curve independently - on a bend the arc-length
    # indices drift apart and the 'own-lane centre' can land LEFT of the
    # centreline (or on it), so every path that follows the lane got
    # rejected by the hard no-cross rule and the planner steered off the
    # road (town hairpin run 2026-08-22: lane_shift candidates all
    # infeasible with cross=+2.x at the apex, only cross-lot arcs left).
    w = _window(r, pos, ahead_m, back_m)
    if len(w) < 4:
        return None
    # Drop the EGO-ANCHOR vertex from the geometry: ``local_route``
    # prepends the car position when it sits more than ~2 m off the
    # centreline (mountain stall 2026-08-22: the ego was 2.5 m outside
    # the route at the hairpin exit).  That vertex is NOT a road point
    # - it is a cross-field connector to the car, and its direction
    # skews every smoothed normal over the first window metres, which
    # flips the left/right boundaries onto the wrong side of the road.
    # With the anchor stripped, the first road vertex (the nearest
    # centreline point) defines the true road tangent/normals, and the
    # ego sits exactly between left and right instead of outside both.
    w_geom = w
    if float(np.linalg.norm(w[0] - pos)) < 0.5 and len(w) >= 5:
        w_geom = w[1:]
    # Extend the BOUNDARY polylines ~1 m behind the first road vertex
    # along the road tangent.  The window was cut at the ego's arc, so
    # the first road vertex sits at (or just past) the car and the
    # boundaries start exactly there - ``_boundary_lateral`` treats a
    # first vertex as a line END and reports the ego as uncovered, which
    # drops lane-edge telemetry and weakens no-cross enforcement right
    # at the car.  The synthetic back vertex lies ON the road line, so
    # it does not skew the smoothed normals (unlike the ego anchor).
    w_geom_b = w_geom
    if len(w_geom) >= 2:
        s0 = w_geom[1] - w_geom[0]
        L0 = float(np.linalg.norm(s0))
        if L0 > 1e-9:
            back_pt = w_geom[0] - s0 / L0 * 1.0
            w_geom_b = np.vstack([back_pt, w_geom])
    # Round sharp corners on the FULL (back-extended) centreline before
    # deriving the lane: the road graph collapses a hairpin into one
    # vertex, and offsetting that vertex makes the lane centre a ~1.5 m
    # arc the car cuts off the road (see map_lane_edges).  Rounding the
    # WINDOWED polyline caps a hairpin near the route start to a ~3.9 m
    # fillet and collapses the bend (fix37-41 first-bend runs), so the
    # full route is rounded (with a backward straight extension for real
    # entry length) and windowed around the ego afterwards.  The rounded
    # centreline is the left boundary; the centre / right edge are
    # offsets of it.
    _r_full = r
    if len(_r_full) >= 2 and float(np.linalg.norm(
            _r_full[0] - pos)) < 0.5:
        _r_full = _r_full[1:]
    if len(_r_full) < 5:
        return None
    _rc_full = _round_corners(_extend_back(_r_full), CORNER_RADIUS_M)
    _rc_full = _resample(_rc_full, step=CORNER_RESAMPLE_M)
    _arc_f = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(
            np.diff(_rc_full, axis=0), axis=1))])
    _i0f = int(np.argmin(np.linalg.norm(_rc_full - pos, axis=1)))
    _lo_f = max(0.0, float(_arc_f[_i0f]) - back_m)
    _hi_f = min(float(_arc_f[-1]), float(_arc_f[_i0f]) + ahead_m)
    _mf = (_arc_f >= _lo_f) & (_arc_f <= _hi_f)
    w_round = _rc_full[_mf]
    if len(w_round) < 4:
        w_round = _round_corners(w_geom_b, CORNER_RADIUS_M)
    # Keep ~1 m of road behind the first vertex so _boundary_lateral
    # covers the ego (see map_lane_local note above).
    if len(w_round) >= 2:
        _s0 = w_round[1] - w_round[0]
        _L0 = float(np.linalg.norm(_s0))
        if _L0 > 1e-9:
            w_round = np.vstack([w_round[0] - _s0 / _L0 * 1.0, w_round])
    norm = _right_normals(w_round)
    _arc_c = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(w_round, axis=0), axis=1))])
    _car_arc = _project_arc(w_round, pos)
    _i0c = int(np.searchsorted(_arc_c, _car_arc))
    _i0c = min(_i0c, len(w_round) - 1)
    _nrm_c = _right_normals(w_round)
    _lat0 = float(np.dot(pos - w_round[_i0c], _nrm_c[_i0c]))
    _blend = np.clip((_arc_c - _car_arc) / CENTER_BLEND_M, 0.0, 1.0)
    _off = _lat0 + (lane_m * 0.5 - _lat0) * _blend
    center = w_round + _nrm_c * _off[:, None]
    left = w_round
    right = _offset_rounded(w_round, lane_m)
    center = _center_forward(center, w_round, pos, heading)

    return center, left, right


def map_lane_edges(nav_route, left_edge, right_edge, pos, heading,
                   ahead_m: float = AHEAD_M, back_m: float = BACK_M):
    """Own-lane reference built from REAL road edges (DecalRoad).

    ``nav_route`` / ``left_edge`` / ``right_edge`` are the aligned
    outputs of ``RoadNetwork.route_with_edges``: the road centreline
    (A* path over the DecalRoad middles) plus the road's LEFT/RIGHT
    edge polylines sampled at the SAME arc positions, with ``right_edge``
    already oriented onto the route's right side of travel.

    Returns ``(center, left_boundary, right_boundary)`` as local,
    ego-forward windows: ``left_boundary`` is the road centreline (the
    own lane's hard LEFT edge in right-hand traffic), ``right_boundary``
    is the real road right edge, and ``center`` is the centre of the
    ego's own lane, NOT the road centre line.

    The lane centre is a parallel offset of a CORNER-ROUNDED copy of the
    centreline into the right lane, NOT the midpoint between the route
    and the edge polyline: at a junction/switchback the left/right edges
    belong to DIFFERENT road segments (the DecalRoad sampling switches
    roads mid-window), and the midpoint line then cuts diagonally across
    the inside of the corner - the car followed that diagonal onto the
    grass/embankment (mountain run 2026-08-27 run_fix30).  Offsetting
    the smoothed route keeps the centre parallel to the road's true
    curve through the junction.  The corner rounding (``_round_corners``)
    replaces the graph's single sharp hairpin vertex with a tangent arc,
    so the offset centre keeps a drivable radius through the bend; a raw
    offset of the sharp vertex is a ~1.5 m arc that the car cuts off the
    road (mountain run 2026-08-27 run_fix31).  The real edges still
    bound the no-cross gate.
    Returns ``None`` when the polylines are unusable; callers then fall
    back to the synthetic ``map_lane_local`` (fixed 3.5 m lane).
    """
    route = np.asarray(nav_route[:, :2], dtype=float) \
        if nav_route is not None and len(nav_route) >= 2 else None
    left = np.asarray(left_edge[:, :2], dtype=float) \
        if left_edge is not None and len(left_edge) >= 2 else None
    right = np.asarray(right_edge[:, :2], dtype=float) \
        if right_edge is not None and len(right_edge) >= 2 else None
    if route is None or left is None or right is None:
        return None
    n = min(len(route), len(left), len(right))
    route = route[:n]
    left = left[:n]
    right = right[:n]
    # Drop rows with any missing value (defensive; route_with_edges
    # already trims to a common finite window) - all three must stay
    # index-aligned, so filter by ONE shared mask.
    keep = (np.all(np.isfinite(route), axis=1)
            & np.all(np.isfinite(left), axis=1)
            & np.all(np.isfinite(right), axis=1))
    route = route[keep]
    left = left[keep]
    right = right[keep]
    if len(route) < 5:
        return None
    pos = np.asarray(pos[:2], dtype=float)
    # Dedup near-duplicate vertices with the same mask on all polylines.
    d = np.linalg.norm(np.diff(route, axis=0), axis=1)
    m = np.concatenate([[True], d > DUP_MIN_M])
    route = route[m]
    left = left[m]
    right = right[m]
    if len(route) < 5:
        return None
    # Keep a FULL deduped copy for the corner-rounded lane centre: the
    # windowed/trimmed polyline below starts ON the hairpin, so rounding
    # it caps the fillet at 45% of a 2 m entry and collapses the bend
    # into a 1 m kink (the first -110 -> -24 deg bend was missed in
    # every fix37-41 run).  The centre is rounded on the extended FULL
    # route and windowed afterwards.
    route_full = route
    if len(route_full) >= 2 and float(np.linalg.norm(
            route_full[0] - pos)) < 0.5:
        route_full = route_full[1:]
    if len(route_full) < 5:
        return None
    # ONE shared arc-length window on the ORIGINAL (deduped) route for
    # the BOUNDARY polylines; the lane centre is rounded separately
    # below, so the boundaries keep the true road edges (the no-cross
    # gate must never be loosened by the corner rounding).
    arc = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))])
    i0 = int(np.argmin(np.linalg.norm(route - pos, axis=1)))
    lo = max(0.0, float(arc[i0]) - back_m)
    hi = min(float(arc[-1]), float(arc[i0]) + ahead_m)
    m = (arc >= lo) & (arc <= hi)
    route = route[m]
    left = left[m]
    right = right[m]
    if len(route) < 4:
        return None
    # Drop the part BEHIND the car along the route (same trim as
    # local_route) so the boundaries start at (or just past) the ego.
    car_arc = _project_arc(route, pos)
    seg_arc = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))])
    cut = seg_arc >= (float(car_arc) - float(lo) - 0.3)
    if int(cut.sum()) < 3:
        return None
    route = route[cut]
    left = left[cut]
    right = right[cut]
    if len(route) < 3:
        return None
    # Extend the BOUNDARY polylines ~1 m behind the first vertex along
    # the road tangent so _boundary_lateral covers the ego (see
    # map_lane_local for the same back extension).
    s0 = route[1] - route[0]
    L0 = float(np.linalg.norm(s0))
    if L0 > 1e-9:
        back_pt = route[0] - s0 / L0 * 1.0
        route = np.vstack([back_pt, route])
        right = np.vstack([right[0] - s0 / L0 * 1.0, right])
        left = np.vstack([left[0] - s0 / L0 * 1.0, left])
    _wd = np.linalg.norm(np.asarray(right, dtype=float)[:, :2]
                         - np.asarray(route, dtype=float)[:, :2], axis=1)
    _wd = _wd[np.isfinite(_wd)]
    _lane_half = (float(np.clip(np.median(_wd) * 0.25,
                                MAP_LANE_HALF_MIN_M, MAP_LANE_HALF_MAX_M))
                  if _wd.size else float(MAP_LANE_HALF_MAX_M))
    # Lane CENTRE from a corner-rounded copy of the FULL (back-extended)
    # deduped route: round the graph's sharp hairpin vertices into
    # tangent arcs, then window the rounded centreline around the ego
    # and offset it into the right lane.  The windowed boundary
    # polylines above are the no-cross gate; the rounded centre stays
    # inside them (validated offline against _route35.npz switchback).
    # The backward extension gives a hairpin near the route START real
    # entry length so ``_round_corners``'s 45% cap keeps the full arc.
    rc_full = _round_corners(_extend_back(route_full), CORNER_RADIUS_M)
    rc = _resample(rc_full, step=CORNER_RESAMPLE_M)
    arc_r = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(rc, axis=0), axis=1))])
    i0r = int(np.argmin(np.linalg.norm(rc - pos, axis=1)))
    lo_r = max(0.0, float(arc_r[i0r]) - back_m)
    hi_r = min(float(arc_r[-1]), float(arc_r[i0r]) + ahead_m)
    m_r = (arc_r >= lo_r) & (arc_r <= hi_r)
    rc = rc[m_r]
    if len(rc) < 4:
        return None
    # Own-lane centre with a CAR-ANCHORED lateral blend: the offset
    # profile starts at the ego's CURRENT lateral offset to the (rounded)
    # road centreline and converges to the own-lane offset over
    # CENTER_BLEND_M.  A plain parallel offset passes beside a car that
    # sits ON the centreline (route start / hairpin entry) - its nearest
    # point lies behind/sideways and the reference steers the car
    # backward past the bend (fix37-41 first-bend runs).  With the
    # car-anchored profile the reference starts AT the ego, follows the
    # road curve through the corner, then settles into the right lane.
    _arc_c = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(rc, axis=0), axis=1))])
    _car_arc = _project_arc(rc, pos)
    _i0c = int(np.searchsorted(_arc_c, _car_arc))
    _i0c = min(_i0c, len(rc) - 1)
    _nrm_c = _right_normals(rc)
    _lat0 = float(np.dot(pos - rc[_i0c], _nrm_c[_i0c]))
    _blend = np.clip((_arc_c - _car_arc) / CENTER_BLEND_M, 0.0, 1.0)
    _off = _lat0 + (_lane_half - _lat0) * _blend
    center = rc + _nrm_c * _off[:, None]
    if len(center) < 3:
        # Degenerate offset (sharp self-touching corner): fall back to
        # the edge midpoint so the lane reference still exists.
        center = (route + right) * 0.5
        return _ego_forward(center, pos)
    center = _center_forward(center, rc, pos, heading)

    if len(center) < 3:
        return None
    # LEFT boundary = the ROUNDED road centreline, not the raw graph
    # polyline: the own-lane centre is a parallel offset of the same
    # rounded centreline, so the no-cross gate must measure against the
    # same shape.  The raw route keeps the graph's sharp hairpin vertex,
    # and the rounded (8 m fillet) centreline lies on the INSIDE of the
    # real curve there - an offset own lane of the rounded road can sit
    # a metre LEFT of the raw centreline at the apex and every in-lane
    # lane_center / lane_shift candidate then reads as a centre-line
    # crossing (first hairpin: lane_center rejected with cross=+5.1 at
    # the spawn, the planner fell back to straight arcs and the car ran
    # past the bend).  map_lane_local already uses the rounded window as
    # its left boundary; this keeps map_lane_edges consistent.  The real
    # right edge stays the hard right boundary (never loosened).
    return center, rc, right


def _round_corners(r: np.ndarray, radius: float = CORNER_RADIUS_M,
                   min_angle_deg: float = CORNER_MIN_ANGLE_DEG,
                   step: float = CORNER_STEP_M) -> np.ndarray:
    """Replace sharp polyline corners with tangent circular arcs.

    A corner is a vertex whose ADJACENT segments turn more than
    ``min_angle_deg`` (the road graph collapses a hairpin into one sharp
    vertex, so the turn is visible in the two segments around it, not in
    a wider chord).  The replacement arc is tangent to the
    incoming/outgoing SEGMENT directions, so the tangent points can sit
    several metres before/after the apex - a fillet limited to the short
    apex segments would stay a ~1 m kink.  The tangent distance is
    capped at 45% of the remaining polyline arc on each side so the arc
    always stays inside the route (a switchback near the route start is
    rounded with a proportionally smaller radius instead of producing an
    out-of-order arc).  The output is rebuilt in arc order: original
    vertices before T1, then the arc, then vertices after T2.  Returns a
    new polyline (denser at the rounded corners).
    """
    r = np.asarray(r, dtype=float)[:, :2]
    n = len(r)
    if n < 4:
        return r
    d = np.linalg.norm(np.diff(r, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(d)])
    total = float(arc[-1])

    corners = []
    for i in range(1, n - 1):
        u1 = r[i] - r[i - 1]
        u2 = r[i + 1] - r[i]
        L1 = float(np.linalg.norm(u1))
        L2 = float(np.linalg.norm(u2))
        if L1 < 0.5 or L2 < 0.5:
            continue
        u1 = u1 / L1
        u2 = u2 / L2
        ang = math.acos(float(np.clip(np.dot(u1, u2), -1.0, 1.0)))
        if math.degrees(ang) < min_angle_deg:
            continue
        t_req = radius * math.tan(ang / 2.0)
        t = min(t_req, 0.45 * float(arc[i]), 0.45 * (total - float(arc[i])))
        if t <= 0.3:
            continue
        # Tangent directions over a WINDOW of the polyline, not just the
        # immediate vertex pair.  The road graph places vertices 1-2 m
        # apart and the exit leg bends a few degrees per vertex, so a
        # fillet tangent to the FIRST segment after the corner lands off
        # the exit polyline and the junction after T2 becomes a ~1 m kink.
        # That kink caps the rounded-route speed profile at ~2 m/s (the
        # first hairpin was taken at a speed the steering cannot execute)
        # and makes the lane-centre offset wobble at the apex.  Averaging
        # the direction over roughly the fillet's own tangent distance
        # keeps T1/T2 on the real entry/exit lines.
        win1 = max(2, int(math.ceil(min(max(t, 0.1), 14.0) / max(L1, 0.5))))
        i1b = max(0, i - win1)
        v1w = r[i] - r[i1b]
        L1w = float(np.linalg.norm(v1w))
        if L1w > 1e-6:
            u1 = v1w / L1w
        win2 = max(2, int(math.ceil(min(max(t, 0.1), 14.0) / max(L2, 0.5))))
        i2f = min(n - 1, i + win2)
        v2w = r[i2f] - r[i]
        L2w = float(np.linalg.norm(v2w))
        if L2w > 1e-6:
            u2 = v2w / L2w
        ang = math.acos(float(np.clip(np.dot(u1, u2), -1.0, 1.0)))
        if math.degrees(ang) < min_angle_deg:
            continue
        t_req = radius * math.tan(ang / 2.0)
        t = min(t_req, 0.45 * float(arc[i]), 0.45 * (total - float(arc[i])))
        if t <= 0.3:
            continue
        T1 = r[i] - u1 * t
        T2 = r[i] + u2 * t
        if corners and float(np.dot(T1 - corners[-1]["T2"], u1)) < 0.0:
            continue
        cross = float(u1[0] * u2[1] - u1[1] * u2[0])
        nl = np.array([-u1[1], u1[0]])
        # Effective radius from the CAPPED tangent distance: the arc must
        # be tangent at BOTH T1 and T2, so when the tangent distance was
        # capped (switchback near the route start) the arc radius shrinks
        # to fit; drawing the requested radius through the two points
        # would leave the arc misaligned with the outgoing segment and
        # create a zigzag after the corner.
        rad_eff = t / math.tan(ang / 2.0)
        C = T1 + rad_eff * nl if cross > 0 else T1 - rad_eff * nl
        a1 = math.atan2(T1[1] - C[1], T1[0] - C[0])
        a2 = math.atan2(T2[1] - C[1], T2[0] - C[0])
        da = (a2 - a1 + math.pi) % (2.0 * math.pi) - math.pi
        if cross > 0:
            if da < 0:
                da += 2.0 * math.pi
        else:
            if da > 0:
                da -= 2.0 * math.pi
        n_arc = max(1, int(math.ceil(abs(da) * rad_eff / step)))
        pts = []
        for k in range(1, n_arc):
            aa = a1 + da * k / n_arc
            pts.append(C + rad_eff * np.array([math.cos(aa), math.sin(aa)]))
        corners.append({"T1": T1, "T2": T2, "t": t, "arc": float(arc[i]),
                        "pts": pts})

    out = []
    i = 0
    for c in corners:
        while i < n and float(arc[i]) < c["arc"] - c["t"] + 1e-9:
            out.append(r[i])
            i += 1
        out.append(c["T1"])
        out.extend(c["pts"])
        out.append(c["T2"])
        while i < n and float(arc[i]) <= c["arc"] + c["t"] + 1e-9:
            i += 1
    while i < n:
        out.append(r[i])
        i += 1
    out = np.asarray(out)
    keep = np.ones(len(out), dtype=bool)
    dd = np.linalg.norm(np.diff(out, axis=0), axis=1)
    keep[1:] = dd > 0.05
    return out[keep]


def _resample_aligned(route, left, right, step: float = RESAMPLE_M):
    """Resample route/left/right at the same arc positions.

    Catmull-Rom resample of the centreline (rounds hairpin kinks); the
    edge polylines are linearly interpolated at the same arc fractions so
    the three stay index-aligned (the planner consumes them as one lane).
    Dense polylines are returned unchanged.
    """
    route = np.asarray(route, dtype=float)[:, :2]
    left = np.asarray(left, dtype=float)[:, :2]
    right = np.asarray(right, dtype=float)[:, :2]
    n = len(route)
    if n < 3:
        return None, None, None
    d = np.linalg.norm(np.diff(route, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(d)])
    total = float(arc[-1])
    if total <= 0.0:
        return None, None, None
    if float(np.median(d)) <= step * 0.75:
        return route, left, right
    route_s = _resample(route, step=step)
    n_out = len(route_s)
    left_s = np.zeros((n_out, 2), dtype=float)
    right_s = np.zeros((n_out, 2), dtype=float)
    ts = np.linspace(0.0, total, n_out)
    i = 0
    for k, s in enumerate(ts):
        while i < n - 2 and arc[i + 1] < s:
            i += 1
        seg = max(1e-9, float(arc[i + 1] - arc[i]))
        t = min(1.0, max(0.0, float(s - arc[i]) / seg))
        left_s[k] = left[i] + t * (left[i + 1] - left[i])
        right_s[k] = right[i] + t * (right[i + 1] - right[i])
    return route_s, left_s, right_s


def _ego_forward(poly, pos, prepend_max_m: float = 0.3):
    """Trim a reference polyline to its forward part and re-anchor at ``pos``.

    ``poly`` is ordered near->far along the lane/route.  The car's
    projection defines the arc where the ego sits; every vertex behind
    that arc is a point the car has already passed and must not steer
    toward (a behind vertex makes the whole path look like it first
    drives backward / sideways).  The returned path starts at (or within
    ``prepend_max_m`` of) ``pos`` and extends forward.  Degenerate
    results keep the original polyline (plus an ego anchor when far).
    """
    poly = np.asarray(poly, dtype=float)[:, :2]
    if len(poly) < 2:
        return poly
    pos = np.asarray(pos[:2], dtype=float)
    car_arc = _project_arc(poly, pos)
    arc = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))])
    seg = poly[arc >= float(car_arc) - 1e-6]
    if len(seg) < 2:
        seg = poly
    if float(np.linalg.norm(seg[0] - pos)) > prepend_max_m:
        seg = np.vstack([pos, seg])
    return seg


def _ego_reanchor(poly, pos, heading, min_ahead_m: float = 1.5):
    """Re-anchor a lane-centre polyline at the ego by FORWARD projection.

    ``_ego_forward`` trims at the ego's arc along the polyline, which for
    an offset own-lane centre picks the PERPENDICULAR point - when the
    car sits ON the road centreline (route start / hairpin entry), that
    point lies BEHIND the car and the path first points backward, so the
    car misses the bend (fix37-41 first-bend runs).  This trims at the
    first vertex at least ``min_ahead_m`` FORWARD of the ego along the
    car's heading and connects it with an ego anchor: the first segment
    is a short convergence (right into the own lane / toward the road),
    then the polyline follows the lane curve.  Degenerate inputs keep
    ``_ego_forward``'s behaviour.
    """
    poly = np.asarray(poly, dtype=float)[:, :2]
    pos = np.asarray(pos[:2], dtype=float)
    if len(poly) < 2:
        return poly
    fwd = np.array([float(np.cos(heading)), float(np.sin(heading))])
    d = poly - pos
    proj = d[:, 0] * fwd[0] + d[:, 1] * fwd[1]
    idx = int(np.argmax(proj >= float(min_ahead_m))) if (proj >= float(min_ahead_m)).any() else -1
    if idx <= 0:
        return _ego_forward(poly, pos)
    seg = np.vstack([pos, poly[idx:]])
    return seg


def _center_forward(center, rc, pos, heading, prepend_max_m: float = 0.6,
                    recover_m: float = 6.0):
    """Trim the offset own-lane centre to the ego's forward part while
    keeping the road's curve.

    ``_ego_forward`` projects the ego onto the OFFSET centre polyline, so
    at a hairpin entry (the ego still approaches along the entry straight
    while the own-lane centre already curves through the apex) the
    projection lands on the POST-apex arc and the whole pre-bend approach
    is cut - the reference then points at the exit and the car runs
    straight past the bend (fix37-41 first-bend runs: the first
    -110 -> -24 deg hairpin was missed every time).  This trims at the
    ego's projection on the ROUNDED ROAD CENTRELINE ``rc`` (the road
    shape the centre is offset from), so the entry/curve stays ahead and
    the car starts steering as soon as the road bends.  When the ego has
    drifted far off-route (recovery) or the road at the projection points
    backward/sideways relative to the ego, the ego-anchored
    ``_ego_forward`` is kept so a backward/sideways reference never
    appears.
    """
    center = np.asarray(center[:, :2], dtype=float)
    rc = np.asarray(rc[:, :2], dtype=float)
    pos = np.asarray(pos[:2], dtype=float)
    if len(center) < 3 or len(rc) < 3 or len(center) != len(rc):
        return _ego_forward(center, pos)
    car_arc = _project_arc(rc, pos)
    arc = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(center, axis=0), axis=1))])
    i0 = int(np.searchsorted(arc, car_arc))
    i0 = min(i0, len(center) - 1)
    if float(np.linalg.norm(rc[i0] - pos)) > recover_m:
        return _ego_forward(center, pos)
    i1 = min(len(rc) - 1, i0 + 1)
    i2 = max(0, i0 - 1)
    t0 = rc[i1] - rc[i2]
    L0 = float(np.linalg.norm(t0))
    fwd = np.array([float(np.cos(heading)), float(np.sin(heading))])
    if L0 < 1e-9 or float(np.dot(t0 / L0, fwd)) <= 0.0:
        return _ego_forward(center, pos)
    seg = center[i0:]
    if len(seg) < 2:
        return _ego_forward(center, pos)
    if float(np.linalg.norm(seg[0] - pos)) > prepend_max_m:
        seg = np.vstack([pos, seg])
    return seg

