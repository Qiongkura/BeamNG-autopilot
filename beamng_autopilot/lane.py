"""Sensor lane framing: markings -> lane centre, LiDAR -> free corridor.

The vision detector returns individual road-marking polylines.  This
module pairs them into left / right lane boundaries and derives the
lane-centre polyline the planner should follow, so the car can stay in
the middle of a single lane under right-hand traffic.  When the camera
has no usable markings, the synthetic LiDAR fan (raycast hits) provides
a free-space corridor between the nearest left / right hits.

No map or road-network data is used here: everything comes from the
front camera and the physics raycast fan.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from beamng_autopilot.vision.lanes import LaneMarking

LANE_WIDTH_DEFAULT_M = 3.5
LANE_WIDTH_MIN_M = 2.2
LANE_PAIR_WIDTH_MIN_M = 1.5
LANE_WIDTH_MAX_M = 6.5
LANE_VISION_PAIR_WIDTH_MAX_M = 6.5
LANE_FUSION_WIDTH_MAX_M = 6.5
LANE_MIN_SPAN_M = 4.0
LANE_BOUNDARY_SPAN_M = 1.5
LANE_PAIR_OVERLAP_M = 1.5
LANE_FRAME_MIN_SPAN_M = 3.0
LANE_PAIRED_VISION_MIN_SPAN_M = 1.5
LANE_EDGE_MAX_M = 5.0
LANE_SINGLE_MIRROR_MAX_M = 3.5
LANE_SINGLE_LIDAR_CENTER_MAX_M = 0.35
LANE_SINGLE_NEAR_MAX_M = 4.0
LANE_SINGLE_NEAR_REQUIRE_M = 6.0
LANE_SINGLE_MED_MIN_M = 0.08
LANE_SINGLE_RIGHT_EDGE_MIN_M = 0.4
LANE_RIDING_LINE_MAX_M = 0.6
LANE_RIGHT_MIRROR_NEAR_M = 3.0
LANE_ONE_NEAR_FAR_START_MAX_M = 8.0
LANE_PAIR_NEAR_MAX_M = 5.5
LANE_PAIR_CENTER_MAX_M = 1.75
LANE_PAIR_NEAR_CENTER_MAX_M = 1.75
LANE_OFF_CENTER_WIDTH_MAX_M = 5.8
AXIS_MERGE_MAX_M = 0.9
LANE_PAIR_CENTER_PREFER_M = 0.7
LANE_MIN_CONF = 0.30
VISION_TRUST_CONF = 0.40
LANE_FUSION_AGREE_MAX_M = 1.5
LANE_FUSION_CENTER_CONT_MAX_M = 0.6
LANE_FUSION_PAIRED_CONF_MAX = 0.62
FUSION_WALL_SAFE_MARGIN_M = 0.45
LANE_VISION_MIRROR_CENTER_MAX_M = 1.0
LANE_VISION_RIGHT_MIRROR_CENTER_MAX_M = 1.75
LANE_VISION_RIGHT_MIRROR_CONF_MAX = 0.45
LANE_FAR_MIRROR_CONF_MAX = 0.42
LANE_FAR_CENTER_PAIR_CONF_MAX = 0.52
LANE_FAR_CENTER_PAIR_MIN_WIDTH_M = 4.5
LANE_FAR_START_MAX_M = 22.0
LANE_FUSION_HOLD_FRAMES = 3
LANE_FUSION_PAIRED_HOLD_FRAMES = 4
LANE_FUSION_HOLD_NONE_FRAMES = 4
MARKING_ALIGNMENT_MIN = 0.65
LIDAR_MAX_DIST_M = 24.0
LIDAR_STATION_M = 1.5
LIDAR_MAX_LAT_M = 6.0
LIDAR_EDGE_MIN_M = 0.5
TRACK_MAX_DIST_M = 20.0
TRACK_STATION_M = 1.5
TRACK_JUMP_MAX_M = 1.2
TRACK_REJECT_MATCH_M = 0.6
TRACK_REJECT_WINDOW_S = 2.5
TRACK_STALE_S = 3.0
TRACK_STALE_M = 10.0


@dataclass
class LaneFrame:
    """A local lane reference in world coordinates.

    ``center`` is the polyline the planner should follow (the middle of
    the detected lane / corridor).  ``left`` / ``right`` are the detected
    boundaries when both sides are available.
    """

    center: np.ndarray          # (N, 2) world points
    left: np.ndarray | None = None   # (M, 2) world points or None
    right: np.ndarray | None = None  # (M, 2) world points or None
    width: float = LANE_WIDTH_DEFAULT_M
    confidence: float = 0.0
    span_m: float = 0.0
    sources: tuple[str, ...] = ("vision",)
    paired: bool = True   # True when both lane edges are real detections
                         # (or a two-sided LiDAR corridor), False for a
                         # single-edge mirror fallback
    left_kind: str | None = None   # kind of the left boundary marking
    right_kind: str | None = None  # kind of the right boundary marking


def _unit_fwd(pos, heading: float, fwd=None) -> np.ndarray:
    if fwd is not None:
        f = np.asarray(fwd, dtype=float)[:2]
        n = float(np.linalg.norm(f))
        if n > 1e-9:
            return f / n
    return np.array([math.cos(heading), math.sin(heading)])


def _marking_axis(world: np.ndarray) -> tuple[np.ndarray, float]:
    pts = np.asarray(world, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 2 or len(pts) < 2:
        return np.array([1.0, 0.0]), 1.0
    d = pts[:, :2] - pts[:, :2].mean(axis=0)
    cov = d.T @ d
    vals, vecs = np.linalg.eigh(cov)
    axis = vecs[:, int(np.argmax(vals))]
    return axis, float(np.linalg.norm(axis))


def _marking_score(mk: LaneMarking, span_m: float) -> float:
    kind_w = {"solid": 1.0, "dashed": 0.9,
              "thin": 0.7, "unknown": 0.55}.get(
        mk.kind, 0.5)
    span_w = min(1.0, span_m / 10.0)
    return float(mk.confidence) * kind_w * (0.5 + 0.5 * span_w)


def _boundary_min_span(mk: LaneMarking, min_span: float) -> float:
    """Real lines may be short; unknown blobs need the caller's span."""
    if mk.kind in ("solid", "dashed", "thin"):
        return LANE_BOUNDARY_SPAN_M
    return min_span


def _boundary_projection(world, pos: np.ndarray, fwd: np.ndarray,
                         left: np.ndarray) -> np.ndarray:
    """Project a marking polyline onto the car frame as (s, lat) rows."""
    pts = np.asarray(world, dtype=float)[:, :2]
    rel = pts - pos
    s = rel @ fwd
    lat = rel @ left
    order = np.argsort(s)
    return np.column_stack([s[order], lat[order]])


def _interp_lat(proj: np.ndarray, stations: np.ndarray) -> np.ndarray:
    if proj is None or len(proj) < 2:
        return np.full(stations.shape, np.nan)
    s = proj[:, 0]
    lat = proj[:, 1]
    return np.interp(stations, s, lat, left=np.nan, right=np.nan)


def _overlap_stations(s_lo: float, s_hi: float,
                      station_step: float,
                      max_stations: int) -> np.ndarray:
    """Longitudinal samples across a candidate overlap.

    The fixed arc step would leave a short but real overlap (a couple of
    metres of painted line beside the car) with only two samples, which
    is not enough to interpolate a pair.  A short overlap is instead
    sampled with at least three stations spread over the whole interval;
    longer overlaps still follow ``station_step`` up to ``max_stations``.
    """
    overlap = float(s_hi - s_lo)
    count = max(3, int(round(overlap / station_step)) + 1)
    count = min(max_stations, count)
    return np.linspace(float(s_lo), float(s_hi), count)


@dataclass
class _LineCandidate:
    """A projected marking usable as a lane boundary."""

    proj: np.ndarray          # (N, 2) (s, lat) rows
    span: float
    conf: float
    kind: str
    color: str
    med_lat: float
    score: float


def _collect_candidates(markings, pos: np.ndarray, fwd: np.ndarray,
                        min_span: float) -> list[_LineCandidate]:
    """Project every aligned marking into the car frame."""
    left = np.array([-fwd[1], fwd[0]])
    cands: list[_LineCandidate] = []
    for mk in markings:
        # A dark pavement patch / tree shadow comes back as ``unknown``.
        # It may be long and confident, but it is not a painted lane line
        # and must never pair with a real line into a fake lane.
        if mk.kind not in ("solid", "dashed", "thin"):
            continue
        world = np.asarray(mk.world, dtype=float)
        if world.ndim != 2 or world.shape[1] < 2 or len(world) < 2:
            continue
        axis, _ = _marking_axis(world)
        if float(axis @ fwd) < 0:
            axis = -axis
        if abs(float(axis @ fwd)) < MARKING_ALIGNMENT_MIN:
            continue
        proj = _boundary_projection(world, pos, fwd, left)
        if len(proj) < 2:
            continue
        span = float(proj[-1, 0] - proj[0, 0])
        if span < _boundary_min_span(mk, min_span):
            continue
        med_lat = float(np.median(proj[:, 1]))
        if abs(med_lat) > LANE_EDGE_MAX_M:
            continue
        cands.append(_LineCandidate(
            proj=proj, span=span, conf=float(mk.confidence),
            kind=mk.kind, color=mk.color, med_lat=med_lat,
            score=_marking_score(mk, span)))
    return cands


def _axis_candidates(cands: list[_LineCandidate], station_step: float,
                     max_stations: int) -> list[_LineCandidate]:
    """Merge the two edges of a line straddled by the car into one axis.

    When the camera sees the left and right edge of the same painted line
    (the car is on or next to it), the pair is too narrow to be a lane.
    Averaging the edges gives the physical line position, which can then
    pair with the far edge of the lane on the other side.
    """
    axes: list[_LineCandidate] = []
    for i, ci in enumerate(cands):
        if ci.med_lat <= 0.08:
            continue
        for cj in cands:
            if cj.med_lat >= -0.08:
                continue
            if ci.med_lat - cj.med_lat > AXIS_MERGE_MAX_M:
                continue
            s_lo = max(0.0, float(ci.proj[0, 0]), float(cj.proj[0, 0]))
            s_hi = min(float(ci.proj[-1, 0]), float(cj.proj[-1, 0]))
            if s_hi - s_lo < LANE_PAIR_OVERLAP_M:
                continue
            stations = _overlap_stations(s_lo, s_hi, station_step,
                                         max_stations)
            la = _interp_lat(ci.proj, stations)
            lb = _interp_lat(cj.proj, stations)
            valid = ~(np.isnan(la) | np.isnan(lb))
            if int(np.sum(valid)) < 3:
                continue
            lat = 0.5 * (la + lb)
            axes.append(_LineCandidate(
                proj=np.column_stack([stations, lat]),
                span=s_hi - s_lo,
                conf=0.5 * (ci.conf + cj.conf),
                kind="axis",
                color="axis",
                med_lat=float(np.median(lat[valid])),
                score=0.5 * min(ci.score, cj.score)))
    return axes


def _yellow_right_ok(c: _LineCandidate) -> bool:
    """A yellow marking is only a right boundary when clearly on the
    right side of the car.

    Under right-hand traffic the centre line is usually the left edge of
    the current lane, so a yellow line sitting just to the right of the
    car is the middle of the road the car is riding, not the lane's right
    edge.  Mirroring it puts the centre across the car and drags the car
    left of the paint.  Only yellow lines well to the right can act as a
    genuine right boundary.
    """
    if c.color != "yellow":
        return True
    near = c.proj[c.proj[:, 0] <= LANE_SINGLE_NEAR_REQUIRE_M]
    if len(near) < 2:
        return False
    return float(np.median(near[:, 1])) <= -1.5


def _near_stats(c: _LineCandidate) -> tuple[int, float | None]:
    """Number of valid near points and their median lateral offset.

    ``None`` for the median means the marking has no trustworthy points
    next to the car, so a far roadside line can still be used as a weak
    single boundary but must not by itself seed a paired lane.
    """
    near = c.proj[c.proj[:, 0] <= LANE_SINGLE_NEAR_REQUIRE_M]
    if len(near) < 2:
        return 0, None
    if not bool(np.all(np.abs(near[:, 1]) <= LANE_PAIR_NEAR_MAX_M)):
        return 0, None
    return int(len(near)), float(np.median(near[:, 1]))


def _best_vision_pair(cands: list[_LineCandidate],
                      axes: list[_LineCandidate],
                      station_step: float, max_stations: int):
    """Choose the most plausible single-lane boundary pair.

    Every left/right candidate combination is checked for a lane-like
    width.  The winning pair is the one that is confident, long, close to
    the car and centred on the car; this stops a far roadside line from
    beating the real edge of the current lane.  One exception is a far
    dashed/thin centre line paired with a real right line: when the two
    form a wide lane the midpoint is still the best read (run 84).
    """
    left_pool = [c for c in cands if c.med_lat > 0.08]
    left_pool += [a for a in axes if a.med_lat >= 0.0]
    right_pool = [c for c in cands if c.med_lat < -0.08]
    right_pool += [a for a in axes if a.med_lat < 0.0]
    right_pool = [c for c in right_pool if _yellow_right_ok(c)]
    # A paired lane usually has to start next to the car on at least one
    # side: real camera lines often begin a few metres ahead on one edge,
    # but a pair of far-away roadside lines must never be mistaken for the
    # current lane (run 57 solid-line false block).  A far centre-line
    # pair is handled per-candidate below.
    left_near = {id(c): _near_stats(c) for c in left_pool}
    right_near = {id(c): _near_stats(c) for c in right_pool}
    best = None
    best_score = 0.0
    for l in left_pool:
        ln, _ = left_near[id(l)]
        for r in right_pool:
            rn, _ = right_near[id(r)]
            if ln < 2 and rn < 2:
                # Two far thin lines are usually roadside / pavement
                # fragments, not a real lane pair: pairing them gives a
                # phantom centre that can drag the car across the road.
                if l.kind == "thin" and r.kind == "thin":
                    continue
                # A wide lane whose centre line and right edge only become
                # visible ahead is still a real two-sided lane.  Without
                # this, run 84 mirrored the right edge and steered beside
                # the right paint instead of between the two lines.
                if not (l.kind in ("dashed", "thin")
                        and r.kind in ("solid", "dashed", "thin")
                        and l.proj[0, 0] <= LANE_FAR_START_MAX_M
                        and r.proj[0, 0] <= LANE_FAR_START_MAX_M):
                    continue
            # The far member of a one-near pair has to look like a real
            # road line, not a short unknown pavement patch.
            if ln < 2 and (l.kind not in ("solid", "dashed", "thin")
                           or l.proj[0, 0] > LANE_FAR_START_MAX_M):
                continue
            if rn < 2 and (r.kind not in ("solid", "dashed", "thin")
                           or r.proj[0, 0] > LANE_FAR_START_MAX_M):
                continue
            # One edge beside the car plus an opposite edge that only
            # starts far ahead is not the current lane: the pair overlap
            # begins metres away and the planner would aim at a phantom
            # lane across the road (run 1786707239).  Keep the near edge
            # as a single-side mirror instead.
            if (ln < 2) != (rn < 2):
                far_c = l if ln < 2 else r
                if far_c.proj[0, 0] > LANE_ONE_NEAR_FAR_START_MAX_M:
                    continue
            s_lo = max(0.0, float(l.proj[0, 0]), float(r.proj[0, 0]))
            s_hi = min(float(l.proj[-1, 0]), float(r.proj[-1, 0]))
            if s_hi - s_lo < LANE_PAIR_OVERLAP_M:
                continue
            stations = _overlap_stations(s_lo, s_hi, station_step,
                                         max_stations)
            left_lat = _interp_lat(l.proj, stations)
            right_lat = _interp_lat(r.proj, stations)
            valid = ~(np.isnan(left_lat) | np.isnan(right_lat))
            if int(np.sum(valid)) < 3:
                continue
            width = float(np.median(left_lat[valid] - right_lat[valid]))
            center = float(np.median(
                0.5 * (left_lat[valid] + right_lat[valid])))
            if abs(center) > LANE_PAIR_CENTER_PREFER_M \
                    and width > LANE_OFF_CENTER_WIDTH_MAX_M:
                continue
            min_w = (LANE_PAIR_WIDTH_MIN_M if abs(center) <= 0.5
                     else LANE_WIDTH_MIN_M)
            if not (min_w <= width <= LANE_VISION_PAIR_WIDTH_MAX_M):
                continue
            if ln < 2 and rn < 2 \
                    and width < LANE_FAR_CENTER_PAIR_MIN_WIDTH_M:
                continue
            # A paired lane whose centre is far from the car usually means
            # the camera paired a near line with a roadside line outside
            # the current lane (run 57).  Such a frame drags the car
            # across the road, so refuse it and let the single-side mirror
            # handle the near boundary instead.
            if abs(center) > LANE_PAIR_CENTER_MAX_M:
                continue
            span = s_hi - s_lo
            avg_conf = 0.5 * (l.conf + r.conf)
            center_ok = math.exp(
                -0.5 * (center / LANE_PAIR_CENTER_PREFER_M) ** 2)
            near_ok = 1.0 / (1.0 + 0.35 * max(abs(l.med_lat),
                                              abs(r.med_lat)))
            near_ok *= 0.55 if min(ln, rn) < 2 else 1.0
            if ln < 2 and rn < 2:
                near_ok *= 0.75
            width_ok = 1.0 - 0.25 * min(1.0, abs(width - 3.5) / 1.3)
            span_ok = 0.5 + 0.5 * min(1.0, span / 14.0)
            score = avg_conf * span_ok * center_ok * near_ok * width_ok
            if score > best_score:
                best_score = score
                best = (l, r, stations, valid, width, center, span,
                        avg_conf, score)
    return best


def _best_single_boundary(cands: list[_LineCandidate],
                          side: int) -> _LineCandidate | None:
    """Best marking on one side for the mirror fallback.

    A far roadside line mirrored at one lane width would put the centre
    metres off the road, so the fallback only trusts lines close to the
    car and prefers the nearest one.  A long solid/dashed line that only
    becomes visible a few metres ahead is still accepted as a weak mirror
    so a real right paint can outrank the wall fallback.
    """
    best = None
    best_score = 0.0
    for c in cands:
        # A single-edge mirror has to be a real painted lane line.  A dark
        # pavement patch / tree shadow comes back as ``unknown`` and would
        # otherwise steer the car toward the roadside.
        if c.kind not in ("solid", "dashed", "thin"):
            continue
        if side > 0 and c.med_lat <= 0.08:
            continue
        if side < 0 and c.med_lat >= -0.08:
            continue
        # A mirror fallback assumes the line is the near edge of the lane.
        # A line whose median sits on top of the car (or on the far side
        # of it) is not the current lane edge, so it can never seed a
        # mirror centre.
        if abs(c.med_lat) < LANE_SINGLE_MED_MIN_M:
            continue
        if abs(c.med_lat) > LANE_SINGLE_NEAR_MAX_M:
            continue
        proj = c.proj
        near = proj[proj[:, 0] <= LANE_SINGLE_NEAR_REQUIRE_M]
        # A ``thin`` line is the skeleton of a wide blob (pavement patch,
        # repair scar).  Far from the car it cannot define the current
        # lane edge by itself; only a thin line beside the car may mirror.
        if len(near) < 2 and c.kind == "thin":
            continue
        if len(near) < 2:
            if c.kind not in ("solid", "dashed", "thin"):
                continue
            if c.proj[0, 0] > LANE_FAR_START_MAX_M:
                continue
        else:
            # A blob whose near end crosses to the other side of the car
            # (or sits in the middle) is exactly the case that mirrors to
            # a bogus centre, so reject it before it can seed the tracker.
            if side > 0 and (float(np.median(near[:, 1])) <= 0.08
                             or bool(np.any(near[:, 1] <= -0.2))):
                continue
            if side < 0 and (float(np.median(near[:, 1])) >= -0.08
                             or bool(np.any(near[:, 1] >= 0.2))):
                continue
        if side < 0 and not _yellow_right_ok(c):
            continue
        if len(near) >= 2:
            near_factor = 1.0 / (1.0 + 0.6 * abs(c.med_lat))
        else:
            start_penalty = 1.0 / (1.0 + 0.12 * c.proj[0, 0])
            near_factor = 0.35 * start_penalty
        score = c.score * (0.4 + 0.6 * near_factor)
        if score > best_score:
            best_score = score
            best = c
    return best


def _frame_from_stations(pos: np.ndarray, fwd: np.ndarray, stations,
                         center_lat: np.ndarray,
                         left_lat: np.ndarray | None = None,
                         right_lat: np.ndarray | None = None):
    left = np.array([-fwd[1], fwd[0]])
    s = stations
    center = pos + s[:, None] * fwd + center_lat[:, None] * left
    left_pts = right_pts = None
    if left_lat is not None:
        left_pts = pos + s[:, None] * fwd + left_lat[:, None] * left
    if right_lat is not None:
        right_pts = pos + s[:, None] * fwd + right_lat[:, None] * left
    return center, left_pts, right_pts


def _cap_mirror_conf(conf: float, stations: np.ndarray,
                     center_lat: np.ndarray, side: int = 0) -> float:
    """Downgrade a mirror whose centre sits far from the car.

    A single painted line two or more metres away is usually the next
    lane's edge or a roadside line; mirroring it puts the centre far from
    the current lane.  Such frames are kept for diagnostics but must not
    be trusted enough to steer the car.

    A right painted line is the primary boundary under right-hand
    traffic, so it keeps a low-trust virtual lane instead of being
    thrown away when it sits closer to the car.
    """
    near = center_lat[stations <= LANE_SINGLE_NEAR_REQUIRE_M]
    near = near[np.isfinite(near)]
    if not len(near):
        return conf
    center = abs(float(np.median(near)))
    if side < 0:
        if center > LANE_VISION_RIGHT_MIRROR_CENTER_MAX_M:
            return min(conf, LANE_VISION_RIGHT_MIRROR_CONF_MAX)
        return conf
    if center > LANE_VISION_MIRROR_CENTER_MAX_M:
        return min(conf, LANE_MIN_CONF - 0.01)
    return conf


def _single_mirror_frame(best, side: int, pos: np.ndarray,
                         fwd: np.ndarray, lane_width: float,
                         station_step: float, max_stations: int,
                         debug: dict | None) -> LaneFrame | None:
    """Mirror one painted edge at an assumed lane width.

    The camera saw only one usable boundary.  The other side is inferred
    at ``lane_width`` and the frame is deliberately a low/medium trust
    single-edge frame; it can still be upgraded later when a LiDAR edge
    on the opposite side arrives.
    """
    proj = best.proj
    span = best.span
    conf = best.conf
    stations = _overlap_stations(
        max(0.0, float(proj[0, 0])), float(proj[-1, 0]),
        station_step, max_stations)
    edge_lat = _interp_lat(proj, stations)
    valid = ~np.isnan(edge_lat)
    if int(np.sum(valid)) < 3:
        return None
    # A painted line this close on the right is being ridden, not the
    # lane's right edge.  Mirroring it puts the centre across the road
    # (typically across the centre line), so let the LiDAR / wall
    # fallback supply the small correction instead.
    if side < 0:
        near = proj[proj[:, 0] <= LANE_SINGLE_NEAR_REQUIRE_M]
        if len(near) >= 2 and abs(float(np.median(near[:, 1]))) \
                <= LANE_RIDING_LINE_MAX_M:
            return None
    if side > 0:
        center_lat = edge_lat - lane_width / 2.0
        left_lat, right_lat = edge_lat, None
        mode = "mirror_left"
    else:
        center_lat = edge_lat + lane_width / 2.0
        left_lat, right_lat = None, edge_lat
        mode = "mirror_right"
    center_lat = np.clip(center_lat, -LANE_SINGLE_MIRROR_MAX_M,
                         LANE_SINGLE_MIRROR_MAX_M)
    conf = (0.42 + 0.22 * min(1.0, span / 14.0)) \
        * (0.6 + 0.4 * min(1.0, conf))
    conf = min(0.82, conf)
    conf = _cap_mirror_conf(conf, stations, center_lat, side)
    # A line that only becomes visible several metres ahead is a weak
    # read: it can still beat a wall fallback, but the planner must only
    # give it a small nudge instead of a full-lane correction.
    if float(proj[0, 0]) > LANE_SINGLE_NEAR_REQUIRE_M:
        conf = min(conf, LANE_FAR_MIRROR_CONF_MAX)
    if debug is not None:
        debug["mode"] = mode
        debug["boundary_med"] = round(float(best.med_lat), 2)
        debug["center0"] = round(float(center_lat[0]), 2)
        debug["conf"] = round(float(conf), 2)
        debug["span"] = round(float(span), 1)
        debug["mirror_ok"] = conf >= LANE_MIN_CONF
    center, left_pts, right_pts = _frame_from_stations(
        pos, fwd, stations, center_lat, left_lat, right_lat)
    return LaneFrame(center=center, left=left_pts, right=right_pts,
                     width=lane_width, confidence=conf, span_m=span,
                     sources=("vision",), paired=False,
                     left_kind=(best.kind if side > 0 else None),
                     right_kind=(best.kind if side < 0 else None))


def pair_lane_markings(
    markings,
    pos,
    heading: float,
    fwd=None,
    lane_width: float = LANE_WIDTH_DEFAULT_M,
    min_span: float = LANE_MIN_SPAN_M,
    station_step: float = 1.5,
    max_stations: int = 18,
    debug: dict | None = None,
) -> LaneFrame | None:
    """Pair detected markings into a lane frame and return its centre.

    Markings are projected into a car-relative frame whose forward axis
    follows ``fwd`` (or the vehicle heading) and whose lateral axis is
    positive to the left of travel.  The best marking on each side is
    chosen, resampled onto common longitudinal stations, and the centre
    is the midpoint of the two boundaries.  When only one side is
    visible the other side is mirrored at ``lane_width``.
    """
    if not markings:
        return None
    pos = np.asarray(pos, dtype=float)[:2]
    fwd = _unit_fwd(pos, heading, fwd)
    cands = _collect_candidates(markings, pos, fwd, min_span)
    if not cands:
        if debug is not None:
            debug["mode"] = "none"
        return None
    axes = _axis_candidates(cands, station_step, max_stations)
    if debug is not None:
        def _cand_summary(c) -> dict:
            near = c.proj[c.proj[:, 0] <= 6.0]
            return {
                "med_lat": round(float(c.med_lat), 2),
                "span": round(float(c.span), 1),
                "conf": round(float(c.conf), 2),
                "kind": c.kind,
                "color": c.color,
                "score": round(float(c.score), 3),
                "start_s": round(float(c.proj[0, 0]), 1),
                "near_n": int(len(near)),
                "near_med": (round(float(np.median(near[:, 1])), 2)
                             if len(near) else None),
            }

        debug["cands"] = [_cand_summary(c) for c in cands]
        debug["axes"] = [_cand_summary(a) for a in axes]
    pair = _best_vision_pair(cands, axes, station_step, max_stations)
    if pair is not None:
        l, r, stations, valid, width, center, span, avg_conf, _ = pair
        left_lat = _interp_lat(l.proj, stations)
        right_lat = _interp_lat(r.proj, stations)
        center_lat = 0.5 * (left_lat + right_lat)
        # A paired frame whose centre sits far from the car, or whose
        # width covers a whole road, is a wrong pairing: a near right
        # line was matched with a far left line / roadside edge.  Drop
        # the pair and let the single-boundary fallback use the near
        # right line instead.
        near = center_lat[stations <= LANE_SINGLE_NEAR_REQUIRE_M]
        near = near[np.isfinite(near)]
        near_center = (float(np.median(near)) if len(near) else 0.0)
        if (abs(near_center) > LANE_PAIR_NEAR_CENTER_MAX_M
                or width > LANE_VISION_PAIR_WIDTH_MAX_M):
            pair = None
    if pair is not None:
        l, r, stations, valid, width, center, span, avg_conf, _ = pair
        left_lat = _interp_lat(l.proj, stations)
        right_lat = _interp_lat(r.proj, stations)
        center_lat = 0.5 * (left_lat + right_lat)
        conf = (0.55 + 0.25 * min(1.0, span / 14.0)) \
            * (0.6 + 0.4 * min(1.0, avg_conf))
        conf = min(0.95, conf)
        left_near_n = int(np.sum(l.proj[:, 0] <= LANE_SINGLE_NEAR_REQUIRE_M))
        right_near_n = int(np.sum(r.proj[:, 0] <= LANE_SINGLE_NEAR_REQUIRE_M))
        if left_near_n < 2 and right_near_n < 2:
            conf = min(conf, LANE_FAR_CENTER_PAIR_CONF_MAX)
        center_pts, left_pts, right_pts = _frame_from_stations(
            pos, fwd, stations, center_lat, left_lat, right_lat)
        if debug is not None:
            debug["mode"] = "pair"
            if left_near_n < 2 and right_near_n < 2:
                debug["far_center_pair"] = True
            debug["center0"] = round(float(center_lat[0]), 2)
            debug["width"] = round(float(width), 2)
            debug["conf"] = round(float(conf), 2)
            debug["span"] = round(float(span), 1)
            debug["left_med"] = round(float(l.med_lat), 2)
            debug["right_med"] = round(float(r.med_lat), 2)
        return LaneFrame(center=center_pts, left=left_pts, right=right_pts,
                         width=width, confidence=conf, span_m=span,
                         sources=("vision",), paired=True,
                         left_kind=l.kind, right_kind=r.kind)

    # No lane-like pair (one side missing, a whole road wide, or the
    # candidate edges are the same painted line): mirror one side.
    # Under right-hand traffic the painted right line is the stronger
    # boundary, so try it first; a short left blob must not shadow a
    # longer, trusted right line.
    left_best = _best_single_boundary(cands, 1)
    right_best = _best_single_boundary(cands, -1)
    if left_best is None and right_best is None:
        return None
    for best, side in ((right_best, -1), (left_best, 1)):
        if best is None:
            continue
        frame = _single_mirror_frame(
            best, side, pos, fwd, lane_width, station_step,
            max_stations, debug)
        if frame is not None:
            return frame
    return None


def build_lidar_corridor(
    hits,
    pos,
    heading: float,
    fwd=None,
    max_dist: float = LIDAR_MAX_DIST_M,
    station: float = LIDAR_STATION_M,
    max_lat: float = LIDAR_MAX_LAT_M,
    min_span: float = LANE_MIN_SPAN_M,
    debug: dict | None = None,
) -> LaneFrame | None:
    """Build a free-space corridor from raw raycast hits.

    At every longitudinal station ahead of the car the nearest hit on the
    left and on the right defines the corridor.  The corridor centre is
    the midpoint, so the car keeps to the middle of the drivable space
    when the camera cannot supply lane markings.  A side that only has
    hits on part of the span is interpolated across the missing stations,
    so a guardrail / wall seen intermittently still yields a real
    two-sided corridor instead of falling back to a rigid one-sided
    mirror.
    """
    if not hits:
        if debug is not None:
            debug["n_hits"] = 0
        return None
    pts = np.asarray(hits, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 2 or len(pts) < 8:
        if debug is not None:
            debug["n_hits"] = 0 if pts.ndim != 2 else int(len(pts))
        return None
    pos = np.asarray(pos, dtype=float)[:2]
    fwd = _unit_fwd(pos, heading, fwd)
    left = np.array([-fwd[1], fwd[0]])
    rel = pts[:, :2] - pos
    lon = rel @ fwd
    lat = rel @ left
    stations = np.arange(station, max_dist + 1e-9, station)
    left_lat = np.full(len(stations), np.nan)
    right_lat = np.full(len(stations), np.nan)
    for i, s in enumerate(stations):
        near = np.abs(lon - s) <= station * 0.55
        if not np.any(near):
            continue
        lats = lat[near]
        lats = lats[np.abs(lats) <= max_lat]
        if not len(lats):
            continue
        pos_l = lats[lats > 0.3]
        neg_l = lats[lats < -0.3]
        if len(pos_l):
            left_lat[i] = float(np.min(pos_l))
        if len(neg_l):
            right_lat[i] = float(np.max(neg_l))
    left_idx = np.where(~np.isnan(left_lat))[0]
    right_idx = np.where(~np.isnan(right_lat))[0]
    valid = ~(np.isnan(left_lat) | np.isnan(right_lat))
    n_direct = int(np.sum(valid))
    if debug is not None:
        debug["n_hits"] = int(len(pts))
        debug["n_direct"] = n_direct
        debug["left_n"] = int(len(left_idx))
        debug["right_n"] = int(len(right_idx))
        debug["max_lat"] = float(max_lat)
    # Both sides only need to be seen somewhere along the span: the
    # missing stations of one side are filled by interpolation so a
    # broken guardrail / sparse wall does not destroy the corridor.
    if len(left_idx) >= 3 and len(right_idx) >= 3:
        span_lo = max(int(left_idx[0]), int(right_idx[0]))
        span_hi = min(int(left_idx[-1]), int(right_idx[-1]))
        if span_hi - span_lo + 1 >= 3:
            mask = np.zeros(len(stations), dtype=bool)
            mask[span_lo:span_hi + 1] = True
            for arr in (left_lat, right_lat):
                idx = np.where(~np.isnan(arr))[0]
                if len(idx):
                    arr[mask] = np.interp(
                        stations[mask], stations[idx], arr[idx],
                        left=arr[idx[0]], right=arr[idx[-1]])
            valid = ~(np.isnan(left_lat) | np.isnan(right_lat))
            n_ok = int(np.sum(valid))
            span = float(stations[valid][-1] - stations[valid][0])
            if n_ok >= 3 and span >= min_span:
                center_lat = (left_lat + right_lat) / 2.0
                width = float(np.median(
                    left_lat[valid] - right_lat[valid]))
                width = min(8.0, max(2.0, width))
                valid_frac = n_ok / len(stations)
                direct_frac = n_direct / n_ok if n_ok else 0.0
                conf = 0.30 + 0.20 * valid_frac \
                    + 0.05 * min(1.0, width / 3.5)
                conf *= 0.75 + 0.25 * direct_frac
                conf = min(0.65, conf)
                if debug is not None:
                    debug["n_ok"] = n_ok
                    debug["span"] = round(float(span), 1)
                    debug["width"] = round(float(width), 2)
                    debug["conf"] = round(float(conf), 2)
                center, left_pts, right_pts = _frame_from_stations(
                    pos, fwd, stations, center_lat, left_lat, right_lat)
                center = center[valid]
                left_pts = None if left_pts is None else left_pts[valid]
                right_pts = None if right_pts is None else right_pts[valid]
                return LaneFrame(center=center, left=left_pts,
                                 right=right_pts, width=width,
                                 confidence=conf, span_m=span,
                                 sources=("lidar",), paired=True)
    fallback = _single_edge_lidar_frame(
        pts, pos, fwd, stations, station, max_lat=LIDAR_MAX_LAT_M,
        debug=debug)
    if fallback is not None:
        return fallback
    if debug is not None:
        debug["n_ok"] = int(np.sum(valid))
    return None


def _single_edge_lidar_frame(pts, pos, fwd, stations, station, max_lat,
                             debug: dict | None) -> LaneFrame | None:
    """One-sided lidar fallback, right-hand traffic first.

    The full two-sided corridor needs hits on both sides at every station,
    which an open road with sparse clutter rarely provides.  In right-hand
    traffic the right-side hit (guardrail / curb / wall) is the boundary a
    single lane must stay clear of, so the fallback mirrors one assumed
    lane width from that edge.  The right side is preferred; the left edge
    is only used when no right-side edge exists.  It is deliberately
    low-confidence so the planner only nudges, never treats the laser edge
    as a full lane map.
    """
    if debug is not None:
        debug.setdefault("fallback", "none")
    left = np.array([-fwd[1], fwd[0]])
    rel = pts[:, :2] - pos
    lon = rel @ fwd
    lat = rel @ left
    candidates = (
        (-1, "right"),
        (1, "left"),
    )
    for side, name in candidates:
        edge_lat = np.full(len(stations), np.nan)
        for i, s in enumerate(stations):
            near = np.abs(lon - s) <= station * 0.55
            if not np.any(near):
                continue
            lats = lat[near]
            if side > 0:
                lats = lats[(lats >= LIDAR_EDGE_MIN_M)
                            & (lats <= max_lat)]
                if len(lats):
                    edge_lat[i] = float(np.min(lats))
            else:
                lats = lats[(lats <= -LIDAR_EDGE_MIN_M)
                            & (lats >= -max_lat)]
                if len(lats):
                    edge_lat[i] = float(np.max(lats))
        valid = ~np.isnan(edge_lat)
        n_ok = int(np.sum(valid))
        if n_ok < 3:
            continue
        span = float(stations[valid][-1] - stations[valid][0])
        if span < LANE_MIN_SPAN_M:
            continue
        center_lat = edge_lat - side * LANE_WIDTH_DEFAULT_M / 2.0
        # A one-sided raycast edge is the drivable boundary, not proof
        # that the car is a full lane width away from it.  Mirroring the
        # full 3.5 m lane from a far guardrail pushed the centre to the
        # wrong side of the car (run 56-58), so the fallback only offers
        # a small centring hint and lets the nav route carry topology.
        center_lat = np.clip(center_lat, -LANE_SINGLE_LIDAR_CENTER_MAX_M,
                             LANE_SINGLE_LIDAR_CENTER_MAX_M)
        valid_frac = n_ok / len(stations)
        conf = 0.40 + 0.12 * valid_frac + 0.05 * min(1.0, span / 14.0)
        conf = min(0.62, conf)
        center, _, _ = _frame_from_stations(
            pos, fwd, stations, center_lat, None, None)
        if side > 0:
            _, edge_pts, _ = _frame_from_stations(
                pos, fwd, stations, center_lat, edge_lat, None)
        else:
            _, _, edge_pts = _frame_from_stations(
                pos, fwd, stations, center_lat, None, edge_lat)
        if debug is not None:
            debug["fallback"] = name
            debug["edge_n"] = n_ok
            debug["edge_span"] = round(float(span), 1)
            debug["edge_med"] = round(float(np.nanmedian(edge_lat)), 2)
            debug["fallback_conf"] = round(float(conf), 2)
        if side > 0:
            return LaneFrame(center=center[valid], left=edge_pts[valid],
                             right=None, width=LANE_WIDTH_DEFAULT_M,
                             confidence=conf, span_m=span,
                             sources=("lidar", "left"), paired=False)
        return LaneFrame(center=center[valid], left=None,
                         right=edge_pts[valid],
                         width=LANE_WIDTH_DEFAULT_M, confidence=conf,
                         span_m=span, sources=("lidar", "right"), paired=False)
    return None


def _to_car_frame(world: np.ndarray, pos: np.ndarray,
                  fwd: np.ndarray) -> np.ndarray:
    left = np.array([-fwd[1], fwd[0]])
    rel = world[:, :2] - pos
    return np.column_stack([rel @ fwd, rel @ left])


def _from_car_frame(rel: np.ndarray, pos: np.ndarray,
                    fwd: np.ndarray) -> np.ndarray:
    left = np.array([-fwd[1], fwd[0]])
    return pos + rel[:, 0][:, None] * fwd + rel[:, 1][:, None] * left


def _resample_rel(rel_pts: np.ndarray, stations: np.ndarray) -> np.ndarray:
    if rel_pts is None or len(rel_pts) < 2:
        return np.full(stations.shape, np.nan)
    s = rel_pts[:, 0]
    lat = rel_pts[:, 1]
    order = np.argsort(s)
    return np.interp(stations, s[order], lat[order],
                     left=np.nan, right=np.nan)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted median of a 1D sample, ignoring NaN values."""
    ok = ~np.isnan(np.asarray(values, dtype=float))
    if not np.any(ok):
        return float("nan")
    v = np.asarray(values, dtype=float)[ok]
    w = np.asarray(weights, dtype=float)[ok]
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    total = float(np.sum(w))
    if total <= 0.0:
        return float(np.median(v))
    half = 0.5 * total
    acc = 0.0
    for val, wt in zip(v, w):
        acc += wt
        if acc >= half:
            return float(val)
    return float(v[-1])


def _median_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Median absolute lateral difference where both samples are valid."""
    ok = ~(np.isnan(a) | np.isnan(b))
    if not np.any(ok):
        return float("inf")
    return float(np.median(np.abs(a[ok] - b[ok])))


class LaneTracker:
    """Time-smooths lane frames in the car-relative frame.

    Raw CV lane centres are noisy frame to frame.  Each incoming frame is
    transformed into the current car frame, resampled onto fixed
    longitudinal stations, and the confidence-weighted median lateral
    offset across the last few frames is used as the smoothed centre, so
    a shaky low-confidence frame cannot drag the lane away.
    """

    def __init__(self, window: int = 4,
                 station_step: float = TRACK_STATION_M,
                 max_dist: float = TRACK_MAX_DIST_M):
        self.window = max(1, int(window))
        self.station_step = float(station_step)
        self.max_dist = float(max_dist)
        self.history: list[tuple[np.ndarray, float, float, float,
                                 bool, np.ndarray | None,
                                 np.ndarray | None]] = []
        self.frame: LaneFrame | None = None
        self.last_rejected = False
        self._reject_world: np.ndarray | None = None
        self._reject_t = 0.0
        self._reject_count = 0
        self._last_ok_t = 0.0
        self._last_ok_pos: np.ndarray | None = None
        self._last_paired_t = 0.0
        self._last_paired_pos: np.ndarray | None = None

    def clear(self) -> None:
        self.history.clear()
        self.frame = None
        self.last_rejected = False
        self._reject_world = None
        self._reject_t = 0.0
        self._reject_count = 0
        self._last_ok_t = 0.0
        self._last_ok_pos = None
        self._last_paired_t = 0.0
        self._last_paired_pos = None

    def update(self, frame: LaneFrame | None, pos, heading: float,
               fwd=None) -> LaneFrame | None:
        now = time.time()
        pos = np.asarray(pos, dtype=float)[:2]
        fwd = _unit_fwd(pos, heading, fwd)
        if frame is None or len(frame.center) < 2:
            self.history.clear()
            self.frame = None
            self.last_rejected = False
            self._reject_world = None
            self._reject_t = 0.0
            self._reject_count = 0
            return None
        # A long gap of rejected frames means the last accepted centre is
        # no longer a useful reference; let the next frame seed fresh.
        if self._last_ok_pos is not None:
            travel = float(np.linalg.norm(pos - self._last_ok_pos))
            if (now - self._last_ok_t > TRACK_STALE_S
                    or travel > TRACK_STALE_M):
                self.history.clear()
                self.frame = None
                self._reject_world = None
                self._reject_t = 0.0
                self._reject_count = 0
                self._last_paired_t = 0.0
                self._last_paired_pos = None
        rel = _to_car_frame(np.asarray(frame.center, dtype=float),
                            pos, fwd)
        stations = np.arange(0.0, self.max_dist + 1e-9,
                             self.station_step)
        new_lat = _resample_rel(rel, stations)
        # Once a two-sided lane has been seen, a one-sided mirror is not
        # allowed to drag the smoothed centre sideways until the paired
        # reference itself goes stale.
        if (not frame.paired and self._last_paired_t > 0.0
                and now - self._last_paired_t <= TRACK_STALE_S):
            if self._last_paired_pos is None:
                return None
            travel = float(np.linalg.norm(pos - self._last_paired_pos))
            if travel <= TRACK_STALE_M:
                return None
        rejected = False
        # The same wrong centre keeps coming back for a while: keep
        # rejecting it instead of letting it re-seed the tracker.
        if (self._reject_world is not None
                and now - self._reject_t <= TRACK_REJECT_WINDOW_S):
            rej_lat = _resample_rel(
                _to_car_frame(self._reject_world, pos, fwd), stations)
            if _median_abs_diff(new_lat, rej_lat) <= TRACK_REJECT_MATCH_M:
                rejected = True
        # A single frame that jumps far from the smoothed centre is a
        # mirror fallback on the wrong edge, not a real lane change.
        if not rejected and self.frame is not None:
            cur_lat = _resample_rel(
                _to_car_frame(np.asarray(self.frame.center, dtype=float),
                              pos, fwd), stations)
            if _median_abs_diff(new_lat, cur_lat) > TRACK_JUMP_MAX_M:
                rejected = True
        if rejected:
            self.last_rejected = True
            self._reject_world = np.asarray(
                frame.center, dtype=float).copy()
            self._reject_t = now
            self._reject_count += 1
            if self._reject_count >= 3:
                self.history.clear()
                self.frame = None
            return None
        self.last_rejected = False
        self._reject_world = None
        self._reject_t = 0.0
        self._reject_count = 0
        self._last_ok_t = now
        self._last_ok_pos = pos.copy()
        if frame.paired:
            # The first paired frame after a mirror stretch replaces the
            # mirror history instead of being averaged into it.
            if self.history and not any(h[4] for h in self.history):
                self.history.clear()
                self.frame = None
            self._last_paired_t = now
            self._last_paired_pos = pos.copy()
        left_rel = right_rel = None
        if frame.left is not None:
            left_rel = _to_car_frame(
                np.asarray(frame.left, dtype=float)[:, :2], pos, fwd)
        if frame.right is not None:
            right_rel = _to_car_frame(
                np.asarray(frame.right, dtype=float)[:, :2], pos, fwd)
        self.history.append((rel, float(frame.width),
                             float(frame.confidence), float(frame.span_m),
                             bool(frame.paired), left_rel, right_rel))
        if len(self.history) > self.window:
            self.history.pop(0)
        lat_mat = np.full((len(self.history), len(stations)), np.nan)
        widths: list[float] = []
        confs: list[float] = []
        spans: list[float] = []
        left_mat = right_mat = None
        for k, (rel_pts, w, c, span, _, lr, rr) in enumerate(self.history):
            lat_mat[k] = _resample_rel(rel_pts, stations)
            if lr is not None:
                if left_mat is None:
                    left_mat = np.full((len(self.history),
                                        len(stations)), np.nan)
                left_mat[k] = _resample_rel(lr, stations)
            if rr is not None:
                if right_mat is None:
                    right_mat = np.full((len(self.history),
                                         len(stations)), np.nan)
                right_mat[k] = _resample_rel(rr, stations)
            widths.append(w)
            confs.append(c)
            spans.append(span)
        weights = np.maximum(np.asarray(confs, dtype=float), 1e-3)
        med_lat = np.array([_weighted_median(lat_mat[:, j], weights)
                            for j in range(len(stations))])
        left_lat = right_lat = None
        if left_mat is not None:
            left_lat = np.array([_weighted_median(left_mat[:, j], weights)
                                 for j in range(len(stations))])
        if right_mat is not None:
            right_lat = np.array([_weighted_median(right_mat[:, j], weights)
                                  for j in range(len(stations))])
        valid = ~np.isnan(med_lat)
        if left_lat is not None:
            valid &= ~np.isnan(left_lat)
        if right_lat is not None:
            valid &= ~np.isnan(right_lat)
        if int(np.sum(valid)) < 3:
            self.history.clear()
            self.frame = None
            return None
        s_pts = stations[valid]
        lat_pts = med_lat[valid]
        center = _from_car_frame(np.column_stack([s_pts, lat_pts]),
                                 pos, fwd)
        left_pts = right_pts = None
        if left_lat is not None:
            left_pts = _from_car_frame(
                np.column_stack([s_pts, left_lat[valid]]), pos, fwd)
        if right_lat is not None:
            right_pts = _from_car_frame(
                np.column_stack([s_pts, right_lat[valid]]), pos, fwd)
        width = float(np.nanmedian(widths) if widths
                      else LANE_WIDTH_DEFAULT_M)
        conf = float(np.nanmedian(confs) if confs else 0.0)
        span = float(max(spans)) if spans else float(s_pts[-1] - s_pts[0])
        self.frame = LaneFrame(center=center, width=width,
                               left=left_pts, right=right_pts,
                               confidence=conf,
                               span_m=float(s_pts[-1] - s_pts[0]),
                               sources=frame.sources or ("vision",),
                               paired=frame.paired,
                               left_kind=frame.left_kind,
                               right_kind=frame.right_kind)
        return self.frame


def lane_frame_usable(frame: LaneFrame | None,
                      min_conf: float = LANE_MIN_CONF) -> bool:
    """A sensor lane only counts as a single drivable lane when it is
    long, confident and roughly one lane wide.  A wider corridor is the
    whole road (or road + verge), so centring it would not keep the car
    in one lane under right-hand traffic."""
    if frame is None:
        return False
    min_w = (LANE_PAIR_WIDTH_MIN_M if frame.paired
             and "vision" in frame.sources else 2.0)
    max_w = (LANE_FUSION_WIDTH_MAX_M
             if frame.paired and len(frame.sources) > 1
             else LANE_WIDTH_MAX_M)
    min_span = (LANE_PAIRED_VISION_MIN_SPAN_M
                if frame.paired and (frame.sources == ("vision",)
                                     or len(frame.sources) > 1)
                else LANE_FRAME_MIN_SPAN_M)
    return (frame.confidence >= min_conf
            and frame.span_m >= min_span
            and min_w <= frame.width <= max_w)


def _frame_near_lat(frame: LaneFrame, pos, heading: float,
                    fwd=None) -> float | None:
    """Median lateral offset of a frame's centre close to the car."""
    center = np.asarray(frame.center, dtype=float)
    if center.ndim != 2 or len(center) < 2:
        return None
    pos = np.asarray(pos, dtype=float)[:2]
    fwd = _unit_fwd(pos, heading, fwd)
    rel = _to_car_frame(center, pos, fwd)
    lat = rel[rel[:, 0] <= LANE_SINGLE_NEAR_REQUIRE_M, 1]
    if len(lat) < 2:
        return None
    return float(np.median(lat))


def _fusion_center_unstable(frame: LaneFrame | None, pos, heading: float,
                            fwd=None) -> bool:
    """A fused lane whose centre is far from the car is a bad pair."""
    if (frame is None or not frame.paired
            or len(frame.sources) < 2 or pos is None):
        return False
    lat = _frame_near_lat(frame, pos, heading, fwd)
    return lat is not None and abs(lat) > LANE_PAIR_NEAR_CENTER_MAX_M


def _boundary_near_lat(world, pos, heading: float,
                       fwd=None) -> float | None:
    """Median lateral offset of a boundary polyline close to the car."""
    pts = np.asarray(world, dtype=float)
    if pts.ndim != 2 or len(pts) < 2:
        return None
    pos = np.asarray(pos, dtype=float)[:2]
    fwd = _unit_fwd(pos, heading, fwd)
    rel = _to_car_frame(pts, pos, fwd)
    lat = rel[rel[:, 0] <= LANE_SINGLE_NEAR_REQUIRE_M, 1]
    if len(lat) < 2:
        return None
    return float(np.median(lat))


def _mirror_near_ok(frame: LaneFrame, pos, heading: float,
                    fwd=None) -> bool:
    """A trusted mirror must centre close to the car, not on a far line.

    A painted line that only appears several metres ahead cannot define
    the current lane by itself: it is kept as a low-trust diagnostics
    frame, but an unpaired mirror must not steer the car before its
    centre is actually visible next to the vehicle.  Such a frame can
    still pair with an opposite-side LiDAR edge, which supplies the
    near boundary the single line is missing.
    """
    if pos is None:
        return True
    lat = _frame_near_lat(frame, pos, heading, fwd)
    if lat is None:
        return False
    max_center = LANE_VISION_MIRROR_CENTER_MAX_M
    if frame.left is None and frame.right is not None:
        max_center = LANE_VISION_RIGHT_MIRROR_CENTER_MAX_M
    return abs(lat) <= max_center


def _mirror_right_ok(frame: LaneFrame, pos, heading: float,
                     fwd=None) -> bool:
    """Trust a right-side mirror when the paint is actually beside the car.

    A painted right line is the strongest boundary once it is clearly to
    the right of the car.  A line that close to the car is usually the
    centre line being ridden, so mirroring it would drag the car across
    the road.  A line that only appears several metres ahead is not the
    current lane edge on its own (run 188), so it is kept for fusion but
    cannot steer an unpaired mirror.
    """
    if frame.right is None:
        return True
    if pos is None:
        return True
    pts = np.asarray(frame.right, dtype=float)
    pos = np.asarray(pos, dtype=float)[:2]
    fwd = _unit_fwd(pos, heading, fwd)
    rel = _to_car_frame(pts, pos, fwd)
    near = rel[rel[:, 0] <= LANE_RIGHT_MIRROR_NEAR_M]
    if len(near) < 2:
        return False
    lat = float(np.median(near[:, 1]))
    return lat <= -LANE_RIDING_LINE_MAX_M


def _vision_lidar_agree(vision_frame: LaneFrame, lidar_frame: LaneFrame,
                        pos, heading: float, fwd=None) -> bool:
    v = _frame_near_lat(vision_frame, pos, heading, fwd)
    l = _frame_near_lat(lidar_frame, pos, heading, fwd)
    return v is not None and l is not None \
        and abs(v - l) <= LANE_FUSION_AGREE_MAX_M


def _active_lidar_reference(state: dict | None,
                            lidar_frame: LaneFrame | None) -> bool:
    """True when the fusion state is currently holding a LiDAR read."""
    if state is None or lidar_frame is None:
        return False
    src = state.get("src")
    return isinstance(src, tuple) and len(src) > 0 \
        and str(src[0]) == "lidar"


def _vision_mirror_keeps_corridor(vision_frame: LaneFrame,
                                  lidar_frame: LaneFrame,
                                  pos, heading: float,
                                  fwd=None) -> bool:
    """A single-edge vision mirror must agree with an active LiDAR
    corridor before it may replace that corridor.

    A mirror assumes the lane width from one painted line.  When its
    inferred centre disagrees with the physical free-space corridor by
    more than a small tolerance, the read is usually a wrong line (the
    other lane's edge / a roadside line) and replacing the corridor with
    it makes the car jump sideways.  The LiDAR corridor stays primary
    until the camera mirror actually matches it.
    """
    if pos is None or lidar_frame is None:
        return True
    v = _frame_near_lat(vision_frame, pos, heading, fwd)
    l = _frame_near_lat(lidar_frame, pos, heading, fwd)
    return v is None or l is None \
        or abs(v - l) <= LANE_FUSION_CENTER_CONT_MAX_M


def _vision_mirror_keeps_reference(vision_frame: LaneFrame,
                                   lidar_frame: LaneFrame | None,
                                   last_frame: LaneFrame | None,
                                   pos, heading: float,
                                   fwd=None) -> bool:
    """A single-edge vision mirror must agree with the active lane read.

    The reference is the last frame the fusion state actually held (a
    LiDAR corridor survives its own short detection gaps) and, when that
    is unavailable, the current LiDAR frame.  A mirror whose inferred
    centre disagrees by more than a small tolerance must not replace
    that corridor, otherwise one wrong painted line yanks the car
    sideways across the road.
    """
    if pos is None:
        return True
    if last_frame is not None:
        last_lat = _frame_near_lat(last_frame, pos, heading, fwd)
        if last_lat is not None:
            v = _frame_near_lat(vision_frame, pos, heading, fwd)
            return v is None \
                or abs(v - last_lat) <= LANE_FUSION_CENTER_CONT_MAX_M
    return _vision_mirror_keeps_corridor(
        vision_frame, lidar_frame, pos, heading, fwd)


def _vision_edge_inside_lidar(vision_frame: LaneFrame | None,
                              lidar_frame: LaneFrame | None,
                              pos, heading: float,
                              fwd=None) -> bool:
    """True when a vision edge stays inside the same-side LiDAR wall.

    A painted line on the right must lie inside (on the road side of) the
    LiDAR right wall / guardrail; a line outside the wall is a far-road
    marking or a projection artefact and must not override the physical
    corridor.  The same applies to the left side.
    """
    if vision_frame is None or lidar_frame is None or pos is None:
        return True
    pos = np.asarray(pos, dtype=float)[:2]
    fwd = _unit_fwd(pos, heading, fwd)
    if vision_frame.right is not None and lidar_frame.right is not None:
        v = _boundary_near_lat(vision_frame.right, pos, heading, fwd)
        l = _boundary_near_lat(lidar_frame.right, pos, heading, fwd)
        if v is not None and l is not None \
                and v <= l + FUSION_WALL_SAFE_MARGIN_M:
            return False
    if vision_frame.left is not None and lidar_frame.left is not None:
        v = _boundary_near_lat(vision_frame.left, pos, heading, fwd)
        l = _boundary_near_lat(lidar_frame.left, pos, heading, fwd)
        if v is not None and l is not None \
                and v >= l - FUSION_WALL_SAFE_MARGIN_M:
            return False
    return True


def _pair_vision_lidar_edges(vision_frame: LaneFrame,
                             lidar_frame: LaneFrame,
                             pos, heading: float,
                             fwd=None) -> LaneFrame | None:
    """Build a two-sided lane from a painted line + a LiDAR edge.

    Under right-hand traffic the lane the car belongs to is bounded on
    the left by the centre line and on the right by the road edge / curb.
    When the camera sees only one painted line and the raycast fan sees a
    right-side boundary beyond it, pairing the two gives a real lane
    width and centre without assuming the line is a lane edge.  A painted
    line is classified by the side it actually sits on, so it only pairs
    with a LiDAR edge on the opposite side of the car.  A two-sided LiDAR
    corridor is accepted too: the side opposite the painted line is used
    as the second boundary, which lets a road-wide corridor pair with a
    detected centre line instead of being discarded.
    """
    if (vision_frame is None or lidar_frame is None
            or vision_frame.paired):
        return None
    pos = np.asarray(pos, dtype=float)[:2]
    fwd = _unit_fwd(pos, heading, fwd)
    vision_pts = vision_frame.left
    if vision_pts is None:
        vision_pts = vision_frame.right
    if vision_pts is None:
        return None
    vision_rel = _to_car_frame(np.asarray(vision_pts, dtype=float),
                               pos, fwd)
    if len(vision_rel) < 2:
        return None
    vision_near = vision_rel[vision_rel[:, 0] <= LANE_SINGLE_NEAR_REQUIRE_M]
    vision_kind = (vision_frame.left_kind
                   if vision_frame.left is not None
                   else vision_frame.right_kind)
    # A far ``thin`` line is usually the skeleton of a dark pavement
    # patch, not the current lane edge.  Pairing it with an opposite
    # LiDAR wall builds a phantom lane across the road.
    if len(vision_near) < 2 and vision_kind == "thin":
        return None
    vision_sample = vision_near if len(vision_near) >= 2 else vision_rel
    vision_lat = float(np.median(vision_sample[:, 1]))
    if not math.isfinite(vision_lat) or abs(vision_lat) <= 0.15:
        return None
    vision_side = 1 if vision_lat > 0.0 else -1
    # A right paint under the car is usually the centre line being
    # ridden, not the lane's right edge.  Pairing it with a left wall
    # would build a lane on the wrong side of the road.
    if vision_side < 0 and abs(vision_lat) <= LANE_RIDING_LINE_MAX_M:
        return None
    # Pick the LiDAR boundary on the opposite side of the painted line.
    if vision_side > 0:
        lidar_pts = lidar_frame.right
        if lidar_pts is None:
            lidar_pts = lidar_frame.left
    else:
        lidar_pts = lidar_frame.left
        if lidar_pts is None:
            lidar_pts = lidar_frame.right
    if lidar_pts is None:
        return None
    lidar_rel = _to_car_frame(np.asarray(lidar_pts, dtype=float),
                              pos, fwd)
    if len(lidar_rel) < 2:
        return None
    lidar_near = lidar_rel[lidar_rel[:, 0] <= LANE_SINGLE_NEAR_REQUIRE_M]
    lidar_sample = lidar_near if len(lidar_near) >= 2 else lidar_rel
    lidar_lat = float(np.median(lidar_sample[:, 1]))
    if not math.isfinite(lidar_lat):
        return None
    if not _vision_edge_inside_lidar(
            vision_frame, lidar_frame, pos, heading, fwd):
        return None
    if vision_side > 0 and lidar_lat >= -0.15:
        return None
    if vision_side < 0 and lidar_lat <= 0.15:
        return None
    if vision_side > 0:
        left_rel, right_rel = vision_rel, lidar_rel
    else:
        left_rel, right_rel = lidar_rel, vision_rel
    # Sample only the longitudinal overlap of the two real boundaries.
    # A right paint that starts several metres ahead still pairs with a
    # wall seen from the car, instead of being discarded for lacking a
    # near point.
    s_lo = max(0.0, float(np.min(left_rel[:, 0])),
               float(np.min(right_rel[:, 0])))
    s_hi = min(float(np.max(left_rel[:, 0])),
               float(np.max(right_rel[:, 0])))
    if s_hi - s_lo < LANE_PAIR_OVERLAP_M:
        return None
    stations = _overlap_stations(s_lo, s_hi, TRACK_STATION_M, 30)
    left_lat = _resample_rel(left_rel, stations)
    right_lat = _resample_rel(right_rel, stations)
    valid = ~(np.isnan(left_lat) | np.isnan(right_lat))
    if int(np.sum(valid)) < 3:
        return None
    width = float(np.median(left_lat[valid] - right_lat[valid]))
    if not (LANE_WIDTH_MIN_M <= width <= LANE_FUSION_WIDTH_MAX_M):
        return None
    span = float(stations[valid][-1] - stations[valid][0])
    # A short real overlap (a right paint that starts a few metres ahead
    # plus a wall seen from the car) is enough to build the two-sided lane;
    # refusing it sends the car back to the low-trust wall fallback.
    if span < LANE_PAIRED_VISION_MIN_SPAN_M:
        return None
    # The painted line is one real boundary and the opposite LiDAR edge
    # is the other: the lane centre is their midpoint.  A bogus pair
    # (near paint + far wall) is rejected below when that midpoint lies
    # too far from the car instead of being pinned to an assumed lane
    # width, which kept the car on the centre line (user case).
    center_lat = np.where(
        valid, 0.5 * (left_lat + right_lat), np.nan)
    # The painted line is the primary boundary, but a real opposite-side
    # wall / guardrail still defines the other edge: keep the fused
    # centre only when it stays close to the car, so a far wall cannot
    # drag the lane across the road.  A pair whose overlap only starts
    # ahead of the car is kept at low trust instead of being dropped.
    near_center = center_lat[np.isfinite(center_lat)
                             & (stations <= LANE_SINGLE_NEAR_REQUIRE_M)]
    if len(near_center):
        # A fused lane whose centre is far from the car is usually a
        # near paint paired with a wall outside the current lane (e.g. a
        # right line 0.2 m away plus a left wall 4.6 m away).  Reject it
        # even when the overlap only has two near stations: the old
        # three-point minimum let exactly that bogus pair through.
        if abs(float(np.median(near_center))) > LANE_PAIR_NEAR_CENTER_MAX_M:
            return None
    elif s_lo > LANE_FAR_START_MAX_M:
        return None
    center, lpts, rpts = _frame_from_stations(
        pos, fwd, stations, center_lat, left_lat, right_lat)
    conf = min(LANE_FUSION_PAIRED_CONF_MAX,
               0.55 + 0.15 * min(1.0, span / 14.0)
               + 0.08 * min(1.0, width / 3.5))
    if len(near_center) < 3:
        conf = min(conf, LANE_FAR_MIRROR_CONF_MAX)
    return LaneFrame(center=center[valid], left=lpts[valid],
                     right=rpts[valid], width=width, confidence=conf,
                     span_m=span, sources=("vision", "lidar"), paired=True)


def choose_sensor_lane(vision_frame: LaneFrame | None,
                       lidar_frame: LaneFrame | None,
                       pos=None, heading: float = 0.0,
                       fwd=None,
                       state: dict | None = None) -> LaneFrame | None:
    """Fuse vision and LiDAR lanes, strongest evidence first.

    A two-sided vision pair is the best lane read.  Next, a vision +
    LiDAR fusion that pairs a painted line with an opposite-side wall /
    guardrail gives a real two-sided lane and wins over either source
    alone.  Then comes a trusted painted right edge: under right-hand
    traffic the right road marking is the boundary the car should follow
    first, so it beats a two-sided LiDAR corridor (a physical
    wall/guardrail corridor is only the fallback when no right marking
    is available).  A LiDAR single-edge fallback is the last resort and
    is deliberately kept as a small centring hint.

    ``state`` is an optional mutable dict used to hold the current fusion
    source across frames.  The key is the full sources tuple, so a LiDAR
    right-edge flip to the left edge is a new source.  A new source has
    to survive ``LANE_FUSION_HOLD_FRAMES`` consecutive frames before it
    may replace the active one, and a short detection gap keeps the last
    lane instead of dropping straight to None.
    """
    vision_ok = lane_frame_usable(vision_frame, LANE_MIN_CONF)
    lidar_ok = lane_frame_usable(lidar_frame, 0.35)
    fused = None
    if (vision_frame is not None and lidar_frame is not None
            and pos is not None and not vision_frame.paired):
        fused = _pair_vision_lidar_edges(
            vision_frame, lidar_frame, pos, heading, fwd)
    if vision_ok and vision_frame.paired:
        chosen = vision_frame
    elif fused is not None:
        chosen = fused
    elif (vision_ok
            and (vision_frame.left is not None
                 or vision_frame.right is not None)
            and _mirror_near_ok(vision_frame, pos, heading, fwd)
            and _vision_mirror_keeps_reference(
                vision_frame, lidar_frame,
                state.get("last") if state is not None else None,
                pos, heading, fwd)
            and (vision_frame.right is None
                 or (_mirror_right_ok(vision_frame, pos, heading, fwd)
                     and _vision_edge_inside_lidar(
                         vision_frame, lidar_frame, pos, heading, fwd)))):
        # As long as a painted lane line is visible it is the primary
        # boundary.  LiDAR only fills the missing opposite side; it must
        # never override a real marking just because the laser corridor
        # is also usable.
        chosen = vision_frame
    elif lidar_ok:
        chosen = lidar_frame
    else:
        chosen = None
    if state is None:
        return chosen
    src = tuple(chosen.sources) if chosen is not None else None
    if src is None:
        misses = int(state.get("misses", 0)) + 1
        if misses > LANE_FUSION_HOLD_NONE_FRAMES:
            state.clear()
            return None
        state["misses"] = misses
        return state.get("last")
    state["misses"] = 0
    if state.get("src") == src:
        state["frames"] = int(state.get("frames", 0)) + 1
        state["last"] = chosen
        return chosen
    if state.get("src") is None:
        state["src"] = src
        state["frames"] = 1
        state["last"] = chosen
        return chosen
    # A two-sided lane (real midpoint) is the best read; a one-sided
    # fallback must not flicker it away.  Unpaired alternatives need a
    # longer consecutive hold before they may replace an active paired
    # lane, while another paired read (e.g. a new fusion source) is
    # allowed to take over with the normal hold.
    active_src = state.get("src")
    active_paired = bool(state.get("last")) and getattr(
        state["last"], "paired", False)
    if (active_paired and not chosen.paired
            and _fusion_center_unstable(
                state.get("last"), pos, heading, fwd)):
        # The held "paired" lane is itself an unphysical fusion (centre
        # metres away from the car).  Do not let it survive the hold
        # window while the sensor read has already recovered.
        state["src"] = src
        state["frames"] = 1
        state["last"] = chosen
        return chosen
    if not chosen.paired and active_paired:
        hold = LANE_FUSION_PAIRED_HOLD_FRAMES
    else:
        hold = LANE_FUSION_HOLD_FRAMES
    if int(state.get("frames", 0)) >= hold:
        state["src"] = src
        state["frames"] = 1
        state["last"] = chosen
        return chosen
    state["frames"] = int(state.get("frames", 0)) + 1
    return state.get("last")
