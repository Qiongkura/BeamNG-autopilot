"""Vision lane marking pairing: markings -> lane centre."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .constants import (
    AXIS_MERGE_MAX_M,
    LANE_BOUNDARY_SPAN_M,
    LANE_EDGE_MAX_M,
    LANE_FAR_CENTER_PAIR_CONF_MAX,
    LANE_FAR_CENTER_PAIR_MIN_WIDTH_M,
    LANE_FAR_MIRROR_CONF_MAX,
    LANE_FAR_START_MAX_M,
    LANE_MIN_CONF,
    LANE_MIN_SPAN_M,
    LANE_OFF_CENTER_WIDTH_MAX_M,
    LANE_ONE_NEAR_FAR_START_MAX_M,
    LANE_PAIR_CENTER_MAX_M,
    LANE_PAIR_CENTER_PREFER_M,
    LANE_PAIR_NEAR_CENTER_MAX_M,
    LANE_PAIR_NEAR_MAX_M,
    LANE_PAIR_OVERLAP_M,
    LANE_PAIR_WIDTH_MIN_M,
    LANE_RIDING_LINE_MAX_M,
    LANE_SINGLE_MIRROR_MAX_M,
    LANE_SINGLE_MED_MIN_M,
    LANE_SINGLE_NEAR_MAX_M,
    LANE_SINGLE_NEAR_REQUIRE_M,
    LANE_VISION_MIRROR_CENTER_MAX_M,
    LANE_VISION_PAIR_WIDTH_MAX_M,
    LANE_VISION_RIGHT_MIRROR_CENTER_MAX_M,
    LANE_VISION_RIGHT_MIRROR_CONF_MAX,
    LANE_WIDTH_DEFAULT_M,
    LANE_WIDTH_MIN_M,
    MARKING_ALIGNMENT_MIN,
)
from beamng_autopilot.vision.lanes import LaneMarking


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
