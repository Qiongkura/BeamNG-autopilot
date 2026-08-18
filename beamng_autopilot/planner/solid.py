"""Solid lane marking detection, noise filtering, and no-cross boundary enforcement."""


from __future__ import annotations

import math

import numpy as np

from .geometry import (
    _point_lat_offset,
    _pts_to_segments,
    _points_to_polyline_lat,
)
from .constants import (
    SOLID_MIN_CONF,
    SOLID_MIN_LEN_M,
    SOLID_MAX_CORRIDOR_DEV_M,
    SOLID_MIN_ALIGNMENT,
    SOLID_ANCHOR_NEAR_M,
    SOLID_BLOCK_MIN_M,
    SOLID_BLOCK_MAX_M,
    SOLID_BLOCK_LANE_CONF,
    SOLID_LINE_MARGIN,
    SOLID_MAX_LAT_SPAN_FACTOR,
    SOLID_MAX_LAT_SPAN_M,
    SOLID_MAX_PERP_SPAN_M,
    SOLID_MAX_PERP_SPAN_FRAC,
    LANE_EDGE_WALL_MIN_ALIGN,
    LANE_EDGE_WALL_MAX_THICK_M,
    LANE_EDGE_WALL_MAX_INTRUDE_M,
    LANE_EDGE_WALL_EDGE_TOL_M,
    ROADSIDE_WALL_MIN_LEN_M,
    ROADSIDE_WALL_MAX_THICK_M,
    ROADSIDE_WALL_MIN_EDGE_M,
    CAR_HALF_WIDTH,
)
from beamng_autopilot.vision.lanes import marking_is_zigzag
from .obstacles import _obstacle_oriented, _obstacle_corners


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
    # Cheap bounding-box reject before any full point-to-segment work:
    # a marking whose box is farther than the corridor tolerance from the
    # path box in either axis cannot be the boundary we drive against.
    # Vision can feed dozens of markings per frame; this keeps the solid
    # stage at a few ms instead of 200+.
    if (w[:, 0].min() > p[:, 0].max() + SOLID_MAX_CORRIDOR_DEV_M
            or w[:, 0].max() < p[:, 0].min() - SOLID_MAX_CORRIDOR_DEV_M
            or w[:, 1].min() > p[:, 1].max() + SOLID_MAX_CORRIDOR_DEV_M
            or w[:, 1].max() < p[:, 1].min() - SOLID_MAX_CORRIDOR_DEV_M):
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

