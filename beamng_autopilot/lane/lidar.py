"""LiDAR raycast corridor: free-space lane estimation from physics rays."""

from __future__ import annotations

import math

import numpy as np

from .constants import (
    LANE_FUSION_WIDTH_MAX_M,
    LANE_MIN_SPAN_M,
    LANE_SINGLE_LIDAR_CENTER_MAX_M,
    LANE_WIDTH_DEFAULT_M,
    LIDAR_EDGE_MIN_M,
    LIDAR_MAX_DIST_M,
    LIDAR_MAX_LAT_M,
    LIDAR_STATION_M,
)
from .pairing import LaneFrame, _frame_from_stations, _unit_fwd


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
                # A LiDAR corridor is the drivable free space between
                # walls / guardrails, not necessarily the current lane:
                # on the highway the guardrails are 10-17 m apart (two or
                # three lanes between them), so their midpoint is NOT the
                # lane the car belongs to - following it dragged the car
                # 8+ m sideways (run 33).  Only a corridor that is about
                # one lane wide can act as the lane centre itself; a wider
                # corridor is dropped and the single-edge / nav fallback
                # keeps the car in its lane.
                if width > LANE_FUSION_WIDTH_MAX_M:
                    if debug is not None:
                        debug["too_wide"] = round(float(width), 2)
                    return None
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
