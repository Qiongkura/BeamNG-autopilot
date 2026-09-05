from __future__ import annotations

import math

import numpy as np

from .constants import (
    FUSION_WALL_SAFE_MARGIN_M,
    LANE_FAR_MIRROR_CONF_MAX,
    LANE_FAR_START_MAX_M,
    LANE_FUSION_AGREE_MAX_M,
    LANE_FUSION_CENTER_CONT_MAX_M,
    LANE_FUSION_HOLD_FRAMES,
    LANE_FUSION_HOLD_NONE_FRAMES,
    LANE_FUSION_PAIRED_CONF_MAX,
    LANE_FUSION_PAIRED_HOLD_FRAMES,
    LANE_FUSION_WIDTH_MAX_M,
    LANE_MIN_CONF,
    LANE_PAIRED_VISION_MIN_SPAN_M,
    LANE_PAIR_NEAR_CENTER_MAX_M,
    LANE_PAIR_OVERLAP_M,
    LANE_RIDING_LINE_MAX_M,
    LANE_SINGLE_NEAR_REQUIRE_M,
    LANE_WIDTH_MIN_M,
    TRACK_STATION_M,
)
from .pairing import LaneFrame, _unit_fwd, _overlap_stations, _frame_from_stations
from .tracking import (
    lane_frame_usable,
    _frame_near_lat,
    _boundary_near_lat,
    _mirror_near_ok,
    _mirror_right_ok,
    _fusion_center_unstable,
)

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
    # Anti-flicker counts CONSECUTIVE frames of the CANDIDATE source.
    # The old code tested the ACTIVE source's tenure, which in steady
    # state is always >= hold, so any one-frame glitch adopted instantly
    # (bug audit 2026-09-06).
    if state.get("cand_src") == src:
        state["cand_frames"] = int(state.get("cand_frames", 0)) + 1
    else:
        state["cand_src"] = src
        state["cand_frames"] = 1
    if int(state.get("cand_frames", 0)) >= hold:
        state["src"] = src
        state["frames"] = 1
        state["last"] = chosen
        state["cand_frames"] = 0
        return chosen
    return state.get("last")
