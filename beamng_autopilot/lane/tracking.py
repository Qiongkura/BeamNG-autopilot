"""Lane frame tracking over time: freshness, confidence decay, and stability checks."""

from __future__ import annotations

import math
import time

import numpy as np

from .constants import (
    LANE_FUSION_WIDTH_MAX_M,
    LANE_FRAME_MIN_SPAN_M,
    LANE_MIN_CONF,
    LANE_PAIR_NEAR_CENTER_MAX_M,
    LANE_PAIR_WIDTH_MIN_M,
    LANE_PAIRED_VISION_MIN_SPAN_M,
    LANE_RIGHT_MIRROR_NEAR_M,
    LANE_RIDING_LINE_MAX_M,
    LANE_SINGLE_NEAR_REQUIRE_M,
    LANE_VISION_MIRROR_CENTER_MAX_M,
    LANE_VISION_RIGHT_MIRROR_CENTER_MAX_M,
    LANE_WIDTH_MAX_M,
    TRACK_JUMP_MAX_M,
    TRACK_MAX_DIST_M,
    TRACK_REJECT_MATCH_M,
    TRACK_REJECT_WINDOW_S,
    TRACK_STALE_M,
    TRACK_STALE_S,
    TRACK_STATION_M,
)
from .pairing import LaneFrame


# ---------------------------------------------------------------------------
# Internal helpers used by the public functions below
# ---------------------------------------------------------------------------

def _unit_fwd(pos, heading: float, fwd=None) -> np.ndarray:
    if fwd is not None:
        f = np.asarray(fwd, dtype=float)[:2]
        n = float(np.linalg.norm(f))
        if n > 1e-9:
            return f / n
    return np.array([math.cos(heading), math.sin(heading)])


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


# ---------------------------------------------------------------------------
# 1. LaneTracker class
# ---------------------------------------------------------------------------

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
                      else LANE_WIDTH_MAX_M)
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


# ---------------------------------------------------------------------------
# 2. lane_frame_usable
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 3. _frame_near_lat
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 4. _fusion_center_unstable
# ---------------------------------------------------------------------------

def _fusion_center_unstable(frame: LaneFrame | None, pos, heading: float,
                            fwd=None) -> bool:
    """A fused lane whose centre is far from the car is a bad pair."""
    if (frame is None or not frame.paired
            or len(frame.sources) < 2 or pos is None):
        return False
    lat = _frame_near_lat(frame, pos, heading, fwd)
    return lat is not None and abs(lat) > LANE_PAIR_NEAR_CENTER_MAX_M


# ---------------------------------------------------------------------------
# 5. _boundary_near_lat
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 6. _mirror_near_ok
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 7. _mirror_right_ok
# ---------------------------------------------------------------------------

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