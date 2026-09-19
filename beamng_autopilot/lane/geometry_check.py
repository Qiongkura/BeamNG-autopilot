"""Geometric consistency of a lane candidate or a left/right pair (plan E4).

The pairing gates that exist are *static*: a width band, a minimum span, a
heading agreement.  The plan asks for the geometric relationships instead,
and names them all:

* curve continuity - each boundary may not bend tighter than a real road;
* curvature rate - and its curvature may not jump between samples;
* left/right spacing - the lane width itself;
* lane-width rate of change - the one the plan calls out by name: "拒绝世界
  坐标中车道宽度突然异常的配对", a pair whose width swings along the lane
  is a mis-pairing even when every single sample sits inside the band;
* vanishing-point consistency - two real boundaries converge somewhere
  AHEAD, so a pair that meets within a few metres is a wedge, not a lane;
* relation to the drivable area - the lane centre must ride on what the
  road mask calls drivable;
* relation to the LiDAR corridor - the lane centre must agree with the
  observed free corridor, not merely be smooth;
* deviation from the previous frame's reference - a jump between frames is
  a perception flip, not a lane change.

Pure logic: the caller passes polylines (and optionally a drivable grid, a
corridor polyline and the previous reference) and gets a report.  Nothing
here reads a map, a route or a lateral offset constant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from beamng_autopilot.lane.constants import LANE_WIDTH_MAX_M, LANE_WIDTH_MIN_M

# A boundary tighter than this radius (1/curvature) is not a road edge.
LANE_MAX_CURVATURE_PER_M = 0.05          # 20 m radius
# |d curvature / d arc|: a real boundary's curvature changes smoothly.
LANE_MAX_CURVATURE_RATE = 0.02
# |d width / d arc|: a lane does not gain or lose centimetres per metre.
LANE_MAX_WIDTH_RATE_M_PER_M = 0.15
# Two boundaries that meet closer than this are a wedge (mis-pairing).
LANE_MIN_VANISH_M = 8.0
# Above this |cos| between the two boundary directions the pair is treated
# as parallel: no useful vanishing distance (which is fine).
LANE_VANISH_PARALLEL_COS = 0.985
# The lane centre must sit on the drivable evidence.
LANE_DRIVABLE_MIN_FRAC = 0.5
LANE_CORRIDOR_MAX_DEV_M = 2.0
LANE_REF_DEV_MAX_M = 1.0


@dataclass
class LaneGeometryLimits:
    """Thresholds for the checks; every one is a policy knob."""

    width_min_m: float = LANE_WIDTH_MIN_M
    width_max_m: float = LANE_WIDTH_MAX_M
    max_width_rate: float = LANE_MAX_WIDTH_RATE_M_PER_M
    max_curvature_per_m: float = LANE_MAX_CURVATURE_PER_M
    max_curvature_rate: float = LANE_MAX_CURVATURE_RATE
    min_vanish_m: float = LANE_MIN_VANISH_M
    vanish_parallel_cos: float = LANE_VANISH_PARALLEL_COS
    drivable_min_frac: float = LANE_DRIVABLE_MIN_FRAC
    corridor_max_dev_m: float = LANE_CORRIDOR_MAX_DEV_M
    ref_dev_max_m: float = LANE_REF_DEV_MAX_M


@dataclass
class LaneGeometryReport:
    """Verdict plus the measured values behind it."""

    ok: bool = True
    reasons: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def digest(self) -> dict:
        def _r(v):
            if v is None:
                return None
            if isinstance(v, float):
                return None if not math.isfinite(v) else round(v, 4)
            return v
        return {"ok": int(bool(self.ok)),
                "why": ",".join(self.reasons) if self.reasons else "",
                "m": {k: _r(v) for k, v in self.metrics.items()}}


# ---------------------------------------------------------------------------
# geometry helpers (lane-local, no planner dependency)
# ---------------------------------------------------------------------------

def resample_polyline(path, step_m: float = 1.0):
    """Resample to uniform arc spacing (near->far order preserved)."""
    pts = np.asarray(path, dtype=float)[:, :2]
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 2:
        return pts, np.zeros(len(pts))
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(arc[-1])
    if total <= 1e-9:
        return pts[:1], np.zeros(1)
    step = max(1e-3, float(step_m))
    n = max(2, int(math.floor(total / step)) + 1)
    grid = np.linspace(0.0, total, n)
    out = np.column_stack([np.interp(grid, arc, pts[:, 0]),
                           np.interp(grid, arc, pts[:, 1])])
    return out, grid


def signed_curvature(pts) -> np.ndarray:
    """Per-vertex signed curvature ``2*cross/(n1*n2*(n1+n2))``.

    Step-independent (the formula the repo already uses for route radii);
    positive means the polyline bends left.
    """
    p = np.asarray(pts, dtype=float)[:, :2]
    if len(p) < 3:
        return np.zeros(max(0, len(p) - 2))
    # consecutive SEGMENT pairs: d[i] and d[i+1] give one curvature value,
    # so both the cross product and the denominator must be built from
    # d[:-1] / d[1:] (mixing d1 with d2 desynchronises the shapes)
    d = np.diff(p, axis=0)                    # (N-1, 2)
    n = np.linalg.norm(d, axis=1)             # (N-1,)
    cross = d[:-1, 0] * d[1:, 1] - d[:-1, 1] * d[1:, 0]
    denom = np.maximum(n[:-1] * n[1:] * (n[:-1] + n[1:]), 1e-9)
    return 2.0 * cross / denom


def point_to_polyline_m(points, poly) -> np.ndarray:
    """Unsigned distance from each point to a polyline."""
    pts = np.asarray(points, dtype=float)[:, :2]
    line = np.asarray(poly, dtype=float)[:, :2]
    if len(line) < 2 or len(pts) == 0:
        return np.full(len(pts), float("inf"))
    a, b = line[:-1], line[1:]
    ab = b - a
    l2 = np.maximum((ab * ab).sum(axis=1), 1e-12)
    rel = pts[:, None, :] - a[None, :, :]
    t = np.clip(np.einsum("ijk,jk->ij", rel, ab) / l2[None, :], 0.0, 1.0)
    proj = a[None, :, :] + t[..., None] * ab[None, :, :]
    d = np.linalg.norm(pts[:, None, :] - proj, axis=2)
    return d.min(axis=1)


def width_profile(left, right, step_m: float = 1.0):
    """``(arc, width)`` of a pair, width = left sample -> right polyline."""
    l_pts, l_arc = resample_polyline(left, step_m)
    r_pts, _ = resample_polyline(right, step_m)
    if len(l_pts) < 2 or len(r_pts) < 2:
        return np.zeros(0), np.zeros(0)
    w = point_to_polyline_m(l_pts, r_pts)
    return l_arc, w


def vanishing_distance_m(left, right, look_ahead_m: float = 30.0):
    """Where the two boundaries' near-ahead directions meet, in metres.

    ``None`` when they are parallel enough that the intersection is
    meaningless (the healthy case for a straight road).
    """
    l_pts, _ = resample_polyline(left, 1.0)
    r_pts, _ = resample_polyline(right, 1.0)
    if len(l_pts) < 2 or len(r_pts) < 2:
        return None
    look = max(2.0, float(look_ahead_m))

    def _dir(pts):
        seg = np.diff(pts, axis=0)
        seg = seg[np.linalg.norm(seg, axis=1) > 1e-6]
        if len(seg) == 0:
            return None, None
        d = seg.mean(axis=0)
        n = float(np.linalg.norm(d))
        if n < 1e-9:
            return None, None
        return d / n, pts

    dl, lp = _dir(l_pts)
    dr, rp = _dir(r_pts)
    if dl is None or dr is None:
        return None
    cross = float(dl[0] * dr[1] - dl[1] * dr[0])
    if abs(cross) < math.sqrt(max(0.0, 1.0 - LANE_VANISH_PARALLEL_COS ** 2)):
        return None                       # parallel: no meaningful meeting
    # solve lp0 + a*dl = rp0 + b*dr for a (distance along the left line)
    rhs = rp[0] - lp[0]
    a = (rhs[0] * dr[1] - rhs[1] * dr[0]) / cross
    return max(0.0, float(a))


def drivable_overlap(center, drivable) -> float | None:
    """Fraction of lane-centre samples whose cell is drivable.

    ``drivable`` is any object exposing ``world_to_cell(x, y)`` and a
    2-D boolean ``drivable`` array (the BEV grid used across the stack);
    None when there is no evidence to check against.
    """
    if drivable is None or center is None:
        return None
    grid = getattr(drivable, "drivable", None)
    fn = getattr(drivable, "world_to_cell", None)
    if grid is None or fn is None:
        return None
    pts = np.asarray(center, dtype=float)[:, :2]
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) == 0:
        return None
    total = 0
    inside = 0
    for x, y in pts:
        cell = fn(float(x), float(y))
        if cell is None:
            continue
        r, c = int(cell[0]), int(cell[1])
        if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
            total += 1
            inside += int(bool(grid[r, c]))
    if total == 0:
        return None
    return inside / float(total)


def polyline_deviation_m(path, ref) -> float | None:
    """Median distance from ``path`` samples to a reference polyline."""
    if ref is None or path is None:
        return None
    p = np.asarray(path, dtype=float)[:, :2]
    p = p[np.isfinite(p).all(axis=1)]
    r = np.asarray(ref, dtype=float)[:, :2]
    r = r[np.isfinite(r).all(axis=1)]
    if len(p) < 2 or len(r) < 2:
        return None
    return float(np.median(point_to_polyline_m(p, r)))


# ---------------------------------------------------------------------------
# the checks
# ---------------------------------------------------------------------------

def check_lane_geometry(*, center, left=None, right=None,
                        limits: LaneGeometryLimits | None = None,
                        drivable=None, corridor=None, prev_ref=None,
                        ) -> LaneGeometryReport:
    """Run every consistency check the plan lists; return the verdict."""
    lim = limits or LaneGeometryLimits()
    rep = LaneGeometryReport()
    center = None if center is None else np.asarray(center, dtype=float)[:, :2]

    # --- per-boundary smoothness: curvature bound and its rate of change
    for name, boundary in (("left", left), ("right", right)):
        if boundary is None:
            continue
        pts, arc = resample_polyline(boundary, 1.0)
        if len(pts) < 3:
            continue
        k = signed_curvature(pts)
        if len(k):
            kmax = float(np.max(np.abs(k)))
            rep.metrics[f"{name}_curv"] = kmax
            if kmax > lim.max_curvature_per_m:
                rep.ok = False
                rep.reasons.append(f"{name}_curve")
            if len(k) >= 2:
                # curvature sample i sits at the junction of segments i and
                # i+1, so the spacing between two samples is the NEXT
                # segment's length - not diff(arc), whose length differs
                dk = np.diff(k)
                seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
                ds = np.maximum(seg[1:1 + len(dk)], 1e-6)
                krate = float(np.max(np.abs(dk / ds))) if len(ds) == len(dk)                     else 0.0
                rep.metrics[f"{name}_curv_rate"] = krate
                if krate > lim.max_curvature_rate:
                    rep.ok = False
                    rep.reasons.append(f"{name}_curv_rate")

    # --- the pair: width band, width rate, vanishing point
    if left is not None and right is not None:
        arc, w = width_profile(left, right)
        if len(w) >= 2:
            w_med = float(np.median(w))
            rep.metrics["width_m"] = w_med
            rep.metrics["width_min"] = float(np.min(w))
            rep.metrics["width_max"] = float(np.max(w))
            if not (lim.width_min_m <= w_med <= lim.width_max_m):
                rep.ok = False
                rep.reasons.append("width_band")
            ds = np.maximum(np.diff(arc[:len(w)]), 1e-6)
            if len(w) >= 2 and len(ds) == len(w) - 1:
                wrate = float(np.max(np.abs(np.diff(w) / ds)))
                rep.metrics["width_rate"] = wrate
                if wrate > lim.max_width_rate:
                    rep.ok = False
                    rep.reasons.append("width_rate")
        vanish = vanishing_distance_m(left, right)
        if vanish is not None:
            rep.metrics["vanish_m"] = vanish
            if vanish < lim.min_vanish_m:
                rep.ok = False
                rep.reasons.append("vanish")

    # --- context: drivable area, LiDAR corridor, previous reference
    if center is not None:
        frac = drivable_overlap(center, drivable)
        if frac is not None:
            rep.metrics["drivable_frac"] = frac
            if frac < lim.drivable_min_frac:
                rep.ok = False
                rep.reasons.append("drivable")
        if corridor is not None:
            dev = polyline_deviation_m(center, corridor)
            if dev is not None:
                rep.metrics["corridor_dev_m"] = dev
                if dev > lim.corridor_max_dev_m:
                    rep.ok = False
                    rep.reasons.append("corridor")
        if prev_ref is not None:
            dev = polyline_deviation_m(center, prev_ref)
            if dev is not None:
                rep.metrics["ref_dev_m"] = dev
                if dev > lim.ref_dev_max_m:
                    rep.ok = False
                    rep.reasons.append("ref_jump")
    return rep
