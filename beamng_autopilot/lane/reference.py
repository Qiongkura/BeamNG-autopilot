"""Single owner of the "where is MY lane" decision for the FSD stack.

The planner, the safety monitor and the candidate generator all need one
answer to the same question: **which centre-line and which hard
boundaries may steer the car right now**.  That policy used to be
assembled inline inside ``FSDStack.tick`` from the sensor lane, the
nav-route map prior, the heading / corner / side gates and the
strict-perception rule - roughly 250 lines buried in the perception
method.

Making it a module of its own does three things:

* the policy becomes a unit with its own tests instead of a branch inside
  a 1500-line orchestrator;
* the map fallback exists in exactly one place, so review can see whether
  map-derived geometry can reach the planner's lateral reference;
* strict perception (``lane_mode="sensor"`` + ``strict_sensor=True``)
  never builds map lane geometry at all - that rule is structural here,
  not a per-branch check (docs/fsd_realism.md §1/§4).

Everything in this module is either strict-perception-exempt (the map
prior, used only in the legacy non-strict mode) or sensor-derived.
"""

from __future__ import annotations

import os

from dataclasses import dataclass, field
import math

import numpy as np

from .constants import LANE_WIDTH_DEFAULT_M

# Plan E4: geometric consistency of the ACCEPTED sensor lane (smoothness,
# width rate, vanishing point, drivable overlap, corridor and previous-
# reference agreement).  Default OFF: like every other gate in this repo it
# is a live A/B lever, and it can only WITHDRAW a sensor lane - never
# relax a gate, never touch a map-derived reference.
LANE_GEOM_ENABLED = os.environ.get("BEAMNG_LANE_GEOM", "0") == "1"
from beamng_autopilot.fsd_realism import (
    SRC_BEV_ROUTE,
    SRC_CORRIDOR,
    SRC_MAP,
    SRC_PAVED,
    SRC_SENSOR,
    SRC_UNAVAILABLE,
)


# Telemetry label per canonical lane source.  ``lane_src_sel`` is the
# machine contract other modules gate on; ``lane_src`` is the label the
# drive loop, the scorecard and the shadow episodes record.  Both are
# published from the SAME final decision in this module, because a second
# derivation is how they drifted apart: FSDStack.tick used to rebuild
# ``lane_src`` from "is there a lane frame / a map lane", so a sensor lane
# that the consistency check had already downgraded to the map prior still
# logged ``lane_src=sensor``, and strict frames logged ``lane_src=bev/route``
# while ``lane_src_sel`` said ``perception-unavailable`` (109 of 217 frames,
# town run 2026-09-07).
_LABEL_BY_SRC = {
    SRC_SENSOR: "sensor",
    SRC_MAP: "map_lane",
    SRC_CORRIDOR: "corridor",
    SRC_PAVED: "paved",
    SRC_UNAVAILABLE: "perception-unavailable",
}


# Heading gate threshold for a PAIRED sensor lane against the nav route.
# The old 60 deg default let a lane locked onto a DIFFERENT road (or a
# centreline fitted from roadside clutter after the ego drifted) through:
# live run 2026-08-27 opt12, the sensor lane at (705,708) pointed
# -56.6 deg while the route ran -111 deg (55 deg off) and the planner
# generated a 51 deg left arc that drove the car into the right-side
# wall.  A real stack trusts its map/nav intent: anything more than
# 35 deg off is a different roadway and falls back to the map-prior own
# lane (which still rounds real hairpins).
LANE_HEADING_MAX_YAW_DEG = 35.0


# Route-turn gate: when the nav route turns more than this in the next
# window the car is at a real corner.  There the map-prior own lane
# (rounded from the same route) is authoritative: vision/LiDAR lane
# pairing reads corner geometry wide (town run 2026-08-28 run11: sensor
# lane -48.6 deg vs route -24.6 deg passed the 35 deg heading gate and
# the car S-curved 2.7 m left then 5.2 m right of the centreline).
LANE_ROUTE_TURN_MAX_DEG = 25.0
LANE_ROUTE_TURN_LOOK_M = 12.0


def _warn(warn, key: str, msg: str) -> None:
    """Emit a one-shot diagnostic through the caller's warner, if any."""
    if warn is None:
        return
    try:
        warn(key, msg)
    except Exception:
        pass


def bev_drivable_center(grid, pos, heading):
    """World centreline of the drivable space in the BEV grid.

    For each longitudinal band of the grid, the lateral centre of the
    drivable cells becomes one point of a lane reference - the "space
    centreline" a real vector-space planner tracks.  Falls back to a
    straight line ahead when few drivable cells exist (unknown road,
    sensor-limited frame).
    """
    drv = getattr(grid, "drivable", None)
    if drv is None or not getattr(drv, "any", lambda: False)():
        return None
    # The lane centre must be over the FREE corridor: drivable road cells
    # that are NOT inside an obstacle footprint.  A roadside or corner
    # wall erases the drivable cells it occupies via obstacle fusion, so
    # using plain "drivable" would pull the centreline into the wall
    # (town corner runs 2026-08-21).  When the corridor is so dense that
    # no free cell survives, fall back to the raw drivable cells so the
    # sensor lane still exists.
    occ = getattr(grid, "obstacle", None)
    if occ is not None and occ.shape == drv.shape:
        free = np.logical_and(drv != 0, occ == 0)
    else:
        free = drv != 0
    if not free.any():
        free = drv != 0
    n = int(getattr(grid, "n_rows", None) or
            getattr(grid, "n_cols", None) or 60)
    res = grid.res
    extent = grid.extent
    step = max(1, n // 24)
    pts = []
    for r in range(0, n, step):
        row = free[r]
        cols = np.nonzero(row)[0]
        if cols.size == 0:
            continue
        c_mid = float(cols.mean())
        ex = extent - (r + 0.5) * res
        ey = extent - (c_mid + 0.5) * res
        # ego -> world
        ch = math.cos(float(heading))
        sh = math.sin(float(heading))
        wx = float(pos[0]) + ex * ch - ey * sh
        wy = float(pos[1]) + ex * sh + ey * ch
        pts.append((wx, wy))
    if len(pts) < 3:
        return None
    arr = np.asarray(pts, dtype=float)
    # Anchor the centreline at the ego and order it near -> far so the
    # planner sees a path it can actually drive from here.  The raw grid
    # rows run far -> near, so without re-anchoring the first point sits
    # metres ahead of the car and every candidate gets scored as
    # off-lane (town runs 2026-08-21: all arcs infeasible at (717.8,754.6)
    # because the reference started 8 m away).
    d = arr - np.asarray(pos[:2], dtype=float)
    fwd_m = d[:, 0] * math.cos(float(heading)) + \
            d[:, 1] * math.sin(float(heading))
    ahead = arr[fwd_m > 0.5]
    if len(ahead) < 3:
        ahead = arr
    d0 = np.linalg.norm(ahead - np.asarray(pos[:2], dtype=float), axis=1)
    ahead = ahead[np.argsort(d0)]          # near -> far
    # Anchor the reference at the ego: when the nearest drivable row is
    # already a couple of metres in front, prepend the ego so the
    # planner's shift candidates start inside the forward-progress gate
    # (a reference whose first point is beyond ~3 m got every lane-shift
    # candidate rejected - town runs 2026-08-21).
    if len(ahead) and float(np.linalg.norm(
            ahead[0] - np.asarray(pos[:2], dtype=float))) > 2.0:
        ahead = np.vstack([np.asarray(pos[:2], dtype=float), ahead])
    return ahead if len(ahead) >= 3 else None


# Width gate for the corridor lane candidate, MEASURED not guessed: over
# 24 town shadow episodes / 4951 frames (scripts/m5_lateral_ref_probe.py,
# in-lane judged against a corrected 0 m ego-lane centre), the candidate
# below scores 84.7% (width 7-8 m), 96.0% (8-9 m) and 100% (9-10 m), then
# collapses to 61.9% (10-12 m) and 0.0% above 12 m - on a wide road the
# corridor's right edge is NOT the ego lane's right boundary and the same
# formula points ~6 m off the lane.  So the candidate must abstain above
# this width and let strict mode fail closed instead.
CORRIDOR_LANE_MAX_WIDTH_M = 10.0


def bev_corridor_lane_center(grid, pos, heading,
                             max_width_m: float = CORRIDOR_LANE_MAX_WIDTH_M,
                             lane_half_m: float | None = None):
    """World polyline of the ego-lane centre from the FREE corridor's right edge.

    .. warning::
        **REFUTED LIVE, do not enable** (2026-09-12, town corridor arm
        ``town_1789142315``): the free corridor's right edge is the ROAD's
        right edge, not the ego lane's right boundary.  Measured on that
        run's own GT frames: painted right line at −1.43 m but drivable
        right edge at −3.75 m — the road mask spills ~1.9 m past the lane
        line (shoulder), so this candidate pointed 1.24 m (p50) right of
        the true lane centre and >1.2 m off on 52.7% of frames.  A whole-
        corridor width gate cannot catch this: "two lanes" and "two lanes
        + right shoulder" have the same total width, and nothing at
        runtime verifies the assumption.  The offline 95% in-lane (24
        episodes) did not transfer.  Kept only as the recorded negative
        result; ``corridor_fallback`` must stay False unless a redesign
        anchors the boundary to painted lines and re-passes the live
        safety gate (line_lat centred, 0 crossing / 0 off-road / 0
        reversing).
    """
    drv = getattr(grid, "drivable", None)
    if drv is None or not getattr(drv, "any", lambda: False)():
        return None
    occ = getattr(grid, "obstacle", None)
    if occ is not None and occ.shape == drv.shape:
        free = np.logical_and(drv != 0, occ == 0)
    else:
        free = drv != 0
    if not free.any():
        return None
    n = int(getattr(grid, "n_rows", None) or
            getattr(grid, "n_cols", None) or 60)
    res = float(grid.res)
    extent = float(grid.extent)
    if lane_half_m is None:
        lane_half_m = 0.5 * float(LANE_WIDTH_DEFAULT_M)
    ch = math.cos(float(heading))
    sh = math.sin(float(heading))
    step = max(1, n // 24)
    pts: list[tuple[float, float]] = []
    widths: list[float] = []
    for r in range(0, n, step):
        cols = np.nonzero(free[r])[0]
        if cols.size == 0:
            continue
        ex = extent - (r + 0.5) * res
        ey_right = extent - (cols.max() + 0.5) * res
        ey_left = extent - (cols.min() + 0.5) * res
        # Gate on the NEAR field only: that is the band the validated
        # width buckets come from, and far rows can spill onto junctions.
        if 3.0 <= ex <= 15.0:
            widths.append(ey_left - ey_right)
        if ex < 0.5:
            continue
        ey = ey_right + lane_half_m
        pts.append((float(pos[0]) + ex * ch - ey * sh,
                    float(pos[1]) + ex * sh + ey * ch))
    if len(widths) < 3 or float(np.median(widths)) > max_width_m:
        return None
    if len(pts) < 3:
        return None
    arr = np.asarray(pts, dtype=float)
    d = arr - np.asarray(pos[:2], dtype=float)
    fwd_m = d[:, 0] * ch + d[:, 1] * sh
    ahead = arr[fwd_m > 0.5]
    if len(ahead) < 3:
        return None
    ahead = ahead[np.argsort(np.linalg.norm(
        ahead - np.asarray(pos[:2], dtype=float), axis=1))]
    if float(np.linalg.norm(
            ahead[0] - np.asarray(pos[:2], dtype=float))) > 2.0:
        ahead = np.vstack([np.asarray(pos[:2], dtype=float), ahead])
    return ahead if len(ahead) >= 3 else None


# --- own lane beside a painted DIVIDER ---------------------------------
# Minimum world length for a marking to be treated as a lane boundary
# rather than a paint fragment.
DIVIDER_LINE_MIN_LEN_M = 5.0
# A marking whose median lateral offset from the nav ROUTE (the road
# centreline) is inside this band is the road's divider / centre paint.
# The route is used as a *road-centre metric* here, exactly as the side
# gate uses it - never as a lateral setpoint.
DIVIDER_ON_ROUTE_BAND_M = 0.85
# The derived own-lane centre may sit at most this far from the divider:
# beyond it the "half a lane to the right" read would be a lane change,
# not a lane-keeping reference.
DIVIDER_MAX_CENTRE_OFFSET_M = 2.6
# Fraction of sampled centre points that must land on drivable, observed
# pavement before the shift may steer the car.
DIVIDER_DRIVABLE_MIN_FRAC = 0.6
# --- 铺装门只回答"在不在观测到的可行驶面上"（单因子开关，默认关）---------
# 实测 2026-10-06（2 次 town、415 settled 帧）：被该门撤销的 202/202 帧**全部**由
# obstacle 合取项触发，n_off_mask（观测到但不在可行驶掩码上）中位 0——即没有一帧是
# 因为"中心不在可行驶面上"被撤销的。而 obstacle 层是 BEV 头的预测
# （fsd_stack.py: `grid.obstacle[:] = (out.bev >= 0.6)`），于是该门在实车链上退化成
# "BEV 头在车道中心误报障碍 → 撤掉整条车道" → plan_blocked=no_perception_lane
# （119-175 帧/次）→ 无路径 → 停车，正是路径可用率只有 ~20-35% 的直接机制。
# 打开后：车道接受只按可行驶掩码判定；obstacle 证据保留在诊断里，且所有**路径级**
# 检查（path_blocked/占位、车体扫掠、风险层）一字不动。
LANE_PAVEMENT_GATE_NO_OCC = os.environ.get(
    "BEAMNG_LANE_PAVEMENT_GATE_NO_OCC", "0") != "0"


def own_lane_beside_divider(markings, pos, heading, route_ref, grid,
                            lane_half_m: float = 0.5 * LANE_WIDTH_DEFAULT_M,
                            debug: dict | None = None):
    """Own-lane centre from a painted divider, inside observed pavement.

    The failure this answers: the car stands ON the road centreline (a
    teleport / placement that could not find a perception lane leaves it
    on the route snap).  Perception then sees the WHOLE road - the divider
    paint beside the car and the opposite edge - so the paired-lane read
    is the road corridor, its centre IS the centreline, and the side gate
    correctly refuses it.  Every read is refused, so a strict car can
    never move off the line it is stuck on: the gate is right and the
    behaviour is still useless.

    A painted divider is unambiguous evidence about where the own lane is
    under right-hand traffic: the lane lies ``lane_half_m`` to the RIGHT
    of that paint.  This builds exactly that centreline and then requires
    it to survive the same safety frame the other candidates do:

    * the divider must be a long solid/dashed marking (a fragment is not
      a boundary) whose median offset from the nav ROUTE is within
      ``DIVIDER_ON_ROUTE_BAND_M`` - i.e. it sits on the road centre,
      which is what makes it a divider and not a lane edge;
    * the derived centre must pass the SIDE gate (right of the route);
    * the derived centre must lie on pavement the sensors actually
      OBSERVED as drivable (CNN road mask + LiDAR), for at least
      ``DIVIDER_DRIVABLE_MIN_FRAC`` of the sampled points - so a wrong
      "divider" cannot park the car on the shoulder;
    * it must stay within ``DIVIDER_MAX_CENTRE_OFFSET_M`` of the divider.

    Returns ``(center, left_boundary, line)`` or ``None``.  Perception
    only: the route decides which SIDE, the paint and the pavement decide
    WHERE.
    """
    if not markings or route_ref is None or len(route_ref) < 2:
        return None
    p = np.asarray(pos[:2], dtype=float)
    fwd = np.array([math.cos(float(heading)), math.sin(float(heading))])
    left_ax = np.array([-fwd[1], fwd[0]])
    best = None
    for mk in markings:
        kind = str(getattr(mk, "kind", "") or "")
        if kind not in ("solid", "dashed"):
            continue
        conf = float(getattr(mk, "confidence", 0.0) or 0.0)
        if conf < 0.4:
            continue
        world = np.asarray(getattr(mk, "world", np.zeros((0, 2))), dtype=float)
        if world.ndim != 2 or world.shape[0] < 2:
            continue
        world = world[:, :2]
        world = world[np.isfinite(world).all(axis=1)]
        if len(world) < 2:
            continue
        seg = np.linalg.norm(np.diff(world, axis=0), axis=1)
        if float(seg.sum()) < DIVIDER_LINE_MIN_LEN_M:
            continue
        rel = world - p
        lon = rel @ fwd
        lat = rel @ left_ax
        near = (lon >= -2.0) & (lon <= 18.0)
        if int(near.sum()) < 3:
            continue
        # Car-frame offset: the divider must be beside the car, not a
        # lane away (then the car is already somewhere else).
        car_lat = float(np.median(lat[near]))
        if abs(car_lat) > DIVIDER_MAX_CENTRE_OFFSET_M:
            continue
        route_lat = _median_lat_vs_ref(world, route_ref, p)
        if route_lat is None or abs(route_lat) > DIVIDER_ON_ROUTE_BAND_M:
            continue
        score = (float(len(world)), conf)
        if best is None or score > best[0]:
            best = (score, world, kind, conf, car_lat, route_lat)
    if best is None:
        if debug is not None:
            debug["reason"] = "no_divider_line"
        return None
    _score, world, kind, conf, car_lat, route_lat = best
    # Order the divider near -> far and build its local right-hand normal.
    order = np.argsort(np.linalg.norm(world - p[None, :], axis=1))
    line = world[order]
    if len(line) < 2:
        return None
    tang = line[-1] - line[0]
    L = float(np.linalg.norm(tang))
    if L < 1e-6:
        return None
    tang = tang / L
    if float(tang @ fwd) < 0.0:          # orient along the travel direction
        tang = -tang
    right = np.array([tang[1], -tang[0]])
    center = line + right[None, :] * float(lane_half_m)
    off = float(np.median((center - p) @ left_ax))
    if abs(off) > DIVIDER_MAX_CENTRE_OFFSET_M:
        if debug is not None:
            debug["reason"] = f"centre_too_far({off:.2f})"
        return None
    side = _median_lat_vs_ref(center, route_ref, p)
    if side is None or float(side) > -0.2:
        if debug is not None:
            debug["reason"] = ("side_unmeasurable" if side is None
                               else f"centre_not_right_of_route({side:.2f})")
        return None
    frac, n_obs = _drivable_fraction(center, grid)
    if n_obs < 3 or frac < DIVIDER_DRIVABLE_MIN_FRAC:
        if debug is not None:
            debug["reason"] = (f"centre_off_drivable(obs={n_obs},"
                               f"frac={frac:.2f})")
        return None
    if debug is not None:
        debug.update({"line_lat_car_m": round(car_lat, 2),
                      "line_lat_route_m": round(route_lat, 2),
                      "centre_lat_car_m": round(off, 2),
                      "centre_lat_route_m": round(float(side), 2),
                      "kind": kind, "conf": round(conf, 2),
                      "pts": int(len(line)),
                      "drivable_frac": round(frac, 3),
                      "drivable_obs": int(n_obs)})
    return center, line, {"line_lat_car_m": round(car_lat, 2),
                          "centre_lat_car_m": round(off, 2),
                          "kind": kind}


def _median_lat_vs_ref(pts, ref, pos) -> float | None:
    """Median signed lateral (left +) of ``pts`` against ``ref``."""
    arr = np.asarray(pts, dtype=float)[:, :2]
    r = np.asarray(ref, dtype=float)[:, :2]
    if len(arr) == 0 or len(r) < 2:
        return None
    seg = r[1:] - r[:-1]
    l2 = np.maximum((seg * seg).sum(axis=1), 1e-12)
    rel = arr[:, None, :] - r[None, :-1, :]
    t = np.clip(np.einsum("kmi,mi->km", rel, seg) / l2[None, :], 0.0, 1.0)
    proj = r[None, :-1, :] + t[..., None] * seg[None, :, :]
    d = np.linalg.norm(proj - arr[:, None, :], axis=2)
    bi = np.argmin(d, axis=1)
    rows = np.arange(len(arr))
    sy = arr[:, 1] - proj[rows, bi, 1]
    sx = arr[:, 0] - proj[rows, bi, 0]
    cross = seg[bi, 0] * sy - seg[bi, 1] * sx
    val = np.where(cross > 0, 1.0, -1.0) * d[rows, bi]
    keep = d[rows, bi] <= 25.0
    if not keep.any():
        return None
    return float(np.median(val[keep]))


def _drivable_fraction(center, grid, *, pos=None,
                       detail: dict | None = None,
                       near_m: float = 8.0,
                       ignore_obstacle: bool = False) -> tuple[float, int]:
    """(fraction of sampled centre points on observed-drivable, samples).

    ``detail`` (diagnostics only, optional): when given, it is filled with
    the per-sample breakdown the single ``frac`` cannot show — how many
    samples were unobserved / obstacle / non-drivable-mask, and the same
    fraction restricted to the NEAR range (where the LiDAR evidence is
    dense).  2026-10-06: 34% of paired frames are revoked by this gate
    (frac < 0.6) and a single number cannot tell "the centre really is off
    the pavement" apart from "the evidence is thin at the far end".
    """
    if grid is None or center is None or len(center) < 2:
        return 0.0, 0
    drv = getattr(grid, "drivable", None)
    occ = getattr(grid, "obstacle", None)
    obs = getattr(grid, "observed", None)
    if drv is None or not getattr(drv, "any", lambda: False)():
        return 0.0, 0
    pts = np.asarray(center, dtype=float)[:, :2]
    rows, cols, kept = [], [], []
    for x, y in pts:
        cell = grid.world_to_cell(float(x), float(y))
        if cell is None:
            continue
        rows.append(int(cell[0]))
        cols.append(int(cell[1]))
        kept.append((float(x), float(y)))
    if not kept:
        return 0.0, 0
    rr = np.asarray(rows)
    cc = np.asarray(cols)
    # ``pts_in`` = exactly the polyline points that mapped into the grid;
    # ``seen`` indexes THAT subset, not the full polyline (2026-10-06: the
    # near-range diagnostic indexed pts[seen] and raised IndexError as soon
    # as some points fell outside the grid — caught by the offline check).
    pts_in = np.asarray(kept, dtype=float)
    if obs is not None and getattr(obs, "size", 0) and obs.shape == drv.shape:
        seen = obs[rr, cc] > 0
    else:
        # No observed layer: the drivable mask itself is the evidence.
        seen = np.ones(len(rr), dtype=bool)
    if int(seen.sum()) < 3:
        if detail is not None:
            detail.update({"n": int(seen.sum()), "n_unobserved": int((~seen).sum()),
                           "n_obstacle": 0, "n_off_mask": 0, "frac_near": None,
                           "n_near": 0})
        return 0.0, int(seen.sum())
    good = drv[rr[seen], cc[seen]] > 0
    blocked = np.zeros(int(seen.sum()), dtype=bool)
    if occ is not None and getattr(occ, "size", 0) and occ.shape == drv.shape:
        blocked = occ[rr[seen], cc[seen]] != 0
        if not ignore_obstacle:
            good = np.logical_and(good, ~blocked)
    if detail is not None:
        _d = {"n": int(seen.sum()), "n_unobserved": int((~seen).sum()),
              "n_obstacle": int(blocked.sum()),
              "n_off_mask": (int((~good).sum()) if ignore_obstacle
                             else int((~good & ~blocked).sum())),
              "obstacle_veto": (not ignore_obstacle)}
        if pos is not None:
            _p = np.asarray(pos, dtype=float).ravel()[:2]
            _dpts = pts_in[seen]
            _dd = np.linalg.norm(_dpts - _p[None, :], axis=1)
            _near = _dd <= float(near_m)
            _d["n_near"] = int(_near.sum())
            _d["frac_near"] = (round(float(np.mean(good[_near])), 3)
                               if int(_near.sum()) >= 3 else None)
        detail.update(_d)
    return float(np.mean(good)), int(seen.sum())


@dataclass
class LaneReference:
    """One tick's lane-keep decision, ready for planner/safety/telemetry."""

    center: np.ndarray | None = None       # ``out.lane_ref`` (BEV incl.)
    left: np.ndarray | None = None         # hard left boundary
    right: np.ndarray | None = None        # hard right boundary
    width: float = 0.0
    src: str = ""                          # ``lane_src_sel``
    src_published: bool = False            # meta["lane_src_sel"] written?
    map_lane: tuple | None = None          # surviving map prior (post strict)
    rejected: bool = False                 # a gate dropped the sensor lane
    reject_reason: str | None = None
    strict: bool = False
    boundaries: bool = False               # publish left/right/width?
    # True when a sensor lane frame existed this tick (the planner Scene
    # keeps a reference for it, unlike the BEV whole-road fallback).
    frame_used: bool = False
    # --- evidence contract (plan §3.2/§3.3, T02) --------------------
    # Both boundaries of this reference are real measurements of one
    # observation.  Only this shape may earn full steering authority.
    two_sided: bool = False
    # The observation behind this reference was taken on THIS tick (not
    # served from a fusion hold / coast / replayed result).
    fresh_obs: bool = False
    # At least one published side comes from a prior (mirror / width).
    inferred: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def geom_id(self) -> str:
        """Identity of the FINAL accepted geometry (plan §3.3-4).

        Every consumer of a tick must carry this same id; a planner that
        reads a different centre than the controller (the slew-limiter
        divergence) shows up as two different ids instead of as a silent
        disagreement.  It is a pure function of the geometry, so it cannot
        drift out of date.
        """
        import hashlib
        h = hashlib.blake2b(digest_size=6)
        for name, arr in (("c", self.center), ("l", self.left),
                          ("r", self.right)):
            if arr is None:
                h.update(b"none")
                continue
            try:
                a = np.asarray(arr, dtype=float)
            except Exception:
                h.update(f"{name}:unreadable".encode())
                continue
            h.update((name + str(a.shape)).encode())
            h.update(np.nan_to_num(a, nan=-999.0).round(3).tobytes())
        h.update(f"w{float(self.width):.3f}".encode())
        return h.hexdigest()

    @property
    def publishable_sensor(self) -> bool:
        """May this decision be published as a ``sensor`` reference?

        Invariant (plan §3.3-1): an accepted ``sensor`` lateral reference
        must carry non-empty, finite centre geometry.  A candidate revoked
        by a gate must be reported unavailable/rejected, never published
        with ``src=sensor`` and an empty centre - the state that let a
        withdrawn read keep the perception label.

        Note what this does NOT say: it does not authorise a centre that
        exists but was withdrawn - the withdrawal clears the geometry
        (``select_lane_reference``) so the two cannot disagree.
        """
        if self.src != SRC_SENSOR:
            return True
        if self.center is None:
            return False
        try:
            arr = np.asarray(self.center, dtype=float)
        except Exception:
            return False
        return bool(arr.ndim == 2 and len(arr) >= 3 and arr.shape[1] >= 2
                    and np.isfinite(arr[:, :2]).all())

    @property
    def scene_ref(self) -> np.ndarray | None:
        """The lateral reference the planner Scene may use.

        The BEV whole-road centre is deliberately excluded: on a two-way
        road it IS the centre line the car must never ride.  Every
        ACCEPTED perception reference reaches the Scene - including the
        ones built without a raw ``LaneFrame`` (pavement-edge and free
        corridor fallbacks), which used to be dropped silently by the
        ``frame_used`` test (plan §3.3-5).
        """
        if self.center is None:
            return None
        if self.src in (SRC_BEV_ROUTE, ""):
            return None
        return self.center


def _lane_gate_verdict(*, lane_frame, lane_ref, route_ref, pos, heading,
                       lane_mode: str, strict_lane: bool) -> dict:
    """三维门（航向 / 转角 / 侧向）对**一个候选感知车道**的裁决。

    2026-10-20 从 ``select_lane_reference`` 里原样搬出来的纯判定：不改任何
    条件顺序、阈值来源或拒绝理由的取值，只是把"哪个门判的、拿什么数判的"
    变成一个可单测的返回值。

    返回 ``{"rejected", "reason", "side_bad", "corner_bad", "meta"}``：

    * ``rejected``：整条感知车道（中心 + 硬边界）都不可信，调用方必须把
      它降级，而不是只换中心——边界若留着，no-cross 规则会拿一条错误的
      边去拦车；
    * ``reason``：``"heading"`` / ``"corner"`` / ``"side"``，与
      ``lane_reject_reason`` 遥测字段同字面量；
    * ``meta``：``lane_side_off_m`` / ``lane_side_limit_m`` / ``pair_*``，
      让"门说不行"可复核（拒绝时也能看到触发它的那个数）。

    靠 hd 头（``lane_heading_ok`` / ``lane_route_turn_ok`` / ``lane_side_ok``）
    的异常照旧抛给调用方：那里有 ``_warn``，换成在这里吞掉会丢诊断。
    """
    from beamng_autopilot.planning.arbiter import (
        lane_heading_ok, lane_route_turn_ok, lane_side_ok,
        lane_side_offset_m)

    side_bad = False
    corner_bad = False
    rejected = False
    try:
        side_off = lane_side_offset_m(lane_ref, route_ref, pos)
    except Exception:                                    # noqa: BLE001
        side_off = None
    # 侧向门的阈值随模式收紧：strict 要求车道中心至少在路线右侧 0.2 m；
    # map 模式允许 0.4 m 的来袭侧容差（弯心），auto/sensor 允许 0.5 m。
    side_limit = (-0.2 if strict_lane
                  else (-0.4 if lane_mode == "map" else 0.5))
    meta = {
        "lane_side_off_m": (None if side_off is None
                            else round(float(side_off), 3)),
        "lane_side_limit_m": float(side_limit),
        # 配对到底产出了什么：车道宽的配对是"一条车道"，路宽的配对是"整条路"。
        "pair_width_m": (round(float(getattr(lane_frame, "width", 0.0)
                                     or 0.0), 2)
                         if lane_frame is not None else None),
        "pair_span_m": (round(float(getattr(lane_frame, "span_m", 0.0)
                                    or 0.0), 2)
                        if lane_frame is not None else None),
        "pair_conf": (round(float(getattr(lane_frame, "confidence", 0.0)
                                  or 0.0), 3)
                      if lane_frame is not None else None),
        "pair_paired": int(bool(getattr(lane_frame, "paired", False))),
        "pair_sources": list(getattr(lane_frame, "sources", ()) or ()),
    }
    if not lane_heading_ok(route_ref, lane_ref, pos, heading,
                           max_yaw_deg=LANE_HEADING_MAX_YAW_DEG):
        rejected = True
    elif lane_mode != "sensor" and not lane_route_turn_ok(
            route_ref, pos,
            look_m=LANE_ROUTE_TURN_LOOK_M,
            max_turn_deg=LANE_ROUTE_TURN_MAX_DEG):
        rejected = True
        corner_bad = True
    elif not lane_side_ok(lane_ref, route_ref, pos, left_max_m=side_limit):
        rejected = True
        side_bad = True
    reason = ("side" if side_bad
              else "corner" if corner_bad else "heading") if rejected else None
    return {"rejected": rejected, "reason": reason, "side_bad": side_bad,
            "corner_bad": corner_bad, "meta": meta}


def _sensor_lane_within_map(lane_ref, map_center, pos, *,
                            max_center_m: float):
    """近场（0.5–8 m）感知中心与**地图本车道中心**是否足够接近。

    2026-10-20 从 ``select_lane_reference`` 原样搬出，语义不变：

    * ``True``：一致，感知车道可以领；
    * ``False``：不一致，调用方降级到地图本车道（并把 src 改回 map）；
    * ``None``：近场点不足 3 个，**判断不了**——搬移前在这种情况下
      既不降级也不报警，这里保持"不下结论"，由调用方维持原状。

    距离用点到地图中心折线的最小距离，取中位数再与阈值比；阈值随模式：
    auto 1.5 m（感知领但要求与地图一致）、sensor 2.5 m（允许过弯时偏离）。
    计算异常照旧抛给调用方——那里有 ``_warn``，在这里吞掉会丢诊断。
    """
    lane_pts = np.asarray(lane_ref[:, :2], dtype=float)
    center_pts = np.asarray(map_center[:, :2], dtype=float)
    ego = np.asarray(pos[:2], dtype=float)
    dist = np.linalg.norm(lane_pts - ego[None, :], axis=1)
    near = np.flatnonzero((dist >= 0.5) & (dist <= 8.0))
    if len(near) < 3:
        return None
    gaps = np.linalg.norm(
        lane_pts[near][:, None, :] - center_pts[None, :, :], axis=2).min(axis=1)
    return bool(float(np.median(gaps)) <= float(max_center_m))


def _lane_width_from_map_edges(ml, mr, *, default_m: float,
                               warn=None) -> float:
    """地图车道左右**实测**边线宽度；取不到就退回 ``default_m``。

    2026-10-20 从 ``select_lane_reference`` 原样搬出。用最近点距离而不是
    逐元素配对：两条边线的弧长参数化可以不同（``map_lane_edges`` 只重采样
    中心线、保留原始边线间距，圆角插值还会插入额外点），逐元素配对在弯道上
    会读出偏大的宽度（实测）。

    读失败时**保留原来的两条行为**：记一条 ``lane_width`` 告警，然后退回
    ``default_m``（``map_lane_width_m``，0 则 ``LANE_WIDTH_DEFAULT_M``）——
    宽度不是一个可以"猜"的量，但缺失时用合同宽度比用 0 安全。
    """
    width = 0.0
    try:
        from beamng_autopilot.planning.geometry import (
            polyline_point_distances)
        left = np.asarray(ml, dtype=float)[:, :2]
        right = np.asarray(mr, dtype=float)[:, :2]
        distances = polyline_point_distances(left, right)
        finite = distances[np.isfinite(distances)]
        if finite.size:
            width = float(np.median(finite))
    except Exception as exc:                         # noqa: BLE001
        _warn(warn, "lane_width", f"lane width read failed: {exc}")
    return width if width > 0.0 else float(
        default_m or LANE_WIDTH_DEFAULT_M)


def _trusted_single_paint(lane_frame, lane_ref, *, sensor_paired: bool) -> bool:
    """剩**一条**感知漆线时，它是否可信到能推出本车道（strict 模式）。

    2026-10-20 从 ``select_lane_reference`` 原样搬出。配对/融合几何已经拒掉
    了远距、错侧与未知标线，所以这里只回答"这条单边够不够格"：

    * 必须来自 **vision**（LiDAR 单边推不出车道）；
    * 不能是已配对的车道（配对有自己的路径）；
    * 明确画线类型（solid/dashed/thin）门槛 0.35——短视场里保住 strict 控制；
      其余类型仍要 0.50；
    * 必须有至少 3 个点的中心折线可推。

    返回 False 时调用方不得从缓存里"复活"单边（见 ``_single_vision`` 的撤销）。
    """
    if lane_frame is None or sensor_paired:
        return False
    sources = tuple(getattr(lane_frame, "sources", ()) or ())
    if "vision" not in sources:
        return False
    kinds = {getattr(lane_frame, "left_kind", None),
             getattr(lane_frame, "right_kind", None)}
    min_conf = 0.35 if kinds & {"solid", "dashed", "thin"} else 0.50
    confidence = float(getattr(lane_frame, "confidence", 0.0) or 0.0)
    return bool(confidence >= min_conf
                and lane_ref is not None and len(lane_ref) >= 3)


def _reference_meta(gate_meta, *, divider_dbg, divider, rejected,
                    reject_reason, src_published, lane_src_sel,
                    strict_lane) -> dict:
    """``LaneReference.meta`` 的装配：来源标签 + 拒绝原因 + divider 证据。

    2026-10-20 从 ``select_lane_reference`` 原样搬出，纯字典构造、无副作用
    （入参 ``gate_meta`` 被复制，调用方后续对 meta 的增改不会回写）。

    ``lane_src`` 是决策的**纯函数**（``_LABEL_BY_SRC``）：遥测里的标签不可能
    与真正把关的来源不一致；未发布来源时落到 ``bev/route``——那正是"既没有
    地图车道也没有感知车道、只有 BEV 自由空间中心"的旧标签。
    """
    meta: dict = dict(gate_meta)
    if divider_dbg:
        meta["lane_divider"] = divider_dbg
        if divider is not None:
            meta["lane_from"] = "divider_right_shift"
    if rejected and reject_reason is not None:
        meta["lane_reject_reason"] = reject_reason
    if src_published:
        meta["lane_src_sel"] = lane_src_sel
    meta["lane_src"] = _LABEL_BY_SRC.get(lane_src_sel, SRC_BEV_ROUTE)
    if strict_lane:
        meta["lane_strict"] = 1
    return meta


def _bev_whole_road_center(grid, pos, heading, *, strict_lane: bool):
    """没有导航路线时的**整条路**中心（BEV 自由走廊），兜底用。

    2026-10-20 从 ``select_lane_reference`` 原样搬出。只在**非 strict** 且真
    有 BEV 网格时给出：整条路中心不是本车道，双向路上它就是中心线，所以
    strict 绝不用它（这是 "never reach the planner's lateral reference" 那
    条约束的执行点）。点不足 4 个也算没有——两三个点的折线推不出横向参考。
    """
    if strict_lane or grid is None:
        return None
    ref = bev_drivable_center(grid, pos, heading)
    if ref is None or len(ref) < 4:
        return None
    return ref


def _paved_boundary_usable(paved_fallback: bool, paved_ref) -> bool:
    """无标线但有铺装时，铺装边界能否当本车道权威（信任顺序第 3 级）。

    2026-10-20 从 ``select_lane_reference`` 原样搬出。AGENTS.md「驾驶约束」
    固定了 trust order：**标线 >> 路面边界 = 护墙 = 围栏**，所以这一级排在
    感知标线之下、任何地图几何之上。

    候选本身已经 abstain 过一次（铺装右缘必须被观测到、跨度是道路量级、
    铺装沿自车轨迹可见），所以这一层**不能凭空在草地上造车道**；它的边线
    会作为本 tick 的**硬边界**发布——这正是"禁止将车辆驾驶到土和草上"的
    执行点。``paved_fallback`` 默认关，开启才考虑。
    """
    if not paved_fallback or paved_ref is None:
        return False
    center = getattr(paved_ref, "center", None)
    return center is not None and len(center) >= 3


def _paved_geometry(paved_ref) -> tuple:
    """铺装车道的 ``(center, left, right, width)``；宽度取实测 ``span_m``。

    只取前两列：``PavedLane`` 的中心折线可能带 z，而横向参考是二维几何。
    宽度缺失(``span_m`` 为 0/None)就按 0 返回，由调用方决定怎么用——
    这里不替它猜一个合同宽度。
    """
    return (np.asarray(paved_ref.center, dtype=float)[:, :2],
            np.asarray(paved_ref.left, dtype=float)[:, :2],
            np.asarray(paved_ref.right, dtype=float)[:, :2],
            float(getattr(paved_ref, "span_m", 0.0) or 0.0))


def _divider_lane(markings, pos, heading, route_ref, grid, *,
                  debug: dict) -> tuple:
    """把路线上的漆线当**分隔线**，返回它右侧的半个车道 ``(center, left)``。

    2026-10-20 从 ``select_lane_reference`` 原样搬出。场景：车正压在（或紧贴）
    道路中心线上，所有整路读法都被侧向门拒绝，strict 车永远离不开它卡住的
    那条线。一条**在路线上的**长实/虚线就是分隔线——本车道在它右侧半个车道
    处，且该中心还要自己过侧向门、落在被观测到的铺装上（后两条仍由调用方
    执行，这里只负责"找分隔线 + 右移"）。

    返回 ``(divider_or_None, debug_dict)``：失败时 debug 被换成
    ``{"reason": ...}``（与搬移前一致），调用方用它写进 meta 供复核。
    """
    dbg: dict = dict(debug)
    try:
        divider = own_lane_beside_divider(
            markings, pos, heading, route_ref, grid,
            lane_half_m=0.5 * float(LANE_WIDTH_DEFAULT_M),
            debug=dbg)
    except Exception as exc:                         # noqa: BLE001
        return None, {"reason": f"divider check failed: {exc}"}
    return divider, dbg


def _enforce_publishable(ref_out: LaneReference) -> LaneReference:
    """末道 fail-closed：``sensor`` 标签但没有可用中心 → 报告 unavailable。

    2026-10-20 从 ``select_lane_reference`` 原样搬出。它**独立于哪条分支**
    造成了不一致（plan §3.3-1）：一个带 ``sensor`` 标签却没有几何的决策，
    必须报成 unavailable 而不是"带标签的空决策"；几何字段一起清掉，任何
    消费者都不能拿半个决定去转向。

    注意它**不**授权"存在但已被撤销的中心"——撤销的那一步已经把几何清了，
    两者不可能互相矛盾（``LaneReference.publishable_sensor`` 的注释）。
    """
    if ref_out.publishable_sensor:
        return ref_out
    meta = ref_out.meta
    meta["lane_ref_invariant"] = (
        "sensor source with empty/invalid centre -> unavailable")
    ref_out.src = SRC_UNAVAILABLE
    ref_out.src_published = True
    ref_out.map_lane = None
    ref_out.center = None
    ref_out.left = None
    ref_out.right = None
    ref_out.width = 0.0
    ref_out.boundaries = False
    meta["lane_src_sel"] = SRC_UNAVAILABLE
    meta["lane_src"] = _LABEL_BY_SRC.get(SRC_UNAVAILABLE, SRC_BEV_ROUTE)
    return ref_out


def _apply_geometry_gate(*, enabled: bool, center, lane_frame, lane_src_sel,
                         grid, corridor, prev_ref, meta: dict) -> tuple:
    """形状级撤销（plan E4，默认关）：返回 ``(center, src, revoked)``。

    ``select_lane_reference`` 上面那些门都是**逐样本**的静态门；这里拒的是
    **形状**不对的车道——宽度来回摆、边界比路还弯、车道夹成一条缝、中心离开
    可行驶证据或离开观测到的 LiDAR 走廊、以及相对上一帧跳变。只作用于
    **感知来源**（``sensor`` / ``paved``；地图车道的几何不是感知主张），
    并且**只能撤销、不能放宽**。默认关闭（``LANE_GEOM_ENABLED``）。

    撤销时调用方还要清 ``boundaries`` / ``lane_width`` / ``map_lane``——
    那三个量属于最终构造，不属于判定，所以留在调用方。

    与搬移前的一处**有意的偏差**：原实现先把车道清空、再去拼接
    ``"geom:" + ",".join(reasons)``，万一拼接抛异常就会"已经撤销了但记成
    检查器错误"。这里先拼字符串、后撤销：拼接失败时决策保持不变，
    只记 ``lane_geom_error``——对 fail-closed 语义更安全，也更可解释。
    """
    if not (enabled and center is not None and len(center) >= 4
            and lane_src_sel in (SRC_SENSOR, SRC_PAVED)):
        return center, lane_src_sel, False
    try:
        from .geometry_check import check_lane_geometry
        left = getattr(lane_frame, "left", None)
        right = getattr(lane_frame, "right", None)
        report = check_lane_geometry(
            center=center,
            left=(None if left is None
                  else np.asarray(left, dtype=float)[:, :2]),
            right=(None if right is None
                   else np.asarray(right, dtype=float)[:, :2]),
            drivable=grid, corridor=corridor, prev_ref=prev_ref)
        meta["lane_geom"] = report.digest()
        if report.ok:
            return center, lane_src_sel, False
        reason = "geom:" + ",".join(report.reasons)
    except Exception as exc:                         # noqa: BLE001
        meta["lane_geom_error"] = str(exc)
        return center, lane_src_sel, False
    meta["lane_reject_reason"] = reason
    meta["lane_src_sel"] = SRC_UNAVAILABLE
    meta["lane_src"] = _LABEL_BY_SRC.get(SRC_UNAVAILABLE, SRC_BEV_ROUTE)
    return None, SRC_UNAVAILABLE, True


def _publishes_hard_boundaries(lane_frame, lane_src_sel, map_lane,
                               divider) -> bool:
    """本 tick 的左右边线能不能当**硬边界**（no-cross 权威）。

    2026-10-20 从 ``select_lane_reference`` 原样搬出。只有**真的被感知到**的
    边才算：双侧配对的感知车道、观测到的铺装边（``SRC_PAVED``）、分隔线推出
    的那条车道（分隔线本身就是检测到的漆线），或者**地图先验自己就是参考**
    的时候。单边镜像不是物理边——no-cross 规则不能拿一条推出来的线去拦车。
    """
    return bool(
        (lane_frame is not None and getattr(lane_frame, "paired", False))
        or lane_src_sel == SRC_PAVED
        or map_lane is not None
        or (divider is not None and lane_src_sel == SRC_SENSOR))


def select_lane_reference(
    *,
    lane_frame,
    pos,
    heading: float,
    route_ref=None,
    has_nav_route: bool = False,
    map_lane_override=None,
    grid=None,
    lane_mode: str = "map",
    strict_sensor: bool = False,
    lane_consistency_m: float = 1.5,
    lane_consistency_sensor_m: float = 2.5,
    map_lane_width_m: float = LANE_WIDTH_DEFAULT_M,
    corridor_fallback: bool = False,
    corridor_max_width_m: float = CORRIDOR_LANE_MAX_WIDTH_M,
    paved_ref=None,
    paved_fallback: bool = False,
    warn=None,
    prev_ref=None,
    corridor=None,
    markings=None,
    tick_id: int = 0,
) -> LaneReference:
    """Decide which lane geometry may steer the car this tick.

    Order of trust (strongest first):

    1. a PAIRED sensor lane (vision markings / LiDAR corridor / fusion)
       that passes the heading, corner and side gates;
    2. a trusted SINGLE painted boundary, whose missing side is inferred
       from the painted-line lane-width contract - strict mode's
       perception-only fallback;
    3. the PAVED BOUNDARY (``paved_ref``, from the soil-stripped road
       mask): a road with no usable marking at all.  AGENTS.md
       「驾驶约束」 fixes the order marking >> pavement edge = guardrail =
       fence, so this ranks below 1/2 and above every map-based level;
    4. the map-prior own lane built from the nav route - **legacy
       non-strict mode only**;
    5. the BEV drivable-space centre - a whole-road centre used only when
       there is no nav route at all (probes / unit stubs), never in
       strict mode.

    Strict mode (``lane_mode == "sensor"`` and ``strict_sensor``) stops at
    (1)-(3): no map geometry is even built, and a tick with none of them
    returns ``src = "perception-unavailable"`` with no centre at all, so
    the caller can only fail closed.
    """
    strict_lane = bool(lane_mode == "sensor" and strict_sensor)
    map_lane = None if strict_lane else map_lane_override

    # Sensor lane seeding: the frame's own centre and (only for a REAL
    # two-sided detection) its hard boundaries.  A single-edge mirror is
    # not a physical edge the no-cross rule may enforce.
    sensor_paired = bool(lane_frame is not None
                         and getattr(lane_frame, "paired", False))
    lane_ref = None
    lane_left = None
    lane_right = None
    lane_width = 0.0
    if lane_frame is not None:
        if getattr(lane_frame, "center", None) is not None \
                and len(lane_frame.center) >= 2:
            lane_ref = np.asarray(lane_frame.center, dtype=float)[:, :2]
        if sensor_paired:
            lane_left = getattr(lane_frame, "left", None)
            lane_right = getattr(lane_frame, "right", None)
            lane_width = float(getattr(lane_frame, "width", 0.0) or 0.0)

    # Map-prior own-lane fallback: when NO sensor lane could be paired
    # this frame, derive the ego lane from the nav route - half a lane
    # width RIGHT of the road centreline (right-hand traffic), with the
    # centreline as the hard left boundary and the road's right edge as
    # the hard right boundary.  Without this the planner tracked the road
    # CENTRE line whenever ``lane_paired=0`` (town runs 2026-08-22:
    # g8/g10/g12 rode the centre line end to end, and the no-cross rule
    # had no boundaries to enforce).  The road-graph route starts at the
    # nearest road node, so it also anchors the car's own lane even when
    # the ego has drifted off the A* polyline.
    #
    # Built whenever a nav route exists: it is both the fallback for a
    # missing sensor lane AND the override when a sensor lane heads into
    # a different road at a junction (heading gate below).  A caller may
    # supply ``map_lane_override`` (real road-edge own-lane window from
    # map_lane_edges); when absent the synthetic map lane is built.
    if has_nav_route and map_lane is None and not strict_lane:
        try:
            from beamng_autopilot.planning.local_route import map_lane_local
            map_lane = map_lane_local(route_ref, pos, heading)
        except Exception as exc:
            map_lane = None
            _warn(warn, "map_lane_local", f"map-lane builder failed: {exc}")

    # Heading gate: a PAIRED sensor lane is only trustworthy when its
    # near-ahead direction agrees with the nav route (or the ego
    # heading).  At a junction the vision/LiDAR pairing can lock onto a
    # DIFFERENT road whose corridor reads clear - following it drives the
    # car off the navigational route (town run 2026-08-22: the paired
    # lane headed into the side road and the car left the road and
    # wedged).  Reject the whole sensor lane (centre + hard boundaries)
    # and fall back to the map-prior own lane.
    #
    # Gate ANY sensor lane (paired or not) against the map-prior own
    # lane: an unpaired vision/LiDAR corridor is often the whole-road
    # centre, which on a two-way road sits on the oncoming side of the
    # own lane.  Without the gate that centre was fed to the planner as
    # the lane reference, the chosen path parked 3.6 m off it and the
    # safety monitor declared "path near lane edge" -> src=none stop at
    # (734.9,753.4) mountain run 2026-08-23.
    #
    # Gate strictness follows lane_mode: map keeps all three gates; auto
    # keeps heading+corner but relaxes the side gate; sensor
    # (perception-led) keeps only heading+side so the sensor lane can
    # lead through corners - the map prior still supplies the hard lane
    # boundaries below (never the sensor's own flickering edges).
    lane_rejected = False
    reject_reason = None
    gate_meta: dict = {}
    # Divider fallback state (strict mode only; declared here so the
    # boundary/meta assembly below can read it on every path).
    _divider = None
    _divider_dbg: dict = {}
    if lane_frame is not None and (map_lane is not None or strict_lane):
        try:
            verdict = _lane_gate_verdict(
                lane_frame=lane_frame, lane_ref=lane_ref, route_ref=route_ref,
                pos=pos, heading=heading, lane_mode=lane_mode,
                strict_lane=strict_lane)
            gate_meta = verdict["meta"]
            if verdict["rejected"]:
                lane_rejected = True
                reject_reason = str(verdict["reason"])
                # 整条感知车道都不可信：中心与（可能错误的）硬边界一起丢弃，
                # 否则 no-cross 规则会拿一条错边去拦车。
                sensor_paired = False
                lane_ref = None
                lane_left = None
                lane_right = None
                lane_width = 0.0
        except Exception as exc:
            _warn(warn, "lane_gates", f"lane gate checks failed: {exc}")

    lane_src_sel = ""
    src_published = False
    if map_lane is not None and not strict_lane:
        mc, ml, mr = map_lane
        lane_src_sel = SRC_MAP
        if sensor_paired and lane_ref is not None and len(lane_ref) >= 3:
            lane_src_sel = SRC_SENSOR
        if lane_mode in ("auto", "sensor"):
            # Perception-led modes: the MAP boundaries are the hard
            # guard-rail (centreline = no-cross, real right edge = no
            # off-road).  The sensor's own paired edges flicker / jump
            # frame to frame and must never be the hard rule.
            lane_left = ml
            lane_right = mr
            if sensor_paired and lane_ref is not None:
                # Only a PAIRED sensor lane may lead, and only when it
                # agrees laterally with the map-prior own lane.  An
                # UNPAIRED sensor centre is the whole-road free corridor
                # - on a two-way road that IS the centre line, never the
                # ego lane (fsd sensor run 2026-08-29: an unpaired centre
                # at the end zone sat 2.5 m off and stopped the car;
                # corner reads steered it off the road edge).  auto =
                # 1.5 m gate (perception-led but map-consistent); sensor
                # = looser 2.5 m gate so the perception lane can lead
                # through corners.
                try:
                    within = _sensor_lane_within_map(
                        lane_ref, mc, pos,
                        max_center_m=(lane_consistency_sensor_m
                                      if lane_mode == "sensor"
                                      else lane_consistency_m))
                    if within is False:
                        lane_ref = mc
                        lane_src_sel = SRC_MAP
                except Exception as exc:
                    lane_ref = mc
                    lane_src_sel = SRC_MAP
                    _warn(warn, "lane_consistency",
                          f"lane consistency check failed: {exc}")
            else:
                # Unpaired sensor centre / no sensor lane: keep the
                # map-prior own-lane centre as the reference.  The map
                # boundaries are already the hard guard-rail.
                lane_ref = mc
                lane_src_sel = SRC_MAP
        else:
            # An unpaired sensor centre is often the whole-road centre
            # (LiDAR free corridor / single-edge mirror) - never trust it
            # as the lane-keep reference over the map-prior own lane.
            if not sensor_paired:
                lane_ref = mc
            if lane_left is None or lane_right is None:
                lane_left = ml if lane_left is None else lane_left
                lane_right = mr if lane_right is None else lane_right
        src_published = True
        if lane_width <= 0.0:
            # Real road-edge width when the map lane comes from DecalRoad
            # edges (median left-right distance); fall back to the fixed
            # map-prior lane width otherwise.
            lane_width = _lane_width_from_map_edges(
                ml, mr, default_m=map_lane_width_m, warn=warn)

    # FSD realism (strict sensor): the lane-keep reference and hard
    # boundaries must come from PERCEPTION only (docs/fsd_realism.md §4).
    # A PAIRED sensor lane is ideal, but a TRUSTED SINGLE PAINTED
    # boundary is also a perception lane: the missing side is inferred
    # from the painted-line lane-width contract, never from the map.
    # Ambiguous/no-vision reads still stop.
    if strict_lane:
        _single_vision = _trusted_single_paint(
            lane_frame, lane_ref, sensor_paired=sensor_paired)
        # Painted-divider fallback: the car stands on (or beside) the road
        # centreline, so every whole-road read is refused by the side gate
        # and a strict car could never leave the line it is stuck on.  A
        # long solid/dashed marking ON the route is the divider; the own
        # lane is half a lane to its right, and that centre must itself
        # pass the side gate and land on observed drivable pavement.
        if not (sensor_paired and lane_ref is not None and len(lane_ref) >= 3) \
                and not _single_vision:
            _divider, _divider_dbg = _divider_lane(
                markings, pos, heading, route_ref, grid,
                debug=_divider_dbg)
            if _divider is not None:
                lane_ref = _divider[0]
                lane_left = _divider[1]
                lane_width = float(LANE_WIDTH_DEFAULT_M)
        # On-pavement gate for EVERY accepted perception lane.  The side
        # gate answers "which side of the road" but says nothing about
        # whether the geometry is on the road at all: measured 2026-09-21
        # a paired vision lane of width 5.24 m was accepted with its
        # centre 4.22 m right of the road centre - past the right edge of
        # a ~7 m road - which would have steered the car onto the
        # shoulder.  The centre must lie where the sensors OBSERVED
        # drivable surface.  No grid / no observed evidence abstains
        # (recorded), and the check can only withdraw.
        _drv_detail: dict = {}
        _drv_frac, _drv_n = _drivable_fraction(
            lane_ref, grid, pos=pos, detail=_drv_detail,
            ignore_obstacle=LANE_PAVEMENT_GATE_NO_OCC)
        _drv_checked = bool(_drv_n >= 3)
        if _drv_checked and _drv_frac < DIVIDER_DRIVABLE_MIN_FRAC:
            _warn(warn, "lane_off_drivable",
                  f"sensor lane centre off observed pavement "
                  f"(frac={_drv_frac:.2f}, n={_drv_n})")
            sensor_paired = False
            lane_ref = None
            lane_left = None
            lane_right = None
            lane_width = 0.0
            gate_meta["lane_drivable"] = {
                "frac": round(float(_drv_frac), 3), "n": int(_drv_n),
                "reason": "centre off observed pavement",
                **_drv_detail}
            # The revocation must kill the pre-gate booleans too.  Both
            # single-edge and divider fallbacks below re-publish the SAME
            # frame's geometry, so leaving them set re-labelled a revoked
            # candidate as ``sensor`` with an empty centre (plan §2.2-B,
            # §3.3-2: never restore a source from a cached pre-veto flag).
            _single_vision = False
            _divider = None
        elif lane_ref is not None:
            gate_meta["lane_drivable"] = {
                "frac": (round(float(_drv_frac), 3) if _drv_checked else None),
                "n": int(_drv_n),
                "reason": ("" if _drv_checked
                           else "not enough observed samples"),
                **_drv_detail}
        if sensor_paired and lane_ref is not None and len(lane_ref) >= 3:
            lane_src_sel = SRC_SENSOR
        elif _single_vision:
            lane_src_sel = SRC_SENSOR
            lane_left = getattr(lane_frame, "left", None)
            lane_right = getattr(lane_frame, "right", None)
            lane_width = float(getattr(lane_frame, "width", 0.0) or 0.0)
        elif _divider is not None:
            lane_src_sel = SRC_SENSOR
            lane_right = None
        elif _paved_boundary_usable(paved_fallback, paved_ref):
            # NO usable marking on a road that IS paved: the pavement
            # boundary is the authority (AGENTS.md「驾驶约束」: 标线 >>
            # 路面边界 = 护墙 = 围栏).  Its edges are published as the
            # tick's HARD boundaries, which is what keeps the car on the
            # pavement ("禁止将车辆驾驶到土和草上").
            lane_ref, lane_left, lane_right, lane_width = \
                _paved_geometry(paved_ref)
            lane_src_sel = SRC_PAVED
        else:
            lane_ref = None
            lane_left = None
            lane_right = None
            lane_width = 0.0
            lane_src_sel = SRC_UNAVAILABLE
            map_lane = None
            # Pairing-free PERCEPTION fallback, flag-gated (default OFF):
            # on a two-lane road the free corridor's right edge + half a
            # lane IS the ego-lane centre - sensor-derived, never map.
            # Roads wider than the validated gate abstain, so strict mode
            # still fails closed exactly where the assumption breaks.
            if corridor_fallback and grid is not None:
                _corridor = bev_corridor_lane_center(
                    grid, pos, heading, max_width_m=corridor_max_width_m)
                if _corridor is not None and len(_corridor) >= 3:
                    lane_ref = _corridor
                    lane_src_sel = SRC_CORRIDOR
        if lane_src_sel in (SRC_SENSOR, SRC_PAVED):
            map_lane = None
        src_published = True

    # BEV drivable-space fallback when there is NO nav route to derive a
    # map-prior own lane from (standalone probes / unit stubs): the
    # lateral centre of the FREE corridor.  In real nav runs the map lane
    # (or a paired sensor lane) takes priority, so this whole-road centre
    # never reaches the planner's lateral reference there.  Strict sensor
    # mode skips it too: the whole-road centre is not the ego lane.
    if lane_ref is None or len(lane_ref) < 4:
        bev_ref = _bev_whole_road_center(grid, pos, heading,
                                         strict_lane=strict_lane)
        if bev_ref is not None:
            lane_ref = bev_ref

    center = (np.asarray(lane_ref, dtype=float)
              if lane_ref is not None else None)
    # Only a REAL sensor detection provides hard lane boundaries: a
    # two-sided lane pair, the observed pavement edges (SRC_PAVED), the
    # divider's own painted edge, or the map prior when IT is the
    # reference.  A single-edge mirror is not a physical edge the no-cross
    # rule may enforce (see _publishes_hard_boundaries).
    boundaries = _publishes_hard_boundaries(
        lane_frame, lane_src_sel, map_lane, _divider)
    meta = _reference_meta(
        gate_meta, divider_dbg=_divider_dbg, divider=_divider,
        rejected=lane_rejected, reject_reason=reject_reason,
        src_published=src_published, lane_src_sel=lane_src_sel,
        strict_lane=strict_lane)
    # Geometric consistency (plan E4, opt-in): static gates above accept a
    # lane sample by sample; this refuses one whose SHAPE is wrong.  It can
    # only withdraw, never relax (see _apply_geometry_gate).
    center, lane_src_sel, _geom_revoked = _apply_geometry_gate(
        enabled=LANE_GEOM_ENABLED, center=center, lane_frame=lane_frame,
        lane_src_sel=lane_src_sel, grid=grid, corridor=corridor,
        prev_ref=prev_ref, meta=meta)
    if _geom_revoked:
        boundaries = False
        lane_width = 0.0
        map_lane = None
    ref_out = LaneReference(
        center=center,
        left=lane_left,
        right=lane_right,
        width=float(lane_width),
        src=lane_src_sel,
        src_published=src_published,
        map_lane=map_lane,
        rejected=lane_rejected,
        reject_reason=reject_reason,
        strict=strict_lane,
        boundaries=boundaries,
        frame_used=bool(lane_frame is not None),
        # Evidence provenance, read from the frame that was actually
        # accepted (plan §3.3-6): ``paired`` alone is not enough - the
        # painted centre line sets it while its right edge is a width
        # prior, so ``two_sided_measured`` is the consumer-facing flag.
        two_sided=bool(lane_frame is not None
                       and getattr(lane_frame, "two_sided_measured", False)),
        fresh_obs=bool(
            lane_frame is not None and tick_id
            and int(getattr(lane_frame, "obs_seq", 0) or 0) == int(tick_id)),
        inferred=bool(lane_frame is not None
                      and getattr(lane_frame, "inferred", False)),
        meta=meta,
    )
    return _enforce_publishable(ref_out)
