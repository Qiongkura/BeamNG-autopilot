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


def _drivable_fraction(center, grid) -> tuple[float, int]:
    """(fraction of sampled centre points on observed-drivable, samples)."""
    if grid is None or center is None or len(center) < 2:
        return 0.0, 0
    drv = getattr(grid, "drivable", None)
    occ = getattr(grid, "obstacle", None)
    obs = getattr(grid, "observed", None)
    if drv is None or not getattr(drv, "any", lambda: False)():
        return 0.0, 0
    pts = np.asarray(center, dtype=float)[:, :2]
    rows, cols, ok = [], [], []
    for x, y in pts:
        cell = grid.world_to_cell(float(x), float(y))
        if cell is None:
            continue
        rows.append(int(cell[0]))
        cols.append(int(cell[1]))
        ok.append(True)
    if not ok:
        return 0.0, 0
    rr = np.asarray(rows)
    cc = np.asarray(cols)
    if obs is not None and getattr(obs, "size", 0) and obs.shape == drv.shape:
        seen = obs[rr, cc] > 0
    else:
        # No observed layer: the drivable mask itself is the evidence.
        seen = np.ones(len(rr), dtype=bool)
    if int(seen.sum()) < 3:
        return 0.0, int(seen.sum())
    good = drv[rr[seen], cc[seen]] > 0
    if occ is not None and getattr(occ, "size", 0) and occ.shape == drv.shape:
        good = np.logical_and(good, occ[rr[seen], cc[seen]] == 0)
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
            from beamng_autopilot.planning.arbiter import (
                lane_heading_ok, lane_route_turn_ok, lane_side_ok,
                lane_side_offset_m)
            # Bearing gate: the lane must HEAD the same way as the route
            # (junction pairing onto a side road is rejected).
            side_bad = False
            corner_bad = False
            # Where the sensor lane centre sits relative to the route, and
            # the threshold the side gate will apply.  A refusal is only
            # reviewable with the number that caused it: without it "the
            # gate said no" cannot be told apart from a pairing error.
            try:
                _side_off = lane_side_offset_m(lane_ref, route_ref, pos)
            except Exception:
                _side_off = None
            _side_limit = (-0.2 if strict_lane
                           else (-0.4 if lane_mode == "map" else 0.5))
            gate_meta = {
                "lane_side_off_m": (None if _side_off is None
                                    else round(float(_side_off), 3)),
                "lane_side_limit_m": float(_side_limit),
                # What the pairing actually produced: a lane-width pair is a
                # lane read, a road-width one is the whole roadway.
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
                lane_rejected = True
            # CORNER gate: at a real turn in the nav route the map-prior
            # own lane is the authority - the sensor lane reads the
            # corner wide and steers the car off the line (see
            # LANE_ROUTE_TURN_MAX_DEG).  The 35 deg heading gate alone
            # does not catch it: a wide corner read can still be within
            # 35 deg of the route.  Only the map mode keeps this;
            # perception-led modes let the sensor lane lead through
            # corners (the map guard-rail still stops a real crossing).
            elif lane_mode != "sensor" and not lane_route_turn_ok(
                    route_ref, pos,
                    look_m=LANE_ROUTE_TURN_LOOK_M,
                    max_turn_deg=LANE_ROUTE_TURN_MAX_DEG):
                lane_rejected = True
                corner_bad = True
            # SIDE gate (safety net, NOT a navigator preference): the lane
            # centre must sit clearly RIGHT of the road centreline
            # (own lane).  A lane locked onto the ONCOMING lane passes
            # the bearing gate (same direction) but still steers the car
            # over the centre line (town runs 2026-08-22: the car rode
            # the centre/oncoming lane end to end with lane_src=sensor).
            # A lane sitting ON the centreline is the WHOLE-ROAD free
            # corridor, not the own lane - trusting it parks the car on
            # the centre line and the switch to the map-prior own lane
            # then forces a 2-3 m over-correction that swings it off the
            # road edge (mountain run 2026-08-27 run_fix31: after the
            # junction the car rode the centre line, then over-corrected
            # right and wedged at (741.2,745.7)).
            #
            # This gate is ALWAYS on - it is the last wall between an
            # unpaired sensor centre (often the whole-road corridor on a
            # two-way road) and the planner.  Strict perception
            # (``lane_mode=sensor + strict_sensor``) tightens it to
            # -0.2 m: the lane centre must sit at least 0.2 m RIGHT of
            # the route (own side only).  ``off=0`` (lane sitting on the
            # road centre line) is rejected - that is the whole-road
            # corridor, not the ego lane, and it is exactly the failure
            # mode the live town run 2026-09-20 exposed (see below).
            # Without strict perception we allow a small oncoming-side
            # read for corner apex (perception-led modes) or the legacy
            # map-prior tolerance (map mode); both still keep the map
            # centreline as the hard no-cross boundary downstream.
            #
            # Regression: live town run 2026-09-20 --strict kept this
            # gate bypassed and the car sat on the road centre line at
            # mean=+1.18 m painted-line lateral (p50=+1.04 m), straddle
            # the divider for 75 of 166 frames and stall on the way to
            # the goal.  See gate_on_3.log and the screenshot in
            # docs/reviews/2026-09-20_strict_centerline.md.
            elif not lane_side_ok(
                    lane_ref, route_ref, pos, left_max_m=_side_limit):
                lane_rejected = True
                side_bad = True
            if lane_rejected:
                reject_reason = ("side" if side_bad
                                 else "corner" if corner_bad else "heading")
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
                    _lr2 = np.asarray(lane_ref[:, :2], dtype=float)
                    _mc2 = np.asarray(mc[:, :2], dtype=float)
                    _p2 = np.asarray(pos[:2], dtype=float)
                    _d2 = np.linalg.norm(_lr2 - _p2[None, :], axis=1)
                    _sel2 = np.flatnonzero((_d2 >= 0.5) & (_d2 <= 8.0))
                    if len(_sel2) >= 3:
                        _dm = np.linalg.norm(
                            _lr2[_sel2][:, None, :]
                            - _mc2[None, :, :], axis=2).min(axis=1)
                        _max_c = (lane_consistency_sensor_m
                                  if lane_mode == "sensor"
                                  else lane_consistency_m)
                        if float(np.median(_dm)) > _max_c:
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
            _w = 0.0
            try:
                from beamng_autopilot.planning.geometry import (
                    polyline_point_distances)
                _ml = np.asarray(ml, dtype=float)[:, :2]
                _mr = np.asarray(mr, dtype=float)[:, :2]
                # the left/right polylines can carry DIFFERENT arc
                # parametrisations (map_lane_edges resamples the
                # centreline but keeps raw edge spacing, and corner
                # interpolation inserts extra points) - element-wise
                # pairing reads an inflated width on any curve.
                # Nearest-point distances are alignment-robust.
                _wa = polyline_point_distances(_ml, _mr)
                _wf = _wa[np.isfinite(_wa)]
                if _wf.size:
                    _w = float(np.median(_wf))
            except Exception as exc:
                _w = 0.0
                _warn(warn, "lane_width",
                      f"lane width read failed: {exc}")
            lane_width = (_w if _w > 0.0
                          else float(map_lane_width_m or LANE_WIDTH_DEFAULT_M))

    # FSD realism (strict sensor): the lane-keep reference and hard
    # boundaries must come from PERCEPTION only (docs/fsd_realism.md §4).
    # A PAIRED sensor lane is ideal, but a TRUSTED SINGLE PAINTED
    # boundary is also a perception lane: the missing side is inferred
    # from the painted-line lane-width contract, never from the map.
    # Ambiguous/no-vision reads still stop.
    if strict_lane:
        _single_conf = float(getattr(lane_frame, "confidence", 0.0) or 0.0) \
            if lane_frame is not None else 0.0
        _single_kinds = {
            getattr(lane_frame, "left_kind", None),
            getattr(lane_frame, "right_kind", None),
        } if lane_frame is not None else set()
        # The pairing/fusion geometry already rejects far, wrong-side and
        # unknown markings.  A near explicit painted edge at 0.35 is enough
        # to keep strict control alive on the short US view; retain 0.50 for
        # generic single-edge frames.
        _single_min_conf = (0.35 if _single_kinds & {"solid", "dashed", "thin"}
                            else 0.50)
        _single_vision = bool(
            lane_frame is not None
            and not sensor_paired
            and "vision" in tuple(getattr(lane_frame, "sources", ()) or ())
            and _single_conf >= _single_min_conf
            and lane_ref is not None and len(lane_ref) >= 3)
        # Painted-divider fallback: the car stands on (or beside) the road
        # centreline, so every whole-road read is refused by the side gate
        # and a strict car could never leave the line it is stuck on.  A
        # long solid/dashed marking ON the route is the divider; the own
        # lane is half a lane to its right, and that centre must itself
        # pass the side gate and land on observed drivable pavement.
        if not (sensor_paired and lane_ref is not None and len(lane_ref) >= 3) \
                and not _single_vision:
            try:
                _divider = own_lane_beside_divider(
                    markings, pos, heading, route_ref, grid,
                    lane_half_m=0.5 * float(LANE_WIDTH_DEFAULT_M),
                    debug=_divider_dbg)
            except Exception as exc:
                _divider = None
                _divider_dbg = {"reason": f"divider check failed: {exc}"}
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
        _drv_frac, _drv_n = _drivable_fraction(lane_ref, grid)
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
                "reason": "centre off observed pavement"}
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
                           else "not enough observed samples")}
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
        elif (paved_fallback and paved_ref is not None
              and getattr(paved_ref, "center", None) is not None
              and len(paved_ref.center) >= 3):
            # NO usable marking on a road that IS paved: the pavement
            # boundary is the authority (AGENTS.md「驾驶约束」: 标线 >>
            # 路面边界 = 护墙 = 围栏).  The candidate already abstained
            # unless the paved RIGHT edge is observed inside the image,
            # the span is road-sized and the pavement is observed along
            # the car's own track, so this branch can not invent a lane on
            # grass; its edges are published as the tick's HARD
            # boundaries, which is what keeps the car on the pavement
            # ("禁止将车辆驾驶到土和草上").
            lane_ref = np.asarray(paved_ref.center, dtype=float)[:, :2]
            lane_left = np.asarray(paved_ref.left, dtype=float)[:, :2]
            lane_right = np.asarray(paved_ref.right, dtype=float)[:, :2]
            lane_width = float(getattr(paved_ref, "span_m", 0.0) or 0.0)
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
        bev_ref = None
        if not strict_lane and grid is not None:
            bev_ref = bev_drivable_center(grid, pos, heading)
        if bev_ref is not None and len(bev_ref) >= 4:
            lane_ref = bev_ref

    center = (np.asarray(lane_ref, dtype=float)
              if lane_ref is not None else None)
    # Only a REAL sensor detection provides hard lane boundaries: a
    # two-sided lane pair, the observed pavement edges (SRC_PAVED), or the
    # map prior when IT is the reference.  A single-edge mirror is not a
    # physical edge the no-cross rule may enforce.
    boundaries = bool(
        (lane_frame is not None and getattr(lane_frame, "paired", False))
        or lane_src_sel == SRC_PAVED
        or map_lane is not None
        # The divider IS a detected painted edge, so the lane it implies
        # publishes it as a real hard boundary (no-cross authority).
        or (_divider is not None and lane_src_sel == SRC_SENSOR))
    meta: dict = dict(gate_meta)
    if _divider_dbg:
        meta["lane_divider"] = _divider_dbg
        if _divider is not None:
            meta["lane_from"] = "divider_right_shift"
    if lane_rejected and reject_reason is not None:
        meta["lane_reject_reason"] = reject_reason
    if src_published:
        meta["lane_src_sel"] = lane_src_sel
    # ``lane_src`` is a pure function of the decision above, so the label
    # in the telemetry can never disagree with the source that gated the
    # planning (see _LABEL_BY_SRC).  An unpublished source means no map
    # lane and no sensor lane were involved - the BEV free-space centre
    # case - which is exactly the legacy "bev/route" label.
    meta["lane_src"] = _LABEL_BY_SRC.get(lane_src_sel, SRC_BEV_ROUTE)
    if strict_lane:
        meta["lane_strict"] = 1
    # Geometric consistency (plan E4, opt-in): the static gates above accept
    # a lane sample by sample; this refuses one whose SHAPE is wrong - a
    # pair whose width swings, a boundary that bends tighter than a road, a
    # "lane" that pinches shut, a centre off the drivable evidence or off
    # the observed LiDAR corridor, or one that jumped since the last frame.
    # It applies to SENSOR-derived lanes only (a map lane's geometry is not
    # a perception claim) and it can only withdraw, never relax.
    if (LANE_GEOM_ENABLED and center is not None and len(center) >= 4
            and lane_src_sel in (SRC_SENSOR, SRC_PAVED)):
        try:
            from .geometry_check import check_lane_geometry
            _left = getattr(lane_frame, "left", None)
            _right = getattr(lane_frame, "right", None)
            _rep = check_lane_geometry(
                center=center,
                left=(None if _left is None
                      else np.asarray(_left, dtype=float)[:, :2]),
                right=(None if _right is None
                       else np.asarray(_right, dtype=float)[:, :2]),
                drivable=grid, corridor=corridor, prev_ref=prev_ref)
            meta["lane_geom"] = _rep.digest()
            if not _rep.ok:
                center = None
                boundaries = False
                lane_width = 0.0
                lane_src_sel = SRC_UNAVAILABLE
                map_lane = None
                meta["lane_reject_reason"] = "geom:" + ",".join(
                    _rep.reasons)
                meta["lane_src_sel"] = SRC_UNAVAILABLE
                meta["lane_src"] = _LABEL_BY_SRC.get(SRC_UNAVAILABLE,
                                                     SRC_BEV_ROUTE)
        except Exception as exc:
            meta["lane_geom_error"] = str(exc)
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
    if not ref_out.publishable_sensor:
        # Fail-closed backstop, independent of which branch produced the
        # inconsistency: a ``sensor`` label without usable centre geometry
        # is reported unavailable instead of steering anything (plan
        # §3.3-1).  The geometry fields are cleared with it so no consumer
        # can act on half a decision.
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
