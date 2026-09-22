"""Separate road-boundary evidence: curb, pavement edge, obstacle entity.

T07 of the round-5 plan.  Three requirements shape this module:

1. **Publish the three kinds SEPARATELY.**  A kerb stone, the edge of the
   pavement and a solid obstacle are different physical objects; averaging
   them into one "boundary" is how a wall becomes a lane edge.  Each
   publisher here returns its own geometry, provenance and confidence, and
   nothing in this module writes into another's output.
2. **Association is a separate, explicit step.**  When two kinds happen to
   describe the same object (a kerb candidates sitting on the pavement
   edge), :func:`associate` reports that - it never merges the geometry.
3. **Thresholds come from the sensor's own geometry**, not from a
   hand-tuned metre constant: the expected spacing between ground returns
   at range ``d`` is ``d * tan(vertical_resolution)``, so a fixed 0.2 m
   step threshold is far too coarse at 5 m and far too fine at 60 m.  See
   :func:`adaptive_step_threshold`.

**Capability record (checked 2026-09-22).**  The plan allows the sliding-
beam method only if the cloud carries a per-beam id / vertical angle /
scan time.  It does not: ``beamngpy``'s LiDAR returns
``{"pointCloud": (N, 3), "colours": (N, 3)}`` and nothing else
(`.venv/Lib/site-packages/beamngpy/sensors/lidar.py`), so the point order
carries no documented beam structure and the per-beam geometric
derivations (Zhang §III, the ``delta_xy``/``delta_z``/``nv`` formulas)
cannot be reproduced.  This module therefore implements the FALLBACK the
plan names: ground height steps and local curvature, with the geometric
spacing used as an adaptive scale rather than as a per-beam prediction.
The gap is recorded, not papered over.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# The sensor's own constants (beamng_autopilot_tech/providers.py): 16 beams
# over +-26.9 deg.  Used to derive the expected ground spacing - NOT as a
# per-beam index, which the cloud does not carry.
LIDAR_VERTICAL_RES = 16
LIDAR_VERTICAL_SPAN_DEG = 2.0 * 26.9
#: Apparent vertical beam pitch, degrees (real sensors of this class are
#: equidistant in angle).
BEAM_PITCH_DEG = LIDAR_VERTICAL_SPAN_DEG / max(1.0, float(LIDAR_VERTICAL_RES))

#: A kerb is ~0.10-0.20 m; the detector looks for a step of at least a
#: fraction of that, scaled by the local sampling.
CURB_HEIGHT_M = 0.15
#: Multiplier on the local ground-point spacing: a step must exceed this
#: many times the expected spacing to be treated as geometry rather than
#: sampling noise.
STEP_K = 1.6
#: Absolute floors/ceilings, so the adaptive rule cannot degenerate.
STEP_MIN_M = 0.05
STEP_MAX_M = 0.35
#: Points further than this are too sparse to judge a step.
MAX_CURB_RANGE_M = 45.0
#: Range-image cell size for the surface-discontinuity test.
RANGE_BIN_M = 0.25
ANGLE_BIN_RAD = math.radians(2.0)
#: Obstacle boxes closer than this to a curb candidate are the same object
#: (the kerb IS the side of the obstacle).
ASSOC_TOL_M = 0.35


def beam_spacing_m(range_m: float,
                   pitch_deg: float = BEAM_PITCH_DEG) -> float:
    """Expected spacing between adjacent ground returns at ``range_m``.

    On flat ground with an equidistant-angle scanner the spacing along the
    surface grows linearly with range: ``d * tan(pitch)``.  This is the
    scale a step threshold has to follow; a constant that works at 20 m is
    off by ~4x at 5 m and by ~3x at 60 m.
    """
    d = float(range_m)
    if not np.isfinite(d) or d <= 0.0:
        return 0.0
    return float(d) * math.tan(math.radians(float(pitch_deg)))


def adaptive_step_threshold(surface_spacing_m: float, *,
                            k: float = STEP_K,
                            floor_m: float = STEP_MIN_M,
                            ceil_m: float = 0.6 * CURB_HEIGHT_M) -> float:
    """Step threshold from the MEASURED along-surface sampling, clamped.

    Two different spacings matter and only one of them is usable:

    * the VERTICAL spacing between rings - ``range * tan(3.36 deg)`` with
      16 beams is 0.29 m at 5 m and 1.17 m at 20 m, i.e. larger than a
      kerb everywhere past a couple of metres, so a threshold based on it
      would miss every kerb (measured on the first version of this file);
    * the ALONG-SURFACE spacing between neighbouring returns on the same
      sweep - millimetres to centimetres, because the azimuthal sampling
      is dense - which is what a height DISCONTINUITY is compared against.

    The clamp's ceiling is half a kerb height: the threshold may never
    grow past the signal it is looking for.
    """
    sp = float(surface_spacing_m)
    if not np.isfinite(sp) or sp < 0.0:
        sp = 0.0
    return float(np.clip(k * sp, float(floor_m), float(ceil_m)))


@dataclass
class BoundaryEvidence:
    """One published kind of boundary evidence (never merged here)."""

    kind: str                     # "curb" | "pavement_edge" | "obstacle"
    points: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=float))
    provenance: str = ""
    confidence: float | None = None
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "n_points": int(len(self.points)),
            "provenance": self.provenance,
            "confidence": (None if self.confidence is None
                           else round(float(self.confidence), 3)),
            "meta": dict(self.meta),
        }


def _local_spacing(points: np.ndarray, idx: int, k: int = 3) -> float:
    """Median distance from point ``idx`` to its k nearest neighbours."""
    if len(points) < 2:
        return 0.0
    d = np.linalg.norm(points - points[idx], axis=1)
    d = np.delete(d, idx)
    if d.size == 0:
        return 0.0
    kk = int(min(max(1, k), d.size))
    return float(np.median(np.sort(d)[:kk]))


# ---------------------------------------------------------------------------
# Line consistency: a boundary is a CURVE beside the car, not a point set.
#
# Measured on a real 8-frame sequence (2026-09-22, italy mountain stretch,
# placed-step replay): the adaptive height-step arm published ~4845
# candidates per frame, spread 2.04 m laterally (p50 PCA residual), 82% of
# them off the pavement, and persistence kept 1156 of them - a detector
# that fires everywhere recurs everywhere, so persistence cannot repair a
# low-precision front end.  Requiring the candidates to form a thin curve
# roughly parallel to the ego path on the SAME frames gives ~12 points,
# 0.036 m residual, 50% of them on the semantic mask edge (15.8% before)
# with the p50 pixel distance to that edge falling from 15.2 to 5.8.
# ---------------------------------------------------------------------------
LINE_FWD_MIN_M = 3.0
LINE_FWD_MAX_M = 30.0
LINE_FWD_BIN_M = 1.0
LINE_LAT_GAP_M = 0.5
#: "Is this cluster thin" tolerance.  A range-adaptive (beam-spacing scaled)
#: tolerance was tried on the same 8 real frames and REJECTED: loosening it
#: merged clusters so the lateral jump between bins exceeded the chain step
#: and four of eight frames published NOTHING, while the last frame mixed
#: several lines into one published offset (measured, see
#: docs/BOUNDARY_EVIDENCE.md).  The fixed tolerance costs usable range
#: (~3-9 m at the 16-beam spacing, recorded as a limit) and keeps the line.
LINE_LAT_SPREAD_M = 0.35
LINE_CHAIN_STEP_M = 0.4
LINE_CHAIN_MIN_BINS = 3
LINE_LAT_MIN_ABS_M = 1.2
#: A lane boundary beside an ego that is ON the road lies within this many
#: metres.  Measured on a real straight stretch (2026-09-22, italy, the
#: wide road at the default spawn): the longest thin chain was a structure
#: 18.9-19.7 m to the LEFT, and publishing it as "the boundary" while the
#: pavement-edge candidates sat 3.6 px from the mask edge was wrong.  The
#: band is an ego-centric relevance window, not a map offset: chains
#: outside it are still recorded, with identity, as ``distant_chains``.
LINE_LAT_MAX_M = 8.0


def _thin_lateral_cluster(lat: np.ndarray, members: list[int]):
    """Median lateral offset of a thin cluster, or None if it is not thin."""
    if len(members) < 3:
        return None
    lv = lat[np.asarray(members, dtype=int)]
    if float(np.percentile(lv, 90) - np.percentile(lv, 10)) > \
            float(LINE_LAT_SPREAD_M):
        return None
    return float(np.median(lv))




def line_consistent_candidates(points, pos, heading: float, *,
                               lat_min_m: float = LINE_LAT_MIN_ABS_M
                               ) -> BoundaryEvidence:
    """Keep only candidates on a thin curve beside the car.

    Bins candidates by forward distance in the ego frame, splits each bin
    at lateral gaps, keeps thin clusters, and keeps the members of every
    chain of thin clusters that spans ``LINE_CHAIN_MIN_BINS`` consecutive
    bins without a lateral jump larger than ``LINE_CHAIN_STEP_M``.

    Everything is in the ego frame at the frame's own pose: no map data,
    no world-frame assumption, and the output is evidence (a curve with a
    measured forward span and lateral offset), not a control input.
    """
    out = BoundaryEvidence(kind="curb_boundary_line",
                           provenance="lidar_height_step+line_consistency")
    p = np.asarray(points, dtype=float) if points is not None else None
    if p is None or p.ndim != 2 or len(p) < 6:
        out.meta["reason"] = "too few candidates"
        return out
    h = float(heading)
    fwd = np.array([math.cos(h), math.sin(h)])
    left = np.array([-fwd[1], fwd[0]])
    rel = p[:, :2] - np.asarray(pos, dtype=float)[:2]
    f = rel @ fwd
    lat = rel @ left
    keep = ((f >= LINE_FWD_MIN_M) & (f <= LINE_FWD_MAX_M)
            & (np.abs(lat) >= float(lat_min_m)))
    out.meta["range_window_m"] = [LINE_FWD_MIN_M, LINE_FWD_MAX_M]
    out.meta["lat_min_abs_m"] = float(lat_min_m)
    if not keep.any():
        out.meta["reason"] = "no candidate beside the car"
        return out
    idx = np.nonzero(keep)[0]
    f, lat = f[keep], lat[keep]
    b_of = np.floor(f / LINE_FWD_BIN_M).astype(int)
    segments: list[tuple[int, float, list[int]]] = []
    for b in sorted(set(int(v) for v in b_of)):
        order = np.nonzero(b_of == b)[0][np.argsort(lat[b_of == b])]
        cur = [int(order[0])]
        for c in order[1:]:
            if float(lat[c]) - float(lat[cur[-1]]) <= float(LINE_LAT_GAP_M):
                cur.append(int(c))
                continue
            med = _thin_lateral_cluster(lat, cur)
            if med is not None:
                segments.append((int(b), med, list(cur)))
            cur = [int(c)]
        med = _thin_lateral_cluster(lat, cur)
        if med is not None:
            segments.append((int(b), med, list(cur)))
    out.meta["segments"] = len(segments)
    if not segments:
        out.meta["reason"] = "no thin cluster"
        return out
    segments.sort(key=lambda s: (s[0], s[1]))
    # Greedy chain tracking: extend the live chain whose last bin is the
    # previous bin and whose lateral offset is closest; otherwise start a
    # new chain.  Each chain is one boundary LINE with its own identity -
    # never an average over several lines (an earlier version published the
    # union and reported the median of two different curbs as one offset).
    chains: list[dict] = []
    for b, med, members in segments:
        best = None
        for ch in chains:
            if ch["last_bin"] != b - 1:
                continue
            jump = abs(med - ch["lat"])
            if jump <= LINE_CHAIN_STEP_M and (best is None
                                              or jump < best[0]):
                best = (jump, ch)
        if best is None:
            chains.append({"first_bin": b, "last_bin": b, "lat": med,
                           "bins": 1, "members": list(members)})
        else:
            ch = best[1]
            ch["last_bin"] = b
            ch["lat"] = med
            ch["bins"] += 1
            ch["members"].extend(members)
    usable = [ch for ch in chains if ch["bins"] >= LINE_CHAIN_MIN_BINS]
    out.meta["chains"] = len(usable)
    if not usable:
        out.meta["reason"] = "no chain of thin clusters"
        return out
    usable.sort(key=lambda ch: (-ch["bins"], ch["first_bin"]))
    # A chain 19 m to the side is evidence of SOMETHING, but not of the lane
    # boundary beside the car: relevance is distance to the ego path, not
    # chain length.  Out-of-band chains are kept (with identity) so nothing
    # is thrown away, and the lane-plausible ones are published.
    band = [ch for ch in usable if abs(float(ch["lat"])) <= LINE_LAT_MAX_M]
    distant = [ch for ch in usable if abs(float(ch["lat"])) > LINE_LAT_MAX_M]
    out.meta["lat_band_m"] = [-LINE_LAT_MAX_M, LINE_LAT_MAX_M]
    out.meta["distant_chains"] = [
        {"bins": int(ch["bins"]),
         "lat_offset_m": round(float(ch["lat"]), 3),
         "points": [[round(float(v), 3) for v in q]
                    for q in p[idx[np.unique(np.asarray(
                        ch["members"], dtype=int))]][:, :2]]}
        for ch in distant[:6]]
    if not band:
        out.meta["reason"] = (f"no chain within {LINE_LAT_MAX_M:.1f} m of "
                              f"the ego path ({len(distant)} distant chain(s) "
                              f"recorded, not published as a lane boundary)")
        return out
    main = band[0]
    sel = np.unique(np.asarray(main["members"], dtype=int))
    out.points = p[idx[sel]][:, :2]
    span = np.sort(f[sel])
    out.meta["fwd_span_m"] = [round(float(span[0]), 3),
                              round(float(span[-1]), 3)]
    out.meta["lat_offset_m"] = round(float(np.median(lat[sel])), 3)
    out.meta["lat_spread_m"] = round(float(
        np.percentile(lat[sel], 90) - np.percentile(lat[sel], 10)), 3)
    out.meta["chain_bins"] = int(main["bins"])
    # each chain keeps its OWN points: a two-sided stretch (the left and the
    # right boundary of one lane) is only measurable if the second line is
    # published with its geometry, not just as a summary
    out.meta["other_chains"] = []
    for ch in band[1:6]:
        sel_ch = np.unique(np.asarray(ch["members"], dtype=int))
        out.meta["other_chains"].append({
            "bins": int(ch["bins"]),
            "lat_offset_m": round(float(ch["lat"]), 3),
            "last_bin": int(ch["last_bin"]),
            "points": [[round(float(v), 3) for v in q]
                       for q in p[idx[sel_ch]][:, :2]],
        })
    out.confidence = float(min(1.0, len(sel) / 20.0))
    return out




def curb_candidates(points, *, ground_z: float, step_k: float = STEP_K,
                    curb_height_m: float = CURB_HEIGHT_M,
                    max_range_m: float = MAX_CURB_RANGE_M,
                    pos=None, min_points: int = 6) -> BoundaryEvidence:
    """Curb candidates from GROUND HEIGHT STEPS in a LiDAR cloud.

    A kerb, a wall foot and a raised shoulder all appear as a local step in
    the ground surface.  The threshold is geometric (see
    :func:`adaptive_step_threshold`): at each candidate range it is derived
    from the measured local point spacing, so the detector does not need a
    hand-tuned metre constant that is wrong at every range but one.

    Nothing here is a control input: this is boundary EVIDENCE, published
    with its threshold so a reviewer can see why a candidate appeared.
    """
    pts = np.asarray(points, dtype=float) if points is not None else None
    out = BoundaryEvidence(kind="curb", provenance="lidar_height_step",
                           meta={"ground_z": float(ground_z),
                                 "step_k": float(step_k),
                                 "capability": "no per-beam id in cloud"})
    if pts is None or pts.ndim != 2 or len(pts) < min_points:
        out.meta["reason"] = "no cloud"
        return out
    p2 = pts[:, :2]
    z = pts[:, 2]
    if pos is not None:
        origin = np.asarray(pos, dtype=float)[:2]
    else:
        origin = np.zeros(2)
    rng = np.linalg.norm(p2 - origin, axis=1)
    near = np.isfinite(z) & (rng <= float(max_range_m)) & (rng > 1.0)
    if int(np.count_nonzero(near)) < min_points:
        out.meta["reason"] = "too few points in range"
        return out
    idxs = np.nonzero(near)[0]
    found: list[list[float]] = []
    thresholds: list[float] = []
    # Order the near points by polar angle WITHIN RANGE BANDS: consecutive
    # entries are then real neighbours along one sweep, which is where a
    # kerb shows up as a height JUMP (sorting all points by angle alone
    # mixes near and far samples and produced 5 m "gaps" between supposed
    # neighbours - measured on the first version).
    # Range image: bin by (range, bearing) and compare each occupied cell
    # with its neighbours in BOTH axes.  A kerb is a surface discontinuity;
    # from a car it is a BEARING-adjacent z jump (the surface steps as you
    # look along it), which is what the (0, +1) comparison finds, while the
    # (1, 0) comparison only helps when the range bins are finer than the
    # ring spacing - they are not beyond a few metres with 16 beams (0.29 m
    # at 5 m, 1.17 m at 20 m), and that limit is recorded here rather than
    # hidden by a permissive threshold.
    ang = np.arctan2(p2[idxs, 1] - origin[1], p2[idxs, 0] - origin[0])
    r_bin = np.floor(rng[idxs] / RANGE_BIN_M).astype(int)
    a_bin = np.floor((ang + math.pi) / ANGLE_BIN_RAD).astype(int)
    cells: dict[tuple[int, int], list[int]] = {}
    for k, i in enumerate(idxs):
        cells.setdefault((int(r_bin[k]), int(a_bin[k])), []).append(int(i))
    z_of = {key: float(np.median(z[v])) for key, v in cells.items()}
    r_of = {key: float(np.median(rng[v])) for key, v in cells.items()}
    pair_gaps: list[float] = []
    jumps: list[tuple[int, int, float]] = []
    for (rb, ab), ids_here in cells.items():
        for dr, da in ((0, 1), (1, 0)):
            nbr = cells.get((rb + dr, ab + da))
            if not nbr:
                continue
            dz = abs(z_of[(rb + dr, ab + da)] - z_of[(rb, ab)])
            dxy = abs(r_of[(rb, ab)] - r_of[(rb + dr, ab + da)])
            pair_gaps.append(dxy)
            jumps.append((ids_here[0], nbr[0], float(dz)))
    if not cells:
        out.meta["reason"] = "no occupied cells"
        return out
    scale = float(np.median(pair_gaps)) if pair_gaps else RANGE_BIN_M
    thr = adaptive_step_threshold(min(scale, RANGE_BIN_M), k=step_k)
    found: list[list[float]] = []
    for ia, ib, dz in jumps:
        if dz <= thr or dz < float(curb_height_m) * 0.5:
            continue
        found.append([float(p2[ia, 0]), float(p2[ia, 1])])
        found.append([float(p2[ib, 0]), float(p2[ib, 1])])
    thresholds = [thr] * len(found)
    gap_medians = [scale] if pair_gaps else []
    out.points = (np.asarray(found, dtype=float) if found
                  else np.empty((0, 2), dtype=float))
    out.meta["local_gap_p50_m"] = (round(float(np.median(gap_medians)), 4)
                                   if gap_medians else None)
    out.meta["threshold_p50_m"] = (round(float(np.median(thresholds)), 3)
                                   if thresholds else None)
    out.meta["threshold_range_m"] = (
        [round(float(min(thresholds)), 3), round(float(max(thresholds)), 3)]
        if thresholds else None)
    out.confidence = (None if not found
                      else float(min(1.0, len(found) / 20.0)))
    return out


def pavement_edges(road_mask, cam_model, pos, heading, *,
                   ground_z: float, min_span_m: float = 3.0) -> BoundaryEvidence:
    """The paved surface's own edge, from the SEMANTIC road mask.

    Deliberately the same evidence the paved fallback already uses
    (``lane/pavement.paved_edge_lane_center``): the plan's trust order is
    marking >> pavement edge = guardrail = fence, so the pavement edge is
    real evidence and not a stand-in for a map line.  It is published here
    as its own kind so it is never averaged with a curb.
    """
    out = BoundaryEvidence(kind="pavement_edge",
                           provenance="semantic_road_mask",
                           meta={"model": "paved_edge_lane_center"})
    if road_mask is None or cam_model is None:
        out.meta["reason"] = "no road mask"
        return out
    try:
        from .pavement import paved_edge_lane_center
        ref = paved_edge_lane_center(np.asarray(road_mask, dtype=bool),
                                     cam_model, pos, heading,
                                     ground_z=ground_z, debug=out.meta)
    except Exception as exc:
        out.meta["reason"] = f"pavement edge failed: {exc}"
        return out
    if ref is None or getattr(ref, "center", None) is None \
            or len(ref.center) < 3:
        out.meta["reason"] = "pavement edge unavailable"
        return out
    edges = []
    if getattr(ref, "left", None) is not None:
        edges.append(np.asarray(ref.left, dtype=float)[:, :2])
    if getattr(ref, "right", None) is not None:
        edges.append(np.asarray(ref.right, dtype=float)[:, :2])
    if not edges:
        out.meta["reason"] = "no edge polylines"
        return out
    pts = np.vstack(edges)
    span = float(getattr(ref, "span_m", 0.0) or 0.0)
    out.points = pts
    out.confidence = float(min(1.0, span / 10.0))
    out.meta["span_m"] = round(span, 2)
    out.meta["min_span_m"] = float(min_span_m)
    return out


def obstacle_entities(obstacles, *, kinds: tuple = ()) -> BoundaryEvidence:
    """Solid obstacles as ENTITIES (their own kind, never a lane edge).

    The plan: a bound tyre or a wall must remain an independent hard
    constraint; it may only be associated with a boundary after the fact.
    """
    out = BoundaryEvidence(kind="obstacle", provenance="obstacle_boxes",
                           meta={"requested_kinds": list(kinds)})
    rows = []
    for ob in obstacles or []:
        kind = str(getattr(ob, "category", "") or "")
        if kinds and kind not in kinds:
            continue
        x = getattr(ob, "x", None)
        y = getattr(ob, "y", None)
        if x is None or y is None:
            continue
        rows.append([float(x), float(y)])
    out.points = (np.asarray(rows, dtype=float) if rows
                  else np.empty((0, 2), dtype=float))
    out.confidence = None if not rows else float(min(1.0, len(rows) / 5.0))
    out.meta["n_entities"] = len(rows)
    return out


def associate(evidence: list[BoundaryEvidence], *,
              tol_m: float = ASSOC_TOL_M) -> dict:
    """Report which evidence describes the SAME object - without merging.

    Returns ``{"same_object": [...], "distinct": [...], "reason": ...}``.
    The geometry of every input is left untouched: the plan forbids fusing
    different semantics into one boundary, so this is a diagnostic that
    says "these two candidates are probably the same kerb" and nothing
    more.
    """
    same: list[dict] = []
    distinct: list[dict] = []
    for i in range(len(evidence)):
        for j in range(i + 1, len(evidence)):
            a, b = evidence[i], evidence[j]
            if a.points.size == 0 or b.points.size == 0:
                continue
            d = np.linalg.norm(a.points[:, None, :] - b.points[None, :, :],
                               axis=2)
            dmin = float(d.min())
            row = {"a": a.kind, "b": b.kind, "min_distance_m": round(dmin, 3),
                   "tol_m": float(tol_m)}
            (same if dmin <= float(tol_m) else distinct).append(row)
    return {"same_object": same, "distinct": distinct,
            "kinds": [e.kind for e in evidence],
            "merged": False,
            "note": "association only reports coincidence; no geometry is "
                    "merged and no kind loses its provenance"}


# --- temporal persistence: the selectivity fix T07 asks for -------------
#: A curb candidate must recur this many times inside the window before it
#: is published as geometry.  Measured 2026-09-22: a single real cloud
#: frame yields ~4790 candidate points, i.e. the detector alone is not
#: selective enough to be evidence; persistence is what separates a kerb
#: (there every tick) from a shadow, a bush or a car (there once).
PERSIST_MIN_HITS = 3
PERSIST_WINDOW_S = 3.0
#: Cell size for the world-space recurrence test, metres.
PERSIST_CELL_M = 0.5


class CurbPersistence:
    """Keep only candidates that RECUR at (nearly) the same world place.

    Pure state + pure functions over world points: no camera, no map, no
    perception internals.  The window is measured in TIME (the plan's rule:
    frame counts break when the tick rate moves) and the recurrence test is
    spatial, so ego motion does not smear a candidate away.
    """

    def __init__(self, *, min_hits: int = PERSIST_MIN_HITS,
                 window_s: float = PERSIST_WINDOW_S,
                 cell_m: float = PERSIST_CELL_M) -> None:
        self.min_hits = int(min_hits)
        self.window_s = float(window_s)
        self.cell_m = float(cell_m)
        self._cells: dict[tuple[int, int], list[float]] = {}

    def _key(self, x: float, y: float) -> tuple[int, int]:
        return (int(np.floor(float(x) / self.cell_m)),
                int(np.floor(float(y) / self.cell_m)))

    def update(self, points, now_s: float) -> BoundaryEvidence:
        """Fold one tick's candidates in; return the PERSISTENT ones."""
        pts = np.asarray(points, dtype=float) if points is not None else None
        now = float(now_s)
        # drop cells whose hits have all aged out of the window
        for k in list(self._cells):
            self._cells[k] = [t for t in self._cells[k]
                              if now - t <= self.window_s]
            if not self._cells[k]:
                del self._cells[k]
        seen: set[tuple[int, int]] = set()
        if pts is not None and pts.ndim == 2:
            for x, y in pts[:, :2]:
                k = self._key(float(x), float(y))
                seen.add(k)
                self._cells.setdefault(k, []).append(now)
        out = BoundaryEvidence(kind="curb", provenance="lidar_height_step+"
                                                       "persistence")
        keep = [k for k, hits in self._cells.items() if len(hits)
                >= self.min_hits]
        out.points = (np.array([[k[0] * self.cell_m, k[1] * self.cell_m]
                                for k in keep], dtype=float)
                      if keep else np.empty((0, 2), dtype=float))
        out.meta = {"min_hits": self.min_hits, "window_s": self.window_s,
                    "cell_m": self.cell_m,
                    "raw_cells": len(seen), "persistent_cells": len(keep)}
        out.confidence = (None if not keep
                          else float(min(1.0, len(keep) / 20.0)))
        return out

    def reset(self) -> None:
        self._cells.clear()
