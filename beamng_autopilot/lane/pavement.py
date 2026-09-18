"""Paved-boundary lateral reference for a paved road with NO marking.

AGENTS.md「驾驶约束」: the right-side trust order is painted marking >>
pavement edge = guardrail = fence, and a dirt shoulder must never count
as road while the route is paved.  The stack implemented the first and
the last part of that; the middle was missing - a tick with no usable
marking published ``lane_src_sel = "perception-unavailable"`` and the
car braked, even on a plainly visible paved road.  This module is that
missing candidate.

What it produces
----------------
The semantic head's PAVED road mask (``strip_soil_from_road`` has already
removed the soil/dirt colour, so the shoulder is not in it) is
back-projected onto the ground plane and turned into three world
polylines:

* ``right`` - the observed pavement right edge (hard boundary);
* ``left``  - the observed pavement left edge (hard boundary);
* ``center`` - the keep-right target ``right_edge + lane_half_m`` (the
  same lane-width contract ``vision.lanes.painted_line_lane_center``
  uses for painted lines), clamped so it never sits closer than
  ``min_clear_m`` to either edge.

The edges are published as the tick's hard boundaries, so a body
crossing them is the same violation the no-cross rule already grades for
a painted lane; path validity is graded against the drivable (paved) BEV
layer in ``planning.constraints``, which is what makes "禁止驾驶到土和草
上" structural instead of a promise.

Gates (each one is the answer to a measured failure of the obvious
implementation)
--------------------------------------------------------------------
* BOTH edges of a band must be OBSERVED INSIDE THE IMAGE: a pavement
  running past the frame border has no observed edge, and its "edge"
  would be the image border - a phantom boundary inside the road;
* the observed span must be road-sized (a plaza or a whole junction is
  not a lane, and above ``max_span_m`` the right edge stops meaning
  "the edge of my road");
* the ego must sit ON the corridor: the pavement has to pass under the
  car's own track in the near field, otherwise the car is already on
  soil/grass and the honest answer is a stop, not a new lateral target;
* the corridor is TRACKED band to band, so a read cannot jump onto a
  neighbouring paved patch (driveway / side road / plaza).

Refuted predecessor
-------------------
``reference.bev_corridor_lane_center`` used the BEV free corridor with
none of these gates and pointed 1.24 m (p50) right of the true lane
centre on 52.7% of frames (town corridor arm 2026-09-12) because that
corridor's right edge was the SHOULDER's edge.  This module is not that:
the mask it reads is soil-stripped, the right edge must be a pavement /
off-pavement transition with both edges inside the image, and the result
is a keep-right target plus hard pavement boundaries - never a
whole-road centre.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from .constants import LANE_WIDTH_DEFAULT_M


# Keep-right lateral target (right-hand traffic): the own-lane centre is
# half a lane width to the LEFT of the observed paved right edge.  Same
# contract as ``painted_line_lane_center``'s ``lane_half_m`` default.
PAVED_LANE_HALF_M = 0.5 * LANE_WIDTH_DEFAULT_M

# Closest the target may ever sit to a pavement edge: the authoritative
# ego half width (config.EGO_HALF_WIDTH_M = 0.9 m) plus a 0.25 m margin.
# A pavement narrower than 2*this has no legal lateral position at all
# and the read abstains instead of picking a side.
PAVED_MIN_CLEAR_M = 1.15

# Observed pavement span gates.  The floor is two minimum clearances -
# 2*PAVED_MIN_CLEAR_M, the width below which no legal lateral position
# exists at all (the target clamps to the middle there, so a genuine
# single-track lane is still drivable).
#
# The CEILING is measured, and it is deliberately NOT the 10 m the
# refuted corridor candidate used: that gate came from inferring a LANE
# CENTRE as the corridor's centre, which collapses when the corridor's
# right edge stops being the lane's.  This candidate never infers a
# centre - it is "keep lane_half_m + margin inside the observed right
# edge", which does not care how wide the pavement is.  Live evidence
# raised to 14 m to admit those bands - and the live run that followed
# drove on the shoulder and into the guardrail.  Reverted.  What the
# stretch really shows is that the MASK reads 8-11 m where the hand
# labels say ~6 m of pavement: the shoulder is classified as road, so the
# "right edge" this candidate measures is the shoulder's edge.  That is a
# model problem, not a gate problem.
PAVED_MIN_SPAN_M = 2.4
#
# The ceiling is deliberately TIGHTER than the 10 m the refuted corridor
# candidate used, because the live failure mode of THIS candidate is a
# mask that swallowed the shoulder: on the east_coast stretch the user's
# hand labels say ~6 m of pavement while the deployed model reads 8-11 m
# there, and a run that accepted those reads drove on the shoulder and
# wedged against the guardrail (2026-09-18).  Below 7.5 m the read can
# not be "the whole road + both shoulders" of that stretch; a genuinely
# wider road abstains and strict mode fails CLOSED (stop), which is the
# legal degradation.  Raise only with hand-labelled evidence on the new
# stretch.
PAVED_MAX_SPAN_M = 7.5

# Longitudinal read window.  The target has to be measurable where the
# car can still act on it, so at least ``min_bands`` bands with both
# edges observed must exist, covering ``min_covered_m`` of road, with the
# nearest of them no further than ``max_first_m``.
PAVED_BAND_M = 3.0
PAVED_NEAR_M = 3.0
PAVED_FAR_M = 24.0
PAVED_MIN_BANDS = 3
PAVED_MIN_COVERED_M = 6.0
PAVED_MAX_FIRST_M = 12.0

# "The ego is on the pavement" gate: the pavement must pass under the
# car's own track (|lateral| <= this) within ``ego_near_max_m`` ahead.
PAVED_EGO_TOL_M = 0.25
PAVED_EGO_NEAR_MAX_M = 6.0

# Pixel border a band edge must stay inside before it counts as OBSERVED.
# A pavement reaching the frame edge continues outside the image; 6 px of
# margin is above the 4 px sampling step.
PAVED_EDGE_BORDER_PX = 6

# A pavement run inside one band must be this long to be pavement rather
# than a stray classification speck.
PAVED_RUN_MIN_M = 0.6

# Longitudinal gap inside one band that still counts as continuous
# pavement.  Paint is ON the pavement but is a different mask class, so
# the road mask carries a hole where the marking is: a US double-yellow
# is ~0.4 m wide and must not split the corridor into "my lane" and "the
# other lane" (a split there would publish the centre line as the hard
# left boundary, which happens to be safe but is an accident of the
# threshold, not a measurement).  Real separate patches are metres apart.
# The limit grows with distance because the sampled point spacing does
# (measured: adjacent-band lat steps reach 0.7 m at 14-16 m with a 4 px
# sampling step), which would otherwise fragment the far bands.
PAVED_RUN_GAP_M = 0.55
PAVED_RUN_GAP_FRAC = 0.06
PAVED_RUN_GAP_MAX_M = 1.2

# The outermost sample of a run is its least reliable one - it is the
# pixel row where the classifier decided to stop.  Measured against the
# hand-labelled pavement boundary (35 frames, 2026-09-18): the deployed
# model's right edge sat OUTSIDE the true pavement on 53.6% of rows
# (median +8 px, tail to +322 px), the fine-tuned one on 39.9% (median
# +1 px).  Discarding the last 5 cm of the run's own extent removes the
# one-pixel outshoots; it is a DISTANCE and not a point count because a
# point count biases the read further inboard the further away the band
# is (adjacent samples are ~0.25 m apart at 15 m but ~0.05 m at 4 m).
PAVED_EDGE_TRIM_M = 0.05
# ... and the same tail, in metres: how much further right than the
# previous band a band's right edge may legitimately move. A road edge
# widens smoothly; a jump outward is the mask spilling onto the
# shoulder.  ``PAVED_EDGE_TOTAL_MAX_M`` bounds the whole read the same
# way, so a spill can not walk outward band by band (0.45 m per 3 m band
# is 3 m over a 24 m read).  Both are one-sided on purpose - moving
# INWARD is always allowed (that only makes the keep-right target more
# conservative).
PAVED_EDGE_STEP_MAX_M = 0.45
PAVED_EDGE_TOTAL_MAX_M = 1.0
# Safety bias of the keep-right target away from the estimated pavement
# edge, in metres.  The edge is an estimate with a measured outward tail
# (fine-tuned model: p90 +48 px ~= 0.7 m at 6-8 m), and being wrong
# toward the shoulder is the one error this feature must not make
# (AGENTS.md「驾驶约束」: 禁止将车辆驾驶到土和草上).  It is the same
# class of constant as ``painted_line_lane_center``'s ``lane_half_m`` -
# an offset from a PERCEIVED boundary, never from a map line.
#
# It stays small on purpose: the bias plus the edge trim must remain far
# inside ONE lane (1.75 + ~0.3 m from the edge leaves about a metre of
# right-lane margin on a 6 m road).  A larger constant would push the
# target over the centre line on two-lane roads, which is a worse
# failure than the spill it would be compensating for - and it could not
# fix a CONTIGUOUS spill anyway (a shoulder region classified as road
# moves the read wholesale; only the model, the band-continuity clamp
# and the live A/B address that).
PAVED_EDGE_MARGIN_M = 0.15

# Corridor tracking: how far a band's run may sit from the previous
# band's before the corridor is considered to have ended.
#
# 2026-09-18 live lesson: this was widened to 4.0 m (and the span ceiling
# to 14 m) purely to raise offline AVAILABILITY on the east_coast
# unmarked stretch - 98% of frames instead of 29%.  The next live run
# used every one of those extra frames to ride the lane line and end up
# wedged against the guardrail ("no drivable path", ``lane=paved``).
# Availability is not correctness: a read that includes the shoulder
# makes the car drive on the shoulder, and the longer/further the read
# runs the more of the road it covers.  Both gates are back at their
# measured-tight values, and the candidate stays OFF by default until a
# pavement edge is trustworthy (see ``paved_fallback``).
PAVED_CONTINUE_MAX_M = 2.0


@dataclass
class PavedLane:
    """One tick's paved-boundary lateral reference (world ``xy``)."""

    center: np.ndarray
    left: np.ndarray
    right: np.ndarray
    span_m: float = 0.0
    bands: int = 0
    # Keep-right target lateral at the first usable band (left = +): the
    # signed distance the car has to move to reach the pavement-relative
    # lane position it should hold.  Telemetry / tests read this.
    first_lat_m: float = 0.0
    first_lon_m: float = 0.0
    meta: dict = field(default_factory=dict)


def _project_paved_points(road_mask, cam, pos, heading, ground_z,
                          step: int, far_m: float):
    """Back-project sampled pavement pixels into ego-frame points.

    Returns ``(ex, ey, u, img_w)``: forward / left metres in the ego
    frame and the IMAGE COLUMN each point came from (the caller needs it
    to tell an observed pavement edge from the frame border).  The ray
    math mirrors ``occupancy.project_road_mask_to_grid``; the projection
    geometry itself stays owned by ``cam.camera_pose``.
    """
    h, w = road_mask.shape[:2]
    C, r_vec, f_vec, u_vec = cam.camera_pose(pos, heading)
    vs = np.arange(0, h, int(step))
    us = np.arange(0, w, int(step))
    uu, vv = np.meshgrid(us, vs)
    ok = np.asarray(road_mask[vv, uu], dtype=bool)
    xc = (uu - float(cam.cx)) / float(cam.fx)
    yc = (vv - float(cam.cy)) / float(cam.fy)
    D = xc[..., None] * r_vec - yc[..., None] * u_vec + f_vec
    Dz = D[..., 2]
    hit = ok & (Dz < -1e-6)
    t = np.zeros_like(Dz)
    np.divide(float(ground_z) - C[2], Dz, out=t, where=hit)
    keep = hit & (t > 0.0) & (t <= float(far_m))
    wx = C[0] + t * D[..., 0]
    wy = C[1] + t * D[..., 1]
    ch, sh = math.cos(float(heading)), math.sin(float(heading))
    dx = wx - float(pos[0])
    dy = wy - float(pos[1])
    ex = dx * ch + dy * sh
    ey = -dx * sh + dy * ch
    keep &= ex * ex + ey * ey <= float(far_m) * float(far_m)
    return ex[keep], ey[keep], uu[keep], int(w)


def _band_runs(ex, ey, u, img_w, band_m: float, far_m: float,
               border_px: int):
    """Contiguous pavement runs per longitudinal band.

    Returns ``[(lon, [(lat_right, lat_left, right_observed,
    left_observed), ...]), ...]`` ordered near -> far.  ``lat_right`` is
    the smaller (more negative) lateral and ``lat_left`` the larger,
    because the ego frame has +y LEFT.  A run whose extreme point sits on
    the frame border is NOT observed - the pavement simply continues
    outside the image there, so that edge carries no information (and
    publishing it would put a phantom boundary inside the road).
    """
    bands: list[tuple[float, list[tuple[float, float, bool, bool]]]] = []
    edges = np.arange(0.0, float(far_m) + float(band_m), float(band_m))
    for i in range(len(edges) - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        sel = (ex >= lo) & (ex < hi)
        if not sel.any():
            continue
        lon = 0.5 * (lo + hi)
        lat = np.sort(ey[sel])
        col = u[sel][np.argsort(ey[sel])]
        gap_max = min(float(PAVED_RUN_GAP_MAX_M),
                      max(float(PAVED_RUN_GAP_M),
                          float(PAVED_RUN_GAP_FRAC) * lon))
        runs: list[tuple[float, float, bool, bool]] = []
        start = 0
        for j in range(1, len(lat) + 1):
            if j < len(lat) and (lat[j] - lat[j - 1]) <= gap_max:
                continue
            seg_lat = lat[start:j]
            seg_col = col[start:j]
            start = j
            if len(seg_lat) < 3 or (seg_lat[-1] - seg_lat[0]) < PAVED_RUN_MIN_M:
                continue
            # Discard the run's outermost slice on the RIGHT
            # (PAVED_EDGE_TRIM_M): the boundary sample is the single least
            # reliable decision the classifier made for this run, and the
            # keep-right reference must never err toward the shoulder.
            # Advances only while the samples are DENSE enough that each
            # dropped point is under the trim - the sampled lat spacing
            # near a band's extreme reaches 0.3 m at 4-6 m range, and
            # stepping across such a gap would bias the read by a whole
            # sample instead of by the trim.  The LEFT extent is kept raw
            # on purpose: trimming it would only shrink the room the
            # narrow-road clamp has, and the left side is not what this
            # reference steers by.
            a = 0
            while (a + 1 < max(3, len(seg_lat) - 2)
                   and (seg_lat[a + 1] - seg_lat[a]) <= PAVED_EDGE_TRIM_M
                   and (seg_lat[a + 1] - seg_lat[0])
                   <= 3.0 * PAVED_EDGE_TRIM_M):
                a += 1
            r_lat = float(seg_lat[a])
            l_lat, l_col = float(seg_lat[-1]), int(seg_col[-1])
            # Observedness is read from the run's RAW extreme, never from
            # the trimmed point: trimming walks inward, so a pavement that
            # runs off the frame would otherwise be re-labelled "edge
            # observed" a few tenths of a metre later and the read would
            # steer by the field-of-view limit.
            right_ok = int(seg_col[0]) <= (img_w - 1 - int(border_px))
            left_ok = l_col >= int(border_px)
            runs.append((r_lat, l_lat, bool(right_ok), bool(left_ok)))
        if runs:
            bands.append((lon, runs))
    return bands


def paved_edge_lane_center(road_mask, cam, pos, heading,
                           ground_z: float | None = None,
                           *, lane_half_m: float = PAVED_LANE_HALF_M,
                           min_clear_m: float = PAVED_MIN_CLEAR_M,
                           min_span_m: float = PAVED_MIN_SPAN_M,
                           max_span_m: float = PAVED_MAX_SPAN_M,
                           near_m: float = PAVED_NEAR_M,
                           far_m: float = PAVED_FAR_M,
                           band_m: float = PAVED_BAND_M,
                           min_bands: int = PAVED_MIN_BANDS,
                           min_covered_m: float = PAVED_MIN_COVERED_M,
                           max_first_m: float = PAVED_MAX_FIRST_M,
                           step: int = 4,
                           debug: dict | None = None) -> PavedLane | None:
    """Keep-right lateral reference from the PAVED road mask, or None.

    ``road_mask`` is the semantic head's road mask (already soil-stripped
    upstream - passing a mask that still contains the shoulder would make
    this function point the car at the shoulder, which is exactly the
    refuted corridor failure).  ``ground_z`` is the ROAD SURFACE plane
    (``pos[2] - config.EGO_ORIGIN_GROUND_GAP_M``), not the ego origin.

    Abstains (returns None, leaving strict mode to fail closed) when the
    pavement is not observed well enough to place a car on it.  ``debug``
    receives the reason and the measured numbers.
    """
    if debug is None:
        debug = {}
    debug["mode"] = "no_read"
    mask = np.asarray(road_mask)
    if mask.size == 0 or not mask.any():
        debug["mode"] = "empty_mask"
        return None
    try:
        if ground_z is None:
            ground_z = float(pos[2]) if len(np.asarray(pos)) > 2 else 0.0
        ex, ey, u, img_w = _project_paved_points(
            mask, cam, pos, heading, float(ground_z), step, float(far_m))
        debug["points"] = int(len(ex))
        if len(ex) < 8:
            debug["mode"] = "too_few_points"
            return None
        bands = _band_runs(ex, ey, u, img_w, band_m, far_m,
                           PAVED_EDGE_BORDER_PX)
        debug["bands"] = len(bands)
        if not bands:
            debug["mode"] = "no_band_runs"
            return None
        # 1) The ego must be ON the pavement: the pavement has to be
        # observed along the car's own track in the near field.  Without
        # this the pavement ahead (or beside) is referenced while the car
        # itself already stands on soil/grass, and the reference would
        # just drive it onward.
        #
        # A more permissive variant - bridging a hole that has pavement on
        # BOTH sides of the car - was measured and rejected: on the live
        # east_coast frames 70/100 (2026-09-18) the car stood IN the
        # vegetation with the pavement visible to one side, and the
        # classifier produced a second patch on the other side, leaving a
        # 2.6-4.0 m hole with the car inside it.  That is geometrically
        # indistinguishable from the car's own hood / shadow hiding the
        # near field (the reason the bridging rule existed), so bridging
        # would have declared "on the pavement" for a car standing on
        # dirt.  Abstaining - and letting strict mode fail closed - is the
        # only safe reading.
        seed_lon = None
        seed = None
        for lon, runs in bands:
            if lon > PAVED_EGO_NEAR_MAX_M:
                break
            under = [r for r in runs
                     if r[0] - PAVED_EGO_TOL_M <= 0.0 <= r[1] + PAVED_EGO_TOL_M]
            if under:
                seed_lon = lon
                seed = min(under, key=lambda r: abs(0.5 * (r[0] + r[1])))
                break
        if seed is None:
            debug["mode"] = "ego_not_on_pavement"
            return None
        # 2) Track the corridor band to band: keep the run whose middle is
        # closest to the previous band's, stop when the pavement no longer
        # continues, and refuse an OUTWARD step bigger than a road edge
        # can make - a jump further right is the mask spilling onto the
        # shoulder.  Inward moves are always allowed (they only make the
        # keep-right target more conservative).
        corridor: list[tuple[float, tuple[float, float, bool, bool]]] = [
            (float(seed_lon), seed)]
        # Continuity clamp on the leftward-widening read.  Two bounds,
        # both one-sided (moving inward is always allowed - it can only
        # make the keep-right target more conservative):
        #   * per band, against the previous band's observed edge, so a
        #     single band can not jump metres outward;
        #   * over the whole read, against the FIRST observed edge, so a
        #     spill can not walk outward band by band (0.45 m per 3 m
        #     band adds up to 3 m over a 24 m read).
        # An UNOBSERVED edge is never an anchor: a band whose right side
        # runs off the frame reports the frame corner, not an edge, and
        # anchoring on it would manufacture an inner boundary out of the
        # field-of-view limit (an 18 m wide pavement then read as a 9.6 m
        # road whose "edge" is 3.4 m right of the car - a target the car
        # could chase for ever, because the frame corner moves with it).
        anchor: float | None = None
        prev_edge: float | None = None
        if seed[2]:
            anchor = prev_edge = float(seed[0])
        for lon, runs in bands:
            prev_lon, prev = corridor[-1]
            if lon <= prev_lon + 1e-9:
                continue
            if (lon - prev_lon) > (band_m + 0.6):
                break                 # a whole band carried no pavement run
            prev_mid = 0.5 * (prev[0] + prev[1])
            # Continuation is judged on the run's MIDDLE first: that is
            # the run the car is driving on.  Judging it on the right edge
            # alone latched onto a 1.2 m mask sliver beside the road on
            # the east_coast unmarked stretch (frame 20 of the 22:08 run:
            # runs (-3.74,-2.55) and (-1.53,8.5) at 10.5 m, the sliver
            # winning on right-edge distance) and the walk then died at
            # the next band.  The right edge is the FALLBACK, for a band
            # that the car's own shadow / a paint hole split so that no
            # run sits near the previous middle while one of them still
            # carries the same right edge.
            best = min(runs, key=lambda r: abs(0.5 * (r[0] + r[1]) - prev_mid))
            if abs(0.5 * (best[0] + best[1]) - prev_mid) \
                    > PAVED_CONTINUE_MAX_M:
                alt = min(runs, key=lambda r: abs(r[0] - prev[0]))
                if abs(alt[0] - prev[0]) <= PAVED_CONTINUE_MAX_M:
                    best = alt
                else:
                    break             # corridor jumped to another patch
            rlat, llat, r_ok, l_ok = best
            if r_ok:
                lim: float | None = None
                if anchor is not None:
                    lim = anchor - PAVED_EDGE_TOTAL_MAX_M
                if prev_edge is not None:
                    _p = prev_edge - PAVED_EDGE_STEP_MAX_M
                    lim = _p if lim is None else max(lim, _p)
                if lim is not None:
                    rlat = max(rlat, lim)
                # The anchor follows the INWARD-most observed edge (larger
                # lat = further from the shoulder), so the outward budget
                # is measured from the most conservative read this tick
                # has already seen.
                anchor = rlat if anchor is None else max(anchor, rlat)
                prev_edge = rlat
            corridor.append((lon, (rlat, llat, r_ok, l_ok)))
        debug["corridor_bands"] = len(corridor)
        # 3) Bands that may carry the reference.  The RIGHT edge must be
        # observed inside the image (that is the edge the target is
        # measured from); an unobserved LEFT edge only means the pavement
        # continues left, which is always safe for a keep-right target -
        # it just can not serve as the narrow-road clamp or as evidence
        # that the road is not a wide paved area.
        usable: list[tuple[float, float, float, float, float]] = []
        for lon, (rlat, llat, r_ok, l_ok) in corridor:
            if lon < float(near_m) or not r_ok:
                continue
            span = llat - rlat
            if span < float(min_span_m):
                continue
            if l_ok and span > float(max_span_m):
                continue
            tgt = rlat + float(lane_half_m) + float(PAVED_EDGE_MARGIN_M)
            lo_lim = rlat + float(min_clear_m)
            hi_lim = llat - float(min_clear_m)
            if lo_lim > hi_lim:
                continue
            usable.append((lon, float(np.clip(tgt, lo_lim, hi_lim)),
                           span, rlat, llat))
        debug["usable_bands"] = len(usable)
        if len(usable) < int(min_bands):
            debug["mode"] = "too_few_bands"
            return None
        covered = usable[-1][0] - usable[0][0]
        debug["covered_m"] = round(float(covered), 2)
        if usable[0][0] > float(max_first_m) or covered < float(min_covered_m):
            debug["mode"] = "too_far_or_short"
            return None
        spans = np.asarray([b[2] for b in usable], dtype=float)
        debug["span_med_m"] = round(float(np.median(spans)), 2)
        ch, sh = math.cos(float(heading)), math.sin(float(heading))
        p2 = np.asarray(pos[:2], dtype=float)

        def _world(lon, lat):
            return np.array([p2[0] + lon * ch - lat * sh,
                             p2[1] + lon * sh + lat * ch], dtype=float)

        first_lat = float(usable[0][1])
        center = [_world(lon, lat) for lon, lat, _, _, _ in usable]
        left = [_world(lon, llat) for lon, _, _, _, llat in usable]
        right = [_world(lon, rlat) for lon, _, _, rlat, _ in usable]
        # Anchor the reference at the ego (near -> far): a centre whose
        # first point sits metres ahead scores every candidate as
        # off-lane and the car crawls.  Boundaries are NOT anchored - a
        # boundary point at the ego would be a phantom edge beside the
        # car.
        if float(np.linalg.norm(center[0] - p2)) > 2.0:
            center = [p2.copy()] + center
        debug["mode"] = "ok"
        return PavedLane(
            center=np.asarray(center, dtype=float),
            left=np.asarray(left, dtype=float),
            right=np.asarray(right, dtype=float),
            span_m=float(np.median(spans)),
            bands=int(len(usable)),
            first_lat_m=first_lat,
            first_lon_m=float(usable[0][0]),
            meta={
                "paved_bands": int(len(usable)),
                "paved_span_m": round(float(np.median(spans)), 2),
                "paved_first_lon_m": round(float(usable[0][0]), 2),
                "paved_first_lat_m": round(first_lat, 2),
                "paved_right_lat_m": round(float(usable[0][3]), 2),
            })
    except Exception as exc:                       # pragma: no cover - guard
        debug["mode"] = "error"
        debug["error"] = str(exc)
        return None
