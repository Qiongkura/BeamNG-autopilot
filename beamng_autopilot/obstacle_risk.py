"""Obstacle risk model: from "an obstacle exists" to "how urgent is it".

The occupancy grid answers WHERE obstacles are; it cannot say how urgent
one is.  A parked car beside the lane, a roadside tree and a vehicle
closing head-on at 12 m/s are three different situations, and the
improvement plan (phase C3) asks for them to be classified and graded
rather than treated as one "obstacle" flag:

* ``hard_collision``    - inside the contact band: act now, no waiting
  for confirmation (the plan's explicit no-wait case).
* ``braking_obstacle``  - in the corridor and either closing (TTC) or
  confirmed and inside the distance the car still needs to stop.
* ``roadside_clutter``  - outside the driven corridor: a lane bound, not
  a brake demand.
* ``unknown``           - inside the corridor but not yet confirmed
  (scattered LiDAR returns): no speed cap may be derived from it, which
  is the plan's C4 "do not brake on a single-frame speck" rule.

Deliberately conservative, like ``prediction.py``:

* the speed cap is the honest stopping-distance bound
  ``v <= sqrt(2*a*(gap - margin - closing*reaction))`` - never a TTC-only
  rule, because a STATIC obstacle has no TTC at all and must be graded by
  the distance the car still needs (the swept-envelope/occupancy gates
  stay the authority for the static case);
* a static obstacle never yields a cap below its gap allows, so a distant
  roadside-to-lane object cannot creep the car;
* ego velocity is taken as ``speed * heading`` - the monitor has no
  lateral velocity estimate, and assuming one would be invention.

Nothing here steers: it returns a verdict the safety monitor folds into
its target speed, exactly like the existing corridor ease band.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from beamng_autopilot.vehicle_body import (
    CORRIDOR_HALF_WIDTH_M,
    HALF_LENGTH_M,
    HALF_WIDTH_M,
)

# Margin around the ego rectangle for the "this is me" rejection, and the
# longitudinal half-window around the ego centre that counts as "under the
# car" (the path is anchored at the centre, so the bumper is
# HALF_LENGTH_M further on - see _inside_ego).
EGO_SELF_MARGIN_M = 0.3
EGO_SELF_ALONG_M = 1.0

RISK_HARD_COLLISION = "hard_collision"
RISK_BRAKING = "braking_obstacle"
RISK_ROADSIDE = "roadside_clutter"
RISK_UNKNOWN = "unknown"

# Inside this distance even an UNCONFIRMED return stops the car: "已经进入
# 近场车身包络" is one of the plan's explicit no-wait cases.
RISK_CONTACT_BAND_M = 3.0
# Time-to-collision below which a closing object demands a braking cap.
RISK_TTC_BRAKE_S = 3.0
# A scattered return needs this many consecutive observations before any
# speed cap may be derived from it (plan C4).
RISK_MIN_CONFIRM_FRAMES = 2
# Comfortable braking used to turn a gap into a speed cap.
RISK_BRAKE_DECEL_MPS2 = 2.5
# Reserve kept between the stop point and the obstacle.
RISK_STOP_MARGIN_M = 2.0
# Reaction time charged to the closing motion before the gap is used.
RISK_REACTION_S = 0.5
# Relative speed below this counts as "not closing".
RISK_MIN_CLOSING_MPS = 0.5
# Prediction horizon for "will it intrude into my corridor", and the step
# used to sample it (a crossing vehicle must be caught when it crosses,
# not only if the crossing lands exactly on the TTC instant).
RISK_PREDICT_HORIZON_S = 3.0
RISK_PREDICT_STEP_S = 0.5
# Speed ceiling applied to a contact-band return the fused occupancy
# contradicts: it may not stop the car, but it must not be ignored either.
CREEP_SPEED_CAP_MPS = 2.0
# An object moving faster than this is a real moving threat; the fused
# raster lags a vehicle, so such a track is never excused by "the grid
# says free".  Static terrain/vegetation returns carry tracker-noise
# velocities of a few tenths of a m/s.
RISK_STATIC_OBJ_MAX_MPS = 1.5


@dataclass
class ObstacleRisk:
    """One obstacle's graded risk against the ego's corridor."""

    track_id: int
    category: str
    kind: str
    distance_m: float            # along-path (or along-heading) distance
    lateral_m: float             # signed (+ left of travel)
    closing_speed_mps: float     # > 0 = the gap is shrinking
    ttc_s: float | None          # None when not closing
    in_corridor: bool
    predicted_intrusion: bool    # will enter the corridor within horizon
    frames_seen: int             # consecutive observations (tracker)
    confirmed: bool
    target_speed_cap: float      # inf when this obstacle caps nothing
    why: str = ""


@dataclass
class RiskVerdict:
    """Aggregate of every obstacle risk for one tick."""

    kind: str = RISK_UNKNOWN
    stop: bool = False
    target_speed_cap: float = float("inf")
    closest_m: float = float("inf")
    min_ttc_s: float | None = None
    n_braking: int = 0
    n_roadside: int = 0
    n_unknown: int = 0
    items: list[ObstacleRisk] = field(default_factory=list)

    def digest(self) -> dict:
        """JSON-safe summary for telemetry (no arrays, no objects)."""
        return {
            "kind": self.kind,
            "stop": int(bool(self.stop)),
            "cap_mps": (None if not math.isfinite(self.target_speed_cap)
                        else round(float(self.target_speed_cap), 2)),
            "closest_m": (None if not math.isfinite(self.closest_m)
                          else round(float(self.closest_m), 2)),
            "min_ttc_s": (None if self.min_ttc_s is None
                          else round(float(self.min_ttc_s), 2)),
            "n_braking": int(self.n_braking),
            "n_roadside": int(self.n_roadside),
            "n_unknown": int(self.n_unknown),
        }


def _project_to_path(points: np.ndarray, path: np.ndarray):
    """``(along, lateral, tangent)`` of each point against a polyline.

    ``along`` is arc length from the path start, ``lateral`` is signed
    (+ left of the local tangent) and ``tangent`` is the unit direction
    of the nearest segment.
    """
    a = path[:-1]
    b = path[1:]
    ab = b - a
    l2 = np.einsum("ij,ij->i", ab, ab)
    seg_len = np.sqrt(np.maximum(l2, 1e-12))
    arc0 = np.concatenate([[0.0], np.cumsum(seg_len)])[:-1]
    rel = points[:, None, :] - a[None, :, :]
    t = np.clip(np.einsum("ijk,jk->ij", rel, ab)
                / np.maximum(l2[None, :], 1e-12), 0.0, 1.0)
    proj = a[None, :, :] + t[..., None] * ab[None, :, :]
    d = np.linalg.norm(points[:, None, :] - proj, axis=2)
    j = np.argmin(d, axis=1)
    rows = np.arange(len(points))
    along = arc0[j] + t[rows, j] * seg_len[j]
    # signed lateral: cross product of segment direction and the offset
    seg = ab[j]
    off = points - proj[rows, j]
    cross = seg[:, 0] * off[:, 1] - seg[:, 1] * off[:, 0]
    lat = np.where(cross >= 0.0, 1.0, -1.0) * d[rows, j]
    tan = seg / np.maximum(seg_len[j], 1e-12)[:, None]
    return along, lat, tan


def stop_distance_m(speed_mps: float,
                    decel_mps2: float = RISK_BRAKE_DECEL_MPS2) -> float:
    """Distance needed to stop from ``speed_mps`` at ``decel_mps2``."""
    v = max(0.0, float(speed_mps))
    a = max(1e-3, float(decel_mps2))
    return v * v / (2.0 * a)


def stopping_margin_m(speed_mps: float, *, latency_s: float,
                      a_min_mps2: float | None,
                      extra_m: float = 0.5) -> tuple[float | None, str]:
    """Conservative distance to stop, or UNKNOWN.

    ``v*latency + v^2/(2*a_min) + extra`` - the plan's starting point.
    ``a_min_mps2`` must come from a measured/declared deceleration; when it
    is missing (or non-positive) the answer is UNKNOWN, because the formula
    without it is an assumption about the road and the brakes, not a
    measurement.  Slope, tyre grip, brake build-up and the swept curve are
    all outside this model and are named in the returned reason.

    This lives here (with ``stop_distance_m``), not in ``lane``: T09's hold
    audit needs it, and a planner module importing the lane package was a
    layering inversion - measured as a real ImportError while checking the
    round-7 commits one by one.
    """
    import math

    try:
        v = float(speed_mps)
        lat = float(latency_s)
    except (TypeError, ValueError):
        return None, "speed/latency unreadable"
    if not (math.isfinite(v) and math.isfinite(lat)) or v < 0.0 or lat < 0.0:
        return None, "speed/latency invalid"
    if a_min_mps2 is None:
        return None, "no declared minimum deceleration"
    try:
        a = float(a_min_mps2)
    except (TypeError, ValueError):
        return None, "deceleration unreadable"
    if not math.isfinite(a) or a <= 0.0:
        return None, "deceleration must be positive"
    declared = abs(a - RISK_BRAKE_DECEL_MPS2) < 1e-9
    dist = v * lat + stop_distance_m(v, a) + max(0.0, float(extra_m))
    return float(dist), (
        "flat ground, dry asphalt, no brake build-up; deceleration is a "
        "declared constant" if declared else
        "flat ground, dry asphalt, no brake build-up")


def ttc_speed_cap(gap_m: float, closing_mps: float = 0.0,
                  decel_mps2: float = RISK_BRAKE_DECEL_MPS2,
                  margin_m: float = RISK_STOP_MARGIN_M,
                  reaction_s: float = RISK_REACTION_S) -> float:
    """Highest ego speed that still stops before ``gap_m``.

    ``gap_m`` is the distance to the obstacle along the driven path,
    ``closing_mps`` the rate at which it is shrinking (obstacle closing
    on the ego).  The reaction term charges the closing motion before the
    usable gap is computed; a gap already inside the margin yields 0.0,
    which the caller reads as "stop".
    """
    usable = (float(gap_m) - max(0.0, float(margin_m))
              - max(0.0, float(closing_mps)) * max(0.0, float(reaction_s)))
    if not math.isfinite(usable) or usable <= 0.0:
        return 0.0
    return math.sqrt(2.0 * max(1e-3, float(decel_mps2)) * usable)


def _inside_ego(track, pos, heading: float,
                along_m: float = EGO_SELF_ALONG_M,
                half_width: float = HALF_WIDTH_M,
                margin_m: float = EGO_SELF_MARGIN_M) -> bool:
    """True when a detection centre sits under the ego's own body.

    A LiDAR cluster or raycast return whose centre is essentially AT the
    car - inside its width, within ``EGO_SELF_ALONG_M`` of the ego centre
    along the heading - is not an object in front of it; it is the car's
    own body or the ground it stands on.  Grading it is how a car standing
    on an open road got a "hard collision at 0.02 m" verdict: measured
    2026-09-21 on the town route, 72-82 LiDAR + 17-20 raycast terrain
    returns per tick with the fused occupancy showing 13.5 m of clear path
    ahead - 37 of 55 frames stopped and the car could never move, so the
    speck never left the contact band.

    The window is deliberately much shorter than the car's own length: the
    path is anchored at the ego CENTRE, so the front bumper sits
    ``HALF_LENGTH_M`` ahead and anything beyond this window may be a real
    object just in front of the nose, which the contact band must still
    grade.  Objects from ``EGO_SELF_ALONG_M`` out are untouched.
    """
    x = float(getattr(track, "x", 0.0) or 0.0)
    y = float(getattr(track, "y", 0.0) or 0.0)
    if not (math.isfinite(x) and math.isfinite(y)):
        return False
    p = np.asarray(pos, dtype=float).ravel()[:2]
    fwd = np.array([math.cos(float(heading)), math.sin(float(heading))])
    left = np.array([-fwd[1], fwd[0]])
    rel = np.array([x, y]) - p
    return (abs(float(rel @ fwd)) <= float(along_m)
            and abs(float(rel @ left)) <= float(half_width) + float(margin_m))


def grid_has_evidence(grid) -> bool:
    """True when the fused occupancy actually detected SOMETHING.

    An all-zero grid is not evidence of free space: it is a grid that
    detected nothing (or a stub with no perception at all).  Only a grid
    that carries occupancy may contradict another sensor's contact return.
    """
    if grid is None:
        return False
    occ = getattr(grid, "obstacle", None)
    if occ is None or not getattr(occ, "size", 0):
        return False
    try:
        return bool((np.asarray(occ) > 0).any())
    except Exception:
        return False


def grid_occupied_at(grid, x: float, y: float) -> bool | None:
    """Is the fused occupancy occupied at (x, y)?  ``None`` = no evidence."""
    if grid is None:
        return None
    occ = getattr(grid, "obstacle", None)
    if occ is None or not getattr(occ, "size", 0):
        return None
    try:
        cell = grid.world_to_cell(float(x), float(y))
    except Exception:
        return None
    if cell is None:
        return None
    r, c = int(cell[0]), int(cell[1])
    if not (0 <= r < occ.shape[0] and 0 <= c < occ.shape[1]):
        return None
    return bool(occ[r, c] > 0)


def assess_obstacles(tracks, pos, heading: float,
                     ego_speed_mps: float = 0.0, *,
                     corridor_half_m: float = CORRIDOR_HALF_WIDTH_M,
                     path=None,
                     grid=None,
                     contact_band_m: float = RISK_CONTACT_BAND_M,
                     ttc_brake_s: float = RISK_TTC_BRAKE_S,
                     min_confirm_frames: int = RISK_MIN_CONFIRM_FRAMES,
                     decel_mps2: float = RISK_BRAKE_DECEL_MPS2,
                     margin_m: float = RISK_STOP_MARGIN_M,
                     reaction_s: float = RISK_REACTION_S,
                     horizon_s: float = RISK_PREDICT_HORIZON_S,
                     ) -> RiskVerdict:
    """Grade every tracked obstacle against the ego's forward corridor.

    ``tracks`` are ``temporal.TrackedObject``-shaped (``x``/``y``/``vx``/
    ``vy``/``matches``/``lost``/``category``); ``path`` is the driven
    trajectory (the corridor follows it) and defaults to the straight
    heading axis when absent.  Tracks behind the ego contribute nothing.
    """
    out = RiskVerdict()
    trs = list(tracks or ())
    if not trs:
        return out
    p = np.asarray(pos, dtype=float).ravel()[:2]
    h = float(heading)
    fwd = np.array([math.cos(h), math.sin(h)])
    left = np.array([-fwd[1], fwd[0]])
    ego_vel = fwd * max(0.0, float(ego_speed_mps))

    xy = np.array([[float(getattr(t, "x", 0.0) or 0.0),
                    float(getattr(t, "y", 0.0) or 0.0)] for t in trs],
                  dtype=float)
    if not np.isfinite(xy).all():
        return out
    if path is not None:
        pth = np.asarray(path, dtype=float)[:, :2]
        valid = np.isfinite(pth).all(axis=1)
        pth = pth[valid]
        if len(pth) < 2:
            pth = None
    else:
        pth = None
    if pth is not None:
        along, lat, tan = _project_to_path(xy, pth)
    else:
        rel = xy - p[None, :]
        along = rel @ fwd
        lat = rel @ left
        tan = np.tile(fwd, (len(xy), 1))

    def _contradicted(track, idx: int) -> bool:
        """Contact-band return the fused occupancy calls free.

        The fused occupancy is the authority the PLANNER scored the path
        against and it is temporally filtered; a terrain/vegetation return
        standing inside the band while that raster shows the cell free is
        the failure this excludes (measured 2026-09-21: 72-82 LiDAR + 17-20
        raycast terrain returns per tick, 13.5 m of clear grid path, and 37
        of 55 frames stopped on one speck - the car could not move, so the
        speck never left the band).

        A FAST object is never contradicted: the raster lags a vehicle,
        and that is exactly what the track layer adds over it.  "Fast" is
        the object's OWN speed, not the approach rate - a static wall the
        car is driving at still closes at the ego speed and must not be
        excused by that.  A grid with no occupancy evidence at all cannot
        contradict anything.
        """
        if not grid_has_evidence(grid):
            return False
        obs_speed = float(math.hypot(
            float(getattr(track, "vx", 0.0) or 0.0),
            float(getattr(track, "vy", 0.0) or 0.0)))
        if obs_speed > RISK_STATIC_OBJ_MAX_MPS:
            return False           # moving threat: never contradicted
        return grid_occupied_at(grid, xy[idx, 0], xy[idx, 1]) is False

    worst_rank = -1
    ranks = {RISK_UNKNOWN: 0, RISK_ROADSIDE: 1,
             RISK_BRAKING: 2, RISK_HARD_COLLISION: 3}
    for i, tr in enumerate(trs):
        if _inside_ego(tr, p, h):
            # A detection centred inside the car's own rectangle is not an
            # object in front of it (own body / the ground under it).
            continue
        gap = float(along[i])
        if not math.isfinite(gap) or gap <= 0.0:
            continue          # behind the ego (or unusable projection)
        lateral = float(lat[i])
        in_corridor = abs(lateral) <= float(corridor_half_m)
        obs_vel = np.array([float(getattr(tr, "vx", 0.0) or 0.0),
                            float(getattr(tr, "vy", 0.0) or 0.0)])
        rel_vel = obs_vel - ego_vel
        closing = float(-(rel_vel @ tan[i]))
        ttc = (gap / closing) if closing > RISK_MIN_CLOSING_MPS else None
        frames_seen = int(getattr(tr, "matches", 1) or 1)
        lost = int(getattr(tr, "lost", 0) or 0)
        confirmed = frames_seen >= int(min_confirm_frames) and lost == 0
        # Will it be INSIDE the corridor at any point within the horizon?
        # Sampling only the TTC instant (the first version) missed a
        # crossing vehicle that reaches the corridor slightly later: it
        # stayed "roadside clutter" until it was already in the way.  The
        # horizon is sampled in steps so the crossing is caught when it
        # happens, not only if it happens to land on the TTC.
        predicted_intrusion = False
        _horizon = max(0.0, float(horizon_s))
        _steps = np.arange(RISK_PREDICT_STEP_S, _horizon + 1e-9,
                           RISK_PREDICT_STEP_S)
        if len(_steps) == 0:
            _steps = np.asarray([_horizon])
        for _t in _steps:
            pred = (xy[i] + obs_vel * float(_t))[None, :]
            if pth is not None:
                _p_along, p_lat, _ = _project_to_path(
                    np.asarray(pred, dtype=float), pth)
                _inside = abs(float(p_lat[0])) <= float(corridor_half_m)
            else:
                rel_p = np.asarray(pred, dtype=float)[0] - p
                _inside = abs(float(rel_p @ left)) <= float(corridor_half_m)
            if _inside:
                predicted_intrusion = True
                break

        cap = float("inf")
        why = ""
        # "Can the car still stop before it?" - the unconfirmed-return
        # gate (plan C4) only defers objects the car can still stop for;
        # "已经无法在剩余距离内刹停" is an explicit no-wait case.
        stop_gap_m = stop_distance_m(ego_speed_mps, decel_mps2) + margin_m
        treat_in_corridor = bool(in_corridor or predicted_intrusion)
        if not in_corridor and not predicted_intrusion:
            # Corridor gate FIRST, before even the contact band: a tree,
            # kerb or parked car BESIDE the lane is a lane bound, and a
            # town street has them within a couple of metres constantly.
            # Ordering the contact band ahead of this gate turned every
            # such object into an immediate collision: the 2026-09-20 town
            # baseline stopped on 144/144 frames ("obstacle contact risk",
            # hard_collision) with fwd_clear 16 m and the lane paired 136
            # frames - the car never moved.  Distance alone is not danger;
            # it is distance WITHIN the driven corridor that is.
            kind = RISK_ROADSIDE
            why = "outside the driven corridor"
        elif gap <= float(contact_band_m) and not _contradicted(tr, i):
            # Inside the corridor and this close: no confirmation wait and
            # no TTC needed.
            kind = RISK_HARD_COLLISION
            cap = 0.0
            why = "inside contact band"
        elif gap <= float(contact_band_m):
            # In the contact band, but a SINGLE-FRAME return the fused
            # occupancy shows as free.  Two evidence sources disagree at
            # contact distance - the grid path is clear while a speck
            # claims a collision - and the grid is the corroborated one
            # (its own clearance over this path reads tens of metres).
            # Stopping on the speck alone parks the car permanently: it
            # never moves, so the speck never leaves the band (measured
            # 2026-09-21: 37 of 55 frames stopped with path_occ_frac 0.0
            # and fwd_clear 13.5 m).  The speck still caps the speed to
            # the creep band, so an approach that becomes real on the
            # next frame is re-graded with confirmation.
            kind = RISK_UNKNOWN
            # Creep toward it, but never closer than the stop margin: the
            # same stopping-distance bound used everywhere else.  If the
            # object IS real and the raster simply has not fused it, the
            # car halts ~``margin_m`` short of it instead of hitting it;
            # if it is a terrain/vegetation return, the car keeps moving
            # instead of parking on a phantom.
            cap = min(float(cap), ttc_speed_cap(gap, 0.0, decel_mps2,
                                                margin_m, reaction_s))
            why = "contact-band return the fused occupancy calls free"
        elif not confirmed and gap > stop_gap_m:
            # Scattered / single-frame return that the car can still stop
            # for: no speed cap may be derived from it (plan C4).
            kind = RISK_UNKNOWN
            why = f"unconfirmed ({frames_seen} frame(s))"
        elif closing > RISK_MIN_CLOSING_MPS and ttc is not None \
                and ttc <= float(ttc_brake_s):
            kind = RISK_BRAKING
            cap = ttc_speed_cap(gap, closing, decel_mps2, margin_m,
                                reaction_s)
            why = f"closing, ttc {ttc:.2f}s"
        elif treat_in_corridor:
            kind = RISK_BRAKING
            cap = ttc_speed_cap(gap, max(0.0, closing), decel_mps2,
                                margin_m, reaction_s)
            why = "stopping-distance bound"
        else:
            kind = RISK_ROADSIDE
            why = "outside the driven corridor"
        item = ObstacleRisk(
            track_id=int(getattr(tr, "track_id", -1) or -1),
            category=str(getattr(tr, "category", "object") or "object"),
            kind=kind, distance_m=gap, lateral_m=lateral,
            closing_speed_mps=closing, ttc_s=ttc,
            in_corridor=bool(in_corridor),
            predicted_intrusion=predicted_intrusion,
            frames_seen=frames_seen, confirmed=bool(confirmed),
            target_speed_cap=float(cap), why=why)
        out.items.append(item)
        out.closest_m = min(out.closest_m, gap)
        if ttc is not None:
            out.min_ttc_s = (ttc if out.min_ttc_s is None
                             else min(out.min_ttc_s, ttc))
        if kind == RISK_BRAKING:
            out.n_braking += 1
        elif kind == RISK_ROADSIDE:
            out.n_roadside += 1
        else:
            out.n_unknown += 1
        if kind == RISK_HARD_COLLISION:
            out.stop = True
        if ranks[kind] > worst_rank:
            worst_rank = ranks[kind]
            out.kind = kind
        # Only capped obstacles may lower the target: an UNKNOWN or
        # ROADSIDE item leaves it untouched.
        out.target_speed_cap = min(out.target_speed_cap, float(cap))
    return out
