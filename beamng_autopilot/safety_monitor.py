"""FSD-style safety monitor: shadow health + minimal-risk fallback.

Tesla FSD runs a *safety monitor* alongside the planner: it watches the
planned trajectory against the fused occupancy, checks sensor freshness
and planning health, and when something is wrong it degrades - to a
more conservative speed, then to a minimal-risk stop.  This module gives
the project that layer as pure, game-free logic:

* ``SafetyVerdict``: Safe / Degraded(reason) / MinimalRisk(reason).
* ``SafetyMonitor``: evaluates one planning tick - the chosen path vs
  the occupancy grid, the path staying inside the lane budget, the
  sensor / planner freshness, and a maximum speed given the closest
  obstacle.  Output is the verdict plus a target speed to run at.

The existing rule planner stays the execution layer; this monitor is the
FSD-style double-check that sits on top of *any* planner output
(FSDStack's best path or the legacy route) and arbitrates the final
speed/steer.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from beamng_autopilot import config
from beamng_autopilot.planning.geometry import polyline_point_distances
from beamng_autopilot.planning.constraints import (
    body_lane_cross_dist_m, body_lane_cross_detail_m,
    body_lane_cross_recovery, body_pose_crosses_lane,
)
from beamng_autopilot.planning.lateral_ref import (
    REF_NONE, REF_SENSOR, lateral_reference,
)
from beamng_autopilot.obstacle_risk import assess_obstacles
from beamng_autopilot.vehicle_body import (
    CORRIDOR_HALF_WIDTH_M,
    HALF_LENGTH_M,
    HALF_WIDTH_M,
)

# Hard body-boundary margin.  The old gate stopped only when the swept
# rectangle had already touched/crossed the detected line; at 3-4 m/s the
# next control burst then carried the car across before the brake took
# effect (split-model live run 2026-09-18: planned cross -> two right-edge
# frames and road_off=2.37 m).  Inflate the same authoritative body by a
# margin so the candidate is rejected before the physical footprint reaches
# the boundary.  This is a safety margin around PERCEPTION geometry, not a
# lateral reference or map offset.
BODY_CROSS_MARGIN_M = 0.25

# How old a perception MODALITY (head / BEV / lane) can be before the
# monitor distrusts it.
STALE_SNAPSHOT_S = 0.8
# The range (LiDAR) modality is reusable by design: FSDStack re-serves a
# motion-compensated scan, bounded by ``RANGE_REUSE_MAX_DT_S``, whenever a
# tick overruns its budget, so its age legitimately climbs past
# STALE_SNAPSHOT_S.  A test pins this against ``fsd_stack``'s reuse cap so
# the two cannot drift apart.
STALE_RANGE_S = 2.0
# The composite snapshot age is sensor-poll + per-tick pipeline latency,
# i.e. roughly the control period, NOT a sensor-health signal: on the
# 2026-09-11 town runs it ran p50 0.58 s / max 0.92 s while no perception
# head exceeded 0.66 s.  Judged against STALE_SNAPSHOT_S it declared the
# stack stale on 217/808 ticks (27%) purely for taking a slow tick, which
# fail-closed the car into crawling (stall 150-190 frames).  It gets its
# own bound instead - above every healthy observed tick, below the Lua
# watchdog's 2.5 s heartbeat abort (``watchdog.timeout``).
STALE_PIPELINE_S = 1.5
# Fraction of path samples inside occupied cells that triggers "blocked".
# Town roads are lined by trees/curbs whose clustered boxes overlap the
# lane margin, so a 0.10 threshold made every FSD path "graze obstacle"
# and the car crawled at rule speed through town (2026-08-21 runs).  The
# corridor_free_band gate is the real "is the way clear" check; this
# fraction only flags a path genuinely weaving through clutter.
OCC_FRACTION_DEGRADE = 0.30
OCC_FRACTION_STOP = 0.40
# Lane-keep: the path must stay within this of the lane reference.
# These are the SOFT, centre-reference budgets (how far the chosen path
# may sit from the lane reference before it degrades) - the HARD gate
# is the full-body rectangle check below, which stops the car whenever
# the one authoritative footprint touches a boundary no matter how
# small the centre deviation is.  Kept separate from the footprint on
# purpose: they describe a path budget, not the size of the car.
LANE_DEV_DEGRADE_M = 3.0
LANE_DEV_STOP_M = 6.0
# Obstacle-approach speed ease: only occupied cells that intrude into
# the driven corridor AHEAD of the ego count.  Roadside trees/curbs
# beside or behind the car are lane bounds, not obstacles - easing to
# the 2 m/s creep for every LiDAR point within 8 m parked the car on
# open mountain roads (run 2026-08-27: plan 6 m/s, monitor crept at
# 2 m/s all run).
#
# The corridor is the shared body corridor (``vehicle_body``: body half
# width + detection margin = 1.6 m), never a standalone car size.
EASE_CORRIDOR_HALF_WIDTH_M = CORRIDOR_HALF_WIDTH_M
EASE_AHEAD_MIN_M = 1.0

# --- Bounded PATH_HOLD (improvement plan phase B) ---------------------
# When a tick produces NO drivable path (strict lane dropout, planner
# decline) the monitor may re-serve the last VERIFIED trajectory for a
# bounded window instead of demanding an instant full stop - that
# instant stop is what produced the stop/restart churn (2026-09-19 live:
# 17 stops in 40 s).  Grace keeps the offered target, after it the hold
# creeps at the minimal-risk speed, past the horizon the hold is CLEARED
# and the tick fails closed to a stop.  The held path is re-checked
# against the CURRENT scene (body, boundaries, occupancy) every serve,
# so only the planner output is reused - never stale perception.
PATH_HOLD_MAX_LAT_M = 2.5     # ego may not drift this far off the held path
PATH_HOLD_MIN_AHEAD_M = 4.0   # the held path must still reach this far ahead
PATH_HOLD_MIN_LEN_M = 6.0     # minimum usable offered trajectory length
# --- current/planned body-cross split (improvement plan phase C1) ----
# A PLANNED sweep crossing detected at least config.FSD_PLANNED_CROSS_
# HARD_M along the path is a far-field risk: degrade (cap speed, let the
# next tick re-plan) instead of the instant full stop the old gate
# applied at ANY crossing distance - a far-end sampling artefact of the
# boundary fit must not stand the car dead.  Near-field planned
# crossings and CURRENT body crossings keep the hard stop.


@dataclass
class SafetyVerdict:
    """The monitor's arbitration result."""

    level: str = "safe"            # "safe" | "degraded" | "minimal_risk"
    reason: str = ""
    target_speed: float = 0.0
    # structured diagnostics
    path_occupied_frac: float = 0.0
    lane_dev_m: float = 0.0
    closest_obs_m: float = 999.0
    stale_sensor: bool = False
    stale_planner: bool = False
    sensor_age_s: float = 0.0
    head_age_s: dict | None = None
    bev_age_s: float | None = None
    lane_age_s: float | None = None
    range_age_s: float | None = None
    corridor_open: bool = True
    # Which lateral reference the lane-keep check used: "sensor" /
    # "envelope" (perception), "route" (legacy map fallback only) or
    # "none".  Telemetry evidence for the FSD realism contract.
    lane_ref_src: str = "none"
    # Bounded PATH_HOLD diagnostics (plan phase B).  ``held_path`` is the
    # re-served trajectory for the drive loop; it is deliberately NOT
    # telemetry-serialised (only its scalars are).
    path_hold_active: bool = False
    path_hold_age_s: float | None = None
    path_hold_phase: str = ""
    held_path: np.ndarray | None = None
    # Structured body-boundary diagnostics (plan phase C1): the
    # current-pose crossing and the planned-sweep crossing are DIFFERENT
    # events with different responses (stop now vs slow down and
    # re-plan), and the planned one carries where on the path it
    # happens.  ``boundary_confidence`` is intentionally absent - the
    # detected boundaries carry no confidence value today and inventing
    # one would fake certainty the sensors did not provide.
    body_cross_current: bool = False
    body_cross_planned: bool = False
    first_crossing_distance_m: float | None = None
    crossing_path_index: int | None = None
    crossing_boundary_side: str = ""
    # Obstacle risk grading (plan phase C3): the worst class seen this
    # tick, the closest confirmed time-to-collision, and the distance of
    # the nearest graded obstacle.  ``risk_kind`` is "" when no track was
    # graded at all (no dynamic perception this tick).
    risk_kind: str = ""
    min_ttc_s: float | None = None
    risk_closest_m: float | None = None

    @property
    def safe(self) -> bool:
        return self.level == "safe"

    @property
    def degraded(self) -> bool:
        return self.level == "degraded"

    @property
    def drivable(self) -> bool:
        """Whether this verdict's path may still be driven.

        ``degraded`` is drivable BY DEFINITION - the monitor computed the
        reduced speed cap (``target_speed``) for exactly that case - so
        only ``minimal_risk`` refuses the path.  Callers that gate on
        ``safe`` instead throw the cap away and stop the car: the
        2026-09-18 live east_coast demo stopped on 122 of 222 ticks, 83 of
        them ``level=degraded`` with ``mon_target`` 3.30 m/s and an open
        corridor, because strict mode has no rule backup to fall through.
        """
        return self.level != "minimal_risk"


@dataclass
class HeldPath:
    """One verified trajectory offered to the bounded hold."""

    path: np.ndarray          # (N, 2) world XY, ego-anchored near->far
    heading: float
    target_speed: float
    offered_at: float
    strict: bool


class PathHold:
    """Bounded reuse of the last verified-safe trajectory (PATH_HOLD).

    Pure time/geometry logic: offer validation, the age phases and the
    ego-consistency bounds.  The scene-dependent re-checks (current body
    crossing, boundary crossing of the held path, occupancy) stay in
    :class:`SafetyMonitor`, which owns the perception data they need.
    An expired hold is refused AND cleared, so a stale trajectory can
    never be revived after its horizon; only a fresh verified offer can
    restart motion, which is the re-start confirmation the plan requires.
    """

    def __init__(self,
                 grace_s: float | None = None,
                 max_s: float | None = None,
                 max_lat_m: float = PATH_HOLD_MAX_LAT_M,
                 min_ahead_m: float = PATH_HOLD_MIN_AHEAD_M,
                 min_len_m: float = PATH_HOLD_MIN_LEN_M):
        self.grace_s = float(grace_s if grace_s is not None
                             else config.FSD_PATH_HOLD_GRACE_S)
        self.max_s = float(max_s if max_s is not None
                           else config.FSD_PATH_HOLD_MAX_S)
        self.max_lat_m = float(max_lat_m)
        self.min_ahead_m = float(min_ahead_m)
        self.min_len_m = float(min_len_m)
        self._held: HeldPath | None = None

    def offer(self, path, heading: float, target_speed: float, *,
              now_s: float, strict: bool = False) -> bool:
        """Store a verified trajectory (any previous one is replaced).

        Only callers that JUST verified the path against the current
        scene (drivable verdict, fresh sensors, perception-derived
        geometry) may offer; this method only rejects structurally
        unusable polylines.
        """
        if path is None:
            return False
        pts = np.asarray(path, dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 2:
            return False
        pts = pts[:, :2]
        if not np.isfinite(pts).all():
            return False
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        if float(seg.sum()) < self.min_len_m:
            return False
        self._held = HeldPath(
            path=pts.copy(), heading=float(heading),
            target_speed=max(0.0, float(target_speed)),
            offered_at=float(now_s), strict=bool(strict))
        return True

    def request(self, pos, now_s: float):
        """Serve the held path when still inside its bounded window.

        Returns ``(held, age_s, phase)`` with ``held`` the :class:`HeldPath`
        and phase ``"grace"`` (keep the offered target) or ``"creep"``
        (decay to the minimal-risk speed), or None.  None also CLEARS an
        expired hold.
        """
        held = self._held
        if held is None:
            return None
        age = max(0.0, float(now_s) - held.offered_at)
        if age > self.max_s:
            self._held = None
            return None
        p = np.asarray(pos, dtype=float).ravel()[:2]
        pts = held.path
        d = np.linalg.norm(pts - p[None, :], axis=1)
        j = int(np.argmin(d))
        if float(d[j]) > self.max_lat_m:
            return None
        arc = np.concatenate([[0.0], np.cumsum(
            np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        if float(arc[-1] - arc[j]) < self.min_ahead_m:
            return None
        phase = "grace" if age <= self.grace_s else "creep"
        return held, age, phase

    def clear(self) -> None:
        self._held = None

    @property
    def active(self) -> bool:
        return self._held is not None

    @property
    def target_speed(self) -> float:
        return self._held.target_speed if self._held is not None else 0.0


def _corridor_ahead_distance(occ_pts, path, half_width_m: float,
                           ahead_min_m: float) -> float | None:
    """Along-path distance of the nearest occupied cell that intrudes
    into the path corridor AHEAD of the ego (None when nothing does).

    The old ``closest`` distance counted every occupied cell within 8 m
    of the path regardless of where it sat laterally or longitudinally,
    so continuous roadside clutter kept the target at the 2 m/s creep.
    Only cells within ``half_width_m`` of the path and at least
    ``ahead_min_m`` along it (the path is ego-anchored) can ease speed.
    """
    pts = np.asarray(occ_pts, dtype=float)
    poly = np.asarray(path, dtype=float)[:, :2]
    if len(pts) == 0 or len(poly) < 2:
        return None
    a = poly[:-1]
    b = poly[1:]
    ab = b - a
    l2 = np.einsum("ij,ij->i", ab, ab)
    seg_len = np.sqrt(np.maximum(l2, 1e-12))
    arc0 = np.concatenate([[0.0], np.cumsum(seg_len)])[:-1]
    rel = pts[:, None, :] - a[None, :, :]
    t = np.clip(np.einsum("ijk,jk->ij", rel, ab)
                / np.maximum(l2[None, :], 1e-12), 0.0, 1.0)
    proj = a[None, :, :] + t[..., None] * ab[None, :, :]
    lat = np.linalg.norm(pts[:, None, :] - proj, axis=2)
    along = arc0[None, :] + t * seg_len[None, :]
    j = np.argmin(lat, axis=1)
    lat_best = lat[np.arange(len(pts)), j]
    along_best = along[np.arange(len(pts)), j]
    sel = (lat_best <= half_width_m) & (along_best >= ahead_min_m)
    if not np.any(sel):
        return None
    return float(np.min(along_best[sel]))


def _perception_freshness(scene, snapshot_age_s: float) -> dict:
    """Collect one age contract from Scene metadata and lane envelope.

    When a canonical ``PerceptionSnapshot`` is present it is authoritative;
    the metadata path remains for legacy/mock scenes.

    Missing/None ages are unknown rather than fresh.  The caller can use
    the returned ``max_s`` for a single stale decision while retaining the
    individual values for telemetry.
    """
    snapshot = getattr(scene, "perception_snapshot", None)
    if snapshot is not None:
        fresh = snapshot.freshness()
        return {
            "head_age_s": dict(snapshot.head_age_s),
            "head_max_s": fresh.get("head_max_s"),
            "range_age_s": fresh.get("range_s"),
            "bev_age_s": fresh.get("bev_s"),
            "lane_age_s": fresh.get("lane_s"),
            "max_s": max(float(snapshot_age_s or 0.0),
                         float(fresh.get("max_s", 0.0))),
        }

    meta = getattr(scene, "meta", {}) or {}
    raw_heads = meta.get("head_age_s", {}) or {}
    heads = {str(k): (None if v is None else float(v))
             for k, v in raw_heads.items()}
    vals = [float(snapshot_age_s)] if snapshot_age_s is not None else []
    if any(v is None for v in heads.values()):
        vals.append(float("inf"))
    else:
        vals.extend(heads.values())
    range_age = meta.get("range_age_s")
    bev_age = meta.get("bev_age_s")
    if range_age is None and "range_age_s" in meta:
        vals.append(float("inf"))
    elif range_age is not None:
        vals.append(float(range_age))
    if bev_age is None and "bev_age_s" in meta:
        vals.append(float("inf"))
    elif bev_age is not None:
        vals.append(float(bev_age))
    lane = getattr(scene, "lane_envelope", None)
    lane_age = None
    if lane is not None:
        lane_age = float(lane.age_s)
        vals.append(lane_age)
    return {"head_age_s": heads,
            "head_max_s": (max(heads.values()) if heads
                           and all(v is not None for v in heads.values())
                           else None),
            "range_age_s": (None if range_age is None else float(range_age)),
            "bev_age_s": (None if bev_age is None else float(bev_age)),
            "lane_age_s": lane_age,
            "max_s": max(vals, default=0.0)}


def _modality_age(freshness: dict, snapshot_age_s: float | None) -> float:
    """Age of the required road-driving modalities.

    ``head_age_s`` also contains optional heads (object / traffic / topology).
    Those heads may be throttled or reused without invalidating a fresh road
    lane: obstacle safety has its own range/BEV checks.  Only the semantic head
    is required for the road-lateral decision; the full max remains available
    in ``freshness['max_s']`` for honest telemetry.
    """
    vals: list[float] = []
    heads = freshness.get("head_age_s") or {}
    if heads:
        value = heads.get("semantic")
        vals.append(float("inf") if value is None else float(value))
    for key in ("bev_age_s", "lane_age_s"):
        value = freshness.get(key)
        if value is not None:
            vals.append(float(value))
    if not vals:
        vals.append(float(snapshot_age_s or 0.0))
    return max(vals)


class SafetyMonitor:
    """Evaluate one planning tick; arbitrate speed / stop decision."""

    def __init__(self,
                 occ_fraction_degrade: float = OCC_FRACTION_DEGRADE,
                 occ_fraction_stop: float = OCC_FRACTION_STOP,
                 lane_dev_degrade_m: float = LANE_DEV_DEGRADE_M,
                 lane_dev_stop_m: float = LANE_DEV_STOP_M,
                 stale_snapshot_s: float = STALE_SNAPSHOT_S,
                 max_speed: float = 15.0,
                 min_risk_speed: float = 2.0,
                 corridor_open_floor_frac: float = 0.55):
        self.occ_degrade = occ_fraction_degrade
        self.occ_stop = occ_fraction_stop
        self.lane_degrade_m = lane_dev_degrade_m
        self.lane_stop_m = lane_dev_stop_m
        self.stale_s = stale_snapshot_s
        self.max_speed = float(max_speed)
        self.min_risk_speed = float(min_risk_speed)
        # When the forward corridor is verified OPEN, roadside clutter
        # only eases the target to this fraction of cruise (never the
        # minimal-risk creep); a closed corridor still creeps/stops.
        self.corridor_open_floor = float(corridor_open_floor_frac)
        # Bounded PATH_HOLD state (plan phase B): the last verified
        # trajectory, re-servable inside its hold window when a tick
        # loses the path.  Replaced wholesale by every fresh verified
        # offer; cleared when it expires or fails a current-scene check.
        self.path_hold = PathHold()

    # ------------------------------------------------------------------
    def offer_verified_path(self, path, heading: float,
                            target_speed: float, *,
                            now_s: float | None = None,
                            strict: bool = False) -> bool:
        """Cache a drivable, freshly-verified trajectory for bounded reuse.

        Callers must offer ONLY perception-derived paths that just passed
        this monitor's checks on fresh sensors (never map/rule fallbacks
        and never stale-sensor ticks) - the hold replays exactly that
        evidence, never a lateral reference the iron rule forbids.
        """
        return self.path_hold.offer(
            path, float(heading), float(target_speed),
            now_s=(time.time() if now_s is None else float(now_s)),
            strict=strict)

    def _serve_hold(self, scene, now_s: float):
        """Re-check the held path against the CURRENT scene, then serve.

        The held trajectory was verified when it was offered; the world
        has moved on, so every check that does not need the (missing)
        planner output is re-run before it may be driven again: the
        current body must not cross a boundary, the held path must not
        now cross a detected boundary, and the occupancy grid must not
        have gone blocked along it.  Returns
        ``(held_path, age_s, phase, target_speed)`` or None.
        """
        req = self.path_hold.request(
            np.asarray(scene.pos[:2], dtype=float), now_s)
        if req is None:
            return None
        held, age, phase = req
        half_len = HALF_LENGTH_M + BODY_CROSS_MARGIN_M
        half_width = HALF_WIDTH_M + BODY_CROSS_MARGIN_M
        if body_pose_crosses_lane(scene, scene.pos, float(scene.heading),
                                  half_len=half_len, half_width=half_width):
            return None
        if body_lane_cross_dist_m(scene, held.path, half_len=half_len,
                                  half_width=half_width) > 0.0:
            return None
        if self._path_occupied_fraction(scene, held.path) >= self.occ_stop:
            return None
        cap = (min(held.target_speed, self.max_speed) if phase == "grace"
               else min(held.target_speed, self.min_risk_speed))
        return held.path, age, phase, max(0.0, cap)

    # ------------------------------------------------------------------
    def _path_occupied_fraction(self, scene, path) -> float:
        if scene.grid is None or path is None or len(path) < 2:
            return 0.0
        path = np.asarray(path, dtype=float)[:, :2]
        pos = np.asarray(scene.pos[:2], dtype=float)
        extent = float(getattr(scene.grid, "extent", 0.0) or 0.0)
        bad = 0
        total = 0
        for x, y in path:
            d = math.hypot(x - pos[0], y - pos[1])
            if d < 2.5:
                continue
            # Beyond the sensor/FOV horizon the world is *unknown*, not
            # blocked. A long nav-route reference must not read as if it
            # were driving through a wall out of sensor range; the forward
            # corridor check owns the "wall ahead" verdict.
            if extent > 0.0 and d > extent:
                continue
            total += 1
            cell = scene.grid.world_to_cell(x, y)
            if cell is None:
                continue
            if scene.grid.obstacle[cell] > 0:
                bad += 1
        return (bad / total) if total else 0.0

    @staticmethod
    def _lane_reference(scene):
        """The lateral lane reference for this tick.

        Delegates to the single lateral-reference policy
        (``planning.lateral_ref``): perception first, the nav route only
        as the documented legacy fallback, and nothing at all in a strict
        scene with no sensor lane - so the caller fails closed instead of
        measuring against the map centre line.
        """
        return lateral_reference(scene)

    def _lane_deviation(self, scene, path) -> tuple[float, str]:
        """(median lateral distance from the lane ref, ref source).

        The source is ``REF_NONE`` when there is no reference to measure
        against at all; a strict scene with no perception lane yields
        that, and ``evaluate`` turns it into a minimal-risk stop.
        """
        ref, src = self._lane_reference(scene)
        if ref is None or path is None or len(path) < 2:
            return 0.0, src
        path = np.asarray(path, dtype=float)[:, :2]
        pos = np.asarray(scene.pos[:2], dtype=float)
        d0 = np.linalg.norm(path - pos, axis=1)
        near = path[d0 <= 25.0]
        if len(near) < 2:
            near = path[: min(4, len(path))]
        offs = polyline_point_distances(near, ref)
        if not len(offs):
            return 0.0, src
        return float(np.median(offs)), src

    # ------------------------------------------------------------------
    def evaluate(self, scene, path, closed_loop_steer: float = 0.0,
                 snapshot_age_s: float = 0.0, planner_age_s: float = 0.0,
                 now_s: float | None = None,
                 ego_speed_mps: float = 0.0) -> SafetyVerdict:
        """Arbitrate one tick.

        ``scene`` is a ``planning.Scene`` (occupancy grid + route/lane).
        ``path`` is the planner-chosen trajectory (or None when none).
        ``snapshot_age_s`` / ``planner_age_s`` are freshness of the
        sensors and the planning output.  ``now_s`` pins the clock for
        the bounded PATH_HOLD phases (tests pass an explicit value; live
        callers use the wall clock).  ``ego_speed_mps`` feeds the
        obstacle TTC model (closing speed is relative to the ego).

        The obstacle risk layer is applied LAST, on whatever verdict the
        core arbitration produced: a degraded branch (scattered clutter,
        a served path hold) may still be driving toward a confirmed
        closing obstacle, and an early return must not skip that check.
        Only a verdict that is already ``minimal_risk`` has nothing left
        to cap.
        """
        v = self._evaluate_core(scene, path, closed_loop_steer,
                                snapshot_age_s, planner_age_s, now_s)
        return self._apply_obstacle_risk(v, scene, path,
                                         float(ego_speed_mps))

    def _apply_obstacle_risk(self, v: SafetyVerdict, scene, path,
                             ego_speed_mps: float) -> SafetyVerdict:
        """Grade tracked obstacles and fold the result into ``v``.

        The occupancy grid says WHERE obstacles are; tracked objects
        carry the velocity it cannot, so each is graded (hard collision /
        braking / roadside / unknown) and a stopping-distance speed cap
        is derived from the closing motion.  Roadside clutter and
        unconfirmed specks cap nothing, and the contact band does not
        wait for confirmation (plan phases C3/C4).
        """
        tracks = getattr(getattr(scene, "perception_snapshot", None),
                         "tracks", None) or []
        risk = assess_obstacles(
            tracks, scene.pos, float(scene.heading), float(ego_speed_mps),
            corridor_half_m=EASE_CORRIDOR_HALF_WIDTH_M,
            path=(path if path is not None and len(path) >= 2 else None))
        if risk.items:
            v.risk_kind = str(risk.kind)
            v.min_ttc_s = risk.min_ttc_s
            v.risk_closest_m = (None if not math.isfinite(risk.closest_m)
                                else float(risk.closest_m))
        if v.level == "minimal_risk":
            return v
        if risk.stop:
            v.level = "minimal_risk"
            v.reason = "obstacle contact risk"
            v.target_speed = 0.0
            return v
        if math.isfinite(risk.target_speed_cap):
            v.target_speed = min(v.target_speed,
                                 float(risk.target_speed_cap))
            if v.target_speed <= 0.0 and v.level == "safe":
                v.level = "degraded"
                v.reason = "obstacle stopping distance"
        return v

    def _evaluate_core(self, scene, path, closed_loop_steer: float = 0.0,
                       snapshot_age_s: float = 0.0,
                       planner_age_s: float = 0.0,
                       now_s: float | None = None) -> SafetyVerdict:
        """The layered arbitration proper (see :meth:`evaluate`)."""
        closed_loop_steer = float(closed_loop_steer)
        path_occ = self._path_occupied_fraction(scene, path)
        lane_dev, lane_ref_src = self._lane_deviation(scene, path)
        body_cross, cross_idx, cross_side = body_lane_cross_detail_m(
            scene, path,
            half_len=HALF_LENGTH_M + BODY_CROSS_MARGIN_M,
            half_width=HALF_WIDTH_M + BODY_CROSS_MARGIN_M)
        body_now_cross = body_pose_crosses_lane(
            scene, scene.pos, float(scene.heading),
            half_len=HALF_LENGTH_M + BODY_CROSS_MARGIN_M,
            half_width=HALF_WIDTH_M + BODY_CROSS_MARGIN_M)
        freshness = _perception_freshness(scene, snapshot_age_s)
        # ``sensor_age`` stays the honest max over every modality for
        # telemetry; the stale decision separates three different things:
        # the live modalities, the reusable range, and the composite
        # pipeline latency (see the constants above).
        sensor_age = float(freshness["max_s"])
        range_age = freshness.get("range_age_s")
        stale_sensor = (
            _modality_age(freshness, snapshot_age_s) > self.stale_s
            or (range_age is not None
                and float(range_age) > STALE_RANGE_S)
            or float(snapshot_age_s or 0.0) > STALE_PIPELINE_S)
        stale_planner = planner_age_s > self.stale_s

        closest = 999.0
        if scene.grid is not None and path is not None and len(path) > 1:
            # nearest corridor-intruding obstacle AHEAD of the ego, not
            # any cell near the path (roadside clutter must not creep
            # the target speed - run 2026-08-27)
            pos = np.asarray(scene.pos[:2], dtype=float)
            path = np.asarray(path, dtype=float)[:, :2]
            rr, cc = np.nonzero(scene.grid.obstacle)
            if len(rr):
                # vectorised: grid cell -> world (same formula as the old
                # per-cell Python loop, but one numpy pass)
                ch = math.cos(getattr(scene.grid, "heading",
                                      scene.heading))
                sh = math.sin(getattr(scene.grid, "heading",
                                      scene.heading))
                ex = scene.grid.max_x - (rr + 0.5) * scene.grid.res
                ey = scene.grid.max_y - (cc + 0.5) * scene.grid.res
                wx = scene.grid.origin[0] + ex * ch - ey * sh
                wy = scene.grid.origin[1] + ex * sh + ey * ch
                occ_pts = np.stack([wx, wy], axis=1)
                _ahead = _corridor_ahead_distance(
                    occ_pts, path, EASE_CORRIDOR_HALF_WIDTH_M,
                    EASE_AHEAD_MIN_M)
                if _ahead is not None:
                    closest = _ahead

        v = SafetyVerdict(
            level="safe", reason="", target_speed=self.max_speed,
            path_occupied_frac=path_occ, lane_dev_m=lane_dev,
            closest_obs_m=closest, stale_sensor=stale_sensor,
            stale_planner=stale_planner,
            sensor_age_s=sensor_age,
            head_age_s=freshness["head_age_s"],
            bev_age_s=freshness["bev_age_s"],
            lane_age_s=freshness["lane_age_s"],
            range_age_s=freshness["range_age_s"])
        v.lane_ref_src = lane_ref_src
        # Structured boundary diagnostics ride EVERY verdict (plan C1):
        # the current-pose and planned-sweep crossings are separate
        # events, and the planned one reports where on the path it
        # happens instead of only "crossed".
        v.body_cross_current = bool(body_now_cross)
        v.body_cross_planned = bool(body_cross > 0.0)
        if body_cross > 0.0:
            v.first_crossing_distance_m = float(body_cross)
            v.crossing_path_index = int(cross_idx)
            v.crossing_boundary_side = str(cross_side)

        # --- stale sensors / planner -> degrade to minimal risk --------
        if stale_sensor or stale_planner:
            v.level = "degraded"
            v.reason = f"stale {'sensor' if stale_sensor else 'planner'}"
            v.target_speed = min(v.target_speed, self.min_risk_speed * 2.0)
            return v

        # --- path missing -> bounded hold, else minimal risk -----------
        # A single frame without a path used to demand an instant full
        # stop, which produced the stop/restart churn of short perception
        # dropouts (17 stops in 40 s, 2026-09-19 live).  A path the
        # monitor JUST verified may be re-served for the bounded hold
        # window - re-checked against the CURRENT scene first - and the
        # hold decays to a creep before failing closed to a stop.
        if path is None or len(path) < 2:
            served = self._serve_hold(
                scene, time.time() if now_s is None else float(now_s))
            if served is not None:
                held_path, hold_age, hold_phase, hold_cap = served
                v.level = "degraded"
                v.reason = f"path hold ({hold_phase})"
                v.target_speed = hold_cap
                v.path_hold_active = True
                v.path_hold_age_s = float(hold_age)
                v.path_hold_phase = str(hold_phase)
                v.held_path = held_path
                return v
            v.level = "minimal_risk"
            v.reason = "no drivable path"
            v.target_speed = 0.0
            return v

        # --- strict perception: no sensor lane -> fail closed -----------
        # A real FSD does not keep driving off the HD map when it cannot
        # see the lane; it degrades.  Never measure lateral position
        # against the nav route in strict mode.
        if lane_ref_src == REF_NONE \
                and getattr(scene, "strict_perception", False):
            v.level = "minimal_risk"
            v.reason = "perception lane unavailable"
            v.target_speed = 0.0
            return v

        # --- occupancy --------------------------------------------------
        corridor_open = False
        if scene.grid is not None:
            from beamng_autopilot.planning import corridor_free_band
            try:
                corridor_open = corridor_free_band(scene)
            except Exception:
                corridor_open = False
        v.corridor_open = corridor_open
        if path_occ >= self.occ_stop and not corridor_open:
            v.level = "minimal_risk"
            v.reason = "path blocked by obstacle"
            v.target_speed = 0.0
            return v
        if path_occ >= self.occ_degrade:
            if corridor_open:
                # Connectivity says a free lateral band exists: scattered
                # roadside/guardrail occupancy is a soft speed cap, not a
                # path-graze failure.  A closed corridor still takes the
                # hard path-graze branch below.
                v.level = "degraded"
                v.reason = "scattered obstacle"
                v.target_speed = min(
                    v.target_speed,
                    self.max_speed * self.corridor_open_floor)
                return v
            v.level = "degraded"
            v.reason = "path grazes obstacle"
            v.target_speed = min(v.target_speed,
                                 self.min_risk_speed * 2.5)
            return v

        # --- lane keep --------------------------------------------------
        # Full-body envelope is a hard safety condition: a path whose
        # centre is inside but whose projected corner crosses a boundary
        # must stop before steering it (not merely degrade its speed).
        if body_now_cross or body_cross > 0.0:
            if body_now_cross and lane_ref_src == REF_SENSOR \
                    and body_lane_cross_recovery(
                        scene, path,
                        half_len=HALF_LENGTH_M + BODY_CROSS_MARGIN_M,
                        half_width=HALF_WIDTH_M + BODY_CROSS_MARGIN_M):
                # The car already sits across the line.  The current-pose
                # flag alone refuses EVERY path - including the one that
                # steers it back - so the car froze in the crossing and
                # the stuck detector then armed a reverse escape (live
                # east_coast 2026-09-18).  A path that CONVERGES is the
                # legal recovery, exactly as the centreline gate already
                # allows (``lane_cross_dist_m``); it is capped to a creep,
                # and a path that keeps or deepens the crossing still
                # takes the hard stop below.
                v.level = "degraded"
                v.reason = "lane boundary recovery"
                v.target_speed = min(v.target_speed,
                                     self.min_risk_speed * 0.5)
                return v
            if not body_now_cross \
                    and body_cross >= config.FSD_PLANNED_CROSS_HARD_M:
                # Far-field PLANNED crossing (plan C1): the current body
                # is clean and the first violation is still ahead - cap
                # the speed and let the next tick re-plan instead of a
                # full stop.  The crossing distance shrinks as the car
                # advances, so a persistent crossing still converges to
                # the hard stop below; only a far-end boundary-fit
                # artefact gets smoothed out.
                v.level = "degraded"
                v.reason = "planned boundary crossing ahead"
                v.target_speed = min(v.target_speed,
                                     self.min_risk_speed * 2.0)
                return v
            v.level = "minimal_risk"
            v.reason = ("current vehicle body crosses lane boundary"
                        if body_now_cross
                        else "planned vehicle body crosses lane boundary")
            v.target_speed = 0.0
            return v
        if lane_dev >= self.lane_stop_m:
            v.level = "minimal_risk"
            v.reason = "path off-lane"
            v.target_speed = 0.0
            return v
        if lane_dev >= self.lane_degrade_m:
            v.level = "degraded"
            v.reason = "path near lane edge"
            v.target_speed = min(v.target_speed, self.min_risk_speed * 3.0)
            return v

        # --- obstacle approach speed ------------------------------------
        # A corridor-intruding obstacle AHEAD of the ego eases speed; a
        # genuinely closed forward corridor (real blockage) is what
        # degrades/stops.  Roadside walls/trees beside the lane are lane
        # bounds and never touch this band (they used to pin the car to
        # the 2 m/s creep on every tree-lined road - run 2026-08-27).
        if closest < 8.0:
            # ease speed as the closest obstacle closes in (brake band)
            k = max(0.0, 1.0 - (8.0 - closest) / 6.0)
            eased = self.max_speed * k
            if corridor_open:
                # Roadside objects are lane bounds: ease, but never
                # crawl.  The planner verified a free band exists, so
                # dense intersection LiDAR must not drop the target to
                # the 2 m/s creep and stall the car (fsd opt21 t=54-60:
                # v 5.3 -> 0.1 -> 3.0 with plan 6.0, junction clutter).
                eased = max(eased, self.max_speed * self.corridor_open_floor)
                # ... except in the last few metres before contact: the
                # open-corridor floor (3.3 m/s at cruise 6) kept the car
                # at full ease speed until it physically brushed a
                # guardrail box at closest 1.2 m (two contact-and-relaunch
                # stalls, east_coast 2026-09-19).  Inside the contact band
                # the proximity ease wins, floored at the minimal-risk
                # speed so the car keeps rolling instead of stalling.
                if closest < 3.0:
                    eased = max(min(eased, self.max_speed * k),
                                self.min_risk_speed)
            else:
                # Real forward blockage: keep the creep / stop reserve.
                eased = max(eased, self.min_risk_speed)
            v.target_speed = min(v.target_speed, eased)
            if not corridor_open and v.target_speed < 1.0:
                v.level = "degraded"
                v.reason = "obstacle very close"
        return v
