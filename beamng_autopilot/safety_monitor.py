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
from dataclasses import dataclass

import numpy as np

from beamng_autopilot.planning.geometry import polyline_point_distances
from beamng_autopilot.planning.constraints import (
    body_lane_cross_dist_m, body_pose_crosses_lane,
)

# How old a range/vision snapshot can be before the monitor distrusts it.
STALE_SNAPSHOT_S = 0.8
# Fraction of path samples inside occupied cells that triggers "blocked".
# Town roads are lined by trees/curbs whose clustered boxes overlap the
# lane margin, so a 0.10 threshold made every FSD path "graze obstacle"
# and the car crawled at rule speed through town (2026-08-21 runs).  The
# corridor_free_band gate is the real "is the way clear" check; this
# fraction only flags a path genuinely weaving through clutter.
OCC_FRACTION_DEGRADE = 0.30
OCC_FRACTION_STOP = 0.40
# Lane-keep: the path must stay within this of the lane reference.
LANE_DEV_DEGRADE_M = 3.0
LANE_DEV_STOP_M = 6.0
# Obstacle-approach speed ease: only occupied cells that intrude into
# the driven corridor AHEAD of the ego count (same 1.6 m corridor as
# ``path_forward_clearance_m``).  Roadside trees/curbs beside or behind
# the car are lane bounds, not obstacles - easing to the 2 m/s creep
# for every LiDAR point within 8 m parked the car on open mountain
# roads (run 2026-08-27: plan 6 m/s, monitor crept at 2 m/s all run).
EASE_CORRIDOR_HALF_WIDTH_M = 1.6
EASE_AHEAD_MIN_M = 1.0


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

    @property
    def safe(self) -> bool:
        return self.level == "safe"

    @property
    def degraded(self) -> bool:
        return self.level == "degraded"


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

    def _lane_deviation(self, scene, path) -> float:
        """Median lateral distance of the near path from the lane ref."""
        ref = getattr(scene, "lane_ref", None)
        if ref is None or len(ref) < 2:
            envelope = getattr(scene, "lane_envelope", None)
            ref = (getattr(envelope, "center", None)
                   if envelope is not None else None)
        if ref is None or len(ref) < 2:
            ref = getattr(scene, "route", None)
        if ref is None or len(ref) < 2 or path is None or len(path) < 2:
            return 0.0
        ref = np.asarray(ref[:, :2], dtype=float)
        path = np.asarray(path, dtype=float)[:, :2]
        pos = np.asarray(scene.pos[:2], dtype=float)
        d0 = np.linalg.norm(path - pos, axis=1)
        near = path[d0 <= 25.0]
        if len(near) < 2:
            near = path[: min(4, len(path))]
        offs = polyline_point_distances(near, ref)
        return float(np.median(offs)) if len(offs) else 0.0

    # ------------------------------------------------------------------
    def evaluate(self, scene, path, closed_loop_steer: float = 0.0,
                 snapshot_age_s: float = 0.0, planner_age_s: float = 0.0
                 ) -> SafetyVerdict:
        """Arbitrate one tick.

        ``scene`` is a ``planning.Scene`` (occupancy grid + route/lane).
        ``path`` is the planner-chosen trajectory (or None when none).
        ``snapshot_age_s`` / ``planner_age_s`` are freshness of the
        sensors and the planning output.
        """
        closed_loop_steer = float(closed_loop_steer)
        path_occ = self._path_occupied_fraction(scene, path)
        lane_dev = self._lane_deviation(scene, path)
        body_cross = body_lane_cross_dist_m(scene, path)
        body_now_cross = body_pose_crosses_lane(
            scene, scene.pos, float(scene.heading))
        freshness = _perception_freshness(scene, snapshot_age_s)
        sensor_age = float(freshness["max_s"])
        stale_sensor = sensor_age > self.stale_s
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

        # --- stale sensors / planner -> degrade to minimal risk --------
        if stale_sensor or stale_planner:
            v.level = "degraded"
            v.reason = f"stale {'sensor' if stale_sensor else 'planner'}"
            v.target_speed = min(v.target_speed, self.min_risk_speed * 2.0)
            return v

        # --- path missing -> minimal risk ------------------------------
        if path is None or len(path) < 2:
            v.level = "minimal_risk"
            v.reason = "no drivable path"
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
            else:
                # Real forward blockage: keep the creep / stop reserve.
                eased = max(eased, self.min_risk_speed)
            v.target_speed = min(v.target_speed, eased)
            if not corridor_open and v.target_speed < 1.0:
                v.level = "degraded"
                v.reason = "obstacle very close"
        return v
