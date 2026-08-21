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

# How old a range/vision snapshot can be before the monitor distrusts it.
STALE_SNAPSHOT_S = 0.8
# Fraction of path samples inside occupied cells that triggers "blocked".
OCC_FRACTION_DEGRADE = 0.10
OCC_FRACTION_STOP = 0.40
# Lane-keep: the path must stay within this of the lane reference.
LANE_DEV_DEGRADE_M = 3.0
LANE_DEV_STOP_M = 6.0


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

    @property
    def safe(self) -> bool:
        return self.level == "safe"

    @property
    def degraded(self) -> bool:
        return self.level == "degraded"


class SafetyMonitor:
    """Evaluate one planning tick; arbitrate speed / stop decision."""

    def __init__(self,
                 occ_fraction_degrade: float = OCC_FRACTION_DEGRADE,
                 occ_fraction_stop: float = OCC_FRACTION_STOP,
                 lane_dev_degrade_m: float = LANE_DEV_DEGRADE_M,
                 lane_dev_stop_m: float = LANE_DEV_STOP_M,
                 stale_snapshot_s: float = STALE_SNAPSHOT_S,
                 max_speed: float = 15.0,
                 min_risk_speed: float = 2.0):
        self.occ_degrade = occ_fraction_degrade
        self.occ_stop = occ_fraction_stop
        self.lane_degrade_m = lane_dev_degrade_m
        self.lane_stop_m = lane_dev_stop_m
        self.stale_s = stale_snapshot_s
        self.max_speed = float(max_speed)
        self.min_risk_speed = float(min_risk_speed)

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
        offs = []
        for px, py in near:
            best = float("inf")
            for k in range(len(ref) - 1):
                ax, ay = ref[k]
                bx, by = ref[k + 1]
                abx, aby = bx - ax, by - ay
                l2 = abx * abx + aby * aby
                if l2 < 1e-12:
                    d = math.hypot(px - ax, py - ay)
                else:
                    t = max(0.0, min(1.0, ((px - ax) * abx +
                                            (py - ay) * aby) / l2))
                    cx, cy = ax + t * abx, ay + t * aby
                    d = math.hypot(px - cx, py - cy)
                if d < best:
                    best = d
            offs.append(best)
        return float(np.median(offs)) if offs else 0.0

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
        stale_sensor = snapshot_age_s > self.stale_s
        stale_planner = planner_age_s > self.stale_s

        closest = 999.0
        if scene.grid is not None and path is not None and len(path) > 1:
            # closest obstacle to the chosen path
            pos = np.asarray(scene.pos[:2], dtype=float)
            path = np.asarray(path, dtype=float)[:, :2]
            rr, cc = np.nonzero(scene.grid.obstacle)
            if len(rr):
                occ_pts = []
                for r, c in zip(rr, cc):
                    ex = scene.grid.max_x - (r + 0.5) * scene.grid.res
                    ey = scene.grid.max_y - (c + 0.5) * scene.grid.res
                    # back to world
                    ch = math.cos(scene.heading)
                    sh = math.sin(scene.heading)
                    wx = scene.grid.origin[0] + ex * ch - ey * sh
                    wy = scene.grid.origin[1] + ex * sh + ey * ch
                    occ_pts.append((wx, wy))
                occ_pts = np.asarray(occ_pts)
                dd = np.linalg.norm(path[:, None, :] - occ_pts[None, :, :],
                                    axis=2)
                best_path_d = float(dd.min()) if dd.size else 999.0
                closest = best_path_d

        v = SafetyVerdict(
            level="safe", reason="", target_speed=self.max_speed,
            path_occupied_frac=path_occ, lane_dev_m=lane_dev,
            closest_obs_m=closest, stale_sensor=stale_sensor,
            stale_planner=stale_planner)

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
        # Nearby roadside walls/cars ease speed; a genuinely closed
        # forward corridor (real blockage) is what degrades/stops.  A
        # narrow town street keeps FSD engaged past buildings (which are
        # normal lane bounds) while slowing, instead of dropping to the
        # rule creep on every frame (town runs 2026-08-21).
        if closest < 8.0:
            # ease speed as the closest obstacle closes in (brake band)
            k = max(0.0, 1.0 - (8.0 - closest) / 6.0)
            eased = self.max_speed * k
            if corridor_open:
                # roadside objects are lane bounds: creep, never stop
                eased = max(eased, self.min_risk_speed)
            v.target_speed = min(v.target_speed, eased)
            if not corridor_open and v.target_speed < 1.0:
                v.level = "degraded"
                v.reason = "obstacle very close"
        return v
