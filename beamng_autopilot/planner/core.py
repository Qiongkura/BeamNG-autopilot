"""Local obstacle-aware path planner and speed planner."""

from __future__ import annotations

import heapq
import math
import time

import numpy as np

from .constants import (
    CAR_HALF_WIDTH, CORRIDOR_HALF_W, DECEL_MPS2, DEV_PENALTY, GRID_AHEAD,
    GRID_ANTICIPATE, GRID_BEHIND, GRID_HALF_W, GRID_RES, GRID_RIGHT_BIAS,
    LANE_BOUNDARY_CLEAR_M, LANE_BOUNDARY_CORRECTION_MAX_M, LANE_BOUNDARY_MAX_M,
    LANE_EDGE_NAV_MAX_DEV_M, LANE_EDGE_NAV_MIN_SIGN_M,
    LANE_EDGE_PULL_MAX_M, LANE_EDGE_RIGHT_MAX_DEV_M,
    LANE_EDGE_RIGHT_VISION_MAX_DEV_M, LANE_FULL_CONF,
    LANE_LIDAR_CORRECTION_MAX_M, LANE_LIDAR_EDGE_CORRECTION_MAX_M,
    LANE_NAV_MAX_DEV_M, LANE_WIDTH_DEFAULT_M, LIDAR_PATH_CLEAR_M,
    MAX_LATERAL_DEV, PASS_BY_MIN_MPS, PLAN_HORIZON_M, RIGHT_OFFSET_M,
    RIGHT_RAMP_M, ROADSIDE_WALL_MIN_EDGE_M, ROADSIDE_WALL_MIN_LEN_M,
    ROADSIDE_WALL_MAX_THICK_M, SAFETY_MARGIN, SHARP_ANGLE_DEG,
    SHARP_CORNER_KPH, SOLID_BLOCK_LANE_CONF, SOLID_LINE_MARGIN,
    SPECK_PASS_BY_MIN_MPS, STOP_MARGIN_M, _MapLaneBoundary,
)
from .geometry import (
    _clamp_path_lateral, _lane_correction_gain, _point_lat_offset,
    _point_route_pos_np, _points_to_polyline_lat, _pts_to_segments,
    _route_spacing_m_impl, _smoothstep, corner_angle_deg,
    corner_angle_max_deg, corner_speed,
)
from .obstacles import (
    _find_blocker, _obstacle_aabb, _obstacle_corners, _obstacle_footprint_area,
    _obstacle_half_extents, _obstacle_oriented, _obstacle_seg_dist,
    _path_clear_m, _path_hit_index, _seg_hits_obstacle,
    _vehicle_speed_along, is_small_lidar_clutter, is_sparse_raycast_speck,
)
from .solid import _clamp_to_solid_lines, is_lane_edge_wall

from beamng_autopilot.lane import (
    LANE_MIN_CONF,
    LANE_WIDTH_MAX_M,
    lane_frame_usable,
)
from beamng_autopilot.traffic import (
    RoadRuleView,
    legal_lane_view,
    road_width_m,
)

class LocalPlanner:
    """Plans a locally drivable path around obstacles and a safe speed."""

    def __init__(
        self,
        horizon_m: float = PLAN_HORIZON_M,
        max_dev: float = MAX_LATERAL_DEV,
        corridor_half_w: float = CORRIDOR_HALF_W,
        margin: float = SAFETY_MARGIN,
        relax_iters: int = 15,
        smooth_passes: int = 3,
        push_gain: float = 0.45,
        lateral_clear: float = 0.6,
        anticipate: float = 5.0,
        right_offset: float = RIGHT_OFFSET_M,
        right_ramp_m: float = RIGHT_RAMP_M,
        sharp_angle_deg: float = SHARP_ANGLE_DEG,
        sharp_corner_kph: float = SHARP_CORNER_KPH,
        grid_right_bias: float = GRID_RIGHT_BIAS,
    ):
        self.horizon_m = horizon_m
        self.max_dev = max_dev
        self.corridor_half_w = corridor_half_w
        self.margin = margin
        self.relax_iters = relax_iters
        self.smooth_passes = smooth_passes
        self.push_gain = push_gain
        # Extra clearance kept beyond the car half width when the elastic
        # band nudges the path around a nearby obstacle.
        self.lateral_clear = lateral_clear
        self.anticipate = anticipate
        self.right_offset = right_offset
        self.right_ramp_m = right_ramp_m
        self.sharp_angle_deg = sharp_angle_deg
        self.sharp_corner_kph = sharp_corner_kph
        self.grid_right_bias = grid_right_bias
        # Obstacles whose center sits this close (laterally) to the original
        # navigation route are treated as "blockers in the lane": the car
        # eases off as it approaches them.  Roadside poles/curbs 3 m off the
        # route do not qualify.
        self.blocker_lat = 2.6
        # Last planning outcome for HUD/telemetry: "follow" (nav route as-is),
        # "deform" (elastic band nudge), "detour" (A* around a blocker) or
        # "blocked" (no drivable way; stop in front).
        self.last_mode = "follow"
        # Last lane-reference source used by plan(): "nav" (the route
        # centre, no right bias by default) or the first entry of the
        # LaneFrame ``sources`` tuple ("vision" / "lidar").
        self.last_lane_mode = "nav"
        # Median lateral path offset applied from the nav route when a
        # single lane boundary (right line / wall / guardrail) defined the
        # drive path.  Diagnostics only.
        self.last_lane_offset = 0.0
        # True while the car sits on the FORBIDDEN side of the map centre
        # line: the plan forces the full legal-lane target so the car
        # recovers immediately, and the autopilot crawls back instead of
        # driving wrong-way.  One second of wrong-way driving is not
        # allowed.
        self.last_map_recover = False
        # When a paired sensor lane was rejected because it deviated too
        # far from the nav route: ("nav", deviation_m) or None.
        self.last_lane_override: tuple[str, float] | None = None
        # When ``last_mode`` is "blocked": (label, distance from the car) of
        # the obstacle that left no drivable way, for HUD diagnostics.
        self.last_blocker: tuple[str, float] | None = None
        self.last_route: np.ndarray | None = None
        # Per-frame planning timing (ms) for slow-frame diagnostics.
        self.last_plan_stages: dict[str, float] = {}
        # Speed-planning diagnostics (read by telemetry / GUI):
        # ``last_corner`` is the cruise limited by path curvature only and
        # ``last_obs_lim`` the kinematic speed limit of the closest obstacle
        # that slowed the car (None when no obstacle did).
        self.last_corner: float | None = None
        self.last_obs_lim: float | None = None
        self.last_sharp: bool = False

    # ---- path planning -------------------------------------------------

    def plan(self, route: np.ndarray | None, obstacles, pos, heading: float,
             nearest: int, solid_lines=None,
             sensor_lane=None,
             road_rule: RoadRuleView | None = None,
             cross_solid: bool = False,
             ) -> tuple[np.ndarray, bool]:
        """Return (drive_path, blocked).

        ``drive_path`` is a dense 2D polyline the car should follow next
        (global route, elastic-band deformation, or an A* detour around a
        blocking obstacle).  ``blocked`` is True when no drivable way was
        found and the car should stop in front of the first obstacle.
        ``solid_lines`` is an optional list of detected solid lane markings
        used as no-cross boundaries.  ``sensor_lane`` is an optional
        ``LaneFrame`` from the camera / LiDAR lane modules: when it is
        confident, a two-sided frame is the lane centre itself and the
        drive path follows it directly.  ``route`` may be None when the
        sensor lane is present: no navigation route is required for
        lane-level driving.  ``road_rule`` is an optional map snapshot
        used to keep the car on the legal side of the road and to block
        paths that would cross the map's opposing-lane boundary.
        ``cross_solid`` permits a detour to cross the *detected* centre
        line (overtaking a stopped car with no oncoming traffic); the
        map's legal-lane boundary - wrong-way / off-road - always applies,
        so the detour still cannot leave the road surface.
        """
        self.last_blocker = None
        self.last_plan_stages = {}
        self._cross_solid = bool(cross_solid)
        _t0 = time.perf_counter()
        # Full obstacle set for the final clearance guarantee, captured
        # before the per-stage filters strip obstacles out.  The detour
        # grid and the final no-contact check must still see a wall that
        # was classified as a "roadside boundary": a detour around a
        # different blocker can otherwise sweep straight into it (fuzz
        # scene 14 drove y=-3.5 through a wall the grid never saw).
        # Sparse raycast specks / small lidar clutter are sensor noise
        # and never participate.
        all_obstacles = [
            ob for ob in (obstacles or [])
            if not is_sparse_raycast_speck(ob)
            and not is_small_lidar_clutter(ob)]

        def _stage(name: str) -> None:
            self.last_plan_stages[name] = (
                time.perf_counter() - _t0) * 1000.0
        # Remember the original reference for diagnostics.  A paired sensor
        # lane replaces the nav route as the lane centre when available.
        self.last_route = (None if route is None
                           else np.asarray(route, dtype=float))
        lane_mode = None
        lane_edge = None
        lane_edge_side = 0.0
        lane_center_hint = False
        lane_center = None
        lane_primary = False
        if lane_frame_usable(sensor_lane):
            lane_center = np.asarray(sensor_lane.center, dtype=float)
            if (lane_center.ndim == 2 and lane_center.shape[1] >= 2
                    and len(lane_center) >= 2):
                src = (sensor_lane.sources[0]
                       if sensor_lane.sources else "sensor")
                if sensor_lane.paired:
                    # A real two-sided lane (painted pair or vision + LiDAR
                    # fusion) defines the current lane: its centre is the
                    # drive path, so a nav route must not pull the car out
                    # of the detected lane.
                    lane_mode = src
                    lane_primary = True
                elif src.startswith("vision"):
                    # A single-edge mirror assumes the lane width from one
                    # painted line, so it cannot prove where the lane is
                    # once a nav route exists.  The nav route stays the
                    # primary centre and the mirror only pushes the path
                    # away from a boundary that is too close.  Without a
                    # nav route the mirror is the only lane reference and
                    # may drive its inferred centre.
                    if route is None:
                        lane_mode = src
                        lane_primary = True
                    elif getattr(sensor_lane, "right", None) is not None:
                        lane_mode = f"{src}_right"
                        lane_edge = sensor_lane.right
                        lane_edge_side = -1.0
                    elif getattr(sensor_lane, "left", None) is not None:
                        lane_mode = f"{src}_left"
                        lane_edge = sensor_lane.left
                        lane_edge_side = 1.0
                    else:
                        lane_mode = src
                        lane_center_hint = True
                elif src.startswith("lidar"):
                    # A single LiDAR edge is a low-trust boundary: it may
                    # nudge the path, but the centre line stays primary.
                    lane_mode = src
                    lane_center_hint = True
                    if getattr(sensor_lane, "right", None) is not None:
                        lane_mode = f"{src}_right"
                        lane_edge = sensor_lane.right
                        lane_edge_side = -1.0
                    elif getattr(sensor_lane, "left", None) is not None:
                        lane_mode = f"{src}_left"
                        lane_edge = sensor_lane.left
                        lane_edge_side = 1.0
                elif getattr(sensor_lane, "right", None) is not None:
                    # Painted right line first: it is the primary boundary
                    # under right-hand traffic and outranks any wall.
                    lane_mode = f"{src}_right"
                    lane_edge = sensor_lane.right
                    lane_edge_side = -1.0
                elif getattr(sensor_lane, "left", None) is not None:
                    lane_mode = f"{src}_left"
                    lane_edge = sensor_lane.left
                    lane_edge_side = 1.0
        self.last_lane_mode = (lane_mode
                               or ("nav" if route is not None else "sensor"))
        self.last_lane_offset = 0.0
        map_lane, map_boundaries = self._map_legal_lane(road_rule)
        if map_lane is not None and not map_lane.legal:
            # The map link has no forward lane in the ego's direction:
            # proceeding would drive the wrong way on a one-way road.
            self.last_mode = "blocked"
            self.last_blocker = ("wrong-way road", 0.0)
            return np.empty((0, 2), dtype=float), True
        # Map legal-lane data is a no-cross / wrong-way safety layer, not a
        # lateral driving target.  The nav route is route-level direction;
        # the position inside the lane comes from the sensor lane.  Keeping
        # ``map_offset`` at ``None`` means no code path may shift the drive
        # path by ``preferred_offset_m``.
        map_offset = None
        # The map knows the exact legal lane centre (link width / lane
        # count / side of travel).  When no trusted two-sided sensor lane
        # is present, this is the lateral driving target: the in-game nav
        # route rides the ROAD centre, which on a two-way street is the
        # centre line itself - following it presses the paint and, in a
        # left curve, cuts into the oncoming side (run 1787134963
        # t=14-27: route_lat -0.5..-1.2).  The map target is only a
        # fallback reference: a paired vision+LiDAR lane still replaces
        # it, obstacles may shrink it (``_safe_lateral_offset`` never
        # flips side), and the map centre-line boundary below is the
        # hard no-cross rule.
        map_target = None
        if (map_lane is not None and map_lane.legal
                and map_lane.preferred_offset_m):
            map_target = float(map_lane.preferred_offset_m)
        # Map wrong-side recovery: when the car already sits on the
        # forbidden side of the map centre line while the road itself
        # stays legal, the full legal-lane target is forced immediately
        # (no obstacle shrink, no sensor-lane detour) so the car returns
        # to its own lane at crawl speed instead of continuing wrong-way
        # (run 1787150245 t=36-40: the offset snapped from 1.75 to 0 on
        # a lidar cluster and the car cut the left bend on the oncoming
        # side).  ``last_map_recover`` lets the autopilot cap the speed.
        self.last_map_recover = False
        if (map_target is not None and map_boundaries
                and route is not None and len(route) >= 2):
            pos2 = np.asarray(pos, dtype=float)[:2]
            for b in map_boundaries:
                wb = np.asarray(b.world[:, :2], dtype=float)
                if len(wb) < 2:
                    continue
                side = float(b.allowed_side)
                if (_point_lat_offset(pos2[0], pos2[1], wb) * side
                        >= -0.35):
                    continue
                rw = np.asarray(
                    self._window(route, nearest)[0][:, :2], dtype=float)
                if len(rw) >= 2:
                    rlats = np.asarray([
                        _point_lat_offset(float(px), float(py), wb)
                        for px, py in rw])
                    if float(np.median(rlats)) * side < -0.1:
                        # The road itself turns away from this link's
                        # straight centre line; not a recovery case.
                        continue
                self.last_map_recover = True
                self.last_lane_override = (
                    "map-recover", round(float(map_target), 2))
                break
        raw_path = None
        # Map-side trust guard: a two-sided sensor lane whose centre sits
        # on the FORBIDDEN side of the map centre line (while the nav
        # route itself stays legal) is a wrong pairing - a far lane's
        # lines or a vision+LiDAR fusion on the oncoming side.  Following
        # it drives wrong-way (run 1787134963 t=17: fusion lane_lat
        # +1.4 m pushed the car to route_lat -1.16 m).  The map boundary
        # is authoritative, so the lane is dropped to nav-primary and the
        # map legal-lane target drives the car back into its own lane.
        if lane_primary and map_boundaries and lane_center is not None:
            pos2 = np.asarray(pos, dtype=float)[:2]
            lane_np = np.asarray(lane_center[:, :2], dtype=float)
            if lane_np.ndim == 2 and len(lane_np) >= 2:
                d2 = np.sum((lane_np - pos2) ** 2, axis=1)
                lane_near = lane_np[d2 <= 30.0 ** 2]
                if len(lane_near) >= 2:
                    for b in map_boundaries:
                        w = np.asarray(b.world[:, :2], dtype=float)
                        if len(w) < 2:
                            continue
                        side = float(b.allowed_side)
                        lane_lats = np.asarray([
                            _point_lat_offset(float(px), float(py), w)
                            for px, py in lane_near])
                        route_ok = True
                        if route is not None and len(route) >= 2:
                            rw = np.asarray(
                                self._window(route, nearest)[0][:, :2],
                                dtype=float)
                            if len(rw) >= 2:
                                route_lats = np.asarray([
                                    _point_lat_offset(
                                        float(px), float(py), w)
                                    for px, py in rw])
                                # The road itself turning / leaving the
                                # link's straight centre line is a real
                                # geometry change; then the lane is not a
                                # wrong pairing.
                                route_ok = bool(
                                    float(np.median(route_lats)) * side
                                    >= -0.1)
                        if (route_ok
                                and float(np.median(lane_lats)) * side
                                < -0.35):
                            lane_primary = False
                            lane_mode = None
                            lane_edge = None
                            lane_center_hint = False
                            self.last_lane_override = (
                                "nav-map",
                                round(float(abs(
                                    np.median(lane_lats))), 2))
                            break
        # Trust guard: a paired sensor lane may replace the nav route as
        # the driving centre only while it stays close to the route.  A
        # wrong pairing (far lane's line, roadside paint, guardrail
        # shadow) sits metres off the route and must not drag the car
        # sideways - that was the lane_lat > 3 m weave / guardrail hit.
        # When the lane deviates too far, drop it to a single-edge
        # protection (nav route primary) instead.
        if self.last_map_recover:
            # Recovery overrides every sensor lane: a wrong-side pairing
            # or a low-trust single edge must not fight the map target.
            lane_primary = False
            lane_mode = None
            lane_edge = None
            lane_center_hint = False
        elif lane_primary and route is not None and len(route) >= 2:
            dev = self._sensor_nav_deviation(lane_center, route, pos)
            if dev is not None and dev > LANE_NAV_MAX_DEV_M:
                lane_primary = False
                lane_mode = None
                lane_edge = None
                lane_center_hint = False
                self.last_lane_override = ("nav", round(dev, 2))
        if lane_primary:
            # The frame centre is the midpoint of the detected markings
            # (or of a marking plus its mirrored/opposite-side boundary),
            # so it is the drive path itself.  The nav route may still
            # exist as a long-range direction, but it must not bias the
            # car right of the lane.
            nav_pts, nav_i0, nav_i1 = self._sensor_window(
                lane_center, pos)
            pts = nav_pts[nav_i0:nav_i1 + 1].copy()
            # A lane centre that collapses to a single point (vision saw
            # a marking only a couple of metres ahead, or the fused
            # overlap degenerates) leaves pure pursuit with no geometry
            # to steer by: the control loop zeroes the steering and the
            # car runs straight off the road.  Fall back to the nav
            # window; the lane mode still nudges the path below.
            if len(pts) < 2 and route is not None and len(route) >= 2:
                lane_primary = False
                lane_mode = None
                nav_pts, nav_i0, nav_i1 = self._window(route, nearest)
                pts = nav_pts[nav_i0:nav_i1 + 1].copy()
                self.last_lane_override = ("nav", 0.0)
            raw_path = pts.copy()
            i0, i1 = 0, len(pts) - 1
        elif lane_mode is not None and route is not None:
            nav_pts, nav_i0, nav_i1 = self._window(route, nearest)
            pts = nav_pts[nav_i0:nav_i1 + 1].copy()
            raw_path = pts.copy()
            i0, i1 = 0, len(pts) - 1
            if src.startswith("lidar") and lane_edge is None:
                if map_target is not None:
                    # Map data gives the exact lane centre; a low-trust
                    # lidar centre hint must not stack on top of it.
                    self.last_lane_offset = 0.0
                else:
                    self._apply_lidar_center_hint(
                        pts, sensor_lane, heading, pos, lane_center)
            elif lane_edge is not None:
                # Single-boundary protection: the nav route is the primary
                # lane centre, so a right paint / wall / guardrail only
                # pushes the path away when the route point is already too
                # close to the edge.  It never actively pulls the car
                # toward a half-lane position.
                edge_dev = self._single_edge_route_dev(
                    lane_edge, lane_edge_side, route, pos)
                edge_ok = edge_dev is None
                if edge_dev is not None:
                    if lane_edge_side < 0.0:
                        # RIGHT edge: it defines the car's own lane,
                        # so a far right edge (3-6 m on a two-way
                        # street) is still the lane boundary, not
                        # "another road".  A LiDAR wall is a
                        # physical edge (real run: 6.85 m off the
                        # road-centre route); a far vision paint is
                        # capped tighter to avoid phantom lines.
                        right_max = (
                            LANE_EDGE_RIGHT_VISION_MAX_DEV_M
                            if src.startswith("vision")
                            else LANE_EDGE_RIGHT_MAX_DEV_M)
                        edge_ok = (LANE_EDGE_NAV_MIN_SIGN_M <= edge_dev
                                   <= right_max)
                    else:
                        edge_ok = (-LANE_EDGE_NAV_MAX_DEV_M <= edge_dev
                                   <= -LANE_EDGE_NAV_MIN_SIGN_M)
                if not edge_ok:
                    # The edge disagrees with the nav route near the car:
                    # it sits on the wrong side (the run 53 "right" paint
                    # flipped to the left of the car and the boundary push
                    # dragged the car off the road) or is another road's
                    # boundary.  Degrade to plain nav-primary: the edge
                    # must not nudge the path, shift the keep-right offset
                    # or feed the lane-edge wall filter.
                    lane_mode = None
                    lane_edge = None
                    lane_edge_side = 0.0
                    lane_center_hint = False
                    self.last_lane_override = (
                        "nav", 0.0 if edge_dev is None
                        else round(abs(edge_dev), 2))
                    self.last_lane_offset = 0.0
                elif map_target is None:
                    self._apply_single_edge_correction(
                        pts, lane_edge, lane_edge_side, src, sensor_lane,
                        heading)
                else:
                    # Map target drives the lane centre; the edge stays a
                    # diagnostics boundary (and feeds the wall filter).
                    self.last_lane_offset = 0.0
        elif route is not None:
            nav_pts, nav_i0, nav_i1 = self._window(route, nearest)
            pts = nav_pts[nav_i0:nav_i1 + 1].copy()
            i0, i1 = 0, len(pts) - 1
        elif lane_mode is not None:
            # No nav route and only a low-trust single-edge frame: use the
            # sensor-derived mirror centre, but still classify it as a
            # single-side fallback for telemetry.
            nav_pts, nav_i0, nav_i1 = self._sensor_window(lane_center, pos)
            pts = nav_pts[nav_i0:nav_i1 + 1].copy()
            raw_path = pts.copy()
            i0, i1 = 0, len(pts) - 1
        else:
            self.last_mode = "no-lane"
            return np.empty((0, 2), dtype=float), False
        if len(pts) < 2:
            self.last_mode = "follow"
            return pts, False
        raw_pts = pts.copy()
        # Sparse raycast artefacts are kept for the speed planner (they
        # still ease off the throttle) but are not allowed to close the
        # whole corridor: a 0.9 m single-hit box or an unlabelled fused
        # blob must not park the car in an empty lane.
        obstacles = [ob for ob in obstacles
                     if not is_sparse_raycast_speck(ob)
                     and not is_small_lidar_clutter(ob)]
        # Lidar clusters far off the road corridor are roadside noise: in
        # dense town scenes they inflate the 20 m-wide A* grid until no
        # detour exists.  Only clusters that can actually reach the
        # corridor (own lane + the opposite lane the detour may use) take
        # part in path blocking; the speed planner still sees the full
        # list, so the car keeps easing off near them.
        if any(ob.category == "lidar" for ob in obstacles):
            _ref = raw_pts if raw_path is not None else pts
            _lats = _points_to_polyline_lat(
                np.asarray([[float(o.x), float(o.y)] for o in obstacles]),
                _ref)
            _lim = 6.0
            if road_rule is not None:
                _w = road_width_m(road_rule)
                if _w is not None and _w > 2.0:
                    _lim = _w / 2.0 + 2.0
            obstacles = [o for o, _la in zip(obstacles, _lats)
                         if not (o.category == "lidar"
                                 and abs(float(_la)) > _lim)]
        if lane_mode is not None:
            # The lane centre already keeps the car inside the detected
            # lane; a thin wall at the lane edge is the boundary itself,
            # so it must not close the whole corridor from the side.
            obstacles = [ob for ob in obstacles
                         if not is_lane_edge_wall(
                             ob, raw_path if raw_path is not None else pts,
                             sensor_lane.width,
                             lane_edge=lane_edge,
                             edge_side=lane_edge_side)]
        if route is not None and (
                lane_mode is None or not (
                    sensor_lane is not None and lane_primary)):
            # A single-edge camera/LiDAR read cannot prove where the lane
            # centre is: it only nudges the path away from a boundary that
            # is too close.  The legal-lane offset from the nav route is
            # the driving reference until a real two-sided lane exists.
            # The offset is never pressed into a wall or a lane blocker; on
            # a wall-lined road the path simply eases back instead of
            # trying to squeeze past the wall.
            if map_target is not None:
                # Legal-lane centre from the map.  During a wrong-side
                # recovery the full target is forced (snapping back to
                # the centre line mid-curve is exactly what cut the
                # corner wrong-way in run 1787150245); otherwise the
                # offset eases back to the route centre when the target
                # side is blocked (never flips to the opposite side).
                if self.last_map_recover:
                    safe_off = float(map_target)
                else:
                    safe_off = self._safe_lateral_offset(
                        raw_pts, i0, i1, heading, obstacles,
                        target=map_target)
            else:
                safe_off = self._safe_right_offset(
                    raw_pts, i0, i1, heading, obstacles,
                    edge_pts=lane_edge, edge_side=lane_edge_side)
            # The MAP lane-centre target is the legal lane centre and is
            # held through bends (real ADAS holds the line in a curve);
            # only the legacy keep-right offset ramps out in bends,
            # because a right-hand offset through a corner pushes the
            # car toward the outside of the bend and onto the shoulder
            # (run 36 exited the highway bend 5 m off the route).
            if map_target is None:
                bend_ang = corner_angle_max_deg(
                    raw_pts if len(raw_pts) >= 4 else pts, 0,
                    ahead_idx=max(16, int(40.0 / max(0.5, _route_spacing_m_impl(raw_pts)))))
                # Even a gentle bend pushes the keep-right offset toward
                # the outside of the curve: at 9 m/s a 16 deg corner with
                # a 1.5 m right offset ran the car off the shoulder
                # (run 40).  Ramp the offset out from 12 deg and remove
                # it entirely above 25 deg.
                if bend_ang >= 25.0:
                    safe_off = 0.0
                elif bend_ang >= 12.0:
                    safe_off *= 1.0 - (bend_ang - 12.0) / 13.0
            self.last_lane_offset = float(safe_off) if abs(safe_off) > 1e-9 \
                else 0.0
            _stage("safe_offset")
            pts = self._right_offset_path(raw_pts, i0, heading,
                                          offset=safe_off)
            _stage("offset")

        # Roadside walls are road boundaries, not blockers.  In dense town
        # scenes the raycast fan and LiDAR see the building fronts 1-3 m
        # off the route, and an axis-aligned wall box inflated by the car
        # width intersects the corridor on every frame, so the planner
        # deformed around every wall (steering weave) and stopped for
        # walls that only lined the road (sudden braking on an empty
        # street).  A wall whose footprint lies entirely on one side of
        # the path is a no-cross boundary: drop it from the path
        # hit/blocker/deform logic.  The keep-right offset above already
        # shrinks away from it and the lane/map clamps keep the path off
        # it; a wall that genuinely spans the corridor fails the one-side
        # test and stays a real blocker.
        if obstacles:
            obstacles = [
                ob for ob in obstacles
                if not self._is_roadside_wall(
                    ob, self._obstacle_route_profile(ob, pts, i0, i1),
                    pts=pts)]
            # Roadside poles / fence posts: a small obstacle whose centre
            # sits clearly outside the car's track is roadside furniture,
            # not a lane blocker.  Rows of guardrail posts beside the road
            # would otherwise trip the A* grid / detour every frame and
            # the car creeps or stops on an open road.  Large objects
            # (vehicles, walls spanning the corridor) keep the full
            # treatment.
            obstacles = [
                ob for ob in obstacles
                if not (abs(_point_route_pos_np(ob.x, ob.y, pts)[1])
                        >= CAR_HALF_WIDTH + 2.0
                        and _obstacle_footprint_area(ob) <= 8.0)]

        hit = -1

        def finish(out, mode: str):
            """Apply the solid-line boundary rule and record the mode."""
            boundaries = list(solid_lines or [])
            # Map lane boundaries are the AUTHORITATIVE no-cross /
            # wrong-way safety layer derived from the road graph.  They
            # ALWAYS apply: a paired two-sided lane proves where the
            # car's lane is, but a wrong pairing on the oncoming side of
            # the centre line must never be followed (one second of
            # wrong-way driving is not allowed), and a SINGLE sensor edge
            # (lidar|left/right, vision_left/right) knows only one side:
            # without the map boundary the path rides the nav route (the
            # road/link centre) and presses the centre line the whole
            # run.  The map centre line therefore always applies, including
            # when a paired lane drives the centre.
            map_only: list = []
            if map_boundaries:
                map_only = list(map_boundaries)
                boundaries.extend(map_boundaries)
            if self._cross_solid and mode == "detour":
                # Overtaking with no oncoming traffic may cross a
                # VISUAL / explicitly supplied centre line, but never
                # the authoritative map boundary (road graph): one
                # second of wrong-way driving is not allowed.  Staying
                # on the road surface is enforced by the road-width
                # clamp below.
                boundaries = [
                    b for b in boundaries if b in map_only]
            if boundaries:
                _t1 = time.perf_counter()
                # A detour deliberately leaves the current lane: if it
                # crosses a detected solid line, reject the lane change and
                # stop in front of the obstacle instead.  For ordinary
                # follow/deform paths a solid-line stop still needs a real
                # two-sided lane read; a single-edge mirror/fallback knows
                # one boundary, not the lane geometry, so it may nudge away
                # from the paint but must not turn a CV line into a full
                # stop.
                allow_block = (mode == "detour"
                               or sensor_lane is None
                               or (sensor_lane.paired
                                   and sensor_lane.confidence
                                   >= SOLID_BLOCK_LANE_CONF))
                # Map boundaries are authoritative: they may always
                # block (stop in front of the centre line) even in
                # single-edge mode.
                allow_block = allow_block or bool(map_boundaries)
                # The map boundary is checked against the NAV route (the
                # road reference), not against the sensor path being
                # tested: a wrong sensor pairing on the oncoming side is a
                # violation even while the road itself stays legal, and a
                # turn / intersection where the nav route itself leaves
                # the link's straight centre line is a genuine road
                # geometry change, not a wrong-way attempt.
                map_cor = raw_pts
                if route is not None and len(route) >= 2:
                    map_cor = self._window(route, nearest)[0]
                out, crossed, cross_dist = _clamp_to_solid_lines(
                    out, boundaries, pos, SOLID_LINE_MARGIN,
                    corridor=raw_pts, allow_block=allow_block,
                    block_near_cross=(mode == "detour"),
                    map_nudge=True, map_corridor=map_cor)
                self.last_plan_stages["solid"] = (
                    time.perf_counter() - _t1) * 1000.0
                if crossed:
                    self.last_blocker = ("solid line",
                                         round(cross_dist, 1))
                    self.last_mode = "blocked"
                    if mode == "detour" and hit >= 0:
                        # Refuse the lane change: stop on the original
                        # lane path in front of the obstacle instead of
                        # following a path that crosses the boundary.
                        stop = pts[: hit + 1] if hit >= 1 else pts[:1]
                        return stop, True
                    return out, True
            # Stay on the road surface: detour/deform paths can drift onto
            # the shoulder in dense obstacle scenes.  When the map knows
            # the link width, pull every path point back inside the road
            # (the corridor may still cross the centre line to overtake,
            # it just never leaves the asphalt).
            if lane_mode is None and road_rule is not None:
                w = road_width_m(road_rule)
                if w is not None and w > 2.0 * (CAR_HALF_WIDTH + 0.5):
                    out = _clamp_path_lateral(
                        out, raw_pts, w / 2.0 - CAR_HALF_WIDTH - 0.3)
            # Final clearance guarantee ("never drive through a wall").
            # The A* grid / lateral-bypass path is verified against the
            # occupancy grid, not the true oriented footprints, so a
            # detour's return leg can still graze a tilted blocker.  Any
            # drive path that would touch an obstacle's real footprint
            # (inflated by the car half width + a small margin) is never
            # handed to the controller: fall back to stopping on the route
            # in front of the first blocker instead.
            if all_obstacles:
                safe_h = CAR_HALF_WIDTH
                hit_out = _path_hit_index(
                    out, 0, len(out) - 1, all_obstacles, safe_h)
                if hit_out >= 0:
                    mode = "blocked"
                    # Stop at the first blocked segment's vertex: keep the
                    # clear prefix before it.  When the very first segment
                    # is blocked (``hit_out == 0``) the car is already at
                    # the wall, so return only the single start point and
                    # let the controller hold the car stopped.
                    out = out[: hit_out + 1] if hit_out >= 1 else out[:1]
            self.last_mode = mode
            return out, mode == "blocked"

        # Obstacles' own footprints (no safety inflation): only a footprint
        # that really intrudes into the lane matters.  Roadside poles/curbs
        # 2.5-3 m off the route must NOT make the car swerve or brake.
        if not obstacles:
            return finish(pts, "follow")
        hit = _path_hit_index(pts, i0, i1, obstacles,
                              CAR_HALF_WIDTH + 0.8)
        _stage("hit")
        if hit < 0:
            # No obstacle footprint touches the corridor: the path is
            # already clear, so do NOT run the elastic-band deform.  In
            # dense town streets the building fronts sit 1-3 m off the
            # route; deform's push radius (half_w + car + clearance
            # ~3-4 m) reaches them anyway, so it nudged the path away
            # from the left wall one frame and the right wall the next
            # - the steering weave / lane crossing the user saw.  The
            # keep-right offset above already keeps the path off a
            # roadside wall; deform is only for obstacles that truly
            # intrude into the corridor.
            return finish(pts, "follow")
        blocker = _find_blocker(pts, i0, i1, obstacles,
                                CAR_HALF_WIDTH + 0.8)
        _stage("blocker")
        if blocker is not None:
            bd = math.hypot(blocker.x - float(pos[0]),
                            blocker.y - float(pos[1]))
            self.last_blocker = (blocker.label or blocker.category, bd)
        # A long wall that only lines the road side is a boundary, not a
        # detour target.  If the lane itself is too narrow for the car and
        # the wall, stop in front of it instead of steering toward/through
        # it.
        if blocker is not None and self._is_roadside_wall(
                blocker, self._obstacle_route_profile(
                    blocker, pts, i0, i1), pts=pts):
            stop = pts[: hit + 1] if hit >= 1 else pts[:1]
            return finish(stop, "blocked")
        detour, reached = self._grid_path(
            pts, all_obstacles, pos, heading, i0, i1, margin=self.margin)
        _stage("grid")
        if detour is not None and len(detour) >= 2 and reached:
            # A detour that actually reaches the route horizon is drivable.
            return finish(detour, "detour")
        # A* could not reach the horizon (wide inflation, a wall of boxes):
        # try a smooth lateral bypass around the first compact blocker
        # before declaring the corridor blocked, so a parked car in the
        # lane becomes a lane change instead of an emergency stop.
        bypass = self._lateral_bypass(pts, all_obstacles, i0, i1)
        _stage("bypass")
        if bypass is not None and len(bypass) >= 2:
            return finish(bypass, "detour")
        if detour is not None and len(detour) >= 2:
            # Truncated "best reachable cell" path: drive up to the
            # obstacle, then stop in front of it instead of creeping on.
            return finish(detour, "blocked")
        # No drivable way at all: stop in front of the first blocker.
        stop = pts[: hit + 1] if hit >= 1 else pts[:1]
        return finish(stop, "blocked")

    def _map_legal_lane(self, road_rule: RoadRuleView | None):
        """Return (LegalLaneView|None, map no-cross boundary list)."""
        view = legal_lane_view(road_rule)
        boundaries: list[_MapLaneBoundary] = []
        if view is None or not view.legal or road_rule is None:
            return view, boundaries
        if not (road_rule.in_pos and road_rule.out_pos and road_rule.right_vec):
            return view, boundaries
        p1 = np.asarray(road_rule.in_pos[:2], dtype=float)
        p2 = np.asarray(road_rule.out_pos[:2], dtype=float)
        right = np.asarray(road_rule.right_vec[:2], dtype=float)
        rn = float(np.linalg.norm(right))
        if rn < 1e-9:
            return view, boundaries
        right = right / rn
        for offset_m, allowed_side in view.boundaries:
            a = p1 + right * offset_m
            b = p2 + right * offset_m
            boundaries.append(_MapLaneBoundary(
                np.vstack([a, b]), allowed_side))
        return view, boundaries

    def _apply_lidar_center_hint(self, pts, sensor_lane, heading,
                                  lane_center) -> None:
        """Pull the nav window toward a low-trust LiDAR centre hint.

        Extracted from the inline fallback: a single LiDAR corridor
        centre is a weak lane reference (it may mirror a guardrail), so
        it only nudges the path toward its centre, capped by
        ``LANE_LIDAR_CORRECTION_MAX_M`` and ramped in over
        ``right_ramp_m``.  Called only when no map legal-lane target is
        available (the map target is authoritative and must not stack
        with a second correction).
        """
        lane_gain = _lane_correction_gain(sensor_lane.confidence)
        corr_max = LANE_LIDAR_CORRECTION_MAX_M
        base = pts.copy()
        n = len(pts)
        d = np.linalg.norm(np.diff(base, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(d)])
        for i in range(n):
            f = _smoothstep(cum[i] / max(1e-9, self.right_ramp_m))
            a = base[max(0, i - 2)]
            b = base[min(n - 1, i + 2)]
            tv = b - a
            tn = float(np.linalg.norm(tv))
            if tn < 1e-9:
                left = np.array([-math.sin(heading),
                                 math.cos(heading)])
            else:
                left = np.array([-tv[1] / tn, tv[0] / tn])
            off = _point_lat_offset(
                base[i, 0], base[i, 1], lane_center)
            off = max(-corr_max, min(corr_max, off))
            pts[i] = base[i] + left * (f * off * lane_gain)
        self.last_lane_offset = 0.0

    def _apply_single_edge_correction(self, pts, lane_edge, lane_edge_side,
                                      src, sensor_lane, heading) -> None:
        """Keep the path in the car's own lane using a single boundary.

        A RIGHT boundary (painted line / wall / guardrail) defines the
        car's lane: the path is actively pulled to the lane centre
        (half an assumed lane width inside the edge, capped by
        ``LANE_EDGE_PULL_MAX_M``) - the nav route alone rides the road
        centre / oncoming lane (run 1787130718).  A LEFT boundary may
        only push the path right when the route is too close to it; it
        never pulls left toward the oncoming lane.  The gain still
        ramps with frame confidence and the ramp smooths it in.
        """
        lane_gain = _lane_correction_gain(sensor_lane.confidence)
        if lane_edge_side < 0.0:
            # Right edge = active lane-centre pull, capped generously.
            corr_max = float(LANE_EDGE_PULL_MAX_M)
            if not src.startswith("vision"):
                # A LiDAR wall is a physical edge: the raw-sensor
                # emergency stop and the final no-contact check are
                # the safety net, so the lane-centre pull is not
                # scaled down by the fallback frame's low confidence
                # (0.45-0.5 would halve the pull and leave the car
                # between the lanes instead of centred in its own).
                lane_gain = 1.0
        else:
            corr_max = (
                LANE_BOUNDARY_CORRECTION_MAX_M
                if src.startswith("vision")
                else LANE_LIDAR_EDGE_CORRECTION_MAX_M)
        edge_pts = np.asarray(lane_edge[:, :2], dtype=float)
        lane_w = float(getattr(sensor_lane, "width",
                               LANE_WIDTH_DEFAULT_M))
        base = pts.copy()
        offsets = []
        n = len(pts)
        d = np.linalg.norm(np.diff(base, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(d)])
        for i in range(n):
            off = self._boundary_path_offset(
                base[i, 0], base[i, 1], edge_pts,
                side=lane_edge_side, lane_width=lane_w,
                corr_max=corr_max)
            if off is None:
                off = 0.0
            offsets.append(off)
            f = _smoothstep(
                cum[i] / max(1e-9, self.right_ramp_m))
            a = base[max(0, i - 2)]
            b = base[min(n - 1, i + 2)]
            tv = b - a
            tn = float(np.linalg.norm(tv))
            if tn < 1e-9:
                right = np.array([math.sin(heading),
                                  -math.cos(heading)])
            else:
                right = np.array([tv[1] / tn, -tv[0] / tn])
            pts[i] = base[i] + right * (
                f * off * lane_gain)
        self.last_lane_offset = float(np.median(offsets))

    def _boundary_path_offset(self, x: float, y: float, edge_pts,
                              side: float, lane_width: float,
                              corr_max: float
                              = LANE_BOUNDARY_CORRECTION_MAX_M
                              ) -> float | None:
        """Signed lateral shift that keeps a path point clear of an edge.

        ``side`` is -1.0 for a right boundary (painted line / wall /
        guardrail) and +1.0 for a left boundary.  The returned offset is
        positive to the right of the path tangent, matching
        ``_point_lat_offset``; ``None`` means the boundary is too far or
        not clearly on the expected side, so the caller keeps the nav
        route point.  The nav route is the lane centre: a far edge adds
        no correction at all.
        """
        edge_pts = np.asarray(edge_pts[:, :2], dtype=float)
        if len(edge_pts) < 2:
            return None
        lane_width = float(lane_width)
        if not math.isfinite(lane_width) or lane_width <= 0.0:
            lane_width = LANE_WIDTH_DEFAULT_M
        boundary_lat = _point_lat_offset(float(x), float(y), edge_pts)
        if not math.isfinite(boundary_lat):
            return None
        # A right boundary may be metres from the route (the route is
        # the road-centre line; real right wall measured 6.85 m).
        # A left boundary farther than a lane is another road and
        # must never pull the car toward it.
        if side < 0.0:
            if abs(boundary_lat) > LANE_EDGE_RIGHT_MAX_DEV_M:
                return None
        elif abs(boundary_lat) > LANE_BOUNDARY_MAX_M:
            return None
        # A right edge must sit to the right of the path point and a
        # left edge to the left: a "right" paint that is actually left
        # of the path (tracker locked onto the opposite line - run 53
        # t=47) would push the path *toward* the wrong-side edge and
        # drag the car off the road.  Contradictory geometry gets no
        # correction at all.
        if side < 0.0 and boundary_lat > 0.0:
            return None
        if side > 0.0 and boundary_lat < 0.0:
            return None
        # ``_point_lat_offset`` is positive to the right of the boundary.
        # A right edge has the road on its negative side, a left edge on
        # its positive side.  Only when the nav point is closer than the
        # car's clearance do we shift away from the edge; a route point
        # that already clears it stays untouched.
        clear = LANE_BOUNDARY_CLEAR_M
        # A LEFT boundary that sits more than ~1.5 m beyond the car's
        # clearance is normal road furniture (guardrail / kerb / wall
        # lining the OPPOSITE side of the road): it must never pull the
        # path toward the oncoming lane (run 33 weave).  A RIGHT
        # boundary is different: the nav route is anchored to the road
        # link centre, so on a two-way street the car's own lane's
        # right edge is 3-6 m right of the route.  Real ADAS drives the
        # lane centre = half an assumed lane width inside that edge;
        # ignoring it let the car ride the centre line / drift right
        # the whole run (1787130718: lat 0.7-3.8 m).  The right edge
        # therefore actively pulls the path back into its own lane.
        if side > 0.0 and abs(boundary_lat) > clear + 1.5:
            return 0.0
        if side < 0.0:
            # RIGHT boundary (painted line / wall / guardrail).
            if boundary_lat <= -clear:
                # Route clears the edge: pull toward the centre of the
                # lane this edge defines (half a lane width inside it).
                # boundary_lat < 0 (edge right of the route point), so
                # the centre offset is -(edge_lat + half_lane), a shift
                # to the RIGHT (positive = right of the route tangent).
                pull_max = float(LANE_EDGE_PULL_MAX_M)
                centre_off = -boundary_lat - lane_width * 0.5
                return max(-pull_max, min(pull_max, centre_off))
            # Edge intrudes into the route: push away with clearance.
            off = -boundary_lat - clear
        else:
            # LEFT boundary.  The car's legal lane lies to the RIGHT of
            # a left edge: it may only PUSH the path right (never pull
            # left toward the oncoming lane - one second of wrong-way
            # driving is not allowed).  When the route already clears
            # the left edge, it stays untouched.
            if boundary_lat >= clear:
                return 0.0
            off = clear - boundary_lat
        corr_max = max(0.0, float(corr_max))
        return max(-corr_max, min(corr_max, off))

    def _right_offset_path(self, pts, i0: int, heading: float,
                           offset: float | None = None):
        """Shift the planning window toward the right-hand side if asked.

        The default is no lateral offset (route centre).  When an offset is
        configured it ramps in over ``right_ramp_m`` so the car leaves the
        nav corridor smoothly instead of jumping sideways at the start.
        Each point is shifted along that point's local route right vector
        (perpendicular to the route tangent), so a bend keeps the offset
        on the same side of the road instead of pushing the path toward
        the outside of the corner.
        """
        target = self.right_offset if offset is None else float(offset)
        if abs(target) < 1e-9:
            return np.asarray(pts, dtype=float)
        src = np.asarray(pts, dtype=float)
        out = src.copy()
        d = np.linalg.norm(np.diff(out, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(d)])
        base = cum[i0]
        n = len(out)
        for i in range(i0, n):
            f = _smoothstep((cum[i] - base)
                            / max(1e-9, self.right_ramp_m))
            a = src[max(0, i - 2)]
            b = src[min(n - 1, i + 2)]
            tv = b - a
            tn = float(np.linalg.norm(tv))
            if tn < 1e-9:
                right = np.array([math.sin(heading), -math.cos(heading)])
            else:
                right = np.array([tv[1] / tn, -tv[0] / tn])
            out[i] = out[i] + right * (f * target)
        return out

    def _obstacle_route_profile(self, ob, pts, i0: int, i1: int):
        """Route-local (lon, lat) extent of an obstacle footprint.

        The window is approximated by one forward direction from ``i0`` to
        ``i1``; lat is positive to the left of travel.  Returns
        ``(lon0, lon1, lat0, lat1)``.
        """
        fwd = pts[i1] - pts[i0]
        fn = float(np.linalg.norm(fwd))
        if fn < 1e-9:
            fwd = np.array([1.0, 0.0])
        else:
            fwd = fwd / fn
        lat = np.array([-fwd[1], fwd[0]])
        p0 = pts[i0]
        corners = _obstacle_corners(ob)
        lons = [float((c[0] - p0[0]) * fwd[0] + (c[1] - p0[1]) * fwd[1])
                for c in corners]
        lats = [float((c[0] - p0[0]) * lat[0] + (c[1] - p0[1]) * lat[1])
                for c in corners]
        return min(lons), max(lons), min(lats), max(lats)

    def _is_roadside_wall(self, ob, profile, pts=None) -> bool:
        """True when the obstacle is a long thin wall beside the route.

        Such a wall is a no-cross road boundary: the planner may drive
        beside it (with clearance) but must never treat it as a compact
        object to swerve around.
        """
        if ob.label != "wall" and ob.category not in ("wall", "raycast"):
            return False
        if _obstacle_oriented(ob):
            length = 2.0 * ob.half_len
            thick = 2.0 * max(0.0, ob.half_thick)
        else:
            length = max(2.0 * ob.half_w, 2.0 * ob.half_h)
            thick = min(2.0 * ob.half_w, 2.0 * ob.half_h)
        # 放宽长度要求：2m 以上的墙都算路边墙
        if length < 2.0:
            return False
        if thick > ROADSIDE_WALL_MAX_THICK_M:
            return False
        # A roadside wall's footprint must stay entirely on one side of the
        # driving polyline.  Using the nearest route segment (instead of the
        # window chord) keeps this correct through bends, where a straight
        # i0->i1 axis makes a diagonal wall look like it crosses the road.
        if pts is not None and len(pts) >= 2:
            # 墙到路线的最小距离取两个度量里更严的那个：(1) 中心距减去
            # 半厚——捕捉垂直横穿道路的墙（四个角都在 2m 外，但墙身从
            # 路中间穿过，中心距是 0）；(2) 四个墙角的实际横向距离——
            # 捕捉斜墙（中心 3m 外、某个角却深探进车道，fuzz scene 7 的
            # 底角实际到 y=0.56）。只有两者都保持在车半宽 + 边缘余量
            # 之外，才允许当作不阻挡的边界。该检查必须排在"角全在单侧"
            # 判定之前。
            lats = [_point_lat_offset(c[0], c[1], pts)
                    for c in _obstacle_corners(ob)]
            if _obstacle_oriented(ob):
                thick = max(0.0, float(getattr(ob, "half_thick", 0.0)))
            else:
                thick = min(float(ob.half_w), float(ob.half_h))
            _, lat_c = _point_route_pos_np(ob.x, ob.y, pts)
            inner = min(abs(lat_c) - thick,
                        min(abs(float(v)) for v in lats))
            if inner < CAR_HALF_WIDTH + ROADSIDE_WALL_MIN_EDGE_M:
                # 墙内缘贴近或越过车道：它是真正的挡路墙，不是路边边界
                return False
            # 内缘足够远之后，单侧判定才是安全的：墙的相邻角至少两个
            # 明显落在线路同一侧 -> 它是可沿其行驶的边界。这时即使墙
            # 中心距路线很远，也不把它当普通障碍物绕行。
            pos = sum(1 for v in lats if v > ROADSIDE_WALL_MIN_EDGE_M)
            neg = sum(1 for v in lats if v < -ROADSIDE_WALL_MIN_EDGE_M)
            if pos >= 2 or neg >= 2:
                return True
            # Through a bend the same wall can straddle the polyline:
            # corner lats land on both sides (pos ~= neg ~= 2) even
            # though the wall only lines the road.  The inner-edge check
            # above already guarantees the wall keeps clear of the
            # corridor, so it is the boundary beside it.
            return True
        lon0, lon1, lat0, lat1 = profile
        lon_span = lon1 - lon0
        lat_span = lat1 - lat0
        if lon_span < 1.5 * lat_span:
            return False
        # Fallback without a polyline: clearly on one side of the window.
        return lat0 > ROADSIDE_WALL_MIN_EDGE_M \
            or lat1 < -ROADSIDE_WALL_MIN_EDGE_M

    def _side_has_roadside_wall(self, pts, i0: int, i1: int,
                                obstacles, side: float) -> bool:
        """True when a roadside wall lines the chosen bypass side."""
        for ob in obstacles:
            profile = self._obstacle_route_profile(ob, pts, i0, i1)
            if not self._is_roadside_wall(ob, profile, pts=pts):
                continue
            lat0, lat1 = profile[2], profile[3]
            if side < 0 and lat1 < -ROADSIDE_WALL_MIN_EDGE_M:
                return True
            if side > 0 and lat0 > ROADSIDE_WALL_MIN_EDGE_M:
                return True
        return False

    def _edge_clear(self, cand, edge_pts, edge_side: float,
                    clearance: float = LANE_BOUNDARY_CLEAR_M) -> bool:
        """True when a candidate path keeps the car clear of one edge."""
        if edge_pts is None or edge_side == 0.0:
            return True
        edge = np.asarray(edge_pts[:, :2], dtype=float)
        if len(edge) < 2:
            return True
        d = edge[-1] - edge[0]
        dn = float(np.linalg.norm(d))
        if dn < 1e-9:
            return True
        d = d / dn
        lon = (edge - edge[0]) @ d
        lon_max = float(lon.max())
        for p in cand:
            p = np.asarray(p, dtype=float)[:2]
            if float((p - edge[0]) @ d) > lon_max + 2.0:
                continue
            lat = _point_lat_offset(float(p[0]), float(p[1]), edge)
            if not math.isfinite(lat):
                continue
            if edge_side < 0.0 and lat > -clearance:
                return False
            if edge_side > 0.0 and lat < clearance:
                return False
        return True

    def _safe_right_offset(self, pts, i0: int, i1: int, heading: float,
                           obstacles, clearance: float = 0.8,
                           edge_pts=None,
                           edge_side: float = 0.0) -> float:
        """Largest keep-right offset that clears every obstacle footprint.

        The full right offset is preferred on open roads; when a wall or a
        parked object sits on the right, the offset shrinks in 0.1 m steps
        so the path never presses the car into the obstacle.

        One extra rule keeps the car from being dragged back onto the
        centre line by sensor boxes that hug the nav corridor itself: when
        even the zero-offset path is already "hit" (a ghost box, a wall
        AABB that overlaps the route, or an obstacle that sits in the
        lane), shrinking the keep-right offset cannot fix the collision,
        so a small right bias is retained instead of returning 0.  A wall
        that only blocks the right-hand side still shrinks the offset to 0
        (``off=0`` is clear), which is the old safe behaviour.
        """
        if abs(self.right_offset) < 1e-9 or (
                not obstacles and (edge_pts is None or edge_side == 0.0)):
            return self.right_offset
        step = 0.1
        n_steps = int(round(self.right_offset / step)) + 1
        # The lateral shift per point scales linearly with the offset, so
        # compute the unit offset direction once and reuse it for every
        # candidate instead of rebuilding the full shifted path.
        src = np.asarray(pts, dtype=float)
        unit = self._right_offset_path(src, i0, heading, offset=1.0)
        off_dir = unit - src
        for k in range(n_steps):
            off = max(0.0, self.right_offset - k * step)
            cand = src + off_dir * off
            if _path_hit_index(cand, i0, i1, obstacles,
                               CAR_HALF_WIDTH + clearance) < 0 \
                    and self._edge_clear(
                        cand, edge_pts, edge_side):
                return off
        if _path_hit_index(src, i0, i1, obstacles,
                           CAR_HALF_WIDTH + clearance) >= 0:
            # The route centre is already inside an obstacle footprint.
            # The fallback may ONLY return an offset whose shifted path is
            # itself proven clear: returning an unchecked bias drove the
            # car straight into a tilted wall that left a ~0.3 m gap (the
            # offset -0.8 path overlapped the wall inner edge while the
            # wall was then dropped as "roadside" - fuzz scene 0).  Try
            # the original bias values, keep the first clear one, and
            # otherwise return the route centre so the downstream hit /
            # deform / blocked pipeline decides with the obstacle still in
            # the list.
            for bias in (0.8, 0.5):
                cand = src + off_dir * bias
                if _path_hit_index(cand, i0, i1, obstacles,
                                   CAR_HALF_WIDTH + clearance) < 0 \
                        and self._edge_clear(cand, edge_pts, edge_side):
                    return bias
            return 0.0
        return 0.0

    def _safe_lateral_offset(self, pts, i0: int, i1: int, heading: float,
                             obstacles, target: float,
                             clearance: float = 0.8) -> float:
        """Largest legal-lane offset that clears every obstacle footprint.

        Works for both LHD (positive-right map offset) and RHD (negative)
        without changing ``RIGHT_OFFSET_M`` default semantics.  Unlike the
        keep-right fallback it never flips to the opposite side: when the
        target side is blocked it eases back to the route centre and lets
        the map boundary clamp / detour logic decide what happens next.
        """
        target = float(target)
        if abs(target) < 1e-9 or not obstacles:
            return target
        src = np.asarray(pts, dtype=float)
        # Roadside walls are road boundaries, not lane blockers.  A wall
        # that lines the road must not shrink the legal-lane centre away
        # (run 1787150245 t=36-40: wall/lidar boxes beside the bend cut
        # the map target 1.75 -> 0 and the car cut the left curve on the
        # oncoming side).  The downstream path-hit stage filters the same
        # walls; do it here too so the map centre is evaluated on the
        # open corridor.
        if any(ob.label == "wall" or ob.category in ("wall", "raycast")
               for ob in obstacles):
            obstacles = [
                ob for ob in obstacles
                if not self._is_roadside_wall(
                    ob, self._obstacle_route_profile(ob, pts, i0, i1),
                    pts=pts)]
        unit = self._right_offset_path(src, i0, heading, offset=1.0)
        off_dir = unit - src
        step = 0.1
        direction = 1.0 if target >= 0.0 else -1.0
        n_steps = int(math.ceil(abs(target) / step))
        for k in range(n_steps + 1):
            off = direction * max(0.0, abs(target) - k * step)
            cand = src + off_dir * off
            if _path_hit_index(cand, i0, i1, obstacles,
                               CAR_HALF_WIDTH + clearance) < 0:
                return off
        return 0.0

    def _lateral_bypass(self, pts, obstacles, i0: int, i1: int):
        """Smooth lane-shift around the first compact blocker.

        Used when the occupancy-grid A* cannot reach the route horizon
        (wide inflated boxes or a wall of obstacles).  The corridor is
        offset laterally across the blocker - like a lane change - and
        blended back into the navigation route after it, so the car
        actually drives around the thing instead of stopping.  A genuine
        wall that fills both sides yields ``None`` and the caller stops.

        Returns a world-frame polyline or None.
        """
        pairs = [ob for ob in obstacles
                 if (ob.half_w >= 0.15 and ob.half_h >= 0.15
                     or _obstacle_oriented(ob))]
        if not pairs:
            return None
        # First obstacle that actually intrudes into the corridor.
        hit_i, hit_ob = -1, None
        for i in range(i0, min(i1, len(pts) - 1)):
            for ob in pairs:
                if _seg_hits_obstacle(
                        pts[i, 0], pts[i, 1],
                        pts[i + 1, 0], pts[i + 1, 1],
                        ob, CAR_HALF_WIDTH + 0.6):
                    hit_i, hit_ob = i, ob
                    break
            if hit_i >= 0:
                break
        if hit_i < 0 or hit_ob is None:
            return None
        # Long roadside walls are boundaries, not obstacles to lane-change
        # around; crossing them (or steering toward them) is what caused the
        # violent left swerve into the wall.
        if self._is_roadside_wall(
                hit_ob, self._obstacle_route_profile(
                    hit_ob, pts, i0, i1), pts=pts):
            return None
        bx, by = hit_ob.x, hit_ob.y
        # Local route frame: forward along the planning window, lateral to
        # the left of the travel direction.
        fwd = pts[i1] - pts[i0]
        fn = float(np.linalg.norm(fwd))
        if fn < 1e-9:
            return None
        fwd = fwd / fn
        lat = np.array([-fwd[1], fwd[0]])
        p0 = pts[i0]
        lon_c = (bx - p0[0]) * fwd[0] + (by - p0[1]) * fwd[1]
        half_lon, half_lat = _obstacle_half_extents(hit_ob, fwd, lat)
        # Lateral path offset needed for the car to clear the box.
        need = half_lat + CAR_HALF_WIDTH + 0.9
        # Clearance zone around the box (footprint + car half width +
        # buffer): the corridor check guarantees this zone stays clear at
        # the chosen offset, so the lane-shift ramp must be *finished*
        # before the path enters it, not taper inside it.
        infl = CAR_HALF_WIDTH + 0.8
        zc0 = lon_c - half_lon - infl
        zc1 = lon_c + half_lon + infl
        hold0 = zc0 - 1.0
        hold1 = zc1 + 1.0
        taper = max(6.0, (zc1 - zc0) / 3.0)
        lon0 = hold0 - taper
        lon1 = hold1 + taper

        def corridor_free(side: float, off: float) -> bool:
            """True when the offset corridor clears every obstacle (the
            blocker itself included) and stays within ``max_dev``."""
            if off > self.max_dev + 1e-9:
                return False
            a = p0 + fwd * lon0 + lat * (off * side)
            b = p0 + fwd * lon1 + lat * (off * side)
            for ob in pairs:
                if _seg_hits_obstacle(
                        a[0], a[1], b[0], b[1], ob,
                        CAR_HALF_WIDTH + 0.8):
                    return False
            return True

        # Prefer the right-hand side (keep-right rule), then the left.
        # A side lined by a roadside wall is skipped entirely: the car must
        # not swerve toward a wall even when the box geometry nominally
        # clears it.
        offset, side = None, 0.0
        for s in (-1.0, 1.0):
            if self._side_has_roadside_wall(pts, i0, i1, obstacles, s):
                continue
            off = need
            while off <= self.max_dev + 1e-9:
                if corridor_free(s, off):
                    offset, side = off, s
                    break
                off += 0.4
            if offset is not None:
                break
        if offset is None:
            return None

        # Longitudinal distance of each window point from i0.
        d = np.linalg.norm(np.diff(pts[i0:i1 + 1], axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(d)])

        def _st(x: float) -> float:
            x = min(1.0, max(0.0, x))
            return x * x * (3.0 - 2.0 * x)

        out = pts[i0:i1 + 1].copy()
        for k in range(len(out)):
            s = float(cum[k])
            if s <= lon0 or s >= lon1:
                f = 0.0
            elif hold1 > hold0 and hold0 <= s <= hold1:
                f = 1.0
            elif s < hold0:
                f = _st((s - lon0) / max(1e-6, hold0 - lon0))
            else:
                f = 1.0 - _st((s - hold1) / max(1e-6, lon1 - hold1))
            out[k] = out[k] + lat * (offset * side * f)
        # One smoothing pass, endpoints stay fixed on the nav corridor.
        nxt = out.copy()
        for i in range(1, len(out) - 1):
            nxt[i] = 0.25 * out[i - 1] + 0.5 * out[i] + 0.25 * out[i + 1]
        return nxt

    def _grid_path(self, pts, obstacles, pos, heading: float,
                   i0: int, i1: int, margin: float = SAFETY_MARGIN):
        """Local occupancy-grid A* detour around blocking obstacles.

        Builds a grid in the car's local frame (x = heading, y = left),
        marks inflated obstacles, and plans a path from the car to the
        global route horizon that hugs the route when possible.  Returns
        a smoothed world-frame polyline (or None when no detour exists) and
        a bool telling whether the path reaches the route horizon.
        """
        import heapq

        hx = math.cos(heading)
        hy = math.sin(heading)
        lx = -hy
        ly = hx
        ox, oy = float(pos[0]), float(pos[1])
        res = GRID_RES
        rows = int(math.ceil((GRID_AHEAD + GRID_BEHIND) / res))
        cols = int(math.ceil(2.0 * GRID_HALF_W / res))
        x0 = -GRID_BEHIND
        y0 = -GRID_HALF_W

        def to_local(wx: float, wy: float):
            dx = wx - ox
            dy = wy - oy
            return dx * hx + dy * hy, dx * lx + dy * ly

        # Car position in grid coordinates (needed to extend obstacles only
        # when they sit clearly ahead of the car).
        def cell_index(wx: float, wy: float):
            f, s = to_local(wx, wy)
            return (int(round((f - x0) / res)), int(round((s - y0) / res)))

        r_start, c_start = cell_index(ox, oy)
        r_start = int(np.clip(r_start, 0, rows - 1))
        c_start = int(np.clip(c_start, 0, cols - 1))
        start = (r_start, c_start)

        # Mark inflated obstacle boxes as occupied (with a 1-cell pad).
        # Each blocking box is extended toward the car by GRID_ANTICIPATE so
        # the detour starts steering well before the obstacle instead of
        # swerving at the last moment.  Only boxes clearly ahead get the
        # extension (a box beside/behind the car must not reach in front of
        # it).
        occ = np.zeros((rows, cols), dtype=bool)
        pad = int(math.ceil(CAR_HALF_WIDTH / res)) + 1
        anticipate_cells = int(math.ceil(GRID_ANTICIPATE / res))
        for ob in obstacles:
            # Obstacle inflation = car half width (the pad below) plus a
            # net clearance gap.  LIDAR_PATH_CLEAR_M already includes the
            # car half width, so subtracting CAR_HALF_WIDTH avoids double
            # counting - before this fix a lidar box in a 7 m street ate
            # 4.4 m and left no drivable lane.  Raycast/vehicle obstacles
            # keep the original SAFETY_MARGIN behaviour (the offline
            # grove scenes are calibrated against it).
            if ob.category == "lidar":
                m = max(0.0, LIDAR_PATH_CLEAR_M - CAR_HALF_WIDTH)
            elif self._is_roadside_wall(
                    ob, self._obstacle_route_profile(ob, pts, i0, i1),
                    pts=pts):
                # Roadside boundary (tree row / wall lining the road):
                # the car legitimately drives beside it, so plan with a
                # tight pad (car half width + 0.5 m) instead of the full
                # margin; the final no-contact check still enforces the
                # physical clearance.
                m = CAR_HALF_WIDTH + 0.5
            else:
                m = margin
            if _obstacle_oriented(ob):
                ux, uy = float(ob.axis[0]), float(ob.axis[1])
                vx, vy = -uy, ux
                hu = ob.half_len + m
                hv = max(0.0, ob.half_thick) + m
                corners = [
                    to_local(ob.x + ux * hu + vx * hv,
                             ob.y + uy * hu + vy * hv),
                    to_local(ob.x + ux * hu - vx * hv,
                             ob.y + uy * hu - vy * hv),
                    to_local(ob.x - ux * hu + vx * hv,
                             ob.y - uy * hu + vy * hv),
                    to_local(ob.x - ux * hu - vx * hv,
                             ob.y - uy * hu - vy * hv),
                ]
            else:
                hw = ob.half_w + m
                hh = ob.half_h + m
                corners = [
                    to_local(ob.x - hw, ob.y - hh),
                    to_local(ob.x + hw, ob.y - hh),
                    to_local(ob.x + hw, ob.y + hh),
                    to_local(ob.x - hw, ob.y + hh),
                ]
            rxs = [c[0] for c in corners]
            rys = [c[1] for c in corners]
            near_r = int(math.floor((min(rxs) - x0) / res)) - pad
            if near_r > r_start + 12:
                # Obstacle is well ahead: pull its near edge toward the car.
                near_r -= anticipate_cells
            r0 = max(0, near_r)
            r1 = min(rows - 1, int(math.floor((max(rxs) - x0) / res)) + pad)
            c0 = max(0, int(math.floor((min(rys) - y0) / res)) - pad)
            c1 = min(cols - 1, int(math.floor((max(rys) - y0) / res)) + pad)
            if r1 >= r0 and c1 >= c0:
                occ[r0:r1 + 1, c0:c1 + 1] = True

        # Lateral-deviation cost per cell: prefer hugging the global route,
        # but allow real detours when the corridor is blocked.
        dev = np.zeros((rows, cols), dtype=float)
        path_p = pts[max(0, i0 - 5): i1 + 1]
        if len(path_p):
            f_vals = x0 + (np.arange(rows) + 0.5) * res
            s_vals = y0 + (np.arange(cols) + 0.5) * res
            wx = ox + f_vals[:, None] * hx + s_vals[None, :] * lx
            wy = oy + f_vals[:, None] * hy + s_vals[None, :] * ly
            cell_xy = np.stack([wx, wy], axis=-1)
            d2 = ((cell_xy[:, :, None, :]
                   - path_p[None, None, :, :]) ** 2).sum(-1)
            dev = np.sqrt(d2.min(axis=2))
        # Small keep-right preference in the A* cost (local y is left of
        # the car): when both sides of a blocker are reachable the planner
        # chooses the right-hand side instead of cutting left.
        s_vals = y0 + (np.arange(cols) + 0.5) * res
        left_cost = np.maximum(0.0, s_vals[None, :])
        cost = (np.ones((rows, cols), dtype=float)
                + dev * DEV_PENALTY
                + self.grid_right_bias * left_cost)
        cost[occ] = np.inf

        if occ[start]:
            nudge = None
            for rr in range(max(0, r_start - 5), min(rows, r_start + 6)):
                for cc in range(max(0, c_start - 5), min(cols, c_start + 6)):
                    if not occ[rr, cc]:
                        nudge = (rr, cc)
                        break
                if nudge:
                    break
            if nudge is None:
                return None, False
            start = nudge

        # Goal: the route horizon point (i1) projected into the grid.  When
        # that cell is occupied or out of bounds the corridor is effectively
        # blocked: A* will not reach it and the best-reachable-cell fallback
        # below stops the car in front of the obstacle.
        f_h, s_h = to_local(float(pts[i1, 0]), float(pts[i1, 1]))
        r_goal = int(np.clip(round((f_h - x0) / res), 0, rows - 1))
        c_goal = int(np.clip(round((s_h - y0) / res), 0, cols - 1))
        goal = (r_goal, c_goal)
        if goal == start:
            return None, False

        open_heap: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
        g_cost = {start: 0.0}
        came: dict[tuple[int, int], tuple[int, int]] = {}
        closed: set[tuple[int, int]] = set()
        found = False
        while open_heap:
            _, cur = heapq.heappop(open_heap)
            if cur == goal:
                found = True
                break
            if cur in closed:
                continue
            closed.add(cur)
            r, c = cur
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < rows and 0 <= nc < cols):
                        continue
                    if occ[nr, nc]:
                        continue
                    if dr != 0 and dc != 0:
                        # no corner cutting between diagonally-touching walls
                        if occ[r + dr, c] or occ[r, c + dc]:
                            continue
                    step = math.hypot(dr, dc)
                    ncost = g_cost[cur] + cost[nr, nc] * step
                    if ncost < g_cost.get((nr, nc), float("inf")):
                        g_cost[(nr, nc)] = ncost
                        came[(nr, nc)] = cur
                        h = math.hypot((nr - r_goal) * res,
                                       (nc - c_goal) * res)
                        heapq.heappush(open_heap, (ncost + h, (nr, nc)))

        if not found:
            # No route to the horizon: keep the best reachable cell (lowest
            # cost + distance to the goal) so we can at least inch forward.
            best = None
            best_score = float("inf")
            for cell, gc in g_cost.items():
                if occ[cell]:
                    continue
                h = math.hypot((cell[0] - r_goal) * res,
                               (cell[1] - c_goal) * res)
                score = gc + h
                if score < best_score:
                    best_score = score
                    best = cell
            if best is None or best == start:
                return None, False
            goal = best
            reached_goal = False
        else:
            reached_goal = True

        path_cells = []
        cur = goal
        while True:
            path_cells.append(cur)
            if cur == start:
                break
            nxt = came.get(cur)
            if nxt is None:
                return None, False
            cur = nxt
        path_cells.reverse()

        world = []
        for r, c in path_cells:
            f = x0 + (r + 0.5) * res
            s = y0 + (c + 0.5) * res
            world.append((ox + f * hx + s * lx, oy + f * hy + s * ly))
        out = np.asarray(world, dtype=float)
        if len(out) < 3:
            return out, reached_goal
        # Smooth the A* zig-zag while keeping both endpoints fixed.
        for _ in range(2):
            nxt = out.copy()
            for i in range(1, len(out) - 1):
                nxt[i] = 0.25 * out[i - 1] + 0.5 * out[i] + 0.25 * out[i + 1]
            out = nxt
        return out, reached_goal

    def _window(self, route: np.ndarray, nearest: int):
        """Local planning window: (pts, i_start, i_end)."""
        pts = np.asarray(route[:, :2], dtype=float)
        n = len(pts)
        end = n - 1
        if n < 2 or nearest >= n - 1:
            return pts, min(nearest, n - 1), end
        d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(d)])
        base = cum[nearest]
        for i in range(nearest + 1, len(cum)):
            if cum[i] - base > self.horizon_m:
                end = i
                break
        return pts, int(nearest), int(end)

    def _sensor_window(self, lane_center, pos):
        """Local planning window on a sensor lane centre near the car."""
        pts = np.asarray(lane_center[:, :2], dtype=float)
        if len(pts) < 2:
            return pts, 0, max(0, len(pts) - 1)
        pos = np.asarray(pos, dtype=float)[:2]
        nearest = int(np.argmin(np.linalg.norm(pts - pos, axis=1)))
        return self._window(pts, nearest)

    def _single_edge_route_dev(self, edge, side: float, route, pos
                               ) -> float | None:
        """Median signed lateral offset of a single detected edge from
        the nav route near the car (positive = right of the route).

        A right edge (``side`` < 0) must sit to the right of the route
        and a left edge to the left.  An edge on the wrong side, or
        metres off the route, is a tracker lock onto the opposite line /
        another road's boundary (run 53: the "right" paint flipped to
        the left of the car and the push dragged the car off the road).
        Returns None when there is no reliable sample.
        """
        edge_xy = np.asarray(edge[:, :2], dtype=float)
        if len(edge_xy) < 2:
            return None
        route_xy = np.asarray(route[:, :2], dtype=float)
        if len(route_xy) < 2:
            return None
        pos = np.asarray(pos, dtype=float)[:2]
        d2 = np.sum((edge_xy - pos) ** 2, axis=1)
        near = edge_xy[d2 <= 30.0 ** 2]
        if len(near) < 2:
            return None
        offs = [_point_lat_offset(float(px), float(py), route_xy)
                for px, py in near]
        offs = [o for o in offs if math.isfinite(o)]
        if len(offs) < 2:
            return None
        return float(np.median(offs))

    def _sensor_nav_deviation(self, lane_center, route, pos) -> float | None:
        """Median lateral deviation of a sensor lane centre from the nav
        route in the near corridor ahead of the car.

        The paired lane is only trusted when it agrees with the route
        geometry: a wrong pairing (far lane's line, roadside paint, a
        guardrail shadow) shows up as a lane centre that sits metres off
        the route close to the car.  Returns None when there is no
        reliable sample (short lane / short route).
        """
        lane = np.asarray(lane_center[:, :2], dtype=float)
        if len(lane) < 2:
            return None
        route_xy = np.asarray(route[:, :2], dtype=float)
        if len(route_xy) < 2:
            return None
        pos = np.asarray(pos, dtype=float)[:2]
        d2 = np.sum((lane - pos) ** 2, axis=1)
        # Sample the lane where it is close to the car (0-30 m ahead /
        # behind).  A short lane (span ~4.5 m, a run 42/47/52 failure)
        # must still be checked: its centre can sit metres off the route
        # while all its points are "near", so require the whole lane here
        # instead of a 2-point near sample that returns None and skips
        # the trust guard.
        near = lane[d2 <= 30.0 ** 2]
        if len(near) < 2:
            return None
        offs = [_point_lat_offset(float(px), float(py), route_xy)
                for px, py in near]
        offs = [o for o in offs if math.isfinite(o)]
        if len(offs) < 2:
            return None
        return float(np.median(np.abs(offs)))

    def deform(self, route: np.ndarray, obstacles, nearest: int) -> np.ndarray:
        """Return a route deformed around obstacles (same length, Nx2)."""
        pts, i0, i1 = self._window(route, nearest)
        if not obstacles or i1 <= i0:
            return pts

        work = pts.copy()
        orig = pts.copy()
        # Local travel direction of the planning window.
        fwd = pts[i1] - pts[i0]
        fn = float(np.linalg.norm(fwd))
        if fn < 1e-9:
            fwd = np.array([0.0, 1.0])
        else:
            fwd = fwd / fn
        lat = np.array([-fwd[1], fwd[0]])  # left of the travel direction
        # The obstacle half extents along (fwd, lat) are constant for the
        # whole elastic-band solve; precompute them once instead of per
        # route point per iteration.
        centers = np.empty((len(obstacles), 2), dtype=float)
        half_fwd = np.empty(len(obstacles), dtype=float)
        half_lat = np.empty(len(obstacles), dtype=float)
        wall_side = np.zeros(len(obstacles), dtype=float)
        for k, ob in enumerate(obstacles):
            centers[k, 0] = ob.x
            centers[k, 1] = ob.y
            hf, hl = _obstacle_half_extents(ob, fwd, lat)
            half_fwd[k] = hf
            half_lat[k] = hl
            if self._is_roadside_wall(
                    ob, self._obstacle_route_profile(ob, pts, i0, i1),
                    pts=pts):
                # A long wall beside the road is a one-sided boundary:
                # it may push the path toward the road but never further
                # onto the shoulder.
                wall_lats = [_point_lat_offset(c[0], c[1], pts)
                             for c in _obstacle_corners(ob)]
                if wall_lats:
                    wall_side[k] = (1.0
                                    if float(np.median(wall_lats)) < 0.0
                                    else -1.0)
        need = half_lat + CAR_HALF_WIDTH + self.lateral_clear
        reach = half_fwd + self.anticipate
        denom = np.maximum(reach - half_fwd, 1e-6)
        active = np.arange(i0 + 1, i1 + 1)
        base = orig[active]
        lat_x = float(lat[0])
        lat_y = float(lat[1])
        for _ in range(self.relax_iters):
            cur = work[active]
            diff = cur[:, None, :] - centers[None, :, :]
            fwd_proj = diff[..., 0] * fwd[0] + diff[..., 1] * fwd[1]
            lat_proj = diff[..., 0] * lat[0] + diff[..., 1] * lat[1]
            mask = ((np.abs(fwd_proj) <= reach[None, :])
                    & (np.abs(lat_proj) < need[None, :]))
            # Taper the push before/after the box so the route blends back
            # into the nav corridor instead of kinking.
            taper = 1.0 - np.maximum(
                0.0, np.abs(fwd_proj) - half_fwd[None, :]) / denom[None, :]
            # Constant repulsion inside the cleared band: the elastic band
            # settles at the band edge (need) instead of decaying to a
            # point still inside the inflated box.
            force = self.push_gain * need[None, :] * taper
            # Roadside walls only repel toward the road.  A right-side
            # wall (wall_side=-1) pushes the path left whenever the path
            # is on the wall's side of the wall centre; a left-side wall
            # pushes right.  Ordinary obstacles still repel from both
            # sides so the path can pass either way.
            wall_mask = wall_side[None, :] != 0.0
            wall_dir = np.where(wall_mask, -wall_side[None, :], 0.0)
            wall_allow = np.where(
                wall_mask,
                lat_proj * wall_side[None, :] < need[None, :],
                True)
            side = np.where(
                wall_mask, wall_dir,
                np.where(lat_proj >= 0.0, 1.0, -1.0))
            use_mask = mask & wall_allow
            fx = np.where(use_mask, lat_x * side * force, 0.0).sum(axis=1)
            fy = np.where(use_mask, lat_y * side * force, 0.0).sum(axis=1)
            moved = cur + np.stack([fx, fy], axis=1)
            # Pull back toward the original nav corridor.
            moved += 0.30 * (base - moved)
            dev = moved - base
            nd = np.linalg.norm(dev, axis=1)
            over = nd > self.max_dev
            if np.any(over):
                scale = self.max_dev / nd[over]
                moved[over] = base[over] + dev[over] * scale[:, None]
            work[active] = moved

        # Smooth the deformed section so steering stays gentle.
        for _ in range(self.smooth_passes):
            nxt = work.copy()
            for i in range(i0 + 1, i1):
                nxt[i] = 0.25 * work[i - 1] + 0.5 * work[i] + 0.25 * work[i + 1]
            work = nxt
        return work

    # ---- speed planning ------------------------------------------------

    def speed(
        self,
        route: np.ndarray,
        obstacles,
        pos,
        heading: float,
        nearest: int,
        cruise: float,
        ahead_idx: int | None = None,
    ) -> tuple[float, float]:
        """Return (target_speed, nearest_obstacle_distance).

        The speed is the cruise speed limited by path curvature and by
        obstacles ahead.  Two distinct cases slow the car:

        * a real blocker - an obstacle whose centre sits inside the car's
          own track on the drive path (a parked car in the lane, a wall) -
          is approached with a kinematic braking curve, so the car eases
          off smoothly instead of charging at it and stops in front of it;
        * a tight pass-by - the (already deformed) path runs within
          ``corridor_half_w`` of the obstacle's own footprint (not inflated
          by the car width) - limits the speed so the car does not zip past
          something nearly scraping its mirrors.  A pass-by never pins the
          speed below ``PASS_BY_MIN_MPS`` (sparse raycast specks use the
          higher ``SPECK_PASS_BY_MIN_MPS``): the box is beside the path,
          not blocking it, so stopping every frame (stop-creep twitch) is
          wrong even when the box projects onto the route start.

        Roadside furniture 3 m or more off the route triggers neither, which
        keeps open roads fast.
        """
        self.last_corner = float(max(cruise, 0.0))
        self.last_obs_lim = None
        self.last_sharp = False
        pts = np.asarray(route[:, :2], dtype=float)
        n = len(pts)
        if n < 2:
            return max(cruise, 0.0), 999.0
        if ahead_idx is not None:
            self.last_sharp = corner_angle_deg(
                pts, nearest, ahead_idx=ahead_idx) \
                >= self.sharp_angle_deg
        else:
            self.last_sharp = corner_angle_deg(
                pts, nearest) >= self.sharp_angle_deg
        v = corner_speed(
            pts, nearest, cruise,
            ahead_idx=(ahead_idx if ahead_idx is not None else 24),
            sharp_angle_deg=self.sharp_angle_deg,
            sharp_speed_mps=self.sharp_corner_kph / 3.6)
        self.last_corner = float(v)
        nearest_obs = 999.0

        # Only inspect obstacles up to ~40 m along the path.
        d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(d)])
        i_end = n - 1
        base = cum[min(nearest, n - 1)]
        for i in range(nearest + 1, len(cum)):
            if cum[i] - base > 40.0:
                i_end = i
                break
        window = slice(min(nearest, n - 1), i_end + 1)
        seg_pts = pts[window]
        if len(seg_pts) < 2:
            return max(v, 0.0), nearest_obs
        win_x0 = float(seg_pts[:, 0].min()) - self.corridor_half_w
        win_x1 = float(seg_pts[:, 0].max()) + self.corridor_half_w
        win_y0 = float(seg_pts[:, 1].min()) - self.corridor_half_w
        win_y1 = float(seg_pts[:, 1].max()) + self.corridor_half_w

        for ob in obstacles:
            # Roadside walls are boundaries, not speed obstacles: a wall
            # lining the road 1-2 m off the route would otherwise pin the
            # speed to a crawl on every frame in town streets (the pass-by
            # corridor test), which is the "random braking" symptom.  A
            # wall that genuinely spans the corridor fails the one-side
            # test and keeps limiting the speed.  All obstacle geometry is
            # judged on the ~40 m window (seg_pts): projecting onto the
            # full nav route (950+ pts) for every obstacle took 400+ ms
            # per frame on long routes (run 49/50 plan=422ms).
            if self._is_roadside_wall(
                    ob, self._obstacle_route_profile(
                        ob, seg_pts, 0, len(seg_pts) - 1), pts=seg_pts):
                continue
            bx0, by0, bx1, by1 = _obstacle_aabb(
                ob, self.corridor_half_w)
            if (bx1 < win_x0 or bx0 > win_x1
                    or by1 < win_y0 or by0 > win_y1):
                continue            # Signed along-route position of the box centre relative to
            # the car.  The polyline projection clamps to the route start
            # (arc_c can never go negative), so also measure along the
            # first segment direction: an obstacle whose centre sits
            # clearly behind the car (roadside furniture we have already
            # passed) must not drag the speed to zero.
            arc_c, lat_c = _point_route_pos_np(ob.x, ob.y, seg_pts)
            u0x = seg_pts[1, 0] - seg_pts[0, 0]
            u0y = seg_pts[1, 1] - seg_pts[0, 1]
            n0 = math.hypot(u0x, u0y)
            rel0 = 0.0 if n0 < 1e-9 else (
                (ob.x - seg_pts[0, 0]) * u0x + (ob.y - seg_pts[0, 1]) * u0y) / n0
            if rel0 < -2.0 and arc_c <= 0.0:
                continue
            if arc_c >= 40.0:
                continue
            # A small roadside object whose centre sits clearly outside the
            # car's track cannot block the lane, no matter how its inflated
            # AABB pokes into the corridor window.  A pole / fence post
            # 3-5 m off the route would otherwise pin the speed to a crawl
            # on every frame (the pass-by corridor test uses the inflated
            # AABB), which is the "random braking on an empty road"
            # symptom.  Large obstacles still get the full treatment below.
            # Objects 2 m off the path still count as a tight pass-by and
            # keep the crawl limit; only clear roadside furniture (>= 3 m)
            # is ignored.
            if (abs(lat_c) >= CAR_HALF_WIDTH + 2.0
                    and _obstacle_footprint_area(ob) <= 8.0):
                continue
            closest = 999.0
            lon = 0.0
            seg_k = 0
            for k in range(len(seg_pts) - 1):
                dd = _obstacle_seg_dist(
                    ob, seg_pts[k, 0], seg_pts[k, 1],
                    seg_pts[k + 1, 0], seg_pts[k + 1, 1])
                if dd < closest:
                    closest = dd
                    lon = cum[min(nearest, n - 1) + k] - base
                    seg_k = k
                if dd < 1e-6:
                    break
            if closest >= 999.0:
                continue
            nearest_obs = min(nearest_obs, closest)
            # The closest point can sit exactly on the window start (the
            # car) even when the box is really a few metres ahead (its
            # footprint reaches back to the car) or off to the side; a
            # bare "0 m" would slam the speed to zero for a box that is
            # still metres away.  Fall back to the box-centre projection
            # in that case.
            if lon <= 0.0:
                lon = max(0.0, arc_c - base)
            if lon >= 40.0:
                continue
            # A box whose centre sits inside the car's own track on the
            # drive path is a real lane blocker: it cannot be driven past,
            # so the kinematic braking curve (possibly down to a stop)
            # applies.  Anything further off the path is a pass-by: only
            # the tight-corridor limit applies and it never pins the speed
            # below a crawl - the car is beside (or will pass beside) the
            # box, so a zero-speed demand every frame is what caused the
            # stop/creep twitch when a roadside grove projected onto the
            # trimmed route start (lon = 0) while the detour ran around it.
            in_lane = abs(lat_c) < CAR_HALF_WIDTH + 0.3
            # A sparse raycast cluster's half-extents are a noise floor
            # (single/few hit points become a fixed 0.9 m box), not a real
            # footprint.  When its centre sits beyond the car half-width
            # (the detour has already routed around it), treating it as an
            # in-lane blocker pins the speed to zero on every frame, which
            # is the twitch/park regression on a roadside grove.  Only a
            # sparse cluster whose centre is truly inside the car's track
            # stays an in-lane blocker, so the car never charges a real
            # obstacle.
            if (in_lane and is_sparse_raycast_speck(ob)
                    and abs(lat_c) >= CAR_HALF_WIDTH):
                in_lane = False
            if in_lane or closest < self.corridor_half_w:
                v_max = math.sqrt(
                    _vehicle_speed_along(ob, seg_pts, seg_k) ** 2
                    + 2.0 * DECEL_MPS2 * max(0.0, lon - STOP_MARGIN_M))
                if not in_lane:
                    floor = (SPECK_PASS_BY_MIN_MPS
                             if is_sparse_raycast_speck(ob)
                             else PASS_BY_MIN_MPS)
                    v_max = max(v_max, floor)
                # Only the obstacle that actually pins the speed may set
                # last_obs_lim; a far-away, higher limit must not mask the
                # creep trigger after a close obstacle already stopped us.
                if v_max < v:
                    v = v_max
                    self.last_obs_lim = float(v_max)
        return max(v, 0.0), nearest_obs


def creep_speed(blocked: bool, obs_lim, desired_speed: float,
                speed: float, since, now: float,
                creep_mps: float = 1.5, hold: float = 1.5):
    """ACC-style creep decision.

    When a kinematic obstacle limit pins the target at 0 but a drivable
    path still exists (``blocked`` is False), inch forward slowly instead
    of parking forever.  Returns ``(target_speed, creep_active, since)``;
    ``since`` is the frame time the pinned state started, or None when the
    car is not in the pinned state.
    """
    if (not blocked and obs_lim is not None
            and obs_lim <= 0.01 and desired_speed <= 0.01
            and speed < 1.0):
        since = since if since is not None else now
        if now - since > hold:
            return float(creep_mps), 1, since
        return float(desired_speed), 0, since
    return float(desired_speed), 0, None