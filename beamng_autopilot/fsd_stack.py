"""FSDStack: the integrated multi-camera perception+planning pipeline.

Wires the FSD-style modules built in previous milestones into one
callable stack that mirrors the data flow of a Tesla FSD stack:

  camera ring -> HydraNets (multi-task heads) -> BEV/vector space
  (occupancy + feature fusion) -> layered planner -> control hint.

It is intentionally *standalone*: it does not touch the existing
``AutopilotSession`` plan path, so the validated rule driving (94.6%
route result) keeps working untouched.  The stack is exercised live by
probes and shadow recording; when it is proven it can replace the old
path without disturbing it first.

Usage::

    stack = FSDStack(conn, mode="tech")
    out = stack.tick()   # one full perception+planning tick
    out.bev             # (N,N) occupancy raster
    out.best_path       # chosen trajectory or None
"""

from __future__ import annotations

import math
import time

import numpy as np

from beamng_autopilot.occupancy import (
    OccupancyGrid,
    fuse_obstacles_to_grid,
    project_road_mask_to_grid,
)
from beamng_autopilot.planning import (
    Constraints,
    Scene,
    sample_arc,
    sample_lane_shift,
    select_trajectory,
)
from beamng_autopilot.planner import CAR_HALF_WIDTH, forward_clearance_m, path_forward_clearance_m
from beamng_autopilot.lane import (
    LANE_WIDTH_DEFAULT_M,
    build_lidar_corridor,
    pair_lane_markings,
    choose_sensor_lane,
)
from beamng_autopilot.runtime import (
    build_camera_ring_provider,
    build_range_provider,
)
from beamng_autopilot.vision.hydra import FrameContext, HydraNet


class FSDTick:
    """One planning-tick result from ``FSDStack``."""

    def __init__(self):
        self.bev: np.ndarray | None = None      # (N, N) occupancy raster
        self.drivable: np.ndarray | None = None
        self.observed: np.ndarray | None = None  # (N, N) sensor-seen mask
        self.best_path: np.ndarray | None = None
        self.lane_ref: np.ndarray | None = None  # sensor lane centreline
        self.lane_left: np.ndarray | None = None  # paired lane boundary (world)
        self.lane_right: np.ndarray | None = None  # paired lane boundary (world)
        self.lane_width: float = 0.0
        self.best_speed: float = 0.0            # planned speed at the start
        self.min_speed: float = 0.0             # lowest speed on the path
        self.n_candidates: int = 0
        self.meta: dict = {}
        self.head_outputs: dict = {}
        self.errors: dict = {}
        self.ray_hits: list = []
        self.forward_clearance: float = float("inf")
        self.path_forward_clearance: float = float("inf")


class FSDStack:
    """Integrated FSD-style perception + planning pipeline."""

    def __init__(self, conn, mode: str = "tech",
                 grid_n: int = 60, grid_res: float = 0.5,
                 ring_roles=None, heads=None,
                 cam_w: int = 320, cam_h: int = 240,
                 temporal: bool = True, tau_s: float = 1.5,
                 range_every_n: int = 1,
                 semantic_every_n: int = 1,
                 object_every_n: int = 1):
        self.conn = conn
        self.grid_n = int(grid_n)
        self.grid_res = float(grid_res)
        self.heads = list(heads) if heads else []

        ring, self.mode = build_camera_ring_provider(
            conn, mode, cam_w, cam_h, roles=ring_roles)
        self.ring = ring
        if self.ring is None:
            self.mode = "steam-front"    # front camera only available
        self.range_prov, _ = build_range_provider(conn, mode)
        # LiDAR scan throttling: a 360 scan + clustering costs ~300-400 ms.
        # range_every_n>1 reuses the previous scan on intermediate ticks
        # (the temporal occupancy filter already smooths over frames); the
        # safety layers still see a fresh wall within one control burst.
        self.range_every_n = max(1, int(range_every_n))
        self._range_skip = 0
        self._last_range = None
        # Per-head throttling: the expensive heads (semantic UNet
        # ~100-300 ms, YOLO object ~100-200 ms on the live 400x300
        # front frame) run every ``semantic_every_n`` / ``object_every_n``
        # ticks and intermediate ticks reuse their last output.  Cheap
        # heads (traffic) still run on every fresh frame; LiDAR is
        # throttled separately by ``range_every_n``.
        self.semantic_every_n = max(1, int(semantic_every_n))
        self.object_every_n = max(1, int(object_every_n))
        self._head_skip: dict[str, int] = {}
        self._last_heads: dict = {}

        self.hydra = HydraNet()
        for head in self.heads:
            self.hydra.add(head)

        self.constraints = Constraints(
            w_collision=5.0, w_curvature=0.5, w_lane_align=1.0)
        self.target_speed = 8.0  # plan cruise speed (m/s); drive can raise
        # Map-prior own-lane width (fallback when sensors see no lane)
        self.map_lane_width_m = LANE_WIDTH_DEFAULT_M
        # Raw-sensor forward corridor half width (car half width + margin)
        # used by the independent FSD safety layer (m5_fsd_drive).
        self.ego_half_width = CAR_HALF_WIDTH + 0.5

        # FSD-style temporal occupancy fusion: single-frame LiDAR glitches
        # must not create a phantom wall or erase a real one.
        self.temporal = bool(temporal)
        self.occ_filter = None
        self._tick_t0 = None
        # Per-frame lane-fusion state (choose_sensor_lane) across ticks
        # to prevent flicker between vision / lidar / fallback.
        self._lane_fusion_state: dict = {}
        if self.temporal:
            from beamng_autopilot.temporal import TemporalOccupancyFilter
            self.occ_filter = TemporalOccupancyFilter(
                n=int(grid_n), res=float(grid_res), tau_s=float(tau_s))

    # ------------------------------------------------------------------
    def tick(self, st=None, route_ref: np.ndarray | None = None,
             include_bev: bool = True,
             map_lane_override=None) -> FSDTick:
        """Run one full perception + planning tick.

        ``st`` is a vehicle state with ``.pos`` / ``.heading`` / ``.speed``
        (falls back to a live ``get_state()`` when None).  ``route_ref``
        is the nav route to plan along (defaults to straight ahead).
        """
        out = FSDTick()
        if st is None:
            st = self.conn.get_state()
        pos = np.asarray(st.pos, dtype=float)
        heading = float(st.heading)
        _tw = time.time()
        _times: dict[str, float] = {}

        # --- 1) camera ring -> HydraNet heads ---------------------------
        snap: dict = {}
        if self.ring is not None:
            try:
                snap = self.ring.grab_ring()
            except Exception as exc:
                out.errors["ring"] = str(exc)
        if snap:
            role = "front_main" if "front_main" in snap \
                else next(iter(snap))
            frame, cam = snap[role]
            ctx = FrameContext(
                frame_rgb=frame, cam=cam, pos=pos, heading=heading,
                ground_z=float(pos[2]) if len(pos) > 2 else 0.0,
                role=role)
            heads: dict = {}
            _skip = getattr(self, '_head_skip', None)
            if _skip is None:
                _skip = {}
                self._head_skip = _skip
            _last = getattr(self, '_last_heads', None)
            if _last is None:
                _last = {}
                self._last_heads = _last
            for _name, _head in self.hydra._heads.items():
                if _name == "semantic":
                    _n = int(getattr(self, 'semantic_every_n', 1))
                elif _name == "object":
                    _n = int(getattr(self, 'object_every_n', 1))
                else:
                    _n = 1
                if _skip.get(_name, 0) > 0 and _last.get(_name) is not None:
                    _skip[_name] = _skip.get(_name, 0) - 1
                    heads[_name] = _last[_name]
                    continue
                try:
                    heads[_name] = _head.run(ctx)
                    _last[_name] = heads[_name]
                    if _n > 1:
                        _skip[_name] = _n - 1
                except Exception as _exc:
                    self.hydra.errors[_name] = str(_exc)
            out.head_outputs = heads
            _times['ring'] = round((time.time() - _tw) * 1000.0, 1)
            _tw = time.time()

        # --- 2) BEV vector space (occupancy grid) -----------------------
        grid = OccupancyGrid(self.grid_n, self.grid_n, self.grid_res,
                             origin=(float(pos[0]), float(pos[1])),
                             heading=heading)
        if include_bev:
            semantic = out.head_outputs.get("semantic")
            if semantic is not None and "road" in semantic.masks \
                    and "front_main" in snap:
                project_road_mask_to_grid(
                    grid, semantic.masks["road"],
                    snap["front_main"][1], pos, heading, step=4)
            try:
                # getattr defaults keep __new__-built test stubs
                # (no __init__) working: they always scan.
                if getattr(self, '_range_skip', 0) <= 0:
                    rng = self.range_prov.scan(pos)
                    self._last_range = rng
                    self._range_skip = max(
                        0, int(getattr(self, 'range_every_n', 1)) - 1)
                else:
                    self._range_skip -= 1
                    rng = getattr(self, '_last_range', None)
                if rng is None:
                    raise RuntimeError("no range sample")
                out.ray_hits = list(getattr(rng, "ray_hits", []) or [])
                out.forward_clearance = forward_clearance_m(
                    out.ray_hits, pos,
                    np.array([math.cos(heading), math.sin(heading)]),
                    half_width=float(getattr(self, "ego_half_width",
                                             CAR_HALF_WIDTH + 0.5)))
                out.meta["fwd_clearance"] = round(
                    float(out.forward_clearance), 3)
                # Fuse only the CLUSTERED obstacle boxes (walls, vehicles,
                # poles) into the BEV.  The raw ray hits are ignored for
                # the obstacle layer: in a tree-lined town road dozens of
                # ground/foliage reflections land inside the drivable lane,
                # push the cell occupancy past 0.6 and the temporal fusion
                # then re-derives obstacle=(bev>=0.6), turning the whole
                # road ahead "occupied" and making every FSD path graze an
                # obstacle (town runs 2026-08-21 - 86% of the forward 4 m
                # corridor was marked occupied).  Clustered boxes keep the
                # real walls/vehicles, the drivable layer keeps the road.
                fuse_obstacles_to_grid(grid, rng.obstacles)
                out.meta["n_obstacles"] = len(rng.obstacles)
            except Exception as exc:
                out.errors["range"] = str(exc)
            # YOLO object head: fuse detected vehicles/pedestrians into
            # the same vector space as the LiDAR clusters (the head may
            # be throttled by object_every_n; its world-space obstacles
            # are re-projected on the next fresh run).
            obj = out.head_outputs.get("object")
            if obj is not None:
                try:
                    _obs = getattr(obj, "obstacles", None) or []
                    if _obs:
                        fuse_obstacles_to_grid(grid, _obs)
                    out.meta["n_object_obstacles"] = len(_obs)
                except Exception as exc:
                    out.errors["object"] = str(exc)
        _times['range'] = round((time.time() - _tw) * 1000.0, 1)
        _tw = time.time()
        out.bev = grid.as_raster()
        out.drivable = grid.drivable
        out.observed = getattr(grid, "observed", None)

        # Temporal fusion: smooth the single-frame occupancy before the
        # planner / safety layers read it (a one-frame glitch is neither a
        # phantom wall nor a vanished one).
        occ_filter = getattr(self, "occ_filter", None)
        if occ_filter is not None:
            now = time.time()
            t0 = getattr(self, "_tick_t0", None)
            dt = (now - t0) if t0 is not None else 0.0
            if dt < 0 or dt > 5.0:
                dt = 0.0
            self._tick_t0 = now
            occ_filter.update(out.bev, dt)
            out.bev = occ_filter.raster()
            # mark the fused occupancy back into the grid obstacle layer so
            # planner collision checks use the smoothed space too
            from beamng_autopilot.occupancy import OccupancyGrid as _OG
            if isinstance(grid, _OG):
                grid.obstacle[:] = (out.bev >= 0.6).astype(np.uint8)
                grid.occupancy[:] = np.asarray(out.bev, dtype=np.float32)

        # --- 3) layered planner -----------------------------------------
        # ``has_nav_route``: only a caller-supplied map/nav route carries
        # real road geometry for the map-prior own-lane fallback; a
        # synthetic straight line does not.
        has_nav_route = route_ref is not None and len(route_ref) >= 2
        if route_ref is None or len(route_ref) < 2:
            xs = np.linspace(0, 40, 41)
            route_ref = np.column_stack(
                [pos[0] + xs * np.cos(heading),
                 pos[1] + xs * np.sin(heading)])
        # Lane reference: use the SENSOR lane centre (vision lane-marking
        # pairing -> LiDAR corridor -> fusion) when available, and only
        # fall back to the BEV drivable-space centreline / nav route.  The
        # BEV centre of the *whole* drivable road is the road centreline -
        # on a two-way road that IS the centre line the car must never
        # ride.  A real stack keeps to the centre of ITS OWN lane, which
        # is what pair_lane_markings / build_lidar_corridor deliver.
        lane_frame = self._sensor_lane(out, pos, heading)
        sensor_paired = (lane_frame is not None
                         and getattr(lane_frame, "paired", False))
        lane_ref = None
        lane_left = None
        lane_right = None
        lane_width = 0.0
        if lane_frame is not None:
            if getattr(lane_frame, "center", None) is not None \
                    and len(lane_frame.center) >= 2:
                lane_ref = np.asarray(lane_frame.center, dtype=float)[:, :2]
            # Only a REAL two-sided detection provides hard lane
            # boundaries; a single-edge mirror is not a physical edge the
            # no-cross rule may enforce.
            if sensor_paired:
                lane_left = getattr(lane_frame, "left", None)
                lane_right = getattr(lane_frame, "right", None)
                lane_width = float(getattr(lane_frame, "width", 0.0) or 0.0)
        # Map-prior own-lane fallback: when NO sensor lane (vision or
        # LiDAR) could be paired this frame, derive the ego lane from the
        # nav route - half a lane width RIGHT of the road centreline
        # (right-hand traffic), with the centreline as the hard left
        # boundary and the road's right edge as the hard right boundary.
        # Without this the planner tracked the road CENTRE line whenever
        # ``lane_paired=0`` (town runs 2026-08-22: g8/g10/g12 rode the
        # centre line end to end, and the no-cross rule had no boundaries
        # to enforce).  The road-graph route starts at the nearest road
        # node, so it also anchors the car's own lane even when the ego
        # has drifted off the A* polyline.
        map_lane = map_lane_override
        # Built whenever a nav route exists: it is both the fallback for a
        # missing sensor lane AND the override when a sensor lane heads into
        # a different road at a junction (heading gate below).  A caller
        # may supply ``map_lane_override`` (real road-edge own-lane window
        # from map_lane_edges); when absent the synthetic map lane is built.
        if has_nav_route and map_lane is None:
            try:
                from beamng_autopilot.planning.local_route import (
                    map_lane_local)
                map_lane = map_lane_local(route_ref, pos, heading)
            except Exception:
                map_lane = None
        # Heading gate: a PAIRED sensor lane is only trustworthy when its
        # near-ahead direction agrees with the nav route (or the ego
        # heading).  At a junction the vision/LiDAR pairing can lock onto a
        # DIFFERENT road whose corridor reads clear - following it drives
        # the car off the navigational route (town run 2026-08-22: the
        # paired lane headed into the side road and the car left the road
        # and wedged).  Reject the whole sensor lane (centre + hard
        # boundaries) and fall back to the map-prior own lane.
        lane_rejected = False
        # Gate ANY sensor lane (paired or not) against the map-prior own
        # lane: an unpaired vision/LiDAR corridor is often the whole-road
        # centre, which on a two-way road sits on the oncoming side of the
        # own lane.  Without the gate that centre was fed to the planner
        # as the lane reference, the chosen path parked 3.6 m off it and
        # the safety monitor declared "path near lane edge" -> src=none
        # stop at (734.9,753.4) mountain run 2026-08-23.
        if lane_frame is not None and map_lane is not None:
            try:
                from beamng_autopilot.planning.arbiter import (
                    lane_heading_ok, lane_side_ok)
                # Bearing gate: the lane must HEAD the same way as the
                # route (junction pairing onto a side road is rejected).
                side_bad = False
                if not lane_heading_ok(route_ref, lane_ref, pos, heading):
                    lane_rejected = True
                # SIDE gate: the lane centre must sit clearly RIGHT of the
                # road centreline (own lane).  A lane locked onto the
                # ONCOMING lane passes the bearing gate (same direction)
                # but still steers the car over the centre line (town runs
                # 2026-08-22: the car rode the centre/oncoming lane end
                # to end with lane_src=sensor).  A lane sitting ON the
                # centreline is the WHOLE-ROAD free corridor, not the own
                # lane - trusting it parks the car on the centre line and
                # the switch to the map-prior own lane then forces a
                # 2-3 m over-correction that swings it off the road edge
                # (mountain run 2026-08-27 run_fix31: after the junction
                # the car rode the centre line, then over-corrected right
                # and wedged at (741.2,745.7)).  Require the lane centre
                # at least 0.4 m right of the route, else the map-prior
                # own lane replaces it.
                elif not lane_side_ok(lane_ref, route_ref, pos,
                                     left_max_m=-0.4):
                    lane_rejected = True
                    side_bad = True
                if lane_rejected:
                    out.meta["lane_reject_reason"] = (
                        "side" if side_bad else "heading")
                    sensor_paired = False
                    lane_ref = None
                    lane_left = None
                    lane_right = None
                    lane_width = 0.0
            except Exception:
                pass
        if map_lane is not None:
            mc, ml, mr = map_lane
            # An unpaired sensor centre is often the whole-road centre
            # (LiDAR free corridor / single-edge mirror) - never trust it
            # as the lane-keep reference over the map-prior own lane.
            if not sensor_paired:
                lane_ref = mc
            if lane_left is None or lane_right is None:
                lane_left = ml if lane_left is None else lane_left
                lane_right = mr if lane_right is None else lane_right
            if lane_width <= 0.0:
                # Real road-edge width when the map lane comes from
                # DecalRoad edges (median left-right distance); fall back
                # to the fixed map-prior lane width otherwise.
                _w = 0.0
                try:
                    _wa = np.linalg.norm(
                        np.asarray(ml, dtype=float)[:, :2]
                        - np.asarray(mr, dtype=float)[:, :2], axis=1)
                    _wf = _wa[np.isfinite(_wa)]
                    if _wf.size:
                        _w = float(np.median(_wf))
                except Exception:
                    _w = 0.0
                lane_width = (_w if _w > 0.0
                              else float(getattr(self, "map_lane_width_m", 0.0)
                                         or LANE_WIDTH_DEFAULT_M))
        if lane_ref is None or len(lane_ref) < 4:
            # BEV drivable-space fallback when there is NO nav route to
            # derive a map-prior own lane from (standalone probes / unit
            # stubs): the lateral centre of the FREE corridor.  In real
            # nav runs the map lane (or a paired sensor lane) takes
            # priority, so this whole-road centre never reaches the
            # planner's lateral reference there.
            bev_ref = self._bev_drivable_center(grid, pos, heading)
            if bev_ref is not None and len(bev_ref) >= 4:
                lane_ref = bev_ref
        out.lane_ref = (np.asarray(lane_ref, dtype=float)
                        if lane_ref is not None else None)
        if (lane_frame is not None and getattr(lane_frame, "paired", False)) \
                or map_lane is not None:
            if lane_left is not None:
                out.lane_left = np.asarray(lane_left, dtype=float)[:, :2]
            if lane_right is not None:
                out.lane_right = np.asarray(lane_right, dtype=float)[:, :2]
            out.lane_width = lane_width
        # Route intent vs sensor lane: the map/nav route is the heading
        # the car must follow (FSD planning consumes the route as the
        # navigational goal), while the sensor lane is the lateral
        # lane-keep reference.  At a junction the free-space centreline
        # alone can point straight ahead through a turn (start->corner
        # town runs 2026-08-21: the car followed a -135 deg sensor centre
        # and drove off-road instead of curving onto the nav route).  So
        # plan ALONG the route when it is ego-anchored (starts near the
        # car), and use the sensor lane as the lateral lane reference.
        route_anchored = (route_ref is not None and len(route_ref) >= 4
                          and float(np.linalg.norm(
                              np.asarray(route_ref[0], dtype=float)[:2]
                              - np.asarray(pos[:2], dtype=float))) <= 6.0)
        # Route intent is the navigational goal, but when that route's near
        # corridor is genuinely occupied while the sensor lane ahead is
        # free, plan along the DRIVABLE sensor lane instead.  A route that
        # cuts through a wall (single-marker ``setPath`` produced a straight
        # line through town buildings) must not keep pushing the car at the
        # wall just because it is ego-anchored (town run 2026-08-21).
        from beamng_autopilot.planning.arbiter import choose_plan_route
        _base_route = np.asarray(route_ref, dtype=float)[:, :2]
        if route_anchored:
            plan_route = choose_plan_route(
                _base_route, lane_ref, pos, heading, grid)
        else:
            plan_route = (lane_ref if lane_ref is not None
                          and len(lane_ref) >= 4 else _base_route)
        if plan_route is None or len(plan_route) < 2:
            plan_route = _base_route
        if lane_ref is None:
            lane_ref = plan_route
        # out.lane_ref drives the *lateral* lane-keep reference (sensor
        # lane centre when available); plan_route stays the navigational
        # intent in the planner's Scene.
        out.meta["lane_src"] = (
            "map_lane" if lane_rejected
            else "sensor" if lane_frame is not None
            else "map_lane" if map_lane is not None else "bev/route")
        out.meta["lane_paired"] = int(
            bool(lane_frame is not None and getattr(lane_frame, "paired", False)))
        if lane_rejected:
            out.meta["lane_reject"] = "heading"
            out.meta["lane_paired"] = 0
        if lane_frame is not None or map_lane is not None:
            out.meta["lane_width"] = round(lane_width, 2)

        # Only pass a sensor lane centre to Scene (the planner uses
        # scene.lane_ref for lateral alignment).  The BEV drivable-space
        # centre is the centre of the whole road — on a two-way road that
        # IS the centre line the car must never ride; passing it as
        # lane_ref would pull the planner toward the centre line.
        # out.lane_ref (set above) still carries the BEV fallback for the
        # safety monitor but the planner Scene must not see it.
        # Only sensor lanes (or the map-prior OWN lane) may steer the
        # planner's lateral alignment; the BEV whole-road centre must not.
        scene_lane_ref = (lane_ref if lane_frame is not None
                          or map_lane is not None else None)
        scene = Scene(pos=pos, heading=heading, grid=grid,
                      route=plan_route, lane_ref=scene_lane_ref,
                      lane_left=lane_left, lane_right=lane_right,
                      lane_width=lane_width,
                      target_speed=getattr(self, "target_speed", 8.0))
        # Town corners need a tighter arc fan than a highway fan: a
        # 5-8 m radius bend is 0.12-0.2 rad/m, and the old 0.10 rad/m
        # cap (10 m radius) could not turn away from a corner wall
        # (town runs 2026-08-21) - it kept pressing the throttle into
        # the wall.  Sample to the physical steer limit and add wider
        # lateral shifts so the planner can actually dodge a near wall.
        fans = sample_arc(pos, heading, speed=max(2.0, float(st.speed)),
                          max_steer=0.5, n_curv=13, max_curv=0.25)
        shifts = sample_lane_shift(plan_route,
                                   offsets=(-3.0, -1.5, 1.5, 3.0))
        for c in shifts.candidates:
            fans.add(c.path, c.meta.get("kind", "shift"),
                     offset=c.meta.get("offset", 0.0))
        # The LANE CENTRE itself is a candidate.  The synthetic shifts
        # blend the route over 8 m, so at low speed a 7 m PurePursuit
        # lookahead sits inside that blend and steers RIGHT while the
        # road curves LEFT (mountain runs 2026-08-26 run_fix8/9: the
        # planner flipped between shift (right) and arc (left) every
        # frame, the car oscillated instead of turning into the first
        # hairpin and stalled off-route at (724.8, 753.2)).  Tracking
        # the lane centre (sensor or map-prior own lane) keeps the car
        # in its lane with no blend wiggle; its alignment cost is ~0 so
        # it wins whenever it is drivable.
        if scene_lane_ref is not None and len(scene_lane_ref) >= 4:
            fans.add(np.asarray(scene_lane_ref, dtype=float)[:, :2],
                     "lane_center", offset=0.0)
        out.n_candidates = len(fans.candidates)
        best, meta = select_trajectory(scene, fans, self.constraints)
        out.best_path = best
        # Path-aware forward clearance: safety layer evaluates the chosen
        # trajectory corridor instead of the raw heading corridor, so a
        # turn away from a wall does not force a stop (town runs 2026-08-21).
        if best is not None and len(best) >= 2 and out.ray_hits:
            out.path_forward_clearance = path_forward_clearance_m(
                best, out.ray_hits,
                half_width=float(getattr(self, "ego_half_width",
                                         CAR_HALF_WIDTH + 0.5)))
            out.meta["path_fwd_clearance"] = round(
                float(out.path_forward_clearance), 3)
        out.meta["planner"] = meta
        out.meta["total_candidates"] = out.n_candidates
        # the chosen path's speed profile (planning-side longitudinal plan)
        sp = meta.get("speed_profile")
        if best is not None and sp is not None and len(sp):
            out.best_speed = float(sp[0])
            out.min_speed = float(np.asarray(sp).min())
            out.meta["best_speed"] = out.best_speed
            out.meta["min_speed"] = out.min_speed

        # Longitudinal plan along the NAV route (route_ref), never the
        # sensor lane: choose_plan_route may pick a straight sensor
        # corridor that does not contain the bend, and the profile then
        # allows 5+ m/s into a hairpin (run 2026-08-23).  The nav local
        # route carries the real curvature; v[0] becomes the bend speed
        # via the look-ahead propagation.
        try:
            from .speed_profile import speed_profile_for_path as _spf
            _sr = _spf(np.asarray(route_ref, dtype=float)[:, :2], scene,
                       target_speed=getattr(self, "target_speed", 8.0))
            if len(_sr):
                out.best_speed = float(_sr[0])
                out.min_speed = float(np.asarray(_sr).min())
                out.meta["best_speed"] = out.best_speed
                out.meta["min_speed"] = out.min_speed
                out.meta["plan_src"] = "route"
        except Exception:
            out.meta["plan_src"] = "candidate"


        # Run the topology head over the sensor lane derived from the
        # semantic markings (its graph needs a real LaneFrame, not just a
        # frame context).
        topology = self.hydra._heads.get("topology")
        if topology is not None:
            lane_frame = self._sensor_lane_from_semantic(
                out.head_outputs, pos, heading)
            ctx = FrameContext(
                frame_rgb=np.zeros((1, 1, 3), dtype=np.uint8),
                cam=None, pos=pos, heading=heading,
                ground_z=float(pos[2]) if len(pos) > 2 else 0.0,
                role="front_main")
            try:
                topo_out = topology.run(ctx, sensor_lane=lane_frame)
                out.head_outputs["topology"] = topo_out
            except Exception:
                pass
        out.meta.update(semantic_to_meta(out.head_outputs))
        _times['plan'] = round((time.time() - _tw) * 1000.0, 1)
        _times['total'] = round(sum(_times.values()), 1)
        out.meta['tick_ms'] = _times
        return out


    def _sensor_lane(self, out, pos, heading):
        """Fused sensor lane: vision markings -> LiDAR corridor -> fusion.

        Returns a ``LaneFrame`` whose centre is the ego lane centre in
        world coordinates and whose left/right (when ``paired``) are the
        detected lane boundaries - the correct lateral reference for a
        real FSD, unlike the whole-road drivable centre.
        """
        try:
            vision = self._sensor_lane_from_semantic(
                out.head_outputs, pos, heading)
        except Exception:
            vision = None
        lidar = None
        if out.ray_hits:
            try:
                lidar = build_lidar_corridor(out.ray_hits, pos, heading)
            except Exception:
                lidar = None
        try:
            frame = choose_sensor_lane(
                vision, lidar, pos, heading,
                state=getattr(self, "_lane_fusion_state", None))
        except Exception:
            frame = vision or lidar
        if frame is None:
            return None
        # Re-anchor the centre at the CURRENT ego.  The fusion state may
        # return a held lane computed at an earlier pose; the planner
        # needs a drivable reference starting at/near the car (near ->
        # far order, ego prepended) just like _bev_drivable_center.
        center = np.asarray(frame.center, dtype=float)[:, :2]
        center = center[np.isfinite(center).all(axis=1)]
        if len(center):
            pos2 = np.asarray(pos[:2], dtype=float)
            fwd = np.array([math.cos(float(heading)), math.sin(float(heading))])
            fwd_m = (center - pos2) @ fwd
            ahead = center[fwd_m > -0.5]
            if len(ahead) >= 2:
                d0 = np.linalg.norm(ahead - pos2, axis=1)
                ahead = ahead[np.argsort(d0)]
                if float(np.linalg.norm(ahead[0] - pos2)) > 2.0:
                    ahead = np.vstack([pos2, ahead])
                frame.center = ahead
            else:
                d0 = np.linalg.norm(center - pos2, axis=1)
                center = center[np.argsort(d0)]
                frame.center = np.vstack([pos2, center])
        return frame

    def _sensor_lane_from_semantic(self, head_outputs, pos, heading):
        """Pair the semantic head's world markings into a LaneFrame."""
        try:
            from beamng_autopilot.lane import pair_lane_markings
            sem = head_outputs.get("semantic")
            if sem is None:
                return None
            markings = sem.meta.get("markings", [])
            if not markings:
                return None
            return pair_lane_markings(markings, pos, heading)
        except Exception:
            return None

    def _bev_drivable_center(self, grid, pos, heading):
        """World centreline of the drivable space in the BEV grid.

        For each longitudinal band of the grid, the lateral centre of
        the drivable cells becomes one point of a lane reference - the
        "space centreline" a real vector-space planner tracks.  Falls
        back to a straight line ahead when few drivable cells exist
        (unknown road, sensor-limited frame).
        """
        drv = getattr(grid, "drivable", None)
        if drv is None or not getattr(drv, "any", lambda: False)():
            return None
        # The lane centre must be over the FREE corridor: drivable road
        # cells that are NOT inside an obstacle footprint.  A roadside or
        # corner wall erases the drivable cells it occupies via obstacle
        # fusion, so using plain "drivable" would pull the centreline into
        # the wall (town corner runs 2026-08-21).  When the corridor is so
        # dense that no free cell survives, fall back to the raw drivable
        # cells so the sensor lane still exists.
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
        # planner sees a path it can actually drive from here.  The raw
        # grid rows run far -> near, so without re-anchoring the first
        # point sits metres ahead of the car and every candidate gets
        # scored as off-lane (town runs 2026-08-21: all arcs infeasible at
        # (717.8,754.6) because the reference started 8 m away).
        d = arr - np.asarray(pos[:2], dtype=float)
        fwd_m = d[:, 0] * math.cos(float(heading)) + \
                d[:, 1] * math.sin(float(heading))
        ahead = arr[fwd_m > 0.5]
        if len(ahead) < 3:
            ahead = arr
        d0 = np.linalg.norm(ahead - np.asarray(pos[:2], dtype=float), axis=1)
        ahead = ahead[np.argsort(d0)]          # near -> far
        # Anchor the reference at the ego: when the nearest drivable row
        # is already a couple of metres in front, prepend the ego so the
        # planner's shift candidates start inside the forward-progress
        # gate (a reference whose first point is beyond ~3 m got every
        # lane-shift candidate rejected - town runs 2026-08-21).
        if len(ahead) and float(np.linalg.norm(ahead[0] - np.asarray(pos[:2],
                                                                     dtype=float))) > 2.0:
            ahead = np.vstack([np.asarray(pos[:2], dtype=float), ahead])
        return ahead if len(ahead) >= 3 else None

    def reset_temporal(self) -> None:
        """Clear the temporal filter - call after a teleport so stale
        occupancy from the previous location never leaks into the new
        scene as a phantom wall."""
        if getattr(self, "occ_filter", None) is not None:
            self.occ_filter.clear()
        self._tick_t0 = None
        self._lane_fusion_state.clear()

    def close(self) -> None:
        if self.ring is not None:
            try:
                self.ring.close()
            except Exception:
                pass
        try:
            rp = getattr(self, "range_prov", None)
            if rp is not None and hasattr(rp, "close"):
                rp.close()
        except Exception:
            pass


def semantic_to_meta(head_outputs: dict) -> dict:
    """Flatten a few useful head outputs into tick meta (telemetry)."""
    meta: dict = {}
    sem = head_outputs.get("semantic")
    if sem is not None:
        meta["lane_markings"] = len(sem.meta.get("markings", []))
    tr = head_outputs.get("traffic")
    if tr is not None:
        meta["signal_state"] = tr.meta.get("signal_state")
        meta["signal_conf"] = tr.meta.get("signal_conf")
    topo = head_outputs.get("topology")
    if topo is not None:
        meta["change_left"] = topo.meta.get("change_left")
        meta["change_right"] = topo.meta.get("change_right")
    return meta

