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

from dataclasses import replace
import math
import os
import time

import numpy as np

from beamng_autopilot.occupancy import (
    OccupancyGrid,
    fuse_obstacles_to_grid,
    project_road_mask_to_grid,
)
from beamng_autopilot.bev_fusion import (
    BEVFeatureMap,
    CameraFeature,
    project_mask_to_ego,
    stamp_signal_bearing,
    world_points_to_ego,
)
from beamng_autopilot.planning import (
    Constraints,
    REF_PERCEPTION,
    REF_ROUTE,
    Scene,
    lateral_reference,
    limit_reference_slew,
    sample_arc,
    sample_lane_shift,
    select_trajectory,
)
from beamng_autopilot.planning.intent import infer_route_intent
from beamng_autopilot.temporal import WorldObjectTracker
from beamng_autopilot.perception_snapshot import PerceptionSnapshot
from beamng_autopilot.planner import forward_clearance_m, path_forward_clearance_m
from beamng_autopilot.vehicle_body import CORRIDOR_HALF_WIDTH_M
from beamng_autopilot.lane import (
    LANE_WIDTH_DEFAULT_M,
    SensorLaneEnvelope,
    build_lidar_corridor,
    choose_sensor_lane,
    select_lane_reference,
)
from beamng_autopilot.runtime import (
    RangeSample,
    build_camera_ring_provider,
    build_range_provider,
)


# Dynamic obstacles picked up by the range channel (other vehicles) carry
# a world velocity.  With ``range_every_n > 1`` the last scan is reused
# for a few ticks; without compensation a moving car's box stays exactly
# where it was scanned and lags behind by up to ~3 ticks while the ego
# plans against a ghost (or misses a car that already moved into its
# path).  FSD-style dynamic occupancy predicts the box into the present
# with its measured velocity over the reuse gap, plus a safety margin for
# the unmeasured acceleration in between - never trusting a stale pose as
# ground truth.
RANGE_REUSE_MAX_DT_S = 2.0
RANGE_REUSE_MAX_PREDICT_M = 10.0
RANGE_REUSE_INFLATE_FRAC = 0.25
RANGE_REUSE_INFLATE_MAX_M = 1.5

# Bounded tick-to-tick slew of the accepted own-lane reference.
# Default OFF (opt-in) because a same-revision, same-game-session,
# interleaved 5-arm-per-condition A/B (2026-09-11, dashed recovery pinned
# on, scripts/m5_live_ab.py) shows no benefit and a real cost:
#   slew off: lane p50 15%, stall p50 138 (135-169), dist p50 48.3,
#             off-road [0,8,5,5,7], crossC [0,5,0,0,0], crossR [0,3,0,0,0]
#   slew on : lane p50 13%, stall p50 180 (151-188), dist p50 32.4,
#             off-road [0,7,11,0,2], crossC [0,0,6,0,0], crossR [0,0,0,0,0]
# Availability is unchanged, stall and distance are worse, and off-road /
# crossing frames occur in BOTH conditions - so this limiter is neither
# the cure nor the cause of the excursions.  The stage stays opt-in:
# ``BEAMNG_LANE_REF_SLEW=1`` enables it.
_LANE_REF_SLEW_ENABLED = os.environ.get("BEAMNG_LANE_REF_SLEW", "0") == "1"


def compensate_range_motion(sample: RangeSample | None,
                            dt_s: float) -> RangeSample | None:
    """Predict where dynamic boxes are now, after ``dt_s`` of reuse.

    Static boxes (walls, lidar clusters, scenario props) carry no
    ``velocity`` and are returned unchanged; the raw ray hits are
    world-frame static surfaces and are also untouched.
    """
    if sample is None or dt_s <= 0.0:
        return sample
    dt = min(float(dt_s), RANGE_REUSE_MAX_DT_S)
    out: list = []
    for ob in getattr(sample, "obstacles", []) or []:
        vel = getattr(ob, "velocity", None)
        if vel is None:
            out.append(ob)
            continue
        v = np.asarray(vel, dtype=float)
        if not np.isfinite(v).all():
            out.append(ob)
            continue
        shift = v * dt
        mag = float(np.hypot(*shift))
        if mag > RANGE_REUSE_MAX_PREDICT_M:
            shift = shift * (RANGE_REUSE_MAX_PREDICT_M / max(mag, 1e-9))
            mag = RANGE_REUSE_MAX_PREDICT_M
        grow = min(RANGE_REUSE_INFLATE_FRAC * mag,
                   RANGE_REUSE_INFLATE_MAX_M)
        out.append(replace(
            ob, x=float(ob.x) + float(shift[0]),
            y=float(ob.y) + float(shift[1]),
            half_w=float(ob.half_w) + grow,
            half_h=float(ob.half_h) + grow,
            # the oriented footprint is what mark_obstacle_region uses
            # when axis is set - inflate it too, or the reuse-gap safety
            # margin silently does not apply to moving vehicles
            half_len=float(getattr(ob, "half_len", 0.0) or 0.0) + grow,
            half_thick=float(getattr(ob, "half_thick", 0.0) or 0.0)
            + grow))
    return RangeSample(
        obstacles=out,
        ray_hits=list(getattr(sample, "ray_hits", []) or []))
from beamng_autopilot.vision.hydra import FrameContext, HydraNet
from beamng_autopilot.fsd_realism import (
    SRC_SENSOR,
    SRC_UNAVAILABLE,
)


# Minimum traffic-head confidence before the lamp is stamped into the
# BEV "sign" channel (below this the colour read is noise and vector
# space stays neutral).
SIGN_MIN_CONF = 0.5


# One-shot diagnostics for guarded blocks: a formerly silent
# ``except Exception`` now warns ONCE per failure kind, so a masked
# error is visible without spamming a per-tick loop.
_WARNED: set[str] = set()


def _warn_once(key: str, msg: str) -> None:
    if key in _WARNED:
        return
    _WARNED.add(key)
    print(f"[fsd-stack] {msg}", flush=True)


def _world_view(source_attr: str, name: str):
    """A read-through view of a world-model field.

    Before the canonical owner (snapshot / planner scene) is bound the
    value staged on the tick is returned; afterwards the owner's value
    is, so a consumer can never observe a second, divergent copy.
    """
    def _get(self):
        src = self.__dict__.get(source_attr)
        if src is not None:
            return getattr(src, name)
        return self.__dict__.get("_" + name)

    def _set(self, value):
        self.__dict__["_" + name] = value

    return property(_get, _set)


class FSDTick:
    """One planning-tick result from ``FSDStack``.

    The world model is published ONCE: the canonical
    ``PerceptionSnapshot`` (sensing) and the planner ``Scene`` (vector
    space).  The flat perception attributes kept for existing consumers
    are read-through views onto those two objects, so a consumer can
    never observe a second copy of the world model that has drifted from
    the one the planner and safety monitor used (docs/fsd_realism.md §2/§4).
    """

    # Sensing outputs: owned by ``self.snapshot`` once it is frozen.
    bev = _world_view("snapshot", "bev")
    drivable = _world_view("snapshot", "drivable")
    observed = _world_view("snapshot", "observed")
    feature_map = _world_view("snapshot", "feature_map")
    lane_envelope = _world_view("snapshot", "lane_envelope")
    frame = _world_view("snapshot", "frame")
    cam = _world_view("snapshot", "cam")
    head_outputs = _world_view("snapshot", "head_outputs")
    tracks = _world_view("snapshot", "tracks")
    errors = _world_view("snapshot", "errors")
    ray_hits = _world_view("snapshot", "ray_hits")
    # World-model fields: owned by the planner ``Scene`` of this tick.
    lane_left = _world_view("scene", "lane_left")
    lane_right = _world_view("scene", "lane_right")
    lane_width = _world_view("scene", "lane_width")
    intent = _world_view("scene", "intent")

    def __init__(self):
        self._bev: np.ndarray | None = None     # (N, N) occupancy raster
        self._drivable: np.ndarray | None = None
        self._observed: np.ndarray | None = None  # (N, N) sensor-seen mask
        self._feature_map = None                # fused multi-camera BEV
                                               # feature map (vector space)
        self._lane_envelope: SensorLaneEnvelope | None = None
        self._frame: np.ndarray | None = None   # front frame used this tick
        self._cam = None                       # its CameraModel
        self._head_outputs: dict = {}
        self._tracks: list = []                # active world-object tracks
        self._errors: dict = {}
        self._ray_hits: list = []
        self._lane_left: np.ndarray | None = None
        self._lane_right: np.ndarray | None = None
        self._lane_width: float = 0.0
        self._intent = None                    # RoutingIntent of nav route
        self.best_path: np.ndarray | None = None
        # Tick-level lane reference for the drive loop / safety monitor.
        # NOT the planner's lateral authority: the planner reads
        # ``scene.lane_ref``, which excludes the BEV whole-road centre.
        self.lane_ref: np.ndarray | None = None
        self.best_speed: float = 0.0            # planned speed at the start
        self.min_speed: float = 0.0             # lowest speed on the path
        self.n_candidates: int = 0
        self.meta: dict = {}
        # The canonical perception result and the planning Scene (world
        # model) of this tick.  Downstream consumers - the safety monitor
        # in particular - must evaluate the SAME objects, never a rebuilt
        # copy with a different lateral reference.
        self.snapshot: PerceptionSnapshot | None = None
        self.scene = None                      # planning.Scene | None
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
                 object_every_n: int = 1,
                 lane_mode: str = "map",
                 strict_sensor: bool = False):
        self.conn = conn
        self.grid_n = int(grid_n)
        self.grid_res = float(grid_res)
        self.heads = list(heads) if heads else []
        self.lane_envelope: SensorLaneEnvelope | None = None

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
        self._last_range_t = 0.0
        # Last accepted own-lane reference + hold start, for the bounded
        # tick-to-tick slew (see ``limit_reference_slew``).
        self._lane_ref_prev = None
        self._lane_ref_hold_t = 0.0
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
        self._head_timestamps: dict[str, float] = {}
        self._tick_num = 0
        self._head_phase: dict[str, int] = {}
        self._head_retry: set[str] = set()
        # Equal heavy cadences (semantic UNet + object YOLO both at n=2/4)
        # would collide on the SAME tick and double the tick cost - the
        # "stutter every few frames" pattern.  Offset the object head by
        # half a cycle so each tick carries at most one expensive head;
        # per-head freshness is unchanged (each still refreshes every n
        # ticks).
        if (self.semantic_every_n > 1 and self.object_every_n > 1
                and self.semantic_every_n == self.object_every_n):
            self._head_phase["object"] = self.object_every_n // 2

        self.hydra = HydraNet()
        for head in self.heads:
            self.hydra.add(head)

        self.constraints = Constraints(
            w_collision=5.0, w_curvature=0.5, w_lane_align=1.0)
        self.target_speed = 8.0  # plan cruise speed (m/s); drive can raise
        # Map-prior own-lane width (fallback when sensors see no lane)
        self.map_lane_width_m = LANE_WIDTH_DEFAULT_M
        # Raw-sensor forward corridor half width used by the independent
        # FSD safety layer (m5_fsd_drive).  Shared body corridor, not a
        # second vehicle size: see ``vehicle_body.CORRIDOR_HALF_WIDTH_M``.
        self.ego_half_width = float(CORRIDOR_HALF_WIDTH_M)

        # FSD-style temporal occupancy fusion: single-frame LiDAR glitches
        # must not create a phantom wall or erase a real one.
        self.temporal = bool(temporal)
        self.occ_filter = None
        self._tick_t0 = None
        # Per-frame lane-fusion state (choose_sensor_lane) across ticks
        # to prevent flicker between vision / lidar / fallback.
        self._lane_fusion_state: dict = {}
        # Lane-keep reference policy:
        #   map    - current rule-stable behaviour: map-prior own lane is
        #            the default, a paired sensor lane only takes over
        #            after strict heading/corner/side gates.
        #   auto   - perception-led when it agrees with the map prior:
        #            side gate relaxed, hard boundaries always from the
        #            map, sensor lane used when within CONSISTENCY_M of
        #            the map lane centre.
        #   sensor - FSD-style perception-led: sensor lane leads through
        #            corners too (corner gate off), map prior remains the
        #            hard guard-rail (centreline + right edge) and the
        #            safety monitor can always stop.
        self.lane_mode = str(lane_mode) if lane_mode in (
            "map", "auto", "sensor") else "map"
        self.strict_sensor = bool(strict_sensor)
        # auto-mode max lateral gap between the sensor lane centre and the
        # map-prior own-lane centre before the map takes over.
        self.lane_consistency_m = 1.5
        # sensor-mode consistency is looser (the perception lane may lead
        # through corners), but a sensor centre that sits > this far from
        # the map-prior lane is a different road / whole-road corridor,
        # not the ego lane.
        self.lane_consistency_sensor_m = 2.5
        if self.temporal:
            from beamng_autopilot.temporal import TemporalOccupancyFilter
            self.occ_filter = TemporalOccupancyFilter(
                n=int(grid_n), res=float(grid_res), tau_s=float(tau_s))
            self.tracker = WorldObjectTracker()
        else:
            self.tracker = None
        # Multi-camera BEV feature map (vector-space channel stack); built
        # lazily on the first tick so __new__-built test stubs stay valid.
        self.fmap: BEVFeatureMap | None = None
        self._trk_t0: float | None = None

    # ------------------------------------------------------------------
    def tick(self, st=None, route_ref: np.ndarray | None = None,
             include_bev: bool = True,
             map_lane_override=None,
             time_budget_s: float | None = None) -> FSDTick:
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
        _tick_cost0 = time.time()
        _budget = (float(time_budget_s)
                   if time_budget_s is not None and time_budget_s > 0.0
                   else None)
        _budget_skips: list[str] = []

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
            out.frame = frame
            out.cam = cam
            ctx = FrameContext(
                frame_rgb=frame, cam=cam, pos=pos, heading=heading,
                ground_z=float(pos[2]) if len(pos) > 2 else 0.0,
                role=role)
            heads: dict = {}
            head_ages: dict[str, float] = {}
            _tick_num = int(getattr(self, '_tick_num', 0))
            _phase = getattr(self, '_head_phase', None)
            if _phase is None:
                _phase = {}
                self._head_phase = _phase
            _retry = getattr(self, '_head_retry', None)
            if _retry is None:
                _retry = set()
                self._head_retry = _retry
            _last = getattr(self, '_last_heads', None)
            if _last is None:
                _last = {}
                self._last_heads = _last
            _head_stamps = getattr(self, "_head_timestamps", None)
            if _head_stamps is None:
                _head_stamps = {}
                self._head_timestamps = _head_stamps
            for _name, _head in self.hydra._heads.items():
                if _name == "topology":
                    # The topology head needs the PAIRED sensor lane,
                    # which only exists after perception fusion - it runs
                    # once at end-of-tick with the real LaneFrame.
                    # Running it here always feeds sensor_lane=None
                    # (has_lane=False) and wastes a lane-graph build.
                    continue
                if _name == "semantic":
                    _n = int(getattr(self, 'semantic_every_n', 1))
                elif _name == "object":
                    _n = int(getattr(self, 'object_every_n', 1))
                else:
                    _n = 1
                _n = max(1, _n)
                _due = ((_tick_num + int(_phase.get(_name, 0))) % _n == 0
                        or _name in _retry)
                if not _due:
                    if _last.get(_name) is not None:
                        heads[_name] = _last[_name]
                        _stamp = _head_stamps.get(
                            _name, _tick_cost0)
                        head_ages[_name] = max(0.0, time.time() - _stamp)
                    continue
                # Tick time-budget governor (smoothness): when a heavy
                # head is due but this tick has already consumed its time
                # budget, defer it and serve the last cached output.  The
                # head stays due (`_head_retry`) and runs on the first
                # later tick the budget allows, so a semantic + YOLO +
                # fresh-LiDAR collision can never freeze the control loop
                # ("stutter every few frames, car barely moves").
                if (_n > 1 and _budget is not None
                        and (time.time() - _tick_cost0) > _budget):
                    if _last.get(_name) is not None:
                        heads[_name] = _last[_name]
                        _stamp = _head_stamps.get(
                            _name, _tick_cost0)
                        head_ages[_name] = max(0.0, time.time() - _stamp)
                    _retry.add(_name)
                    _budget_skips.append(_name)
                    continue
                try:
                    out_head = _head.run(ctx)
                    heads[_name] = out_head
                    _last[_name] = out_head
                    _head_stamps[_name] = time.time()
                    head_ages[_name] = 0.0
                    _retry.discard(_name)
                except Exception as _exc:
                    self.hydra.errors[_name] = str(_exc)
            self._tick_num = _tick_num + 1
            # Every registered head gets an explicit age.  A missing output
            # is not silently "fresh": without a previous result it is
            # represented as +inf and SafetyMonitor can fail closed.
            for _name in self.hydra._heads:
                if _name not in head_ages:
                    _stamp = _head_stamps.get(_name)
                    head_ages[_name] = (max(0.0, time.time() - _stamp)
                                        if _stamp is not None else float("inf"))
            out.head_outputs = heads
            out.meta["head_age_s"] = {
                k: round(float(v), 3) if np.isfinite(v) else None
                for k, v in head_ages.items()}
            out.meta["object_head"] = int("object" in self.hydra._heads)
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
                    if (_budget is not None
                            and (time.time() - _tick_cost0) > _budget
                            and getattr(self, '_last_range', None)
                            is not None):
                        _dt = (time.time()
                               - float(getattr(self, '_last_range_t', 0.0)))
                        rng = compensate_range_motion(
                            self._last_range, _dt)
                        _budget_skips.append("range")
                    else:
                        rng = self.range_prov.scan(pos)
                        self._last_range = rng
                        self._last_range_t = time.time()
                        self._range_skip = max(
                            0, int(getattr(self, 'range_every_n', 1)) - 1)
                else:
                    self._range_skip -= 1
                    _dt = (time.time()
                           - float(getattr(self, '_last_range_t', 0.0)))
                    rng = compensate_range_motion(
                        getattr(self, '_last_range', None), _dt)
                if rng is None:
                    raise RuntimeError("no range sample")
                out.ray_hits = list(getattr(rng, "ray_hits", []) or [])
                out.forward_clearance = forward_clearance_m(
                    out.ray_hits, pos,
                    np.array([math.cos(heading), math.sin(heading)]),
                    half_width=float(getattr(self, "ego_half_width",
                                             CORRIDOR_HALF_WIDTH_M)))
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
                # Geometric obstacle identity (tree / guardrail / wall):
                # per-class counts + nearest distance, refreshed on every
                # fresh scan and cached between scans so every tick's
                # telemetry carries the last-known read.
                from beamng_autopilot.perception import obstacle_class_counts
                self._cls_counts, self._cls_nearest = obstacle_class_counts(
                    rng.obstacles, pos)
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
            # Multi-camera BEV feature map (the FSD vector-space channel
            # stack).  The semantic road / lane masks are back-projected
            # into ego space and accumulated per channel together with the
            # LiDAR + YOLO obstacle points, so the stack keeps a persistent
            # fused feature map (later the learning input) instead of only
            # the transient occupancy grid.
            try:
                # The feature map is a per-tick ego snapshot, not a
                # persistent world map: semantic points and detections are
                # already in the CURRENT ego frame, so reusing the old
                # raster would leave ghosts behind as the car moves.
                fmap = BEVFeatureMap(
                    n=int(getattr(self, "grid_n", 60)),
                    res=float(getattr(self, "grid_res", 0.5)))
                self.fmap = fmap
                _gnd = float(pos[2]) if len(pos) > 2 else 0.0
                _role0 = "front_main" if "front_main" in snap \
                    else (next(iter(snap)) if snap else None)
                if semantic is not None and _role0 is not None:
                    if "road" in semantic.masks:
                        _pts = project_mask_to_ego(
                            semantic.masks["road"], snap[_role0][1], pos,
                            heading, ground_z=_gnd, channel="drivable",
                            step=8, max_ahead_m=40.0)
                        for _p in _pts:
                            fmap.accumulate(CameraFeature(
                                _role0, "drivable", _p, confidence=0.7))
                    if "line" in semantic.masks:
                        _pts = project_mask_to_ego(
                            semantic.masks["line"], snap[_role0][1], pos,
                            heading, ground_z=_gnd, channel="lane",
                            step=8, max_ahead_m=40.0)
                        for _p in _pts:
                            fmap.accumulate(CameraFeature(
                                _role0, "lane", _p, confidence=0.75))
                # Traffic-signal head -> "sign" channel: a confidently
                # detected lamp is stamped along its pixel bearing so the
                # vector space carries "traffic control ahead" instead of
                # a dead (never-written) channel.
                _sig = out.head_outputs.get("traffic")
                if _sig is not None and _role0 is not None:
                    _smeta = getattr(_sig, "meta", {}) or {}
                    _sst = str(_smeta.get("signal_state") or "none")
                    _scf = float(_smeta.get("signal_conf", 0.0) or 0.0)
                    _spx = _smeta.get("signal_px")
                    if (_sst in ("red", "yellow", "green")
                            and _scf >= SIGN_MIN_CONF
                            and _spx is not None):
                        stamp_signal_bearing(
                            fmap, snap[_role0][1], _spx,
                            confidence=0.5 + 0.5 * _scf)
                _obs_pts: list[tuple[float, float, float]] = []
                _rng = locals().get("rng", None)
                _obs_pts.extend(
                    (float(_o.x), float(_o.y), 0.0)
                    for _o in (getattr(_rng, "obstacles", None) or []))
                if obj is not None:
                    _obs_pts.extend(
                        (float(_o.x), float(_o.y), 0.0)
                        for _o in (getattr(obj, "obstacles", None) or []))
                if _obs_pts:
                    # obstacle detections are WORLD coordinates; fmap points
                    # are EGO coordinates.  Transform once before stamping
                    # so nonzero origin/heading cannot move a wall to a
                    # fictitious cell (canonical BEV contract).
                    _ego_obs = world_points_to_ego(
                        np.asarray(_obs_pts, dtype=float).reshape(-1, 3),
                        pos, heading)
                    fmap.accumulate(CameraFeature(
                        "sensor_fusion", "obstacle", _ego_obs,
                        confidence=0.85))
                out.feature_map = fmap
            except Exception as exc:
                out.errors["bev_fusion"] = str(exc)
        _cc = getattr(self, "_cls_counts", None)
        if _cc:
            out.meta["cls_counts"] = dict(_cc)
            out.meta["cls_nearest"] = dict(getattr(self, "_cls_nearest", {}))
        _range_stamp = float(getattr(self, "_last_range_t", 0.0) or 0.0)
        if _range_stamp > 0.0:
            out.meta["range_age_s"] = round(
                max(0.0, time.time() - _range_stamp), 3)
        _times['range'] = round((time.time() - _tw) * 1000.0, 1)
        _tw = time.time()
        if _budget_skips:
            out.meta["tick_budget_skips"] = list(_budget_skips)
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

        # World-object tracking: match LiDAR + YOLO detections frame to
        # frame into persistent velocity tracks (the FSD object-tracking
        # layer).  The tracks fill the gap between throttled raw sweeps -
        # their extrapolated positions keep occupying cells so a vehicle
        # does not vanish from vector space between updates.
        tracker = getattr(self, "tracker", None)
        if tracker is not None:
            try:
                _now = time.time()
                _t0 = getattr(self, "_trk_t0", None)
                _dt = (_now - _t0) if _t0 is not None else None
                self._trk_t0 = _now
                _dets: list[tuple[float, float, str]] = []
                _rng = locals().get("rng", None)
                for _o in (getattr(_rng, "obstacles", None) or []):
                    _dets.append((float(_o.x), float(_o.y),
                                  str(getattr(_o, "category", "object")
                                      or "object")))
                _obj = out.head_outputs.get("object")
                for _o in (getattr(_obj, "obstacles", None) or []):
                    _dets.append((float(_o.x), float(_o.y),
                                  str(getattr(_o, "category", "object")
                                      or "object")))
                _active = tracker.update(_dets, dt=_dt)
                out.tracks = list(_active)
                out.meta["n_tracks"] = len(_active)
                if _active:
                    from beamng_autopilot.occupancy import (
                        OccupancyGrid as _OG2)
                    if isinstance(grid, _OG2):
                        from beamng_autopilot.perception import Obstacle
                        _boxes = [
                            Obstacle(float(t.x), float(t.y), 1.0, 1.0,
                                     category=t.category)
                            for t in _active]
                        fuse_obstacles_to_grid(grid, _boxes)
            except Exception as exc:
                out.errors["tracker"] = str(exc)

        # Tracker fusion above mutates the canonical planner grid.  Publish
        # the SAME post-tracker raster to safety/E2E/recorder; otherwise
        # those consumers see the pre-tracker occupancy and a tracked
        # moving object vanishes at the safety boundary.
        out.bev = grid.as_raster()
        out.drivable = grid.drivable
        out.observed = getattr(grid, "observed", None)

        # --- 3) lane perception -----------------------------------------
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
        self.lane_envelope = SensorLaneEnvelope.from_lane_frame(
            lane_frame, captured_at=time.time()) if lane_frame is not None else None
        out.lane_envelope = self.lane_envelope
        if self.lane_envelope is not None:
            out.meta["lane_envelope"] = self.lane_envelope.as_meta()
        # One owner decides which lane geometry may steer the car this
        # tick (sensor lane -> trusted single painted boundary -> map
        # prior in legacy mode only -> BEV free-space centre when there
        # is no nav route).  Strict FSD never builds map lane geometry
        # and returns no centre when perception cannot supply one, so
        # the planner can only fail closed.
        lane_mode = getattr(self, "lane_mode", "map")
        lane_ref_out = select_lane_reference(
            lane_frame=lane_frame,
            pos=pos,
            heading=heading,
            route_ref=route_ref,
            has_nav_route=has_nav_route,
            map_lane_override=map_lane_override,
            grid=grid,
            lane_mode=lane_mode,
            # Defensive read: the stub stacks built with ``__new__`` in tests
            # and probes never run ``__init__``.  The inline code this call
            # replaced short-circuited on ``lane_mode == "sensor"`` and so
            # never touched the attribute; a call argument is always
            # evaluated, so the default has to live here.
            strict_sensor=getattr(self, "strict_sensor", False),
            lane_consistency_m=getattr(self, "lane_consistency_m", 1.5),
            lane_consistency_sensor_m=getattr(
                self, "lane_consistency_sensor_m", 2.5),
            map_lane_width_m=getattr(
                self, "map_lane_width_m", LANE_WIDTH_DEFAULT_M),
            warn=_warn_once,
        )
        lane_ref = lane_ref_out.center
        # Single owner, no sideways teleport: the accepted own-lane
        # reference is the ONE reference the planner and the safety Scene
        # both consume, so it is limited HERE, before either reads it.
        # Measured 2026-09-11 (dashed recovery on): the accepted sensor
        # reference jumped up to 2.20 m laterally between consecutive
        # ticks (5.3% > 1.0 m) while the envelope moved at most 0.41 m.
        # ``BEAMNG_LANE_REF_SLEW=0`` disables it: the live A/B is not yet
        # decisive at this sample size, so the switch is the rollback and
        # comparison lever.
        if lane_ref is not None and _LANE_REF_SLEW_ENABLED:
            lane_ref, self._lane_ref_hold_t = limit_reference_slew(
                getattr(self, "_lane_ref_prev", None),
                getattr(self, "_lane_ref_hold_t", 0.0),
                lane_ref, time.time(), pos, heading)
        self._lane_ref_prev = (None if lane_ref is None
                               else np.asarray(lane_ref, dtype=float)[:, :2])
        lane_left = lane_ref_out.left
        lane_right = lane_ref_out.right
        lane_width = lane_ref_out.width
        map_lane = lane_ref_out.map_lane
        lane_rejected = lane_ref_out.rejected
        lane_src_sel = lane_ref_out.src
        strict_lane = lane_ref_out.strict
        out.meta.update(lane_ref_out.meta)
        out.lane_ref = lane_ref
        if lane_ref_out.boundaries:
            if lane_left is not None:
                out.lane_left = np.asarray(lane_left, dtype=float)[:, :2]
            if lane_right is not None:
                out.lane_right = np.asarray(lane_right, dtype=float)[:, :2]
            out.lane_width = lane_width
        # Perception head that needs the fused lane: the topology head
        # consumes a real LaneFrame (not just a frame context), so it runs
        # once here - after fusion, before the snapshot is frozen.  It used
        # to run after planning, which made planning a sibling of a head
        # instead of a consumer of the complete perception result.
        topology = self.hydra._heads.get("topology")
        if topology is not None:
            _topo_lane = self._sensor_lane_from_semantic(
                out.head_outputs, pos, heading)
            _topo_ctx = FrameContext(
                frame_rgb=np.zeros((1, 1, 3), dtype=np.uint8),
                cam=None, pos=pos, heading=heading,
                ground_z=float(pos[2]) if len(pos) > 2 else 0.0,
                role="front_main")
            try:
                _topo_out = topology.run(_topo_ctx, sensor_lane=_topo_lane)
                getattr(self, "_head_timestamps", {})["topology"] = time.time()
                out.meta.setdefault("head_age_s", {})["topology"] = 0.0
                out.head_outputs["topology"] = _topo_out
            except Exception as exc:
                _warn_once("topology", f"topology head failed: {exc}")
        out.meta.update(semantic_to_meta(out.head_outputs))
        # --- 4) canonical perception snapshot ---------------------------
        # Sensing is complete here: ring -> heads -> BEV -> tracking ->
        # lane envelope.  Freeze it into the ONE snapshot every downstream
        # stage reads this tick - the planner builds its Scene from it, the
        # safety monitor verifies against it, telemetry and shadow
        # recording publish it.  Building it HERE rather than after
        # planning is what makes planning provably a consumer of
        # perception instead of a sibling that re-reads raw tick fields
        # (docs/fsd_realism.md §2).
        out.meta.setdefault("bev_age_s",
                            0.0 if out.bev is not None else None)
        out.meta.setdefault("range_age_s", None)
        out.snapshot = PerceptionSnapshot(
            captured_at=float(_tick_cost0),
            tick_id=int(getattr(self, "_tick_num", 0)),
            pos=pos.copy(), heading=heading,
            frame=out.frame, cam=out.cam,
            head_outputs=dict(out.head_outputs),
            head_age_s=dict(out.meta.get("head_age_s", {})),
            errors=dict(out.errors), ray_hits=list(out.ray_hits),
            tracks=list(out.tracks), bev=out.bev,
            drivable=out.drivable, observed=out.observed,
            feature_map=out.feature_map,
            lane_envelope=getattr(out, "lane_envelope", None),
            range_age_s=out.meta.get("range_age_s"),
            bev_age_s=out.meta.get("bev_age_s"))
        out.meta["snapshot"] = out.snapshot.meta()
        # --- 5) layered planner -----------------------------------------
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
                _base_route, lane_ref, pos, heading, grid,
                strict_perception=strict_lane)
        elif strict_lane:
            plan_route = (lane_ref if lane_ref is not None
                          and len(lane_ref) >= 4 else None)
        else:
            plan_route = (lane_ref if lane_ref is not None
                          and len(lane_ref) >= 4 else _base_route)
        if plan_route is None or len(plan_route) < 2:
            if not strict_lane:
                plan_route = _base_route
        if lane_ref is None and not strict_lane:
            lane_ref = plan_route
        # out.lane_ref drives the *lateral* lane-keep reference (sensor
        # lane centre when available); plan_route stays the navigational
        # intent in the planner's Scene.
        # ``lane_src`` / ``lane_src_sel`` / ``lane_reject_reason`` all come
        # from the single policy owner (``lane.reference``) via
        # ``out.meta.update(lane_ref_out.meta)`` above.  Do not re-derive
        # them here: a second derivation is exactly how the two labels
        # drifted apart in the 2026-09-07 town logs.
        out.meta["lane_paired"] = int(
            bool(lane_frame is not None and getattr(lane_frame, "paired", False)))
        if lane_rejected:
            out.meta["lane_paired"] = 0
            # historical key, now carrying the TRUE gate reason instead of
            # a hardcoded "heading"
            if lane_ref_out.reject_reason is not None:
                out.meta["lane_reject"] = lane_ref_out.reject_reason
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
        scene_lane_ref = lane_ref_out.scene_ref
        # Routing intent: classify what the nav route does ahead (turn /
        # straight / u-turn) - the FSD Routing layer output.  It does not
        # steer by itself; it only informs the longitudinal plan (slow
        # down for an up-coming turn) and the HUD.
        intent = None
        try:
            if has_nav_route and len(route_ref) >= 8 and \
                    len(np.asarray(route_ref, dtype=float)[:, :2]) >= 8:
                intent = infer_route_intent(route_ref, pos, heading)
        except Exception as exc:
            _warn_once("intent", f"route intent failed: {exc}")
            intent = None
        out.intent = intent
        if intent is not None:
            out.meta["intent"] = intent.label
            out.meta["intent_turn_deg"] = round(float(intent.turn_deg), 1)
            out.meta["intent_speed"] = round(float(intent.suggested_speed), 2)
        # A turn ahead lowers the plan cruise speed (never raises it);
        # the curvature-based profile in speed_profile.py still does the
        # fine-grained braking into the bend.
        _target = float(getattr(self, "target_speed", 8.0))
        if intent is not None and getattr(intent, "is_turn", False):
            _target = min(_target, float(intent.suggested_speed))
        scene = Scene(pos=pos, heading=heading, grid=grid,
                      route=plan_route, lane_ref=scene_lane_ref,
                      lane_left=lane_left, lane_right=lane_right,
                      lane_width=lane_width,
                      lane_envelope=self.lane_envelope,
                      perception_snapshot=out.snapshot,
                      target_speed=_target, intent=intent,
                      strict_perception=strict_lane)
        # Publish the world model: the safety layer evaluates the same
        # Scene the planner planned against (one world model per tick).
        out.scene = scene
        # Candidate families use the same lateral policy as the planner
        # and safety layer.  Strict FSD mode therefore never seeds a
        # map-centre path from the nav route when perception is missing.
        candidate_ref, candidate_src = lateral_reference(scene)
        out.meta["lateral_candidate_src"] = candidate_src
        out.meta["lateral_candidate_pts"] = (
            int(len(candidate_ref)) if candidate_ref is not None else 0)
        # Town corners need a tighter arc fan than a highway fan: a
        # 5-8 m radius bend is 0.12-0.2 rad/m, and the old 0.10 rad/m
        # cap (10 m radius) could not turn away from a corner wall
        # (town runs 2026-08-21) - it kept pressing the throttle into
        # the wall.  Sample to the physical steer limit and add wider
        # lateral shifts so the planner can actually dodge a near wall.
        fans = sample_arc(pos, heading, speed=max(2.0, float(st.speed)),
                          max_steer=0.5, n_curv=13, max_curv=0.25)
        if candidate_ref is not None and len(candidate_ref) >= 4:
            shifts = sample_lane_shift(candidate_ref,
                                       offsets=(-3.0, -1.5, 1.5, 3.0))
            for c in shifts.candidates:
                kind = c.meta.get("kind", "shift")
                # sample_lane_shift includes its reference as a candidate.
                # When that reference is the perception lane, the explicit
                # lane_center candidate below represents it; keeping the
                # duplicate would win ties under the generic "reference"
                # label and hide the perception-led choice from telemetry.
                if kind == "reference" and candidate_src in REF_PERCEPTION:
                    continue
                fans.add(c.path, kind,
                         offset=c.meta.get("offset", 0.0))
        elif candidate_src == REF_ROUTE:
            # The policy always returns the route with REF_ROUTE; this
            # guard keeps a future policy change from silently dropping
            # the legacy candidate family.
            _warn_once("candidate_route",
                       "route lateral candidate unavailable in legacy mode")
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
        lane_center_ref = (scene_lane_ref
                           if scene_lane_ref is not None
                           else candidate_ref
                           if candidate_src in REF_PERCEPTION else None)
        if lane_center_ref is not None and len(lane_center_ref) >= 4:
            fans.add(np.asarray(lane_center_ref, dtype=float)[:, :2],
                     "lane_center", offset=0.0)
        out.n_candidates = len(fans.candidates)
        best, meta = select_trajectory(scene, fans, self.constraints)
        out.best_path = best
        # One fail-closed decision, published in telemetry: strict mode
        # with no perception lane declines every candidate in the
        # constraint layer (raw arcs carry no lane geometry either), so
        # the runtime has nothing to steer and the stack degrades to the
        # legal set - stop, hold heading, or a safe road point
        # (docs/fsd_realism.md §4).
        if best is None and strict_lane and lane_src_sel != SRC_SENSOR:
            out.meta["plan_blocked"] = "no_perception_lane"
        # Path-aware forward clearance: safety layer evaluates the chosen
        # trajectory corridor instead of the raw heading corridor, so a
        # turn away from a wall does not force a stop (town runs 2026-08-21).
        if best is not None and len(best) >= 2 and out.ray_hits:
            out.path_forward_clearance = path_forward_clearance_m(
                best, out.ray_hits,
                half_width=float(getattr(self, "ego_half_width",
                                         CORRIDOR_HALF_WIDTH_M)))
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
            # NB: the speed_profile module lives in planning/ - the old
            # ``from .speed_profile import ...`` here never resolved and
            # this whole block silently never ran (surfaced by the
            # warn-once diagnostics, real run 2026-09-05).
            from .planning import speed_profile_for_path as _spf
            _rr = np.asarray(route_ref, dtype=float)[:, :2]
            # Skip the ego-anchor dogleg: local_route prepends the ego
            # pose, and the lateral jump onto the road reads as a
            # sub-metre hairpin for the curvature profile, pinning
            # best_speed to ~1.5 m/s on an r=15 bend (offline r15 closed
            # loop 2026-09-05).  Profile the road geometry from ~2.5 m
            # ahead - the same footprint skip the collision layers use.
            _keep = np.hypot(_rr[:, 0] - float(pos[0]),
                             _rr[:, 1] - float(pos[1])) >= 2.5
            if int(np.count_nonzero(_keep)) >= 4:
                _rr = _rr[_keep]
            _sr = _spf(_rr, scene, target_speed=_target)
            if len(_sr):
                out.best_speed = float(_sr[0])
                out.min_speed = float(np.asarray(_sr).min())
                out.meta["best_speed"] = out.best_speed
                out.meta["min_speed"] = out.min_speed
                out.meta["plan_src"] = "route"
        except Exception as exc:
            out.meta["plan_src"] = "candidate"
            _warn_once("plan_speed", f"route speed profile failed: {exc}")


        _times['plan'] = round((time.time() - _tw) * 1000.0, 1)
        _times['total'] = round(sum(_times.values()), 1)
        out.meta['tick_ms'] = _times
        # Publish the tick's provenance on the Scene the planner already
        # planned against, so the safety monitor reads the same freshness
        # contract (the snapshot itself was frozen before planning).
        scene.meta = out.meta
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
        except Exception as exc:
            _warn_once("sensor_lane_vision", f"vision lane failed: {exc}")
            vision = None
        lidar = None
        if out.ray_hits:
            try:
                lidar = build_lidar_corridor(out.ray_hits, pos, heading)
            except Exception as exc:
                _warn_once("sensor_lane_lidar",
                           f"lidar corridor failed: {exc}")
                lidar = None
        try:
            frame = choose_sensor_lane(
                vision, lidar, pos, heading,
                state=getattr(self, "_lane_fusion_state", None))
        except Exception as exc:
            _warn_once("sensor_lane_fusion", f"lane fusion failed: {exc}")
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
        except Exception as exc:
            _warn_once("pair_lane_markings",
                       f"vision lane pairing failed: {exc}")
            return None

    def reset_temporal(self) -> None:
        """Clear the temporal filter - call after a teleport so stale
        occupancy from the previous location never leaks into the new
        scene as a phantom wall."""
        if getattr(self, "occ_filter", None) is not None:
            self.occ_filter.clear()
        self._tick_t0 = None
        self._lane_fusion_state.clear()
        self.lane_envelope = None
        # world-object tracks and the fused feature map are location-bound
        # too: an object tracked at the old teleport can ghost into the
        # new scene as a false obstacle.
        _trk = getattr(self, "tracker", None)
        if _trk is not None:
            _trk.tracks = []
        self._trk_t0 = None
        _fm = getattr(self, "fmap", None)
        if _fm is not None:
            _fm.clear()
        # The LiDAR reuse cache is location-bound too: compensate_range_
        # motion only shifts boxes that carry a velocity, so a cached
        # pre-teleport scan would flood the NEW grid with static boxes
        # at their OLD world coordinates for up to range_every_n ticks.
        self._last_range = None
        self._last_range_t = 0.0
        self._range_skip = 0
        # The accepted lane reference is location-bound as well: after a
        # teleport the previous polyline is somewhere else entirely and
        # must not be held against the new tick's selection.
        self._lane_ref_prev = None
        self._lane_ref_hold_t = 0.0

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
