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

from beamng_autopilot import geometry
from beamng_autopilot.occupancy import (
    OccupancyGrid,
    fuse_obstacles_to_grid,
    nearfield_coverage,
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
from beamng_autopilot.workers import LatestJobRunner
from beamng_autopilot.prediction import predict_tracks, prediction_digest
from beamng_autopilot.perception_snapshot import PerceptionSnapshot
from beamng_autopilot.planner import forward_clearance_m, path_forward_clearance_m
from beamng_autopilot.vehicle_body import CORRIDOR_HALF_WIDTH_M
from beamng_autopilot.lane import (
    LANE_WIDTH_DEFAULT_M,
    ReferenceStabilityTracker,
    SensorLaneEnvelope,
    build_lidar_corridor,
    choose_sensor_lane,
    select_lane_reference,
)
from beamng_autopilot.lane.pavement import paved_edge_lane_center
from beamng_autopilot.config import EGO_ORIGIN_GROUND_GAP_M
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


# --- near-field drivable evidence (review handoff P1-1) -----------------
# The front camera's nearest visible ground is ~3.5 m, so the 0-4 m band
# carries no drivable evidence and strict mode reports "no drivable path"
# - the largest single stall cause measured.  This switch is the INPUT
# half of the experiment: it changes the camera input only (no planner
# threshold, no safety gate, no model change).
#
#   off      - current behaviour (front_main only)
#   fisheye  - run the same Segmenter on the front FISHEYE frame and
#              project its road mask into the same grid
#   fuse     - front_main + fisheye (union)
#   band     - **UPPER BOUND ONLY**: inject a fixed-width drivable strip in
#              the 0-NEARFIELD_BAND_AHEAD_M window.  It answers "with
#              near-field evidence, does the planner stop anyway?" and must
#              never be shipped as a solution - it is not perception.
NEARFIELD_CAM = os.environ.get("BEAMNG_NEARFIELD_CAM", "off").strip().lower()
NEARFIELD_EVERY_N = max(1, int(os.environ.get("BEAMNG_NEARFIELD_EVERY_N", "1")
                               or 1))
NEARFIELD_MAX_AHEAD_M = 10.0     # fisheye projection horizon
NEARFIELD_BAND_AHEAD_M = 4.0     # injected upper-bound band length
NEARFIELD_BAND_HALF_M = 2.2      # injected band half width


def nearfield_step(stack, out, snap, semantic, grid, pos, heading,
                   tick_num) -> dict | None:
    """Add near-field drivable evidence for this tick (P1-1).

    Returns the telemetry for the step, or None when the mode is ``off``
    or the pass did not run this tick.  Never raises into the tick: a
    missing fisheye camera degrades to "no near-field evidence", which is
    exactly the state the experiment is measuring.
    """
    _asked = NEARFIELD_CAM
    if _asked in ("", "off", "0", "none"):
        return None
    # ``fisheye`` and ``fuse`` are THE SAME code path: the main view's road
    # is already in the grid by the time this runs, so adding the fisheye is
    # the fusion - there is no second algorithm.  The two names are kept
    # (old experiment commands still run) but the telemetry reports the
    # canonical mode and the alias separately, so nobody reads "fuse" as a
    # distinct method (plan T05: name the real difference or merge them).
    _alias = None
    _mode = _asked
    if _asked == "fuse":
        _mode = "fisheye"
        _alias = "fuse"
    meta: dict = {"nearfield_mode": _mode}
    if _alias:
        meta["nearfield_alias"] = _alias
        meta["nearfield_alias_note"] = (
            "fuse == fisheye: one code path, the main view is already fused "
            "into the same grid")
    try:
        if _mode == "band":
            # Upper bound: paint the strip the car would need to see.
            res = float(grid.res)
            n = int(grid.n_rows)
            rows = max(1, int(round(NEARFIELD_BAND_AHEAD_M / res)))
            cols = max(1, int(round(NEARFIELD_BAND_HALF_M / res)))
            c0 = max(0, n // 2 - cols)
            c1 = min(n, n // 2 + cols)
            # Ahead = smaller row index (see nearfield_coverage).
            r0 = max(0, n // 2 - rows)
            r1 = n // 2 + 1
            grid.drivable[r0:r1, c0:c1] = 1.0
            if getattr(grid, "observed", None) is not None:
                grid.observed[r0:r1, c0:c1] = 1
            meta["nearfield_injected_band_m"] = NEARFIELD_BAND_AHEAD_M
            return meta
        if tick_num % NEARFIELD_EVERY_N != 0:
            meta["nearfield_skipped"] = "cadence"
            return meta
        if "front_fisheye" not in snap:
            meta["nearfield_skipped"] = "no fisheye frame this tick"
            return meta
        # The SEGMENTER lives on the head INSTANCE, not on its TaskOutput
        # (``head_outputs["semantic"]`` is the output object) - looking for
        # it there raised every tick and silently turned the whole pass
        # into a recorded error.
        head = None
        heads = getattr(getattr(stack, "hydra", None), "_heads", None)
        if hasattr(heads, "get"):
            head = heads.get("semantic")
        if head is None or not hasattr(head, "_get_segmenter"):
            meta["nearfield_skipped"] = "no semantic head instance"
            return meta
        import numpy as _np
        t0 = time.time()
        seg = head._get_segmenter()
        fish_rgb, fish_cam = snap["front_fisheye"]
        road_fish, _line_fish = seg.predict(fish_rgb)
        ground_z = (float(pos[2]) - float(EGO_ORIGIN_GROUND_GAP_M)
                    if len(pos) > 2 else None)
        project_road_mask_to_grid(
            grid, _np.asarray(road_fish, dtype=bool), fish_cam, pos, heading,
            max_ahead_m=NEARFIELD_MAX_AHEAD_M, step=3, ground_z=ground_z)
        meta["nearfield_ms"] = round((time.time() - t0) * 1000.0, 1)
        meta["nearfield_road_px"] = int(_np.count_nonzero(road_fish))
        return meta
    except Exception as exc:                      # never break the tick
        meta["nearfield_error"] = str(exc)
        return meta


# Heavy-head worker timeout (plan A4): a head that runs longer than this
# is counted in telemetry, never waited for - the served output simply
# ages and the freshness contract (safety monitor) owns the decision.
ASYNC_HEAD_TIMEOUT_S = 1.5

# Default OFF, like every behaviour-changing perception switch in this
# repo: moving the object head off the tick changes an observable
# contract - its YOLO obstacles are fused into the BEV of the SAME tick
# (test_fsd_stack.py pins it), and a one-tick delay before an NPC enters
# vector space is a safety-relevant change that has to be decided on live
# evidence, not offline.  With ``BEAMNG_ASYNC_HEADS=1`` the mechanism is
# active and the tick-level contract (never wait for a head, serve the
# newest available output with an honest age) is covered by
# tests/test_fsd_stack_async.py.
ASYNC_HEADS_ENABLED = os.environ.get("BEAMNG_ASYNC_HEADS", "0") == "1"


# --- scheduler keep-alive (2026-09-20) ---------------------------------
# The tick-budget governor defers a due modality whenever the tick has
# already spent its budget.  The instinct is right (perception must not
# freeze the control loop) but the rule had no floor, and the ring alone
# costs 376-672 ms against a 450 ms budget - so ``range`` was deferred on
# 151/151 and 120/120 frames of the 2026-09-20 town runs and ``object`` on
# 147/151, ``range_age`` reached p50 61.9 s, the freshness contract read
# "stale sensor" on every frame, and the car crawled (speed p50 0.00 m/s).
#
# This is not a new failure mode.  ``range_every_n=3`` had already produced
# the same shape (fsd_drive's strict-sensor comment: 4-6 s range age,
# "every reuse cycle fail-closed into stale sensor stops"), and setting
# ``range_every_n=1`` only handed the starvation to the budget gate.  Both
# reuse paths were missing one rule:
#
#   a reused modality may not be pushed past its keep-alive bound.
#
# Past the bound the modality is STARVED, not deferred, and it refreshes
# regardless of the budget or the every-n throttle.  This is deliberately
# NOT "these modalities ignore the budget": the cost stays bounded at one
# extra refresh per bound, so the governor still owns smoothness.
#
#   range : 1.0 s - the safety-critical channel and the one the stale
#                   verdict actually reads.  STALE_RANGE_S = 2.0 and
#                   RANGE_REUSE_MAX_DT_S = 2.0 say the compensation is
#                   only meaningful that far, so the reuse must end well
#                   inside it - one slow tick of margin.
#   object: 2.0 s - YOLO is not a stale trigger (the freshness contract
#                   requires the semantic head only), so this is the
#                   lower tier: bounded, not tight.
#
# Default OFF, like every behaviour-changing perception switch in this
# repo: the floor trades tick time for sensor freshness and that trade
# needs same-condition live evidence.  ``BEAMNG_SCHED_KEEPALIVE=1``.
RANGE_KEEPALIVE_S = 1.0
OBJECT_KEEPALIVE_S = 2.0
SCHED_KEEPALIVE_ENABLED = os.environ.get("BEAMNG_SCHED_KEEPALIVE", "0") == "1"


def _keepalive_s(name: str) -> float | None:
    """Scheduler keep-alive bound for a modality (None = no floor)."""
    if not SCHED_KEEPALIVE_ENABLED:
        return None
    if name == "range":
        return RANGE_KEEPALIVE_S
    if name == "object":
        return OBJECT_KEEPALIVE_S
    return None


def _keepalive_expired(name: str, age_s: float | None,
                       *, keepalive_s: float | None = None) -> bool:
    """Whether a held/reused output has reached its keep-alive bound.

    ``None`` age (nothing to reuse yet) is never expired - there is no
    bound to protect when the modality has no output at all.

    ``keepalive_s`` overrides the module switch (``None`` = keep reading it),
    so a test can exercise the floor on and off without re-importing the
    module - the switch is otherwise frozen at import time.
    """
    keep = _keepalive_s(name) if keepalive_s is None else keepalive_s
    if keep is None or age_s is None:
        return False
    return float(age_s) >= float(keep)



def _async_allowed(name: str, *, strict: bool) -> bool:
    """Which heavy heads may run in the background worker.

    The semantic lane is strict mode's lateral safety input, and a
    deferred/stale semantic reads as "stale sensor" and fail-closes the
    car - the exact reason the tick-budget governor never defers it in
    strict mode.  So the semantic head is asynchronous ONLY outside
    strict mode; the object head is asynchronous always: the freshness
    contract makes only the semantic head REQUIRED for the road-lateral
    decision, and obstacle safety has its own range/BEV checks.
    """
    if not ASYNC_HEADS_ENABLED:
        return False
    if name == "object":
        return True
    if name == "semantic":
        return not bool(strict)
    return False


def _budget_defers(name: str, *, strict: bool, every_n: int,
                   budget: float | None, elapsed: float,
                   age_s: float | None = None,
                   keepalive_s: float | None = None) -> bool:
    """Whether the tick-budget governor defers this head.

    Smoothness: a heavy head due on a tick that already consumed its
    budget is deferred and the last cached output served, so a semantic +
    YOLO + fresh-LiDAR collision can never freeze the control loop.

    STRICT exception: the semantic lane is the lateral safety input, and
    serving a stale one reads as "stale sensor" -> minimal-risk stop
    (live base_emergency run: 9 stall frames, "tick budget: 19 frames
    deferred 35 heavy heads").  In strict sensor mode the semantic head
    is never deferred - the tick overruns its budget instead, and the
    drive loop's stale-control guard owns the overly-long-tick case.
    YOLO still defers.

    KEEP-ALIVE floor (2026-09-20): ``elapsed`` is the WHOLE tick's spend,
    not this head's, so a ring that costs more than the budget makes every
    later deferral unconditional - 151/151 frames of the 2026-09-20 town
    baseline deferred ``range`` and the car fail-closed into "stale
    sensor" for the entire run.  A deferral is therefore only legal while
    the head's own output is still inside its keep-alive bound; past it
    the head is starved and runs whatever the budget says.  A head with no
    output at all (``age_s=None``) is never deferred either: there is
    nothing to serve from cache.  A disabled floor keeps the old behaviour
    exactly.
    """
    if every_n <= 1 or budget is None:
        return False
    if elapsed <= float(budget):
        return False
    if name == "semantic" and bool(strict):
        return False
    keep = _keepalive_s(name) if keepalive_s is None else keepalive_s
    if keep is None:
        return True                      # floor off: the old behaviour
    if age_s is None:
        # Nothing to reuse yet.  Deferring here does not "serve the last
        # cached output", it leaves the modality absent for the whole run
        # - the 2026-09-20 town baseline had n_object_obstacles = 0 on
        # 151/151 frames for exactly this reason.
        return False
    return float(age_s) < float(keep)


def range_schedule(*, budget: float | None, elapsed: float,
                   age_s: float | None, has_prev: bool,
                   keepalive_s: float | None) -> tuple[str, str]:
    """What the range scan does this tick: ``(action, state)``.

    Extracted from the tick body so the behaviour can be tested with a FAKE
    clock instead of by driving town repeatedly.  The question the A/B could
    not answer - "does the keep-alive floor do anything?" - is exactly the
    ``("scan", "keepalive_forced")`` cell, and it needs (a) the budget already
    blown at this decision point AND (b) the reused scan past its bound.  On
    the 8 clean A/B runs (b) held 26 times but (a) never coincided, which is
    why the floor never fired and the two arms ran identical logic.

    ``keepalive_s`` is passed explicitly rather than read from the module
    switch so a test can turn the floor on and off without re-importing.
    ``None`` means "no floor" - the pre-floor behaviour, deferred
    unconditionally.
    """
    over = budget is not None and float(elapsed) > float(budget)
    starved = (keepalive_s is not None and age_s is not None
               and float(age_s) >= float(keepalive_s))
    if over and has_prev and not starved:
        # Still inside its bound: serving the compensated cache is legal.
        return "defer", "budget_deferred"
    if over and starved:
        # Past its bound: the budget does not get to reuse it.  This is the
        # cell the A/B never reached.
        return "scan", "keepalive_forced"
    return "scan", "scanned"


def sched_record(head: str, state: str, *, source_seq, result_seq,
                 eligible_t, source_t=None, age_s=None, compute_ms=None,
                 reason="", dispatch_t=None, finish_t=None,
                 publish_t=None) -> dict:
    """One head's record for this tick, in the traceability vocabulary.

    The original keys (``state`` / ``age_s`` / ``compute_ms`` / ``reason``)
    keep their exact meaning so existing consumers are unaffected; the added
    ones are what let an anomaly land on a STAGE instead of on a bare age
    (see ``telemetry_contract``).

    Times are ``time.time()`` wall-clock seconds, matching the drive
    command trace.  They are NOT monotonic; duration measurements use
    separate ``perf_counter()`` differences.  ``None`` means "did not
    happen" and is never filled in with a plausible number.
    """
    return {
        "state": state,
        "age_s": age_s,
        "compute_ms": compute_ms,
        "reason": reason,
        # --- traceability contract (plan P1) ---
        "head": head,
        "source_seq": source_seq,
        "result_seq": result_seq,
        "decision_state": state,
        "source_t": source_t,
        "eligible_t": eligible_t,
        "dispatch_t": dispatch_t,
        "finish_t": finish_t,
        "publish_t": publish_t,
    }


def _run_head_job(head, ctx, trace):
    """Measure a connection-free head on the command trace's wall clock."""
    try:
        return head.run(ctx)
    finally:
        trace["finish_t"] = time.time()


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


def _reference_geometry_id(ref) -> str | None:
    """Geom id of an accepted reference, or None when there is none.

    None is not the same as "a hash of an empty reference": a frame with no
    accepted reference at all must publish no version, or a comparison
    between two absent references looks like two matching geometries.  (A
    live run on 2026-09-22 showed exactly that: 16 of 20 frames carried a
    hex ``lane_ref_geom_id`` for a reference that did not exist.)
    """
    if ref is None:
        return None
    try:
        if getattr(ref, "center", None) is None:
            return None
        return ref.geom_id
    except Exception:
        return None


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
                 strict_sensor: bool = False,
                 corridor_fallback: bool = False,
                 paved_fallback: bool = False):
        self.conn = conn
        # Pairing-free strict-mode lane fallback (bev corridor right edge
        # + half a lane, width-gated).  Default OFF: it changes which
        # frames strict mode drives on, so it goes through live A/B
        # before it becomes a default.
        self.corridor_fallback = bool(corridor_fallback)
        # PAVED-boundary candidate for a paved road with NO marking: the
        # road mask's edges are the authority, the reference is half a
        # lane in from the paved right edge and the pavement edges are the
        # hard boundaries (AGENTS.md「驾驶约束」).
        #
        # Default OFF, and it stays that way until a pavement edge is
        # trustworthy.  Measured live 2026-09-18 (east_coast unmarked
        # stretch, deployed model): with the candidate enabled the car
        # rode the lane line and wedged against the guardrail
        # (``lane=paved``, "no drivable path"), because the model reads
        # the gravel shoulder as pavement - its right edge sits 15 px
        # (median) OUTSIDE the hand-labelled pavement (53.6% of rows), so
        # "keep a lane width inside that edge" is a place on the
        # shoulder, and publishing those edges as the hard boundaries
        # WEAKENS the no-cross rule instead of enforcing it (the real
        # lane line is no longer a boundary at all).  Enable only with
        # --paved-lane after the edge is trustworthy.
        self.paved_fallback = bool(paved_fallback)
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
        # Cross-tick stability of the accepted reference (P1-2): side +
        # near-field centre must agree for NEED_TICKS before the reference
        # earns full steering authority; unpaired reads never do.
        self._ref_stability = ReferenceStabilityTracker()
        # Per-head throttling: the expensive heads (semantic UNet
        # ~100-300 ms, YOLO object ~100-200 ms on the live 400x300
        # front frame) run every ``semantic_every_n`` / ``object_every_n``
        # ticks and intermediate ticks reuse their last output.  Cheap
        # heads (traffic) still run on every fresh frame; LiDAR is
        # throttled separately by ``range_every_n``.
        self.semantic_every_n = max(1, int(semantic_every_n))
        self.object_every_n = max(1, int(object_every_n))
        self._head_skip: dict[str, int] = {}
        # Background workers for the heavy heads that may run async (plan
        # A4).  Created lazily on first use so __new__-built test stubs
        # and non-driving callers never spawn a thread.
        self._head_workers: dict[str, LatestJobRunner] = {}
        self._head_async: set[str] = set()
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
    def _range_async_step(self, pos, budget, tick_t0, budget_skips):
        """Off-thread scan step (plan A4): fetch here, cluster in a worker.

        The provider's ``fetch`` takes only the connection reads (under
        the connector lock, on THIS thread); the 200+ ms of clustering
        goes to a latest-wins worker.  Until the result lands the caller
        serves the motion-compensated cached scan - the same reuse
        semantics (and the same ``RANGE_REUSE_*`` bounds) the synchronous
        path already uses when a scan is skipped.  Returns
        ``(sample, error, state)``; ``sample`` may be None when nothing has
        been produced yet, and ``state`` is the scheduler record for this
        tick (which of "adopted / submitted / in flight / deferred /
        keep-alive forced / failed" actually happened).
        """
        runner = getattr(self, "_range_worker", None)
        if runner is None:
            runner = LatestJobRunner("range",
                                     timeout_s=ASYNC_HEAD_TIMEOUT_S)
            self._range_worker = runner
            _async = getattr(self, "_head_async", None)
            if _async is None:
                _async = set()
                self._head_async = _async
            _async.add("range")
        error = None
        rng = None
        ran_ms = None
        _adopted = _submitted = _scanned = _deferred = _forced = False
        result = runner.poll()
        if result is not None:
            if result.ok:
                self._last_range = result.value
                # the cloud was captured when the job was FETCHED, so that
                # is the honest age of the evidence it produced
                self._last_range_t = float(
                    getattr(self, "_range_job_t", 0.0) or time.time())
                rng = result.value
                ran_ms = round(float(result.duration_s) * 1000.0, 1)
                _adopted = True
            else:
                error = str(result.error)
        if rng is None:
            _dt = time.time() - float(getattr(self, "_last_range_t", 0.0))
            rng = compensate_range_motion(
                getattr(self, "_last_range", None), _dt)
        _age = (time.time() - float(getattr(self, "_last_range_t", 0.0))
                if getattr(self, "_last_range", None) is not None else None)
        if not runner.busy:
            # Keep-alive floor: the budget may not push the reused scan past
            # its bound either (the same rule the synchronous path and the
            # every-n throttle obey).
            _forced = _keepalive_expired("range", _age)
            if (budget is not None and (time.time() - tick_t0) > budget
                    and not _forced):
                # over budget this tick: keep the compensated cache, fetch
                # on the next affordable tick (same rule as the sync path)
                budget_skips.append("range")
                _deferred = True
            else:
                payload = None
                try:
                    payload = self.range_prov.fetch(pos)
                except Exception as exc:
                    error = str(exc)
                if payload is not None:
                    self._range_job_t = time.time()
                    runner.submit(self.range_prov.process, payload, pos)
                    _submitted = True
                else:
                    # the provider declared a split but produced no
                    # payload: fall back to the synchronous scan
                    _t_scan = time.perf_counter()
                    rng = self.range_prov.scan(pos)
                    ran_ms = round(
                        (time.perf_counter() - _t_scan) * 1000.0, 1)
                    self._last_range = rng
                    self._last_range_t = time.time()
                    _scanned = True
        if error is not None:
            state = "error"
        elif _forced and (_submitted or _scanned):
            state = "keepalive_forced"
        elif _adopted:
            state = "async_adopted"
        elif _submitted:
            state = "async_submitted"
        elif _scanned:
            state = "scanned_fallback"
        elif _deferred:
            state = "budget_deferred"
        elif runner.busy:
            state = "async_in_flight"
        else:
            state = "reused"
        return rng, error, {"state": state, "age_s": _age,
                            "compute_ms": ran_ms}

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
        # T05: the MEASURED attitude (quaternion) and the ONE ground plane
        # every consumer uses this tick.  Publishing both makes "which
        # geometry produced this number" answerable from the run record
        # instead of from the source.
        _rotation = getattr(st, "rotation", None)
        if not geometry.POSE_ROTATION_ENABLED:
            _rotation = None
        _ground_z = geometry.projection_ground_z(pos)
        out.meta["pose_label"] = geometry.pose_label(
            _rotation, enabled=geometry.POSE_ROTATION_ENABLED)
        out.meta["ground_model"] = geometry.GROUND_MODEL_FLAT
        out.meta["ground_z"] = round(float(_ground_z), 4)
        out.meta["ego_ground_gap_m"] = float(geometry.EGO_GROUND_GAP_M)
        _tw = time.time()
        _times: dict[str, float] = {}
        _tick_cost0 = time.time()
        _budget = (float(time_budget_s)
                   if time_budget_s is not None and time_budget_s > 0.0
                   else None)
        _budget_skips: list[str] = []
        # Per-head scheduler record (2026-09-20): the age says a modality
        # is stale, the error says whether it threw, and this says WHY it
        # did not run - "not due", "budget deferred", "starved past the
        # keep-alive bound", "async in flight", "async failed", or the
        # compute time it actually spent.  Without it "the head is 113 s
        # old" cannot be told apart from "the head is broken".
        _sched: dict[str, dict] = {}
        _perception_ms = {name: None for name in (
            "camera_acquire", "heads", "heads_sync", "heads_async_poll",
            "heads_async_dispatch")}
        out.meta["perception_ms"] = _perception_ms
        out.meta["head_trace_clock"] = "wall_time"
        out.meta["head_source_clock_basis"] = "acquire_return"
        # Result sequence per head: incremented whenever a NEW result is
        # produced, so "the control tick consumed an older version" becomes
        # detectable instead of invisible.  Without it, a reused output and
        # a fresh one are indistinguishable in the telemetry.
        _res_seq = getattr(self, "_head_result_seq", None)
        if _res_seq is None:
            _res_seq = {}
            self._head_result_seq = _res_seq
        _result_trace = getattr(self, "_head_result_trace", None)
        if _result_trace is None:
            _result_trace = {}
            self._head_result_trace = _result_trace
        _new_heads: set[str] = set()

        # --- 1) camera ring -> HydraNet heads ---------------------------
        snap: dict = {}
        if self.ring is not None:
            _acquire_t0 = time.perf_counter()
            try:
                snap = self.ring.grab_ring()
            except Exception as exc:
                out.errors["ring"] = str(exc)
            finally:
                _perception_ms["camera_acquire"] = (
                    time.perf_counter() - _acquire_t0) * 1000.0
        # The provider exposes no exposure timestamp.  This is the observed
        # acquisition-return boundary, not a claim about sensor capture time.
        _source_t = time.time()
        if snap:
            _heads_t0 = time.perf_counter()
            role = "front_main" if "front_main" in snap \
                else next(iter(snap))
            frame, cam = snap[role]
            out.frame = frame
            out.cam = cam
            ctx = FrameContext(
                frame_rgb=frame, cam=cam, pos=pos, heading=heading,
                # the road plane, not the ego-origin plane (T05)
                ground_z=float(_ground_z),
                rotation=_rotation,
                role=role, timestamp=float(_tick_cost0),
                # Per-camera frame counter: the evidence layer keys its
                # votes on (source, capture) so a reprocessed frame cannot
                # add a vote (plan §3.2/T03).
                seq=int(getattr(self, "_tick_num", 0)))
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
                # The head's own output age BEFORE this tick's decision:
                # the keep-alive floor and the deferral telemetry are both
                # about what we would serve, not about what we just ran.
                _stamp_now = _head_stamps.get(_name)
                _age_now = (None if _stamp_now is None
                            else max(0.0, time.time() - float(_stamp_now)))
                # When this head became a candidate for this tick.  All the
                # contract times use time.time(), matching command traces;
                # only the separate duration counters are monotonic.
                _eligible_t = time.time()
                if not _due:
                    _pub = _eligible_t
                    if _last.get(_name) is not None:
                        heads[_name] = _last[_name]
                        _stamp = _head_stamps.get(
                            _name, _tick_cost0)
                        head_ages[_name] = max(0.0, time.time() - _stamp)
                    _sched[_name] = sched_record(
                        _name, "not_due", source_seq=_tick_num,
                        result_seq=_res_seq.get(_name),
                        eligible_t=_pub, source_t=_source_t, age_s=_age_now,
                        reason=f"every_n={_n}", publish_t=_pub)
                    continue
                # Tick time-budget governor (smoothness): when a heavy
                # head is due but this tick has already consumed its time
                # budget, defer it and serve the last cached output.  The
                # head stays due (`_head_retry`) and runs on the first
                # later tick the budget allows, so a semantic + YOLO +
                # fresh-LiDAR collision can never freeze the control loop
                # ("stutter every few frames, car barely moves").
                #
                # STRICT exception: the semantic lane is the lateral
                # safety input, and serving a stale one reads as "stale
                # sensor" -> minimal-risk stop (live base_emergency run:
                # 9 stall frames, "tick budget: 19 frames deferred 35
                # heavy heads").  In strict sensor mode the semantic head
                # is never deferred - the tick overruns its budget
                # instead, and the drive loop's own stale-control guard
                # owns the overly-long-tick case.  YOLO still defers.
                if _async_allowed(
                        _name,
                        strict=bool(getattr(self, "strict_sensor", False))):
                    # Heavy head in the background (plan A4): submit the
                    # current frame and serve the newest AVAILABLE output
                    # this tick - the tick never waits for the head, so a
                    # slow UNet/YOLO costs freshness, not control cadence.
                    # Only pure-CPU heads may go async: anything touching
                    # the BeamNGpy connection stays on this thread.
                    _workers = getattr(self, "_head_workers", None)
                    if _workers is None:
                        _workers = {}
                        self._head_workers = _workers
                    _async_set = getattr(self, "_head_async", None)
                    if _async_set is None:
                        _async_set = set()
                        self._head_async = _async_set
                    _runner = _workers.get(_name)
                    if _runner is None:
                        _runner = LatestJobRunner(
                            _name, timeout_s=ASYNC_HEAD_TIMEOUT_S)
                        _workers[_name] = _runner
                        _async_set.add(_name)
                    _frames = getattr(self, "_head_job_frame_t", None)
                    if _frames is None:
                        _frames = {}
                        self._head_job_frame_t = _frames
                    _job_maps = getattr(self, "_head_job_traces", None)
                    if _job_maps is None:
                        _job_maps = {}
                        self._head_job_traces = _job_maps
                    _jobs = _job_maps.setdefault(_name, {})
                    _poll_t0 = time.perf_counter()
                    _res = _runner.poll() if _due or _runner.busy else None
                    _perception_ms["heads_async_poll"] = (
                        (_perception_ms["heads_async_poll"] or 0.0)
                        + (time.perf_counter() - _poll_t0) * 1000.0)
                    _job = (_jobs.pop(_res.token, {})
                            if _res is not None else {})
                    _finish_t = _job.get("finish_t")
                    _ran_ms = None
                    if _res is not None and _res.ok:
                        heads[_name] = _res.value
                        _last[_name] = _res.value
                        _retry.discard(_name)
                        # a NEW result entered the cache: bump its version
                        _res_seq[_name] = _res_seq.get(_name, 0) + 1
                        # the output's age starts at the frame that
                        # produced it, not at this tick
                        _head_stamps[_name] = float(
                            _job.get("frame_t", _frames.get(_name, _tick_cost0)))
                        _result_trace[_name] = {
                            key: _job.get(key) for key in (
                                "source_seq", "source_t", "eligible_t",
                                "dispatch_t", "finish_t")}
                        _result_trace[_name]["publish_t"] = time.time()
                        _new_heads.add(_name)
                        _ran_ms = round(float(_res.duration_s) * 1000.0, 1)
                    if _name not in heads and _last.get(_name) is not None:
                        heads[_name] = _last[_name]
                    if _res is not None and not _res.ok:
                        self.hydra.errors[_name] = str(_res.error)
                    _stamp = _head_stamps.get(_name)
                    head_ages[_name] = (
                        float("inf") if _stamp is None
                        else max(0.0, time.time() - float(_stamp)))
                    _dispatch_t = None
                    if _due and not _runner.busy:
                        _dispatch_t = time.time()
                        _job_trace = {
                            "source_seq": _tick_num, "source_t": _source_t,
                            "eligible_t": _eligible_t,
                            "dispatch_t": _dispatch_t, "finish_t": None,
                            "frame_t": float(_tick_cost0)}
                        _submit_t0 = time.perf_counter()
                        _token = _runner.submit(_run_head_job, _head, ctx,
                                                _job_trace)
                        _perception_ms["heads_async_dispatch"] = (
                            (_perception_ms["heads_async_dispatch"] or 0.0)
                            + (time.perf_counter() - _submit_t0) * 1000.0)
                        _jobs[_token] = _job_trace
                        # At most one unread result plus one running job.
                        for _old_token in sorted(_jobs)[:-2]:
                            del _jobs[_old_token]
                        _frames[_name] = float(_tick_cost0)
                    if _res is not None and not _res.ok:
                        # The failed job and the newly submitted job are
                        # different attempts; keep the failure's own clock.
                        _dispatch_t = _job.get("dispatch_t")
                    # An async head has four distinct "did not run this
                    # tick" meanings and only the runner knows which one
                    # applies: failed, still in flight, just submitted, or
                    # idle with no output at all.
                    _pub_t = time.time()
                    if _res is not None and not _res.ok:
                        _sched[_name] = sched_record(
                            _name, "async_failed", source_seq=_job.get("source_seq"),
                            result_seq=_res_seq.get(_name),
                            eligible_t=_eligible_t, source_t=_job.get("source_t"),
                            age_s=_age_now,
                            compute_ms=_ran_ms, reason=str(_res.error),
                            dispatch_t=_dispatch_t, finish_t=_finish_t)
                    elif _res is not None and _res.ok:
                        _sched[_name] = sched_record(
                            _name, "async_adopted", source_seq=_tick_num,
                            result_seq=_res_seq.get(_name),
                            eligible_t=_eligible_t, source_t=_source_t, age_s=_age_now,
                            compute_ms=_ran_ms, dispatch_t=_dispatch_t,
                            finish_t=_finish_t, publish_t=_pub_t)
                    elif _runner.busy:
                        _sched[_name] = sched_record(
                            _name, "async_in_flight", source_seq=_tick_num,
                            result_seq=_res_seq.get(_name),
                            eligible_t=_eligible_t, source_t=_source_t, age_s=_age_now,
                            reason=(f"in_flight_s={_runner.in_flight_s():.2f}"),
                            dispatch_t=_dispatch_t)
                    elif _due:
                        _sched[_name] = sched_record(
                            _name, "async_submitted", source_seq=_tick_num,
                            result_seq=_res_seq.get(_name),
                            eligible_t=_eligible_t, source_t=_source_t, age_s=_age_now,
                            dispatch_t=_dispatch_t)
                    else:
                        _sched[_name] = sched_record(
                            _name, "async_idle_no_output",
                            source_seq=_tick_num,
                            result_seq=_res_seq.get(_name),
                            eligible_t=_eligible_t, source_t=_source_t, age_s=_age_now,
                            reason="no result and not due")
                    continue
                _elapsed_now = time.time() - _tick_cost0
                # "Starved, not deferred": the tick is over budget and the
                # head is due, so only the keep-alive floor can let it run
                # - record it as a forced refresh, not as a plain run.
                _keepalive_forced = (
                    _budget is not None
                    and _elapsed_now > float(_budget)
                    and _n > 1
                    and _keepalive_expired(_name, _age_now))
                if _budget_defers(
                        _name, strict=bool(getattr(self, "strict_sensor",
                                                   False)),
                        every_n=_n, budget=_budget,
                        elapsed=_elapsed_now, age_s=_age_now):
                    if _last.get(_name) is not None:
                        heads[_name] = _last[_name]
                        _stamp = _head_stamps.get(
                            _name, _tick_cost0)
                        head_ages[_name] = max(0.0, time.time() - _stamp)
                    _retry.add(_name)
                    _budget_skips.append(_name)
                    _sched[_name] = sched_record(
                        _name, "budget_deferred", source_seq=_tick_num,
                        result_seq=_res_seq.get(_name),
                        eligible_t=_eligible_t, source_t=_source_t, age_s=_age_now,
                        reason=(f"tick {_elapsed_now:.2f}s > budget "
                                f"{float(_budget):.2f}s"),
                        # the cached result is what gets published
                        publish_t=time.time())
                    continue
                _dispatch_t = None
                try:
                    _t_run = time.perf_counter()
                    _dispatch_t = time.time()
                    out_head = _head.run(ctx)
                    _finish_t = time.time()
                    _head_ms = round(
                        (time.perf_counter() - _t_run) * 1000.0, 1)
                    heads[_name] = out_head
                    _last[_name] = out_head
                    _head_stamps[_name] = time.time()
                    head_ages[_name] = 0.0
                    _retry.discard(_name)
                    _res_seq[_name] = _res_seq.get(_name, 0) + 1
                    _result_trace[_name] = {
                        "source_seq": _tick_num, "source_t": _source_t,
                        "eligible_t": _eligible_t,
                        "dispatch_t": _dispatch_t, "finish_t": _finish_t,
                        "publish_t": time.time()}
                    _new_heads.add(_name)
                    _sched[_name] = sched_record(
                        _name,
                        "keepalive_forced" if _keepalive_forced else "ran",
                        source_seq=_tick_num,
                        result_seq=_res_seq.get(_name),
                        eligible_t=_eligible_t, source_t=_source_t, age_s=_age_now,
                        compute_ms=_head_ms, dispatch_t=_dispatch_t,
                        finish_t=_finish_t, publish_t=time.time())
                except Exception as _exc:
                    self.hydra.errors[_name] = str(_exc)
                    # finish_t IS set: the work ended, it just failed, so
                    # the stage is "finished but not published" - not
                    # "in flight", which would look like it is still coming.
                    _sched[_name] = sched_record(
                        _name, "error", source_seq=_tick_num,
                        result_seq=_res_seq.get(_name),
                        eligible_t=_eligible_t, source_t=_source_t, age_s=_age_now,
                        reason=str(_exc), dispatch_t=_dispatch_t,
                        finish_t=time.time())
                finally:
                    _perception_ms["heads_sync"] = (
                        (_perception_ms["heads_sync"] or 0.0)
                        + (time.perf_counter() - _t_run) * 1000.0)
            self._tick_num = _tick_num + 1
            # Keep scheduling attempts distinct from the evidence actually
            # served: a new submission must not relabel an older result.
            for _name, _record in _sched.items():
                _record["attempt_source_seq"] = _record.get("source_seq")
                _record["attempt_source_t"] = _record.get("source_t")
                _record["attempt_eligible_t"] = _record.get("eligible_t")
                _record["attempt_dispatch_t"] = _record.get("dispatch_t")
                _record["attempt_finish_t"] = _record.get("finish_t")
                _record["result_available"] = _name in heads
                if _name in heads:
                    _trace = _result_trace.get(_name, {})
                    for _key in ("source_seq", "source_t", "eligible_t",
                                 "dispatch_t", "finish_t", "publish_t"):
                        _record[_key] = _trace.get(_key)
                else:
                    _record["result_seq"] = None
            _perception_ms["heads"] = (
                time.perf_counter() - _heads_t0) * 1000.0
            if "semantic" in _new_heads:
                _sem_meta = getattr(heads.get("semantic"), "meta", {}) or {}
                for _key in ("semantic_ms", "segmentation_ms"):
                    if isinstance(_sem_meta.get(_key), dict):
                        out.meta[_key] = _sem_meta[_key]
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
            # Per-head scheduler record: WHY each head did not run, not
            # only how old its output is.  Ages are rounded like the rest
            # of the telemetry so the frames stay small.
            if _sched:
                out.meta["head_sched"] = {
                    k: {**v,
                        "age_s": (None if v.get("age_s") is None
                                  else round(float(v["age_s"]), 3))}
                    for k, v in _sched.items()}
            out.meta["object_head"] = int("object" in self.hydra._heads)
            _async = getattr(self, "_head_async", None)
            if _async:
                out.meta["head_async"] = sorted(_async)
                out.meta["head_worker"] = {
                    _n: _r.digest()
                    for _n, _r in getattr(self, "_head_workers", {}).items()
                    if _n in _async}
            # WHICH head threw, in words.  ``head_age_s`` above says a head
            # stopped refreshing; the exception that stopped it is kept on
            # ``hydra.errors`` (dropped again the moment the head succeeds)
            # and was never published - the 2026-09-20 town runs show the
            # object head's age climbing to 113 s with no way to tell a
            # crashed head from a merely slow one.  Sticky by design: it is
            # the LAST error per head, not "an error happened this tick".
            _head_errs = dict(getattr(self.hydra, "errors", None) or {})
            if _head_errs:
                out.meta["head_errors"] = _head_errs
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
                    snap["front_main"][1], pos, heading, step=4,
                    ground_z=float(_ground_z))
            # --- near-field drivable evidence (P1-1) --------------------
            # ``semantic`` is bound inside the guard above and does not
            # exist when the ring returned nothing; read it here instead.
            _nf = nearfield_step(self, out, snap,
                                 out.head_outputs.get("semantic"), grid, pos,
                                 heading, int(getattr(self, "_tick_num", 0)))
            if _nf:
                out.meta.update(_nf)
            # What the planner actually sees in the band that stalls it:
            # reported every tick, in every mode (including ``off``), so the
            # A/B compares the same measurement.
            out.meta["nearfield_cov"] = nearfield_coverage(
                grid, pos, heading, ahead_m=NEARFIELD_BAND_AHEAD_M)
            try:
                # getattr defaults keep __new__-built test stubs
                # (no __init__) working: they always scan.
                if ASYNC_HEADS_ENABLED and bool(
                        getattr(self.range_prov, "range_split", False)):
                    # LiDAR decoupling (plan A4): the provider declared the
                    # fetch/process split, so the clustering runs off this
                    # thread and the tick serves the compensated cache in
                    # the meantime; a slow cloud costs freshness (bounded
                    # by STALE_RANGE_S / RANGE_REUSE_MAX_DT_S), not the
                    # control cadence.
                    rng, _range_err, _range_state = self._range_async_step(
                        pos, _budget, _tick_cost0, _budget_skips)
                    if _range_err:
                        out.errors["range"] = _range_err
                    out.meta["range_async"] = 1
                    out.meta["range_sched"] = _range_state
                    _rworker = getattr(self, "_range_worker", None)
                    if _rworker is not None:
                        # The same health summary the async heads publish:
                        # from ``range_age_s`` alone a WEDGED clustering
                        # worker (busy=1, in_flight_s still growing, no new
                        # submit) is indistinguishable from a healthy-but-
                        # slow one - and range age is what trips the stale
                        # verdict.
                        out.meta["range_worker"] = _rworker.digest()
                elif getattr(self, '_range_skip', 0) <= 0:
                    _has_prev = getattr(self, '_last_range', None) is not None
                    _r_age = ((time.time()
                               - float(getattr(self, '_last_range_t', 0.0)))
                              if _has_prev else None)
                    # Keep-alive floor: the budget may only reuse the scan
                    # while it is still inside its bound.  Past it the
                    # modality is STARVED, not deferred, and the scan runs
                    # whatever the budget says - this is the rule whose
                    # absence deferred range on 151/151 and 120/120 frames
                    # of the 2026-09-20 town runs.
                    # Same decision as before, but named and testable: the
                    # inline version could only be exercised by driving town.
                    _r_elapsed = time.time() - _tick_cost0
                    _action, _state = range_schedule(
                        budget=_budget, elapsed=_r_elapsed,
                        age_s=_r_age, has_prev=_has_prev,
                        keepalive_s=_keepalive_s("range"))
                    if _action == "defer":
                        rng = compensate_range_motion(
                            self._last_range, _r_age)
                        _budget_skips.append("range")
                        out.meta["range_sched"] = {
                            "state": _state, "age_s": _r_age,
                            "compute_ms": None}
                    else:
                        _t_scan = time.perf_counter()
                        rng = self.range_prov.scan(pos)
                        _scan_ms = round(
                            (time.perf_counter() - _t_scan) * 1000.0, 1)
                        self._last_range = rng
                        self._last_range_t = time.time()
                        self._range_skip = max(
                            0, int(getattr(self, 'range_every_n', 1)) - 1)
                        out.meta["range_sched"] = {
                            "state": _state, "age_s": _r_age,
                            "compute_ms": _scan_ms}
                else:
                    _dt = (time.time()
                           - float(getattr(self, '_last_range_t', 0.0)))
                    if _keepalive_expired("range", _dt):
                        # The every-n throttle may not push the safety scan
                        # past its bound either: that reuse is exactly the
                        # shape that produced 4-6 s range age before
                        # range_every_n was forced to 1 in strict mode, and
                        # fixing the throttle alone only handed the
                        # starvation to the budget gate.
                        _t_scan = time.perf_counter()
                        rng = self.range_prov.scan(pos)
                        _scan_ms = round(
                            (time.perf_counter() - _t_scan) * 1000.0, 1)
                        self._last_range = rng
                        self._last_range_t = time.time()
                        self._range_skip = max(
                            0, int(getattr(self, 'range_every_n', 1)) - 1)
                        out.meta["range_sched"] = {
                            "state": "keepalive_forced", "age_s": _dt,
                            "compute_ms": _scan_ms}
                    else:
                        self._range_skip -= 1
                        rng = compensate_range_motion(
                            getattr(self, '_last_range', None), _dt)
                        out.meta["range_sched"] = {
                            "state": "every_n_reuse", "age_s": _dt,
                            "compute_ms": None}
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
                # Prediction layer, published but NOT yet consumed: the
                # tracker has always given every track a smoothed vx/vy and
                # nothing used it, so planning scored candidates against
                # occupancy as of THIS tick and a crossing vehicle was only
                # avoided once it was already inside the corridor.  This is
                # the telemetry-first step - the planner's collision cost
                # consuming predicted poses is a separate, live-validated
                # change (see beamng_autopilot/prediction.py).
                out.meta["prediction"] = prediction_digest(
                    predict_tracks(_active))
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
        if getattr(self, "_last_lane_fusion_debug", None):
            out.meta["lane_fusion_debug"] = dict(
                self._last_lane_fusion_debug)
        if getattr(self, "_last_lane_pair_debug", None):
            out.meta["lane_pair_debug"] = dict(
                self._last_lane_pair_debug)
        self.lane_envelope = SensorLaneEnvelope.from_lane_frame(
            lane_frame, captured_at=time.time()) if lane_frame is not None else None
        out.lane_envelope = self.lane_envelope
        if self.lane_envelope is not None:
            out.meta["lane_envelope"] = self.lane_envelope.as_meta()
        # One owner decides which lane geometry may steer the car this
        # tick (sensor lane -> trusted single painted boundary -> paved
        # boundary on a road with no marking -> map prior in legacy mode
        # only -> BEV free-space centre when there is no nav route).
        # Strict FSD never builds map lane geometry and returns no centre
        # when perception cannot supply one, so the planner can only fail
        # closed.
        #
        # The PAVED candidate is computed here because perception (the
        # semantic road mask + the camera that produced it) lives here;
        # the trust order and the strict gate stay in the lane owner.
        # Skipped when a two-sided sensor lane already exists - that is
        # the one case where the pavement edge can never win, and it is
        # the common case, so the extra back-projection stays off the hot
        # path.
        paved_ref = None
        paved_dbg: dict = {}
        if (bool(getattr(self, "paved_fallback", False))
                and not bool(getattr(lane_frame, "paired", False))):
            try:
                _sem_p = out.head_outputs.get("semantic")
                _role_p = ("front_main" if "front_main" in snap
                           else (next(iter(snap)) if snap else None))
                if (_sem_p is not None and _role_p is not None
                        and "road" in getattr(_sem_p, "masks", {})):
                    # Ground plane, not the vehicle origin: the road mask
                    # is a picture of the ROAD SURFACE (config comment:
                    # the origin plane would bias the read 0.5 m at 5 m).
                    _gz_p = float(pos[2]) - float(EGO_ORIGIN_GROUND_GAP_M)
                    paved_ref = paved_edge_lane_center(
                        _sem_p.masks["road"], snap[_role_p][1], pos,
                        heading, ground_z=_gz_p, debug=paved_dbg)
            except Exception as _exc:
                paved_dbg["mode"] = "error"
                paved_dbg["error"] = str(_exc)
        out.meta["paved_edge_debug"] = dict(paved_dbg)
        if paved_ref is not None:
            out.meta["paved_edge"] = dict(paved_ref.meta)
        lane_mode = getattr(self, "lane_mode", "map")
        _sem_head = (out.head_outputs or {}).get("semantic")
        _sem_marks = list(getattr(getattr(_sem_head, "meta", {}), "get",
                                  lambda *a, **k: [])("markings", []) or [])
        # Count published to telemetry: the lane policy may only use the
        # painted lines it was actually handed, and an empty list here is
        # otherwise indistinguishable from "no line was detected".
        out.meta["lane_marks_n"] = len(_sem_marks)
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
            corridor_fallback=getattr(self, "corridor_fallback", False),
            paved_ref=paved_ref,
            paved_fallback=bool(getattr(self, "paved_fallback", False)),
            warn=_warn_once,
            # Plan E4 context: the previous tick's ACCEPTED reference (the
            # slew limiter already keeps it, and it is updated only after
            # this call, so it is genuinely the previous frame's) and the
            # observed LiDAR corridor for this tick - both only consumed
            # when BEAMNG_LANE_GEOM is on.
            prev_ref=getattr(self, "_lane_ref_prev", None),
            corridor=self._lane_geom_corridor(out, pos, heading),
            markings=_sem_marks,
            # The tick's own observation number: a reference whose frame
            # carries the same number rests on THIS tick's measurement.
            tick_id=int(getattr(self, "_tick_num", 0)),
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
        # Strict perception-led mode enables the slew limiter regardless of
        # the env switch: the accepted reference jumping 1.8 m laterally
        # between ticks (line_lat +1.43 -> -0.33, live base_dirtfix
        # 2026-09-19) bangs the steering full-lock left/right at 3 m/s and
        # put the car into a tree.  The 2026-09-11 A/B that disabled it ran
        # the legacy non-strict regime; in strict mode a reference jump is
        # a perception flip, not a lane change.
        if lane_ref is not None and (_LANE_REF_SLEW_ENABLED
                                     or bool(getattr(self, "strict_sensor",
                                                     False))):
            lane_ref, self._lane_ref_hold_t = limit_reference_slew(
                getattr(self, "_lane_ref_prev", None),
                getattr(self, "_lane_ref_hold_t", 0.0),
                lane_ref, time.time(), pos, heading)
            # The limiter's result becomes THE accepted geometry: writing
            # it back into the one published object is what keeps the
            # planner Scene, the safety monitor and the controller on the
            # same centre.  Sleighing a local copy while the Scene read
            # ``scene_ref`` produced two different references in one tick
            # (plan §2.2-C / §3.3-3).
            self._publish_reference_geometry(lane_ref_out, lane_ref)
        self._lane_ref_prev = (None if lane_ref is None
                               else np.asarray(lane_ref, dtype=float)[:, :2])
        # --- cross-tick reference stability (review handoff P1-2) --------
        # ``paired`` = a two-sided perception read; ``fresh`` = the source
        # is live perception this tick (a hold / rule / stale reference
        # must not accumulate stability just by surviving).
        try:
            _tracker = getattr(self, "_ref_stability", None)
            if _tracker is None:
                _tracker = self._ref_stability = ReferenceStabilityTracker()
            _st = _tracker.update(
                ref=lane_ref, pos=pos, heading=heading,
                # ``two_sided`` = both boundaries are real measurements of
                # one observation (the painted-centre-line frame also sets
                # ``paired=True`` while its right edge is a width prior);
                # ``fresh_obs`` = the observation behind it was taken THIS
                # tick.  Feeding the raw ``lane_frame.paired`` / a source
                # label here let a held constructed reference promote
                # itself to full authority (plan §2.2-A).
                paired=bool(lane_ref_out.two_sided),
                fresh=bool(lane_ref_out.fresh_obs))
        except Exception as exc:
            _warn_once("ref_stability", f"stability tracker failed: {exc}")
            _tracker = getattr(self, "_ref_stability", None)
            if _tracker is not None:
                _tracker.reset()
            _st = None
        if _st is not None:
            out.meta["ref_authority"] = _st.authority
            out.meta["ref_stability_reason"] = _st.reason
            out.meta["ref_side"] = _st.side
            out.meta["ref_lat_m"] = _st.lat_m
            out.meta["ref_stable_ticks"] = int(_st.stable_ticks)
            out.meta["ref_side_flips"] = int(_st.flips_total)
            out.meta["ref_flip"] = int(bool(_st.flip))
        # --- T06 shadow lateral state (READ-ONLY) -----------------------
        # (e, e_dot, theta, theta_dot) + covariance relative to the accepted
        # reference, in the same identity, for comparison against the
        # current behaviour.  Nothing here can move the steering: the drive
        # loop only records it, and the control path never imports it.
        try:
            _shadow = getattr(self, "_lane_shadow", None)
            if _shadow is None:
                from beamng_autopilot.lane.shadow_state import (
                    LateralShadowEstimator)
                _shadow = self._lane_shadow = LateralShadowEstimator()
            _shadow_state = _shadow.update(
                ref=lane_ref_out.center, pos=pos, heading=heading,
                speed_mps=float(getattr(st, "speed", 0.0) or 0.0),
                now=time.time(),
                two_sided=bool(lane_ref_out.two_sided),
                inferred=bool(lane_ref_out.inferred),
                fresh_obs=bool(lane_ref_out.fresh_obs),
                width_m=float(lane_ref_out.width or 0.0),
                lane_id=f"{lane_ref_out.src}|{out.meta.get('ref_side') or '?'}")
            out.meta["lane_shadow"] = _shadow_state.as_dict()
        except Exception as exc:
            out.meta["lane_shadow_error"] = str(exc)
        # --- T09 bounded lateral risk (read-only) -----------------------
        # Signed gaps to the published boundaries, the first crossing (only
        # while the car actually closes on one), the stopping margin with
        # its declared inputs, and how current the evidence is.  UNKNOWN is
        # published explicitly; nothing here is divided by a near-zero rate.
        try:
            from beamng_autopilot.lane.lateral_risk import lateral_risk
            from beamng_autopilot.obstacle_risk import RISK_BRAKE_DECEL_MPS2
            _sh = out.meta.get("lane_shadow") or {}
            _risk = lateral_risk(
                body_half_width_m=float(geometry.FOOTPRINT_HALF_WIDTH_M),
                lat_left_m=(None if out.lane_left is None
                            else float(np.median(np.asarray(
                                out.lane_left, dtype=float)[:, 1])
                                - float(pos[1]))),
                lat_right_m=(None if out.lane_right is None
                             else float(np.median(np.asarray(
                                 out.lane_right, dtype=float)[:, 1])
                                 - float(pos[1]))),
                e_m=_sh.get("e_m"), e_dot_mps=_sh.get("e_dot_mps"),
                speed_mps=float(getattr(st, "speed", 0.0) or 0.0),
                latency_s=float(getattr(self, "risk_latency_s", 0.35)),
                a_min_mps2=float(getattr(self, "a_min_mps2",
                                         RISK_BRAKE_DECEL_MPS2)),
                evidence_age_s=out.meta.get("line_evidence_age_s"),
                history_only_frac=(out.meta.get("lane_ref_support") or {}
                                   ).get("history_only_frac")
                if isinstance(out.meta.get("lane_ref_support"), dict) else None)
            out.meta["lateral_risk"] = _risk.as_dict()
        except Exception as exc:
            out.meta["lateral_risk_error"] = str(exc)
        lane_left = lane_ref_out.left
        lane_right = lane_ref_out.right
        lane_width = lane_ref_out.width
        map_lane = lane_ref_out.map_lane
        lane_rejected = lane_ref_out.rejected
        lane_src_sel = lane_ref_out.src
        strict_lane = lane_ref_out.strict
        out.meta.update(lane_ref_out.meta)
        out.lane_ref = lane_ref
        # Version tags for the one-reference contract: the accepted object
        # the controller consumes, and the geometry the planner Scene was
        # built from.  They must match every tick; a live run or a test can
        # check that instead of assuming it (plan §3.3-4).
        out.meta["lane_ref_geom_id"] = _reference_geometry_id(lane_ref_out)
        out.meta["lane_ref_two_sided"] = int(bool(lane_ref_out.two_sided))
        out.meta["lane_ref_fresh_obs"] = int(bool(lane_ref_out.fresh_obs))
        out.meta["lane_ref_inferred"] = int(bool(lane_ref_out.inferred))
        # How much of the ACCEPTED reference's own geometry is current
        # evidence vs only remembered history (plan T03: "选中边界 current
        # 支持与 history-only 支持"), plus the per-band local ages that a
        # single global ratio hides ("far refreshed, near expired").  Reads
        # only - it never adds a vote.
        try:
            _ev_acc = getattr(
                self.hydra._heads.get("semantic"), "_evidence", None)
            if _ev_acc is not None and out.lane_ref is not None:
                _now_ev = time.time()
                _sup = getattr(_ev_acc, "support_digest", None)
                if callable(_sup):
                    # The published BOUNDARIES are the geometry that can sit
                    # on paint cells; the lane CENTRE is a constructed
                    # offset from them, so its own support is expected to be
                    # low and is reported separately rather than conflated.
                    if out.lane_left is not None:
                        out.meta["lane_left_support"] = _sup(
                            out.lane_left, now=_now_ev,
                            role="lane_left_boundary")
                    if out.lane_right is not None:
                        out.meta["lane_right_support"] = _sup(
                            out.lane_right, now=_now_ev,
                            role="lane_right_boundary")
                    out.meta["lane_ref_support"] = _sup(
                        out.lane_ref, now=_now_ev,
                        role="lane_centre (offset from the paint by "
                             "construction)")
                _bands = getattr(_ev_acc, "local_bands", None)
                if callable(_bands):
                    out.meta["line_evidence_bands"] = _bands(
                        pos, heading, now=_now_ev)
        except Exception as exc:
            out.meta["lane_ref_support_error"] = str(exc)
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
        # The planner's copy must be the SAME geometry the controller got
        # (``out.lane_ref`` above, post-slew).  Publishing its id here makes
        # the two comparable per tick: equal ids = one reference; different
        # ids = the divergence this plan forbids (plan §3.3-4).
        out.meta["scene_ref_geom_id"] = (
            None if scene_lane_ref is None
            else _reference_geometry_id(lane_ref_out))
        # A frame either has no reference on either side (both ids None) or
        # one geometry that both consumers share.  Anything else is the
        # divergence this contract forbids, and it is flagged rather than
        # left for a reader to notice.
        if out.meta.get("scene_ref_geom_id") != out.meta.get("lane_ref_geom_id"):
            out.meta["scene_ref_geom_mismatch"] = (
                "planner Scene reference and the accepted reference the "
                "controller consumes are not the same geometry (or one of "
                "them is missing)")
        elif scene_lane_ref is not None and out.lane_ref is not None:
            _a = np.asarray(scene_lane_ref, dtype=float)
            _b = np.asarray(out.lane_ref, dtype=float)
            if _a.shape != _b.shape or not np.allclose(
                    _a, _b, atol=1e-6, equal_nan=True):
                out.meta["scene_ref_geom_mismatch"] = (
                    "planner Scene reference differs from the accepted "
                    "reference the controller consumes")
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
        if (candidate_ref is not None and len(candidate_ref) >= 4
                and not strict_lane):
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
        elif candidate_ref is not None and strict_lane:
            # Strict perception owns the lane and its detected boundaries.
            # Synthetic +/-1.5/3.0 m shifts are map-style lane changes, not
            # evasive paths justified by this tick's sensor evidence; on a
            # centre-paint-only frame they put the body across the physical
            # centreline and trigger the planned-body-cross stop every
            # tick.  Keep the sensor lane centre and the physical arc fan
            # only.  A future obstacle manoeuvre must be generated from an
            # observed free corridor, not from a fixed lateral shift.
            out.meta["strict_lane_shift_disabled"] = 1
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
        # Candidate hysteresis (plan phase D1): the fan is re-scored every
        # tick and two near-equal candidates would otherwise win on
        # alternate frames, flipping the steering intent every 0.5 s.
        # The choice sticks while the kept candidate is STILL feasible
        # this tick and nothing beats it by more than the margin; the
        # decision is published for telemetry (switch count / reason /
        # age).  Lazy creation keeps ``__new__``-built test stubs valid.
        _hyst = getattr(self, "candidate_hysteresis", None)
        if _hyst is None:
            from beamng_autopilot.planning import CandidateHysteresis
            _hyst = CandidateHysteresis()
            self.candidate_hysteresis = _hyst
        best, meta = select_trajectory(scene, fans, self.constraints,
                                       hysteresis=_hyst,
                                       now_s=time.time())
        if meta.get("hysteresis") is not None:
            out.meta["hysteresis"] = dict(meta["hysteresis"])
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
        # WHY every candidate was rejected this tick (review P1-3): the
        # constraint layer used to drop the reason, so "no drivable path"
        # could not be told apart from a boundary gate, a blind-evidence
        # gate or an empty drivable layer.  Published whenever the planner
        # declined, empty dict otherwise.
        _rej = meta.get("rejects")
        if _rej:
            out.meta["plan_rejects"] = dict(_rej)
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


    def _lane_geom_corridor(self, out, pos, heading):
        """The LiDAR corridor polyline for the E4 check (None when off).

        Built only when the geometry gate is enabled: it is a projection
        over the tick's ray hits, and with the switch off it would be
        pure waste.  A failure here degrades to "no corridor evidence",
        which the check reports as a missing metric rather than a
        rejection.
        """
        try:
            from beamng_autopilot.lane.reference import LANE_GEOM_ENABLED
        except Exception:
            return None
        if not LANE_GEOM_ENABLED:
            return None
        hits = list(getattr(out, "ray_hits", None) or [])
        if not hits:
            return None
        try:
            frame = build_lidar_corridor(hits, pos, heading)
        except Exception as exc:
            _warn_once("lane_geom_corridor", f"corridor build failed: {exc}")
            return None
        center = getattr(frame, "center", None)
        if center is None:
            return None
        center = np.asarray(center, dtype=float)[:, :2]
        return center if len(center) >= 2 else None

    @staticmethod
    def _publish_reference_geometry(ref, geometry) -> bool:
        """Write the FINAL accepted geometry back into the reference object.

        One tick has exactly one accepted lateral reference (plan §3.3):
        whoever limits or smooths it must update the object every consumer
        reads, not a local copy.  Returns True when the geometry was
        adopted, False when there was nothing to write.
        """
        if ref is None or geometry is None:
            return False
        try:
            arr = np.asarray(geometry, dtype=float)
            if arr.ndim != 2 or len(arr) < 3:
                return False
            ref.center = arr[:, :2]
            return True
        except Exception:
            return False

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
        # Stamp WHICH observation each freshly built frame carries, before
        # fusion can adopt or hold it.  A frame served from the hold/coast
        # keeps the old number, so "the reference survived another tick"
        # can never be read as "we measured it again" (T02/§3.2).
        _obs_tick = int(getattr(self, "_tick_num", 0))
        for _f in (vision, lidar):
            if _f is not None:
                try:
                    _f.obs_seq = _obs_tick
                except Exception:
                    pass
        if frame is None:
            try:
                from beamng_autopilot.lane import lane_frame_usable
                self._last_lane_fusion_debug = {
                    "vision": (None if vision is None else {
                        "usable": bool(lane_frame_usable(vision)),
                        "paired": bool(vision.paired),
                        "confidence": round(float(vision.confidence), 3),
                        "span_m": round(float(vision.span_m), 2),
                        "width_m": round(float(vision.width), 2),
                        "sources": tuple(vision.sources),
                    }),
                    "lidar": (None if lidar is None else {
                        "usable": bool(lane_frame_usable(lidar, 0.35)),
                        "paired": bool(lidar.paired),
                        "confidence": round(float(lidar.confidence), 3),
                        "span_m": round(float(lidar.span_m), 2),
                        "sources": tuple(lidar.sources),
                    }),
                    "chosen": None,
                }
            except Exception:
                pass
            return None
        try:
            from beamng_autopilot.lane import lane_frame_usable
            self._last_lane_fusion_debug = {
                "vision": (None if vision is None else {
                    "usable": bool(lane_frame_usable(vision)),
                    "paired": bool(vision.paired),
                    "confidence": round(float(vision.confidence), 3),
                    "span_m": round(float(vision.span_m), 2),
                    "sources": tuple(vision.sources),
                }),
                "lidar": (None if lidar is None else {
                    "usable": bool(lane_frame_usable(lidar, 0.35)),
                    "paired": bool(lidar.paired),
                    "confidence": round(float(lidar.confidence), 3),
                    "span_m": round(float(lidar.span_m), 2),
                    "sources": tuple(lidar.sources),
                }),
                "chosen": {
                    "paired": bool(frame.paired),
                    "confidence": round(float(frame.confidence), 3),
                    "span_m": round(float(frame.span_m), 2),
                    "sources": tuple(frame.sources),
                },
            }
        except Exception:
            pass
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
        debug: dict = {}
        self._last_lane_pair_debug = debug
        try:
            from beamng_autopilot.lane import pair_lane_markings
            sem = head_outputs.get("semantic")
            if sem is None:
                debug["mode"] = "no_semantic"
                return None
            markings = sem.meta.get("markings", [])
            debug["marking_count"] = len(markings)
            if not markings:
                debug["mode"] = "no_markings"
                return None
            frame = pair_lane_markings(markings, pos, heading,
                                       debug=debug)
            debug["result"] = (None if frame is None else {
                "paired": bool(frame.paired),
                "confidence": round(float(frame.confidence), 3),
                "span_m": round(float(frame.span_m), 2),
            })
            return frame
        except Exception as exc:
            debug["mode"] = "error"
            debug["error"] = str(exc)
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
        # Drop any in-flight/received async head output: a result computed
        # from a pre-teleport frame must never be adopted at the new pose.
        for _r in getattr(self, "_head_workers", {}).values():
            try:
                _r.poll()
            except Exception:
                pass
        _rr = getattr(self, "_range_worker", None)
        if _rr is not None:
            try:
                _rr.poll()
            except Exception:
                pass
            self._last_range = None
            self._last_range_t = 0.0
        for _n in getattr(self, "_head_async", ()) or ():
            for _d in (getattr(self, "_head_timestamps", None),
                       getattr(self, "_head_job_frame_t", None)):
                try:
                    _d.pop(_n, None)
                except Exception:
                    pass
        # Heads may hold location-bound temporal state of their own
        # (semantic head: world-space line evidence) - clear it too.
        for _h in getattr(self, "heads", None) or []:
            _rst = getattr(_h, "reset", None)
            if callable(_rst):
                _rst()

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
        # Line-evidence provenance (plan phase E3): how much of the fused
        # line mask is a fresh observation vs. held history, and whether
        # continuous loss has invalidated that history.
        ev = sem.meta.get("line_evidence")
        if isinstance(ev, dict):
            meta["line_conf_current"] = ev.get("current_confidence")
            meta["line_conf_history"] = ev.get("history_confidence")
            meta["line_evidence_age_s"] = ev.get("since_observation_s")
            meta["line_evidence_expired"] = int(bool(ev.get("expired")))
            # T03 provenance: the added pixels split by ORIGIN (this
            # observation / re-projected history / yellow prior) and the
            # source-event counters that say whether a refresh actually
            # produced new source evidence.
            meta["line_added_current_px"] = ev.get("added_pixels_current")
            meta["line_added_history_px"] = ev.get("added_pixels_history")
            meta["line_added_yellow_px"] = ev.get("added_pixels_yellow")
            meta["line_yellow_in_line_px"] = ev.get("yellow_pixels_in_line")
            meta["line_evidence_events"] = ev.get("source_events")
    tr = head_outputs.get("traffic")
    if tr is not None:
        meta["signal_state"] = tr.meta.get("signal_state")
        meta["signal_conf"] = tr.meta.get("signal_conf")
    topo = head_outputs.get("topology")
    if topo is not None:
        meta["change_left"] = topo.meta.get("change_left")
        meta["change_right"] = topo.meta.get("change_right")
    return meta
