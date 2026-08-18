"""AutopilotSession: main loop and all state for the M5 autopilot assistant.

Refactored out of scripts/m5_autopilot.py so that the entry point stays
thin and the business logic lives in a reusable class.
"""

from __future__ import annotations

import argparse
import atexit
import ctypes
import faulthandler
import json
import math
import os
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from beamng_autopilot import config
from beamng_autopilot.bridge import ControlBridge
from beamng_autopilot.connector import BeamNGConnector
from beamng_autopilot.control import gearbox
from beamng_autopilot.control.handover import handover_vehicle
from beamng_autopilot.control.pure_pursuit import PurePursuit
from beamng_autopilot.control.speed import SpeedController
from beamng_autopilot.hotkeys import (
    HotkeyListener,
    MOD_CONTROL,
    VK_F8,
    VK_F9,
    VK_F10,
    VK_F11,
    VK_F12,
    VK_Q,
)
from beamng_autopilot.hud import LiveHUD
from beamng_autopilot.lane import (
    LaneTracker,
    _boundary_near_lat,
    build_lidar_corridor,
    choose_sensor_lane,
    lane_frame_usable,
    pair_lane_markings,
)
from beamng_autopilot.perception import (
    drop_vision_waypoint_ghosts,
    errors_active,
    errors_summary,
    last_error,
    merge_obstacles,
    filter_self_overlap,
)
from beamng_autopilot.planner import (
    RIGHT_OFFSET_M,
    SHARP_ANGLE_DEG,
    SHARP_CORNER_KPH,
    LocalPlanner,
    _point_route_pos,
    adaptive_lookahead_idx,
    corner_angle_deg,
    creep_speed,
    is_sparse_raycast_speck,
)
from beamng_autopilot.roadnet import RoadNetwork
from beamng_autopilot.traffic import (
    RoadRuleView,
    SignalRule,
    OvertakeStateMachine,
    apply_rule_speed,
    find_lead_vehicle,
    follow_speed,
    oncoming_vehicle_ahead,
    select_signal_rule,
    signal_action_label,
    solid_marking_left,
    vehicle_along_speed,
    signal_distance,
    signal_requires_stop,
)
from beamng_autopilot.runtime import build_camera_provider, build_range_provider
from beamng_autopilot.telemetry import TelemetryBroadcaster
from beamng_autopilot.telemetry_chart import plot_telemetry
from beamng_autopilot.vision.projection import default_camera
from beamng_autopilot.vision.lanes import LaneDetector, MarkingSmoother
from beamng_autopilot.vision.tracking import VisionTrack, update_vision_tracks
from beamng_autopilot.watchdog import (
    arm as wd_arm,
    disarm as wd_disarm,
    heartbeat as wd_heartbeat,
)
from beamng_autopilot.visionview import (
    WorldOverlay,
    render_birdview,
    render_camera_overlay,
)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

CAM_W, CAM_H = 1076, 806
GOAL_RADIUS_M = 8.0
RAMP_ACCEL = 2.5
RAMP_DECEL = 3.5
STEER_SMOOTH = 0.35
STEER_RATE = 1.2
CREEP_MPS = 1.5
RULE_POLL_INTERVAL_S = 0.8
HEADING_DEV_DEG = 8.0
HEADING_DEV_RELEASE = 6.0
HEADING_DEV_CAP = 2.5
HEADING_DEV_CRAWL = 0.6
REVERSE_SPEED_MPS = -0.3
REVERSE_ENGAGE_S = 0.30
REVERSE_HOLD_S = 1.0
REVERSE_STOP_DIST_M = 1.0
REVERSE_SOFT_BRAKE = 0.35
REVERSE_MOVEMENT_M = 0.10
REVERSE_BACK_RATIO = 0.30
PLAN_INTERVAL_S = 0.15
RANGE_SCAN_INTERVAL_S = 0.2
VIS_TRACK_MATCH_M = 1.8
VIS_TRACK_CONFIRM = 2
VIS_TRACK_TTL_S = 8.0
VIS_TRACK_EGO_GATE_M = 0.8
VIS_TRACK_RIDE_RATIO = 0.6
VIS_LANE_REUSE_MISS = 6
VIS_LANE_REUSE_TTL_S = 1.0
VIS_LANE_REUSE_EGO_M = 6.0


# ---------------------------------------------------------------------------
# Module-level utility functions
# ---------------------------------------------------------------------------

def lane_reuse_expired(miss: int, age_s: float, drive_m: float) -> bool:
    """A vision lane is stale once too many scans missed it, too much wall
    time passed, or the car travelled farther than the frame can be
    transformed reliably."""
    return (miss > VIS_LANE_REUSE_MISS
            or age_s > VIS_LANE_REUSE_TTL_S
            or drive_m > VIS_LANE_REUSE_EGO_M)


def heading_deviation_deg(route, nearest, fwd) -> float:
    """Degrees between the ego forward vector and the route tangent at the
    nearest waypoint (0.0 when the route is degenerate)."""
    if route is None or nearest is None or len(route) < 2:
        return 0.0
    i1 = max(0, min(len(route) - 1, nearest - 1))
    i2 = max(0, min(len(route) - 1, nearest + 1))
    tv = np.asarray(route[i2][:2], dtype=float) - np.asarray(
        route[i1][:2], dtype=float)
    hv = np.asarray(fwd[:2], dtype=float)
    n_t, n_h = float(np.linalg.norm(tv)), float(np.linalg.norm(hv))
    if n_t < 1e-6 or n_h < 1e-6:
        return 0.0
    cos_a = float(np.clip(np.dot(hv / n_h, tv / n_t), -1.0, 1.0))
    return math.degrees(math.acos(cos_a))


def nearest_route_point(route, pos, fwd) -> int:
    """Index of the route point nearest to ``pos``, consistent with the
    ego heading."""
    route = np.asarray(route[:, :2], dtype=float)
    pos = np.asarray(pos[:2], dtype=float)
    fwd = np.asarray(fwd[:2], dtype=float)
    n = len(route)
    if n < 3:
        return 0
    d = np.linalg.norm(route - pos, axis=1)
    raw = int(np.argmin(d))
    fn = float(np.linalg.norm(fwd))
    if fn < 1e-6:
        return raw
    fwd = fwd / fn
    tv = np.zeros_like(route)
    tv[1:-1] = route[2:] - route[:-2]
    tv[0] = route[1] - route[0]
    tv[-1] = route[-1] - route[-2]
    tn = np.linalg.norm(tv, axis=1)
    align = np.zeros(n)
    ok = tn > 1e-6
    align[ok] = (tv[ok] / tn[ok, None]) @ fwd
    d_gated = np.where(align >= -0.2, d, np.inf)
    gated = int(np.argmin(d_gated))
    if d_gated[gated] <= max(25.0, 3.0 * float(d[raw])):
        return gated
    return raw


def smooth_steer(prev: float, new: float, dt: float,
                 rate: float = STEER_RATE) -> float:
    """Rate-limited steering smoothing (normalized input units)."""
    max_step = float(rate * max(1e-3, dt))
    return prev + float(np.clip(new - prev, -max_step, max_step))


def heading_dev_speed_cap(dev_deg: float,
                          engaged: bool = False) -> float | None:
    """Speed cap (m/s) once the car points away from the route direction;
    None inside the dead zone."""
    limit = HEADING_DEV_RELEASE if engaged else HEADING_DEV_DEG
    if dev_deg <= limit:
        return None
    t = min(1.0, max(0.0, (90.0 - dev_deg) / (90.0 - HEADING_DEV_DEG)))
    return HEADING_DEV_CRAWL + (HEADING_DEV_CAP - HEADING_DEV_CRAWL) * t


def _route_spacing_m(route) -> float:
    """Median segment length (m) of a route polyline."""
    r = np.asarray(route[:, :2], dtype=float)
    if len(r) < 2:
        return 2.0
    seg = np.linalg.norm(np.diff(r, axis=0), axis=1)
    if not len(seg):
        return 2.0
    return float(np.median(seg))


def beamng_process_running() -> bool:
    """True when a BeamNG game process exists (even without the comms port)."""
    import psutil
    names = config.BEAMNG_PROCESS_NAMES
    for proc in psutil.process_iter(["name"]):
        try:
            if proc.info["name"] in names:
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False


def _acquire_single_instance(name: str) -> bool:
    """Return True if this process is the only m5 instance (named mutex)."""
    ERROR_ALREADY_EXISTS = 183
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.CreateMutexW(None, False, name)
    if not handle:
        return False
    if kernel32.GetLastError() == ERROR_ALREADY_EXISTS:
        kernel32.CloseHandle(handle)
        return False
    _acquire_single_instance._handle = handle  # keep alive for process lifetime
    return True


# ---------------------------------------------------------------------------
# AutopilotSession
# ---------------------------------------------------------------------------

class AutopilotSession:
    """Main autopilot session: owns all state, workers, and the control loop.

    Usage::

        session = AutopilotSession(args)
        session.run()
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.cruise_speed: float = args.speed

        # --- connections / services ------------------------------------------------
        self.conn = BeamNGConnector(
            args.map, args.vehicle,
            port=(args.port or config.runtime_port(args.runtime)),
            home=config.runtime_home(args.runtime))
        self.vision_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.range_lock = threading.Lock()
        self.latest_st = None
        self.range_snapshot: dict = {
            "seq": 0, "ts": 0.0, "obstacles": [], "ray_hits": [],
            "sensor_ok": True, "failures": 0, "error": None,
        }
        self.last_range_seq = 0
        self.vision_snapshot: dict = {
            "seq": 0, "ts": 0.0, "frame": None, "lanes": [],
            "lane_frame": None, "lane_n": 0, "lane_miss": 0,
            "lane_reuse": 0, "lane_reuse_age": 0.0,
            "lane_reuse_drive": 0.0, "vis_obs": [], "det_boxes": [],
            "hud_boxes": [], "vision_n": 0, "failures": 0,
            "black_frames": 0, "error": None,
        }
        self.hotkeys = HotkeyListener(
            bindings={
                VK_F8: "vision",
                VK_F9: "autopilot",
                VK_F10: "navroute",
                VK_F11: "clear",
                VK_F12: "quit",
            },
            modifier_alternates={(MOD_CONTROL, VK_Q): "quit"},
        )
        self.ctl = ControlBridge()
        self.ctl_seen = self.ctl.current_seq()
        self.roadnet = RoadNetwork()
        self.telemetry = TelemetryBroadcaster()
        self.overlay = WorldOverlay(self.conn.bng)
        self.hud = None if args.no_hud else LiveHUD(show_camera=True)
        self.cam_model = default_camera(CAM_W, CAM_H)
        self.camera_provider = None
        self.range_provider = None
        self.pp = PurePursuit(lookahead=6.0)
        self.planner = LocalPlanner(
            right_offset=args.right_offset,
            sharp_angle_deg=args.sharp_angle,
            sharp_corner_kph=args.sharp_corner_kph,
        )
        self.speed_ctrl = SpeedController()

        # --- state ---------------------------------------------------------------
        self.route: np.ndarray | None = None
        self.autopilot = False
        self.vision = True
        self.quit_flag = False
        self.fwd_gear = 2
        self.saved_gearbox: str | None = None
        self.gearbox_switched = False
        self.session_t0 = 0.0
        self.hist: dict[str, list] = {
            "t": [], "throttle": [], "brake": [], "speed": [],
        }

        self.last_overlay = 0.0
        self.last_hud = 0.0
        self.last_scan = 0.0
        self.last_wspd = 0.0
        self.wheel_speed: float | None = None
        self.wspd_warned = False
        self.last_status = 0.0
        self.last_idle = 0.0
        self.last_ctrl = time.time()
        self.last_plan_t = 0.0
        self.last_plan_route: np.ndarray | None = None
        self.last_plan_rule: RoadRuleView | None = None
        self.corner_hold_until = 0.0
        self.corner_held_speed: float | None = None
        self.cached_drive_route: np.ndarray | None = None
        self.cached_blocked = False
        self.last_vision = 0.0
        self.last_vision_seq = 0
        self.vision_det = None
        self.lane_det = None
        self.last_lanes: list = []
        self.last_lane_frame = None
        self.last_lidar_hits: list = []
        self.lane_tracker = LaneTracker()
        self.lane_fusion_state: dict = {}
        self.vision_clear_requested = False
        self.vision_clear_generation = 0
        self.last_vision_generation = 0
        self.lane_miss = 0
        self.lane_n = 0
        self.vision_n = 0
        self.vision_failures = 0
        self.black_frames = 0
        self.vis_tracks: list[VisionTrack] = []
        self.vision_conf_n = 0
        self.det_boxes: list = []
        self.hud_boxes: list = []
        self.nearest = 0
        self.last_st = None
        self.obstacles: list = []
        self.obs_dist = 999.0
        self.sensor_ok = True
        self.scan_failures = 0
        self.last_beat = 0.0
        self.last_rearm = 0.0
        self.target_speed = 0.0
        self.prev_steer = 0.0
        self.creep_since: float | None = None
        self.hdg_engaged = False
        self.cur_env: dict = {}
        self.last_env = 0.0
        self.road_polys: list = []
        self.last_roads = 0.0
        self.road_rule: RoadRuleView | None = None
        self.signal_rules: list[SignalRule] = []
        self.selected_signal: SignalRule | None = None
        self.rule_speed_limit: float | None = None
        self.rule_limit: float | None = None
        self.rule_reason: str | None = None
        self.signal_state: str | None = None
        self.signal_action: int = 0
        self.signal_dist: float | None = None
        self.signal_name: str = ""
        self.last_rule = 0.0
        self.last_progress_pos: np.ndarray | None = None
        self.last_progress_t = 0.0
        self.stuck_recover_until = 0.0
        self.stuck_retries = 0
        self.force_end_reason = ""
        self.rev_start_pos: np.ndarray | None = None
        self.rev_dist = 0.0
        self.rev_since = 0.0
        self.rev_hold_until = 0.0
        self.rev_warned = False
        self.rev_guard_logged = False
        self.last_route_arc: float | None = None
        self.route_rev_dist = 0.0
        self.last_pos2d: np.ndarray | None = None
        self.last_rev_nearest = 0

        # Watchdog blocking-phase state
        self.blocking_active = False
        self.blocking_since = 0.0

        # Overlay display helpers (written during _control_tick)
        self._lane_src: str = ""
        self._lane_conf: float = 0.0
        self._last_blk_log: float = 0.0

    # ------------------------------------------------------------------
    # Context manager for watchdog blocking phases
    # ------------------------------------------------------------------

    @contextmanager
    def _wd_blocking(self) -> Iterator[None]:
        self.blocking_active = True
        self.blocking_since = time.time()
        try:
            yield
        finally:
            self.blocking_active = False

    # ------------------------------------------------------------------
    # Helper methods (originally nested functions in main())
    # ------------------------------------------------------------------

    def toast(self, msg: str) -> None:
        print(f"[m5] {msg}")
        try:
            with self.conn.io_lock:
                self.conn.bng.display_gui_message(msg)
        except Exception:
            pass

    def roads_json(self, xy) -> list:
        """Nearby road centre lines as JSON-safe, downsampled polylines."""
        try:
            polys = self.roadnet.nearby_polylines(xy, 90.0)
            out = []
            for pl in polys:
                step = max(1, int(len(pl) // 80))
                out.append([[round(float(x), 1), round(float(y), 1)]
                            for x, y in pl[::step]])
            return out[:16]
        except Exception:
            return []

    def markings_json(self, lanes) -> list:
        """Detected lane markings as JSON-safe, downsampled polylines."""
        out = []
        try:
            for mk in lanes[:20]:
                world = getattr(mk, "world", None)
                if world is None:
                    continue
                pts = np.asarray(world, dtype=float)
                if pts.ndim != 2 or pts.shape[1] < 2 or len(pts) < 2:
                    continue
                step = max(1, int(len(pts) // 24))
                out.append({
                    "color": str(getattr(mk, "color", "white")),
                    "kind": str(getattr(mk, "kind", "unknown")),
                    "poly": [[round(float(x), 1), round(float(y), 1)]
                             for x, y in pts[::step]],
                })
        except Exception:
            return []
        return out

    def release_control(self) -> None:
        """Hand the vehicle back to the player without leaving it rolling."""
        with self._wd_blocking():
            handover_vehicle(self.conn, self.saved_gearbox,
                             self.gearbox_switched)
        self.gearbox_switched = False

    def read_gearbox_mode(self) -> str | None:
        """Return the current gearbox mode name or None when unavailable."""
        with self.conn.io_lock:
            return gearbox.read_gearbox_mode(self.conn.vehicle)

    def set_gearbox_mode(self, mode: str) -> None:
        with self.conn.io_lock:
            gearbox.set_gearbox_mode(self.conn.vehicle, mode)

    def read_gear(self) -> str | None:
        with self.conn.io_lock:
            return gearbox.read_gear(self.conn)

    def forward_gear_input(self) -> int:
        with self._wd_blocking():
            return gearbox.forward_gear_input(self.conn)

    def restore_gearbox(self) -> None:
        """No-op kept for call-site compatibility."""
        return

    def finish_session(self, show_chart: bool = True) -> None:
        if self.hist["t"]:
            ts = time.strftime("%Y%m%d_%H%M%S")
            p = config.LOGS_DIR / "telemetry" / (f"m5_telemetry_{ts}.png")
            plot_telemetry(self.hist, p, block=False,
                           show=show_chart and not self.args.no_show)
            print(f"[m5] telemetry chart saved -> {p}")
            try:
                t_arr = np.asarray(self.hist["t"], dtype=float)
                spd = np.asarray(self.hist["speed"], dtype=float)
                thr = np.asarray(self.hist["throttle"], dtype=float)
                brk = np.asarray(self.hist["brake"], dtype=float)
                summary = {
                    "ts": ts,
                    "png": str(p),
                    "duration": (round(float(t_arr[-1]), 1)
                                 if len(t_arr) else 0.0),
                    "max_speed": (round(float(spd.max()), 2)
                                  if len(spd) else 0.0),
                    "avg_speed": (round(float(spd.mean()), 2)
                                  if len(spd) else 0.0),
                    "throttle_ratio": (round(float((thr > 0.02).mean()), 3)
                                       if len(thr) else 0.0),
                    "brake_ratio": (round(float((brk > 0.02).mean()), 3)
                                    if len(brk) else 0.0),
                }
                lp = config.LOGS_DIR / "telemetry" / "last_session.json"
                lp.write_text(json.dumps(summary, ensure_ascii=False,
                                        indent=2), encoding="utf-8")
                print(f"[m5] session summary -> {lp}")
            except Exception as exc:
                print(f"[m5] session summary write failed: {exc}")
        self.hist = {"t": [], "throttle": [], "brake": [], "speed": []}

    # ------------------------------------------------------------------
    # Worker threads
    # ------------------------------------------------------------------

    def _vision_worker(self) -> None:
        lane_det_worker = None
        lane_smoother_worker = None
        seg_worker = None
        vision_det_worker = None
        last_lanes_worker: list = []
        last_lanes_ts_worker = 0.0
        last_lanes_pos_worker = None
        last_lane_frame_worker = None
        last_lane_ts_worker = 0.0
        last_lane_pos_worker = None
        lane_miss_worker = 0
        last_ts = 0.0
        while not self.quit_flag:
            if not self.autopilot:
                time.sleep(0.05)
                continue
            now = time.time()
            rate = max(1.0, self.args.vision_rate)
            if now - last_ts < 1.0 / rate:
                time.sleep(0.02)
                continue
            last_ts = now
            try:
                with self.state_lock:
                    st_worker = self.latest_st
                if st_worker is None:
                    continue
                img = self.camera_provider.grab()
                frame_worker = cv2.resize(img, (CAM_W, CAM_H))
                vw, vh = img.shape[1], img.shape[0]
                vmodel_worker = self.camera_provider.camera_model(
                    st_worker.pos, st_worker.heading, vw, vh,
                    fallback=default_camera(vw, vh),
                    rotation=st_worker.rotation)
                lanes_worker: list = []
                lane_frame_worker = None
                debug_lane: dict = {}
                if not self.args.no_lanes:
                    seg_model_path = self.args.seg_model
                    if seg_model_path is None:
                        try:
                            from beamng_autopilot.vision.segmentation \
                                import default_model_path
                            seg_model_path = default_model_path()
                        except Exception:
                            seg_model_path = None
                    if seg_model_path is not None:
                        if seg_worker is None:
                            from beamng_autopilot.vision.segmentation \
                                import Segmenter
                            try:
                                seg_worker = Segmenter(
                                    model_path=seg_model_path)
                                print(f"[m5] 学习式分割就绪: "
                                      f"{seg_model_path}")
                            except Exception as exc:
                                seg_worker = False
                                print(f"[m5] WARNING: 分割模型加载失败"
                                      f"（回退 CV）: {exc}")
                    if seg_worker:
                        raw_lanes_worker = seg_worker.detect_lines(
                            img, vmodel_worker, st_worker.pos,
                            st_worker.heading,
                            ground_z=(float(st_worker.pos[2])
                                      - config.EGO_ORIGIN_GROUND_GAP_M
                                      if len(st_worker.pos) > 2 else 0.0))
                    else:
                        if lane_det_worker is None:
                            lane_det_worker = LaneDetector()
                        raw_lanes_worker = lane_det_worker.detect(
                            img, vmodel_worker, st_worker.pos,
                            st_worker.heading,
                            ground_z=(float(st_worker.pos[2])
                                      - config.EGO_ORIGIN_GROUND_GAP_M
                                      if len(st_worker.pos) > 2 else 0.0))
                    if lane_smoother_worker is None:
                        lane_smoother_worker = MarkingSmoother()
                    lanes_worker = lane_smoother_worker.update(
                        raw_lanes_worker, vmodel_worker, st_worker.pos,
                        st_worker.heading,
                        ground_z=(float(st_worker.pos[2])
                                  - config.EGO_ORIGIN_GROUND_GAP_M
                                  if len(st_worker.pos) > 2 else 0.0),
                        warmup=True, speed=float(st_worker.speed),
                        now=now)
                    if lanes_worker:
                        try:
                            lane_frame_worker = pair_lane_markings(
                                lanes_worker, st_worker.pos,
                                st_worker.heading, fwd=st_worker.dir,
                                debug=debug_lane)
                        except Exception:
                            lane_frame_worker = None
                lane_debug_path = os.environ.get("M5_LANE_DEBUG_PATH")
                if lane_debug_path and lanes_worker and debug_lane:
                    row = {
                        "t": round(now, 3),
                        "pos": [round(float(v), 2)
                                for v in st_worker.pos[:2]],
                        "heading": round(float(st_worker.heading), 3),
                        "lane_n": len(lanes_worker),
                        "pair_ok": int(lane_frame_worker is not None),
                        "debug": debug_lane,
                    }
                    try:
                        with open(lane_debug_path, "a",
                                  encoding="utf-8") as fh:
                            fh.write(json.dumps(
                                row, ensure_ascii=False) + "\n")
                    except Exception:
                        pass
                if lanes_worker:
                    last_lanes_worker = lanes_worker
                    last_lanes_ts_worker = now
                    last_lanes_pos_worker = tuple(
                        float(x) for x in st_worker.pos[:2])
                    lane_miss_worker = 0
                else:
                    lane_miss_worker += 1
                    lanes_age = now - last_lanes_ts_worker
                    if last_lanes_pos_worker is None:
                        lanes_drive = float("inf")
                    else:
                        lanes_drive = float(np.linalg.norm(
                            np.asarray(st_worker.pos[:2], dtype=float)
                            - np.asarray(last_lanes_pos_worker,
                                         dtype=float)))
                    if lane_reuse_expired(
                            lane_miss_worker, lanes_age, lanes_drive):
                        last_lanes_worker = []
                        last_lanes_ts_worker = 0.0
                        last_lanes_pos_worker = None
                lane_reuse_age = now - last_lane_ts_worker
                if last_lane_pos_worker is None:
                    lane_reuse_drive = float("inf")
                else:
                    lane_reuse_drive = float(np.linalg.norm(
                        np.asarray(st_worker.pos[:2], dtype=float)
                        - np.asarray(last_lane_pos_worker, dtype=float)))
                lane_reused_worker = False
                if lane_frame_worker is not None:
                    last_lane_frame_worker = lane_frame_worker
                    last_lane_ts_worker = now
                    last_lane_pos_worker = tuple(
                        float(x) for x in st_worker.pos[:2])
                elif lane_reuse_expired(
                        lane_miss_worker, lane_reuse_age,
                        lane_reuse_drive):
                    last_lane_frame_worker = None
                    last_lane_ts_worker = 0.0
                    last_lane_pos_worker = None
                    lane_frame_worker = None
                elif last_lane_frame_worker is not None:
                    lane_frame_worker = last_lane_frame_worker
                    lane_reused_worker = True
                else:
                    lane_frame_worker = None
                vis_obs_worker: list = []
                det_boxes_worker: list = []
                if not self.args.no_vision_obstacles:
                    if vision_det_worker is None:
                        from beamng_autopilot.vision.detection \
                                import VisionDetector
                        vision_det_worker = VisionDetector(
                            conf=self.args.vision_conf)
                    vis_obs_worker, det_boxes_worker = \
                        vision_det_worker.detect(
                            img, vmodel_worker, st_worker.pos,
                            st_worker.heading)
                hud_boxes_worker: list = []
                if det_boxes_worker:
                    sx = CAM_W / vw
                    sy = CAM_H / vh
                    hud_boxes_worker = [
                        (int(a * sx), int(b * sy),
                         int(c * sx), int(d * sy), lab, cf)
                        for (a, b, c, d, lab, cf) in det_boxes_worker]
                with self.vision_lock:
                    if self.vision_clear_requested:
                        self.vision_clear_requested = False
                        if lane_smoother_worker is not None:
                            lane_smoother_worker.reset()
                        last_lanes_worker = []
                        last_lanes_ts_worker = 0.0
                        last_lanes_pos_worker = None
                        last_lane_frame_worker = None
                        last_lane_ts_worker = 0.0
                        last_lane_pos_worker = None
                        lane_miss_worker = 0
                        lanes_worker = []
                        lane_frame_worker = None
                    self.vision_snapshot.update({
                        "seq": int(self.vision_snapshot.get("seq", 0)) + 1,
                        "ts": now,
                        "frame": frame_worker,
                        "gen": self.vision_clear_generation,
                        "lanes": last_lanes_worker,
                        "lane_frame": lane_frame_worker,
                        "lane_n": len(lanes_worker),
                        "pair_ok": int(lane_frame_worker is not None),
                        "lane_miss": lane_miss_worker,
                        "lane_reuse": int(lane_reused_worker),
                        "lane_reuse_age": lane_reuse_age,
                        "lane_reuse_drive": lane_reuse_drive,
                        "vis_obs": vis_obs_worker,
                        "det_boxes": det_boxes_worker,
                        "hud_boxes": hud_boxes_worker,
                        "vision_n": len(vis_obs_worker),
                        "failures": 0,
                        "error": None,
                    })
                last_error["vision"] = None
            except Exception as exc:
                err_txt = str(exc)
                with self.vision_lock:
                    self.vision_snapshot["failures"] += 1
                    if "black frame" in err_txt:
                        self.vision_snapshot["black_frames"] = int(
                            self.vision_snapshot.get("black_frames", 0)) + 1
                    self.vision_snapshot["error"] = str(exc)[:120]
                    vfail = int(self.vision_snapshot["failures"])
                if vfail <= 3 or vfail % 20 == 0:
                    print(f"[m5] vision scan error: {exc}")

    def _range_worker(self) -> None:
        last_scan_worker = 0.0
        while not self.quit_flag:
            if not self.autopilot:
                time.sleep(0.05)
                continue
            now = time.time()
            if now - last_scan_worker < RANGE_SCAN_INTERVAL_S:
                time.sleep(0.02)
                continue
            last_scan_worker = now
            try:
                with self.state_lock:
                    st_worker = self.latest_st
                if st_worker is None:
                    continue
                sample = self.range_provider.scan(
                    st_worker.pos, ego_vid=self.conn.vehicle.vid,
                    radius=55.0)
                with self.range_lock:
                    self.range_snapshot.update({
                        "seq": int(self.range_snapshot["seq"]) + 1,
                        "ts": now,
                        "obstacles": list(sample.obstacles),
                        "ray_hits": list(sample.ray_hits),
                        "sensor_ok": not errors_active(),
                        "failures": 0,
                        "error": None,
                    })
            except Exception as exc:
                with self.range_lock:
                    self.range_snapshot["failures"] += 1
                    self.range_snapshot["error"] = str(exc)[:120]
                    rf = int(self.range_snapshot["failures"])
                if rf <= 3 or rf % 20 == 0:
                    print(f"[m5] range scan error: {exc}")

    def _wd_beat_daemon(self, stop_event: threading.Event) -> None:
        logged_since: float | None = None
        while not stop_event.wait(0.5):
            if not self.blocking_active:
                logged_since = None
                continue
            if time.time() - self.blocking_since > 20.0:
                continue
            if logged_since != self.blocking_since:
                logged_since = self.blocking_since
                print("[m5] watchdog daemon beating during blocking phase")
            try:
                with self.conn.io_lock:
                    beat_ok = wd_heartbeat(self.conn)
                if not beat_ok and not stop_event.is_set():
                    with self.conn.io_lock:
                        wd_arm(self.conn)
                    print("[m5] watchdog re-armed by daemon")
            except Exception:
                pass

    def _atexit_safety(self) -> None:
        """Last-ditch safety so a normal/Ctrl+C exit never leaves the car
        latched with throttle/brake or stuck in a mode it did not start in."""
        try:
            self.conn.control(throttle=0.0, brake=0.0, steering=0.0)
            self.conn.step(3)
            if self.gearbox_switched and self.saved_gearbox:
                try:
                    self.conn.control(throttle=0.0, brake=0.0,
                                      steering=0.0, parkingbrake=1.0, gear=0)
                    self.conn.step(5)
                except Exception:
                    pass
                with self.conn.io_lock:
                    gearbox.set_gearbox_mode(self.conn.vehicle,
                                             self.saved_gearbox)
                self.conn.step(3)
            self.conn.control(throttle=0.0, brake=0.0, steering=0.0,
                              parkingbrake=1.0)
            self.conn.step(3)
        except Exception:
            pass
        try:
            with self.conn.io_lock:
                wd_disarm(self.conn)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Connection / sensor setup
    # ------------------------------------------------------------------

    def _setup_connect(self) -> bool:
        """Connect to BeamNG, build sensors, arm watchdog.
        Returns False on failure (callers should return)."""
        try:
            if self.args.attach:
                self.conn.attach_vehicle(vid=self.args.attach_vid)
                print("[m5] attached: press F9 to drive on lane vision, "
                      "or open the big map (M), pick a destination, "
                      "then F10")
            else:
                attached = False
                try:
                    self.conn.open(launch=False)
                    try:
                        self.conn.attach_vehicle(
                            vid=self.args.attach_vid, already_open=True)
                        attached = True
                        print("[m5] attached to your running map: "
                              "press F9 to drive on lane vision, or M "
                              "+ F10 for a nav route")
                    except Exception as exc:
                        print(f"[m5] instance found but no vehicle to "
                              f"attach ({exc}); loading a fresh "
                              "scenario")
                except Exception as exc:
                    if beamng_process_running():
                        print("[m5] BeamNG is running but the "
                              "communication port "
                              f"is closed ({exc}).")
                        print("[m5] Launch the game via "
                              "`python scripts/launch_game.py` "
                              "(adds -tcom -tport 64256) or add those "
                              "flags to the Steam launch options, then "
                              "rerun this script.")
                        raise RuntimeError(
                            "BeamNG running without comms port")
                    print(f"[m5] no running instance ({exc}); "
                          "launching a fresh scenario")
                if not attached:
                    self.conn.open(launch=True)
                    self.conn.load_scenario()
                    print("[m5] scenario loaded: press F9 to drive on "
                          "lane vision, or M + F10 for a nav route")
                    t0 = time.time()
                    while (not self.roadnet.ready
                           and time.time() - t0 < 90.0):
                        with self.conn.io_lock:
                            road_ready = self.roadnet.build(
                                self.conn.bng)
                        if road_ready:
                            print(f"[m5] roadnet ready: "
                                  f"{self.roadnet.info}")
                            break
                        time.sleep(1.0)
                    if self.conn.reposition_on_road(self.roadnet):
                        self.toast("car placed on road network")
            if self.args.front_camera:
                self.conn.set_front_camera()
        except Exception as exc:
            print(f"[m5] connect failed: {exc}")
            self.hotkeys.close()
            self.telemetry.close()
            return False

        # Nav world line
        nav_world_visible = self.conn.read_nav_world_visible()
        want_nav_world = bool(self.args.nav_world)
        if want_nav_world != nav_world_visible:
            ok = self.conn.set_nav_world_visible(
                want_nav_world, persist=not want_nav_world)
            if ok:
                nav_world_visible = want_nav_world
                print(f"[m5] nav world line "
                      f"{'shown' if want_nav_world else 'hidden'} "
                      "(map route stays active)")
            else:
                print("[m5] WARNING: failed to set nav world line to "
                      f"{'visible' if want_nav_world else 'hidden'}")
        print(f"[m5] nav world line visible={nav_world_visible}")

        # Sensor providers
        try:
            self.camera_provider, runtime_mode = build_camera_provider(
                self.conn, self.args.runtime, CAM_W, CAM_H)
        except Exception as exc:
            print(f"[m5] camera provider init failed: {exc}")
            self.hotkeys.close()
            self.telemetry.close()
            self.conn.close()
            return False
        try:
            self.range_provider, _ = build_range_provider(
                self.conn, self.args.runtime)
        except Exception as exc:
            print(f"[m5] range provider init failed: {exc}")
            try:
                self.camera_provider.close()
            except Exception:
                pass
            self.hotkeys.close()
            self.telemetry.close()
            self.conn.close()
            return False
        try:
            st0 = self.conn.get_state()
            self.cam_model = self.camera_provider.camera_model(
                st0.pos, st0.heading, CAM_W, CAM_H,
                fallback=default_camera(CAM_W, CAM_H),
                rotation=st0.rotation)
            print(f"[m5] runtime={runtime_mode}")
        except Exception as exc:
            print(f"[m5] sensor state init failed: {exc}")
            try:
                self.camera_provider.close()
            except Exception:
                pass
            try:
                self.range_provider.close()
            except Exception:
                pass
            self.hotkeys.close()
            self.telemetry.close()
            self.conn.close()
            return False

        # Clear any control inputs a previous instance may have left
        try:
            self.conn.control(throttle=0.0, brake=0.0, steering=0.0)
            self.conn.step(5)
        except Exception:
            pass

        # Arm the game-side input watchdog
        try:
            with self.conn.io_lock:
                wd_armed = wd_arm(self.conn)
            if wd_armed:
                print("[m5] input watchdog armed (stops the car if m5 "
                      "dies)")
            else:
                print("[m5] WARNING: input watchdog failed to arm")
        except Exception as exc:
            print(f"[m5] WARNING: input watchdog error: {exc}")

        return True

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the autopilot session (blocking until quit)."""
        if not _acquire_single_instance("BeamNGAutopilot_M5"):
            print("[m5] another m5 instance is already running; exiting.")
            return

        if not self._setup_connect():
            return

        # Start watchdog beat daemon
        wd_beat_stop = threading.Event()
        wd_beat_thread = threading.Thread(
            target=self._wd_beat_daemon, args=(wd_beat_stop,),
            daemon=True)
        wd_beat_thread.start()

        self.toast("m5 ready: F9 autopilot (lane vision or nav route), "
                   "F8 vision, F12 quit")

        # Query the actual running map / vehicle once
        try:
            self.cur_env = self.conn.current_env()
        except Exception as exc:
            print(f"[m5] env query failed: {exc}")
            self.cur_env = {"map": self.conn.map_name,
                            "vehicle": self.conn.vehicle_model}

        # Warm the YOLO model + CUDA context in the background
        if not self.args.no_vision_obstacles:
            def _prewarm_vision() -> None:
                try:
                    from beamng_autopilot.vision.detection \
                        import VisionDetector
                    VisionDetector(
                        conf=self.args.vision_conf)._ensure_model()
                    print("[m5] vision detector ready (YOLOv8n)")
                except Exception as exc:
                    print(f"[m5] WARNING: vision detector "
                          f"unavailable: {exc}")

            threading.Thread(target=_prewarm_vision,
                             daemon=True).start()

        # Start vision / range worker threads
        if ((not self.args.no_lanes)
                or (not self.args.no_vision_obstacles)):
            threading.Thread(target=self._vision_worker,
                             daemon=True).start()
        if self.range_provider is not None:
            threading.Thread(target=self._range_worker,
                             daemon=True).start()

        # Register atexit safety
        atexit.register(self._atexit_safety)

        # ---- Main loop --------------------------------------------------------
        self._main_loop()

        # ---- Cleanup ----------------------------------------------------------
        self._cleanup(wd_beat_stop, wd_beat_thread)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _main_loop(self) -> None:
        conn = self.conn
        args = self.args
        nav_world_visible = conn.read_nav_world_visible()

        while not self.quit_flag:
            try:
                faulthandler.dump_traceback_later(3.0, exit=False)
                keys = list(self.hotkeys.pump())
                new_cmds, self.ctl_seen = self.ctl.poll(self.ctl_seen)
                keys.extend(new_cmds)
                for item in keys:
                    key, cmd_value = (
                        item if isinstance(item, tuple)
                        else (item, None))
                    if key == "set_speed":
                        try:
                            new_speed = float(cmd_value)
                        except (TypeError, ValueError):
                            new_speed = 0.0
                        if 1.0 <= new_speed <= 60.0:
                            self.cruise_speed = new_speed
                            self.toast(
                                f"cruise speed "
                                f"{new_speed * 3.6:.0f} km/h")
                            print(f"[m5] cruise speed -> "
                                  f"{new_speed:.2f} m/s "
                                  f"({new_speed * 3.6:.0f} km/h)")
                        else:
                            self.toast(
                                "invalid cruise speed (1-60 m/s)")
                    elif key == "quit":
                        self.quit_flag = True
                    elif key == "vision":
                        self.vision = not self.vision
                        self.toast(
                            "vision ON" if self.vision
                            else "vision OFF")
                    elif key == "nav_world":
                        want_visible = bool(
                            float(cmd_value or 0.0) > 0.0)
                        ok = conn.set_nav_world_visible(
                            want_visible, persist=not want_visible)
                        if ok:
                            nav_world_visible = want_visible
                        self.toast(
                            f"nav line "
                            f"{'visible' if want_visible else 'hidden'} "
                            f"({'ok' if ok else 'FAILED'})")
                    elif key == "autopilot":
                        if self.autopilot:
                            self.autopilot = False
                            self.release_control()
                            self.restore_gearbox()
                            self.toast("autopilot OFF")
                            self.finish_session()
                        elif ((self.route is not None
                               and len(self.route) >= 2)
                              or not args.no_lanes):
                            self._engage_autopilot()
                        else:
                            self.toast(
                                "no route or lane vision - grab a "
                                "route with F10 or enable lane "
                                "vision")
                    elif key == "navroute":
                        if self.autopilot:
                            self.toast("stop autopilot first (F9)")
                        else:
                            nav = conn.read_navigation_route()
                            if nav is not None:
                                self.route = (
                                    nav[:, :2]
                                    if nav.ndim == 2
                                    and nav.shape[1] >= 3
                                    else nav)
                                self.toast(
                                    f"navigation route grabbed: "
                                    f"{len(self.route)} pts - "
                                    "press F9")
                            else:
                                self.toast(
                                    "no navigation route - press M "
                                    "in game and pick a destination")
                    elif key == "clear":
                        self.route = None
                        self.cached_drive_route = None
                        self.cached_blocked = False
                        self.last_plan_route = None
                        self.last_plan_rule = None
                        self.toast("route cleared")

                now = time.time()
                # Keep the watchdog heartbeat fresh
                if now - self.last_beat > 0.5:
                    self.last_beat = now
                    try:
                        with conn.io_lock:
                            beat_ok = wd_heartbeat(conn)
                        if (not beat_ok
                                and now - self.last_rearm > 5.0):
                            self.last_rearm = now
                            with conn.io_lock:
                                wd_arm(conn)
                    except Exception:
                        pass
                display_route = self.route

                if self.autopilot:
                    display_route = self._control_tick(
                        now, nav_world_visible)
                else:
                    self._idle_tick(now, nav_world_visible)

                # HUD
                if (self.hud is not None
                        and now - self.last_hud > 0.05):
                    self.last_hud = now
                    try:
                        if not self.autopilot:
                            self.last_st = conn.get_state()
                        cam = None
                        img = None
                        if self.vision and self.last_st is not None:
                            if self.autopilot:
                                with self.vision_lock:
                                    img = self.vision_snapshot.get(
                                        "frame")
                                if img is not None:
                                    img = img.copy()
                            else:
                                img = self.camera_provider.grab()
                                img = cv2.resize(
                                    img, (CAM_W, CAM_H))
                        if img is not None:
                            img = render_camera_overlay(
                                img, display_route,
                                self.last_st.pos,
                                self.last_st.heading,
                                self.cam_model,
                                obstacles=self.obstacles,
                                det_boxes=self.hud_boxes,
                                lane_markings=self.last_lanes)
                            bv = np.full(
                                (CAM_H, CAM_H, 3),
                                (22, 24, 30), np.uint8)
                            start_xy = (
                                display_route[0][:2]
                                if display_route is not None
                                and len(display_route)
                                else None)
                            goal = (
                                display_route[-1][:2]
                                if display_route is not None
                                and len(display_route)
                                else None)
                            render_birdview(
                                bv, route_xy=display_route,
                                obstacles=self.obstacles,
                                waypoints=(
                                    [start_xy]
                                    if start_xy is not None
                                    else None),
                                goal_xy=goal,
                                pos=self.last_st.pos,
                                heading=self.last_st.heading,
                                lane_markings=self.last_lanes)
                            cam = np.hstack([img, bv])
                        data = self.telemetry.latest
                        if (data is None
                                and self.last_st is not None):
                            data = {
                                "t": 0.0,
                                "speed": self.last_st.speed,
                                "throttle": 0.0, "brake": 0.0,
                                "steer": 0.0,
                                "heading": self.last_st.heading}
                        if not self.hud.update(data, cam=cam):
                            self.quit_flag = True
                    except Exception as exc:
                        print(f"[m5] hud error: {exc}")

                # 3D world overlay at ~3 Hz
                if now - self.last_overlay > 0.33:
                    self.last_overlay = now
                    pos3 = (0.0, 0.0, 0.0)
                    if self.last_st is not None:
                        pos3 = tuple(
                            float(v) for v in self.last_st.pos)
                    if self.autopilot:
                        mode = getattr(
                            self.planner, "last_mode", "follow")
                        if mode == "blocked":
                            mode_txt = "BLOCKED-stop"
                        elif mode == "detour":
                            mode_txt = "AVOIDING"
                        elif mode == "deform":
                            mode_txt = "nudging"
                        else:
                            mode_txt = "cruise"
                        sen_txt = ("SENSOR OK" if self.sensor_ok
                                   else "SENSOR FAIL")
                        lane_txt = (
                            f" | lane {self._lane_src} "
                            f"{self._lane_conf:.2f}"
                            if self._lane_src else "")
                        status = (
                            f"AUTOPILOT {mode_txt} | {sen_txt} | "
                            f"obs {len(self.obstacles)} nearest "
                            f"{self.obs_dist:.0f}m | "
                            f"target {self.target_speed:.0f} m/s"
                            + lane_txt)
                    elif (display_route is not None
                          and len(display_route) > 0):
                        status = (
                            f"ROUTE {len(display_route)} pts  "
                            "press F9 to start")
                    else:
                        status = ("press F9 to drive on lane "
                                  "vision")
                    try:
                        start_xy = (
                            display_route[0][:2]
                            if display_route is not None
                            and len(display_route) else None)
                        goal_xy = (
                            display_route[-1][:2]
                            if display_route is not None
                            and len(display_route) else None)
                        with conn.io_lock:
                            self.overlay.update(
                                route_xy=display_route,
                                obstacles=(
                                    self.obstacles
                                    if self.vision else None),
                                waypoints=(
                                    [start_xy]
                                    if start_xy is not None
                                    else None),
                                goal_xy=goal_xy,
                                status_text=status,
                                status_pos=pos3,
                                enabled=(self.vision
                                         and not args.no_overlay),
                                markers=not args.no_markers)
                    except Exception as exc:
                        print(f"[m5] overlay error: {exc}")
            except KeyboardInterrupt:
                print("[m5] interrupted")
                break
            except Exception as exc:
                print(f"[m5] loop error: {exc}")
                time.sleep(0.2)

    # ------------------------------------------------------------------
    # Autopilot engagement
    # ------------------------------------------------------------------

    def _engage_autopilot(self) -> None:
        """Handle F9 press: engage autopilot with gearbox setup."""
        self.autopilot = True
        with self._wd_blocking():
            self.lane_tracker.clear()
            self.last_lane_frame = None
            self.last_lidar_hits = []
            self.saved_gearbox = (
                self.read_gearbox_mode() or "arcade")
            self.set_gearbox_mode("realistic")
            self.gearbox_switched = True
            self.fwd_gear = self.forward_gear_input()
            print(f"[m5] forward gear input = {self.fwd_gear}")
            self.conn.control(
                throttle=0.0, brake=0.0, steering=0.0,
                parkingbrake=0.0, gear=self.fwd_gear)
            self.conn.step(3)
            print("[m5] gearbox switched to realistic for "
                  "autopilot (will be restored on exit)")
            self.session_t0 = time.time()
            self.nearest = 0
            self.speed_ctrl.reset()
            self.obstacles = []
            self.obs_dist = 999.0
            self.vis_tracks = []
            self.vision_conf_n = 0
            self.target_speed = 0.0
            self.prev_steer = 0.0
            self.last_ctrl = time.time()
            self.last_plan_t = 0.0
            self.last_plan_route = None
            self.last_plan_rule = None
            self.cached_drive_route = None
            self.cached_blocked = False
            self.hdg_engaged = False
            self.hist = {
                "t": [], "throttle": [], "brake": [], "speed": []}
            self.last_progress_pos = None
            self.last_progress_t = 0.0
            self.stuck_recover_until = 0.0
            self.stuck_retries = 0
            self.force_end_reason = ""
            self.rev_start_pos = None
            self.rev_dist = 0.0
            self.rev_since = 0.0
            self.rev_hold_until = 0.0
            self.rev_warned = False
            self.rev_guard_logged = False
            self.last_route_arc = None
            self.route_rev_dist = 0.0
            self.last_pos2d = None
            self.last_rev_nearest = 0
            self.road_rule = None
            self.signal_rules = []
            self.selected_signal = None
            self.rule_speed_limit = None
            self.rule_limit = None
            self.rule_reason = None
            self.signal_state = None
            self.signal_action = 0
            self.signal_dist = None
            self.signal_name = ""
            self.last_rule = 0.0
            self.toast("autopilot ON")

    # ------------------------------------------------------------------
    # Single control tick
    # ------------------------------------------------------------------

    def _control_tick(self, now: float,
                      nav_world_visible: bool):
        """Execute one control-loop iteration.
        Returns the display_route."""
        conn = self.conn
        args = self.args
        display_route = self.route
        _ft0 = time.perf_counter()
        _st0 = _ft0
        _stages: dict[str, float] = {}

        def _mark(name: str) -> None:
            nonlocal _st0
            _stages[name] = time.perf_counter() - _st0
            _st0 = time.perf_counter()

        st = conn.get_state()
        with self.state_lock:
            self.latest_st = st
        self.last_st = st
        speed = st.speed
        _mark("get_state")

        # --- Range snapshot ---------------------------------------------------
        with self.range_lock:
            rs_seq = int(self.range_snapshot["seq"])
            rs_new = rs_seq != self.last_range_seq
            if rs_new:
                self.last_range_seq = rs_seq
                self.obstacles = list(
                    self.range_snapshot["obstacles"])
                self.last_lidar_hits = list(
                    self.range_snapshot["ray_hits"])
                self.sensor_ok = bool(
                    self.range_snapshot["sensor_ok"])
                self.scan_failures = int(
                    self.range_snapshot["failures"])
            else:
                self.sensor_ok = bool(
                    self.range_snapshot["sensor_ok"])
                self.scan_failures = int(
                    self.range_snapshot["failures"])
        if rs_new:
            self.obstacles = filter_self_overlap(
                self.obstacles, st.pos,
                categories=("vision", "vehicle", "scenario",
                            "raycast", "lidar"))
            if errors_active():
                self.sensor_ok = False
                self.scan_failures += 1
                if (self.scan_failures <= 3
                        or self.scan_failures % 10 == 0):
                    print(f"[m5] sensor warning: "
                          f"{errors_summary()}")
            else:
                self.sensor_ok = True
                self.scan_failures = 0
            try:
                with conn.io_lock:
                    wd_heartbeat(conn)
            except Exception:
                pass
        _mark("scan")

        # --- Vision snapshot --------------------------------------------------
        snap_new = False
        snap_vis_obs: list = []
        snap_det_boxes: list = []
        snap_hud_boxes: list = []
        snap_lane_frame = None
        snap_lane_n = 0
        snap_pair_ok = 0
        snap_generation = self.last_vision_generation

        with self.vision_lock:
            snap_new = (self.vision_snapshot["seq"]
                        != self.last_vision_seq)
            if snap_new:
                self.last_vision_seq = int(
                    self.vision_snapshot["seq"])
                snap_lanes = list(
                    self.vision_snapshot["lanes"])
                snap_lane_frame = self.vision_snapshot.get(
                    "lane_frame")
                snap_lane_n = int(
                    self.vision_snapshot["lane_n"])
                snap_pair_ok = int(
                    self.vision_snapshot.get("pair_ok", 0))
                snap_vis_obs = list(
                    self.vision_snapshot["vis_obs"])
                snap_det_boxes = list(
                    self.vision_snapshot["det_boxes"])
                snap_hud_boxes = list(
                    self.vision_snapshot["hud_boxes"])
                snap_generation = int(
                    self.vision_snapshot.get("gen", 0))
            else:
                snap_vis_obs = []
                snap_det_boxes = []
                snap_hud_boxes = []
                snap_lane_frame = None
                snap_lane_n = 0
                snap_pair_ok = 0
                snap_generation = self.last_vision_generation
        if snap_new:
            if snap_generation >= self.last_vision_generation:
                self.last_lanes = snap_lanes
                self.last_lane_frame = snap_lane_frame
                self.lane_n = snap_lane_n
            else:
                self.last_lanes = []
                self.last_lane_frame = None
                self.lane_n = 0
            self.last_vision_generation = max(
                self.last_vision_generation, snap_generation)
            self.vision_n = len(snap_vis_obs)
            self.det_boxes = snap_det_boxes
            self.hud_boxes = snap_hud_boxes
            vis_obs = snap_vis_obs
            if vis_obs:
                anchors = []
                if (self.route is not None
                        and len(self.route) > 0):
                    near_i = int(np.argmin(np.linalg.norm(
                        self.route[:, :2] - st.pos[:2],
                        axis=1)))
                    for idx in (0, near_i,
                                len(self.route) - 1):
                        anchors.append(
                            self.route[idx][:2])
                vis_obs = drop_vision_waypoint_ghosts(
                    vis_obs, anchors)
            self.vis_tracks, confirmed_vis_obs = \
                update_vision_tracks(
                    self.vis_tracks, vis_obs, st.pos[:2],
                    now, match_m=VIS_TRACK_MATCH_M,
                    confirm_hits=VIS_TRACK_CONFIRM,
                    ttl_s=VIS_TRACK_TTL_S,
                    ego_gate_m=VIS_TRACK_EGO_GATE_M,
                    ride_along_ratio=VIS_TRACK_RIDE_RATIO)
        else:
            self.det_boxes = []
            self.hud_boxes = []
            confirmed_vis_obs = []
        self.vision_conf_n = len(confirmed_vis_obs)
        if confirmed_vis_obs:
            self.obstacles = merge_obstacles(
                self.obstacles + confirmed_vis_obs)
        _mark("vision")

        # --- Lane centre -----------------------------------------------------
        lane_src = ""
        lane_conf = 0.0
        lane_span = 0.0
        lidar_conf = 0.0
        lane_lat = None
        lane_w = 0.0
        vision_lane = self.lane_tracker.update(
            self.last_lane_frame, st.pos, st.heading,
            fwd=st.dir)
        lidar_dbg: dict = {}
        lidar_frame = build_lidar_corridor(
            self.last_lidar_hits, st.pos, st.heading,
            fwd=st.dir, debug=lidar_dbg)
        lane_frame = choose_sensor_lane(
            vision_lane, lidar_frame, st.pos, st.heading,
            fwd=st.dir, state=self.lane_fusion_state)
        if lane_frame is not None:
            lane_src = ("|".join(lane_frame.sources)
                        if lane_frame.sources else "sensor")
            lane_conf = float(lane_frame.confidence)
            lane_span = float(lane_frame.span_m)
            lane_w = float(lane_frame.width)
            if "lidar" in lane_src.split("|"):
                lidar_conf = lane_conf
            fwd2 = np.asarray(st.dir[:2], dtype=float)
            fn = float(np.linalg.norm(fwd2))
            if fn > 1e-9:
                fwd2 = fwd2 / fn
            else:
                fwd2 = np.array([math.cos(st.heading),
                                 math.sin(st.heading)])
            left2 = np.array([-fwd2[1], fwd2[0]])
            c0 = np.asarray(
                lane_frame.center[0], dtype=float)[:2]
            lane_lat = float(
                (c0 - np.asarray(st.pos[:2], dtype=float))
                @ left2)

        # --- ACC lead vehicle -------------------------------------------------
        lead_vehicle, lead_dist, _lead_lat = find_lead_vehicle(
            self.obstacles,
            (self.route
             if (self.route is not None
                 and len(self.route) >= 2)
             else None),
            st.pos, heading=st.heading)
        if (lead_vehicle is not None
                and lead_vehicle.velocity is not None):
            lead_speed = vehicle_along_speed(
                lead_vehicle,
                (self.route
                 if (self.route is not None
                     and len(self.route) >= 2)
                 else None),
                st.pos, heading=st.heading)
        else:
            lead_speed = 0.0
        follow_active = lead_vehicle is not None

        oncoming = oncoming_vehicle_ahead(
            self.obstacles,
            (self.route
             if (self.route is not None
                 and len(self.route) >= 2)
             else None),
            st.pos, heading=st.heading)
        ovk = OvertakeStateMachine()
        ovk_state = ovk.update(
            time.time(), lead_vehicle is not None,
            lead_speed, lead_dist, self.cruise_speed,
            speed, oncoming=oncoming,
            solid_left=solid_marking_left(
                self.last_lanes, st.pos, st.heading))
        overtake_requested = ovk_state == "active"
        desired_speed = self.cruise_speed
        blocked = False
        hdg_dev = 0.0
        hdg_guard = False
        plan_ran = False
        drive_route = None
        has_route = (self.route is not None
                     and len(self.route) > 0)
        has_lane = (lane_frame is not None
                    and lane_frame_usable(lane_frame))

        if has_route or has_lane:
            dt = max(1e-3, now - self.last_ctrl)
            self.last_ctrl = now
            plan_route = self.route if has_route else None
            plan_nearest = self.nearest if has_route else 0
            if has_route:
                self.nearest = nearest_route_point(
                    self.route, st.pos, st.dir)
                plan_nearest = self.nearest
            drive_route = self.cached_drive_route
            blocked = self.cached_blocked
            last_cross_solid = not oncoming
            if (drive_route is None
                    or self.route is not self.last_plan_route
                    or (self.road_rule is not self.last_plan_rule)
                    or ((not oncoming) != last_cross_solid)
                    or (now - self.last_plan_t
                        >= PLAN_INTERVAL_S)):
                drive_route, blocked = self.planner.plan(
                    plan_route, self.obstacles, st.pos,
                    st.heading, plan_nearest,
                    solid_lines=self.last_lanes,
                    sensor_lane=lane_frame,
                    road_rule=self.road_rule,
                    cross_solid=not oncoming)
                plan_ran = True
                drive_route = np.asarray(
                    drive_route, dtype=float)
                self.cached_drive_route = drive_route
                self.cached_blocked = blocked
                self.last_plan_route = plan_route
                self.last_plan_rule = self.road_rule
                self.last_plan_t = time.time()
            if blocked:
                blk = getattr(
                    self.planner, "last_blocker", None)
                desc = ("unknown obstacle" if blk is None
                        else f"{blk[0]} @ {blk[1]:.1f}m")
                if now - self._last_blk_log > 2.5:
                    self._last_blk_log = now
                    print(f"[m5] BLOCKED by {desc} "
                          "(no drivable way; stopping)")
            if len(drive_route) >= 2:
                d0 = np.linalg.norm(
                    drive_route[:, :2] - st.pos[:2],
                    axis=1)
                start_i = int(np.argmin(d0))
                if (start_i > 0
                        and len(drive_route) - start_i >= 2):
                    drive_route = drive_route[start_i:]
            if len(drive_route) >= 2:
                display_route = drive_route
                speed_route = (
                    self.route[:, :2]
                    if self.route is not None
                    and len(self.route) >= 2
                    else drive_route)
                speed_nearest = nearest_route_point(
                    speed_route, st.pos, st.dir)
                ahead_idx = adaptive_lookahead_idx(
                    float(speed),
                    spacing_m=max(
                        0.5,
                        _route_spacing_m(speed_route)))
                self.planner.last_sharp = (
                    corner_angle_deg(
                        speed_route, speed_nearest,
                        ahead_idx=ahead_idx)
                    >= self.planner.sharp_angle_deg)
                desired_speed, self.obs_dist = \
                    self.planner.speed(
                        speed_route, self.obstacles,
                        st.pos, st.heading,
                        speed_nearest, self.cruise_speed,
                        ahead_idx=ahead_idx)
                corner_v = float(getattr(
                    self.planner, "last_corner",
                    desired_speed))
                if corner_v < self.cruise_speed * 0.9:
                    self.corner_hold_until = (
                        time.time() + 2.5)
                    self.corner_held_speed = min(
                        corner_v,
                        (self.corner_held_speed
                         if self.corner_held_speed
                         is not None else corner_v))
                if ((self.corner_held_speed is not None
                        and time.time()
                        < self.corner_hold_until
                        and speed
                        > self.corner_held_speed + 1.0)):
                    desired_speed = min(
                        desired_speed,
                        self.corner_held_speed)
                if ((self.corner_held_speed is not None
                        and time.time()
                        >= self.corner_hold_until
                        and corner_v
                        >= self.cruise_speed * 0.9)):
                    self.corner_held_speed = None
                if blocked:
                    if ((lead_vehicle is not None
                            and (lead_vehicle.velocity
                                 is not None)
                            and lead_speed > 0.0)):
                        desired_speed = min(
                            desired_speed,
                            follow_speed(
                                self.cruise_speed,
                                lead_dist, lead_speed,
                                speed))
                    else:
                        desired_speed = 0.0
                elif (lead_vehicle is not None
                        and not overtake_requested):
                    desired_speed = min(
                        desired_speed,
                        follow_speed(
                            self.cruise_speed,
                            lead_dist, lead_speed,
                            speed))
                corner_v = getattr(
                    self.planner, "last_corner",
                    desired_speed)
                obs_lim = getattr(
                    self.planner, "last_obs_lim", None)
                self.pp.lookahead = (
                    self.pp.adaptive_lookahead(speed))
                steer_rad, _, _ = self.pp.steering(
                    st.pos, st.heading, drive_route, 0)
                new_steer = float(np.clip(
                    -steer_rad / 0.6, -1.0, 1.0))
                v_sq = max(speed * speed, 2.0)
                steer_cap = max(
                    0.10,
                    min(1.0, 5.0 * 2.9 / v_sq / 0.6))
                new_steer = float(np.clip(
                    new_steer, -steer_cap, steer_cap))
                steer = smooth_steer(
                    self.prev_steer, new_steer, dt)
                self.prev_steer = steer
                steer_angle = abs(steer) * 0.6
                steer_capped = False
                if ((steer_angle > 0.09
                        and corner_v
                        < self.cruise_speed * 0.85)):
                    steer_radius = (
                        2.9 / math.tan(steer_angle))
                    capped = float(
                        math.sqrt(7.0 * steer_radius))
                    if capped < desired_speed:
                        desired_speed = capped
                        steer_capped = True
            else:
                steer, desired_speed = 0.0, 0.0
                corner_v, obs_lim = 0.0, None
                steer_angle, steer_capped = 0.0, False
                self.obs_dist = 999.0
                self.prev_steer = 0.0
        else:
            steer, desired_speed = 0.0, 0.0
            corner_v, obs_lim = 0.0, None
            steer_angle, steer_capped = 0.0, False
            self.obs_dist = 999.0
            self.prev_steer = 0.0
            dt = max(1e-3, now - self.last_ctrl)
            self.last_ctrl = now

        _mark("plan")

        # --- Traffic rules ----------------------------------------------------
        if now - self.last_rule > RULE_POLL_INTERVAL_S:
            self.last_rule = now
            self.road_rule = conn.read_current_road_rule(
                st.pos, st.dir)
            self.signal_rules = []
            self.selected_signal = None
            self.rule_speed_limit = None
            self.rule_limit = None
            self.rule_reason = None
            self.signal_state = None
            self.signal_action = 0
            self.signal_dist = None
            self.signal_name = ""
            if self.road_rule is not None:
                self.rule_speed_limit = (
                    self.road_rule.speed_limit_mps)
                if (self.road_rule.n1
                        and self.road_rule.n2):
                    try:
                        self.signal_rules = \
                            conn.read_signal_snapshot(
                                conn.vehicle.vid,
                                self.road_rule.n1,
                                self.road_rule.n2)
                    except Exception:
                        self.signal_rules = []
                self.selected_signal = select_signal_rule(
                    self.signal_rules, st.pos,
                    heading=st.heading, dir_vec=st.dir)

        # --- ACC creep --------------------------------------------------------
        desired_speed, creep, self.creep_since = \
            creep_speed(
                blocked, obs_lim, desired_speed, speed,
                self.creep_since, now, CREEP_MPS)

        # --- Legal caps -------------------------------------------------------
        desired_speed, self.rule_reason, self.rule_limit = \
            apply_rule_speed(
                desired_speed, self.road_rule,
                self.selected_signal)
        if (signal_requires_stop(self.selected_signal)
                and desired_speed <= 0.01):
            creep = 0
            self.creep_since = None
        self.signal_state = (
            self.selected_signal.state
            if self.selected_signal is not None else None)
        self.signal_action = (
            self.selected_signal.action
            if self.selected_signal is not None else 0)
        self.signal_dist = signal_distance(
            self.selected_signal)
        self.signal_name = (
            self.selected_signal.name
            if self.selected_signal is not None else "")
        _mark("rule")

        # --- Spin guard -------------------------------------------------------
        hdg_dev = heading_deviation_deg(
            self.route, self.nearest, st.dir)
        if (self.route is None
                and drive_route is not None
                and len(drive_route) >= 2):
            hdg_dev = heading_deviation_deg(
                drive_route, 0, st.dir)
        hdg_cap = heading_dev_speed_cap(
            hdg_dev, self.hdg_engaged)
        hdg_guard = hdg_cap is not None
        self.hdg_engaged = hdg_guard
        if hdg_guard:
            desired_speed = min(desired_speed, hdg_cap)

        # --- Reverse guard ----------------------------------------------------
        signed = 0.0
        if st.vel is not None and st.dir is not None:
            signed = float(np.dot(
                np.asarray(st.vel[:2], dtype=float),
                np.asarray(st.dir[:2], dtype=float)))
        rev_pos = np.asarray(st.pos[:2], dtype=float)
        route_arc = 0.0
        route_back = 0.0
        if (self.route is not None
                and len(self.route) >= 2):
            try:
                route_arc, _ = _point_route_pos(
                    st.pos[0], st.pos[1], self.route)
            except Exception:
                route_arc = self.last_route_arc or 0.0
            if self.last_route_arc is not None:
                arc_delta = (self.last_route_arc
                             - route_arc)
                if arc_delta > 0.03 and arc_delta < 3.0:
                    route_back = arc_delta
                elif arc_delta >= 3.0:
                    self.last_route_arc = route_arc
            if (self.last_pos2d is not None
                    and self.last_rev_nearest > 0):
                disp = rev_pos - self.last_pos2d
                disp_norm = float(
                    np.linalg.norm(disp))
                if disp_norm > REVERSE_MOVEMENT_M:
                    i0 = max(0, min(
                        len(self.route) - 1,
                        self.last_rev_nearest - 1))
                    i1 = max(0, min(
                        len(self.route) - 1,
                        self.last_rev_nearest + 1))
                    tv = (np.asarray(
                        self.route[i1][:2], dtype=float)
                        - np.asarray(
                            self.route[i0][:2],
                            dtype=float))
                    tn = float(np.linalg.norm(tv))
                    if tn > 1e-6:
                        back_share = max(0.0, float(
                            -(disp @ (tv / tn))
                            / disp_norm))
                        if back_share >= REVERSE_BACK_RATIO:
                            route_back = max(
                                route_back, disp_norm)
        route_reverse = (
            route_back > 0.12 and signed < 0.3)
        reverse_now = (
            signed < REVERSE_SPEED_MPS or route_reverse)
        rev_sustained = False
        if reverse_now:
            if self.rev_since <= 0.0:
                self.rev_since = now
                self.rev_start_pos = rev_pos
                self.rev_dist = 0.0
                self.route_rev_dist = 0.0
                self.rev_guard_logged = False
                self.speed_ctrl.reset()
            else:
                self.rev_dist = max(
                    self.rev_dist,
                    float(np.linalg.norm(
                        rev_pos - self.rev_start_pos)))
            if route_reverse:
                self.route_rev_dist += route_back
            rev_total = max(
                self.rev_dist, self.route_rev_dist)
            rev_sustained = (
                (now - self.rev_since)
                >= REVERSE_ENGAGE_S)
            if rev_sustained:
                if not self.rev_guard_logged:
                    self.rev_guard_logged = True
                    rev_gear = self.read_gear()
                    print(f"[m5] REVERSE-GUARD engaged: "
                          f"signed={signed:.2f} m/s "
                          f"route={route_back:.2f}m "
                          f"back={rev_total:.2f}m "
                          f"gear={rev_gear}")
                    with self.vision_lock:
                        self.vision_clear_requested = True
                        self.vision_clear_generation += 1
                    self.last_lanes = []
                    self.lane_n = 0
                    self.obstacles = [
                        ob for ob in self.obstacles
                        if not is_sparse_raycast_speck(ob)]
                    self.cached_drive_route = None
                    self.cached_blocked = False
                    print("[m5] REVERSE-GUARD cleared "
                          "lane/sensor ghosts; will replan")
                self.rev_hold_until = (
                    now + REVERSE_HOLD_S)
                if ((rev_total > REVERSE_STOP_DIST_M)
                        and not self.rev_warned):
                    self.rev_warned = True
                    print(f"[m5] REVERSE {rev_total:.1f}m; "
                          "braking to a stop")
                    self.force_end_reason = "reverse stuck"
        else:
            self.rev_since = 0.0
            self.rev_start_pos = None
            self.rev_dist = 0.0
            self.route_rev_dist = 0.0
            self.rev_warned = False
            self.rev_guard_logged = False
        self.last_route_arc = route_arc
        self.last_pos2d = rev_pos.copy()
        self.last_rev_nearest = self.nearest
        rev_total = max(
            self.rev_dist, self.route_rev_dist)
        reverse_hold = now < self.rev_hold_until
        if reverse_now or reverse_hold:
            desired_speed = 0.0
        _mark("reverse")

        # --- Speed ramp -------------------------------------------------------
        if desired_speed > self.target_speed:
            self.target_speed = min(
                desired_speed,
                self.target_speed + RAMP_ACCEL * dt)
        else:
            self.target_speed = max(
                desired_speed,
                self.target_speed - RAMP_DECEL * dt)
        _mark("ramp")

        if now - self.last_wspd > 0.1:
            self.last_wspd = now
            self.wheel_speed = conn.get_wheel_speed()
            if (self.wheel_speed is None
                    and not self.wspd_warned):
                self.wspd_warned = True
                print("[m5] wheel-speed unavailable; "
                      "slip guard inactive")
        throttle, brake = self.speed_ctrl.update(
            self.target_speed, speed,
            dt=min(0.25, max(0.01, dt)),
            wheel_speed=self.wheel_speed)
        if ((speed < 6.0 and hdg_dev > 5.0)
                and not reverse_now
                and not reverse_hold):
            throttle = min(throttle, 0.35)
        if reverse_now or reverse_hold:
            throttle = 0.0
            if rev_sustained or reverse_hold:
                brake = max(brake, 0.85)
            else:
                brake = max(brake, REVERSE_SOFT_BRAKE)
        slip = bool(self.speed_ctrl.slip_active)
        _mark("speed")

        # --- Telemetry --------------------------------------------------------
        t = now - self.session_t0
        self.hist["t"].append(t)
        self.hist["throttle"].append(float(throttle))
        self.hist["brake"].append(float(brake))
        self.hist["speed"].append(float(speed))

        mode = getattr(self.planner, "last_mode", "follow")
        sen = "OK" if self.sensor_ok else "FAIL"
        goal_dist = 0.0
        if (self.route is not None
                and len(self.route) > 0):
            goal_dist = float(np.linalg.norm(
                self.route[-1][:2] - st.pos[:2]))

        # No-progress recovery
        progress_pos = np.asarray(
            st.pos[:2], dtype=float)
        if (self.route is not None
                and len(self.route) >= 2):
            ri0 = max(0, self.nearest - 1)
            ri1 = min(
                len(self.route) - 1,
                self.nearest + 1)
            tv = (np.asarray(
                self.route[ri1][:2], dtype=float)
                - np.asarray(
                    self.route[ri0][:2], dtype=float))
            tn = float(np.linalg.norm(tv))
            fwd_dir = (
                tv / tn if tn > 1e-6
                else np.asarray(st.dir[:2], dtype=float))
        else:
            fwd_dir = np.asarray(
                st.dir[:2], dtype=float)
        if reverse_now:
            self.last_progress_pos = progress_pos
            self.last_progress_t = now
        if ((self.target_speed < 1.0
                or goal_dist <= 12.0
                or signed >= 0.5)):
            self.last_progress_pos = progress_pos
            self.last_progress_t = now
        elif self.last_progress_pos is None:
            self.last_progress_pos = progress_pos
            self.last_progress_t = now
        else:
            moved_fwd = float(
                (progress_pos - self.last_progress_pos)
                @ fwd_dir)
            if moved_fwd >= 0.6:
                self.last_progress_pos = progress_pos
                self.last_progress_t = now
            elif now - self.last_progress_t > 2.5:
                if now < self.stuck_recover_until:
                    desired_speed = min(
                        desired_speed, 0.5)
                    self.target_speed = min(
                        self.target_speed, 0.5)
                else:
                    self.stuck_retries += 1
                    if self.stuck_retries > 2:
                        print("[m5] STUCK (retry failed; "
                              "stopping)")
                        self.force_end_reason = "stuck"
                        desired_speed = 0.0
                        self.target_speed = 0.0
                    else:
                        self.stuck_recover_until = (
                            now + 4.0)
                        self.last_progress_pos = (
                            progress_pos)
                        self.last_progress_t = now
                        self.last_lanes = []
                        self.obstacles = [
                            ob for ob in self.obstacles
                            if not is_sparse_raycast_speck(ob)]
                        desired_speed = min(
                            desired_speed, 0.5)
                        self.target_speed = min(
                            self.target_speed, 0.5)
                        brake = max(brake, 0.15)
                        print("[m5] STUCK (no forward "
                              "progress); clearing "
                              "lane/sensor ghosts, "
                              "retrying")

        boxes = [
            [round(o.x, 1), round(o.y, 1),
             round(o.half_w, 1), round(o.half_h, 1),
             o.label or o.category,
             (None if o.axis is None else [
                 round(float(o.axis[0]), 3),
                 round(float(o.axis[1]), 3)]),
             round(o.half_len, 2),
             round(o.half_thick, 2)]
            for o in self.obstacles[:12]]
        rte_pts: list = []
        if (self.route is not None
                and len(self.route) > 0):
            rstep = max(1, len(self.route) // 200)
            rte_pts = [
                [round(float(p[0]), 1),
                 round(float(p[1]), 1)]
                for p in self.route[::rstep]]
        if now - self.last_roads > 1.0:
            self.last_roads = now
            self.road_polys = self.roads_json(st.pos)
        if now - self.last_env > 10.0:
            self.last_env = now
            try:
                self.cur_env = conn.current_env()
            except Exception:
                pass

        self._lane_src = lane_src
        self._lane_conf = lane_conf

        self.telemetry.publish(
            t=t, speed=float(speed),
            throttle=throttle, brake=brake,
            steer=steer, vel=st.vel,
            dir_vec=st.dir, up_vec=st.up,
            pos=st.pos, heading=float(st.heading),
            nearest=int(self.nearest),
            extra={
                "auto": 1,
                "nav_world": int(nav_world_visible),
                "mode": mode,
                "cruise": round(
                    float(self.cruise_speed), 1),
                "speed_limit": (
                    None
                    if self.rule_speed_limit is None
                    else round(
                        float(self.rule_speed_limit), 1)),
                "rule_limit": (
                    None
                    if self.rule_limit is None
                    else round(
                        float(self.rule_limit), 1)),
                "rule_reason": self.rule_reason,
                "signal_state": self.signal_state,
                "signal_action": self.signal_action,
                "signal_dist": (
                    None
                    if self.signal_dist is None
                    else round(
                        float(self.signal_dist), 1)),
                "signal_name": self.signal_name,
                "target": round(
                    float(self.target_speed), 1),
                "desired": round(
                    float(desired_speed), 1),
                "corner": round(float(corner_v), 1),
                "obslim": (
                    None if obs_lim is None
                    else round(float(obs_lim), 1)),
                "creep": int(creep),
                "slip": int(slip),
                "hdg_dev": round(float(hdg_dev), 1),
                "hdg_g": int(hdg_guard),
                "steer_rad": round(
                    float(steer_angle), 3),
                "sen": sen,
                "vis": int(self.vision_n),
                "vconf": int(self.vision_conf_n),
                "black_frames": int(self.black_frames),
                "lanes": len(self.last_lanes),
                "pair_ok": snap_pair_ok,
                "tracker_ok": int(
                    lane_frame is not None),
                "lane_src": lane_src,
                "lane_paired": int(bool(getattr(
                    lane_frame, "paired", False))),
                "lane_override": (
                    None
                    if getattr(
                        self.planner,
                        "last_lane_override",
                        None) is None
                    else list(
                        self.planner.last_lane_override)),
                "lane_jump": int(bool(getattr(
                    self.lane_tracker,
                    "last_rejected", False))),
                "lane_conf": round(lane_conf, 2),
                "lane_span": round(lane_span, 1),
                "lidar_conf": round(lidar_conf, 2),
                "lidar_hits": len(
                    self.last_lidar_hits),
                "lidar_obs": sum(
                    1 for o in self.obstacles
                    if o.category == "lidar"),
                "lidar_dbg": {
                    k: (round(float(v), 2)
                        if isinstance(v, (int, float))
                        else v)
                    for k, v in lidar_dbg.items()},
                "lane_lat": (
                    None if lane_lat is None
                    else round(lane_lat, 2)),
                "edge_lat": (
                    (None
                     if (edge_lat := _boundary_near_lat(
                        lane_frame.left
                        if getattr(lane_frame, "right",
                                   None) is None
                        else lane_frame.right,
                        st.pos, st.heading,
                        fwd=st.dir)) is None
                     else round(edge_lat, 2))
                    if (lane_frame is not None
                        and (getattr(lane_frame,
                                     "left", None)
                             is not None
                             or getattr(lane_frame,
                                        "right", None)
                             is not None))
                    else None),
                "lane_w": round(lane_w, 1),
                "sharp": int(getattr(
                    self.planner, "last_sharp",
                    False)),
                "plan_mode": getattr(
                    self.planner, "last_lane_mode",
                    "nav"),
                "plan_offset": round(float(getattr(
                    self.planner,
                    "last_lane_offset", 0.0)), 2),
                "obs": len(self.obstacles),
                "obs_d": round(
                    float(self.obs_dist), 1),
                "lead_d": (
                    round(float(lead_dist), 1)
                    if lead_vehicle is not None
                    else None),
                "lead_v": round(
                    float(lead_speed), 1),
                "follow": int(follow_active),
                "overtake": int(overtake_requested),
                "ovk": ovk_state,
                "goal_d": round(goal_dist, 1),
                "rev": round(rev_total, 2),
                "rev_route": round(
                    self.route_rev_dist, 2),
                "rev_g": int(reverse_now),
                "rev_s": (
                    round(float(now - self.rev_since), 2)
                    if reverse_now else 0.0),
                "rev_h": int(reverse_hold),
                "route": (0 if self.route is None
                          else len(self.route)),
                "blk": (
                    (f"{self.planner.last_blocker[0]}"
                     f"@{self.planner.last_blocker[1]:.0f}m")
                    if getattr(self.planner,
                               "last_blocker", None)
                    else ""),
                "boxes": boxes,
                "rte": rte_pts,
                "env": self.cur_env,
                "roads": self.road_polys,
                "markings": self.markings_json(
                    self.last_lanes),
            },
        )
        _mark("telemetry")

        # Hold brake at standstill while blocked
        if blocked and speed < 0.5:
            brake = max(brake, 0.12)
        conn.control(
            throttle=throttle, steering=steer,
            brake=brake, gear=self.fwd_gear)
        _mark("control")

        if now - self.last_status > 1.0:
            self.last_status = now
            mode = getattr(
                self.planner, "last_mode", "follow")
            sen = ("OK" if self.sensor_ok else "FAIL")
            gear_txt = self.read_gear() or "?"
            print(
                f"[m5] mode={mode} sen={sen} "
                f"obs={len(self.obstacles)} "
                f"vis={self.vision_n}/{self.vision_conf_n} "
                f"nearest={self.obs_dist:.0f}m "
                f"v={speed:.1f} "
                f"target={self.target_speed:.1f} "
                f"desired={desired_speed:.1f} "
                f"corner={corner_v:.1f} "
                f"rule={(self.rule_reason or '-')} "
                f"sig={signal_action_label(self.signal_action)} "
                f"lanes={len(self.last_lanes)} "
                f"lane={lane_src or '-'} {lane_conf:.2f} "
                f"sharp={int(getattr(self.planner, 'last_sharp', False))} "
                f"cap={int(steer_capped)} "
                f"obslim={(obs_lim if obs_lim is None else round(obs_lim, 1))} "
                f"creep={creep} slip={int(slip)} "
                f"hdg={hdg_dev:.0f} "
                f"gear={gear_txt} "
                f"rev={rev_total:.2f}/{int(reverse_now)} "
                f"throttle={throttle:.2f} "
                f"brake={brake:.2f}")

        # --- End conditions ---------------------------------------------------
        ended, reason = False, ""
        if self.force_end_reason:
            ended, reason = True, self.force_end_reason
        elif goal_dist > 0 and goal_dist < GOAL_RADIUS_M:
            ended, reason = True, "goal reached"
        elif t > args.max_run:
            ended, reason = True, "timeout"

        if ended:
            faulthandler.cancel_dump_traceback_later()
            self.autopilot = False
            conn.control(
                throttle=0.0, brake=1.0, steering=0.0,
                gear=self.fwd_gear)
            conn.step(20)
            self.release_control()
            self.restore_gearbox()
            self.toast(f"autopilot ended: {reason}")
            try:
                self.telemetry.publish(
                    t=t, speed=float(speed),
                    throttle=0.0, brake=1.0,
                    steer=0.0, vel=st.vel,
                    dir_vec=st.dir, up_vec=st.up,
                    pos=st.pos,
                    heading=float(st.heading),
                    nearest=int(self.nearest),
                    extra={
                        "auto": 0, "mode": "ENDED",
                        "sen": sen,
                        "nav_world": int(
                            nav_world_visible),
                        "target": 0.0,
                        "cruise": round(
                            float(self.cruise_speed), 1),
                        "speed_limit": (
                            None
                            if self.rule_speed_limit
                            is None
                            else round(float(
                                self.rule_speed_limit),
                                1)),
                        "rule_limit": (
                            None
                            if self.rule_limit is None
                            else round(float(
                                self.rule_limit), 1)),
                        "rule_reason": self.rule_reason,
                        "signal_state": self.signal_state,
                        "signal_action": self.signal_action,
                        "signal_dist": (
                            None
                            if self.signal_dist is None
                            else round(float(
                                self.signal_dist), 1)),
                        "signal_name": self.signal_name,
                        "vis": int(self.vision_n),
                        "vconf": int(
                            self.vision_conf_n),
                        "lanes": len(self.last_lanes),
                        "sharp": int(getattr(
                            self.planner,
                            "last_sharp", False)),
                        "obs": len(self.obstacles),
                        "obs_d": 999.0,
                        "goal_d": round(goal_dist, 1),
                        "route": 0, "blk": "",
                        "slip": int(slip),
                        "hdg_dev": round(
                            float(hdg_dev), 1),
                        "hdg_g": int(hdg_guard),
                        "boxes": [], "rte": [],
                        "env": self.cur_env,
                        "roads": self.road_polys,
                        "markings": self.markings_json(
                            self.last_lanes),
                        "ended": 1, "reason": reason,
                    },
                )
            except Exception:
                pass
            self.finish_session()
        else:
            conn.step(1, wait=False)
            _stages["step"] = (
                time.perf_counter() - _st0)
            if time.perf_counter() - _ft0 > 0.35:
                detail = " ".join(
                    f"{k}={v*1000:.0f}ms"
                    for k, v in _stages.items())
                if (plan_ran
                        and self.planner.last_plan_stages):
                    detail += " | plan_stages=" + " ".join(
                        f"{k}={v:.0f}ms"
                        for k, v in
                        self.planner.last_plan_stages
                        .items())
                print(f"[m5] SLOW-FRAME {detail}")

        return display_route

    # ------------------------------------------------------------------
    # Idle tick
    # ------------------------------------------------------------------

    def _idle_tick(self, now: float,
                   nav_world_visible: bool) -> None:
        conn = self.conn
        if now - self.last_idle > 0.5:
            self.last_idle = now
            try:
                st_idle = conn.get_state()
                self.last_st = st_idle
                if now - self.last_roads > 1.0:
                    self.last_roads = now
                    self.road_polys = self.roads_json(
                        st_idle.pos)
                if now - self.last_env > 10.0:
                    self.last_env = now
                    try:
                        self.cur_env = conn.current_env()
                    except Exception:
                        pass
                self.telemetry.publish(
                    t=0.0, speed=float(st_idle.speed),
                    throttle=0.0, brake=0.0, steer=0.0,
                    vel=st_idle.vel,
                    dir_vec=st_idle.dir,
                    up_vec=st_idle.up,
                    pos=st_idle.pos,
                    heading=float(st_idle.heading),
                    extra={
                        "auto": 0, "mode": "IDLE",
                        "sen": "OK",
                        "nav_world": int(
                            nav_world_visible),
                        "target": 0.0,
                        "cruise": round(
                            float(self.cruise_speed), 1),
                        "vis": 0, "obs": 0,
                        "obs_d": 999.0, "goal_d": 0.0,
                        "route": (0
                                  if self.route is None
                                  else len(self.route)),
                        "boxes": [], "rte": [],
                        "env": self.cur_env,
                        "roads": self.road_polys,
                        "markings": [],
                    },
                )
            except Exception:
                pass
        time.sleep(0.02)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _cleanup(self, wd_beat_stop: threading.Event,
                 wd_beat_thread: threading.Thread) -> None:
        conn = self.conn
        if self.autopilot:
            try:
                self.finish_session(show_chart=False)
            except Exception as exc:
                print(f"[m5] quit session summary failed: {exc}")
            try:
                conn.control(
                    throttle=0.0, brake=1.0,
                    steering=0.0, gear=self.fwd_gear)
                conn.step(20)
                self.release_control()
                self.restore_gearbox()
            except Exception:
                pass
        try:
            with conn.io_lock:
                self.overlay.close()
        except Exception:
            pass
        try:
            faulthandler.cancel_dump_traceback_later()
        except Exception:
            pass
        wd_beat_stop.set()
        wd_beat_thread.join(timeout=2.0)
        self.ctl.clear()
        self.hotkeys.close()
        self.telemetry.close()
        if self.hud is not None:
            self.hud.close()
        try:
            self.camera_provider.close()
        except Exception:
            pass
        try:
            self.range_provider.close()
        except Exception:
            pass
        try:
            with conn.io_lock:
                wd_disarm(conn)
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass
        print("[m5] bye")
