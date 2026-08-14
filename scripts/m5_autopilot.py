"""M5 autopilot assistant: manual in-game activation, Tesla-style vision
overlay, and post-drive throttle/brake/speed bar charts.

Workflow (recommended: you are already in a map with a vehicle in BeamNG):
    1. Run the script. It first probes for your running session and attaches
       to your current map/vehicle automatically (no --attach needed). If the
       game is not running at all, it launches one and loads a fresh scenario.
    2. Press F9 to toggle autopilot ON/OFF. Without a nav route the car
       keeps the sensor lane centre from camera lane lines / LiDAR walls.
       In the game open the big map (M), pick a destination, then press
       F10 to add the navigation route. F11 clears the route.
    3. F8 toggles the Tesla-style vision overlay (3D world route + front
       camera projection + bird view). When a session ends, a 3-panel
       throttle/brake/speed bar chart pops up.

Hotkeys (global, work while the game has focus):
    F8   vision overlay on/off
    F9   autopilot on/off
    F10  grab the in-game navigation route (set a destination on map M first)
    F11  clear route
    F12  quit
"""

from __future__ import annotations

import argparse
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
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
)
from beamng_autopilot.perception import (
    filter_self_overlap,
)
from beamng_autopilot.planner import (
    RIGHT_OFFSET_M,
    SHARP_ANGLE_DEG,
    SHARP_CORNER_KPH,
    LocalPlanner,
    _point_route_pos,
    creep_speed,
    is_sparse_raycast_speck,
)
from beamng_autopilot.roadnet import RoadNetwork
from beamng_autopilot.traffic import (
    RoadRuleView,
    SignalRule,
    apply_rule_speed,
    select_signal_rule,
    signal_action_label,
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

CAM_W, CAM_H = 1076, 806
GOAL_RADIUS_M = 8.0
# Target speed is ramped toward the planner's demand so the pedals never
# slam: throttle/brake changes stay linear instead of step-like.
RAMP_ACCEL = 2.5    # m/s^2 max rate the target speed may rise
RAMP_DECEL = 3.5    # m/s^2 max rate the target speed may drop
STEER_SMOOTH = 0.35  # low-pass coefficient for the steering command
STEER_RATE = 1.2     # max normalized steering change per second.  The
                     # control loop can be blocked by a ~1 Hz sensor scan,
                     # so a per-frame low-pass lag turns a bend into a wall.
CREEP_MPS = 1.5     # ACC-style creep speed when an obstacle limit pins 0
RULE_POLL_INTERVAL_S = 0.8   # road-rule/signal snapshot cadence
HEADING_DEV_DEG = 18.0   # heading vs route direction at which we slow down
HEADING_DEV_RELEASE = 13.0  # hysteresis: release the cap only below this
HEADING_DEV_CAP = 2.5    # m/s speed cap right at the threshold
HEADING_DEV_CRAWL = 0.6  # m/s floor when the car is fully sideways/spinning
REVERSE_SPEED_MPS = -0.3  # signed forward speed below which we are rolling back
REVERSE_ENGAGE_S = 0.30   # sustained backward motion before the hard guard
REVERSE_HOLD_S = 1.0      # keep the speed demand at zero after reverse
REVERSE_STOP_DIST_M = 1.0  # backward travel that triggers a hard stop warning
REVERSE_SOFT_BRAKE = 0.35  # transient-backward brake (no long hold)
REVERSE_MOVEMENT_M = 0.10  # backward displacement that counts as reversing
REVERSE_BACK_RATIO = 0.30  # minimum backward share of the frame displacement
PLAN_INTERVAL_S = 0.15     # minimum wall time between full local re-plans
VIS_TRACK_MATCH_M = 1.8    # world-space gate for matching a vision detection
VIS_TRACK_CONFIRM = 2      # scans before a vision obstacle may act as a blocker
VIS_TRACK_TTL_S = 8.0      # forget a vision track after this many seconds
VIS_TRACK_EGO_GATE_M = 0.8  # ego motion needed before a track can confirm
VIS_TRACK_RIDE_RATIO = 0.6  # obstacle/ego drift ratio treated as a phantom
VIS_LANE_REUSE_MISS = 6     # consecutive lane misses before a lane expires
VIS_LANE_REUSE_TTL_S = 1.0  # a reused lane may not be older than this
VIS_LANE_REUSE_EGO_M = 6.0  # ego travel that expires a reused lane


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


def smooth_steer(prev: float, new: float, dt: float,
                 rate: float = STEER_RATE) -> float:
    """Rate-limited steering smoothing (normalized input units)."""
    max_step = float(rate * max(1e-3, dt))
    return prev + float(np.clip(new - prev, -max_step, max_step))


def heading_dev_speed_cap(dev_deg: float,
                          engaged: bool = False) -> float | None:
    """Speed cap (m/s) once the car points away from the route direction;
    None inside the dead zone.  Caps 2.5 m/s at 18 deg down to a 0.6 m/s
    crawl at 90 deg and beyond, so a drifting car slows before it can
    reach the outside wall and a spinning car cannot dig in.

    ``engaged`` adds hysteresis: once the guard has tripped it stays on
    until the heading drops below HEADING_DEV_RELEASE, so a heading that
    wobbles around the threshold cannot flick the throttle on/off."""
    limit = HEADING_DEV_RELEASE if engaged else HEADING_DEV_DEG
    if dev_deg <= limit:
        return None
    t = min(1.0, max(0.0, (90.0 - dev_deg) / (90.0 - HEADING_DEV_DEG)))
    return HEADING_DEV_CRAWL + (HEADING_DEV_CAP - HEADING_DEV_CRAWL) * t


def beamng_process_running() -> bool:
    """True when a BeamNG game process exists (even without the comms port)."""
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


def main() -> None:
    if not _acquire_single_instance("BeamNGAutopilot_M5"):
        print("[m5] another m5 instance is already running; exiting.")
        return
    ap = argparse.ArgumentParser(description="M5 in-game autopilot assistant")
    ap.add_argument("--map", default=config.DEFAULT_MAP)
    ap.add_argument("--vehicle", default=config.DEFAULT_VEHICLE)
    ap.add_argument("--attach", action="store_true",
                    help="attach to the vehicle in a running BeamNG session")
    ap.add_argument("--attach-vid", default=None,
                    help="vehicle id to attach to (default: first active)")
    ap.add_argument("--speed", type=float, default=20.0,
                    help="cruise speed in m/s")
    ap.add_argument("--port", type=int, default=config.PORT)
    ap.add_argument("--runtime", choices=("auto", "steam", "tech"),
                    default=config.RUNTIME_MODE,
                    help="game runtime: auto detects Steam/tech after connect")
    ap.add_argument("--max-run", type=float, default=600.0,
                    help="max seconds for one autopilot session")
    ap.add_argument("--no-hud", action="store_true",
                    help="disable the live telemetry HUD window")
    ap.add_argument("--no-show", action="store_true",
                    help="save the telemetry chart to PNG without showing it")
    ap.add_argument("--front-camera", action="store_true",
                    help="switch the in-game camera to a fixed view ahead of "
                         "the car (default: leave the game camera and UI alone)")
    ap.add_argument("--no-vision-obstacles", action="store_true",
                    help="disable YOLO front-camera obstacle detection "
                         "(keeps raycast + vehicle sources)")
    ap.add_argument("--no-lanes", action="store_true",
                    help="disable front-camera lane-marking detection "
                         "(keeps raycast + vehicle sources)")
    ap.add_argument("--no-markers", action="store_true",
                    help="do not draw the yellow start / red goal spheres "
                         "in the game world")
    ap.add_argument("--nav-world", type=int, choices=(0, 1), default=0,
                    help="in-world nav line: 0 hides it (default) while the "
                         "route stays on the map; 1 shows arrows/ground "
                         "markers in the world")
    ap.add_argument("--no-overlay", action="store_true",
                    help="disable the 3D world overlay (route/obstacles)")
    ap.add_argument("--vision-conf", type=float, default=0.35,
                    help="YOLO confidence threshold for obstacle boxes")
    ap.add_argument("--vision-rate", type=float, default=3.0,
                    help="vision obstacle scan rate in Hz (default 3)")
    ap.add_argument("--right-offset", type=float, default=RIGHT_OFFSET_M,
                    help="optional right-hand offset from the nav route "
                         "(m, default 0 = follow the route centre)")
    ap.add_argument("--sharp-angle", type=float, default=SHARP_ANGLE_DEG,
                    help="corner angle threshold for the speed cap (deg)")
    ap.add_argument("--sharp-corner-kph", type=float,
                    default=SHARP_CORNER_KPH,
                    help="max speed through a sharp corner (km/h)")
    args = ap.parse_args()

    # Cruise speed is mutable at runtime via the launcher's set_speed command.
    cruise_speed = args.speed

    conn = BeamNGConnector(
        args.map, args.vehicle, port=args.port,
        home=config.runtime_home(args.runtime))
    vision_lock = threading.Lock()
    state_lock = threading.Lock()
    latest_st = None
    vision_snapshot: dict = {
        "seq": 0,
        "ts": 0.0,
        "frame": None,
        "lanes": [],
        "lane_frame": None,
        "lane_n": 0,
        "lane_miss": 0,
        "lane_reuse": 0,
        "lane_reuse_age": 0.0,
        "lane_reuse_drive": 0.0,
        "vis_obs": [],
        "det_boxes": [],
        "hud_boxes": [],
        "vision_n": 0,
        "failures": 0,
        "black_frames": 0,
        "error": None,
    }
    # F12 may be taken by another app; provide Ctrl+Q as a fallback so the
    # script can always be quit from inside the game.  Esc is deliberately
    # NOT bound: in-game menus use it and it must never kill the autopilot.
    hotkeys = HotkeyListener(
        bindings={
            VK_F8: "vision",
            VK_F9: "autopilot",
            VK_F10: "navroute",
            VK_F11: "clear",
            VK_F12: "quit",
        },
        modifier_alternates={(MOD_CONTROL, VK_Q): "quit"},
    )
    ctl = ControlBridge()
    ctl_seen = ctl.current_seq()
    roadnet = RoadNetwork()
    telemetry = TelemetryBroadcaster()
    overlay = WorldOverlay(conn.bng)
    hud = None if args.no_hud else LiveHUD(show_camera=True)
    cam_model = default_camera(CAM_W, CAM_H)
    camera_provider = None
    range_provider = None
    pp = PurePursuit(lookahead=6.0)
    planner = LocalPlanner(
        right_offset=args.right_offset,
        sharp_angle_deg=args.sharp_angle,
        sharp_corner_kph=args.sharp_corner_kph,
    )
    speed_ctrl = SpeedController()

    route: np.ndarray | None = None
    autopilot = False
    vision = True
    quit_flag = False
    fwd_gear = 2  # refreshed whenever autopilot engages
    saved_gearbox: str | None = None
    gearbox_switched = False
    session_t0 = 0.0
    hist: dict[str, list] = {"t": [], "throttle": [], "brake": [], "speed": []}

    last_overlay = 0.0
    last_hud = 0.0
    last_scan = 0.0
    last_wspd = 0.0
    wheel_speed: float | None = None
    wspd_warned = False
    last_status = 0.0
    last_idle = 0.0
    last_ctrl = time.time()
    last_plan_t = 0.0
    last_plan_route: np.ndarray | None = None
    last_plan_rule: RoadRuleView | None = None
    cached_drive_route: np.ndarray | None = None
    cached_blocked = False
    last_vision = 0.0
    last_vision_seq = 0
    vision_det = None
    lane_det = None
    last_lanes: list = []
    last_lane_frame = None
    last_lidar_hits: list = []
    lane_tracker = LaneTracker()
    lane_fusion_state: dict = {}
    vision_clear_requested = False
    vision_clear_generation = 0
    last_vision_generation = 0
    lane_miss = 0
    lane_n = 0
    vision_n = 0
    vision_failures = 0
    black_frames = 0
    vis_tracks: list[VisionTrack] = []
    vision_conf_n = 0
    det_boxes: list = []
    hud_boxes: list = []
    nearest = 0
    last_st = None
    obstacles: list = []
    obs_dist = 999.0
    sensor_ok = True
    scan_failures = 0
    last_beat = 0.0
    last_rearm = 0.0
    target_speed = 0.0
    prev_steer = 0.0
    creep_since: float | None = None
    hdg_engaged = False
    cur_env: dict = {}
    last_env = 0.0
    road_polys: list = []
    last_roads = 0.0
    road_rule: RoadRuleView | None = None
    signal_rules: list[SignalRule] = []
    selected_signal: SignalRule | None = None
    rule_speed_limit: float | None = None
    rule_limit: float | None = None
    rule_reason: str | None = None
    signal_state: str | None = None
    signal_action: int = 0
    signal_dist: float | None = None
    signal_name: str = ""
    last_rule = 0.0
    last_progress_pos: np.ndarray | None = None
    last_progress_t = 0.0
    stuck_recover_until = 0.0
    stuck_retries = 0
    force_end_reason = ""
    rev_start_pos: np.ndarray | None = None
    rev_dist = 0.0
    rev_since = 0.0
    rev_hold_until = 0.0
    rev_warned = False
    rev_guard_logged = False
    last_route_arc: float | None = None
    route_rev_dist = 0.0
    last_pos2d: np.ndarray | None = None
    last_rev_nearest = 0

    # The main loop normally keeps the game-side watchdog heartbeat fresh.
    # During deliberate multi-second blocking phases (F9 gearbox setup, the
    # handover sequence) the main thread cannot beat, so the watchdog daemon
    # below beats instead while these flags say such a phase is active.
    blocking_active = False
    blocking_since = 0.0

    @contextmanager
    def _wd_blocking() -> Iterator[None]:
        nonlocal blocking_active, blocking_since
        blocking_active = True
        blocking_since = time.time()
        try:
            yield
        finally:
            blocking_active = False

    def toast(msg: str) -> None:
        print(f"[m5] {msg}")
        try:
            with conn.io_lock:
                conn.bng.display_gui_message(msg)
        except Exception:
            pass

    def roads_json(xy) -> list:
        """Nearby road centre lines as JSON-safe, downsampled polylines."""
        try:
            polys = roadnet.nearby_polylines(xy, 90.0)
            out = []
            for pl in polys:
                step = max(1, int(len(pl) // 80))
                out.append([[round(float(x), 1), round(float(y), 1)]
                            for x, y in pl[::step]])
            return out[:16]
        except Exception:
            return []

    def markings_json(lanes) -> list:
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

    def release_control() -> None:
        """Hand the vehicle back to the player without leaving it rolling.

        Delegates to the shared handover sequence: force realistic, brake to
        a full standstill at any speed, probe and engage D/1st, then restore
        the player's gearbox while fully stopped with the parking brake
        engaged and zero pedals (the arcade box then sits in N and only
        engages D when the player presses throttle - it never latches R and
        the car never creeps or rolls on its own).
        """
        nonlocal gearbox_switched, blocking_active, blocking_since
        with _wd_blocking():
            handover_vehicle(conn, saved_gearbox, gearbox_switched)
        gearbox_switched = False

    def read_gearbox_mode() -> str | None:
        """Return the current gearbox mode name or None when unavailable."""
        with conn.io_lock:
            return gearbox.read_gearbox_mode(conn.vehicle)

    def set_gearbox_mode(mode: str) -> None:
        with conn.io_lock:
            gearbox.set_gearbox_mode(conn.vehicle, mode)

    def read_gear() -> str | None:
        with conn.io_lock:
            return gearbox.read_gear(conn)

    def forward_gear_input() -> int:
        # Each gearbox helper takes conn.io_lock per socket call, so an
        # outer lock here would leave the watchdog daemon unable to beat
        # while this multi-second brake/probe sequence runs.
        with _wd_blocking():
            return gearbox.forward_gear_input(conn)

    def restore_gearbox() -> None:
        """No-op kept for call-site compatibility.

        handover_vehicle() already restored the player's gearbox while the
        car was still rolling forward, so nothing is left to do here.
        """
        return
    def finish_session(show_chart: bool = True) -> None:
        nonlocal hist
        if hist["t"]:
            ts = time.strftime("%Y%m%d_%H%M%S")
            p = config.LOGS_DIR / "telemetry" / (
                f"m5_telemetry_{ts}.png")
            plot_telemetry(hist, p, block=False,
                           show=show_chart and not args.no_show)
            print(f"[m5] telemetry chart saved -> {p}")
            # Publish a small JSON summary so the GUI launcher can show the
            # chart + stats the moment a session ends (it watches the file's
            # mtime and re-renders the PNG directly in the window).
            try:
                t_arr = np.asarray(hist["t"], dtype=float)
                spd = np.asarray(hist["speed"], dtype=float)
                thr = np.asarray(hist["throttle"], dtype=float)
                brk = np.asarray(hist["brake"], dtype=float)
                summary = {
                    "ts": ts,
                    "png": str(p),
                    "duration": round(float(t_arr[-1]), 1) if len(t_arr) else 0.0,
                    "max_speed": round(float(spd.max()), 2) if len(spd) else 0.0,
                    "avg_speed": round(float(spd.mean()), 2) if len(spd) else 0.0,
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
        hist = {"t": [], "throttle": [], "brake": [], "speed": []}

    try:
        if args.attach:
            conn.attach_vehicle(vid=args.attach_vid)
            print("[m5] attached: press F9 to drive on lane vision, or open "
                  "the big map (M), pick a destination, then F10")
        else:
            # Default behavior: prefer a session the user already opened.
            # Probe without launching; only load our own scenario when no
            # instance is running at all, so the user's map is never replaced.
            attached = False
            try:
                conn.open(launch=False)  # raises if no instance is running
                try:
                    conn.attach_vehicle(vid=args.attach_vid, already_open=True)
                    attached = True
                    print("[m5] attached to your running map: "
                          "press F9 to drive on lane vision, or M + F10 "
                          "for a nav route")
                except Exception as exc:
                    print(f"[m5] instance found but no vehicle to attach "
                          f"({exc}); loading a fresh scenario")
            except Exception as exc:
                if beamng_process_running():
                    # The user launched the game manually (Steam etc.) which
                    # does not open the beamngpy comms port by default. Do not
                    # silently start a second instance over their map.
                    print("[m5] BeamNG is running but the communication port "
                          f"is closed ({exc}).")
                    print("[m5] Launch the game via "
                          "`python scripts/launch_game.py` (adds -tcom "
                          "-tport 64256) or add those flags to the Steam "
                          "launch options, then rerun this script.")
                    raise RuntimeError("BeamNG running without comms port")
                print(f"[m5] no running instance ({exc}); "
                      "launching a fresh scenario")
            if not attached:
                conn.open(launch=True)
                conn.load_scenario()
                print("[m5] scenario loaded: press F9 to drive on lane "
                      "vision, or M + F10 for a nav route")
                # Pull the car onto the nearest road node so it never spawns
                # below terrain on maps whose origin is not on a road.
                t0 = time.time()
                while not roadnet.ready and time.time() - t0 < 90.0:
                    with conn.io_lock:
                        road_ready = roadnet.build(conn.bng)
                    if road_ready:
                        print(f"[m5] roadnet ready: {roadnet.info}")
                        break
                    time.sleep(1.0)
                if conn.reposition_on_road(roadnet):
                    toast("car placed on road network")
        if args.front_camera:
            conn.set_front_camera()
    except Exception as exc:
        print(f"[m5] connect failed: {exc}")
        hotkeys.close()
        telemetry.close()
        return

    nav_world_visible = conn.read_nav_world_visible()
    want_nav_world = bool(args.nav_world)
    if want_nav_world != nav_world_visible:
        # Hide by default and persist that choice; a temporary "show" is
        # deliberately not persisted so later sessions stay clean.
        ok = conn.set_nav_world_visible(
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

    # Build runtime sensor providers only after the vehicle exists: Steam
    # keeps the Lua screen/raycast path, Tech lazily opens Camera/LiDAR.
    try:
        camera_provider, runtime_mode = build_camera_provider(
            conn, args.runtime, CAM_W, CAM_H)
    except Exception as exc:
        print(f"[m5] camera provider init failed: {exc}")
        hotkeys.close()
        telemetry.close()
        conn.close()
        return
    try:
        range_provider, _ = build_range_provider(conn, args.runtime)
    except Exception as exc:
        print(f"[m5] range provider init failed: {exc}")
        try:
            camera_provider.close()
        except Exception:
            pass
        hotkeys.close()
        telemetry.close()
        conn.close()
        return
    try:
        st0 = conn.get_state()
        cam_model = camera_provider.camera_model(
            st0.pos, st0.heading, CAM_W, CAM_H,
            fallback=default_camera(CAM_W, CAM_H))
        print(f"[m5] runtime={runtime_mode}")
    except Exception as exc:
        print(f"[m5] sensor state init failed: {exc}")
        try:
            camera_provider.close()
        except Exception:
            pass
        try:
            range_provider.close()
        except Exception:
            pass
        hotkeys.close()
        telemetry.close()
        conn.close()
        return

    # Clear any control inputs a previous (killed) instance may have left
    # latched on the vehicle - BeamNG keeps applying the last client inputs
    # even after the client disconnects, which is how the car keeps driving
    # / reversing on its own.  Zeroing here also heals a car that is already
    # running away when this script is launched.
    try:
        conn.control(throttle=0.0, brake=0.0, steering=0.0)
        conn.step(5)
    except Exception:
        pass

    # Arm the game-side input watchdog: if this process dies or is killed
    # (no Python cleanup can run), the watchdog takes over after ~2.5 s -
    # zeroing throttle/steering, braking while rolling and parking the car -
    # so a dead autopilot can never leave the car driving / reversing by
    # itself.  Arming also heals a watchdog already engaged by a previously
    # killed run.  On a normal exit we disarm it again so the player can
    # drive manually without the watchdog pulling the handbrake.
    try:
        with conn.io_lock:
            wd_armed = wd_arm(conn)
        if wd_armed:
            print("[m5] input watchdog armed (stops the car if m5 dies)")
        else:
            print("[m5] WARNING: input watchdog failed to arm")
    except Exception as exc:
        print(f"[m5] WARNING: input watchdog error: {exc}")

    # The main loop's heartbeat cannot run while it is synchronously blocked
    # inside a deliberate multi-second operation (F9 gearbox setup, handover).
    # This daemon covers exactly those phases - it beats only while
    # _wd_blocking() is active - so the game-side watchdog no longer engages
    # mid-setup, while a main-loop hang outside a blocking phase still trips
    # the watchdog as before.  A hard-killed process kills this thread too.
    wd_beat_stop = threading.Event()

    def _wd_beat_daemon() -> None:
        nonlocal blocking_active, blocking_since
        logged_since: float | None = None
        while not wd_beat_stop.wait(0.5):
            if not blocking_active:
                logged_since = None
                continue
            if time.time() - blocking_since > 20.0:
                # Never mask a genuinely stuck main thread indefinitely.
                continue
            if logged_since != blocking_since:
                logged_since = blocking_since
                print("[m5] watchdog daemon beating during blocking phase")
            try:
                with conn.io_lock:
                    beat_ok = wd_heartbeat(conn)
                if not beat_ok and not wd_beat_stop.is_set():
                    with conn.io_lock:
                        wd_arm(conn)
                    print("[m5] watchdog re-armed by daemon")
            except Exception:
                pass

    wd_beat_thread = threading.Thread(target=_wd_beat_daemon, daemon=True)
    wd_beat_thread.start()

    toast("m5 ready: F9 autopilot (lane vision or nav route), "
          "F8 vision, F12 quit")

    # Query the actual running map / vehicle once so the GUI settings can
    # auto-fetch them (attach mode always uses the user's own map).
    try:
        cur_env = conn.current_env()
    except Exception as exc:
        print(f"[m5] env query failed: {exc}")
        cur_env = {"map": conn.map_name, "vehicle": conn.vehicle_model}

    # Warm the YOLO model + CUDA context in the background so the first
    # autopilot scan never stalls the control loop on a multi-second load.
    if not args.no_vision_obstacles:
        def _prewarm_vision() -> None:
            try:
                from beamng_autopilot.vision.detection import VisionDetector

                VisionDetector(conf=args.vision_conf)._ensure_model()
                print("[m5] vision detector ready (YOLOv8n)")
            except Exception as exc:
                print(f"[m5] WARNING: vision detector unavailable: {exc}")

        threading.Thread(target=_prewarm_vision, daemon=True).start()

    # YOLO and lane detection run in a worker thread.  The control loop only
    # consumes the latest finished snapshot, so a ~3.5 s camera scan can no
    # longer stall steering / throttle updates or let the car overshoot.
    if (not args.no_lanes) or (not args.no_vision_obstacles):
        def _vision_worker() -> None:
            nonlocal vision_clear_requested, vision_clear_generation
            lane_det_worker = None
            lane_smoother_worker = None
            vision_det_worker = None
            last_lanes_worker: list = []
            last_lanes_ts_worker = 0.0
            last_lanes_pos_worker = None
            last_lane_frame_worker = None
            last_lane_ts_worker = 0.0
            last_lane_pos_worker = None
            lane_miss_worker = 0
            last_ts = 0.0
            while not quit_flag:
                if not autopilot:
                    time.sleep(0.05)
                    continue
                now = time.time()
                rate = max(1.0, args.vision_rate)
                if now - last_ts < 1.0 / rate:
                    time.sleep(0.02)
                    continue
                last_ts = now
                try:
                    with state_lock:
                        st_worker = latest_st
                    if st_worker is None:
                        continue
                    img = camera_provider.grab()
                    frame_worker = cv2.resize(img, (CAM_W, CAM_H))
                    vw, vh = img.shape[1], img.shape[0]
                    vmodel_worker = camera_provider.camera_model(
                        st_worker.pos, st_worker.heading, vw, vh,
                        fallback=default_camera(vw, vh))
                    lanes_worker: list = []
                    lane_frame_worker = None
                    debug_lane: dict = {}
                    if not args.no_lanes:
                        if lane_det_worker is None:
                            lane_det_worker = LaneDetector()
                        if lane_smoother_worker is None:
                            lane_smoother_worker = MarkingSmoother()
                        raw_lanes_worker = lane_det_worker.detect(
                            img, vmodel_worker, st_worker.pos,
                            st_worker.heading,
                            ground_z=(float(st_worker.pos[2])
                                      if len(st_worker.pos) > 2 else 0.0))
                        lanes_worker = lane_smoother_worker.update(
                            raw_lanes_worker, vmodel_worker, st_worker.pos,
                            st_worker.heading,
                            ground_z=(float(st_worker.pos[2])
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
                    # Keep the last usable lane reference across brief
                    # misses/weak detections so lane tracking does not
                    # reset to None every other vision scan.
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
                    if not args.no_vision_obstacles:
                        if vision_det_worker is None:
                            from beamng_autopilot.vision.detection \
                                    import VisionDetector

                            vision_det_worker = VisionDetector(
                                conf=args.vision_conf)
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
                    with vision_lock:
                        if vision_clear_requested:
                            vision_clear_requested = False
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
                        vision_snapshot.update({
                            "seq": int(vision_snapshot.get("seq", 0)) + 1,
                            "ts": now,
                            "frame": frame_worker,
                            "gen": vision_clear_generation,
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
                    with vision_lock:
                        vision_snapshot["failures"] += 1
                        if "black frame" in err_txt:
                            vision_snapshot["black_frames"] = int(
                                vision_snapshot.get("black_frames", 0)) + 1
                        vision_snapshot["error"] = str(exc)[:120]
                        vfail = int(vision_snapshot["failures"])
                    if vfail <= 3 or vfail % 20 == 0:
                        print(f"[m5] vision scan error: {exc}")

        threading.Thread(target=_vision_worker, daemon=True).start()

    def _atexit_safety() -> None:
        """Last-ditch safety so a normal/Ctrl+C exit never leaves the car
        latched with throttle/brake or stuck in a mode it did not start in:
        zero the pedals, restore the player's gearbox and engage the parking
        brake.  A hard kill (task manager) cannot run this - use
        scripts/m5_emergency_stop.py for that case."""
        try:
            conn.control(throttle=0.0, brake=0.0, steering=0.0)
            conn.step(3)
            if gearbox_switched and saved_gearbox:
                # Shift to N first: handing realistic D back into arcade
                # leaves the arcade box in D and the car creeps forward.
                try:
                    conn.control(throttle=0.0, brake=0.0, steering=0.0,
                                 parkingbrake=1.0, gear=0)
                    conn.step(5)
                except Exception:
                    pass
                with conn.io_lock:
                    gearbox.set_gearbox_mode(conn.vehicle, saved_gearbox)
                conn.step(3)
            conn.control(throttle=0.0, brake=0.0, steering=0.0,
                         parkingbrake=1.0)
            conn.step(3)
        except Exception:
            pass
        # Normal-exit safety net (also covers early exits where the socket
        # is still open): the car is parked and pedals zeroed, so disarm the
        # watchdog - otherwise it keeps pulling the handbrake when the
        # player takes over manually.  A hard kill cannot run this.
        try:
            with conn.io_lock:
                wd_disarm(conn)
        except Exception:
            pass

    import atexit
    atexit.register(_atexit_safety)

    while not quit_flag:
        try:
            # If the loop ever stalls inside a blocking BeamNG call - also
            # during F9 gearbox setup, which happens before the control-loop
            # heartbeat - dump the main-thread stack after 3 s so the exact
            # call is visible in stderr instead of the run just going quiet.
            faulthandler.dump_traceback_later(3.0, exit=False)
            keys = list(hotkeys.pump())
            new_cmds, ctl_seen = ctl.poll(ctl_seen)
            keys.extend(new_cmds)
            for item in keys:
                key, cmd_value = (
                    item if isinstance(item, tuple) else (item, None))
                if key == "set_speed":
                    try:
                        new_speed = float(cmd_value)
                    except (TypeError, ValueError):
                        new_speed = 0.0
                    if 1.0 <= new_speed <= 60.0:
                        cruise_speed = new_speed
                        toast(f"cruise speed {new_speed * 3.6:.0f} km/h")
                        print(f"[m5] cruise speed -> {new_speed:.2f} m/s "
                              f"({new_speed * 3.6:.0f} km/h)")
                    else:
                        toast("invalid cruise speed (1-60 m/s)")
                elif key == "quit":
                    quit_flag = True
                elif key == "vision":
                    vision = not vision
                    toast("vision ON" if vision else "vision OFF")
                elif key == "nav_world":
                    want_visible = bool(float(cmd_value or 0.0) > 0.0)
                    ok = conn.set_nav_world_visible(
                        want_visible, persist=not want_visible)
                    if ok:
                        nav_world_visible = want_visible
                    toast(f"nav line {'visible' if want_visible else 'hidden'} "
                          f"({'ok' if ok else 'FAILED'})")
                elif key == "autopilot":
                    if autopilot:
                        autopilot = False
                        release_control()
                        restore_gearbox()
                        toast("autopilot OFF")
                        finish_session()
                    elif (route is not None and len(route) >= 2
                          or not args.no_lanes):
                        autopilot = True
                        # Gearbox setup can take several seconds (brake to a
                        # standstill + gear probes); keep the watchdog alive
                        # for the duration instead of letting it engage and
                        # kill the just-started session.
                        with _wd_blocking():
                            lane_tracker.clear()
                            last_lane_frame = None
                            last_lidar_hits = []
                            # ALWAYS run the control loop in realistic mode:
                            # in arcade, brake-at-standstill is a reverse
                            # request and drives the car backward by itself.
                            # Save the player's mode (arcade fallback when
                            # unreadable) so the handover can restore it at a
                            # standstill.
                            saved_gearbox = read_gearbox_mode() or "arcade"
                            set_gearbox_mode("realistic")
                            gearbox_switched = True
                            # Engage a real forward gear (D/1st) before
                            # driving.  forward_gear_input() brakes to a
                            # standstill in realistic mode (where braking
                            # never latches R), probes the gearbox and leaves
                            # it in D/1st with the parking brake on.  This
                            # also fixes autopilot start while the player is
                            # reversing: arcade leaves R, and without this
                            # the autopilot would keep reversing.
                            fwd_gear = forward_gear_input()
                            print(f"[m5] forward gear input = {fwd_gear}")
                            # Release the parking brake so a parked car can
                            # actually drive off.
                            conn.control(throttle=0.0, brake=0.0,
                                         steering=0.0, parkingbrake=0.0,
                                         gear=fwd_gear)
                            conn.step(3)
                            print("[m5] gearbox switched to realistic for "
                                  "autopilot (will be restored on exit)")
                            session_t0 = time.time()
                            nearest = 0
                            speed_ctrl.reset()
                            obstacles = []
                            obs_dist = 999.0
                            vis_tracks = []
                            vision_conf_n = 0
                            last_blk_log = 0.0
                            target_speed = 0.0
                            prev_steer = 0.0
                            last_ctrl = time.time()
                            last_plan_t = 0.0
                            last_plan_route = None
                            last_plan_rule = None
                            cached_drive_route = None
                            cached_blocked = False
                            hdg_engaged = False
                            hist = {"t": [], "throttle": [], "brake": [],
                                    "speed": []}
                            last_progress_pos = None
                            last_progress_t = 0.0
                            stuck_recover_until = 0.0
                            stuck_retries = 0
                            force_end_reason = ""
                            rev_start_pos = None
                            rev_dist = 0.0
                            rev_since = 0.0
                            rev_hold_until = 0.0
                            rev_warned = False
                            rev_guard_logged = False
                            last_route_arc = None
                            route_rev_dist = 0.0
                            last_pos2d = None
                            last_rev_nearest = 0
                            road_rule = None
                            signal_rules = []
                            selected_signal = None
                            rule_speed_limit = None
                            rule_limit = None
                            rule_reason = None
                            signal_state = None
                            signal_action = 0
                            signal_dist = None
                            signal_name = ""
                            last_rule = 0.0
                            toast("autopilot ON")
                    else:
                        toast("no route or lane vision - grab a route "
                              "with F10 or enable lane vision")
                elif key == "navroute":
                    if autopilot:
                        toast("stop autopilot first (F9)")
                    else:
                        nav = conn.read_navigation_route()
                        if nav is not None:
                            route = nav[:, :2] if nav.ndim == 2 \
                                and nav.shape[1] >= 3 else nav
                            toast(f"navigation route grabbed: "
                                  f"{len(route)} pts - press F9")
                        else:
                            toast("no navigation route - press M in game and "
                                  "pick a destination")
                elif key == "clear":
                    route = None
                    cached_drive_route = None
                    cached_blocked = False
                    last_plan_route = None
                    last_plan_rule = None
                    toast("route cleared")

            now = time.time()
            # Keep the watchdog heartbeat fresh (0.5 s cadence vs 2.5 s
            # timeout); a hard-killed m5 trips the watchdog instead.
            if now - last_beat > 0.5:
                last_beat = now
                try:
                    with conn.io_lock:
                        beat_ok = wd_heartbeat(conn)
                    if not beat_ok and now - last_rearm > 5.0:
                        last_rearm = now
                        with conn.io_lock:
                            wd_arm(conn)
                except Exception:
                    pass
            display_route = route

            if autopilot:
                _ft0 = time.perf_counter()
                _st0 = _ft0
                _stages: dict[str, float] = {}

                def _mark(name: str) -> None:
                    nonlocal _st0
                    _stages[name] = time.perf_counter() - _st0
                    _st0 = time.perf_counter()

                st = conn.get_state()
                with state_lock:
                    latest_st = st
                last_st = st
                speed = st.speed
                _mark("get_state")
                if now - last_scan > 0.2:
                    last_scan = now
                    sample = range_provider.scan(
                        st.pos, ego_vid=conn.vehicle.vid, radius=55.0)
                    obstacles, last_lidar_hits = (
                        sample.obstacles, sample.ray_hits)
                    # Guard against any sensor ghost whose footprint covers
                    # the ego itself (vision chase-cam self-detection, a
                    # scenario object registered under another id, ...):
                    # an obstacle sitting on top of the car can never be a
                    # real blocker, only a stuck car.  Cover every channel:
                    # the vehicle registry can also report the ego under a
                    # second id on some maps, and that ghost box is just as
                    # harmful as the vision chase-cam one.
                    obstacles = filter_self_overlap(
                        obstacles, st.pos,
                        categories=("vision", "vehicle", "scenario",
                                    "raycast"))
                    if errors_active():
                        sensor_ok = False
                        scan_failures += 1
                        if scan_failures <= 3 or scan_failures % 10 == 0:
                            print(f"[m5] sensor warning: {errors_summary()}")
                    else:
                        sensor_ok = True
                        scan_failures = 0
                    # A scan can take a moment; refresh the watchdog right
                    # after so a slow scan never looks like a dead client.
                    try:
                        with conn.io_lock:
                            wd_heartbeat(conn)
                    except Exception:
                        pass
                _mark("scan")
                # Consume the newest finished camera snapshot without ever
                # waiting for the worker's YOLO / lane scan to complete.
                with vision_lock:
                    snap_new = vision_snapshot["seq"] != last_vision_seq
                    if snap_new:
                        last_vision_seq = int(vision_snapshot["seq"])
                        snap_lanes = list(vision_snapshot["lanes"])
                        snap_lane_frame = vision_snapshot.get("lane_frame")
                        snap_lane_n = int(vision_snapshot["lane_n"])
                        snap_pair_ok = int(vision_snapshot.get(
                            "pair_ok", 0))
                        snap_vis_obs = list(vision_snapshot["vis_obs"])
                        snap_det_boxes = list(vision_snapshot["det_boxes"])
                        snap_hud_boxes = list(vision_snapshot["hud_boxes"])
                        snap_vision_n = int(vision_snapshot["vision_n"])
                        snap_failures = int(vision_snapshot["failures"])
                        snap_black_frames = int(vision_snapshot.get(
                            "black_frames", 0))
                        snap_generation = int(vision_snapshot.get("gen", 0))
                    else:
                        snap_vis_obs = []
                        snap_det_boxes = []
                        snap_hud_boxes = []
                        snap_lane_frame = None
                        snap_lane_n = 0
                        snap_pair_ok = 0
                        snap_vision_n = 0
                        snap_failures = 0
                        snap_black_frames = 0
                        snap_generation = last_vision_generation
                if snap_new:
                    if snap_generation >= last_vision_generation:
                        last_lanes = snap_lanes
                        last_lane_frame = snap_lane_frame
                        lane_n = snap_lane_n
                    else:
                        last_lanes = []
                        last_lane_frame = None
                        lane_n = 0
                    last_vision_generation = max(
                        last_vision_generation, snap_generation)
                    vision_n = snap_vision_n
                    det_boxes = snap_det_boxes
                    hud_boxes = snap_hud_boxes
                    vision_failures = snap_failures
                    black_frames = snap_black_frames
                    vis_obs = snap_vis_obs
                    if vis_obs:
                        # Vision ghosts from the route markers: the yellow
                        # start ball and the red goal ball are detected as
                        # "person" and back-project onto the road, pinning
                        # the speed so the car stutters on an empty road.
                        anchors = []
                        if route is not None and len(route) > 0:
                            near_i = int(np.argmin(np.linalg.norm(
                                route[:, :2] - st.pos[:2], axis=1)))
                            for idx in (0, near_i, len(route) - 1):
                                anchors.append(route[idx][:2])
                        vis_obs = drop_vision_waypoint_ghosts(
                            vis_obs, anchors)
                    # Only multi-frame tracks that survived the ego-motion
                    # check may act as blockers; a single-frame "car" is
                    # usually a phantom riding along with the camera.
                    vis_tracks, confirmed_vis_obs = update_vision_tracks(
                        vis_tracks, vis_obs, st.pos[:2], now,
                        match_m=VIS_TRACK_MATCH_M,
                        confirm_hits=VIS_TRACK_CONFIRM,
                        ttl_s=VIS_TRACK_TTL_S,
                        ego_gate_m=VIS_TRACK_EGO_GATE_M,
                        ride_along_ratio=VIS_TRACK_RIDE_RATIO)
                else:
                    # No new camera snapshot: leave tracks alone instead of
                    # re-confirming a stale object from every loop iteration.
                    det_boxes = []
                    hud_boxes = []
                    confirmed_vis_obs = []
                vision_conf_n = len(confirmed_vis_obs)
                if confirmed_vis_obs:
                    obstacles = merge_obstacles(
                        obstacles + confirmed_vis_obs)
                _mark("vision")
                # Lane centre from the camera markings (time-smoothed) with
                # the raw raycast fan as a free-space fallback.  The chosen
                # frame overrides the keep-right nav offset in plan().
                lane_src = ""
                lane_conf = 0.0
                lane_span = 0.0
                lidar_conf = 0.0
                lane_lat = None
                lane_w = 0.0
                vision_lane = lane_tracker.update(
                    last_lane_frame, st.pos, st.heading, fwd=st.dir)
                lidar_dbg: dict = {}
                lidar_frame = build_lidar_corridor(
                    last_lidar_hits, st.pos, st.heading, fwd=st.dir,
                    debug=lidar_dbg)
                lane_frame = choose_sensor_lane(
                    vision_lane, lidar_frame, st.pos, st.heading,
                    fwd=st.dir, state=lane_fusion_state)
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
                        fwd2 = np.array(
                            [math.cos(st.heading), math.sin(st.heading)])
                    left2 = np.array([-fwd2[1], fwd2[0]])
                    c0 = np.asarray(
                        lane_frame.center[0], dtype=float)[:2]
                    lane_lat = float(
                        (c0 - np.asarray(st.pos[:2], dtype=float)) @ left2)
                desired_speed = cruise_speed
                blocked = False
                hdg_dev = 0.0
                hdg_guard = False
                plan_ran = False
                drive_route = None
                has_route = route is not None and len(route) > 0
                has_lane = lane_frame is not None and lane_frame_usable(
                    lane_frame)
                if has_route or has_lane:
                    dt = max(1e-3, now - last_ctrl)
                    last_ctrl = now
                    plan_route = route if has_route else None
                    plan_nearest = nearest if has_route else 0
                    if has_route:
                        d = np.linalg.norm(
                            route[:, :2] - st.pos[:2], axis=1)
                        nearest = int(np.argmin(d))
                        plan_nearest = nearest
                    drive_route = cached_drive_route
                    blocked = cached_blocked
                    if (drive_route is None
                            or route is not last_plan_route
                            or road_rule is not last_plan_rule
                            or now - last_plan_t >= PLAN_INTERVAL_S):
                        drive_route, blocked = planner.plan(
                            plan_route, obstacles, st.pos, st.heading,
                            plan_nearest,
                            solid_lines=last_lanes, sensor_lane=lane_frame,
                            road_rule=road_rule)
                        plan_ran = True
                        drive_route = np.asarray(drive_route, dtype=float)
                        cached_drive_route = drive_route
                        cached_blocked = blocked
                        last_plan_route = plan_route
                        last_plan_rule = road_rule
                        last_plan_t = time.time()
                    if blocked:
                        blk = getattr(planner, "last_blocker", None)
                        desc = ("unknown obstacle" if blk is None
                                else f"{blk[0]} @ {blk[1]:.1f}m")
                        if now - last_blk_log > 2.5:
                            last_blk_log = now
                            print(f"[m5] BLOCKED by {desc} "
                                  f"(no drivable way; stopping)")
                    # plan() returns either the full route, a local window or
                    # an A* detour; always trim so it starts at the car and
                    # pure pursuit can use index 0.
                    if len(drive_route) >= 2:
                        d0 = np.linalg.norm(
                            drive_route[:, :2] - st.pos[:2], axis=1)
                        start_i = int(np.argmin(d0))
                        if start_i > 0 and len(drive_route) - start_i >= 2:
                            drive_route = drive_route[start_i:]
                    if len(drive_route) >= 2:
                        display_route = drive_route
                        desired_speed, obs_dist = planner.speed(
                            drive_route, obstacles, st.pos, st.heading,
                            0, cruise_speed)
                        if blocked:
                            desired_speed = 0.0
                        # Speed-planning diagnostics for the telemetry/GUI:
                        # corner = curvature limit only, obslim = kinematic
                        # obstacle limit, desired = final planner demand.
                        corner_v = getattr(
                            planner, "last_corner", desired_speed)
                        obs_lim = getattr(planner, "last_obs_lim", None)
                        pp.lookahead = pp.adaptive_lookahead(speed)
                        steer_rad, _, _ = pp.steering(
                            st.pos, st.heading, drive_route, 0)
                        new_steer = float(np.clip(
                            -steer_rad / 0.6, -1.0, 1.0))
                        steer = smooth_steer(prev_steer, new_steer, dt)
                        prev_steer = steer
                        # Curvature cap only for real corners; small tracking
                        # wobble on a straight road must not limit the speed.
                        # ``corner_v`` is the planner's curvature-only limit:
                        # on a straight it stays at the cruise limit, on a
                        # real bend it drops well below, so use it to gate
                        # the steer cap.
                        steer_angle = abs(steer) * 0.6
                        steer_capped = False
                        if (steer_angle > 0.09
                                and corner_v < cruise_speed * 0.85):
                            steer_radius = 2.9 / math.tan(steer_angle)
                            capped = float(math.sqrt(7.0 * steer_radius))
                            if capped < desired_speed:
                                desired_speed = capped
                                steer_capped = True
                    else:
                        steer, desired_speed = 0.0, 0.0
                        corner_v, obs_lim = 0.0, None
                        steer_angle, steer_capped = 0.0, False
                        obs_dist = 999.0
                        prev_steer = 0.0
                else:
                    steer, desired_speed = 0.0, 0.0
                    corner_v, obs_lim = 0.0, None
                    steer_angle, steer_capped = 0.0, False
                    obs_dist = 999.0
                    prev_steer = 0.0
                    dt = max(1e-3, now - last_ctrl)
                    last_ctrl = now

                _mark("plan")

                # Traffic-rule snapshot: road speed limit plus signals on
                # the current link.  Polled slower than the control loop;
                # the decision layer stays pure Python for both runtimes.
                if now - last_rule > RULE_POLL_INTERVAL_S:
                    last_rule = now
                    road_rule = conn.read_current_road_rule(st.pos, st.dir)
                    signal_rules = []
                    selected_signal = None
                    rule_speed_limit = None
                    rule_limit = None
                    rule_reason = None
                    signal_state = None
                    signal_action = 0
                    signal_dist = None
                    signal_name = ""
                    if road_rule is not None:
                        rule_speed_limit = road_rule.speed_limit_mps
                        if road_rule.n1 and road_rule.n2:
                            try:
                                signal_rules = conn.read_signal_snapshot(
                                    conn.vehicle.vid,
                                    road_rule.n1, road_rule.n2)
                            except Exception:
                                signal_rules = []
                        selected_signal = select_signal_rule(
                            signal_rules, st.pos, heading=st.heading,
                            dir_vec=st.dir)

                # ACC-style creep: when a kinematic obstacle limit pins the
                # target at 0 but a drivable path still exists (not blocked),
                # inch forward slowly instead of parking forever.  The brake
                # curve re-engages once the blocker is within STOP_MARGIN.
                desired_speed, creep, creep_since = creep_speed(
                    blocked, obs_lim, desired_speed, speed,
                    creep_since, now, CREEP_MPS)

                # Legal caps are a hard longitudinal limit: reapply after
                # ACC creep so a red/stop signal is never inched past.
                desired_speed, rule_reason, rule_limit = apply_rule_speed(
                    desired_speed, road_rule, selected_signal)
                if signal_requires_stop(selected_signal) \
                        and desired_speed <= 0.01:
                    creep = 0
                    creep_since = None
                signal_state = (
                    selected_signal.state if selected_signal is not None
                    else None)
                signal_action = (
                    selected_signal.action if selected_signal is not None
                    else 0)
                signal_dist = signal_distance(selected_signal)
                signal_name = (
                    selected_signal.name if selected_signal is not None
                    else "")
                _mark("rule")

                # Spin guard: when the car points more than 18 deg away
                # from the route direction (a wide entry into a bend, sand /
                # gravel / a spin just starting), full throttle only digs
                # the wheels in.  Pin the demand to a crawl until the
                # heading lines up again.
                hdg_dev = heading_deviation_deg(route, nearest, st.dir)
                if route is None and drive_route is not None \
                        and len(drive_route) >= 2:
                    hdg_dev = heading_deviation_deg(drive_route, 0, st.dir)
                hdg_cap = heading_dev_speed_cap(hdg_dev, hdg_engaged)
                hdg_guard = hdg_cap is not None
                hdg_engaged = hdg_guard
                if hdg_guard:
                    desired_speed = min(desired_speed, hdg_cap)

                # Reverse guard: D must never accumulate backward travel.
                # A wall bounce or deformed wheel can flicker the signed
                # speed negative for a few frames; only the sustained
                # backward case gets the hard 1.5 s brake hold, so a
                # transient bounce cannot lock the car in place.
                signed = 0.0
                if st.vel is not None and st.dir is not None:
                    signed = float(
                        np.dot(np.asarray(st.vel[:2], dtype=float),
                               np.asarray(st.dir[:2], dtype=float)))
                rev_pos = np.asarray(st.pos[:2], dtype=float)
                # Route-arc progress catches backward motion even when the
                # 1 Hz telemetry of a slow scan misses the negative velocity
                # sample.  Small jitter in the nearest-point search is
                # ignored; a route jump (cutting a corner) rebases instead
                # of counting as reverse travel.
                route_arc = 0.0
                route_back = 0.0
                if route is not None and len(route) >= 2:
                    try:
                        route_arc, _ = _point_route_pos(
                            st.pos[0], st.pos[1], route)
                    except Exception:
                        route_arc = last_route_arc or 0.0
                    if last_route_arc is not None:
                        arc_delta = last_route_arc - route_arc
                        if arc_delta > 0.03 and arc_delta < 3.0:
                            route_back = arc_delta
                        elif arc_delta >= 3.0:
                            last_route_arc = route_arc
                    # A slow main loop can miss the one negative-speed
                    # frame during a wall bounce; compare consecutive
                    # positions along the route tangent as well.  Only
                    # count a frame whose motion is mostly backward so a
                    # lateral slide or route-jump does not look like reverse.
                    if last_pos2d is not None and last_rev_nearest > 0:
                        disp = rev_pos - last_pos2d
                        disp_norm = float(np.linalg.norm(disp))
                        if disp_norm > REVERSE_MOVEMENT_M:
                            i0 = max(
                                0, min(len(route) - 1,
                                       last_rev_nearest - 1))
                            i1 = max(
                                0, min(len(route) - 1,
                                       last_rev_nearest + 1))
                            tv = np.asarray(route[i1][:2], dtype=float) - \
                                np.asarray(route[i0][:2], dtype=float)
                            tn = float(np.linalg.norm(tv))
                            if tn > 1e-6:
                                back_share = max(0.0, float(
                                    -(disp @ (tv / tn)) / disp_norm))
                                if back_share >= REVERSE_BACK_RATIO:
                                    route_back = max(route_back, disp_norm)
                route_reverse = route_back > 0.12 and signed < 0.3
                reverse_now = signed < REVERSE_SPEED_MPS or route_reverse
                rev_sustained = False
                if reverse_now:
                    if rev_since <= 0.0:
                        rev_since = now
                        rev_start_pos = rev_pos
                        rev_dist = 0.0
                        route_rev_dist = 0.0
                        rev_guard_logged = False
                        speed_ctrl.reset()
                    else:
                        rev_dist = max(
                            rev_dist,
                            float(np.linalg.norm(rev_pos - rev_start_pos)))
                    if route_reverse:
                        route_rev_dist += route_back
                    rev_total = max(rev_dist, route_rev_dist)
                    rev_sustained = (now - rev_since) >= REVERSE_ENGAGE_S
                    if rev_sustained:
                        if not rev_guard_logged:
                            rev_guard_logged = True
                            rev_gear = read_gear()
                            print(f"[m5] REVERSE-GUARD engaged: "
                                  f"signed={signed:.2f} m/s "
                                  f"route={route_back:.2f}m "
                                  f"back={rev_total:.2f}m "
                                  f"gear={rev_gear}")
                            # A wall bounce can leave a stale "solid line
                            # under the car" or a sparse raycast ghost that
                            # replans the same wall every frame.  Drop the
                            # lane cache and ghosts so the next plan starts
                            # from a clean corridor instead of repeating
                            # the exact collision.
                            with vision_lock:
                                vision_clear_requested = True
                                vision_clear_generation += 1
                            last_lanes = []
                            lane_n = 0
                            obstacles = [
                                ob for ob in obstacles
                                if not is_sparse_raycast_speck(ob)]
                            cached_drive_route = None
                            cached_blocked = False
                            print("[m5] REVERSE-GUARD cleared lane/sensor "
                                  "ghosts; will replan")
                        rev_hold_until = now + REVERSE_HOLD_S
                        if (rev_total > REVERSE_STOP_DIST_M
                                and not rev_warned):
                            rev_warned = True
                            print(f"[m5] REVERSE {rev_total:.1f}m; "
                                  f"braking to a stop")
                            force_end_reason = "reverse stuck"
                else:
                    rev_since = 0.0
                    rev_start_pos = None
                    rev_dist = 0.0
                    route_rev_dist = 0.0
                    rev_warned = False
                    rev_guard_logged = False
                last_route_arc = route_arc
                last_pos2d = rev_pos.copy()
                last_rev_nearest = nearest
                rev_total = max(rev_dist, route_rev_dist)
                reverse_hold = now < rev_hold_until
                if reverse_now or reverse_hold:
                    desired_speed = 0.0
                _mark("reverse")

                # Ramp the target speed so throttle/brake stay linear.
                if desired_speed > target_speed:
                    target_speed = min(desired_speed,
                                       target_speed + RAMP_ACCEL * dt)
                else:
                    target_speed = max(desired_speed,
                                       target_speed - RAMP_DECEL * dt)
                _mark("ramp")

                if now - last_wspd > 0.1:
                    last_wspd = now
                    wheel_speed = conn.get_wheel_speed()
                    if wheel_speed is None and not wspd_warned:
                        wspd_warned = True
                        print("[m5] wheel-speed unavailable; "
                              "slip guard inactive")
                throttle, brake = speed_ctrl.update(
                    target_speed, speed, dt=min(0.25, max(0.01, dt)),
                    wheel_speed=wheel_speed)
                if reverse_now or reverse_hold:
                    throttle = 0.0
                    if rev_sustained or reverse_hold:
                        brake = max(brake, 0.85)
                    else:
                        brake = max(brake, REVERSE_SOFT_BRAKE)
                slip = bool(speed_ctrl.slip_active)
                _mark("speed")

                t = now - session_t0
                hist["t"].append(t)
                hist["throttle"].append(float(throttle))
                hist["brake"].append(float(brake))
                hist["speed"].append(float(speed))

                mode = getattr(planner, "last_mode", "follow")
                sen = "OK" if sensor_ok else "FAIL"
                goal_dist = 0.0
                if route is not None and len(route) > 0:
                    goal_dist = float(np.linalg.norm(
                        route[-1][:2] - st.pos[:2]))
                # No-progress recovery: the car can sit in D against a
                # wall/kerb with speed 0 and full throttle (it looks like
                # reversing but is just deformation).  Brake, clear the
                # lane/sensor ghosts, retry forward briefly, then stop.
                progress_pos = np.asarray(st.pos[:2], dtype=float)
                if route is not None and len(route) >= 2:
                    ri0 = max(0, nearest - 1)
                    ri1 = min(len(route) - 1, nearest + 1)
                    tv = np.asarray(route[ri1][:2], dtype=float) - np.asarray(
                        route[ri0][:2], dtype=float)
                    tn = float(np.linalg.norm(tv))
                    fwd_dir = (tv / tn if tn > 1e-6
                               else np.asarray(st.dir[:2], dtype=float))
                else:
                    fwd_dir = np.asarray(st.dir[:2], dtype=float)
                if reverse_now:
                    last_progress_pos = progress_pos
                    last_progress_t = now
                if (target_speed < 1.0 or goal_dist <= 12.0
                        or signed >= 0.5):
                    last_progress_pos = progress_pos
                    last_progress_t = now
                elif last_progress_pos is None:
                    last_progress_pos = progress_pos
                    last_progress_t = now
                else:
                    moved_fwd = float(
                        (progress_pos - last_progress_pos) @ fwd_dir)
                    if moved_fwd >= 0.6:
                        last_progress_pos = progress_pos
                        last_progress_t = now
                    elif now - last_progress_t > 2.5:
                        if now < stuck_recover_until:
                            desired_speed = min(desired_speed, 0.5)
                            target_speed = min(target_speed, 0.5)
                        else:
                            stuck_retries += 1
                            if stuck_retries > 2:
                                print("[m5] STUCK (retry failed; stopping)")
                                force_end_reason = "stuck"
                                desired_speed = 0.0
                                target_speed = 0.0
                            else:
                                stuck_recover_until = now + 4.0
                                last_progress_pos = progress_pos
                                last_progress_t = now
                                last_lanes = []
                                obstacles = [
                                    ob for ob in obstacles
                                    if not is_sparse_raycast_speck(ob)]
                                desired_speed = min(desired_speed, 0.5)
                                target_speed = min(target_speed, 0.5)
                                brake = max(brake, 0.15)
                                print("[m5] STUCK (no forward progress); "
                                      "clearing lane/sensor ghosts, retrying")
                boxes = [[round(o.x, 1), round(o.y, 1),
                          round(o.half_w, 1), round(o.half_h, 1),
                          o.label or o.category,
                          (None if o.axis is None else [
                              round(float(o.axis[0]), 3),
                              round(float(o.axis[1]), 3)]),
                          round(o.half_len, 2), round(o.half_thick, 2)]
                         for o in obstacles[:12]]
                rte_pts: list = []
                if route is not None and len(route) > 0:
                    rstep = max(1, len(route) // 200)
                    rte_pts = [[round(float(p[0]), 1),
                                round(float(p[1]), 1)]
                               for p in route[::rstep]]
                if now - last_roads > 1.0:
                    last_roads = now
                    road_polys = roads_json(st.pos)
                if now - last_env > 10.0:
                    last_env = now
                    try:
                        cur_env = conn.current_env()
                    except Exception:
                        pass
                telemetry.publish(
                    t=t, speed=float(speed), throttle=throttle, brake=brake,
                    steer=steer, vel=st.vel, dir_vec=st.dir, up_vec=st.up,
                    pos=st.pos, heading=float(st.heading),
                    nearest=int(nearest),
                    extra={
                        "auto": 1,
                        "nav_world": int(nav_world_visible),
                        "mode": mode,
                        "cruise": round(float(cruise_speed), 1),
                        "speed_limit": (
                            None if rule_speed_limit is None
                            else round(float(rule_speed_limit), 1)),
                        "rule_limit": (
                            None if rule_limit is None
                            else round(float(rule_limit), 1)),
                        "rule_reason": rule_reason,
                        "signal_state": signal_state,
                        "signal_action": signal_action,
                        "signal_dist": (
                            None if signal_dist is None
                            else round(float(signal_dist), 1)),
                        "signal_name": signal_name,
                        "target": round(float(target_speed), 1),
                        "desired": round(float(desired_speed), 1),
                        "corner": round(float(corner_v), 1),
                        "obslim": (None if obs_lim is None
                                   else round(float(obs_lim), 1)),
                        "creep": int(creep),
                        "slip": int(slip),
                        "hdg_dev": round(float(hdg_dev), 1),
                        "hdg_g": int(hdg_guard),
                        "steer_rad": round(float(steer_angle), 3),
                        "sen": sen,
                        "vis": int(vision_n),
                        "vconf": int(vision_conf_n),
                        "black_frames": int(black_frames),
                        "lanes": len(last_lanes),
                        "pair_ok": snap_pair_ok,
                        "tracker_ok": int(lane_frame is not None),
                        "lane_src": lane_src,
                        "lane_paired": int(bool(getattr(
                            lane_frame, "paired", False))),
                        "lane_jump": int(bool(getattr(
                            lane_tracker, "last_rejected", False))),
                        "lane_conf": round(lane_conf, 2),
                        "lane_span": round(lane_span, 1),
                        "lidar_conf": round(lidar_conf, 2),
                        "lidar_hits": len(last_lidar_hits),
                        "lidar_dbg": {k: (round(float(v), 2) if
                                          isinstance(v, (int, float))
                                          else v)
                                      for k, v in lidar_dbg.items()},
                        "lane_lat": (None if lane_lat is None
                                     else round(lane_lat, 2)),
                        "edge_lat": (
                            (None
                             if (edge_lat := _boundary_near_lat(
                                lane_frame.left
                                if getattr(lane_frame, "right", None) is None
                                else lane_frame.right,
                                st.pos, st.heading, fwd=st.dir)) is None
                             else round(edge_lat, 2))
                            if lane_frame is not None and (
                                getattr(lane_frame, "left", None) is not None
                                or getattr(lane_frame, "right", None)
                                is not None)
                            else None),
                        "lane_w": round(lane_w, 1),
                        "sharp": int(getattr(planner, "last_sharp", False)),
                        "plan_mode": getattr(planner, "last_lane_mode", "nav"),
                        "plan_offset": round(float(getattr(
                            planner, "last_lane_offset", 0.0)), 2),
                        "obs": len(obstacles),
                        "obs_d": round(float(obs_dist), 1),
                        "goal_d": round(goal_dist, 1),
                        "rev": round(rev_total, 2),
                        "rev_route": round(route_rev_dist, 2),
                        "rev_g": int(reverse_now),
                        "rev_s": (round(float(now - rev_since), 2)
                                  if reverse_now else 0.0),
                        "rev_h": int(reverse_hold),
                        "route": 0 if route is None else len(route),
                        "blk": (f"{planner.last_blocker[0]}@{planner.last_blocker[1]:.0f}m"
                                if getattr(planner, "last_blocker", None)
                                else ""),
                        "boxes": boxes,
                        "rte": rte_pts,
                        "env": cur_env,
                        "roads": road_polys,
                        "markings": markings_json(last_lanes),
                    },
                )
                _mark("telemetry")

                # Hold brake at a standstill while blocked so the car does
                # not roll back on a slope waiting for the player to act.
                if blocked and speed < 0.5:
                    brake = max(brake, 0.12)
                # Keep the shifter pinned to the forward gear every step:
                # a repeated identical gear request is a no-op in realistic
                # mode (verified live) and guarantees a gearbox hiccup can
                # never leave the car in R/P while autopilot is driving.
                conn.control(throttle=throttle, steering=steer, brake=brake,
                             gear=fwd_gear)
                _mark("control")

                if now - last_status > 1.0:
                    last_status = now
                    mode = getattr(planner, "last_mode", "follow")
                    sen = "OK" if sensor_ok else "FAIL"
                    gear_txt = read_gear() or "?"
                    print(f"[m5] mode={mode} sen={sen} obs={len(obstacles)} "
                          f"vis={vision_n}/{vision_conf_n} "
                          f"nearest={obs_dist:.0f}m "
                          f"v={speed:.1f} "
                          f"target={target_speed:.1f} "
                          f"desired={desired_speed:.1f} "
                          f"corner={corner_v:.1f} "
                          f"rule={(rule_reason or '-')} "
                          f"sig={signal_action_label(signal_action)} "
                          f"lanes={len(last_lanes)} "
                          f"lane={lane_src or '-'} {lane_conf:.2f} "
                          f"sharp={int(getattr(planner, 'last_sharp', False))} "
                          f"cap={int(steer_capped)} "
                          f"obslim={(obs_lim if obs_lim is None else round(obs_lim, 1))} "
                          f"creep={creep} slip={int(slip)} "
                          f"hdg={hdg_dev:.0f} "
                          f"gear={gear_txt} "
                          f"rev={rev_total:.2f}/{int(reverse_now)} "
                          f"throttle={throttle:.2f} "
                          f"brake={brake:.2f}")

                ended, reason = False, ""
                if force_end_reason:
                    ended, reason = True, force_end_reason
                elif goal_dist > 0 and goal_dist < GOAL_RADIUS_M:
                    ended, reason = True, "goal reached"
                elif t > args.max_run:
                    ended, reason = True, "timeout"

                if ended:
                    faulthandler.cancel_dump_traceback_later()
                    autopilot = False
                    conn.control(throttle=0.0, brake=1.0, steering=0.0,
                                 gear=fwd_gear)
                    conn.step(20)
                    release_control()
                    restore_gearbox()
                    toast(f"autopilot ended: {reason}")
                    # Publish one final "ended" frame so the GUI can pop the
                    # last-session chart immediately (it also watches
                    # last_session.json as the robust fallback).
                    try:
                        telemetry.publish(
                            t=t, speed=float(speed), throttle=0.0, brake=1.0,
                            steer=0.0, vel=st.vel, dir_vec=st.dir,
                            up_vec=st.up, pos=st.pos,
                            heading=float(st.heading), nearest=int(nearest),
                            extra={"auto": 0, "mode": "ENDED", "sen": sen,
                                   "nav_world": int(nav_world_visible),
                                   "target": 0.0,
                                   "cruise": round(float(cruise_speed), 1),
                                   "speed_limit": (
                                       None if rule_speed_limit is None
                                       else round(float(rule_speed_limit), 1)),
                                   "rule_limit": (
                                       None if rule_limit is None
                                       else round(float(rule_limit), 1)),
                                   "rule_reason": rule_reason,
                                   "signal_state": signal_state,
                                   "signal_action": signal_action,
                                   "signal_dist": (
                                       None if signal_dist is None
                                       else round(float(signal_dist), 1)),
                                   "signal_name": signal_name,
                                   "vis": int(vision_n),
                                   "vconf": int(vision_conf_n),
                                   "lanes": len(last_lanes),
                                   "sharp": int(getattr(planner,
                                                        "last_sharp", False)),
                                   "obs": len(obstacles), "obs_d": 999.0,
                                   "goal_d": round(goal_dist, 1),
                                   "route": 0, "blk": "", "slip": int(slip),
                                   "hdg_dev": round(float(hdg_dev), 1),
                                   "hdg_g": int(hdg_guard),
                                   "boxes": [],
                                   "rte": [], "env": cur_env,
                                   "roads": road_polys,
                                   "markings": markings_json(last_lanes),
                                   "ended": 1, "reason": reason},
                        )
                    except Exception:
                        pass
                    finish_session()
                else:
                    conn.step(1, wait=False)
                    _stages["step"] = time.perf_counter() - _st0
                    if time.perf_counter() - _ft0 > 0.35:
                        detail = " ".join(
                            f"{k}={v*1000:.0f}ms"
                            for k, v in _stages.items())
                        if plan_ran and planner.last_plan_stages:
                            detail += " | plan_stages=" + " ".join(
                                f"{k}={v:.0f}ms"
                                for k, v in planner.last_plan_stages.items())
                        print(f"[m5] SLOW-FRAME {detail}")
            else:
                # Keep the EID alive while the autopilot is off: publish a
                # light idle snapshot (speed / heading / position) so the
                # launcher GUI never goes stale between sessions.
                if now - last_idle > 0.5:
                    last_idle = now
                    try:
                        st_idle = conn.get_state()
                        last_st = st_idle
                        if now - last_roads > 1.0:
                            last_roads = now
                            road_polys = roads_json(st_idle.pos)
                        if now - last_env > 10.0:
                            last_env = now
                            try:
                                cur_env = conn.current_env()
                            except Exception:
                                pass
                        telemetry.publish(
                            t=0.0, speed=float(st_idle.speed),
                            throttle=0.0, brake=0.0, steer=0.0,
                            vel=st_idle.vel, dir_vec=st_idle.dir,
                            up_vec=st_idle.up, pos=st_idle.pos,
                            heading=float(st_idle.heading),
                            extra={"auto": 0, "mode": "IDLE", "sen": "OK",
                                   "nav_world": int(nav_world_visible),
                                   "target": 0.0,
                                   "cruise": round(float(cruise_speed), 1),
                                   "vis": 0, "obs": 0,
                                   "obs_d": 999.0, "goal_d": 0.0,
                                   "route": 0 if route is None else len(route),
                                   "boxes": [], "rte": [],
                                   "env": cur_env, "roads": road_polys,
                                   "markings": []},
                        )
                    except Exception:
                        pass
                time.sleep(0.02)

            # HUD: front camera overlay + bird view when vision is enabled
            if hud is not None and now - last_hud > 0.05:
                last_hud = now
                try:
                    if not autopilot:
                        last_st = conn.get_state()
                    cam = None
                    img = None
                    if vision and last_st is not None:
                        if autopilot:
                            with vision_lock:
                                img = vision_snapshot.get("frame")
                            if img is not None:
                                img = img.copy()
                        else:
                            img = camera_provider.grab()
                            img = cv2.resize(img, (CAM_W, CAM_H))
                    if img is not None:
                        img = render_camera_overlay(
                            img, display_route, last_st.pos,
                            last_st.heading, cam_model,
                            obstacles=obstacles, det_boxes=hud_boxes,
                            lane_markings=last_lanes)
                        bv = np.full((CAM_H, CAM_H, 3), (22, 24, 30), np.uint8)
                        start_xy = (display_route[0][:2]
                                    if display_route is not None
                                    and len(display_route) else None)
                        goal = (display_route[-1][:2]
                                if display_route is not None
                                and len(display_route) else None)
                        render_birdview(
                            bv, route_xy=display_route,
                            obstacles=obstacles,
                            waypoints=[start_xy] if start_xy is not None
                            else None,
                            goal_xy=goal, pos=last_st.pos,
                            heading=last_st.heading,
                            lane_markings=last_lanes)
                        cam = np.hstack([img, bv])
                    data = telemetry.latest
                    if data is None and last_st is not None:
                        data = {"t": 0.0, "speed": last_st.speed,
                                "throttle": 0.0, "brake": 0.0, "steer": 0.0,
                                "heading": last_st.heading}
                    if not hud.update(data, cam=cam):
                        quit_flag = True
                except Exception as exc:
                    print(f"[m5] hud error: {exc}")

            # 3D world overlay at ~3 Hz
            if now - last_overlay > 0.33:
                last_overlay = now
                pos3 = (0.0, 0.0, 0.0)
                if last_st is not None:
                    pos3 = tuple(float(v) for v in last_st.pos)
                if autopilot:
                    mode = getattr(planner, "last_mode", "follow")
                    if mode == "blocked":
                        mode_txt = "BLOCKED-stop"
                    elif mode == "detour":
                        mode_txt = "AVOIDING"
                    elif mode == "deform":
                        mode_txt = "nudging"
                    else:
                        mode_txt = "cruise"
                    sen_txt = "SENSOR OK" if sensor_ok else "SENSOR FAIL"
                    lane_txt = (f" | lane {lane_src} {lane_conf:.2f}"
                                if lane_src else "")
                    status = (f"AUTOPILOT {mode_txt} | {sen_txt} | "
                              f"obs {len(obstacles)} nearest {obs_dist:.0f}m | "
                              f"target {target_speed:.0f} m/s"
                              + lane_txt)
                elif display_route is not None and len(display_route) > 0:
                    status = (f"ROUTE {len(display_route)} pts  "
                              f"press F9 to start")
                else:
                    status = "press F9 to drive on lane vision"
                try:
                    start_xy = (display_route[0][:2]
                                if display_route is not None
                                and len(display_route) else None)
                    goal_xy = (display_route[-1][:2]
                               if display_route is not None
                               and len(display_route) else None)
                    with conn.io_lock:
                        overlay.update(
                            route_xy=display_route,
                            obstacles=obstacles if vision else None,
                            waypoints=[start_xy] if start_xy is not None
                            else None,
                            goal_xy=goal_xy,
                            status_text=status, status_pos=pos3,
                            enabled=vision and not args.no_overlay,
                            markers=not args.no_markers)
                except Exception as exc:
                    print(f"[m5] overlay error: {exc}")
        except KeyboardInterrupt:
            print("[m5] interrupted")
            break
        except Exception as exc:
            print(f"[m5] loop error: {exc}")
            time.sleep(0.2)

    # cleanup
    if autopilot:
        try:
            # Quitting mid-run (launcher Stop button / Ctrl+C) should leave
            # the same session record as F9-off and goal-reached.
            finish_session(show_chart=False)
        except Exception as exc:
            print(f"[m5] quit session summary failed: {exc}")
        try:
            conn.control(throttle=0.0, brake=1.0, steering=0.0,
                         gear=fwd_gear)
            conn.step(20)
            release_control()
            restore_gearbox()
        except Exception:
            pass
    try:
        with conn.io_lock:
            overlay.close()
    except Exception:
        pass
    try:
        faulthandler.cancel_dump_traceback_later()
    except Exception:
        pass
    wd_beat_stop.set()
    wd_beat_thread.join(timeout=2.0)
    ctl.clear()
    hotkeys.close()
    telemetry.close()
    if hud is not None:
        hud.close()
    try:
        camera_provider.close()
    except Exception:
        pass
    try:
        range_provider.close()
    except Exception:
        pass
    # Normal exit: the car is parked and pedals zeroed, so disarm the
    # watchdog - otherwise it keeps pulling the handbrake when the player
    # takes over manually.  Must happen before conn.close() (the socket).
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


if __name__ == "__main__":
    main()
