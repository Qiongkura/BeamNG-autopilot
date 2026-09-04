"""FSD-mode live driving: FSDStack planning -> safety monitor -> control.

This is the optional *real-driving* path of the FSD-style stack: instead
of only recording shadow data, it drives the car with the layered
planner's chosen trajectory, arbitrated every frame by the safety
monitor (which can degrade to a stop when the path is blocked, sensors
go stale, or the trajectory leaves the lane).  It is a separate entry
point from ``m5_autopilot.py`` so the proven rule autopilot (94.6%
route result) is never touched.

Usage::
    .venv\\Scripts\\python.exe scripts\\m5_fsd_drive.py --runtime tech \\
        --attach --seconds 30 --speed 8
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beamng_autopilot import config
from beamng_autopilot.connector import BeamNGConnector, angle_to_quat
from beamng_autopilot.control import gearbox
from beamng_autopilot.control.reverse_guard import ReverseGuard
from beamng_autopilot.control.reverse_maneuver import ReverseManeuver
from beamng_autopilot.control.pure_pursuit import PurePursuit
from beamng_autopilot.control.speed import (
    SpeedController, rate_limit_pedal,
)
from beamng_autopilot.fsd_stack import FSDStack
from beamng_autopilot.lane import perception_curve_speed
from beamng_autopilot.neural.e2e_runtime import (
    DEFAULT_E2E_WEIGHTS, E2ERuntime,
)
from beamng_autopilot.occupancy import OccupancyGrid
from beamng_autopilot.planning import (
    Scene, anchored_rule_ref, arbitrate, local_route,
)
from beamng_autopilot.planning.constraints import _boundary_lateral
from beamng_autopilot.planning.local_route import (
    map_lane_edges, _project_arc, _route_turn_deg,
)
from beamng_autopilot.planning.speed_profile import \
    MIN_SPEED as _PROF_MIN_SPEED, \
    speed_profile_for_path as _spf_raw
from beamng_autopilot.roadnet import RoadNetwork
from beamng_autopilot.autopilot import nearest_route_point, smooth_steer
from beamng_autopilot.planner import (
    LocalPlanner,
    emergency_speed_limit_mps,
    emergency_stop_clearance_m,
    path_grid_clearance_m,
)
from beamng_autopilot.safety_monitor import SafetyMonitor
from beamng_autopilot.vision.heads import (
    ObjectHead, SemanticHead, TrafficSignalHead,
)
from beamng_autopilot.vision.lanes import (
    PaintedLineLateralCorrector,
    painted_line_correction_active,
    painted_line_direction,
    painted_line_lane_center,
    painted_line_markings,
    polyline_dir_at,
)

# Reverse guard: the car must never drive backwards under the FSD mode.
# A real factory stack has lane/gear protections - m5_autopilot does too
# (REVERSE_ENGAGE_S/REVERSE_HOLD_S/gear=D).  Without this the FSD drive
# reversed into walls after an impact ("dumb reversing" seen on probes).
REVERSE_THRESHOLD_MPS = -0.35
REVERSE_CLEAR_MPS = 0.2
# Slope-creep assist: seconds of full throttle before a "stuck" car is
# allowed to reverse (a car stopped at the bottom of a dip facing uphill
# is not wedged - it needs torque, not a backward roll).
CLIMB_ASSIST_S = 3.5
# Real-time control loop: the sim runs continuously and the loop paces
# itself to ~REALTIME_CTRL_HZ, so the car never freezes between ticks
# (the old paused-step design moved 0.33 s then froze ~1.2 s - the
# stutter).  Each tick now drives with the previous control for at most
# ~0.5 s, so the stale-control window is the tick time itself (ticks are
# ~0.4-0.6 s after warm-up, vs the 5-10 m stale drift of fix44-51).
REALTIME_CTRL_HZ = 2.0
# Warm-up crawl: the first FSD ticks load YOLO and settle camera/LiDAR
# (observed 4-6 s); in real time the car would otherwise run open-loop
# during that.  Crawl until the object head is live, capped at WARMUP_S.
WARMUP_S = 8.0
WARMUP_SPEED_MPS = 1.5
# If a tick ever takes longer than this, the car has been driving
# open-loop for that long - keep this frame slow instead of trusting
# stale controls.
STALE_CTRL_S = 1.2
STALE_CTRL_SPEED_MPS = 2.0
# Tick time-budget governor (smoothness): the FSD tick cost swings
# widely - a semantic + YOLO + fresh-LiDAR tick can hit 0.5-0.7 s while
# a reuse tick is ~40-100 ms, so without a bound the control loop
# visibly stutters every few frames.  The stack defers a due heavy head
# once the tick exceeds the budget and serves the cached output instead
# (the head stays due and runs on the next affordable tick).  The budget
# adapts to the measured frame time: fast runs never constrain a head,
# slow runs stay bounded.
TICK_BUDGET_MIN_S = 0.25
TICK_BUDGET_MAX_S = 0.45
TICK_BUDGET_FRAC = 0.85          # of the EMA frame time
TICK_BUDGET_EMA = 0.30           # EMA weight for the frame-time tracker
# Target-speed smoothing: the plan speed changes by whole m/s between
# ticks (corner governor, obstacle caps).  Feeding it straight into the
# SpeedController made the pedals oscillate throttle -> brake -> throttle
# every other frame (fix54/fix56).  Ramp the effective target toward the
# plan at a bounded rate instead.
SPEED_TARGET_RAMP_MPS = 1.5     # m/s per sim second
# SpeedController pedal band: brake enters at err < -(deadband+hyst),
# throttle at err > +(deadband+hyst).  The old 0.35/0.4 pair made the
# car coast a +/-0.75 m/s band around cruise (5.25 <-> 7.0), which
# read as a visible accelerate -> brake -> accelerate wave; 0.2/0.25
# keeps a real hysteresis but halves the band to +/-0.45 m/s.
SPEED_DEADBAND_MPS = 0.2        # SpeedController coast deadband
SPEED_HYST_MPS = 0.25           # SpeedController brake/throttle hysteresis
# Plan-speed brake governor hysteresis: enter at +1.0 m/s overshoot,
# release once back within +0.5 m/s, and brake gently (0.25) instead of
# 1.0 - BeamNG's brake is highly nonlinear and even 0.4-0.7 stands the
# car dead in one 0.33 s burst, which then re-triggers the full-throttle
# stall loop (fix61-64).  The downhill-start guard below prevents the
# overshoot in the first place.
# Heading-error speed scrub: with a ~0.6 s control tick a 7-8 m/s
# car keeps rotating long after the pure-pursuit correction, and
# the loop oscillates instead of converging (opt12 2026-08-27: the
# heading swung -99 -> -131 -> -94 deg and the car drifted into the
# right-side wall).  Once the nose points more than START_DEG away
# from the nav route, ramp the target down to FLOOR at FULL_DEG so
# the steering loop can catch the rotation at a speed it can
# control.  Real bends (hairpins) are already slowed by the corner
# governor, so this only trims the oscillation case.
HEADING_DEV_START_DEG = 12.0
HEADING_DEV_FULL_DEG = 40.0
HEADING_DEV_CAP_MPS = 5.0
HEADING_DEV_FLOOR_MPS = 1.5
# Map road-edge guard: the real DecalRoad left/right edges are the
# hard map prior for "where the road is".  Once the ego crosses an
# edge the car is on grass/verge (right) or in the oncoming lane
# (left) - cut the target hard so it never cruises off-road (opt13:
# with no nav route the straight-line reference drove straight onto
# grass and crept there).  Thresholds are beyond the own-lane centre
# so normal right-lane driving is never punished.
ROAD_OFF_EDGE_M = 1.0     # past an edge: crawl (stop if it worsens)
ROAD_OFF_CRAWL_MPS = 0.5
ROAD_EDGE_SLOW_MPS = 2.0
# Past STOP_M the car is definitively off the road - stop creeping and
# steer back to the road centre (town opt24: after a 90-deg corner cut
# the car drove 20 m on grass because route_local anchors at the car
# and the old guard only capped at 0.5 m/s).
ROAD_OFF_STOP_M = 3.0   # beyond this = definitely off-road (hard stop);
                    # junction/start offsets stay ~2 m and only crawl
ROAD_RETURN_LOOK_M = 40.0  # road-centre target for the return steering
ROAD_RETURN_STEER_MAX = 0.55
# Map-prior lane centre EMA: map_lane_edges re-derives the own-lane
# centre every frame from the road edges; at junctions / edge
# sampling switches it can wiggle a few degrees between frames and
# feed the steering loop (opt12: map_lane bearing swung ~7 deg).
# Blend the new centre with the previous one (arc-aligned) so the
# reference stays smooth; boundaries are left untouched.
MAP_LANE_EMA = 0.6
# Start placement is PERCEPTION ONLY: after the semantic head warms up,
# the drive re-positions itself into its own lane by measuring the
# painted line (line right side + own-lane half width).  The pre-warm
# ground snap must not ride the route centre line, and it must not use a
# fixed "route centre + offset" constant either - that is the map-prior
# shortcut the FSD realism rules forbid.  (2026-09-03: SNAP_LANE_OFFSET_M
# removed; placement now comes from painted-line perception.)
# End-of-route handling: the nav route is finite.  When the local
# forward window reaches the route END, the car has arrived; without
# an explicit stop it creeps onto the road end / kerb and parks over
# the edge line (opt15 2026-08-28: parked at the destination with
# lat_right -0.3~-0.5 m, i.e. over the right line).  Slow from 10 m
# out and stop from 4 m out, while the planner still keeps the lane
# centre, so the car ends gently in its lane before the road end.
END_START_SLOW_M = 12.0
# Lateral re-centring starts earlier than the braking zone: the car
# enters the final stretch ~1.0-1.4 m from the centreline and the
# 12-6 m stop zone alone cannot pull it to the lane centre before the
# stop (opt35-41 parked 1.4-1.7 m off-centre).  From 20 m out the
# steering reference aims at the own-lane centre ahead; braking still
# starts at END_START_SLOW_M.
END_PULL_START_M = 20.0
END_STOP_M = 6.0
END_SLOW_MPS = 1.0
END_BRAKE = 0.7
# End-zone last-good perception anchor: if the painted line flickers
# or fades inside the final stop zone, keep converging to the LAST
# perceived own-lane centre (still perception - never a map prior)
# for a short window, so a line dropout does not silently drop the
# lateral pull and park the car wherever it entered the zone.
END_PLC_HOLD_S = 2.0
END_PLC_MAX_LAT_M = 3.0
END_PLC_MAX_FWD_M = 15.0
# Steady painted-line lateral corrector (FSD realism, perception only):
# telemetry showed ``lane_src_sel=map`` for 100% of a run while the
# semantic LINE mask gave a confident own-lane centre - the map-prior
# own lane rides on (or too close to) the painted centre line and the
# car never uses its perception lateral reference in steady driving.  A
# real FSD stack places the car where ITS sensors say the lane is: while
# cruising with the map lane leading, nudge the near path toward the
# perceived own-lane centre (line right side + lane half width) at a
# bounded rate.  No nav-centreline + offset constant anywhere - the
# target is the same perception rule as start placement / end zone.
PLC_MAX_SHIFT_M = 1.0
PLC_RATE_MPS = 1.2
PLC_HORIZON_M = 12.0
PLC_HOLD_S = 2.0
PLC_MIN_SPEED_MPS = 0.5
PLC_MIN_ENGAGE_M = 0.02
GOV_ON_MPS = 0.8
GOV_OFF_MPS = 0.4
GOV_BRAKE = 0.25
# Tight-bend governor gate: only apply the hairpin speed cap when the
# 12 m look-ahead actually TURNS this much.  A rounded junction corner /
# resample wiggle can measure R~3 m while turning <30 deg; capping there
# made the car crawl 5+ s through a widening junction (fsd opt23).
BEND_GOV_MIN_TURN_DEG = 40.0
# Plan-speed rate limit (m/s per sim second): the full-route profile is
# re-profiled every frame against the LIVE LiDAR grid, and junction /
# end-zone clutter appears and disappears between ticks.  Without a rate
# limit the plan snapped 6.0 <-> 1.0 and the controller slammed the
# brakes then relaunched (opt22 t=50.6: plan=1.00 brk=0.75, next frame
# plan=6.00).  The down rate stays above SPEED_TARGET_RAMP so real bend
# deceleration still binds; the up rate is gentler so a transient
# obstruction never ends in a relaunch kick.
PLAN_DOWN_RATE_MPS2 = 2.0
PLAN_UP_RATE_MPS2 = 1.0



def _trim_backtrack(route):
    """Drop a terminal backtracking tail from a road-graph route.

    When the goal is NOT on the road network, ``route_with_edges`` appends
    a straight segment from the last road node to the off-road goal - the
    route doubles back against the road direction and the final ~2 m of
    centreline / edge data point the wrong way (the end-zone reference
    then aimed left and the car parked 1.4-1.7 m off the lane centre,
    opt35-38).  The end-zone reference must only use the real road part,
    so cut the route at the first terminal direction reversal.
    """
    r = np.asarray(route[:, :2], dtype=float)
    n = len(r)
    if n < 6:
        return r
    seg = np.diff(r, axis=0)
    ang = np.arctan2(seg[:, 1], seg[:, 0])
    for k in range(n - 3, 2, -1):
        a = ang[max(0, k - 4):k]
        b = ang[k:k + 2]
        if len(a) and len(b):
            da = np.arctan2(np.sin(a - b[0]), np.cos(a - b[0]))
            if float(np.median(np.abs(da))) > 2.0:
                return r[:k]
    return r


def _painted_line_lat(out, pos, heading, marks=None):
    """Painted centre-line lateral relative to the ego (left = +).

    Projects the semantic LINE mask's near-field pixels to the ground
    with the tick's own camera model, so every run self-reports whether
    the car sits LEFT (oncoming) or RIGHT (own lane) of the painted
    line - the objective check that replaces eyeballing telemetry.
    Returns None when no line is detected this frame.
    """
    if out is None or out.frame is None or out.cam is None:
        return None
    sem = out.head_outputs.get("semantic")
    if sem is None or "line" not in getattr(sem, "masks", {}):
        return None
    try:
        if marks is None:
            ground_z = (float(pos[2]) - config.EGO_ORIGIN_GROUND_GAP_M
                        if len(pos) > 2 else None)
            marks = painted_line_markings(sem, out.cam, pos, heading,
                                          ground_z=ground_z)
        if not marks:
            return None
        fwd = np.array([math.cos(float(heading)), math.sin(float(heading))])
        left = np.array([-fwd[1], fwd[0]])
        lats = []
        p2 = np.asarray(pos[:2], dtype=float)
        for m in marks:
            pts = np.asarray(m.world[:, :2], dtype=float)
            near = pts[np.linalg.norm(pts - p2, axis=1) < 25.0]
            if len(near):
                lats.append(float(((near - p2) @ left).mean()))
        return round(float(np.mean(lats)), 3) if lats else None
    except Exception:
        return None


def _ref_bearing(ref, pos, min_m: float = 1.5, max_m: float = 20.0):
    """Bearing (deg) of the near-ahead part of a reference polyline."""
    if ref is None or len(ref) < 2:
        return None
    r = np.asarray(ref[:, :2], dtype=float)
    p = np.asarray(pos[:2], dtype=float)
    d = np.linalg.norm(r - p, axis=1)
    sel = np.flatnonzero((d >= min_m) & (d <= max_m))
    if len(sel) < 2:
        sel = np.flatnonzero(d >= min_m)
    if len(sel) < 2:
        return None
    i, j = int(sel[0]), int(sel[-1])
    v = r[j] - r[i]
    L = float(np.linalg.norm(v))
    if L < 1e-9:
        return None
    return round(float(math.degrees(math.atan2(v[1], v[0]))), 1)


def _route_lateral_off_m(pos, nav_route, road_left, road_right,
                         window: int = 10, cap_m: float = 12.0,
                         default_m: float = 7.0):
    """Signed lateral + metres beyond the local road edge vs the NAV
    centreline.

    Returns ``(lat, beyond, half_width)``: ``lat`` is the ego's signed
    offset from the route centreline (positive = right of travel),
    ``half_width`` the local road half-width (max edge distance in a
    rolling window, capped), and ``beyond = max(0, |lat| - half_width)``.

    The guard measures against the centreline instead of the raw
    DecalRoad edge polylines because those fold at junctions (two road
    segments stitched by the graph) and go stale past the last graph
    node - the nearest-edge point then jumps to a far-away corner and
    reports a fake 3 m off-road on a straight section while the car is
    on the centreline (town run10 t=11 at (737,751)).
    """
    p = np.asarray(pos[:2], dtype=float)
    r = np.asarray(nav_route[:, :2], dtype=float)
    n = len(r)
    if n < 2:
        return 0.0, 0.0, default_m
    i = int(np.argmin(np.linalg.norm(r - p, axis=1)))
    i0 = max(0, i - 2)
    i1 = min(n - 1, i + 2)
    tv = r[i1] - r[i0]
    L = float(np.linalg.norm(tv))
    if L < 1e-9:
        return 0.0, 0.0, default_m
    rn = np.array([tv[1] / L, -tv[0] / L])
    lat = float(np.dot(p - r[i], rn))
    hw = []
    for k in range(max(0, i - window), min(n, i + window + 1)):
        lk = road_left[k]
        rk = road_right[k]
        if np.all(np.isfinite(lk)) and np.all(np.isfinite(rk)):
            hw.append(float(np.linalg.norm(r[k] - lk)))
            hw.append(float(np.linalg.norm(r[k] - rk)))
    hw_m = min(cap_m, max(hw)) if hw else default_m
    return lat, max(0.0, abs(lat) - hw_m), hw_m


def _path_curvature_ff(path, pos, heading, near_m: float = 1.5,
                       horizon_m: float = 8.0, wheelbase: float = 2.9,
                       ratio: float = 0.6, max_ff: float = 0.40) -> float:
    """Feed-forward steering from the chosen path's near-ahead curvature.

    The ~1.4 s control loop only reacts to the PurePursuit target at the
    lookahead point, so at 2 m/s the car has already passed the entry of
    a hairpin before the pursuit asks for the turn (fix37-41 runs: the
    first -110 -> -24 deg bend was missed every time and the car ran
    straight past the apex).  A feed-forward term from the path curvature
    2-10 m ahead starts the turn as soon as the path bends.

    Returns a NORMALIZED steering input (negative = left), scaled by how
    aligned the ego heading is with the path so a sideways rejoin is not
    fought.  ``ratio`` is the rad-per-normalized-input steering ratio used
    by the pursuit conversion below.
    """
    if path is None or len(path) < 4:
        return 0.0
    p = np.asarray(path[:, :2], dtype=float)
    pos2 = np.asarray(pos[:2], dtype=float)
    n = len(p)
    d = np.linalg.norm(p - pos2, axis=1)
    i0 = int(np.argmin(d))
    arc = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    base = float(arc[i0])
    idxs = [i0]
    for tgt in (near_m, near_m + horizon_m):
        j = i0
        while j < n - 1 and float(arc[j]) - base < tgt:
            j += 1
        idxs.append(j)
    i1, i2 = idxs[1], idxs[2]
    if i2 - i1 < 2:
        return 0.0

    def _tangent(i: int) -> np.ndarray:
        a = max(0, i - 1)
        b = min(n - 1, i + 1)
        v = p[b] - p[a]
        L = float(np.linalg.norm(v))
        return (v / L) if L > 1e-9 else np.array([1.0, 0.0])

    t1 = _tangent(i1)
    t2 = _tangent(i2)
    th1 = math.atan2(float(t1[1]), float(t1[0]))
    th2 = math.atan2(float(t2[1]), float(t2[0]))
    dth = (th2 - th1 + math.pi) % (2.0 * math.pi) - math.pi
    ds = max(1e-3, float(arc[i2] - arc[i1]))
    kappa = dth / ds
    align = float(np.clip(
        math.cos(th1 - float(heading)), 0.0, 1.0))
    ff = -kappa * wheelbase / ratio   # left curve (kappa>0) -> negative input
    return float(np.clip(ff * (0.3 + 0.7 * align), -max_ff, max_ff))


def main() -> int:
    ap = argparse.ArgumentParser(description="FSD-mode live driving")
    ap.add_argument("--runtime", choices=("auto", "steam", "tech"),
                    default="auto")
    ap.add_argument("--attach", action="store_true")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--speed", type=float, default=6.0)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--cam-w", type=int, default=400)
    ap.add_argument("--cam-h", type=int, default=300)
    ap.add_argument("--teleport", nargs=3, type=float, default=None,
                    metavar=("X", "Y", "YAW_DEG"),
                    help="teleport to an open stretch before driving")
    ap.add_argument("--out", type=str, default=None,
                    help="path for per-frame JSON telemetry export")
    ap.add_argument("--lane-mode", choices=("map", "auto", "sensor"),
                    default="map",
                    help="lane-keep reference policy: map (rule-stable "
                         "default), auto (sensor leads only when it agrees "
                         "with the map lane), sensor (perception-led; map "
                         "prior stays the hard guard-rail)")
    ap.add_argument("--strict", action="store_true",
                    help="FSD realism mode (docs/fsd_realism.md): with "
                         "--lane-mode sensor the map lane may NEVER lead; "
                         "no paired perception lane -> no-lane degradation")
    ap.add_argument("--e2e-model", type=str, default=None,
                    help="trained E2ENetTorch checkpoint to rank as the "
                         "neural planning candidate (default: "
                         "logs/m5_e2e/best_temporal.pt)")
    ap.add_argument("--no-e2e", action="store_true",
                    help="disable the E2E neural planning candidate")
    ap.add_argument("--goal", nargs=2, type=float, default=None,
                    metavar=("X", "Y"),
                    help="set an in-game navigation route to this goal "
                         "before driving (else reuse the active nav route)")
    args = ap.parse_args()

    conn = BeamNGConnector(
        "italy", "etk800",
        port=config.runtime_port(args.runtime),
        home=config.runtime_home(args.runtime))
    pp = PurePursuit(lookahead=5.0)
    speed_ctrl = SpeedController(deadband=SPEED_DEADBAND_MPS,
                                hyst_mps=SPEED_HYST_MPS)
    monitor = SafetyMonitor(max_speed=args.speed)
    try:
        conn.open(launch=not args.attach)
        try:
            conn.attach_vehicle(already_open=True)
        except Exception:
            conn.load_scenario()
        if args.teleport is not None:
            x, y, yaw = args.teleport
            # Ground-safe teleport: the connector measures the real surface
            # with a cast ray and re-checks after settling, so no map can
            # ever drop the car below terrain (hardcoded z heights did on
            # maps whose surface sits much higher - 2026-08-28).
            conn.safe_teleport(float(x), float(y), heading_deg=float(yaw))
            st1 = conn.get_state()
            print(f"[fsd-drive] teleport -> "
                  f"({float(st1.pos[0]):.1f}, {float(st1.pos[1]):.1f}, "
                  f"{float(st1.pos[2]):.1f})")

        # Navigation route: a real stack plans ALONG the destination
        # route (FSD vector-space planner), not a straight line ahead -
        # a straight reference drives the car into the building on the
        # first town bend (observed town run 2026-08-21).  A single-marker
        # ``core_groundMarkers.setPath`` route is just a straight
        # interpolation to the marker, NOT a road-following nav route, so
        # it cuts across roads/walls (town run 2026-08-21: a straight line
        # pointed at a wall at (711.8,730.8) and the car wedged there).
        # Prefer the road-graph A* route built from the actual DecalRoad
        # centre-lines; fall back to the in-game route only when the road
        # graph is unavailable.
        nav_route = None
        nav_route_ref = None
        road_left = None
        road_right = None
        st0 = conn.get_state()
        p0 = np.asarray(st0.pos[:2], dtype=float)
        if args.goal is not None:
            rn = RoadNetwork()
            t_road = time.time()
            while not rn.ready and time.time() - t_road < 90.0:
                try:
                    if rn.build(conn.bng):
                        break
                except Exception:
                    pass
                time.sleep(1.0)
            if rn.ready:
                # Start the route at the nearest ROAD node, not the raw
                # (possibly off-road) ego/teleport point: interpolating the
                # route from an off-road start crosses terrain before the
                # car reaches the road, and leaves the auto-snap disabled
                # (route[0] == start so d0 == 0).  Snap will then place the
                # car on the road and face it along the route.
                n0 = rn.nodes[rn._nearest(p0)]
                _rwe = rn.route_with_edges(
                    n0, np.asarray(args.goal, dtype=float))
                if _rwe[0] is not None and len(_rwe[0]) >= 4:
                    nav_route = np.asarray(_rwe[0][:, :2], dtype=float)
                    # Real-road reference for the end-zone steering: cut
                    # the off-road goal tail so the last metres point
                    # along the actual road (see _trim_backtrack).
                    nav_route_ref = _trim_backtrack(nav_route)
                    road_left = _rwe[1]
                    road_right = _rwe[2]
                    dseg = np.linalg.norm(np.diff(nav_route, axis=0), axis=1)
                    print(f"[fsd-drive] road-graph route: "
                          f"{len(nav_route)} pts, "
                          f"{float(np.sum(dseg)):.1f} m "
                          f"({rn.info})")
                else:
                    print("[fsd-drive] road-graph A* found no route; "
                          "falling back to in-game nav route")
            else:
                print("[fsd-drive] road graph unavailable; "
                      "falling back to in-game nav route")
        if nav_route is None and args.goal is not None:
            conn.bng.control.queue_lua_command(
                "core_groundMarkers.setPath({vec3(%.3f, %.3f, 0)})\n"
                "return 'ok'" % (float(args.goal[0]), float(args.goal[1])),
                response=True)
            time.sleep(0.8)
            nav = conn.read_navigation_route()
            if nav is not None and len(nav) >= 4:
                nav_route = np.asarray(nav[:, :2], dtype=float)
                nav_route_ref = _trim_backtrack(nav_route)
        elif nav_route is None:
            nav = conn.read_navigation_route()
            if nav is not None and len(nav) >= 4:
                nav_route = np.asarray(nav[:, :2], dtype=float)
                nav_route_ref = _trim_backtrack(nav_route)
        if nav_route is not None and len(nav_route) >= 4:
            dseg = np.linalg.norm(np.diff(nav_route, axis=0), axis=1)
            print(f"[fsd-drive] nav route: {len(nav_route)} pts, "
                  f"{float(np.sum(dseg)):.1f} m")
            # If the car was spawned far from the nav route (or the
            # supplied --teleport missed the road), driving straight at
            # the route cuts across terrain/walls and wedges the car
            # (town runs 2026-08-21: 17-24 m off-route starts all ended
            # against a wall at ~1.8 m raw clearance).  Snap determinist-
            # ically onto the route and face along it so the FSD stack
            # verifies route-following instead of a cross-country path.
            st0 = conn.get_state()
            p0 = np.asarray(st0.pos[:2], dtype=float)
            h0 = float(st0.heading)
            f0 = np.array([float(np.cos(h0)), float(np.sin(h0))])
            d0 = np.linalg.norm(nav_route - p0, axis=1)
            i = int(np.argmin(d0))
            # Route start direction: which way the map route leaves the
            # nearest route vertex (forward vs backward along the polyline).
            rdir = None
            if i + 1 < len(nav_route):
                rv = nav_route[i + 1] - nav_route[i]
                if np.linalg.norm(rv) > 1e-9:
                    rdir = rv / np.linalg.norm(rv)
            elif i > 0:
                rv = nav_route[i] - nav_route[i - 1]
                if np.linalg.norm(rv) > 1e-9:
                    rdir = rv / np.linalg.norm(rv)
            heading_ok = True
            if rdir is not None:
                cos_a = float(np.dot(f0, rdir))
                heading_ok = cos_a >= 0.0
            # Snap when the car is off the road centre OR faces more than
            # ~60 deg away from the route direction (town runs 2026-08-22:
            # a 0.4 m off-route start with the nose pointing the wrong way
            # made the local route fall back to a straight line and drove
            # across the town into a wall; mountain run 2026-08-27
            # run_fix22: a start ON the route but facing 71 deg off drove
            # straight onto the grass and spun).  Align the nose along the
            # route so the planner follows the road graph, not a
            # cross-field line - a real stack never starts a run facing
            # across its own lane.
            snap_heading = False
            if rdir is not None:
                cos_a = float(np.dot(f0, rdir))
                snap_heading = cos_a < 0.5
            if d0[i] > 1.5 or not heading_ok or snap_heading:
                rx, ry = float(nav_route[i, 0]), float(nav_route[i, 1])
                if i + 1 < len(nav_route):
                    ndx, ndy = (float(nav_route[i + 1, 0] - rx),
                                float(nav_route[i + 1, 1] - ry))
                else:
                    ndx, ndy = (float(rx - nav_route[i - 1, 0]),
                                float(ry - nav_route[i - 1, 1]))
                h = float(np.arctan2(ndy, ndx))
                # Ground-safe snap: same connector helper as --teleport, so
                # the car is always placed on the real surface (never below
                # terrain) and facing the route direction.  No lateral
                # offset constant here: once the semantic head is warm, the
                # perception placement below moves the car into its own
                # lane from the painted line.
                conn.safe_teleport(rx, ry, heading_deg=math.degrees(h))
                st1 = conn.get_state()
                print(f"[fsd-drive] snapped onto nav route "
                      f"({float(st1.pos[0]):.1f}, {float(st1.pos[1]):.1f}, "
                      f"{float(st1.pos[2]):.1f}) "
                      f"(was {d0[i]:.1f} m off route)")
        else:
            print("[fsd-drive] no nav route set; falling back to "
                  "straight-ahead reference")

        # Proven rule planner as the arbitration fallback: it rounds
        # switchback corners into drivable arcs (the 94.6% rule-autopilot
        # path), so the FSD drive never stops dead at a hairpin apex or
        # full-locks across a kinked map-prior lane.
        rule_planner = LocalPlanner()
        stack = FSDStack(conn, args.runtime,
                         heads=[SemanticHead(), TrafficSignalHead(),
                                ObjectHead()],
                         ring_roles=('front_main',),
                         lane_mode=args.lane_mode,
                         strict_sensor=args.strict,
                         cam_w=args.cam_w, cam_h=args.cam_h,
                         temporal=True, range_every_n=3,
                         # LiDAR every 3rd tick: a fresh scan costs
                         # ~280 ms while a reuse is ~40 ms; the world-frame
                         # hits stay valid for static walls and the
                         # temporal occupancy filter bridges the gap.
                         semantic_every_n=2, object_every_n=2)
        stack.reset_temporal()  # stale occupancy before start must not leak
        # Neural (E2E) planner candidate: a trained end-to-end network
        # over the same perception (rgb + segmentation label + BEV)
        # provides the second planning layer a real FSD stack has -
        # ranked below the layered planner but above the map/rule
        # fallback.  Loading failure (missing weights / bad checkpoint)
        # disables the candidate silently; the drive never depends on it.
        e2e_rt = None
        if not args.no_e2e:
            try:
                e2e_rt = E2ERuntime(args.e2e_model or DEFAULT_E2E_WEIGHTS,
                                    device=None)
                if e2e_rt.loaded:
                    _ck = e2e_rt.ckpt or {}
                    print(f"[fsd-drive] E2E neural planner: "
                          f"{e2e_rt.weights} "
                          f"(img={e2e_rt.img_w}x{e2e_rt.img_h}, "
                          f"history={e2e_rt.history}, "
                          f"epoch={_ck.get('epoch', '?')}, "
                          f"device={e2e_rt.device})")
                else:
                    e2e_rt = None
            except Exception as _e2e_e:
                print(f"[fsd-drive] E2E planner disabled: {_e2e_e}")
                e2e_rt = None
        if e2e_rt is not None:
            e2e_rt.reset()   # no stale frames from before the run
        # Realistic gearbox locked into a forward gear (D).  A real stack
        # never leaves the car in reverse; keep the D input on every
        # control frame so an impact can never leave the gearbox in R.
        fwd_gear = gearbox.forward_gear_input(conn)
        conn.control(throttle=0.0, brake=0.0, steering=0.0,
                     parkingbrake=0.0, gear=fwd_gear)
        conn.step(3)
        # Real-time control: DO NOT pause the sim.  With ticks now
        # ~0.4-0.6 s (after warm-up) the stale-control window is a few
        # metres at cruise and a fraction of a metre at bend speeds, so
        # the sim runs continuously - the car is always moving, which
        # removes the paused-step stutter.  The warm-up crawl and
        # stale-tick scrub below bound the open-loop windows.
        # Pre-warm the pipeline BEFORE driving: the first FSD ticks load
        # YOLO and settle camera/LiDAR (observed 4-6 s, opt11: a 5.3 s
        # tick).  Running that with the car braked means the long tick
        # cannot drive the car open-loop off the road; release once the
        # object head is live (or WARMUP_S elapsed).
        _pw_out = None
        try:
            conn.control(throttle=0.0, brake=1.0, steering=0.0,
                         parkingbrake=1.0, gear=fwd_gear)
            _pw_t0 = time.time()
            while time.time() - _pw_t0 < WARMUP_S:
                _pw_state = conn.get_state()
                _pw_route = local_route(
                    np.asarray(_pw_state.pos[:2], dtype=float),
                    float(_pw_state.heading), nav_route)
                try:
                    _pw_out = stack.tick(st=_pw_state, route_ref=_pw_route)
                    if _pw_out.meta.get("object_head"):
                        break
                except Exception:
                    pass
            print(f"[fsd-drive] pipeline warm: "
                  f"{time.time() - _pw_t0:.1f}s, "
                  f"object_head={bool(_pw_out.meta.get('object_head'))}")
            conn.control(throttle=0.0, brake=0.0, steering=0.0,
                         parkingbrake=0.0, gear=fwd_gear)
            conn.step(3)
        except Exception as _pw_e:
            print(f"[fsd-drive] pre-warm skipped: {_pw_e}")
        # Perception-only lane placement (NO map-centre / offset constant):
        # if the warmed semantic head saw the painted line, put the car in
        # its OWN lane - line right side + lane half width - so a start or
        # restart never parks on (or straddles) the centre line.
        _percep_ok = False
        try:
            if _pw_out is not None and _pw_out.frame is not None:
                _st_sp = conn.get_state()
                _sp_tgt = painted_line_lane_center(
                    _pw_out.head_outputs.get("semantic"), _pw_out.cam,
                    _st_sp.pos, float(_st_sp.heading),
                    ground_z=(float(_st_sp.pos[2])
                              - config.EGO_ORIGIN_GROUND_GAP_M))
                if _sp_tgt is not None:
                    conn.safe_teleport(
                        _sp_tgt[0], _sp_tgt[1],
                        heading_deg=math.degrees(float(_st_sp.heading)))
                    _st_sp2 = conn.get_state()
                    _percep_ok = True
                    print(f"[fsd-drive] perception lane placement -> "
                          f"({float(_st_sp2.pos[0]):.1f}, "
                          f"{float(_st_sp2.pos[1]):.1f}, "
                          f"{float(_st_sp2.pos[2]):.1f}) "
                          f"(painted line right lane, no offset constant)")
        except Exception as _spe:
            print(f"[fsd-drive] perception lane placement failed: {_spe}")
        if not _percep_ok:
            print("[fsd-drive] painted line not perceived; keeping the "
                  "ground-safe route snap")
        conn.control(throttle=0.0, brake=0.0, steering=0.0,
                     parkingbrake=0.0, gear=fwd_gear)
        conn.step(3)
        print(f"[fsd-drive] gearbox realistic, forward gear input = {fwd_gear}")
        rguard = ReverseGuard(threshold_mps=REVERSE_THRESHOLD_MPS,
                              clear_mps=REVERSE_CLEAR_MPS)
        rman = ReverseManeuver(fwd_gear=fwd_gear)
        print(f"[fsd-drive] runtime={stack.mode} FSD pipeline driving "
              f"for {args.seconds}s at {args.speed} m/s "
              f"(lane_mode={args.lane_mode})")

        prev_steer = 0.0  # rate-limited steering state (rule-autopilot convention)
        map_mc_smooth = None   # EMA-smoothed map-prior lane centre
        end_plc_cache = None   # (own-lane centre xy, t_seen) last-good
                               # perception anchor for the end zone
        plc_corr = PaintedLineLateralCorrector(
            max_shift_m=PLC_MAX_SHIFT_M, horizon_m=PLC_HORIZON_M,
            rate_m_s=PLC_RATE_MPS, hold_s=PLC_HOLD_S,
            min_speed_mps=PLC_MIN_SPEED_MPS)
        last_h = None      # previous heading for the yaw-rate steering damper
        climb_t = 0.0      # seconds spent in slope-creep assist
        stuck_t = 0.0    # seconds at near-standstill with a "safe" plan
        t_end = time.time() + args.seconds
        frames = 0
        stopps = 0
        t0 = time.time()
        warmup_until = time.time() + WARMUP_S
        target_sm = float(args.speed)
        plan_sm = float(args.speed)
        prev_thr = 0.0
        prev_brk = 0.0
        gov_brake = False
        # First-frame steering: with last_t set to NOW the first tick has
        # dt ~= 0, smooth_steer cannot move the wheel, and the car runs
        # the whole first ~7 m straight past the corner entry before any
        # steering appears (fix50: steer stayed -0.02 while the car went
        # from the spawn to (726.9,756.8)).  Pretend one control interval
        # has already elapsed so the first frame can steer immediately.
        last_t = time.time() - 1.5
        # Static full-route geometry: the nav route never changes during
        # a run, so dedup/extend/round/resample + arc lengths + per-vertex
        # radii are computed ONCE before the loop instead of every frame
        # (the old per-frame rebuild of the whole rounded route cost
        # ~10-20 ms on a long route, plus a repeated curvature scan for
        # the bend governor).
        route_round = None
        route_arc = None
        route_rad = None
        if nav_route is not None and len(nav_route) >= 4:
            try:
                from beamng_autopilot.planning.local_route import (
                    _dedup as _rdd, _extend_back as _reb,
                    _round_corners as _rrc, _resample as _rrs,
                    CORNER_RADIUS_M as _CR, CORNER_RESAMPLE_M as _CSM)
                route_round = _rrs(_rrc(_reb(_rdd(nav_route[:, :2])), _CR),
                                   _CSM)
                route_arc = np.concatenate(
                    [[0.0], np.cumsum(np.linalg.norm(
                        np.diff(route_round, axis=0), axis=1))])
                _rv1 = np.diff(route_round, axis=0)
                _rn1 = np.linalg.norm(_rv1, axis=1)
                _rn2 = _rn1[1:]
                _rcr = (_rv1[:-1, 0] * _rv1[1:, 1]
                        - _rv1[:-1, 1] * _rv1[1:, 0])
                _rcurv = 2.0 * np.abs(_rcr) / (
                    _rn1[:-1] * _rn2 * (_rn1[:-1] + _rn2))
                route_rad = np.full(len(route_round), np.inf)
                _rm = _rcurv > 1e-6
                route_rad[1:-1][_rm] = 1.0 / _rcurv[_rm]
            except Exception:
                route_round = route_arc = route_rad = None
        hist: list[dict] = []
        _ema_tick = 0.35          # adaptive tick-budget EMA (seeded: a
                                  # typical warm tick is ~0.3-0.4 s)
        while time.time() < t_end:
            _f0 = time.time()
            st = conn.get_state()
            pos = np.asarray(st.pos, dtype=float)
            heading = float(st.heading)
            v = float(st.speed)
            signed = 0.0
            if st.vel is not None and st.dir is not None:
                signed = float(np.dot(
                    np.asarray(st.vel[:2], dtype=float),
                    np.asarray(st.dir[:2], dtype=float)))
            # Control-loop dt in real time: the sim runs continuously, so
            # wall time between ticks IS the driving time.  Clamp the
            # warm-up frames (which can take seconds) so the stuck /
            # reverse / climb state machines do not count a camera stall
            # as seconds of standstill (fix53) - dt caps at 0.5 s.
            now_t = time.time()
            _wall_dt = max(0.0, now_t - last_t)
            last_t = now_t
            dt = min(0.5, max(0.05, _wall_dt))
            # Long-tick brake guard: if the PREVIOUS tick took too long
            # the car just drove open-loop for that long.  Brake now
            # (before the next, possibly long, tick) so no more distance
            # is added uncontrolled; the tick below re-plans and resumes.
            if _wall_dt > STALE_CTRL_S and v > 1.0:
                try:
                    conn.control(throttle=0.0, brake=0.5,
                                 steering=prev_steer, gear=fwd_gear)
                except Exception:
                    pass
            rev_brk, reversing = rguard.decide(signed, dt=dt)
            # Yaw rate for the steering damper: a low-speed car at full
            # lock keeps rotating for seconds after the wheel is centred
            # (the fix37/38 loop - the car swung -130 deg around the
            # junction because the bang-bang pursuit had no damping).
            yaw_rate = 0.0
            if last_h is not None and dt > 0.05:
                _d = float(heading) - float(last_h)
                _d = (_d + math.pi) % (2.0 * math.pi) - math.pi
                yaw_rate = _d / dt
            last_h = float(heading)

            # one full FSD tick -> best trajectory (planned along the
            # LOCAL forward route anchored at the ego; the full nav
            # route tail is a map prior that can cut through a corner
            # wall when the car drifts (town runs 2026-08-21) - the
            # local forward route is what the planner may follow.
            route_local = local_route(pos, heading, nav_route)
            map_lane = None
            if road_left is not None and road_right is not None:
                try:
                    map_lane = map_lane_edges(
                        nav_route, road_left, road_right, pos, heading)
                except Exception:
                    map_lane = None
            if map_lane is not None:
                _mc = np.asarray(map_lane[0], dtype=float)[:, :2]
                if map_mc_smooth is not None and len(_mc) >= 3 \
                        and len(map_mc_smooth) >= 3:
                    try:
                        _a_new = np.concatenate([[0.0], np.cumsum(
                            np.linalg.norm(np.diff(_mc, axis=0), axis=1))])
                        _a_old = np.concatenate([[0.0], np.cumsum(
                            np.linalg.norm(np.diff(map_mc_smooth, axis=0),
                                           axis=1))])
                        _al = np.empty_like(_mc)
                        for _j in range(len(_mc)):
                            _k = int(np.clip(np.searchsorted(
                                _a_old, _a_new[_j]), 0,
                                len(map_mc_smooth) - 1))
                            _al[_j] = map_mc_smooth[_k]
                        _mc = _al * (1.0 - MAP_LANE_EMA) + _mc * MAP_LANE_EMA
                    except Exception:
                        pass
                map_mc_smooth = _mc.copy()
                map_lane = (_mc, map_lane[1], map_lane[2])
            # End-of-route remaining distance, measured on the FULL nav
            # route (arc from the ego's projection to the route END).
            # The old end-stop checked route_local[-1] against
            # nav_route[-1]; right at the end the local window collapses
            # to a straight fallback (its last point jumps ~40 m away),
            # so rem_end became None exactly at the goal, the end-stop
            # released, and the car crept onto the centre line and
            # parked ON it (opt17: lat_left 0.00 -> +0.97 inside oncoming
            # -> parked lat_left 0.00).  Computed here so the steering
            # zone override and the target cap both use it.
            rem_end = None
            if nav_route is not None and len(nav_route) >= 2:
                try:
                    _r2 = np.asarray(nav_route[:, :2], dtype=float)
                    _arc2 = np.concatenate([[0.0], np.cumsum(np.linalg.norm(
                        np.diff(_r2, axis=0), axis=1))])
                    _proj2 = _project_arc(_r2, pos[:2])
                    rem_end = float(max(0.0, _arc2[-1] - _proj2))
                except Exception:
                    rem_end = None
            _ta = time.time()  # local route / map lane / FSD tick split
            # Adaptive shared time budget for the whole FSD tick (see
            # TICK_BUDGET_* above): the stack drops to cached outputs
            # once the tick exceeds the budget so one heavy tick cannot
            # freeze the control loop.
            _budget = float(np.clip(
                _ema_tick * TICK_BUDGET_FRAC,
                TICK_BUDGET_MIN_S, TICK_BUDGET_MAX_S))
            out = stack.tick(st=st, route_ref=route_local,
                             map_lane_override=map_lane,
                             time_budget_s=_budget)
            _tb = time.time()
            _ema_tick = (TICK_BUDGET_EMA * (_tb - _f0)
                         + (1.0 - TICK_BUDGET_EMA) * _ema_tick)
            best = out.best_path
            # Painted centre-line lateral (objective lane-side check).
            # The semantic LINE mask is back-projected to world markings
            # ONCE per frame and shared with the steady lateral corrector
            # below, so a frame of line detection is never done twice.
            _plmarks = None
            if out is not None and out.cam is not None and \
                    getattr(out, "head_outputs", None):
                try:
                    _semx = out.head_outputs.get("semantic")
                    if _semx is not None and \
                            "line" in getattr(_semx, "masks", {}):
                        _plmarks = painted_line_markings(
                            _semx, out.cam, pos, float(heading),
                            ground_z=(float(pos[2])
                                      - config.EGO_ORIGIN_GROUND_GAP_M
                                      if len(pos) > 2 else None))
                except Exception:
                    _plmarks = None
            line_lat = _painted_line_lat(out, pos, heading, _plmarks)

            # safety arbitration on the chosen path: evaluate against the
            # tick's FUSED occupancy (the planner's own vector space), not
            # a fresh empty grid - an empty grid read every path as
            # "grazes obstacle" and kicked the FSD path out on the first
            # bend (observed town run 2026-08-21).
            grid = OccupancyGrid(stack.grid_n, stack.grid_n,
                                 stack.grid_res,
                                 origin=(float(pos[0]), float(pos[1])),
                                 heading=heading)
            # Lane reference for the safety monitor: prefer the stack's
            # vision/LiDAR drivable centreline (the lane the planner
            # actually follows); the map nav route only when the sensor
            # lane is not available.
            local = route_local
            if out.lane_ref is not None and len(out.lane_ref) >= 4:
                local = np.asarray(out.lane_ref, dtype=float)
            if out.bev is not None and out.bev.shape == grid.occupancy.shape:
                grid.occupancy[:] = np.asarray(out.bev, dtype=np.float32)
                grid.obstacle[:] = (np.asarray(out.bev) >= 0.6
                                    ).astype(np.uint8)
                if out.drivable is not None and \
                        out.drivable.shape == grid.drivable.shape:
                    grid.drivable[:] = np.asarray(out.drivable)
                if out.observed is not None and \
                        out.observed.shape == grid.drivable.shape:
                    grid.observed[:] = np.asarray(out.observed)
                scene = Scene(pos=pos, heading=heading, grid=grid,
                              route=local,
                              lane_ref=local,
                              lane_left=out.lane_left,
                              lane_right=out.lane_right,
                              lane_width=out.lane_width,
                              target_speed=args.speed)
                verd = monitor.evaluate(scene, best,
                                        planner_age_s=0.0)
            else:
                verd = monitor.evaluate(Scene(pos=pos, heading=heading),
                                        best)
            _tc = time.time()

            # Neural (E2E) planner candidate: the trained end-to-end
            # network consumes the same perception the layered planner
            # saw (front RGB + segmentation label + BEV) and regresses
            # an ego-relative trajectory, inverse-transformed to world
            # here.  It is arbitrated BELOW the layered planner but
            # ABOVE the map/rule fallback - a real FSD stack ranks its
            # neural planner over the kinematic backup.  Only the
            # trajectory steers the car; the net's raw action is
            # telemetry-only, and the safety monitor re-verifies the
            # path before arbitration can select it.
            e2e_path = None
            e2e_safe = False
            e2e_ms = None
            e2e_act = None
            e2e_ext = None
            _ve = None
            if e2e_rt is not None:
                try:
                    e2e_path, e2e_act, e2e_ms = e2e_rt.step(
                        out, pos, heading, float(v))
                    if e2e_path is not None and len(e2e_path) >= 2:
                        _ve = monitor.evaluate(scene, e2e_path,
                                               planner_age_s=0.0)
                        e2e_safe = bool(_ve.safe)
                        e2e_ext = float(np.hypot(
                            e2e_path[-1, 0] - pos[0],
                            e2e_path[-1, 1] - pos[1]))
                except Exception:
                    e2e_path = None
                    e2e_safe = False

            # planner arbitration: FSD path first; when the layered
            # planner declined (even to minimal risk) fall back to the
            # rule straight-ahead reference IN WORLD COORDINATES - the
            # car must not stop dead on a transient "no drivable path"
            # unless the rule path is also unusable (then and only then
            # a minimal-risk stop).  A body-frame reference handed to
            # PurePursuit points at a wrong world target and spins the
            # car (the "dumb reversing" seen in probes).
            # The rule fallback must also be a path the car can actually
            # drive from here: anchored at the ego and heading forward.  A
            # mis-anchored map prior sitting metres away must not be an
            # excuse to push the wall - when no drivable path exists, the
            # correct FSD behaviour is a minimal-risk stop (town runs
            # 2026-08-21 pushed a wall under a far-away route reference).
            # Proven rule-autopilot fallback: the LocalPlanner rounds
            # switchback corners and keeps the car in its own lane, so the
            # FSD drive does not stop dead at a hairpin apex when the
            # layered planner declines every kinked map-prior candidate.
            # The path must still be ego-anchored and head forward; the
            # FSD safety monitor re-verifies it below (and can stop).
            rule_ref = None
            _need_rule = (best is None or len(best) < 2 or not verd.safe)
            # FSD realism (strict): with no PAIRED perception lane the
            # car must stop, not drive the map/nav route through the rule
            # fallback (docs/fsd_realism.md §4).  The rule planner below
            # is exactly that map fallback, so it is disabled here.
            _strict_no_lane = bool(args.strict and
                                   str(out.meta.get("lane_src_sel", ""))
                                   != "sensor")
            if _strict_no_lane:
                _need_rule = False
            if _need_rule and nav_route is not None and len(nav_route) >= 2:
                try:
                    _fwd = (np.asarray(st.dir[:2], dtype=float)
                            if st.dir is not None else np.array(
                                [math.cos(heading), math.sin(heading)]))
                    _nidx = nearest_route_point(nav_route[:, :2], pos, _fwd)
                    _rd, _rblk = rule_planner.plan(
                        np.asarray(nav_route[:, :2], dtype=float), [],
                        pos, heading, _nidx,
                        sensor_lane=None, road_rule=None, cross_solid=False)
                    if _rd is not None and len(_rd) >= 2 and not _rblk:
                        rule_ref = anchored_rule_ref(
                            pos, heading, np.asarray(_rd, dtype=float)[:, :2])
                except Exception:
                    rule_ref = None
            chosen = arbitrate(
                best, rule_ref,
                fsd_safe=verd.safe and best is not None and len(best) >= 2,
                e2e_path=e2e_path,
                e2e_safe=e2e_safe,
                prefer_rule=False)
            # Re-verify the verdict against the path the car actually
            # runs: the FSD verdict above was computed on the FSD best
            # (which may be None / minimal-risk).  Arbitration then
            # handed a drivable rule path to control, but the old
            # minimal-risk stop stayed latched and braked the fallback
            # to a standstill every frame (run_fix6: src=rule with
            # brk=1.0 and v=0 for 60 s after a transient no-path frame).
            if chosen.path is not None and len(chosen.path) >= 2 and \
                    (best is None or len(best) < 2 or not verd.safe):
                try:
                    if chosen.source == "e2e" and _ve is not None:
                        # already verified this tick against the same
                        # scene; reuse to avoid a second evaluation
                        verd = _ve
                    else:
                        verd = monitor.evaluate(scene, chosen.path,
                                                planner_age_s=0.0)
                except Exception:
                    pass
            # End-zone steering reference: inside the final stop zone the
            # raw route tail sits ON the road centre line at the
            # destination (the road ends / edges swap there) and the
            # near-end map lane can collapse onto the centreline -
            # following it parked the car ON the line (opt17: lat_left
            # 0.00 -> +0.97 inside oncoming -> parked lat_left 0.00).
            # The stop reference is therefore a STRAIGHT ray whose
            # LATERAL anchor comes from the perceived painted line
            # (own-lane centre - perception only, no nav-route offset),
            # and which falls back to holding the current heading when
            # no line is visible, so the car brakes to a stop centred
            # in its lane instead of turning onto the line.
            steer_path = chosen.path
            _end_ref = 0  # 0=straight hold 1=last-good perception hold
                          # 2=live perception (telemetry)
            _dir_src = "none"
            if rem_end is not None and rem_end < END_PULL_START_M:
                _bear = None
                _anchor = np.asarray(pos[:2], dtype=float)
                # Perception-only end-zone lateral reference (FSD
                # realism rule 2026-09-03: lateral placement from a
                # nav route + offset is BANNED - a real self-driving
                # stack puts the car where the SENSORS say its lane
                # is).  When the semantic head sees the painted line,
                # aim the straight stop reference at the perceived
                # own-lane centre (line right side + lane half width,
                # the same perception helper as the start placement),
                # so the final stop converges into the lane instead of
                # riding the route centreline.  If the line is
                # invisible (faded / degenerate road end) the car keeps
                # converging to the LAST perceived own-lane centre for
                # a short window, and only falls back to holding the
                # current heading straight when that expires - still no
                # map lateral pull anywhere in the chain.
                _plc = None
                try:
                    _sem = (out.head_outputs.get("semantic")
                            if out is not None
                            and getattr(out, "head_outputs", None)
                            else None)
                    if _sem is not None and out.cam is not None:
                        # Own-lane centre = line right side + measured
                        # lane half width.  The half width comes from the
                        # PAIRED sensor lane when it is live (perception),
                        # else the same 1.5 m default as start placement -
                        # never a map-centre offset constant.
                        _lh = 1.5
                        try:
                            if out is not None:
                                _lw = float(getattr(out, "lane_width",
                                                    0.0) or 0.0)
                                _sel = str(out.meta.get("lane_src_sel", ""))
                                if _lw >= 2.4 and _sel == "sensor":
                                    _lh = float(np.clip(_lw / 2.0,
                                                        1.2, 2.6))
                        except Exception:
                            _lh = 1.5
                        _plc = painted_line_lane_center(
                            _sem, out.cam, pos, float(heading),
                            ground_z=(float(pos[2])
                                      - config.EGO_ORIGIN_GROUND_GAP_M
                                      if len(pos) > 2 else None),
                            lane_half_m=_lh, marks=_plmarks)
                except Exception:
                    _plc = None
                # Travel orientation for the stop ray - PERCEPTION FIRST,
                # never a lateral offset.  1) the painted line's own
                # direction (the sensors' answer to "which way is my
                # lane"); 2) the paired sensor lane centreline's local
                # heading; 3) the nav route tangent as a plain
                # orientation fallback (the route tail folds onto the
                # centreline at road ends, which is exactly why the nose
                # used to park angled); 4) the current heading.
                _dir3 = None
                if out is not None:
                    try:
                        if _sem is not None and out.cam is not None:
                            _pd = painted_line_direction(
                                _sem, out.cam, pos, float(heading),
                                ground_z=(float(pos[2])
                                          - config.EGO_ORIGIN_GROUND_GAP_M
                                          if len(pos) > 2 else None))
                            if _pd is not None:
                                _dir3 = np.asarray(_pd[:2], dtype=float)
                                _dir_src = "painted"
                        if _dir3 is None and \
                                str(out.meta.get("lane_src_sel", "")) \
                                == "sensor":
                            # Sensor lane centreline local heading; sanity-
                            # gate it to the travel direction so a stray
                            # geometry read cannot aim the stop sideways.
                            _dl = polyline_dir_at(out.lane_ref, _anchor)
                            _hf = np.array([math.cos(float(heading)),
                                            math.sin(float(heading))])
                            if _dl is not None and \
                                    float(_dl @ _hf) >= 0.5:
                                _dir3 = _dl
                                _dir_src = "sensor_lane"
                    except Exception:
                        _dir3 = None
                if _dir3 is None:
                    _nav_ref = nav_route_ref if nav_route_ref is not None \
                        else nav_route
                    if _nav_ref is not None and len(_nav_ref) >= 4:
                        _r3 = np.asarray(_nav_ref[:, :2], dtype=float)
                        _d3 = np.linalg.norm(
                            _r3 - _anchor[None, :], axis=1)
                        _i3 = int(np.argmin(_d3))
                        _i3a = max(0, _i3 - 2)
                        _i3b = min(len(_r3) - 1, _i3 + 2)
                        _tv3 = _r3[_i3b] - _r3[_i3a]
                        _L3 = float(np.linalg.norm(_tv3))
                        if _L3 > 1e-9:
                            _dir3 = _tv3 / _L3
                            _dir_src = "route"
                if _dir3 is None:
                    _dir3 = np.array([math.cos(float(heading)),
                                      math.sin(float(heading))])
                _ref_xy = None
                if _plc is not None:
                    _ref_xy = np.asarray(_plc, dtype=float)[:2]
                    end_plc_cache = (_ref_xy, time.time())
                    _end_ref = 2
                elif end_plc_cache is not None:
                    # Line dropout: reuse the last perceived own-lane
                    # centre while it is still fresh AND the car is
                    # still near the same lane line (pedal to the
                    # cached straight reference, projected ahead of the
                    # ego), so the stop keeps converging instead of
                    # freezing at the entry offset.
                    _cxy, _t_seen = end_plc_cache
                    _age = time.time() - float(_t_seen)
                    _s_proj = float((_anchor - _cxy) @ _dir3)
                    _perp = float((_anchor - _cxy)
                                  @ np.array([-_dir3[1], _dir3[0]]))
                    if _age <= END_PLC_HOLD_S \
                            and _s_proj >= -1.0 \
                            and _s_proj <= END_PLC_MAX_FWD_M \
                            and abs(_perp) <= END_PLC_MAX_LAT_M:
                        _ref_xy = _cxy
                        _end_ref = 1
                if _ref_xy is not None:
                    _tgt3 = _ref_xy + _dir3 * 5.0
                    _v3 = _tgt3 - _anchor
                    if float(np.linalg.norm(_v3)) > 1e-6:
                        _bear = float(math.atan2(_v3[1], _v3[0]))
                if _bear is None:
                    # No live line, no fresh last-good anchor: hold the
                    # current heading straight instead of following the
                    # degenerate tail or any map prior.
                    _bear = float(heading)
                _f3 = np.array([math.cos(_bear), math.sin(_bear)])
                steer_path = _anchor + _f3 * np.arange(
                    0.0, 8.0, 0.8)[:, None]
            # Steady painted-line lateral corrector (cruising only): the
            # end zone above already owns its perception reference, so the
            # nudge engages while the car is driving normally.  When the
            # map lane keeps leading (lane_src_sel != sensor) but the
            # semantic LINE mask gives a confident own-lane centre, shift
            # the near path toward it at a bounded rate - perception pulls
            # the car into its own lane instead of hugging the centre
            # line.  The corrector holds the last perceived shift across a
            # line dropout and decays it, so the car never jerks back to
            # the map prior mid-line.  Rule-source frames (FSD declined,
            # obstacle/unsafe fallback) must not get a superimposed
            # centring pull - the fallback path already carries its own
            # avoidance shape, so the shift just holds then decays.
            _plc_shift = 0.0
            _plc_desired = None
            _plc_active = painted_line_correction_active(
                str(out.meta.get("lane_src_sel", "")),
                str(chosen.source),
                rem_end, END_PULL_START_M)
            if _plc_active:
                try:
                    _sem0 = (out.head_outputs.get("semantic")
                             if out is not None
                             and getattr(out, "head_outputs", None)
                             else None)
                    if _sem0 is not None and out.cam is not None:
                        _olc = painted_line_lane_center(
                            _sem0, out.cam, pos, float(heading),
                            ground_z=(float(pos[2])
                                      - config.EGO_ORIGIN_GROUND_GAP_M
                                      if len(pos) > 2 else None),
                            marks=_plmarks)
                        if _olc is not None:
                            _plc_desired = plc_corr.desired_shift(
                                _olc, pos, float(heading),
                                max_shift_m=PLC_MAX_SHIFT_M)
                except Exception:
                    pass
                _plc_shift = plc_corr.update(_plc_desired, dt, v)
                if steer_path is not None \
                        and abs(_plc_shift) >= PLC_MIN_ENGAGE_M:
                    steer_path = plc_corr.apply(
                        steer_path, pos, float(heading))
            steer = 0.0
            pp_alpha = None
            pp_tgt = None
            ff_steer = 0.0
            if steer_path is not None and len(steer_path) >= 2:
                # Same conversion as the proven rule autopilot: PurePursuit
                # returns the steering ANGLE (rad), BeamNG expects a
                # normalized input with the OPPOSITE sign (left target ->
                # positive angle -> negative input).  Feeding the raw angle
                # (as before 2026-08-22) steered the car the WRONG WAY at
                # the town junction - the route bent left, the raw positive
                # steer was applied as right, yaw swung -32 -> -70 deg and
                # every candidate crossed the lane boundaries -> wedged.
                # Also rate-limit like m5_autopilot so the wheel does not
                # slam from lock to lock on a flickering reference.
                # Speed-adaptive lookahead like the proven rule autopilot:
                # a fixed 5 m lookahead at the hairpin lands the target ON the
                # bend instead of past it, so the wheel barely turns and the
                # car understeers off the road before the corner (mountain run
                # 2026-08-26 run_fix11: at (727.1,757.5) the controller only
                # asked -0.12 and the car missed the first hairpin, then only
                # looping arcs were left).  Computed from the BASE lookahead
                # each frame (no compounding ratchet).
                pp.lookahead = float(np.clip(
                    5.0 + 0.55 * max(0.0, v), 4.0, 16.0))
                steer_rad, pp_tgt, pp_near = pp.steering(
                    pos, heading, np.asarray(steer_path))
                steer_rad = float(steer_rad)
                ff_steer = _path_curvature_ff(steer_path, pos, heading)
                new_steer = float(np.clip(
                    -steer_rad / 0.6 + ff_steer, -1.0, 1.0))
                # Speed-adaptive steering cap (same as the proven rule
                # autopilot): at speed a full-lock correction swings the
                # car far past the lane direction (the FSD runs showed
                # 40-60 deg over-rotation at the junction exits, mountain
                # runs 2026-08-27 fix31/32).  The rule autopilot caps the
                # wheel by v^2 so high-speed corrections stay gentle.
                v_sq = max(v * v, 2.0)
                steer_cap = max(0.10, min(1.0, 5.0 * 2.9 / v_sq / 0.6))
                # Low-speed cap: full lock at 2 m/s is a ~2.9 m radius
                # circle and builds a yaw rate the slow control loop
                # cannot catch - the car swings around the junction
                # instead of converging onto the lane.  0.45 (~10.7 m
                # radius) cannot track the 8 m hairpin fillet, so the
                # first -110 -> -24 deg bend was run wide every time
                # (fix44: lat_l=-7.7 at the apex, then reverse-loops).
                # 0.55 (~8.8 m radius) matches the hairpin geometry; the
                # old 0.55 over-rotation (fix40, 9.7 m/s downhill) is
                # now blocked by the hard speed governor below.
                steer_cap = min(steer_cap, 0.55)
                new_steer = float(np.clip(new_steer, -steer_cap, steer_cap))
                # Yaw-rate damper: oppose a fast rotation that is not
                # being commanded (left rotation -> steer right).  It must
                # NOT fight a hard commanded turn - in the hairpin the car
                # needs ~0.35 rad/s of yaw and the damper cut the full-lock
                # input from -0.55 to -0.46, so the car ran wide off the
                # road (fix49: lat_left -7.5 m at the apex).  Only damp
                # while the wheel is not already at a strong commanded
                # angle.
                if abs(yaw_rate) > 0.3 and abs(new_steer) < 0.35:
                    new_steer = float(np.clip(
                        new_steer + 0.25 * yaw_rate,
                        -steer_cap, steer_cap))
                steer = smooth_steer(prev_steer, new_steer, dt,
                                 rate=0.8)
                prev_steer = steer
                _tv = np.asarray(pp_tgt, dtype=float)[:2] - pos[:2]
                pp_alpha = round(float(math.degrees(
                    math.atan2(_tv[1], _tv[0]) - heading)), 1)

            # Longitudinal plan from the FULL RAW nav route, never the
            # local resampled window: the local window starts ON the
            # corner and its Catmull-Rom resample rounds the hairpin into
            # a 4-5 m sweep, so the curvature profile reads 4-5 m/s into
            # the bend and the car understeers off the road (mountain run
            # 2026-08-23, run_fix6: plan_v 4.9-6.0 at the first hairpin,
            # car left the road 3-10 m west of the route and never came
            # back).  The raw road-graph polyline keeps the 90-degree
            # kink - its look-ahead profile caps the entry speed at
            # ~1.7 m/s, which is the speed the bend can actually take.
            if route_round is not None and route_arc is not None \
                    and route_rad is not None:
                _rfull = route_round
                _ga = route_arc
                # Arc-length projection (not nearest-vertex): the nearest
                # vertex can flip between two close route samples frame to
                # frame, which snapped the sampled profile speed by whole
                # m/s on straight segments; the projection stays stable.
                _proj_s = float(_project_arc(_rfull, pos[:2]))
                try:
                    # Profile the ROUNDED full route, not the raw road-graph
                    # polyline: the graph collapses the first hairpin into a
                    # sharp vertex whose curvature profile caps the bend at
                    # ~1.7 m/s - at that speed the tyres scrub and the car
                    # cannot even turn (steering probe 2026-08-27).  The
                    # rounded route (same 8 m fillet the lane centre uses)
                    # lets the bend be taken at a speed the steering can
                    # actually execute.  The rounded polyline, its arc
                    # lengths and per-vertex radii are precomputed once
                    # before the loop (route_round/route_arc/route_rad).
                    _sp_raw = _spf_raw(
                        _rfull, scene,
                        target_speed=float(args.speed),
                        # Corridor-open clutter (dense junction/end-zone
                        # LiDAR that still leaves a free band) must not
                        # pin the full-route plan to the 1 m/s MIN_SPEED -
                        # the safety monitor keeps the same cruise floor
                        # when its corridor is open.
                        obstacle_min_speed=(
                            max(_PROF_MIN_SPEED,
                                0.4 * float(args.speed))
                            if verd.corridor_open else _PROF_MIN_SPEED))
                    if len(_sp_raw):
                        # The full-route profile is indexed along the WHOLE
                        # route; [0] is the speed at the ROUTE START, not at
                        # the car.  Sample the profile at the nearest route
                        # point so a hairpin 100 m into the route still caps
                        # the speed when the car reaches it (run_fix25:
                        # plan_speed stayed at the start speed into the bend).
                        _i = int(np.clip(
                            int(np.searchsorted(_ga, _proj_s)),
                            0, len(_rfull) - 1))
                        out.best_speed = float(_sp_raw[_i])
                        out.meta["plan_src"] = "nav_round"
                except Exception:
                    pass
                # Tight-bend entry governor: the ~1.4 s control tick lets the
                # car overshoot the profiled corner speed by ~+1 m/s mid-tick
                # (fix45: 4.4 target -> ~5.5 actual at the first hairpin, ran
                # wide off the left edge).  For a bend tighter than 15 m in
                # the next 12 m, cap the plan speed at sqrt(1.5*R) so the
                # actual peak stays near 4 m/s, where the 0.55 steering cap
                # (~8.8 m radius) can track the 8 m hairpin fillet.
                try:
                    _gi = int(np.clip(
                        int(np.searchsorted(_ga, _proj_s)),
                        0, len(_rfull) - 1))
                    _ghi = int(np.searchsorted(_ga, _proj_s + 12.0))
                    _lo = max(1, _gi)
                    _hi = min(_ghi, len(route_rad) - 2) + 1
                    _rmin = 1e9
                    if _lo < _hi:
                        _rmin = float(np.min(route_rad[_lo:_hi]))
                    if _rmin < 15.0:
                        # Turn-angle gate: only a REAL bend deserves the
                        # hairpin speed cap.  A rounded junction corner /
                        # resample wiggle can measure R~3 m over 12 m while
                        # turning <30 deg total; capping there parked the
                        # plan at sqrt(1.3*3)=1.97 and the car crawled 5+ s
                        # through a widening junction (fsd opt23 t=36-45,
                        # lat_right 1.5->4.2).  A true hairpin turns
                        # 60-180 deg over the same window.
                        if _route_turn_deg(route_round, _lo, _hi) >= \
                                BEND_GOV_MIN_TURN_DEG:
                            # Floor the implied radius: the 0.8 m resample
                            # can measure a hairpin fillet edge as R~0.8 m
                            # (three nearly-collinear points), which caps
                            # the plan at ~1 m/s and stands the car dead
                            # on the approach (fix65: plan=1.00 at the
                            # second bend, v=5.6 -> brake-to-0).  Real
                            # roads never bend tighter than ~3 m; anything
                            # smaller is a sampling artifact.
                            _rmin = max(_rmin, 3.0)
                            out.best_speed = float(min(
                                out.best_speed, math.sqrt(1.3 * _rmin)))
                            out.meta["plan_src"] = "nav_round+gov"
                except Exception:
                    pass
                # FSD-realism speed cap: slow on what the sensors SEE.
                # The nav-route profile above is navigation intent; when
                # the BEV road mask curves ahead, the plan is capped from
                # PERCEPTION only (docs/fsd_realism.md §2).
                try:
                    if out.drivable is not None:
                        _pg = OccupancyGrid(
                            60, 60, 0.5,
                            origin=(float(pos[0]), float(pos[1])),
                            heading=float(heading))
                        _pg.drivable = np.asarray(out.drivable, dtype=float)
                        _pcap = perception_curve_speed(_pg, out.best_speed)
                        if _pcap < out.best_speed:
                            out.best_speed = _pcap
                            out.meta["plan_src"] = "perception-curve"
                except Exception:
                    pass
            # control from the (possibly degraded) target speed, but never
            # exceed the *planned* speed along the chosen trajectory - the
            # FSD longitudinal plan (bend deceleration, obstacle brake
            # band) must govern the actual pedals.
            plan_speed = out.best_speed if out.best_speed > 0.0 \
                else float(args.speed)
            if _strict_no_lane and chosen.source != "e2e":
                # no perception lane -> minimal-risk stop (never map-drive).
                # The E2E neural path is perception-driven (not the map
                # fallback), so when the monitor green-lit it the car may
                # keep rolling on the learned trajectory.
                plan_speed = 0.0
                plan_sm = 0.0
            # a rule fallback does not get the FSD plan speed; cap it to a
            # cautious creep so the L2 fallback is gentle
            if chosen.source == "rule":
                plan_speed = min(plan_speed, 3.0)
            plan_raw_speed = float(plan_speed)
            # Rate-limit the plan (PLAN_* constants): transient LiDAR
            # clutter at junctions/end zones must not snap the plan
            # 6.0 <-> 1.0 between ticks and make the controller brake
            # then relaunch (opt22).  Real bends still decelerate - the
            # profile drops smoothly over many frames, well inside the
            # down rate - and the safety monitor / force-stop remain the
            # authority for genuine emergencies.
            _dplan = float(np.clip(
                plan_raw_speed - plan_sm,
                -PLAN_DOWN_RATE_MPS2 * dt,
                PLAN_UP_RATE_MPS2 * dt))
            plan_sm = float(plan_sm + _dplan)
            plan_speed = plan_sm
            target = min(verd.target_speed, plan_speed, float(args.speed))
            # Heading-error speed scrub: the nav route is the intent;
            # when the nose drifts off it (oscillation / over-rotation)
            # slow down so the steering loop converges instead of
            # feeding the swing.  Falls back to no-op when the local
            # route bearing cannot be measured.
            _rh_b = _ref_bearing(route_local, pos)
            if _rh_b is not None:
                _hdg_dev = abs((float(heading)
                                - math.radians(_rh_b) + math.pi)
                               % (2.0 * math.pi) - math.pi)
                _hdg_deg = math.degrees(_hdg_dev)
                if _hdg_deg > HEADING_DEV_START_DEG:
                    _k = min(1.0, (_hdg_deg - HEADING_DEV_START_DEG)
                             / (HEADING_DEV_FULL_DEG
                                - HEADING_DEV_START_DEG))
                    _hdg_cap = (HEADING_DEV_FLOOR_MPS
                                + (HEADING_DEV_CAP_MPS
                                   - HEADING_DEV_FLOOR_MPS)
                                * (1.0 - _k))
                    target = min(target, _hdg_cap)
            # No-route guard: without a nav route there is no map prior to
            # keep the car on the road - a straight-line reference drives
            # straight onto grass (opt13 2026-08-28: no route after a game
            # restart, car crept on the grass at 0-4 m/s).  Never cruise
            # without a route; the caller must pass --goal.
            if nav_route is None:
                target = min(target, 1.0)
            # Map road-edge guard: the nav centreline + real DecalRoad
            # edge rows are the map prior for "where the road is".  Once
            # the ego is beyond the local road edge (grass/verge on the
            # right, oncoming lane on the left) the car must not keep
            # driving - crawl at 0.5 m/s; the monitor still stops it if
            # the path is blocked.  The centreline is used instead of the
            # raw edge polylines because edge rows fold at junctions and
            # go stale past the last graph node (town run10: a folded
            # edge corner reported 3.2 m off-road on a straight section
            # while the car sat on the centreline).
            road_off = 0.0
            off_recover = False
            if (nav_route is not None and len(nav_route) >= 2
                    and road_left is not None and road_right is not None):
                _lat2, _beyond2, _hw2 = _route_lateral_off_m(
                    pos, nav_route, road_left, road_right)
                road_off = float(_beyond2)
                if road_off > ROAD_OFF_STOP_M:
                    # Hard stop + return steering (recovery block
                    # below), not a 0.5 m/s grass cruise.
                    target = min(target, ROAD_OFF_CRAWL_MPS)
                    off_recover = True
                elif road_off > ROAD_OFF_EDGE_M:
                    target = min(target, ROAD_OFF_CRAWL_MPS)
                elif road_off > 0.0:
                    target = min(target, ROAD_EDGE_SLOW_MPS)
            # End-of-route (rem_end computed above from the FULL nav route
            # arc, so it never goes None when the local window collapses):
            # ease to a stop while still in the lane instead of parking
            # over the edge line at the road end (opt15).
            if rem_end is not None:
                if rem_end < END_STOP_M:
                    target = 0.0
                elif rem_end < END_START_SLOW_M:
                    target = min(target, END_SLOW_MPS)
            # Warm-up crawl + stale-tick scrub (real-time mode): before
            # the object head is live, or after an unusually long tick,
            # the car has been driving open-loop - keep it slow.
            if time.time() < warmup_until and not out.meta.get("object_head"):
                target = min(target, WARMUP_SPEED_MPS)
            if _wall_dt > STALE_CTRL_S:
                target = min(target, STALE_CTRL_SPEED_MPS)
            # Safety clearance along the CHOSEN path (FSD vector-space
            # safety layer): the grid obstacle layer the planner itself
            # scored against is the authority, so a wall beside the nose
            # that the chosen arc turns away from does not park the car
            # (town corner run 2026-08-21: planner picked a feasible arc,
            # raw LiDAR foliage read 0.5 m and the emergency layer
            # force-stopped every frame).  A path that really is blocked
            # still forces the same stop.  ``inf`` from
            # ``path_grid_clearance_m`` MEANS the path is clear - it must
            # not be treated as "missing" (run 2026-08-22: the fallback
            # replaced a clean-path inf with the raw heading corridor
            # 0.19 m at a town corner and parked a car that was steering
            # fine).  The raw-sensor heading corridor is only the last
            # line when there is NO planned path at all.
            force_stop = False
            fwd_clear = float("inf")
            if chosen.path is not None and len(chosen.path) >= 2:
                fwd_clear = path_grid_clearance_m(chosen.path, grid)
            else:
                fwd_clear = float(out.forward_clearance)
            if np.isfinite(fwd_clear):
                need = emergency_stop_clearance_m(v)
                force_stop, cap = emergency_speed_limit_mps(fwd_clear, need)
                target = min(target, cap if not force_stop else 0.0)
            # Smooth the effective target: ramp toward the raw plan at a
            # bounded rate (sim time), so the corner governor stepping the
            # plan from 6 to 3.2 m/s in one tick cannot flip the pedals.
            # A safety force-stop bypasses the ramp and brakes immediately.
            if force_stop:
                target_sm = 0.0
            else:
                _dmax = SPEED_TARGET_RAMP_MPS * dt
                if target > target_sm:
                    target_sm = min(target, target_sm + _dmax)
                else:
                    target_sm = max(target, target_sm - _dmax)
                # Never cruise above the planned corner speed.  The ramp
                # can still be converging down from a high initial target
                # (first frames), which let the car overshoot the bend
                # plan and trip the hard governor every tick (fix61:
                # v=4.45 against plan 3.23 -> brake 1.0 -> stall ->
                # full throttle again).  Capping the smoothed target by
                # plan_speed keeps the pedals inside the plan from the
                # very first tick.
                target_sm = min(target_sm, plan_speed)
            thr, brk = speed_ctrl.update(
                target_sm, v, dt=min(0.25, max(0.01, dt)))
            # Downhill-start throttle guard: on the descent the car
            # accelerates by gravity alone, so feeding throttle near the
            # plan overshoots the bend speed and the governor then brakes
            # it to a standstill every tick (fix61-64: v 0 -> 4.4 -> 0).
            # Below a low speed, hold the pedal near idle and let gravity
            # bring the speed up to the plan; the controller resumes once
            # the speed is there.
            if v < 2.5 and signed > 0.3 and target_sm <= plan_speed:
                thr = min(thr, 0.25)
            # Downhill acceleration cap: once the car is rolling, full
            # throttle demand plus gravity overshoots the plan in one
            # 0.66 s burst (opt8: 0.80 throttle -> 2.7 -> 7.3 m/s).
            # Cap the pedal while approaching the target so gravity does
            # most of the work; the plan governor trims the rest.
            if v > 2.5 and signed > 0.5 and v < target_sm - 0.5:
                thr = min(thr, 0.35)
            # (The old "corner brake zone" here is gone: with the sim
            # paused and stepped in 0.33 s bursts the speed controller
            # reacts within one burst, while the zone caused an
            # accelerate -> brake-to-stop oscillation - fix54 reached
            # v=3.6 then brk=0.8 stopped it dead at every tick.  The
            # plan-speed governor below still hard-brakes overshoot.)
            # Soft overspeed governor: never let the car exceed the
            # commanded cruise speed by more than 1 m/s regardless of
            # the smoothed pedal state.  The old brk=1.0 here stopped
            # the car DEAD on the downhill, then the controller relaunched
            # with full throttle -> 0 <-> 7.8 m/s bang-bang (opt8).
            # Taper the throttle off between +0.5 and +1.3 m/s overshoot
            # instead of the old hard cut at +1.0.  A hard cut to 0 then
            # a full re-launch made a +/-0.9 m/s speed wave around cruise
            # (opt21: 49 throttle on/off flips in 169 frames); a gradual
            # taper removes the relaunch kick while the plan governor's
            # gentle GOV_BRAKE still trims the overshoot.
            _ov = v - (float(args.speed) + 0.5)
            if _ov > 0.0:
                thr *= float(np.clip((0.8 - _ov) / 0.8, 0.0, 1.0))
            # Plan-speed governor: never let the car exceed the planned
            # corner speed by more than 0.8 m/s even within one tick -
            # the profile alone is sampled at tick boundaries and the
            # car can overshoot a 4.4 m/s hairpin plan to ~5.5 mid-tick.
            if v > plan_speed + GOV_ON_MPS:
                gov_brake = True
            elif not (v > plan_speed + GOV_OFF_MPS):
                gov_brake = False
            if gov_brake:
                thr, brk = 0.0, max(brk, GOV_BRAKE)
            # Stuck detection: the planner can keep reporting "safe" while
            # the car physically cannot move (wedged against a guardrail /
            # embankment after an over-correction).  Holding throttle
            # against the obstruction forever is the "spinning in place"
            # failure - after a couple of seconds at near-standstill with
            # a commanded forward path, treat it as "no forward path" so
            # the bounded reverse escape backs out and re-plans (mountain
            # run 2026-08-27 run_fix31: wedged at (741.2,745.7) with
            # thr=0.53 and v=0 for 50 s).
            if (chosen.path is not None and not force_stop
                    and v < 0.35 and thr > 0.0):
                stuck_t += max(0.0, float(dt))
            else:
                stuck_t = 0.0
            stuck = stuck_t >= 2.5
            # hard stop when no path remains, the raw-sensor forward
            # clearance is inside the braking reserve (never grind into a
            # wall / wedge the car into a too-narrow gap), or the car is
            # stuck spinning against an obstruction
            pb = 0.0  # handbrake: hold the car on a slope while stopped
            # Slope-creep assist: the planner keeps a CLEAR forward path
            # while the car cannot move (e.g. stopped at the bottom of a
            # dip facing uphill).  Full throttle for a bounded window lets
            # it climb before the reverse escape is allowed to arm - the
            # fix41 east-side dip at (756.7,740.6) had fwd=8.6 clear but
            # the stuck detector sent the car backwards downhill instead
            # of giving it torque.
            climb = False
            if (stuck and not force_stop and not off_recover
                    and chosen.path is not None
                    and len(chosen.path) >= 2
                    and np.isfinite(fwd_clear) and fwd_clear > 3.0):
                if climb_t < CLIMB_ASSIST_S:
                    climb = True
                    climb_t += max(0.0, float(dt))
                else:
                    climb_t = 0.0  # give up climbing -> allow reverse
            if (chosen.path is None or force_stop or stuck) \
                    and not climb and not off_recover:
                thr, brk = 0.0, 1.0
                steer = 0.0
                stopps += 1
                pb = 1.0
            if climb:
                thr, brk = 1.0, 0.0
                steer = 0.0
                pb = 0.0
            # Controlled reverse escape: when NO drivable forward path
            # remains (dead-end / wedged nose) and the space BEHIND the
            # car is clear, back up a bounded distance in R, then let the
            # planner re-attempt a forward path.  This is the "reverse to
            # find a feasible path" behaviour; it never reverses while a
            # forward path exists and never reverses blindly (no rear
            # clearance data -> stay stopped).
            has_forward_path = (chosen.path is not None
                                and len(chosen.path) >= 2
                                and not force_stop and not stuck) or climb \
                or off_recover
            # Arrived at the destination: once inside the stop zone and
            # nearly stopped, treat the forward path as present so the
            # reverse escape never backs the car over the line at the end
            # (opt18: rev=1 at the goal moved the car onto the oncoming
            # lane and it parked over the line).
            if rem_end is not None and rem_end < END_STOP_M and v < 0.6:
                has_forward_path = True
            rear_clear_m = None
            if not has_forward_path and grid is not None:
                try:
                    _bh = float(heading) + math.pi
                    _ln = np.array([math.cos(_bh), math.sin(_bh)])
                    _lt = np.array([-_ln[1], _ln[0]])
                    _best = float("inf")
                    for _lat in (0.0, -1.5, 1.5):
                        _o = pos[:2] + _lat * _lt
                        for _ds in np.arange(1.0, 10.0, 0.4):
                            _wx = float(_o[0] + _ds * _ln[0])
                            _wy = float(_o[1] + _ds * _ln[1])
                            _cell = grid.world_to_cell(_wx, _wy)
                            if _cell is not None:
                                _r, _c = int(_cell[0]), int(_cell[1])
                                if (0 <= _r < grid.obstacle.shape[0]
                                        and 0 <= _c < grid.obstacle.shape[1]
                                        and grid.obstacle[_r, _c] > 0):
                                    _best = min(_best, _ds - 0.5)
                                    break
                    rear_clear_m = (float(_best) if _best < float("inf")
                                     else 40.0)
                except Exception:
                    rear_clear_m = None
            rm = rman.decide(has_forward_path=has_forward_path,
                             rear_clear_m=rear_clear_m,
                             signed_speed=signed,
                             pos2d=pos[:2], dt=dt)
            gear_use = fwd_gear
            if rm.active:
                # Bounded reverse: gear R, slow, straight back (steering
                # centred) until the state machine releases the attempt.
                gear_use = rm.gear
                steer = 0.0
                pb = 0.0  # the car must roll for the escape
                if signed >= -0.05:
                    # Gentle R throttle: 0.25 ramped the car to -3 m/s in
                    # ~1.5 s (fix37); 0.10 still reached -3.0 on the
                    # east-side slope (fix41).  0.06 keeps the escape
                    # near the -0.4 m/s target even on a mild grade.
                    # No throttle once the car is ALREADY rolling back
                    # (signed < -0.05): on a downhill slope gravity does
                    # the backing, adding throttle only rolls it further
                    # before the brake catches (east-side roll-back).
                    thr, brk = 0.06, 0.0
                elif signed > rm.target_speed_mps:
                    # Approaching the reverse target: back off the
                    # throttle and brake softly instead of waiting for a
                    # full overshoot (control ticks are ~1.4 s apart).
                    thr, brk = 0.0, 0.6
                else:
                    # Backward speed already beyond the target (downhill
                    # roll): brake hard AND handbrake - 0.8 alone let the
                    # car run away to -7 m/s on a slope (run_fix17).
                    thr, brk, pb = 0.0, 1.0, 1.0
                # Extra stop margin when the rear space is nearly gone.
                if rear_clear_m is not None and rear_clear_m <= 2.0:
                    thr, brk, pb = 0.0, 1.0, 1.0
            elif reversing:
                # Passive reverse guard (unintended backward motion, e.g.
                # a wall bounce): brake, centre the wheel, no throttle.
                # Handbrake too while still moving backwards so a slope
                # cannot roll the car away.
                thr, brk = 0.0, max(brk, float(rev_brk))
                steer = 0.0
                if signed < -0.1:
                    pb = 1.0
            # Off-road recovery: the hard DecalRoad edges are the ground
            # truth for "on the road".  Once the car has LEFT the road
            # (grass/verge on the right, oncoming side on the left) it
            # must not keep driving - creeping on the verge understeered
            # 2 m -> 10 m further away at the town corner (opt24).  Hard
            # stop + hold like the end zone; the driver / next teleport
            # repositions.  Reverse escape is suppressed while off-road
            # (has_forward_path above) so it never backs further off.
            if off_recover:
                thr = 0.0
                brk = max(brk, END_BRAKE)
                steer = 0.0
                if v < 0.4:
                    pb = 1.0
            # End-of-route hard stop + hold: target=0 alone only asks the
            # speed controller for a gentle ramp (brk ~0.16 at 1.3 m/s),
            # so the car rolled through the whole stop zone, the lane
            # reference collapsed onto the centre line and it parked ON
            # the line (opt18: ll -1.00 -> 0.00 at rem 4->0, final parked
            # over the line after the reverse escape).  Inside the stop
            # zone brake hard so the car stops BEFORE the degenerate road
            # end, and hold with the handbrake once stopped.
            if rem_end is not None and rem_end < END_STOP_M:
                thr = 0.0
                brk = max(brk, END_BRAKE)
                if v < 0.5:
                    # Already stopped: centre the wheel.  The pursuit
                    # reference is gone / points sideways once the car
                    # sits in the stop zone, and holding its steering
                    # parks the nose angled across the lane (opt32/33:
                    # hdg -146 vs lane -133, steer pinned at -0.47).
                    steer = 0.0
                if v < 0.4:
                    pb = 1.0
            # Pedal rate limit: the branches above (downhill cap, taper,
            # governor, climb/reverse/hard-stop) can step thr/brk by a
            # whole pedal in one tick - a relaunch then reads as a speed
            # kick (opt23: 13 speed jumps >1.5 m/s per tick).  Ramp the
            # FINAL commanded pedals toward the previous tick's at bounded
            # rates; safety branches bypass on purpose (hard stop, climb,
            # reverse escape, end-zone hold).
            _hard_pedal = bool(
                force_stop or stuck or climb or rm.active or reversing
                or (rem_end is not None and rem_end < END_STOP_M))
            if not _hard_pedal:
                thr, brk = rate_limit_pedal(
                    thr, brk, prev_thr, prev_brk, dt)
            prev_thr, prev_brk = thr, brk
            conn.control(throttle=thr, brake=brk, steering=steer,
                         gear=gear_use, parkingbrake=pb)
            # Lane-position telemetry: signed lateral offset of the ego from
            # each DETECTED lane boundary (left: + = inside oncoming traffic;
            # right: - = off the road edge).  None when the boundary does
            # not extend to the ego or no lane was detected this frame.
            lat_left = lat_right = None
            fwd_lane = np.array([float(np.cos(heading)),
                                 float(np.sin(heading))])
            if out.lane_left is not None:
                try:
                    _ll, _cl = _boundary_lateral(
                        float(pos[0]), float(pos[1]), out.lane_left, fwd_lane)
                    lat_left = round(float(_ll), 3) if _cl else None
                except Exception:
                    pass
            if out.lane_right is not None:
                try:
                    _lr, _cr = _boundary_lateral(
                        float(pos[0]), float(pos[1]), out.lane_right, fwd_lane)
                    lat_right = round(float(_lr), 3) if _cr else None
                except Exception:
                    pass
            # snapshot for offline stability evaluation (safe / degraded
            # ratio over a long route); written once at the end.
            hist.append({
                "t": round(time.time() - t0, 3),
                "pos": [round(float(p), 3) for p in pos[:3]],
                "heading": round(float(heading), 4),
                "speed": round(float(v), 3),
                "signed": round(float(signed), 3),
                "level": str(verd.level),
                "reason": verd.reason or "-",
                "source": str(chosen.source),
                "e2e": int(e2e_path is not None and len(e2e_path) >= 2),
                "e2e_safe": int(bool(e2e_safe)),
                "e2e_ms": (round(float(e2e_ms), 1)
                           if e2e_ms is not None else None),
                "e2e_extent": (round(float(e2e_ext), 2)
                               if e2e_ext is not None else None),
                "e2e_act": ([round(float(a), 3) for a in e2e_act]
                            if e2e_act is not None else None),
                "plan_speed": round(float(plan_speed), 2),
                "plan_raw": round(float(plan_raw_speed), 2),
                "target_sm": round(float(target_sm), 2),
                "plan_src": str(out.meta.get("plan_src", "?")),
                "tick_ms": out.meta.get("tick_ms"),
                "tick_wall_ms": round((_tb - _f0) * 1000.0, 1),
                "budget_s": round(float(_budget), 3),
                "budget_skips": list(
                    out.meta.get("tick_budget_skips") or []),
                'frame_ms': {
                    'local': round((_ta - _f0) * 1000.0, 1),
                    'tick': round((_tb - _ta) * 1000.0, 1),
                    'grid_mon': round((_tc - _tb) * 1000.0, 1),
                    'rest': round((time.time() - _tc) * 1000.0, 1),
                },
                "throttle": round(float(thr), 4),
                "brake": round(float(brk), 4),
                "steer": round(float(steer), 4),
                "reversing": int(bool(reversing)),
                "rev_state": str(rman.state),
                "rev_active": int(bool(rm.active)),
                "rear_clear": (round(float(rear_clear_m), 2)
                               if rear_clear_m is not None else None),
                "fwd_clear": float(out.forward_clearance)
                    if np.isfinite(out.forward_clearance) else None,
                "emergency": int(bool(force_stop)),
                "stuck": int(bool(stuck)),
                "lane_src": str(out.meta.get("lane_src", "?")),
                "lane_mode": str(args.lane_mode),
                "lane_reject": str(out.meta.get("lane_reject_reason", "")),
                "lane_sel": str(out.meta.get("lane_src_sel", "")),
                "lane_paired": int(out.meta.get("lane_paired", 0)),
                "n_object_obstacles": int(
                    out.meta.get("n_object_obstacles", 0)),
                "object_head": int(out.meta.get("object_head", 0)),
                "lane_dev_m": round(float(getattr(verd, "lane_dev_m", 0.0)), 3),
                "lat_left": lat_left,
                "lat_right": lat_right,
                "kind": str(out.meta.get("planner", {}).get("kind", "?")),
                "cost": round(float(out.meta.get("planner", {}).get("cost", 0.0)), 3),
                "n_eval": int(out.meta.get("planner", {}).get("n_eval", 0)),
                "pp_alpha": pp_alpha,
                "yaw_rate": round(float(yaw_rate), 3),
                "ff_steer": round(float(ff_steer), 3),
                "climb": int(bool(climb)),
                "road_off": round(float(road_off), 3),
                "off_recover": int(off_recover),
                "rem_end": (round(float(rem_end), 2)
                            if rem_end is not None else None),
                "mon_target": round(float(verd.target_speed), 2),
                "closest_obs": (round(float(verd.closest_obs_m), 2)
                                if verd.closest_obs_m < 900.0 else None),
                "pp_tgt": ([round(float(v), 2) for v in pp_tgt[:2]]
                           if pp_tgt is not None else None),
                "line_lat": line_lat,
                "plc_active": int(_plc_active),
                "plc_shift": round(float(_plc_shift), 3),
                "plc_desired": (round(float(_plc_desired), 3)
                                if _plc_desired is not None else None),
                "end_ref": _end_ref,
                "end_dir_src": _dir_src,
                "lane_bear": _ref_bearing(out.lane_ref, pos),
                "route_bear": _ref_bearing(route_local, pos),
                "best_bear": _ref_bearing(out.best_path, pos),
            })
            # Real-time cadence: the sim keeps running; pace the control
            # loop to ~2 Hz so the car is never left without a fresh
            # command for long.
            _elapsed = time.time() - _f0
            _slack = (1.0 / REALTIME_CTRL_HZ) - _elapsed
            if _slack > 0.0:
                time.sleep(_slack)
            frames += 1
            if frames % 4 == 1:
                _e2e_s = (f"e2e={e2e_ms:.0f}ms "
                          if e2e_ms is not None else "")
                print(f"[fsd-drive] t={time.time()-t0:5.1f} v={v:4.1f} "
                      f"level={verd.level} src={chosen.source:4s} "
                      f"reason={verd.reason or '-':22s} "
                      f"steer={steer:+.2f} thr={thr:.2f} "
                      f"plan_v={plan_speed:.1f} {_e2e_s}"
                      f"rev={int(reversing)} signed={signed:+.2f} "
                      f"lane={out.meta.get('lane_src', '?')}/"
                      f"{'P' if out.meta.get('lane_paired') else '1'} "
                      f"dev={getattr(verd, 'lane_dev_m', 0.0):.2f}")
        _ll = [f["line_lat"] for f in hist if f.get("line_lat") is not None]
        if _ll:
            _arr = np.asarray(_ll, dtype=float)
            print(f"[fsd-drive] painted line lateral (left=+): "
                  f"mean={_arr.mean():+.2f}m p50="
                  f"{np.percentile(_arr, 50):+.2f}m "
                  f"min={_arr.min():+.2f} max={_arr.max():+.2f} "
                  f"({int((_arr < -0.5).sum())} frames car left of line)")
        _ps = [float(f.get("plc_shift", 0.0)) for f in hist]
        _nplc = sum(1 for f in hist
                    if abs(f.get("plc_shift", 0.0)) > PLC_MIN_ENGAGE_M)
        if _nplc:
            print(f"[fsd-drive] painted-line steady corrector: "
                  f"active {_nplc}/{len(hist)} frames, "
                  f"mean|shift|={np.mean(np.abs(_ps)):.2f}m")
        _er = [(f.get("end_ref", 0), f.get("rem_end")) for f in hist
               if f.get("rem_end") is not None
               and f["rem_end"] < END_PULL_START_M]
        if _er:
            _n_live = sum(1 for _r, _ in _er if _r == 2)
            _n_hold = sum(1 for _r, _ in _er if _r == 1)
            _n_flat = sum(1 for _r, _ in _er if _r == 0)
            print(f"[fsd-drive] end-zone ref: live-perception={_n_live} "
                  f"last-good-hold={_n_hold} straight-hold={_n_flat}")
        print(f"[fsd-drive] done: {frames} frames, {stopps} stops")
        _n_skip_fr = sum(1 for f in hist if f.get("budget_skips"))
        if _n_skip_fr:
            _n_skip_hd = sum(len(f.get("budget_skips") or [])
                             for f in hist)
            _max_budget = max(float(f.get("budget_s") or 0.0)
                              for f in hist)
            print(f"[fsd-drive] tick budget: {_n_skip_fr} frames "
                  f"deferred {_n_skip_hd} heavy head(s) "
                  f"(budget capped at {_max_budget:.2f}s)")
    finally:
        # ensure the car stops
        try:
            conn.control(throttle=0.0, brake=1.0, steering=0.0,
                         gear=locals().get("fwd_gear"))
            conn.step(3)
        except Exception:
            pass
        # Remove the Tech camera/LiDAR sensors so a next attach process
        # does not pile up sensors in the running game (leftover sensors
        # made later camera polls fail intermittently).
        try:
            stack.close()
        except Exception:
            pass
        conn.close()
        if hist and args.out:
            try:
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                Path(args.out).write_text(
                    json.dumps(hist, ensure_ascii=False), encoding="utf-8")
                print(f"[fsd-drive] telemetry -> {args.out} ({len(hist)} frames)")
            except Exception as _e:
                print(f"[fsd-drive] telemetry write failed: {_e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())








