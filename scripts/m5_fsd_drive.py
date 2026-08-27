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
from beamng_autopilot.connector import BeamNGConnector
from beamng_autopilot.control import gearbox
from beamng_autopilot.control.reverse_guard import ReverseGuard
from beamng_autopilot.control.reverse_maneuver import ReverseManeuver
from beamng_autopilot.control.pure_pursuit import PurePursuit
from beamng_autopilot.control.speed import SpeedController
from beamng_autopilot.fsd_stack import FSDStack
from beamng_autopilot.occupancy import OccupancyGrid
from beamng_autopilot.planning import Scene, local_route
from beamng_autopilot.roadnet import RoadNetwork
from beamng_autopilot.autopilot import nearest_route_point, smooth_steer
from beamng_autopilot.planner import (
    LocalPlanner,
    emergency_speed_limit_mps,
    emergency_stop_clearance_m,
    path_grid_clearance_m,
)
from beamng_autopilot.safety_monitor import SafetyMonitor
from beamng_autopilot.vision.heads import SemanticHead, TrafficSignalHead

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
# Physics steps advanced per control tick while the sim is paused
# (1/60 s per step => 20 steps = 0.33 s of driving per control).  The
# FSD tick itself takes ~0.7-2 s of WALL time; without pause the car
# drove that whole wall time with the PREVIOUS control.  Paused and
# stepped in 0.33 s bursts, the control spacing in sim time is fixed and
# much finer than the old wall-time drift (fix44-51: 5-10 m of stale
# control per frame at the hairpin).
CTRL_BURST_STEPS = 20
# Target-speed smoothing: the plan speed changes by whole m/s between
# ticks (corner governor, obstacle caps).  Feeding it straight into the
# SpeedController made the pedals oscillate throttle -> brake -> throttle
# every other frame (fix54/fix56).  Ramp the effective target toward the
# plan at a bounded rate instead.
SPEED_TARGET_RAMP_MPS = 1.5     # m/s per sim second
SPEED_HYST_MPS = 0.4            # SpeedController brake/throttle hysteresis
# Plan-speed brake governor hysteresis: enter at +1.0 m/s overshoot,
# release once back within +0.5 m/s, and brake gently (0.25) instead of
# 1.0 - BeamNG's brake is highly nonlinear and even 0.4-0.7 stands the
# car dead in one 0.33 s burst, which then re-triggers the full-throttle
# stall loop (fix61-64).  The downhill-start guard below prevents the
# overshoot in the first place.
GOV_ON_MPS = 1.0
GOV_OFF_MPS = 0.5
GOV_BRAKE = 0.25



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
    speed_ctrl = SpeedController(hyst_mps=SPEED_HYST_MPS)
    monitor = SafetyMonitor(max_speed=args.speed)
    try:
        conn.open(launch=not args.attach)
        try:
            conn.attach_vehicle(already_open=True)
        except Exception:
            conn.load_scenario()
        if args.teleport is not None:
            from beamng_autopilot.connector import angle_to_quat
            x, y, yaw = args.teleport
            # ground ray: from high above straight down, take the real
            # terrain z and lift the car above it - a hardcoded z puts the
            # car underground and the camera sees nothing useful.
            resp = conn.bng.control.queue_lua_command(
                f"local r = Engine.castRay(vec3({x:.3f}, {y:.3f}, 10000), "
                f"vec3({x:.3f}, {y:.3f}, -1000), true, false)\n"
                "if r and r.pt then return string.format('%.3f', r.pt.z) "
                "end\nreturn 'nil'", response=True)
            z = 154.1
            if resp and str(resp).strip() != "nil":
                try:
                    z = float(str(resp).strip()) + 0.6
                except ValueError:
                    pass
            conn.vehicle.teleport(pos=(float(x), float(y), z),
                                  rot_quat=angle_to_quat((0, 0, -float(yaw) - 90.0)))
            conn.step(8)
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
        elif nav_route is None:
            nav = conn.read_navigation_route()
            if nav is not None and len(nav) >= 4:
                nav_route = np.asarray(nav[:, :2], dtype=float)
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
                # Connector convention (see connector.py spawn): the snap
                # passes angle_to_quat the yaw directly with
                # ``yaw_deg = -degrees(h) - 90`` so the resulting STATE
                # heading equals h (route direction).  Do NOT use the
                # --teleport convention here (that one adds the -90 itself).
                yaw_deg = -math.degrees(h) - 90.0
                from beamng_autopilot.connector import angle_to_quat
                resp = conn.bng.control.queue_lua_command(
                    f"local r = Engine.castRay(vec3({rx:.3f}, {ry:.3f}, 10000), "
                    f"vec3({rx:.3f}, {ry:.3f}, -1000), true, false)\n"
                    "if r and r.pt then return string.format('%.3f', r.pt.z) "
                    "end\nreturn 'nil'", response=True)
                z = 154.1
                if resp and str(resp).strip() != "nil":
                    try:
                        z = float(str(resp).strip()) + 0.6
                    except ValueError:
                        pass
                conn.vehicle.teleport(pos=(rx, ry, z),
                                      rot_quat=angle_to_quat((0, 0, yaw_deg)))
                conn.step(8)
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
                         heads=[SemanticHead(), TrafficSignalHead()],
                         ring_roles=('front_main',),
                         cam_w=args.cam_w, cam_h=args.cam_h,
                         temporal=True, range_every_n=2)
        stack.reset_temporal()  # stale occupancy before start must not leak
        # Realistic gearbox locked into a forward gear (D).  A real stack
        # never leaves the car in reverse; keep the D input on every
        # control frame so an impact can never leave the gearbox in R.
        fwd_gear = gearbox.forward_gear_input(conn)
        conn.control(throttle=0.0, brake=0.0, steering=0.0,
                     parkingbrake=0.0, gear=fwd_gear)
        conn.step(3)
        # Deterministic control: pause the simulation for the control
        # loop.  Without this the sim keeps running in REAL TIME while
        # the FSD tick computes (~1.4-7 s per frame): the first frame
        # takes ~7 s and the car rolls ~9 m straight down the slope
        # before the first steering is ever sent (diag2: f=0 "before"
        # position was already (725.6,753.1), the hairpin entry), and
        # every later frame the car drives several metres with the
        # PREVIOUS tick's controls.  Paused, each tick is: sense ->
        # plan -> control -> step(2), a tight closed loop.
        try:
            conn.bng.pause()
        except Exception as _pe:
            print(f"[fsd-drive] pause failed: {_pe}")
        print(f"[fsd-drive] gearbox realistic, forward gear input = {fwd_gear}")
        rguard = ReverseGuard(threshold_mps=REVERSE_THRESHOLD_MPS,
                              clear_mps=REVERSE_CLEAR_MPS)
        rman = ReverseManeuver(fwd_gear=fwd_gear)
        print(f"[fsd-drive] runtime={stack.mode} FSD pipeline driving "
              f"for {args.seconds}s at {args.speed} m/s")

        prev_steer = 0.0  # rate-limited steering state (rule-autopilot convention)
        last_h = None      # previous heading for the yaw-rate steering damper
        climb_t = 0.0      # seconds spent in slope-creep assist
        stuck_t = 0.0    # seconds at near-standstill with a "safe" plan
        t_end = time.time() + args.seconds
        frames = 0
        stopps = 0
        t0 = time.time()
        target_sm = float(args.speed)
        gov_brake = False
        # First-frame steering: with last_t set to NOW the first tick has
        # dt ~= 0, smooth_steer cannot move the wheel, and the car runs
        # the whole first ~7 m straight past the corner entry before any
        # steering appears (fix50: steer stayed -0.02 while the car went
        # from the spawn to (726.9,756.8)).  Pretend one control interval
        # has already elapsed so the first frame can steer immediately.
        last_t = time.time() - 1.5
        hist: list[dict] = []
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
            # Control-loop dt in SIMULATION time: the sim is paused and
            # advanced in CTRL_BURST_STEPS bursts, so each tick is exactly
            # 0.33 s of driving regardless of the wall-time the FSD tick
            # took (which can be 2-7 s).  Using wall dt made the stuck /
            # reverse / climb state machines count a 7 s computation as
            # 7 s of standstill and reverse after every stop (fix53).
            dt = CTRL_BURST_STEPS / 60.0
            now_t = time.time()
            _wall_dt = max(0.0, now_t - last_t)
            last_t = now_t
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
                    from beamng_autopilot.planning.local_route import (
                        map_lane_edges)
                    map_lane = map_lane_edges(
                        nav_route, road_left, road_right, pos, heading)
                except Exception:
                    map_lane = None
            _ta = time.time()  # local route / map lane / FSD tick split
            out = stack.tick(st=st, route_ref=route_local,
                                   map_lane_override=map_lane)
            _tb = time.time()
            best = out.best_path

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
            from beamng_autopilot.planning import anchored_rule_ref, arbitrate
            from beamng_autopilot.planning.constraints import _boundary_lateral
            # Proven rule-autopilot fallback: the LocalPlanner rounds
            # switchback corners and keeps the car in its own lane, so the
            # FSD drive does not stop dead at a hairpin apex when the
            # layered planner declines every kinked map-prior candidate.
            # The path must still be ego-anchored and head forward; the
            # FSD safety monitor re-verifies it below (and can stop).
            rule_ref = None
            _need_rule = (best is None or len(best) < 2 or not verd.safe)
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
                    verd = monitor.evaluate(scene, chosen.path,
                                            planner_age_s=0.0)
                except Exception:
                    pass
            steer = 0.0
            pp_alpha = None
            pp_tgt = None
            ff_steer = 0.0
            if chosen.path is not None and len(chosen.path) >= 2:
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
                    pos, heading, np.asarray(chosen.path))
                steer_rad = float(steer_rad)
                ff_steer = _path_curvature_ff(chosen.path, pos, heading)
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
            _corner_d_ahead = None
            try:
                from beamng_autopilot.planning.local_route import (
                    _dedup as _rdd, _extend_back as _reb,
                    _round_corners as _rrc, _resample as _rrs,
                    CORNER_RADIUS_M as _CR, CORNER_RESAMPLE_M as _CSM,
                    DUP_MIN_M as _DUP)
                from beamng_autopilot.planning.speed_profile import \
                    speed_profile_for_path as _spf_raw
                # Profile the ROUNDED full route, not the raw road-graph
                # polyline: the graph collapses the first hairpin into a
                # sharp vertex whose curvature profile caps the bend at
                # ~1.7 m/s - at that speed the tyres scrub and the car
                # cannot even turn (steering probe 2026-08-27).  The
                # rounded route (same 8 m fillet the lane centre uses)
                # lets the bend be taken at a speed the steering can
                # actually execute.
                _rfull = _rrs(_rrc(_reb(_rdd(nav_route[:, :2])), _CR),
                              _CSM)
                _sp_raw = _spf_raw(_rfull, scene,
                                   target_speed=float(args.speed))
                if len(_sp_raw):
                    # The full-route profile is indexed along the WHOLE
                    # route; [0] is the speed at the ROUTE START, not at
                    # the car.  Sample the profile at the nearest route
                    # point so a hairpin 100 m into the route still caps
                    # the speed when the car reaches it (run_fix25:
                    # plan_speed stayed at the start speed into the bend).
                    _i = int(np.argmin(np.linalg.norm(
                        _rfull - pos[:2], axis=1)))
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
                _ga = np.concatenate(
                    [[0.0], np.cumsum(np.linalg.norm(
                        np.diff(_rfull, axis=0), axis=1))])
                _gi = int(np.argmin(np.linalg.norm(
                    _rfull - pos[:2], axis=1)))
                _ghi = int(np.searchsorted(_ga, _ga[_gi] + 12.0))
                _rmin = 1e9
                for _j in range(max(1, _gi),
                                min(_ghi + 1, len(_rfull) - 1)):
                    _a, _b, _cc = (_rfull[_j - 1], _rfull[_j],
                                   _rfull[_j + 1])
                    _v1 = _b - _a
                    _v2 = _cc - _b
                    _n1 = float(np.linalg.norm(_v1))
                    _n2 = float(np.linalg.norm(_v2))
                    if _n1 < 1e-9 or _n2 < 1e-9:
                        continue
                    _cr = abs(_v1[0] * _v2[1] - _v1[1] * _v2[0])
                    _curv = 2.0 * _cr / (_n1 * _n2 * (_n1 + _n2))
                    if _curv > 1e-6:
                        _rmin = min(_rmin, 1.0 / _curv)
                if _rmin < 15.0:
                    # Floor the implied radius: the 0.8 m resample can
                    # measure a hairpin fillet edge as R~0.8 m (three
                    # nearly-collinear points), which caps the plan at
                    # ~1 m/s and stands the car dead on the approach
                    # (fix65: plan=1.00 at the second bend, v=5.6 ->
                    # brake-to-0).  Real roads never bend tighter than
                    # ~3 m; anything smaller is a sampling artifact.
                    _rmin = max(_rmin, 3.0)
                    out.best_speed = float(min(
                        out.best_speed, math.sqrt(1.3 * _rmin)))
                    out.meta["plan_src"] = "nav_round+gov"
                # Distance (m) from the car to the first bend tighter
                # than 15 m within the 12 m window; used by the corner
                # brake zone below.
                _corner_d_ahead = None
                for _j in range(max(1, _gi),
                                min(_ghi + 1, len(_rfull) - 1)):
                    _a, _b, _cc = (_rfull[_j - 1], _rfull[_j],
                                   _rfull[_j + 1])
                    _v1 = _b - _a
                    _v2 = _cc - _b
                    _n1 = float(np.linalg.norm(_v1))
                    _n2 = float(np.linalg.norm(_v2))
                    if _n1 < 1e-9 or _n2 < 1e-9:
                        continue
                    _cr = abs(_v1[0] * _v2[1] - _v1[1] * _v2[0])
                    _curv = 2.0 * _cr / (_n1 * _n2 * (_n1 + _n2))
                    if _curv > 1e-6 and 1.0 / _curv < 15.0:
                        _corner_d_ahead = float(_ga[_j] - _ga[_gi])
                        break
            except Exception:
                pass
            # control from the (possibly degraded) target speed, but never
            # exceed the *planned* speed along the chosen trajectory - the
            # FSD longitudinal plan (bend deceleration, obstacle brake
            # band) must govern the actual pedals.
            plan_speed = out.best_speed if out.best_speed > 0.0 \
                else float(args.speed)
            # a rule fallback does not get the FSD plan speed; cap it to a
            # cautious creep so the L2 fallback is gentle
            if chosen.source == "rule":
                plan_speed = min(plan_speed, 3.0)
            target = min(verd.target_speed, plan_speed, float(args.speed))
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
                thr = min(thr, 0.08)
            # (The old "corner brake zone" here is gone: with the sim
            # paused and stepped in 0.33 s bursts the speed controller
            # reacts within one burst, while the zone caused an
            # accelerate -> brake-to-stop oscillation - fix54 reached
            # v=3.6 then brk=0.8 stopped it dead at every tick.  The
            # plan-speed governor below still hard-brakes overshoot.)
            # Hard speed governor: the ~1.4 s control loop can miss a
            # downhill acceleration (fix40: 5.0 -> 9.7 m/s in one tick
            # on the east-side descent).  Never let the car exceed the
            # commanded cruise speed by more than 1 m/s regardless of
            # the smoothed pedal state.
            if v > float(args.speed) + 1.0:
                thr, brk = 0.0, 1.0
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
            if (stuck and not force_stop and chosen.path is not None
                    and len(chosen.path) >= 2
                    and np.isfinite(fwd_clear) and fwd_clear > 3.0):
                if climb_t < CLIMB_ASSIST_S:
                    climb = True
                    climb_t += max(0.0, float(dt))
                else:
                    climb_t = 0.0  # give up climbing -> allow reverse
            if (chosen.path is None or force_stop or stuck) and not climb:
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
                                and not force_stop and not stuck) or climb
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
                "plan_speed": round(float(plan_speed), 2),
                "target_sm": round(float(target_sm), 2),
                "plan_src": str(out.meta.get("plan_src", "?")),
                "tick_ms": out.meta.get("tick_ms"),
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
                "lane_paired": int(out.meta.get("lane_paired", 0)),
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
                "pp_tgt": ([round(float(v), 2) for v in pp_tgt[:2]]
                           if pp_tgt is not None else None),
                "lane_bear": _ref_bearing(out.lane_ref, pos),
                "route_bear": _ref_bearing(route_local, pos),
                "best_bear": _ref_bearing(out.best_path, pos),
            })
            conn.step(CTRL_BURST_STEPS)
            frames += 1
            if frames % 4 == 1:
                print(f"[fsd-drive] t={time.time()-t0:5.1f} v={v:4.1f} "
                      f"level={verd.level} src={chosen.source:4s} "
                      f"reason={verd.reason or '-':22s} "
                      f"steer={steer:+.2f} thr={thr:.2f} "
                      f"plan_v={plan_speed:.1f} "
                      f"rev={int(reversing)} signed={signed:+.2f} "
                      f"lane={out.meta.get('lane_src', '?')}/"
                      f"{'P' if out.meta.get('lane_paired') else '1'} "
                      f"dev={getattr(verd, 'lane_dev_m', 0.0):.2f}")
        print(f"[fsd-drive] done: {frames} frames, {stopps} stops")
    finally:
        # ensure the car stops
        try:
            conn.control(throttle=0.0, brake=1.0, steering=0.0,
                         gear=locals().get("fwd_gear"))
            conn.step(3)
        except Exception:
            pass
        # leave the game running (the control loop paused it)
        try:
            conn.bng.resume()
        except Exception:
            pass
        conn.close()
        if hist and args.out:
            try:
                from pathlib import Path as _P
                _P(args.out).parent.mkdir(parents=True, exist_ok=True)
                _P(args.out).write_text(
                    json.dumps(hist, ensure_ascii=False), encoding="utf-8")
                print(f"[fsd-drive] telemetry -> {args.out} ({len(hist)} frames)")
            except Exception as _e:
                print(f"[fsd-drive] telemetry write failed: {_e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())








