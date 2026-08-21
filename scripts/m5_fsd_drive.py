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
from beamng_autopilot.control.pure_pursuit import PurePursuit
from beamng_autopilot.control.speed import SpeedController
from beamng_autopilot.fsd_stack import FSDStack
from beamng_autopilot.occupancy import OccupancyGrid
from beamng_autopilot.planning import Scene
from beamng_autopilot.safety_monitor import SafetyMonitor
from beamng_autopilot.vision.heads import SemanticHead, TrafficSignalHead

# Reverse guard: the car must never drive backwards under the FSD mode.
# A real factory stack has lane/gear protections - m5_autopilot does too
# (REVERSE_ENGAGE_S/REVERSE_HOLD_S/gear=D).  Without this the FSD drive
# reversed into walls after an impact ("dumb reversing" seen on probes).
REVERSE_THRESHOLD_MPS = -0.35
REVERSE_CLEAR_MPS = 0.2


def _local_route(pos, heading, nav_route, ahead_m=32.0) -> np.ndarray:
    """The route in front of the ego, re-anchored at the car.

    The in-game navigation route is a map prior that can start far from
    the ego when the car drifts; planning and lane-keep must measure
    against the *forward* part of the route only - the far tail of a
    long route reads as a wall / lane edge and stalls the car on the
    first town bend.  Falls back to a straight line ahead without a
    nav route.
    """
    pos = np.asarray(pos[:2], dtype=float)
    fwd = np.array([float(np.cos(heading)), float(np.sin(heading))])
    if nav_route is not None and len(nav_route) >= 2:
        r = np.asarray(nav_route[:, :2], dtype=float)
        i0 = int(np.argmin(np.linalg.norm(r - pos, axis=1)))
        d_prev = float("inf")
        d_next = float("inf")
        if i0 > 0:
            v = r[i0] - r[i0 - 1]
            d_prev = float(np.dot(v, fwd))
        if i0 + 1 < len(r):
            v = r[i0 + 1] - r[i0]
            d_next = float(np.dot(v, fwd))
        if d_next > 0 or d_prev == float("inf"):
            seg = r[i0:]
        else:
            seg = r[i0::-1]  # route runs opposite; walk it backwards
        # keep only the near, ahead part (the far route is unknown space)
        dseg = np.linalg.norm(seg - pos, axis=1)
        seg = seg[dseg <= ahead_m + 8.0]
        if len(seg) >= 4:
            return seg
    xs = np.linspace(0, ahead_m, 25)
    return np.column_stack([pos[0] + xs * np.cos(heading),
                            pos[1] + xs * np.sin(heading)])


def main() -> int:
    ap = argparse.ArgumentParser(description="FSD-mode live driving")
    ap.add_argument("--runtime", choices=("auto", "steam", "tech"),
                    default="auto")
    ap.add_argument("--attach", action="store_true")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--speed", type=float, default=6.0)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--cam-w", type=int, default=536)
    ap.add_argument("--cam-h", type=int, default=403)
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
    speed_ctrl = SpeedController()
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
                                  rot_quat=angle_to_quat((0, 0, float(yaw))))
            conn.step(8)
            st1 = conn.get_state()
            print(f"[fsd-drive] teleport -> "
                  f"({float(st1.pos[0]):.1f}, {float(st1.pos[1]):.1f}, "
                  f"{float(st1.pos[2]):.1f})")

        # Navigation route: a real stack plans ALONG the destination
        # route (FSD vector-space planner), not a straight line ahead -
        # a straight reference drives the car into the building on the
        # first town bend (observed town run 2026-08-21).  Use the
        # in-game navigation route when available.
        nav_route = None
        if args.goal is not None:
            conn.bng.control.queue_lua_command(
                "core_groundMarkers.setPath({vec3(%.3f, %.3f, 0)})\n"
                "return 'ok'" % (float(args.goal[0]), float(args.goal[1])),
                response=True)
            time.sleep(0.8)
        nav = conn.read_navigation_route()
        if nav is not None and len(nav) >= 4:
            nav_route = np.asarray(nav[:, :2], dtype=float)
            dseg = np.linalg.norm(np.diff(nav_route, axis=0), axis=1)
            print(f"[fsd-drive] nav route: {len(nav_route)} pts, "
                  f"{float(np.sum(dseg)):.1f} m")
        else:
            print("[fsd-drive] no nav route set; falling back to "
                  "straight-ahead reference")

        stack = FSDStack(conn, args.runtime,
                         heads=[SemanticHead(), TrafficSignalHead()],
                         cam_w=args.cam_w, cam_h=args.cam_h,
                         temporal=True)
        stack.reset_temporal()  # stale occupancy before start must not leak
        # Realistic gearbox locked into a forward gear (D).  A real stack
        # never leaves the car in reverse; keep the D input on every
        # control frame so an impact can never leave the gearbox in R.
        fwd_gear = gearbox.forward_gear_input(conn)
        conn.control(throttle=0.0, brake=0.0, steering=0.0,
                     parkingbrake=0.0, gear=fwd_gear)
        conn.step(3)
        print(f"[fsd-drive] gearbox realistic, forward gear input = {fwd_gear}")
        rguard = ReverseGuard(threshold_mps=REVERSE_THRESHOLD_MPS,
                              clear_mps=REVERSE_CLEAR_MPS)
        print(f"[fsd-drive] runtime={stack.mode} FSD pipeline driving "
              f"for {args.seconds}s at {args.speed} m/s")

        t_end = time.time() + args.seconds
        frames = 0
        stopps = 0
        t0 = time.time()
        last_t = time.time()
        hist: list[dict] = []
        while time.time() < t_end:
            st = conn.get_state()
            pos = np.asarray(st.pos, dtype=float)
            heading = float(st.heading)
            v = float(st.speed)
            signed = 0.0
            if st.vel is not None and st.dir is not None:
                signed = float(np.dot(
                    np.asarray(st.vel[:2], dtype=float),
                    np.asarray(st.dir[:2], dtype=float)))
            now_t = time.time()
            dt = max(0.0, now_t - last_t)
            last_t = now_t
            rev_brk, reversing = rguard.decide(signed, dt=dt)

            # one full FSD tick -> best trajectory (planned along the
            # navigation route when one exists)
            out = stack.tick(st=st, route_ref=nav_route)
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
            local = _local_route(pos, heading, nav_route)
            if out.lane_ref is not None and len(out.lane_ref) >= 4:
                local = np.asarray(out.lane_ref, dtype=float)
            if out.bev is not None and out.bev.shape == grid.occupancy.shape:
                grid.occupancy[:] = np.asarray(out.bev, dtype=np.float32)
                grid.obstacle[:] = (np.asarray(out.bev) >= 0.6
                                    ).astype(np.uint8)
                if out.drivable is not None and \
                        out.drivable.shape == grid.drivable.shape:
                    grid.drivable[:] = np.asarray(out.drivable)
                scene = Scene(pos=pos, heading=heading, grid=grid,
                              route=local,
                              lane_ref=local, target_speed=args.speed)
                verd = monitor.evaluate(scene, best,
                                        planner_age_s=0.0)
            else:
                verd = monitor.evaluate(Scene(pos=pos, heading=heading),
                                        best)

            # planner arbitration: FSD path first; when the layered
            # planner declined (even to minimal risk) fall back to the
            # rule straight-ahead reference IN WORLD COORDINATES - the
            # car must not stop dead on a transient "no drivable path"
            # unless the rule path is also unusable (then and only then
            # a minimal-risk stop).  A body-frame reference handed to
            # PurePursuit points at a wrong world target and spins the
            # car (the "dumb reversing" seen in probes).
            rule_ref = local
            from beamng_autopilot.planning import arbitrate
            chosen = arbitrate(
                best, rule_ref,
                fsd_safe=verd.safe and best is not None and len(best) >= 2,
                prefer_rule=False)
            steer = 0.0
            if chosen.path is not None and len(chosen.path) >= 2:
                steer = float(pp.steering(
                    pos, heading, np.asarray(chosen.path))[0])

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
            thr, brk = speed_ctrl.update(target, v)
            # hard stop ONLY when no path at all remains (arbitration none)
            if chosen.path is None:
                thr, brk = 0.0, 1.0
                steer = 0.0
                stopps += 1
            # Reverse guard is the final control authority: while the car
            # is (still) moving backwards, brake with steering centred and
            # no throttle until forward motion returns (hysteresis in
            # ReverseGuard prevents brake flap around standstill).
            if reversing:
                thr, brk = 0.0, max(brk, float(rev_brk))
                steer = 0.0
            conn.control(throttle=thr, brake=brk, steering=steer,
                         gear=fwd_gear)
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
                "throttle": round(float(thr), 4),
                "brake": round(float(brk), 4),
                "steer": round(float(steer), 4),
                "reversing": int(bool(reversing)),
            })
            conn.step(args.steps)
            frames += 1
            if frames % 4 == 1:
                print(f"[fsd-drive] t={time.time()-t0:5.1f} v={v:4.1f} "
                      f"level={verd.level} src={chosen.source:4s} "
                      f"reason={verd.reason or '-':22s} "
                      f"steer={steer:+.2f} thr={thr:.2f} "
                      f"plan_v={plan_speed:.1f} "
                      f"rev={int(reversing)} signed={signed:+.2f}")
        print(f"[fsd-drive] done: {frames} frames, {stopps} stops")
    finally:
        # ensure the car stops
        try:
            conn.control(throttle=0.0, brake=1.0, steering=0.0,
                         gear=locals().get("fwd_gear"))
            conn.step(3)
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