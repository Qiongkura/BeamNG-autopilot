"""Scan the active nav route at fixed distances without driving.

Attaches to the running game, reads the same navigation route the
autopilot uses, then runs the real perception + planning pipeline at
probe points along the route.  The vehicle is not teleported and no
control inputs are sent, so the user's session is left untouched.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beamng_autopilot import config
from beamng_autopilot.connector import BeamNGConnector
from beamng_autopilot.perception import scan_obstacles_all
from beamng_autopilot.planner import (
    CAR_HALF_WIDTH,
    LocalPlanner,
    _find_blocker,
    _path_hit_index,
    _point_lat_offset,
)
from beamng_autopilot.runtime import resolve_runtime


def _route_heading(route, k: int) -> float:
    n = len(route)
    a = max(0, k - 2)
    b = min(n - 1, k + 2)
    dx = route[b, 0] - route[a, 0]
    dy = route[b, 1] - route[a, 1]
    return math.atan2(dy, dx)


def _ob_axis_str(ob) -> str:
    if getattr(ob, "axis", None) is not None:
        return (f"axis=({float(ob.axis[0]):.2f},{float(ob.axis[1]):.2f}) "
                f"len={ob.half_len:.2f} thick={ob.half_thick:.2f}")
    return "axis=none"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=config.DEFAULT_MAP)
    ap.add_argument("--vehicle", default=config.DEFAULT_VEHICLE)
    ap.add_argument("--start-m", type=float, default=0.0)
    ap.add_argument("--end-m", type=float, default=90.0)
    ap.add_argument("--step-m", type=float, default=5.0)
    ap.add_argument("--radius", type=float, default=55.0)
    ap.add_argument("--goal-x", type=float, default=None)
    ap.add_argument("--goal-y", type=float, default=None)
    ap.add_argument("--telemetry-json", default=None,
                    help="fallback route source (logs/telemetry/live.json)")
    ap.add_argument("--runtime", choices=("auto", "steam", "tech"),
                    default=config.RUNTIME_MODE,
                    help="game runtime: auto detects after connecting")
    args = ap.parse_args()

    conn = BeamNGConnector(
        args.map, args.vehicle, home=config.runtime_home(args.runtime))
    planner = LocalPlanner()
    try:
        conn.open(launch=False)
        conn.attach_vehicle(already_open=True)
        runtime_mode = resolve_runtime(conn, args.runtime)
        print(f"[probe] runtime={runtime_mode}")
        nav = conn.read_navigation_route()
        if nav is None or len(nav) < 4:
            src = args.telemetry_json
            if src is None:
                src = str(Path(__file__).resolve().parent.parent
                          / "logs" / "telemetry" / "live.json")
            telemetry_path = Path(src)
            if telemetry_path.exists():
                data = json.loads(telemetry_path.read_text(encoding="utf-8"))
                rte = ((data or {}).get("extra") or {}).get("rte") or []
                if len(rte) >= 4:
                    raw = np.asarray(rte, dtype=float)
                    pts = raw[:, :2]
                    dseg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
                    cum = np.concatenate([[0.0], np.cumsum(dseg)])
                    total = float(cum[-1])
                    if total > 10.0:
                        n = max(2, int(round(total / 2.0)) + 1)
                        s = np.linspace(0.0, total, n)
                        nav = np.empty((n, 3), dtype=float)
                        for k in range(2):
                            nav[:, k] = np.interp(
                                s, cum, pts[:, k])
                        nav[:, 2] = 0.0
                        print("[probe] using telemetry route fallback")
            if nav is None or len(nav) < 4:
                if args.goal_x is None or args.goal_y is None:
                    raise RuntimeError("no nav route and no fallback goal given")
                conn.bng.control.queue_lua_command(
                    "core_groundMarkers.setPath({vec3(%.3f, %.3f, 0)})\n"
                    "return 'ok'" % (float(args.goal_x), float(args.goal_y)),
                    response=True)
                time.sleep(0.6)
                nav = conn.read_navigation_route()
        if nav is None or len(nav) < 4:
            raise RuntimeError("no in-game navigation route available")
        route = nav[:, :2]
        dseg = np.linalg.norm(np.diff(route, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(dseg)])
        total = float(cum[-1])
        print(f"[probe] route pts={len(route)} length={total:.1f} m "
              f"head={route[0].tolist()} tail={route[-1].tolist()}")

        dist = args.start_m
        while dist <= args.end_m and dist <= total:
            k = int(np.argmin(np.abs(cum - dist)))
            px, py = route[k]
            pz = float(nav[k, 2]) if nav.shape[1] >= 3 else 0.0
            heading = _route_heading(route, k)
            # Probe points are remote world positions, so both runtimes use
            # the world-space Lua scans; a vehicle-attached LiDAR only sees
            # the car's actual location.
            obstacles = scan_obstacles_all(
                conn.bng, conn.vehicle.vid, (px, py, pz),
                radius=args.radius)
            nearest = k
            lat = _point_lat_offset(px, py, route)
            pts, i0, i1 = planner._window(route, nearest)
            safe_off = planner._safe_right_offset(
                pts, i0, i1, heading, obstacles)
            raw_pts = pts.copy()
            off_pts = planner._right_offset_path(
                raw_pts, i0, heading, offset=safe_off)
            raw_hit = _path_hit_index(
                raw_pts, i0, i1, obstacles, CAR_HALF_WIDTH + 0.8)
            off_hit = _path_hit_index(
                off_pts, i0, i1, obstacles, CAR_HALF_WIDTH + 0.8)
            drive, blocked = planner.plan(
                route, obstacles, (px, py), heading, nearest)
            blocker = _find_blocker(
                off_pts, i0, i1, obstacles, CAR_HALF_WIDTH + 0.8)
            grid, reached = planner._grid_path(
                off_pts, obstacles, (px, py), heading, i0, i1)
            bypass = planner._lateral_bypass(
                off_pts, obstacles, i0, i1)
            print(f"[probe] d={dist:5.1f} pos=({px:.1f},{py:.1f}) "
                  f"lat={lat:.2f} hdg={math.degrees(heading):6.1f} "
                  f"mode={planner.last_mode} blocked={blocked} "
                  f"safe_off={safe_off:.2f} obs={len(obstacles)} "
                  f"drive={len(drive)} raw_hit={raw_hit} off_hit={off_hit}")
            if blocker is not None:
                bprof = planner._obstacle_route_profile(
                    blocker, off_pts, i0, i1)
                print(f"  blocker {blocker.category}/{blocker.label!r} "
                      f"c=({blocker.x:.1f},{blocker.y:.1f}) "
                      f"len={blocker.half_len:.2f} "
                      f"thick={blocker.half_thick:.2f} "
                      f"profile={tuple(round(v, 2) for v in bprof)} "
                      f"roadside={planner._is_roadside_wall(blocker, bprof, pts=route)}")
            if grid is not None:
                print(f"  grid len={len(grid)} reached={reached}")
            if bypass is not None:
                print(f"  bypass len={len(bypass)}")
            for ob in obstacles:
                dx = ob.x - px
                dy = ob.y - py
                od = math.hypot(dx, dy)
                if od > 45.0:
                    continue
                profile = planner._obstacle_route_profile(
                    ob, pts, i0, i1)
                road = planner._is_roadside_wall(
                    ob, profile, pts=route)
                olat = _point_lat_offset(ob.x, ob.y, route)
                side = "L" if olat > 0.5 else ("R" if olat < -0.5 else "C")
                print(f"  obs {ob.category}/{ob.label!r} "
                      f"c=({ob.x:.1f},{ob.y:.1f}) "
                      f"hw={ob.half_w:.2f} hh={ob.half_h:.2f} "
                      f"{_ob_axis_str(ob)} d={od:5.1f} "
                      f"lat={olat:6.2f} {side} "
                      f"profile=({profile[0]:.1f},{profile[1]:.1f},"
                      f"{profile[2]:.1f},{profile[3]:.1f}) "
                      f"roadside={road}")
            dist += args.step_m
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
