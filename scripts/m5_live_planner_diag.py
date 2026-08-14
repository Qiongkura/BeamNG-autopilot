"""Live planner diagnostic: print every obstacle and the planner verdict.

Connects to the running game, teleports to the route start, reads the nav
route, then runs the same planner pipeline the autopilot uses and prints
the obstacle footprints, route-local profiles, roadside-wall checks and the
final mode/blocker.  Used to diagnose "car does not move" and "stops at a
wall" without relying on screenshots.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
from beamngpy.misc.quat import angle_to_quat

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beamng_autopilot import config
from beamng_autopilot.connector import BeamNGConnector
from beamng_autopilot.perception import scan_obstacles_all
from beamng_autopilot.planner import (
    LocalPlanner,
    _point_lat_offset,
)


def _ground_z(conn, x: float, y: float) -> float | None:
    chunk = (
        f"local res = Engine.castRay(vec3({x:.3f}, {y:.3f}, 10000), "
        f"vec3({x:.3f}, {y:.3f}, -1000), true, false)\n"
        "if res and res.pt then "
        "return string.format('%.3f,%.3f,%.3f', "
        "res.pt.x, res.pt.y, res.pt.z) end\n"
        "return 'nil'"
    )
    resp = conn.bng.control.queue_lua_command(chunk, response=True)
    if resp and str(resp).strip() != "nil":
        parts = str(resp).split(",")
        if len(parts) == 3:
            return float(parts[2])
    return None


def _teleport(conn, x: float, y: float, goal_xy):
    heading = math.atan2(goal_xy[1] - y, goal_xy[0] - x)
    yaw_deg = -math.degrees(float(heading)) - 90.0
    st0 = conn.get_state()
    z = float(st0.pos[2]) if len(st0.pos) > 2 else 0.0
    ground_z = _ground_z(conn, float(x), float(y))
    if ground_z is not None:
        z = ground_z + 0.6
    conn.vehicle.teleport(
        (float(x), float(y), z),
        rot_quat=angle_to_quat((0.0, 0.0, yaw_deg)))
    conn.control(throttle=0.0, brake=0.0, steering=0.0, parkingbrake=0.0)
    conn.step(30)
    st = conn.get_state()
    print(f"[diag] teleport ({x:.1f}, {y:.1f}, z={z:.2f}) -> "
          f"({st.pos[0]:.1f}, {st.pos[1]:.1f}, {st.pos[2]:.1f}) "
          f"heading={math.degrees(float(st.heading)):.1f}")
    return st


def _profile_str(planner, ob, pts, i0, i1) -> str:
    lon0, lon1, lat0, lat1 = planner._obstacle_route_profile(
        ob, pts, i0, i1)
    road = planner._is_roadside_wall(ob, (lon0, lon1, lat0, lat1))
    return (f"lon=({lon0:.1f},{lon1:.1f}) lat=({lat0:.1f},{lat1:.1f}) "
            f"roadside={road}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=config.DEFAULT_MAP)
    ap.add_argument("--vehicle", default=config.DEFAULT_VEHICLE)
    ap.add_argument("--start-x", type=float, default=726.6)
    ap.add_argument("--start-y", type=float, default=755.9)
    ap.add_argument("--goal-x", type=float, default=555.8)
    ap.add_argument("--goal-y", type=float, default=394.2)
    ap.add_argument("--goal-dist", type=float, default=330.0)
    ap.add_argument("--loop-secs", type=float, default=8.0)
    ap.add_argument("--load-scenario", action="store_true",
                    help="load a fresh scenario instead of attaching")
    args = ap.parse_args()

    conn = BeamNGConnector(args.map, args.vehicle)
    planner = LocalPlanner()
    try:
        goal_xy = np.array([args.goal_x, args.goal_y], dtype=float)
        if args.load_scenario:
            conn.open(launch=False)
            heading = math.atan2(
                goal_xy[1] - args.start_y, goal_xy[0] - args.start_x)
            conn.load_scenario(
                spawn_pos=(args.start_x, args.start_y, 178.5),
                spawn_heading=heading)
            conn.step(30)
        else:
            conn.open(launch=False)
            conn.attach_vehicle(already_open=True)
        st = _teleport(conn, args.start_x, args.start_y, goal_xy)

        conn.bng.control.queue_lua_command(
            "core_groundMarkers.setPath({vec3(%.3f, %.3f, 0)})\nreturn 'ok'"
            % (float(args.goal_x), float(args.goal_y)), response=True)
        time.sleep(0.6)
        nav = conn.read_navigation_route()
        if nav is None or len(nav) < 4:
            raise RuntimeError("no in-game navigation route available")
        route = nav[:, :2]
        dseg = np.linalg.norm(np.diff(route, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(dseg)])
        end = int(np.argmin(np.abs(cum - args.goal_dist)))
        route = route[: end + 1]
        print(f"[diag] nav route: {len(route)} pts length={cum[end]:.1f} m")
        print("[diag] route head:")
        for k in range(min(8, len(route))):
            print(f"  [{k}] ({route[k, 0]:.1f}, {route[k, 1]:.1f})")

        t0 = time.time()
        while time.time() - t0 < args.loop_secs:
            st = conn.get_state()
            obstacles = scan_obstacles_all(
                conn.bng, conn.vehicle.vid, st.pos, radius=55.0)
            d = np.linalg.norm(route[:, :2] - st.pos[:2], axis=1)
            nearest = int(np.argmin(d))
            drive_route, blocked = planner.plan(
                route, obstacles, st.pos, st.heading, nearest)
            print(f"[diag] t={time.time() - t0:.1f}s pos=({st.pos[0]:.1f}, "
                  f"{st.pos[1]:.1f}, {st.pos[2]:.1f}) speed={st.speed:.2f} "
                  f"nearest={nearest} mode={planner.last_mode} "
                  f"blocked={blocked} blocker={planner.last_blocker} "
                  f"obs={len(obstacles)}")
            _, i0, i1 = planner._window(route, nearest)
            for k, ob in enumerate(obstacles):
                dx = ob.x - st.pos[0]
                dy = ob.y - st.pos[1]
                dist = math.hypot(dx, dy)
                print(f"  obs[{k}] cat={ob.category} label='{ob.label}' "
                      f"center=({ob.x:.1f},{ob.y:.1f}) hw={ob.half_w:.1f} "
                      f"hh={ob.half_h:.1f} dist={dist:.1f} "
                      f"{_profile_str(planner, ob, route, i0, i1)}")
            lat = _point_lat_offset(
                float(st.pos[0]), float(st.pos[1]), route)
            print(f"[diag] lat_offset={lat:.2f} drive_pts={len(drive_route)}")
            conn.step(30)
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
