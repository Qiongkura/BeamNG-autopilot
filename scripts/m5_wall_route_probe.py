"""Batch wall-route probe: teleport along a route and inspect planner.

Attaches to the running session once, reads the same nav route the
autopilot uses, then teleports to a list of known trouble points and runs
the real perception + planning pipeline at each one.  It prints whether
the planner still reports a wall blocker so perception clustering changes
can be validated against the actual map geometry.
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
from beamng_autopilot.planner import LocalPlanner, _point_lat_offset


PROBE_POINTS = [
    (727.0, 756.0),
    (715.0, 730.0),
    (710.0, 715.0),
    (705.0, 700.0),
    (701.0, 690.0),
    (700.0, 684.0),
    (699.0, 682.0),
    (697.0, 674.0),
    (695.0, 675.0),
    (693.0, 671.0),
    (690.0, 668.0),
    (688.0, 664.0),
    (700.0, 650.0),
    (680.0, 650.0),
]


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
    return conn.get_state()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=config.DEFAULT_MAP)
    ap.add_argument("--vehicle", default=config.DEFAULT_VEHICLE)
    ap.add_argument("--goal-x", type=float, default=555.8)
    ap.add_argument("--goal-y", type=float, default=394.2)
    ap.add_argument("--goal-dist", type=float, default=330.0)
    args = ap.parse_args()

    conn = BeamNGConnector(args.map, args.vehicle)
    planner = LocalPlanner()
    try:
        conn.open(launch=False)
        conn.attach_vehicle(already_open=True)
        goal_xy = np.array([args.goal_x, args.goal_y], dtype=float)
        first = PROBE_POINTS[0]
        _teleport(conn, first[0], first[1], goal_xy)
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
        print(f"[wall-route] nav route: {len(route)} pts "
              f"length={cum[end]:.1f} m")

        for k, (px, py) in enumerate(PROBE_POINTS):
            st = _teleport(conn, px, py, goal_xy)
            obstacles = scan_obstacles_all(
                conn.bng, conn.vehicle.vid, st.pos, radius=55.0)
            d = np.linalg.norm(route[:, :2] - st.pos[:2], axis=1)
            nearest = int(np.argmin(d))
            drive, blocked = planner.plan(
                route, obstacles, st.pos, st.heading, nearest)
            lat = _point_lat_offset(float(st.pos[0]), float(st.pos[1]), route)
            big = [ob for ob in obstacles
                   if ob.half_w > 1.4 or ob.half_h > 1.4]
            print(f"[wall-route] {k + 1}/{len(PROBE_POINTS)} "
                  f"pos=({st.pos[0]:.1f},{st.pos[1]:.1f}) "
                  f"lat={lat:.2f} nearest={nearest} "
                  f"mode={planner.last_mode} blocked={blocked} "
                  f"blocker={planner.last_blocker} obs={len(obstacles)} "
                  f"drive={len(drive)}")
            _, i0, i1 = planner._window(route, nearest)
            for ob in big:
                lon0, lon1, lat0, lat1 = planner._obstacle_route_profile(
                    ob, route, i0, i1)
                road = planner._is_roadside_wall(
                    ob, (lon0, lon1, lat0, lat1))
                dx = ob.x - st.pos[0]
                dy = ob.y - st.pos[1]
                print(f"  obs cat={ob.category} label='{ob.label}' "
                      f"center=({ob.x:.1f},{ob.y:.1f}) "
                      f"hw={ob.half_w:.1f} hh={ob.half_h:.1f} "
                      f"dist={math.hypot(dx, dy):.1f} "
                      f"lon=({lon0:.1f},{lon1:.1f}) "
                      f"lat=({lat0:.1f},{lat1:.1f}) roadside={road}")
            conn.step(10)
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
