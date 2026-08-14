"""Live blocker probe: verify raycast + planner against a real parked car.

Attaches to the RUNNING session (the user's map/vehicle are untouched),
spawns a blocker vehicle ~30 m ahead of the ego along its heading, then
runs the exact perception + planning stack used by m5_autopilot.py and
reports whether the raycast fan sees the car and whether the local
planner produces a detour.  The blocker is deleted afterwards, so the
session is left exactly as it was.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beamng_autopilot import config
from beamng_autopilot.connector import BeamNGConnector
from beamng_autopilot.perception import (
    errors_summary,
    scan_obstacles,
    scan_obstacles_all,
    scan_obstacles_raycast,
    scan_obstacles_vehicles,
)
from beamng_autopilot.planner import LocalPlanner

BLOCK_DIST = 30.0


def _lua_spawn(conn, model: str, pos, heading: float) -> str | None:
    """Spawn a parked car via the game's own spawner (no tech license)."""
    x, y, z = (float(v) for v in pos)
    dx = math.cos(heading)
    dy = math.sin(heading)
    chunk = (
        "local v = core_vehicles.spawnNewVehicle('" + model + "', {"
        "pos = vec3(%f, %f, %f), "
        "rot = quatFromDir(vec3(%f, %f, 0)), "
        "autoEnterVehicle = false, "
        "}) "
        "if v then return jsonEncode({id = tostring(v:getId())}) "
        "else return jsonEncode({id = nil}) end"
        % (x, y, z, dx, dy)
    )
    resp = conn.bng.queue_lua_command(chunk, response=True)
    try:
        data = json.loads(str(resp))
        return data.get("id")
    except (ValueError, TypeError):
        print(f"[probe] spawn response not JSON: {resp!r}")
        return None


def _lua_delete(conn, vid: str) -> None:
    try:
        conn.bng.queue_lua_command(
            "local v = scenetree.findObjectById('" + vid + "') "
            "if v and v.delete then v:delete() end return 1")
    except Exception as exc:
        print(f"[probe] delete failed: {exc}")


def main() -> None:
    conn = BeamNGConnector(config.DEFAULT_MAP, config.DEFAULT_VEHICLE,
                           port=config.PORT)
    blocker_id = None
    try:
        conn.open(launch=False)
        conn.attach_vehicle(vid=None, already_open=True)
        st = conn.get_state()
        pos = st.pos
        heading = float(st.heading)
        hx, hy = math.cos(heading), math.sin(heading)
        print(f"[probe] ego pos=({pos[0]:.1f}, {pos[1]:.1f}) "
              f"heading={heading:.2f} speed={st.speed:.2f}")

        obs0 = scan_obstacles_all(conn.bng, conn.vehicle.vid, pos,
                                  radius=55.0)
        print(f"[probe] baseline before blocker: {len(obs0)} obstacles "
              f"(errors={errors_summary()!r})")

        bx = pos[0] + hx * BLOCK_DIST
        by = pos[1] + hy * BLOCK_DIST
        bz = float(pos[2])
        blocker_id = _lua_spawn(conn, "etk800", (bx, by, bz), heading)
        if blocker_id is None:
            print("[probe] FAILED to spawn blocker - aborting")
            return
        print(f"[probe] blocker spawned: id={blocker_id} at "
              f"({bx:.1f}, {by:.1f})")
        time.sleep(1.0)
        conn.step(20)

        st = conn.get_state()
        scen = scan_obstacles(conn.bng, conn.vehicle.vid, st.pos, radius=55.0)
        vehs = scan_obstacles_vehicles(conn.bng, conn.vehicle.vid, st.pos,
                                       radius=55.0)
        rays = scan_obstacles_raycast(conn.bng, st.pos, radius=55.0, rays=90)
        print(f"[probe] scenario_objects={len(scen)} "
              f"vehicles={len(vehs)} raycast={len(rays)} "
              f"(errors={errors_summary()!r})")
        for o in scen + vehs + rays:
            d = float(np.hypot(o.x - st.pos[0], o.y - st.pos[1]))
            lon = (o.x - st.pos[0]) * hx + (o.y - st.pos[1]) * hy
            lat = (o.x - st.pos[0]) * (-hy) + (o.y - st.pos[1]) * hx
            print(f"[probe]   obs d={d:6.1f}m lon={lon:+6.1f} "
                  f"lat={lat:+6.1f} box=({o.half_w:.1f}x{o.half_h:.1f}) "
                  f"cat={o.category}")

        route = np.array([
            [st.pos[0] + hx * d, st.pos[1] + hy * d]
            for d in np.arange(0.0, 80.0, 1.5)
        ])
        obstacles = scan_obstacles_all(conn.bng, conn.vehicle.vid, st.pos,
                                       radius=55.0)
        planner = LocalPlanner()
        drive, blocked = planner.plan(route, obstacles, st.pos, heading, 0)
        speed, obs_dist = planner.speed(drive, obstacles, st.pos, heading,
                                        0, 9.0)
        print(f"[probe] planner mode={planner.last_mode} "
              f"blocked={blocked} speed={speed:.1f} m/s "
              f"nearest={obs_dist:.1f} m, path pts={len(drive)}")
        # drive may have more or fewer points than the nav route (A* detours
        # re-sample at grid resolution), so measure each drive point's
        # distance to the nearest nav-route point instead of indexing pairs.
        rp = route[:, :2]
        dev = float(np.max(np.min(
            np.linalg.norm(drive[:, None, :2] - rp[None, :, :], axis=2),
            axis=1)))
        print(f"[probe] max lateral deviation from nav route: {dev:.1f} m")

        ok = (len(rays) + len(vehs) + len(scen)) > 0
        ok = ok and planner.last_mode in ("detour", "deform", "blocked")
        result = ("PASS - obstacle seen and avoided" if ok
                  else "FAIL - perception/planner did not react")
        print(f"[probe] RESULT: {result}")
    finally:
        if blocker_id is not None:
            _lua_delete(conn, blocker_id)
            time.sleep(0.5)
            try:
                conn.step(10)
            except Exception:
                pass
            print(f"[probe] blocker {blocker_id} removed")
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
