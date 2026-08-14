"""Live comparison: Engine.castRay vs castRayStatic against a real blocker.

The game's own AI uses the global ``castRayStatic(origin, dir, dist)`` for
obstacle detection - it returns a plain hit *distance* (number), and any
return >= ``dist`` means "no hit".  ``Engine.castRay`` returns a table and
missed a parked car 30 m ahead in an earlier probe, so this probe spawns a
blocker, casts a horizontal fan with BOTH APIs, and also checks downward
rays (ground) so we can pick the source that actually sees the world.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from beamng_autopilot import config
from beamng_autopilot.connector import BeamNGConnector


_LUA = r"""
local px, py, pz = %(x)s, %(y)s, %(z)s
local hx, hy = %(hx)s, %(hy)s
local R = %(radius)s
local o = vec3(px, py, pz + 1.15)
local out = {engine = {}, static = {}}
local N = %(n)s
for i = 0, N - 1 do
  local a = (i / N) * 2 * math.pi
  local c, s = math.cos(a), math.sin(a)
  -- castRayStatic(origin, dir, dist): returns distance, >= dist = no hit.
  -- Direction is passed normalized; the engine also accepts full vectors.
  local sd = castRayStatic(o, vec3(c, s, 0), R)
  if sd < R then
    out.static[#out.static + 1] = {
      x = px + c * sd, y = py + s * sd, z = pz + 1.15, d = sd,
    }
  end
  -- Engine.castRay absolute-target form, both flags on.
  local t = vec3(px + c * R, py + s * R, pz + 1.15)
  local res = Engine.castRay(o, t, true, true)
  if res then
    out.engine[#out.engine + 1] = {
      x = res.pt.x, y = res.pt.y, z = res.pt.z,
      d = math.sqrt((res.pt.x - px)^2 + (res.pt.y - py)^2),
    }
  end
end
-- Ground sanity checks.
local up = vec3(px, py, pz + 40)
local down = vec3(px, py, pz - 5)
out.ground_static = castRayStatic(up, vec3(0, 0, -1), 50)
local rg = Engine.castRay(up, down, true, true)
out.ground_engine = rg and {x = rg.pt.x, y = rg.pt.y, z = rg.pt.z} or "nil"
return jsonEncode(out)
"""


def _lua_spawn(conn, model: str, pos, heading: float) -> str | None:
    x, y, z = (float(v) for v in pos)
    dx, dy = math.cos(heading), math.sin(heading)
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
        return None


def _lua_delete(conn, vid: str) -> None:
    try:
        conn.bng.queue_lua_command(
            "local v = scenetree.findObjectById('" + vid + "') "
            "if v and v.delete then v:delete() end return 1")
    except Exception:
        pass


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
        print(f"[probe] ego pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) "
              f"heading={heading:.3f} fwd=({hx:.3f}, {hy:.3f})")

        # Baseline: cast the fan before any blocker exists.
        chunk = _LUA % {"x": pos[0], "y": pos[1], "z": pos[2],
                        "hx": hx, "hy": hy, "radius": 55.0, "n": 90}
        resp = conn.bng.queue_lua_command(chunk, response=True)
        data = json.loads(str(resp))
        print(f"[probe] BASELINE engine={len(data['engine'])} "
              f"static={len(data['static'])} "
              f"ground_static={data['ground_static']:.2f} "
              f"ground_engine={data['ground_engine']}")

        # Spawn a parked car 30 m ahead.
        bx = pos[0] + hx * 30.0
        by = pos[1] + hy * 30.0
        blocker_id = _lua_spawn(conn, "etk800", (bx, by, pos[2]), heading)
        print(f"[probe] blocker spawned: {blocker_id}")
        time.sleep(1.0)
        conn.step(20)

        resp = conn.bng.queue_lua_command(chunk, response=True)
        data = json.loads(str(resp))
        eng = data["engine"]
        sta = data["static"]
        print(f"[probe] WITH BLOCKER engine={len(eng)} static={len(sta)}")
        fwd_eng = [e for e in eng if abs(e["d"] - 30.0) < 12.0]
        fwd_sta = [s for s in sta if abs(s["d"] - 30.0) < 12.0]
        print(f"[probe]   engine hits near 30m: {len(fwd_eng)} "
              f"(sample d={fwd_eng[0]['d']:.1f} "
              f"pt=({fwd_eng[0]['x']:.1f},{fwd_eng[0]['y']:.1f})" if fwd_eng
              else "[probe]   engine hits near 30m: 0")
        print(f"[probe]   static hits near 30m: {len(fwd_sta)} "
              f"(sample d={fwd_sta[0]['d']:.1f} "
              f"pt=({fwd_sta[0]['x']:.1f},{fwd_sta[0]['y']:.1f})" if fwd_sta
              else "[probe]   static hits near 30m: 0")
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
