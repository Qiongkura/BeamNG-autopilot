"""Diagnose why newly spawned TSStatic walls are invisible to castRayStatic.

Attaches to the RUNNING session, spawns a wall with ``setPosRot``, waits,
tries ``enableCollision``, then compares ``castRayStatic`` and
``Engine.castRay`` full 2D fans around the ego.  The wall is deleted
before the script exits, so the session is left untouched.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beamng_autopilot import config
from beamng_autopilot.connector import BeamNGConnector

WALL_DIST = 15.0
WALL_LEN_M = 12.0
WALL_HEIGHT_M = 3.0
WALL_THICK_M = 1.2
RADIUS = 55.0
RAYS = 90


def _spawn_wall(conn, pos, heading: float) -> str | None:
    x, y, z = (float(v) for v in pos)
    chunk = (
        "local q = quatFromEuler(0, 0, %f + math.pi / 2.0)\n"
        "local obj = createObject('TSStatic')\n"
        "obj:setField('shapeName', 0, "
        "'/levels/smallgrid/art/shapes/misc/gm_cube_1m.dae')\n"
        "obj.scale = vec3(%f, %f, %f)\n"
        "obj:setField('collisionType', 0, 'Collision Mesh')\n"
        "obj:setField('decalType', 0, 'Collision Mesh')\n"
        "obj.canSave = false\n"
        "obj:registerObject('wall_probe2')\n"
        "local grp = scenetree.MissionGroup\n"
        "if grp then grp:addObject(obj) end\n"
        "obj:setPosRot(%f, %f, %f, q.x, q.y, q.z, q.w)\n"
        "local ok, msg = pcall(function() obj:enableCollision() end)\n"
        "return jsonEncode({id = tostring(obj:getId()), enable = ok, "
        "msg = tostring(msg)})"
        % (heading, WALL_LEN_M, WALL_THICK_M, WALL_HEIGHT_M, x, y, z)
    )
    resp = conn.bng.queue_lua_command(chunk, response=True)
    try:
        data = json.loads(str(resp))
        print(f"[probe] enableCollision -> ok={data.get('enable')} "
              f"msg={data.get('msg')!r}")
        return data.get("id")
    except (ValueError, TypeError) as exc:
        print(f"[probe] spawn response not JSON: {resp!r} ({exc})")
        return None


def _delete_wall(conn) -> None:
    try:
        conn.bng.queue_lua_command(
            "local o = scenetree.findObject('wall_probe2')\n"
            "if o and o.delete then o:delete() end\n"
            "return 1")
    except Exception as exc:
        print(f"[probe] delete failed: {exc}")


def _lua_diag(conn, pos, heading):
    x, y, z = (float(v) for v in pos)
    hx, hy = math.cos(heading), math.sin(heading)
    chunk = r"""
local x, y, z = %(x)s, %(y)s, %(z)s
local hx, hy = %(hx)s, %(hy)s
local R = %(radius)s
local N = %(rays)s
local out = {}

-- Object state.
local o = scenetree.findObject('wall_probe2')
if o then
  local okp, p = pcall(function() return o:getPosition() end)
  out.pos = okp and {x = p.x, y = p.y, z = p.z} or nil
  local okb, b = pcall(function() return o:getWorldBox() end)
  if okb and b then
    out.box = {
      min = {x = b.minExtents.x, y = b.minExtents.y, z = b.minExtents.z},
      max = {x = b.maxExtents.x, y = b.maxExtents.y, z = b.maxExtents.z},
      str = tostring(b),
    }
  else
    out.box = {str = tostring(b)}
  end
  local okc, c = pcall(function() return o:getField('collisionType', 0) end)
  out.collisionType = okc and tostring(c) or nil
  local oks, s = pcall(function() return o:getField('shapeName', 0) end)
  out.shapeName = oks and tostring(s) or nil
  local okr, r = pcall(function() return o:getField('rotation', 0) end)
  out.rotationField = okr and tostring(r) or nil
end

-- Scenario object database.
local objs = scenario_objects and scenario_objects.getObjects()
out.scenario_count = objs and #objs or -1
local near = {}
if objs then
  for i = 1, #objs do
    local e = objs[i]
    if e and e.pos then
      local dx = e.pos.x - (x + hx * %(wd)s)
      local dy = e.pos.y - (y + hy * %(wd)s)
      if dx * dx + dy * dy < 9.0 then
        near[#near + 1] = {id = tostring(e.id), x = e.pos.x, y = e.pos.y,
                           sx = e.size and e.size.x or nil,
                           sy = e.size and e.size.y or nil}
      end
    end
  end
end
out.scenario_near_wall = near

-- Full fans.
local static_hits = {}
local engine_hits = {}
local engine_hits_geom = {}
local o0 = vec3(x, y, z + 1.05)
for i = 0, N - 1 do
  local a = (i / N) * 2 * math.pi
  local c, s = math.cos(a), math.sin(a)
  local sd = castRayStatic(o0, vec3(c, s, 0), R)
  if sd and sd < R then
    static_hits[#static_hits + 1] = {x = x + c * sd, y = y + s * sd,
                                     z = z + 1.05, d = sd}
  end
  local t = vec3(x + c * R, y + s * R, z + 1.05)
  local r1 = Engine.castRay(o0, t, true, false)
  if r1 then
    local d1 = math.sqrt((r1.pt.x - x)^2 + (r1.pt.y - y)^2)
    engine_hits[#engine_hits + 1] = {x = r1.pt.x, y = r1.pt.y,
                                     z = r1.pt.z, d = d1}
  end
  local r2 = Engine.castRay(o0, t, true, true)
  if r2 then
    local d2 = math.sqrt((r2.pt.x - x)^2 + (r2.pt.y - y)^2)
    engine_hits_geom[#engine_hits_geom + 1] = {x = r2.pt.x, y = r2.pt.y,
                                               z = r2.pt.z, d = d2}
  end
end
out.static_hits = static_hits
out.engine_hits = engine_hits
out.engine_hits_geom = engine_hits_geom
return jsonEncode(out)
""" % {
        "x": x, "y": y, "z": z, "hx": hx, "hy": hy,
        "radius": RADIUS, "rays": RAYS, "wd": WALL_DIST,
    }
    return conn.bng.queue_lua_command(chunk, response=True)


def _summarize(name, hits, pos, wall_xy=None):
    if not hits:
        print(f"[probe] {name}: 0 hits")
        return
    ds = sorted(float(h["d"]) for h in hits)
    print(f"[probe] {name}: {len(hits)} hits, min={ds[0]:.1f}m "
          f"max={ds[-1]:.1f}m")
    near_wall = [h for h in hits if abs(h["d"] - WALL_DIST) < 12.0]
    if wall_xy is not None:
        near_wall = [
            h for h in hits
            if math.hypot(h["x"] - wall_xy[0], h["y"] - wall_xy[1]) < 5.0
        ]
    print(f"[probe] {name}: hits at wall: {len(near_wall)}")
    for h in near_wall[:5]:
        print(f"[probe]   d={h['d']:.1f} pt=({h['x']:.1f}, {h['y']:.1f}, "
              f"{h['z']:.2f})")


def main() -> None:
    conn = BeamNGConnector(config.DEFAULT_MAP, config.DEFAULT_VEHICLE,
                           port=config.PORT)
    wall_id = None
    try:
        conn.open(launch=False)
        conn.attach_vehicle(vid=None, already_open=True)
        st = conn.get_state()
        pos = st.pos
        heading = float(st.heading)
        print(f"[probe] ego pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) "
              f"heading={heading:.3f}")

        wx = pos[0] + math.cos(heading) * WALL_DIST
        wy = pos[1] + math.sin(heading) * WALL_DIST
        wz = float(pos[2]) - 0.35
        wall_id = _spawn_wall(conn, (wx, wy, wz), heading)
        if wall_id is None:
            print("[probe] spawn failed, aborting")
            return
        print(f"[probe] wall spawned: {wall_id} at "
              f"({wx:.1f}, {wy:.1f}, {wz:.1f})")
        time.sleep(2.0)
        conn.step(60)

        resp = _lua_diag(conn, pos, heading)
        print(f"[probe] diag response length={len(str(resp))}")
        try:
            data = json.loads(str(resp))
        except (ValueError, TypeError) as exc:
            print(f"[probe] diag response not JSON: {resp!r} ({exc})")
            return

        print(f"[probe] object pos={data.get('pos')} "
              f"box={data.get('box', {}).get('str')}")
        if "min" in data.get("box", {}):
            b = data["box"]
            print(f"[probe] world box min={b['min']} max={b['max']}")
        print(f"[probe] collisionType={data.get('collisionType')} "
              f"shape={data.get('shapeName')} "
              f"rotationField={data.get('rotationField')}")
        print(f"[probe] scenario_objects count={data.get('scenario_count')} "
              f"near_wall={data.get('scenario_near_wall')}")

        _summarize("castRayStatic", data.get("static_hits") or [], pos,
                   (wx, wy))
        _summarize("Engine.castRay", data.get("engine_hits") or [], pos,
                   (wx, wy))
        _summarize("Engine.castRay+geom", data.get("engine_hits_geom") or [],
                   pos, (wx, wy))
    finally:
        if wall_id is not None:
            _delete_wall(conn)
            time.sleep(0.5)
            try:
                conn.step(10)
            except Exception:
                pass
            print("[probe] wall removed")
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
