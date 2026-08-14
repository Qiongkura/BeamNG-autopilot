"""Compare ray fan configurations against a spawned runtime wall.

Attaches to the running session, spawns the same 12 m wall used by the
wall probe, then casts a few Engine.castRay fan variants (full circle vs
forward sector, terrain on/off, render geometry on/off) and reports how
many hits land on the wall and what obstacle box the autopilot clustering
would produce.  The wall is deleted before exit.
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
from beamng_autopilot.perception import _cluster_points

WALL_DIST = 15.0
WALL_LEN_M = 12.0
WALL_HEIGHT_M = 3.0
WALL_THICK_M = 1.2
RADIUS = 55.0


def _spawn(conn, pos, heading: float) -> str | None:
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
        "obj:registerObject('wall_fan_probe')\n"
        "local grp = scenetree.MissionGroup\n"
        "if grp then grp:addObject(obj) end\n"
        "obj:setPosRot(%f, %f, %f, q.x, q.y, q.z, q.w)\n"
        "local ok, msg = pcall(function() obj:enableCollision() end)\n"
        "return jsonEncode({id = tostring(obj:getId()), "
        "enable = ok, msg = tostring(msg)})"
        % (heading, WALL_LEN_M, WALL_THICK_M, WALL_HEIGHT_M,
           x, y, z)
    )
    resp = conn.bng.queue_lua_command(chunk, response=True)
    try:
        return json.loads(str(resp)).get("id")
    except (ValueError, TypeError):
        print(f"[probe] spawn response not JSON: {resp!r}")
        return None


def _delete(conn) -> None:
    try:
        conn.bng.queue_lua_command(
            "local o = scenetree.findObject('wall_fan_probe')\n"
            "if o and o.delete then o:delete() end\n"
            "return 1")
    except Exception as exc:
        print(f"[probe] delete failed: {exc}")


def _fan(conn, pos, heading, n, terrain, render_geom, sector_deg=360.0):
    x, y, z = (float(v) for v in pos)
    hx, hy = math.cos(heading), math.sin(heading)
    half = math.radians(sector_deg / 2.0)
    a0 = heading - half
    chunk = r"""
local x, y, z = %(x)s, %(y)s, %(z)s
local a0 = %(a0)s
local sector = %(sector)s
local R = %(radius)s
local N = %(n)s
local terrain = %(terrain)s
local geom = %(geom)s
local out = {}
local o = vec3(x, y, z + 1.05)
for i = 0, N - 1 do
  local a = a0 + (i / math.max(1, N - 1)) * sector
  local c, s = math.cos(a), math.sin(a)
  local t = vec3(x + c * R, y + s * R, z + 1.05)
  local res = Engine.castRay(o, t, terrain, geom)
  if res and res.pt then
    local d = math.sqrt((res.pt.x - x)^2 + (res.pt.y - y)^2)
    if d < R then
      out[#out + 1] = {x = res.pt.x, y = res.pt.y, z = res.pt.z, d = d}
    end
  end
end
return jsonEncode(out)
""" % {"x": x, "y": y, "z": z, "a0": a0, "sector": math.radians(sector_deg),
       "radius": RADIUS, "n": n,
       "terrain": "true" if terrain else "false",
       "geom": "true" if render_geom else "false"}
    resp = conn.bng.queue_lua_command(chunk, response=True)
    try:
        data = json.loads(str(resp))
        return data or [], len(str(resp))
    except (ValueError, TypeError):
        return [], len(str(resp))


def _wall_hits(hits, center, heading):
    cx, cy = center
    px, py = -math.sin(heading), math.cos(heading)
    on = []
    for h in hits:
        dx = h["x"] - cx
        dy = h["y"] - cy
        along = abs(dx * px + dy * py)
        across = abs(dx * math.cos(heading) + dy * math.sin(heading))
        if along <= WALL_LEN_M / 2.0 + 0.6 and \
                across <= max(2.0, WALL_THICK_M / 2.0 + 1.0):
            on.append((h["x"], h["y"]))
    return on


def _report(conn, pos, center, heading, name, n, terrain, geom, sector):
    t0 = time.time()
    hits, size = _fan(conn, pos, heading, n, terrain, geom, sector)
    dt = time.time() - t0
    on = _wall_hits(hits, center, heading)
    pts = [(float(h["x"]), float(h["y"])) for h in hits]
    boxes = _cluster_points(pts, cell=2.0)
    wall_boxes = [b for b in boxes if b.label == "wall"]
    print(f"[probe] {name}: rays={n} terrain={terrain} geom={geom} "
          f"sector={sector:.0f} -> total_hits={len(hits)} wall_hits={len(on)} "
          f"resp={size}B time={dt:.2f}s boxes={len(boxes)} "
          f"wall_boxes={len(wall_boxes)}")
    if on:
        offs = [(p[0] - center[0]) * -math.sin(heading) +
                (p[1] - center[1]) * math.cos(heading) for p in on]
        print(f"[probe]   wall span={max(offs) - min(offs):.1f}m "
              f"points={['({:.1f},{:.1f})'.format(*p) for p in on]}")
    for b in wall_boxes:
        print(f"[probe]   wall box=({b.half_w:.1f}x{b.half_h:.1f}) "
              f"at=({b.x:.1f}, {b.y:.1f})")


def main() -> None:
    conn = BeamNGConnector(config.DEFAULT_MAP, config.DEFAULT_VEHICLE,
                           port=config.PORT)
    try:
        conn.open(launch=False)
        conn.attach_vehicle(vid=None, already_open=True)
        st = conn.get_state()
        pos = st.pos
        heading = float(st.heading)
        cx = float(pos[0]) + math.cos(heading) * WALL_DIST
        cy = float(pos[1]) + math.sin(heading) * WALL_DIST
        wz = float(pos[2]) - 0.35
        wid = _spawn(conn, (cx, cy, wz), heading)
        if wid is None:
            print("[probe] spawn failed")
            return
        print(f"[probe] ego ({pos[0]:.1f}, {pos[1]:.1f}) wall {wid} "
              f"at ({cx:.1f}, {cy:.1f})")
        time.sleep(1.0)
        conn.step(20)
        _report(conn, pos, (cx, cy), heading, "full90_terrain", 90, True,
                False, 360.0)
        _report(conn, pos, (cx, cy), heading, "full180_terrain", 180, True,
                False, 360.0)
        _report(conn, pos, (cx, cy), heading, "full360_noterrain", 360, False,
                False, 360.0)
        _report(conn, pos, (cx, cy), heading, "fwd180_noterrain", 180, False,
                False, 90.0)
        _report(conn, pos, (cx, cy), heading, "fwd180_noterrain_geom", 180,
                False, True, 90.0)
    finally:
        _delete(conn)
        try:
            conn.step(5)
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
