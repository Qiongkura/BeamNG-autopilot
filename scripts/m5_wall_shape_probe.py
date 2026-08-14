"""Compare how different TSStatic shapes respond to the 90-ray engine fan.

Spawns each candidate wall 15 m ahead of the ego, casts the same
Engine.castRay fan used by the autopilot, and reports how many hits land
on the wall and how far they span along the wall.  Every wall is deleted
before the next one is spawned.
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
WALL_THICK_M = 1.0
RADIUS = 55.0
RAYS = 90

SHAPES = [
    "/art/shapes/objects/jerseybarrier_3m.dae",
    "/art/shapes/objects/s_drywall.dae",
    "/art/shapes/objects/s_precast_block.dae",
    "/art/shapes/race/s_concrete_race_barrier.dae",
    "/levels/smallgrid/art/shapes/misc/gm_cube_1m.dae",
]


def _spawn(conn, shape: str, pos, heading: float) -> str | None:
    x, y, z = (float(v) for v in pos)
    chunk = (
        "local q = quatFromEuler(0, 0, %f + math.pi / 2.0)\n"
        "local obj = createObject('TSStatic')\n"
        "obj:setField('shapeName', 0, '%s')\n"
        "obj.scale = vec3(%f, %f, %f)\n"
        "obj.useInstanceRenderData = true\n"
        "obj:setField('instanceColor', 0, '1 1 1 1')\n"
        "obj:setField('collisionType', 0, 'Collision Mesh')\n"
        "obj:setField('decalType', 0, 'Collision Mesh')\n"
        "obj.canSave = false\n"
        "obj:registerObject('wall_shape_probe')\n"
        "local grp = scenetree.MissionGroup\n"
        "if grp then grp:addObject(obj) end\n"
        "obj:setPosRot(%f, %f, %f, q.x, q.y, q.z, q.w)\n"
        "local ok, msg = pcall(function() obj:enableCollision() end)\n"
        "return jsonEncode({id = tostring(obj:getId()), "
        "enable = ok, msg = tostring(msg)})"
        % (heading, shape, WALL_LEN_M, WALL_THICK_M, WALL_HEIGHT_M,
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
            "local o = scenetree.findObject('wall_shape_probe')\n"
            "if o and o.delete then o:delete() end\n"
            "return 1")
    except Exception as exc:
        print(f"[probe] delete failed: {exc}")


def _fan(conn, pos, center) -> list[dict]:
    x, y, z = (float(v) for v in pos)
    wx, wy = center
    chunk = r"""
local x, y, z = %(x)s, %(y)s, %(z)s
local wx, wy = %(wx)s, %(wy)s
local R = %(radius)s
local N = %(rays)s
local out = {}
local o = vec3(x, y, z + 1.05)
for i = 0, N - 1 do
  local a = (i / N) * 2 * math.pi
  local c, s = math.cos(a), math.sin(a)
  local t = vec3(x + c * R, y + s * R, z + 1.05)
  local res = Engine.castRay(o, t, true, false)
  if res and res.pt then
    local d = math.sqrt((res.pt.x - x)^2 + (res.pt.y - y)^2)
    if d < R and math.abs(res.pt.x - wx) < 20 and
       math.abs(res.pt.y - wy) < 20 then
      out[#out + 1] = {x = res.pt.x, y = res.pt.y, d = d}
    end
  end
end
return jsonEncode(out)
""" % {"x": x, "y": y, "z": z, "wx": wx, "wy": wy,
       "radius": RADIUS, "rays": RAYS}
    resp = conn.bng.queue_lua_command(chunk, response=True)
    try:
        return json.loads(str(resp))
    except (ValueError, TypeError):
        print(f"[probe] fan response not JSON: {resp!r}")
        return []


def main() -> None:
    conn = BeamNGConnector(config.DEFAULT_MAP, config.DEFAULT_VEHICLE,
                           port=config.PORT)
    try:
        conn.open(launch=False)
        conn.attach_vehicle(vid=None, already_open=True)
        st = conn.get_state()
        pos = st.pos
        heading = float(st.heading)
        hx, hy = math.cos(heading), math.sin(heading)
        print(f"[probe] ego pos=({pos[0]:.1f}, {pos[1]:.1f}) "
              f"heading={heading:.2f}")
        for shape in SHAPES:
            cx = pos[0] + hx * WALL_DIST
            cy = pos[1] + hy * WALL_DIST
            wz = float(pos[2]) - 0.35
            wid = _spawn(conn, shape, (cx, cy, wz), heading)
            if wid is None:
                print(f"[probe] {shape}: spawn failed")
                continue
            time.sleep(0.5)
            conn.step(10)
            hits = _fan(conn, pos, (cx, cy))
            if hits:
                # Offset along the wall axis (perpendicular to heading).
                px, py = -hy, hx
                on_wall = []
                for h in hits:
                    dx = h["x"] - cx
                    dy = h["y"] - cy
                    along = dx * px + dy * py
                    across = abs(dx * hx + dy * hy)
                    if abs(along) <= WALL_LEN_M / 2.0 + 0.5 and \
                            across <= max(2.0, WALL_THICK_M / 2.0 + 1.0):
                        on_wall.append(h)
                hits = on_wall
                offs = [(h["x"] - cx) * px + (h["y"] - cy) * py
                        for h in hits]
                if not offs:
                    print(f"[probe] {shape}: hits on wall=0")
                    _delete(conn)
                    time.sleep(0.3)
                    conn.step(5)
                    continue
                span = max(offs) - min(offs)
                ds = sorted(h["d"] for h in hits)
                print(f"[probe] {shape}: hits={len(hits)} "
                      f"span={span:.1f}m min_d={ds[0]:.1f}m "
                      f"max_d={ds[-1]:.1f}m")
            else:
                print(f"[probe] {shape}: hits=0")
            _delete(conn)
            time.sleep(0.3)
            conn.step(5)
    finally:
        try:
            _delete(conn)
            conn.step(5)
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
