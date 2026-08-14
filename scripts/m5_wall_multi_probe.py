"""Probe a runtime wall built from several collision-complete pieces.

A single scaled TSStatic only exposes a small collision patch around its
origin (observed ~3 m of a 12 m cube), so this probe builds a 12 m wall
from four unscaled jersey barriers and checks whether the forward
Engine.castRay fan produces a long, dense hit chain that clusters into a
``wall`` box.
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
from beamng_autopilot.perception import (
    _cluster_points,
    _oriented_dims,
    scan_obstacles_all,
)

WALL_DIST = 15.0
WALL_LEN_M = 12.0
PIECES = 4
PIECE_GAP = 0.15
RADIUS = 55.0

SHAPE = "/art/shapes/objects/jerseybarrier_3m.dae"


def _spawn_piece(conn, name, x, y, z, heading) -> str | None:
    chunk = (
        "local q = quatFromEuler(0, 0, %f + math.pi / 2.0)\n"
        "local obj = createObject('TSStatic')\n"
        "obj:setField('shapeName', 0, '%s')\n"
        "obj.scale = vec3(1, 1, 1)\n"
        "obj.useInstanceRenderData = true\n"
        "obj:setField('instanceColor', 0, '1 1 1 1')\n"
        "obj:setField('collisionType', 0, 'Collision Mesh')\n"
        "obj:setField('decalType', 0, 'Collision Mesh')\n"
        "obj.canSave = false\n"
        "obj:registerObject('%s')\n"
        "local grp = scenetree.MissionGroup\n"
        "if grp then grp:addObject(obj) end\n"
        "obj:setPosRot(%f, %f, %f, q.x, q.y, q.z, q.w)\n"
        "local ok, msg = pcall(function() obj:enableCollision() end)\n"
        "return jsonEncode({id = tostring(obj:getId())})"
        % (heading, SHAPE, name, x, y, z)
    )
    resp = conn.bng.queue_lua_command(chunk, response=True)
    try:
        return json.loads(str(resp)).get("id")
    except (ValueError, TypeError):
        print(f"[probe] spawn {name} failed: {resp!r}")
        return None


def _diag_pieces(conn) -> None:
    chunk = r"""
local out = {}
for i = 0, 3 do
  local o = scenetree.findObject('wall_piece_' .. i)
  if o then
    local okp, p = pcall(function() return o:getPosition() end)
    local okb, b = pcall(function() return o:getWorldBox() end)
    local oks, s = pcall(function() return o:getField('shapeName', 0) end)
    local okc, c = pcall(function() return o:getField('collisionType', 0) end)
    out[#out + 1] = {
      id = tostring(o:getId()),
      pos = okp and {x = p.x, y = p.y, z = p.z} or nil,
      box = okb and tostring(b) or nil,
      shape = oks and tostring(s) or nil,
      collision = okc and tostring(c) or nil,
    }
  end
end
return jsonEncode(out)
"""
    try:
        resp = conn.bng.queue_lua_command(chunk, response=True)
        data = json.loads(str(resp))
    except (ValueError, TypeError) as exc:
        print(f"[probe] piece diag failed: {exc!r} {resp!r}")
        return
    for row in data or []:
        print(f"[probe]   piece {row}")


def _delete_all(conn, count: int) -> None:
    for i in range(count):
        try:
            conn.bng.queue_lua_command(
                "local o = scenetree.findObject('wall_piece_%d')\n"
                "if o and o.delete then o:delete() end\n"
                "return 1" % i)
        except Exception as exc:
            print(f"[probe] delete {i} failed: {exc}")


def _fan(conn, pos, heading, n=180, sector_deg=90.0):
    x, y, z = (float(v) for v in pos)
    a0 = heading - math.radians(sector_deg / 2.0)
    chunk = r"""
local x, y, z = %(x)s, %(y)s, %(z)s
local a0 = %(a0)s
local sector = %(sector)s
local R = %(radius)s
local N = %(n)s
local out = {}
local h = %(h)s
local o = vec3(x, y, z + h)
for i = 0, N - 1 do
  local a = a0 + (i / math.max(1, N - 1)) * sector
  local c, s = math.cos(a), math.sin(a)
  local t = vec3(x + c * R, y + s * R, z + h)
  local res = Engine.castRay(o, t, false, false)
  if res and res.pt then
    local d = math.sqrt((res.pt.x - x)^2 + (res.pt.y - y)^2)
    if d < R then
      out[#out + 1] = {x = res.pt.x, y = res.pt.y, d = d}
    end
  end
end
return jsonEncode(out)
""" % {"x": x, "y": y, "z": z, "a0": a0,
       "sector": math.radians(sector_deg), "radius": RADIUS, "n": n,
       "h": 0.45}
    resp = conn.bng.queue_lua_command(chunk, response=True)
    try:
        data = json.loads(str(resp))
        return data or [], len(str(resp))
    except (ValueError, TypeError):
        print(f"[probe] fan response not JSON: {resp!r}")
        return [], len(str(resp))


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
        px, py = -math.sin(heading), math.cos(heading)
        ids = []
        piece_len = (WALL_LEN_M - (PIECES - 1) * PIECE_GAP) / PIECES
        start = -(WALL_LEN_M / 2.0)
        for i in range(PIECES):
            along = start + i * (piece_len + PIECE_GAP)
            wx = cx + px * along
            wy = cy + py * along
            wid = _spawn_piece(conn, f"wall_piece_{i}", wx, wy, wz, heading)
            if wid is None:
                print("[probe] abort: piece spawn failed")
                return
            ids.append(wid)
        print(f"[probe] wall pieces={ids} along "
              f"({cx - px * WALL_LEN_M / 2:.1f}, {cy - py * WALL_LEN_M / 2:.1f})"
              f" -> ({cx + px * WALL_LEN_M / 2:.1f}, "
              f"{cy + py * WALL_LEN_M / 2:.1f})")
        time.sleep(1.0)
        conn.step(20)
        _diag_pieces(conn)

        hits, size = _fan(conn, pos, heading)
        on = []
        for h in hits:
            dx = h["x"] - cx
            dy = h["y"] - cy
            along = dx * px + dy * py
            across = abs(dx * math.cos(heading) + dy * math.sin(heading))
            if abs(along) <= WALL_LEN_M / 2.0 + 1.0 and across <= 1.5:
                on.append((h["x"], h["y"]))
        offs = [p[0] * px + p[1] * py for p in on]
        if on:
            xs = [p[0] for p in on]
            ys = [p[1] for p in on]
            print(f"[probe] wall x={min(xs):.1f}..{max(xs):.1f} "
                  f"y={min(ys):.1f}..{max(ys):.1f} "
                  f"span={max(offs) - min(offs):.1f}m "
                  f"bbox=({max(xs) - min(xs):.1f}x{max(ys) - min(ys):.1f})")
        print(f"[probe] fan hits={len(hits)} on_wall={len(on)} "
              f"span={max(offs) - min(offs) if offs else 0:.1f}m "
              f"resp={size}B")
        pts = [(float(h["x"]), float(h["y"])) for h in hits]
        boxes = _cluster_points(pts, cell=2.0)
        wall_boxes = [b for b in boxes if b.label == "wall"]
        wall_only = _cluster_points(
            [(float(x), float(y)) for x, y in on], cell=2.0)
        if on:
            major, minor = _oriented_dims(on)
            print(f"[probe] wall oriented major={major:.1f}m "
                  f"minor={minor:.1f}m")
        print(f"[probe] clustered boxes={len(boxes)} wall_boxes={len(wall_boxes)}")
        print(f"[probe] wall-only boxes={len(wall_only)}")
        for b in boxes:
            lon = (b.x - cx) * math.cos(heading) + \
                (b.y - cy) * math.sin(heading)
            lat = (b.x - cx) * px + (b.y - cy) * py
            print(f"[probe]   box=({b.half_w:.1f}x{b.half_h:.1f}) "
                  f"lon={lon:.1f} lat={lat:.1f} label={b.label!r}")
        for b in wall_only:
            lon = (b.x - cx) * math.cos(heading) + \
                (b.y - cy) * math.sin(heading)
            lat = (b.x - cx) * px + (b.y - cy) * py
            print(f"[probe]   wall_only=({b.half_w:.1f}x{b.half_h:.1f}) "
                  f"lon={lon:.1f} lat={lat:.1f} label={b.label!r}")

        obs = scan_obstacles_all(
            conn.bng, conn.vehicle.vid, pos, radius=RADIUS)
        print(f"[probe] scan_obstacles_all={len(obs)}")
        for ob in obs:
            dx = ob.x - pos[0]
            dy = ob.y - pos[1]
            lon = dx * math.cos(heading) + dy * math.sin(heading)
            lat = dx * px + dy * py
            print(f"[probe]   obs=({ob.half_w:.1f}x{ob.half_h:.1f}) "
                  f"lon={lon:.1f} lat={lat:.1f} cat={ob.category} "
                  f"label={ob.label!r}")
    finally:
        _delete_all(conn, PIECES)
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
