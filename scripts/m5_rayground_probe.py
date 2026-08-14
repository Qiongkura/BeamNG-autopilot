"""Cast rays that MUST hit and inspect the raw result semantics."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beamng_autopilot import config
from beamng_autopilot.connector import BeamNGConnector


_LUA = r"""
local out = {}
local x, y, z = %(x)s, %(y)s, %(z)s

-- 1) straight down from well above ground, terrain NOT ignored -> must hit
local r1 = Engine.castRay(vec3(x, y, z + 40), vec3(0, 0, -1), 80, false, false)
if r1 then
  out["down40"] = {ptx=r1.pt.x, pty=r1.pt.y, ptz=r1.pt.z}
else
  out["down40"] = "nil"
end

-- 2) straight down from 40 m up, terrain ignored -> likely no hit
local r2 = Engine.castRay(vec3(x, y, z + 40), vec3(0, 0, -1), 80, true, false)
out["down40_ignoreTerrain"] = r2 and {ptx=r2.pt.x, pty=r2.pt.y, ptz=r2.pt.z} or "nil"

-- 3) forward horizontal, terrain NOT ignored
local r3 = Engine.castRay(vec3(x, y, z + 1.15), vec3(1, 0, 0), 80, false, false)
out["forward_terrainOn"] = r3 and {ptx=r3.pt.x, pty=r3.pt.y, ptz=r3.pt.z} or "nil"

-- 4) the same forward ray but from 30 m behind the car in world terms, at
--    height 1.15: passes through/over where the car is.  If pt is a local
--    offset, this also reports direction-ish garbage; if it is real, we
--    should see a hit ~30 m ahead.
local hx, hy = math.cos(%(h)s), math.sin(%(h)s)
local ox = x - hx * 30
local oy = y - hy * 30
local r4 = Engine.castRay(vec3(ox, oy, z + 1.15), vec3(hx, hy, 0), 90, true, false)
out["towardCar"] = r4 and {ptx=r4.pt.x, pty=r4.pt.y, ptz=r4.pt.z} or "nil"

return jsonEncode(out)
"""


def main() -> None:
    conn = BeamNGConnector(config.DEFAULT_MAP, config.DEFAULT_VEHICLE,
                           port=config.PORT)
    try:
        conn.open(launch=False)
        conn.attach_vehicle(vid=None, already_open=True)
        st = conn.get_state()
        pos = st.pos
        heading = float(st.heading)
        print(f"[probe] ego pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) "
              f"heading={heading:.3f}")
        chunk = _LUA % {"x": pos[0], "y": pos[1], "z": pos[2], "h": heading}
        resp = conn.bng.queue_lua_command(chunk, response=True)
        print(f"[probe] results:\n{resp}")
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
