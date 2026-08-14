"""Dump the full structure returned by Engine.castRay on this build."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beamng_autopilot import config
from beamng_autopilot.connector import BeamNGConnector


_DUMP_LUA = r"""
local origin = vec3(0, 0, 1.15)
local res = Engine.castRay(origin, vec3(1, 0, 0), 80, true, false)
local out = {}
if res == nil then
  out["RESULT"] = "nil"
  return jsonEncode(out)
end
out["RESULT"] = type(res)
local mt = getmetatable(res)
out["metatable"] = mt and tostring(mt) or "nil"

-- Probe likely fields one at a time, safely.
local candidates = {
  "pt", "point", "pos", "position", "t", "distance", "d", "len",
  "n", "normal", "hit", "hasHit", "blocked", "entity", "object",
  "x", "y", "z",
}
for _, f in ipairs(candidates) do
  local ok, v = pcall(function() return res[f] end)
  if ok then
    out["field_" .. f] = "type=" .. type(v) .. " val=" .. tostring(v)
  else
    out["field_" .. f] = "ERROR: " .. tostring(v)
  end
end

-- Any metatable __index keys?
if mt and mt.__index then
  local ok, idx = pcall(function() return mt.__index end)
  if ok then
    local keys = {}
    local ok2, klist = pcall(function()
      for k in pairs(idx) do keys[#keys + 1] = tostring(k) end
    end)
    out["mt_index_keys"] = ok2 and table.concat(keys, ",") or "n/a"
  end
end

-- Alternate call shapes.
local r2 = Engine.castRay(origin, vec3(1, 0, 0), 80, false, false)
out["alt_ignoreTerrain=false"] = r2 and "hit" or "nil"
local r3 = Engine.castRay(origin, vec3(1, 0, 0), 80)
out["alt_noFlags"] = r3 and "hit" or "nil"
return jsonEncode(out)
"""


def main() -> None:
    conn = BeamNGConnector(config.DEFAULT_MAP, config.DEFAULT_VEHICLE,
                           port=config.PORT)
    try:
        conn.open(launch=False)
        conn.attach_vehicle(vid=None, already_open=True)
        print("[probe] attached")
        resp = conn.bng.queue_lua_command(_DUMP_LUA, response=True)
        print(f"[probe] castRay structure dump:\n{resp}")
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
