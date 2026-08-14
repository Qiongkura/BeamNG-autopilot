"""Game-side input watchdog (vehicle VM, comms-only injection).

BeamNG keeps applying the last client inputs after the Python process
disconnects or is killed, which is exactly what makes the car keep driving
or reversing on its own after an autopilot exit.  A killed process cannot
run any Python cleanup, so we inject a tiny Lua extension into the vehicle
VM (no file is written anywhere) that watches a heartbeat:

* ``arm()`` + regular ``heartbeat()`` calls from Python tell the watchdog
  the client is alive - it stays idle;
* when the heartbeat goes stale (client died / hard kill / crash), the
  watchdog zeroes throttle/steering/clutch, brakes while the car is still
  rolling, and engages the parking brake once it is (nearly) stopped - so
  the car always ends up parked and never drives itself.

The brake logic is deliberate: in arcade gearbox mode a brake input at
(nearly) zero speed is a REVERSE request, so the watchdog only brakes above
2 m/s and switches to the parking brake below that, where reverse can
never latch.
"""

from __future__ import annotations

import json


_MODULE_LUA = r"""
local watchdog = {}
watchdog.timeout = 2.5
watchdog.armed = false
watchdog.engaged = false
watchdog.heartbeat_ts = os.clockhp()

function watchdog.heartbeat()
  watchdog.heartbeat_ts = os.clockhp()
  watchdog.engaged = false
end

function watchdog.arm()
  watchdog.armed = true
  watchdog.heartbeat_ts = os.clockhp()
  watchdog.engaged = false
end

function watchdog.disarm()
  watchdog.armed = false
  watchdog.engaged = false
end

function watchdog._stop()
  input.event("throttle", 0, 1)
  input.event("steering", 0, 1)
  input.event("clutch", 0, 1)
  local spd = electrics.values.airspeed or 0
  if spd > 2.0 then
    -- rolling: brake hard.  Above 2 m/s even arcade treats brake as a
    -- brake, never as a reverse request.
    input.event("brake", 1, 1)
    input.event("parkingbrake", 0, 1)
  else
    -- (nearly) stopped: NEVER brake in arcade (brake-at-standstill is a
    -- reverse request); hold with the parking brake instead.
    input.event("brake", 0, 1)
    input.event("parkingbrake", 1, 1)
  end
end

function watchdog.updateGFX(dtSim)
  if not watchdog.armed then return end
  if watchdog.engaged then
    -- keep re-asserting until a fresh heartbeat arrives: a re-armed
    -- client (or player restart) is the only thing that disengages.
    watchdog._stop()
    return
  end
  if os.clockhp() - watchdog.heartbeat_ts > watchdog.timeout then
    watchdog.engaged = true
    watchdog._stop()
  end
end

return watchdog
"""

_INSTALL_CHUNK = (
    'package.loaded["autopilot/watchdog"] = (function()\n'
    + _MODULE_LUA
    + "\nend)()\n"
    'extensions.load("autopilot/watchdog")\n'
    'return rawget(_G, "autopilot_watchdog") ~= nil'
)

_MODULE_REF = "rawget(_G, 'autopilot_watchdog')"


def _cmd(conn, chunk: str, response: bool = True):
    """Run a Lua chunk on the attached vehicle; parse JSON when present."""
    try:
        resp = conn.vehicle.queue_lua_command(chunk, response=response)
    except Exception as exc:
        print(f"[watchdog] lua failed: {exc}")
        return None
    if resp is None:
        return None
    if isinstance(resp, str):
        try:
            return json.loads(resp)
        except (ValueError, TypeError):
            return resp
    return resp


def is_installed(conn) -> bool:
    """True when the watchdog module lives in the current vehicle VM."""
    return _cmd(conn, f"return {_MODULE_REF} ~= nil") is True


def install(conn) -> bool:
    """Inject (or force-reinject) the watchdog into the vehicle VM.

    BeamNG's ``extensions.load()`` is a no-op when the extension name is
    already registered, so an old injected copy (e.g. a previous version
    with a different timeout) would keep running forever.  Always unload
    any existing copy first and inject fresh, so the module version this
    process ships is guaranteed to be the live one.
    """
    _cmd(conn,
         'if rawget(_G, "autopilot_watchdog") ~= nil then '
         'pcall(function() extensions.unload("autopilot_watchdog") end) '
         'end return 1')
    ok = _cmd(conn, _INSTALL_CHUNK)
    return ok is True


def status(conn) -> dict:
    """Snapshot of the watchdog module ({} when not installed)."""
    resp = _cmd(
        conn,
        f"if {_MODULE_REF} then return jsonEncode("
        "{armed=autopilot_watchdog.armed,"
        "engaged=autopilot_watchdog.engaged,"
        "timeout=autopilot_watchdog.timeout}) "
        "else return jsonEncode({installed=false}) end",
    )
    if isinstance(resp, dict):
        resp.setdefault("installed", True)
        return resp
    return {"installed": False}


def arm(conn) -> bool:
    """Arm the watchdog (install first if needed) and beat once."""
    if not install(conn):
        return False
    return _cmd(conn, f"autopilot_watchdog.arm(); return 1") is not None


def heartbeat(conn) -> bool:
    """Refresh the watchdog timestamp; True when the module is live."""
    if not is_installed(conn):
        return False
    return _cmd(conn, f"autopilot_watchdog.heartbeat(); return 1") is not None


def disarm(conn) -> None:
    """Disarm the watchdog (safe to call when never armed)."""
    try:
        _cmd(conn, f"autopilot_watchdog.disarm(); return 1")
    except Exception:
        pass
