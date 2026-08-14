# Probe the gearbox part internals so we can compute the numeric control
# input for a forward gear ("D" for autos, "1" for manuals) instead of
# hardcoding beamngpy's -1/0/1 convention.

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beamng_autopilot import config
from beamng_autopilot.connector import BeamNGConnector


def lua(conn, chunk: str):
    resp = conn.vehicle.queue_lua_command(chunk, response=True)
    if resp is None:
        return None
    if isinstance(resp, str):
        try:
            return json.loads(resp)
        except (ValueError, TypeError):
            return resp
    return resp


def main() -> None:
    conn = BeamNGConnector(config.DEFAULT_MAP, config.DEFAULT_VEHICLE,
                           port=config.PORT)
    try:
        conn.open(launch=False)
        conn.attach_vehicle(vid=None, already_open=True)
        print("[diag] attached")

        chunks = [
            ("gearbox part", "local v=vehicle\n"
                             "local gb=v:getGearbox()\n"
                             "if not gb then return jsonEncode({err='no gb'}) end\n"
                             "return jsonEncode({type=gb.type, gears=gb.gears, "
                             "current=electrics.values.gear_index, "
                             "gear=tostring(electrics.values.gear)})"),
            ("controller gearbox", "local c=controller\n"
                                   "local gb=c and c.gearbox\n"
                                   "if not gb then return jsonEncode({err='no c.gearbox'}) end\n"
                                   "return jsonEncode({gears=gb.gears})"),
            ("gearboxGears()", "local c=controller\n"
                               "local f=c.getGearboxGears\n"
                               "if not f then return jsonEncode({err='no fn'}) end\n"
                               "return jsonEncode({gears=f()})"),
        ]
        for label, chunk in chunks:
            try:
                print(f"  {label}: {lua(conn, chunk)}")
            except Exception as exc:
                print(f"  {label}: error {exc}")

        # In realistic mode with the car stationary and parking brake on,
        # cycle gear input and record resulting gear string.
        lua(conn, 'controller.mainController.setGearboxMode("realistic")')
        conn.control(throttle=0.0, brake=0.0, steering=0.0, parkingbrake=1.0)
        conn.step(10)
        for g in [-1, 0, 1, 2, 3, 6]:
            conn.control(throttle=0.0, brake=0.0, steering=0.0,
                         parkingbrake=1.0, gear=g)
            conn.step(10)
            resp = lua(conn, "return jsonEncode({"
                             "gear=tostring(electrics.values.gear),"
                             "gear_input=electrics.values.gear_input,"
                             "gear_index=electrics.values.gear_index})")
            print(f"  input {g:>2} -> {resp}")

        # Verify repeated D input is a no-op while in D.
        conn.control(throttle=0.0, brake=0.0, steering=0.0,
                     parkingbrake=1.0, gear=2)
        conn.step(10)
        for _ in range(3):
            conn.control(throttle=0.0, brake=0.0, steering=0.0,
                         parkingbrake=1.0, gear=2)
            conn.step(10)
        resp = lua(conn, "return jsonEncode({"
                         "gear=tostring(electrics.values.gear),"
                         "mode=tostring(electrics.values.gearboxMode)})")
        print(f"  repeat gear=2 -> {resp}")

        # Manual-style gear list probe on the same vehicle: check gears list
        # ordering so forward_gear_input can be computed generally.
        conn.control(throttle=0.0, brake=0.0, steering=0.0,
                     parkingbrake=1.0, gear=0)
        conn.step(10)
    finally:
        try:
            lua(conn, 'controller.mainController.setGearboxMode("arcade")')
            conn.control(throttle=0.0, brake=0.0, steering=0.0, parkingbrake=1.0)
            conn.step(20)
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
