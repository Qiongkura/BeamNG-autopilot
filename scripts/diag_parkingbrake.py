# Live diagnostic: does the Control message control the parking brake?
# Attaches to the running game and compares parking-brake release/engage via:
#   1. conn.control(parkingbrake=0.0 / 1.0)  (beamngpy Control message)
#   2. Lua input.event("parkingbrake", 0 / 1)
#   3. writing electrics.values.parkingbrake directly
# Prints electrics.values.parkingbrake, gearboxMode, gear and speed after
# each attempt so we can see which channel actually changes the brake.

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


def snapshot(conn, label: str) -> None:
    try:
        st = conn.get_state()
        signed = float(st.vel[0] * st.dir[0] + st.vel[1] * st.dir[1])
    except Exception as exc:
        signed = float("nan")
        print(f"  [state error] {exc}")
    info = lua(conn, "return jsonEncode({"
                      "pb=electrics.values.parkingbrake,"
                      "mode=tostring(electrics.values.gearboxMode),"
                      "gear=electrics.values.gear,"
                      "gear_input=electrics.values.gear_input,"
                      "eng=electrics.values.engineRunning})")
    print(f"  {label}: signed={signed:+.2f} {info}")


def main() -> None:
    conn = BeamNGConnector(config.DEFAULT_MAP, config.DEFAULT_VEHICLE,
                           port=config.PORT)
    try:
        conn.open(launch=False)
        conn.attach_vehicle(vid=None, already_open=True)
        print("[diag] attached")

        # 1. baseline
        snapshot(conn, "baseline")
        time.sleep(0.5)
        snapshot(conn, "baseline+0.5s")

        # 2. Control message: release
        conn.control(throttle=0.0, brake=0.0, steering=0.0, parkingbrake=0.0)
        conn.step(10)
        snapshot(conn, "after control(pb=0)")
        time.sleep(0.5)
        snapshot(conn, "after control(pb=0)+0.5s")

        # 3. Control message: engage
        conn.control(throttle=0.0, brake=0.0, steering=0.0, parkingbrake=1.0)
        conn.step(10)
        snapshot(conn, "after control(pb=1)")
        time.sleep(0.5)
        snapshot(conn, "after control(pb=1)+0.5s")

        # 4. Lua input.event: release
        lua(conn, 'input.event("parkingbrake", 0)')
        conn.step(10)
        snapshot(conn, "after lua input.event(pb=0)")
        time.sleep(0.5)
        snapshot(conn, "after lua input.event(pb=0)+0.5s")

        # 5. Lua input.event: engage
        lua(conn, 'input.event("parkingbrake", 1)')
        conn.step(10)
        snapshot(conn, "after lua input.event(pb=1)")
        time.sleep(0.5)
        snapshot(conn, "after lua input.event(pb=1)+0.5s")

        # 6. Direct electrics write: release
        lua(conn, "electrics.values.parkingbrake = 0")
        conn.step(10)
        snapshot(conn, "after electrics.pb=0")
        time.sleep(0.5)
        snapshot(conn, "after electrics.pb=0+0.5s")

        # 7. Direct electrics write: engage
        lua(conn, "electrics.values.parkingbrake = 1")
        conn.step(10)
        snapshot(conn, "after electrics.pb=1")
        time.sleep(0.5)
        snapshot(conn, "after electrics.pb=1+0.5s")

        # 8. test gearbox mode switches still work
        for mode in ("realistic", "arcade"):
            lua(conn, f'controller.mainController.setGearboxMode("{mode}")')
            conn.step(10)
            snapshot(conn, f"after setGearboxMode({mode})")

        # Leave car parked and safe.
        conn.control(throttle=0.0, brake=0.0, steering=0.0, parkingbrake=1.0)
        conn.step(10)
        snapshot(conn, "final (pb=1)")
    finally:
        try:
            conn.control(throttle=0.0, brake=0.0, steering=0.0,
                         parkingbrake=1.0)
            conn.step(5)
        except Exception:
            pass
        conn.close()


if __name__ == "__main__":
    main()
