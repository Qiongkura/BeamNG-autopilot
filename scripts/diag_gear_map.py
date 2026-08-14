# Probe: map numeric gear input -> actual gear string in realistic mode.

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


def snap(conn, label: str) -> None:
    info = lua(conn, "return jsonEncode({"
                      "mode=tostring(electrics.values.gearboxMode),"
                      "gear=tostring(electrics.values.gear),"
                      "gear_index=electrics.values.gear_index,"
                      "gear_input=electrics.values.gear_input,"
                      "rpm=electrics.values.rpm})")
    print(f"  {label}: {info}")


def main() -> None:
    conn = BeamNGConnector(config.DEFAULT_MAP, config.DEFAULT_VEHICLE,
                           port=config.PORT)
    try:
        conn.open(launch=False)
        conn.attach_vehicle(vid=None, already_open=True)
        print("[diag] attached")

        lua(conn, 'controller.mainController.setGearboxMode("realistic")')
        conn.control(throttle=0.0, brake=0.0, steering=0.0, parkingbrake=1.0)
        conn.step(10)
        snap(conn, "start (pb=1)")

        for g in range(-3, 8):
            conn.control(throttle=0.0, brake=0.0, steering=0.0,
                         parkingbrake=1.0, gear=g)
            for _ in range(4):
                conn.step(10)
                time.sleep(0.1)
            snap(conn, f"gear={g}")

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
