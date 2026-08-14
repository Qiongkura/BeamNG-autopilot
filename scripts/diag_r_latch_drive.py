# Live probe: after arcade latches R at standstill, does switching to
# realistic and sending throttle (WITHOUT gear=1) drive the car backward?
# Also verify that sending gear=1 shifts out of R.

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
    try:
        st = conn.get_state()
        signed = float(st.vel[0] * st.dir[0] + st.vel[1] * st.dir[1])
    except Exception as exc:
        signed = float("nan")
        print(f"  [state error] {exc}")
    info = lua(conn, "return jsonEncode({"
                      "mode=tostring(electrics.values.gearboxMode),"
                      "gear=electrics.values.gear,"
                      "gear_input=electrics.values.gear_input,"
                      "thr=electrics.values.throttle_input,"
                      "brk=electrics.values.brake_input,"
                      "rpm=electrics.values.rpm})")
    print(f"  {label}: signed={signed:+.2f} {info}")


def park(conn) -> None:
    try:
        lua(conn, 'controller.mainController.setGearboxMode("arcade")')
        conn.control(throttle=0.0, brake=0.0, steering=0.0, parkingbrake=1.0)
        conn.step(20)
    except Exception as exc:
        print(f"  [park failed] {exc}")


def main() -> None:
    conn = BeamNGConnector(config.DEFAULT_MAP, config.DEFAULT_VEHICLE,
                           port=config.PORT)
    try:
        conn.open(launch=False)
        conn.attach_vehicle(vid=None, already_open=True)
        print("[diag] attached")

        # 1. Latch R in arcade with a gentle brake at standstill.
        lua(conn, 'controller.mainController.setGearboxMode("arcade")')
        conn.control(throttle=0.0, brake=0.0, steering=0.0, parkingbrake=0.0)
        conn.step(10)
        conn.control(throttle=0.0, brake=0.12, steering=0.0)
        conn.step(10)
        snap(conn, "arcade brake=0.12 (expect R latch)")
        conn.control(throttle=0.0, brake=0.0, steering=0.0)
        conn.step(10)
        snap(conn, "brake released (still R?)")

        # 2. Switch to realistic and give throttle WITHOUT gear input.
        lua(conn, 'controller.mainController.setGearboxMode("realistic")')
        conn.step(10)
        snap(conn, "realistic (gear still R?)")
        conn.control(throttle=0.4, brake=0.0, steering=0.0)
        for i in range(6):
            conn.step(10)
            time.sleep(0.1)
        snap(conn, "realistic throttle=0.4 no gear (backward?)")
        conn.control(throttle=0.0, brake=0.0, steering=0.0)
        conn.step(10)

        # 3. Now send gear=1 with throttle: does it shift out of R?
        conn.control(throttle=0.4, brake=0.0, steering=0.0, gear=1)
        for i in range(6):
            conn.step(10)
            time.sleep(0.1)
        snap(conn, "realistic throttle=0.4 gear=1 (forward?)")

        # 4. Shift back to 1 explicitly and drive a bit forward.
        conn.control(throttle=0.4, brake=0.0, steering=0.0, gear=1)
        for i in range(10):
            conn.step(10)
            time.sleep(0.1)
        snap(conn, "continue gear=1 (forward speed?)")

        conn.control(throttle=0.0, brake=0.0, steering=0.0, gear=1)
        conn.step(10)
        park(conn)
        snap(conn, "parked")
    finally:
        try:
            park(conn)
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
