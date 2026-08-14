# Verify: switching realistic -> arcade while fully stopped with the
# parking brake engaged does NOT latch R or let the car move on its own.

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
    except Exception:
        signed = float("nan")
    info = lua(conn, "return jsonEncode({"
                      "mode=tostring(electrics.values.gearboxMode),"
                      "gear=tostring(electrics.values.gear),"
                      "pb=electrics.values.parkingbrake,"
                      "thr=electrics.values.throttle_input,"
                      "brk=electrics.values.brake_input})")
    print(f"  {label}: signed={signed:+.2f} {info}")


def main() -> None:
    conn = BeamNGConnector(config.DEFAULT_MAP, config.DEFAULT_VEHICLE,
                           port=config.PORT)
    try:
        conn.open(launch=False)
        conn.attach_vehicle(vid=None, already_open=True)
        print("[diag] attached")

        # Stop the car in realistic mode, D gear, parking brake on.
        lua(conn, 'controller.mainController.setGearboxMode("realistic")')
        conn.control(throttle=0.0, brake=1.0, steering=0.0, gear=2)
        for _ in range(150):
            conn.step(2)
            st = conn.get_state()
            signed = float(st.vel[0] * st.dir[0] + st.vel[1] * st.dir[1])
            if abs(signed) < 0.2:
                break
        conn.control(throttle=0.0, brake=0.0, steering=0.0,
                     parkingbrake=1.0, gear=2)
        conn.step(10)
        snap(conn, "realistic D stopped pb=1")

        # Switch to arcade while stopped + parking brake, watch 4 s.
        lua(conn, 'controller.mainController.setGearboxMode("arcade")')
        conn.step(10)
        snap(conn, "arcade just after switch (pb=1)")
        worst = 0.0
        for i in range(40):
            time.sleep(0.1)
            st = conn.get_state()
            signed = float(st.vel[0] * st.dir[0] + st.vel[1] * st.dir[1])
            worst = min(worst, signed)
            if i % 10 == 0:
                snap(conn, f"arcade t={i * 0.1:.1f}s")
        print(f"[diag] RESULT worst backward roll={worst:+.2f} m/s")
    finally:
        try:
            conn.control(throttle=0.0, brake=0.0, steering=0.0,
                         parkingbrake=1.0)
            conn.step(10)
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
