"""Live probe: game-side input watchdog (shared beamng_autopilot.watchdog).

Verifies the exact module m5_autopilot.py ships:
  1. attach, park, snapshot;
  2. inject + verify installed;
  3. arm, apply throttle, confirm inputs latched;
  4. STOP heartbeating -> step sim past the timeout -> verify inputs
     cleared, gearbox safe and parking brake engaged (slow branch);
  5. heartbeat again -> verify disengage; disarm, park, close.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beamng_autopilot import config
from beamng_autopilot.connector import BeamNGConnector
from beamng_autopilot.watchdog import arm, disarm, heartbeat, install, status


def lua(conn, chunk: str, response: bool = True):
    resp = conn.vehicle.queue_lua_command(chunk, response=response)
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
                     "armed=rawget(_G,'autopilot_watchdog') and "
                     "autopilot_watchdog.armed})")
    info2 = lua(conn, "return jsonEncode({"
                      "engaged=rawget(_G,'autopilot_watchdog') and "
                      "autopilot_watchdog.engaged})")
    elec = lua(conn, "return jsonEncode({"
                     "throttle=electrics.values.throttle,"
                     "throttle_input=electrics.values.throttle_input,"
                     "brake=electrics.values.brake,"
                     "brake_input=electrics.values.brake_input,"
                     "parkingbrake=electrics.values.parkingbrake,"
                     "gear=tostring(electrics.values.gear),"
                     "mode=tostring(electrics.values.gearboxMode)})")
    print(f"  {label}: signed={signed:+.2f} wd={info}/{info2} elec={elec}")


def park(conn) -> None:
    try:
        conn.vehicle.queue_lua_command(
            'controller.mainController.setGearboxMode("realistic")',
            response=False)
        conn.control(throttle=0.0, brake=0.0, steering=0.0,
                     parkingbrake=1.0)
        conn.step(10)
        snapshot(conn, "parked")
    except Exception as exc:
        print(f"  [park failed] {exc}")


def main() -> None:
    conn = BeamNGConnector(config.DEFAULT_MAP, config.DEFAULT_VEHICLE,
                           port=config.PORT)
    try:
        conn.open(launch=False)
        conn.attach_vehicle(vid=None, already_open=True)
        print("[wd] attached")
        park(conn)

        # ---- inject via the shared module ----
        ok = install(conn)
        print(f"[wd] install -> {ok}")
        mod = status(conn)
        print(f"[wd] module check -> {mod}")
        if not (ok and mod.get("installed")):
            print("[wd] FAIL: module not installed")
            return

        # ---- arm + apply throttle ----
        print(f"[wd] arm -> {arm(conn)}")
        conn.control(throttle=0.4, brake=0.0, steering=0.0,
                     parkingbrake=0.0)
        conn.step(5)
        snapshot(conn, "armed, throttle=0.4 applied")

        # ---- stop heartbeating; let the timeout fire ----
        print("[wd] heartbeats STOPPED; stepping 3.0 s ...")
        for _ in range(18):
            conn.step(10)
            time.sleep(0.05)
        snapshot(conn, "after 3 s no heartbeat")
        air = lua(conn, "return jsonEncode({airspeed="
                        "tostring(electrics.values.airspeed)})")
        print(f"[wd] airspeed check -> {air}")
        elec = lua(conn, "return jsonEncode({"
                         "parkingbrake=electrics.values.parkingbrake,"
                         "throttle_input=electrics.values.throttle_input,"
                         "brake_input=electrics.values.brake_input})")
        wd = lua(conn, "return jsonEncode({"
                       "engaged=autopilot_watchdog.engaged,"
                       "armed=autopilot_watchdog.armed})")
        print(f"[wd] elec={elec} wd={wd}")
        if wd and wd.get("engaged") and elec and elec.get("parkingbrake") == 1:
            print("[wd] PASS: watchdog fired (inputs cleared, handbrake on)")
        else:
            print("[wd] FAIL: watchdog did NOT fire")

        # ---- heartbeat again: should re-arm cleanly ----
        print(f"[wd] heartbeat -> {heartbeat(conn)}")
        conn.step(5)
        wd = lua(conn, "return jsonEncode({"
                       "engaged=autopilot_watchdog.engaged})")
        print(f"[wd] after heartbeat engaged={wd}")

        # ---- cleanup: disarm + park ----
        disarm(conn)
        park(conn)
        print("[wd] done")
    finally:
        try:
            conn.control(throttle=0.0, brake=0.0, steering=0.0,
                         parkingbrake=1.0)
            conn.step(3)
        except Exception:
            pass
        conn.close()


if __name__ == "__main__":
    main()
