# Live diagnostic: does BeamNG keep applying client inputs after the
# client disconnects?  And does arcade brake-at-standstill latch R?
# Attaches to the running game, mimics an autopilot hold (realistic,
# brake held), then hard-disconnects WITHOUT zeroing inputs, waits, then
# reconnects and reports mode/gear/speed.  Restores a parked car at the
# end.

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
                      "throttle_input=electrics.values.throttle_input,"
                      "brake_input=electrics.values.brake_input})")
    print(f"  {label}: signed={signed:+.2f} {info}")


def park(conn) -> None:
    try:
        conn.vehicle.queue_lua_command(
            'controller.mainController.setGearboxMode("arcade")')
        conn.control(throttle=0.0, brake=0.0, steering=0.0, parkingbrake=1.0)
        conn.step(15)
        snapshot(conn, "parked")
    except Exception as exc:
        print(f"  [park failed] {exc}")


def main() -> None:
    conn = BeamNGConnector(config.DEFAULT_MAP, config.DEFAULT_VEHICLE,
                           port=config.PORT)
    try:
        conn.open(launch=False)
        conn.attach_vehicle(vid=None, already_open=True)
        print("[diag] attached")

        # ---- Test A: arcade brake-at-standstill latches R? ----
        print("[A] arcade brake-at-standstill -> R latch test")
        lua(conn, 'controller.mainController.setGearboxMode("arcade")')
        conn.step(5)
        conn.control(throttle=0.0, brake=0.12, steering=0.0, parkingbrake=0.0)
        conn.step(10)
        snapshot(conn, "arcade brake=0.12")
        for _ in range(5):
            conn.step(10)
            time.sleep(0.1)
        snapshot(conn, "arcade brake=0.12 held 0.5s")
        # release
        conn.control(throttle=0.0, brake=0.0, steering=0.0)
        conn.step(10)
        snapshot(conn, "arcade brake released")
        park(conn)

        # ---- Test B: hard disconnect while holding brake (no zero) ----
        print("[B] hard disconnect while realistic + brake held")
        lua(conn, 'controller.mainController.setGearboxMode("realistic")')
        conn.step(5)
        conn.control(throttle=0.0, brake=0.5, steering=0.0, parkingbrake=0.0)
        conn.step(10)
        snapshot(conn, "holding brake=0.5 (realistic)")
        # disconnect WITHOUT zeroing inputs (simulates task-manager kill)
        try:
            conn.bng.disconnect()
            print("[B] disconnected (inputs NOT zeroed)")
        except Exception as exc:
            print(f"[B] disconnect error: {exc}")
        time.sleep(3.0)

        # reconnect and read the vehicle state
        conn2 = BeamNGConnector(config.DEFAULT_MAP, config.DEFAULT_VEHICLE,
                                port=config.PORT)
        try:
            conn2.open(launch=False)
            conn2.attach_vehicle(vid=None, already_open=True)
            print("[B] reconnected after 3 s")
            snapshot(conn2, "after hard disconnect +3s")
            park(conn2)
        except Exception as exc:
            print(f"[B] reconnect failed: {exc}")
        finally:
            try:
                conn2.close()
            except Exception:
                pass
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
