"""Live reproduction of the 'stuck in reverse after autopilot exit' bug.

Attaches to the ego vehicle in the user's running session and exercises the
exact handover path m5_autopilot.py uses when autopilot is turned off while
the car is (nearly) stopped:

    1. if the car is rolling backward, brake it to a stop first;
    2. switch gearbox to realistic (like autopilot start does);
    3. engage a forward gear (D/1st) and roll forward;
    4. call handover_vehicle() with saved_gearbox="arcade";
    5. wait 4 seconds and report whether the car stayed stationary, or
       latched R and started reversing.

This is the same code path the user triggers with F9, so a FAIL here is the
bug they keep reporting.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beamng_autopilot import config
from beamng_autopilot.connector import BeamNGConnector
from beamng_autopilot.control.gearbox import forward_gear_input
from beamng_autopilot.control.handover import handover_vehicle
from beamng_autopilot.runtime import resolve_runtime


def lua_state(conn) -> dict:
    resp = conn.vehicle.queue_lua_command(
        "return jsonEncode({"
        "gear=tostring(electrics.values.gear),"
        "gear_input=electrics.values.gear_input,"
        "throttle=electrics.values.throttle,"
        "throttle_input=electrics.values.throttle_input,"
        "brake=electrics.values.brake,"
        "brake_input=electrics.values.brake_input,"
        "mode=tostring(electrics.values.gearboxMode),"
        "rpm=electrics.values.rpm,"
        "engineRunning=electrics.values.engineRunning"
        "})", response=True)
    if not resp:
        return {}
    if isinstance(resp, str):
        try:
            resp = json.loads(resp)
        except (ValueError, TypeError):
            return {}
    return resp if isinstance(resp, dict) else {}


def main() -> None:
    ap = argparse.ArgumentParser(description="Handover regression test")
    ap.add_argument("--runtime", choices=("auto", "steam", "tech"),
                    default=config.RUNTIME_MODE,
                    help="game runtime: auto detects after connecting")
    args = ap.parse_args()

    conn = BeamNGConnector(config.DEFAULT_MAP, config.DEFAULT_VEHICLE,
                           port=config.PORT,
                           home=config.runtime_home(args.runtime))
    try:
        conn.open(launch=False)
        conn.attach_vehicle(vid=None, already_open=True)
        runtime_mode = resolve_runtime(conn, args.runtime)
        print(f"[test] runtime={runtime_mode}")

        # Switch to realistic BEFORE touching the pedals: in arcade,
        # brake-at-standstill is a reverse request and would latch R during
        # the test's own setup (the exact bug this test checks for).
        conn.vehicle.queue_lua_command(
            'controller.mainController.setGearboxMode("realistic")')
        conn.step(5)

        st = conn.get_state()
        signed = float(st.vel[0] * st.dir[0] + st.vel[1] * st.dir[1])
        print(f"[test] start: speed={st.speed:.2f} signed={signed:.2f} "
              f"state={lua_state(conn)}")

        # Emergency: stop any backward roll first (the car may be reversing).
        if signed < 0.5:
            print("[test] car rolling backward or stopped -> braking to stop")
            for _ in range(150):
                conn.control(throttle=0.0, brake=1.0, steering=0.0)
                conn.step(2)
                st = conn.get_state()
                signed = float(st.vel[0] * st.dir[0] + st.vel[1] * st.dir[1])
                if signed > -0.2:
                    break
        conn.control(throttle=0.0, brake=0.0, steering=0.0)
        conn.step(3)
        st = conn.get_state()
        print(f"[test] stopped: speed={st.speed:.2f} state={lua_state(conn)}")

        # Simulate autopilot start: remember arcade, confirm realistic.
        mode = lua_state(conn).get("mode")
        print(f"[test] gearbox mode before: {mode}")
        conn.vehicle.queue_lua_command(
            'controller.mainController.setGearboxMode("realistic")')
        conn.step(5)
        print(f"[test] after switch to realistic: {lua_state(conn)}")

        # Engage a forward gear (D/1st) like autopilot start does, then
        # roll forward like autopilot driving away.
        fwd = forward_gear_input(conn)
        print(f"[test] forward gear input = {fwd}: {lua_state(conn)}")
        conn.control(throttle=0.45, brake=0.0, steering=0.0,
                     parkingbrake=0.0, gear=fwd)
        for _ in range(80):
            conn.step(2)
            st = conn.get_state()
            if st.speed > 1.6:
                break
        st = conn.get_state()
        print(f"[test] rolling forward: speed={st.speed:.2f} "
              f"state={lua_state(conn)}")
        conn.control(throttle=0.0, brake=0.0, steering=0.0, gear=fwd)
        conn.step(3)

        # The exact handover used by m5_autopilot (restore arcade).
        print("[test] calling handover_vehicle(conn, 'arcade', True) ...")
        handover_vehicle(conn, "arcade", True)
        print(f"[test] after handover: state={lua_state(conn)}")

        # Watch for 4 seconds: does R latch and the car start reversing, or
        # does arcade D creep and the car roll forward on its own?
        worst = 0.0
        best = 0.0
        for i in range(40):
            time.sleep(0.1)
            st = conn.get_state()
            signed = float(st.vel[0] * st.dir[0] + st.vel[1] * st.dir[1])
            worst = min(worst, signed)
            best = max(best, signed)
            if i % 10 == 0:
                print(f"[test] t={i * 0.1:.1f}s signed={signed:.2f} "
                      f"state={lua_state(conn)}")
        print(f"[test] RESULT: roll range [{worst:.2f}, {best:.2f}] m/s")
        if worst < -0.5:
            print("[test] FAIL: car is reversing after handover")
        elif best > 0.5:
            print("[test] FAIL: car is creeping forward after handover")
        else:
            print("[test] PASS: car stayed put after handover")
    finally:
        try:
            conn.control(throttle=0.0, brake=0.0, steering=0.0)
            conn.step(3)
        except Exception:
            pass
        conn.close()


if __name__ == "__main__":
    main()
