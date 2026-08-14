"""Emergency stop: clear residual control and park the car safely.

If a previous autopilot process was killed abruptly, its last control
inputs stay latched on the vehicle - BeamNG keeps applying them even
after the client disconnects, which is exactly what makes the car keep
driving / reversing on its own.  This script clears them and hands the
car back through the same safe sequence as the autopilot:

  1. force the gearbox to realistic first (in arcade, brake-at-standstill
     is a reverse request - the cause of the "stuck in R" bug);
  2. handover_vehicle() brakes to a full standstill in 1st at any speed,
     shifts to N with the parking brake on, restores the player's gearbox
     while fully stopped, and hands back with all pedals zeroed - so the
     car can never creep forward or latch R and reverse on its own.

Run it any time the car drives itself after an autopilot exit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beamng_autopilot import config
from beamng_autopilot.connector import BeamNGConnector
from beamng_autopilot.control.handover import handover_vehicle
from beamng_autopilot.runtime import resolve_runtime
from beamng_autopilot.watchdog import (
    arm as wd_arm,
    disarm as wd_disarm,
)


def signed_speed(st) -> float:
    return float(st.vel[0] * st.dir[0] + st.vel[1] * st.dir[1])


def main() -> None:
    ap = argparse.ArgumentParser(description="Emergency stop / handover")
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
        print(f"[stop] runtime={runtime_mode}")

        # Arm the input watchdog first: it heals a watchdog already engaged
        # by a killed autopilot, and if THIS script dies mid-cleanup the
        # watchdog takes over and stops the car again.
        try:
            wd_arm(conn)
        except Exception as exc:
            print(f"[stop] watchdog arm failed: {exc}")

        # Clear any residual latched inputs from the dead client immediately
        # (the watchdog is idle while heartbeats are fresh, so the stale
        # throttle/brake would otherwise keep driving the car).
        conn.control(throttle=0.0, brake=0.0, steering=0.0)
        conn.step(5)

        st = conn.get_state()
        print(f"[stop] before: speed={st.speed:.2f} signed="
              f"{signed_speed(st):.2f}")

        # Force realistic BEFORE any braking: in arcade, brake-at-standstill
        # latches R and drives the car backward on its own.
        conn.vehicle.queue_lua_command(
            'controller.mainController.setGearboxMode("realistic")')
        conn.step(5)

        # Hand back to the player (restores arcade only while rolling
        # forward, so the box latches D instead of R).
        handover_vehicle(conn, "arcade", True)

        st = conn.get_state()
        signed = signed_speed(st)
        resp = conn.vehicle.queue_lua_command(
            "return jsonEncode({throttle=electrics.values.throttle,"
            "brake=electrics.values.brake,"
            "parkingbrake=electrics.values.parkingbrake,"
            "gear=electrics.values.gear,"
            "mode=tostring(electrics.values.gearboxMode)})",
            response=True)
        print(f"[stop] after: speed={st.speed:.2f} signed={signed:.2f} "
              f"elec={resp}")
        if abs(signed) > 0.5:
            print("[stop] WARNING: car still moving - run again")
        else:
            print("[stop] done: pedals zeroed, parked, gearbox restored")
        # Car is parked and pedals zeroed: disarm the watchdog so the player
        # can drive manually without it pulling the handbrake.
        try:
            wd_disarm(conn)
        except Exception:
            pass
    finally:
        conn.close()


if __name__ == "__main__":
    main()
