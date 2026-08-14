"""Time each communication primitive used by the m5 control loop.

Read-only apart from zero-input control/step samples; the car is expected
to be parked with the parking brake engaged before running this probe.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beamng_autopilot import config
from beamng_autopilot.connector import BeamNGConnector
from beamng_autopilot.control.gearbox import read_gear, read_gearbox_mode
from beamng_autopilot.runtime import build_range_provider, resolve_runtime
from beamng_autopilot.watchdog import heartbeat as wd_heartbeat


def timed(label, fn, repeat=3):
    vals = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        try:
            out = fn()
        except Exception as exc:
            print(f"  {label}: ERROR {exc!r}")
            return None
        vals.append(time.perf_counter() - t0)
    best = min(vals)
    avg = sum(vals) / len(vals)
    print(f"  {label}: min={best*1000:.1f} ms avg={avg*1000:.1f} ms")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="M5 communication timing probe")
    ap.add_argument("--runtime", choices=("auto", "steam", "tech"),
                    default=config.RUNTIME_MODE,
                    help="game runtime: auto detects after connecting")
    args = ap.parse_args()

    conn = BeamNGConnector(config.DEFAULT_MAP, config.DEFAULT_VEHICLE,
                           port=config.PORT,
                           home=config.runtime_home(args.runtime))
    range_provider = None
    try:
        conn.open(launch=False)
        conn.attach_vehicle(vid=None, already_open=True)
        runtime_mode = resolve_runtime(conn, args.runtime)
        range_provider, _ = build_range_provider(conn, runtime_mode)
        print(f"[probe] runtime={runtime_mode}")
        print(f"[probe] attached vid={conn.vehicle.vid} "
              f"sensors={list(conn.vehicle.sensors.data.keys())}")
        st = conn.get_state()
        print(f"[probe] pos=({st.pos[0]:.1f},{st.pos[1]:.1f},{st.pos[2]:.1f}) "
              f"speed={st.speed:.2f}")
        timed("get_state", conn.get_state, repeat=5)
        timed("get_wheel_speed", conn.get_wheel_speed, repeat=3)
        timed("read_gear", lambda: read_gear(conn), repeat=3)
        timed("read_gearbox_mode",
              lambda: read_gearbox_mode(conn.vehicle), repeat=3)
        timed("wd_heartbeat", lambda: wd_heartbeat(conn), repeat=3)
        timed("range_scan",
              lambda: range_provider.scan(
                  st.pos, conn.vehicle.vid, radius=55.0).obstacles,
              repeat=3)
        timed("lua return 1",
              lambda: conn.bng.queue_lua_command("return 1", response=True),
              repeat=3)
        timed("control zero",
              lambda: conn.control(throttle=0.0, steering=0.0, brake=0.0),
              repeat=3)
        timed("step(1, wait=True)",
              lambda: conn.step(1, wait=True), repeat=3)
        timed("step(1, wait=False)",
              lambda: conn.step(1, wait=False), repeat=3)
        timed("read_navigation_route", conn.read_navigation_route, repeat=2)
    finally:
        if range_provider is not None:
            try:
                range_provider.close()
            except Exception:
                pass
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
