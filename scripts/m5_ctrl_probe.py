"""Minimal control-path probe: send one control, read electrics back."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beamng_autopilot import config
from beamng_autopilot.connector import BeamNGConnector
from beamng_autopilot.runtime import resolve_runtime


def electrics(conn) -> dict:
    resp = conn.vehicle.queue_lua_command(
        "return jsonEncode({throttle=electrics.values.throttle,"
        "brake=electrics.values.brake,"
        "parkingbrake=electrics.values.parkingbrake,"
        "gear=electrics.values.gear,"
        "mode=tostring(electrics.values.gearboxMode),"
        "throttle_input=electrics.values.throttle_input,"
        "brake_input=electrics.values.brake_input})",
        response=True)
    return resp or {}


def main() -> None:
    ap = argparse.ArgumentParser(description="Control-path probe")
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
        print(f"[probe] runtime={runtime_mode}")
        st = conn.get_state()
        signed = float(st.vel[0] * st.dir[0] + st.vel[1] * st.dir[1])
        print(f"[probe] before: speed={st.speed:.2f} signed={signed:.2f} "
              f"elec={electrics(conn)}")

        conn.control(throttle=0.0, brake=1.0, steering=0.0)
        conn.step(3)
        st = conn.get_state()
        signed = float(st.vel[0] * st.dir[0] + st.vel[1] * st.dir[1])
        print(f"[probe] after control(brake=1): speed={st.speed:.2f} "
              f"signed={signed:.2f} elec={electrics(conn)}")

        resp = conn.vehicle.queue_lua_command(
            "local ok, err = pcall(function() "
            "v.controller:setInputs({throttle=0, brake=1, steering=0}) end)"
            "return jsonEncode({ok=ok, err=tostring(err)})", response=True)
        print(f"[probe] setInputs via lua: {resp}")
        conn.step(3)
        st = conn.get_state()
        signed = float(st.vel[0] * st.dir[0] + st.vel[1] * st.dir[1])
        print(f"[probe] after lua setInputs: speed={st.speed:.2f} "
              f"signed={signed:.2f} elec={electrics(conn)}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
