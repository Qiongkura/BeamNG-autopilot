"""Read-only probe: electrics inputs, AI state and current route info."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beamng_autopilot import config
from beamng_autopilot.connector import BeamNGConnector
from beamng_autopilot.runtime import resolve_runtime


def main() -> None:
    ap = argparse.ArgumentParser(description="Electrics / AI state probe")
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
        resp = conn.vehicle.queue_lua_command(
            "return jsonEncode({"
            "throttle=electrics.values.throttle,"
            "throttle_input=electrics.values.throttle_input,"
            "brake=electrics.values.brake,"
            "brake_input=electrics.values.brake_input,"
            "parkingbrake=electrics.values.parkingbrake,"
            "gear=electrics.values.gear,"
            "gear_input=electrics.values.gear_input,"
            "mode=tostring(electrics.values.gearboxMode),"
            "rpm=electrics.values.rpm,"
            "engineRunning=electrics.values.engineRunning,"
            "aiEnabled=v.controller and v.controller.isAIControlled and "
            "v.controller:isAIControlled() or false,"
            "scriptName=v and v.script and v.script:getName() or 'nil'"
            "})", response=True)
        st = conn.get_state()
        print(f"[probe] speed={st.speed:.2f} pos=({st.pos[0]:.0f},"
              f"{st.pos[1]:.0f},{st.pos[2]:.2f})")
        print(f"[probe] electrics={resp}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
