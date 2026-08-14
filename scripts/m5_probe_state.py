"""Read-only probe: current vehicle gearbox / gear / motion state.

Attaches to the ego vehicle in a running BeamNG session and prints the
gearbox mode, current gear, speed and signed longitudinal velocity without
changing anything.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beamng_autopilot import config
from beamng_autopilot.connector import BeamNGConnector
from beamng_autopilot.runtime import resolve_runtime


def main() -> None:
    ap = argparse.ArgumentParser(description="Gearbox / motion state probe")
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
        resp = conn.vehicle.queue_lua_command(
            "return jsonEncode({mode=tostring(electrics.values.gearboxMode), "
            "gear=electrics.values.gear, "
            "gear_input=electrics.values.gear_input, "
            "shift=electrics.values.shift})", response=True)
        print(f"[probe] speed={st.speed:.2f} m/s "
              f"vel=({st.vel[0]:.2f},{st.vel[1]:.2f},{st.vel[2]:.2f})")
        print(f"[probe] dir=({st.dir[0]:.2f},{st.dir[1]:.2f}) "
              f"pos=({st.pos[0]:.1f},{st.pos[1]:.1f},{st.pos[2]:.1f})")
        signed = float(st.vel[0] * st.dir[0] + st.vel[1] * st.dir[1])
        print(f"[probe] signed longitudinal speed={signed:.2f} m/s")
        print(f"[probe] lua={resp}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
