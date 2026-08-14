"""Live probe: what does the perception stack actually see right now?

Attaches to the ego vehicle in the running session and reports, for each of
the three obstacle sources, how many obstacles were returned, their distance
from the car, and whether any sensor error is flagged.  This answers the
"is it really sensing the environment or just following the nav route"
question with live data instead of guesswork.  On BeamNG.tech the sources
are reported as one merged lidar + scenario + vehicle scan.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beamng_autopilot import config
from beamng_autopilot.connector import BeamNGConnector
from beamng_autopilot.perception import errors_summary
from beamng_autopilot.runtime import build_range_provider, resolve_runtime


def main() -> None:
    ap = argparse.ArgumentParser(description="M5 live perception probe")
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
        print(f"[probe] runtime={runtime_mode}")
        st = conn.get_state()
        pos = st.pos
        heading = float(st.heading)
        print(f"[probe] pos=({pos[0]:.1f}, {pos[1]:.1f}) "
              f"heading={heading:.2f} speed={st.speed:.2f}")
        hx, hy = np.cos(heading), np.sin(heading)

        def dump(label: str, obs) -> None:
            print(f"[probe] {label}: {len(obs)} obstacles")
            for ob in sorted(
                obs, key=lambda o: math_dist(o, pos))[:12]:
                dx = ob.x - pos[0]
                dy = ob.y - pos[1]
                lon = dx * hx + dy * hy
                lat = dx * (-hy) + dy * hx
                print(f"    d={math_dist(ob, pos):6.1f}m "
                      f"lon={lon:+6.1f} lat={lat:+6.1f} "
                      f"box=({ob.half_w:.1f}x{ob.half_h:.1f}) "
                      f"cat={ob.category}")

        def math_dist(ob, p) -> float:
            return float(np.hypot(ob.x - p[0], ob.y - p[1]))

        if runtime_mode == "tech":
            range_provider, _ = build_range_provider(conn, runtime_mode)
            sample = range_provider.scan(
                pos, conn.vehicle.vid, radius=55.0)
            dump("merged lidar+scenario+vehicles", sample.obstacles)
            print(f"[probe] ray/lidar hits: {len(sample.ray_hits)}")
        else:
            from beamng_autopilot.perception import (
                scan_obstacles,
                scan_obstacles_raycast,
                scan_obstacles_vehicles,
            )

            # Scenario objects
            try:
                obs = scan_obstacles(
                    conn.bng, conn.vehicle.vid, pos, radius=55.0)
                dump("scenario", obs)
            except Exception as exc:
                print(f"[probe] scenario FAILED: {exc}")

            # Live vehicles
            try:
                obs = scan_obstacles_vehicles(
                    conn.bng, conn.vehicle.vid, pos, radius=60.0)
                dump("vehicles", obs)
            except Exception as exc:
                print(f"[probe] vehicles FAILED: {exc}")

            # Raycast lidar (2 sweeps to show stability)
            for sweep in (1, 2):
                try:
                    obs = scan_obstacles_raycast(conn.bng, pos, radius=55.0,
                                                 rays=90)
                    dump(f"raycast sweep{sweep}", obs)
                except Exception as exc:
                    print(f"[probe] raycast FAILED: {exc}")
                time.sleep(0.3)

        print(f"[probe] perception errors = {errors_summary()!r}")

        # Raw single ray straight ahead, 60 m - does Engine.castRay work at all?
        # Engine.castRay(origin, target, includeTerrain, renderGeometry): the
        # second argument is an absolute world target point, NOT a direction.
        try:
            tx = pos[0] + hx * 60.0
            ty = pos[1] + hy * 60.0
            resp = conn.bng.queue_lua_command(
                "local r = Engine.castRay(vec3(%f, %f, %f), "
                "vec3(%f, %f, %f), true, false) "
                "if r then return jsonEncode({hit=true, x=r.pt.x, y=r.pt.y}) "
                "else return jsonEncode({hit=false}) end"
                % (pos[0], pos[1], pos[2] + 1.15,
                   tx, ty, pos[2] + 1.15), response=True)
            print(f"[probe] single straight-ahead ray: {resp}")
        except Exception as exc:
            print(f"[probe] single ray FAILED: {exc}")
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
