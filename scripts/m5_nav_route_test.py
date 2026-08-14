"""E2E check: grab the in-game navigation route from Python.

Loads a fresh scenario, builds a navigation route to a distant road node via
the same Lua call the big map uses (core_groundMarkers.setPath), then reads
it back with BeamNGConnector.read_navigation_route().  The game must already
be running with the tcom port open (scripts/launch_game.py).

Usage:
    .venv\\Scripts\\python.exe scripts\\launch_game.py
    .venv\\Scripts\\python.exe scripts\\m5_nav_route_test.py
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
from beamng_autopilot.roadnet import RoadNetwork


def main() -> int:
    ap = argparse.ArgumentParser(description="M5 navigation-route probe")
    ap.add_argument("--map", default=config.DEFAULT_MAP)
    ap.add_argument("--vehicle", default=config.DEFAULT_VEHICLE)
    ap.add_argument("--runtime", choices=("auto", "steam", "tech"),
                    default=config.RUNTIME_MODE,
                    help="game runtime: auto detects after connecting")
    args = ap.parse_args()

    conn = BeamNGConnector(
        args.map, args.vehicle, home=config.runtime_home(args.runtime))
    try:
        conn.open(launch=False)
        conn.load_scenario()
        conn.attach_vehicle(vid="ego", already_open=True)
        print("[test] scenario loaded")

        roadnet = RoadNetwork()
        t0 = time.time()
        while not roadnet.ready and time.time() - t0 < 60.0:
            if roadnet.build(conn.bng):
                break
            time.sleep(1.0)

        st = conn.get_state()
        if roadnet.ready:
            d = np.linalg.norm(roadnet.nodes - st.pos[:2], axis=1)
            dest = roadnet.nodes[int(np.argmax(d))]
            print(f"[test] roadnet ready: {roadnet.info}")
        else:
            dest = np.asarray([150.0, 0.0])
            print("[test] roadnet not ready; using fixed destination")
        print(f"[test] car pos      = {st.pos[:2].tolist()}")
        print(f"[test] destination  = {dest.tolist()}")

        # Same call the big map makes when you pick a destination.  The
        # destination must be a single vec3, NOT a plain {x, y, z} table:
        # setPath treats a plain table as a list of waypoints and the path
        # builder would explode on the raw numbers.
        lua = (
            "core_groundMarkers.setPath({vec3(%.3f, %.3f, 0)})\n"
            "return 'ok'"
        ) % (float(dest[0]), float(dest[1]))
        resp = conn.bng.control.queue_lua_command(lua, response=True)
        print(f"[test] setPath resp = {resp!r}")

        time.sleep(0.3)
        nav = conn.read_navigation_route()
        if nav is None:
            print("[test] FAIL: read_navigation_route() returned None")
            return 1
        print(f"[test] OK: grabbed {len(nav)} route pts")
        print(f"[test] first pt = {nav[0].tolist()}")
        print(f"[test] last  pt = {nav[-1].tolist()}")
        length = float(np.sum(np.linalg.norm(np.diff(nav[:, :2], axis=0),
                                             axis=1)))
        print(f"[test] route length = {length:.1f} m")
        return 0
    finally:
        try:
            # Leave the game without an active nav route: a leftover route
            # keeps being tracked by the game and can degrade into garbage
            # points (see connector.read_navigation_route validation).
            conn.bng.control.queue_lua_command(
                "core_groundMarkers.setPath(nil)\nreturn 'ok'",
                response=True)
        except Exception:
            pass
        try:
            conn.close()  # disconnect only; the game stays open
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
