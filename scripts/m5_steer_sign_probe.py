"""Live probe: which way does a positive BeamNG steering input turn?

Teleports to the Italy crossroads, engages a real forward gear, then
applies a short throttle burst with steering = +0.4 and -0.4.  The
heading change and signed lateral drift identify the steering sign.
The vehicle is handed back parked in N in the player's gearbox mode.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
from beamngpy.misc.quat import angle_to_quat

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beamng_autopilot import config
from beamng_autopilot.connector import BeamNGConnector
from beamng_autopilot.control.gearbox import (
    forward_gear_input,
    read_gearbox_mode,
    set_gearbox_mode,
)
from beamng_autopilot.control.handover import handover_vehicle
from beamng_autopilot.runtime import resolve_runtime


START_XY = (726.6, 755.9)
GOAL_XY = (572.0, 533.5)


def _ground_z(conn, x: float, y: float) -> float:
    chunk = (
        f"local res = Engine.castRay(vec3({x:.3f}, {y:.3f}, 10000), "
        f"vec3({x:.3f}, {y:.3f}, -1000), true, false)\n"
        "if res and res.pt then "
        "return string.format('%.3f,%.3f,%.3f', "
        "res.pt.x, res.pt.y, res.pt.z) end\n"
        "return 'nil'"
    )
    resp = conn.bng.control.queue_lua_command(chunk, response=True)
    if resp and str(resp).strip() != "nil":
        parts = str(resp).split(",")
        if len(parts) == 3:
            return float(parts[2])
    return 0.0


def _teleport(conn, start_xy, goal_xy) -> None:
    heading = math.atan2(goal_xy[1] - start_xy[1],
                         goal_xy[0] - start_xy[0])
    yaw_deg = -math.degrees(float(heading)) - 90.0
    st0 = conn.get_state()
    z = float(st0.pos[2]) if len(st0.pos) > 2 else 0.0
    z = _ground_z(conn, float(start_xy[0]), float(start_xy[1])) + 0.6
    conn.vehicle.teleport(
        (float(start_xy[0]), float(start_xy[1]), z),
        rot_quat=angle_to_quat((0.0, 0.0, yaw_deg)))
    conn.control(throttle=0.0, brake=0.0, steering=0.0,
                 parkingbrake=0.0)
    conn.step(20)


def _snap(conn, label: str) -> None:
    st = conn.get_state()
    heading = float(st.heading)
    pos = np.asarray(st.pos[:2], dtype=float)
    signed = float(np.dot(np.asarray(st.vel[:2], dtype=float),
                          np.asarray(st.dir[:2], dtype=float)))
    print(f"[probe] {label}: pos=({pos[0]:.2f},{pos[1]:.2f}) "
          f"hdg={math.degrees(heading):+7.1f} signed={signed:+.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--burst-s", type=float, default=0.8)
    ap.add_argument("--throttle", type=float, default=0.32)
    ap.add_argument("--steer", type=float, default=0.4)
    ap.add_argument("--runtime", choices=("auto", "steam", "tech"),
                    default=config.RUNTIME_MODE,
                    help="game runtime: auto detects after connecting")
    args = ap.parse_args()

    conn = BeamNGConnector(config.DEFAULT_MAP, config.DEFAULT_VEHICLE,
                           port=config.PORT,
                           home=config.runtime_home(args.runtime))
    saved = None
    switched = False
    try:
        conn.open(launch=False)
        conn.attach_vehicle(vid=None, already_open=True)
        runtime_mode = resolve_runtime(conn, args.runtime)
        print(f"[probe] runtime={runtime_mode}")
        saved = read_gearbox_mode(conn.vehicle) or "arcade"
        _teleport(conn, START_XY, GOAL_XY)
        set_gearbox_mode(conn.vehicle, "realistic")
        conn.step(5)
        fwd = forward_gear_input(conn, force=True)
        switched = True
        print(f"[probe] forward gear input = {fwd}")

        for sign in (1.0, -1.0):
            _teleport(conn, START_XY, GOAL_XY)
            _snap(conn, f"before steer={sign:+.1f}")
            t0 = time.time()
            steps = max(1, int(args.burst_s * 60))
            for _ in range(steps):
                conn.control(throttle=args.throttle,
                             steering=sign * args.steer, brake=0.0)
                conn.step(1)
            conn.control(throttle=0.0, steering=0.0, brake=1.0)
            for _ in range(120):
                conn.step(2)
                st = conn.get_state()
                if abs(float(np.dot(np.asarray(st.vel[:2], dtype=float),
                                    np.asarray(st.dir[:2], dtype=float)))) \
                        < 0.15:
                    break
            _snap(conn, f"after  steer={sign:+.1f}")
            print(f"[probe] burst wall-time={time.time() - t0:.2f}s")
    finally:
        try:
            handover_vehicle(conn, saved, switched)
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
