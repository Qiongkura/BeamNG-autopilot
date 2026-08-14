"""Live gearbox diagnostic for the route probe's "car does not move" case.

Teleports to the route start, enumerates realistic-mode gear inputs 1..6,
prints the reported gear string for each, then engages the first forward
gear it finds and applies throttle for a couple of seconds to see whether
the vehicle actually moves.
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
    read_gear,
    read_gearbox_mode,
    set_gearbox_mode,
)


def _ground_z(conn, x: float, y: float) -> float | None:
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
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=config.DEFAULT_MAP)
    ap.add_argument("--vehicle", default=config.DEFAULT_VEHICLE)
    ap.add_argument("--start-x", type=float, default=726.6)
    ap.add_argument("--start-y", type=float, default=755.9)
    ap.add_argument("--goal-x", type=float, default=555.8)
    ap.add_argument("--goal-y", type=float, default=394.2)
    args = ap.parse_args()

    conn = BeamNGConnector(args.map, args.vehicle)
    try:
        conn.open(launch=False)
        conn.attach_vehicle(already_open=True)
        heading = math.atan2(args.goal_y - args.start_y,
                             args.goal_x - args.start_x)
        yaw_deg = -math.degrees(float(heading)) - 90.0
        ground_z = _ground_z(conn, args.start_x, args.start_y)
        z = (ground_z + 0.6) if ground_z is not None else 178.0
        conn.vehicle.teleport(
            (float(args.start_x), float(args.start_y), z),
            rot_quat=angle_to_quat((0.0, 0.0, yaw_deg)))
        conn.control(throttle=0.0, brake=1.0, steering=0.0,
                     parkingbrake=0.0)
        conn.step(60)
        st = conn.get_state()
        print(f"[gearbox] at ({st.pos[0]:.1f}, {st.pos[1]:.1f}, "
              f"{st.pos[2]:.1f}) speed={st.speed:.2f} "
              f"mode={read_gearbox_mode(conn.vehicle)}")

        set_gearbox_mode(conn.vehicle, "realistic")
        conn.step(10)
        print(f"[gearbox] mode after set = {read_gearbox_mode(conn.vehicle)}")
        for g in (1, 2, 3, 4, 5, 6):
            conn.control(throttle=0.0, brake=0.0, steering=0.0,
                         parkingbrake=1.0, gear=g)
            conn.step(10)
            gear = read_gear(conn)
            print(f"[gearbox] input={g} -> gear={gear!r}")

        # Pick the first input whose reported gear is D or a forward number.
        chosen = None
        for g in (2, 1, 3, 4, 5, 6):
            conn.control(throttle=0.0, brake=0.0, steering=0.0,
                         parkingbrake=1.0, gear=g)
            conn.step(10)
            gear = read_gear(conn) or ""
            if gear == "D" or (gear and gear[0] in "12345678"):
                chosen = (g, gear)
                break
        if chosen is None:
            print("[gearbox] no forward gear found")
            return
        g, gear = chosen
        print(f"[gearbox] chosen input={g} gear={gear!r}; throttle test")
        conn.control(throttle=0.35, brake=0.0, steering=0.0,
                     parkingbrake=0.0, gear=g)
        for k in range(6):
            conn.step(40)
            st = conn.get_state()
            resp = conn.vehicle.queue_lua_command(
                "return jsonEncode({rpm=electrics.values.enginerpm, "
                "throttle=electrics.values.throttle, "
                "brake=electrics.values.brake, "
                "parking=electrics.values.parkingbrake, "
                "clutch=electrics.values.clutch, "
                "gear=electrics.values.gear, "
                "wheelspeed=electrics.values.wheelspeed, "
                "speed=electrics.values.airspeed})",
                response=True)
            print(f"[gearbox] t+{k * 0.66:.1f}s speed={st.speed:.2f} "
                  f"pos=({st.pos[0]:.1f}, {st.pos[1]:.1f}) gear="
                  f"{read_gear(conn)} electrics={resp}")
            if st.speed > 3.0:
                break
        conn.control(throttle=0.0, brake=1.0, steering=0.0,
                     parkingbrake=1.0, gear=g)
        conn.step(20)
        fwd = forward_gear_input(conn, force=True)
        print(f"[gearbox] forward_gear_input(force=True) -> {fwd}")
        time.sleep(0.2)
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
