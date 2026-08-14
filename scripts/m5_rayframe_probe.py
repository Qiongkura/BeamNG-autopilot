"""Determine the coordinate frame of Engine.castRay hit points.

The earlier probe found a straight-ahead ray "hitting" at (-0.736, -0.677),
which is suspiciously equal to the heading unit vector (cos(-2.40),
sin(-2.40)) * ~1.0 - i.e. the hit looks like it is reported in the ego
vehicle's LOCAL frame, not in world coordinates.  If true, every obstacle
the lidar "sees" is placed at the wrong place in world space (usually tens
of km away), which would make the planner see nothing and explain why the
car just charges along the nav route.

This probe casts rays at several headings and prints the raw hit points
next to the ego position + heading, so the frame can be confirmed
unambiguously.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beamng_autopilot import config
from beamng_autopilot.connector import BeamNGConnector


def main() -> None:
    conn = BeamNGConnector(config.DEFAULT_MAP, config.DEFAULT_VEHICLE,
                           port=config.PORT)
    try:
        conn.open(launch=False)
        conn.attach_vehicle(vid=None, already_open=True)
        st = conn.get_state()
        pos = st.pos
        heading = float(st.heading)
        print(f"[probe] ego pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) "
              f"heading={heading:.4f} rad")
        hx, hy = math.cos(heading), math.sin(heading)
        print(f"[probe] heading unit vec=({hx:.4f}, {hy:.4f})")

        # Rays: forward, left, right, backward, and a couple of diagonals.
        angles = [0.0, math.pi / 2, -math.pi / 2, math.pi,
                  math.pi / 4, -math.pi / 4, 0.6, -2.4]
        chunk_lines = []
        for i, a in enumerate(angles):
            c, s = math.cos(a), math.sin(a)
            chunk_lines.append(
                "local r%d = Engine.castRay(vec3(%.3f, %.3f, %.3f), "
                "vec3(%.4f, %.4f, 0), 80, true, false)" % (
                    i, pos[0], pos[1], pos[2] + 1.15, c, s))
        chunk_lines.append("return jsonEncode({")
        for i in range(len(angles)):
            chunk_lines.append(
                "r%d = r%d and {x=r%d.pt.x, y=r%d.pt.y, z=r%d.pt.z} or nil,"
                % (i, i, i, i, i))
        chunk_lines.append("})")
        chunk = "\n".join(chunk_lines)
        resp = conn.bng.queue_lua_command(chunk, response=True)
        print(f"[probe] raw response: {resp}")
        try:
            data = json.loads(str(resp))
        except (ValueError, TypeError):
            print("[probe] could not parse response")
            return
        for i, a in enumerate(angles):
            hit = data.get(f"r{i}")
            if not hit:
                print(f"[probe] ray {math.degrees(a):6.1f}deg: no hit")
                continue
            x, y, z = hit["x"], hit["y"], hit["z"]
            # Interpretation A: world coords.  Distance from ego in world.
            dw = math.hypot(x - pos[0], y - pos[1])
            # Interpretation B: local offset (x forward, y left).
            lx, ly = x, y
            dl = math.hypot(lx, ly)
            # Interpretation B world reconstruction:
            wx = pos[0] + lx * hx - ly * hy
            wy = pos[1] + lx * hy + ly * hx
            print(f"[probe] ray {math.degrees(a):6.1f}deg: "
                  f"raw=({x:.3f}, {y:.3f}, z={z:.3f}) "
                  f"as-world d={dw:.2f}m | as-local-offset d={dl:.2f}m "
                  f"-> world({wx:.1f}, {wy:.1f})")
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
