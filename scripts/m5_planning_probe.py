"""Probe the FSD-style layered planner on BeamNG.tech.

Builds a Scene from the live BEV occupancy grid, samples an arc + lane-
shift candidate fan, runs the constraint scorer and selector, and prints
the ranking + chosen path so the layered planner is verifiable live.

Usage::
    .venv\\Scripts\\python.exe scripts\\m5_planning_probe.py --runtime tech --attach
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beamng_autopilot import config
from beamng_autopilot.connector import BeamNGConnector
from beamng_autopilot.runtime import (
    build_camera_ring_provider,
    build_range_provider,
)
from beamng_autopilot.occupancy import (
    OccupancyGrid,
    fuse_obstacles_to_grid,
    project_road_mask_to_grid,
)
from beamng_autopilot.planning import (
    Constraints,
    Scene,
    cost_collision,
    cost_curvature,
    cost_lane_align,
    sample_arc,
    sample_lane_shift,
    select_trajectory,
)
from beamng_autopilot.vision.hydra import FrameContext, HydraNet
from beamng_autopilot.vision.heads.semantic import SemanticHead


def main() -> int:
    ap = argparse.ArgumentParser(description="FSD layered planner probe")
    ap.add_argument("--runtime", choices=("auto", "steam", "tech"),
                    default="auto")
    ap.add_argument("--attach", action="store_true")
    args = ap.parse_args()

    conn = BeamNGConnector(
        "italy", "etk800",
        port=config.runtime_port(args.runtime),
        home=config.runtime_home(args.runtime))
    try:
        conn.open(launch=not args.attach)
        try:
            conn.attach_vehicle(already_open=True)
        except Exception:
            conn.load_scenario()
        ring, mode = build_camera_ring_provider(conn, args.runtime, 320, 240)
        range_prov, _ = build_range_provider(conn, args.runtime)
        if ring is None:
            print(f"[plan] runtime={mode}: no ring (front-only Steam)")
            return 0

        st = conn.get_state()
        pos = np.asarray(st.pos, dtype=float)
        heading = float(st.heading)
        grid = OccupancyGrid(60, 60, 0.5,
                             origin=(float(pos[0]), float(pos[1])),
                             heading=heading)

        net = HydraNet()
        try:
            net.add(SemanticHead())
        except Exception:
            pass
        snap = ring.grab_ring()
        if "front_main" in snap:
            frame, cam = snap["front_main"]
            ctx = FrameContext(frame_rgb=frame, cam=cam, pos=pos,
                               heading=heading, ground_z=float(pos[2]),
                               role="front_main")
            out = net.run(ctx).get("semantic")
            if out is not None and "road" in out.masks:
                project_road_mask_to_grid(grid, out.masks["road"], cam,
                                          pos, heading, step=4)
        rng = range_prov.scan(pos)
        fuse_obstacles_to_grid(grid, rng.obstacles, rng.ray_hits)

        # route reference: straight ahead when no nav route is set
        xs = np.linspace(0, 40, 41)
        route = np.column_stack(
            [pos[0] + xs * np.cos(heading),
             pos[1] + xs * np.sin(heading)])
        scene = Scene(pos=pos, heading=heading, grid=grid, route=route,
                      obstacles=rng.obstacles, target_speed=8.0)

        # candidate fan: arcs + lateral shifts of the straight reference
        fans = sample_arc(pos, heading, speed=8.0, max_steer=0.4, n_curv=9)
        shifts = sample_lane_shift(route, offsets=(-2.0, -1.0, 1.0, 2.0))
        for c in shifts.candidates:
            fans.add(c.path, c.meta.get("kind", "shift"),
                     offset=c.meta.get("offset", 0.0))

        cons = Constraints(w_collision=5.0, w_curvature=0.5,
                           w_lane_align=1.0)
        best, meta = select_trajectory(scene, fans, cons)
        print(f"[plan] runtime={mode} candidates={len(fans.candidates)} "
              f"grid_drivable={int(grid.drivable.sum())} "
              f"occupied={int((grid.obstacle > 0).sum())}")
        for c in fans.candidates:
            col = cost_collision(scene, c.path, cons.collision_fraction_max)
            curv = cost_curvature(c.path)
            align = cost_lane_align(scene, c.path)
            star = " *" if (best is not None and len(best) == len(c.path)
                            and np.allclose(best, c.path)) else ""
            print(f"  {c.meta.get('kind','?'):10s} "
                  f"coll={col:.2f} curv={curv:.3f} align={align:.2f}"
                  f"{star}")
        if best is not None:
            print(f"[plan] chosen {meta} ahead={float(np.linalg.norm(best[-1][:2] - pos[:2])):.1f}m")
        else:
            print(f"[plan] no feasible trajectory ({meta.get('why')})")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())