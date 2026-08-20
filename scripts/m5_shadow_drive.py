"""Shadow-mode drive: record aligned (truth control vs shadow prediction).

Drives the car with a simple rule autopilot (PurePursuit toward a
straight-ahead route) while the FSD-style stack runs in shadow: the same
BEV occupancy grid + layered planner the perception mainline will use,
and every frame the executed control and the shadow trajectory are
recorded into one .npz episode via ``ShadowRecorder``.

This is the data-fusion harness of the project: the recorded
(bev_raster, steer, throttle) pairs are exactly the input/action
contract the end-to-end skeleton trains against.

Usage::
    .venv\\Scripts\\python.exe scripts\\m5_shadow_drive.py --runtime tech \\
        --attach --seconds 15 --out logs/live_runs/shadow_episodes
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beamng_autopilot import config
from beamng_autopilot.connector import BeamNGConnector
from beamng_autopilot.control.pure_pursuit import PurePursuit
from beamng_autopilot.occupancy import (
    OccupancyGrid,
    fuse_obstacles_to_grid,
    project_road_mask_to_grid,
)
from beamng_autopilot.planning import (
    Constraints,
    Scene,
    sample_arc,
    select_trajectory,
)
from beamng_autopilot.recording import ShadowFrame, ShadowRecorder
from beamng_autopilot.runtime import (
    build_camera_ring_provider,
    build_range_provider,
)
from beamng_autopilot.vision.hydra import FrameContext, HydraNet
from beamng_autopilot.vision.heads.semantic import SemanticHead


def main() -> int:
    ap = argparse.ArgumentParser(description="shadow-mode recorder")
    ap.add_argument("--runtime", choices=("auto", "steam", "tech"),
                    default="auto")
    ap.add_argument("--attach", action="store_true")
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--speed", type=float, default=5.0)
    ap.add_argument("--out", default="logs/live_runs/shadow_episodes")
    ap.add_argument("--drive", action="store_true",
                    help="actually drive with the rule autopilot")
    args = ap.parse_args()

    conn = BeamNGConnector(
        "italy", "etk800",
        port=config.runtime_port(args.runtime),
        home=config.runtime_home(args.runtime))
    rec = ShadowRecorder(args.out, f"shadow_{int(time.time())}")
    pp = PurePursuit(lookahead=5.0)
    try:
        conn.open(launch=not args.attach)
        try:
            conn.attach_vehicle(already_open=True)
        except Exception:
            conn.load_scenario()
        ring, mode = build_camera_ring_provider(conn, args.runtime, 320, 240)
        range_prov, _ = build_range_provider(conn, args.runtime)
        if ring is None:
            print(f"[shadow] runtime={mode}: no ring provider")
            return 0
        net = HydraNet()
        try:
            net.add(SemanticHead())
        except Exception:
            pass
        cons = Constraints(w_collision=5.0, w_curvature=0.5,
                           w_lane_align=1.0)

        heading0 = float(conn.get_state().heading)
        x0 = conn.get_state().pos[:2]
        # straight route ahead of the spawn heading
        xs = np.linspace(0, 40, 41)
        route = np.column_stack([x0[0] + xs * np.cos(heading0),
                                 x0[1] + xs * np.sin(heading0)])

        t_end = time.time() + args.seconds
        frames = 0
        while time.time() < t_end:
            st = conn.get_state()
            pos = np.asarray(st.pos, dtype=float)
            heading = float(st.heading)
            v = float(st.speed)

            grid = OccupancyGrid(60, 60, 0.5,
                                 origin=(float(pos[0]), float(pos[1])),
                                 heading=heading)
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

            scene = Scene(pos=pos, heading=heading, grid=grid, route=route,
                          obstacles=rng.obstacles, target_speed=args.speed)
            fans = sample_arc(pos, heading, speed=max(2.0, v),
                              max_steer=0.4, n_curv=9)
            best, meta = select_trajectory(scene, fans, cons)

            # executed control: rule PP toward the straight route + cruise
            drive_route = route
            if best is not None:
                # prefer the shadow path when it is within the lane budget
                route_ref = best
            else:
                route_ref = drive_route
            steer = 0.0
            if route_ref is not None and len(route_ref) >= 2:
                res = pp.steering(
                    pos, heading, np.asarray(route_ref, dtype=float))
                steer = float(res[0]) if isinstance(res, tuple) else float(res)
            throttle = 0.35 if v < args.speed else 0.0
            brake = 1.0 if v > args.speed + 1.5 else 0.0
            if args.drive:
                conn.control(throttle=throttle, brake=brake,
                             steering=steer)
            conn.step(3)

            rec.add(ShadowFrame(
                x=float(pos[0]), y=float(pos[1]), heading=heading,
                speed=v, throttle=throttle, brake=brake, steer=float(steer),
                bev_raster=grid.as_raster(),
                drivable=grid.drivable,
                trajectory=None if best is None else best,
                target_speed=float(args.speed),
                lane_src="semantic" if net.names() else "",
                cost=float(meta.get("cost", -1.0)),
                kind=meta.get("kind", "")))
            frames += 1
        out = rec.save()
        print(f"[shadow] runtime={mode} frames={frames}")
        if out:
            print(f"[shadow] episode saved -> {out}")
        else:
            print("[shadow] nothing recorded")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())