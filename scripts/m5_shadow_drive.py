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

import math

from beamng_autopilot import config
from beamng_autopilot.connector import BeamNGConnector
from beamng_autopilot.control import gearbox
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
from beamng_autopilot.roadnet import RoadNetwork
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
    ap.add_argument("--teleport", nargs=3, type=float, default=None,
                    metavar=("X", "Y", "YAW_DEG"),
                    help="teleport to an open stretch before driving")
    ap.add_argument("--goal", nargs=2, type=float, default=None,
                    metavar=("X", "Y"),
                    help="follow the road-graph A* route to this goal "
                         "(FSD-style) instead of a straight line")
    ap.add_argument("--min-quality", type=float, default=0.5,
                    help="drop shadow frames with a gated-out prediction "
                         "(no feasible trajectory); 0 keeps everything")
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
        if args.teleport is not None:
            x, y, yaw = args.teleport
            conn.safe_teleport(float(x), float(y), heading_deg=float(yaw))
            st1 = conn.get_state()
            print(f"[shadow] teleport -> "
                  f"({float(st1.pos[0]):.1f}, {float(st1.pos[1]):.1f}, "
                  f"{float(st1.pos[2]):.1f})")
        ring, mode = build_camera_ring_provider(
            conn, args.runtime, 320, 240, roles=("front_main",))
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

        st0 = conn.get_state()
        heading0 = float(st0.heading)
        x0 = np.asarray(st0.pos[:2], dtype=float)
        # Straight-route fallback: a real stack plans ALONG the road graph,
        # so when --goal is given build the same A* route the FSD drive
        # uses and snap the car onto it, facing its direction.
        nav_route = None
        if args.goal is not None:
            rn = RoadNetwork()
            t_road = time.time()
            while not rn.ready and time.time() - t_road < 90.0:
                try:
                    if rn.build(conn.bng):
                        break
                except Exception:
                    pass
                time.sleep(1.0)
            if rn.ready:
                n0 = rn.nodes[rn._nearest(x0)]
                _rwe = rn.route_with_edges(
                    n0, np.asarray(args.goal, dtype=float))
                if _rwe[0] is not None and len(_rwe[0]) >= 4:
                    nav_route = np.asarray(_rwe[0][:, :2], dtype=float)
                    dseg = np.linalg.norm(np.diff(nav_route, axis=0),
                                          axis=1)
                    print(f"[shadow] road-graph route: {len(nav_route)} "
                          f"pts, {float(np.sum(dseg)):.1f} m ({rn.info})")
            if nav_route is None:
                print("[shadow] road graph gave no route; "
                      "falling back to a straight line")
            else:
                # Snap onto the route, facing along it (same rule as the
                # FSD drive: never start a run across its own lane).
                d0 = np.linalg.norm(nav_route - x0, axis=1)
                i = int(np.argmin(d0))
                if i + 1 < len(nav_route):
                    ndx, ndy = (float(nav_route[i + 1, 0] - nav_route[i, 0]),
                                float(nav_route[i + 1, 1] - nav_route[i, 1]))
                else:
                    ndx, ndy = (float(nav_route[i, 0] - nav_route[i - 1, 0]),
                                float(nav_route[i, 1] - nav_route[i - 1, 1]))
                h = float(np.arctan2(ndy, ndx))
                conn.safe_teleport(float(nav_route[i, 0]),
                                   float(nav_route[i, 1]),
                                   heading_deg=math.degrees(h))
                st1 = conn.get_state()
                print(f"[shadow] snapped onto route "
                      f"({float(st1.pos[0]):.1f}, {float(st1.pos[1]):.1f}, "
                      f"{float(st1.pos[2]):.1f})")
                heading0 = float(st1.heading)
                x0 = np.asarray(st1.pos[:2], dtype=float)
        # Lock the box into a forward gear (D on the etk800) so a teleport
        # can never leave the car in park/reverse and the recorder actually
        # drives; release the parking brake the helper engages.
        fwd_gear = gearbox.forward_gear_input(conn)
        conn.control(throttle=0.0, brake=0.0, steering=0.0,
                     parkingbrake=0.0, gear=fwd_gear)
        conn.step(3)
        # straight route ahead of the spawn heading (fallback reference)
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
            # Shadow data is front-camera only: poll just FRONT_MAIN
            # instead of the full 8-camera ring so recording ticks stay
            # fast (ring polls were the per-frame bottleneck).
            frame_rgb = np.ascontiguousarray(ring.grab(), dtype=np.uint8)
            cam = ring.camera_model(pos, heading, ring.width, ring.height)
            label_map = None
            ctx = FrameContext(frame_rgb=frame_rgb, cam=cam, pos=pos,
                               heading=heading, ground_z=float(pos[2]),
                               role="front_main")
            out = net.run(ctx).get("semantic")
            if out is not None and "road" in out.masks:
                project_road_mask_to_grid(grid, out.masks["road"], cam,
                                          pos, heading, step=4)
                h, w = frame_rgb.shape[:2]
                label_map = np.zeros((h, w), dtype=np.uint8)
                label_map[out.masks["road"]] = 1
                label_map[out.masks["line"]] = 2
            rng = range_prov.scan(pos)
            fuse_obstacles_to_grid(grid, rng.obstacles, rng.ray_hits)

            scene_route = nav_route if nav_route is not None else route
            scene = Scene(pos=pos, heading=heading, grid=grid,
                          route=scene_route, obstacles=rng.obstacles,
                          target_speed=args.speed)
            fans = sample_arc(pos, heading, speed=max(2.0, v),
                              max_steer=0.4, n_curv=9)
            best, meta = select_trajectory(scene, fans, cons)

            # executed control: rule PP toward the route (road graph when
            # available, else straight ahead) + cruise
            drive_route = nav_route if nav_route is not None else route
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
                             steering=steer, parkingbrake=0.0,
                             gear=fwd_gear)
            conn.step(3)

            # Shadow-prediction quality gate: only frames with a feasible
            # shadow trajectory are worth training on; bad predictions
            # would otherwise poison the end-to-end dataset.
            cost = float(meta.get("cost", -1.0))
            quality = 1.0 if best is not None else 0.0
            if args.min_quality > 0.0 and quality < args.min_quality:
                continue
            rec.add(ShadowFrame(
                x=float(pos[0]), y=float(pos[1]), heading=heading,
                speed=v, throttle=throttle, brake=brake, steer=float(steer),
                bev_raster=grid.as_raster(),
                drivable=grid.drivable,
                trajectory=None if best is None else best,
                target_speed=float(args.speed),
                lane_src="semantic" if net.names() else "",
                cost=cost,
                kind=meta.get("kind", ""),
                rgb=frame_rgb,
                label=label_map,
                quality=quality))
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