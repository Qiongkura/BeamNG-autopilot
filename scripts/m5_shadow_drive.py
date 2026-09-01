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
    sample_lane_shift,
    select_trajectory,
)
from beamng_autopilot.planning.local_route import map_lane_edges
from beamng_autopilot.lane import perception_lateral_guard
from beamng_autopilot.recording import ShadowFrame, ShadowRecorder
from beamng_autopilot.roadnet import RoadNetwork
from beamng_autopilot.runtime import (
    build_camera_ring_provider,
    build_range_provider,
)
from beamng_autopilot.vision.hydra import FrameContext, HydraNet
from beamng_autopilot.vision.heads.semantic import SemanticHead


# Snap/restart offset into the RIGHT lane (same as fsd_drive): the
# recorder must label a car in its OWN lane, not one riding the route
# centre line into the oncoming lane.
SNAP_LANE_OFFSET_M = 1.6


def _path_curvature_ff(path, pos, heading, near_m: float = 1.5,
                       horizon_m: float = 8.0, wheelbase: float = 2.9,
                       ratio: float = 0.6, max_ff: float = 0.40) -> float:
    """Feed-forward steering from the chosen path's near-ahead curvature.

    PurePursuit only reacts at the lookahead point, so on a hairpin the
    car reaches the entry still pointing straight and runs into the
    outside wall (first -110 -> -24 deg bend on the mountain route).  A
    feed-forward term from the path curvature 2-10 m ahead starts the
    turn as soon as the path bends.  Returns a NORMALIZED steering input
    (negative = left), scaled by how aligned the ego heading is with the
    path so a sideways rejoin is not fought.
    """
    if path is None or len(path) < 4:
        return 0.0
    p = np.asarray(path[:, :2], dtype=float)
    pos2 = np.asarray(pos[:2], dtype=float)
    n = len(p)
    d = np.linalg.norm(p - pos2, axis=1)
    i0 = int(np.argmin(d))
    arc = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    base = float(arc[i0])
    idxs = [i0]
    for tgt in (near_m, near_m + horizon_m):
        j = i0
        while j < n - 1 and float(arc[j]) - base < tgt:
            j += 1
        idxs.append(j)
    i1, i2 = idxs[1], idxs[2]
    if i2 - i1 < 2:
        return 0.0

    def _tangent(i: int) -> np.ndarray:
        a = max(0, i - 1)
        b = min(n - 1, i + 1)
        v = p[b] - p[a]
        L = float(np.linalg.norm(v))
        return (v / L) if L > 1e-9 else np.array([1.0, 0.0])

    t1 = _tangent(i1)
    t2 = _tangent(i2)
    th1 = math.atan2(float(t1[1]), float(t1[0]))
    th2 = math.atan2(float(t2[1]), float(t2[0]))
    dth = (th2 - th1 + math.pi) % (2.0 * math.pi) - math.pi
    ds = max(1e-3, float(arc[i2] - arc[i1]))
    kappa = dth / ds
    align = float(np.clip(
        math.cos(th1 - float(heading)), 0.0, 1.0))
    ff = -kappa * wheelbase / ratio   # left curve (kappa>0) -> negative input
    return float(np.clip(ff * (0.3 + 0.7 * align), -max_ff, max_ff))


def _curve_speed_mps(route: np.ndarray | None, pos: np.ndarray,
                    heading: float, horizon_m: float = 10.0) -> float:
    """Cap speed ahead of a bend (rule-driver curve governor).

    Measures how much the route direction rotates over the next
    ``horizon_m`` from the nearest route point and returns a safe speed
    (2 m/s for a hairpin, 3.5 for a normal bend, 6+ for straight road)
    so the simple PP driver can round corners instead of wedging into
    the outside wall.
    """
    if route is None or len(route) < 4:
        return 8.0
    pos = np.asarray(pos, dtype=float)[:2]
    d = np.linalg.norm(route - pos, axis=1)
    i = int(np.argmin(d))
    seg = 0.0
    j = i
    while j + 1 < len(route) and seg < horizon_m:
        seg += float(np.linalg.norm(route[j + 1] - route[j]))
        j += 1
    if j <= i + 1:
        return 8.0
    a0 = float(np.arctan2(route[i + 1, 1] - route[i, 1],
                          route[i + 1, 0] - route[i, 0]))
    a1 = float(np.arctan2(route[j, 1] - route[j - 1, 1],
                          route[j, 0] - route[j - 1, 0]))
    deg = abs(float(np.degrees((a1 - a0 + np.pi) % (2 * np.pi) - np.pi)))
    if deg >= 70.0:
        return 2.2
    if deg >= 40.0:
        return 3.5
    return 8.0


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
    ap.add_argument("--min-quality", type=float, default=0.0,
                    help="frames with a gated-out shadow prediction "
                         "(no feasible trajectory) are still saved for "
                         "action-only supervision; 0.5 keeps only "
                         "trajectory-valid frames")
    ap.add_argument("--lidar-every", type=int, default=1,
                    help="reuse the last LiDAR scan for N-1 of every N "
                         "frames (range scan is the per-frame bottleneck)")
    args = ap.parse_args()

    conn = BeamNGConnector(
        "italy", "etk800",
        port=config.runtime_port(args.runtime),
        home=config.runtime_home(args.runtime))
    rec = ShadowRecorder(args.out, f"shadow_{int(time.time())}")
    pp = PurePursuit(lookahead=2.5)
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
        road_left = None
        road_right = None
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
                    road_left = (np.asarray(_rwe[1][:, :2], dtype=float)
                                 if _rwe[1] is not None else None)
                    road_right = (np.asarray(_rwe[2][:, :2], dtype=float)
                                  if _rwe[2] is not None else None)
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
                # Right-lane start: right normal = (sin h, -cos h).  A
                # recorder starting on the centre line labels a car that
                # rides the line / cuts the hairpin into oncoming.
                route_start_xy = (
                    float(nav_route[i, 0]) + SNAP_LANE_OFFSET_M * math.sin(h),
                    float(nav_route[i, 1]) - SNAP_LANE_OFFSET_M * math.cos(h))
                route_start_deg = math.degrees(h)
                conn.safe_teleport(*route_start_xy,
                                   heading_deg=route_start_deg)
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
        route_start_xy = (float(x0[0]), float(x0[1]))
        route_start_deg = math.degrees(heading0)
        # straight route ahead of the spawn heading (fallback reference)
        xs = np.linspace(0, 40, 41)
        route = np.column_stack([x0[0] + xs * np.cos(heading0),
                                 x0[1] + xs * np.sin(heading0)])

        t_end = time.time() + args.seconds
        frames = 0
        rng_last = None
        stuck_t0 = None
        while time.time() < t_end:
            st = conn.get_state()
            pos = np.asarray(st.pos, dtype=float)
            heading = float(st.heading)
            v = float(st.speed)
            drive_route = nav_route if nav_route is not None else route
            # End-stop: once the remaining route is short, brake to a
            # standstill in the lane instead of driving off the route end
            # onto the roadside grass (the route ends at a junction and
            # the simple driver kept going past it).
            if drive_route is None or len(drive_route) < 4:
                near_end = False
            else:
                d_end = np.linalg.norm(drive_route - pos[:2], axis=1)
                i_end = int(np.argmin(d_end))
                arc = np.concatenate([[0.0], np.cumsum(
                    np.linalg.norm(np.diff(drive_route, axis=0), axis=1))])
                near_end = float(arc[-1] - arc[i_end]) < 12.0
            # Stuck restart: the rule driver can wedge against a bend's
            # outside wall (first hairpin on the mountain route); rather
            # than burning the rest of the episode on identical static
            # frames, teleport back to the route start and attack the
            # corner again - every episode then contains several
            # approach/steer attempts.
            if near_end:
                stuck_t0 = None
            elif v < 0.3:
                if stuck_t0 is None:
                    stuck_t0 = time.time()
                elif time.time() - stuck_t0 > 3.0:
                    print(f"[shadow] stuck at ({float(pos[0]):.1f}, "
                          f"{float(pos[1]):.1f}) - restarting route")
                    conn.safe_teleport(*route_start_xy,
                                       heading_deg=route_start_deg)
                    conn.control(throttle=0.0, brake=0.0, steering=0.0,
                                 parkingbrake=0.0, gear=fwd_gear)
                    conn.step(3)
                    stuck_t0 = None
                    continue
            else:
                stuck_t0 = None

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
            if args.lidar_every > 1 and frames > 0 and \
                    frames % args.lidar_every != 0:
                rng = rng_last
            else:
                rng = range_prov.scan(pos)
                rng_last = rng
            fuse_obstacles_to_grid(grid, rng.obstacles, rng.ray_hits)

            scene_route = nav_route if nav_route is not None else route
            # Own-lane map reference (same as the FSD drive): the recorder
            # must label a car in ITS OWN lane, not one riding the route
            # centre line (which wedges on hairpins and labels centre-line
            # driving).  The map lane centre is the lane_center candidate
            # and the fallback when the arc fan declines.
            scene_lane_ref = None
            lane_left = lane_right = None
            lane_width = 0.0
            if nav_route is not None and road_left is not None \
                    and road_right is not None:
                try:
                    map_lane = map_lane_edges(
                        nav_route, road_left, road_right, pos, heading)
                except Exception:
                    map_lane = None
                if map_lane is not None:
                    mc, ml, mr = map_lane
                    scene_lane_ref = np.asarray(mc, dtype=float)[:, :2]
                    lane_left = np.asarray(ml, dtype=float)[:, :2]
                    lane_right = np.asarray(mr, dtype=float)[:, :2]
                    _n2 = min(len(lane_right), len(lane_left))
                    _wa = np.linalg.norm(
                        lane_right[:_n2] - lane_left[:_n2], axis=1)
                    _wf = _wa[np.isfinite(_wa)]
                    if _wf.size:
                        lane_width = float(np.median(_wf))
            scene = Scene(pos=pos, heading=heading, grid=grid,
                          route=scene_route, lane_ref=scene_lane_ref,
                          lane_left=lane_left, lane_right=lane_right,
                          lane_width=lane_width,
                          obstacles=rng.obstacles,
                          target_speed=args.speed)
            fans = sample_arc(pos, heading, speed=max(2.0, v),
                              max_steer=0.4, n_curv=9)
            if scene_lane_ref is not None and len(scene_lane_ref) >= 4:
                fans.add(np.asarray(scene_lane_ref, dtype=float)[:, :2],
                         "lane_center", offset=0.0)
                _sh = sample_lane_shift(scene_lane_ref,
                                        offsets=(-1.5, 1.5))
                for _c in _sh.candidates:
                    fans.add(_c.path, _c.meta.get("kind", "shift"),
                             offset=_c.meta.get("offset", 0.0))
            best, meta = select_trajectory(scene, fans, cons)

            # executed control: follow the shadow planner's ``best`` arc
            # when it exists (it is smooth through bends) and fall back
            # to the own-lane centre (or the road graph when no map lane
            # is available) otherwise; the lateral guard below keeps the
            # car inside the road instead of cutting onto grass.
            fallback_ref = drive_route
            if best is None and scene_lane_ref is not None \
                    and len(scene_lane_ref) >= 4:
                fallback_ref = np.asarray(scene_lane_ref)[:, :2]
            route_ref = best if best is not None else fallback_ref
            steer = 0.0
            if route_ref is not None and len(route_ref) >= 2:
                res = pp.steering(
                    pos, heading, np.asarray(route_ref, dtype=float))
                steer_rad = float(res[0]) if isinstance(res, tuple) \
                    else float(res)
                # PurePursuit returns radians: normalise by the steering
                # ratio (and flip sign to BeamNG's left-negative input) and
                # add the curvature feed-forward so bends are pre-turned
                # instead of run wide into the outside wall.
                # Stay-on-road is PERCEPTION-only: the BEV drivable mask
                # (semantic road back-projection + LiDAR free space) says
                # where the road is.  No map, no nav route - the same
                # guard runs on any map from the sensors alone.
                steer = float(np.clip(
                    -steer_rad / 0.6 +
                    _path_curvature_ff(route_ref, pos, heading) +
                    perception_lateral_guard(grid),
                    -1.0, 1.0))
            v_target = min(args.speed,
                           _curve_speed_mps(drive_route, pos, heading))
            # Without a shadow arc the rule fallback follows the raw
            # road-graph polyline, which kinks too sharply to track at
            # speed (wedged a right bend at 6.6 m/s); crawl instead.
            if best is None:
                v_target = min(v_target, 3.0)
            if near_end:
                throttle = 0.0
                brake = 1.0 if v > 0.2 else 0.3
            else:
                throttle = 0.35 if v < v_target else 0.0
                brake = 1.0 if v > v_target + 1.5 else 0.0
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