"""M5 end-to-end test: scripted autopilot run against the real game.

Simulates the user flow without needing keyboard input:
    1. connect (attach to a running session, or load a scenario)
    2. mark a waypoint ahead of the car (as if F10 was pressed)
    3. enable autopilot (as if F9 was pressed)
    4. run the obstacle-aware planner + smooth speed controller loop until
       the goal is reached / timeout
    5. save Tesla-style overlay frames (camera projection + bird view)
    6. plot and save the post-drive throttle/brake/speed chart

Usage:
    .venv\\Scripts\\python.exe scripts\\m5_e2e_test.py                 # own scenario
    .venv\\Scripts\\python.exe scripts\\m5_e2e_test.py --attach        # running game
    .venv\\Scripts\\python.exe scripts\\m5_e2e_test.py --runtime tech  # BeamNG.tech
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beamng_autopilot import config
from beamng_autopilot.connector import BeamNGConnector
from beamng_autopilot.control.gearbox import (
    forward_gear_input,
    read_gearbox_mode,
    set_gearbox_mode,
)
from beamng_autopilot.control.handover import handover_vehicle
from beamng_autopilot.control.pure_pursuit import PurePursuit
from beamng_autopilot.control.speed import SpeedController
from beamng_autopilot.perception import filter_self_overlap
from beamng_autopilot.planner import LocalPlanner, _point_lat_offset
from beamng_autopilot.roadnet import RoadNetwork
from beamng_autopilot.runtime import (
    build_camera_provider,
    build_range_provider,
    resolve_runtime,
)
from beamng_autopilot.telemetry import TelemetryBroadcaster
from beamng_autopilot.telemetry_chart import plot_telemetry
from beamng_autopilot.vision.lanes import LaneDetector
from beamng_autopilot.vision.projection import CameraModel, default_camera
from beamng_autopilot.visionview import (
    render_birdview,
    render_camera_overlay,
)

CAM_W, CAM_H = 1076, 806
GOAL_RADIUS_M = 8.0
RAMP_ACCEL = 2.5
RAMP_DECEL = 3.5
STEER_SMOOTH = 0.35


def main() -> None:
    ap = argparse.ArgumentParser(description="M5 end-to-end test")
    ap.add_argument("--map", default=config.DEFAULT_MAP)
    ap.add_argument("--vehicle", default=config.DEFAULT_VEHICLE)
    ap.add_argument("--attach", action="store_true",
                    help="attach to a vehicle in a running session")
    ap.add_argument("--runtime", choices=("auto", "steam", "tech"),
                    default=config.RUNTIME_MODE,
                    help="game runtime: auto detects Steam/tech after connect")
    ap.add_argument("--speed", type=float, default=10.0,
                    help="cruise speed in m/s")
    ap.add_argument("--goal-dist", type=float, default=120.0,
                    help="drive about this far (m) along the road network")
    ap.add_argument("--max-run", type=float, default=240.0,
                    help="max seconds for one autopilot session")
    ap.add_argument("--save-every", type=float, default=10.0,
                    help="save an overlay frame every N seconds")
    args = ap.parse_args()

    conn = BeamNGConnector(
        args.map, args.vehicle, home=config.runtime_home(args.runtime))
    roadnet = RoadNetwork()
    telemetry = TelemetryBroadcaster()
    cam_model = default_camera(CAM_W, CAM_H)
    camera_provider = None
    range_provider = None
    lane_det: LaneDetector | None = None
    pp = PurePursuit(lookahead=6.0)
    planner = LocalPlanner()
    speed_ctrl = SpeedController()

    out_dir = config.LOGS_DIR / "telemetry"
    out_dir.mkdir(parents=True, exist_ok=True)
    hist: dict[str, list] = {"t": [], "throttle": [], "brake": [], "speed": []}
    saved_mode: str | None = None

    try:
        if args.attach:
            conn.attach_vehicle()
            print("[e2e] attached to running session")
        else:
            conn.open(launch=True)
            conn.load_scenario()
            print("[e2e] scenario loaded")
            # Pull the car onto the nearest road node so it never spawns
            # below terrain on maps whose origin is not on a road.
            t0 = time.time()
            while time.time() - t0 < 90.0:
                if roadnet.build(conn.bng):
                    if roadnet.node_count >= config.ROADNET_REPOSITION_MIN_NODES:
                        print(f"[e2e] roadnet ready: {roadnet.info}")
                        break
                    print(f"[e2e] roadnet loading: {roadnet.info}")
                time.sleep(1.0)
            if conn.reposition_on_road(roadnet):
                print("[e2e] vehicle placed on road network")
        runtime_mode = resolve_runtime(conn, args.runtime)
        print(f"[e2e] runtime={runtime_mode}")
        if runtime_mode == "steam":
            conn.set_front_camera()
        camera_provider, _ = build_camera_provider(
            conn, runtime_mode, CAM_W, CAM_H)
        range_provider, _ = build_range_provider(conn, runtime_mode)

        # Mirror autopilot start: remember the player's gearbox mode and
        # force realistic so braking at a standstill can never latch R.
        saved_mode = read_gearbox_mode(conn.vehicle) or "arcade"
        set_gearbox_mode(conn.vehicle, "realistic")
        fwd_gear = forward_gear_input(conn)
        conn.control(throttle=0.0, brake=0.0, steering=0.0,
                     parkingbrake=0.0, gear=fwd_gear)
        conn.step(5)
        print(f"[e2e] gearbox: saved={saved_mode} -> realistic, "
              f"forward gear input={fwd_gear}, parking brake released")

        # wait for the road network (up to 120 s)
        last_try, t0 = 0.0, time.time()
        while (not roadnet.ready
               or roadnet.node_count < config.ROADNET_REPOSITION_MIN_NODES) \
                and time.time() - t0 < 120.0:
            if time.time() - last_try > 3.0:
                last_try = time.time()
                if roadnet.build(conn.bng):
                    print(f"[e2e] roadnet ready: {roadnet.info}")
            time.sleep(0.2)
        print(f"[e2e] roadnet state: {roadnet.info}")

        st = conn.get_state()
        start_xy = st.pos[:2].copy()
        goal_xy = None
        if roadnet.ready:
            goal_xy = roadnet.goal_along_route(start_xy, args.goal_dist)
        if goal_xy is None:
            # map has no DecalRoad data: fall back to a straight-line goal
            heading = float(st.heading)
            goal_xy = start_xy + args.goal_dist * np.array(
                [math.cos(heading), math.sin(heading)])
            print("[e2e] no road data - straight-line goal ahead")
        goal_xy = np.asarray(goal_xy, dtype=float)
        waypoints = [goal_xy.tolist()]
        print(f"[e2e] scripted goal -> ({goal_xy[0]:.1f}, {goal_xy[1]:.1f}) "
              f"~{args.goal_dist:.0f}m {'along road network' if roadnet.ready else 'ahead'}")

        seg = None
        if roadnet.ready:
            seg = roadnet.route(start_xy, goal_xy, step=1.5)
            print(f"[e2e] roadnet route: {roadnet.info}")
            if seg is None:
                raise RuntimeError(f"A* found no path: {roadnet.info}")
        else:
            seg = roadnet._interpolate(start_xy, goal_xy, 1.5)
            print("[e2e] no road data - straight-line route")
        route = np.asarray(seg, dtype=float)
        print(f"[e2e] route generated: {len(route)} points")

        session_t0 = time.time()
        nearest = 0
        last_save = 0.0
        last_scan = 0.0
        last_lane_scan = 0.0
        last_lanes: list = []
        lane_miss = 0
        lane_frames = 0
        solid_markings = 0
        sharp_frames = 0
        sharp_max_speed = 0.0
        solid_blocked_frames = 0
        lat_offsets: list[float] = []
        obstacles = []
        obs_dist = 999.0
        ended = False
        reason = "timeout"
        max_speed = 0.0
        target_speed = 0.0
        prev_steer = 0.0
        last_ctrl = time.time()
        display_route: np.ndarray | None = None

        while time.time() - session_t0 < args.max_run:
            st = conn.get_state()
            speed = st.speed
            max_speed = max(max_speed, speed)
            if time.time() - last_scan > 0.2:
                last_scan = time.time()
                sample = range_provider.scan(
                    st.pos, conn.vehicle.vid, radius=55.0)
                # Mirror m5_autopilot: drop sensor ghosts whose footprint
                # covers the ego (a Tech LiDAR/scenario box can wrap the
                # car itself and pin the speed to zero).
                obstacles = filter_self_overlap(
                    sample.obstacles, st.pos,
                    categories=("vision", "vehicle", "scenario", "raycast"))
            if time.time() - last_lane_scan > 0.5:
                last_lane_scan = time.time()
                try:
                    if lane_det is None:
                        lane_det = LaneDetector()
                    img = camera_provider.grab()
                    if img is None or getattr(img, "size", 0) == 0:
                        raise RuntimeError("empty screenshot")
                    vw, vh = img.shape[1], img.shape[0]
                    vmodel = camera_provider.camera_model(
                        st.pos, st.heading, vw, vh, fallback=cam_model)
                    cam_model = CameraModel(
                        offset=vmodel.offset,
                        fwd_local=vmodel.fwd_local,
                        up_local=vmodel.up_local,
                        fov_deg=vmodel.fov_deg,
                        width=CAM_W,
                        height=CAM_H,
                    )
                    lanes = lane_det.detect(
                        img, vmodel, st.pos, st.heading,
                        ground_z=(float(st.pos[2]) - config.EGO_ORIGIN_GROUND_GAP_M
                                  if len(st.pos) > 2 else 0.0))
                    if lanes:
                        lane_miss = 0
                        last_lanes = lanes
                        lane_frames += 1
                        solid_markings += sum(
                            1 for mk in lanes if mk.kind == "solid")
                    else:
                        lane_miss += 1
                        if lane_miss > 6:
                            last_lanes = []
                except Exception as exc:
                    print(f"[e2e] lane scan error: {exc}")
            desired_speed = args.speed
            blocked = False
            if route is not None and len(route) > 0:
                d = np.linalg.norm(route[:, :2] - st.pos[:2], axis=1)
                nearest = int(np.argmin(d))
                lat_offsets.append(float(_point_lat_offset(
                    float(st.pos[0]), float(st.pos[1]), route)))
                drive_route, blocked = planner.plan(
                    route, obstacles, st.pos, st.heading, nearest,
                    solid_lines=last_lanes)
                drive_route = np.asarray(drive_route, dtype=float)
                if len(drive_route) >= 2:
                    d0 = np.linalg.norm(
                        drive_route[:, :2] - st.pos[:2], axis=1)
                    start_i = int(np.argmin(d0))
                    if start_i > 0 and len(drive_route) - start_i >= 2:
                        drive_route = drive_route[start_i:]
                display_route = drive_route
                desired_speed, obs_dist = planner.speed(
                    drive_route, obstacles, st.pos, st.heading,
                    0, args.speed)
                if blocked:
                    desired_speed = 0.0
                if planner.last_sharp:
                    sharp_frames += 1
                    sharp_max_speed = max(sharp_max_speed, speed)
                if (planner.last_blocker is not None
                        and planner.last_blocker[0] == "solid line"):
                    solid_blocked_frames += 1
                pp.lookahead = pp.adaptive_lookahead(speed)
                steer_rad, _, _ = pp.steering(
                    st.pos, st.heading, drive_route, 0)
                new_steer = float(np.clip(-steer_rad / 0.6, -1.0, 1.0))
                steer = prev_steer + STEER_SMOOTH * (new_steer - prev_steer)
                prev_steer = steer
                steer_angle = abs(steer) * 0.6
                if steer_angle > 0.09:
                    steer_radius = 2.9 / math.tan(steer_angle)
                    desired_speed = min(
                        desired_speed,
                        float(math.sqrt(7.0 * steer_radius)))
            else:
                steer, desired_speed = 0.0, args.speed
                obs_dist = 999.0
                prev_steer = 0.0

            now = time.time()
            dt = max(1e-3, now - last_ctrl)
            last_ctrl = now
            if desired_speed > target_speed:
                target_speed = min(desired_speed,
                                   target_speed + RAMP_ACCEL * dt)
            else:
                target_speed = max(desired_speed,
                                   target_speed - RAMP_DECEL * dt)

            throttle, brake = speed_ctrl.update(target_speed, speed)

            t = time.time() - session_t0
            hist["t"].append(t)
            hist["throttle"].append(float(throttle))
            hist["brake"].append(float(brake))
            hist["speed"].append(float(speed))
            telemetry.publish(
                t=t, speed=float(speed), throttle=throttle, brake=brake,
                steer=steer, vel=st.vel, dir_vec=st.dir, up_vec=st.up,
                pos=st.pos, heading=float(st.heading),
                nearest=int(nearest),
                extra={"mode": "E2E", "wp": len(waypoints),
                       "obs": len(obstacles), "obs_d": round(float(obs_dist), 1)},
            )

            conn.control(throttle=throttle, steering=steer, brake=brake)

            if route is not None and len(route) > 0:
                goal_dist = float(np.linalg.norm(route[-1][:2] - st.pos[:2]))
                if goal_dist < GOAL_RADIUS_M:
                    ended, reason = True, "goal reached"

            if t - last_save >= args.save_every:
                last_save = t
                try:
                    img = camera_provider.grab()
                    img = cv2.resize(img, (CAM_W, CAM_H))
                    img = render_camera_overlay(
                        img, display_route, st.pos, st.heading, cam_model,
                        obstacles=obstacles, lane_markings=last_lanes)
                    bv = np.full((CAM_H, CAM_H, 3), (22, 24, 30), np.uint8)
                    render_birdview(
                        bv, route_xy=display_route, waypoints=waypoints,
                        goal_xy=goal_xy, pos=st.pos, heading=st.heading,
                        obstacles=obstacles, lane_markings=last_lanes)
                    frame = np.hstack([img, bv])
                    p = out_dir / f"m5_e2e_t{int(t)}s.png"
                    cv2.imwrite(str(p), frame)
                    print(f"[e2e] overlay frame -> {p}")
                except Exception as exc:
                    print(f"[e2e] frame error: {exc}")

            if ended:
                break
            conn.step(1)

        # end of session: stop and hand back without latching reverse
        handover_vehicle(conn, saved_mode, True)

        chart_path = out_dir / (
            f"m5_e2e_chart_{time.strftime('%Y%m%d_%H%M%S')}.png")
        plot_telemetry(hist, chart_path, block=False, show=False)
        print(f"[e2e] telemetry chart -> {chart_path}")

        final = conn.get_state().pos[:2]
        dist = float(np.linalg.norm(final - start_xy))
        print(f"[e2e] RESULT reason={reason} elapsed={t:.1f}s "
              f"dist={dist:.1f}m max_speed={max_speed:.1f}m/s")
        print(f"[e2e] lanes: detected_frames={lane_frames} "
              f"solid_markings={solid_markings} last={len(last_lanes)}")
        print(f"[e2e] sharp corners: frames={sharp_frames} "
              f"max_speed_while_sharp={sharp_max_speed:.1f}m/s "
              f"cap={planner.sharp_corner_kph / 3.6:.1f}m/s")
        print(f"[e2e] solid-line blocks: {solid_blocked_frames}")
        if lat_offsets:
            lat = np.asarray(lat_offsets, dtype=float)
            right_frac = float(np.mean(lat < -0.3))
            print(f"[e2e] right-side driving: median_lat={np.median(lat):.2f}m "
                  f"mean_lat={lat.mean():.2f}m "
                  f"right_fraction={right_frac * 100.0:.1f}%")

    except Exception as exc:
        print(f"[e2e] FAILED: {exc}")
        raise
    finally:
        try:
            handover_vehicle(conn, saved_mode, True)
        except Exception:
            pass
        telemetry.close()
        if camera_provider is not None:
            try:
                camera_provider.close()
            except Exception:
                pass
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
