"""Live route probe: drive a real in-game navigation route through a sharp
corner and report right-side offset, lane markings, solid-line rule and
the >45 deg corner speed cap."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
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
from beamng_autopilot.control.pure_pursuit import PurePursuit
from beamng_autopilot.control.speed import SpeedController
from beamng_autopilot.perception import scan_obstacles_all
from beamng_autopilot.planner import (
    LocalPlanner,
    _point_lat_offset,
    corner_angle_deg,
)
from beamng_autopilot.vision.detection import live_camera_model
from beamng_autopilot.vision.lanes import LaneDetector
from beamng_autopilot.vision.projection import CameraModel, default_camera
from beamng_autopilot.visionview import (
    render_birdview,
    render_camera_overlay,
)

CAM_W, CAM_H = 1076, 806
GOAL_RADIUS_M = 10.0


def _ground_z(conn, x: float, y: float) -> float | None:
    """Raycast straight down from high above to find the ground surface."""
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
    ap = argparse.ArgumentParser(description="M5 live route probe")
    ap.add_argument("--map", default=config.DEFAULT_MAP)
    ap.add_argument("--vehicle", default=config.DEFAULT_VEHICLE)
    ap.add_argument("--speed", type=float, default=17.0,
                    help="cruise speed in m/s")
    ap.add_argument("--goal-dist", type=float, default=330.0,
                    help="drive about this far along the nav route")
    ap.add_argument("--start-x", type=float, default=726.6)
    ap.add_argument("--start-y", type=float, default=755.9)
    ap.add_argument("--goal-x", type=float, default=555.8)
    ap.add_argument("--goal-y", type=float, default=394.2)
    ap.add_argument("--teleport", action="store_true", default=True,
                    help="teleport the ego to --start-x/y before routing")
    ap.add_argument("--no-teleport", action="store_false", dest="teleport")
    ap.add_argument("--max-run", type=float, default=150.0)
    ap.add_argument("--save-every", type=float, default=12.0)
    args = ap.parse_args()

    conn = BeamNGConnector(args.map, args.vehicle)
    out_dir = config.LOGS_DIR / "m5_route"
    out_dir.mkdir(parents=True, exist_ok=True)
    hist: dict[str, list] = {"t": [], "throttle": [], "brake": [], "speed": []}
    saved_mode: str | None = None

    try:
        conn.open(launch=False)
        conn.attach_vehicle(already_open=True)
        st0 = conn.get_state()
        start_xy = np.array([args.start_x, args.start_y], dtype=float)
        goal_xy = np.array([args.goal_x, args.goal_y], dtype=float)
        if args.teleport:
            heading = math.atan2(
                goal_xy[1] - start_xy[1], goal_xy[0] - start_xy[0])
            yaw_deg = -math.degrees(float(heading)) - 90.0
            z = float(st0.pos[2]) if len(st0.pos) > 2 else 0.0
            ground_z = _ground_z(conn, float(start_xy[0]), float(start_xy[1]))
            if ground_z is not None:
                z = ground_z + 0.6
            conn.vehicle.teleport(
                (float(start_xy[0]), float(start_xy[1]), z),
                rot_quat=angle_to_quat((0.0, 0.0, yaw_deg)))
            conn.control(throttle=0.0, brake=0.0, steering=0.0,
                         parkingbrake=0.0)
            conn.step(30)
            st0 = conn.get_state()
            print(f"[route] teleported to ({start_xy[0]:.1f}, "
                  f"{start_xy[1]:.1f}, z={z:.2f}) heading="
                  f"{math.degrees(float(st0.heading)):.1f} deg")
        conn.bng.control.queue_lua_command(
            "core_groundMarkers.setPath({vec3(%.3f, %.3f, 0)})\nreturn 'ok'"
            % (float(goal_xy[0]), float(goal_xy[1])), response=True)
        time.sleep(0.6)
        nav = conn.read_navigation_route()
        if nav is None or len(nav) < 4:
            raise RuntimeError("no in-game navigation route available")
        route = nav[:, :2]
        dseg = np.linalg.norm(np.diff(route, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(dseg)])
        end = int(np.argmin(np.abs(cum - args.goal_dist)))
        route = route[: end + 1]
        goal_xy = route[-1].copy()
        print(f"[route] nav route: {len(route)} pts, "
              f"length={cum[end]:.1f} m, goal={goal_xy.tolist()}")

        conn.set_front_camera()
        saved_mode = read_gearbox_mode(conn.vehicle) or "arcade"
        set_gearbox_mode(conn.vehicle, "realistic")
        fwd_gear = forward_gear_input(conn)
        conn.control(throttle=0.0, brake=0.0, steering=0.0,
                     parkingbrake=0.0, gear=fwd_gear)
        conn.step(5)
        print(f"[route] forward gear input={fwd_gear}, parking brake released")

        cam_model = default_camera(CAM_W, CAM_H)
        lane_det = LaneDetector()
        pp = PurePursuit(lookahead=6.0)
        planner = LocalPlanner()
        speed_ctrl = SpeedController()

        session_t0 = time.time()
        nearest = 0
        last_save = 0.0
        last_scan = 0.0
        last_lane_scan = 0.0
        last_diag_t = -1
        last_lanes: list = []
        lane_miss = 0
        lane_frames = 0
        solid_markings = 0
        sharp_frames = 0
        sharp_max_speed = 0.0
        sharp_max_angle = 0.0
        solid_blocked_frames = 0
        lat_offsets: list[float] = []
        obstacles = []
        obs_dist = 999.0
        max_speed = 0.0
        target_speed = 0.0
        prev_steer = 0.0
        last_ctrl = time.time()
        display_route: np.ndarray | None = None
        ended = False
        reason = "timeout"

        while time.time() - session_t0 < args.max_run:
            st = conn.get_state()
            speed = st.speed
            max_speed = max(max_speed, speed)
            if time.time() - last_scan > 0.2:
                last_scan = time.time()
                obstacles = scan_obstacles_all(
                    conn.bng, conn.vehicle.vid, st.pos, radius=55.0)
            if time.time() - last_lane_scan > 0.5:
                last_lane_scan = time.time()
                try:
                    img = conn._grab_lua_screenshot(timeout=6.0)
                    if img is None or getattr(img, "size", 0) == 0:
                        raise RuntimeError("empty screenshot")
                    vw, vh = img.shape[1], img.shape[0]
                    vmodel = live_camera_model(
                        conn.bng, vw, vh, st.pos, st.heading)
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
                        ground_z=(float(st.pos[2])
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
                    print(f"[route] lane scan error: {exc}")

            desired_speed = args.speed
            blocked = False
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
                drive_route, obstacles, st.pos, st.heading, 0, args.speed)
            if blocked:
                desired_speed = 0.0
            if planner.last_sharp:
                sharp_frames += 1
                sharp_max_speed = max(sharp_max_speed, speed)
                sharp_max_angle = max(
                    sharp_max_angle,
                    corner_angle_deg(route, nearest))
            if (planner.last_blocker is not None
                    and planner.last_blocker[0] == "solid line"):
                solid_blocked_frames += 1
            pp.lookahead = pp.adaptive_lookahead(speed)
            steer_rad, _, _ = pp.steering(
                st.pos, st.heading, drive_route, 0)
            steer = float(np.clip(-steer_rad / 0.6, -1.0, 1.0))
            steer = prev_steer + 0.35 * (steer - prev_steer)
            prev_steer = steer
            steer_angle = abs(steer) * 0.6
            if steer_angle > 0.09:
                steer_radius = 2.9 / math.tan(steer_angle)
                desired_speed = min(
                    desired_speed, float(math.sqrt(7.0 * steer_radius)))

            now = time.time()
            dt = max(1e-3, now - last_ctrl)
            last_ctrl = now
            if desired_speed > target_speed:
                target_speed = min(desired_speed,
                                   target_speed + 2.5 * dt)
            else:
                target_speed = max(desired_speed,
                                   target_speed - 3.5 * dt)
            throttle, brake = speed_ctrl.update(target_speed, speed)
            t = now - session_t0
            if int(t) != last_diag_t and int(t) % 10 == 0:
                last_diag_t = int(t)
                print(f"[route] t={t:.0f}s speed={speed:.1f} m/s "
                      f"pos=({st.pos[0]:.1f}, {st.pos[1]:.1f}) "
                      f"mode={planner.last_mode} blocker={planner.last_blocker} "
                      f"obs={len(obstacles)} obs_dist={obs_dist:.1f} m "
                      f"target={target_speed:.1f} m/s nearest={nearest}")
            hist["t"].append(t)
            hist["throttle"].append(float(throttle))
            hist["brake"].append(float(brake))
            hist["speed"].append(float(speed))

            conn.control(throttle=throttle, steering=steer, brake=brake,
                         gear=fwd_gear)
            goal_dist = float(np.linalg.norm(goal_xy - st.pos[:2]))
            if goal_dist < GOAL_RADIUS_M:
                ended, reason = True, "goal reached"

            if t - last_save >= args.save_every:
                last_save = t
                try:
                    img = conn.grab_screen()
                    img = cv2.resize(img, (CAM_W, CAM_H))
                    img = render_camera_overlay(
                        img, display_route, st.pos, st.heading, cam_model,
                        obstacles=obstacles, lane_markings=last_lanes)
                    bv = np.full((CAM_H, CAM_H, 3), (22, 24, 30), np.uint8)
                    render_birdview(
                        bv, route_xy=display_route, waypoints=[goal_xy.tolist()],
                        goal_xy=goal_xy, pos=st.pos, heading=st.heading,
                        obstacles=obstacles, lane_markings=last_lanes)
                    frame = np.hstack([img, bv])
                    p = out_dir / f"route_probe_t{int(t)}s.png"
                    cv2.imwrite(str(p), frame)
                    print(f"[route] overlay frame -> {p}")
                except Exception as exc:
                    print(f"[route] frame error: {exc}")

            if ended:
                break
            conn.step(1)

        handover_vehicle(conn, saved_mode, True)
        lat = np.asarray(lat_offsets, dtype=float) if lat_offsets \
            else np.zeros(1)
        cap = planner.sharp_corner_kph / 3.6
        print(f"[route] RESULT reason={reason} elapsed={t:.1f}s "
              f"max_speed={max_speed:.1f} m/s")
        print(f"[route] lanes: frames={lane_frames} "
              f"solid_markings={solid_markings} last={len(last_lanes)}")
        print(f"[route] right-side: median_lat={np.median(lat):.2f} m "
              f"mean_lat={lat.mean():.2f} m "
              f"right_fraction={float(np.mean(lat < -0.3)) * 100.0:.1f}%")
        print(f"[route] sharp>45deg: frames={sharp_frames} "
              f"max_angle={sharp_max_angle:.1f} deg "
              f"max_speed_while_sharp={sharp_max_speed:.2f} m/s "
              f"cap={cap:.2f} m/s ({planner.sharp_corner_kph:.0f} km/h)")
        print(f"[route] solid-line blocks: {solid_blocked_frames}")
    finally:
        try:
            handover_vehicle(conn, saved_mode, True)
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
