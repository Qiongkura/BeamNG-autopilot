"""Obstacle-avoidance proof test against the real game (no keyboard needed).

Loads a small scenario with the ego car AND a blocker vehicle parked in the
middle of the generated route ~35 m ahead.  Then drives the exact autopilot
loop used by m5_autopilot.py (raycast + scenario-object perception -> local
planner -> ramped throttle) and records whether the car actually steers
around the blocker instead of blindly charging into it.

Outputs:
    * console log: sensor status, obstacle counts, planner mode transitions
    * overlay frames (front-camera projection + bird view) proving the detour
    * telemetry chart (throttle / brake / speed) proving linear ramps

Usage:
    .venv\\Scripts\\python.exe scripts\\m5_obstacle_test.py
    .venv\\Scripts\\python.exe scripts\\m5_obstacle_test.py --runtime tech
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from beamngpy import Scenario, Vehicle
from beamngpy.misc.quat import angle_to_quat

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beamng_autopilot import config
from beamng_autopilot.connector import BeamNGConnector
from beamng_autopilot.control.gearbox import (
    read_gearbox_mode,
    set_gearbox_mode,
)
from beamng_autopilot.control.handover import handover_vehicle
from beamng_autopilot.control.pure_pursuit import PurePursuit
from beamng_autopilot.control.speed import SpeedController
from beamng_autopilot.perception import (
    errors_active,
    errors_summary,
    scan_obstacles,
    scan_obstacles_raycast,
    scan_obstacles_vehicles,
)
from beamng_autopilot.planner import LocalPlanner
from beamng_autopilot.roadnet import RoadNetwork
from beamng_autopilot.runtime import (
    build_camera_provider,
    build_range_provider,
    resolve_runtime,
)
from beamng_autopilot.telemetry import TelemetryBroadcaster
from beamng_autopilot.telemetry_chart import plot_telemetry
from beamng_autopilot.vision.projection import default_camera
from beamng_autopilot.visionview import (
    render_birdview,
    render_camera_overlay,
)

CAM_W, CAM_H = 1076, 806
GOAL_RADIUS_M = 8.0
RAMP_ACCEL = 2.5
RAMP_DECEL = 3.5
STEER_SMOOTH = 0.35
CRUISE = 9.0          # m/s - modest cruise so the test is safe on smallgrid
GOAL_DIST = 140.0     # route length in m
BLOCK_DIST = 35.0     # blocker placed this far along the route


def _yaw_deg_from_heading(heading: float) -> float:
    # Same world-heading -> BeamNG yaw convention as connector.load_scenario.
    return math.degrees(float(heading)) + 90.0


def main() -> None:
    ap = argparse.ArgumentParser(description="M5 obstacle-avoidance test")
    ap.add_argument("--runtime", choices=("auto", "steam", "tech"),
                    default=config.RUNTIME_MODE,
                    help="game runtime: auto detects after connecting")
    args = ap.parse_args()

    conn = BeamNGConnector(
        "smallgrid", "etk800", home=config.runtime_home(args.runtime))
    roadnet = RoadNetwork()
    telemetry = TelemetryBroadcaster()
    cam_model = default_camera(CAM_W, CAM_H)
    camera_provider = None
    range_provider = None
    pp = PurePursuit(lookahead=6.0)
    planner = LocalPlanner()
    speed_ctrl = SpeedController()

    out_dir = config.LOGS_DIR / "telemetry"
    out_dir.mkdir(parents=True, exist_ok=True)
    hist: dict[str, list] = {"t": [], "throttle": [], "brake": [], "speed": []}
    modes: list[tuple[float, str]] = []
    saved_mode: str | None = None

    try:
        conn.open(launch=True)
        # Own scenario with TWO vehicles: ego + a blocker parked in the lane.
        scenario = Scenario("smallgrid", "autopilot_obstacle")
        ego = Vehicle("ego", model="etk800", color="Red")
        blocker = Vehicle("blocker", model="etk800", color="Blue")
        quat = angle_to_quat((0.0, 0.0, 90.0))
        scenario.add_vehicle(ego, pos=(0.0, 0.0, 0.0), rot_quat=quat,
                             cling=True)
        scenario.add_vehicle(blocker, pos=(8.0, 0.0, 0.0), rot_quat=quat,
                             cling=True)
        scenario.make(conn.bng)
        conn.bng.scenario.load(scenario)
        conn.bng.scenario.start()
        conn.scenario = scenario
        conn.vehicle = ego
        print("[obstacle] scenario loaded")
        runtime_mode = resolve_runtime(conn, args.runtime)
        print(f"[obstacle] runtime={runtime_mode}")
        if runtime_mode == "steam":
            conn.set_front_camera()
        camera_provider, _ = build_camera_provider(
            conn, runtime_mode, CAM_W, CAM_H)
        range_provider, _ = build_range_provider(conn, runtime_mode)

        # Mirror autopilot start: remember the player's gearbox mode and
        # force realistic so braking at a standstill can never latch R.
        saved_mode = read_gearbox_mode(conn.vehicle) or "arcade"
        set_gearbox_mode(conn.vehicle, "realistic")
        conn.step(5)
        print(f"[obstacle] gearbox: saved={saved_mode} -> realistic")

        t0 = time.time()
        while not roadnet.ready and time.time() - t0 < 90.0:
            if roadnet.build(conn.bng):
                print(f"[obstacle] roadnet ready: {roadnet.info}")
                break
            time.sleep(1.0)
        if conn.reposition_on_road(roadnet):
            print("[obstacle] ego placed on road network")

        st = conn.get_state()
        start_xy = st.pos[:2].copy()
        if roadnet.ready:
            goal_xy = roadnet.goal_along_route(start_xy, GOAL_DIST)
        else:
            goal_xy = None
        if goal_xy is None:
            # No DecalRoad data (smallgrid etc.): drive straight along the
            # vehicle heading - the planner still has to dodge the blocker.
            heading = float(st.heading)
            goal_xy = start_xy + GOAL_DIST * np.array(
                [math.cos(heading), math.sin(heading)])
            print("[obstacle] no road data - straight-line route along "
                  "vehicle heading")
        goal_xy = np.asarray(goal_xy, dtype=float)
        if roadnet.ready:
            seg = roadnet.route(start_xy, goal_xy, step=1.5)
        else:
            seg = None
        if seg is None:
            seg = RoadNetwork._interpolate(start_xy, goal_xy, 1.5)
        route = np.asarray(seg, dtype=float)
        print(f"[obstacle] route: {len(route)} pts, goal {goal_xy.round(1)}")

        # ---- park the blocker in the lane 35 m along the route ----
        d = np.linalg.norm(route[:, :2] - start_xy, axis=1)
        nearest = int(np.argmin(d))
        cum = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(route[:, :2], axis=0),
                                             axis=1))])
        target = None
        for i in range(nearest, len(route) - 1):
            if cum[i] - cum[nearest] >= BLOCK_DIST:
                target = i
                break
        if target is None:
            raise RuntimeError("route too short to place the blocker")
        bp = route[target]
        fwd = route[min(target + 3, len(route) - 1)] - bp
        if np.linalg.norm(fwd) < 1e-6:
            fwd = route[target] - route[max(target - 3, 0)]
        heading_at = float(np.arctan2(fwd[1], fwd[0]))

        blocker = conn.scenario.get_vehicle("blocker")
        z = 0.4
        if roadnet.heights is not None:
            hi = int(np.argmin(np.linalg.norm(roadnet.nodes - bp, axis=1)))
            z = float(roadnet.heights[hi]) + 0.4
        blocker.teleport(
            (float(bp[0]), float(bp[1]), z),
            rot_quat=angle_to_quat((0.0, 0.0, _yaw_deg_from_heading(heading_at))))
        conn.step(30)
        print(f"[obstacle] blocker parked at ({bp[0]:.1f}, {bp[1]:.1f}) "
              f"heading {math.degrees(heading_at):.0f}deg, ~{BLOCK_DIST:.0f}m "
              "along the route")

        # ---- verify the perception actually sees the blocker ----
        st = conn.get_state()
        scen = scan_obstacles(conn.bng, conn.vehicle.vid, st.pos, radius=55.0)
        rays = scan_obstacles_raycast(conn.bng, st.pos, radius=55.0)
        vehs = scan_obstacles_vehicles(conn.bng, conn.vehicle.vid, st.pos,
                                       radius=55.0)
        print(f"[obstacle] perception check: scenario_objects={len(scen)} "
              f"vehicles={len(vehs)} raycast={len(rays)} "
              f"(error={errors_summary()!r})")
        for o in scen + vehs + rays:
            print(f"[obstacle]   obstacle at ({o.x:.1f}, {o.y:.1f}) "
                  f"half=({o.half_w:.1f}, {o.half_h:.1f}) cat={o.category}")

        # ---- drive ----
        session_t0 = time.time()
        nearest = 0
        last_save = 0.0
        last_scan = 0.0
        last_status = 0.0
        obstacles: list = []
        obs_dist = 999.0
        blocked = False
        target_speed = 0.0
        prev_steer = 0.0
        last_ctrl = time.time()
        max_dev = 0.0
        min_speed_near_blocker = 999.0
        passed_blocker = False
        stuck_t0 = time.time()
        last_mode = None
        ended = False
        reason = "timeout"
        max_speed = 0.0

        while time.time() - session_t0 < 90.0:
            st = conn.get_state()
            speed = float(st.speed)
            max_speed = max(max_speed, speed)
            now = time.time()

            if now - last_scan > 0.2:
                last_scan = now
                sample = range_provider.scan(
                    st.pos, conn.vehicle.vid, radius=55.0)
                obstacles = sample.obstacles
                if errors_active():
                    print(f"[obstacle] sensor warning: {errors_summary()}")

            desired_speed = CRUISE
            blocked = False
            drive_route = route
            if len(route) > 0:
                d = np.linalg.norm(route[:, :2] - st.pos[:2], axis=1)
                nearest = int(np.argmin(d))
                drive_route, blocked = planner.plan(
                    route, obstacles, st.pos, st.heading, nearest)
                drive_route = np.asarray(drive_route, dtype=float)
                if len(drive_route) >= 2:
                    d0 = np.linalg.norm(
                        drive_route[:, :2] - st.pos[:2], axis=1)
                    start_i = int(np.argmin(d0))
                    if start_i > 0 and len(drive_route) - start_i >= 2:
                        drive_route = drive_route[start_i:]
                desired_speed, obs_dist = planner.speed(
                    drive_route, obstacles, st.pos, st.heading,
                    0, CRUISE)
                if blocked:
                    desired_speed = 0.0
                pp.lookahead = pp.adaptive_lookahead(speed)
                steer_rad, _, _ = pp.steering(
                    st.pos, st.heading, drive_route, 0)
                new_steer = float(np.clip(-steer_rad / 0.6, -1.0, 1.0))
                steer = prev_steer + STEER_SMOOTH * (new_steer - prev_steer)
                prev_steer = steer

                # Lateral deviation of the car vs the nav route.
                lat = 0.0
                for k in range(len(route) - 1):
                    ax, ay = route[k, :2]
                    bx, by = route[k + 1, :2]
                    abx, aby = bx - ax, by - ay
                    l2 = abx * abx + aby * aby
                    if l2 < 1e-9:
                        continue
                    t = max(0.0, min(1.0, (
                        (st.pos[0] - ax) * abx + (st.pos[1] - ay) * aby) / l2))
                    px, py = ax + t * abx, ay + t * aby
                    lat = min(lat, abs(((st.pos[0] - px) * aby
                                        - (st.pos[1] - py) * abx)
                                       / math.sqrt(l2)))
                max_dev = max(max_dev, lat)
                dist_to_bp = float(np.linalg.norm(st.pos[:2] - bp))
                if dist_to_bp < 6.0:
                    min_speed_near_blocker = min(min_speed_near_blocker, speed)
                if dist_to_bp < 2.5:
                    passed_blocker = True

            mode = getattr(planner, "last_mode", "follow")
            if mode != last_mode:
                last_mode = mode
                modes.append((round(time.time() - session_t0, 1), mode))
                print(f"[obstacle] MODE -> {mode} at t={modes[-1][0]:.1f}s "
                      f"v={speed:.1f} obs={len(obstacles)}")

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
                extra={"mode": "OBSTACLE_TEST", "obs": len(obstacles),
                       "obs_d": round(float(obs_dist), 1)},
            )
            if blocked and speed < 0.5:
                brake = max(brake, 0.12)
            conn.control(throttle=throttle, steering=steer, brake=brake)

            if now - last_status > 1.0:
                last_status = now
                print(f"[obstacle] mode={mode} obs={len(obstacles)} "
                      f"nearest={obs_dist:.0f}m v={speed:.1f} "
                      f"target={target_speed:.1f} thr={throttle:.2f} "
                      f"brk={brake:.2f} dev={lat:.2f}")

            if now - last_save >= 4.0:
                last_save = now
                try:
                    img = camera_provider.grab()
                    img = cv2.resize(img, (CAM_W, CAM_H))
                    cam_model = camera_provider.camera_model(
                        st.pos, st.heading, CAM_W, CAM_H,
                        fallback=cam_model)
                    if len(drive_route) >= 2:
                        img = render_camera_overlay(
                            img, drive_route, st.pos, st.heading, cam_model,
                            obstacles=obstacles)
                    bv = np.full((CAM_H, CAM_H, 3), (22, 24, 30), np.uint8)
                    render_birdview(
                        bv, route_xy=drive_route, obstacles=obstacles,
                        goal_xy=goal_xy, pos=st.pos, heading=st.heading)
                    frame = np.hstack([img, bv])
                    p = out_dir / f"m5_obstacle_t{int(t)}s_mode{mode}.png"
                    cv2.imwrite(str(p), frame)
                except Exception as exc:
                    print(f"[obstacle] frame error: {exc}")

            if mode == "blocked":
                if t > 8.0:
                    ended, reason = True, "blocked-stop confirmed"
                    break
            elif speed < 0.35 and t > 12.0:
                if now - stuck_t0 > 10.0:
                    ended, reason = True, "car stuck (aborted)"
                    break
            else:
                stuck_t0 = now

            goal_dist = float(np.linalg.norm(route[-1][:2] - st.pos[:2]))
            if goal_dist < GOAL_RADIUS_M:
                ended, reason = True, "goal reached"
            if ended:
                break
            conn.step(1)

        # ---- finish ----
        handover_vehicle(conn, saved_mode, True)

        chart_path = out_dir / (
            f"m5_obstacle_chart_{time.strftime('%Y%m%d_%H%M%S')}.png")
        plot_telemetry(hist, chart_path, block=False, show=False)
        print(f"[obstacle] telemetry chart -> {chart_path}")

        final = conn.get_state().pos[:2]
        dist = float(np.linalg.norm(final - start_xy))
        print(f"[obstacle] RESULT reason={reason} elapsed={t:.1f}s "
              f"dist={dist:.1f}m max_speed={max_speed:.1f}")
        print(f"[obstacle] max lateral deviation vs nav route: {max_dev:.2f}m")
        print(f"[obstacle] min speed within 6m of blocker: "
              f"{min_speed_near_blocker if min_speed_near_blocker < 999 else float('nan'):.1f}m/s")
        print(f"[obstacle] passed within 2.5m of blocker: {passed_blocker}")
        print(f"[obstacle] mode timeline: {modes}")

    except Exception as exc:
        print(f"[obstacle] FAILED: {exc}")
        import traceback
        traceback.print_exc()
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
