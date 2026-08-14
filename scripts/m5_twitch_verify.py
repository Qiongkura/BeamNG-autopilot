"""M5 抽搐修复的实机验证：在 21:02 出事现场复现整条控制链路。

把车刷在旧会话记录的位置/航向，加载当时的行驶路线，然后逐帧执行与
m5_autopilot 完全相同的 规划 -> 裁剪 -> 限速 -> 蠕行 -> 踏板 链路，
统计是否还出现“停车/蠕动”振荡：

  * pin_frames:        目标速度被压到 <0.5 m/s 而车还在动 (>1.5 m/s)
  * obslim_zero_frames: 规划器把障碍限速钉死到 0.0 的帧数（修复前特征）
  * stop_creep_cycles:  3 秒内从 <0.5 m/s 弹回 >4 m/s 的次数

三者全为 0 且实际前进超过 30 m 才算 PASS。结束后不退出游戏。
"""

from __future__ import annotations

import argparse
import json
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
    read_gearbox_mode,
    set_gearbox_mode,
)
from beamng_autopilot.control.handover import handover_vehicle
from beamng_autopilot.control.pure_pursuit import PurePursuit
from beamng_autopilot.control.speed import SpeedController
from beamng_autopilot.planner import LocalPlanner, creep_speed
from beamng_autopilot.roadnet import RoadNetwork
from beamng_autopilot.runtime import (
    build_camera_provider,
    build_range_provider,
    resolve_runtime,
)
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
CREEP_MPS = 1.5
SCENE_FILE = Path(__file__).resolve().parent / "_twitch_route.json"


def load_scene() -> dict:
    if SCENE_FILE.exists():
        with open(SCENE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def main() -> None:
    ap = argparse.ArgumentParser(description="M5 twitch-fix in-game verify")
    ap.add_argument("--map", default=config.DEFAULT_MAP)
    ap.add_argument("--vehicle", default=config.DEFAULT_VEHICLE)
    ap.add_argument("--cruise", type=float, default=8.0)
    ap.add_argument("--max-run", type=float, default=150.0)
    ap.add_argument("--runtime", choices=("auto", "steam", "tech"),
                    default=config.RUNTIME_MODE,
                    help="game runtime: auto detects after connecting")
    args = ap.parse_args()

    scene = load_scene()
    spawn_pos = tuple(scene.get("spawn_pos", [0.0, 0.0, 0.0]))
    spawn_heading = float(scene.get("spawn_heading", 0.0))
    recorded_route = scene.get("route")

    conn = BeamNGConnector(
        args.map, args.vehicle, home=config.runtime_home(args.runtime))
    roadnet = RoadNetwork()
    cam_model = default_camera(CAM_W, CAM_H)
    camera_provider = None
    range_provider = None
    pp = PurePursuit(lookahead=6.0)
    planner = LocalPlanner()
    speed_ctrl = SpeedController()

    out_dir = config.LOGS_DIR / "telemetry"
    out_dir.mkdir(parents=True, exist_ok=True)
    hist: dict[str, list] = {"t": [], "throttle": [], "brake": [], "speed": []}
    frames: list[dict] = []
    saved_mode: str | None = None

    try:
        conn.open(launch=True)
        conn.load_scenario(spawn_pos=spawn_pos, spawn_heading=spawn_heading)
        print(f"[verify] spawned at {spawn_pos} heading={spawn_heading:.3f}")
        runtime_mode = resolve_runtime(conn, args.runtime)
        print(f"[verify] runtime={runtime_mode}")
        if runtime_mode == "steam":
            conn.set_front_camera()
        camera_provider, _ = build_camera_provider(
            conn, runtime_mode, CAM_W, CAM_H)
        range_provider, _ = build_range_provider(conn, runtime_mode)

        # Best-effort roadnet for the fallback route (recorded route wins).
        t0 = time.time()
        while not roadnet.ready and time.time() - t0 < 60.0:
            if roadnet.build(conn.bng):
                print(f"[verify] roadnet ready: {roadnet.info}")
                break
            time.sleep(0.5)

        route: np.ndarray | None = None
        if recorded_route and len(recorded_route) >= 4:
            route = np.asarray(recorded_route, dtype=float)
            print(f"[verify] using recorded route: {len(route)} pts")
        elif roadnet.ready:
            st = conn.get_state()
            goal_xy = roadnet.goal_along_route(st.pos[:2], 250.0)
            seg = None if goal_xy is None else roadnet.route(
                st.pos[:2], goal_xy, step=1.5)
            if seg is not None:
                route = np.asarray(seg, dtype=float)
                # Make sure the route leads away from the car's heading.
                d = np.linalg.norm(route[:, :2] - st.pos[:2], axis=1)
                i = int(np.argmin(d))
                if i < len(route) - 2:
                    fwd = route[i + 1, :2] - route[i, :2]
                    if np.dot(fwd, st.dir[:2]) < 0.0:
                        route = route[::-1].copy()
                print(f"[verify] roadnet fallback route: {len(route)} pts")
        if route is None or len(route) < 4:
            raise RuntimeError("no route available for verification")

        saved_mode = read_gearbox_mode(conn.vehicle) or "arcade"
        set_gearbox_mode(conn.vehicle, "realistic")
        conn.step(5)
        print(f"[verify] gearbox: saved={saved_mode} -> realistic")

        session_t0 = time.time()
        nearest = 0
        last_scan = 0.0
        last_save = 0.0
        last_wspd = 0.0
        wheel_speed = None
        obstacles: list = []
        obs_dist = 999.0
        target_speed = 0.0
        prev_steer = 0.0
        last_ctrl = time.time()
        creep_since = 0.0
        ended = False
        reason = "timeout"
        max_speed = 0.0
        last_pos = None
        stationary_since = time.time()

        while time.time() - session_t0 < args.max_run:
            now = time.time()
            st = conn.get_state()
            speed = st.speed
            max_speed = max(max_speed, speed)
            if speed < 0.3:
                if now - stationary_since > 25.0:
                    reason = "stuck"
                    break
            else:
                stationary_since = now

            if now - last_scan > 0.2:
                last_scan = now
                obstacles = range_provider.scan(
                    st.pos, conn.vehicle.vid, radius=55.0).obstacles

            desired_speed = args.cruise
            blocked = False
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
            display_route = drive_route
            desired_speed, obs_dist = planner.speed(
                drive_route, obstacles, st.pos, st.heading, 0, args.cruise)
            if blocked:
                desired_speed = 0.0
            corner_v = getattr(planner, "last_corner", desired_speed)
            obs_lim = getattr(planner, "last_obs_lim", None)

            pp.lookahead = pp.adaptive_lookahead(speed)
            steer_rad, _, _ = pp.steering(
                st.pos, st.heading, drive_route, 0)
            new_steer = float(np.clip(-steer_rad / 0.6, -1.0, 1.0))
            steer = prev_steer + STEER_SMOOTH * (new_steer - prev_steer)
            prev_steer = steer
            steer_angle = abs(steer) * 0.6
            if (steer_angle > 0.09
                    and corner_v < args.cruise * 0.85):
                steer_radius = 2.9 / math.tan(steer_angle)
                capped = float(math.sqrt(7.0 * steer_radius))
                if capped < desired_speed:
                    desired_speed = capped

            desired_speed, creep, creep_since = creep_speed(
                blocked, obs_lim, desired_speed, speed,
                creep_since, now, CREEP_MPS)

            dt = max(1e-3, now - last_ctrl)
            last_ctrl = now
            if desired_speed > target_speed:
                target_speed = min(desired_speed,
                                   target_speed + RAMP_ACCEL * dt)
            else:
                target_speed = max(desired_speed,
                                   target_speed - RAMP_DECEL * dt)

            if now - last_wspd > 0.1:
                last_wspd = now
                wheel_speed = conn.get_wheel_speed()
            throttle, brake = speed_ctrl.update(
                target_speed, speed, dt=min(0.05, dt),
                wheel_speed=wheel_speed)

            t = now - session_t0
            hist["t"].append(t)
            hist["throttle"].append(float(throttle))
            hist["brake"].append(float(brake))
            hist["speed"].append(float(speed))
            frames.append({
                "t": round(t, 3),
                "speed": round(float(speed), 3),
                "desired": round(float(desired_speed), 3),
                "obslim": None if obs_lim is None else round(float(obs_lim), 3),
                "blocked": bool(blocked),
                "creep": bool(creep),
                "n_obs": len(obstacles),
                "throttle": round(float(throttle), 3),
                "brake": round(float(brake), 3),
                "pos": [round(float(v), 2) for v in st.pos[:2]],
            })

            conn.control(throttle=throttle, steering=steer, brake=brake)

            goal_dist = float(np.linalg.norm(route[-1][:2] - st.pos[:2]))
            if goal_dist < GOAL_RADIUS_M:
                ended, reason = True, "goal reached"
                break

            if t - last_save >= 10.0:
                last_save = t
                try:
                    img = camera_provider.grab()
                    img = cv2.resize(img, (CAM_W, CAM_H))
                    cam_model = camera_provider.camera_model(
                        st.pos, st.heading, CAM_W, CAM_H,
                        fallback=cam_model)
                    if len(display_route) >= 2:
                        img = render_camera_overlay(
                            img, display_route, st.pos, st.heading, cam_model)
                    bv = np.full((CAM_H, CAM_H, 3), (22, 24, 30), np.uint8)
                    render_birdview(
                        bv, route_xy=display_route, pos=st.pos,
                        heading=st.heading)
                    frame = np.hstack([img, bv])
                    p = out_dir / f"twitch_verify_t{int(t)}s.png"
                    cv2.imwrite(str(p), frame)
                    print(f"[verify] frame -> {p}")
                except Exception as exc:
                    print(f"[verify] frame error: {exc}")

        # ---- metrics --------------------------------------------------
        speeds = np.array([f["speed"] for f in frames])
        desired = np.array([f["desired"] for f in frames])
        ts = np.array([f["t"] for f in frames])
        n = len(frames)
        pin_frames = int(np.sum((desired < 0.5) & (speeds > 1.5)))
        obslim_zero = int(np.sum(
            [1 for f in frames if f["obslim"] is not None
             and f["obslim"] < 0.01]))
        below_floor = int(np.sum(
            [1 for f in frames if f["obslim"] is not None
             and 0.0 < f["obslim"] < 2.0]))

        cycles = 0
        i = 0
        while i < n:
            if speeds[i] < 0.5:
                t0 = ts[i]
                j = i
                while j < n and ts[j] - t0 <= 3.0:
                    if speeds[j] > 4.0:
                        cycles += 1
                        break
                    j += 1
                while i < n and speeds[i] < 0.5:
                    i += 1
            else:
                i += 1

        dist = 0.0
        last_xy = None
        for f in frames:
            xy = np.array(f["pos"])
            if last_xy is not None:
                dist += float(np.linalg.norm(xy - last_xy))
            last_xy = xy
        stationary = float(np.sum(speeds < 0.5))
        avg_speed = float(np.mean(speeds)) if n else 0.0
        p5 = float(np.percentile(speeds, 5)) if n else 0.0
        p50 = float(np.percentile(speeds, 50)) if n else 0.0
        blocked_frames = int(np.sum([1 for f in frames if f["blocked"]]))

        ok = (pin_frames == 0 and obslim_zero == 0 and cycles == 0
              and dist >= 30.0 and reason != "stuck")
        verdict = "PASS" if ok else "FAIL"

        report = {
            "verdict": verdict,
            "reason": reason,
            "frames": n,
            "pin_frames": pin_frames,
            "obslim_zero_frames": obslim_zero,
            "obslim_below_floor": below_floor,
            "stop_creep_cycles": cycles,
            "dist_m": round(dist, 1),
            "avg_speed": round(avg_speed, 2),
            "p5_speed": round(p5, 2),
            "p50_speed": round(p50, 2),
            "max_speed": round(max_speed, 2),
            "stationary_s": round(stationary, 1),
            "blocked_frames": blocked_frames,
            "spawn_pos": list(spawn_pos),
            "spawn_heading": spawn_heading,
            "cruise": args.cruise,
        }
        stamp = time.strftime("%Y%m%d_%H%M%S")
        report_path = out_dir / f"twitch_verify_{stamp}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        chart_path = out_dir / f"twitch_verify_{stamp}.png"
        if n >= 2:
            plot_telemetry(hist, chart_path, block=False, show=False)
        print(f"[verify] report -> {report_path}")
        print(f"[verify] chart  -> {chart_path}")
        print("[verify] RESULT:", json.dumps(report, ensure_ascii=False))

        # Restore control to the driver without latching reverse.
        handover_vehicle(conn, saved_mode, True)
        print(f"[verify] VERDICT: {verdict}  "
              f"(pins={pin_frames} obslim0={obslim_zero} "
              f"cycles={cycles} dist={dist:.1f}m)")
    except Exception as exc:
        print(f"[verify] FAILED: {exc}")
        raise
    finally:
        try:
            if saved_mode:
                handover_vehicle(conn, saved_mode, True)
        except Exception:
            pass
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
            conn.close()  # disconnect only; the game stays open
        except Exception:
            pass


if __name__ == "__main__":
    main()
