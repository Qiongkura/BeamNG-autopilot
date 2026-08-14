"""M1-循迹：Pure Pursuit + PID 让车沿闭环轨迹自动跑圈。"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beamng_autopilot import config
from beamng_autopilot.connector import BeamNGConnector
from beamng_autopilot.control.pid import PID
from beamng_autopilot.control.pure_pursuit import PurePursuit
from beamng_autopilot.telemetry import TelemetryBroadcaster
from beamng_autopilot.hud import LiveHUD
from beamng_autopilot.track import (
    generate_rounded_rectangle,
    heading_from_path,
    load_track,
)


def corner_speed(points, nearest, base_speed, a_lat=4.5, back_m=8.0, ahead_m=18.0):
    """Limit speed by track curvature around the car (m/s)."""
    pts = points[:, :2]
    n = len(pts)

    def walk(start, step, limit_m):
        i = start
        traveled = 0.0
        segs = []
        while traveled < limit_m:
            j = (i + step) % n
            seg = pts[j] - pts[i]
            seg_len = float(np.linalg.norm(seg))
            if seg_len < 1e-9:
                i = j
                continue
            segs.append(seg)
            traveled += seg_len
            i = j
        return segs

    segs = [-s for s in reversed(walk(nearest, -1, back_m))] + walk(nearest, 1, ahead_m)
    total_len = sum(float(np.linalg.norm(s)) for s in segs)
    ang = np.unwrap(np.arctan2([s[1] for s in segs], [s[0] for s in segs]))
    total_da = abs(float(ang[-1] - ang[0]))
    if total_da < 1e-6:
        return base_speed
    radius = total_len / total_da
    return min(base_speed, float(np.sqrt(a_lat * radius)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=config.DEFAULT_MAP)
    ap.add_argument("--vehicle", default=config.DEFAULT_VEHICLE)
    ap.add_argument("--track", default=None, help="录制的 npz 轨迹；缺省使用解析闭环")
    ap.add_argument("--speed", type=float, default=12.0, help="目标巡航速度 m/s")
    ap.add_argument("--laps", type=int, default=2)
    ap.add_argument("--port", type=int, default=config.PORT)
    ap.add_argument("--no-hud", action="store_true",
                    help="disable the live telemetry HUD window")
    ap.add_argument("--duration", type=float, default=600.0, help="最大运行秒数")
    ap.add_argument("--runtime", choices=("auto", "steam", "tech"),
                    default=config.RUNTIME_MODE,
                    help="game runtime: auto prefers BeamNG.tech when installed")
    args = ap.parse_args()

    if args.track:
        points, headings = load_track(args.track)
        print(f"[follow] 使用录制轨迹: {args.track} ({len(points)} 点)")
    else:
        points = generate_rounded_rectangle()
        headings = heading_from_path(points)
        print(f"[follow] 使用解析闭环轨迹 ({len(points)} 点)")

    start = points[0]
    heading0 = float(headings[0])
    n = len(points)
    lap_len = float(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1).sum())

    pp = PurePursuit(lookahead=6.0)
    pid_throttle = PID(kp=0.55, ki=0.15, kd=0.06, output_limits=(0.0, 1.0))

    telemetry = []
    live_telemetry = TelemetryBroadcaster()
    hud = None if args.no_hud else LiveHUD(show_camera=False)
    with BeamNGConnector(
            args.map, args.vehicle, port=args.port,
            home=config.runtime_home(args.runtime)) as conn:
        conn.load_scenario(spawn_pos=(float(start[0]), float(start[1]), 0.0), spawn_heading=heading0)
        print(f"[follow] 场景已开始，环路 {lap_len:.0f} m，目标 {args.speed} m/s，跑 {args.laps} 圈")

        nearest, lap, t0 = 0, 0, time.time()
        last_status = 0.0
        stalled_since = None
        completed = False

        while time.time() - t0 < args.duration and lap < args.laps:
            st = conn.get_state()
            speed = st.speed

            # 预瞄距离随速度自适应
            pp.lookahead = pp.adaptive_lookahead(speed)
            steer_rad, target, nearest = pp.steering(st.pos, st.heading, points, nearest)
            # 转向角(rad) -> 转向输入（满舵约 0.6 rad）
            # Pure Pursuit 的 steer 约定正值为左转，BeamNG 正值为右转，取反
            steer = float(np.clip(-steer_rad / 0.6, -1.0, 1.0))

            target_speed = corner_speed(points, nearest, args.speed)
            steer_angle = abs(steer) * 0.6
            if steer_angle > 0.05:
                steer_radius = 2.9 / math.tan(steer_angle)
                target_speed = min(target_speed, float(np.sqrt(4.5 * steer_radius)))
            target_speed = max(target_speed, 2.5)
            throttle = pid_throttle.update(target_speed - speed)
            brake = 0.0
            if speed > target_speed + 2.0:
                brake = min(0.7, (speed - target_speed) / 8.0)
                throttle = 0.0
            if speed < 0.5:
                throttle = max(throttle, 0.6)  # 起步/低速补油

            live_telemetry.publish(
                t=time.time() - t0,
                speed=float(speed),
                throttle=throttle,
                brake=brake,
                steer=steer,
                vel=st.vel,
                dir_vec=st.dir,
                up_vec=st.up,
                pos=st.pos,
                heading=float(st.heading),
                lap=lap,
                nearest=int(nearest),
            )

            if hud is not None and not hud.update(live_telemetry.latest):
                print("[follow] HUD closed, stopping")
                break

            conn.control(throttle=throttle, steering=steer, brake=brake)
            conn.step(1)

            # 圈数检测：最近点索引从高跳到低 => 过了一圈
            if len(telemetry) > 0:
                prev_nearest = telemetry[-1][8]
                if prev_nearest > 0.7 * n and nearest < 0.3 * n:
                    lap += 1
                    print(f"[follow] 完成第 {lap} 圈！t={time.time() - t0:.1f}s")

            telemetry.append(
                [
                    time.time() - t0, st.pos[0], st.pos[1], st.heading, speed,
                    throttle, steer, brake, nearest, lap,
                ]
            )

            if time.time() - last_status > 2.0:
                last_status = time.time()
                prog = nearest / n * 100
                print(
                    f"[follow] t={time.time() - t0:5.1f}s 圈={lap} 进度={prog:5.1f}% "
                    f"speed={speed:5.2f} 目标={target_speed:5.2f} 转向={steer:+.2f} 油门={throttle:.2f} 刹车={brake:.2f}"
                )

            # 卡住检测：5 秒内速度一直 < 0.5
            if speed < 0.5:
                stalled_since = stalled_since if stalled_since is not None else time.time()
                if time.time() - stalled_since > 5.0:
                    print("[follow] 车辆似乎卡住，加大油门尝试脱困")
                    conn.control(throttle=0.9, steering=steer, brake=0.0)
                    conn.step(30)
                    stalled_since = None
            else:
                stalled_since = None

        # 结束：刹车
        conn.control(throttle=0.0, brake=1.0)
        conn.step(30)
        completed = lap >= args.laps
        print(f"[follow] {'完成目标圈数' if completed else '超时结束'}，共跑 {lap} 圈")

    # 保存遥测
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = config.LOGS_DIR / f"m1_follow_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "x", "y", "heading", "speed", "throttle", "steer", "brake", "nearest", "lap"])
        writer.writerows(telemetry)
    live_telemetry.close()
    if hud is not None:
        hud.close()
    print(f"[follow] 遥测已保存: {csv_path}")


if __name__ == "__main__":
    main()
