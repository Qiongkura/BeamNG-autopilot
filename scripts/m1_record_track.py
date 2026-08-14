"""M1-录制：让游戏内 AI 沿闭环跑，录制真实行驶轨迹到 data/track_<map>.npz。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beamng_autopilot import config
from beamng_autopilot.connector import BeamNGConnector
from beamng_autopilot.track import (
    generate_rounded_rectangle,
    heading_from_path,
    clean_closed_lap,
    save_track,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=config.DEFAULT_MAP)
    ap.add_argument("--vehicle", default=config.DEFAULT_VEHICLE)
    ap.add_argument("--speed", type=float, default=12.0, help="AI 巡航速度 m/s")
    ap.add_argument("--laps", type=float, default=2.2, help="录制圈数")
    ap.add_argument("--port", type=int, default=config.PORT)
    ap.add_argument("--runtime", choices=("auto", "steam", "tech"),
                    default=config.RUNTIME_MODE,
                    help="game runtime: auto prefers BeamNG.tech when installed")
    args = ap.parse_args()

    points = generate_rounded_rectangle()
    start = points[0]
    heading0 = float(heading_from_path(points)[0])

    with BeamNGConnector(
            args.map, args.vehicle, port=args.port,
            home=config.runtime_home(args.runtime)) as conn:
        conn.load_scenario(spawn_pos=(float(start[0]), float(start[1]), 0.0), spawn_heading=heading0)
        conn.ai_set_line(points, speed=args.speed)

        # 估算录制时长：轨迹周长 / 车速 * 圈数 + 起步缓冲
        seg = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
        lap_len = float(seg.sum())
        duration = lap_len / args.speed * args.laps + 10.0
        print(f"[record] 环路长度 {lap_len:.0f} m，预计录制 {duration:.0f} s")

        poses, headings, speeds, t0 = [], [], [], time.time()
        last_print = time.time()
        while time.time() - t0 < duration:
            st = conn.get_state()
            poses.append(st.pos)
            headings.append(st.heading)
            speeds.append(st.speed)
            conn.step(3)  # 20 Hz 采样
            if time.time() - last_print > 5.0:
                last_print = time.time()
                print(f"[record] t={time.time() - t0:5.1f}s  speed={st.speed:5.2f} m/s  记录点数={len(poses)}")

        conn.ai_disable()
        conn.control(throttle=0.0, brake=1.0)
        conn.step(30)

    # 去掉结尾停车阶段的重复点（AI 结束时减速到 0）
    if len(speeds) > 20:
        i = len(speeds) - 1
        while i > 0 and speeds[i] < 1.0:
            i -= 1
        poses = poses[: i + 2]

    track = np.column_stack([np.array([p[:2] for p in poses]), np.zeros(len(poses))])
    track = clean_closed_lap(track, step=1.0)
    save_track(track, config.DATA_DIR / f"track_{args.map}.npz")


if __name__ == "__main__":
    main()
