"""Passive map frame dumper: you drive, this records front-camera frames.

No routes, no lane placement, no autopilot - the car is driven manually
(WASD in the game window) while this script dumps front_main camera frames
to PNG for the grid-labeling loop.  The most robust way to collect frames
on maps the FSD stack cannot yet drive.

Usage::
    .venv\\Scripts\\python.exe scripts\\m5_map_frame_dump.py \\
        --runtime tech --map east_coast_usa --attach
    .venv\\Scripts\\python.exe scripts\\m5_map_frame_dump.py \\
        --runtime tech --map utah          # launch a fresh game into the map
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2

from beamng_autopilot import config


def main() -> int:
    ap = argparse.ArgumentParser(description="被动采帧：人工驾驶，脚本存图")
    ap.add_argument("--runtime", choices=("auto", "steam", "tech"),
                    default="tech")
    ap.add_argument("--map", type=str, default=None,
                    help="地图（--launch 时必填；--attach 留空=沿用当前关卡）")
    ap.add_argument("--attach", action="store_true",
                    help="接管正在运行的游戏实例")
    ap.add_argument("--vehicle", type=str, default="etk800")
    ap.add_argument("--fps", type=float, default=2.0,
                    help="采帧频率（默认 2 帧/秒）")
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="采集时长（0 = 一直采到 Ctrl+C）")
    ap.add_argument("--out", type=str, default=None,
                    help="输出目录（默认 logs/labeling/mapdump_<map>）")
    args = ap.parse_args()

    from beamng_autopilot.connector import BeamNGConnector
    from beamng_autopilot.runtime import build_camera_ring_provider

    the_map = args.map or "italy"
    out_dir = (Path(args.out) if args.out
               else config.LOGS_DIR / "labeling" / f"mapdump_{the_map}")
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = BeamNGConnector(the_map, args.vehicle,
                           port=config.runtime_port(args.runtime),
                           home=config.runtime_home(args.runtime))
    print(f"[dump] connecting ({'attach' if args.attach else 'launch'}) "
          f"map={the_map} -> {out_dir}")
    conn.open(launch=not args.attach)
    try:
        conn.attach_vehicle(already_open=True)
    except Exception:
        conn.load_scenario()
    try:
        # 之前 FSD 驱动装的游戏侧输入看门狗会在进程死后锁死车辆（刹停），
        # 被动采集必须解除并把控制权还给玩家键盘
        from beamng_autopilot.watchdog import disarm as wd_disarm
        wd_disarm(conn)
        conn.control(throttle=0.0, brake=0.0, steering=0.0, parkingbrake=0.0)
        print("[dump] watchdog disarmed, controls released to player")
    except Exception as exc:
        print(f"[dump] watchdog disarm skipped: {exc}")
    try:
        # 新地图默认出生点可能在野地：把车吸附到最近的道路节点上
        import math
        from beamng_autopilot.roadnet import RoadNetwork
        net = RoadNetwork()
        if net.build(conn.bng) and net.node_count:
            st = conn.get_state()
            xyz = net.nearest_node_xyz((float(st.pos[0]), float(st.pos[1])))
            if xyz is not None:
                h = net.road_heading_at((float(st.pos[0]), float(st.pos[1])))
                conn.safe_teleport(float(xyz[0]), float(xyz[1]),
                                   heading_deg=math.degrees(float(h)))
                print(f"[dump] snapped to road ({float(xyz[0]):.1f}, "
                      f"{float(xyz[1]):.1f})")
    except Exception as exc:
        print(f"[dump] road snap skipped: {exc}")
    ring, _ = build_camera_ring_provider(conn, args.runtime, 536, 403,
                                         roles=("front_main",))
    interval = 1.0 / max(0.1, args.fps)
    t_end = (time.time() + args.seconds) if args.seconds > 0 else None
    n = 0
    t_last = 0.0
    print(f"[dump] recording at {args.fps} fps - 切到游戏窗口开车，"
          f"Ctrl+C 结束")
    try:
        while t_end is None or time.time() < t_end:
            snap = ring.grab_ring()
            role = "front_main" if "front_main" in snap else next(iter(snap))
            now = time.time()
            if now - t_last >= interval:
                rgb = snap[role][0]
                fp = out_dir / f"frame_{n:05d}.png"
                cv2.imwrite(str(fp), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                n += 1
                t_last = now
                if n % 50 == 0:
                    print(f"[dump] {n} frames", flush=True)
            conn.step(5)
    except KeyboardInterrupt:
        pass
    finally:
        print(f"[dump] saved {n} frames -> {out_dir}")
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
