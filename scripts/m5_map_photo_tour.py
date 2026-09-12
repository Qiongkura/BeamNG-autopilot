"""Teleport photo tour: collect road frames without driving.

The road graph is a ready-made list of camera positions.  This script
teleports the ego to sampled road nodes across the map, settles, and
captures a front frame at each stop - no driving, no lane placement,
works on any map regardless of FSD-stack competence.

Usage::
    .venv\\Scripts\\python.exe scripts\\m5_map_photo_tour.py \\
        --runtime tech --map east_coast_usa --stops 250
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2

from beamng_autopilot import config


def main() -> int:
    ap = argparse.ArgumentParser(description="传送观光采帧：无需驾驶")
    ap.add_argument("--runtime", choices=("auto", "steam", "tech"),
                    default="tech")
    ap.add_argument("--map", type=str, required=True)
    ap.add_argument("--vehicle", type=str, default="etk800")
    ap.add_argument("--stops", type=int, default=250)
    ap.add_argument("--min-dist", type=float, default=120.0,
                    help="相邻机位最小间距（米），避免重复取景")
    ap.add_argument("--settle-steps", type=int, default=15)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--launch", action="store_true",
                    help="新开游戏实例（默认 attach 现有实例）")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    from beamng_autopilot.connector import BeamNGConnector
    from beamng_autopilot.roadnet import RoadNetwork
    from beamng_autopilot.runtime import build_camera_ring_provider
    from beamng_autopilot.watchdog import disarm as wd_disarm

    out_dir = (Path(args.out) if args.out
               else config.LOGS_DIR / "labeling" / f"tour_{args.map}")
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = BeamNGConnector(args.map, args.vehicle,
                           port=config.runtime_port(args.runtime),
                           home=config.runtime_home(args.runtime))
    print(f"[tour] connecting map={args.map} -> {out_dir}", flush=True)
    conn.open(launch=args.launch)
    try:
        conn.attach_vehicle(already_open=True)
    except Exception:
        conn.load_scenario()
    try:
        wd_disarm(conn)
        conn.control(throttle=0.0, brake=0.0, steering=0.0, parkingbrake=0.0)
    except Exception as exc:
        print(f"[tour] watchdog disarm skipped: {exc}", flush=True)

    net = RoadNetwork()
    if not net.build(conn.bng) or net.node_count == 0:
        print("[tour] roadnet build failed")
        conn.close()
        return 1
    print(f"[tour] roadnet {net.node_count} nodes", flush=True)

    # 远离点采样：随机序 + 最小间距过滤，机位覆盖全图不扎堆
    rng = random.Random(args.seed)
    order = list(range(net.node_count))
    rng.shuffle(order)
    stops: list[int] = []
    for i in order:
        if len(stops) >= args.stops:
            break
        xy = net.nodes[i]
        if all(float((xy - net.nodes[j]) @ (xy - net.nodes[j])) ** 0.5
               >= args.min_dist for j in stops):
            stops.append(i)
    print(f"[tour] {len(stops)} stops sampled "
          f"(min dist {args.min_dist}m)", flush=True)

    ring, _ = build_camera_ring_provider(conn, args.runtime, 536, 403,
                                         roles=("front_main",))
    n = 0
    t0 = time.time()
    for k, i in enumerate(stops):
        x, y = float(net.nodes[i][0]), float(net.nodes[i][1])
        try:
            h = float(net.road_heading_at((x, y)))
            conn.safe_teleport(x, y, heading_deg=math.degrees(h))
            conn.step(args.settle_steps)
            snap = ring.grab_ring()
            role = "front_main" if "front_main" in snap else next(iter(snap))
            rgb = snap[role][0]
            cv2.imwrite(str(out_dir / f"frame_{n:05d}.png"),
                        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            n += 1
        except Exception as exc:
            print(f"[tour] stop {i} failed: {exc}", flush=True)
        if (k + 1) % 25 == 0:
            rate = (k + 1) / max(time.time() - t0, 1.0)
            print(f"[tour] {k + 1}/{len(stops)} stops, {n} frames "
                  f"({rate:.1f} stops/s)", flush=True)
    print(f"[tour] done: {n} frames -> {out_dir}", flush=True)
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
