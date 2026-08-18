"""M5 分割训练数据采集：Tech colour + annotation 同步保存。

游戏 AI 沿路行驶（span 模式），同时保存 RGB 帧与 3 类标签
（0=背景 / 1=路面 ASPHALT / 2=标线 SOLID+DASHED+ZEBRA），
半分辨率 (536, 403) 存储，每帧一个 npz。

--segments > 1 时为多样采集：每段从路网随机节点出发（远离当前点），
AI 沿路行驶 frames-per-seg 帧，覆盖不同路段，提升模型泛化。

用法（Tech 实例需在运行，默认端口 64257）:
    .venv\Scripts\python.exe scripts\m5_collect_seg.py --segments 4 --frames-per-seg 225
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
from beamng_autopilot.roadnet import RoadNetwork
from beamng_autopilot_tech.annotations import to_label

W, H = 536, 403  # 半分辨率（原始 1076x806）

MIN_SEGMENT_DIST_M = 200.0  # 新段起点至少离当前位置这么远


def _random_road_point(roadnet: RoadNetwork, cur_xy, rng) -> tuple:
    """随机选一个离当前点足够远的路网节点 (x, y, z, heading)。"""
    for _ in range(30):
        idx = int(rng.integers(0, roadnet.node_count))
        xy = roadnet.nodes[idx]
        if np.hypot(*(xy - np.asarray(cur_xy[:2], dtype=float))) \
                < MIN_SEGMENT_DIST_M:
            continue
        z = float(roadnet.heights[idx]) if roadnet.heights is not None \
            else 0.0
        hdg = roadnet.road_heading_at(xy)
        return float(xy[0]), float(xy[1]), z, hdg
    raise RuntimeError("roadnet 中找不到足够远的随机节点")


def _teleport_to(conn, xyz, heading, lift: float = 0.6,
                 no_step: bool = False) -> None:
    """把车 teleport 到路网节点并按道路方向摆正。"""
    from beamngpy.misc.quat import angle_to_quat

    yaw_deg = -math.degrees(float(heading)) - 90.0
    quat = angle_to_quat((0.0, 0.0, yaw_deg))
    with conn.io_lock:
        conn.vehicle.teleport((float(xyz[0]), float(xyz[1]),
                               float(xyz[2]) + lift), rot_quat=quat)
        if no_step:
            time.sleep(1.0)  # 让持有 step 的客户端推进物理结算
        else:
            conn.step(30)


def main() -> None:
    ap = argparse.ArgumentParser(description="分割数据采集")
    ap.add_argument("--port", type=int, default=64257)
    ap.add_argument("--runtime", choices=("auto", "steam", "tech"),
                    default="tech",
                    help="game runtime: tech (default) or steam")
    ap.add_argument("--frames", type=int, default=None,
                    help="总帧数（默认 segments x frames-per-seg）")
    ap.add_argument("--segments", type=int, default=1,
                    help="多样采集段数（每段从路网随机节点出发，>1 提升泛化）")
    ap.add_argument("--frames-per-seg", type=int, default=225,
                    help="每段帧数（默认 225）")
    ap.add_argument("--rate", type=float, default=4.0)
    ap.add_argument("--speed", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--run", default=None,
                    help="运行名（默认自动时间戳）")
    ap.add_argument("--no-step", action="store_true",
                    help="共享模式：不 blocking step（另一客户端如 "
                         "lane_state_view 在推进 sim），只 poll 相机")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    from beamngpy.sensors import Camera
    from beamng_autopilot_tech.providers import (
        CAMERA_POS, CAMERA_DIR, CAMERA_UP, CAMERA_FOV_DEG)

    segments = max(1, args.segments)
    per_seg = max(20, args.frames_per_seg)
    total = args.frames or segments * per_seg

    conn = BeamNGConnector(
        port=(args.port or config.runtime_port(args.runtime)),
        home=config.runtime_home(args.runtime))
    conn.open(launch=False)
    try:
        conn.attach_vehicle(already_open=True)
    except Exception:
        conn.load_scenario()
        conn.step(60)

    cam = Camera("seg_collect_cam", conn.bng, conn.vehicle,
                 requested_update_time=0.05, pos=CAMERA_POS, dir=CAMERA_DIR,
                 up=CAMERA_UP, resolution=(1076, 806),
                 field_of_view_y=CAMERA_FOV_DEG, near_far_planes=(0.05, 150.0),
                 is_using_shared_memory=True, is_render_colours=True,
                 is_render_annotations=True, is_render_instance=False,
                 is_render_depth=False, is_visualised=False)

    run = args.run or time.strftime("%Y%m%d_%H%M%S")
    out_dir = config.LOGS_DIR / "m5_seg" / f"run_{run}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 多样采集：构建路网用于随机取点
    roadnet = None
    rng = np.random.default_rng(args.seed)
    if segments > 1:
        roadnet = RoadNetwork()
        t0 = time.time()
        while not roadnet.ready and time.time() - t0 < 90.0:
            with conn.io_lock:
                if roadnet.build(conn.bng):
                    print(f"[collect] roadnet ready: {roadnet.info}")
                    break
            time.sleep(1.0)
        if not roadnet.ready:
            print(f"[collect] WARNING: roadnet 不可用，"
                  f"退回单段采集（{roadnet.info}）")
            segments = 1

    meta = {"port": args.port, "speed": args.speed, "w": W, "h": H,
            "classes": ["background", "asphalt", "line"],
            "segments": segments, "frames_per_seg": per_seg,
            "frames": [], "segment_starts": []}
    line_px_total = 0
    frame_i = 0
    seg_start_xy = None

    def start_segment(seg: int) -> None:
        nonlocal seg_start_xy
        if seg > 0 and roadnet is not None:
            st = conn.get_state()
            x, y, z, hdg = _random_road_point(roadnet, st.pos[:2], rng)
            _teleport_to(conn, (x, y, z), hdg, no_step=args.no_step)
            print(f"[collect] segment {seg}: teleport -> "
                  f"({x:.0f}, {y:.0f})")
            meta["segment_starts"].append([round(x, 1), round(y, 1)])
        else:
            meta["segment_starts"].append(None)
        try:
            with conn.io_lock:
                conn.vehicle.ai.set_mode("span")
                conn.vehicle.ai.set_speed(args.speed, mode="limit")
        except Exception as exc:
            print(f"[collect] WARNING: AI 启动失败 ({exc})，原地采集")

    for seg in range(segments):
        start_segment(seg)
        seg_frames = min(per_seg, total - frame_i)
        for k in range(seg_frames):
            t0 = time.time()
            if not args.no_step:
                conn.step(10)
            with conn.io_lock:
                data = cam.poll()
            colour = np.ascontiguousarray(
                np.asarray(data["colour"], dtype=np.uint8))
            ann = np.ascontiguousarray(
                np.asarray(data["annotation"], dtype=np.uint8))
            st = conn.get_state()

            small_colour = cv2.resize(colour, (W, H),
                                      interpolation=cv2.INTER_AREA)
            small_ann = cv2.resize(ann, (W, H),
                                   interpolation=cv2.INTER_NEAREST)
            label = to_label(small_ann)

            np.savez_compressed(
                out_dir / f"frame_{frame_i:05d}.npz",
                colour=small_colour, label=label)
            meta["frames"].append({
                "i": frame_i, "seg": seg,
                "pos": [round(float(v), 2) for v in st.pos],
                "heading": round(float(st.heading), 4),
            })
            line_px_total += int((label == 2).sum())
            frame_i += 1

            if (frame_i) % 25 == 0 or frame_i == total:
                print(f"[collect] {frame_i}/{total}  段 {seg + 1}/{segments}  "
                      f"标线px/帧={line_px_total / max(1, frame_i):.0f}")

            rem = 1.0 / args.rate - (time.time() - t0)
            if rem > 0:
                time.sleep(rem)
            if frame_i >= total:
                break

    try:
        with conn.io_lock:
            conn.vehicle.ai.set_mode("disabled")
    except Exception:
        pass
    with conn.io_lock:
        cam.remove()
    conn.close()

    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[collect] 完成: {frame_i} 帧 / {segments} 段 -> {out_dir}")
    print(f"[collect] 标线像素总量 {line_px_total} "
          f"({line_px_total / max(1, frame_i):.0f}/帧)")


if __name__ == "__main__":
    main()