"""M5 分割训练数据采集：Tech colour + annotation 同步保存。

游戏 AI 沿路行驶（span 模式），同时保存 RGB 帧与 3 类标签
（0=背景 / 1=路面 ASPHALT / 2=标线 SOLID+DASHED+ZEBRA），
半分辨率 (536, 403) 存储，每帧一个 npz。

用法（Tech 实例需在运行，默认端口 64257）:
    .venv\Scripts\python.exe scripts\m5_collect_seg.py --frames 500
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beamng_autopilot import config
from beamng_autopilot.connector import BeamNGConnector

W, H = 536, 403  # 半分辨率（原始 1076x806）

ANN_ASPHALT = (128, 128, 128)
ANN_SOLID_LINE = (255, 196, 128)
ANN_DASHED_LINE = (196, 196, 255)
ANN_ZEBRA = (255, 128, 128)
_LINE_COLORS = {ANN_SOLID_LINE, ANN_DASHED_LINE, ANN_ZEBRA}


def to_label(ann_rgb: np.ndarray) -> np.ndarray:
    """annotation RGB 图 -> 3 类标签图 (H, W) uint8。"""
    label = np.zeros(ann_rgb.shape[:2], dtype=np.uint8)
    label[(ann_rgb == np.asarray(ANN_ASPHALT, dtype=np.uint8)).all(axis=2)] = 1
    for c in _LINE_COLORS:
        label[(ann_rgb == np.asarray(c, dtype=np.uint8)).all(axis=2)] = 2
    return label


def main() -> None:
    ap = argparse.ArgumentParser(description="分割数据采集")
    ap.add_argument("--port", type=int, default=64257)
    ap.add_argument("--frames", type=int, default=500)
    ap.add_argument("--rate", type=float, default=4.0)
    ap.add_argument("--speed", type=float, default=8.0)
    ap.add_argument("--run", default=None,
                    help="运行名（默认自动时间戳）")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    from beamngpy.sensors import Camera
    from beamng_autopilot_tech.providers import (
        CAMERA_POS, CAMERA_DIR, CAMERA_UP, CAMERA_FOV_DEG)

    conn = BeamNGConnector(port=args.port)
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

    try:
        with conn.io_lock:
            conn.vehicle.ai.set_mode("span")
            conn.vehicle.ai.set_speed(args.speed, mode="limit")
        print(f"[collect] AI 行驶 {args.speed} m/s, 采集 {args.frames} 帧 "
              f"-> {out_dir}")
    except Exception as exc:
        print(f"[collect] WARNING: AI 启动失败 ({exc})，原地采集")

    meta = {"port": args.port, "speed": args.speed, "w": W, "h": H,
            "classes": ["background", "asphalt", "line"], "frames": []}
    line_px_total = 0
    for i in range(args.frames):
        t0 = time.time()
        conn.step(10)
        with conn.io_lock:
            data = cam.poll()
        colour = np.ascontiguousarray(np.asarray(data["colour"], dtype=np.uint8))
        ann = np.ascontiguousarray(np.asarray(data["annotation"], dtype=np.uint8))
        st = conn.get_state()

        small_colour = cv2.resize(colour, (W, H), interpolation=cv2.INTER_AREA)
        small_ann = cv2.resize(ann, (W, H), interpolation=cv2.INTER_NEAREST)
        label = to_label(small_ann)

        np.savez_compressed(
            out_dir / f"frame_{i:05d}.npz",
            colour=small_colour, label=label)
        meta["frames"].append({
            "i": i, "pos": [round(float(v), 2) for v in st.pos],
            "heading": round(float(st.heading), 4),
        })
        line_px_total += int((label == 2).sum())

        if (i + 1) % 25 == 0 or i == args.frames - 1:
            print(f"[collect] {i + 1}/{args.frames}  "
                  f"标线像素/帧={line_px_total / max(1, i + 1):.0f}")

        rem = 1.0 / args.rate - (time.time() - t0)
        if rem > 0:
            time.sleep(rem)

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
    n_line = sum(1 for f in meta["frames"] if True)
    print(f"[collect] 完成: {n_line} 帧 -> {out_dir}")
    print(f"[collect] 标线像素总量 {line_px_total} "
          f"({line_px_total / max(1, n_line):.0f}/帧)")


if __name__ == "__main__":
    main()