"""M5 车道真值对比探针：CV 检测 vs BeamNG.tech annotation 像素真值。

量化当前传统 CV 车道线/路面边界检测的真实水平，输出三项指标：
  1. 路面分割 IoU：cv 色度分类 vs annotation ASPHALT
  2. 标线检测 precision/recall/IoU：LaneDetector vs SOLID/DASHED/ZEBRA
  3. 边界横向距离误差（米）：estimate_pavement_edges vs 真值掩码
     （同一几何提取管道，输入换成完美真值掩码，公平对比）

用法（需要 BeamNG.tech 实例，端口默认 64257）:
    .venv\\Scripts\\python.exe scripts\\m5_lane_truth_probe.py --frames 30

结果输出到控制台报表，对比帧存 logs/m5_lane_truth/。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beamng_autopilot import config
from beamng_autopilot.connector import BeamNGConnector
from beamng_autopilot.vision.lane_overlay import (
    _offroad_mask,
    _side_pavement_lat,
    estimate_pavement_edges,
    unit_fwd,
    PAVEMENT_STATIONS_M,
)
from beamng_autopilot.vision.lanes import LaneDetector

# ---- BeamNG.tech annotation 真值颜色（tech/annotations.json）----
ANN_ASPHALT = (128, 128, 128)
ANN_SOLID_LINE = (255, 196, 128)
ANN_DASHED_LINE = (196, 196, 255)  # json 里 256 溢出截断为 255
ANN_ZEBRA = (255, 128, 128)
ANN_SIDEWALK = (89, 118, 155)
ANN_GUARD_RAIL = (0, 128, 128)
_LINE_COLORS = {ANN_SOLID_LINE, ANN_DASHED_LINE, ANN_ZEBRA}


def _mask_for(ann, color) -> np.ndarray:
    return (ann == np.asarray(color, dtype=np.uint8)).all(axis=2)


def _lines_mask(ann) -> np.ndarray:
    out = np.zeros(ann.shape[:2], dtype=bool)
    for c in _LINE_COLORS:
        out |= _mask_for(ann, c)
    return out


def _markings_pixel_mask(markings, shape) -> np.ndarray:
    """把 CV 检测到的标线 polyline 画成像素掩码（与真值同空间对比）。"""
    mask = np.zeros(shape, dtype=np.uint8)
    for mk in markings:
        pts = np.asarray(mk.pixels, dtype=np.int32)
        if len(pts) < 2:
            continue
        cv2.polylines(mask, [pts], False, 255, 3, cv2.LINE_AA)
    return mask > 0


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return inter / union if union else 1.0


def _truth_edge_lats(mask_offroad, cam, pos, heading, fwd, left,
                     ground_z: float) -> dict:
    """用与 CV 完全相同的几何管道从真值掩码提取左右边界。"""
    lats = {s: None for s in PAVEMENT_STATIONS_M}
    for s_m in PAVEMENT_STATIONS_M:
        l = _side_pavement_lat(mask_offroad, cam, pos, heading, fwd, left,
                               s_m, 1.0, ground_z)
        r = _side_pavement_lat(mask_offroad, cam, pos, heading, fwd, left,
                               s_m, -1.0, ground_z)
        lats[s_m] = (l, r)
    return lats


def main() -> None:
    ap = argparse.ArgumentParser(description="车道真值对比探针")
    ap.add_argument("--port", type=int, default=64257,
                    help="Tech 实例端口（默认 64257）")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--rate", type=float, default=4.0)
    ap.add_argument("--drive", action="store_true",
                    help="用游戏 AI 沿路行驶采集（标线通常在行驶路段上）")
    ap.add_argument("--drive-speed", type=float, default=8.0,
                    help="AI 行驶速度 m/s（默认 8）")
    ap.add_argument("--save", action="store_true",
                    help="保存对比帧到 logs/m5_lane_truth/")
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--model", default=None,
                    help="学习式分割模型路径（缺省用 CV 检测；传路径则对比模型）")
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
    print(f"[probe] attached vehicle '{conn.vehicle.vid}'")

    if args.drive:
        # 游戏原生 AI 沿当前道路行驶（span 模式），把车带到有标线的路段。
        try:
            with conn.io_lock:
                conn.vehicle.ai.set_mode("span")
                conn.vehicle.ai.set_speed(args.drive_speed, mode="limit")
            print(f"[probe] AI driving at {args.drive_speed} m/s (span)")
        except Exception as exc:
            print(f"[probe] WARNING: AI drive failed ({exc}); "
                  "collecting while stationary")

    cam = Camera("truth_probe_cam", conn.bng, conn.vehicle,
                 requested_update_time=0.05, pos=CAMERA_POS, dir=CAMERA_DIR,
                 up=CAMERA_UP, resolution=(1076, 806),
                 field_of_view_y=CAMERA_FOV_DEG, near_far_planes=(0.05, 150.0),
                 is_using_shared_memory=True, is_render_colours=True,
                 is_render_annotations=True, is_render_instance=False,
                 is_render_depth=False, is_visualised=False)

    if args.model:
        from beamng_autopilot.vision.segmentation import Segmenter

        segmenter = Segmenter(model_path=args.model)
        detector = None
        print(f"[probe] 使用学习式分割: {args.model}")
    else:
        segmenter = None
        detector = LaneDetector()
        print("[probe] 使用经典 CV 检测")
    out_dir = config.LOGS_DIR / "m5_lane_truth"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 指标累计
    road_ious, line_prec, line_rec, line_iou = [], [], [], []
    edge_left_err, edge_right_err = [], []
    dash_colors_seen: dict[tuple, int] = {}
    line_px_total = 0
    n_line_frames = 0
    MIN_LINE_PX = 1500  # 真值标线像素低于此值的帧不算"有标线帧"

    for i in range(args.frames):
        t0 = time.time()
        conn.step(10)
        with conn.io_lock:
            data = cam.poll()
        colour = np.ascontiguousarray(np.asarray(data["colour"], dtype=np.uint8))
        ann = np.ascontiguousarray(np.asarray(data["annotation"], dtype=np.uint8))
        st = conn.get_state()
        h, w = colour.shape[:2]
        fwd = unit_fwd(st)
        heading = float(st.heading)
        ground_z = float(st.pos[2])

        from beamng_autopilot.vision.projection import default_camera
        cam_model = default_camera(w, h)  # 标定外参（与 Tech provider 等价）

        # ---- 检测（CV 或学习式）----
        if segmenter is not None:
            cv_markings = segmenter.detect_lines(
                colour, cam_model, st.pos, heading, ground_z=ground_z)
            cv_edges = estimate_pavement_edges(
                colour, cam_model, st.pos, heading, ground_z=ground_z,
                offroad_mask=segmenter.offroad_mask(colour))
            cv_road = ~segmenter.offroad_mask(colour)
        else:
            cv_markings = detector.detect(colour, cam_model, st.pos, heading,
                                          ground_z=ground_z)
            cv_edges = estimate_pavement_edges(colour, cam_model, st.pos,
                                               heading, ground_z=ground_z)
            cv_road = ~_offroad_mask(colour)

        # ---- 真值 ----
        truth_road = _mask_for(ann, ANN_ASPHALT)
        truth_lines = _lines_mask(ann)
        for c, n in zip(*np.unique(ann.reshape(-1, 3), axis=0,
                                   return_counts=True)):
            c = tuple(int(v) for v in c)
            if c in _LINE_COLORS:
                dash_colors_seen[c] = dash_colors_seen.get(c, 0) + int(n)

        # 1) 路面 IoU（所有帧）
        road_ious.append(_iou(cv_road, truth_road))

        # 2) 标线（只在真值标线充足的帧上统计）
        line_px = int(truth_lines.sum())
        line_px_total += line_px
        if line_px >= MIN_LINE_PX:
            n_line_frames += 1
            cv_line_mask = _markings_pixel_mask(cv_markings,
                                                truth_lines.shape)
            tp = int(np.logical_and(cv_line_mask, truth_lines).sum())
            fp = int(cv_line_mask.sum()) - tp
            fn = int(truth_lines.sum()) - tp
            line_prec.append(tp / (tp + fp) if tp + fp else 1.0)
            line_rec.append(tp / (tp + fn) if tp + fn else 1.0)
            line_iou.append(_iou(cv_line_mask, truth_lines))

        # 3) 边界误差：CV near lat vs 真值 near lat
        if cv_edges is not None:
            truth_off = ~truth_road
            t_lats = _truth_edge_lats(truth_off, cam_model, st.pos, heading,
                                      fwd, np.array([-fwd[1], fwd[0]]),
                                      ground_z)
            near_s = [s for s in PAVEMENT_STATIONS_M
                      if s <= 6.5 and t_lats[s][0] is not None
                      and t_lats[s][1] is not None]
            if near_s and cv_edges.get("left_lat") is not None \
                    and cv_edges.get("right_lat") is not None:
                t_left = float(np.median([t_lats[s][0] for s in near_s]))
                t_right = float(np.median([t_lats[s][1] for s in near_s]))
                edge_left_err.append(abs(cv_edges["left_lat"] - t_left))
                edge_right_err.append(abs(cv_edges["right_lat"] - t_right))

        if args.save and i % args.save_every == 0:
            vis = ann.copy()
            vis[truth_lines] = (0, 255, 0)      # 真值标线 -> 绿
            vis[_mask_for(ann, ANN_ASPHALT)] = (128, 128, 128)
            cv_vis = colour.copy()
            for mk in cv_markings:
                pts = np.asarray(mk.pixels, dtype=np.int32)
                if len(pts) >= 2:
                    cv2.polylines(cv_vis, [pts], False, (0, 200, 255), 3)
            panel = np.hstack([colour, vis, cv_vis])
            cv2.imwrite(str(out_dir / f"truth_cmp_{i:03d}.png"),
                        cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

        if (i + 1) % 5 == 0 or i == args.frames - 1:
            print(f"[probe] {i + 1}/{args.frames}  road_iou="
                  f"{np.median(road_ious):.3f} line_rec="
                  f"{np.median(line_rec):.3f}")

        rem = 1.0 / args.rate - (time.time() - t0)
        if rem > 0:
            time.sleep(rem)

    if args.drive:
        try:
            with conn.io_lock:
                conn.vehicle.ai.set_mode("disabled")
        except Exception:
            pass

    with conn.io_lock:
        cam.remove()
    conn.close()

    def _med(x):
        return float(np.median(x)) if x else float("nan")

    print()
    print("=" * 62)
    print("车道识别 vs 像素真值（BeamNG.tech annotation）")
    print("=" * 62)
    print(f"帧数: {args.frames}" + ("  (AI 沿路行驶)" if args.drive else ""))
    print(f"标线真值颜色出现统计: {dash_colors_seen}")
    print(f"标线像素总量: {line_px_total}  "
          f"有标线帧(>={MIN_LINE_PX}px): {n_line_frames}/{args.frames}")
    print(f"路面分割 IoU          : 中位 {_med(road_ious):.3f}")
    if n_line_frames:
        print(f"标线 precision        : 中位 {_med(line_prec):.3f}")
        print(f"标线 recall           : 中位 {_med(line_rec):.3f}")
        print(f"标线 IoU              : 中位 {_med(line_iou):.3f}")
    else:
        print("标线指标: 无有效帧（行驶路段上没有足够标线，加大 --frames 或换路段）")
    print(f"左边界误差 (m)        : 中位 {_med(edge_left_err):.3f}"
          f"  (n={len(edge_left_err)})")
    print(f"右边界误差 (m)        : 中位 {_med(edge_right_err):.3f}"
          f"  (n={len(edge_right_err)})")
    if args.save:
        print(f"对比帧已存: {out_dir}/truth_cmp_*.png")
    print("=" * 62)


if __name__ == "__main__":
    main()