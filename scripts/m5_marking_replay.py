"""live 帧回放 + mask→marking 诊断（预注册 T16_MARKING_REPLAY_PREREG_20261104）。

对 `m5_pairing_probe --save-frames` 保存的每帧离线回放：
  seg.predict -> seg.detect_lines -> pair_lane_markings
按 H-M0/M1/M2/M3/ok 逐帧分类，输出定位表。

用法：
  .venv\\Scripts\\python.exe scripts\\m5_marking_replay.py \\
      --frames logs/m5_seg/marking_replay_east_r0.frames \\
      --scene logs/experiments/t16_scenes_20261102_140335/scene_known_line_a0s0.json \\
      --edge-margin 1.0 --model logs/m5_seg/seg_model_v28_sslA/best.pt \\
      --out logs/m5_seg/marking_replay_east.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NEAR_TOL_M = 0.6      # marking 与 GT 右线的横向命中容差
MASK_RADIUS_PX = 5    # 掩码在 GT 投影点附近的检查半径
STATION_M = 8.0       # GT 取距 ego 8m 的横向


def pre_scale(frame: np.ndarray, width: int) -> np.ndarray:
    """输入域匹配：按目标宽度做 INTER_AREA 缩小（训练域预处理）。"""
    tw = int(width)
    th = max(1, int(round(frame.shape[0] * tw / float(frame.shape[1]))))
    return cv2.resize(np.asarray(frame), (tw, th),
                      interpolation=cv2.INTER_AREA)


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def _right_lane_lats(half_width_m: float, edge_margin_m: float):
    """复用采集器的 right_lane 约定（scripts 不是包，按文件加载）。"""
    import importlib.util
    entry = ROOT / "scripts" / "m5_controlled_scenes.py"
    spec = importlib.util.spec_from_file_location("m5cs_replay", entry)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.right_lane_lats(half_width_m, edge_margin_m=edge_margin_m)


def right_line_world(scene_doc: dict, edge_margin_m: float,
                     stations=(4.0, 8.0, 12.0, 16.0, 20.0)):
    """按 right_lane 约定生成右线（黄线）世界点。"""
    s = scene_doc["scene"]
    mid = np.asarray(s["mid"], dtype=float)
    d = _unit(s["dir"])
    left2d = np.array([-d[1], d[0]])
    _ll, rl = _right_lane_lats(float(s["half_width_m"]),
                               float(edge_margin_m))
    slope = float(s.get("slope_pct") or 0.0) / 100.0
    pts = []
    for st in stations:
        p = mid + d * st + np.array([left2d[0], left2d[1], 0.0]) * rl
        p[2] = mid[2] + slope * st
        pts.append(p)
    return np.asarray(pts)


def ego_left(pos, heading: float) -> np.ndarray:
    fwd = np.array([math.cos(heading), math.sin(heading)])
    return np.array([-fwd[1], fwd[0]])


def gt_right_lat(scene_doc: dict, edge_margin_m: float, pos, heading: float,
                 station_m: float = STATION_M) -> float:
    pts = right_line_world(scene_doc, edge_margin_m, stations=(station_m,))
    return float((pts[0, :2] - np.asarray(pos, dtype=float)[:2])
                 @ ego_left(pos, heading))


def marking_near_lat(mk, pos, heading: float) -> float | None:
    """marking 近场横向（车体系左正）；无 world 点返回 None。"""
    try:
        world = np.asarray(mk.world, dtype=float)[:, :2]
        if len(world) == 0:
            return None
        rel = world - np.asarray(pos, dtype=float)[:2]
        fwd = np.array([math.cos(heading), math.sin(heading)])
        s = rel @ fwd
        lat = rel @ ego_left(pos, heading)
        near = lat[np.abs(s) <= 15.0]
        return float(np.median(near if len(near) else lat))
    except Exception:  # noqa: BLE001
        return None


def classify_frame(gt_lat: float, marking_lats: list[float],
                   cand_lats: list[float], mask_near: bool,
                   paired: bool) -> str:
    """H-M0..M3 / ok 的逐帧互斥分类（纯函数，单测覆盖）。"""
    def near(v):
        return v is not None and abs(float(v) - float(gt_lat)) <= NEAR_TOL_M

    if not any(near(v) for v in marking_lats):
        return "H-M1" if mask_near else "H-M0"
    if any(near(v) for v in cand_lats):
        return "ok" if paired else "H-M3"
    return "H-M2"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True, help=".frames 目录（npz）")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--edge-margin", type=float, default=0.6)
    ap.add_argument("--model", required=True)
    ap.add_argument("--input-width", type=int, default=None,
                    help="输入域匹配（默认 None）：先把帧缩到该宽度再送模型，"
                         "掩码放大回原尺寸继续")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from beamng_autopilot.vision.segmentation import Segmenter
    from beamng_autopilot.lane import pair_lane_markings

    scene_doc = json.loads(Path(args.scene).read_text(encoding="utf-8"))
    seg = Segmenter(model_path=args.model, device=args.device)
    rows = []
    for fp in sorted(glob.glob(str(Path(args.frames) / "frame_*.npz"))):
        d = np.load(fp, allow_pickle=True)
        frame = d["frame"]
        pos = d["pos"]
        heading = float(d["heading"])
        cam = d["cam"][0]
        frame_in = frame
        pre = bool(args.input_width)
        if pre:
            frame_in = pre_scale(frame, int(args.input_width))
        road, line = seg.predict(frame_in)
        if pre:
            road = cv2.resize(road.astype(np.uint8),
                              (frame.shape[1], frame.shape[0]),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
            line = cv2.resize(line.astype(np.uint8),
                              (frame.shape[1], frame.shape[0]),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
        dbg: dict = {}
        markings = seg.detect_lines(frame, cam, pos, heading,
                                    line_mask=line, road_mask=road,
                                    debug=dbg)
        dbg2: dict = {}
        pair = pair_lane_markings(markings, pos, heading, debug=dbg2)
        g_lat = gt_right_lat(scene_doc, args.edge_margin, pos, heading)
        gt_pts = right_line_world(scene_doc, args.edge_margin)
        u, v, valid = cam.project(gt_pts, pos, heading)
        near_mask = False
        if np.any(valid):
            dil = cv2.dilate(np.asarray(line, dtype=np.uint8),
                             np.ones((2 * MASK_RADIUS_PX + 1,
                                      2 * MASK_RADIUS_PX + 1), np.uint8))
            for uu, vv, ok in zip(u, v, valid):
                if ok and 0 <= int(vv) < dil.shape[0] \
                        and 0 <= int(uu) < dil.shape[1] and dil[int(vv), int(uu)]:
                    near_mask = True
                    break
        m_lats = [x for x in (marking_near_lat(m, pos, heading)
                              for m in markings) if x is not None]
        c_lats = [float(c["med_lat"]) for c in (dbg2.get("cands") or [])
                  if c.get("med_lat") is not None]
        label = classify_frame(g_lat, m_lats, c_lats, near_mask,
                               pair is not None)
        rows.append({
            "file": Path(fp).name,
            "gt_right_lat": round(g_lat, 3),
            "markings": len(markings),
            "marking_lats": [round(x, 3) for x in m_lats],
            "cand_lats": [round(x, 3) for x in c_lats],
            "mask_near_gt": bool(near_mask),
            "pair": pair is not None,
            "collect_drops": dbg2.get("collect_drops") or {},
            "label": label,
        })
    counts = {}
    for r in rows:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    summary = {"frames": len(rows), "labels": counts,
               "ok_rate": round(counts.get("ok", 0) / max(1, len(rows)), 4)}
    out = {"schema": "marking-replay-v1", "scene": args.scene,
           "edge_margin": args.edge_margin, "model": args.model,
           "input_width": args.input_width,
           "summary": summary, "frames": rows}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
        print(f"[replay] -> {args.out}", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
