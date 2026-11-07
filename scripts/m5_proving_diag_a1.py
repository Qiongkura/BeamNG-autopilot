# -*- coding: utf-8 -*-
"""A1 对照实验分析：右线渲染偏移定位（诊断，不进训练池）。

预注册修订案 A1（``docs/T16_PROVING_EXPORT_PILOT_BLOCKER_20261105.md``）：
对采集帧逐帧测量"渲染橙色相对构建节点的横向偏移"，并支持 phantom 模式
——把参照 built.json 的某条线投影进**另一个场景**的帧里，统计其走廊内的
橙色存在率。两种模式一起回答：右线位置的橙色到底是不是右线节点生成的。

模式::

    # scan：每条构建线输出 offset(depth) 曲线（渲染横向位置 - 节点横向）
    python scripts/m5_proving_diag_a1.py --mode scan \
        --frames-dir logs/m5_seg/.../probe.frames \
        --built logs/m5_seg/.../probe.built.json --out report.json

    # phantom：参照 built.json 的 --phantom-role 线投影进本帧集，走廊内
    # 橙色存在率（预期：只左线场景中右线走廊 ≈ 0 存在率）
    python scripts/m5_proving_diag_a1.py --mode phantom \
        --frames-dir logs/m5_seg/.../probe.frames \
        --built logs/m5_seg/.../probe.built.json \
        --phantom-built logs/m5_seg/proving_capture/S1/noon_clear/probe.built.json \
        --phantom-role right --out report.json

只读帧与构建记录，不写任何训练池产物。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MIN_DEPTH_M = 5.0
MAX_DEPTH_M = 25.0
HIT_RADIUS_PX = 6.0
SCAN_HALFWIDTH_M = 3.0
SCAN_STEP_M = 0.05
MAX_POINTS_IN_REPORT = 400


def _load_export_module():
    spec = importlib.util.spec_from_file_location(
        "m5_proving_export_a1", str(ROOT / "scripts" / "m5_proving_export.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def line_lateral_dir(nodes: np.ndarray) -> np.ndarray:
    """线节点链的横向单位向量（左正），由相邻节点水平差分推断。"""
    d = np.diff(np.asarray(nodes, float)[:, :2], axis=0)
    seg = d[np.linalg.norm(d, axis=1) > 1e-6]
    if not len(seg):
        raise ValueError("degenerate node chain")
    fwd = seg[len(seg) // 2]
    fwd = fwd / np.linalg.norm(fwd)
    return np.array([-fwd[1], fwd[0]])


def _nearest_paint_dist(paint_pts: np.ndarray, us: np.ndarray,
                        vs: np.ndarray) -> np.ndarray:
    """每个投影点到最近橙色像素的像素距离；无橙色时全 inf。"""
    if not paint_pts.size:
        return np.full(len(us), np.inf)
    dd = np.hypot(paint_pts[:, 0][None, :] - us[:, None],
                  paint_pts[:, 1][None, :] - vs[:, None])
    return dd.min(axis=1)


def scan_line_offsets(cam, pos, heading, rgb_shape, nodes, paint_pts, *,
                      min_depth=MIN_DEPTH_M, max_depth=MAX_DEPTH_M,
                      halfwidth=SCAN_HALFWIDTH_M, step=SCAN_STEP_M) -> list[dict]:
    """逐节点找与橙色像素最近的横向偏移（渲染横向 - 节点横向）。

    返回 ``[{"depth_m", "offset_m", "dist_px"}]``；该深度附近完全无橙色时
    offset 为 None。扫描半径与另一条构建线至少相距 0.6m（车道宽 3.6m），
    不会把对侧漆色误吸入本线扫描。
    """
    lat2d = line_lateral_dir(nodes)
    pos2 = np.asarray(pos, float)[:2]
    deltas = np.arange(-halfwidth, halfwidth + 1e-9, step)
    h, w = rgb_shape[:2]
    out = []
    for node in np.asarray(nodes, float):
        depth = float(np.linalg.norm(node[:2] - pos2))
        if not (min_depth <= depth <= max_depth):
            continue
        cands2 = node[:2][None, :] + lat2d[None, :] * deltas[:, None]
        pts = np.column_stack([cands2, np.full(len(deltas), node[2])])
        u, v, ok = cam.project(pts, pos, heading)
        in_frame = (ok & np.isfinite(u) & np.isfinite(v)
                    & (u >= 0) & (u < w) & (v >= 0) & (v < h))
        if not in_frame.any():
            continue
        dist = _nearest_paint_dist(paint_pts, u[in_frame], v[in_frame])
        bi = int(np.argmin(dist))
        if not np.isfinite(dist[bi]):
            out.append({"depth_m": round(depth, 2), "offset_m": None,
                        "dist_px": None})
            continue
        out.append({"depth_m": round(depth, 2),
                    "offset_m": round(float(deltas[in_frame][bi]), 4),
                    "dist_px": round(float(dist[bi]), 2)})
    return out


def analyze_frames(frames_dir: Path, built_doc: dict, mode: str,
                   phantom_doc: dict | None, phantom_role: str,
                   *, max_frames: int) -> dict:
    exp = _load_export_module()
    frames = sorted(frames_dir.glob("frame_*.npz"))[:max_frames]
    lines = {ln["role"]: ln for ln in built_doc.get("lines", [])}
    phantom_truth = []
    if phantom_doc is not None:
        for ln in phantom_doc.get("lines", []):
            if ln.get("role") == phantom_role:
                phantom_truth = ln.get("truth_points") or []
                break
        if not phantom_truth:
            raise SystemExit(f"参照 built 里没有 {phantom_role} 线真值点")
    scan_acc: dict[str, list[dict]] = {role: [] for role in lines}
    phantom_frames = []
    used = 0
    for fp in frames:
        z = np.load(fp, allow_pickle=True)
        rgb = np.asarray(z["frame"], np.uint8)
        cam = z["cam"][0]
        pos = np.asarray(z["pos"], float)
        heading = float(z["heading"])
        paint = exp.paint_mask(rgb)
        ys, xs = np.nonzero(paint)
        paint_pts = (np.column_stack([xs, ys]) if len(xs)
                     else np.zeros((0, 2), float))
        used += 1
        if mode == "scan":
            for role, ln in lines.items():
                scan_acc[role].extend(
                    scan_line_offsets(cam, pos, heading, rgb.shape,
                                      ln["nodes"], paint_pts))
        else:
            us, vs = _phantom_project(cam, pos, heading, phantom_truth)
            rec = _phantom_frame_stats(us, vs, rgb.shape, paint_pts)
            rec["file"] = fp.name
            phantom_frames.append(rec)
    report: dict = {"schema": "proving_diag_a1_v1", "mode": mode,
                    "frames_dir": str(frames_dir),
                    "built": str(built_doc.get("scene", "?")),
                    "frames_analyzed": used}
    if mode == "scan":
        report["scan"] = {role: _summarize_scan(recs)
                          for role, recs in scan_acc.items()}
    else:
        report["phantom"] = _summarize_phantom(phantom_frames, phantom_role)
    return report


def _phantom_project(cam, pos, heading, truth_points):
    worlds = [np.asarray(p["world"], float) for p in truth_points]
    if not worlds:
        return np.zeros(0), np.zeros(0)
    u, v, ok = cam.project(np.asarray(worlds, float), pos, heading)
    return u, v


def _phantom_frame_stats(us, vs, rgb_shape, paint_pts) -> dict:
    h, w = rgb_shape[:2]
    in_frame = (np.isfinite(us) & np.isfinite(vs)
                & (us >= 0) & (us < w) & (vs >= 0) & (vs < h))
    n = int(in_frame.sum())
    if not n or not paint_pts.size:
        return {"checked": n, "hits": 0,
                "coverage": (0.0 if n else None),
                "dist_median_px": None}
    dist = _nearest_paint_dist(paint_pts, us[in_frame], vs[in_frame])
    hits = int((dist <= HIT_RADIUS_PX).sum())
    return {"checked": n, "hits": hits, "coverage": hits / n,
            "dist_median_px": round(float(np.median(dist)), 2)}


def _summarize_scan(recs: list[dict]) -> dict:
    offsets = [r["offset_m"] for r in recs if r["offset_m"] is not None]
    by_depth: dict[int, list[float]] = {}
    for r in recs:
        if r["offset_m"] is not None:
            by_depth.setdefault(int(round(r["depth_m"])), []).append(r["offset_m"])
    hitable = [r["dist_px"] for r in recs if r["dist_px"] is not None]
    return {
        "points_total": len(recs),
        "points_with_paint": len(offsets),
        "hit_ratio": (round(len(offsets) / len(recs), 3) if recs else None),
        "offset_median_m": (round(float(np.median(offsets)), 3)
                            if offsets else None),
        "offset_by_depth_m": {str(d): round(float(np.median(v)), 3)
                              for d, v in sorted(by_depth.items())},
        "dist_median_px": (round(float(np.median(hitable)), 2)
                           if hitable else None),
        "points": recs[:MAX_POINTS_IN_REPORT],
    }


def _summarize_phantom(frames: list[dict], role: str) -> dict:
    checked = sum(f["checked"] for f in frames)
    hits = sum(f["hits"] for f in frames)
    covs = [f["coverage"] for f in frames if f["coverage"] is not None]
    med = [f["dist_median_px"] for f in frames
           if f["dist_median_px"] is not None]
    return {"role": role, "frames": len(frames),
            "points_total": checked, "hits_total": hits,
            "coverage": (round(hits / checked, 3) if checked else None),
            "frame_coverage_median": (round(float(np.median(covs)), 3)
                                      if covs else None),
            "dist_median_px": (round(float(np.median(med)), 2) if med else None),
            "per_frame": frames[:MAX_POINTS_IN_REPORT]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("scan", "phantom"), default="scan")
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--built", required=True, help="本帧集的 probe.built.json")
    ap.add_argument("--phantom-built", default=None,
                    help="phantom 模式的参照构建记录")
    ap.add_argument("--phantom-role", default="right")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-frames", type=int, default=40)
    args = ap.parse_args()

    built_doc = json.loads(Path(args.built).read_text(encoding="utf-8"))
    phantom_doc = None
    if args.mode == "phantom":
        if not args.phantom_built:
            raise SystemExit("phantom 模式需要 --phantom-built")
        phantom_doc = json.loads(
            Path(args.phantom_built).read_text(encoding="utf-8"))
    report = analyze_frames(Path(args.frames_dir), built_doc, args.mode,
                            phantom_doc, args.phantom_role,
                            max_frames=int(args.max_frames))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items()
                      if k in ("mode", "frames_analyzed", "scan", "phantom")},
                     ensure_ascii=False, indent=1)[:2000])
    print(f"[a1-diag] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
