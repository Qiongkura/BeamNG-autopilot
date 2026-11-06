# -*- coding: utf-8 -*-
"""试车场采集帧导出：几何真值标签 + 冻结门禁（预注册 §3）。

**PILOT 阻塞（2026-11-05）**：S1/noon_rain 51 帧全部被拒——右线渲染位置
比构建节点向外偏约 0.8m（左线 0.55px、右线 10px）。在修订案 A1 的对照
实验定论前，本工具**不用于训练**；证据见
``docs/T16_PROVING_EXPORT_PILOT_BLOCKER_20261105.md``。

输入 ``logs/m5_seg/proving_capture/<site>/<cell>/probe.frames/frame_*.npz``
（rgb/pos/heading/cam），输出每帧 ``rgb + line(2) + road(1)`` 的 npz 与
manifest。真值来自生成侧几何（right_lane 线位 + 线宽），不新增人工标注。

门禁（冻结，不因结果调整）：覆盖率 ≥0.8、投影残差中位 ≤1px/最大 ≤2px、
漆色一致率 ≥0.9、环境读回有效、内容 hash 去重；不合格帧隔离并保留原因。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CAP_DIR = ROOT / "logs" / "m5_seg" / "proving_capture"
POOL_DIR = ROOT / "logs" / "m5_seg" / "proving_pool"
SITES_DIR = ROOT / "logs" / "m5_seg" / "proving_sites"

LINE_WIDTH_M = 0.15
LINE_SPAN_M = 34.0
LINE_STEP_M = 1.0
TRUTH_STEP_M = 1.7
EXPAND_M = 1.0          # 车道区域外扩（road 标签）
COVERAGE_MIN = 0.80
RESIDUAL_MED_MAX_PX = 1.0
RESIDUAL_MAX_PX = 2.0
COLOR_MIN = 0.90
RADIUS_PX = 6
CHECK_MIN_M = 5.0       # 可核对距离带（近于 5m 出画/远于 25m 线宽 <2px）
CHECK_MAX_M = 25.0
#: 历史探索阈值，仅供候选规则测试；正式导出仍阻断，协议未冻结。
POSE_MIN_XY_M = 0.5
POSE_MIN_HEADING_RAD = 0.02


def find_pose_duplicate(pos, heading, accepted, *, min_xy=POSE_MIN_XY_M,
                        min_heading=POSE_MIN_HEADING_RAD):
    """查找全体已接受位姿中的重复项；无匹配返回 None。

    返回 (索引, 平面距离, 最短角差)。距离和角差均小于阈值才重复；
    因此原地转向达到角度阈值可作为不同姿态。此规则仅供离线验证，
    尚未经预注册准入，不能据此解除正式导出阻断。
    """
    p = np.asarray(pos, dtype=float).reshape(-1)
    h = float(heading)
    dx = float(min_xy)
    dh = float(min_heading)
    if p.size < 2 or not np.isfinite(p[:2]).all() or not np.isfinite(h):
        raise ValueError("invalid pose")
    if not np.isfinite(dx) or not np.isfinite(dh) or dx <= 0 or dh <= 0:
        raise ValueError("invalid pose thresholds")
    for index, (old_pos, old_heading) in enumerate(accepted):
        previous = np.asarray(old_pos, dtype=float).reshape(-1)
        oh = float(old_heading)
        if previous.size < 2 or not np.isfinite(previous[:2]).all() or not np.isfinite(oh):
            raise ValueError("invalid accepted pose")
        distance = float(np.linalg.norm(p[:2] - previous[:2]))
        delta = abs(float(np.arctan2(np.sin(h - oh), np.cos(h - oh))))
        if distance < dx and delta < dh:
            return index, distance, delta
    return None


def _load_scenes_module():
    spec = importlib.util.spec_from_file_location(
        "mcs_export", str(ROOT / "scripts" / "m5_controlled_scenes.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def site_geometry(site_doc: dict) -> list[dict]:
    """站点 JSON -> 两条线的 (role, center, left_rail, right_rail) 世界点。"""
    mcs = _load_scenes_module()
    sc = site_doc["scene"]
    site = {"mid": [float(x) for x in sc["mid"]],
            "dir": [float(x) for x in sc["dir"]],
            "half_width_m": float(sc["half_width_m"]),
            "slope_pct": float(sc.get("slope_pct", 0.0)),
            "curve_deg": float(sc.get("curve_deg", 0.0))}
    edge = float(site_doc.get("capture", {}).get("edge_margin_m", 0.6))
    ll, rl = mcs.right_lane_lats(site["half_width_m"],
                                 edge_margin_m=edge)
    out = []
    for role, lat in (("left", ll), ("right", rl)):
        center = np.asarray(mcs._line_nodes(site, lat, span_m=LINE_SPAN_M,
                                            step_m=LINE_STEP_M), float)
        d = np.asarray(site["dir"][:2], float)
        d = d / max(1e-9, float(np.linalg.norm(d)))
        left = np.array([-d[1], d[0]])
        off = np.zeros_like(center)
        off[:, :2] = left[None, :] * (LINE_WIDTH_M / 2.0)
        out.append({"role": role, "center": center,
                    "left": center + off,
                    "right": center - off})
    return out


def truth_points(geom: list[dict]) -> list[dict]:
    pts = []
    for line in geom:
        c = line["center"]
        for i in range(len(c) - 1):
            a, b = c[i], c[i + 1]
            n = max(1, int(np.linalg.norm(b - a) / TRUTH_STEP_M))
            for k in range(n):
                p = a + (b - a) * (k / n)
                pts.append({"world": [float(v) for v in p], "class": 2,
                            "role": line["role"]})
    return pts


def paint_mask(rgb: np.ndarray) -> np.ndarray:
    """试车场橙漆掩码（line_yellow_double 在 GridMap 渲染为 255,128,0 量级）。

    项目通用 ``paint_like_mask`` 的黄漆窗（|r-g|<=45、g>=135）不覆盖这种
    橙色，因此在试车场域内用色相/饱和度判据；低照度下 val 门放宽到 30，
    只要求色相 0-25、sat>=120、r-b>=80（排除米色地面/雨雾）。
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    r, b = rgb[:, :, 0].astype(np.int16), rgb[:, :, 2].astype(np.int16)
    return ((h >= 0) & (h <= 25) & (s >= 120) & (v >= 30)
            & ((r - b) >= 80))


def _project(cam, pos, heading, pts):
    u, v, ok = cam.project(np.asarray(pts, float), pos, heading)
    return np.asarray(u, float), np.asarray(v, float), np.asarray(ok, bool)


def rasterize_strip(shape, cam, pos, heading, rail_a, rail_b,
                    value: int, mask: np.ndarray) -> None:
    """把两条投影后的轨线围成的条带逐段填进 mask。"""
    ua, va, oka = _project(cam, pos, heading, rail_a)
    ub, vb, okb = _project(cam, pos, heading, rail_b)
    for i in range(len(ua) - 1):
        if not (oka[i] and oka[i + 1] and okb[i] and okb[i + 1]):
            continue
        quad = np.array([[ua[i], va[i]], [ua[i + 1], va[i + 1]],
                         [ub[i + 1], vb[i + 1]], [ub[i], vb[i]]],
                        np.int32)
        cv2.fillPoly(mask, [quad], int(value))


def build_masks(rgb, cam, pos, heading, geom):
    h, w = rgb.shape[:2]
    line = np.zeros((h, w), np.uint8)
    road = np.zeros((h, w), np.uint8)
    for g in geom:
        rasterize_strip((h, w), cam, pos, heading, g["left"], g["right"],
                        2, line)
    # 车道区域：左线外侧到右线外侧（外扩 EXPAND_M）
    left_line, right_line = geom[0], geom[1]
    left_n = (left_line["left"][0, :2] - left_line["center"][0, :2])
    left_n = left_n / max(1e-9, float(np.linalg.norm(left_n)))
    lane_a = left_line["left"][:, :2] + left_n * EXPAND_M
    lane_b = right_line["right"][:, :2] - left_n * EXPAND_M
    lane_a3 = np.column_stack([lane_a, left_line["center"][:, 2]])
    lane_b3 = np.column_stack([lane_b, right_line["center"][:, 2]])
    rasterize_strip((h, w), cam, pos, heading, lane_a3, lane_b3, 1, road)
    road = np.maximum(road, (line > 0).astype(np.uint8))
    return line, road


def gate_frame(rgb, cam, pos, heading, geom, pts) -> dict:
    line, road = build_masks(rgb, cam, pos, heading, geom)
    paint = paint_mask(rgb)
    # 覆盖率 / 残差：真值点 -> 最近漆色像素
    ys, xs = np.nonzero(paint)
    paint_pts = np.column_stack([xs, ys]) if len(xs) else np.zeros((0, 2))
    u, v, ok = _project(cam, pos, heading,
                        [p["world"] for p in pts])
    world = np.asarray([p["world"] for p in pts], float)
    dist = np.linalg.norm(world[:, :2] - np.asarray(pos, float)[:2], axis=1)
    checkable = (dist >= CHECK_MIN_M) & (dist <= CHECK_MAX_M)
    in_frame = (ok & checkable & np.isfinite(u) & np.isfinite(v)
                & (u >= 0) & (u < rgb.shape[1])
                & (v >= 0) & (v < rgb.shape[0]))
    residuals = []
    covered = 0
    color_hits = 0
    if paint_pts.size:
        for uu, vv in zip(u[in_frame], v[in_frame]):
            dd = np.hypot(paint_pts[:, 0] - uu, paint_pts[:, 1] - vv)
            dmin = float(dd.min())
            residuals.append(dmin)
            if dmin <= RADIUS_PX:
                covered += 1
            if dmin <= 2.0:
                color_hits += 1
    denom = int(in_frame.sum())
    coverage = (covered / denom) if denom else 0.0
    med = float(np.median(residuals)) if residuals else float("inf")
    mx = float(np.max(residuals)) if residuals else float("inf")
    line_px = int((line > 0).sum())
    color = (float(color_hits) / denom) if denom else 0.0
    reasons = []
    if coverage < COVERAGE_MIN:
        reasons.append(f"coverage={coverage:.3f}")
    if med > RESIDUAL_MED_MAX_PX:
        reasons.append(f"residual_med={med:.2f}")
    if mx > RESIDUAL_MAX_PX:
        reasons.append(f"residual_max={mx:.2f}")
    if color < COLOR_MIN:
        reasons.append(f"color={color:.3f}")
    return {"ok": not reasons, "reasons": reasons, "line": line, "road": road,
            "coverage": coverage, "residual_med": med, "residual_max": mx,
            "color": color, "line_px": line_px, "truth_points": denom}


def export_cell(site: str, cell: str, *, target_frames: int = 30) -> int:
    # Pilot is diagnostic only: actual built nodes, independent environment
    # verification and the preregistered per-pixel colour gate are not yet
    # wired. Refuse before touching output or existing quarantine evidence.
    raise RuntimeError("BLOCKED: 导出资格链尚未验证；禁止生成训练池")
    src = CAP_DIR / site / cell
    manifest_path = src / "capture.json"
    if not manifest_path.is_file():
        raise SystemExit(f"缺采集清单 {manifest_path}")
    cap = json.loads(manifest_path.read_text(encoding="utf-8"))
    if cap.get("status") != "ok" or not cap.get("env", {}).get("tod_after"):
        raise SystemExit(f"采集未完成/环境无效：{site}/{cell} -> {cap.get('status')}")
    doc = json.loads((SITES_DIR / f"{site}.json").read_text(encoding="utf-8"))
    geom = site_geometry(doc)
    pts = truth_points(geom)
    out_dir = POOL_DIR / site / cell
    q_dir = out_dir / "quarantine"
    out_dir.mkdir(parents=True, exist_ok=True)
    q_dir.mkdir(exist_ok=True)
    frames = sorted((src / "probe.frames").glob("frame_*.npz"))
    passed, rejected, seen = [], [], set()
    # 位姿去重：SHA-1 只按像素算，仿真器光影/雨滴每帧微变，静止帧的哈希
    # 各不相同（2026-11-06 实测：S1/noon_clear 437 帧 437 个不同哈希，
    # 但独立位姿只有 1 个）。所以必须再按位姿距离筛，否则同一位置的
    # 静止画面会被当成 30 个训练样本送进池子。
    last_pose = None
    for fp in frames:
        z = np.load(fp, allow_pickle=True)
        rgb = np.asarray(z["frame"], np.uint8)
        h = hashlib.sha1(rgb.tobytes()).hexdigest()
        if h in seen:
            rejected.append({"file": fp.name, "reasons": ["duplicate"]})
            continue
        seen.add(h)
        cam = z["cam"][0]
        pos = np.asarray(z["pos"], float)
        heading = float(z["heading"])
        if last_pose is not None:
            d_xy = float(np.linalg.norm(pos[:2] - last_pose[:2]))
            d_h = abs(float(heading) - float(last_pose[2]))
            if d_xy < POSE_MIN_XY_M and d_h < POSE_MIN_HEADING_RAD:
                rejected.append({"file": fp.name,
                                 "reasons": [f"pose_repeat({d_xy:.2f}m,"
                                             f"{d_h:.3f}rad)"]})
                continue
        g = gate_frame(rgb, cam, pos, heading, geom, pts)
        if not g["ok"]:
            rejected.append({"file": fp.name, "reasons": g["reasons"]})
            np.savez_compressed(q_dir / fp.name,
                                rgb=rgb, line=g["line"], road=g["road"])
            continue
        passed.append((fp, rgb, g))
        last_pose = (float(pos[0]), float(pos[1]), float(heading))
    # 均匀抽 target_frames 帧
    if len(passed) > target_frames:
        idx = np.linspace(0, len(passed) - 1, target_frames).round().astype(int)
        passed = [passed[i] for i in sorted(set(idx.tolist()))]
    kept = []
    for fp, rgb, g in passed:
        dst = out_dir / f"{site}_{cell}_{len(kept):04d}.npz"
        np.savez_compressed(
            dst, rgb=rgb, line=g["line"], road=g["road"],
            site=site, cell=cell,
            coverage=float(g["coverage"]),
            residual_med=float(g["residual_med"]),
            residual_max=float(g["residual_max"]),
            color=float(g["color"]))
        kept.append({"file": dst.name, "coverage": g["coverage"],
                     "residual_med": g["residual_med"],
                     "residual_max": g["residual_max"],
                     "color": g["color"],
                     "sha1": hashlib.sha1(rgb.tobytes()).hexdigest()})
    manifest = {"schema": "proving_pool_v1", "site": site, "cell": cell,
                "split": cap.get("split"), "source_frames": len(frames),
                "kept": len(kept), "rejected": len(rejected),
                "target_frames": target_frames,
                "gates": {"coverage": COVERAGE_MIN,
                          "residual_med_px": RESIDUAL_MED_MAX_PX,
                          "residual_max_px": RESIDUAL_MAX_PX,
                          "color": COLOR_MIN,
                          "radius_px": RADIUS_PX},
                "frames": kept, "rejections": rejected[:200]}
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in
                      ("site", "cell", "source_frames", "kept",
                       "rejected")}, ensure_ascii=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", required=True, choices=("S1", "S2", "S3", "P"))
    ap.add_argument("--cell", required=True)
    ap.add_argument("--target-frames", type=int, default=30)
    args = ap.parse_args()
    return export_cell(args.site, args.cell,
                       target_frames=int(args.target_frames))


if __name__ == "__main__":
    raise SystemExit(main())
