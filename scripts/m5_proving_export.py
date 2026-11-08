# -*- coding: utf-8 -*-
"""试车场采集帧导出：渲染漆色组件标签 + A1 门禁（预注册修订案 A1 §2–§3）。

**A1 定论（2026-11-07）**：右线 decal 渲染存在固定 -0.85m 横向标定偏移
（向路外），渲染严格跟随节点；左线无偏移。证据与标定记录见
``docs/T16_PROVING_EXPORT_PILOT_BLOCKER_20261105.md`` §A1 结果节。
据此本工具按修订案重写：

1. 几何**绑定** ``probe.built.json`` 的实际节点（不重算；§6.2-5 修复）；
2. 线标签 = **渲染漆色连通域**（paint_mask 连通域按最近投影线分配到
   left/right，宽 2–25px、纵向跨度 ≥10m 的组件才入标签）；
3. 门禁（A1 §3，冻结）：每侧组件纵向跨度 ≥10m、宽度 2–25px、组件中位
   横向偏移 vs 节点 ≤1.0m（记录实测偏移）；真值点近场（5–25m）落入
   1.0m 走廊比例 ≥0.8；原像素残差门（coverage/med/max）只对左线保留，
   右线按 -0.85m 标定豁免；
4. 产物带**溯源哈希**（built/capture/逐帧源 npz 的 SHA-256）；重复导出
   不覆盖——旧产物移入 ``stale/``，quarantine 证据永不删除；
5. 池状态标 ``pilot``、``training_qualified=false``：入池 ≠ 可训练，
   训练消费仍受 §6.2 状态门修复与全格重采约束（§10.5）。

输入 ``logs/m5_seg/proving_capture/<site>/<cell>/probe.frames/frame_*.npz``
（rgb/pos/heading/cam），输出每帧 ``rgb + line(2) + road(1)`` 的 npz 与
manifest。本工具不新增人工标注。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import time
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
EXPAND_M = 1.0          # 车道区域外扩（road 标签）
RADIUS_PX = 6
CHECK_MIN_M = 5.0       # 可核对距离带（近于 5m 出画/远于 25m 线宽 <2px）
CHECK_MAX_M = 25.0

# ── A1 门禁（修订案 §3，冻结，不因结果调整）────────────────────────────
COMPONENT_MAX_ASSIGN_M = 1.5    # 组件归属最近线的最大中位距离（>0.85 偏移+余量）
COMPONENT_THICKNESS_MIN_PX = 2  # 组件垂直厚度下限（雨滴/噪点被此挡住）
COMPONENT_THICKNESS_MAX_PX = 25
COMPONENT_SPAN_MIN_M = 10.0     # 每侧组件纵向（深度带内）跨度
COMPONENT_OFFSET_MAX_M = 1.0    # 组件中位横向偏移 vs 节点（记录实测值）
CORRIDOR_M = 1.0                # 真值点走廊半宽（容纳 -0.85m 标定偏移）
CORRIDOR_MIN_RATIO = 0.8        # 近场真值点落入走廊的比例门
# 左线保留的原冻结像素残差门：
LEFT_COVERAGE_MIN = 0.80
LEFT_RESIDUAL_MED_MAX_PX = 1.0
LEFT_RESIDUAL_MAX_PX = 2.0
# 历史探索阈值，仅供候选规则测试；不是入池资格的一部分。
POSE_MIN_XY_M = 0.5
POSE_MIN_HEADING_RAD = 0.02
DENSIFY_STEP_M = 0.2            # 组件归属用的节点链加密步长


def find_pose_duplicate(pos, heading, accepted, *, min_xy=POSE_MIN_XY_M,
                        min_heading=POSE_MIN_HEADING_RAD):
    """查找全体已接受位姿中的重复项；无匹配返回 None。

    返回 (索引, 平面距离, 最短角差)。距离和角差均小于阈值才重复；
    因此原地转向达到角度阈值可作为不同姿态。此规则供导出去重，
    静止帧（0 位移 0 转角）必须被判为重复。
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


def select_pose_candidates(records, *, min_xy=POSE_MIN_XY_M,
                           min_heading=POSE_MIN_HEADING_RAD):
    """对已通过其他资格门的记录执行候选去重，不读取或写入文件。

    输入按调用方提供的顺序处理；重复关系不具传递性，先接受者优先。
    每条记录包含唯一 source_id、pos 和 heading。
    """
    find_pose_duplicate((0.0, 0.0), 0.0, [], min_xy=min_xy,
                        min_heading=min_heading)
    accepted = []
    rejected = []
    history = []
    identifiers = set()
    for record in records:
        source_id = record["source_id"]
        if not isinstance(source_id, str) or not source_id or source_id in identifiers:
            raise ValueError("source_id must be a unique nonempty string")
        identifiers.add(source_id)
        try:
            match = find_pose_duplicate(record["pos"], record["heading"], history,
                                        min_xy=min_xy, min_heading=min_heading)
        except (ValueError, TypeError, KeyError, OverflowError):
            rejected.append({"source_id": source_id, "reason": "invalid_pose"})
            continue
        if match is None:
            history.append((record["pos"], record["heading"]))
            accepted.append(record)
        else:
            index, distance, angle = match
            rejected.append({"source_id": source_id, "reason": "pose_repeat",
                             "matched_source_id": accepted[index]["source_id"],
                             "distance_m": distance, "angle_rad": angle})
    return accepted, rejected


def _load_scenes_module():
    spec = importlib.util.spec_from_file_location(
        "mcs_export", str(ROOT / "scripts" / "m5_controlled_scenes.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def validate_built_lines(record):
    """Validate recorded built nodes without asserting rendered pixel truth.

    A structurally valid record is still NOT proof that it belongs to the
    current run or that its decals rendered. Caller must independently bind
    the record digest to a fresh capture and check annotation/projection.
    """
    if not isinstance(record, dict) or not isinstance(record.get("lines"), list):
        raise ValueError("missing built lines")
    found = {}
    for line in record["lines"]:
        if not isinstance(line, dict):
            raise ValueError("invalid line record")
        role = line.get("role")
        if role not in ("left", "right") or role in found:
            raise ValueError("invalid or duplicate role")
        try:
            nodes = np.asarray(line["nodes"], dtype=float)
            width = float(line["width_m"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("invalid built geometry") from exc
        if (nodes.ndim != 2 or nodes.shape[1] != 3 or nodes.shape[0] < 2
                or not np.isfinite(nodes).all() or not np.isfinite(width)
                or width <= 0 or not np.any(np.linalg.norm(np.diff(nodes, axis=0), axis=1) > 1e-9)):
            raise ValueError("invalid built geometry")
        truth = []
        raw_truth = line.get("truth_points") or []
        if not isinstance(raw_truth, list):
            raise ValueError("invalid truth points")
        for pt in raw_truth:
            if (not isinstance(pt, dict)
                    or not isinstance(pt.get("world"), list)
                    or len(pt["world"]) != 3):
                raise ValueError("invalid truth points")
            w = np.asarray(pt["world"], dtype=float)
            if not np.isfinite(w).all():
                raise ValueError("invalid truth points")
            truth.append({"world": [float(v) for v in w],
                          "class": pt.get("class"), "role": pt.get("role")})
        found[role] = {"role": role, "nodes": nodes.copy(), "width_m": width,
                       "truth_points": truth}
    if set(found) != {"left", "right"}:
        raise ValueError("both line roles required")
    return [found["left"], found["right"]]


def site_geometry(site_doc: dict) -> list[dict]:
    """站点 JSON -> 两条线的 (role, center, left, right) 世界点。

    保留为**规格几何**参考（与 built.json 的实际节点对照用）；导出标签
    一律以 built 节点为准（A1：不重算）。
    """
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
    """线实例的真值点（规格几何口径，class=2 带 role）。"""
    pts = []
    for line in geom:
        c = line["center"]
        for i in range(len(c) - 1):
            a, b = c[i], c[i + 1]
            n = max(1, int(np.linalg.norm(b - a) / 1.7))
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


def _line_lateral_normal(center: np.ndarray) -> np.ndarray:
    """节点链的横向（左正）单位向量，取中段方向避开两端噪声。"""
    d = np.diff(center[:, :2], axis=0)
    seg = d[np.linalg.norm(d, axis=1) > 1e-6]
    if not len(seg):
        raise ValueError("degenerate node chain")
    fwd = seg[len(seg) // 2]
    fwd = fwd / np.linalg.norm(fwd)
    return np.array([-fwd[1], fwd[0]])


def _densify(center: np.ndarray, step_m: float = DENSIFY_STEP_M) -> np.ndarray:
    """节点链沿折线加密（组件归属/距离用，z 线性插值）。"""
    out = [center[0]]
    for a, b in zip(center[:-1], center[1:]):
        seg = float(np.linalg.norm(b[:2] - a[:2]))
        n = max(1, int(seg / step_m))
        for k in range(1, n + 1):
            out.append(a + (b - a) * (k / n))
    return np.asarray(out, float)


def geometry_from_built(built_lines: list[dict]) -> list[dict]:
    """built 节点 -> 每线 (center, dense, rails, truth_points)。

    rails = center ± 横向法线*(width/2)：road 多边形与条带栅格化用；
    truth_points 直接取 built 记录（不重算，A1）。
    """
    out = []
    for line in built_lines:
        center = line["nodes"]
        normal = _line_lateral_normal(center)
        off = np.zeros_like(center)
        off[:, :2] = normal[None, :] * (float(line["width_m"]) / 2.0)
        out.append({"role": line["role"], "center": center,
                    "dense": _densify(center),
                    "left": center + off, "right": center - off,
                    "normal": normal,
                    "truth_points": line.get("truth_points") or []})
    return out


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


def build_road_mask(rgb_shape, cam, pos, heading, geom) -> np.ndarray:
    """车道区域：左线外侧到右线外侧（外扩 EXPAND_M），built 节点栅格化。"""
    h, w = rgb_shape[:2]
    road = np.zeros((h, w), np.uint8)
    left = next(g for g in geom if g["role"] == "left")
    right = next(g for g in geom if g["role"] == "right")
    n = left["normal"]
    lane_a = left["left"][:, :2] + n[None, :] * EXPAND_M
    lane_b = right["right"][:, :2] - n[None, :] * EXPAND_M
    lane_a3 = np.column_stack([lane_a, left["left"][:, 2]])
    lane_b3 = np.column_stack([lane_b, right["right"][:, 2]])
    rasterize_strip((h, w), cam, pos, heading, lane_a3, lane_b3, 1, road)
    return road


def _component_scan(cam, pos, heading, line, pixels, *,
                    halfwidth: float = 4.0, step: float = 0.1) -> dict:
    """把链按世界横向 δ 平移扫描，找组件像素与平移链的最佳对齐。

    返回 {"delta_m": 世界横向偏移（渲染-节点）， "median_dist_px"/
    "dist_p90_px": 像素到平移链**折线**的精确距离（段投影细化，与链密度
    无关）， "depths": 最近点深度}。
    **为什么用世界 δ 扫描而不是图像最近距离**：斜视角下两条世界平行线的
    图像间距被透视压缩（垂直于图像线方向的分量只占横向 offset 的一小部分），
    图像最近距离换算的"米"会系统性偏小；而 δ 平移后的链与渲染条带在
    同深度重合，折线距离才等于真实厚度，δ 本身才是真实横向偏移
    （与 scripts/m5_proving_diag_a1.py 的 scan 同口径）。
    """
    dense = line["dense"]
    normal = line["normal"]
    deltas = np.arange(-halfwidth, halfwidth + 1e-9, step)
    pos2 = np.asarray(pos, float)[None, :2]
    best = None
    px = pixels
    n = len(dense)
    for delta in deltas:
        shifted = dense.copy()
        shifted[:, :2] = dense[:, :2] + normal[None, :] * float(delta)
        u, v, ok = _project(cam, pos, heading, shifted)
        if not (ok & np.isfinite(u) & np.isfinite(v)).all():
            # 有链点不可投影时退化为点距离（用大数屏蔽无效点）
            uu = np.where(ok & np.isfinite(u), u, 1e9)
            vv = np.where(ok & np.isfinite(v), v, 1e9)
            d2 = (px[:, 0][:, None] - uu[None, :]) ** 2 + \
                 (px[:, 1][:, None] - vv[None, :]) ** 2
            j = np.argmin(d2, axis=1)
            dists = np.sqrt(d2[np.arange(len(px)), j])
            depth = np.linalg.norm(shifted[:, :2] - pos2, axis=1)
        else:
            d2 = (px[:, 0][:, None] - u[None, :]) ** 2 + \
                 (px[:, 1][:, None] - v[None, :]) ** 2
            j = np.argmin(d2, axis=1)
            # 段细化：最近点所在链段（j-1,j）与（j,j+1）的精确折线距离
            ax, ay = u[j], v[j]
            best_d = np.sqrt(d2[np.arange(len(px)), j])
            for k in (j - 1, j):
                lo = np.clip(k, 0, n - 1)
                hi = np.clip(k + 1, 0, n - 1)
                abx, aby = u[hi] - u[lo], v[hi] - v[lo]
                L2 = np.maximum(abx * abx + aby * aby, 1e-9)
                t = np.clip(((px[:, 0] - u[lo]) * abx
                             + (px[:, 1] - v[lo]) * aby) / L2, 0.0, 1.0)
                ex, ey = u[lo] + t * abx, v[lo] + t * aby
                best_d = np.minimum(best_d, np.hypot(px[:, 0] - ex,
                                                     px[:, 1] - ey))
            dists = best_d
            depth = np.linalg.norm(shifted[:, :2] - pos2, axis=1)
        med = float(np.median(dists))
        # depth 是逐链点数组，最近点深度按 j 取（退化分支此前已误索引一次，
        # 2026-11-08 实测 IndexError: index 105 out of bounds for size 7——
        # 近处链点被 _project 判为 depth<=0.1 时进入退化分支即崩）
        p90 = float(np.percentile(dists, 90))
        if best is None or med < best["median_dist_px"]:
            best = {"delta_m": round(float(delta), 3),
                    "median_dist_px": med,
                    "dist_p90_px": p90,
                    "depths": depth[j]}
    return best if best is not None else {"delta_m": None,
                                          "median_dist_px": None,
                                          "dist_p90_px": None,
                                          "depths": None}


def _assign_paint_components(paint: np.ndarray, cam, pos, heading,
                             geom) -> dict:
    """橙色连通域 -> 按 |δ| 最小的投影线归属（A1 §2：最近投影线）。

    每组件对 left/right 各做一次世界 δ 扫描：δ_best = 渲染相对该线节点的
    横向偏移，median_dist_px = 对齐残差（≈条带半厚），深度取最近点深度
    （跨度门用）。归属 = |δ| 更小的那条线。
    """
    n_comp, labels = cv2.connectedComponents(paint.astype(np.uint8),
                                             connectivity=8)
    components = []
    unassigned = 0
    for cid in range(1, n_comp):
        ys, xs = np.nonzero(labels == cid)
        if len(xs) < 4:
            unassigned += 1
            continue
        px = np.column_stack([xs, ys]).astype(float)
        scans = {}
        for g in geom:
            scans[g["role"]] = _component_scan(cam, pos, heading, g, px)
        role = min(scans, key=lambda r: abs(scans[r]["delta_m"]
                                            if scans[r]["delta_m"] is not None
                                            else 1e9))
        best = scans[role]
        if best["delta_m"] is None:
            unassigned += 1
            continue
        # 厚度 = 2×p90(像素到平移链距离)：均匀条带的 p90≈0.45w，中位数会把
        # 宽条带测成一半宽（31px 条带被测成 16px 而漏过 25px 门）
        thickness_px = 2.0 * float(best.get("dist_p90_px")
                                   or best["median_dist_px"])
        depths = best["depths"]
        span_m = (float(np.max(depths) - np.min(depths))
                  if depths is not None and len(depths) else 0.0)
        components.append({"role": role, "pixels": px,
                           "delta_m": best["delta_m"],
                           "median_dist_px": best["median_dist_px"],
                           "thickness_px": thickness_px,
                           "span_m": span_m})
    return {"components": components, "unassigned": unassigned}


def label_and_gate_frame(rgb, cam, pos, heading, geom) -> dict:
    """A1 §2 标签 + §3 门禁。返回 line/road 掩码、reasons 与指标。"""
    h, w = rgb.shape[:2]
    line = np.zeros((h, w), np.uint8)
    paint = paint_mask(rgb)
    ys, xs = np.nonzero(paint)
    paint_pts = (np.column_stack([xs, ys]).astype(float) if len(xs)
                 else np.zeros((0, 2), float))
    reasons = []
    metrics: dict = {}

    assign = _assign_paint_components(paint, cam, pos, heading, geom)
    road = build_road_mask(rgb.shape, cam, pos, heading, geom)

    # 每侧：厚度门内组件入标签；跨度门与偏移门（A1 §3）
    for role in ("left", "right"):
        comps = [c for c in assign["components"] if c["role"] == role]
        good = [c for c in comps
                if COMPONENT_THICKNESS_MIN_PX <= c["thickness_px"]
                <= COMPONENT_THICKNESS_MAX_PX]
        rejected_thick = len(comps) - len(good)
        if (not good
                or max(c["span_m"] for c in good) < COMPONENT_SPAN_MIN_M):
            reasons.append(f"{role}_component_span_or_missing")
        for c in good:
            line[c["pixels"][:, 1].astype(int), c["pixels"][:, 0].astype(int)] = 2
        offsets = [c["delta_m"] for c in good]
        med_off = (round(float(np.median(offsets)), 3) if offsets else None)
        if med_off is None or abs(med_off) > COMPONENT_OFFSET_MAX_M:
            reasons.append(f"{role}_offset={med_off}")
        metrics[f"{role}_component"] = {
            "components": len(comps), "kept": len(good),
            "rejected_by_thickness": rejected_thick,
            "span_m": round(max((c["span_m"] for c in good), default=0.0), 2)}
        metrics[f"{role}_offset_m"] = med_off

    # 真值点走廊门（A1 §3）：横向判据 = 该侧渲染线相对节点的世界偏移
    # δ_role（组件中位，已算出）≤ CORRIDOR_M——远深度处逐点像素容差会
    # 折算成 ±0.65m 的世界距离，逐点 δ 扫描病态，故横向用帧级 δ_role；
    # 纵向判据 = 逐真值点检查渲染线（节点+δ_role）上有漆覆盖。
    pos2 = np.asarray(pos, float)[:2]
    corridor_hits = 0
    corridor_total = 0
    for g in geom:
        role = g["role"]
        normal3 = np.array([g["normal"][0], g["normal"][1], 0.0])
        delta_role = metrics.get(f"{role}_offset_m")
        lateral_ok = (delta_role is not None
                      and abs(delta_role) <= CORRIDOR_M)
        for pt in g["truth_points"]:
            world = np.asarray(pt["world"], float)
            depth = float(np.linalg.norm(world[:2] - pos2))
            if not (CHECK_MIN_M <= depth <= CHECK_MAX_M):
                continue
            if not paint_pts.size:
                continue
            shifted = world + normal3 * float(delta_role or 0.0)
            u, v, ok = _project(cam, pos, heading, [shifted])
            if not ok[0] or not np.isfinite(u[0]):
                continue
            uu, vv = float(u[0]), float(v[0])
            if not (0 <= uu < w and 0 <= vv < h):
                continue
            corridor_total += 1
            if not lateral_ok:
                continue
            dd = np.hypot(paint_pts[:, 0] - uu, paint_pts[:, 1] - vv)
            if float(dd.min()) <= RADIUS_PX:
                corridor_hits += 1
    ratio = (corridor_hits / corridor_total) if corridor_total else 0.0
    metrics["corridor_ratio"] = round(ratio, 3)
    metrics["corridor_points"] = corridor_total
    if corridor_total == 0 or ratio < CORRIDOR_MIN_RATIO:
        reasons.append(f"corridor={ratio:.3f}")

    # 左线保留原冻结像素残差门（coverage/med/max；右线按 -0.85m 标定豁免）
    fx = float(cam.fx)
    left_g = next(g for g in geom if g["role"] == "left")
    left_cov_pts = 0
    left_cov_hits = 0
    left_res = []
    for pt in left_g["truth_points"]:
        world = np.asarray(pt["world"], float)
        depth = float(np.linalg.norm(world[:2] - pos2))
        if not (CHECK_MIN_M <= depth <= CHECK_MAX_M):
            continue
        u, v, ok = _project(cam, pos, heading, [world])
        if not ok[0] or not np.isfinite(u[0]):
            continue
        uu, vv = float(u[0]), float(v[0])
        if not (0 <= uu < w and 0 <= vv < h):
            continue
        left_cov_pts += 1
        if paint_pts.size:
            dd = np.hypot(paint_pts[:, 0] - uu, paint_pts[:, 1] - vv)
            dmin = float(dd.min())
            left_res.append(dmin)
            if dmin <= RADIUS_PX:
                left_cov_hits += 1
    coverage = (left_cov_hits / left_cov_pts) if left_cov_pts else 0.0
    med = float(np.median(left_res)) if left_res else float("inf")
    mx = float(np.max(left_res)) if left_res else float("inf")
    metrics["left_coverage"] = round(coverage, 3)
    metrics["left_residual_med_px"] = round(med, 2)
    metrics["left_residual_max_px"] = round(mx, 2)
    if left_cov_pts == 0 or coverage < LEFT_COVERAGE_MIN:
        reasons.append(f"left_coverage={coverage:.3f}")
    if med > LEFT_RESIDUAL_MED_MAX_PX:
        reasons.append(f"left_residual_med={med:.2f}")
    if mx > LEFT_RESIDUAL_MAX_PX:
        reasons.append(f"left_residual_max={mx:.2f}")
    metrics["unassigned_components"] = assign["unassigned"]

    return {"ok": not reasons, "reasons": reasons, "line": line,
            "road": road, "metrics": metrics}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_attempt_frames(src: Path) -> tuple[Path, dict | None]:
    """定位一格的帧目录与 attempt 结果（§6.2 attempt 布局）。

    3591c22 起每次采集是独立 ``attempt_*`` 目录，帧在
    ``attempt_*/probe.frames``；旧布局 ``<cell>/probe.frames`` 仍兼容
    （历史格）。优先取 EVIDENCE_READY 的最新 attempt——只有过了采集门
    的 attempt 才允许导出，避免把 INCOMPLETE 的残留帧当证据。
    """
    attempts = sorted((p for p in src.glob("attempt_*") if p.is_dir()),
                      key=lambda p: p.stat().st_mtime, reverse=True)
    for att in attempts:
        result = att / "attempt_result.json"
        if not result.is_file():
            continue
        try:
            rec = json.loads(result.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if rec.get("status") != "EVIDENCE_READY":
            continue
        frames = att / "probe.frames"
        if frames.is_dir() and any(frames.glob("frame_*.npz")):
            return frames, rec
    legacy = src / "probe.frames"
    if legacy.is_dir() and any(legacy.glob("frame_*.npz")):
        return legacy, None
    raise SystemExit(
        f"无可用帧 {src}：既没有 EVIDENCE_READY 的 attempt_*/probe.frames，"
        f"也没有旧布局 probe.frames")


def export_cell(site: str, cell: str, built_path: str, *,
                target_frames: int = 30, force: bool = False) -> int:
    """按 A1 门禁把一个采集格导出入池（pilot 状态）。

    几何必须显式给 ``--built``（probe.built.json；noon_rain 等早期格
    用同站点同几何的 noon_clear 记录，溯源记录在 manifest）。
    """
    src = CAP_DIR / site / cell
    capture_path = src / "capture.json"
    if not capture_path.is_file():
        raise SystemExit(f"缺采集清单 {capture_path}")
    cap = json.loads(capture_path.read_text(encoding="utf-8"))
    built_file = Path(built_path)
    if not built_file.is_file():
        raise SystemExit(f"缺构建记录 {built_file}")
    built_doc = json.loads(built_file.read_text(encoding="utf-8"))
    built_lines = validate_built_lines(built_doc)
    geom = geometry_from_built(built_lines)

    frames_dir, attempt_rec = resolve_attempt_frames(src)
    frames = sorted(frames_dir.glob("frame_*.npz"))
    if not frames:
        raise SystemExit(f"无帧文件 {frames_dir}")

    out_dir = POOL_DIR / site / cell
    q_dir = out_dir / "quarantine"
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists() and not force:
        raise SystemExit(f"{manifest_path} 已存在；重导出请加 --force "
                         f"（旧产物会移入 stale/，quarantine 保留）")
    out_dir.mkdir(parents=True, exist_ok=True)
    q_dir.mkdir(exist_ok=True)
    if force and manifest_path.exists():
        stale = out_dir / "stale" / time.strftime("%Y%m%d_%H%M%S")
        stale.mkdir(parents=True, exist_ok=True)
        for old in sorted(out_dir.glob("*.npz")) + [manifest_path]:
            shutil.move(str(old), str(stale / old.name))

    passed, rejected, seen = [], [], set()
    last_pose = None
    for fp in frames:
        z = np.load(fp, allow_pickle=True)
        rgb = np.asarray(z["frame"], np.uint8)
        digest = hashlib.sha1(rgb.tobytes()).hexdigest()
        if digest in seen:
            rejected.append({"file": fp.name, "reasons": ["duplicate"]})
            continue
        seen.add(digest)
        cam = z["cam"][0]
        pos = np.asarray(z["pos"], float)
        heading = float(z["heading"])
        if last_pose is not None:
            d_xy = float(np.linalg.norm(pos[:2] - np.asarray(last_pose[:2])))
            d_h = abs(heading - float(last_pose[2]))
            if d_xy < POSE_MIN_XY_M and d_h < POSE_MIN_HEADING_RAD:
                rejected.append({"file": fp.name,
                                 "reasons": [f"pose_repeat({d_xy:.2f}m,"
                                             f"{d_h:.3f}rad)"]})
                continue
        g = label_and_gate_frame(rgb, cam, pos, heading, geom)
        if not g["ok"]:
            rejected.append({"file": fp.name, "reasons": g["reasons"],
                             "metrics": g["metrics"]})
            np.savez_compressed(q_dir / fp.name,
                                rgb=rgb, line=g["line"], road=g["road"])
            continue
        passed.append((fp, rgb, g))
        last_pose = (float(pos[0]), float(pos[1]), heading)

    if len(passed) > target_frames:
        idx = np.linspace(0, len(passed) - 1, target_frames).round().astype(int)
        passed = [passed[i] for i in sorted(set(idx.tolist()))]
    kept = []
    for fp, rgb, g in passed:
        dst = out_dir / f"{site}_{cell}_{len(kept):04d}.npz"
        np.savez_compressed(
            dst, rgb=rgb, line=g["line"], road=g["road"],
            site=site, cell=cell, metrics=json.dumps(g["metrics"]))
        kept.append({"file": dst.name,
                     "sha256": _sha256_file(dst),
                     "source_npz": fp.name,
                     "source_npz_sha1": hashlib.sha1(rgb.tobytes()).hexdigest(),
                     "metrics": g["metrics"]})

    manifest = {
        "schema": "proving_pool_v2",
        "site": site, "cell": cell,
        "split": cap.get("split"),
        "pool_status": "pilot",
        "training_qualified": False,
        "qualification_note": "pool consumption gated by T16 §6.2 repairs "
                              "and full capture rerun (ROADMAP §10.5)",
        "built_source": {"path": str(built_file),
                         "sha256": _sha256_file(built_file)},
        "capture_source": {"path": str(capture_path),
                           "sha256": _sha256_file(capture_path),
                           "env": cap.get("env"),
                           "env_independently_verified": False},
        "frames_source": {"path": str(frames_dir),
                          "attempt": attempt_rec.get("status") if attempt_rec
                          else "legacy",
                          "attempt_result_sha256": (
                              _sha256_file(frames_dir.parent
                                           / "attempt_result.json")
                              if attempt_rec else None)},
        "source_frames": len(frames),
        "kept": len(kept), "rejected": len(rejected),
        "target_frames": target_frames,
        "gates": {
            "component_span_min_m": COMPONENT_SPAN_MIN_M,
            "component_thickness_px": [COMPONENT_THICKNESS_MIN_PX,
                                       COMPONENT_THICKNESS_MAX_PX],
            "component_offset_max_m": COMPONENT_OFFSET_MAX_M,
            "corridor_m": CORRIDOR_M,
            "corridor_min_ratio": CORRIDOR_MIN_RATIO,
            "left_coverage_min": LEFT_COVERAGE_MIN,
            "left_residual_med_max_px": LEFT_RESIDUAL_MED_MAX_PX,
            "left_residual_max_px": LEFT_RESIDUAL_MAX_PX,
            "right_line_render_offset_m": -0.85,
        },
        "frames": kept, "rejections": rejected[:200],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in
                      ("site", "cell", "source_frames", "kept", "rejected")},
                     ensure_ascii=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", required=True, choices=("S1", "S2", "S3", "P"))
    ap.add_argument("--cell", required=True)
    ap.add_argument("--built", required=True,
                    help="probe.built.json 路径（几何真值源，不重算）")
    ap.add_argument("--target-frames", type=int, default=30)
    ap.add_argument("--force", action="store_true",
                    help="允许重导出：旧 npz/manifest 移入 stale/（quarantine 保留）")
    args = ap.parse_args()
    return export_cell(args.site, args.cell, args.built,
                       target_frames=int(args.target_frames),
                       force=bool(args.force))


if __name__ == "__main__":
    raise SystemExit(main())
