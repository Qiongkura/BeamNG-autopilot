"""量开发集（人工标签）的车道几何约定：线像素 → 地面横向米数（T16 §4.4）。

为什么需要它：上一轮自动真值增量实验里身份率升了 0.028 但**角色一致率掉到
0.609**——生成场景用固定 ±1.8 m 车道，与开发集（italy 城镇/环路）的实际车道
约定不一致。这个入口从**人工标签**实测约定：把每帧的线像素按采集侧同一套相机
模型反投影到地面（`z = ego.z - EGO_ORIGIN_GROUND_GAP_M`），算车体坐标下的
有符号横向偏移（左为正），给出分布（分位数 + 直方图）。

用途：把实测的中位/众数写进生成器（`LAT_LANE_HALF`），让生成场景的车道与
目标域一致；报告里同时给出"角色约定"证据（同一帧左右线的 |横向| 配对）。

用法::

    .venv\\Scripts\\python.exe scripts\\m5_lane_geometry_measure.py `
        --pack-dir logs\\experiments\\review_pack_20260926 --trees reviewed_full `
        --out logs\\experiments\\lane_geometry_dev_20260928.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beamng_autopilot import config  # noqa: E402
from beamng_autopilot.vision.ring import camera_ring_models  # noqa: E402

#: 采集侧分辨率（m5_collect_seg_ring.py 的 W, H）
RING_W, RING_H = 536, 403
#: 只量近场（画面下部的行）：远处线像素的横向估计对深度/俯仰极敏感
NEAR_ROW_FRAC = 0.55


def back_project_ground(u, v, cam, pos, heading, *, ground_z):
    """像素 -> 地面点（世界系）；射线与地面平面求交。返回 ``(x, y, z)`` 或 None。"""
    C, r, f, up = cam.camera_pose(np.asarray(pos, dtype=float), float(heading))
    x = (float(u) - cam.cx) / cam.fx
    z = -(float(v) - cam.cy) / cam.fy
    d = r * x + f + up * z          # depth=1 的射线方向（与 project 互逆）
    if abs(float(d[2])) < 1e-9:
        return None
    t = (float(ground_z) - float(C[2])) / float(d[2])
    if t <= 0:
        return None
    return C + t * d


def lateral_of(point, pos, heading) -> float:
    """世界点相对车体的**有符号横向**（左为正，米）。"""
    h = float(heading)
    left = np.array([-np.sin(h), np.cos(h), 0.0])
    return float((np.asarray(point, dtype=float)[:2]
                  - np.asarray(pos, dtype=float)[:2]) @ left[:2])


def measure_frame(label, cam, pos, heading, *, ground_z, min_row_frac: float):
    """一帧的线像素横向偏移列表。"""
    lab = np.asarray(label)
    if lab.ndim != 2 or not lab.size:
        return []
    h, w = lab.shape
    rows, cols = np.nonzero(lab == 2)
    keep = rows >= int(h * float(min_row_frac))
    rows, cols = rows[keep], cols[keep]
    out = []
    for y, x in zip(rows.tolist(), cols.tolist()):
        p = back_project_ground(x, y, cam, pos, heading, ground_z=ground_z)
        if p is None:
            continue
        out.append(lateral_of(p, pos, heading))
    return out


def summarize(values: list) -> dict:
    a = np.asarray([v for v in values if v is not None], dtype=float)
    if not a.size:
        return {"n": 0}
    edges = [-6.0, -4.0, -3.0, -2.5, -2.0, -1.75, -1.5, -1.25, -1.0, -0.75,
             -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0,
             2.5, 3.0, 4.0, 6.0]
    hist, _ = np.histogram(a, bins=edges)
    return {"n": int(a.size),
            "p05": round(float(np.percentile(a, 5)), 3),
            "p25": round(float(np.percentile(a, 25)), 3),
            "p50": round(float(np.percentile(a, 50)), 3),
            "p75": round(float(np.percentile(a, 75)), 3),
            "p95": round(float(np.percentile(a, 95)), 3),
            "abs_p50": round(float(np.percentile(np.abs(a), 50)), 3),
            "hist": {f"{edges[i]:+.2f}..{edges[i + 1]:+.2f}":
                     int(hist[i]) for i in range(len(hist)) if hist[i]}}


def measure_dir(d: Path, *, view: str, width: int, height: int,
                min_row_frac: float) -> dict:
    meta = None
    for cand in (d / "meta.json", d.parent / "meta.json"):
        if cand.is_file():
            meta = json.loads(cand.read_text(encoding="utf-8"))
            break
    if not meta:
        return {"dir": str(d), "error": "no meta.json"}
    cam = camera_ring_models(width, height).get(view)
    if cam is None:
        return {"dir": str(d), "error": f"no camera model for view {view!r}"}
    vals: list = []
    n_frames = n_with_line = 0
    for fr in (meta.get("frames") or []):
        if str(fr.get("view") or view) != view:
            continue
        p = Path(str(fr.get("path") or ""))
        npz = d / p
        if not npz.is_file():
            continue
        pos = fr.get("pos")
        heading = fr.get("heading")
        if pos is None or heading is None:
            continue
        n_frames += 1
        z = np.load(npz, allow_pickle=False)
        if "label" not in z.files:
            continue
        lab = np.asarray(z["label"])
        if int((lab == 2).sum()) == 0:
            continue
        n_with_line += 1
        ground_z = float(np.asarray(pos, dtype=float)[2]) - \
            float(config.EGO_ORIGIN_GROUND_GAP_M)
        vals += measure_frame(lab, cam, pos, heading, ground_z=ground_z,
                              min_row_frac=min_row_frac)
    return {"dir": str(d), "n_frames": n_frames, "n_frames_with_line": n_with_line,
            "lateral": summarize(vals)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack-dir", required=True)
    ap.add_argument("--trees", nargs="+", default=["reviewed_full"])
    ap.add_argument("--view", default="front_main")
    ap.add_argument("--width", type=int, default=RING_W)
    ap.add_argument("--height", type=int, default=RING_H)
    ap.add_argument("--min-row-frac", type=float, default=NEAR_ROW_FRAC)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pack = Path(args.pack_dir)
    dirs = []
    for tree in args.trees:
        for d in sorted((pack / tree).glob(f"*/{args.view}")):
            dirs.append(d)
    out = {"pack_dir": str(pack), "view": args.view,
           "width": args.width, "height": args.height,
           "min_row_frac": args.min_row_frac,
           "ground_gap_m": float(config.EGO_ORIGIN_GROUND_GAP_M),
           "dirs": [], "pooled": None}
    pooled: list = []
    for d in dirs:
        rec = measure_dir(d, view=args.view, width=args.width,
                          height=args.height, min_row_frac=args.min_row_frac)
        out["dirs"].append(rec)
        print(f"[lane-geom] {d.parent.name}: 帧 {rec.get('n_frames')}"
              f"（有线 {rec.get('n_frames_with_line')}）| 横向 "
              f"{json.dumps(rec.get('lateral') or {}, ensure_ascii=False)[:160]}",
              flush=True)
    # 池化：把每个目录的原始值合起来再统计（各目录帧数不同，不平均比率）
    all_vals: list = []
    for d in dirs:
        rec = measure_dir(d, view=args.view, width=args.width,
                          height=args.height, min_row_frac=args.min_row_frac)
        # 重新取原始值（上面已经算过；这里直接按同一路径再取一次代价小）
        all_vals += _raw_values(d, view=args.view, width=args.width,
                               height=args.height,
                               min_row_frac=args.min_row_frac)
    out["pooled"] = summarize(all_vals)
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False),
                              encoding="utf-8")
    print(f"[lane-geom] 池化: {json.dumps(out['pooled'], ensure_ascii=False)[:400]}")
    print(f"[lane-geom] -> {args.out}")
    return 0


def _raw_values(d: Path, *, view: str, width: int, height: int,
                min_row_frac: float) -> list:
    meta = None
    for cand in (d / "meta.json", d.parent / "meta.json"):
        if cand.is_file():
            meta = json.loads(cand.read_text(encoding="utf-8"))
            break
    if not meta:
        return []
    cam = camera_ring_models(width, height).get(view)
    if cam is None:
        return []
    vals: list = []
    for fr in (meta.get("frames") or []):
        if str(fr.get("view") or view) != view:
            continue
        npz = d / Path(str(fr.get("path") or ""))
        if not npz.is_file() or fr.get("pos") is None or fr.get("heading") is None:
            continue
        z = np.load(npz, allow_pickle=False)
        if "label" not in z.files:
            continue
        ground_z = float(np.asarray(fr["pos"], dtype=float)[2]) - \
            float(config.EGO_ORIGIN_GROUND_GAP_M)
        vals += measure_frame(np.asarray(z["label"]), cam, fr["pos"],
                              fr["heading"], ground_z=ground_z,
                              min_row_frac=min_row_frac)
    return vals


if __name__ == "__main__":
    raise SystemExit(main())
