"""离线测量：铺装路面边界参考在真实录制帧上到底可用吗？

数据源：`logs/m5_e2e/shadow_fsd_*.npz`（实车 shadow 录制）。每帧带
`rgb`（真实前视帧）、`label`（**当帧模型输出**：1=铺装路面，2=标线）、
`x/y/heading`、`lane_src`（当时选中的横向源）。

本探针只做测量，不训练、不改运行时：

1. 用 `paved_edge_lane_center` 在每帧上跑铺装边界候选，统计可用率、
   放弃原因、宽度分布；
2. 在**同一帧**里用 `label==2` 反投影出漆画线的横向位置，比较
   铺装右边界 vs 漆画线：若铺装右边界比漆画线远出很多，说明路面掩码
   溢出到了土肩（这正是历史上 corridor 候选被证伪的原因）；
3. 分桶给出「当时 source=perception-unavailable 的帧里，铺装候选能救回
   多少」。

    用法：
        .venv\\Scripts\\python.exe scripts\\m5_paved_edge_probe.py \\
            --episode logs/m5_e2e/shadow_fsd_*.npz [--limit 200] [--json out]
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beamng_autopilot.lane.pavement import paved_edge_lane_center
from beamng_autopilot.vision import lanes as vision_lanes
from beamng_autopilot.vision.ring import camera_ring_models


def _cam_for(w: int, h: int):
    return camera_ring_models(int(w), int(h))["front_main"]


def _line_lats(line_mask, cam, pos, heading, ground_z=0.0,
               near_m=18.0, max_lat_m=8.0):
    """Painted-line lateral offsets (left = +) from the frame's line mask."""
    try:
        mask = (np.asarray(line_mask, dtype=np.uint8) * 255)
        marks = vision_lanes._mask_to_markings(
            mask, "white", cam, pos, heading, ground_z=ground_z)
    except Exception:
        return []
    fwd = np.array([np.cos(float(heading)), np.sin(float(heading))])
    left = np.array([-fwd[1], fwd[0]])
    p2 = np.asarray(pos[:2], dtype=float)
    out = []
    for m in marks or []:
        wpts = np.asarray(getattr(m, "world", []), dtype=float)
        if wpts.ndim != 2 or len(wpts) == 0:
            continue
        rel = wpts[:, :2] - p2[None, :]
        lon = rel @ fwd
        lat = rel @ left
        sel = (lon >= -2.0) & (lon <= near_m) & (np.abs(lat) <= max_lat_m)
        if int(sel.sum()) >= 3:
            out.append({"kind": str(getattr(m, "kind", "")),
                        "lat": round(float(np.median(lat[sel])), 3),
                        "lon": round(float(np.median(lon[sel])), 2)})
    return out


def _quant(vals, qs=(10, 50, 90)):
    if not len(vals):
        return None
    a = np.asarray(vals, dtype=float)
    return {f"p{q}": round(float(np.percentile(a, q)), 3) for q in qs}


def run_episode(path: Path, limit: int, stride: int, debug_out: bool):
    d = np.load(path, allow_pickle=True)
    n = len(d["x"])
    n = min(n, limit) if limit else n
    idx = range(0, n, max(1, stride))
    x = d["x"]
    y = d["y"]
    hd = d["heading"]
    rgb = d["rgb"]
    lab = d["label"]
    src = d["lane_src"]
    h, w = int(rgb.shape[1]), int(rgb.shape[2])
    cam = _cam_for(w, h)
    rows = []
    for i in idx:
        # Ground plane at z=0 with the ego origin 0.17 m above it: only
        # the HEIGHT DIFFERENCE reaches the projection, so a recorded
        # episode needs no elevation to be replayed exactly on flat
        # ground.
        pos = (float(x[i]), float(y[i]), 0.17)
        heading = float(hd[i])
        road = lab[i] == 1
        line = lab[i] == 2
        dbg: dict = {}
        ref = paved_edge_lane_center(road, cam, pos, heading,
                                     ground_z=0.0, debug=dbg)
        lats = _line_lats(line, cam, pos, heading)
        # Only real paint kinds (the classic/learned extractor also emits
        # weak fragments) - grouped by side.
        neg = [m["lat"] for m in lats if m["lat"] < -0.3]
        pos_ = [m["lat"] for m in lats if m["lat"] > 0.3]
        row = {
            "i": int(i),
            "src": str(src[i]),
            "mode": dbg.get("mode"),
            "bands": dbg.get("usable_bands"),
            "span": dbg.get("span_med_m"),
            "paved": ref is not None,
            "right_lat": (None if ref is None
                          else round(ref.meta["paved_right_lat_m"], 3)),
            "target_lat": (None if ref is None
                           else round(ref.meta["paved_first_lat_m"], 3)),
            "first_lon": (None if ref is None
                          else ref.meta["paved_first_lon_m"]),
            "line_right": (min(neg) if neg else None),
            "line_left": (max(pos_) if pos_ else None),
            "n_line": len(lats),
        }
        if ref is not None and neg:
            # The paint closest to the paved right edge (the right-hand
            # boundary line, if painted at all).
            row["edge_vs_paint_m"] = round(row["right_lat"]
                                           - max(neg), 3)
        rows.append(row)
        if debug_out and ref is None:
            print(f"  frame {i}: abstain {dbg.get('mode')} {dbg}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="铺装边界候选离线测量")
    ap.add_argument("--episode", nargs="+", required=True,
                    help="shadow episode npz（支持通配符）")
    ap.add_argument("--limit", type=int, default=0, help="每段最多取多少帧")
    ap.add_argument("--stride", type=int, default=1, help="帧抽样步长")
    ap.add_argument("--json", type=str, default=None, help="明细输出文件")
    ap.add_argument("--show-abstains", action="store_true",
                    help="打印放弃原因样例")
    args = ap.parse_args()

    paths: list[Path] = []
    for pat in args.episode:
        paths.extend(sorted(Path(p) for p in glob.glob(pat)))
    if not paths:
        print("[paved-probe] 没找到 episode")
        return 2
    all_rows: list[dict] = []
    for p in paths:
        try:
            rows = run_episode(p, args.limit, args.stride,
                               args.show_abstains)
        except Exception as exc:
            print(f"[paved-probe] {p.name}: 读取失败 {exc}")
            continue
        ok = sum(1 for r in rows if r["paved"])
        print(f"[paved-probe] {p.name}: {len(rows)} 帧, "
              f"铺装候选可用 {ok} ({100.0 * ok / max(1, len(rows)):.1f}%)")
        for r in rows:
            r["episode"] = p.name
        all_rows.extend(rows)

    n = len(all_rows)
    if not n:
        return 0
    ok = [r for r in all_rows if r["paved"]]
    print(f"\n== 汇总 {len(paths)} 段 / {n} 帧 ==")
    print(f"铺装候选可用: {len(ok)} ({100.0 * len(ok) / n:.1f}%)")
    modes: dict[str, int] = {}
    for r in all_rows:
        if not r["paved"]:
            modes[str(r["mode"])] = modes.get(str(r["mode"]), 0) + 1
    print("放弃原因: " + ", ".join(
        f"{k}={v}" for k, v in sorted(modes.items(),
                                      key=lambda kv: -kv[1])))
    print("铺装跨度 m:", _quant([r["span"] for r in ok]))
    print("目标横向 m(左+) :", _quant([r["target_lat"] for r in ok]))
    print("起始可用距离 m:", _quant([r["first_lon"] for r in ok]))
    # 关键安全指标：铺装右边界 vs 同帧漆画线（右侧最近的线）
    dv = [r["edge_vs_paint_m"] for r in ok if r.get("edge_vs_paint_m")
          is not None]
    print(f"铺装右边界 − 右侧漆画线（{len(dv)} 帧）:", _quant(dv))
    if dv:
        arr = np.asarray(dv, dtype=float)
        for tol in (0.3, 0.6, 1.0, 1.9):
            print(f"  |差| <= {tol} m: {100.0 * float((np.abs(arr) <= tol).mean()):.1f}%")
    # 无标线帧（没有漆画线）的可用率——用户真正关心的场景
    no_paint = [r for r in all_rows if not r["n_line"]]
    ok_np = [r for r in no_paint if r["paved"]]
    print(f"\n无漆画线帧: {len(no_paint)}，其中铺装候选可用 "
          f"{len(ok_np)} ({100.0 * len(ok_np) / max(1, len(no_paint)):.1f}%)")
    unavail = [r for r in all_rows if r["src"] != "sensor"]
    ok_un = [r for r in unavail if r["paved"]]
    print(f"当时 source != sensor 的帧: {len(unavail)}，铺装候选可用 "
          f"{len(ok_un)} ({100.0 * len(ok_un) / max(1, len(unavail)):.1f}%)")
    if args.json:
        Path(args.json).write_text(
            json.dumps(all_rows, ensure_ascii=False), encoding="utf-8")
        print(f"[paved-probe] 明细 -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
