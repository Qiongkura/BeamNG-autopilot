"""离线评估"候选横向参考"：覆盖率 + 中心是否落在本车道（无需游戏）。

背景：strict FSD 的横向参考只在约 26% 的帧里能靠"双侧漆画线配对"成立，
其余帧失败关闭停车。自然的想法是找一个不需要配对的替代参考（可行驶带
中心、道路右边缘等）。本探针用**已录制的影子集**直接量化这些候选，判据与
``m5_seg_task_eval.py`` 一致：本车道中心 = 标线右侧半车道宽（右行交通）=
车体坐标系 −1.75 m，容差 ±1.2 m。

它消费 episode 里已存的 BEV 栅格（``drivable`` / ``bev`` 占用），因此不需要
游戏、不需要重跑模型：

* ``drivable_center``      - 前向带内所有可行驶格的中位横向（= 整条路的中线）
* ``drivable_right_edge``  - 前向带内最右可行驶格（= 道路右边缘）
* ``right_edge_plus_half`` - 道路右边缘 + 半车道宽（"本车道 = 最右车道"假设）
* ``obstacle_right_wall``  - 前向带内最右占用格（物理右墙/护栏）

结论（2026-09-11，城镇影子集，in_lane 判据目标已修正为 0）：

* ``drivable_center`` 中位横向 ≈ **+1.75 m**（= 道路中线，车在右车道），
  **不是本车道中心**——与 ``fsd_stack.py`` "BEV 可行驶带中心就是道路中线、
  车不能骑"的注释一致。"可行驶带中心"这条替代路线被否证。
* ``obstacle_right_wall`` 在 −14.75 m（BEV 栅格边缘），城镇段没有可用的
  右侧物理边界。
* ``right_edge_plus_half``（= 可行驶右边缘 + 半车道宽）在**双车道**段给出
  ≈ 0.00 m，正是本车道中心；但在 13 m 宽的多车道路段给出 −4.50 m 而失效
  ——它隐含"本车在最右车道、且右边缘就是本车道右边界"，仅在双车道路成立。

**实车否决（2026-09-12，town_1789142315）**：``right_edge_plus_half`` 的
假设在实车上不成立——城镇路面掩码的右边缘（−3.75 m）越过真值右车道线
（−1.43 m）向右多出 ~1.9 m 路肩，候选指向真值中心右侧 1.24 m（52.7% 帧
误差 >1.2 m）。总带宽门控无法区分"双车道"与"双车道+右路肩"（总宽相同），
该假设运行时不可验证 → **结构上不可靠，不要启用**（`bev_corridor_lane_center`
docstring 有完整记录）。离线 in-lane 数字（含本探针的 95.1%）不构成上线
依据。

Usage::

    .venv\\Scripts\\python.exe scripts\\m5_lateral_ref_probe.py --episodes 3
    .venv\\Scripts\\python.exe scripts\\m5_lateral_ref_probe.py ^
        --episode-names shadow_fsd_1789138050_20260911_224931.npz --out logs\\latref.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from beamng_autopilot import config

# The ego-lane centre reads ~0 m in the car frame when the car sits in its
# own lane (left positive).  Same contract as m5_seg_task_eval.py, which was
# corrected from -1.75 to 0.0 on 2026-09-11: -1.75 is the "centred" value of
# a single-line offset (``line_lat``), not of a lane centre.
EGO_LANE_CENTRE_M = 0.0
IN_LANE_TOL_M = 1.2
# BEV grid resolution used by FSDStack (occupancy.py default).
GRID_RES_M = 0.5
# Forward band where the lane reference actually steers the car.
BAND_NEAR_M = 3.0
BAND_FAR_M = 15.0
# Minimum evidence cells before a candidate counts as "available".
MIN_CELLS = 12
# Occupancy above this is obstacle evidence (bev raster is 0..1).
OBSTACLE_LEVEL = 0.5
LANE_HALF_M = 1.75


def _ego_axes(n: int, res: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Ego-frame (x forward, y left) centre of every cell.

    ``OccupancyGrid``: row 0 is the most forward (+x), col 0 the most
    leftward (+y); ``ego_to_cell`` maps ``r = (extent - ex)/res``.
    """
    extent = 0.5 * n * res
    ex = extent - (np.arange(n) + 0.5) * res
    ey = extent - (np.arange(n) + 0.5) * res
    return ex, ey, np.broadcast_to(ex[:, None], (n, n)), \
        np.broadcast_to(ey[None, :], (n, n))


def _candidate_stats(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0, "lat_med": None, "in_lane": None}
    a = np.asarray(vals, dtype=float)
    in_lane = np.mean(np.abs(a - EGO_LANE_CENTRE_M) < IN_LANE_TOL_M)
    return {"n": len(a), "lat_med": round(float(np.median(a)), 3),
            "lat_mean": round(float(a.mean()), 3),
            "in_lane": round(float(in_lane), 3)}


def measure_episode(ep: str, *, res: float = GRID_RES_M) -> dict:
    d = np.load(ep, allow_pickle=True)
    drv = d["drivable"] if "drivable" in d.files else None
    bev = d["bev"] if "bev" in d.files else None
    src = d["lane_src"] if "lane_src" in d.files else None
    if drv is None:
        return {"episode": os.path.basename(ep), "error": "no drivable raster"}

    n = int(drv.shape[-1])
    ex, _ey, EX, EY = _ego_axes(n, res)
    band = (ex >= BAND_NEAR_M) & (ex <= BAND_FAR_M)
    band2 = np.broadcast_to(band[:, None], (n, n))

    cands: dict[str, list[float]] = {k: [] for k in (
        "drivable_center", "drivable_right_edge", "right_edge_plus_half",
        "obstacle_right_wall")}
    widths: list[float] = []
    # Per-frame (band width, right_edge_plus_half lat) so the pooled
    # analysis can answer WHERE the "rightmost lane" assumption holds.
    width_lat: list[tuple[float, float]] = []
    avail: dict[str, int] = {k: 0 for k in cands}
    frames = 0
    src_hist: dict[str, int] = {}

    for i in range(len(drv)):
        frames += 1
        if src is not None:
            k = str(src[i])
            src_hist[k] = src_hist.get(k, 0) + 1

        mask = drv[i].astype(bool) & band2
        n_drv = int(mask.sum())
        if n_drv >= MIN_CELLS:
            eyy = EY[mask]
            widths.append(float(eyy.max() - eyy.min()))
            cands["drivable_center"].append(float(np.median(eyy)))
            avail["drivable_center"] += 1
            right_edge = float(eyy.min())
            cands["drivable_right_edge"].append(right_edge)
            avail["drivable_right_edge"] += 1
            re_half = right_edge + LANE_HALF_M
            cands["right_edge_plus_half"].append(re_half)
            width_lat.append((float(eyy.max() - eyy.min()), re_half))
            avail["right_edge_plus_half"] += 1

        if bev is not None:
            obs = (bev[i] >= OBSTACLE_LEVEL) & band2
            if int(obs.sum()) >= 1:
                obs_r = float(EY[obs].min())    # rightmost obstacle
                if obs_r < -0.5:                # right of the car
                    cands["obstacle_right_wall"].append(obs_r)
                    avail["obstacle_right_wall"] += 1

    out: dict = {
        "episode": os.path.basename(ep),
        "frames": frames,
        "band_m": [BAND_NEAR_M, BAND_FAR_M],
        "drivable_width_m_p50": (round(float(np.median(widths)), 2)
                                 if widths else None),
        "recorded_lane_src": src_hist,
        "candidates": {},
    }
    if width_lat:
        out["width_lat_n"] = len(width_lat)
        out["_width_lat"] = [(round(w, 2), round(v, 2)) for w, v in width_lat]
    for k, vals in cands.items():
        st = _candidate_stats(vals)
        st["coverage"] = round(avail[k] / max(1, frames), 3)
        out["candidates"][k] = st
    return out


def _print_report(r: dict) -> None:
    print(f"\n=== {r['episode']} ===")
    if "error" in r:
        print(f"  {r['error']}")
        return
    print(f"  frames {r['frames']}   band {r['band_m'][0]}-{r['band_m'][1]} m   "
          f"drivable width p50 {r['drivable_width_m_p50']} m")
    print(f"  recorded lane_src: {r['recorded_lane_src']}")
    print(f"  {'candidate':22s} {'cover':>6} {'lat_p50':>8} {'lat_mean':>9} "
          f"{'in_lane':>8}")
    for k, v in r["candidates"].items():
        lat = "-" if v["lat_med"] is None else f"{v['lat_med']:+.2f}"
        mean = "-" if v["lat_mean"] is None else f"{v['lat_mean']:+.2f}"
        inl = "-" if v["in_lane"] is None else f"{v['in_lane']:.1%}"
        print(f"  {k:22s} {v['coverage']:>6.1%} {lat:>8} {mean:>9} {inl:>8}")
    print(f"  (in_lane = |lat - ({EGO_LANE_CENTRE_M:+.2f})| < {IN_LANE_TOL_M} m)")


def main() -> int:
    ap = argparse.ArgumentParser(description="offline lateral-reference probe")
    ap.add_argument("--data", type=str, default=None,
                    help="episode directory (default logs/m5_e2e)")
    ap.add_argument("--pattern", type=str, default="shadow_fsd_*.npz")
    ap.add_argument("--episodes", type=int, default=3,
                    help="how many NEWEST episodes (ignored when --episode-names)")
    ap.add_argument("--episode-names", type=str, default=None,
                    help="comma-separated pinned episode file names "
                         "(shadow sets drift - pin them for comparability)")
    ap.add_argument("--res", type=float, default=GRID_RES_M,
                    help="BEV grid resolution in metres")
    ap.add_argument("--out", type=str, default=None,
                    help="write the report JSON here")
    args = ap.parse_args()

    data_dir = Path(args.data) if args.data else config.LOGS_DIR / "m5_e2e"
    if args.episode_names:
        eps = [str(data_dir / n.strip())
               for n in args.episode_names.split(",") if n.strip()]
    else:
        fs = sorted(glob.glob(str(data_dir / args.pattern)),
                    key=os.path.getmtime)
        if not fs:
            print(f"no episodes matching {args.pattern} in {data_dir}")
            return 1
        eps = fs[-max(1, args.episodes):]

    reports = [measure_episode(ep, res=args.res) for ep in eps]
    for r in reports:
        _print_report(r)

    if len(reports) > 1:
        ok = [r for r in reports if "error" not in r]
        tot = sum(r["frames"] for r in ok)
        print(f"\n=== ALL {len(ok)} EPISODES (frames={tot}) ===")
        for k in ("drivable_center", "drivable_right_edge",
                  "right_edge_plus_half", "obstacle_right_wall"):
            vals = [r["candidates"][k] for r in ok]
            n = sum(v["n"] for v in vals)
            print(f"  {k:22s} coverage {n / max(1, tot):>6.1%}")

        # Where does the "ego lane = rightmost lane" assumption hold?
        # Pool every (band width, right_edge_plus_half) frame and bucket by
        # width: a width gate on this candidate must come from this table,
        # not from intuition - on wide roads the right edge is NOT the ego
        # lane's right boundary and the candidate steers off the road.
        pairs = [(w, v) for r in ok for w, v in r.get("_width_lat", [])]
        if pairs:
            buckets = ((0.0, 7.0), (7.0, 8.0), (8.0, 9.0), (9.0, 10.0),
                       (10.0, 12.0), (12.0, 99.0))
            print(f"\n  right_edge_plus_half by drivable band width "
                  f"(n={len(pairs)} frames, in_lane vs {EGO_LANE_CENTRE_M:+.2f}):")
            for lo, hi in buckets:
                sel = np.asarray([v for w, v in pairs if lo <= w < hi],
                                 dtype=float)
                if not len(sel):
                    continue
                inl = float(np.mean(np.abs(sel - EGO_LANE_CENTRE_M)
                                    < IN_LANE_TOL_M))
                print(f"      width {lo:4.1f}-{hi:<4.1f} m  n={len(sel):5d}  "
                      f"in_lane {inl:6.1%}  lat_p50 {np.median(sel):+.2f} m")

    out = (Path(args.out) if args.out
           else config.LOGS_DIR / "m5_lateral_ref_probe.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reports, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print(f"report -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
