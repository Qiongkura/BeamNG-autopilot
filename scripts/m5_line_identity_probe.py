"""离线验证「单检出线 + 路面右边缘」能否判别线的身份（P1b，无需游戏）。

背景（docs/lateral_reference_diag §19 悬置问题）：strict 模式大量帧只检出
**一条**漆画线，而"这条线是本车道右边界还是道路中线"决定了横向参考是
`s + 半车道宽` 还是 `s − 半车道宽`——判错就把车放到对向侧（实测放宽镜像门
曾把中心漂到 +0.16 m）。corridor 否决轮（town_1789142315）留下一个线索：
路面掩码的右边缘（含路肩）与漆画线的**间距**携带身份信息——

* 若单线是**右车道边界**：线到路缘的间距 = 路肩宽（实测城镇 ~2.3 m）；
* 若单线是**中线**：间距 = 路肩 + 车道宽（实测 ~5.1 m）。

以车道宽（3.5 m）为分界即可判别：``gap = line_lat − road_edge_lat``，
``gap < LANE_W − margin ⇒ 右边界``（中心 = line + 半车道宽）；
``gap > LANE_W + margin ⇒ 中线``（中心 = line − 半车道宽）；其间弃权。

本探针在**钉住 10 集**上用 GT annotation 当真值量化这条规则：每帧取
GT 左/右线中位作为"单边观测"（身份由构造已知），与可行驶栅格右边缘组成
观测 (s, R)，判别后与 GT 车道中心（左右线中点）比误差。输出混淆矩阵、
准确率、弃权率、中心误差分布——**判别力达标才允许进入实车安全门**
（line_lat 居中 + 0 压线/出路面/倒车）。

Usage::

    .venv\\Scripts\\python.exe scripts/m5_line_identity_probe.py ^
        --episodes logs/m5_seg/pinned_town_set.json
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
from beamng_autopilot.vision.detection import back_project
from beamng_autopilot.vision.hydra import FrameContext
from beamng_autopilot.vision.ring import CAMERA_RING, FRONT_MAIN

LANE_W_M = 3.5
LANE_HALF_M = LANE_W_M / 2.0
# gap decision margins around the lane-width boundary (tunable; the report
# prints the gap distributions so the operating point can be re-derived).
GAP_LO_M = LANE_W_M - 0.5     # below: the line is the right boundary
GAP_HI_M = LANE_W_M + 0.5     # above: the line is the centre line
BAND_NEAR_M, BAND_FAR_M = 3.0, 12.0
GRID_RES_M = 0.5
GT_MIN_PX = 400
LABEL_LINE = 2


def _episode_list(arg: str) -> list[Path]:
    p = Path(arg)
    if p.suffix == ".json":
        names = json.loads(p.read_text(encoding="utf-8"))
        base = p.parent.parent / "m5_e2e"
        return [base / n for n in names]
    if os.path.isdir(arg):
        fs = sorted(glob.glob(str(p / "shadow_fsd_*.npz")),
                    key=os.path.getmtime)
        return [Path(f) for f in fs[-10:]]
    return [Path(arg)]


def measure(ep: Path, *, cam_w: int, cam_h: int,
            lo: float, hi: float) -> dict:
    d = np.load(ep, allow_pickle=True)
    if "label" not in d.files or "drivable" not in d.files:
        return {"episode": ep.name, "error": "missing label/drivable"}
    x, y, h = d["x"], d["y"], d["heading"]
    lab, drv = d["label"], d["drivable"]
    mount = next(m for m in CAMERA_RING if m.role == FRONT_MAIN)
    cam = mount.camera_model(cam_w, cam_h)
    n_grid = int(drv.shape[-1])
    extent = 0.5 * n_grid * GRID_RES_M
    ey = extent - (np.arange(n_grid) + 0.5) * GRID_RES_M
    ex = extent - (np.arange(n_grid) + 0.5) * GRID_RES_M
    EY = np.broadcast_to(ey[None, :], (n_grid, n_grid))
    band = np.broadcast_to(((ex >= BAND_NEAR_M) & (ex <= BAND_FAR_M))[:, None],
                           (n_grid, n_grid))
    rng = np.random.default_rng(7)

    obs: list[dict] = []
    for i in range(len(x)):
        pos = np.array([float(x[i]), float(y[i]), 0.0])
        hd = float(h[i])
        fwd = np.array([np.cos(hd), np.sin(hd)])
        left = np.array([-fwd[1], fwd[0]])
        rr, cc = np.where(lab[i] == LABEL_LINE)
        if len(rr) > GT_MIN_PX:
            sel = rng.choice(len(rr), GT_MIN_PX, replace=False)
            rr, cc = rr[sel], cc[sel]
        lats_l: list[float] = []
        lats_r: list[float] = []
        for r_, c_ in zip(rr, cc):
            wp = back_project(float(c_), float(r_), cam, pos, hd, 0.0)
            if wp is None:
                continue
            rel = np.asarray(wp) - pos[:2]
            lon = float(rel @ fwd)
            lat = float(rel @ left)
            if BAND_NEAR_M <= lon <= BAND_FAR_M:
                (lats_l if lat > 0.3 else
                 lats_r if lat < -0.3 else []).append(lat)
        if not (lats_l and lats_r):
            continue
        mask = drv[i].astype(bool) & band
        if int(mask.sum()) < 12:
            continue
        road_edge = float(EY[mask].min())          # rightmost drivable
        s_r, s_l = float(np.median(lats_r)), float(np.median(lats_l))
        centre = 0.5 * (s_r + s_l)                  # GT own-lane centre
        # PER-SIDE rule (zero-wrong by construction): on a right-hand
        # two-lane road the centre line is never to the RIGHT of the ego,
        # so a right-side detection may only be "boundary" (if the
        # shoulder gap is plausible) or abstain - never "centre".  The
        # wide-shoulder tail of right-side gaps overlaps the centre-gap
        # range, and mis-reading it would steer ~3.5 m into oncoming.
        for side, s in (("right", s_r), ("left", s_l)):
            gap = s - road_edge
            if side == "right":
                if gap <= lo:
                    verdict, c_pred = "boundary", s + LANE_HALF_M
                else:
                    verdict, c_pred = "abstain", None
            else:
                if gap >= hi:
                    verdict, c_pred = "centre", s - LANE_HALF_M
                else:
                    verdict, c_pred = "abstain", None
            obs.append({
                "episode": ep.name, "frame": i, "side": side,
                "true": "boundary" if side == "right" else "centre",
                "gap": gap, "verdict": verdict, "centre_true": centre,
                "centre_pred": c_pred,
            })
    return {"episode": ep.name, "n_frames": len(x), "obs": obs}


def measure_detector(ep: Path, *, sem, lo: float, hi: float,
                     cam_w: int, cam_h: int,
                     agg: str = "median") -> dict:
    """Single-side observations from the DETECTOR's markings (not GT).

    Population per frame: paired (both sides -> the pairing path owns it),
    single-right / single-left (this rule's input), none.  A side counts
    as seen when the detector produced >=1 marking with near-field
    (3-12 m) points on that side; the observation is that side's median
    marking lateral.  Road edge comes from the recorded drivable raster,
    and the GT centre (when the annotation has both lines) is used ONLY
    to score the implied centre.
    """
    d = np.load(ep, allow_pickle=True)
    rgb, x, y, h = d["rgb"], d["x"], d["y"], d["heading"]
    drv = d["drivable"]
    lab = d.get("label") if "label" in d.files else None
    mount = next(m for m in CAMERA_RING if m.role == FRONT_MAIN)
    cam = mount.camera_model(cam_w, cam_h)
    n_grid = int(drv.shape[-1])
    extent = 0.5 * n_grid * GRID_RES_M
    ey = extent - (np.arange(n_grid) + 0.5) * GRID_RES_M
    ex = extent - (np.arange(n_grid) + 0.5) * GRID_RES_M
    EY = np.broadcast_to(ey[None, :], (n_grid, n_grid))
    band = np.broadcast_to(((ex >= BAND_NEAR_M) &
                            (ex <= BAND_FAR_M))[:, None], (n_grid, n_grid))
    rng = np.random.default_rng(7)

    obs: list[dict] = []
    pop = {"paired": 0, "single_right": 0, "single_left": 0, "none": 0}
    for i in range(len(x)):
        pos = np.array([float(x[i]), float(y[i]), 0.0])
        hd = float(h[i])
        fwd = np.array([np.cos(hd), np.sin(hd)])
        left = np.array([-fwd[1], fwd[0]])
        out = sem.run(FrameContext(
            frame_rgb=np.asarray(rgb[i], np.uint8), cam=cam, pos=pos,
            heading=hd, ground_z=0.0, role="front_main"))
        marks = list(out.meta.get("markings") or [])
        lat_r: list[float] = []
        lat_l: list[float] = []
        pts_r: list[float] = []
        pts_l: list[float] = []
        for mk in marks:
            w = np.asarray(getattr(mk, "world", None), dtype=float)
            if w.ndim != 2 or w.shape[1] < 2 or len(w) < 2:
                continue
            rel = w[:, :2] - pos[:2]
            lon = rel @ fwd
            lat = rel @ left
            near = lat[(lon >= BAND_NEAR_M) & (lon <= BAND_FAR_M)]
            if len(near) >= 2:
                med = float(np.median(near))
                if med > 0.3:
                    lat_l.append(med)
                    pts_l.extend(float(v) for v in near)
                elif med < -0.3:
                    lat_r.append(med)
                    pts_r.extend(float(v) for v in near)
        mask = drv[i].astype(bool) & band
        if int(mask.sum()) < 12:
            pop["none"] += 1
            continue
        road_edge = float(EY[mask].min())
        gt_c = None
        if lab is not None:
            rr, cc = np.where(lab[i] == LABEL_LINE)
            if len(rr) > GT_MIN_PX:
                sel = rng.choice(len(rr), GT_MIN_PX, replace=False)
                rr, cc = rr[sel], cc[sel]
            gl, gr = [], []
            for r_, c_ in zip(rr, cc):
                wp = back_project(float(c_), float(r_), cam, pos, hd, 0.0)
                if wp is None:
                    continue
                rel = np.asarray(wp) - pos[:2]
                lon = float(rel @ fwd)
                lat = float(rel @ left)
                if BAND_NEAR_M <= lon <= BAND_FAR_M:
                    (gl if lat > 0.3 else
                     gr if lat < -0.3 else []).append(lat)
            if gl and gr:
                gt_c = 0.5 * (float(np.median(gl)) + float(np.median(gr)))
        sides = (("left" if lat_l else "") +
                 ("right" if lat_r else ""))
        if lat_l and lat_r:
            pop["paired"] += 1
            continue
        if not lat_l and not lat_r:
            pop["none"] += 1
            continue
        side = "right" if lat_r else "left"
        pop[f"single_{side}"] += 1
        # observation: the side's point-level mean uses every near-field
        # sample; the median-of-per-marking-medians gives a fragmentary
        # marking the same weight as a long one.
        pool = pts_r if lat_r else pts_l
        s = (float(np.mean(pool)) if agg == "points"
             else float(np.median(lat_r if lat_r else lat_l)))
        gap = s - road_edge
        if side == "right":
            verdict, c_pred = (("boundary", s + LANE_HALF_M)
                               if gap <= lo else ("abstain", None))
            true = "boundary"
        else:
            verdict, c_pred = (("centre", s - LANE_HALF_M)
                               if gap >= hi else ("abstain", None))
            true = "centre"
        obs.append({"episode": ep.name, "frame": i, "side": side,
                    "true": true, "gap": gap, "verdict": verdict,
                    "centre_true": (gt_c if gt_c is not None else 0.0),
                    "centre_pred": c_pred,
                    "gt_scored": gt_c is not None})
    return {"episode": ep.name, "n_frames": len(x), "obs": obs,
            "population": pop}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="line-identity discriminator validation (offline)")
    ap.add_argument("--episodes",
                    default=str(config.LOGS_DIR / "m5_seg"
                                / "pinned_town_set.json"),
                    help="pinned-set JSON / dir / single episode")
    ap.add_argument("--gap-lo", type=float, default=GAP_LO_M)
    ap.add_argument("--gap-hi", type=float, default=GAP_HI_M)
    ap.add_argument("--source", choices=("gt", "detector"), default="gt",
                    help="gt = annotation lines (rule upper bound); "
                         "detector = SemanticHead markings as the live "
                         "path would see them")
    ap.add_argument("--agg", choices=("median", "points"), default="median",
                    help="detector observation: median of per-marking "
                         "medians, or mean over ALL near-field points "
                         "of the side")
    ap.add_argument("--model", default=None,
                    help="segmentation checkpoint for --source detector "
                         "(default: deployed best.pt)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    lo, hi = args.gap_lo, args.gap_hi

    eps = _episode_list(args.episodes)
    if not eps:
        print(f"no episodes at {args.episodes}")
        return 1

    sem = None
    if args.source == "detector":
        from beamng_autopilot.vision.heads.semantic import SemanticHead
        from beamng_autopilot.vision.segmentation import Segmenter
        seg = Segmenter(model_path=args.model)
        sem = SemanticHead(segmenter=seg)
        print(f"detector source: {seg.model_path} (device={seg.device})")

    all_obs: list[dict] = []
    total_pop: dict = {}
    for ep in eps:
        if args.source == "detector":
            r = measure_detector(ep, sem=sem, lo=lo, hi=hi,
                                 cam_w=536, cam_h=403, agg=args.agg)
        else:
            r = measure(ep, cam_w=536, cam_h=403, lo=lo, hi=hi)
        if "error" in r:
            print(f"  {r['episode']}: {r['error']}")
            continue
        all_obs.extend(r["obs"])
        if "population" in r:
            for k, v in r["population"].items():
                total_pop[k] = total_pop.get(k, 0) + v
            print(f"  {r['episode']}: {len(r['obs'])} single-side obs "
                  f"(population so far {total_pop})")
        else:
            print(f"  {r['episode']}: {len(r['obs'])} single-line observations")
    if not all_obs:
        print("no usable observations")
        return 1

    g = np.asarray([o["gap"] for o in all_obs], dtype=float)
    true = np.asarray([o["true"] for o in all_obs])
    verd = np.asarray([o["verdict"] for o in all_obs])
    ct = np.asarray([o["centre_true"] for o in all_obs], dtype=float)
    cp = np.asarray([np.nan if o["centre_pred"] is None
                     else o["centre_pred"] for o in all_obs], dtype=float)

    src_label = ("DETECTOR markings" if args.source == "detector"
                 else "GT annotation lines")
    print(f"\n=== line-identity discriminator on {len(eps)} pinned episodes "
          f"({len(all_obs)} observations, source={src_label}) ===")
    if total_pop:
        n_all = sum(total_pop.values())
        print("  frame population: " +
              ", ".join(f"{k}={v} ({v / max(1, n_all):.1%})"
                        for k, v in total_pop.items()))
    for t in ("boundary", "centre"):
        sel = g[true == t]
        print(f"  true {t:8s}: gap p10/p50/p90 = "
              f"{np.percentile(sel, 10):+.2f}/{np.median(sel):+.2f}/"
              f"{np.percentile(sel, 90):+.2f} m  (n={len(sel)})")

    print(f"\n  per-side rule (zero-wrong): right-side gap <= {lo:.2f} -> boundary; "
          f"left-side gap >= {hi:.2f} -> centre; otherwise abstain")
    for t in ("boundary", "centre"):
        sel = verd[true == t]
        n = len(sel)
        acc = float((sel == t).mean())
        absn = float((sel == "abstain").mean())
        wrong = float((sel == ("centre" if t == "boundary"
                               else "boundary")).mean())
        print(f"  true {t:8s}: correct {acc:6.1%}  wrong {wrong:5.1%}  "
              f"abstain {absn:5.1%}  (n={n})")

    decided = verd != "abstain"
    # detector mode: only observations whose frame had a GT line pair can
    # be scored against a true centre (others carry a 0.0 placeholder).
    scored = np.asarray([bool(o.get("gt_scored", True))
                         for o in all_obs])
    decided &= scored
    if decided.any():
        err = np.abs(cp[decided] - ct[decided])
        print(f"\n  implied centre vs GT centre (decided & GT-scored "
              f"{int(decided.sum())}):"
              f" p50={np.median(err):.2f} m  <=1.2 m: "
              f"{float((err <= 1.2).mean()):.1%}  <=0.5 m: "
              f"{float((err <= 0.5).mean()):.1%}")
    cov_frames = len(all_obs)
    print(f"\n  coverage: {cov_frames} single-line observations "
          f"(road edge available on all of them)")

    out = Path(args.out) if args.out else \
        config.LOGS_DIR / "m5_line_identity_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"episodes": [e.name for e in eps],
         "gap_lo": lo, "gap_hi": hi,
         "n_obs": len(all_obs),
         "obs": all_obs}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"report -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
