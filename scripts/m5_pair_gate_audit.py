"""离线审计配对几何门：「两侧都在却不配对」的帧强制配对值不值（无需游戏）。

钉住 10 集 + hand 模型的漏斗归因（``m5_lane_continuity.py --diagnose``，
lateral_reference_diag §25）显示最大可疑损失是 ``both_sides_but_no_pair``
（两侧候选都活过门、``pair_lane_markings`` 仍拒绝，525 帧 / 2102），其次
``pair_rejected_mirror_refused``（32 帧）。本审计对这两类帧取存活候选的
左/右近场中位、**强制中点**作为车道中心，与 GT 车道中心（annotation 左右
线中点，修正后口径 0）比误差；已接受的配对帧作为参照系一起报。

判读：强制中点误差小 → 几何门过严，这部分帧可回收（配对率 33%→~58% 的
空间）；误差大 → 门拒得对，损失归检测/几何质量。**本工具只测量，不改任何
驾驶行为；集成需另过实车安全门。**

Usage::

    .venv\\Scripts\\python.exe scripts/m5_pair_gate_audit.py ^
        --episodes logs/m5_seg/pinned_town_set.json ^
        --model logs/m5_seg/seg_model_hand/best_task.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from beamng_autopilot import config
from beamng_autopilot.lane import pair_lane_markings
from beamng_autopilot.vision.detection import back_project
from beamng_autopilot.vision.heads.semantic import SemanticHead
from beamng_autopilot.vision.hydra import FrameContext
from beamng_autopilot.vision.ring import CAMERA_RING, FRONT_MAIN
from beamng_autopilot.vision.segmentation import Segmenter

BAND_NEAR_M, BAND_FAR_M = 3.0, 12.0
GRID_RES_M = 0.5
LABEL_LINE = 2
GT_MIN_PX = 400
SIDE_BAND_M = 0.3


def _episode_list(arg: str) -> list[Path]:
    p = Path(arg)
    if p.suffix == ".json":
        names = json.loads(p.read_text(encoding="utf-8"))
        return [p.parent.parent / "m5_e2e" / n for n in names]
    return [Path(arg)]


def _gt_centre(label, pos, hd, cam, rng) -> tuple[float | None, float | None]:
    """GT own-lane centre + right-line lateral from annotation paint."""
    if label is None:
        return None, None
    rr, cc = np.where(label == LABEL_LINE)
    if len(rr) < GT_MIN_PX:
        return None, None
    sel = rng.choice(len(rr), GT_MIN_PX, replace=False)
    rr, cc = rr[sel], cc[sel]
    fwd = np.array([np.cos(hd), np.sin(hd)])
    left = np.array([-fwd[1], fwd[0]])
    gl, gr = [], []
    for r_, c_ in zip(rr, cc):
        wp = back_project(float(c_), float(r_), cam, pos, hd, 0.0)
        if wp is None:
            continue
        rel = np.asarray(wp) - pos[:2]
        lon = float(rel @ fwd)
        lat = float(rel @ left)
        if BAND_NEAR_M <= lon <= BAND_FAR_M:
            (gl if lat > SIDE_BAND_M else
             gr if lat < -SIDE_BAND_M else []).append(lat)
    if not (gl and gr):
        return None, None
    return (0.5 * (float(np.median(gl)) + float(np.median(gr))),
            float(np.median(gr)))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="pair-gate audit: forced-midpoint vs GT (offline)")
    ap.add_argument("--episodes",
                    default=str(config.LOGS_DIR / "m5_seg"
                                / "pinned_town_set.json"))
    ap.add_argument("--model", default=None,
                    help="segmentation checkpoint (default deployed best.pt)")
    ap.add_argument("--relax", action="append", default=[],
                    metavar="CONST=VALUE",
                    help="override a lane.pairing module constant for this "
                         "run (controlled relaxation experiment; the audit "
                         "centre quality is the acceptance)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.relax:
        import beamng_autopilot.lane.pairing as pairing_mod
        for kv in args.relax:
            k, _, v = kv.partition("=")
            setattr(pairing_mod, k.strip(), float(v))
        print("relaxed: " + ", ".join(
            f"{k}={v}" for k, v in (kv.partition("=")[0::2]
                                    for kv in args.relax)))

    eps = _episode_list(args.episodes)
    if not eps:
        print(f"no episodes at {args.episodes}")
        return 1
    seg = Segmenter(model_path=args.model)
    sem = SemanticHead(segmenter=seg)
    print(f"model: {seg.model_path} (device={seg.device})")
    mount = next(m for m in CAMERA_RING if m.role == FRONT_MAIN)
    rng = np.random.default_rng(11)

    rows: list[dict] = []
    for ep in eps:
        if not Path(ep).is_file():
            print(f"  missing {ep}")
            continue
        d = np.load(ep, allow_pickle=True)
        meta = json.loads(bytes(d["meta"]).decode())
        cam = mount.camera_model(int(meta["cam_w"]), int(meta["cam_h"]))
        xs, ys, hds, rgbs = d["x"], d["y"], d["heading"], d["rgb"]
        lab = d["label"] if "label" in d.files else None
        for i in range(len(xs)):
            pos = np.array([float(xs[i]), float(ys[i]), 0.0])
            hd = float(hds[i])
            out = sem.run(FrameContext(
                frame_rgb=np.asarray(rgbs[i], np.uint8), cam=cam, pos=pos,
                heading=hd, ground_z=0.0, role="front_main"))
            marks = list(out.meta.get("markings") or [])
            if not marks:
                continue
            dbg: dict = {}
            frame = pair_lane_markings(marks, pos, hd, debug=dbg)
            cands = dbg.get("cands") or []

            def _side(c):
                v = c.get("near_med")
                return float(c.get("med_lat", 0.0) if v is None else v)

            lats = [_side(c) for c in cands]
            l_med = [v for v in lats if v > SIDE_BAND_M]
            r_med = [v for v in lats if v < -SIDE_BAND_M]
            if not (l_med and r_med):
                continue                     # not a both-sides frame
            paired = bool(frame is not None
                          and getattr(frame, "paired", False))
            cause = ("accepted" if paired else
                     ("mirror_refused" if not dbg.get("mode")
                      else "no_pair"))
            gt_c, gt_r = _gt_centre(
                lab[i] if lab is not None else None, pos, hd, cam, rng)
            forced = 0.5 * (float(np.median(l_med))
                            + float(np.median(r_med)))
            mod_c = None
            if paired and frame is not None:
                c = np.asarray(frame.center, dtype=float)[:, :2]
                near = c[np.linalg.norm(c - pos[:2], axis=1) <= 12.0]
                if len(near):
                    mod_c = float(((near - pos[:2]) @ (
                        np.array([np.cos(hd), np.sin(hd)]) * 0
                        + np.array([-np.sin(hd), np.cos(hd)]))).mean())
            rows.append({"episode": Path(ep).name, "frame": i,
                         "cause": cause, "forced": forced,
                         "module": mod_c, "gt": gt_c, "gt_right": gt_r,
                         "rejects": dbg.get("pair_rejects") or {},
                         "width": dbg.get("width"),
                         "mode": dbg.get("mode")})
        print(f"  {Path(ep).name}: done")

    if not rows:
        print("no both-sides frames found")
        return 1

    print(f"\n=== pair-gate audit on {len(eps)} pinned episodes "
          f"({len(rows)} both-sides frames) ===")
    # rejection histogram per cause: which gate killed every combo
    from collections import Counter
    for cause in ("no_pair", "mirror_refused", "accepted"):
        agg: Counter = Counter()
        n = 0
        for r in rows:
            if r["cause"] != cause:
                continue
            n += 1
            for k, v in (r.get("rejects") or {}).items():
                agg[k] += int(v)
        if n:
            top = ", ".join(f"{k}={v}" for k, v in agg.most_common(6))
            print(f"  {cause:14s} frames={n:4d}  combo-rejects: {top}")
    for cause in ("accepted", "no_pair", "mirror_refused"):
        sel = [r for r in rows if r["cause"] == cause and r["gt"] is not None]
        if not sel:
            continue
        fe = np.array([abs(r["forced"] - r["gt"]) for r in sel])
        line = (f"  {cause:14s} n={len(sel):4d}  FORCED-mid vs GT: "
                f"p50={np.median(fe):.2f} m  <=1.2m {float((fe <= 1.2).mean()):.1%}"
                f"  <=0.5m {float((fe <= 0.5).mean()):.1%}")
        mod = [r for r in sel if r["module"] is not None]
        if mod:
            me = np.array([abs(r["module"] - r["gt"]) for r in mod])
            line += (f"  | module vs GT: p50={np.median(me):.2f} "
                     f"<=1.2m {float((me <= 1.2).mean()):.1%}")
        print(line)

    out = Path(args.out) if args.out else \
        config.LOGS_DIR / "m5_pair_gate_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print(f"report -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
