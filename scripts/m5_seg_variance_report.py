"""训练波动归因：同一配方下 `road_iou` 的 seed 间差到底从哪儿来。

下一阶段方案第 2 项要求"固定数据版本、初始权重、优化步数和成对 seed，重跑基线；
按 seed、路段、epoch 查看差异，检查是否由少量场景、标签或选取 checkpoint 的方式
造成"，并明确"只有测量稳定后才增加 seed 或比较新因子，不直接投入几十次重复训练"。

本工具**不训练**，只用磁盘上已有的 checkpoint 做测量：

* 逐 (arm, seed, epoch) 评估：总 `road_iou`、**逐开发路段**、逐帧 road IoU、推理 p50/p95；
* 方差分解：seed 间 vs seed 内(epoch) vs 路段间；
* 帧主导性：最差的若干帧吃掉多少"均值差"（少量场景主导的判据）；
* checkpoint 选取：`epoch_*` 各轮之间、以及它们与最后/最佳轮的差距
  （"取最后一轮"是不是一个会制造波动的选择）。

用法::

    pwsh> .venv\\Scripts\\python.exe scripts\\m5_seg_variance_report.py `
            --runs logs\\experiments\\t14_3h_roads3f logs\\experiments\\t14_3h_roads3d `
            --dev-runs logs\\m5_seg\\diverse_wide_20260924\\front_main `
                       logs\\m5_seg\\diverse_plain_20260924\\front_main `
            --out logs\\experiments\\t14_variance\\variance.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def road_frame_metrics(pred_road, label) -> dict:
    """逐帧路面统计（纯函数；分母为 0 时 iou 为 None，不返回 0 冒充测量）。"""
    import numpy as np
    pred = np.asarray(pred_road).astype(bool)
    lab = np.asarray(label)
    known = lab != 255
    gt = lab == 1
    pr = pred & known
    tp = int((gt & pr).sum())
    fp = int((pr & ~gt).sum())
    fn = int((gt & ~pr & known).sum())
    denom = tp + fp + fn
    return {"tp": tp, "fp": fp, "fn": fn,
            "iou": None if denom == 0 else round(tp / denom, 5),
            "gt_px": int(gt.sum())}


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return None if not xs else sum(xs) / len(xs)


def variance_decomposition(records: list) -> dict:
    """把 (arm, seed, epoch) 的 road_iou 记录分解成三个来源。

    ``records`` 每项需含 ``arm``/``seed``/``epoch``/``road_iou``（可为 None）。
    分量用"离差平方和"口径（不含自由度修正，便于相加=总平方和）：

    * ``between_seed``：同一 arm 内 seed 均值的离散；
    * ``within_seed_epoch``：同 seed 下不同 epoch 的离散（"取哪一轮"的影响）；
    * ``between_arm``：不同 arm 均值的离散（数据/因子本身的影响）。
    """
    vals = [r for r in records if r.get("road_iou") is not None]
    out = {"n": len(vals), "grand_mean": None, "total_ss": 0.0,
           "between_arm": 0.0, "between_seed": 0.0, "within_seed_epoch": 0.0,
           "arms": {}}
    if not vals:
        return out
    grand = sum(r["road_iou"] for r in vals) / len(vals)
    out["grand_mean"] = round(grand, 5)
    out["total_ss"] = round(sum((r["road_iou"] - grand) ** 2 for r in vals), 6)
    by_arm: dict = {}
    for r in vals:
        by_arm.setdefault(str(r.get("arm")), []).append(r)
    for arm, rs in by_arm.items():
        arm_mean = sum(x["road_iou"] for x in rs) / len(rs)
        out["between_arm"] += len(rs) * (arm_mean - grand) ** 2
        by_seed: dict = {}
        for x in rs:
            by_seed.setdefault(str(x.get("seed")), []).append(x)
        info = {"n": len(rs), "mean": round(arm_mean, 5), "seeds": {}}
        for seed, ss in by_seed.items():
            seed_mean = sum(y["road_iou"] for y in ss) / len(ss)
            out["between_seed"] += len(ss) * (seed_mean - arm_mean) ** 2
            within = sum((y["road_iou"] - seed_mean) ** 2 for y in ss)
            out["within_seed_epoch"] += within
            info["seeds"][seed] = {
                "n": len(ss), "mean": round(seed_mean, 5),
                "min": round(min(y["road_iou"] for y in ss), 5),
                "max": round(max(y["road_iou"] for y in ss), 5),
                "epochs": {str(y["epoch"]): y["road_iou"] for y in ss}}
        out["arms"][arm] = info
    for k in ("between_arm", "between_seed", "within_seed_epoch"):
        out[k] = round(out[k], 6)
    return out


def dominant_frame_share(per_frame: list, *, worst_k: int = 5) -> dict:
    """最差的 k 帧对"帧均值"的拉低占多少（判断是否少量场景主导）。

    口径：把逐帧 IoU 的均值当作指标，算去掉最差 k 帧后的均值提升，
    以及最差帧的共享权重；同时给出"没有分母(iou=None)的帧数"。
    """
    got = [f for f in (per_frame or []) if f.get("iou") is not None]
    none_n = len([f for f in (per_frame or []) if f.get("iou") is None])
    if not got:
        return {"n": 0, "n_no_denominator": none_n, "mean": None,
                "worst": [], "mean_without_worst": None,
                "lift_from_dropping_worst": None}
    vals = sorted(f["iou"] for f in got)
    mean = sum(vals) / len(vals)
    k = min(int(worst_k), len(vals))
    rest = vals[k:]
    mean_rest = (sum(rest) / len(rest)) if rest else None
    worst = sorted(got, key=lambda f: f["iou"])[:k]
    return {"n": len(got), "n_no_denominator": none_n,
            "mean": round(mean, 5),
            "worst": [{"frame": f.get("frame"), "iou": f["iou"],
                       "gt_px": f.get("gt_px")} for f in worst],
            "mean_without_worst": (None if mean_rest is None
                                   else round(mean_rest, 5)),
            "lift_from_dropping_worst": (None if mean_rest is None
                                         else round(mean_rest - mean, 5))}


def checkpoints_in(run_dir: Path) -> list:
    """``[(arm, seed, epoch, path), ...]``。

    认两种布局：``<run>/<arm>/seed*/epoch_*.pt``（rounds 的运行）与
    ``<run>/seed*/epoch_*.pt``（单臂训练，arm 记作运行目录名）。
    """
    run_dir = Path(run_dir)
    out = []

    def _collect(arm_name: str, seed_root: Path) -> None:
        for seed_dir in sorted(seed_root.glob("seed*")):
            for ck in sorted(seed_dir.glob("epoch_*.pt")):
                try:
                    ep = int(ck.stem.split("_")[-1])
                except ValueError:
                    continue
                try:
                    seed = int(seed_dir.name.replace("seed", ""))
                except ValueError:
                    seed = -1
                out.append((arm_name, seed, ep, ck))

    if any(run_dir.glob("seed*/epoch_*.pt")):
        _collect(run_dir.name, run_dir)
    for arm_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        if arm_dir.name.startswith(".") or arm_dir.name == "candidates":
            continue
        _collect(arm_dir.name, arm_dir)
    return out


def evaluate_checkpoint(path: Path, dev: dict, *, device: str = "cuda") -> dict:
    """在一个 checkpoint 上评估所有开发路段：总/逐路段指标 + 逐帧 road IoU。"""
    import numpy as np
    from beamng_autopilot.vision.segmentation import Segmenter
    seg = Segmenter(model_path=str(path))
    per_frame, ms = [], []
    rows: dict = {}
    for name, frames in dev.items():
        acc = {"tp": 0, "fp": 0, "fn": 0}
        for _label, colour, lab in frames:
            road, _line, _p = seg.predict_with_probs(colour)
            ms.append(float((seg.last_timing_ms or {}).get("total") or 0.0))
            m = road_frame_metrics(road, lab)
            for k in acc:
                acc[k] += m[k]
            per_frame.append({"road": name, "frame": _label, "iou": m["iou"],
                              "gt_px": m["gt_px"]})
        denom = acc["tp"] + acc["fp"] + acc["fn"]
        rows[name] = {"tp": acc["tp"], "fp": acc["fp"], "fn": acc["fn"],
                      "road_iou": (None if denom == 0
                                   else round(acc["tp"] / denom, 5))}
    tp = sum(r["tp"] for r in rows.values())
    fp = sum(r["fp"] for r in rows.values())
    fn = sum(r["fn"] for r in rows.values())
    # 平凡基线（把全部已知像素预测成路面）＝开发标签里的路面占比：类别不平衡下
    # 它会白送一个高分（实测 0.4024），所以每个 checkpoint 都要和它比
    gt_px = sum(int((np.asarray(f[2]) == 1).sum())
                for frames in dev.values() for f in frames)
    known_px = sum(int((np.asarray(f[2]) != 255).sum())
                   for frames in dev.values() for f in frames)
    trivial = (None if not known_px else round(gt_px / known_px, 5))
    a = np.asarray(ms, dtype=float) if ms else np.asarray([], dtype=float)
    return {"road_iou": (None if (tp + fp + fn) == 0
                         else round(tp / (tp + fp + fn), 5)),
            "road_iou_trivial_all_road": trivial,
            "above_trivial": (None if (tp + fp + fn) == 0 or trivial is None
                              else round(tp / (tp + fp + fn) - trivial, 5)),
            "per_road": rows, "per_frame": per_frame,
            "n_frames": len(per_frame),
            "inference_ms_p50": (None if a.size == 0
                                 else round(float(np.percentile(a, 50)), 2)),
            "inference_ms_p95": (None if a.size == 0
                                 else round(float(np.percentile(a, 95)), 2))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="训练波动归因（只读，不训练）")
    ap.add_argument("--runs", nargs="+", required=True,
                    help="运行目录（含 baseline/ 与 round*/seed*/epoch_*.pt）")
    ap.add_argument("--dev-runs", nargs="+", required=True,
                    help="开发路段目录（逐帧 npz），每条单独统计并汇总")
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--worst-k", type=int, default=5)
    ap.add_argument("--max-frames", type=int, default=0,
                    help="每条路段最多用多少帧（0=全部）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    sys.path.insert(0, str(ROOT / "scripts"))
    import m5_seg_eval_matrix as em

    dev: dict = {}
    for d in args.dev_runs:
        frames = em.load_frames([Path(d)])
        if args.max_frames:
            frames = frames[:int(args.max_frames)]
        dev[Path(d).parent.name or Path(d).name] = frames
    print(f"[variance] 开发路段 {len(dev)} 条，共 "
          f"{sum(len(v) for v in dev.values())} 帧", flush=True)

    records, t0 = [], time.time()
    for run in args.runs:
        cks = checkpoints_in(Path(run))
        print(f"[variance] {Path(run).name}: {len(cks)} 个 epoch checkpoint",
              flush=True)
        for arm, seed, ep, path in cks:
            m = evaluate_checkpoint(path, dev, device=args.device)
            records.append({"run": Path(run).name, "arm": arm, "seed": seed,
                            "epoch": ep, "ckpt": str(path),
                            "road_iou": m["road_iou"],
                            "trivial": m["road_iou_trivial_all_road"],
                            "above_trivial": m["above_trivial"],
                            "per_road": m["per_road"],
                            "inference_ms_p50": m["inference_ms_p50"],
                            "inference_ms_p95": m["inference_ms_p95"],
                            "frames": dominant_frame_share(m["per_frame"],
                                                          worst_k=args.worst_k)})
            print(f"    {arm}/seed{seed}/ep{ep}: road_iou={m['road_iou']} "
                  f"(平凡基线 {m['road_iou_trivial_all_road']}, 领先 "
                  f"{m['above_trivial']:+}) {m['n_frames']} 帧", flush=True)

    dec = variance_decomposition(records)
    # 逐 arm 的"取最后一轮 vs 取最好一轮"差异：checkpoint 选取本身制造多少波动
    pick = {}
    for arm in sorted({r["arm"] for r in records}):
        rs = [r for r in records if r["arm"] == arm and r["road_iou"] is not None]
        last = [r for r in rs if r["epoch"] == max(x["epoch"] for x in rs)]
        best = max(rs, key=lambda r: r["road_iou"]) if rs else None
        pick[arm] = {
            "last_epoch_mean": (None if not last else
                                round(_mean([r["road_iou"] for r in last]), 5)),
            "best_over_epochs": (None if best is None else
                                 {"seed": best["seed"], "epoch": best["epoch"],
                                  "road_iou": best["road_iou"]}),
            "spread_epochs": (None if not rs else
                              round(max(r["road_iou"] for r in rs)
                                    - min(r["road_iou"] for r in rs), 5))}
    blob = {"dev_runs": list(dev), "n_checkpoints": len(records),
            "decomposition": dec, "checkpoint_pick": pick,
            "records": records,
            "wall_s": round(time.time() - t0, 1)}
    print("\n[variance] 方差分解（离差平方和口径）")
    print(f"  总平方和 {dec['total_ss']}  =  臂间 {dec['between_arm']}"
          f" + seed 间 {dec['between_seed']} + seed 内(epoch) "
          f"{dec['within_seed_epoch']}")
    for arm, info in dec["arms"].items():
        seeds = ", ".join(f"seed{s}:{v['mean']}({v['min']}..{v['max']})"
                          for s, v in sorted(info["seeds"].items()))
        print(f"  {arm}: 均值 {info['mean']} | {seeds}")
    for arm, p in pick.items():
        print(f"  {arm}: epoch 极差 {p['spread_epochs']} | "
              f"最后一轮均值 {p['last_epoch_mean']} | 最好一轮 "
              f"{p['best_over_epochs']}")
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(blob, indent=1, ensure_ascii=False),
                     encoding="utf-8")
        print(f"[variance] -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
