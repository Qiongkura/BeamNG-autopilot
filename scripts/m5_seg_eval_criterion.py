"""复核评估口径：`checkpoint_last.pt`（定轮）vs `best.pt`（按训练内验证选优）。

背景：`rounds` 现在只用 ``checkpoint_last.pt``。这对"等步数/预算"类对照是对的
（同一个固定轮数，没有早停），但第 5 轮暴露了一个副作用——96-epoch 的臂里有 seed
末段大跳（spread 0.33），对没到平台期的臂 ``last`` 未必是它最好的状态；而
``best.pt`` 是按训练内验证集选的，会引入"每臂各自选一个 epoch"的选择步骤。
两种口径回答的问题不同，所以**不能顺手换**：先用同一批 checkpoint 把两种口径都算出来，
看结论会不会翻转，再决定。

本脚本做的事：

1. 对 ``<run>/<arm>/seed<k>/`` 下**同时存在**的 ``checkpoint_last.pt`` 与 ``best.pt``
   各评估一次（同一开发集、同一实现，来自 `m5_seg_eval_matrix`）；
2. 用**成对（按 seed）**比较分别给出两种口径下的 delta / 均值 / ci95 / verdict；
3. 报告两口径的结论是否一致，以及逐 seed 的优劣变化（翻转的 seed 要能看见）。

用法::

    python scripts/m5_seg_eval_criterion.py \\
        --run-dir logs/experiments/<run> --arm baseline --arm round0 \\
        --dev-runs logs/m5_seg/diverse_wide_20260924/front_main \\
                   logs/m5_seg/diverse_plain_20260924/front_main \\
        --out logs/experiments/<run>/eval_criterion.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: 两种口径的文件名（先 last / 后 best；报告里按这个顺序）
CHECKPOINTS = ("checkpoint_last.pt", "best.pt")


def collect_arms(run_dir: Path, arms: list, dev_runs: list, *,
                 metric: str = "road_iou", device: str = "cuda",
                 evaluate=None, load_frames=None) -> dict:
    """``{arm: {seed: {ckpt: {"metric": v, ...}}}}``；缺文件记 None，不猜。"""
    if evaluate is None or load_frames is None:
        import m5_seg_eval_matrix as em
        evaluate = evaluate or em.evaluate_model
        load_frames = load_frames or em.load_frames
    out: dict = {}
    for arm in arms:
        d = Path(run_dir) / arm
        if not d.exists():
            out[arm] = {}
            continue
        per_seed: dict = {}
        for seed_dir in sorted(p for p in d.iterdir() if p.is_dir()
                               and p.name.startswith("seed")):
            seed = seed_dir.name.replace("seed", "")
            got: dict = {}
            for ck in CHECKPOINTS:
                p = seed_dir / ck
                if not p.exists():
                    got[ck] = {"missing": str(p)}
                    continue
                frames = load_frames([Path(r) for r in dev_runs])
                m = evaluate(p, frames, device=device) or {}
                got[ck] = {"metric": m.get(metric),
                           "inference_ms_p95": m.get("inference_ms_p95"),
                           "inference_ms_p50": m.get("inference_ms_p50")}
            per_seed[seed] = got
        out[arm] = per_seed
    return out


def compare(per_arm: dict, *, arm_a: str, arm_b: str, metric: str = "road_iou",
            lower_is_better: bool = False) -> dict:
    """按 seed 成对比较（两种口径各一份）；缺测的 seed 直接不参与，不补 0。"""
    from beamng_autopilot.experiments.gates import paired_compare
    a, b = per_arm.get(arm_a) or {}, per_arm.get(arm_b) or {}
    seeds = sorted(set(a) & set(b), key=lambda s: int(s) if s.isdigit() else 0)
    out: dict = {"seeds": seeds, metric: {}}
    for ck in CHECKPOINTS:
        va, vb = [], []
        used = []
        for s in seeds:
            x = (a[s].get(ck) or {})
            y = (b[s].get(ck) or {})
            if x.get("missing") or y.get("missing"):
                continue
            if x.get("metric") is None or y.get("metric") is None:
                continue
            va.append(float(x["metric"]))
            vb.append(float(y["metric"]))
            used.append(s)
        if len(used) < 2:
            out[metric][ck] = {"n": len(used), "verdict": "needs_evidence",
                               "why": "可用 seed 少于 2 个"}
            continue
        res = paired_compare(metric, va, vb,
                             lower_is_better=lower_is_better)
        res["seeds_used"] = used
        out[metric][ck] = res
    return out


def criterion_report(run_dir: Path, arms: list, dev_runs: list, *,
                     metric: str = "road_iou", device: str = "cuda",
                     evaluate=None, load_frames=None,
                     per_arm: dict | None = None) -> dict:
    per_arm = per_arm or collect_arms(run_dir, arms, dev_runs, metric=metric,
                                      device=device, evaluate=evaluate,
                                      load_frames=load_frames)
    arm_a, arm_b = arms[0], arms[1]
    cmp_ = compare(per_arm, arm_a=arm_a, arm_b=arm_b, metric=metric)
    last_v = (cmp_[metric].get("checkpoint_last.pt") or {}).get("verdict")
    best_v = (cmp_[metric].get("best.pt") or {}).get("verdict")
    flips = []
    for s in cmp_["seeds"]:
        row = {}
        for ck in CHECKPOINTS:
            x = ((per_arm.get(arm_a) or {}).get(s) or {}).get(ck) or {}
            y = ((per_arm.get(arm_b) or {}).get(s) or {}).get(ck) or {}
            if x.get("metric") is None or y.get("metric") is None:
                row[ck] = None
                continue
            row[ck] = float(y["metric"]) - float(x["metric"])
        if row.get("checkpoint_last.pt") is not None and \
                row.get("best.pt") is not None and \
                (row["checkpoint_last.pt"] > 0) != (row["best.pt"] > 0):
            flips.append({"seed": s, **{k: round(v, 5) for k, v in row.items()
                                        if v is not None}})
    return {"run_dir": str(run_dir), "arms": arms, "metric": metric,
            "per_arm": per_arm, "compare": cmp_,
            "verdict_last": last_v, "verdict_best": best_v,
            "same_verdict": last_v == best_v, "flipped_seeds": flips}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--arm", action="append", required=True,
                    help="可重复；第一个当基线，第二个当候选")
    ap.add_argument("--dev-runs", nargs="+", required=True)
    ap.add_argument("--metric", default="road_iou")
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if len(args.arm) != 2:
        print("[criterion] 需要两个 --arm（基线、候选）")
        return 2
    rep = criterion_report(Path(args.run_dir), list(args.arm),
                           [str(p) for p in args.dev_runs],
                           metric=args.metric, device=args.device)
    c = rep["compare"][args.metric]
    for ck in CHECKPOINTS:
        row = c.get(ck) or {}
        print(f"[criterion] {ck}: n={row.get('n')} "
              f"mean={row.get('mean_delta')} ci95={row.get('ci95_halfwidth')} "
              f"verdict={row.get('verdict')}")
    print(f"[criterion] 两口径结论一致：{rep['same_verdict']}"
          + (f"；翻转的 seed：{rep['flipped_seeds']}" if rep["flipped_seeds"] else ""))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False),
                                  encoding="utf-8")
        print(f"[criterion] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
