"""E0 可信基线：同一配方逐 seed 重训 + 分场景/逐 seed 评价（方案 §9.2 E0）。

E0 的问题是"**独立标签下错误究竟是什么**"，产出是"分场景、逐 seed 的完整基线"，
**不宣布优化收益**。本脚本只做两件事：

1. 用**同一配方**逐 seed 训练（命令形状与 `rounds` 的 `train_cmd` 一致：
   `--split tail --val-frac 0.2 --save-every-epoch --metrics-run ...`），
   已有 checkpoint 的 seed 直接复用（重跑会覆盖，需显式 `--retrain`）；
2. 逐 seed、逐场景评价：像素层（road/line/offroad，分母口径与评估矩阵一致）
   + 候选口径（身份/覆盖/左右角色，走 identity probe）+ 实际步数与训练样本数
   （来自 checkpoint 的 `train_args`，不靠"配置里写了多少"）。

评价标签不是独立真值时，这份基线只能作为**研究基线**（`research_only`），
脚本会在输出里写清 `truth_level`，不假装它是可信基线。

用法::

    .venv\\Scripts\\python.exe scripts\\m5_e0_baseline.py --run-id t14_e0_20260925 \\
        --train-runs <训练目录> [<训练目录> ...] \\
        --eval-runs <开发目录> [<开发目录> ...] \\
        --seeds 42 43 44 45 46 --epochs 24 --batch 4 --lr 1e-3 \\
        --paint-source agent_revision [--device cuda] [--reuse] [--limit 1]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from beamng_autopilot.experiments.manifest import dir_group  # noqa: E402
from beamng_autopilot.experiments.protocol import (  # noqa: E402
    COVERAGE_GATE_FROZEN, protocol_blob, snapshot_hash,
)


def _load_loop():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "m5_seg_autoloop_e0", SCRIPTS / "m5_seg_autoloop.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["m5_seg_autoloop_e0"] = mod
    spec.loader.exec_module(mod)
    return mod


def train_one(loop, args, seed: int, out: Path) -> dict:
    """一个 seed 的训练（命令形状与 rounds 的 train_cmd 一致）。"""
    metrics_run = f"{args.run_id}-e0-s{seed}"
    cmd = [sys.executable, str(SCRIPTS / "m5_train_seg.py"),
           "--runs", *[str(r) for r in args.train_runs],
           "--split", "tail", "--val-frac", "0.2",
           "--epochs", str(args.epochs), "--batch", str(args.batch),
           "--lr", str(args.lr), "--seed", str(seed),
           "--device", args.device, "--save-every-epoch",
           "--metrics-run", metrics_run, "--out", str(out)]
    if args.paint_source:
        cmd += ["--paint-source", args.paint_source]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True,
                       timeout=int(args.timeout_s))
    return {"cmd": cmd, "rc": int(r.returncode),
            "wall_minutes": round((time.time() - t0) / 60.0, 2),
            "tail": (r.stdout or "")[-600:] + (r.stderr or "")[-400:]}


def eval_one(loop, args, seed: int, ckpt: Path) -> dict:
    """逐 seed 评价：分场景像素层 + 分场景候选口径 + 实际步数/样本数。"""
    import m5_seg_eval_matrix as em
    by: dict = {}
    for r in args.eval_runs:
        by.setdefault(dir_group(r), []).extend(em.load_frames([Path(r)]))
    px = em.evaluate_model_per_group(ckpt, by, device=args.device)
    ident = loop.identity_metrics(ckpt, [str(r) for r in args.eval_runs])
    keep = ("line_recall", "line_precision", "line_iou", "road_iou",
            "offroad_false_frac_of_pred", "offroad_false_line_px",
            "n_frames", "inference_ms_p50", "inference_ms_p95",
            "pred_unknown_line_px")
    per_scene = {}
    for g, m in (px.get("per_group") or {}).items():
        per_scene[g] = {k: m.get(k) for k in keep}
        # 分场景候选口径（方案 §10.2：只上报，不进硬门）
        per_scene[g]["candidate"] = (ident.get("per_group") or {}).get(g) or {}
    targs = {}
    try:
        import torch
        blob = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        targs = dict(blob.get("train_args") or {})
    except Exception as exc:                               # noqa: BLE001
        targs = {"error": f"{type(exc).__name__}: {exc}"}
    n_train = targs.get("n_train")
    return {
        "seed": int(seed), "checkpoint": str(ckpt),
        "checkpoint_sha16": loop._ckpt_sha16(ckpt),
        "n_train_frames": n_train,
        "steps": (loop.steps_per_epoch(n_train, args.batch) * int(args.epochs)
                  if n_train else None),
        "epochs": int(args.epochs),
        "overall": {k: px.get(k) for k in keep},
        "candidate_overall": {k: v for k, v in ident.items() if k != "per_group"},
        "per_scene": per_scene,
        "train_args": {k: targs.get(k) for k in
                       ("runs", "seed", "epochs", "batch", "lr",
                        "paint_source", "n_train", "arch_args")},
    }


#: 越小越好的指标：算"最差场景"时要取**最大**值（方向搞反会把最好的场景
#: 报成最差——实测踩到：offroad 0.0018 被当成最差，而真正最差是 0.71）
LOWER_IS_BETTER = ("offroad_false_frac_of_pred", "inference_ms_p95")


def worst_scene_table(seeds: list, *,
                      keys=("line_recall", "line_precision", "line_iou",
                            "offroad_false_frac_of_pred")) -> dict:
    """逐场景最差：每个指标给出最差的那一次测量（场景 + seed）。"""
    worst: dict = {}
    for info in seeds or []:
        ev = info.get("eval") or {}
        for g, m in (ev.get("per_scene") or {}).items():
            for key in keys:
                v = m.get(key)
                if v is None:
                    continue
                lower = key in LOWER_IS_BETTER
                cur = worst.get(key)
                if cur is None or ((float(v) > float(cur["value"])) if lower
                                   else (float(v) < float(cur["value"]))):
                    worst[key] = {"value": v, "scene": g, "seed": info.get("seed"),
                                  "lower_is_better": lower}
    return worst


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--train-runs", nargs="+", required=True)
    ap.add_argument("--eval-runs", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--paint-source", default="agent_revision")
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--reuse", action="store_true",
                    help="已有 checkpoint 直接复用（默认也复用；--retrain 才重训）")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个 seed（冒烟）")
    ap.add_argument("--timeout-s", type=float, default=3600.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    args.train_runs = [Path(p) for p in args.train_runs]
    args.eval_runs = [Path(p) for p in args.eval_runs]

    loop = _load_loop()
    run_dir = Path(loop.config.LOGS_DIR) / "experiments" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(args.seeds)[:int(args.limit)] if args.limit else list(args.seeds)
    print(f"[e0] run={args.run_id} seeds={seeds} epochs={args.epochs} "
          f"batch={args.batch} lr={args.lr} device={args.device}")
    print(f"[e0] 训练：{len(args.train_runs)} 个目录；评价：{len(args.eval_runs)} 个目录")
    snap = {**protocol_blob(), "hash": snapshot_hash(protocol_blob())}
    result = {
        "run_id": args.run_id, "recipe": {
            "train_runs": [str(p) for p in args.train_runs],
            "eval_runs": [str(p) for p in args.eval_runs],
            "seeds": seeds, "epochs": args.epochs, "batch": args.batch,
            "lr": args.lr, "paint_source": args.paint_source},
        "protocol": snap, "coverage_gate_frozen": COVERAGE_GATE_FROZEN,
        "truth_level": ("research_only: 评价标签来源 "
                        f"{args.paint_source}，不是独立 verified 真值——"
                        "这份基线**不用于晋级**"),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seeds": [],
    }
    for seed in seeds:
        out = run_dir / f"seed{seed}"
        ckpt = out / "checkpoint_last.pt"
        info = {"seed": int(seed), "out": str(out)}
        if ckpt.exists() and not args.retrain:
            info["train"] = {"reused": True}
            print(f"[e0] seed {seed}: 复用 {ckpt}")
        else:
            print(f"[e0] seed {seed}: 训练 -> {out}")
            info["train"] = train_one(loop, args, seed, out)
            print(f"[e0] seed {seed}: rc={info['train']['rc']} "
                  f"{info['train']['wall_minutes']} min")
            if info["train"]["rc"] != 0:
                info["eval"] = None
                result["seeds"].append(info)
                Path(args.out or run_dir / "e0_baseline.json").write_text(
                    json.dumps(result, indent=1, ensure_ascii=False),
                    encoding="utf-8")
                print(f"[e0] 训练失败：{info['train']['tail'][-300:]}")
                return 1
        if not ckpt.exists():
            info["eval"] = None
            info["note"] = "checkpoint_last.pt 不存在：无法评价"
            result["seeds"].append(info)
            continue
        info["eval"] = eval_one(loop, args, seed, ckpt)
        ov = info["eval"]["overall"]
        print(f"[e0] seed {seed}: line_iou={ov.get('line_iou')} "
              f"recall={ov.get('line_recall')} precision={ov.get('line_precision')} "
              f"offroad={ov.get('offroad_false_frac_of_pred')} "
              f"p95={ov.get('inference_ms_p95')} "
              f"steps={info['eval']['steps']} n_train={info['eval']['n_train_frames']}")
        for g, m in sorted((info["eval"]["per_scene"] or {}).items()):
            print(f"[e0]   {g}: line_iou={m.get('line_iou')} "
                  f"recall={m.get('line_recall')} "
                  f"offroad={m.get('offroad_false_frac_of_pred')} "
                  f"n={m.get('n_frames')}")
        result["seeds"].append(info)
    # 逐场景最差与逐 seed 汇总（方案 §10.2：最差场景必须保留）
    result["worst_scene"] = worst_scene_table(result["seeds"])
    result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    out_path = Path(args.out) if args.out else (run_dir / "e0_baseline.json")
    out_path.write_text(json.dumps(result, indent=1, ensure_ascii=False),
                        encoding="utf-8")
    print(f"[e0] -> {out_path}")
    for key, w in sorted(result["worst_scene"].items()):
        arrow = "最大" if w.get("lower_is_better") else "最小"
        print(f"[e0] 最差 {key}（{arrow}）: {w['value']} @ {w['scene']} "
              f"(seed {w['seed']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
