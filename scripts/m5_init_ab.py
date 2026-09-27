"""初始化对照实验（T16 §3.3/§3.4）：**相同数据/步数/seed，唯一差异 `--init`**。

方案原文："先独立检验'相同数据/步数下随机初始化与已有模型初始化'；选定新基线后，
再做自动数据增量实验，不把初始化与数据组成同时变化解释成单一因子收益。"
以及 §3.6："效果结论使用预先冻结的配对 seeds 42/43/44。"

这个入口只做一件事：两臂跑同一批 seed，逐 seed 评估（像素层 + 候选身份层 +
负例），再用判定器同一条成对比较（`gates.paired_compare`）给出区间——
**不做晋级判定**（初始化对照不是候选晋级，产物只用于选基线与报告）。

用法::

    .venv\\Scripts\\python.exe scripts\\m5_init_ab.py `
        --runs <正例目录...> --eval-runs <开发集目录...> `
        --seeds 42 43 44 --total-steps 120 `
        --init logs\\experiments\\t14_e1_promotable_20260927\\baseline\\seed42\\checkpoint_last.pt `
        --out logs\\experiments\\t16_init_ab_20260927
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

#: 逐 seed 收的指标（像素层 + 候选身份层 + 负例）
PIXEL_KEYS = ("line_iou", "line_precision", "line_recall", "road_iou",
              "offroad_false_line_px")
IDENT_KEYS = ("candidate_identity_rate", "candidate_reference_coverage",
              "left_right_role_agreement")
NEG_KEYS = ("eligible_frames", "false_positive_frames",
            "false_positive_frame_rate", "false_positive_pixel_fraction",
            "false_positive_max_cc_px")
#: 越低越好的指标（成对比较的方向）
LOWER_IS_BETTER = ("offroad_false_line_px", "false_positive_frame_rate",
                   "false_positive_pixel_fraction", "false_positive_max_cc_px")


def _load_autoloop():
    spec = importlib.util.spec_from_file_location(
        "m5_seg_autoloop", ROOT / "scripts" / "m5_seg_autoloop.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["m5_seg_autoloop"] = mod
    spec.loader.exec_module(mod)
    return mod


def train_cmd(args, arm: str, seed: int, out: Path) -> list:
    """两臂共用的命令构造：**只有 `--init` 不同**（随机臂不加）。"""
    cmd = [sys.executable, str(ROOT / "scripts" / "m5_train_seg.py"),
           "--runs", *[str(r) for r in args.runs],
           "--split", "tail", "--val-frac", "0.2",
           "--total-steps", str(int(args.total_steps)),
           "--batch", str(int(args.batch)), "--lr", str(args.lr),
           "--seed", str(seed), "--device", args.device,
           "--save-every-epoch", "--metrics-run", f"{args.run_id}-{arm}-s{seed}",
           "--out", str(out)]
    if getattr(args, "paint_source", None):
        cmd += ["--paint-source", str(args.paint_source)]
    if arm == "init" and getattr(args, "init", None):
        cmd += ["--init", str(args.init)]
    return cmd


def summarize_pair(per_seed: dict, metric: str, *, lower_is_better: bool) -> dict:
    """两臂的成对比较（逐 seed 配对；缺 seed 不进配对，不补 0）。

    ``per_seed``: ``{"random": {seed: {metric: v}}, "init": {seed: {metric: v}}}``。
    返回 ``{"n", "random_mean", "init_mean", "delta", "compare"}``；``compare``
    用判定器同一条成对 t 区间（``gates.paired_compare``）。
    """
    from beamng_autopilot.experiments.gates import paired_compare
    seeds = sorted(set(per_seed.get("random", {}))
                   & set(per_seed.get("init", {})))
    a, b = [], []
    for s in seeds:
        va = (per_seed["random"][s] or {}).get(metric)
        vb = (per_seed["init"][s] or {}).get(metric)
        if va is None or vb is None:
            continue
        a.append(float(va))
        b.append(float(vb))
    out = {"n": len(a), "metric": metric, "lower_is_better": bool(lower_is_better),
           "random_mean": (None if not a else round(sum(a) / len(a), 6)),
           "init_mean": (None if not b else round(sum(b) / len(b), 6)),
           "delta": (None if not a else round(sum(b) / len(b) - sum(a) / len(a), 6)),
           "seeds": [str(s) for s in seeds]}
    if a:
        out["compare"] = paired_compare(metric, a, b,
                                        lower_is_better=lower_is_better)
    return out


def run_arm(loop, args, arm: str, seed: int, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    cmd = train_cmd(args, arm, seed, out)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True,
                       cwd=str(ROOT), timeout=float(args.timeout_s))
    rec = {"arm": arm, "seed": seed, "rc": int(r.returncode),
           "minutes": round((time.time() - t0) / 60.0, 2),
           "cmd": cmd}
    if r.returncode != 0:
        rec["stderr_tail"] = (r.stdout[-400:] + r.stderr[-400:])
        return rec
    ck = out / "checkpoint_last.pt"
    m = loop._pixel_eval(ck, [str(p) for p in args.eval_runs], args.device)
    ident = loop.identity_metrics(ck, [str(p) for p in args.eval_runs])
    hist = {}
    hp = out / "train_hist.json"
    if hp.exists():
        hist = json.loads(hp.read_text(encoding="utf-8"))
    rec.update({k: m.get(k) for k in PIXEL_KEYS})
    rec.update({k: ident.get(k) for k in IDENT_KEYS})
    rec["counts"] = ident.get("counts")
    neg = m.get("negative_line") or {}
    rec["negative_line"] = {k: neg.get(k) for k in NEG_KEYS}
    rec["steps_done"] = hist.get("steps_done")
    rec["stopped_by"] = hist.get("stopped_by")
    rec["init"] = hist.get("init")
    rec["sampler_report"] = {k: (hist.get("sampler_report") or {}).get(k)
                             for k in ("unique_available", "unique_seen",
                                       "exposures_total")}
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--eval-runs", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--total-steps", type=int, default=120)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--init", default=None,
                    help="初始化臂的父 checkpoint；不给就只跑随机臂")
    ap.add_argument("--paint-source", default="human_revision")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--timeout-s", type=float, default=3600.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not args.init:
        print("[init-ab] 没有 --init：只跑随机臂（对照组）")
    if not args.run_id:
        args.run_id = f"t16_init_ab_{time.strftime('%Y%m%d_%H%M%S')}"

    loop = _load_autoload_loop()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    arms = ["random"] + (["init"] if args.init else [])
    per_seed: dict = {a: {} for a in arms}
    runs: list = []
    for arm in arms:
        for seed in args.seeds:
            rec = run_arm(loop, args, arm, seed, out_root / arm / f"seed{seed}")
            runs.append(rec)
            per_seed[arm][str(seed)] = rec
            print(f"[init-ab] {arm} seed {seed}: rc={rec['rc']} "
                  f"line_iou={rec.get('line_iou')} "
                  f"precision={rec.get('line_precision')} "
                  f"recall={rec.get('line_recall')} "
                  f"offroad_px={rec.get('offroad_false_line_px')} "
                  f"steps={rec.get('steps_done')}/{args.total_steps} "
                  f"({rec.get('minutes')} min)", flush=True)

    summary: dict = {"run_id": args.run_id, "args": vars(args), "runs": runs,
                     "per_seed": per_seed, "comparisons": {}}
    if args.init:
        for metric in (PIXEL_KEYS + IDENT_KEYS + NEG_KEYS):
            summary["comparisons"][metric] = summarize_pair(
                per_seed, metric, lower_is_better=metric in LOWER_IS_BETTER)
    (out_root / "init_ab.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False, default=str),
        encoding="utf-8")
    print(f"[init-ab] 产物 -> {out_root / 'init_ab.json'}")
    for metric, c in (summary.get("comparisons") or {}).items():
        if c.get("n"):
            print(f"[init-ab] {metric}: random={c['random_mean']} "
                  f"init={c['init_mean']} delta={c['delta']} "
                  f"({c['compare'].get('verdict') if isinstance(c.get('compare'), dict) else ''})")
    return 0 if all(r["rc"] == 0 for r in runs) else 1


def _load_autoload_loop():
    return _load_autoloop()


if __name__ == "__main__":
    raise SystemExit(main())
