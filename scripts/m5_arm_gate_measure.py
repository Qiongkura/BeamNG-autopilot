"""臂级门表：同一套提取/匹配口径下，逐 arm（checkpoint）量全部任务门。

为什么要有它：`rounds` 的判定只对**当轮两臂**出数，而"提取器/协议换版后旧判定
不可比"（协议 v7 起）需要**把已有 checkpoint 在同一口径下重测**——本脚本就是
那个入口：给若干个 `名字=checkpoint`，在**同一开发集、同一实现**上跑探针，
输出覆盖率 / 身份率 / 角色一致率 / 掩码 recall-precision-IoU / 路外候选。

用法::

    .venv\\Scripts\\python.exe scripts\\m5_arm_gate_measure.py `
        --arm devdist-s42=logs/experiments/<run>/round0/seed42/checkpoint_last.pt `
        --arm baseline-s42=logs/experiments/<run>/baseline/seed42/checkpoint_last.pt `
        --out logs/experiments/<run>/gate_v7.json

不训练、不启动游戏；只读 checkpoint 与开发帧。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

DEFAULT_DEV = "logs/experiments/review_pack_20260926/reviewed_full"


def summarize_rows(rows: list) -> dict:
    """逐帧探针行 -> 臂级计数与比率（纯函数，可单测）。

    只累加**整数计数**再算一次比率（与判定同口径）；掩码指标取逐帧均值并
    同时给出样本数（缺测不静默当 0）。
    """
    from beamng_autopilot.experiments import candidate_metrics as cm
    acc = cm.empty()
    mask = {"recall": [], "precision": [], "iou": []}
    off_road = n_cand = frames = 0
    merges = merged = 0
    for r in rows or []:
        c = r.get("counts") or {}
        cm.accumulate(acc, c)
        frames += 1
        for k in mask:
            if r.get(k) is not None:
                mask[k].append(float(r[k]))
        off_road += int(r.get("candidates_off_road") or 0)
        n_cand += int(r.get("n_candidates") or 0)
        g = r.get("line_candidate_gate") or {}
        mg = (g.get("line_candidate_merge") or {}) if isinstance(g, dict) else {}
        merges += int(mg.get("groups") or 0)
        merged += int(mg.get("merged") or 0)
    rt = cm.ratios(acc)
    out = {"frames": frames, **{k: int(acc[k]) for k in ("P_frames", "C", "R",
                                                         "M", "L", "A",
                                                         "C_outside_P")},
           "candidate_reference_coverage": rt["candidate_reference_coverage"],
           "candidate_identity_rate": rt["candidate_identity_rate"],
           "left_right_role_agreement": rt["left_right_role_agreement"],
           "candidates_off_road": off_road, "n_candidates": n_cand,
           "off_road_frac": (round(off_road / n_cand, 4) if n_cand else None),
           "merge_groups": merges, "merge_merged": merged}
    for k, v in mask.items():
        out[f"mask_{k}_mean"] = (round(sum(v) / len(v), 4) if v else None)
        out[f"mask_{k}_n"] = len(v)
    return out


def _probe():
    spec = importlib.util.spec_from_file_location(
        "m5_marking_identity_probe", ROOT / "scripts"
        / "m5_marking_identity_probe.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["m5_marking_identity_probe"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True,
                    metavar="NAME=CHECKPOINT")
    ap.add_argument("--dev-runs", nargs="*", default=None,
                    help=f"开发集目录（默认 {DEFAULT_DEV}/*/front_main）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dirs = ([Path(p) for p in args.dev_runs] if args.dev_runs
            else sorted((ROOT / DEFAULT_DEV).glob("*/front_main")))
    if not dirs:
        raise SystemExit(f"开发集为空：{args.dev_runs or DEFAULT_DEV}")
    ip = _probe()
    out = {"dev_runs": [str(d) for d in dirs], "arms": {}}
    for spec in args.arm:
        name, _, ck = str(spec).partition("=")
        if not ck:
            raise SystemExit(f"--arm 需要 NAME=CHECKPOINT，收到 {spec!r}")
        rows = []
        for d in dirs:
            meta_p = d / "meta.json"
            meta = (json.loads(meta_p.read_text(encoding="utf-8"))
                    if meta_p.is_file() else None)
            res = ip.probe(d, meta, view=d.name, model_path=str(ROOT / ck),
                           device=args.device)
            rows.extend(res.get("rows") or [])
        s = summarize_rows(rows)
        out["arms"][name] = {"checkpoint": str(ck), **s}
        print(f"[gate] {name:16s} C={s['C']:4d} R={s['R']:4d} M={s['M']:4d} "
              f"L={s['L']:4d} A={s['A']:4d} | 覆盖={s['candidate_reference_coverage']} "
              f"身份={s['candidate_identity_rate']} 角色={s['left_right_role_agreement']} "
              f"| 掩码 r/p/iou={s['mask_recall_mean']}/{s['mask_precision_mean']}/"
              f"{s['mask_iou_mean']} | 路外候选 {s['candidates_off_road']}"
              f"（{s['off_road_frac']}）| 合并 {s['merge_merged']}", flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                  encoding="utf-8")
        print(f"[gate] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
