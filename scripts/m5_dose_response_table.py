"""剂量-效应表：把若干轮判定文件里的"剂量"与"配对差值"汇总成一张表（T16 §22–§25）。

为什么要它：负例通道的剂量-效应现在是主线证据（1× +0.0214 / 4× +0.0245 /
6× +0.0452…），而每轮的剂量（线占比、负例占比、包数）散落在各判定文件的
`synthetic_line_dose` 与 `candidate_runs` 里，跨臂比较要手工数目录（实测踩过：
4×/6× 臂的负例剂量在判定里一度不可见）。本入口把它们读出来，按**负例帧数**
排序输出，并给出去重后的"剂量点"。

用法::

    .venv\\Scripts\\python.exe scripts\\m5_dose_response_table.py `
        --run t16_negdose4x_20261001 --run t16_negdose6x_20261001 `
        --run t16_negdose95x_20261001 --out logs/experiments/dose_response.json

只读判定文件；不训练、不启动游戏。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: 关心哪些指标的配对差值（报告里的主指标 + 门相关项）
METRICS = ("candidate_identity_rate", "left_right_role_agreement",
           "line_precision", "line_recall", "line_iou",
           "offroad_false_line_px")


def summarize_decision(blob: dict) -> dict:
    """判定文件 -> 一行剂量-效应记录（纯函数，可单测）。

    剂量字段优先取 `synthetic_line_dose`（新版判定都有）；负例帧数在旧判定里
    可能缺失——那时按"追加 run 数 × 8 帧 − 合成线帧"估，并把 `dose_source`
    记成 `estimated`（估算值不得与实测值混着引用）。
    """
    pr = blob.get("pairings") or {}
    dz = blob.get("synthetic_line_dose") or {}
    added = [str(r) for r in (blob.get("candidate_runs") or [])
             if str(r) not in set(blob.get("baseline_runs") or [])]
    line_frames = dz.get("synthetic_line_frames")
    neg_frames = dz.get("negative_frames")
    dose_source = "measured" if (line_frames is not None
                                and neg_frames is not None) else "unknown"
    if neg_frames is None:
        # 旧判定：用 8 帧/包估（本仓库的生成包都是 8 帧），并标明是估算——
        # 估算的负例帧数**不得**与实测值混着引用（dose_source=estimated）
        neg_frames = max(0, len(added) * 8 - int(line_frames or 0))
        dose_source = "estimated"
    out = {
        "run_id": blob.get("_run_id") or "",
        "candidate_id": blob.get("candidate_id"),
        "seeds": len((pr.get("candidate_identity_rate") or {})
                     .get("candidate") or []),
        "added_runs": len(added),
        "synthetic_line_frames": line_frames,
        "negative_frames": neg_frames,
        "negative_x": (None if not line_frames
                       else round(float(neg_frames) / float(line_frames), 2)),
        "line_share": dz.get("share"),
        "negative_share": dz.get("negative_frac"),
        "dose_level": dz.get("level"),
        "dose_source": dose_source,
        "all_at_plateau": blob.get("all_at_plateau"),
        "outcome": blob.get("round_outcome"),
    }
    for k in METRICS:
        p = pr.get(k) or {}
        out[k] = {"mean_delta": p.get("mean_delta"),
                  "ci95": p.get("ci95_halfwidth"),
                  "verdict": p.get("verdict"),
                  "champion_mean": (None if not p.get("champion")
                                    else round(sum(p["champion"])
                                               / len(p["champion"]), 4)),
                  "candidate_mean": (None if not p.get("candidate")
                                     else round(sum(p["candidate"])
                                                 / len(p["candidate"]), 4))}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True,
                    help="run-id（logs/experiments/<run-id>/decision_*.json）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows: list = []
    for rid in args.run:
        d = ROOT / "logs" / "experiments" / str(rid)
        files = sorted(d.glob("decision_*.json"))
        if not files:
            print(f"[dose] {rid}: 没有判定文件（跳过）")
            continue
        for f in files:
            blob = json.loads(f.read_text(encoding="utf-8"))
            blob["_run_id"] = str(rid)
            rows.append(summarize_decision(blob))
    rows.sort(key=lambda r: (r["negative_frames"] or 0))
    hdr = (f"{'run':26s} {'cand':14s} {'seeds':>5} {'线帧':>5} {'负帧':>5} "
           f"{'负/线':>6} {'身份Δ':>8} {'身份判定':>18} {'IoUΔ':>8} {'召回Δ':>8}")
    print(hdr)
    for r in rows:
        idm = r["candidate_identity_rate"]
        print(f"{r['run_id']:26s} {str(r['candidate_id']):14s} {r['seeds']:5d} "
              f"{str(r['synthetic_line_frames']):>5} {str(r['negative_frames']):>5} "
              f"{str(r['negative_x']):>6} {str(idm['mean_delta']):>8} "
              f"{str(idm['verdict']):>18} "
              f"{str((r['line_iou'] or {}).get('mean_delta')):>8} "
              f"{str((r['line_recall'] or {}).get('mean_delta')):>8}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=1, ensure_ascii=False),
                                  encoding="utf-8")
        print(f"[dose] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
