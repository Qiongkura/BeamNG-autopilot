"""Score the seg model's line-presence against the human grid labels.

Reads the grid-labeler JSONL, runs the segmentation model on every labeled
frame, and reports agreement (AUC + best-threshold confusion + the top
disagreement frames = model miss/false-line cases, i.e. the next frames a
human should re-check).  Writes a {frame: line_fraction} score JSON that
feeds the active-learning loop directly:

    .venv\\Scripts\\python.exe scripts\\m5_line_grid_labeler.py \\
        --src <dir> --strategy score \\
        --scores logs\\labeling\\line_presence_scores.json

Usage:
    .venv\\Scripts\\python.exe scripts\\m5_line_presence_eval.py
    .venv\\Scripts\\python.exe scripts\\m5_line_presence_eval.py \\
        --src logs\\m5_e2e\\worst -r        # 顺带给未标帧也打分
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beamng_autopilot import config
from beamng_autopilot.labeling.presence import (
    load_label_records, presence_agreement, score_images)


def main() -> int:
    ap = argparse.ArgumentParser(description="标线存在性：模型 vs 人工标签")
    ap.add_argument("--labels", type=str, default=None,
                    help="宫格标注 JSONL（默认 logs/labeling/line_grid_labels.jsonl）")
    ap.add_argument("--model", type=str, default=None,
                    help="分割模型 checkpoint（默认 seg_model/best.pt）")
    ap.add_argument("--src", type=str, default=None,
                    help="可选：顺带给这个目录所有图片打分（未标帧也进分数 json）")
    ap.add_argument("-r", "--recursive", action="store_true")
    ap.add_argument("--scores-out", type=str, default=None,
                    help="分数 JSON 输出（默认 logs/labeling/line_presence_scores.json）")
    ap.add_argument("--top", type=int, default=10,
                    help="打印前 N 条不一致帧")
    args = ap.parse_args()

    from beamng_autopilot.labeling.grid_labeler import scan_images

    labels_path = (Path(args.labels) if args.labels
                   else config.LOGS_DIR / "labeling" / "line_grid_labels.jsonl")
    if not labels_path.exists():
        print(f"[presence] no labels at {labels_path} - 先跑 m5_line_grid_labeler")
        return 1
    records = load_label_records(labels_path)
    if not records:
        print(f"[presence] {labels_path} 里没有可用标签")
        return 1

    paths = [Path(r["path"]) for r in records]
    print(f"[presence] {len(paths)} labeled frames | model={args.model or 'best.pt'}")
    scores = score_images(args.model, paths, logger=print)

    if args.src:
        src = Path(args.src).resolve()
        extra = [p for p in scan_images(src, recursive=args.recursive)
                 if str(p) not in scores]
        if extra:
            print(f"[presence] scoring {len(extra)} extra frames from {src}")
            scores.update(score_images(args.model, extra, logger=print))

    agreement = presence_agreement(
        {r["path"]: int(r["has_line"]) for r in records}, scores)
    print(f"\n[presence] labeled: {agreement['n']} "
          f"(pos {agreement['pos']} / neg {agreement['neg']})")
    print(f"[presence] AUC = {agreement['auc']}  "
          f"best-threshold = {agreement['threshold']}")
    print(f"[presence] acc {agreement['accuracy']} | "
          f"precision {agreement['precision']} | recall {agreement['recall']} | "
          f"tp/fp/tn/fn = {agreement['tp']}/{agreement['fp']}/"
          f"{agreement['tn']}/{agreement['fn']}")
    for d in agreement["disagreements"][:max(0, args.top)]:
        tag = "模型漏检" if d["label"] == 1 else "模型误报"
        print(f"[presence] {tag}: score={d['score']:.4f} "
              f"{Path(d['key']).name}")

    out = (Path(args.scores_out) if args.scores_out
           else config.LOGS_DIR / "labeling" / "line_presence_scores.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    by_name = {Path(k).name: v for k, v in scores.items()}
    out.write_text(json.dumps(by_name, ensure_ascii=False, indent=0),
                   encoding="utf-8")
    print(f"\n[presence] scores ({len(by_name)} frames) -> {out}")
    print(f"[presence] 下一轮优先标不一致帧: "
          f"--strategy score --scores {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
