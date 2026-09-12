"""Train the line-presence head on human grid labels (no game needed).

Input : the grid-labeler JSONL (logs/labeling/line_grid_labels.jsonl),
        deduplicated per frame, latest record wins.
Model : frozen SegUNet backbone (deployed seg model) -> pooled mid
        features -> small MLP head; probability of "contains lane paint".
Output: logs/labeling/presence_model/best.pt (+ val AUC/acc/threshold).

The trained head is the reward/screening signal for RL and the score
source for the next active-labeling round::

    .venv\\Scripts\\python.exe scripts\\m5_line_grid_labeler.py \\
        --src <dir> --strategy score --scores <uncertainty.json>
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


def main() -> int:
    ap = argparse.ArgumentParser(description="标线存在性分类器训练")
    ap.add_argument("--labels", type=str, default=None,
                    help="宫格标注 JSONL（默认 logs/labeling/line_grid_labels.jsonl）")
    ap.add_argument("--backbone", type=str, default=None,
                    help="分割 checkpoint（默认 seg_model/best.pt）")
    ap.add_argument("--out", type=str, default=None,
                    help="输出 ckpt（默认 logs/labeling/presence_model/best.pt）")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    from beamng_autopilot.labeling.presence_model import train_and_save

    labels_path = (Path(args.labels) if args.labels
                   else config.LOGS_DIR / "labeling" / "line_grid_labels.jsonl")
    backbone = (Path(args.backbone) if args.backbone
                else config.LOGS_DIR / "m5_seg" / "seg_model" / "best.pt")
    out = (Path(args.out) if args.out
           else config.LOGS_DIR / "labeling" / "presence_model" / "best.pt")
    if not labels_path.exists():
        print(f"[presence-train] no labels at {labels_path}")
        return 1
    if not backbone.exists():
        print(f"[presence-train] no backbone at {backbone}")
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    report = train_and_save(labels_path, backbone, out,
                            val_frac=args.val_frac, epochs=args.epochs,
                            seed=args.seed, device=device, logger=print)
    print(f"\n[presence-train] RESULT val_auc={report['auc']:.4f} "
          f"val_acc={report['acc']:.4f} threshold={report['threshold']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
