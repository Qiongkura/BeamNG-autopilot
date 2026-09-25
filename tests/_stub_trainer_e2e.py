"""测试桩：一个"训练一步都不做、但产物合法"的训练器。

用途：让 `rounds` 走到**写判定**那一步（只测 `--plan-only` 或只测被拦下的路径都
抓不到写判定时的错，实测踩到过一次 UnboundLocalError：训练全跑完却拿不到判定）。

它写出与真训练器同形状的产物：

* ``train_hist.json``：末段 spread 很小（让平台期判据判 True）；
* ``checkpoint_last.pt`` / ``best.pt``：真的 SegUNet state_dict + ``train_args``
  （含 ``arch_args.width``，所以容量因子也能被评估链加载）；
* 尊重 ``--width``：桩也按宽度建模型，这样 width=2 的候选臂同样能过评估。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beamng_autopilot.vision.segmentation import SegUNet  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+")
ap.add_argument("--out", required=True)
ap.add_argument("--width", type=float, default=1.0)
for flag in ("--split", "--val-frac", "--epochs", "--batch", "--lr", "--seed",
             "--device", "--metrics-run", "--paint-source",
             "--max-train-frames", "--run-weights", "--line-weight",
             "--line-tversky-weight", "--line-cldice-weight"):
    ap.add_argument(flag)
ap.add_argument("--save-every-epoch", action="store_true")
ap.add_argument("--ignore-line-class", action="store_true")
args, _unknown = ap.parse_known_args()

d = pathlib.Path(args.out)
d.mkdir(parents=True, exist_ok=True)
(d / "train_hist.json").write_text(json.dumps({
    "epoch": [0, 1, 2, 3],
    "train_loss": [1.0, 0.8, 0.75, 0.74],
    "val_miou": [0.50, 0.60, 0.601, 0.602],
    "val_acc": [0.90, 0.91, 0.911, 0.912],
    "line_ignored_frames": [0, 0, 0, 0],
    "val_line_iou": [None, None, None, None],
}), encoding="utf-8")

model = SegUNet(width=float(args.width))
ck = {
    "state_dict": model.state_dict(),
    "n_classes": 3,
    "dataset_id": "stub",
    "train_args": {
        "epochs": int(args.epochs or 24),
        "batch": int(args.batch or 2),
        "n_train": 2,
        "arch": "SegUNet",
        "arch_args": {"width": float(args.width)},
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "ignore_line_class": bool(args.ignore_line_class),
    },
}
torch.save(ck, d / "checkpoint_last.pt")
torch.save(ck, d / "best.pt")
print(f"[stub] wrote {d} (width={args.width})", flush=True)
sys.exit(0)
