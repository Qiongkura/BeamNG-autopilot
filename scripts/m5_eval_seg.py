"""M5 分割模型离线评估：在采集的 npz 数据上评估，无需游戏。

用法:
    .venv\Scripts\python.exe scripts\m5_eval_seg.py ^
        --runs logs\m5_seg\run_* --model logs\m5_seg\seg_model\best.pt

输出: mIoU、各类 IoU、像素准确率；--save 保存可视化对比帧
（colour | 真值 | 预测）到 logs/m5_seg/eval/。
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beamng_autopilot import config
from beamng_autopilot.vision.segmentation import Segmenter, N_CLASSES, CLASS_NAMES


def main() -> None:
    ap = argparse.ArgumentParser(description="分割模型离线评估")
    ap.add_argument("--runs", nargs="+", required=True,
                    help="数据目录（可多个）")
    ap.add_argument("--model", required=True, help="模型路径 best.pt")
    ap.add_argument("--temporal", action="store_true",
                    help="enable Segmenter temporal line-mask smoothing "
                         "(component hysteresis; ablation: slightly "
                         "negative on pixel line IoU)")
    ap.add_argument("--save", action="store_true",
                    help="保存可视化对比帧")
    ap.add_argument("--save-n", type=int, default=12,
                    help="最多保存多少帧（默认 12，均匀采样）")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    files = []
    for rd in args.runs:
        files.extend(sorted(glob.glob(str(Path(rd) / "frame_*.npz"))))
    if not files:
        raise SystemExit(f"没有找到数据: {args.runs}")
    print(f"[eval] 数据 {len(files)} 帧")

    seg = Segmenter(model_path=args.model,
                    temporal_smooth=args.temporal)
    print(f"[eval] 模型加载完成: {args.model} "
          f"(temporal_smooth={args.temporal})")

    if args.save:
        out_dir = config.LOGS_DIR / "m5_seg" / "eval"
        out_dir.mkdir(parents=True, exist_ok=True)
        step = max(1, len(files) // args.save_n)

    ious = np.zeros(N_CLASSES)
    n_correct = n_pix = 0
    n_frames = 0
    per_frame = []
    # 近场标线 IoU：只统计画面下半（贴近车头 10-15m 内的路面），
    # 远场细线/小目标对驾驶几乎无影响，分开报才能看出感知到底可不可用。
    line_iou_near = 0.0
    for idx, f in enumerate(files):
        d = np.load(f)
        colour, label = d["colour"], d["label"]
        road, line = seg.predict(colour)
        pred = np.zeros(label.shape, dtype=np.uint8)
        pred[line] = 2
        pred[road] = 1
        correct = int((pred == label).sum())
        n_correct += correct
        n_pix += int(label.size)
        for c in range(N_CLASSES):
            p = pred == c
            t = label == c
            inter = int((p & t).sum())
            union = int((p | t).sum())
            ious[c] += inter / union if union else 1.0
        y0 = label.shape[0] // 2
        p_n = pred[y0:] == 2
        t_n = label[y0:] == 2
        inter_n = int((p_n & t_n).sum())
        union_n = int((p_n | t_n).sum())
        line_iou_near += inter_n / union_n if union_n else 1.0
        n_frames += 1
        if args.save and idx % step == 0:
            vis_true = (colour.copy(),)
            vis_true = colour.copy()
            vis_pred = colour.copy()
            vis_true[label == 1] = (128, 128, 128)
            vis_true[label == 2] = (0, 255, 0)
            vis_pred[pred == 1] = (128, 128, 128)
            vis_pred[pred == 2] = (0, 255, 0)
            panel = np.hstack([colour, vis_true, vis_pred])
            cv2.imwrite(str(out_dir / f"eval_{idx:05d}.png"),
                        cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

    ious /= max(1, n_frames)
    print()
    print("=" * 56)
    print("分割模型离线评估")
    print("=" * 56)
    for c in range(N_CLASSES):
        print(f"  {CLASS_NAMES[c]:<12} IoU = {ious[c]:.4f}")
    print(f"  {'line IoU(近场)':<12} = {line_iou_near / max(1, n_frames):.4f} "
          f"(下半画面)")
    print(f"  {'mIoU':<12}      = {ious.mean():.4f}")
    print(f"  像素准确率          = {n_correct / max(1, n_pix):.4f}")
    print(f"  帧数               = {n_frames}")
    if args.save:
        print(f"  可视化 -> {config.LOGS_DIR / 'm5_seg' / 'eval'}/eval_*.png")
    print("=" * 56)


if __name__ == "__main__":
    main()