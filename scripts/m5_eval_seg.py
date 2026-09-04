"""M5 分割模型离线评估：在采集的 npz 数据上评估，无需游戏。

用法:
    .venv\Scripts\python.exe scripts\m5_eval_seg.py ^
        --runs logs\m5_seg\run_* --model logs\m5_seg\seg_model\best.pt

输出: mIoU、各类 IoU、像素准确率；--save 保存可视化对比帧
（colour | 真值 | 预测）到 logs/m5_seg/eval/。IoU 用全局累加
inter/union 计算（iou_from_accum），未出现标线的帧不再虚高为满分，
held-out 的 line IoU 数字才可信。

除总体指标外，还按 run 分别给出 mIoU / line IoU / 近场 line IoU /
像素准确率 / 帧数 / 标线像素占比——评估 held-out 时能立刻看出模型
在哪类路段弱、哪个 run 是劣质（标线极稀疏）数据，避免被混合均值带偏。
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
from beamng_autopilot.vision.segmentation import (
    Segmenter, N_CLASSES, CLASS_NAMES, iou_from_accum,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="分割模型离线评估")
    ap.add_argument("--runs", nargs="+", required=True,
                    help="数据目录（可多个）")
    ap.add_argument("--model", required=True, help="模型路径 best.pt")
    ap.add_argument("--device", default=None,
                    help="推理设备（cuda/cpu），默认自动；训练占用 GPU 时"
                         "建议 --device cpu 避免互抢")
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

    files = []  # (run_name, 文件路径)，按 run 归档以输出分组统计
    for rd in args.runs:
        run_name = Path(rd).name
        for f in sorted(glob.glob(str(Path(rd) / "frame_*.npz"))):
            files.append((run_name, Path(f)))
    if not files:
        raise SystemExit(f"没有找到数据: {args.runs}")
    print(f"[eval] 数据 {len(files)} 帧")

    seg = Segmenter(model_path=args.model, device=args.device,
                    temporal_smooth=args.temporal)
    print(f"[eval] 模型加载完成: {args.model} "
          f"(temporal_smooth={args.temporal})")

    save_dir = config.LOGS_DIR / "m5_seg" / "eval"
    if args.save:
        save_dir.mkdir(parents=True, exist_ok=True)
        step = max(1, len(files) // args.save_n)

    total_inter = np.zeros(N_CLASSES)
    total_union = np.zeros(N_CLASSES)
    n_correct = n_pix = 0
    n_frames = 0
    near_inter = 0
    near_union = 0
    # 按 run 聚合：总体指标之外，每组单独报 mIoU / line IoU / 近场 /
    # 像素准确率 / 标线像素占比。
    run_stat: dict[str, dict] = {}
    for idx, (run_name, f) in enumerate(files):
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
            total_inter[c] += int((p & t).sum())
            total_union[c] += int((p | t).sum())
        # 近场标线 IoU：只统计画面下半（贴近车头 10-15m 内的路面），
        # 远场细线/小目标对驾驶几乎无影响，分开报才能看出感知可不可用。
        y0 = label.shape[0] // 2
        p_n = pred[y0:] == 2
        t_n = label[y0:] == 2
        near_inter += int((p_n & t_n).sum())
        near_union += int((p_n | t_n).sum())
        st = run_stat.setdefault(run_name, {
            "n": 0, "inter": np.zeros(N_CLASSES),
            "union": np.zeros(N_CLASSES),
            "correct": 0, "pix": 0,
            "near_inter": 0, "near_union": 0, "line_px": 0,
        })
        st["n"] += 1
        for c in range(N_CLASSES):
            p = pred == c
            t = label == c
            st["inter"][c] += int((p & t).sum())
            st["union"][c] += int((p | t).sum())
        st["correct"] += int((pred == label).sum())
        st["pix"] += int(label.size)
        st["near_inter"] += int((p_n & t_n).sum())
        st["near_union"] += int((p_n | t_n).sum())
        st["line_px"] += int((label == 2).sum())
        n_frames += 1
        if args.save and idx % step == 0:
            vis_true = colour.copy()
            vis_pred = colour.copy()
            vis_true[label == 1] = (128, 128, 128)
            vis_true[label == 2] = (0, 255, 0)
            vis_pred[pred == 1] = (128, 128, 128)
            vis_pred[pred == 2] = (0, 255, 0)
            panel = np.hstack([colour, vis_true, vis_pred])
            cv2.imwrite(str(save_dir / f"eval_{idx:05d}.png"),
                        cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

    ious = iou_from_accum(total_inter, total_union)
    present_all = total_union > 0
    near_iou = (near_inter / near_union) if near_union else 0.0
    print()
    print("=" * 56)
    print("分割模型离线评估")
    print("=" * 56)
    for c in range(N_CLASSES):
        print(f"  {CLASS_NAMES[c]:<12} IoU = {ious[c]:.4f}")
    print(f"  {'line IoU(近场)':<12} = {near_iou:.4f} (下半画面)")
    print(f"  {'mIoU':<12}      = "
          f"{ious[present_all].mean() if present_all.any() else 0.0:.4f}")
    print(f"  像素准确率          = {n_correct / max(1, n_pix):.4f}")
    print(f"  帧数               = {n_frames}")
    print("-" * 56)
    print("按 run 分组（诊断数据构成/路段泛化）")
    print("-" * 56)
    for name in sorted(run_stat):
        st = run_stat[name]
        line_frac = st["line_px"] / max(1, st["pix"])
        ri = iou_from_accum(st["inter"], st["union"])
        rp = st["union"] > 0
        near = (st["near_inter"] / st["near_union"]
                if st["near_union"] else 0.0)
        print(f"  {name:<22} 帧{st['n']:>4} "
              f"line_px_frac={line_frac:.5f}  "
              f"lineIoU={ri[2]:.4f} "
              f"near={near:.4f} "
              f"mIoU={ri[rp].mean() if rp.any() else 0.0:.4f} "
              f"acc={st['correct'] / max(1, st['pix']):.4f}")
    if args.save:
        print(f"  可视化 -> {config.LOGS_DIR / 'm5_seg' / 'eval'}/eval_*.png")
    print("=" * 56)


if __name__ == "__main__":
    main()
