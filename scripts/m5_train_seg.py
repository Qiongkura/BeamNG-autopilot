"""M5 路面/标线分割训练：轻量 UNet（beamng_autopilot.vision.segmentation）。

数据：scripts/m5_collect_seg.py 采集的 npz 帧（colour + label）。
划分：按帧序时间划分（前 80% 训练，后 20% 验证，与 M3 相同做法）。
损失：交叉熵 + 中位频率类别加权（标线像素极少，不加权学不动）。
指标：mIoU + 各类 IoU + 像素准确率。自动保存最优模型与训练曲线。

用法:
    .venv\Scripts\python.exe scripts\m5_train_seg.py --runs logs\m5_seg\run_*
        --epochs 40 --out logs\m5_seg\seg_model
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beamng_autopilot import config
from beamng_autopilot.vision.segmentation import SegUNet, N_CLASSES, CLASS_NAMES


def load_frames(run_dirs: list[Path]) -> list[tuple[np.ndarray, np.ndarray]]:
    """读取所有 npz 帧，按文件名排序（时间序）。"""
    files = []
    for rd in run_dirs:
        files.extend(sorted(glob.glob(str(rd / "frame_*.npz"))))
    if not files:
        raise SystemExit(f"没有找到数据: {run_dirs}")
    frames = []
    for f in files:
        d = np.load(f)
        frames.append((d["colour"], d["label"]))
    return frames


def median_freq_weights(labels: list[np.ndarray],
                       line_weight: float = 2.0) -> torch.Tensor:
    """Median frequency balancing：权重与类别频率成反比。"""
    hist = np.zeros(N_CLASSES, dtype=np.float64)
    for _, lab in labels:
        hist += np.bincount(lab.ravel(), minlength=N_CLASSES)
    hist /= max(1.0, hist.sum())
    med = float(np.median(hist[hist > 0]))
    w = np.array([med / max(h, 1e-6) for h in hist])
    w = np.clip(w, 0.1, 20.0)
    w[2] *= line_weight  # line 类再放大：细线目标需要更强的监督
    return torch.tensor(w, dtype=torch.float32)


def _augment(frame, rng: np.random.Generator):
    """在线数据增强（colour/label 同步变换），提升路段泛化。

    关键：色相/饱和度扰动 + 模糊，逼模型学"标线结构"而不是记住
    特定路段的颜色纹理（旧模型过拟合训练路段：换路 recall 0%）。
    """
    import cv2

    colour, label = frame
    if rng.random() < 0.5:                      # 水平翻转
        colour = np.fliplr(colour).copy()
        label = np.fliplr(label).copy()
    if rng.random() < 0.8:                      # 亮度/对比度扰动（光照鲁棒：
        a = float(0.6 + 0.8 * rng.random())     # 覆盖晨/午/昏/夜差异）
        b = float(-35.0 + 70.0 * rng.random())
        colour = np.clip(colour.astype(np.float32) * a + b,
                         0, 255).astype(np.uint8)
    if rng.random() < 0.5:                      # 色温扰动：R/B 通道独立增益
        rg = float(0.88 + 0.24 * rng.random())  # （模拟清晨偏红/黄昏偏橙）
        bg = float(0.88 + 0.24 * rng.random())
        c = colour.astype(np.float32)
        c[..., 0] *= rg
        c[..., 2] *= bg
        colour = np.clip(c, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:                      # 色相/饱和度扰动（换路面颜色）
        hsv = cv2.cvtColor(colour, cv2.COLOR_RGB2HSV).astype(np.int16)
        hsv[..., 0] = (hsv[..., 0] + int(rng.integers(-20, 21))) % 180
        s_gain = float(0.6 + 1.0 * rng.random())
        hsv[..., 1] = np.clip(hsv[..., 1] * s_gain, 0, 255)
        v_gain = float(0.85 + 0.3 * rng.random())
        hsv[..., 2] = np.clip(hsv[..., 2] * v_gain, 0, 255)
        colour = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    if rng.random() < 0.35:                     # 模糊（模拟行驶运动模糊）
        k = int(rng.integers(3, 8)) | 1
        colour = cv2.GaussianBlur(colour, (k, k), 0)
    if rng.random() < 0.7:                      # 随机裁剪后缩放回原尺寸
        h, w = colour.shape[:2]
        ch, cw = int(h * 0.85), int(w * 0.85)
        y0 = int(rng.integers(0, h - ch + 1))
        x0 = int(rng.integers(0, w - cw + 1))
        colour = cv2.resize(colour[y0:y0 + ch, x0:x0 + cw], (w, h),
                            interpolation=cv2.INTER_AREA)
        label = cv2.resize(label[y0:y0 + ch, x0:x0 + cw], (w, h),
                           interpolation=cv2.INTER_NEAREST)
    if rng.random() < 0.45 and (label == 2).any():
        # 标线形态扰动：模拟远距离细线/断线/磨损，逼模型学标线结构
        # 而不是记住粗线。粗线->弥合（dilate），细线->收缩（erode），
        # 或随机擦除几段（虚线/磨损），三种都以真实物理为约束：
        # 擦除掉的标线底下是沥青路面（1），不是背景。
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        line = (label == 2).astype(np.uint8)
        r = rng.random()
        if r < 0.4:                             # 弥合断裂：标线变粗/连上
            line2 = cv2.dilate(line, k, iterations=1).astype(bool)
            label[line2] = 2
        elif r < 0.75:                          # 远处细线：标线收缩变细
            line2 = cv2.erode(line, k, iterations=1).astype(bool)
            label[line.astype(bool) & ~line2] = 1
        else:                                   # 随机擦除段：虚线/磨损
            erase = np.zeros_like(label, dtype=bool)
            for _ in range(int(rng.integers(1, 4))):
                ys = int(rng.integers(0, label.shape[0] - 6))
                xs = int(rng.integers(0, label.shape[1] - 6))
                hh = int(rng.integers(2, 12))
                ww = int(rng.integers(2, 12))
                erase[ys:ys + hh, xs:xs + ww] = True
            label[erase & line.astype(bool)] = 1
    return colour, label


def main() -> None:
    ap = argparse.ArgumentParser(description="分割训练")
    ap.add_argument("--runs", nargs="+", required=True,
                    help="数据目录（可多个，如 logs/m5_seg/run_*）")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--line-weight", type=float, default=2.0,
                    help="extra multiplier on the line class loss weight")
    ap.add_argument("--out", default=str(config.LOGS_DIR / "m5_seg" / "seg_model"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--vram-frac", type=float, default=0.6,
                    help="max fraction of GPU VRAM training may use; keeps "
                         "headroom for the running game so its rendering "
                         "never starves (white windows)")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device}")
    if device == "cuda":
        # Cap training VRAM so a concurrently running game keeps enough
        # memory to render.  Without this, batch 16 training filled the
        # whole 12 GB card and the game's window went white.
        torch.cuda.set_per_process_memory_fraction(
            max(0.1, min(1.0, args.vram_frac)))

    frames = load_frames([Path(p) for p in args.runs])
    n = len(frames)
    n_val = max(1, int(n * args.val_frac))
    train_frames = frames[:n - n_val]   # 时间序：前段训练
    val_frames = frames[n - n_val:]
    print(f"[train] 共 {n} 帧: 训练 {len(train_frames)} / 验证 {len(val_frames)}")
    print(f"[train] 类别: {CLASS_NAMES}")

    weights = median_freq_weights(train_frames,
                                   line_weight=args.line_weight)
    print(f"[train] 类别权重: {weights.tolist()}")

    model = SegUNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    crit = nn.CrossEntropyLoss(weight=weights.to(device))

    def to_tensor(frame, dev):
        colour, label = frame
        x = torch.from_numpy(colour).permute(2, 0, 1).float() / 255.0
        y = torch.from_numpy(label).long()
        return x.to(dev), y.to(dev)

    def run_epoch(fr, train: bool):
        model.train(train)
        total_loss, correct, n_pix = 0.0, 0, 0
        ious = np.zeros(N_CLASSES)
        rng = np.random.default_rng(args.seed + len(fr))
        for i in range(0, len(fr), args.batch):
            batch = fr[i:i + args.batch]
            if train:
                batch = [_augment(f, rng) for f in batch]
            xs = torch.stack([to_tensor(f, device)[0] for f in batch])
            ys = torch.stack([to_tensor(f, device)[1] for f in batch])
            if train:
                opt.zero_grad()
            # Validation must not build autograd graphs: it halves the
            # peak VRAM and keeps a concurrently running game rendering.
            with torch.set_grad_enabled(train):
                logits = model(xs)
                loss = crit(logits, ys)
            if train:
                loss.backward()
                opt.step()
            total_loss += float(loss.item()) * len(batch)
            pred = logits.argmax(dim=1)
            correct += int((pred == ys).sum())
            n_pix += int(ys.numel())
            for c in range(N_CLASSES):
                p = (pred == c)
                t = (ys == c)
                inter = int((p & t).sum())
                union = int((p | t).sum())
                ious[c] += inter / union if union else 1.0
        n_b = max(1, (len(fr) + args.batch - 1) // args.batch)
        ious /= n_b
        return total_loss / max(1, len(fr)), correct / max(1, n_pix), ious

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_miou = -1.0
    hist = {"epoch": [], "train_loss": [], "val_miou": [], "val_acc": []}
    for ep in range(args.epochs):
        t0 = time.time()
        tr_loss, tr_acc, _ = run_epoch(train_frames, train=True)
        sched.step()
        va_loss, va_acc, va_ious = run_epoch(val_frames, train=False)
        m_iou = float(va_ious.mean())
        hist["epoch"].append(ep)
        hist["train_loss"].append(round(tr_loss, 4))
        hist["val_miou"].append(round(m_iou, 4))
        hist["val_acc"].append(round(va_acc, 4))
        print(f"[train] ep {ep:02d}  loss={tr_loss:.4f} "
              f"val_acc={va_acc:.3f} val_mIoU={m_iou:.4f} "
              f"({time.time() - t0:.0f}s)")
        if m_iou > best_miou:
            best_miou = m_iou
            torch.save({
                "state_dict": model.state_dict(),
                "n_classes": N_CLASSES,
                "class_names": CLASS_NAMES,
                "val_miou": round(m_iou, 4),
                "val_ious": [round(float(v), 4) for v in va_ious],
                "val_acc": round(float(va_acc), 4),
                "weights": weights.tolist(),
            }, out_dir / "best.pt")
            print(f"[train] 保存最优 mIoU={m_iou:.4f} -> "
                  f"{out_dir / 'best.pt'}")

    (out_dir / "train_hist.json").write_text(
        json.dumps(hist, ensure_ascii=False, indent=1), encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot(hist["epoch"], hist["train_loss"], label="train loss")
        plt.plot(hist["epoch"], hist["val_miou"], label="val mIoU")
        plt.xlabel("epoch")
        plt.legend()
        plt.tight_layout()
        plt.savefig(str(out_dir / "curve.png"), dpi=110)
        print(f"[train] 曲线 -> {out_dir / 'curve.png'}")
    except Exception as exc:
        print(f"[train] 曲线保存失败: {exc}")

    print(f"[train] 完成: 最优验证 mIoU={best_miou:.4f} "
          f"-> {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()