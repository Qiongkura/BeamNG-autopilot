"""线专用自监督预训练（去噪自编码器）——研究臂，不晋级。

动机（T16_LINE_RECOGNITION_CLOSEOUT_20261102 的可续接路径 1）：分割模型
在 71 族/214 帧池上已到平台（3 折宏平均 0.446–0.454），训练侧监督杠杆
全部证伪。本脚本用**无标签重建任务**预训练 SegUNet 的编码器
（e1/e2/e3/mid），产出的 checkpoint 可被 ``m5_train_seg.py --init`` 严格
加载（编码器权重来自自监督，解码器/头为随机初始化）。

用法（fold-A 探针）：
    python scripts/m5_ssl_pretrain.py \
        --runs <fold-A 训练集的 run 目录...> \
        --out logs/m5_seg/ssl_pretrain_foldA.pt --epochs 20
然后：
    python scripts/m5_train_seg.py --init logs/m5_seg/ssl_pretrain_foldA.pt \
        --runs <同训练集> --out logs/m5_seg/seg_model_v28_sslA ...

纪束：产物只落 logs/；不进训练池、不进导出池、不改冻结门、不晋级。
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beamng_autopilot.vision.segmentation import SegUNet  # noqa: E402


class _AutoEncoder(nn.Module):
    """SegUNet 编码器 + 轻量转置卷积解码器（只训练编码器部分有用）。"""

    def __init__(self):
        super().__init__()
        seg = SegUNet(width=1.0)
        self.e1 = seg.e1
        self.e2 = seg.e2
        self.e3 = seg.e3
        self.mid = seg.mid
        self.pool = seg.pool
        self.u2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.d2 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1),
                                nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.u1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.d1 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1),
                                nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.u0 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.d0 = nn.Sequential(nn.Conv2d(16, 3, 3, padding=1),
                                nn.ReLU(inplace=True))

    def encode(self, x):
        x1 = self.e1(x)
        x2 = self.e2(self.pool(x1))
        x3 = self.e3(self.pool(x2))
        m = self.mid(self.pool(x3))
        return x1, x2, x3, m

    def forward(self, x):
        x1, x2, x3, m = self.encode(x)
        u2 = F.interpolate(self.u2(m), size=x3.shape[-2:], mode="bilinear",
                           align_corners=False)
        d2 = self.d2(u2)
        u1 = F.interpolate(self.u1(d2), size=x2.shape[-2:], mode="bilinear",
                           align_corners=False)
        d1 = self.d1(u1)
        u0 = F.interpolate(self.u0(d1), size=x1.shape[-2:], mode="bilinear",
                           align_corners=False)
        return self.d0(u0)


def _load_frames(runs: list[str]) -> torch.Tensor:
    """读取 run 目录下全部 npz 的 colour（无标签，自监督用）。"""
    imgs = []
    for r in runs:
        for f in sorted(glob.glob(str(Path(r) / "frame_*.npz"))):
            z = np.load(f)
            c = z["colour"] if "colour" in z else z["rgb"]
            imgs.append(torch.from_numpy(np.ascontiguousarray(c)).permute(2, 0, 1).float() / 255.0)
    if not imgs:
        raise SystemExit("没有可用的帧")
    return torch.stack(imgs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="无标签帧的 run 目录")
    ap.add_argument("--out", required=True, help="SSL checkpoint 路径（.pt）")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0,
                    help="SSL 自身随机种子（重复复核时换种子）")
    args = ap.parse_args()
    torch.manual_seed(int(args.seed))

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    x = _load_frames(args.runs)
    n = x.shape[0]
    print(f"[ssl] frames={n} device={dev} epochs={args.epochs}", flush=True)

    ae = _AutoEncoder().to(dev)
    opt = torch.optim.Adam(ae.parameters(), lr=args.lr)
    g = torch.Generator().manual_seed(int(args.seed))
    t0 = time.time()
    for ep in range(args.epochs):
        perm = torch.randperm(n, generator=g)
        tot, cnt = 0.0, 0
        ae.train()
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            xb = x[idx].to(dev)
            # 去噪：输入加噪声，目标原图
            xb_noisy = (xb + 0.05 * torch.randn_like(xb)).clamp(0, 1)
            rec = ae(xb_noisy)
            loss = F.l1_loss(rec, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
            cnt += len(idx)
        print(f"[ssl] ep {ep} L1={tot / max(1, cnt):.5f} ({time.time() - t0:.0f}s)", flush=True)

    # 组装可被 --init 严格加载的 SegUNet state_dict：编码器来自 SSL，其余随机
    seg = SegUNet(width=1.0)
    sd = seg.state_dict()
    enc = ae.state_dict()
    for k in list(sd.keys()):
        if k.split(".")[0] in ("e1", "e2", "e3", "mid"):
            sd[k] = enc[k]
    ck = {
        "state_dict": sd,
        "optimizer": None,
        "scheduler": None,
        "scaler": None,
        "next_epoch": args.epochs,
        "best_miou": None,
        "best_task": None,
        "hist": [],
        "train_args": {
            "init_from": None,
            "init_sha16": None,
            "init_source": "random",   # 链根（编码器来自无监督重建，非监督预训练）
            "ssl_pretrain": {
                "frames": n,
                "epochs": args.epochs,
                "batch": args.batch,
                "lr": args.lr,
                "runs": list(args.runs),
                "encoder_keys": ["e1", "e2", "e3", "mid"],
            },
            "epochs": args.epochs,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, out)
    print(f"[ssl] saved -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
