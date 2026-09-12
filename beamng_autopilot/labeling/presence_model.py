"""标线存在性分类器：SegUNet 骨干（冻结）+ 池化小头.

宫格人工标签（has_line 0/1）训练二分类头，输出"含标线概率"。骨干特征
来自已训练的分割模型（mid 瓶颈 128 通道），推理时一次前向同时可得
分割与存在性。与 line 掩码面积阈值法（presence.py）互补：学习式头
在新地图/新光照下更稳，也直接可用作强化学习的奖励/筛选信号。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from beamng_autopilot.labeling.presence import _auc

FEATURE_DIM = 128          # SegUNet mid 瓶颈通道数
INFER_W, INFER_H = 536, 403


class PresenceNet:
    """冻结 SegUNet 骨干 + 可训练池化头（推理用组合）。"""

    def __init__(self, backbone, head):
        self.backbone = backbone
        self.head = head

    def features(self, x):
        """SegUNet 编码器到 mid 瓶颈，全局平均池化 -> N×128。"""
        import torch
        import torch.nn.functional as F
        b = self.backbone
        with torch.no_grad():
            x1 = b.e1(x)
            x2 = b.e2(b.pool(x1))
            x3 = b.e3(b.pool(x2))
            m = b.mid(b.pool(x3))
        return F.adaptive_avg_pool2d(m, 1).flatten(1)

    def predict_logit(self, x):
        """x: NCHW float tensor -> N logits。"""
        return self.head(self.features(x)).squeeze(-1)


def build_presence_model(backbone_ckpt_path: Path, device: str = "cuda"):
    """从分割 checkpoint 构建冻结骨干 + 随机初始化头。"""
    import torch
    from beamng_autopilot.vision.segmentation import N_CLASSES, SegUNet

    ckpt = torch.load(backbone_ckpt_path, map_location=device)
    backbone = SegUNet(n_classes=int(ckpt.get("n_classes", N_CLASSES)))
    backbone.load_state_dict(ckpt["state_dict"])
    backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    head = torch.nn.Sequential(
        torch.nn.Linear(FEATURE_DIM, 64), torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(0.2), torch.nn.Linear(64, 1))
    head.to(device)
    return PresenceNet(backbone, head)


def load_labeled_images(labels_path: Path):
    """JSONL -> [(唯一 path, has_line)] 按路径去重，后写覆盖。"""
    latest: dict[str, int] = {}
    for line in Path(labels_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("path") and rec.get("has_line") is not None:
            latest[rec["path"]] = int(rec["has_line"])
    return sorted(latest.items())


def extract_features(model: PresenceNet, paths: list[str], device: str,
                     logger=None) -> np.ndarray:
    """每张图 forward 一次，返回 2N×128 特征（偶数行原图、奇数行水平翻转）。"""
    import cv2
    import torch

    feats = []
    buf: list = []

    def flush():
        if buf:
            feats.append(torch.cat(buf).cpu().numpy())
            buf.clear()

    for i, p in enumerate(paths):
        bgr = cv2.imread(p)
        if bgr is None:
            raise FileNotFoundError(p)
        rgb = cv2.cvtColor(cv2.resize(bgr, (INFER_W, INFER_H),
                                      interpolation=cv2.INTER_AREA),
                           cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)
        x = x.unsqueeze(0).to(device)
        buf.append(torch.cat([model.features(x),
                              model.features(torch.flip(x, dims=[3]))]))
        if len(buf) >= 16:
            flush()
        if logger and (i + 1) % 100 == 0:
            logger(f"[presence-train] features {i + 1}/{len(paths)}")
    flush()
    return np.concatenate(feats)


def _head_metrics(scores: list[float], ys: list[int]) -> dict:
    auc = _auc(ys, scores)
    best_acc, best_thr = -1.0, 0.0
    for t in sorted(set(scores)):
        acc = float(np.mean([(1 if v >= t else 0) == yy
                             for v, yy in zip(scores, ys)]))
        if acc > best_acc:
            best_acc, best_thr = acc, t
    return {"auc": auc, "acc": best_acc, "threshold": best_thr}


def train_head(features: np.ndarray, labels: list[int], val_frac: float = 0.2,
               epochs: int = 300, lr: float = 1e-3, seed: int = 42,
               device: str = "cuda") -> tuple[dict, dict]:
    """冻结特征上训小头：分层划分、BCE（pos_weight 平衡）、翻转增强。

    返回 (report, best_head_state_dict)。report 含 val AUC / acc / 阈值。
    """
    import random
    import torch

    rng = random.Random(seed)
    idx_pos = [i for i, y in enumerate(labels) if y == 1]
    idx_neg = [i for i, y in enumerate(labels) if y == 0]
    rng.shuffle(idx_pos)
    rng.shuffle(idx_neg)
    val_idx = set(idx_pos[:max(1, round(len(idx_pos) * val_frac))]
                  + idx_neg[:max(1, round(len(idx_neg) * val_frac))])
    tr_idx = [i for i in range(len(labels)) if i not in val_idx]
    va_idx = sorted(val_idx)

    n = len(labels)
    x = torch.from_numpy(features).float().to(device)       # 2N×128
    y = torch.tensor(labels * 2, dtype=torch.float32, device=device)
    tr = tr_idx + [i + n for i in tr_idx]                   # 原图+翻转都进训练
    pos = float(y[tr].sum())
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor((len(tr) - pos) / max(pos, 1.0),
                                device=device))
    head = torch.nn.Sequential(
        torch.nn.Linear(FEATURE_DIM, 64), torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(0.2), torch.nn.Linear(64, 1)).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)

    best = {"auc": -1.0, "state": None}
    for ep in range(epochs):
        head.train()
        opt.zero_grad()
        loss = loss_fn(head(x[tr]).squeeze(-1), y[tr])
        loss.backward()
        opt.step()
        if (ep + 1) % 10 == 0 or ep == epochs - 1:
            head.eval()
            with torch.no_grad():
                s = head(x[va_idx]).squeeze(-1).cpu().numpy().tolist()
            m = _head_metrics(s, [labels[i] for i in va_idx])
            if m["auc"] > best["auc"]:
                best = {"auc": m["auc"], "acc": m["acc"],
                        "threshold": m["threshold"], "epoch": ep,
                        "state": {k: v.detach().clone()
                                  for k, v in head.state_dict().items()}}

    state = best.pop("state")
    return best, state


def train_and_save(labels_path: Path, backbone_ckpt: Path, out_path: Path,
                   val_frac: float = 0.2, epochs: int = 300, seed: int = 42,
                   device: str = "cuda", logger=None) -> dict:
    """完整流程：读标签 -> 特征 -> 训头 -> 存 checkpoint。返回验证报告。"""
    import torch

    labeled = load_labeled_images(labels_path)
    if len(labeled) < 10:
        raise ValueError(f"标签太少: {len(labeled)}")
    paths = [p for p, _ in labeled]
    labels = [y for _, y in labeled]
    if logger:
        logger(f"[presence-train] {len(paths)} labeled frames "
               f"(pos {sum(labels)} / neg {len(labels) - sum(labels)})")

    model = build_presence_model(backbone_ckpt, device)
    feats = extract_features(model, paths, device, logger=logger)
    report, state = train_head(feats, labels, val_frac=val_frac,
                               epochs=epochs, seed=seed, device=device)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"head_state_dict": state,
                "backbone_ckpt": str(backbone_ckpt),
                "feature_dim": FEATURE_DIM,
                "threshold": report["threshold"],
                "val_auc": report["auc"],
                "val_acc": report["acc"],
                "n_labeled": len(paths)}, out_path)
    if logger:
        logger(f"[presence-train] val AUC {report['auc']:.4f} "
               f"acc {report['acc']:.4f} thr {report['threshold']:.4f} "
               f"-> {out_path}")
    return report


def score_images(ckpt_path: Path, paths: list[str], device: str = "cuda",
                 logger=None) -> dict[str, float]:
    """用训练好的存在性头给一批图片打分 -> {路径: 含标线概率}。"""
    import cv2
    import torch

    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_presence_model(Path(ckpt["backbone_ckpt"]), device)
    head = torch.nn.Sequential(
        torch.nn.Linear(ckpt["feature_dim"], 64),
        torch.nn.ReLU(inplace=True), torch.nn.Dropout(0.2),
        torch.nn.Linear(64, 1)).to(device)
    head.load_state_dict(ckpt["head_state_dict"])
    model.head = head

    out: dict[str, float] = {}
    buf: list = []
    names: list = []

    def flush():
        if buf:
            with torch.no_grad():
                s = torch.sigmoid(
                    model.predict_logit(torch.cat(buf))).cpu().numpy()
            for nm, v in zip(names, s):
                out[nm] = round(float(v), 6)
            buf.clear()
            names.clear()

    for i, p in enumerate(paths):
        bgr = cv2.imread(p)
        if bgr is None:
            if logger:
                logger(f"[presence] skip unreadable {p}")
            continue
        rgb = cv2.cvtColor(cv2.resize(bgr, (INFER_W, INFER_H),
                                      interpolation=cv2.INTER_AREA),
                           cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)
        buf.append(x.unsqueeze(0).to(device))
        names.append(p)
        if len(buf) >= 32:
            flush()
        if logger and (i + 1) % 200 == 0:
            logger(f"[presence] scored {i + 1}/{len(paths)}")
    flush()
    return out
