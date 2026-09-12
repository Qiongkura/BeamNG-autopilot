"""标线存在性打分与人工标签一致性评估.

用分割模型的 line 掩码面积当"含标线"连续分数，与宫格标注的 0/1 人工
标签对比：AUC + 最优阈值混淆矩阵 + 不一致帧清单。不一致帧即模型漏检/
误检，是下一批最值得人工复核的图（主动学习闭环）；分数可直接喂
``m5_line_grid_labeler.py --strategy score --scores``。
"""

from __future__ import annotations

import json
from pathlib import Path


def line_fraction(line_mask) -> float:
    """line 掩码占整图像素比，作为"含标线"的连续分数。"""
    import numpy as np
    mask = np.asarray(line_mask)
    if mask.size == 0:
        return 0.0
    return float(mask.astype(bool).mean())


def score_images(model_path, paths: list[Path], logger=None) -> dict[str, float]:
    """对一批图片跑分割模型，返回 {路径字符串: line 面积分数}。

    torch/cv2 延迟导入，纯逻辑测试不需要加载模型。
    """
    import cv2
    from beamng_autopilot.vision.segmentation import Segmenter

    seg = Segmenter(model_path=model_path)
    scores: dict[str, float] = {}
    for i, p in enumerate(paths):
        p = Path(p)
        bgr = cv2.imread(str(p))
        if bgr is None:
            if logger:
                logger(f"[presence] skip unreadable {p.name}")
            continue
        _, line = seg.predict(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        scores[str(p)] = line_fraction(line)
        if logger and (i + 1) % 50 == 0:
            logger(f"[presence] scored {i + 1}/{len(paths)}")
    return scores


def _auc(labels: list[int], scores: list[float]) -> float:
    """秩和（Mann-Whitney）AUC，正类排名越高 AUC 越接近 1，含并列处理。"""
    pos = [s for y, s in zip(labels, scores) if y == 1]
    neg = [s for y, s in zip(labels, scores) if y == 0]
    if not pos or not neg:
        return float("nan")
    ranked = sorted(enumerate(scores), key=lambda t: t[1])
    # 并列分数取平均秩（1-based）
    rank = [0.0] * len(scores)
    i = 0
    while i < len(ranked):
        j = i
        while j + 1 < len(ranked) and ranked[j + 1][1] == ranked[i][1]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            rank[ranked[k][0]] = avg
        i = j + 1
    r_pos = sum(rank[idx] for idx, _ in enumerate(labels) if labels[idx] == 1)
    return (r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def _best_threshold(labels: list[int], scores: list[float]) -> float:
    """按准确率最优的分数阈值（并列分数同侧，取中点）。"""
    pairs = sorted(zip(scores, labels))
    best_thr, best_acc = pairs[0][0] - 1.0, -1.0
    for i in range(len(pairs)):        # 阈值落在 pairs[i] 与 pairs[i+1] 之间
        thr = pairs[i][0]
        tp = tn = 0
        for s, y in pairs:
            pred = 1 if s >= thr else 0
            tp += pred == 1 and y == 1
            tn += pred == 0 and y == 0
        acc = (tp + tn) / len(pairs)
        if acc > best_acc:
            best_acc, best_thr = acc, thr
    return best_thr


def presence_agreement(labels: dict[str, int], scores: dict[str, float]) -> dict:
    """人工 0/1 标签 vs 模型连续分数的一致性报告（纯数学，可离线回归）。"""
    keys = [k for k in labels if k in scores]
    if not keys:
        return {"n": 0}
    ys = [int(labels[k]) for k in keys]
    ss = [float(scores[k]) for k in keys]
    auc = _auc(ys, ss)
    thr = _best_threshold(ys, ss)
    tp = fp = tn = fn = 0
    disagreements = []
    for k, y, s in zip(keys, ys, ss):
        pred = 1 if s >= thr else 0
        if pred == 1 and y == 1:
            tp += 1
        elif pred == 1 and y == 0:
            fp += 1
        elif pred == 0 and y == 0:
            tn += 1
        else:
            fn += 1
        if pred != y:
            disagreements.append({"key": k, "label": y, "score": round(s, 6)})
    disagreements.sort(key=lambda d: abs(d["score"] - d["label"]), reverse=True)
    n = len(keys)
    pos = sum(ys)
    return {
        "n": n, "pos": pos, "neg": n - pos,
        "auc": round(auc, 4) if auc == auc else None,
        "threshold": round(thr, 6),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": round((tp + tn) / n, 4),
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "recall": round(tp / (tp + fn), 4) if tp + fn else None,
        "disagreements": disagreements,
    }


def load_label_records(path: Path) -> list[dict]:
    """读宫格标注 JSONL 为记录列表（坏行跳过）。"""
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("path") is not None and rec.get("has_line") is not None:
            records.append(rec)
    return records
