"""Presence-model regressions: head training and label dedup (torch ok)."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from beamng_autopilot.labeling.presence_model import (
    FEATURE_DIM, PresenceNet, load_labeled_images, train_head)


class _TinyBackbone:
    """Minimal stand-in exposing e1/e2/e3/pool/mid as 1x1 convs."""

    def __init__(self):
        def blk(cin, cout):
            return torch.nn.Conv2d(cin, cout, 1)

        self.e1 = blk(3, 4)
        self.e2 = blk(4, 4)
        self.e3 = blk(4, 4)
        self.pool = torch.nn.MaxPool2d(2)
        self.mid = blk(4, FEATURE_DIM)


def test_presence_net_feature_and_logit_shapes():
    model = PresenceNet(_TinyBackbone(),
                        torch.nn.Linear(FEATURE_DIM, 1))
    x = torch.rand(2, 3, 32, 32)
    assert model.features(x).shape == (2, FEATURE_DIM)
    assert model.predict_logit(x).shape == (2,)


def test_train_head_separates_toy_features():
    # 两团可分特征 + 两个可分标签 -> 训完验证 AUC 应到 1.0
    import numpy as np
    single = torch.cat([torch.randn(8, FEATURE_DIM) + 3.0,
                        torch.randn(8, FEATURE_DIM) - 3.0]).numpy()
    feats = np.concatenate([single, single])   # 契约：2N 行（原图+翻转）
    labels = [1] * 8 + [0] * 8
    report, state = train_head(feats, labels, val_frac=0.25, epochs=150,
                               device="cpu")
    assert report["auc"] == 1.0
    assert set(state.keys()) >= {"0.weight", "0.bias"}


def test_load_labeled_images_dedupes_latest(tmp_path):
    p = tmp_path / "l.jsonl"
    rows = [
        {"path": "C:/a.png", "has_line": 0},
        {"path": "C:/b.png", "has_line": 1},
        {"path": "C:/a.png", "has_line": 1},   # 后写覆盖
        "not json",
        {"path": None, "has_line": 1},
    ]
    p.write_text("\n".join(json.dumps(r) if isinstance(r, dict) else r
                           for r in rows), encoding="utf-8")
    got = load_labeled_images(p)
    assert got == [("C:/a.png", 1), ("C:/b.png", 1)]
