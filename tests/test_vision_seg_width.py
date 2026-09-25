"""SegUNet 的容量旋钮（``width``）：默认值必须与原模型**逐位一致**。

为什么单独钉：加宽是"数据因子与训练预算都测到回报边界之后"的第三条杠杆
（第 5 轮结论：单段新数据 <1 点且需 14 seed；预算 120→240 步 +0.0092、
240→480 步没有可判定增益）。但容量对照只有在**默认值不改变现有行为**时才安全：

* ``width=1.0`` 必须与原版同层名、同通道、同参数量 → 已有 checkpoint 照旧能加载；
* 加宽只改通道数，不改接口（输入/输出形状不变），否则评估链要跟着改；
* 通道数必须是整数且不小于 8（否则小宽度会把通道压到 0，建不出模型）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beamng_autopilot.vision.segmentation import SegUNet  # noqa: E402


def _n_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def test_width_one_matches_the_default_bit_for_bit():
    a = SegUNet()
    b = SegUNet(width=1.0)
    assert set(a.state_dict()) == set(b.state_dict()), "层名必须一致"
    assert _n_params(a) == _n_params(b) == 834931
    # 默认模型存出来的权重必须能装进显式 width=1.0 的模型（反之亦然）
    b.load_state_dict(a.state_dict())
    a.load_state_dict(b.state_dict())
    assert a.width == 1.0 and b.width == 1.0


def test_widening_increases_capacity_without_changing_the_interface():
    narrow, wide = SegUNet(), SegUNet(width=2.0)
    n1, n2 = _n_params(narrow), _n_params(wide)
    assert n2 > 3 * n1, f"加宽两倍参数应约 4 倍：{n1} -> {n2}"
    assert set(narrow.state_dict()) == set(wide.state_dict()), \
        "层名不能变（否则评估/部署链的加载代码要跟着改）"
    x = torch.zeros(1, 3, 64, 48)
    assert narrow(x).shape == wide(x).shape == (1, 3, 64, 48)


def test_channel_rounding_is_integral_and_never_zero():
    m = SegUNet(width=1.5)
    # 32/64/128 × 1.5 = 48/96/192
    assert m.e1[0].out_channels == 48
    assert m.e2[0].out_channels == 96
    assert m.e3[0].out_channels == 192
    tiny = SegUNet(width=0.01)
    assert tiny.e1[0].out_channels == 8, "通道下限 8，不能被压到 0"
    with pytest.raises(TypeError):
        SegUNet(width=None)          # 不猜、不静默取默认值


def test_a_width_one_checkpoint_still_loads_after_the_change(tmp_path):
    """旧权重（没有 width 概念的 checkpoint）必须原样可用。"""
    old = tmp_path / "old.pt"
    torch.save({"state_dict": SegUNet().state_dict(),
                "train_args": {"epochs": 3, "batch": 4, "lr": 0.001}}, old)
    blob = torch.load(old, map_location="cpu", weights_only=True)
    got = SegUNet().load_state_dict(blob["state_dict"])
    assert list(got.missing_keys) == [] and list(got.unexpected_keys) == []

def test_the_evaluator_reads_the_width_from_the_checkpoint(tmp_path):
    """容量对照的 checkpoint 必须能被**评估链**加载。

    实测缺陷（2026-09-25）：评估链按默认宽度建 SegUNet，候选臂（width=2）的
    checkpoint 装进去就是一片 size mismatch，整轮判定直接崩——训练明明成功了。
    修法：宽度从 checkpoint 的 `train_args.arch_args.width` 读出来再建模型；
    结构仍然不符时给**清楚的错**，而不是静默按随机权重跑。
    """
    import numpy as np

    from beamng_autopilot.vision.segmentation import Segmenter

    good = tmp_path / "w2.pt"
    torch.save({"state_dict": SegUNet(width=2.0).state_dict(),
                "train_args": {"arch_args": {"width": 2.0}},
                "n_classes": 3}, good)
    seg = Segmenter(model_path=str(good), device="cpu", use_half=False)
    assert seg.width == 2.0
    road, line = seg.predict(np.zeros((64, 48, 3), np.uint8))
    assert road.shape == (64, 48) and line.shape == (64, 48)

    old = tmp_path / "w1.pt"
    torch.save({"state_dict": SegUNet().state_dict(),
                "train_args": {"epochs": 1}}, old)
    seg1 = Segmenter(model_path=str(old), device="cpu", use_half=False)
    assert seg1.width == 1.0, "老 checkpoint 没有 width 字段 -> 按 1.0，逐位一致"

    bad = tmp_path / "bad.pt"
    torch.save({"state_dict": SegUNet().state_dict(),
                "train_args": {"arch_args": {"width": 2.0}}}, bad)
    with pytest.raises(RuntimeError) as exc:
        Segmenter(model_path=str(bad), device="cpu", use_half=False)
    # torch 的 shape 检查先炸（"size mismatch ..."），缺键/多键由我们自己的
    # 检查兜住（默认 load_state_dict 对缺键只 warning，会静默按随机权重跑）
    msg = str(exc.value)
    assert "size mismatch" in msg or "不匹配" in msg, msg
