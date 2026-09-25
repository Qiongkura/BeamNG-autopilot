"""Tests for the line-class region losses (seg_losses)."""

from __future__ import annotations

import pytest
import numpy as np
import torch

from beamng_autopilot.vision.seg_losses import (
    LineSegLoss, soft_cldice_line_loss, soft_skeleton, tversky_line_loss,
)


def _mask(lines: list[tuple[int, int, int, int]], h: int = 64, w: int = 64,
          thick: int = 2) -> torch.Tensor:
    """Row-band line segments as (1, 1, H, W) float masks."""
    m = torch.zeros(1, 1, h, w)
    for r0, r1, c0, c1 in lines:
        m[:, :, r0:r1, c0:c1 + max(1, thick)] = 1.0
    return m


def test_soft_skeleton_thins_a_band():
    band = _mask([(20, 22, 10, 40)], thick=2)
    skel = soft_skeleton(band, iters=4)
    assert skel.shape == band.shape
    assert float(skel.sum()) > 0.0
    # skeleton must not exceed the original band
    assert float(skel.sum()) <= float(band.sum()) + 1e-4


def test_cldice_prefers_connected_prediction():
    target = _mask([(20, 21, 8, 48)])            # single-row band
    valid = torch.ones_like(target)
    perfect = target.clone()
    broken = target.clone()
    broken[:, :, 20, 20:32] = 0.0        # middle of the line wiped out
    off = torch.zeros_like(target)
    off[:, :, 40:42, 8:48] = 1.0         # same length, wrong place
    l_perfect = float(soft_cldice_line_loss(perfect, target, valid))
    l_broken = float(soft_cldice_line_loss(broken, target, valid))
    l_off = float(soft_cldice_line_loss(off, target, valid))
    assert l_perfect < 0.05
    assert l_broken > l_perfect
    assert l_off > l_perfect


def test_tversky_penalises_missed_lines_harder():
    target = _mask([(20, 21, 8, 48)])            # single-row band, len 41
    valid = torch.ones_like(target)
    # identical 20-px overlap with the target; the remaining 21 px are
    # either MISSED (fn=21, fp=0) or HALLUCINATED (fp=21, fn=0)
    half_miss = torch.zeros_like(target)
    half_miss[:, :, 20, 8:28] = 1.0
    full_plus_fp = torch.zeros_like(target)
    full_plus_fp[:, :, 20, 8:49] = 1.0
    full_plus_fp[:, :, 40, 8:29] = 1.0
    l_miss = float(tversky_line_loss(half_miss, target, valid, 0.3, 0.7))
    l_fp = float(tversky_line_loss(full_plus_fp, target, valid, 0.3, 0.7))
    assert l_miss > l_fp                 # FN costs more than equal FP


def test_empty_cases_stay_finite():
    empty_pred = torch.zeros(1, 1, 32, 32)
    empty_tgt = torch.zeros(1, 1, 32, 32)
    valid = torch.ones_like(empty_tgt)
    assert float(tversky_line_loss(empty_pred, empty_tgt, valid)) >= 0.0
    assert float(soft_cldice_line_loss(empty_pred, empty_tgt, valid)) == 0.0
    blob = _mask([(5, 7, 4, 20)], 32, 32)
    # no target skeleton -> clDice is 0 by design (Tversky carries it)
    assert float(soft_cldice_line_loss(blob, empty_tgt, valid)) == 0.0
    # ... and Tversky there is a full penalty: finite and maximal
    assert float(tversky_line_loss(blob, empty_tgt, valid)) == pytest.approx(1.0)


def test_linesegloss_backward_and_ignore_mask():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 32, 32, requires_grad=True)
    target = torch.full((2, 32, 32), 1)
    target[:, 16, 8:24] = 2              # a line
    target[:, 0, :] = 255                # ignored border
    crit = LineSegLoss(w_tversky=1.0, w_cldice=1.0)
    loss = crit(logits, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_linesegloss_zero_weights_equal_ce():
    torch.manual_seed(1)
    logits = torch.randn(1, 3, 16, 16)
    target = torch.randint(0, 3, (1, 16, 16))
    crit = LineSegLoss(w_tversky=0.0, w_cldice=0.0)
    ce = torch.nn.functional.cross_entropy(logits, target)
    assert float(crit(logits, target)) == pytest.approx(float(ce))


def test_a_masked_class_gets_exactly_zero_gradient():
    """被屏蔽的类必须**既不收正样本也不收负样本**：line 列梯度恒为 0。

    这是"整条通道忽略"的判据。只把类别权重置零做不到这一点——未标注像素仍
    在 softmax 分母里当负样本（等于教"未标注的可见漆线=背景"）。
    """
    import torch

    from beamng_autopilot.vision.seg_losses import (LINE_CLASS,
                                                    masked_cross_entropy)
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 6, 6, requires_grad=True)
    target = torch.randint(0, 2, (2, 6, 6))          # 只有背景/路面
    mask = torch.tensor([True, True, False])         # 屏蔽 line
    loss = masked_cross_entropy(logits, target, mask)
    loss.backward()
    assert torch.count_nonzero(logits.grad[:, LINE_CLASS]) == 0, \
        "line 通道拿到了梯度，说明它仍在分母里当负样本"
    assert torch.count_nonzero(logits.grad[:, :2]) > 0, "其它类要正常学习"

    # 逐样本掩码：第 1 个样本屏蔽 line、第 0 个不屏蔽
    logits2 = torch.randn(2, 3, 6, 6, requires_grad=True)
    per_sample = torch.tensor([[True, True, True], [True, True, False]])
    masked_cross_entropy(logits2, target, per_sample).backward()
    assert torch.count_nonzero(logits2.grad[1, LINE_CLASS]) == 0
    assert torch.count_nonzero(logits2.grad[0, LINE_CLASS]) > 0


def test_a_blocked_class_may_not_appear_as_a_target():
    """自相矛盾的标签（屏蔽了 line 却又有 line 目标）要报错，不能算成 inf。"""
    import pytest as _pytest
    import torch

    from beamng_autopilot.vision.seg_losses import masked_cross_entropy
    logits = torch.randn(1, 3, 4, 4)
    target = torch.zeros(1, 4, 4, dtype=torch.long)
    target[0, 1, 1] = 2
    with _pytest.raises(ValueError, match="被屏蔽的类出现在目标里"):
        masked_cross_entropy(logits, target, torch.tensor([True, True, False]))


def test_masking_the_line_channel_drops_the_region_terms():
    """整通道屏蔽时不计算 line 区域项（Tversky/clDice 拿不到线目标）。"""
    import pytest as _pytest
    import torch

    from beamng_autopilot.vision.seg_losses import (LineSegLoss,
                                                    masked_cross_entropy)
    torch.manual_seed(1)
    logits = torch.randn(2, 3, 8, 8)
    target = torch.randint(0, 2, (2, 8, 8))
    mask = torch.tensor([True, True, False])
    crit = LineSegLoss(w_tversky=1.0, w_cldice=1.0)
    masked = crit(logits, target, class_mask=mask)
    ce_only = masked_cross_entropy(logits, target, mask)
    assert float(masked) == pytest.approx(float(ce_only), rel=1e-6), \
        "区域项不该在整通道屏蔽时混进来"

    # 混批（部分样本可信 line）+ 开着的区域项 = 调用方该拆批，直接报错
    mixed = torch.tensor([[True, True, True], [True, True, False]])
    with _pytest.raises(ValueError, match="请拆批"):
        crit(logits, target, class_mask=mixed)
    # 关掉区域项后，混批是可算的
    assert float(LineSegLoss(w_tversky=0.0, w_cldice=0.0)(
        logits, target, class_mask=mixed)) > 0.0


def _logits(*, line_hot: bool, h=16, w=16) -> torch.Tensor:
    """(1, 3, h, w) 的 logits；``line_hot`` 决定在中间一行预测标线。"""
    x = torch.zeros(1, 3, h, w)
    x[:, 0] = 2.0                    # background 低
    x[:, 1] = 1.0                    # road 中
    if line_hot:
        x[:, 2, h // 2, :] = 5.0     # 中间一行强烈预测 line
    return x


def _label(*, unknown_band=False, line=True, h=16, w=16) -> torch.Tensor:
    lab = np.zeros((1, h, w), np.uint8)
    lab[0, 4:12, :] = 1              # road
    if line:
        lab[0, h // 2, :] = 2
    if unknown_band:
        lab[0, 0:2, :] = 255         # 顶部一条未知区
    return torch.from_numpy(lab.astype(np.int64))


def test_an_unknown_region_contributes_no_supervision():
    """1：把某区域改成 UNKNOWN 后，那里预测什么都不影响损失。"""
    crit = LineSegLoss()
    lab = _label(unknown_band=True)
    a = _logits(line_hot=False)      # 未知区里没预测线
    b = _logits(line_hot=True)       # 未知区外才有线；这里只改"未知区外"的预测
    base = float(crit(a, lab, class_mask=None))
    # 在**未知区之外**改预测 -> 损失必须变（说明监督还在）
    assert abs(float(crit(b, lab, class_mask=None)) - base) > 1e-6
    # 在**未知区之内**改预测 -> 损失必须不变
    c = a.clone()
    c[:, 2, 0:2, :] = 5.0            # 未知区里强烈预测 line
    assert abs(float(crit(c, lab, class_mask=None)) - base) < 1e-6, \
        "未知区不得贡献任何监督"
