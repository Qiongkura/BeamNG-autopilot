"""Tests for the line-class region losses (seg_losses)."""

from __future__ import annotations

import pytest
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
