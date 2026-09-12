"""Region losses for the M5-B road/line segmenter's thin line class.

Weighted cross entropy alone collapses on the line class: at 0.07-0.57%
pixel share the cheap solution is "almost never predict line" (held-out
line IoU 0.04-0.13 across runs, logs/m5_seg/eval_v12_matrix.json).  Two
region-level terms attack that directly, both on the line probability
channel vs the line target channel with ignore pixels excluded:

- Tversky (alpha < beta): penalises missed line pixels (FN) harder than
  false positives, lifting thin-line recall.
- soft-clDice: skeleton-overlap loss that keeps the predicted line
  CONNECTED.  Broken-line frames are what starves the per-tick painted
  lateral reference downstream (docs/lateral_reference_diag_20260911.md).

Soft-skeleton follows the differentiable clDice formulation
(Milletari et al. 2021, github.com/jocpae/clDice), 2D cross-shaped
structuring element via separable max-pooling so it stays cheap and
AMP-safe.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

LINE_CLASS = 2
CLDICE_ITERS = 3
_EPS = 1e-6


def _soft_erode(img: torch.Tensor) -> torch.Tensor:
    p1 = -F.max_pool2d(-img, (3, 1), 1, (1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), 1, (0, 1))
    return torch.min(p1, p2)


def _soft_dilate(img: torch.Tensor) -> torch.Tensor:
    p1 = F.max_pool2d(img, (3, 1), 1, (1, 0))
    p2 = F.max_pool2d(img, (1, 3), 1, (0, 1))
    return torch.max(p1, p2)


def _soft_open(img: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(img))


def soft_skeleton(img: torch.Tensor, iters: int = CLDICE_ITERS) -> torch.Tensor:
    """Differentiable soft skeleton of a (N, 1, H, W) probability map."""
    img1 = _soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iters):
        img = _soft_erode(img)
        img1 = _soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def tversky_line_loss(prob: torch.Tensor, target: torch.Tensor,
                      valid: torch.Tensor,
                      alpha: float = 0.3, beta: float = 0.7) -> torch.Tensor:
    """Tversky loss on the line channel; beta>alpha favours line recall."""
    p = prob * valid
    t = target * valid
    tp = (p * t).sum()
    fp = (p * (1.0 - t) * valid).sum()
    fn = ((1.0 - p) * t * valid).sum()
    return 1.0 - tp / (tp + alpha * fp + beta * fn + _EPS)


def soft_cldice_line_loss(prob: torch.Tensor, target: torch.Tensor,
                          valid: torch.Tensor,
                          iters: int = CLDICE_ITERS) -> torch.Tensor:
    """1 - clDice between the skeletons of prediction and target.

    Undefined on an empty side (no skeleton to overlap): returns 0 there
    and lets the Tversky term carry the penalty instead of a NaN.
    """
    skel_p = soft_skeleton(prob * valid, iters)
    skel_t = soft_skeleton(target * valid, iters)
    sp = skel_p.sum()
    st = skel_t.sum()
    if float(sp.detach()) < _EPS or float(st.detach()) < _EPS:
        return prob.new_zeros(())
    tprec = ((skel_p * target * valid).sum() / (sp + _EPS)).clamp(0.0, 1.0)
    trec = ((skel_t * prob * valid).sum() / (st + _EPS)).clamp(0.0, 1.0)
    return 1.0 - 2.0 * tprec * trec / (tprec + trec + _EPS)


class LineSegLoss(nn.Module):
    """Weighted CE + line-channel Tversky + soft-clDice.

    ``w_tversky`` / ``w_cldice`` of 0 disable the respective region term
    and reduce to the historical weighted-CE behaviour.
    """

    def __init__(self, weight: torch.Tensor | None = None,
                 ignore_index: int = 255,
                 w_tversky: float = 1.0, w_cldice: float = 1.0,
                 cldice_iters: int = CLDICE_ITERS,
                 tversky_alpha: float = 0.3,
                 tversky_beta: float = 0.7) -> None:
        super().__init__()
        self.ignore_index = int(ignore_index)
        self.w_tversky = float(w_tversky)
        self.w_cldice = float(w_cldice)
        self.cldice_iters = int(cldice_iters)
        self.tversky_alpha = float(tversky_alpha)
        self.tversky_beta = float(tversky_beta)
        self.register_buffer("weight",
                             None if weight is None
                             else torch.as_tensor(weight, dtype=torch.float32),
                             persistent=False)

    def forward(self, logits: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.weight,
                             ignore_index=self.ignore_index)
        if self.w_tversky <= 0.0 and self.w_cldice <= 0.0:
            return ce
        prob = F.softmax(logits.float(), dim=1)[:, LINE_CLASS]
        tgt = (target == LINE_CLASS).float()
        valid = (target != self.ignore_index).float()
        loss = ce
        if self.w_tversky > 0.0:
            loss = loss + self.w_tversky * tversky_line_loss(
                prob, tgt, valid, self.tversky_alpha, self.tversky_beta)
        if self.w_cldice > 0.0:
            loss = loss + self.w_cldice * soft_cldice_line_loss(
                prob, tgt, valid, self.cldice_iters)
        return loss
