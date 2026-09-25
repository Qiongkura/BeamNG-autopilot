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


def masked_cross_entropy(logits: torch.Tensor, target: torch.Tensor,
                         class_mask, *, weight: torch.Tensor | None = None,
                         ignore_index: int = 255) -> torch.Tensor:
    """类屏蔽交叉熵：被屏蔽的类**从 softmax 的分母里去掉**。

    为什么不能只把该类的权重置零：权重只影响"正样本"的拉力，未标注像素仍留在
    分母里当负样本——那等于教模型"未标注的可见漆线就是背景"，正是 T14 §1 禁止的
    错误监督（实测：引擎标注把可见漆线标成沥青）。把该类的 logit 置 ``-inf`` 后，
    softmax 只在允许的类上归一化，该通道既无正样本也无负样本，反向梯度恒为 0。

    ``class_mask`` 支持 ``[C]``（整批一致）与 ``[B, C]``（逐样本）；
    被屏蔽的类**不允许**作为目标出现（否则是自相矛盾的标签，直接报错而不是
    让损失变成 inf 混过去）。
    """
    mask = torch.as_tensor(class_mask, dtype=torch.bool, device=logits.device)
    if mask.dim() == 1:
        blocked = (~mask).view(1, -1, 1, 1).expand_as(logits)
    elif mask.dim() == 2:
        blocked = (~mask).view(mask.shape[0], mask.shape[1], 1, 1).expand_as(
            logits)
    else:
        raise ValueError(f"class_mask 维度只支持 [C] 或 [B,C]，得到 {mask.shape}")
    bad = blocked & (target.unsqueeze(1) == torch.arange(
        logits.shape[1], device=logits.device).view(1, -1, 1, 1))
    if bool(bad.any()):
        raise ValueError(
            "被屏蔽的类出现在目标里：先去掉该类的目标像素（例如 "
            "mask_line_for_loss 把已标注 line 改成 255），否则损失无定义")
    # 全 ignore 的帧（或全 ignore 的批）：`F.cross_entropy` 对"零个有效像素"
    # 求均值会得到 **NaN**（实测：一张全 255 的标签就能让整步变 NaN）。
    # 这里返回"零损失但保留计算图"：不 NaN、也不制造假的梯度。
    if not bool((target != int(ignore_index)).any()):
        return logits.sum() * 0.0
    return F.cross_entropy(logits.masked_fill(blocked, float("-inf")), target,
                           weight=weight, ignore_index=ignore_index)


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

    def forward(self, logits: torch.Tensor, target: torch.Tensor,
                class_mask=None) -> torch.Tensor:
        """``class_mask`` 见 :func:`masked_cross_entropy`。

        整条通道被屏蔽时（``class_mask`` 不含 ``LINE_CLASS``）**不计算**
        line 的区域项：区域项拿不到"哪些像素是线"，硬算等于用未标注的像素当
        负样本。逐样本掩码若与本批的 line 权重同用，说明这一批混了"可信/不可信"
        两种帧——那是调用方该拆批的场景，直接报错而不是悄悄改损失口径。
        """
        # 整帧/整批都是 ignore：没有任何监督可算。直接交给 F.cross_entropy
        # 会得到 **NaN**（零个有效像素求均值），NaN 会顺着反向传播污染整步权重。
        # 返回"零损失但保留计算图"：不 NaN、不制造假梯度（方案 W1 §6.2 的第 4 条）。
        if not bool((target != self.ignore_index).any()):
            return logits.sum() * 0.0
        line_allowed = True
        if class_mask is not None:
            m = torch.as_tensor(class_mask, dtype=torch.bool)
            line_allowed = bool(m[LINE_CLASS]) if m.dim() == 1                 else bool(m[:, LINE_CLASS].all())
            if not line_allowed and (self.w_tversky > 0.0
                                     or self.w_cldice > 0.0):
                if m.dim() == 2 and bool(m[:, LINE_CLASS].any()):
                    raise ValueError(
                        "这一批同时含'可信 line'与'不可信 line'的帧，而 line 区域"
                        "项是开着的一一请拆批，或把 --line-tversky-weight / "
                        "--line-cldice-weight 设为 0（路面通道实验的常规配方）")
                # 整批禁用：丢掉区域项（不静默改口径，调用方在 CLI 里会看到提示）
                region_w = 0.0
            else:
                region_w = 1.0
        else:
            region_w = 1.0
        if class_mask is not None:
            ce = masked_cross_entropy(logits, target, class_mask,
                                      weight=self.weight,
                                      ignore_index=self.ignore_index)
        else:
            ce = F.cross_entropy(logits, target, weight=self.weight,
                                 ignore_index=self.ignore_index)
        if not region_w or (self.w_tversky <= 0.0 and self.w_cldice <= 0.0):
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
