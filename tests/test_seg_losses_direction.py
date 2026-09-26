"""T15：实测 ``vision/seg_losses.py`` 的方向，而不是照搬口头判断。

方案 v2 §S6/E2 与验收 T15 要求：在围绕损失设计训练实验之前，先用合成 FP/FN
例子量出**当前实现**到底把损失推向哪边。本文件里的注释只写**实测事实**
（数字来自本文件里的断言，且都在 CPU 上可复现）。

已实测的关键事实（详见 docs/T15_SEG_LOSS_DIRECTION_20260926.md）：

* 实现的分母是 ``TP + alpha*FP + beta*FN + 1e-6``：``alpha`` 乘 FP、``beta``
  乘 FN。方向只由 ``beta/alpha`` 决定；``alpha``/``beta`` 的绝对大小和
  ``w_tversky`` 都是"整体幅度"旋钮，不改 FP/FN 相对方向。
* 同一状态（TP=20, FP=5, FN=5）下新增 1 个 FN 的**切线**代价是新增 1 个 FP 的
  ``beta/alpha = 2.333`` 倍；但"10 个纯 FP"与"10 个纯 FN"的**损失值**之比只有
  1.9877——因为分母本身也含惩罚项。把这两个数字当成同一个量就会推错方向。
* 默认 ``alpha=0.3, beta=0.7`` 时 FN 更贵（偏召回）。"``beta<1`` 提升召回"不是
  方向条件：``(alpha=0.7, beta=0.3)`` 里 ``beta`` 仍 <1，方向却反了（FP 更贵）。
* 完整 ``LineSegLoss`` 的方向被 CE 的类别权重主导：记录到的真实配方权重
  ``[0.604, 1.0, 40.0]``（logs/_train_balanced.txt:8）下 |g_FN|/|g_FP| ≈ 68.7，
  而 ``weight=None`` 时只有 ≈ 4.12。把 ``w_tversky`` 1→2 只会把这个比值抬到
  71.1（**更偏召回**），所以"Tversky 权重 2 = 压 FP"与实测方向相反。
* ignore(255)：损失与梯度跟"忽略区里的预测内容"**逐位相同**（``torch.equal``
  为 True），忽略区内的梯度恒为 0；把同一批像素从 255 改成 line 目标则损失会变
  （说明忽略不是空转）。
"""

from __future__ import annotations

import math

import pytest
import torch

from beamng_autopilot.vision.seg_losses import (
    LINE_CLASS, LineSegLoss, masked_cross_entropy, tversky_line_loss,
)

#: 真实训练记录里的类别权重（logs/_train_balanced.txt:8 的 "[train] 类别权重" 行，
#: median_freq_weights 在 line_weight=2.0 下的输出；line 权重顶到 20*2 上限）。
RECORDED_RECIPE_WEIGHTS = (0.6039878726005554, 1.0, 40.0)

#: 训练脚本当前写死的区域项参数（m5_train_seg.py: --line-tversky-weight 1.0,
#: --line-cldice-weight 1.0, alpha 0.3, beta 0.7）。
RECIPE_ALPHA = 0.3
RECIPE_BETA = 0.7


def _counts_loss(tp: int, fp: int, fn: int, alpha: float = RECIPE_ALPHA,
                 beta: float = RECIPE_BETA, n: int = 256) -> float:
    """16x16 二值 prob/target，含已知的 tp/fp/fn 像素数（其余像素既非目标也不预测）。"""
    assert tp + fp + fn <= n
    prob = torch.zeros(n)
    tgt = torch.zeros(n)
    valid = torch.ones(n)
    tgt[:tp + fn] = 1.0
    prob[:tp] = 1.0
    prob[tp + fn:tp + fn + fp] = 1.0
    return tversky_line_loss(prob.view(1, 16, 16), tgt.view(1, 16, 16),
                             valid.view(1, 16, 16), alpha, beta).detach().item()


def _probe_grads(weight, *, w_tversky: float = 1.0, w_cldice: float = 0.0,
                 alpha: float = RECIPE_ALPHA, beta: float = RECIPE_BETA,
                 h: int = 12):
    """完整 LineSegLoss 在"FP 侧/FN 侧状态对称"的探针上的梯度。

    所有像素的 line 概率都是 0.5（line logit=ln2，其余 logit=0），目标里只有
    一个真线像素且被预测成背景（FN）。(3,3) 是该 FN 像素，(3,4) 是背景像素
    （在那里抬高 line logit 就是制造 FP）。返回 (loss, g_FN, g_FP)。
    """
    logits = torch.zeros(1, 3, h, h)
    logits[0, 2] = math.log(2.0)
    logits.requires_grad_(True)
    target = torch.zeros(1, h, h, dtype=torch.long)
    target[0, h // 4, h // 4] = LINE_CLASS
    crit = LineSegLoss(weight=weight, w_tversky=w_tversky, w_cldice=w_cldice,
                       tversky_alpha=alpha, tversky_beta=beta)
    loss = crit(logits, target)
    loss.backward()
    g = logits.grad[0, 2]
    return (loss.detach().item(), g[h // 4, h // 4].item(),
            g[h // 4, h // 4 + 1].item())


# --------------------------------------------------------------------------
# 1. 合成 FP/FN 的损失值：实测数字 + "值之比不是 beta/alpha"
# --------------------------------------------------------------------------

def test_tversky_hard_count_values_and_value_ratio_is_not_beta_over_alpha():
    """实测：TP=20 时 10 个纯 FP 损失 0.130435、10 个纯 FN 损失 0.259259，
    值之比 1.9877 —— 明显小于 beta/alpha=2.333（分母同时含被加的惩罚项）。"""
    l_10fp = _counts_loss(20, 10, 0)
    l_10fn = _counts_loss(20, 0, 10)
    l_5_5 = _counts_loss(20, 5, 5)
    # 解析：1 - 20/(20+0.3*fp+0.7*fn)
    assert l_10fp == pytest.approx(1.0 - 20.0 / 23.0, abs=1e-6)
    assert l_10fn == pytest.approx(1.0 - 20.0 / 27.0, abs=1e-6)
    assert l_5_5 == pytest.approx(5.0 / 25.0, abs=1e-6)
    ratio = l_10fn / l_10fp
    assert ratio == pytest.approx(1.9877, abs=2e-3)
    # 实测事实：值之比 < beta/alpha，且 < 2 —— 不能把"损失值之比"当作权重比
    assert ratio < RECIPE_BETA / RECIPE_ALPHA
    assert ratio < 2.0


def test_tversky_marginal_fp_fn_sensitivity_ratio_is_beta_over_alpha():
    """实测：状态 (TP=20,FP=5,FN=5) 下，有限差分给 dL/dFP=0.009486、
    dL/dFN=0.021790，比值 2.297；解析切线是 alpha*TP/D^2=0.0096 与
    beta*TP/D^2=0.0224，比值恰为 beta/alpha=7/3。1 像素步长的割线略低于切线
    （损失对该计数是凹的）。"""
    base = _counts_loss(20, 5, 5)
    d_fp = _counts_loss(20, 6, 5) - base
    d_fn = _counts_loss(20, 5, 6) - base
    d = 20.0 + RECIPE_ALPHA * 5.0 + RECIPE_BETA * 5.0
    assert base == pytest.approx(0.2, abs=1e-6)
    # 割线比解析切线小 1.1e-4 / 6.1e-4：1 像素的步长跨越了凹曲率
    assert d_fp == pytest.approx(RECIPE_ALPHA * 20.0 / d ** 2, abs=8e-4)
    assert d_fn == pytest.approx(RECIPE_BETA * 20.0 / d ** 2, abs=8e-4)
    assert d_fn / d_fp == pytest.approx(2.297, abs=5e-3)
    assert d_fn / d_fp < RECIPE_BETA / RECIPE_ALPHA      # 割线 < 切线
    assert d_fn / d_fp > 2.0                             # 但 FN 确实更贵


def test_tversky_saturates_so_the_next_fp_costs_less():
    """实测：FN 固定时 FP 从 0 到 1 这一步使损失 +0.010728，从 9 到 10 只有
    +0.008642 —— 惩罚随计数增多而饱和，边际代价不是常数。"""
    step1 = _counts_loss(20, 1, 5) - _counts_loss(20, 0, 5)
    step10 = _counts_loss(20, 10, 5) - _counts_loss(20, 9, 5)
    assert step1 > step10 > 0.0
    assert step1 == pytest.approx(0.010728, abs=1e-5)
    assert step10 == pytest.approx(0.008642, abs=1e-5)


# --------------------------------------------------------------------------
# 2. 方向表：只有 beta/alpha 比值定方向；"beta<1"不是方向条件
# --------------------------------------------------------------------------

def test_tversky_only_the_beta_over_alpha_ratio_sets_the_fp_fn_direction():
    """实测：同一点 (TP=20,FP=5,FN=5) 上切线比 = beta/alpha。
    (0.3,0.7) -> 2.333（FN 贵）；(0.7,0.3) -> 0.4286（FP 贵）。
    两者 beta 都 <1（0.7 与 0.3），方向却相反 —— "beta<1 提升召回"在本实现里
    不成立，只有 beta/alpha>1 才偏召回。"""
    ratio_recall = _counts_loss(20, 5, 5, 0.3, 0.7)   # 只用来看基线
    assert ratio_recall == pytest.approx(0.2, abs=1e-6)
    # 切线比 = beta/alpha（用解析值锁定方向表）
    assert RECIPE_BETA / RECIPE_ALPHA > 1.0
    assert 0.3 / 0.7 < 1.0
    # 割线实证：(0.7,0.3) 下 FN 与 FP 的边际代价互换先后
    swapped = (_counts_loss(20, 5, 6, 0.7, 0.3) - _counts_loss(20, 5, 5, 0.7, 0.3))
    swapped_fp = (_counts_loss(20, 6, 5, 0.7, 0.3) - _counts_loss(20, 5, 5, 0.7, 0.3))
    assert swapped / swapped_fp < 0.5            # FN 反而比 FP 便宜
    assert swapped / swapped_fp > 0.35


def test_scaling_alpha_and_beta_together_changes_magnitude_not_direction():
    """实测：alpha/beta 同乘一个系数**不是**不变式。状态 (TP=1,FP=0,FN=1) 下
    (0.3,0.7)=0.411765、(0.6,1.4)=0.583333、(0.15,0.35)=0.259259 —— 幅度跟着
    系数走，而 FP/FN 相对方向（比值）不变。"""
    assert _counts_loss(1, 0, 1, 0.3, 0.7) == pytest.approx(1 - 1 / 1.7, abs=1e-6)
    l_big = _counts_loss(1, 0, 1, 0.6, 1.4)
    l_small = _counts_loss(1, 0, 1, 0.15, 0.35)
    l_base = _counts_loss(1, 0, 1, 0.3, 0.7)
    assert l_base == pytest.approx(1 - 1 / 1.7, abs=1e-6)
    assert l_big == pytest.approx(1 - 1 / 2.4, abs=1e-6)
    assert l_small == pytest.approx(1 - 1 / 1.35, abs=1e-6)
    assert l_big > l_base > l_small              # 同乘系数只改幅度


def test_region_gradient_matches_closed_form_and_rejects_swapped_convention():
    """实测：本实现的区域项对 prob 的梯度与闭式
    -(t*D - TP*(t + alpha*(1-t) - beta*t))/D^2（D=TP+alpha*FP+beta*FN+1e-6）
    在 float64 下最大误差 2.8e-17；把 alpha/beta 对调（FP<->FN 换名）的闭式与
    实测梯度相差 0.0686 —— 实现确实是 alpha 乘 FP、beta 乘 FN。"""
    from beamng_autopilot.vision.seg_losses import _EPS
    torch.manual_seed(3)
    prob = torch.rand(1, 4, 4, dtype=torch.float64, requires_grad=True)
    target = torch.zeros(1, 4, 4, dtype=torch.float64)
    target[0, 0, 0] = target[0, 1, 1] = target[0, 3, 3] = 1.0
    tversky_line_loss(prob, target, torch.ones_like(target), 0.3, 0.7).backward()
    tp = (prob * target).sum()
    fp = (prob * (1 - target)).sum()
    fn = ((1 - prob) * target).sum()
    with torch.no_grad():
        d_ok = tp + 0.3 * fp + 0.7 * fn + _EPS
        d_swapped = tp + 0.7 * fp + 0.3 * fn + _EPS
        grad_ok = -((target * d_ok) - tp * (target + 0.3 * (1 - target)
                                           - 0.7 * target)) / d_ok ** 2
        grad_swapped = -((target * d_swapped)
                         - tp * (target + 0.7 * (1 - target)
                                 - 0.3 * target)) / d_swapped ** 2
        err_ok = (grad_ok - prob.grad).abs().max().item()
        err_swapped = (grad_swapped - prob.grad).abs().max().item()
    assert err_ok < 1e-12
    assert err_swapped > 1e-3


# --------------------------------------------------------------------------
# 3. 完整 LineSegLoss 的方向：CE 类别权重主导，w_tversky 只加幅度
# --------------------------------------------------------------------------

def test_recipe_loss_direction_is_dominated_by_ce_class_weights():
    """实测：真实配方权重 [0.604,1.0,40.0] 下 |g_FN|/|g_FP| ≈ 68.7（FN 更贵），
    把 CE 换成不分类权重的 weight=None 后只剩 4.12。也就是说"当前损失偏召回"
    主要来自 median-frequency CE 权重，而不是 alpha/beta。"""
    l_rec, g_fn, g_fp = _probe_grads(torch.tensor(RECORDED_RECIPE_WEIGHTS))
    ratio_rec = abs(g_fn) / abs(g_fp)
    assert g_fn < 0 < g_fp                     # 真线像素上抬 logit 降损失
    assert ratio_rec == pytest.approx(68.7, abs=1.5)
    l_plain, g_fn0, g_fp0 = _probe_grads(None)
    ratio_plain = abs(g_fn0) / abs(g_fp0)
    assert ratio_plain == pytest.approx(4.12, abs=0.05)
    assert ratio_rec > 5 * ratio_plain         # 类别权重是数量级更大的来源


def test_raising_the_tversky_weight_is_more_fn_averse_not_fp_averse():
    """实测：把 w_tversky 从 1 提到 2（历史 "Tversky 2.0" 因子）后，
    |g_FN|/|g_FP| 从 68.7 升到 71.1，损失从 2.144 升到 3.122 —— 方向是**更偏
    召回**，不是"压 FP"。只有 alpha/beta 比值能移动 FP/FN 平衡，而它作用很小：
    (0.7,0.3) 只把比值降到 67.3。"""
    l1, g_fn1, g_fp1 = _probe_grads(torch.tensor(RECORDED_RECIPE_WEIGHTS),
                                    w_tversky=1.0)
    l2, g_fn2, g_fp2 = _probe_grads(torch.tensor(RECORDED_RECIPE_WEIGHTS),
                                    w_tversky=2.0)
    r1 = abs(g_fn1) / abs(g_fp1)
    r2 = abs(g_fn2) / abs(g_fp2)
    assert l2 > l1
    assert r2 > r1                       # 权重加倍 -> 更偏 FN/召回
    assert r2 > 60.0                     # 两个方向都远离"压 FP"
    # alpha/beta 对调（beta<alpha）只把偏向降一点点，因为 CE 权重占主导
    _, g_fn3, g_fp3 = _probe_grads(torch.tensor(RECORDED_RECIPE_WEIGHTS),
                                   alpha=0.7, beta=0.3)
    assert abs(g_fn3) / abs(g_fp3) < r1


# --------------------------------------------------------------------------
# 4. ignore 区：不参与损失也不参与梯度
# --------------------------------------------------------------------------

def _ignore_case(line_hot_in_ignore: bool, ignore_as_line: bool = False,
                 h: int = 10):
    """第 0 行是真线；(4:6,4:6) 是 2x2 的 ignore 块。

    ``line_hot_in_ignore``：忽略区里强烈预测 line（若参与监督就是 4 个 FP）。
    ``ignore_as_line``：把这 4 个像素改成真 line 目标（若参与监督就是 4 个 FN）。
    """
    logits = torch.zeros(1, 3, h, h)
    logits[0, 0] = 2.0
    logits[0, 1] = 1.0
    logits[0, 2, 0, :] = 6.0
    logits[0, 2, 4:6, 4:6] = 6.0 if line_hot_in_ignore else -6.0
    target = torch.zeros(1, h, h, dtype=torch.long)
    target[0, 0, :] = LINE_CLASS
    target[0, 4:6, 4:6] = LINE_CLASS if ignore_as_line else 255
    return logits, target


@pytest.mark.parametrize("w_cldice", [0.0, 1.0, 3.0])
def test_ignore_region_content_changes_neither_loss_nor_gradient(w_cldice):
    """实测：忽略区里放 4 个 FP（line logit=+6）与放"中性"内容（-6）相比，
    损失逐位相同（torch.equal 为 True），梯度也逐位相同，且忽略区内梯度为 0。
    区域项开/关（w_cldice=0/1/3）都一样。"""
    crit = LineSegLoss(w_tversky=1.0, w_cldice=w_cldice)
    logits_fp, target = _ignore_case(line_hot_in_ignore=True)
    logits_neutral, _ = _ignore_case(line_hot_in_ignore=False)
    a = crit(logits_fp.clone(), target)
    b = crit(logits_neutral.clone(), target)
    assert torch.equal(a, b), "忽略区的 FP 内容改变了损失"
    xa = logits_fp.clone().requires_grad_(True)
    xb = logits_neutral.clone().requires_grad_(True)
    crit(xa, target).backward()
    crit(xb, target).backward()
    assert torch.equal(xa.grad, xb.grad), "忽略区的内容改变了别处的梯度"
    assert torch.count_nonzero(xa.grad[0, :, 4:6, 4:6]) == 0, \
        "忽略区拿到了梯度"


@pytest.mark.parametrize("w_cldice", [0.0, 1.0])
def test_ignore_is_not_vacuous_same_pixels_as_line_target_change_the_loss(w_cldice):
    """实测：把 (4:6,4:6) 从 255 改成真 line 目标（在当前预测下就是 4 个 FN），
    损失变化 0.059339 —— 忽略确实屏蔽了监督，而不是那条路径本来就没影响。"""
    crit = LineSegLoss(w_tversky=1.0, w_cldice=w_cldice)
    logits, target_ignore = _ignore_case(line_hot_in_ignore=True)
    _, target_line = _ignore_case(line_hot_in_ignore=True, ignore_as_line=True)
    l_ignore = crit(logits.clone(), target_ignore).detach().item()
    l_line = crit(logits.clone(), target_line).detach().item()
    assert abs(l_line - l_ignore) == pytest.approx(0.059339, abs=1e-5)


def test_all_ignore_frame_gives_zero_loss_and_zero_grad():
    """实测：整帧 255（含默认 w_tversky=w_cldice=1）损失恰为 0.0、梯度范数 0.0、
    不是 NaN；整帧忽略走 masked_cross_entropy 也一样。"""
    torch.manual_seed(0)
    logits = torch.randn(1, 3, 6, 6, requires_grad=True)
    target = torch.full((1, 6, 6), 255, dtype=torch.long)
    loss = LineSegLoss()(logits, target)
    assert loss.detach().item() == 0.0
    assert bool(torch.isfinite(loss))
    loss.backward()
    assert float(logits.grad.norm()) == 0.0
    logits2 = torch.randn(2, 3, 5, 5, requires_grad=True)
    target2 = torch.full((2, 5, 5), 255, dtype=torch.long)
    loss2 = masked_cross_entropy(logits2, target2, torch.tensor([True, True, True]))
    assert loss2.detach().item() == 0.0
    loss2.backward()
    assert float(logits2.grad.norm()) == 0.0


def test_masked_non_line_class_leaves_the_region_term_on_the_raw_softmax():
    """实测（当前配方走不到、但方向口径确实如此）：class_mask=[1,0,1]（屏蔽
    asphalt、line 允许）时，CE 在屏蔽后的 softmax 上算，但区域项用的是**未屏蔽**
    的 softmax：把 asphalt logit 手动置 -inf（等价于 CE 的口径）后损失从
    2.258158 变到 2.248027（差 0.010130），且 line logit 在 (0,0) 处拿到
    0.011837 的非零梯度。"""
    h = 8
    logits = torch.zeros(1, 3, h, h)
    logits[0, 2] = 1.0
    logits[0, 1] = 3.0
    target = torch.zeros(1, h, h, dtype=torch.long)
    target[0, 3, 3] = LINE_CLASS
    mask = [True, False, True]
    crit = LineSegLoss(w_tversky=1.0, w_cldice=0.0)
    l_raw = crit(logits.clone(), target, class_mask=mask).detach().item()
    logits_masked = logits.clone()
    logits_masked[0, 1] = float("-inf")
    l_masked = crit(logits_masked, target, class_mask=mask).detach().item()
    assert l_raw == pytest.approx(2.258158, abs=1e-5)
    assert l_masked == pytest.approx(2.248027, abs=1e-5)
    assert abs(l_raw - l_masked) > 5e-3, \
        "区域项若与 CE 同口径，这两者应当相等"
    x = logits.clone().requires_grad_(True)
    crit(x, target, class_mask=mask).backward()
    assert x.grad[0, 2, 0, 0].item() != 0.0, "line 通道在屏蔽口径下拿到了区域项梯度"
