"""漆面外观判据（白/暖白/黄）：**掩码门**用，不作真值判据。

为什么单独成模块：同一组阈值现在有三个消费者——自动真值的外观证据
（`experiments.auto_truth.line_evidence_mask(mode="appearance")`）、导出器的
漆面一致门、以及掩码侧的**外观门**（`segmentation`，单因子开关）。三处各写一份
阈值迟早漂移，这里只留一份。

标定与边界（都实测过，别把结论用反）：

* 阈值来自 2026-09-27 开发集人工标签的线像素实测（亮无彩脊 / 黄漆）；
* **作真值判据不成立**：`mode="appearance"` 在**已验证的无线负例**上有
  13k–68k px/8 帧命中（亮铺装、亮碎石都命中"白/暖白"）——所以它是**门**，
  不是"哪里有线的证据"；
* 作掩码门是**单调**的：`line & paint_like` 只会删像素、不会加，所以精度与
  负例假阳性只可能改善，唯一代价是召回（实测 R3-b：P 0.138→0.557、
  R 0.807→0.769；dev：P 0.111→0.755、R 0.793→0.600，见
  `docs/T16_NEXT_FACTOR_FAR_OFFROAD_PROPOSAL_20261004.md` §9）。
"""

from __future__ import annotations

import numpy as np

#: 白漆下限（三通道最大值）
LINE_WHITE_MIN = 185
#: 白漆色度上限（max-min）
LINE_WHITE_CHROMA = 30
#: 暖白：r-b 至少这么大也算白漆（沥青上的旧漆偏暖）
LINE_WARM_WHITE_RB = 25
#: 黄漆：r、g 下限
LINE_YELLOW_RG_MIN = 135
#: 黄漆：b 上限
LINE_YELLOW_B_MAX = 130
#: 黄漆：r-b 下限
LINE_YELLOW_RB_MIN = 90
#: 黄漆：|r-g| 上限（排除橙色/棕色）
LINE_YELLOW_RG_MAX_DIFF = 45


def paint_like_mask(rgb) -> np.ndarray:
    """``HxWx3`` 图像 -> 漆面外观布尔掩码（白 / 暖白 / 黄）。

    只读 RGB；不做遮挡/深度判断（那是调用方的事）。形状不对时抛
    ``ValueError``——不静默返回全 False（那会把门变成"全删"）。
    """
    a = np.asarray(rgb)
    if a.ndim != 3 or a.shape[2] < 3 or a.size == 0:
        raise ValueError("rgb must be a nonempty HxWx3 image")
    win = a[:, :, :3].astype(np.int16)
    mx = win.max(axis=2)
    mn = win.min(axis=2)
    r, g, b = win[:, :, 0], win[:, :, 1], win[:, :, 2]
    white = ((mx >= LINE_WHITE_MIN)
             & (((mx - mn) <= LINE_WHITE_CHROMA)
                | ((r - b) >= LINE_WARM_WHITE_RB)))
    yellow = ((r >= LINE_YELLOW_RG_MIN) & (g >= LINE_YELLOW_RG_MIN)
              & (b <= LINE_YELLOW_B_MAX)
              & (np.abs(r - g) <= LINE_YELLOW_RG_MAX_DIFF)
              & ((r - b) >= LINE_YELLOW_RB_MIN))
    return white | yellow
