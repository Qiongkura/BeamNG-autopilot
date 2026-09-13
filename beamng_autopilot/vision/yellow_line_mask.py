"""黄色漆线掩码：HSV 阈值 + 形态学清理.

italy 标线全为白色，US 地图（east/west coast）大量使用黄色中线——
分割模型没学过黄漆，跨地图漏检集中于此。用颜色先验自动生成候选
掩码，与模型白线掩码取并集即为伪标签的 line 通道。
"""

from __future__ import annotations

import cv2
import numpy as np

# OpenCV HSV（H 0-180）：黄漆 H≈20-35、高饱和、中高亮度；路面/树影
# 的黄色偏灰（低饱和）被 S 下限挡掉
H_LO, H_HI = 18, 38
S_LO = 90
V_LO = 110
# 只在画面下部 2/3 找（路面上方是树/天，误源密集）
ROI_TOP_FRAC = 0.30
_MIN_AREA = 40


def yellow_line_mask(rgb: np.ndarray) -> np.ndarray:
    """RGB 帧 -> bool 掩码（黄色漆线候选，已清理小噪块）。"""
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)
    h, w = hsv.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    roi_top = int(h * ROI_TOP_FRAC)
    region = hsv[roi_top:, :]
    cand = cv2.inRange(region,
                       (H_LO, S_LO, V_LO), (H_HI, 255, 255)).astype(bool)
    # 形态学：开运算去孤立点，闭运算连虚线段
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9))
    cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_OPEN, k)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT,
                                                      (5, 15)))
    # 连通域面积过滤
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cand, 8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= _MIN_AREA:
            mask[roi_top:, :][lab == i] = 1
    return mask.astype(bool)
