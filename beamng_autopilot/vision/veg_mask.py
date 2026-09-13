"""绿色植被掩码： HSV 阈值，用于把 US 地图上被误判为路面的草地
从 road 伪标签里剔除（road 过膨胀修正的负向约束）。

保守口径：只抓强绿色的草本/树冠表面（H 35-85、中高饱和）；
阴影里的暗绿和灰绿路面不在范围内。仅作为 road 的扣除项使用，
不作为独立类别。
"""

from __future__ import annotations

import cv2
import numpy as np

H_LO, H_HI = 35, 85
S_LO = 60
V_LO = 40


def green_vegetation_mask(rgb: np.ndarray) -> np.ndarray:
    """RGB 帧 -> bool 掩码（强绿色植被区域）。"""
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)
    m = cv2.inRange(hsv, (H_LO, S_LO, V_LO), (H_HI, 255, 255))
    return m.astype(bool)
