"""Pure Pursuit 循迹控制器。"""

from __future__ import annotations

import numpy as np


def wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


class PurePursuit:
    def __init__(self, lookahead: float = 6.0, wheelbase: float = 2.9, search_window: int = 60):
        self.lookahead = lookahead
        self.wheelbase = wheelbase
        self.search_window = search_window

    def adaptive_lookahead(self, speed: float) -> float:
        """速度越快，预瞄越远（限制在安全区间）。"""
        return float(np.clip(self.lookahead + 0.35 * speed, 4.0, 14.0))

    def find_target(self, pos, path: np.ndarray, nearest_idx: int = 0):
        """在闭环轨迹上找预瞄点。返回 (目标点, 预瞄索引, 最近点索引)。"""
        p = np.asarray(pos[:2], dtype=float)
        pts = path[:, :2]
        n = len(pts)
        # Closed loop: wrap the search window around the path end.
        idxs = (np.arange(nearest_idx - 10, nearest_idx + self.search_window)) % n
        dists = np.linalg.norm(pts[idxs] - p, axis=1)
        nearest = int(idxs[np.argmin(dists)])

        # 从最近点向前累计弧长，找到距离 = lookahead 的点（线性插值）
        ahead = nearest
        traveled = 0.0
        for _ in range(n):
            nxt = (ahead + 1) % n
            seg_len = float(np.linalg.norm(pts[nxt] - pts[ahead]))
            if traveled + seg_len >= self.lookahead:
                t = (self.lookahead - traveled) / max(seg_len, 1e-6)
                target = pts[ahead] + t * (pts[nxt] - pts[ahead])
                return target, ahead, nearest
            traveled += seg_len
            ahead = nxt
        return pts[nearest], nearest, nearest

    def steering(self, pos, heading: float, path: np.ndarray, nearest_idx: int = 0):
        """计算转向角（弧度）。返回 (转向角, 目标点, 最近索引)。"""
        target, _, nearest = self.find_target(pos, path, nearest_idx)
        dx = target[0] - pos[0]
        dy = target[1] - pos[1]
        alpha = wrap_angle(np.arctan2(dy, dx) - heading)
        steer = np.arctan2(2.0 * self.wheelbase * np.sin(alpha), self.lookahead)
        return float(steer), target, nearest
