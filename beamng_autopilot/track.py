"""参考轨迹：解析闭环的生成、轨迹的录制存取与重采样。"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _spaced(p0, p1, step):
    dist = np.hypot(p1[0] - p0[0], p1[1] - p0[1])
    n = max(1, int(np.ceil(dist / step)))
    ts = np.linspace(0.0, 1.0, n, endpoint=False)
    return np.column_stack(
        [p0[0] + (p1[0] - p0[0]) * ts, p0[1] + (p1[1] - p0[1]) * ts]
    )


def _arc(cx, cy, radius, a0_deg, a1_deg, step):
    """顺时针圆弧（角度递减），间距约 step。"""
    sweep = (a0_deg - a1_deg) % 360.0
    n = max(2, int(np.ceil(np.deg2rad(sweep) * radius / step)))
    ang = np.linspace(np.deg2rad(a0_deg), np.deg2rad(a1_deg), n, endpoint=False)
    return np.column_stack([cx + radius * np.cos(ang), cy + radius * np.sin(ang)])


def generate_rounded_rectangle(
    half_extent=(90.0, 90.0),
    corner_radius: float = 25.0,
    step: float = 1.5,
    center=(0.0, 0.0),
) -> np.ndarray:
    """生成顺时针圆角矩形闭环轨迹，返回 Nx3（x, y, z=0），首尾相接。"""
    hx, hy = half_extent
    r = min(corner_radius, hx, hy)
    cx0, cy0 = center

    parts = []
    # 1) 右边直线：从 (hx, hy-r) 向下
    parts.append(_spaced((hx, hy - r), (hx, -hy + r), step))
    # 2) 右下圆角：中心 (hx-r, -hy+r)，0° -> -90°
    parts.append(_arc(hx - r, -hy + r, r, 0.0, -90.0, step))
    # 3) 底边直线：向左
    parts.append(_spaced((hx - r, -hy), (-hx + r, -hy), step))
    # 4) 左下圆角：中心 (-hx+r, -hy+r)，-90° -> 180°
    parts.append(_arc(-hx + r, -hy + r, r, -90.0, 180.0, step))
    # 5) 左边直线：向上
    parts.append(_spaced((-hx, -hy + r), (-hx, hy - r), step))
    # 6) 左上圆角：中心 (-hx+r, hy-r)，180° -> 90°
    parts.append(_arc(-hx + r, hy - r, r, 180.0, 90.0, step))
    # 7) 顶边直线：向右
    parts.append(_spaced((-hx + r, hy), (hx - r, hy), step))
    # 8) 右上圆角：中心 (hx-r, hy-r)，90° -> 0°
    parts.append(_arc(hx - r, hy - r, r, 90.0, 0.0, step))

    xy = np.vstack(parts)
    xy = np.vstack([xy, xy[0:1]])  # 首尾相接，形成闭环
    xy[:, 0] += cx0
    xy[:, 1] += cy0
    return np.column_stack([xy[:, 0], xy[:, 1], np.zeros(len(xy))])


def heading_from_path(points: np.ndarray) -> np.ndarray:
    """由轨迹点计算每点航向角（弧度），用相邻点差分。"""
    d = np.gradient(points[:, :2], axis=0)
    return np.arctan2(d[:, 1], d[:, 0])


def resample_uniform(points: np.ndarray, step: float = 1.0) -> np.ndarray:
    """把任意密度轨迹重采样为等间距（闭环，不重复终点）。"""
    pts = points[:, :2]
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    n = max(2, int(np.ceil(total / step)))
    ds = np.linspace(0.0, total, n, endpoint=False)
    out = np.empty((n, 3))
    for i, d in enumerate(ds):
        j = int(np.searchsorted(cum, d, side="right")) - 1
        j = min(max(j, 0), len(pts) - 2)
        t = (d - cum[j]) / max(seg[j], 1e-9)
        p = pts[j] + t * (pts[j + 1] - pts[j])
        out[i] = [p[0], p[1], points[0, 2]]
    return out


def clean_closed_lap(points: np.ndarray, step: float = 1.0) -> np.ndarray:
    """把（可能多圈的）记录路径裁成一条干净闭环：
    1) 去掉起步反转造成的航向急变抖动（30m 内跳变 >100 度的尾部剔除）
    2) 走过半圈以后，取离起点最近的位置截断成一圈
    3) 等间距重采样。
    """
    pts = np.asarray(points[:, :2], dtype=float)
    headings = heading_from_path(points)

    # 1) 起步反转抖动：前 30 个点内航向跳变 >100 度的最后位置
    start = 0
    lim = min(30, len(pts) - 2)
    for i in range(1, lim):
        d = (headings[i] - headings[i - 1]) % (2.0 * np.pi)
        if d > np.pi:
            d -= 2.0 * np.pi
        if abs(d) > np.deg2rad(100.0):
            start = i + 1
    if start > 0:
        start = min(start + 1, len(pts) - 2)

    sub = pts[start:]
    seg = np.linalg.norm(np.diff(sub, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]

    # 2) 走过一半以后再找离起点最近的位置（那正是一整圈回来的地方）
    cand = np.where(cum > 0.5 * total)[0]
    if len(cand) == 0:
        cand = np.arange(len(sub))
    j = int(cand[np.argmin(np.linalg.norm(sub[cand] - sub[0], axis=1))])
    loop = np.vstack([sub[: j + 1], sub[0]])

    # 3) 等间距重采样
    z = float(points[0, 2]) if points.shape[1] > 2 else 0.0
    loop3 = np.column_stack([loop, np.full(len(loop), z)])
    return resample_uniform(loop3, step=step)


def save_track(points: np.ndarray, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, points=np.asarray(points, dtype=float), headings=heading_from_path(points))
    print(f"[track] 已保存轨迹: {path} ({len(points)} 点)")


def load_track(path):
    data = np.load(path)
    return data["points"], data["headings"]
