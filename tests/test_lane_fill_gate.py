# -*- coding: utf-8 -*-
"""斜向细线的 bbox 填充率修正（too_sparse 门）。

2026-11-05 GridMap 运动后右线整条被 `too_sparse` 丢弃：斜线在轴对齐
bbox 里填充率随透视变化从 ~10% 掉到 9.4%。线状组件在最小外接旋转矩形
里是致密的，真正的散点/空心团块仍然稀疏。
"""
import cv2
import numpy as np

from beamng_autopilot.vision.lanes import _component_fill_ok


def _components(mask):
    return cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)


def test_diagonal_stroke_is_not_sparse():
    mask = np.zeros((200, 200), np.uint8)
    cv2.line(mask, (10, 190), (190, 10), 1, 4)
    n, labels, stats, _ = _components(mask)
    assert n == 2
    x, y, w, h, area = stats[1]
    assert area / (w * h) < 0.10          # bbox 口径确实稀疏
    assert _component_fill_ok(labels, 1, x, y, w, h, area, roi=True)
    assert _component_fill_ok(labels, 1, x, y, w, h, area, roi=False)


def test_hollow_square_still_rejected():
    mask = np.zeros((200, 200), np.uint8)
    cv2.rectangle(mask, (50, 50), (149, 149), 1, 1)
    n, labels, stats, _ = _components(mask)
    x, y, w, h, area = stats[1]
    assert not _component_fill_ok(labels, 1, x, y, w, h, area, roi=True)


def test_solid_block_accepted():
    mask = np.zeros((200, 200), np.uint8)
    cv2.rectangle(mask, (50, 50), (99, 99), 1, -1)
    n, labels, stats, _ = _components(mask)
    x, y, w, h, area = stats[1]
    assert _component_fill_ok(labels, 1, x, y, w, h, area, roi=True)
