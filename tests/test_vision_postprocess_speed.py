"""后处理的两个连通域循环：判据必须等价，耗时必须与"碎块数"脱钩。

背景（2026-09-25 实测）：推理 p95 在不同 seed 之间差 **8.7 倍**（12.5–121.7 ms），
静默复测确认不是负载污染。逐 stage 拆开发现全在**后处理**，而且在
`constrain_line_to_road` 里：它对每个连通域做 `labels == i`——**每块扫一遍全帧**
（536×403）。漆线掩码碎掉的帧上块数上千 → 这个 stage 从 ~2 ms 涨到 **59 ms/帧**，
整条链的 p95 就是它撑起来的（驾驶 deadline 看 p95）。

修法：判据一个字没改，改成一次直方图 + LUT（`np.bincount` / `stats` 包围盒）。
本文件钉两件事：

1. **逐位等价**：用"每块扫全帧"的参考实现与新的向量化实现对比，随机碎块数
   从几十到上千，输出必须完全一致（这块是感知链里的判据，改一个像素都可能
   让历史结论失效）；
2. **耗时与碎块数脱钩**：1100+ 块的掩码必须在几十毫秒内处理完
   （旧实现 ~180 ms；这里给 100 ms 的宽松上限，避免 CI 抖动误报）。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beamng_autopilot.vision.segmentation import (  # noqa: E402
    _LINE_MIN_AREA_PX, constrain_line_to_road, fill_interior_holes,
    filter_line_shape,
)

H, W = 403, 536


def _ref_constrain(line, road, *, ksize=7, keep_frac=0.5,
                   elongated_frac=0.25):
    """参考实现：**每块扫全帧**（改之前的写法），用来做逐位等价校验。"""
    m = np.asarray(line, bool)
    if not m.any():
        return m
    road_b = np.asarray(road, bool)
    if float(road_b.mean()) < 0.005:
        return m
    rd = fill_interior_holes(road_b).astype(np.uint8)
    rd = cv2.dilate(rd, cv2.getStructuringElement(
        cv2.MORPH_RECT, (ksize, ksize))).astype(bool)
    rows_known = rd.any(axis=1)
    n, labels, _s, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    keep = np.zeros_like(m)
    for i in range(1, n):
        comp = labels == i
        judged = comp & rows_known[:, None]
        nj = int(judged.sum())
        if nj <= 0:
            keep |= comp
            continue
        inside = float(np.count_nonzero(comp & rd)) / float(nj)
        if inside >= float(keep_frac):
            keep |= comp
            continue
        if elongated_frac is None:
            continue
        if inside < float(elongated_frac):
            continue
        ys, xs = np.nonzero(comp)
        if len(ys) == 0:
            continue
        ls = max(int(xs.max() - xs.min()) + 1, int(ys.max() - ys.min()) + 1)
        ss = min(int(xs.max() - xs.min()) + 1, int(ys.max() - ys.min()) + 1)
        if ls >= 20 and ss >= 1 and ls >= 2.5 * ss:
            keep |= comp
    return keep


def _ref_filter(line):
    line = np.asarray(line, bool)
    if not line.any():
        return line
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        line.astype(np.uint8), 8)
    keep = np.zeros_like(line)
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        ls, ss = max(cw, ch), min(cw, ch)
        if area >= _LINE_MIN_AREA_PX or (ls >= 20 and ss >= 2
                                         and ls >= 2.5 * ss):
            keep[labels == i] = True
    return keep


def _fragmented_mask(rng, n_specks: int, *, with_stroke=True):
    m = np.zeros((H, W), bool)
    for _ in range(n_specks):
        y, x = rng.integers(0, H - 3), rng.integers(0, W - 3)
        m[y:y + 3, x:x + 3] = True
    if with_stroke:
        m[200:260, 250:252] = True          # 一条细长线（要被保留）
    return m


def _road_mask():
    road = np.zeros((H, W), bool)
    road[100:400, 60:480] = True
    return road


def test_constrain_is_bit_identical_to_the_reference():
    rng = np.random.default_rng(3)
    road = _road_mask()
    for n_specks in (0, 20, 200, 600, 1100):
        line = _fragmented_mask(rng, n_specks)
        new = constrain_line_to_road(line, road)
        old = _ref_constrain(line, road)
        assert np.array_equal(new, old), f"碎块 {n_specks} 个时输出不一致"
        # 第三档关掉时也要一致（配置项不能被向量化改坏）
        new_off = constrain_line_to_road(line, road, elongated_frac=None)
        old_off = _ref_constrain(line, road, elongated_frac=None)
        assert np.array_equal(new_off, old_off), "elongated_frac=None 时不一致"
        # 0.0 不是"关掉"，是"这一档全放行"（计划 T01 的语义）
        assert np.array_equal(constrain_line_to_road(line, road,
                                                     elongated_frac=0.0),
                              _ref_constrain(line, road, elongated_frac=0.0))


def test_filter_is_bit_identical_to_the_reference():
    rng = np.random.default_rng(11)
    for n_specks in (0, 30, 400, 1200):
        line = _fragmented_mask(rng, n_specks)
        assert np.array_equal(filter_line_shape(line), _ref_filter(line)), \
            f"碎块 {n_specks} 个时输出不一致"
    assert np.array_equal(filter_line_shape(np.zeros((H, W), bool)),
                          np.zeros((H, W), bool))


def test_a_fragmented_mask_does_not_blow_up():
    """1100+ 块的掩码：新实现几毫秒，旧实现约 180 ms（见报告 §12）。"""
    rng = np.random.default_rng(3)
    line = _fragmented_mask(rng, 1100)
    road = _road_mask()
    n_comp = cv2.connectedComponentsWithStats(line.astype(np.uint8), 8)[0] - 1
    assert n_comp > 900, n_comp
    constrain_line_to_road(line, road)            # 预热
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        constrain_line_to_road(line, road)
        times.append((time.perf_counter() - t0) * 1000)
    best = min(times)
    assert best < 100.0, (
        f"碎块 {n_comp} 个时 constrain_line_to_road 花了 {best:.1f} ms："
        f"判据里又出现了'每块扫全帧'（旧写法约 180 ms）")
    t0 = time.perf_counter()
    filter_line_shape(line)
    assert (time.perf_counter() - t0) * 1000 < 100.0


def test_the_elongated_stroke_is_still_kept():
    """向量化不能把"细长笔画要保留"这条判据弄丢。"""
    road = _road_mask()
    line = np.zeros((H, W), bool)
    line[10:70, 250:252] = True          # 完全在路面外（y<100 的已知行之外）
    kept = constrain_line_to_road(line, road)
    assert kept.sum() > 0, "路面掩码以外的细长笔画应按第二档保留"


def test_segmenter_passes_the_thresholds_through(tmp_path):
    """离路约束的阈值可配：None = 模块默认（旧行为不变）。

    为什么要可配：硬门里 `offroad_false_ratio` / `line_precision`
    都由这一步决定，配上才能用**同一批 checkpoint** 扫阈值（不必重训）。
    实测结论（report.md §19）：跨 seed 方向不一致，所以默认值不动。
    """
    import numpy as np
    import torch

    from beamng_autopilot.vision.segmentation import (
        LINE_ROAD_ELONGATED_FRAC, LINE_ROAD_KEEP_FRAC, Segmenter, SegUNet,
    )

    ck = tmp_path / "m.pt"
    torch.save({"state_dict": SegUNet().state_dict(),
                "train_args": {"epochs": 1}, "n_classes": 3}, ck)
    seg = Segmenter(model_path=str(ck), device="cpu", use_half=False)
    assert seg.line_road_keep_frac is None, "默认不改行为"
    seg2 = Segmenter(model_path=str(ck), device="cpu", use_half=False,
                     line_road_keep_frac=0.8, line_road_elongated_frac=None,
                     line_road_ksize=9)
    assert seg2.line_road_keep_frac == 0.8
    assert seg2.line_road_elongated_frac is None
    assert seg2.line_road_ksize == 9
    assert (LINE_ROAD_KEEP_FRAC, LINE_ROAD_ELONGATED_FRAC) == (0.5, 0.25), (
        "模块默认值不能被调参惄改掉")
    frame = np.zeros((64, 48, 3), np.uint8)
    road_a, line_a = seg.predict(frame)
    road_b, line_b = seg2.predict(frame)
    assert road_a.shape == road_b.shape == (64, 48)
