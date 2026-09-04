"""分割训练数据工具（load_frames 密度过滤 / per-run 验证划分）纯逻辑回归。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


_TRAIN_PATH = (Path(__file__).resolve().parent.parent /
               "scripts" / "m5_train_seg.py")
_spec = importlib.util.spec_from_file_location("m5_train_seg", _TRAIN_PATH)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)


def _write_frame(run_dir: Path, name: str, line_px: int) -> None:
    """写一张 8x8 合成帧：line 像素数可指定，其余为路面。"""
    colour = np.full((8, 8, 3), 90, dtype=np.uint8)
    label = np.ones((8, 8), dtype=np.uint8)
    label.ravel()[:line_px] = 2
    np.savez(run_dir / name, colour=colour, label=label)


def test_load_frames_filters_sparse(tmp_path):
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    run_a.mkdir()
    run_b.mkdir()
    for i in range(3):
        _write_frame(run_a, f"frame_{i:05d}.npz", line_px=8)   # 8/64
    for i in range(2):
        _write_frame(run_b, f"frame_{i:05d}.npz", line_px=1)   # 1/64

    frames, per_run = _mod.load_frames([run_a, run_b], min_line_frac=0.05)
    assert len(frames) == 3                       # 稀疏 run_b 全部被过滤
    assert per_run["run_a"]["kept"] == 3
    assert per_run["run_b"]["kept"] == 0
    assert per_run["run_b"]["line_px_frac"] == pytest.approx(1 / 64, abs=1e-6)


def test_split_frames_per_run_keeps_each_tail():
    frames = [(np.zeros((4, 4, 3), np.uint8), np.zeros((4, 4), np.uint8))
              for _ in range(20)]
    per_run = {
        "a": {"kept": 10, "start": 0, "end": 10},
        "b": {"kept": 10, "start": 10, "end": 20},
    }
    tr, va = _mod.split_frames(frames, per_run, "per-run", 0.2)
    assert len(tr) == 16 and len(va) == 4
    # 每个 run 各取尾部 20%：a 的最后 2 帧与 b 的最后 2 帧进验证
    assert va == [frames[8], frames[9], frames[18], frames[19]]
    assert tr == frames[:8] + frames[10:18]


def test_split_frames_tail_global():
    frames = [(np.zeros((4, 4, 3), np.uint8), np.zeros((4, 4), np.uint8))
              for _ in range(10)]
    tr, va = _mod.split_frames(frames, {}, "tail", 0.2)
    assert len(tr) == 8 and len(va) == 2
    assert va == frames[8:]


def test_iou_from_accum_no_inflation_for_absent_class():
    ious = _mod.iou_from_accum(
        np.array([10.0, 5.0, 0.0]), np.array([20.0, 10.0, 0.0]))
    assert ious[0] == pytest.approx(0.5)
    assert ious[1] == pytest.approx(0.5)
    assert ious[2] == 0.0          # 未出现类别：实数 0，而不是虚高 1.0


def test_balanced_indices_each_run_equal():
    rng = np.random.default_rng(7)
    idx = _mod.balanced_indices([(0, 5), (5, 15)], rng)
    assert len(idx) == 20                       # 每 run 补齐到最长 10 帧
    assert sum(0 <= j < 5 for j in idx) == 10   # run a：5 帧循环补齐到 10
    assert sum(5 <= j < 15 for j in idx) == 10  # run b：自身 10 帧
    assert len(np.unique(idx)) == 15            # 全部来源帧都被覆盖


def test_balanced_indices_deterministic():
    a = _mod.balanced_indices([(0, 5), (5, 15)],
                              np.random.default_rng(3))
    b = _mod.balanced_indices([(0, 5), (5, 15)],
                              np.random.default_rng(3))
    assert np.array_equal(a, b)


def test_train_run_bounds_matches_per_run_split():
    frames = [(np.zeros((4, 4, 3), np.uint8), np.zeros((4, 4), np.uint8))
              for _ in range(20)]
    per_run = {"a": {"kept": 10, "start": 0, "end": 10},
               "b": {"kept": 10, "start": 10, "end": 20}}
    tr, _ = _mod.split_frames(frames, per_run, "per-run", 0.2)
    bounds = _mod.train_run_bounds(frames, per_run, "per-run", 0.2)
    assert bounds == [(0, 8), (8, 16)]
    assert len(tr) == 16


def test_train_run_bounds_tail_clips_last_run():
    frames = [(np.zeros((4, 4, 3), np.uint8), np.zeros((4, 4), np.uint8))
              for _ in range(20)]
    per_run = {"a": {"kept": 10, "start": 0, "end": 10},
               "b": {"kept": 10, "start": 10, "end": 20}}
    tr, va = _mod.split_frames(frames, per_run, "tail", 0.2)
    bounds = _mod.train_run_bounds(frames, per_run, "tail", 0.2)
    assert bounds == [(0, 10), (10, 16)]  # 全局尾部 4 帧进验证，b 被截断
    assert len(tr) == 16 and len(va) == 4
