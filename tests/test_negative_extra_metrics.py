"""负例 v2 解释量：最大连通域 + 控制区域假候选数，且 v1 旧累计不翻转。

方案 §7（T16）：负例帧率饱和到 1.0 时，靠**最大连通域**和**进入控制区域的假
候选数**继续解释模型行为；原始帧率/像素占比一个不隐去。计数契约升到 v2，但
``negative_line_summary`` 默认仍按 v1 读旧累计——否则 54 个旧判定会被新键
"缺键"重判成不完整（历史翻转）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beamng_autopilot.experiments import negative_scenes as ns  # noqa: E402

RANK = "verified"


def _frame_with_two_blocks() -> np.ndarray:
    """8x8 帧：2x2 小块（4 px）+ 3x3 大块（9 px），4-连通下互不相连。"""
    pred = np.zeros((8, 8), bool)
    pred[0:2, 0:2] = True          # 4 px
    pred[4:7, 4:7] = True          # 9 px
    return pred


def test_max_connected_component_is_the_biggest_block_not_the_sum():
    pred = _frame_with_two_blocks()
    c = ns.negative_line_counts(pred, np.zeros((8, 8), np.uint8),
                                label_rank=RANK)
    assert c["negative_line_false_positive_px"] == 13, c
    assert c["negative_line_false_positive_max_cc_px"] == 9, c
    # 干净帧与不合格帧都没有连通域读数（0 是"测到 0"，不是"没测"）
    clean = ns.negative_line_counts(np.zeros((8, 8), bool),
                                    np.zeros((8, 8), np.uint8),
                                    label_rank=RANK)
    assert clean["negative_line_false_positive_max_cc_px"] == 0, clean
    unverified = ns.negative_line_counts(pred, np.zeros((8, 8), np.uint8),
                                         label_rank="agent")
    assert unverified["negative_line_false_positive_max_cc_px"] == 0, unverified
    assert unverified["negative_line_unverified_frames"] == 1, unverified


def test_control_region_counts_components_inside_the_region():
    pred = _frame_with_two_blocks()
    label = np.zeros((8, 8), np.uint8)
    left = np.zeros((8, 8), bool)
    left[:, :3] = True                     # 只罩住 2x2 小块
    c = ns.negative_line_counts(pred, label, label_rank=RANK,
                                control_region=left)
    assert c["negative_line_control_region_false_candidates"] == 1, c
    assert c["negative_line_false_positive_control_region_frames"] == 1, c
    # 全帧区域：两个不相连的块 -> 2
    whole = np.ones((8, 8), bool)
    c2 = ns.negative_line_counts(pred, label, label_rank=RANK,
                                 control_region=whole)
    assert c2["negative_line_control_region_false_candidates"] == 2, c2
    # 区域里没有预测 -> 0；None -> 0 且不参与资格（伴随标记也是 0）
    right = np.zeros((8, 8), bool)
    right[:, 5:] = True
    none_in_region = ns.negative_line_counts(
        np.zeros((8, 8), bool), label, label_rank=RANK, control_region=right)
    assert none_in_region["negative_line_control_region_false_candidates"] == 0
    no_region = ns.negative_line_counts(pred, label, label_rank=RANK)
    assert no_region["negative_line_control_region_false_candidates"] == 0
    assert no_region["negative_line_false_positive_control_region_frames"] == 0


def test_control_region_shape_mismatch_fails_loudly():
    with pytest.raises(ValueError, match="control_region"):
        ns.negative_line_counts(np.zeros((4, 4), bool),
                                np.zeros((4, 4), np.uint8), label_rank=RANK,
                                control_region=np.zeros((2, 2), bool))


def _old_v1_acc(frames) -> dict:
    """模拟 v1 旧累加器：只保留 v1 键（新键整批缺席）。"""
    acc: dict = {}
    for pred, label in frames:
        for k, v in ns.negative_line_counts(pred, label,
                                            label_rank=RANK).items():
            if any(k == ns.PREFIX + old for old in ns.COUNTERS_V1):
                acc[k] = acc.get(k, 0) + v
    return acc


def test_v1_old_accumulator_stays_complete_with_extra_counters_absent():
    frames = [(_frame_with_two_blocks(), np.zeros((8, 8), np.uint8)),
              (np.zeros((8, 8), bool), np.zeros((8, 8), np.uint8))]
    acc = _old_v1_acc(frames)
    out = ns.negative_line_summary(acc, n_frames=2)      # 默认 v1 语义
    assert out["status"] == "measured", out
    assert out["extra_counters"] == "absent", out
    assert out["counter_version"] == 1, out
    # 新键按 0 报，但 extra_counters 已说明"根本没测"（0 不等于干净）
    assert out["false_positive_max_cc_px"] == 0, out
    assert out["false_positive_max_cc_px_max"] == 0, out
    assert out["false_positive_control_region_frames"] == 0, out
    # 原始帧率/像素占比不受影响
    assert out["false_positive_frame_rate"] == 0.5, out
    assert out["false_positive_pixel_fraction"] == 13 / 128, out


def test_v2_missing_new_keys_is_incomplete():
    frames = [(_frame_with_two_blocks(), np.zeros((8, 8), np.uint8))]
    acc = _old_v1_acc(frames)
    out = ns.negative_line_summary(acc, n_frames=1, counter_version=2)
    assert out["status"] == "missing_counters", out
    assert out["false_positive_frame_rate"] is None, out
    for key in ns.COUNTERS_V2_NEW:
        assert key in out["missing_reason"], (key, out["missing_reason"])
    # 不支持的版本要显式报错，不静默按旧版读
    with pytest.raises(ValueError, match="counter_version"):
        ns.negative_line_summary(acc, n_frames=1, counter_version=3)


def test_v2_merge_counts_reports_frame_max_and_control_region_frames():
    pred_big = _frame_with_two_blocks()                      # max cc = 9
    small = np.zeros((8, 8), bool); small[0:2, 0:2] = True   # max cc = 4
    region = np.zeros((8, 8), bool); region[:, :3] = True
    label = np.zeros((8, 8), np.uint8)
    acc: dict = {}
    ns.merge_counts(acc, ns.negative_line_counts(pred_big, label,
                                                 label_rank=RANK,
                                                 control_region=region))
    ns.merge_counts(acc, ns.negative_line_counts(small, label, label_rank=RANK))
    out = ns.negative_line_summary(acc, n_frames=2, counter_version=2)
    assert out["status"] == "measured", out
    assert out["extra_counters"] == "present", out
    # 帧间最大 = max(9, 4) = 9；普通求和会得到 13（这就是 merge_counts 的理由）
    assert out["false_positive_max_cc_px_max"] == 9, out
    assert out["false_positive_control_region_frames"] == 1, out
    assert out["control_region_false_candidates"] == 1, out
    # 原始字段照旧：帧率 2/2（两帧都有假线）、像素占比 (13+4)/128
    assert out["false_positive_frame_rate"] == 1.0, out
    assert out["false_positive_pixel_fraction"] == 17 / 128, out


def test_max_counters_are_max_merged_not_summed():
    acc: dict = {}
    ns.merge_counts(acc, {ns.PREFIX + "false_positive_max_cc_px": 9,
                          ns.PREFIX + "false_positive_px": 13})
    ns.merge_counts(acc, {ns.PREFIX + "false_positive_max_cc_px": 4,
                          ns.PREFIX + "false_positive_px": 4})
    assert acc[ns.PREFIX + "false_positive_max_cc_px"] == 9, acc
    assert acc[ns.PREFIX + "false_positive_px"] == 17, acc


def test_original_rate_and_pixel_fields_are_additive_only():
    """加新键不能改动任何旧字段的取值（判定/看板按旧键读）。"""
    frames = [(_frame_with_two_blocks(), np.zeros((8, 8), np.uint8)),
              (np.zeros((8, 8), bool), np.zeros((8, 8), np.uint8))]
    old_acc = _old_v1_acc(frames)
    new_acc: dict = {}
    for pred, label in frames:
        ns.merge_counts(new_acc, ns.negative_line_counts(
            pred, label, label_rank=RANK, control_region=np.ones((8, 8), bool)))
    old = ns.negative_line_summary(old_acc, n_frames=2)
    new = ns.negative_line_summary(new_acc, n_frames=2)
    for key in ("frames", "eligible_frames", "clean_frames",
                "false_positive_frames", "false_positive_px", "eligible_px",
                "positive_frames", "unknown_frames", "empty_frames",
                "unverified_frames", "unverified_pred_line_px", "status",
                "false_positive_frame_rate", "false_positive_pixel_fraction",
                "excluded_frames"):
        assert new[key] == old[key], (key, new[key], old[key])
    assert ns.COUNTER_VERSION == 2
    assert ns.COUNTERS[:len(ns.COUNTERS_V1)] == ns.COUNTERS_V1
    assert set(ns.COUNTERS_V2_NEW) <= set(ns.COUNTERS)
