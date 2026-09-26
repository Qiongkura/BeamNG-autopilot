"""无标线误报评价：空预测、假线和未知标注不能混为一类。"""

from __future__ import annotations

import numpy as np
import pytest

from beamng_autopilot.experiments.negative_scenes import (
    negative_line_counts, negative_line_summary,
)


#: 这些用例测的是**内容**语义（空预测/未知像素/有真线），所以显式声明 verified 档位；
#: 档位门槛本身（非 verified 不算合格负例）见文件末尾的 T10 反例。
RANK = "verified"


def summarize(*pairs):
    acc = {}
    for pred, label in pairs:
        for key, value in negative_line_counts(pred, label,
                                               label_rank=RANK).items():
            acc[key] = acc.get(key, 0) + value
    return negative_line_summary(acc, n_frames=len(pairs))


def test_correct_negative_is_measured_zero_false_positives():
    out = summarize((np.zeros((2, 3)), np.ones((2, 3))))
    assert out["status"] == "measured"
    assert out["eligible_frames"] == out["clean_frames"] == 1
    assert out["false_positive_frame_rate"] == 0
    assert out["false_positive_pixel_fraction"] == 0


def test_false_positive_rates_use_negative_frames_and_pixel_totals():
    # 一张 4 像素帧有 1 个假线，一张 12 像素帧干净；不是逐帧像素比平均。
    out = summarize((np.eye(2, dtype=bool) & [[1, 0], [0, 0]],
                     np.zeros((2, 2))),
                    (np.zeros((3, 4)), np.ones((3, 4))),
                    (np.ones((4, 4)), np.full((4, 4), 2)))
    assert out["frames"] == 3
    assert out["positive_frames"] == 1
    assert out["eligible_frames"] == 2
    assert out["false_positive_frames"] == out["clean_frames"] == 1
    assert out["false_positive_frame_rate"] == .5
    assert out["false_positive_pixel_fraction"] == 1 / 16


@pytest.mark.parametrize("label", [
    [[255, 255], [255, 255]], [[0, 1], [1, 255]], [[2, 1], [1, 255]],
])
def test_unknown_pixels_prevent_whole_frame_negative_claim(label):
    out = summarize((np.ones((2, 2)), np.array(label)))
    assert out["unknown_frames"] == 1
    assert out["eligible_frames"] == 0
    assert out["status"] == "no_eligible_frames"
    assert out["false_positive_frame_rate"] is None
    assert out["false_positive_pixel_fraction"] is None


def test_empty_image_is_not_a_successful_negative():
    out = summarize((np.zeros((0, 0)), np.zeros((0, 0))))
    assert out["empty_frames"] == 1
    assert out["status"] == "no_eligible_frames"


def test_old_and_partially_upgraded_accumulators_remain_unknown():
    old = negative_line_summary({"gt_line_px": 0}, n_frames=3)
    assert old["status"] == "missing_counters"
    assert old["eligible_frames"] is None
    acc = negative_line_counts(np.zeros((2, 2)), np.zeros((2, 2)),
                            label_rank=RANK)
    mixed = negative_line_summary(acc, n_frames=3)
    assert mixed["status"] == "missing_counters"
    assert mixed["false_positive_frame_rate"] is None


@pytest.mark.parametrize("pred,label", [
    (np.zeros((2, 2)), np.zeros((3, 2))),
    (np.zeros(2), np.zeros(2)),
    (np.zeros((2, 2)), np.full((2, 2), 3)),
])
def test_bad_inputs_fail_instead_of_manufacturing_clean_negatives(pred, label):
    with pytest.raises(ValueError):
        negative_line_counts(pred, label, label_rank=RANK)


def test_t10_a_non_verified_label_cannot_be_a_clean_negative():
    """T10：engine/agent 档的"标签全零"**不构成**确认无线（方案 v2 §3.5）。

    实测缺口：`negative_line_counts` 原来只看标签内容——engine 档的全零帧会被算成
    "合格负例零误报"。现在档位不是 verified（含未声明）时记 `unverified_frames`
    （排除量）与 `unverified_pred_line_px`（可上报的观测），**不进**合格分母。
    """
    empty_pred = np.zeros((4, 4), bool)
    all_zero = np.zeros((4, 4), np.uint8)      # 标签全零（"看起来没有线"）
    # 未声明档位 -> 不可信（不默认通过）
    a = negative_line_counts(empty_pred, all_zero)
    assert a["negative_line_unverified_frames"] == 1, a
    assert a["negative_line_eligible_frames"] == 0, a
    # engine / agent 档同理
    for rank in ("unreliable", "agent", "pseudo", "absent"):
        r = negative_line_counts(empty_pred, all_zero, label_rank=rank)
        assert r["negative_line_eligible_frames"] == 0, (rank, r)
        assert r["negative_line_unverified_frames"] == 1, (rank, r)
    # verified 档才是合格负例（对照）
    v = negative_line_counts(empty_pred, all_zero, label_rank="verified")
    assert v["negative_line_eligible_frames"] == 1, v
    # 混合来源：排除量要报出来，合格分母只含 verified
    out = negative_line_summary(
        {k: a.get(k, 0) + v.get(k, 0) for k in a}, n_frames=2)
    assert out["eligible_frames"] == 1, out
    assert out["unverified_frames"] == 1, out
    assert out["excluded_frames"] == 1, out
    assert "not verified" in (out.get("excluded_reason") or ""), out
    # 在不可信标签上预测了线：像素数可上报，但不算假线结论
    pred = np.ones((4, 4), bool)
    u = negative_line_counts(pred, all_zero, label_rank="agent")
    assert u["negative_line_unverified_pred_line_px"] == 16, u
    assert u["negative_line_false_positive_px"] == 0, u
