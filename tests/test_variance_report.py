"""方差归因工具的纯函数：逐帧路面统计与三方分解。

下一阶段方案第 2 项要求"按 seed、路段、epoch 查看差异"，并明确"只有测量稳定后
才增加 seed 或比较新因子"。这里的分解口径必须能被手算复核，否则报告里的
"波动来自哪里"就是一句空话。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "m5_seg_variance_report", ROOT / "scripts" / "m5_seg_variance_report.py")
var = importlib.util.module_from_spec(_spec)
sys.modules["m5_seg_variance_report"] = var
_spec.loader.exec_module(var)


def test_road_frame_metrics_counts_and_none_denominator():
    lab = np.zeros((4, 4), np.uint8)
    lab[0:2, :] = 1                      # 8 px road
    pred = np.zeros((4, 4), bool)
    pred[0:2, :] = True                  # 全对
    m = var.road_frame_metrics(pred, lab)
    assert (m["tp"], m["fp"], m["fn"]) == (8, 0, 0) and m["iou"] == 1.0
    # 预测一半
    pred2 = np.zeros((4, 4), bool)
    pred2[0, :] = True
    m2 = var.road_frame_metrics(pred2, lab)
    assert m2["tp"] == 4 and m2["fn"] == 4 and abs(m2["iou"] - 0.5) < 1e-9
    # 有预测但该帧没有路面真值：分母存在，IoU=0 是**真实测量**（"这里没有路面却画了"）
    empty = np.zeros((4, 4), np.uint8)
    assert var.road_frame_metrics(pred, empty)["iou"] == 0.0
    # 既没有预测也没有真值（分母为 0）-> None，不许返回 0 冒充"测过"
    none_pred = np.zeros((4, 4), bool)
    assert var.road_frame_metrics(none_pred, empty)["iou"] is None
    # ignore(255) 不计入
    lab3 = np.full((4, 4), 255, np.uint8)
    lab3[0, 0] = 1
    m3 = var.road_frame_metrics(pred, lab3)
    # ignore(255) 不进 known：只有那 1 个像素是真值，且被预测覆盖 -> fn=0
    assert m3["gt_px"] == 1 and m3["fn"] == 0


def test_variance_decomposition_splits_seed_arm_and_epoch():
    """手算一组：两个 arm、每个 2 seed、每 seed 2 epoch。"""
    recs = [
        # arm A：seed 1 稳（0.40,0.40），seed 2 稳（0.60,0.60） -> 波动在 seed 间
        {"arm": "A", "seed": 1, "epoch": 0, "road_iou": 0.40},
        {"arm": "A", "seed": 1, "epoch": 1, "road_iou": 0.40},
        {"arm": "A", "seed": 2, "epoch": 0, "road_iou": 0.60},
        {"arm": "A", "seed": 2, "epoch": 1, "road_iou": 0.60},
        # arm B：seed 3 两轮跳（0.30,0.50） -> 波动在 seed 内
        {"arm": "B", "seed": 3, "epoch": 0, "road_iou": 0.30},
        {"arm": "B", "seed": 3, "epoch": 1, "road_iou": 0.50},
        {"arm": "B", "seed": 4, "epoch": 0, "road_iou": 0.40},
        {"arm": "B", "seed": 4, "epoch": 1, "road_iou": 0.40},
    ]
    d = var.variance_decomposition(recs)
    assert d["n"] == 8
    assert abs(d["grand_mean"] - 0.45) < 1e-9
    # 总平方和 = 臂间 + seed 间 + seed 内
    assert abs(d["total_ss"] - (d["between_arm"] + d["between_seed"]
                                + d["within_seed_epoch"])) < 1e-6
    # A 的 seed 间明显（0.40/0.60 各 2 次），B 的 seed 内明显（0.30/0.50）
    assert d["arms"]["A"]["seeds"]["1"]["mean"] == 0.4
    assert d["arms"]["B"]["seeds"]["3"]["epochs"] == {"0": 0.3, "1": 0.5}
    # 没有测量的记录被跳过而不是当 0
    d2 = var.variance_decomposition(recs + [
        {"arm": "A", "seed": 9, "epoch": 0, "road_iou": None}])
    assert d2["n"] == 8


def test_dominant_frame_share_reports_what_the_worst_frames_cost():
    per_frame = [{"frame": i, "iou": v, "gt_px": 100}
                 for i, v in enumerate([0.9, 0.8, 0.7, 0.2, 0.0])]
    per_frame.append({"frame": 99, "iou": None, "gt_px": 0})
    r = var.dominant_frame_share(per_frame, worst_k=2)
    assert r["n"] == 5 and r["n_no_denominator"] == 1
    assert abs(r["mean"] - (0.9 + 0.8 + 0.7 + 0.2 + 0.0) / 5) < 1e-9
    assert [w["frame"] for w in r["worst"]] == [4, 3]
    assert abs(r["mean_without_worst"] - (0.9 + 0.8 + 0.7) / 3) < 1e-9
    assert r["lift_from_dropping_worst"] > 0
    # 全都没有分母时如实报
    r2 = var.dominant_frame_share([{"frame": 0, "iou": None}])
    assert r2["mean"] is None and r2["n"] == 0


def test_the_trivial_all_road_prediction_is_reported_as_a_reference():
    """平凡基线参照：类别不平衡会让指标"白送"高分。

    实测：开发集标签里路面占 40.2%，而"把全部像素预测成路面"就能拿到 IoU≈0.42
    ——看起来稳定的 0.4239 读数其实几乎全是白送的。评估必须同时给出这个参照，
    否则 0.42 会被读成"模型学会了路面"。
    """
    import numpy as np

    from scripts import m5_seg_eval_matrix as em
    lab = np.zeros((10, 10), np.uint8)
    lab[0:4, :] = 1                     # 40% 路面，60% 背景
    all_road = np.ones((10, 10), bool)
    m = em.road_pixel_metrics(all_road, lab)
    acc = {}
    em.accumulate(acc, m)
    em.accumulate(acc, em.line_pixel_metrics(np.zeros((10, 10), bool), lab))
    out = em.totals_to_metrics(acc, n_frames=1, ms=[10.0])
    assert out["road_iou"] == 0.4
    assert out["road_iou_trivial_all_road"] == 0.4, "平凡基线要与它相等"
    assert out["road_iou_trivial_all_background"] == 0.0
    # 全背景预测的 IoU 是 0（分母有真值），而不是 None
    m2 = em.road_pixel_metrics(np.zeros((10, 10), bool), lab)
    acc2 = {}
    em.accumulate(acc2, m2)
    assert em.totals_to_metrics(acc2, n_frames=1,
                                ms=[10.0])["road_iou"] == 0.0
