"""选样评分与批次配额（方案 §7.5–7.7）。

三条纪律各有反例：
* **无真值区域只能进复核队列**，不能自动当 hard negative（未复核的帧不许直接
  教模型"这里不是线"）；
* 同一地点连续帧与同一相机占比要有上限——否则"一批 20 帧"其实是同一个场景的
  20 个近邻帧，等于没加数据；
* 覆盖（无线负例/左右侧/铺装土肩）不能被"分数最高的那批"冲掉。

评分的每个分量都要能查：分数不是黑箱，报告里能看到它为什么高。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beamng_autopilot.experiments.selection import (  # noqa: E402
    BATCH_SIZE, MAX_CAMERA_SHARE, MAX_PER_GROUP, frame_score, pick_batch,
)


def _f(name, group="italy/ring_a", view="front_main", exposure=0, **kw):
    rec = {"frame": name, "group": group, "view": view, "exposure": exposure,
           "label_source": "human_revision"}
    rec.update(kw)
    return rec


def test_an_unverified_frame_may_only_enter_the_review_queue():
    """§7.5：无真值/未复核区域**不能**自动当 hard negative。"""
    raw = frame_score(_f("f0", label_source="", unknown_frac=0.9,
                         disagreement=0.8, verified_errors=0))
    assert raw["route"] == "review" and raw["may_teach_negative"] is False, raw
    assert "not a confirmed negative" in raw["route_reason"], raw
    # agent 来源：**可以**进研究训练，但同样不许教负例、不能当评价参考
    agent = frame_score(_f("f1", label_source="agent_revision", unknown_frac=0.5))
    assert agent["route"] == "train_research", agent
    assert agent["may_teach_negative"] is False, agent
    # 人工复核过的帧才可以进训练
    ok = frame_score(_f("f2", label_source="human_revision", unknown_frac=0.2))
    assert ok["route"] == "train" and ok["may_teach_negative"] is True, ok


def test_the_score_is_decomposable_and_prefers_uncertainty():
    """评分要能看出为什么高：未知程度/模型分歧/已验证错误/场景缺口逐项列出。"""
    a = frame_score(_f("a", unknown_frac=0.0, disagreement=0.0,
                       verified_errors=0, scene_gap=0.0))
    b = frame_score(_f("b", unknown_frac=0.8, disagreement=0.6,
                       verified_errors=2, scene_gap=1.0))
    assert b["score"] > a["score"], (a, b)
    assert set(b["parts"]) == {"unknown_frac", "disagreement",
                               "verified_errors", "scene_gap"}, b["parts"]
    assert b["parts"]["verified_errors"] == 2
    assert a["score"] == 0.0, a


def test_one_group_cannot_dominate_a_batch():
    """§7.6：同一地点（组）的帧数上限——相邻帧不是独立样本。"""
    pool = [_f(f"g{i}", group="italy/ring_a", exposure=i) for i in range(30)]
    picked = pick_batch([frame_score(r) for r in pool], limit=12)
    per_group = {}
    for it in picked["items"]:
        per_group[it["group"]] = per_group.get(it["group"], 0) + 1
    assert per_group["italy/ring_a"] <= MAX_PER_GROUP, per_group
    assert picked["dropped_by_group_limit"] >= 30 - MAX_PER_GROUP, picked
    assert picked["n_items"] == MAX_PER_GROUP


def test_one_camera_cannot_dominate_a_batch():
    """§7.6：同一相机占比上限（视角不同不等于场景不同）。"""
    pool = []
    for i in range(20):
        pool.append(_f(f"front{i}", group=f"italy/ring_{i}", view="front_main",
                       unknown_frac=0.9))
    for i in range(10):
        pool.append(_f(f"pill{i}", group=f"italy/ring_p{i}", view="pillar_left",
                       unknown_frac=0.1))
    picked = pick_batch([frame_score(r) for r in pool], limit=10)
    views = [it["view"] for it in picked["items"]]
    share = views.count("front_main") / max(1, len(views))
    assert share <= MAX_CAMERA_SHARE, (share, views)


def test_equal_scores_are_spread_across_cameras():
    """同分帧要按相机铺开：否则一批全是同一路相机（实测 640 帧池子前 18 个
    全是 front_fisheye），"加了 20 帧"其实只加了一个视角。"""
    pool = []
    for i in range(12):
        pool.append(_f(f"front{i:02d}", group="italy/ring_a",
                       view="front_main", exposure=i))
    for i in range(12):
        pool.append(_f(f"pill{i:02d}", group="italy/ring_a",
                       view="pillar_left", exposure=i))
    picked = pick_batch([frame_score(r) for r in pool], limit=6)
    views = sorted({it["view"] for it in picked["items"]})
    assert views == ["front_main", "pillar_left"], views
    # 分数仍然优先：高分的单帧不会被同分轮转挤掉
    hot = frame_score(_f("hot", group="italy/ring_z", verified_errors=5))
    picked2 = pick_batch([frame_score(r) for r in pool] + [hot], limit=1)
    assert picked2["items"][0]["frame"] == "hot", picked2["items"]


def test_coverage_kinds_are_kept_even_when_they_score_lower():
    """§7.6：无线负例/土肩这类覆盖不能被"高分帧"冲掉。"""
    pool = []
    for i in range(12):
        pool.append(_f(f"line{i}", group=f"italy/ring_{i}",
                       unknown_frac=0.9, kinds=["line"]))
    pool.append(_f("noline0", group="italy/plain_x", unknown_frac=0.0,
                   kinds=["no_line"], verified_errors=0))
    picked = pick_batch([frame_score(r) for r in pool], limit=4,
                        require_kinds=["no_line"])
    kinds = {k for it in picked["items"] for k in it["kinds"]}
    assert "no_line" in kinds, picked
    assert picked["coverage_filled"], picked
    assert picked["n_items"] <= BATCH_SIZE


def test_a_batch_is_small_by_design_and_reports_what_it_dropped():
    """§7.6：每次只加预设小批，并说明丢了多少、为什么。"""
    pool = [_f(f"x{i}", group=f"italy/ring_{i % 5}", view="front_main",
               unknown_frac=i / 100.0) for i in range(100)]
    picked = pick_batch([frame_score(r) for r in pool], limit=15)
    assert picked["n_items"] <= 15 <= BATCH_SIZE
    assert picked["n_pool"] == 100
    assert picked["dropped_by_group_limit"] + picked["dropped_by_camera_limit"]         + picked["n_items"] <= 100
    assert picked["reasons"], "批次必须给出可读的取舍理由"
