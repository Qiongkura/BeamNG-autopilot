"""全池配额采样器（T16 §3.2）纯逻辑回归。

钉住的语义（都是判定层会读的口径，不能用"看起来对"代替）：

* ``unique_seen`` 只数**真实进过 batch** 的不同内容 hash——预算不足时必须
  小于池容量，绝不能拿池大小/目录文件数冒充"已学习的帧"；
* 一轮（epoch）内不重复抽同一帧；跨轮重新洗牌且相邻两轮顺序不同；
* run 配额按帧数比例，小 run 不被饿死；
* ``state()/from_state()`` 之后序列与曝光计数逐位一致（--resume 的依据）；
* ``plan_epochs`` 的取整数学与训练器的循环上界一致。

只依赖 numpy，不 import torch（采样库本身也不 import）。
"""

from __future__ import annotations

import numpy as np
import pytest

from beamng_autopilot.vision.sampling import (
    QuotaSampler, allocate_run_quotas, frame_identity, plan_epochs,
)


def _frame(k: int, size: int = 4):
    """第 k 张合成帧（colour 内容随 k 变，label 固定）。"""
    rng = np.random.default_rng(1000 + k)
    colour = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    return colour, np.zeros((size, size), np.uint8)


def _pool(spec, *, line_every: int = 2, view=True):
    """spec = [(run, n), ...] -> 池；hash 用真实的 frame_identity。"""
    pool, k = [], 0
    for run, n in spec:
        for i in range(n):
            f = _frame(k)
            pool.append({
                "frame": f, "run": run,
                "view": (f"{run}_v" if view else None),
                "hash": frame_identity(f),
                "has_line": bool(line_every and i % line_every == 0),
            })
            k += 1
    return pool


def _batches(sampler, n: int) -> list[list[int]]:
    return [sampler.next_batch() for _ in range(n)]


# --------------------------------------------------------------- 覆盖/预算
def test_the_whole_pool_is_seen_once_the_budget_covers_a_round():
    pool = _pool([("a", 5), ("b", 3), ("c", 1)])
    s = QuotaSampler(pool, batch=4, seed=42, run_key="ds")
    steps = s.steps_per_epoch
    assert steps == 3                      # ceil(9/4)
    got = _batches(s, steps)
    flat = [j for b in got for j in b]
    assert len(flat) == len(pool), "一轮就是把池走一遍"
    assert len(set(flat)) == len(pool), "一轮内不重复"
    rep = s.report()
    assert rep["unique_available"] == 9
    assert rep["unique_seen"] == 9
    assert rep["exposures_total"] == 9
    assert (rep["frames_seen_with_line"]
            + rep["frames_seen_without_line"]) == rep["unique_seen"]


def test_a_short_budget_does_not_claim_unseen_frames():
    """预算不足：unique_seen = 去重后的 batch×batch，且严格小于池容量。"""
    pool = _pool([("a", 5), ("b", 3), ("c", 1)])
    s = QuotaSampler(pool, batch=4, seed=42, run_key="ds")
    got = _batches(s, 1)
    flat = [j for b in got for j in b]
    rep = s.report()
    assert rep["unique_seen"] == len(set(flat)) == 4
    assert rep["unique_seen"] < rep["unique_available"] == 9
    assert rep["exposures_total"] == 4
    assert rep["pool_size"] == 9


def test_exposures_count_repeats_and_unique_seen_does_not():
    pool = _pool([("a", 2)])
    s = QuotaSampler(pool, batch=1, seed=1, run_key="")
    _batches(s, 5)
    rep = s.report()
    assert rep["unique_seen"] == 2, "只有 2 个不同帧，重复曝光不算新帧"
    assert rep["exposures_total"] == 5


# ---------------------------------------------------------------- 轮次纪律
def test_no_frame_repeats_within_one_round():
    pool = _pool([("a", 7), ("b", 5), ("c", 2)])
    s = QuotaSampler(pool, batch=3, seed=7, run_key="k")
    flat = [j for b in _batches(s, s.steps_per_epoch) for j in b]
    assert sorted(flat) == list(range(len(pool))), "一轮 = 池的一个排列"


def test_consecutive_rounds_never_repeat_the_same_order():
    """池很小时随机洗牌完全可能连着给出同一顺序，必须显式打破。"""
    pool = _pool([("a", 2)])
    s = QuotaSampler(pool, batch=2, seed=3, run_key="")
    orders = [_batches(s, 1)[0] for _ in range(6)]     # 每轮正好一批
    for prev, nxt in zip(orders, orders[1:]):
        assert prev != nxt, f"相邻两轮顺序相同：{prev}"
    assert all(sorted(o) == [0, 1] for o in orders)


def test_runs_are_balanced_by_size_and_the_small_run_is_not_starved():
    pool = _pool([("big", 20), ("mid", 4), ("tiny", 1)])
    s = QuotaSampler(pool, batch=5, seed=11, run_key="q")
    for _ in range(s.steps_per_epoch * 3):             # 整整 3 轮
        s.next_batch()
    rep = s.report()
    exp = rep["exposures_by_run"]
    assert exp["big"] == 60 and exp["mid"] == 12 and exp["tiny"] == 3
    # 比例与帧数比例一致（成正比），而不是"每 run 等量"
    assert exp["big"] / exp["mid"] == pytest.approx(20 / 4, abs=0.01)
    assert exp["mid"] / exp["tiny"] == pytest.approx(4 / 1, abs=0.01)
    assert rep["unique_seen"] == 25, "三轮之后全池覆盖"


def test_view_exposures_are_reported_per_view():
    pool = _pool([("a", 3), ("b", 2)])
    s = QuotaSampler(pool, batch=5, seed=5, run_key="")
    s.next_batch()
    rep = s.report()
    assert rep["exposures_by_view"] == {"a_v": 3, "b_v": 2}


def test_a_frame_without_a_view_is_grouped_under_unknown():
    pool = _pool([("a", 2)], view=False)
    s = QuotaSampler(pool, batch=2, seed=5, run_key="")
    s.next_batch()
    assert s.report()["exposures_by_view"] == {"unknown": 2}


# ------------------------------------------------------------ 确定性/恢复
def test_same_seed_and_pool_gives_the_same_sequence():
    pool = _pool([("a", 4), ("b", 2)])
    s1 = QuotaSampler(pool, batch=3, seed=42, run_key="ds")
    s2 = QuotaSampler(pool, batch=3, seed=42, run_key="ds")
    a, b = _batches(s1, 5), _batches(s2, 5)
    assert a == b
    assert s1.report() == s2.report()
    assert s1.state_digest() == s2.state_digest()
    # 不同 run_key 是另一条独立流（配对实验里用 dataset_id 稳定它）
    s3 = QuotaSampler(pool, batch=3, seed=42, run_key="other")
    assert _batches(s3, 5) != a


def test_state_round_trip_resumes_the_exact_sequence():
    pool = _pool([("a", 6), ("b", 3), ("c", 2)])
    s = QuotaSampler(pool, batch=4, seed=9, run_key="ds")
    _batches(s, 2)                                  # 走到轮中间
    blob = s.state()
    s2 = QuotaSampler.from_state(blob, pool, batch=4)
    assert s2.state_digest() == s.state_digest()
    cont_a, cont_b = _batches(s, 6), _batches(s2, 6)
    assert cont_a == cont_b, "恢复后的序列必须与不中断逐位一致"
    assert s.report() == s2.report()


def test_state_round_trip_across_a_round_boundary():
    """停在轮边界也要一致：下一轮用同一个 rng 状态重新生成。"""
    pool = _pool([("a", 3), ("b", 2)])
    s = QuotaSampler(pool, batch=5, seed=4, run_key="")   # 一轮恰好一批
    _batches(s, 1)
    s2 = QuotaSampler.from_state(s.state(), pool, batch=5)
    assert _batches(s, 4) == _batches(s2, 4)


def test_state_is_json_safe_and_a_resumed_state_is_rejected_on_a_different_pool():
    import json
    pool = _pool([("a", 2), ("b", 2)])
    s = QuotaSampler(pool, batch=2, seed=1, run_key="ds")
    s.next_batch()
    blob = s.state()
    json.dumps(blob)                       # checkpoints store it as plain JSON
    assert isinstance(blob["rng"], dict)
    with pytest.raises(ValueError):
        QuotaSampler.from_state(blob, _pool([("a", 4)]), batch=2)
    with pytest.raises(ValueError):
        QuotaSampler.from_state(blob, pool, batch=3)
    with pytest.raises(ValueError):
        QuotaSampler.from_state(dict(blob, version=99), pool, batch=2)


# ------------------------------------------------------------------ 纯函数
def test_plan_epochs_math():
    assert plan_epochs(5, 6, 4) == {"steps_per_epoch": 2, "epochs": 3,
                                    "total_steps": 5}
    assert plan_epochs(1, 5, 4) == {"steps_per_epoch": 2, "epochs": 1,
                                    "total_steps": 1}
    assert plan_epochs(8, 4, 4) == {"steps_per_epoch": 1, "epochs": 8,
                                    "total_steps": 8}
    for total, n, batch in ((72, 12, 4), (5, 6, 4), (1000, 173, 8)):
        p = plan_epochs(total, n, batch)
        assert p["steps_per_epoch"] == max(1, -(-n // batch))
        assert p["epochs"] * p["steps_per_epoch"] >= total
        assert (p["epochs"] - 1) * p["steps_per_epoch"] < total, "轮数取最少"
    with pytest.raises(ValueError):
        plan_epochs(0, 6, 4)               # 0 = 关闭预算，不走这个函数


def test_allocate_run_quotas_keeps_the_ratio_and_protects_tiny_runs():
    assert allocate_run_quotas([5, 3, 1], 9) == [5, 3, 1]
    assert allocate_run_quotas([100, 1], 50) == [49, 1]
    # 极小的 run 被取整成 0 时，从最大配额借 1 帧，而不是把它饿死
    quotas = allocate_run_quotas([997, 2, 1], 500)
    assert sum(quotas) == 500
    assert min(quotas) >= 1
    # 轮长小于 run 数时不可能人人 1 帧：不虚报，直接按比例给
    assert sum(allocate_run_quotas([1, 1, 1], 2)) == 2


def test_frame_identity_is_content_based_and_stable():
    a = _frame(0)
    same = (a[0].copy(), a[1].copy())
    other = _frame(1)
    assert frame_identity(a) == frame_identity(same)
    assert frame_identity(a) != frame_identity(other)
    assert len(frame_identity(a)) == 16
    # dict 形式与 (colour, label) 元组形式同一个身份
    assert frame_identity({"colour": a[0], "label": a[1]}) == frame_identity(a)
    # 与 manifest 的内容哈希同一个命名空间（跨模块可对账）
    from beamng_autopilot.experiments.manifest import content_sha16
    assert frame_identity(a) == content_sha16(a[0])
