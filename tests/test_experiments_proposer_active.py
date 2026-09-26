"""S5：后台提议与资源管理补漏——线通道屏蔽（road-only）下的因子有效性。

背景（已实测）：``scripts/m5_seg_autoloop.py`` 的 road-only 模式会给两臂都加
``--ignore-line-class --line-tversky-weight 0 --line-cldice-weight 0``，此时
**任何**线损失因子都不会生效；历史上出现过 5 轮"road-only + 线损失因子"白跑
（``applied_flags`` 里有参数，实际不起作用）。这套测试钉住三件事：

* :func:`factor_activity` 在 road-only 下把每个线损失键判 inactive，而数据 /
  轮次类键照常 active（不能一刀切禁掉所有提议）；
* :func:`classify_round_outcome` 的优先级：资格失败 > 缺标注 > 无效因子 >
  有意义的无收益，只有"有意义"的一种能计入连续无收益；
* ``propose(line_supervision=False)`` 不产出线损失提议，且默认行为逐字不变。

只跑本文件：``.venv/Scripts/python.exe -m pytest
tests/test_experiments_proposer_active.py -q``
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).resolve().parents[1]))

from beamng_autopilot.experiments.proposer import (  # noqa: E402
    LINE_LOSS_KEYS, bucket_errors, classify_round_outcome, factor_activity,
    propose,
)

#: 训练器（TRAINER_FLAG_FACTORS）支持的线损失键：与 proposer 的键表钉在一起。
TRAINER_LINE_KEYS = ("line_weight", "line_tversky_weight",
                     "line_cldice_weight", "line_tversky_beta")


# ---------------------------------------------------------------------------
# factor_activity：road-only 下线损失键一律 inactive
def test_road_only_marks_every_line_loss_key_inactive() -> None:
    assert set(LINE_LOSS_KEYS) == set(TRAINER_LINE_KEYS), \
        "线损失键表漂移了：训练器的线损失键必须逐个覆盖"
    for key in TRAINER_LINE_KEYS:
        act = factor_activity({key: 1.5}, line_supervision=False)
        assert act["active"] is False, f"{key} 在 road-only 下必须判 inactive"
        assert key in act["inactive_keys"], \
            f"inactive_keys 要逐个列出被屏蔽的线损失键：{act}"
        assert act["why"], "必须写明为什么无效（applied_flags 有参数也没用）"
        assert "line" in act["why"], act["why"]


def test_line_loss_keys_are_active_when_line_channel_is_supervised() -> None:
    for key in TRAINER_LINE_KEYS:
        act = factor_activity({key: 1.5}, line_supervision=True)
        assert act["active"] is True, f"{key} 在有 line 监督时必须 active"
        assert act["inactive_keys"] == []
        assert act["why"], "active 也要说明根据"


def test_road_only_keeps_non_line_factors_active() -> None:
    """不能一刀切：数据组成/轮次/学习率/容量在 road-only 下照常生效。"""
    factors = ({"add_runs": ["logs/m5_seg/newrun/front_main"]},
               {"drop_runs": ["logs/m5_seg/oldrun/front_main"]},
               {"group_weights": {"0": 2.0}},
               {"epochs": 6},
               {"lr": 5e-4},
               {"width": 64})
    for factor in factors:
        act = factor_activity(factor, line_supervision=False)
        assert act["active"] is True, f"road-only 下 {factor} 仍然生效：{act}"
        assert act["inactive_keys"] == [], act


def test_empty_and_unknown_factors_are_inactive_with_a_reason() -> None:
    for empty in ({}, None):
        act = factor_activity(empty, line_supervision=True)
        assert act["active"] is False, "空因子改不了任何东西，不能当一轮实验"
        assert act["inactive_keys"] == []
        assert act["why"], "空因子也要写原因"
    # 未知键：不猜它是否生效
    act = factor_activity({"magic_weight": 3.0}, line_supervision=True)
    assert act["active"] is False
    assert act["inactive_keys"] == ["magic_weight"]
    assert "magic_weight" in act["why"]
    # 已知键 + 未知键的混合因子：整条判 inactive（提议纪律：一次只改一个因子）
    mixed = factor_activity({"epochs": 6, "magic_weight": 3.0},
                            line_supervision=True)
    assert mixed["active"] is False
    assert mixed["inactive_keys"] == ["magic_weight"]
    # 非字典输入也不猜
    assert factor_activity("epochs", line_supervision=True)["active"] is False


# ---------------------------------------------------------------------------
# classify_round_outcome：五种归因与优先级
def test_classify_round_outcome_covers_the_five_verdicts() -> None:
    assert classify_round_outcome(
        decision="rejected", factor_active=True, labels_ok=True,
        eligibility_ok=True) == "meaningful_no_gain"
    assert classify_round_outcome(
        decision="needs_evidence", factor_active=True, labels_ok=True,
        eligibility_ok=True) == "meaningful_no_gain"
    assert classify_round_outcome(
        decision="rejected", factor_active=False, labels_ok=True,
        eligibility_ok=True) == "invalid_factor"
    assert classify_round_outcome(
        decision="needs_evidence", factor_active=True, labels_ok=False,
        eligibility_ok=True) == "missing_labels"
    assert classify_round_outcome(
        decision="rejected", factor_active=True, labels_ok=True,
        eligibility_ok=False) == "qualification_failure"
    assert classify_round_outcome(
        decision="shadow_candidate", factor_active=True, labels_ok=True,
        eligibility_ok=True) == "promoted"


def test_classify_priority_is_qualification_labels_factor_then_gain() -> None:
    # 资格失败压过一切（缺标注 + 无效因子同时成立也归资格）
    assert classify_round_outcome(
        decision="needs_evidence", factor_active=False, labels_ok=False,
        eligibility_ok=False) == "qualification_failure"
    # 缺标注压过无效因子（题目点名的组合）
    assert classify_round_outcome(
        decision="needs_evidence", factor_active=False, labels_ok=False,
        eligibility_ok=True) == "missing_labels"
    # 无效因子压过"有意义的无收益"
    assert classify_round_outcome(
        decision="rejected", factor_active=False, labels_ok=True,
        eligibility_ok=True) == "invalid_factor"


def test_only_meaningful_no_gain_may_feed_the_stop_streak() -> None:
    """停止条件的连续无收益只能数"有意义"的那一种，其余是单独原因。"""
    bad_inputs = (
        {"decision": "rejected", "factor_active": False, "labels_ok": True,
         "eligibility_ok": True},
        {"decision": "rejected", "factor_active": True, "labels_ok": False,
         "eligibility_ok": True},
        {"decision": "needs_evidence", "factor_active": True, "labels_ok": True,
         "eligibility_ok": False},
        {"decision": "shadow_candidate", "factor_active": True, "labels_ok": True,
         "eligibility_ok": True},
        {"decision": "failed", "factor_active": True, "labels_ok": True,
         "eligibility_ok": True},
    )
    for kw in bad_inputs:
        got = classify_round_outcome(**kw)
        assert got != "meaningful_no_gain", (kw, got)
    # 非判定（failed/paused/queued）两边都不算
    assert classify_round_outcome(
        decision="paused", factor_active=True, labels_ok=True,
        eligibility_ok=True) == "not_a_verdict"


# ---------------------------------------------------------------------------
# propose 接线：road-only 不产出线损失提议，默认行为不变
def _buckets_missed_line_top():
    """让 top 桶是 missed_true_line（loss_weights 族唯一会产提议的桶）。"""
    return bucket_errors(pixel={
        "offroad_false_line_px": 0, "pred_line_px": 30000,
        "missed_true_line_px": 1500, "gt_line_px": 20000, "model": "m"})


def test_propose_never_emits_line_loss_proposals_under_road_only() -> None:
    buckets = _buckets_missed_line_top()
    blocked: list = []
    props = propose(buckets=buckets, dataset={"n_train_frames": 173},
                    champion={"steps": 129, "epochs": 3},
                    blocked=blocked, line_supervision=False)
    assert props, "road-only 不是一刀切禁掉所有提议（epochs 仍然可提）"
    assert all(not (set(p.factor) & set(LINE_LOSS_KEYS)) for p in props), \
        f"road-only 下不许出现线损失因子：{[p.factor for p in props]}"
    assert [p.family for p in props] == ["epochs"]
    # 被屏蔽的线损失族要单独记账（它不是"模型没有提升空间"）
    assert any(b["family"] == "loss_weights"
               and b.get("inactive_keys") == ["line_tversky_weight"]
               for b in blocked), blocked


def test_propose_default_line_supervision_is_unchanged() -> None:
    """默认 line_supervision=True：提议与旧版逐字一致（含线损失提议）。"""
    buckets = _buckets_missed_line_top()
    default_props = propose(buckets=buckets,
                            dataset={"n_train_frames": 173},
                            champion={"steps": 129, "epochs": 3})
    explicit = propose(buckets=buckets, dataset={"n_train_frames": 173},
                       champion={"steps": 129, "epochs": 3},
                       line_supervision=True)
    assert [p.candidate_id for p in default_props] == \
        [p.candidate_id for p in explicit]
    assert any("line_tversky_weight" in p.factor for p in default_props), \
        "默认（有 line 监督）下仍要产出线损失提议"


def test_line_loss_key_table_matches_the_trainer_whitelist() -> None:
    """提议器与训练入口的白名单不能各说各话（照 test_experiments.py 的做法）。"""
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "m5_seg_autoloop_for_activity", root / "scripts" / "m5_seg_autoloop.py")
    loop = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loop)
    trainer_line = tuple(k for k in loop.TRAINER_FLAG_FACTORS if "line" in k)
    assert set(trainer_line) == set(LINE_LOSS_KEYS), \
        f"线损失键表漂移：提议器 {LINE_LOSS_KEYS} vs 训练器 {trainer_line}"
