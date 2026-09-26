"""停止条件的归因口径（方案 v2 §S5）：只有"有意义的无收益"才计入连胜。

背景（实测反例）：旧口径把任何 `rejected`/`needs_evidence` 都数成"连续 N 轮
无收益"。于是无效因子（road-only 下的线损失键）、缺标注、来源资格失败的轮次
都被当成了"模型已经没有提升空间"的证据，把预算推向停止。
"""

from __future__ import annotations


def test_should_stop_counts_only_meaningful_no_gain() -> None:
    """归因字段接入停止条件：无效因子/缺标注/资格失败不算平台期证据。"""
    from beamng_autopilot.experiments.controller import (
        LoopConfig, RoundRecord, should_stop)

    cfg = LoopConfig(max_rounds_without_gain=3)
    for reason in ("invalid_factor", "missing_labels", "qualification_failure",
                   "not_a_verdict"):
        hist = [RoundRecord(i, f"c{i}", "rejected", ["no gain"],
                            gain=0.0, outcome=reason) for i in range(3)]
        st = should_stop(cfg, hist, gpu_minutes_today=10, candidates_used=3)
        assert not st["stop"], f"{reason} 不得计入连续无收益：{st}"
        assert st["no_gain_streak"] == 0, st
    hist = [RoundRecord(i, f"c{i}", "rejected", ["no gain"],
                        gain=0.0, outcome="meaningful_no_gain")
            for i in range(3)]
    st = should_stop(cfg, hist, gpu_minutes_today=10, candidates_used=3)
    assert st["stop"] and st["no_gain_streak"] == 3, st
    # 有效因子带来的晋级会清零连胜（即使前面已经数了两轮）
    hist2 = hist[:2] + [RoundRecord(2, "c2", "shadow_candidate",
                                    ["improved"], gain=0.1,
                                    outcome="promoted")]
    assert should_stop(cfg, hist2, gpu_minutes_today=10,
                       candidates_used=3)["no_gain_streak"] == 0


def test_legacy_records_without_outcome_keep_the_old_rule() -> None:
    """旧记录没有归因字段：回退旧口径，不猜、不重算。"""
    from beamng_autopilot.experiments.controller import (
        LoopConfig, RoundRecord, should_stop)

    cfg = LoopConfig(max_rounds_without_gain=3)
    hist = [RoundRecord(i, f"c{i}", "rejected", ["no gain"], gain=0.0)
            for i in range(3)]
    st = should_stop(cfg, hist, gpu_minutes_today=10, candidates_used=3)
    assert st["stop"] and st["no_gain_streak"] == 3, st
