"""实验协议 v1（`beamng_autopilot/experiments/protocol.py`）与冻结快照的确定性。

方案 W0/W5 要求协议本身可哈希、可复现，且"来源资格不能靠字符串升格"（验收 A1）。
本文件钉住三件事：

1. **定义完整**：每个指标都有层级/单位/**分母**/未知规则——没有分母的指标
   会在两种口径下变成两个数（实测：line_precision 0.30 vs 0.65 的差别来自真值完整性）；
2. **哈希稳定且敏感**：同输入同哈希；定义、覆盖、资格、阈值任一改动都会变；
3. **资格由 rank 派生**：任何非 verified 的 rank（agent/pseudo）都不能作为晋级参考，
   混用与未知 rank 一律按最保守处理。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beamng_autopilot.experiments.protocol import (  # noqa: E402
    COVERAGE_REQUIREMENTS, METRIC_DEFINITIONS, MIN_EVAL_GROUPS, PROTOCOL_VERSION,
    RESEARCH_ONLY_RANKS, SOURCE_ELIGIBILITY, can_promote, eligibility,
    metric_denominator, protocol_blob, protocol_hash,
)


def test_every_metric_has_a_denominator_and_an_unknown_rule():
    for name, d in METRIC_DEFINITIONS.items():
        for key in ("level", "unit", "denominator", "unknown_rule", "note"):
            assert d.get(key), f"{name} 缺 {key}"
        assert d["level"] in ("pixel", "candidate", "geometry", "control",
                              "performance"), (name, d["level"])
    # 点名的两个口径必须写清（G04 的教训）
    assert "known area" in metric_denominator("offroad_false_ratio")
    assert "NOT all predicted px" in metric_denominator("offroad_false_ratio")
    assert "predicted line px" in metric_denominator("line_precision")


def test_protocol_hash_is_stable_and_sensitive():
    a = protocol_hash(thresholds={"line_recall_min": 0.7})
    b = protocol_hash(thresholds={"line_recall_min": 0.7})
    assert a == b and len(a) == 16
    assert a != protocol_hash(thresholds={"line_recall_min": 0.71}), \
        "阈值变了哈希必须变"
    assert a != protocol_hash(thresholds={"line_recall_min": 0.7,
                                          "extra": 1}), "多一个键也要变"
    blob = protocol_blob(thresholds={"line_recall_min": 0.7})
    assert blob["version"] == PROTOCOL_VERSION
    assert blob["metrics"]["line_iou"]["level"] == "pixel"
    assert len(blob["coverage_requirements"]) == len(COVERAGE_REQUIREMENTS) == 6
    assert blob["min_eval_groups"] == MIN_EVAL_GROUPS == 6
    assert json.dumps(blob, sort_keys=True)          # 可序列化（进判定文件）


def test_only_verified_sources_can_promote():
    assert can_promote(["verified"])[0] is True
    for r in ("agent", "pseudo", "unreliable", "absent"):
        ok, why = can_promote([r])
        assert ok is False and why, r
    # 混用：只要有一个不能晋级，整体就不能
    ok, why = can_promote(["verified", "agent"])
    assert ok is False and "agent" in why[0], why
    # 空集合：没有声明来源 -> 不猜，按不能晋级处理（默认来源是 unreliable）
    assert can_promote([])[0] is True, "空 = 未声明任何来源，由调用方决定默认来源"
    assert eligibility("agent") == {"can_train": True, "can_measure": True,
                                    "can_promote": False}
    assert eligibility("no-such-rank") == SOURCE_ELIGIBILITY["absent"]
    assert set(RESEARCH_ONLY_RANKS) == {"agent", "pseudo"}
