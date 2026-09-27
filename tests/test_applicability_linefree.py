"""适用性接线：已证明无线的场景，线通道**缺测**记 N/A 而不是 UNKNOWN（T16 §3.5/P0）。

实测缺陷（`logs/experiments/t14_line_promotable2_20260927/decision_line-add-verified-r0.json`）：
同一场景在 `scene_applicability` 里是 `not_applicable`，却在 `missing_metrics`
里被索要 `line_recall`（自相矛盾）。根源 = `gates.scene_report` 的像素度量路
没有像 `scene_count_violations` 那样过滤无线场景。修复纪律（三条都要钉住）：

1. 只有**已证明**无线（rank verified + 负例合格计数/零线像素证据）才 N/A；
2. 未证明（rank 非 verified、负例计数不合格、有线真值）仍是 UNKNOWN，
   不得靠档位默认放行；
3. 不放宽任何门：有线场景的低召回照旧违反，"永远不画线"仍被召回门拦下；
   已测到的线通道坏值（如无线场景里画出假线 -> precision 0）不因 N/A 洗白。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beamng_autopilot.experiments import gates  # noqa: E402


def _t() -> gates.Thresholds:
    return gates.Thresholds()


# ---------------------------------------------------------------------------
# 1) proven_line_free 四档判据
# ---------------------------------------------------------------------------
def test_strong_tier_needs_verified_rank_and_eligible_negative_counters():
    p = gates.proven_line_free(
        p_frames=0, label_rank="verified",
        negative={"eligible_frames": 5, "positive_frames": 0,
                  "unknown_frames": 0})
    assert p["proven"] is True and p["basis"] == "negative_eligible_counters", p
    assert set(p) == {"proven", "basis", "why"}, p
    # 带 negative_line_ 前缀的键也认（reduce 调用方的拼接方式不同）
    p2 = gates.proven_line_free(
        p_frames=0, label_rank="verified",
        negative={"negative_line_eligible_frames": 2,
                  "negative_line_positive_frames": 0,
                  "negative_line_unknown_frames": 0})
    assert p2["proven"] is True, p2
    assert p2["basis"] == "negative_eligible_counters", p2


def test_weak_tier_is_rank_verified_zero_line_pixels_without_counters():
    """没有负例计数时弱一档放行，但 basis 必须不同（判定里可分辨）。"""
    for negative in (None, {}):
        p = gates.proven_line_free(p_frames=0, label_rank="verified",
                                   negative=negative)
        assert p["proven"] is True and p["basis"] == (
            "rank_verified_zero_line_pixels"), p
        assert p["basis"] != "negative_eligible_counters", p


def test_unverified_rank_never_proves_line_free():
    for rank in ("", "absent", "unreliable", "agent", "pseudo",
                 "unverified"):
        p = gates.proven_line_free(p_frames=0, label_rank=rank)
        assert p["proven"] is False and p["basis"] == "unverified_labels", p
        assert p["why"], p
    # p_frames 不是整数读数：连真值有没有都判不了，不许 N/A
    p = gates.proven_line_free(p_frames=None, label_rank="verified")
    assert p["proven"] is False and p["basis"] == "unknown_p_frame_count", p


def test_a_scene_with_line_truth_is_never_n_a():
    p = gates.proven_line_free(
        p_frames=3, label_rank="verified",
        negative={"eligible_frames": 9, "positive_frames": 0,
                  "unknown_frames": 0})
    assert p["proven"] is False and p["basis"] == "has_line_truth", p


def test_incomplete_negative_counters_do_not_fall_back_to_the_weak_tier():
    """给了 negative 但不合格时**不退回弱档**：已有证据不支持无线，不能靠档位放行。"""
    bad = [
        {"eligible_frames": 0, "positive_frames": 0, "unknown_frames": 0},
        {"eligible_frames": 4, "positive_frames": 2, "unknown_frames": 0},
        {"eligible_frames": 4, "positive_frames": 0, "unknown_frames": 1},
        {"eligible_frames": 4},                       # 缺 unknown_frames
    ]
    for negative in bad:
        p = gates.proven_line_free(p_frames=0, label_rank="verified",
                                   negative=negative)
        assert p["proven"] is False, (negative, p)
        assert p["basis"] == "negative_counters_not_eligible", (negative, p)


# ---------------------------------------------------------------------------
# 2) scene_report 的 N/A 接线：默认逐字不变，带集合只搬缺测的线通道字段
# ---------------------------------------------------------------------------
#: 固定输入（含两个无线场景：一个有线通道全缺测，一个 precision=0 的假线读数）
_PER_SCENE = {
    "italy/ring_wired": {"line_recall": 0.50, "line_precision": 0.90,
                         "offroad_false_ratio": 0.01, "inference_ms_p95": 20.0},
    "italy/ring_noline": {"line_recall": None, "line_precision": 0.0,
                          "offroad_false_ratio": 0.02,
                          "inference_ms_p95": 19.0},
    "italy/ring_empty": {"inference_ms_p95": 18.0},
}

_DEFAULT_OUT = {
    "violations": [
        "scene italy/ring_noline: line_precision: 0.0 < 0.4",
        "scene italy/ring_wired: line_recall: 0.5 < 0.7",
    ],
    "missing": [
        "scene italy/ring_empty: line_recall: UNKNOWN (hard gate needs a measurement)",
        "scene italy/ring_empty: line_precision: UNKNOWN (hard gate needs a measurement)",
        "scene italy/ring_empty: offroad_false_ratio: UNKNOWN (hard gate needs a measurement)",
        "scene italy/ring_noline: line_recall: UNKNOWN (hard gate needs a measurement)",
    ],
}

_LINE_FREE = {
    "italy/ring_empty": {
        "basis": "negative_eligible_counters",
        "why": "12 verified negative frames, positive=unknown=0"},
    "italy/ring_noline": {
        "basis": "rank_verified_zero_line_pixels",
        "why": "rank verified and p_frames=0 (no negative counters supplied)"},
}


def test_scene_report_without_the_optional_set_is_byte_identical():
    """默认路径逐字不变（旧判定重放的前提）。"""
    out = gates.scene_report(_PER_SCENE, _t())
    assert out == _DEFAULT_OUT, out
    assert set(out) == {"violations", "missing"}, out
    assert gates.scene_report(_PER_SCENE, _t(),
                              line_free_scenes=None) == _DEFAULT_OUT


def test_scene_report_moves_missing_line_fields_to_not_applicable():
    out = gates.scene_report(_PER_SCENE, _t(), line_free_scenes=_LINE_FREE)
    # 线通道缺测不再进 missing（自相矛盾的根源）
    assert not any("line_recall" in m for m in out["missing"]), out
    assert out["missing"] == [
        "scene italy/ring_empty: offroad_false_ratio: UNKNOWN "
        "(hard gate needs a measurement)"], out
    # 有线场景、以及有线通道的硬门违反一个不动（门没被取消）
    assert out["violations"] == _DEFAULT_OUT["violations"], out
    # N/A 条目带场景名/字段名/basis/why，可原样落盘并重放
    assert out["not_applicable"] == [
        {"scene": "italy/ring_empty", "field": "line_recall",
         "basis": "negative_eligible_counters",
         "why": _LINE_FREE["italy/ring_empty"]["why"], "measured": None},
        {"scene": "italy/ring_empty", "field": "line_precision",
         "basis": "negative_eligible_counters",
         "why": _LINE_FREE["italy/ring_empty"]["why"], "measured": None},
        {"scene": "italy/ring_noline", "field": "line_recall",
         "basis": "rank_verified_zero_line_pixels",
         "why": _LINE_FREE["italy/ring_noline"]["why"], "measured": None},
    ], out["not_applicable"]
    for g, info in _LINE_FREE.items():
        assert out["scenes"][g]["applicability"] == "not_applicable", out
        assert out["scenes"][g]["basis"] == info["basis"], out
    # 证据可 JSON 化（主 agent 写进判定 blob -> replay 读回同一划分）
    assert json.loads(json.dumps(out, ensure_ascii=False)) == out


def test_scene_report_refuses_n_a_without_recorded_basis():
    """没有 basis 的"无线"声明 = 无证据豁免：直接报错，不许静默记 N/A。"""
    import pytest
    for bad in ({"basis": "", "why": "x"}, {}, None):
        with pytest.raises(ValueError, match="basis"):
            gates.scene_report(_PER_SCENE, _t(),
                               line_free_scenes={"italy/ring_noline": bad})


def test_an_unproven_line_free_scene_still_goes_unknown():
    """主 agent 接线模拟：只把 proven 的场景放进集合；未证明的仍 UNKNOWN。"""
    counts = {
        "wire_proven": {"P_frames": 0, "rank": "verified",
                        "negative": {"eligible_frames": 8,
                                     "positive_frames": 0, "unknown_frames": 0}},
        "wire_unproven": {"P_frames": 0, "rank": "agent",
                          "negative": {"eligible_frames": 8,
                                       "positive_frames": 0,
                                       "unknown_frames": 0}},
    }
    lf = {}
    for g, c in counts.items():
        pr = gates.proven_line_free(p_frames=c["P_frames"],
                                    label_rank=c["rank"],
                                    negative=c["negative"])
        if pr["proven"]:
            lf[g] = {"basis": pr["basis"], "why": pr["why"]}
    assert sorted(lf) == ["wire_proven"], lf
    per_scene = {
        "wire_proven": {"line_recall": None, "line_precision": None,
                        "offroad_false_ratio": 0.0, "inference_ms_p95": 9.0},
        "wire_unproven": {"line_recall": None, "line_precision": 0.0,
                          "offroad_false_ratio": 0.0,
                          "inference_ms_p95": 9.0},
    }
    out = gates.scene_report(per_scene, _t(), line_free_scenes=lf)
    assert not any("wire_proven" in m for m in out["missing"]), out
    # 未证明：line_recall 仍是 UNKNOWN -> needs_evidence（不是 N/A、不是通过）
    assert any("wire_unproven" in m and "line_recall" in m
               for m in out["missing"]), out["missing"]


# ---------------------------------------------------------------------------
# 3) 门没被放宽
# ---------------------------------------------------------------------------
def test_a_wired_scene_that_never_draws_a_line_still_violates_recall():
    """反例："永远不画线"在有线场景仍是漏线（recall 0），不得被 N/A 通道洗白。"""
    never_draw = {"P_frames": 66, "rank": "verified",
                  "negative": {"eligible_frames": 0, "positive_frames": 60,
                               "unknown_frames": 0}}
    pf = gates.proven_line_free(p_frames=never_draw["P_frames"],
                                label_rank=never_draw["rank"],
                                negative=never_draw["negative"])
    assert pf["proven"] is False and pf["basis"] == "has_line_truth", pf
    per_scene = {"italy/ring_wired": {"line_recall": 0.0,
                                      "line_precision": None,
                                      "offroad_false_ratio": 0.0,
                                      "inference_ms_p95": 20.0}}
    # 即便调用方错误地把有线场景塞进了 N/A 集合，已测到的 recall=0 也照旧违反
    out = gates.scene_report(
        per_scene, _t(),
        line_free_scenes={"italy/ring_wired": {
            "basis": "forged", "why": "must not hide a measured value"}})
    assert any("line_recall: 0.0 < 0.7" in v for v in out["violations"]), out
    # 池化路径同样：never-draw 的 line_recall=0 仍然违反
    assert any("line_recall: 0.0 < 0.7" in v
               for v in gates.threshold_violations({"line_recall": 0.0}, _t()))


def test_threshold_violations_default_is_unchanged_and_drop_fields_is_opt_in():
    t = _t()
    measured = {"line_precision": 0.1}
    expected = [
        "line_precision: 0.1 < 0.4",
        "candidate_identity_rate: UNKNOWN (hard gate needs a measurement)",
        "line_recall: UNKNOWN (hard gate needs a measurement)",
        "offroad_false_ratio: UNKNOWN (hard gate needs a measurement)",
        "inference_ms_p95: UNKNOWN (hard gate needs a measurement)",
    ]
    assert gates.threshold_violations(measured, t) == expected
    assert gates.threshold_violations(measured, t, drop_fields=None) == expected
    assert gates.threshold_violations(measured, t, drop_fields=()) == expected
    # 显式豁免：违反与缺测两个通道一起移除（由调用方决定并落盘）
    dropped = gates.threshold_violations(
        measured, t, drop_fields=("line_precision", "line_recall",
                                  "offroad_false_ratio", "inference_ms_p95"))
    assert dropped == [
        "candidate_identity_rate: UNKNOWN (hard gate needs a measurement)"], \
        dropped
    # 没点名的字段不受影响
    partial = gates.threshold_violations(measured, t, drop_fields=("line_recall",))
    assert "line_precision: 0.1 < 0.4" in partial, partial
    assert not any(x.startswith("line_recall") for x in partial), partial


def test_line_channel_fields_are_derived_from_the_scene_hard_fields():
    """线通道豁免名单必须从 SCENE_HARD_FIELDS 派生，不能自己另立一份。"""
    assert gates.LINE_CHANNEL_FIELDS == ("line_recall", "line_precision"), \
        gates.LINE_CHANNEL_FIELDS
    assert all(f in gates.SCENE_HARD_FIELDS for f in gates.LINE_CHANNEL_FIELDS)
    # offroad_false_ratio / 耗时不是线通道字段：无线场景里它们仍按硬门检查
    assert "offroad_false_ratio" not in gates.LINE_CHANNEL_FIELDS
    assert "inference_ms_p95" not in gates.LINE_CHANNEL_FIELDS
