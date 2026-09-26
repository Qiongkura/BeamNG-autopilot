"""候选计数与聚合的确定性验收（方案 v2 §3.3/§3.4/§5 的 T03–T07/T09）。

为什么单独一个文件：计数口径是这一轮的核心修复——实测缺口是"入口把各目录的
**比率**取平均、并用全部候选当分母"，于是确定性输入（有参考场景 C=10/R=8/M=6
+ 无参考场景 C=10）算出覆盖率 0.40，而正确口径是 0.80。测试必须能直接抓住
"用比率平均""分母含无参考候选""用 C 冒充样本量"这三种退化。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beamng_autopilot.experiments import candidate_metrics as cm  # noqa: E402


def _scene(P=3, C=10, R=8, M=6, L=6, A=5, outside=0):
    return {"P_frames": P, "C": C, "C_outside_P": outside, "R": R, "M": M,
            "L": L, "A": A}


def test_t03_a_reference_scene_and_a_line_free_scene_do_not_mix_denominators():
    """T03/T04：有线场景 C=10/R=8/M=6 + 真无线场景 C=10。

    覆盖率 = 8/10 = 0.80（**只算 P 帧**）、身份率 = 6/8 = 0.75；无线场景的
    10 条候选单独记 `C_outside_P`，既不进覆盖率分母，也不被当成假线。
    逐帧判断（T04 的同组混合帧由探针的逐帧 `P_frame` 保证）。
    """
    ref = _scene()
    no_line = _scene(P=0, C=0, R=0, M=0, L=0, A=0, outside=10)
    acc = cm.totals([ref, no_line])
    r = cm.ratios(acc)
    assert r["candidate_reference_coverage"] == 0.80, r
    assert r["candidate_identity_rate"] == 0.75, r
    assert r["left_right_role_agreement"] == round(5 / 6, 4), r
    assert acc["C_outside_P"] == 10 and acc["C"] == 10, acc
    # 无线场景的适用性是 not_applicable（不是"通过"，也不是缺测）
    ap = cm.applicability(no_line, has_line_truth=False)
    assert ap["status"] == "not_applicable", ap


def test_t05_micro_is_count_sum_not_ratio_average():
    """T05：两组 (M/R)=1/1 与 1/9 -> micro=2/10=0.20，绝不是比率平均 0.556。"""
    a = _scene(P=1, C=1, R=1, M=1, L=1, A=1)
    b = _scene(P=1, C=9, R=9, M=1, L=1, A=1)
    mm = cm.micro_macro({"g_a": a, "g_b": b})
    assert mm["micro"]["candidate_identity_rate"] == 0.20, mm
    assert mm["micro"]["candidate_identity_rate_numerator"] == 2
    assert mm["micro"]["candidate_identity_rate_denominator"] == 10
    assert mm["macro"]["candidate_identity_rate"] == round((1.0 + 1 / 9) / 2, 4)
    assert mm["macro"]["candidate_identity_rate_units"] == 2, mm
    assert "one unit per" in mm["macro_note"], mm


def test_t06_t07_the_sample_floor_counts_R_not_C():
    """T06/T07：样本量下限按**身份率实际分母 R** 计，不能用总候选数冒充。

    C=100 而 R=1 时，身份率只有 1 个可判候选——"100 条候选"是覆盖率的分母，
    不是身份样本数。
    """
    thin = _scene(P=5, C=100, R=29, M=20, L=20, A=18)
    ok = _scene(P=5, C=100, R=30, M=20, L=20, A=18)
    rep = cm.scene_report({"g_thin": thin, "g_ok": ok}, min_candidates=30)
    assert rep["scenes"]["g_thin"]["sample"] == "insufficient", rep
    assert rep["scenes"]["g_ok"]["sample"] == "ok", rep
    assert any("R=29 < 30" in x for x in rep["low_sample"]), rep
    # C=100 但 R=1：同样是不足，且原因里带 C（让人看到"总候选多不代表样本够"）
    one = cm.scene_report({"g_one": _scene(P=1, C=100, R=1, M=1, L=1, A=1)},
                          min_candidates=30)
    assert one["scenes"]["g_one"]["sample"] == "insufficient", one
    assert "C=100" in one["low_sample"][0], one


def test_zero_denominators_are_null_with_reasons_never_zero_or_one():
    """零分母 -> None + 原因；不许写 0，也不许写满分（方案 v2 §3.3/§3.4）。"""
    acc = cm.totals([_scene(P=2, C=0, R=0, M=0, L=0, A=0)])
    r = cm.ratios(acc)
    assert r["candidate_reference_coverage"] is None
    assert r["candidate_identity_rate"] is None
    assert r["left_right_role_agreement"] is None
    assert "denominator C=0" in r["candidate_reference_coverage_missing"], r
    # 有线真值但一条候选都没有：unknown（"没东西可判"不是 0 分）
    ap = cm.applicability(acc, has_line_truth=True)
    assert ap["status"] == "unknown", ap
    # 有 R 但没有可判角色的候选：角色率 UNKNOWN（L=0），覆盖率/身份率仍实测
    l0 = cm.totals([_scene(P=1, C=10, R=8, M=6, L=0, A=0)])
    r2 = cm.ratios(l0)
    assert r2["left_right_role_agreement"] is None
    assert r2["candidate_identity_rate"] == 0.75, r2
    rep = cm.scene_report({"g": _scene(P=1, C=10, R=8, M=6, L=0, A=0)},
                          min_candidates=30)
    assert any("L=0" in x for x in rep["missing"]), rep


def test_unverified_labels_cannot_be_quoted_as_a_result():
    """非 verified 档：计数可以诊断，但适用性必须标 unverified_labels。"""
    acc = cm.totals([_scene()])
    ap = cm.applicability(acc, has_line_truth=True, label_rank="agent")
    assert ap["status"] == "unverified_labels", ap
    assert cm.applicability(acc, has_line_truth=True,
                            label_rank="verified")["status"] == "measured"


def test_accumulate_only_adds_integers_never_ratios():
    """累加器只吃整数计数：把比率塞进去不会污染计数（键不同即忽略）。"""
    acc = cm.empty()
    cm.accumulate(acc, {"C": 3, "R": 2, "candidate_identity_rate": 0.9,
                        "match_rate": 0.5})
    assert acc["C"] == 3 and acc["R"] == 2, acc
    assert "candidate_identity_rate" not in acc, acc
    cm.accumulate(acc, {"C": 4, "R": 1})
    assert acc["C"] == 7 and acc["R"] == 3, acc
