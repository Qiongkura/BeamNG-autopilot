"""S3 剩余：门控的样本量/适用性判定与旧判定重放兼容（方案 v2 §S3.2/§3.4/§S3.7）。

与本文件的分工：``tests/test_candidate_metrics.py`` 卡的是**计数层**（T03–T05、
micro/macro），这里卡的是**判定层**——``gates.scene_count_violations`` 是否按 R
（身份率分母）判样本量、适用性是否只派发 not_applicable/unknown 而不是 0 分、
以及旧判定文件能不能按新分母重放（不能就说清楚，且**不许补零**）。

四个必须抓住的退化（都踩过或方案点名）：
* 用总候选 C 冒充身份样本量（C=100、R=1 被判"样本够"）；
* 给无标线真值的场景记缺测/记 0 分（它不是不通过，是**不适用**）；
* 把零分母写成 0 或满分（"没测到"变成"测得 0"）；
* 旧判定文件被补上默认计数后重判（分母是凭空造的）。
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beamng_autopilot.experiments import candidate_metrics as cm  # noqa: E402
from beamng_autopilot.experiments import gates  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _counts(P=40, C=32, R=30, M=25, L=20, A=15, outside=0):
    """一个场景的 v5 整数计数（缺省是"够样本"的一组）。"""
    return {"P_frames": P, "C": C, "C_outside_P": outside, "R": R, "M": M,
            "L": L, "A": A}


def _declared(**kw) -> gates.Thresholds:
    """v3/v4 的候选门数值（显式声明）；测试里只覆盖需要的那几个键。"""
    base = dict(candidate_reference_coverage_min=0.80,
                left_right_role_agreement_min=0.70,
                per_scene_min_candidates=30)
    base.update(kw)
    return gates.Thresholds(**base)


def test_t06_the_sample_floor_is_counted_on_r_and_only_when_declared():
    """T06：per_scene_min_candidates=30 卡的是 R；未声明时旧语义不变。

    R 是 ``M/R`` 的分母——身份率能判的样本数。R=29 与 R=30 只差一条候选，
    判定必须在这一条上翻转；而旧阈值文件（``None``）里这个门**未声明**，
    29 和 30 都不得进 ``low_sample``（否则历史判定会被新门重判）。
    """
    t = _declared()
    thin = {"italy/ring_a": _counts(C=31, R=29, M=24, L=20, A=15)}
    rep = gates.scene_count_violations(thin, t)
    assert len(rep["low_sample"]) == 1 and "R=29" in rep["low_sample"][0], rep
    assert rep["missing"] == [], rep            # 有测量，只是样本不够
    assert rep["scenes"]["italy/ring_a"]["sample"] == "insufficient", rep
    assert rep["scenes"]["italy/ring_a"]["applicability"] == "measured", rep
    ok = {"italy/ring_a": _counts(C=33, R=30, M=25, L=20, A=15)}
    rep_ok = gates.scene_count_violations(ok, t)
    assert rep_ok["low_sample"] == [] and rep_ok["missing"] == [], rep_ok
    assert rep_ok["scenes"]["italy/ring_a"]["sample"] == "ok", rep_ok
    # 下限未声明（旧阈值文件）：R=29/30 都不进 low_sample，也不进缺测
    old = gates.Thresholds()
    assert old.per_scene_min_candidates is None, old
    for per_scene in (thin, ok):
        rep_old = gates.scene_count_violations(per_scene, old)
        assert rep_old["low_sample"] == [], rep_old
        assert rep_old["missing"] == [], rep_old


def test_t07_c_100_with_r_1_is_still_insufficient():
    """T07：C=100 但 R=1 -> 仍判不足，且原因里带 C。

    覆盖率的分母是 C、身份率的分母是 R，两者不能互换；把"总候选 100 条"
    当成"身份样本 100 条"正是方案 v2 §S3.2 点名的退化。
    """
    t = _declared()
    per_scene = {"long_scene": _counts(P=60, C=100, R=1, M=1, L=1, A=1)}
    rep = gates.scene_count_violations(per_scene, t)
    msgs = [m for m in rep["low_sample"] if "R=1" in m]
    assert len(msgs) == 1, rep
    msg = msgs[0]
    assert "C=100" in msg and "< 30" in msg, msg   # 看到了 C，但仍然按 R 判
    assert rep["missing"] == [], rep              # 不是缺测：有 1 条可判
    # L=1 同时触发"角色分母太小"的提醒（两个分母各自独立上报）
    assert any("L=1" in m for m in rep["low_sample"]), rep
    entry = rep["scenes"]["long_scene"]
    assert entry["candidate_identity_rate_denominator"] == 1, entry
    assert entry["candidate_identity_rate"] == 1.0, entry
    # 覆盖率继续按 C 算（1/100），说明两个分母各归其位
    assert entry["candidate_reference_coverage_denominator"] == 100, entry
    assert entry["candidate_reference_coverage"] == 0.01, entry


def test_t09_applicability_not_applicable_and_unknown_are_not_zero_scores():
    """T09：适性三态——not_applicable（无真值）、unknown（R=0）、measured。

    * 无标线真值的场景（P_frames=0）只标 not_applicable：既不通过也不缺测，
      由负例/边界任务评价（记成缺测会永久挡住所有实验）；
    * 有真值但 R=0 -> ``missing`` + UNKNOWN（没有分母就不是 0 分）；
    * L=0 且 R>0 -> 角色率 UNKNOWN 可见，同场景的覆盖率/身份率照实测。
    """
    t = _declared()
    per_scene = {
        # 人确认无线：候选记 C_outside_P，不进覆盖率分母
        "no_line": _counts(P=0, C=0, R=0, M=0, L=0, A=0, outside=12),
        # 有线真值但没候选有可用参考
        "truth_but_no_reference": _counts(P=25, C=40, R=0, M=0, L=0, A=0),
        # 有参考、够样本，但没有可判左右角色的候选
        "no_role": _counts(P=40, C=35, R=30, M=25, L=0, A=0),
    }
    rep = gates.scene_count_violations(per_scene, t)
    # 1) 无线场景：not_applicable，且 missing/low_sample 两个通道都不进
    nl = rep["scenes"]["no_line"]
    assert nl["applicability"] == "not_applicable", nl
    assert "no line truth" in nl["why"], nl
    assert not any("no_line" in m for m in rep["missing"] + rep["low_sample"]), rep
    # 2) 有线真值但 R=0：UNKNOWN（进缺测通道），不是 low_sample、不是 0 分
    assert rep["scenes"]["truth_but_no_reference"]["applicability"] == "unknown"
    assert not any("truth_but_no_reference" in m
                   for m in rep["low_sample"]), rep
    msgs = [m for m in rep["missing"] if "truth_but_no_reference" in m]
    assert len(msgs) == 1, rep["missing"]
    assert "R=0" in msgs[0] and "UNKNOWN" in msgs[0] and "not 0" in msgs[0], msgs
    tbr = rep["scenes"]["truth_but_no_reference"]
    assert tbr["candidate_identity_rate"] is None, tbr
    assert "denominator R=0" in tbr["candidate_identity_rate_missing"], tbr
    assert tbr["candidate_reference_coverage"] == 0.0, tbr   # 40 条候选全没有参考
    # 3) L=0 且 R>0：角色率 UNKNOWN 可见；身份率照报且样本判定仍为 ok
    nr = rep["scenes"]["no_role"]
    assert nr["role_denominator_L"] == 0 and nr["sample"] == "ok", nr
    assert nr["applicability"] == "measured", nr
    assert nr["candidate_identity_rate"] == round(25 / 30, 4), nr
    assert nr["left_right_role_agreement"] is None, nr
    assert nr["left_right_role_agreement_denominator"] == 0, nr
    role_msgs = [m for m in rep["missing"] if "no_role" in m and "role" in m]
    assert role_msgs and "UNKNOWN" in role_msgs[0], rep["missing"]


def test_t12_one_set_of_denominators_across_candidate_metrics_and_gates():
    """T12：同一组整数计数只有一套分母（gates 直接复用 candidate_metrics）。

    判定的分子分母必须与被判定的计数来自同一实现：这里逐场景比对
    ``gates.scene_count_violations`` 的条目与 ``candidate_metrics.ratios``，
    并确认总体口径是 micro（整数求和后再算一次），不是逐场景比率平均——
    两者在样例里刻意不同（覆盖率 0.9091 vs 0.95、身份率 0.55 vs 0.75）。
    """
    per_scene = {
        "s1": _counts(P=10, C=2, R=2, M=2, L=2, A=2),
        "s2": _counts(P=90, C=20, R=18, M=9, L=9, A=6),
    }
    rep = gates.scene_count_violations(per_scene, _declared())
    for g, c in per_scene.items():
        r = cm.ratios(c)
        entry = rep["scenes"][g]
        for name in cm.RATIO_SPECS:
            assert entry[name] == r[name], (g, name, entry[name], r[name])
            assert (entry[f"{name}_denominator"]
                    == r[f"{name}_denominator"]), (g, name)
    # 判定与计数层完全同源：没有无真值场景时，缺测/低样本逐字相同
    direct = cm.scene_report(per_scene, min_candidates=30)
    assert rep["missing"] == direct["missing"], rep
    assert rep["low_sample"] == direct["low_sample"], rep
    # 总体：micro = 计数求和后算一次比率
    mm = cm.micro_macro(per_scene)
    micro = cm.ratios(cm.totals(list(per_scene.values())))
    for name in cm.RATIO_SPECS:
        assert mm["micro"][name] == micro[name], (name, mm["micro"], micro)
    assert micro["candidate_reference_coverage"] == round(20 / 22, 4), micro
    assert micro["candidate_identity_rate"] == 0.55, micro
    # macro 是"每场景一个单位"的平均：与 micro 不同，且必须带单位说明
    assert mm["macro"]["candidate_reference_coverage"] == 0.95, mm
    assert mm["macro"]["candidate_identity_rate"] == 0.75, mm
    assert mm["macro"]["candidate_identity_rate"] != micro[
        "candidate_identity_rate"], mm
    assert "one unit per scene" in mm["macro_note"], mm


def test_t13_a_legacy_decision_is_not_re_judged_and_never_backfilled():
    """T13：旧判定没有 v5 计数 -> 说明"不能按新分母重判"，且不改写 blob。

    旧 blob 只有比率（``hard_by_seed``/``pairings``），重判要用的 R/C、M/R、A/L
    一个都没有；把缺字段补成 0 会让"没测到"变成"测得 0 分"，所以函数只给说明。
    """
    legacy = {"candidate_id": "r3_cand_x",
              "decision": {"decision": "rejected", "reasons": ["..."]},
              "hard_by_seed": {"42": {"line_recall": 0.80, "line_precision": 0.50,
                                      "candidate_identity_rate": 0.62}},
              "pairings": {"line_recall": {"n": 3, "mean_delta": -0.02}},
              "per_scene": {"italy/ring_a": {"line_recall": 0.5}}}
    before = copy.deepcopy(legacy)
    note = gates.legacy_replay_note(legacy)
    assert note == gates.LEGACY_REPLAY_NOTE, note
    assert "predates the v5 counting contract" in note, note
    assert "cannot be re-judged" in note and "re-measure" in note, note
    assert "do not backfill zeros" in note, note
    # 纯函数：不改写判定 blob（也不往里面补任何计数）
    assert legacy == before, "legacy_replay_note 必须只读"
    assert "counts" not in legacy and "counts_by_group" not in legacy, legacy
    # 空/全 None 的计数等于"没测到"：仍然不能重判（不补零）
    assert gates.legacy_replay_note({"counts": {}}) == gates.LEGACY_REPLAY_NOTE
    assert gates.legacy_replay_note({"counts": {"R": None, "C": None}}) == \
        gates.LEGACY_REPLAY_NOTE
    assert gates.legacy_replay_note({}) == gates.LEGACY_REPLAY_NOTE
    assert gates.legacy_replay_note(None) == gates.LEGACY_REPLAY_NOTE


def test_t13_v5_counters_make_the_decision_replayable():
    """T13（另一半）：带 v5 整数计数的判定 -> 可以按同一分母重放（返回 None）。"""
    counts = _counts(P=40, C=32, R=30, M=25, L=20, A=15)
    with_counts = {"candidate_id": "r4_cand_y",
                   "counts": dict(counts),
                   "counts_by_group": {"front_main": dict(counts)},
                   "decision": {"decision": "shadow_candidate"}}
    assert gates.legacy_replay_note(with_counts) is None
    # 逐场景累加器（autoloop 落盘的 scene_counts）也算
    assert gates.legacy_replay_note(
        {"scene_counts": {"italy/ring_a": dict(counts)}}) is None
    # 逐 seed 新口径：计数内联在 hard_by_seed 或挂在它的 counts 下
    assert gates.legacy_replay_note(
        {"hard_by_seed": {"42": {**dict(counts), "line_recall": 0.8}}}) is None
    assert gates.legacy_replay_note(
        {"hard_by_seed": {"42": {"line_recall": 0.8,
                                 "counts": dict(counts)}}}) is None
    # 实测到 0 也是计数（P_frames=0 是"确认过没有标线帧"的读数）
    assert gates.legacy_replay_note(
        {"counts": {"P_frames": 0, "C": 0, "R": 0, "M": 0, "L": 0, "A": 0}}) \
        is None
    # 仍然只读
    snapshot = copy.deepcopy(with_counts)
    gates.legacy_replay_note(with_counts)
    assert with_counts == snapshot, "legacy_replay_note 必须只读"


def test_the_v4_thresholds_file_restates_v3_numbers_under_the_v5_wording():
    """阈值文件 v4：数值与 v3 逐字段相同（哈希自洽），只有 note/source 说清 v5。

    v2/v3 是只读历史：v4 必须能通过 ``Thresholds(**blob["thresholds"])`` 的
    哈希校验，且 ``config_hash`` 与 v3 相同（数值没动的证据）。任何一个数字
    被改都会在这里被抓到。
    """
    v3_p = ROOT / "docs" / "t14_thresholds_v3.json"
    v4_p = ROOT / "docs" / "t14_thresholds_v4.json"
    assert v4_p.is_file(), "v4 阈值文件必须存在（v3 不得原地改）"
    v3 = json.loads(v3_p.read_text(encoding="utf-8"))
    v4 = json.loads(v4_p.read_text(encoding="utf-8"))
    a, b = dict(v3["thresholds"]), dict(v4["thresholds"])
    # 字段集合与数值一字不改（source/frozen_at 是说明性字段，允许重写）
    assert set(a) == set(b), (sorted(set(a) - set(b)), sorted(set(b) - set(a)))
    for k in sorted(a):
        if k in ("source", "frozen_at"):
            continue
        assert a[k] == b[k], (k, a[k], b[k])
    # 显式点名四道候选门 + 主指标（逐字段比较之外的"读者一眼能查"清单）
    assert (b["candidate_identity_rate_min"],
            b["candidate_reference_coverage_min"],
            b["left_right_role_agreement_min"],
            b["per_scene_min_candidates"], b["line_recall_min"]) == \
        (0.60, 0.80, 0.70, 30, 0.70), b
    # 哈希自洽 + 与 v3 相同；历史文件（v3）未被改动
    t = gates.Thresholds(**b)
    assert t.config_hash == v4["config_hash"], (t.config_hash, v4["config_hash"])
    assert v4["config_hash"] == v3["config_hash"] == "a02e4db5b46feeaa", v4
    assert v3["config_hash"] == "a02e4db5b46feeaa", "v3 是只读历史，不得被改"
    # note/source 写清 v5 口径：P 帧分母、R 计下限、四档适用性、负例无阈值
    text = f"{v4['note']} {b['source']}"
    for needle in ("v5", "R/C", "P frames", "R (the identity denominator)",
                   "measured/not_applicable/unknown/unverified_labels",
                   "no threshold"):
        assert needle in text, (needle, text)
    assert "v5" in b["source"] and "v5" in v4["note"], text
