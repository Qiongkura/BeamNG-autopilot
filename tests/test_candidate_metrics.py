"""候选计数与聚合的确定性验收（方案 v2 §3.3/§3.4/§5 的 T03–T07/T09）。

为什么单独一个文件：计数口径是这一轮的核心修复——实测缺口是"入口把各目录的
**比率**取平均、并用全部候选当分母"，于是确定性输入（有参考场景 C=10/R=8/M=6
+ 无参考场景 C=10）算出覆盖率 0.40，而正确口径是 0.80。测试必须能直接抓住
"用比率平均""分母含无参考候选""用 C 冒充样本量"这三种退化。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

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


def test_attribution_tool_scopes_candidates_to_p_frames():
    """归因脚本的候选口径必须与协议一致：无线真值的帧记 `C_outside_P`。

    实测踩到（2026-09-30）：两个 P_frames=0 的开发场景贡献了 477/531 个
    "无参考候选"，于是"模型到处乱画"的结论是从**不判线**的帧里读出来的
    （协议口径 C=385，脚本却报 1021）。修法是按帧适用性分流：不在 P 的帧
    只累加 `candidates_outside_p` 并 `continue`，不进 C/R/M。

    这里是源码级回归守卫（脚本要跑真模型才能端到端测）：分流语句与计数键
    必须同时在，且 `continue` 必须在累加之后。
    """
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "scripts"
           / "m5_candidate_failure_attribution.py").read_text(encoding="utf-8")
    assert "candidates_outside_p" in src, "必须单独记 C_outside_P"
    assert src.count("candidates_outside_p") >= 4, \
        "totals/agg/累加/打印 四处都要有（否则口径只改了一半）"
    i = src.index('if not has_truth:')
    j = src.index('candidates_outside_p', i)
    k = src.index('continue', j)
    assert i < j < k, "分流必须先累加 C_outside_P 再 continue"
    # 比率口径要写明只含 P 帧
    assert '"scope"' in src and "frames with line truth only" in src


def test_arm_gate_measure_summarizes_rows():
    """臂级门表：只累加整数计数再算比率；缺测的掩码指标记 None 不记 0。

    用途（T16 §20）：协议/提取器换版后**旧判定不可比**，要把已有 checkpoint
    在同一口径下重测——本聚合函数是那个入口的纯逻辑部分。
    """
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "m5_arm_gate_measure", root / "scripts" / "m5_arm_gate_measure.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["m5_arm_gate_measure"] = m
    spec.loader.exec_module(m)
    rows = [
        {"counts": {"P_frames": 1, "C": 4, "R": 4, "M": 2, "L": 2, "A": 1,
                    "C_outside_P": 3}, "recall": 0.5, "precision": 0.25,
         "iou": 0.2, "candidates_off_road": 2, "n_candidates": 4,
         "line_candidate_gate": {"line_candidate_merge": {"merged": 1,
                                                          "groups": 1}}},
        {"counts": {"P_frames": 1, "C": 6, "R": 5, "M": 3, "L": 3, "A": 2,
                    "C_outside_P": 0}, "recall": 0.7, "precision": 0.5,
         "iou": 0.4, "candidates_off_road": 1, "n_candidates": 6},
        # 掩码指标缺测（None）：不进均值，样本数分开记
        {"counts": {"P_frames": 0, "C": 0, "R": 0, "M": 0, "L": 0, "A": 0,
                    "C_outside_P": 1}, "candidates_off_road": 0,
         "n_candidates": 0},
    ]
    s = m.summarize_rows(rows)
    assert s["C"] == 10 and s["R"] == 9 and s["M"] == 5 and s["A"] == 3
    assert s["C_outside_P"] == 4
    assert s["candidate_identity_rate"] == round(5 / 9, 4)
    assert s["left_right_role_agreement"] == round(3 / 5, 4)
    assert s["mask_recall_n"] == 2 and s["mask_recall_mean"] == 0.6
    assert s["mask_precision_mean"] == 0.375
    assert s["off_road_frac"] == round(3 / 10, 4)
    assert s["merge_merged"] == 1 and s["merge_groups"] == 1
    assert s["P_frames"] == 2 and s["frames"] == 3
    # 全空：比率 None（不写 0）
    e = m.summarize_rows([])
    assert e["candidate_identity_rate"] is None and e["C"] == 0


def test_arm_gate_measure_reports_per_scene_counts(tmp_path, monkeypatch):
    """逐场景明细必须随臂输出：冻结口径的 per_scene_min_candidates=30 在 R 上，
    池化均值会掩盖"某个场景样本不足"（R3 设计 §7 教训 1）。逐场景计数之和
    必须等于池化计数（同一批行、两种聚合，不允许漏行）。
    """
    import importlib.util
    from pathlib import Path as _P
    root = _P(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "m5_arm_gate_measure_ps", root / "scripts" / "m5_arm_gate_measure.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["m5_arm_gate_measure_ps"] = m
    spec.loader.exec_module(m)
    for name in ("pkg_a", "pkg_b"):
        d = tmp_path / name / "front_main"
        d.mkdir(parents=True)
        (tmp_path / name / "meta.json").write_text("{}", encoding="utf-8")
    rows_by_dir = {
        "pkg_a": [{"counts": {"P_frames": 1, "C": 4, "R": 3, "M": 1, "L": 1,
                              "A": 1, "C_outside_P": 0}}],
        "pkg_b": [{"counts": {"P_frames": 1, "C": 2, "R": 2, "M": 2, "L": 2,
                              "A": 1, "C_outside_P": 0}}]}

    class FakeProbe:
        @staticmethod
        def probe(d, meta, view=None, model_path=None, device=None):
            return {"rows": rows_by_dir[_P(d).parent.name]}

    monkeypatch.setattr(m, "_probe", lambda: FakeProbe)
    out = tmp_path / "gate.json"
    ck = tmp_path / "ck.pt"
    ck.write_bytes(b"x")
    monkeypatch.setattr(sys, "argv", [
        "m5_arm_gate_measure.py", "--arm", f"a={ck}",
        "--dev-runs", str(tmp_path / "pkg_a" / "front_main"),
        str(tmp_path / "pkg_b" / "front_main"), "--out", str(out)])
    assert m.main() == 0
    import json as _json
    blob = _json.loads(out.read_text(encoding="utf-8"))
    ps = blob["arms"]["a"]["per_scene"]
    assert set(ps) == {"pkg_a/front_main", "pkg_b/front_main"}
    assert ps["pkg_a/front_main"]["R"] == 3 and ps["pkg_b/front_main"]["R"] == 2
    assert (ps["pkg_a/front_main"]["R"] + ps["pkg_b/front_main"]["R"]
            == blob["arms"]["a"]["R"])


def test_arm_gate_measure_decomposes_identity_by_role():
    """身份率 ~0.5 时必须能分解到角色/认证实例：弥散噪声 vs 某条线系统性不匹配。

    按模型自报角色（by_role）与按参考角色（by_reference_role，即认证实例）
    两个视角都要有；R 为空时 identity=None（不写 0）。
    """
    import importlib.util
    from pathlib import Path as _P
    root = _P(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "m5_arm_gate_measure_role", root / "scripts" / "m5_arm_gate_measure.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["m5_arm_gate_measure_role"] = m
    spec.loader.exec_module(m)
    rows = [{"counts": {"P_frames": 1, "C": 4, "R": 4, "M": 2, "L": 2, "A": 1,
                        "C_outside_P": 0},
             "candidates": [
                 {"role": "near_left", "engine_role": "near_left",
                  "reference_available": True, "matched": True},
                 {"role": "near_left", "engine_role": "near_left",
                  "reference_available": True, "matched": False},
                 {"role": "near_right", "engine_role": "near_right",
                  "reference_available": True, "matched": False},
                 {"role": "near_right", "engine_role": "near_right",
                  "reference_available": True, "matched": False},
                 {"role": "far_left", "reference_available": False,
                  "matched": False}]}]
    s = m.summarize_rows(rows)
    assert s["by_role"]["near_left"] == {"C": 2, "R": 2, "M": 1,
                                         "identity": 0.5}
    assert s["by_role"]["far_left"] == {"C": 1, "R": 0, "M": 0,
                                        "identity": None}
    # 匹配对上的角色混淆（model_role × engine_role）与逐实例被匹配数：
    # 1 个 near_left 候选匹配到了 near_left 参考 -> 混淆 {near_left:{near_left:1}}
    assert s["role_confusion"] == {"near_left": {"near_left": 1}}
    assert s["reference_instances_matched"] == {"near_left": 1}


def test_dose_response_summarize_decision():
    """剂量-效应表：从判定文件抽出剂量与配对差值；旧判定缺负例帧数时按 8 帧/包
    估算并**标明 estimated**（估算值不得与实测值混着引用）。"""
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "m5_dose_response_table", root / "scripts" / "m5_dose_response_table.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["m5_dose_response_table"] = m
    spec.loader.exec_module(m)
    blob = {
        "candidate_id": "x", "candidate_runs": ["b1", "g1", "n1", "n2"],
        "baseline_runs": ["b1"], "all_at_plateau": False,
        "round_outcome": "meaningful_no_gain",
        "synthetic_line_dose": {"synthetic_line_frames": 8, "negative_frames": 16,
                                "share": 0.1, "negative_frac": 0.2,
                                "level": "ok"},
        "pairings": {"candidate_identity_rate": {
            "champion": [0.4, 0.5], "candidate": [0.45, 0.55],
            "mean_delta": 0.05, "ci95_halfwidth": 0.01,
            "verdict": "candidate_better"}},
    }
    r = m.summarize_decision(blob)
    assert r["seeds"] == 2 and r["added_runs"] == 3
    assert r["negative_x"] == 2.0 and r["dose_source"] == "measured"
    assert r["candidate_identity_rate"]["mean_delta"] == 0.05
    assert r["candidate_identity_rate"]["champion_mean"] == 0.45
    assert r["candidate_identity_rate"]["candidate_mean"] == 0.5
    # 旧判定：没有 synthetic_line_dose -> 负例帧数按 8 帧/包估，标明 estimated
    old = {"candidate_id": "y", "candidate_runs": ["b1", "g1", "n1", "n2"],
           "baseline_runs": ["b1"], "pairings": {}}
    r2 = m.summarize_decision(old)
    assert r2["negative_frames"] == 24 and r2["dose_source"] == "estimated"
    assert r2["negative_x"] is None      # 线帧未知 -> 倍数未知，不猜


def test_arm_gate_measure_surface_scope_dual():
    """双口径对照（§10.3 提案）：表面口径只排除"标签说在背景上"的候选，
    匹配上的候选一条都不该丢（matched_lost 必须为 0 才能说"无召回代价"）。"""
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "m5_arm_gate_measure", root / "scripts" / "m5_arm_gate_measure.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["m5_arm_gate_measure"] = m
    spec.loader.exec_module(m)
    rows = [{
        "counts": {"P_frames": 1, "C": 4, "R": 4, "M": 2, "L": 2, "A": 1,
                   "C_outside_P": 0},
        "n_candidates": 4, "candidates_off_road": 2,
        "candidates": [
            {"matched": True, "reference_available": True,
             "off_road_frac": 0.0, "role_agrees": True},
            {"matched": True, "reference_available": True,
             "off_road_frac": 0.0, "role_agrees": False},
            # 未匹配但落在标签的背景类上（"线状结构"）-> 表面口径排除
            {"matched": False, "reference_available": True,
             "off_road_frac": 1.0, "role_agrees": None},
            {"matched": False, "reference_available": True,
             "off_road_frac": 0.7, "role_agrees": None},
        ],
    }]
    s = m.summarize_rows(rows)
    assert s["candidate_identity_rate"] == 0.5          # 现有口径 2/4
    ss = s["surface_scope"]
    assert ss["R"] == 2 and ss["M"] == 2 and ss["matched_lost"] == 0
    assert ss["excluded_R"] == 2 and ss["excluded_C"] == 2
    assert ss["candidate_identity_rate"] == 1.0         # 表面口径 2/2
    assert ss["left_right_role_agreement"] == 0.5       # 角色不变（L/A 只数匹配上的）
    # "横向近、像素在背景上"的匹配：表面口径排除它（M 也减），代价可见
    rows2 = [{
        "counts": {"P_frames": 1, "C": 2, "R": 2, "M": 1, "L": 1, "A": 1,
                   "C_outside_P": 0},
        "n_candidates": 2, "candidates_off_road": 1,
        "candidates": [
            {"matched": True, "reference_available": True,
             "off_road_frac": 0.0, "role_agrees": True},
            {"matched": True, "reference_available": True,
             "off_road_frac": 1.0, "role_agrees": True},
        ],
    }]
    s2 = m.summarize_rows(rows2)
    assert s2["candidate_identity_rate"] == 0.5         # 现口径 1/2
    ss2 = s2["surface_scope"]
    assert ss2["M"] == 1 and ss2["R"] == 1 and ss2["matched_lost"] == 1
    assert ss2["candidate_identity_rate"] == 1.0


def test_surface_scope_counts_and_cost():
    """表面范围口径（§10.3 提案，只报不判）：计数、比率与**召回代价**一起给。

    判据用标签自己的类（`off_road_frac >= 0.5` = 标签说不在路面/漆线上）；
    `matched_lost`（横向匹配但像素在背景上）必须可见——提案要求收益与代价并列，
    不许只报收益。
    """
    from beamng_autopilot.experiments.candidate_metrics import (
        SURFACE_OFF_ROAD_MAX, surface_scope)
    rows = [{"candidates": [
        {"matched": True, "reference_available": True, "off_road_frac": 0.0,
         "role_agrees": True},
        {"matched": True, "reference_available": True, "off_road_frac": 0.8,
         "role_agrees": True},          # 擦边匹配：表面口径的召回代价
        {"matched": False, "reference_available": True, "off_road_frac": 1.0},
        {"matched": False, "reference_available": False, "off_road_frac": 0.0},
    ]}]
    s = surface_scope(rows)
    assert SURFACE_OFF_ROAD_MAX == 0.5
    assert s["reported_only"] is True
    # C=2（第 1、4 个在表面上）、R=1（只有第 1 个既在表面上又有参考）、
    # M/L/A=1；第 2 个是"擦边匹配"-> matched_lost=1；第 3、4 个被排除
    assert s["counts"]["C"] == 2 and s["counts"]["R"] == 1
    assert s["counts"]["M"] == 1 and s["counts"]["L"] == 1 and s["counts"]["A"] == 1
    # 被排除的 2 条里，两条都"该侧有参考"（第 2 条擦边匹配、第 3 条纯假线）
    assert s["excluded_C"] == 2 and s["excluded_R"] == 2
    assert s["matched_lost"] == 1
    assert s["candidate_identity_rate"] == 1.0
    # 空输入：比率 None（不写 0）
    e = surface_scope([])
    assert e["candidate_identity_rate"] is None and e["matched_lost"] == 0


def test_frame_structure_density_is_label_only():
    """R3 场景筛选判据：只读标签的"近场背景类结构密度"（纯函数，模型无关）。

    干净路段（线之间只有路面）≈ 0；线外有墙/护栏/背景的帧明显更高；
    全帧无线时 None（不猜、不写 0）。
    """
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "m5_scene_structure_density",
        root / "scripts" / "m5_scene_structure_density.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["m5_scene_structure_density"] = m
    spec.loader.exec_module(m)
    # 干净：两行近场（每行 6 列）在两条线之间全是路面（1）
    clean = np.ones((4, 10), dtype=np.uint8) * 1
    clean[2:, 2] = 2
    clean[2:, 7] = 2
    s = m.frame_structure_density(clean)
    assert s["density"] == 0.0 and s["rows_with_line"] == 2
    # 结构：线之间的路面被背景（0）替换
    dirty = clean.copy()
    dirty[2:, 4:6] = 0
    d = m.frame_structure_density(dirty)
    assert d["density"] is not None and d["density"] > 0.2
    assert d["structure_px"] == 4          # 2 行 × 2 列
    # 全帧无线：None（无线帧由负例通道管）
    nol = np.ones((4, 10), dtype=np.uint8)
    assert m.frame_structure_density(nol)["density"] is None

def test_candidate_scope_scan_transforms():
    """候选集扫描的纯变换：lateral 只丢超限（可按通道过滤）；parallel 同侧合并
    且 matched 优先、否则保留 |lat| 小者；跨侧不合并（不同线的重复不可能跨侧）。
    """
    import importlib.util
    from pathlib import Path as _P
    root = _P(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "m5_candidate_scope_scan",
        root / "scripts" / "m5_candidate_scope_scan.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["m5_candidate_scope_scan"] = m
    spec.loader.exec_module(m)
    cands = [
        {"lat_m": 1.6, "matched": True, "reference_available": True},
        {"lat_m": 1.7, "matched": False, "reference_available": True},
        {"lat_m": -6.0, "matched": False, "reference_available": True,
         "learned_frac": None},
        {"lat_m": -6.2, "matched": True, "reference_available": True,
         "learned_frac": 0.9},
        {"lat_m": 7.5, "matched": False, "reference_available": True,
         "learned_frac": 0.5},
    ]
    # lateral：全部通道（-6.0 与 -6.2 都超 5.0，都丢）
    out = m.apply_scope(cands, mode="lateral", param=5.0)
    assert [c["lat_m"] for c in out] == [1.6, 1.7]
    # lateral：只丢经典（-6.0 经典丢；-6.2/7.5 是学习通道，留）
    out = m.apply_scope(cands, mode="lateral", param=5.0, only_classic=True)
    assert [c["lat_m"] for c in out] == [1.6, 1.7, -6.2, 7.5]
    # parallel：1.6/1.7 合并保留 matched 的 1.6；-6.0/-6.2 合并保留 matched 的 -6.2
    out = m.apply_scope(cands, mode="parallel", param=0.5)
    lats = sorted(c["lat_m"] for c in out)
    assert lats == [-6.2, 1.6, 7.5], lats
    assert sum(c.get("_merged") or 0 for c in out) == 2
    # 跨侧不合并（+7.5 与 -6.2 差 13.7 > 0.5 本来就不合；构造近距跨侧用例）
    cross = [{"lat_m": 0.2, "matched": False}, {"lat_m": -0.3, "matched": False}]
    out = m.apply_scope(cross, mode="parallel", param=0.8)
    assert len(out) == 2
    # 都未匹配时保留 |lat| 小者
    both = [{"lat_m": -1.2, "matched": False}, {"lat_m": -1.5, "matched": False}]
    out = m.apply_scope(both, mode="parallel", param=0.5)
    assert len(out) == 1 and out[0]["lat_m"] == -1.2
