"""候选统计：**先累加整数计数，再算比率**（方案 v2 §3.3）。

为什么单独建库：同一批逐帧数据必须只有一个实现。实测缺口（2026-09-26）：入口
`identity_metrics` 把各目录的**比率**做平均、并用"全部候选"当分母，于是确定性输入
（有参考场景 C=10/R=8/M=6 + 无参考场景 C=10）算出覆盖率 0.40、身份率 0.60，
而正确口径是 **0.80 / 0.75**。修法不是改公式，而是把计数集中到这里：
探针给逐帧整数 → 这里累加 → 比率只在最后算一次。

冻结定义（与协议一起版本化）：

| 计数 | 含义 |
| --- | --- |
| ``P_frames`` | 真值明确存在漆线的合格帧数（**逐帧判断**，不因同组另一帧有线而纳入） |
| ``C`` | P 帧内由实际感知链产生的、纳入评价的漆线候选数 |
| ``R`` | C 中**自身所在侧有可用真值参考**的候选数 |
| ``M`` | R 中满足冻结匹配条件的候选数（**M ⊆ R**，无参考候选不得混入分子） |
| ``L`` | M 中预测与参考都有可判左右角色的候选数 |
| ``A`` | L 中左右角色一致的候选数 |

公式：覆盖率 = R/C；身份匹配率 = M/R；角色一致率 = A/L。
分母为 0 时比率为 ``None`` 并附原因；候选数量是**整数相加**，不做比率平均。
"""

from __future__ import annotations

#: 全部计数键（顺序固定，便于报告与测试）。
#: ``C`` 只统计 **P 帧内**的候选（覆盖率分母）；P 帧之外的候选记
#: ``C_outside_P``（例如人确认无线的场景），**只上报、不进覆盖率分母**——
#: 否则"没有参考可判"会被算成模型的覆盖率问题（方案 v2 §3.4/§3.5）。
COUNTERS = ("P_frames", "C", "R", "M", "L", "A", "C_outside_P")

#: 比率的分子/分母（覆盖率、身份、角色）
RATIO_SPECS = {
    "candidate_reference_coverage": ("R", "C"),
    "candidate_identity_rate": ("M", "R"),
    "left_right_role_agreement": ("A", "L"),
}


def empty() -> dict:
    return {k: 0 for k in COUNTERS}


def accumulate(acc: dict, counts: dict) -> dict:
    """把一帧（或一个目录/场景）的整数计数加进累加器。只加整数，不算比率。"""
    for k in COUNTERS:
        v = (counts or {}).get(k)
        if v is None:
            continue
        acc[k] = int(acc.get(k, 0)) + int(v)
    return acc


def totals(frame_counts: list) -> dict:
    """逐帧/逐目录计数列表 -> 整数总计。"""
    acc = empty()
    for c in frame_counts or []:
        accumulate(acc, c)
    return acc


def ratios(acc: dict) -> dict:
    """由整数总计算比率；分母 0 -> None + 原因（不写 0、不写满分）。"""
    out = {}
    for name, (num, den) in RATIO_SPECS.items():
        n, d = int((acc or {}).get(num, 0)), int((acc or {}).get(den, 0))
        out[name] = None if d == 0 else round(n / d, 4)
        out[f"{name}_numerator"] = n
        out[f"{name}_denominator"] = d
        if d == 0:
            out[f"{name}_missing"] = (
                f"denominator {den}=0: nothing to divide by "
                f"({'no candidates' if den == 'C' else 'no reference-matched candidates' if den == 'R' else 'no role-comparable candidates'})")
    return out


#: 适用性取值（方案 v2 §3.4）：measured / not_applicable / unknown /
#: unverified_labels。**只有可信输入与冻结任务范围**能派生出 not_applicable，
#: 调用方不能自行写入以旁路硬门。
APPLICABILITY = ("measured", "not_applicable", "unknown", "unverified_labels")


def applicability(acc: dict, *, has_line_truth: bool | None,
                  label_rank: str = "") -> dict:
    """这批计数的适用性：能否用来判覆盖率/身份率。

    * 帧**没有**标线真值（人确认无线）-> ``not_applicable``：无线场景不进入
      有线参考覆盖门（它们由负例/边界任务评价），但**也不是通过**；
    * 标线真值存在但一个候选都没有 -> ``unknown``（没东西可判，不是 0 分）；
    * 标签档位不是 verified -> ``unverified_labels``（计数可诊断，不得当结论）；
    * 其余 -> ``measured``。
    """
    if has_line_truth is False:
        return {"status": "not_applicable",
                "why": "no line truth in this set (confirmed line-free): the "
                       "candidate coverage/identity gates do not apply here; "
                       "judge it by the negative-line/boundary task instead"}
    if has_line_truth is None:
        return {"status": "unknown",
                "why": "line truth could not be established for this set"}
    if label_rank and label_rank != "verified":
        return {"status": "unverified_labels",
                "why": f"label rank {label_rank!r} is not verified: counts are "
                       "diagnostic only and must not be quoted as a result"}
    if int(acc.get("C", 0)) == 0:
        return {"status": "unknown",
                "why": "line truth exists but the perception produced no "
                       "candidates: nothing to measure (this is not a 0 score)"}
    return {"status": "measured", "why": ""}


def scene_report(per_scene: dict, *, min_candidates: int | None) -> dict:
    """逐场景：整数计数 + 比率 + 样本量判定（下限按 **R** 计，不是 C）。

    方案 v2 §S3.2：「不能用 30 条总候选、其中只有 1 条可判，冒充足够身份样本」——
    所以样本量下限检查的是身份率的**实际分母 R**。``L`` 单独报（角色分母），
    L=0 记 UNKNOWN，非零但很小标低样本。
    """
    out = {"scenes": {}, "violations": [], "missing": [], "low_sample": []}
    for g in sorted(per_scene or {}):
        acc = per_scene[g]
        r = ratios(acc)
        R, L, C = (int(acc.get("R", 0)), int(acc.get("L", 0)),
                   int(acc.get("C", 0)))
        entry = {**{k: int(acc.get(k, 0)) for k in COUNTERS}, **r,
                 "role_denominator_L": L}
        if min_candidates is not None:
            if 0 < R < int(min_candidates):
                entry["sample"] = "insufficient"
                out["low_sample"].append(
                    f"scene {g}: R={R} < {int(min_candidates)} usable references "
                    f"(C={C}): too few judgeable candidates for a per-scene "
                    "identity verdict")
            elif R == 0:
                entry["sample"] = "none"
                out["missing"].append(
                    f"scene {g}: R=0 (C={C}) - no candidate has a usable "
                    "reference: identity is UNKNOWN here, not 0")
            else:
                entry["sample"] = "ok"
        if L == 0 and R > 0:
            entry["role_sample"] = "none"
            out["missing"].append(
                f"scene {g}: L=0 with R={R} - no role-comparable candidate: "
                "role agreement is UNKNOWN here")
        elif 0 < L < 5:
            entry["role_sample"] = "low"
            out["low_sample"].append(
                f"scene {g}: L={L} is a very small role denominator: report the "
                "number, do not claim high confidence")
        out["scenes"][g] = entry
    return out


def micro_macro(per_scene: dict) -> dict:
    """micro = 计数汇总（本轮主口径）；macro = 逐场景比率的等权平均（需注明单位）。

    方案 v2 §3.3：主门用 micro；macro 若展示必须写清"每场景一个单位"，
    不能让读者把它当成"每帧/每候选一个单位"。
    """
    acc = empty()
    for c in (per_scene or {}).values():
        accumulate(acc, c)
    macro_vals: dict = {}
    for name, (num, den) in RATIO_SPECS.items():
        # 用**未四舍五入**的逐场景比率求平均，最后只舍入一次：先舍入再平均会
        # 叠加误差（实测 1/1 与 1/9 的平均会算出 0.5555 而不是 0.5556）。
        vals = []
        for c in (per_scene or {}).values():
            d = int((c or {}).get(den, 0))
            if d:
                vals.append(int((c or {}).get(num, 0)) / d)
        macro_vals[name] = (None if not vals
                            else round(sum(vals) / len(vals), 4))
        macro_vals[f"{name}_units"] = len(vals)
    return {"micro": {**{k: int(acc.get(k, 0)) for k in COUNTERS},
                      **ratios(acc)},
            "macro": macro_vals,
            "macro_note": ("macro averages per-scene ratios, one unit per "
                           "scene that had a denominator; it is not a per-frame "
                           "or per-candidate rate")}
