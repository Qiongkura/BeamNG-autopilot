"""实验协议 v1：指标定义、覆盖要求与标签来源资格（可哈希、可复现）。

为什么要有这个模块（方案 W0/W5）：

* 指标不能只留一个名字。"line_precision 0.30" 在**分母含不含未知区**、
  参考缺失的候选算不算进来这两种口径下是两个数，混着比就是假结论。
  所以每个指标都带：层级、单位、**分母**、有效区域要求、未知怎么记。
* 来源资格必须与真值**凭证**绑定，而不是与命令行字符串绑定：写 ``human_revision``
  不能把 agent 数据升格（方案 §6.1、验收 A1）。
* 协议本身要能进哈希：定义、覆盖要求、来源资格、阈值版本一起算，
  变了就是新协议，旧结论不能悄悄沿用（方案 §10.2）。

本模块只放**定义与纯函数**，不做 I/O、不读日志、不起进程。
"""

from __future__ import annotations

import hashlib
import json

from .labels import PAINT_SOURCE_RANK

#: 协议版本：定义/覆盖/资格任一处改动都要递增，并写进判定文件
PROTOCOL_VERSION = "t14-protocol-v5"

# ---------------------------------------------------------------------------
# 指标字典：name -> 定义
#   level: pixel | candidate | geometry | control | performance
#   unit:  ratio | ms | count | m
#   denominator: 分母口径（**必须写清**，含有效区域规则）
#   unknown_rule: 未知/缺测时怎么记（一律 UNKNOWN，不写 0）
# ---------------------------------------------------------------------------
METRIC_DEFINITIONS: dict = {
    "road_iou": {
        "level": "pixel", "unit": "ratio",
        "denominator": "road TP+FP+FN over the known area (label != 255)",
        "unknown_rule": "no prediction or no known area -> UNKNOWN (never 0)",
        "note": "道路像素代理；不能证明没有把土肩当道路（那要铺装/土肩单独指标）",
    },
    "line_iou": {
        "level": "pixel", "unit": "ratio",
        "denominator": "line TP+FP+FN over the known area (label != 255)",
        "unknown_rule": "no predicted line px or no known line px -> UNKNOWN",
        "note": "研究结论用；单独改善不能变成模型晋级（见 gates.PRIMARY_ORDER）",
    },
    "line_precision": {
        "level": "pixel", "unit": "ratio",
        "denominator": "predicted line px **inside the known area**",
        "unknown_rule": "no predicted line px -> UNKNOWN",
        "note": "未知区里的预测不参与分子分母，但要单独计数上报",
    },
    "line_recall": {
        "level": "pixel", "unit": "ratio",
        "denominator": "reference line px inside the known area",
        "unknown_rule": "no reference line px -> UNKNOWN",
        "note": "只在有效标线参考上统计；**必须报告场景覆盖**（参考缺失的帧不进分母）",
    },
    "offroad_false_ratio": {
        "level": "pixel", "unit": "ratio",
        "denominator": "predicted line px inside the known area (NOT all predicted px)",
        "unknown_rule": "no known-area prediction -> UNKNOWN",
        "note": "实测口径缺陷（G04）：旧实现用全部预测像素当分母，未知区的预测会稀释它",
    },
    "pred_unknown_line_px": {
        "level": "pixel", "unit": "count",
        "denominator": "count",
        "unknown_rule": "always a count (0 is a real value here)",
        "note": "落在未知区的预测像素：不算成功、不算假线，但必须上报",
    },
    "candidate_identity_rate": {
        "level": "candidate", "unit": "ratio",
        "denominator": "candidates that have a usable reference (projection match within the frozen tolerance)",
        "unknown_rule": "no reference for a candidate -> that candidate is UNKNOWN, not a miss",
        "note": "当前实现是投影候选匹配率；不等于左右车道身份正确，必须与角色一致率分开报",
    },
    "candidate_reference_coverage": {
        "level": "candidate", "unit": "ratio",
        "denominator": "all candidates in the evaluated frames",
        "unknown_rule": "no candidates at all -> coverage UNKNOWN",
        "note": "可测候选覆盖率：覆盖率本身要过门，不能只在容易区域算高匹配率",
    },
    "left_right_role_agreement": {
        "level": "candidate", "unit": "ratio",
        "denominator": "matched candidates with a side label on both sides",
        "unknown_rule": "missing side label -> UNKNOWN",
        "note": "左右角色一致率：与 identity_rate 分开报",
    },
    "inference_ms_p95": {
        "level": "performance", "unit": "ms",
        "denominator": "n/a (percentile over per-frame totals)",
        "unknown_rule": "no clean repeat -> UNKNOWN (polluted repeats are recorded, not averaged)",
        "note": "离线分割链（预处理+前向+后处理）；**不能**替代完整驾驶 tick deadline",
    },
    "tick_deadline_miss_ratio": {
        "level": "control", "unit": "ratio",
        "denominator": "all ticks in the run",
        "unknown_rule": "no Tech run -> UNKNOWN",
        "note": "完整驾驶闭环指标，离线实验一律未测",
    },
}

# ---------------------------------------------------------------------------
# 覆盖要求：首批人工评价集起步量（方案 §6.3；数量是起步工作量，不是统计充分性）
# ---------------------------------------------------------------------------
COVERAGE_REQUIREMENTS: tuple = (
    {"scene": "clear_paint", "what": "左/右边缘、中央线、白/黄线、实/虚线",
     "min_frames": 20, "min_groups": 2},
    {"scene": "curve_and_junction", "what": "斜线、断线、交汇、不同远近尺度",
     "min_frames": 20, "min_groups": 2},
    {"scene": "degraded_occluded", "what": "阴影、磨损、车辆遮挡、亮度变化",
     "min_frames": 20, "min_groups": 2},
    {"scene": "confusable_texture", "what": "路缘石、排水槽、护栏反光、墙面、轮胎印",
     "min_frames": 20, "min_groups": 2},
    {"scene": "no_paint_paved", "what": "能确认整片评价区域无线，保留边界信息",
     "min_frames": 20, "min_groups": 2},
    {"scene": "pavement_vs_shoulder", "what": "铺装/土肩与纯土路分别标记",
     "min_frames": 20, "min_groups": 2},
)
#: 至少 6 个完整路段组；关键场景尽量两个独立组，达不到要显式记缺口
MIN_EVAL_GROUPS = 6
#: 评价主视角；其他相机要单独声明覆盖
PRIMARY_EVAL_VIEW = "front_main"

# ---------------------------------------------------------------------------
# 来源资格：由 PAINT_SOURCE_RANK 的真实取值派生，**不由命令行字符串决定**
#   can_train: 能进训练监督；can_measure: 能用于测量（报数）；
#   can_promote: 能作为晋级评价参考
# ---------------------------------------------------------------------------
SOURCE_ELIGIBILITY: dict = {
    "verified": {"can_train": True, "can_measure": True, "can_promote": True},
    "agent": {"can_train": True, "can_measure": True, "can_promote": False},
    "pseudo": {"can_train": True, "can_measure": True, "can_promote": False},
    "unreliable": {"can_train": False, "can_measure": False, "can_promote": False},
    "absent": {"can_train": False, "can_measure": False, "can_promote": False},
}
#: 允许研究使用（训练/测量）但不能晋级的来源 rank
RESEARCH_ONLY_RANKS = tuple(sorted(
    r for r, e in SOURCE_ELIGIBILITY.items() if e["can_measure"] and not e["can_promote"]))


def eligibility(rank: str) -> dict:
    """某个 rank 的资格；未知 rank 按最保守处理（全 False）。"""
    return dict(SOURCE_ELIGIBILITY.get(str(rank), SOURCE_ELIGIBILITY["absent"]))


def can_promote(ranks) -> tuple:
    """``(ok, reasons)``：给定一组来源 rank，能否作为晋级评价参考。

    任何一个是"只许研究"的 rank 就不能晋级；rank 未知同样不能（不猜）。
    """
    bad = []
    for r in ranks or ():
        e = eligibility(r)
        if not e["can_promote"]:
            bad.append(str(r))
    if bad:
        return False, [f"paint source rank {r!r} is not admissible as a promotion "
                       f"reference" for r in sorted(set(bad))]
    return True, []


def effective_source(declared: str, credential: str | None) -> tuple:
    """``(source, notes)``：**以凭证为准**，命令行只能降低、不能抬高。

    方案 §6.1 的两条硬要求在这里落地：

    * 没有凭证时声明 ``verified`` **不认**（降级到 ``engine_annotation``，
      并在 notes 里写明原因）——"在命令行写 human_revision"不能把 agent 数据升格；
    * 有凭证时凭证优先：凭证说 ``human_revision``（真的逐帧复核过）就按 verified，
      凭证说 ``agent_revision`` 就按 agent（即使命令行写 human）。

    ``credential=None`` 表示"没有凭证"；``""`` 表示"有凭证文件但没声明来源"。
    """
    notes: list = []
    dec_rank = PAINT_SOURCE_RANK.get(str(declared), "absent")
    if credential is None:
        if dec_rank == "verified":
            notes.append(f"declared {declared!r} without a per-frame credential: "
                         f"not accepted as verified (downgraded)")
            return "engine_annotation", notes
        return str(declared), notes
    cred = str(credential)
    if not cred:
        if dec_rank == "verified":
            notes.append(f"declared {declared!r} but the credential file does "
                         f"not declare a source: not accepted as verified")
            return "engine_annotation", notes
        return str(declared), notes
    cred_rank = PAINT_SOURCE_RANK.get(cred, "absent")
    if dec_rank == "verified" and cred_rank != "verified":
        notes.append(f"declared {declared!r} but the credential says {cred!r}: "
                     f"using the credential")
        return cred, notes
    if cred_rank == "verified" and dec_rank != "verified":
        notes.append(f"credential says {cred!r} (verified) while the caller "
                     f"declared {declared!r}: using the credential")
        return cred, notes
    return cred, notes


def sources_can_promote(sources) -> tuple:
    """``(ok, reasons)``：一组**已解析**的来源（名字或 rank）能否作为晋级参考。"""
    ranks = []
    for s in sources or ():
        s = str(s)
        ranks.append(PAINT_SOURCE_RANK.get(s, s) if s in PAINT_SOURCE_RANK
                     else s)
    return can_promote(ranks)


def metric_denominator(name: str) -> str:
    """指标的分母口径；未定义的名字返回空串（调用方必须显式报"未定义"）。"""
    return str((METRIC_DEFINITIONS.get(str(name)) or {}).get("denominator") or "")


#: 空间隔离缓冲（米）：冻结值，依据前向相机 536×403 的可见范围（约 40–60 m）取保守值。
#: 改相机/分辨率必须重新标定并递增协议版本（方案 §7.3："空间缓冲依据相机可见范围
#: 制定并冻结"）。判据实现见 ``experiments/spatial.py``。
SPATIAL_BUFFER_M = 50.0
SPATIAL_BUFFER_NOTE = ("frozen from the front camera's visible range at 536x403 "
                       "(~40-60 m of road): a conservative 50 m buffer. "
                       "Changing the camera or resolution requires recalibration "
                       "and a new protocol version.")

#: 统计方法（方案 §10.3）：方法/置信水平/自由度规则/样本量估计一起进哈希——
#: "聚合方式"变了就是新协议，旧结论不能悄悄沿用。
STATISTICS: dict = {
    "interval": "paired t interval",
    "confidence": 0.95,
    "df_rule": "n_pairs - 1",
    "assumption": "paired deltas iid approximately normal",
    "t_table": "T95_TABLE (df 1..30) + normal 1.96 beyond (no scipy dependency)",
    "seeds_needed_formula": "n = (2*sd/|delta|)^2 (conservative: strict paired "
                            "t design would use (1.96+0.84)^2=7.84)",
    "raw_distribution_reported": True,
}

#: 聚合方式（方案 §10.2："分组、聚合方式…一起进哈希"）。为什么它必须进协议：
#: 同一批数字在"跨 seed 取均值"与"每个 seed 各自过门"下是两个结论——4 个 seed
#: 的 line_recall 都 0.9、第 5 个 0.3，均值 0.78 过 0.70，但那个 checkpoint
#: 根本不该部署。改聚合规则就是改验收，所以要哈希。
AGGREGATION: dict = {
    "hard_gate": "every seed and every evaluated scene must pass on its own; "
                 "the cross-seed mean may not rescue a bad seed or scene",
    "per_seed": "measured per seed; missing counts as UNKNOWN -> needs_evidence",
    "per_scene": "measured per map/source_id group; a scene with no measurement "
                 "is UNKNOWN -> needs_evidence",
    "reported_alongside": ["mean", "macro average", "worst scene", "each seed"],
}

#: 可测候选覆盖率门槛：**2026-09-26 已标定并冻结**（见
#: `docs/CANDIDATE_GATE_CALIBRATION_20260926.md`）。标定集为 136 帧权威人工真值：
#: 有标线参考场景上实测 0.890–0.916，门取 0.80（要求驱动：至少 4/5 的候选要有
#: 参考可判）；无标线场景没有参考、覆盖率天然为 0，**不计入该门**。
COVERAGE_GATE_FROZEN = True

#: 候选计数契约（v5，方案 v2 §3.3/§3.4）：**先累加整数、再算比率**。
#: 实现在 `experiments/candidate_metrics.py`，所有入口共用这一份。
CANDIDATE_COUNTING = {
    "counters": {
        "P_frames": "frames whose truth explicitly contains line pixels "
                    "(judged per frame, never per group)",
        "C": "candidates produced inside P frames (the coverage denominator)",
        "C_outside_P": "candidates in frames without line truth (reported only; "
                       "never in the coverage denominator)",
        "R": "candidates whose own side has a usable truth reference",
        "M": "matched candidates (M is a subset of R)",
        "L": "matched candidates with a judgeable left/right role (subset of M)",
        "A": "role-agreeing candidates (subset of L)",
    },
    "ratios": {"candidate_reference_coverage": "R/C",
               "candidate_identity_rate": "M/R",
               "left_right_role_agreement": "A/L"},
    "aggregation": "integer sums (micro) for the primary gate; macro may be "
                   "shown only with its unit (one unit per scene with a "
                   "denominator); never average per-directory ratios",
    "zero_denominator": "null plus a reason - never 0 and never a full score",
    "sample_floor": "per_scene_min_candidates counts R (the identity "
                    "denominator), not C: C=100 with R=1 is still insufficient",
    "legacy": {"candidate_identity_rate_legacy_match_rate":
               "the old raw match_rate (denominator = all candidates, "
               "including those without a reference); kept explicitly named, "
               "not consumed by any gate"},
}

#: 适用性取值（方案 v2 §3.4）：只有可信输入与冻结任务范围能派生 not_applicable，
#: 调用方不能自行写入以旁路硬门。
APPLICABILITY = {
    "measured": "counts are usable as a result",
    "not_applicable": "confirmed line-free scene with complete reference: the "
                      "coverage/recall gates do not apply (judge by the "
                      "negative-line/boundary task instead)",
    "unknown": "no line truth established, or no candidates to judge",
    "unverified_labels": "labels are not verified: diagnostic only",
}

#: 负例（无标线）诊断：**版本化诊断指标**，本轮不设阈值、不进晋级门。
NEGATIVE_DIAGNOSTIC = {
    "source": "experiments/negative_scenes.py (exact per-frame counters)",
    "eligibility": "verified labels, complete line supervision, non-empty "
                   "evaluation area with no UNKNOWN, and no true line pixels",
    "metrics": {
        "false_positive_frame_rate": "frames with >=1 predicted line pixel / "
                                     "eligible negative frames",
        "false_positive_pixel_fraction": "false-positive pixels / total pixels "
                                         "of those frames",
    },
    "not_comparable_with": "offroad_false_ratio (denominator = predicted line "
                           "pixels in the known area)",
    "gate": "none this round: no threshold before an independent calibration",
}

#: 候选口径门槛（v4 标定值；进 `protocol_blob()` 哈希）。
CANDIDATE_GATES = {
    "candidate_reference_coverage_min": 0.80,
    "left_right_role_agreement_min": 0.70,
    "per_scene_min_candidates": 30,
    "identity_rate_min": 0.60,
    "coverage_denominator": ("candidates produced on frames that HAVE a line "
                             "reference; scenes with no line truth are negative "
                             "scenes judged by the negative-line metric instead"),
    "calibrated_on": "2026-09-26, 136-frame authoritative human-truth set",
    "evidence": "docs/CANDIDATE_GATE_CALIBRATION_20260926.md",
}


def protocol_blob(*, thresholds: dict | None = None) -> dict:
    """协议快照：定义 + 覆盖 + 资格 + 统计 + 聚合 + 阈值版本（判定文件里存这份）。"""
    return {
        "version": PROTOCOL_VERSION,
        "metrics": METRIC_DEFINITIONS,
        "coverage_requirements": list(COVERAGE_REQUIREMENTS),
        "min_eval_groups": int(MIN_EVAL_GROUPS),
        "primary_eval_view": PRIMARY_EVAL_VIEW,
        "source_eligibility": SOURCE_ELIGIBILITY,
        "research_only_ranks": list(RESEARCH_ONLY_RANKS),
        "statistics": STATISTICS,
        "aggregation": AGGREGATION,
        "candidate_gates": CANDIDATE_GATES,
        "candidate_counting": CANDIDATE_COUNTING,
        "applicability": APPLICABILITY,
        "negative_diagnostic": NEGATIVE_DIAGNOSTIC,
        "coverage_gate_frozen": bool(COVERAGE_GATE_FROZEN),
        "spatial_buffer_m": SPATIAL_BUFFER_M,
        "spatial_buffer_note": SPATIAL_BUFFER_NOTE,
        "thresholds": dict(thresholds or {}),
    }


#: 参与哈希的字段（顺序无关，哈希前会排序）。``hash`` 本身不在内——
#: 否则重算时会把旧的哈希值也算进去。
SNAPSHOT_FIELDS = ("version", "metrics", "coverage_requirements",
                   "min_eval_groups", "primary_eval_view", "source_eligibility",
                   "research_only_ranks", "statistics", "aggregation",
                   "candidate_gates", "candidate_counting", "applicability",
                   "negative_diagnostic", "coverage_gate_frozen",
                   "spatial_buffer_m", "spatial_buffer_note", "thresholds")


def snapshot_hash(snap: dict) -> str:
    """对**判定文件里记下来的**协议快照重算哈希。

    为什么需要：判定文件必须能自查"我这份快照有没有被改过、是不是按当前口径
    写的"。只比对 ``snap["hash"]`` 等于当前哈希不够——手改快照内容而留旧哈希
    就查不出来（方案 §10.3：判定文件保存完整协议快照及哈希）。
    """
    view = {k: snap.get(k) for k in SNAPSHOT_FIELDS}
    return hashlib.sha256(
        json.dumps(view, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:16]


def verify_snapshot(snap: dict) -> dict:
    """校验一份记录下来的协议快照：内容自洽 + 与当前口径是否一致。"""
    got = snapshot_hash(snap)
    recorded = snap.get("hash")
    cur = protocol_hash(thresholds=snap.get("thresholds") or None)
    reasons = []
    if recorded is None:
        reasons.append("snapshot has no hash: cannot tell which protocol it was "
                       "written under")
    elif str(recorded) != got:
        reasons.append(f"snapshot content does not match its recorded hash "
                       f"({got}) - the decision file was edited or truncated")
    return {"ok": not reasons, "recorded_hash": recorded,
            "recomputed_hash": got, "current_hash": cur,
            "matches_current_protocol": bool(recorded == cur),
            "reasons": reasons}


def protocol_hash(*, thresholds: dict | None = None) -> str:
    """协议哈希：定义/覆盖/资格/聚合/阈值任一变化都会变。"""
    return snapshot_hash(protocol_blob(thresholds=thresholds))
