"""逐类别标签质量：road / paint / pavement 分开判，缺真值不当负例。

方案（§"先纠正数据前提"）实测的事实：Tech annotation 在本图提供密集
**路面**类别，但可见漆画线常被渲染成 ``ASPHALT``（一帧有上千白线像素而
``SOLID_LINE`` 只有 3 px）。因此"有 annotation"不能推断"标线标签可靠"，
三个通道必须**分别**给有效标记：

===========  ==========================================================
road_valid   路面类别可用（采集器已保证，仍需检查 palette 质量与非空）
paint_valid  标线类别可用：需要人工修订或单独验证的模拟器真值
pavement_valid  铺装/土肩区分可用（"road 类"不等于"可行驶铺装"）
===========  ==========================================================

**缺标线真值时的纪律**（方案 §1）：可以贡献可信路面损失，但**不得**把
未标出的可见漆线当成负例——所以 ``line`` 通道在 ``paint_valid=False`` 的
帧上被换成 ignore(255)，任何损失项与指标都必须尊重它。本模块只产出标记
与掩码，由 ``beamng_autopilot.vision`` 的损失/指标消费。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

#: 训练契约的类别编号（与 vision.segmentation.N_CLASSES 一致）。
CLS_BACKGROUND, CLS_ROAD, CLS_LINE = 0, 1, 2
IGNORE = 255

#: 漆线真值来源的强度分级。``pseudo`` 只能进独立研究臂（方案点名
#: ``m5_inject_yellow_labels.py`` 的 HSV 注入属于伪标签）。
PAINT_SOURCE_RANK = {
    "human_revision": "verified",     # 人工逐帧修订
    "engine_verified": "verified",    # 单独验证过的模拟器真值
    # 引擎标注的漆线类：**存在但不完整**。2026-09-25 更正：此前写的
    # "漆线被画成 ASPHALT / line 类为空"在这批采集上不成立——逐帧目视 + 像素统计
    # （6 个采集 × 8 帧）：引擎线像素覆盖了 RGB 漆线候选的 ~0.61（precision ~0.68）。
    # 所以它不能当**门槛真值**（未标注的漆线会被算成假阳），但可以当**弱监督**
    # 用于研究臂：先让模型学会"产出标线"（当前 road-only 配方完全不产出），
    # 之后才谈得上压线判断。
    "engine_annotation": "unreliable",
    "engine_annotation_partial": "pseudo",  # 弱监督（研究臂，不得晋级）
    # agent 逐帧核对式标注（2026-09-25，scripts/m5_line_truth_agent.py）：
    # 机器提议（引擎标线 ∪ 细长亮条，宽亮带判背景，其余碎亮斑写 255=ignore）
    # + **逐帧目视复核**（32 帧 4 视角全部看过，抓到并修掉 pillar_right 的一处假阳）。
    # 它比引擎弱标签完整（补回虚线中线与右边缘线），但仍是**机器画的**：
    # 可训练、可测，晋级仍需人确认 → valid=False, usable=True。
    "agent_revision": "agent",
    "pseudo": "pseudo",               # HSV 注入等，不得当真值
    "none": "absent",
}


@dataclass
class ClassQuality:
    """一个通道的判定：**可学性**（usable）+ **可判定性**（valid）+ 原因 + 计数。

    这两件事必须分开，实测教训（2026-09-25）：引擎标注的漆线类**存在但不完整**
    （覆盖 RGB 漆线候选 ~0.61），它不能当门槛真值（`valid=False`：未标注的漆线会
    被算成假阳），但完全可以当**弱监督**让模型先学会产出标线（`usable=True`）。
    只用一个 `valid` 会把"不能判"误当成"不能学"，等于白白丢掉唯一的标线监督。
    默认 `usable == valid`，保持既有语义不变。
    """

    valid: bool
    reason: str
    pixels: int = 0
    usable: bool | None = None        # None = 跟 valid 一致（老调用方语义不变）
    #: 标签来源档位（verified/agent/pseudo/unreliable/absent）。看板与报告要按
    #: 档位分别计数（"640 生成帧"不等于"640 人工真值"），从散文 reason 里抠
    #: 档位是脆的，所以直接记下来。
    rank: str = ""

    def __post_init__(self) -> None:
        if self.usable is None:
            self.usable = bool(self.valid)

    def as_dict(self) -> dict:
        return {"valid": bool(self.valid), "reason": self.reason,
                "pixels": int(self.pixels), "usable": bool(self.usable),
                "rank": str(self.rank or "")}


@dataclass
class LabelAudit:
    road: ClassQuality
    paint: ClassQuality
    pavement: ClassQuality
    unknown_reason: str = ""
    line_masked_px: int = 0
    frame_unknown_frac: float = 0.0
    label_sha256_16: str = ""
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"road": self.road.as_dict(), "paint": self.paint.as_dict(),
                "pavement": self.pavement.as_dict(),
                "unknown_reason": self.unknown_reason,
                "line_masked_px": int(self.line_masked_px),
                "frame_unknown_frac": round(float(self.frame_unknown_frac), 5),
                "label_sha256_16": self.label_sha256_16,
                "notes": list(self.notes)}

    @property
    def trainable(self) -> bool:
        """能不能进有监督训练：至少一个通道有可用真值。"""
        return self.road.valid or self.paint.valid


def label_sha16(label: np.ndarray) -> str:
    a = np.ascontiguousarray(np.asarray(label, dtype=np.uint8))
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


def audit_label(label, *, paint_source: str = "engine_annotation",
                palette_ok: bool = True, road_min_px: int = 200,
                has_rgb: bool = True) -> LabelAudit:
    """逐帧判三个通道的可用性。

    ``paint_source`` 来自采集元数据的 ``label_source`` 或人工修订记录；
    ``engine_annotation``（本图默认）被判为 **unreliable**：可见漆线可能
    完全没有标注，此时 ``paint_valid=False`` 并把 line 通道屏蔽。
    """
    lab = np.asarray(label)
    if lab.ndim != 2:
        raise ValueError(f"label must be 2-D, got shape {lab.shape}")
    n_px = int(lab.size)
    road_px = int((lab == CLS_ROAD).sum())
    line_px = int((lab == CLS_LINE).sum())
    unknown_px = int((lab == IGNORE).sum())

    notes: list[str] = []
    if not has_rgb:
        notes.append("no RGB for this frame: it cannot be a training sample")
    if not palette_ok:
        notes.append("palette quality flagged by the collector audit")

    rank = PAINT_SOURCE_RANK.get(str(paint_source), "absent")
    if rank == "verified":
        paint = ClassQuality(True, f"paint truth from {paint_source}",
                             line_px, rank=rank)
    elif rank == "pseudo":
        # 弱监督：可以学（usable=True），但不能当门槛真值（valid=False）
        paint = ClassQuality(
            False,
            f"weak/pseudo paint labels ({paint_source}): trainable as weak "
            "supervision, never a gate", line_px, usable=True, rank=rank)
    elif rank == "agent":
        # agent 逐帧核对式标注：可训练、可测（报告里写清来源），但要晋级仍需人确认
        paint = ClassQuality(
            False,
            f"agent-reviewed paint labels ({paint_source}): machine-drawn with "
            "a per-frame visual pass - trainable and measurable, but promotion "
            "still needs human sign-off", line_px, usable=True, rank=rank)
    elif line_px == 0 and rank == "unreliable":
        paint = ClassQuality(False,
                             "engine annotation provides no reliable paint "
                             "class; 0 line px is NOT evidence of no paint",
                             0, rank=rank)
    else:
        paint = ClassQuality(
            False,
            "engine annotation's paint class exists but is incomplete "
            "(measured 2026-09-25: covers ~0.61 of RGB paint candidates, "
            "precision ~0.68) - unannotated paint would score as false "
            "positives, so it is not admissible as gating truth", line_px,
            rank=rank)

    road = ClassQuality(road_px >= int(road_min_px) and has_rgb,
                        "" if road_px >= int(road_min_px) else
                        f"road px {road_px} < {int(road_min_px)}", road_px)
    # 铺装/土肩需要独立确认：默认不声称可用，直到有单独验证的标签来源。
    pavement = ClassQuality(False,
                            "pavement vs shoulder is not separable from the "
                            "road class alone; needs its own verified labels",
                            road_px)

    unknown_reason = ""
    if not road.valid:
        unknown_reason = road.reason
    elif not paint.valid:
        unknown_reason = f"paint truth unusable: {paint.reason}"

    return LabelAudit(road=road, paint=paint, pavement=pavement,
                      unknown_reason=unknown_reason,
                      line_masked_px=line_px, frame_unknown_frac=unknown_px / max(1, n_px),
                      label_sha256_16=label_sha16(lab), notes=notes)


def line_channel_mask(label, audit: LabelAudit) -> np.ndarray:
    """该帧的 ``line`` 通道掩码：``True`` = 该像素的 line 标签可信。

    不可信时整通道为 ``False`` —— 调用方据此把 line 损失与 line 指标都
    排除在这一帧之外，而不是把可见漆线当负例。
    """
    lab = np.asarray(label)
    if audit.paint.valid:
        return np.ones(lab.shape, dtype=bool)
    return np.zeros(lab.shape, dtype=bool)


def mask_line_for_loss(label, audit: LabelAudit) -> np.ndarray:
    """把不可信帧的 line 像素改成 ignore(255)，返回新标签。

    这样现有损失函数（按 ``label != 255`` 取有效区）**无需改动**就尊重
    "缺真值不当负例"；测试固定了"改了之后 line 类在有效区里不再出现"。
    """
    lab = np.array(label, dtype=np.uint8, copy=True)
    if not audit.paint.valid:
        lab[lab == CLS_LINE] = IGNORE
    return lab


def audit_summary(audits: list[LabelAudit]) -> dict:
    """一组帧的逐通道覆盖：方案要求的"每类有效帧/像素与 UNKNOWN"。"""
    n = len(audits)
    out = {"n_frames": n,
           "road_valid_frames": sum(1 for a in audits if a.road.valid),
           "paint_valid_frames": sum(1 for a in audits if a.paint.valid),
           "pavement_valid_frames": sum(1 for a in audits
                                        if a.pavement.valid),
           "trainable_frames": sum(1 for a in audits if a.trainable),
           "road_px": sum(a.road.pixels for a in audits),
           "paint_px": sum(a.paint.pixels for a in audits),
           "line_masked_px": sum(a.line_masked_px for a in audits
                                 if not a.paint.valid),
           "unknown_reasons": {}}
    for a in audits:
        if a.unknown_reason:
            key = a.unknown_reason[:60]
            out["unknown_reasons"][key] = out["unknown_reasons"].get(key, 0) + 1
    # 每个比例都带分母，避免"没有可比样本"被读成 0%
    out["paint_valid_frac"] = (None if n == 0
                               else round(out["paint_valid_frames"] / n, 4))
    return out
