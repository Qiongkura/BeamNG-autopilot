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
    #: 单独验证过的模拟器真值——**声明档位**。注意：这个字典只是"这个字符串
    #: 声称什么"，engine_verified 是否真的算 verified 必须经
    #: :func:`resolve_paint_rank` 用凭证里的真值证明判定（T16 §4.3 反伪造）；
    #: 人工修订（human_revision）有逐帧复核记录，按原语义保持 verified。
    "engine_verified": "verified",
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

#: engine_verified 凭证要求的真值契约版本（T16 Order0 §3.4）。
TRUTH_CONTRACT_V1 = "v1"


def engine_verified_proof(credential: dict | None) -> dict:
    """凭证是否**支持** engine_verified（P1 反伪造；T16 §4.3 / Order0 §3.4）。

    返回 ``{"proven": bool, "basis": str, "why": str}``。成立条件（任一）：
    ``credential["truth_verified"] is True``（真值验证器的派生结论），或
    ``credential["truth_provenance"]`` 满足 ``truth_contract == "v1"`` 且
    ``report.verified is True`` 且 ``report.verifier_version`` 非空。

    **为什么不能只看 sidecar 字符串**：``label_source`` 是声明，谁都能写；
    "引擎采集过"不等于"这批标签被独立验证过"（方案 §4.3：不得仅改 sidecar
    字符串把旧不完整标签提升为 engine_verified）。缺键/None 一律按**不成立**
    处理——这里只读 ``read_dir_credentials`` 返回 dict 里的键，不 import 凭证
    模块，避免模块间顺序耦合。
    """
    if not isinstance(credential, dict):
        return {"proven": False, "basis": "no_credentials",
                "why": ("no readable credential was supplied: engine_verified "
                        "is only a declared string here")}
    if credential.get("truth_verified") is True:
        return {"proven": True, "basis": "truth_verified_flag",
                "why": ("the credential carries truth_verified=True (the "
                        "verifier's conclusion was recorded)")}
    prov = credential.get("truth_provenance")
    if isinstance(prov, dict):
        report = prov.get("report") if isinstance(prov.get("report"), dict) else {}
        contract = str(credential.get("truth_contract")
                       or prov.get("truth_contract") or "")
        verifier = str(report.get("verifier_version") or "").strip()
        if (contract == TRUTH_CONTRACT_V1 and report.get("verified") is True
                and verifier):
            return {"proven": True,
                    "basis": "truth_contract_v1_report_verified",
                    "why": (f"truth_provenance report says verified=True under "
                            f"truth_contract={TRUTH_CONTRACT_V1!r} "
                            f"(verifier_version={verifier!r})")}
        return {"proven": False, "basis": "provenance_incomplete",
                "why": ("truth_provenance is present but is not a verified v1 "
                        f"report: truth_contract={contract!r}, "
                        f"report.verified={report.get('verified')!r}, "
                        f"verifier_version={verifier!r}")}
    return {"proven": False, "basis": "no_truth_provenance",
            "why": ("the credential declares engine_verified but carries no "
                    "truth_provenance/truth_verified keys")}


def resolve_paint_rank(label_source: str, *,
                       credential: dict | None = None) -> dict:
    """按凭证解析漆线来源**档位**（P1）：返回 ``{rank, declared_rank, notes,
    proof}``。

    * 非 ``engine_verified``：与 ``PAINT_SOURCE_RANK`` 直接查表**逐字相同**
      （human_revision / agent / pseudo / engine_annotation 语义不变）；
    * ``engine_verified``：凭证证明成立 -> ``"verified"``；否则 -> ``"absent"``
      （不得仅凭 sidecar 字符串升格），notes 里写明
      "engine_verified declared but no verifier proof (truth_provenance
      missing/unverified): treated as absent" 与证据标签。

    调用链（主 agent 接线）：``read_dir_credentials`` 读目录凭证 -> 本函数把
    ``label_source + credential`` 解析成最终档位；``audit_label`` 已接上它。
    """
    src = str(label_source or "")
    declared = PAINT_SOURCE_RANK.get(src, "absent")
    if src != "engine_verified":
        return {"rank": declared, "declared_rank": declared, "notes": [],
                "proof": {"proven": False, "basis": "not_engine_verified",
                          "why": (f"{src or '(empty)'!r} does not require "
                                  "verifier proof")}}
    proof = engine_verified_proof(credential)
    if proof["proven"]:
        return {"rank": "verified", "declared_rank": declared,
                "notes": [f"engine_verified supported by credential "
                          f"({proof['basis']})"],
                "proof": proof}
    note = ("engine_verified declared but no verifier proof (truth_provenance "
            "missing/unverified): treated as absent")
    return {"rank": "absent", "declared_rank": declared,
            "notes": [f"{note} [{proof['basis']}: {proof['why']}]"],
            "proof": proof}


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
    #: 逐像素路型计数（沥青/碎石/路肩）；空 = 该帧没有路型图（老帧）
    road_type_px: dict = None                     # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.road_type_px is None:
            self.road_type_px = {}
    label_sha256_16: str = ""
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"road": self.road.as_dict(), "paint": self.paint.as_dict(),
                "pavement": self.pavement.as_dict(),
                "road_type_px": dict(self.road_type_px),
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


#: 路型图取值（标注器的 road_type 列）：1=沥青 2=碎石 3=路肩
ROAD_TYPE_NAMES = {1: "asphalt", 2: "gravel", 3: "shoulder"}


def audit_label(label, *, paint_source: str = "engine_annotation",
                palette_ok: bool = True, road_min_px: int = 200,
                has_rgb: bool = True, road_type=None,
                credential: dict | None = None) -> LabelAudit:
    """逐帧判三个通道的可用性。

    ``paint_source`` 来自采集元数据的 ``label_source`` 或人工修订记录；
    ``engine_annotation``（本图默认）被判为 **unreliable**：可见漆线可能
    完全没有标注，此时 ``paint_valid=False`` 并把 line 通道屏蔽。

    ``credential``（可选，默认 None = 旧调用逐字不变）：``read_dir_credentials``
    读回的目录凭证。只有它带了真值证明（``truth_verified`` /
    ``truth_contract=="v1"`` + ``report.verified``）时，``engine_verified``
    才算 verified；否则档位降 ``absent`` 且整通道屏蔽（反伪造，方案 §4.3）。
    人工/agent/pseudo 档位不受影响。
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

    resolved = resolve_paint_rank(paint_source, credential=credential)
    rank = resolved["rank"]
    notes.extend(resolved["notes"])
    if rank == "verified":
        paint = ClassQuality(True, f"paint truth from {paint_source}",
                             line_px, rank=rank)
    elif rank == "absent" and str(paint_source or "") == "engine_verified":
        # 反伪造：声明 engine_verified 但凭证不成立 -> 连弱监督都不给
        # （`absent` 在 protocol.SOURCE_ELIGIBILITY 里全 False）。
        paint = ClassQuality(False, resolved["notes"][0], line_px,
                             usable=False, rank=rank)
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
    # 铺装/土肩：label 的 0/1/2/255 表达不了材质，所以要么有**逐像素路型图**
    # （标注器另存的一列：1=沥青 2=碎石 3=路肩），要么判"不可分"。路肩按背景
    # 处理（有铺装时土肩不算道路），因此"路面材质"这一列是可判的独立证据。
    _rt_counts: dict = {}
    _rt_ok = False
    _rt_reason = ("pavement vs shoulder is not separable from the road class "
                  "alone; needs its own verified labels")
    if road_type is not None:
        rt = np.asarray(road_type)
        if rt.shape == lab.shape:
            for _v, _name in ROAD_TYPE_NAMES.items():
                _n = int((rt == _v).sum())
                if _n:
                    _rt_counts[_name] = _n
            _rt_ok = bool(_rt_counts)
            if _rt_ok:
                _rt_reason = (
                    "road type is marked per pixel by the reviewer ("
                    + ", ".join(f"{k}: {v}px" for k, v in sorted(
                        _rt_counts.items()))
                    + "); shoulder is background, so it never counts as road")
    pavement = ClassQuality(bool(_rt_ok), _rt_reason, road_px,
                            usable=bool(_rt_ok), rank=rank)

    unknown_reason = ""
    if not road.valid:
        unknown_reason = road.reason
    elif not paint.valid:
        unknown_reason = f"paint truth unusable: {paint.reason}"

    return LabelAudit(road=road, paint=paint, pavement=pavement,
                      unknown_reason=unknown_reason,
                      line_masked_px=line_px, frame_unknown_frac=unknown_px / max(1, n_px),
                      label_sha256_16=label_sha16(lab), notes=notes,
                      road_type_px=dict(_rt_counts))


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
           #: 可训练/可测量的漆线监督（弱监督与 agent 档算 usable、不算 valid）。
           #: 训练准入看它；晋级门仍然只看 paint_valid_frames（方案 §6.1）。
           "paint_usable_frames": sum(1 for a in audits if a.paint.usable),
           #: 有路型图（沥青/碎石/路肩逐像素标记）的帧数：pavement 通道可判的前提
           "road_type_frames": sum(1 for a in audits if a.road_type_px),
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
    out["paint_usable_frac"] = (None if n == 0
                                else round(out["paint_usable_frames"] / n, 4))
    return out
