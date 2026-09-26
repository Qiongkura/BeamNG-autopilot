"""无标线场景的误报统计；不把 IoU 的零分母改写为满分。"""

from __future__ import annotations

import numpy as np


COUNTERS = (
    "frames", "eligible_frames", "clean_frames", "false_positive_frames",
    "false_positive_px", "eligible_px", "positive_frames",
    "unknown_frames", "empty_frames",
    #: 档位不是 verified 的帧（含未声明档位）：**排除量**，不进合格负例分母
    "unverified_frames",
    #: 这些帧里预测的标线像素（可上报的观测，不能当"假线结论"）
    "unverified_pred_line_px",
)
PREFIX = "negative_line_"


def negative_line_counts(pred_line: np.ndarray, label: np.ndarray, *,
                         label_rank: str | None = None) -> dict:
    """只有**档位 verified**、非空、全像素已知且无 class 2 的帧才是完整负例。

    255 是未知；即使已知区域没有标线，也不能推断整帧没有标线。
    **档位门槛（方案 v2 §3.5）**：engine/agent/unknown 来源的"标签全零"不构成
    确认无线——它们记进 `unverified_frames`（排除量），不进合格负例分母。
    `label_rank=None` 表示调用方**没有声明档位**，同样按不可信处理（不默认通过）。
    正例、含未知像素帧、空帧、未验证帧互斥分类，所有计数可直接累加。
    """
    pred = np.asarray(pred_line, dtype=bool)
    lab = np.asarray(label)
    if pred.shape != lab.shape or lab.ndim != 2:
        raise ValueError("negative-line evaluation requires matching 2D masks")
    if not np.isin(lab, (0, 1, 2, 255)).all():
        raise ValueError("unsupported segmentation label (expected 0/1/2/255)")
    out = dict.fromkeys(COUNTERS, 0)
    out["frames"] = 1
    if str(label_rank or "") != "verified":
        # 档位不可信：既不算合格负例、也不算"未知像素"（那是标签内容问题），
        # 单独记排除量；预测像素可上报但不能当假线结论。
        out["unverified_frames"] = 1
        out["unverified_pred_line_px"] = int(pred.sum())
        return {PREFIX + k: v for k, v in out.items()}
    if not lab.size:
        out["empty_frames"] = 1
    elif (lab == 255).any():
        out["unknown_frames"] = 1
    elif (lab == 2).any():
        out["positive_frames"] = 1
    else:
        fp = int(pred.sum())
        out.update(eligible_frames=1, clean_frames=int(fp == 0),
                   false_positive_frames=int(fp > 0), false_positive_px=fp,
                   eligible_px=int(lab.size))
    return {PREFIX + k: v for k, v in out.items()}


def negative_training_eligibility(dirs: list[dict], *, research: bool) -> dict:
    """E1 前置：训练用的"困难负例"目录资格（方案 §S6/E1 + §3.5/T10）。

    ``dirs``：逐目录汇总 ``[{"dir", "n_frames", "n_line_frames", "rank"}]``
    （``n_line_frames`` = 该目录里有标线真值的帧数；``rank`` = 该目录标线标签
    的档位）。

    判定：

    * ``n_line_frames > 0`` -> 普通训练数据（有正有负），不受本条约束；
    * 全零标线 + ``rank == "verified"`` -> **合格负例**（人工确认无线）；
    * 全零标线 + 非 verified -> 只能作**研究臂**负例：``research=False`` 时
      **拒绝**（"全零"不构成确认无线——与 T10 同一逻辑；混进可晋级训练等于
      教模型"这里没有线"，而实际可能有线），``research=True`` 时允许，但逐条
      记为弱负例（必须在报告里可见，不得当成"已确认负例"）。

    实测依据（2026-09-26，`logs/experiments/e1_readiness_20260926.json`）：
    仓库里**没有**同时满足"评价集之外 + 有身份/位姿 + 人工确认无线"的负例目录
    —— dirt_road_* 无身份（审计直接拒收）、引擎采集的 line 类"游戏不提供"
    （全零是缺失而非确认）、评价包里的 verified 负例是开发帧（训练禁用）。
    """
    confirmed: list = []
    weak: list = []
    rejected: list = []
    for d in dirs or []:
        n_frames = int(d.get("n_frames") or 0)
        n_line = int(d.get("n_line_frames") or 0)
        rank = str(d.get("rank") or "")
        if n_line > 0 or n_frames <= 0:
            continue                     # 不是"全零"目录：普通训练数据
        if rank == "verified":
            confirmed.append({"dir": d.get("dir"), "n_frames": n_frames,
                              "rank": rank})
        elif research:
            weak.append({"dir": d.get("dir"), "n_frames": n_frames,
                         "rank": rank or "absent",
                         "why": ("all-zero line labels with a non-verified "
                                 "rank: weak research negative, not a "
                                 "confirmed one")})
        else:
            rejected.append({"dir": d.get("dir"), "n_frames": n_frames,
                             "rank": rank or "absent",
                             "why": ("a training dir whose line labels are all "
                                     "zero and not verified cannot be used as "
                                     "a confirmed hard negative in a "
                                     "promotion-eligible run (T10)")})
    return {"confirmed": confirmed, "weak": weak, "rejected": rejected,
            "note": (f"confirmed={len(confirmed)} weak={len(weak)} "
                     f"rejected={len(rejected)} research={bool(research)}")}


def negative_line_summary(acc: dict, *, n_frames: int) -> dict:
    """从逐帧计数汇总；旧产物或混合新旧计数不得冒充完整覆盖。"""
    missing = [k for k in COUNTERS if PREFIX + k not in acc]
    complete = not missing and acc[PREFIX + "frames"] == n_frames
    out = {k: acc.get(PREFIX + k) for k in COUNTERS}
    out.update(status="missing_counters", false_positive_frame_rate=None,
               false_positive_pixel_fraction=None)
    if not complete:
        out["missing_reason"] = (
            "missing per-frame counters: " + ", ".join(missing) if missing
            else "per-frame counter coverage differs from n_frames")
        return out
    n, pixels = out["eligible_frames"], out["eligible_px"]
    out["status"] = "measured" if n else "no_eligible_frames"
    if n:
        out["false_positive_frame_rate"] = out["false_positive_frames"] / n
        out["false_positive_pixel_fraction"] = out["false_positive_px"] / pixels
    # 排除量必须可见（方案 §3.5：混合来源分开计数并报告排除量）：合格分母之外
    # 的帧分成"档位不可信 / 含未知像素 / 空帧"三类，各自可查。
    out["excluded_frames"] = (int(out["unverified_frames"] or 0)
                              + int(out["unknown_frames"] or 0)
                              + int(out["empty_frames"] or 0))
    if out["unverified_frames"]:
        out["excluded_reason"] = (
            f"{out['unverified_frames']} frame(s) excluded: label rank is not "
            "verified, so an all-zero label does not confirm 'no line' "
            "(engine/agent labels are not negative evidence)")
    return out
