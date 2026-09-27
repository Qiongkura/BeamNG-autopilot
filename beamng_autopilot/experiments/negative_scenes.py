"""无标线场景的误报统计；不把 IoU 的零分母改写为满分。

方案 §7（T16）：负例要同时报**假线帧率、像素占比、最大连通域**与**进入控制
区域的假候选数**——帧率饱和到 1.0 时，后三者才是能继续区分模型的解释量
（原始帧率/像素占比一个都不隐去）。计数分两版：

* v1（历史）：帧率/像素类计数，旧判定按这些键判完整；
* v2（本文件 ``COUNTER_VERSION = 2``）：加 ``false_positive_max_cc_px``（该帧
  预测漆线的最大 4-连通域像素数）与 ``control_region_false_candidates``
  （预测漆线在控制相关区域内的连通域个数）。

兼容纪律：``negative_line_summary`` 默认按 **v1** 读——旧累计没有新键仍判
``complete``（新键按 0 报，但用 ``extra_counters="absent"`` 标出"根本没测"，
防止 0 被读成"测过且干净"）；只有显式 ``counter_version=2`` 才因缺新键判不完整。
"""

from __future__ import annotations

import numpy as np


#: v1（历史）计数键：**顺序与含义冻结**，旧累计/旧判定按它们判完整。
COUNTERS_V1 = (
    "frames", "eligible_frames", "clean_frames", "false_positive_frames",
    "false_positive_px", "eligible_px", "positive_frames",
    "unknown_frames", "empty_frames",
    #: 档位不是 verified 的帧（含未声明档位）：**排除量**，不进合格负例分母
    "unverified_frames",
    #: 这些帧里预测的标线像素（可上报的观测，不能当"假线结论"）
    "unverified_pred_line_px",
)

#: v2 新增计数键（方案 §7）。
COUNTERS_V2_NEW = ("false_positive_max_cc_px", "control_region_false_candidates")

#: 当前计数契约版本（新产物带 v2 键）。
COUNTER_VERSION = 2

#: 全部计数键（v1 + v2；顺序固定，便于报告与测试）。
COUNTERS = COUNTERS_V1 + COUNTERS_V2_NEW

#: 需要按 **max** 聚合的计数键（最大连通域是"最大"语义）：逐帧求和会得到
#: Σ 每帧最大值——无意义且会夸大。调用方用 :func:`merge_counts` 聚合。
MAX_COUNTERS = ("false_positive_max_cc_px",)

PREFIX = "negative_line_"

#: 汇总里派生字段要读的"该帧有假候选进控制区域"0/1 标记（不在 COUNTERS 里：
#: 它是 ``control_region_false_candidates`` 的伴随量，缺了不影响 v2 完整性）。
_CONTROL_REGION_FRAME_KEY = "false_positive_control_region_frames"


def _connected_component_sizes(mask: np.ndarray) -> list:
    """4-连通域的像素数列表（纯 numpy + 显式栈，不引 scipy 等新依赖）。

    为什么自己写：负例帧很小、预测漆线通常稀疏，BFS 只访问 True 像素；
    不引新依赖就不会改变仓库的可复现环境。前景全满的极端帧会退化成 O(h*w)，
    这是可以接受的（离线评价，不在驾驶 tick 里）。
    """
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2:
        raise ValueError(f"connected components need a 2-D mask, got {m.shape}")
    h, w = m.shape
    if h == 0 or w == 0:
        return []
    flat = m.ravel()
    total = flat.size
    visited = bytearray(total)
    sizes: list = []
    for start in np.flatnonzero(flat).tolist():
        if visited[start]:
            continue
        stack = [start]
        visited[start] = 1
        n = 0
        while stack:
            i = stack.pop()
            n += 1
            if i >= w and flat[i - w] and not visited[i - w]:
                visited[i - w] = 1
                stack.append(i - w)
            if i < total - w and flat[i + w] and not visited[i + w]:
                visited[i + w] = 1
                stack.append(i + w)
            if i % w and flat[i - 1] and not visited[i - 1]:
                visited[i - 1] = 1
                stack.append(i - 1)
            if (i + 1) % w and flat[i + 1] and not visited[i + 1]:
                visited[i + 1] = 1
                stack.append(i + 1)
        sizes.append(n)
    return sizes


def merge_counts(acc: dict, counts: dict) -> dict:
    """把逐帧（或逐目录）计数并入累加器：普通键**求和**，MAX_COUNTERS **取最大**。

    为什么单独给一个聚合函数：``false_positive_max_cc_px`` 是帧间最大连通域，
    用普通求和聚合会把它变成 Σ 每帧最大值（既非最大也非平均，纯伪数）。调用方
    应统一用本函数，而不是各写一份 ``acc[k] += v``。
    """
    max_keys = {PREFIX + k for k in MAX_COUNTERS} | set(MAX_COUNTERS)
    for key, value in (counts or {}).items():
        if value is None:
            continue
        try:
            v = int(value)
        except (TypeError, ValueError):
            continue
        if key in max_keys:
            cur = acc.get(key)
            acc[key] = v if cur is None else max(int(cur), v)
        else:
            acc[key] = int(acc.get(key, 0)) + v
    return acc


def negative_line_counts(pred_line: np.ndarray, label: np.ndarray, *,
                         label_rank: str | None = None,
                         control_region: np.ndarray | None = None) -> dict:
    """只有**档位 verified**、非空、全像素已知且无 class 2 的帧才是完整负例。

    255 是未知；即使已知区域没有标线，也不能推断整帧没有标线。
    **档位门槛（方案 v2 §3.5）**：engine/agent/unknown 来源的"标签全零"不构成
    确认无线——它们记进 `unverified_frames`（排除量），不进合格负例分母。
    `label_rank=None` 表示调用方**没有声明档位**，同样按不可信处理（不默认通过）。
    正例、含未知像素帧、空帧、未验证帧互斥分类，所有计数可直接累加。

    ``control_region``（可选；bool mask，True = 控制相关区域）：只在**合格负例帧**
    上统计"预测漆线在该区域内的 4-连通域个数"（0 = 没进控制区域）；``None`` 记 0。
    形状必须与标签一致（不一致直接报错，不静默忽略）。
    """
    pred = np.asarray(pred_line, dtype=bool)
    lab = np.asarray(label)
    if pred.shape != lab.shape or lab.ndim != 2:
        raise ValueError("negative-line evaluation requires matching 2D masks")
    if not np.isin(lab, (0, 1, 2, 255)).all():
        raise ValueError("unsupported segmentation label (expected 0/1/2/255)")
    region = None
    if control_region is not None:
        region = np.asarray(control_region, dtype=bool)
        if region.shape != pred.shape:
            raise ValueError(
                f"control_region shape {region.shape} != prediction shape "
                f"{pred.shape}")
    out = dict.fromkeys(COUNTERS, 0)
    out[_CONTROL_REGION_FRAME_KEY] = 0
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
        blocks = _connected_component_sizes(pred)
        region_hits = (len(_connected_component_sizes(pred & region))
                       if region is not None else 0)
        out.update(eligible_frames=1, clean_frames=int(fp == 0),
                   false_positive_frames=int(fp > 0), false_positive_px=fp,
                   eligible_px=int(lab.size),
                   false_positive_max_cc_px=(max(blocks) if blocks else 0),
                   control_region_false_candidates=int(region_hits))
        # 伴随 0/1 标记：普通求和聚合即可得到"有多少帧有假候选进控制区域"。
        out[_CONTROL_REGION_FRAME_KEY] = int(region_hits > 0)
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


def negative_line_summary(acc: dict, *, n_frames: int,
                          counter_version: int = 1) -> dict:
    """从逐帧计数汇总；旧产物或混合新旧计数不得冒充完整覆盖。

    ``counter_version``（默认 **1** = 保持旧语义，旧判定/旧测试不翻转）：

    * ``1``：完整性只看 v1 键——旧累计（没有 v2 新键）仍判 ``complete``；
      新键按 **0** 报，但 ``extra_counters="absent"`` 标出"这些 0 不是测出来的"；
    * ``2``：完整性看全部 v2 键，缺任一 -> ``missing_counters``（不完整）。

    新增解释量（方案 §7）：``false_positive_max_cc_px_max``（帧间最大连通域）、
    ``false_positive_control_region_frames``（有假候选进控制区域的帧数）、
    ``extra_counters``（present/partial/absent）。原始帧率与像素占比字段原样保留
    ——帧率饱和到 1.0 时靠新量解释，不隐去原值。
    """
    ver = int(counter_version)
    if ver not in (1, 2):
        raise ValueError(f"unsupported negative counter_version "
                         f"{counter_version!r} (expected 1 or 2)")
    required = COUNTERS_V1 if ver == 1 else COUNTERS
    missing = [k for k in required if PREFIX + k not in acc]
    complete = not missing and acc[PREFIX + "frames"] == n_frames
    out = {k: acc.get(PREFIX + k) for k in required}
    present_new = [k for k in COUNTERS_V2_NEW if PREFIX + k in acc]
    # v1 读法下新键可能整批缺席：按 0 报，但 extra_counters 说明它们没被测过。
    for k in COUNTERS_V2_NEW:
        if k not in out or out[k] is None:
            out[k] = int(acc.get(PREFIX + k) or 0)
    out["counter_version"] = ver
    out["extra_counters"] = ("present" if len(present_new) == len(COUNTERS_V2_NEW)
                             else "partial" if present_new else "absent")
    out.update(status="missing_counters", false_positive_frame_rate=None,
               false_positive_pixel_fraction=None,
               false_positive_max_cc_px_max=None,
               false_positive_control_region_frames=None)
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
    # 帧间最大：累加器里的 false_positive_max_cc_px 必须由 merge_counts() 以
    # max 语义聚合（普通求和会得到 Σ 每帧最大值，不是最大）。
    out["false_positive_max_cc_px_max"] = int(out["false_positive_max_cc_px"] or 0)
    cr_frames = acc.get(PREFIX + _CONTROL_REGION_FRAME_KEY)
    if cr_frames is None:
        # 没有伴随 0/1 标记时：候选总数为 0 才能确定"0 帧进控制区域"；
        # 总数 > 0 却说不出帧数 -> UNKNOWN（不拿总数冒充帧数）。
        if int(out["control_region_false_candidates"] or 0) == 0:
            cr_frames = 0
        else:
            out["false_positive_control_region_frames_missing"] = (
                "the per-frame 0/1 marker for control-region hits was not "
                "accumulated while the candidate total is > 0: the frame count "
                "is UNKNOWN, not 0")
    out["false_positive_control_region_frames"] = (
        None if cr_frames is None else int(cr_frames))
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
