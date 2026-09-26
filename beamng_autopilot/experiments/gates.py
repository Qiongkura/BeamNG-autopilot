"""阶段 0 冻结协议：主指标优先级、非劣门槛、预算与最终集规则。

方案要求"阈值先在基线与开发数据上定并写入版本化配置，搜索开始后不随候选
成绩改动"。所以这里的每个数字都带 **``frozen_at`` 与来源说明**，并且
``Thresholds.save`` 写盘后拒绝被静默改写：要改就只能产生一个新的
``config_hash``（= 新协议），旧候选的判定理由因此仍然可复现。

指标优先级（方案 §128，按优先级冻结）：

1. 可验证的本车道漆线身份/几何与右侧边界来源
2. 漏线及路外假线
3. 铺装/土肩
4. 延迟和资源
   像素 IoU 是训练诊断与辅助门槛，**不得单独用于选模型**。
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field

#: 主指标：先看身份/几何，再看漏线与路外假线；像素只作辅助。
PRIMARY_ORDER = ("candidate_identity_rate", "line_recall", "line_precision",
                 "offroad_false_line_px", "inference_ms_p95")
AUXILIARY = ("line_iou", "val_miou", "train_loss")


@dataclass(frozen=True)
class Thresholds:
    """事前冻结的非劣门槛。``lower_is_better`` 决定比较方向。"""

    # 主指标（方案 §128 的优先级顺序）
    candidate_identity_rate_min: float = 0.60     # 漆线身份确认率下限
    line_recall_min: float = 0.70                 # 漏线：不够就是漏真线
    line_precision_min: float = 0.40              # 路外假线：不够就是画错线
    offroad_false_ratio_max: float = 0.10         # 路外假线占预测标线像素比
    inference_ms_p95_max: float = 45.0            # 单帧分割 p95（离线）
    # 非劣余量：候选在这些指标上不得比 champion 差超过该比例
    noninferior_margin: float = 0.05
    #: 成对 seed 结论所需的最小 seed 数与差值/不确定度规则
    min_seeds: int = 3
    #: --- 候选口径门槛（2026-09-26 标定，见 docs/CANDIDATE_GATE_CALIBRATION_20260926.md）
    #: **None = 该门未声明**（旧阈值文件没有这些键 -> 哈希不变、不启用新门）。
    #: 覆盖率只在"有标线参考的帧/场景"上算：无标线场景没有参考，覆盖率天然为 0，
    #: 按全部场景算会让任何模型永久失败（标定实测 0.89–0.92，门 0.80）。
    candidate_reference_coverage_min: float | None = None
    #: 左右角色一致率：角色混淆超过 30% 时左右身份不足以支撑横向参考（实测 0.74）。
    left_right_role_agreement_min: float | None = None
    #: 逐场景判身份的样本量下限：低于 30 条候选时 p=0.5 的 95% 区间宽约 ±0.18。
    per_scene_min_candidates: int | None = None
    #: --- 续训等价性容差（GPU）-----------------------------------------
    #: CUDA 上没有 `nll_loss2d` 的确定性实现（本项目交叉熵就用它），所以
    #: "中断续训 = 未中断"在 GPU 上只能按容差判。实测（3 seed、AMP、CUDA、
    #: 2 轮中断→3 轮、`scripts/m5_seg_resume_tolerance.py`）：
    #:   控制组（同配置两次未中断）最大相对差 = 3.26e-3
    #:   续训组最大相对差 = 2.87e-3，比值 0.88 ⇒ 落在噪声带内
    #: 判据取"相对差 ≤ 控制组噪声的 factor 倍"，即下面的派生阈值。
    resume_tolerance_factor: float = 3.0
    resume_control_max_rel_diff: float = 0.0033
    resume_max_rel_diff: float = 0.01
    frozen_at: str = "2026-09-24"
    source: str = ("T13 measured baselines (docs/T13_LOCAL_TRAINING_20260924.md) "
                   "+ plan T14 §5; frozen before the T14 search starts")

    @property
    def config_hash(self) -> str:
        # None 值**不进哈希**：新加的门槛字段在旧阈值文件里没有键，保持 None ->
        # 旧文件的 config_hash 不变（冻结校验照过）；显式给值就进哈希（改值即新配置）。
        blob = json.dumps({k: v for k, v in asdict(self).items()
                           if k not in ("frozen_at", "source") and v is not None},
                          sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def resume_within_tolerance(self, max_rel_diff: float) -> dict:
        """GPU 上"续训≈未中断"的判据：相对差是否落在噪声带内。"""
        ok = float(max_rel_diff) <= float(self.resume_max_rel_diff)
        return {"max_rel_diff": float(max_rel_diff),
                "limit": float(self.resume_max_rel_diff),
                "control_noise": float(self.resume_control_max_rel_diff),
                "factor": float(self.resume_tolerance_factor),
                "within_tolerance": bool(ok),
                "basis": ("measured on CUDA+AMP, 3 seeds, "
                          "scripts/m5_seg_resume_tolerance.py")}

    def save(self, path, *, force: bool = False):
        import pathlib
        p = pathlib.Path(path)
        blob = {"schema": 1, "thresholds": asdict(self),
                "config_hash": self.config_hash,
                "primary_order": list(PRIMARY_ORDER),
                "auxiliary": list(AUXILIARY),
                "note": "frozen: changing any value creates a NEW config_hash"}
        if p.exists() and not force:
            old = json.loads(p.read_text(encoding="utf-8"))
            if old.get("config_hash") != self.config_hash:
                raise FileExistsError(
                    f"{p} already holds config {old.get('config_hash')}; "
                    f"refusing to overwrite with {self.config_hash}")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(blob, indent=1, ensure_ascii=False),
                     encoding="utf-8")
        return p


def seeds_needed_for_effect(sd: float, mean_delta: float, *,
                            alpha_half: float = 2.0,
                            max_seeds: int = 200) -> int | None:
    """判出 ``mean_delta`` 大约需要多少 seed（**保守近似**）。

    公式 ``n ≈ (2*sd/|delta|)^2``：等价于"乘子取 2"的样本量估计。严格的
    配对 t 检验（α=0.05、power≈0.8）乘子应为 ``(1.96+0.84)=2.8``，平方后
    7.84，比这里的 16 小一半——所以本估计**偏保守**（报出来的 seed 数偏多）。
    保守方向是安全的（宁可少下结论），但不代表统计充分性保证。
    """
    """要让"观测到的效应"脱离噪声带，还需要多少个 seed（粗估）。

    为什么值得算：3 个 seed 上看到 +0.04 而 sd≈0.07 时，区间宽到覆盖 0，
    再盲目加 seed 也可能永远判不出——先算一下所需规模，才知道这条因子
    在这个尺度上**是否值得继续投 GPU**。用与 ``PairedResult.summary``
    相同的粗略口径（半宽 = alpha_half * sd / sqrt(n)，n=1 时不可估）。

    返回达到"半宽 < |mean_delta|"所需的最小 n；效应为 0、sd 为 0 或超过
    ``max_seeds`` 时返回 None（=在可行规模内不可分辨，别再空转）。
    """
    try:
        sd = float(sd)
        mean_delta = abs(float(mean_delta))
    except (TypeError, ValueError):
        return None
    if sd <= 0.0:
        return None
    if mean_delta <= 0.0:
        return None
    need = (alpha_half * sd / mean_delta) ** 2
    n = int(math.ceil(need))
    return n if n <= int(max_seeds) else None


#: 双侧 95% t 分位（df 1..30）。**不依赖 scipy**：判定路径要能在最小环境里
#: 复现，表值本身由测试与 scipy 对照校验。df>30 用正态近似 1.96（与 t 差 <2%）。
T95_TABLE = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
    14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
    20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def t_critical(df: int, *, alpha: float = 0.05) -> float:
    """双侧 ``1-alpha`` 的 t 分位；``df <= 0`` 返回 ``inf``（一个样本给不出区间）。

    方案 §10.3：不确定度必须用**明确记录的方法**。这里用成对差值的 t 区间
    （假设差值 iid 近似正态），并把自由度与方法写进判定文件——
    ``2*sd/sqrt(n)`` 只是"乘子取 2"的近似，小样本下会低估区间
    （n=3 时 t=4.303，是它的 3.7 倍）。
    """
    if df <= 0:
        return float("inf")
    if alpha == 0.05:
        return T95_TABLE.get(int(df), 1.96)
    try:                                  # 非 0.05 时才用 scipy（可选依赖）
        from scipy import stats
        return float(stats.t.ppf(1.0 - alpha / 2.0, int(df)))
    except Exception:                     # noqa: BLE001
        raise ValueError(f"alpha={alpha} 需要 scipy 才能取 t 分位")

@dataclass
class PairedResult:
    """成对 seed 比较：差值、分布与不确定度都要跟着结论走。"""

    metric: str
    champion: list = field(default_factory=list)
    candidate: list = field(default_factory=list)
    lower_is_better: bool = False

    @property
    def n(self) -> int:
        return min(len(self.champion), len(self.candidate))

    @property
    def deltas(self) -> list:
        return [float(c) - float(b) for b, c in
                zip(self.champion, self.candidate)]

    def summary(self) -> dict:
        d = self.deltas
        if not d:
            return {"metric": self.metric, "n": 0, "mean_delta": None,
                    "missing": "no paired seeds measured"}
        mean = sum(d) / len(d)
        sd = (math.sqrt(sum((x - mean) ** 2 for x in d) / (len(d) - 1))
              if len(d) > 1 else 0.0)
        # 成对差值的 t 区间（方案 §10.3）：写清方法、自由度与假设。
        # 旧口径（乘子取 2）仍照实报告，便于与历史结论对照。
        df = len(d) - 1
        half_approx = (2.0 * sd / math.sqrt(len(d))
                       if len(d) > 1 else float("inf"))
        half = (t_critical(df) * sd / math.sqrt(len(d))
                if len(d) > 1 else float("inf"))
        lo, hi = mean - half, mean + half
        if self.lower_is_better:
            # 候选更好 = 差值为负；区间整体 < 0 才算可信改善
            verdict = ("candidate_better" if hi < 0 else
                       "champion_better" if lo > 0 else "inconclusive")
        else:
            verdict = ("candidate_better" if lo > 0 else
                       "champion_better" if hi < 0 else "inconclusive")
        need = seeds_needed_for_effect(sd, mean)
        return {"metric": self.metric, "n": len(d),
                "seeds_needed_for_effect": need,
                "champion": [round(float(x), 5) for x in self.champion],
                "candidate": [round(float(x), 5) for x in self.candidate],
                "deltas": [round(x, 5) for x in d],
                "mean_delta": round(mean, 5), "sd": round(sd, 5),
                "ci95_halfwidth": (None if half == float("inf")
                                   else round(half, 5)),
                "ci95_halfwidth_approx_2sd": (
                    None if half_approx == float("inf")
                    else round(half_approx, 5)),
                "ci_method": ("paired t interval, two-sided 95%, df=n-1, "
                              "assumes iid normal deltas"),
                "df": df,
                "verdict": verdict,
                "lower_is_better": bool(self.lower_is_better)}


def paired_compare(metric: str, champion: list, candidate: list, *,
                   lower_is_better: bool = False) -> dict:
    return PairedResult(metric=metric, champion=list(champion),
                        candidate=list(candidate),
                        lower_is_better=lower_is_better).summary()


def decide(*, pairings: dict, thresholds: Thresholds,
           missing_metrics: list | None = None,
           production_mismatch: bool = False,
           definition_drift: bool = False,
           confirmation_issues: list | None = None,
           hard_gate_violations: list | None = None) -> dict:
    """晋级判定：``rejected`` / ``needs_evidence`` / ``shadow_candidate``。

    规则（方案 §5）：
    * UNKNOWN、指标定义漂移、``production_mismatch`` 或硬门槛违反 → 不是
      晋级，而是 ``rejected``/``needs_evidence``；
    * 小预算预筛**只能淘汰**；
    * 完整 seed 评估通过才给 ``shadow_candidate``（且不覆盖生产模型）。
    """
    reasons: list[str] = []
    if production_mismatch:
        return {"decision": "rejected", "reasons": [
            "production_mismatch on the production arm: this round's metrics "
            "are void"]}
    if definition_drift:
        return {"decision": "rejected", "reasons": [
            "metric definitions drifted within the round"]}
    if confirmation_issues:
        # 方案 §10.3 第一条：输入不一致 -> invalid/rejected。最终确认记录对不上
        # （换了协议/权重/最终集）就是在拿另一次确认冒充这一次。
        return {"decision": "rejected", "reasons": [
            "final-set confirmation does not match this candidate: "
            + "; ".join(confirmation_issues)]}
    if hard_gate_violations:
        # 硬门槛失败时也要把成对比较的结论一起给出：否则读者只看到"缺测/越界"，
        # 看不到主指标到底有没有改善（两者是不同的问题）。
        also = []
        for name, p in (pairings or {}).items():
            if p.get("verdict"):
                also.append(f"{name}: {p['verdict']} "
                            f"(mean {p.get('mean_delta')})")
        return {"decision": "rejected",
                "reasons": list(hard_gate_violations) + also}
    if missing_metrics:
        # 缺测时也把成对比较的结论一起给出（与硬门分支同一条理由）：只写"缺测"
        # 会让读者看不到主指标到底有没有改善（方案点名的可读性要求）。
        _also = [f"{name}: {p['verdict']} (mean {p.get('mean_delta')})"
                 for name, p in (pairings or {}).items() if p.get("verdict")]
        return {"decision": "needs_evidence",
                "reasons": [f"{m}: not measured" for m in missing_metrics]
                + _also}

    # 只按**有测量**的配对算 seed 数：没测到的口径（n=0）走
    # missing_metrics 去 needs_evidence，不能把整个判定一起拉成“样本不足”
    # （实测踩到：新增两个未测口径后，本来能晋级的判定变成了
    # needs_evidence）。
    _measured = [p for p in (pairings or {}).values() if (p.get("n") or 0) > 0]
    n_seeds = min((p.get("n") for p in _measured), default=0)
    # 一个任务主指标都没测到：像素代理（IoU）不能替它下结论（方案 §10.3：
    # "像素 candidate_better 只是研究结论，不能直接变成模型晋级"）。
    if not [n for n in PRIMARY_ORDER if (pairings.get(n) or {}).get("n")]:
        return {"decision": "needs_evidence", "reasons": [
            "no primary (task) metric was measured: a pixel proxy alone "
            "cannot promote"]}
    if n_seeds < int(thresholds.min_seeds):
        return {"decision": "needs_evidence", "reasons": [
            f"only {n_seeds} paired seeds < required "
            f"{thresholds.min_seeds}: a single best seed must not promote"]}

    # 逐项证据照实写进理由（读者要能看到"不显著"与"明显变差"的区别）。
    # 拦截规则：
    #   * 主指标（身份/漏线/精度/路外假线/延迟）：至少一项**可信改善**，
    #     且不得有**可信变差**；inconclusive 只算"分不出来"，不拦。
    #   * 辅助指标（IoU 等）：只要求**非劣**（不超出余量）。
    improved, primary_worse, inconclusive = [], [], []
    for name in PRIMARY_ORDER:
        p = pairings.get(name)
        if not p:
            continue
        worse = _worse_delta(p)
        if p.get("verdict") == "candidate_better":
            improved.append(name)
        elif p.get("verdict") == "champion_better":
            primary_worse.append(
                f"{name}: champion_better - worse than the champion by "
                f"{worse:.4g} (ci95 ±{p.get('ci95_halfwidth')})")
        else:
            inconclusive.append(name)
    reasons += primary_worse
    reasons += [f"{name}: inconclusive (mean {pairings[name].get('mean_delta')}"
                f", ci95 ±{pairings[name].get('ci95_halfwidth')})"
                for name in inconclusive]
    blockers = []
    for name, p in pairings.items():
        if name in PRIMARY_ORDER:
            continue
        worse = _worse_delta(p)
        margin = _noninferior_margin(p, thresholds)
        if worse is not None and worse > margin:
            blockers.append(f"{name}: worse than the champion by {worse:.4g} "
                            f"(non-inferiority margin {margin:.4g})")
    if not improved:
        return {"decision": "rejected",
                "reasons": reasons + ["no primary metric shows a credible "
                                      "improvement over the champion"]}
    if primary_worse or blockers:
        return {"decision": "needs_evidence",
                "reasons": reasons + blockers + [
                    f"improved: {improved} - but the above are not "
                    f"non-inferior / not separable from seed noise"]}
    return {"decision": "shadow_candidate",
            "reasons": reasons + [f"improved: {improved}",
                                  "all compared metrics non-inferior",
                                  "shadow only: the production model is "
                                  "NOT replaced"],
            "note": "requires an independent Tech closed-loop before any "
                    "deployment decision"}


def _worse_delta(p: dict) -> float | None:
    """正值 = 候选比 champion 差多少（按指标方向归一）。"""
    d = p.get("mean_delta")
    if d is None:
        return None
    return float(d) if p.get("lower_is_better") else -float(d)


def _noninferior_margin(p: dict, thresholds: Thresholds) -> float:
    """非劣余量：按 champion 均值的相对比例换算成该指标的绝对量。

    champion 均值为 0（例如"本来就没错"的计数）时用 0——此时任何正的变差
    都算超出余量，因为相对比例没有意义。
    """
    champ = [float(x) for x in (p.get("champion") or []) if x is not None]
    if not champ:
        return 0.0
    mean = sum(champ) / len(champ)
    return abs(mean) * float(thresholds.noninferior_margin)


#: 硬门检查表：``(指标, 门槛字段, 越小越好)``。逐 seed、分场景与总体共用同一张表
#: ——换口径只改这里一处，避免三个地方各写一份"什么算过门"。
#: 后两项（覆盖率、左右角色）的门槛字段默认 None = **未声明**，只在阈值文件显式
#: 给值时才启用（旧阈值文件因此保持原语义；见 Thresholds 的注释）。
HARD_CHECKS = (
    ("candidate_identity_rate", "candidate_identity_rate_min", False),
    ("line_recall", "line_recall_min", False),
    ("line_precision", "line_precision_min", False),
    ("offroad_false_ratio", "offroad_false_ratio_max", True),
    ("inference_ms_p95", "inference_ms_p95_max", True),
    ("candidate_reference_coverage", "candidate_reference_coverage_min", False),
    ("left_right_role_agreement", "left_right_role_agreement_min", False),
)


#: 分场景**可当硬门**的指标：像素层三项 + 该场景自己的推理耗时。
#: 为什么身份/候选类指标不在里面：``candidate_identity_rate`` 的门槛是**总体
#: 冻结口径**下的值，单场景候选数可能只有几个，按它逐场景卡门会把噪声当结论；
#: 而"每场景至少多少候选才能单独判身份"这条门槛**尚未标定**——按方案 §10.2
#: 只能先上报（进 ``per_scene`` 明细），标定后再进硬门。
SCENE_HARD_FIELDS = ("line_recall", "line_precision", "offroad_false_ratio",
                     "inference_ms_p95")


def hard_split(measured: dict, thresholds: Thresholds,
               *, fields=None) -> dict:
    """硬门检查的**拆分**结果：``{"violations": [...], "missing": [指标名]}``。

    方案 §10.3 的判定顺序要求分开："完整测量但硬门失败" → ``rejected``；
    "证据缺失" → ``needs_evidence``。合并成一个列表就分不出这两种，
    于是逐 seed / 分场景调用方只能用 ``threshold_violations()``（旧行为，
    把缺测也写成违反）。
    """
    violations, missing = [], []
    wanted = tuple(fields) if fields is not None else None
    for name, limit_attr, lower_is_better in HARD_CHECKS:
        if wanted is not None and name not in wanted:
            continue
        limit = getattr(thresholds, limit_attr)
        if limit is None:
            continue          # 该门未声明（旧阈值文件）：不检查、也不算缺测
        v = (measured or {}).get(name)
        if v is None:
            missing.append(name)
            continue
        if lower_is_better and float(v) > float(limit):
            violations.append(f"{name}: {v} > {limit}")
        if not lower_is_better and float(v) < float(limit):
            violations.append(f"{name}: {v} < {limit}")
    return {"violations": violations, "missing": missing}


def per_seed_gate_violations(per_seed: dict, thresholds: Thresholds,
                            *, fields=None) -> list:
    """逐 seed 硬门：**每个 seed 的 checkpoint 自己**必须满足门槛（方案 §10.2）。

    ``fields``：只查这些指标（整通道被屏蔽的实验用，例如 road-only 下
    标线指标是"未测"而不是"很差"）。

    为什么不能只看跨 seed 均值：4 个 seed 的 ``line_recall`` 都 0.9、第 5 个
    0.3，均值 0.78 照样过 0.70——真正要部署的是具体 checkpoint，均值会把这个
    坏模型藏掉。缺测的 seed 不在这里当违反（走 :func:`per_seed_missing`，
    判定是 ``needs_evidence`` 而不是"测了不好"）。
    """
    out: list[str] = []
    for seed in sorted(per_seed, key=str):
        for r in hard_split(per_seed[seed] or {}, thresholds,
                            fields=fields)["violations"]:
            out.append(f"seed {seed}: {r}")
    return out


def per_seed_missing(per_seed: dict, thresholds: Thresholds,
                     *, fields=None) -> list:
    """逐 seed 缺测项（进 ``missing_metrics`` → ``needs_evidence``）。"""
    out: list[str] = []
    for seed in sorted(per_seed, key=str):
        for name in hard_split(per_seed[seed] or {}, thresholds,
                               fields=fields)["missing"]:
            out.append(f"seed {seed}: {name}: UNKNOWN (hard gate needs a "
                       f"measurement)")
    return out


def scene_report(per_scene: dict, thresholds: Thresholds, *, fields=None) -> dict:
    """分场景硬门：坏场景不能被合并均值抵消，缺测场景记 UNKNOWN（§10.2/A7）。

    ``per_scene``：``{场景/组键: 该场景自己的硬门度量}``。返回
    ``{"violations": [...], "missing": [...]}``，两者分别进判定的硬门与缺测通道。
    ``fields`` 缺省用 :data:`SCENE_HARD_FIELDS`（像素层 + 该场景耗时）。
    """
    wanted = SCENE_HARD_FIELDS if fields is None else tuple(fields)
    violations, missing = [], []
    for name in sorted(per_scene, key=str):
        sp = hard_split(per_scene[name] or {}, thresholds, fields=wanted)
        violations += [f"scene {name}: {r}" for r in sp["violations"]]
        missing += [f"scene {name}: {m}: UNKNOWN (hard gate needs a "
                    f"measurement)" for m in sp["missing"]]
    return {"violations": violations, "missing": missing}


def missing_metrics_for(pairings: dict, *,
                        coverage_gate_frozen: bool = True) -> list:
    """判定要用的缺测清单（在线判定与 replay 共用，保证逐字可复现）。

    ``n == 0`` 的配对一律算缺测；另外，**可测候选覆盖率**这个新硬门的阈值尚未
    标定（``COVERAGE_GATE_FROZEN=False``）时，即使测到了也不许当通过
    （方案 §10.2：新硬门未完成基线标定前阻止晋级）。把它放在这个纯函数里，
    replay 才能从判定文件里的 ``coverage_gate_frozen`` 重放出同一结论——
    否则"在线判 needs_evidence、重放判 rejected"。
    """
    out = [k for k, v in (pairings or {}).items() if not (v or {}).get("n")]
    cov = (pairings or {}).get("candidate_reference_coverage") or {}
    if not coverage_gate_frozen and cov.get("n"):
        out.append("candidate_reference_coverage: gate not calibrated "
                   "(unfrozen)")
    return out


def threshold_violations(measured: dict, thresholds: Thresholds) -> list:
    """硬门槛检查：返回违反项（空列表 = 全过）。缺测不算通过。"""
    sp = hard_split(measured, thresholds)
    return sp["violations"] + [
        f"{n}: UNKNOWN (hard gate needs a measurement)" for n in sp["missing"]]


def budget_report(*, gpu_minutes_used: float, gpu_minutes_limit: float,
                  candidates_used: int, candidates_limit: int) -> dict:
    left = max(0.0, float(gpu_minutes_limit) - float(gpu_minutes_used))
    return {"gpu_minutes_used": round(float(gpu_minutes_used), 1),
            "gpu_minutes_limit": float(gpu_minutes_limit),
            "gpu_minutes_left": round(left, 1),
            "candidates_used": int(candidates_used),
            "candidates_limit": int(candidates_limit),
            "exhausted": left <= 0.0 or int(candidates_used) >= int(
                candidates_limit),
            "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
