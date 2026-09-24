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
        blob = json.dumps({k: v for k, v in asdict(self).items()
                           if k not in ("frozen_at", "source")},
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
        # 小样本下用 t≈2 的粗略半宽，随 n 报告——不声称显著性
        half = 2.0 * sd / math.sqrt(len(d)) if len(d) > 1 else float("inf")
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
        return {"decision": "needs_evidence", "reasons": [
            f"{m}: not measured" for m in missing_metrics]}

    n_seeds = min((p.get("n") or 0) for p in pairings.values()) \
        if pairings else 0
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


def threshold_violations(measured: dict, thresholds: Thresholds) -> list:
    """硬门槛检查：返回违反项（空列表 = 全过）。缺测不算通过。"""
    out: list[str] = []
    checks = (
        ("candidate_identity_rate", thresholds.candidate_identity_rate_min,
         False),
        ("line_recall", thresholds.line_recall_min, False),
        ("line_precision", thresholds.line_precision_min, False),
        ("offroad_false_ratio", thresholds.offroad_false_ratio_max, True),
        ("inference_ms_p95", thresholds.inference_ms_p95_max, True),
    )
    for name, limit, lower_is_better in checks:
        v = measured.get(name)
        if v is None:
            out.append(f"{name}: UNKNOWN (hard gate needs a measurement)")
            continue
        if lower_is_better and float(v) > float(limit):
            out.append(f"{name}: {v} > {limit}")
        if not lower_is_better and float(v) < float(limit):
            out.append(f"{name}: {v} < {limit}")
    return out


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
