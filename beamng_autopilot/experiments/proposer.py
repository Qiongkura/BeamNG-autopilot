"""错误分桶与有限实验提议器：一次只改一个因子。

输入是**已有评估产物**（T13 的评估矩阵、identity probe、hard-negative
清单、事件流），输出是若干"可执行且有限"的候选配方。三条纪律：

1. **只从训练/开发集挖错误**；最终集不参与搜索（方案 §3/§4）。
2. 缺标线真值的错误**不能**直接当负例训练——它们进 ``needs_review``
   队列（方案 §3）。可验证区域之外不计算错误。
3. 提议顺序遵循方案 §4：先数据质量/场景配比，再困难样本采样、损失权重、
   训练轮次；且每个提议都带**总优化步数**，避免把"训得更久"当成"数据更好"。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

#: 提议族按方案给定的优先级排列（先试前面的）。
FAMILIES = ("scene_mix", "hard_negative_sampling", "loss_weights", "epochs")

#: 每个族**允许**改的参数。
ALLOWED_KEYS = {
    "scene_mix": ("add_runs", "drop_runs", "group_weights"),
    "hard_negative_sampling": ("oversample_runs", "hard_neg_manifest"),
    "loss_weights": ("line_weight", "line_tversky_weight",
                     "line_cldice_weight", "line_tversky_beta"),
    "epochs": ("epochs", "lr"),
}

#: 训练器/``rounds`` 真正能**应用**的键（`add_runs`/`drop_runs` 改训练输入，
#: 其余是训练器开关）。提议器只产出这里的键：产出一个没人实现的键，会让
#: `rounds` 拒绝训练（实测：旧 scene_mix 族发 `group_weights`，而训练器只有
#: 已标注像素 ignore + 类别权重），循环就在"因子未生效"上停止。
#: 与 ``scripts/m5_seg_autoloop.py`` 的 DATA_FACTORS|TRAINER_FLAG_FACTORS
#: 必须一致（有测试钉住两边不漂移）。
APPLICABLE_KEYS = ("add_runs", "drop_runs", "run_weights", "epochs", "lr",
                   "line_weight", "line_tversky_weight",
                   "line_cldice_weight", "line_tversky_beta")

#: 这些桶说明问题在数据/场景，不该退到"训练更久"。
_DATA_SIDE_BUCKETS = ("offroad_false_line", "candidates_not_on_paint",
                      "candidate_off_road")


@dataclass
class ErrorBucket:
    """一类错误：名称、计数、分母、证据路径。缺分母的计数不作排序依据。"""

    name: str
    count: int
    denominator: int | None
    evidence: list = field(default_factory=list)
    labels_verified: bool = False

    @property
    def rate(self) -> float | None:
        if not self.denominator:
            return None
        return self.count / self.denominator

    def as_dict(self) -> dict:
        return {**asdict(self), "rate": self.rate}


@dataclass
class Proposal:
    candidate_id: str
    family: str
    factor: dict
    hypothesis: str
    expected_steps: int | None = None
    evidence: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def bucket_errors(*, pixel: dict | None = None, identity: dict | None = None,
                  produce_graded: bool = False) -> list[ErrorBucket]:
    """从评估产物里分桶。``produce`` 之外的桶都要求标签可验证。"""
    out: list[ErrorBucket] = []
    if pixel:
        pred = pixel.get("pred_line_px") or 0
        out.append(ErrorBucket(
            "offroad_false_line", int(pixel.get("offroad_false_line_px") or 0),
            pred, evidence=[pixel.get("model", "")],
            labels_verified=True))
        out.append(ErrorBucket(
            "missed_true_line", int(pixel.get("missed_true_line_px") or 0),
            int(pixel.get("gt_line_px") or 0) or None,
            evidence=[pixel.get("model", "")], labels_verified=True))
    if identity:
        tot = int(identity.get("candidates_total") or 0)
        out.append(ErrorBucket(
            "candidate_off_road", int(identity.get("candidates_off_road")
                                      or 0), tot,
            evidence=[identity.get("run", "")], labels_verified=True))
        out.append(ErrorBucket(
            "role_mismatch", int(round((1.0 - float(
                identity.get("role_agreement_rate") or 0.0)) * tot)), tot,
            evidence=[identity.get("run", "")], labels_verified=True))
        if identity.get("candidate_paint_recall_p50") is not None \
                and float(identity["candidate_paint_recall_p50"]) < 0.02:
            out.append(ErrorBucket(
                "candidates_not_on_paint",
                int(round((1.0 - float(identity["candidate_paint_recall_p50"]))
                          * tot)), tot,
                evidence=["engine paint is not a verified paint truth on this "
                          "map: review queue, NOT a training negative"],
                labels_verified=False))
    if produce_graded:
        out.append(ErrorBucket("pavement_misgraded", 0, None,
                               evidence=["needs its own verified labels"],
                               labels_verified=False))
    return out


def needs_review(buckets: list[ErrorBucket]) -> list[dict]:
    """缺可信真值的错误进复核队列（不得直接当负例训练）。"""
    return [{"name": b.name, "count": b.count,
             "why": "labels not verified: human revision or separately "
                    "verified simulator truth required before training",
             "evidence": b.evidence}
            for b in buckets if not b.labels_verified and b.count > 0]


def propose(*, buckets: list[ErrorBucket], dataset: dict, champion: dict,
            family_order: list | None = None, max_proposals: int = 3,
            history: list | None = None, available_runs: list | None = None,
            blocked: list | None = None) -> list[Proposal]:
    """按优先级给出**有限**个单因子提议。

    ``dataset`` 给出当前版本的组与帧数（用于场景配比提议），``champion``
    给出它的配方与总优化步数（等步数对照的依据）。``available_runs`` 是
    **可用的、尚未入训的数据组**（场景配比族的输入；空 = 没有新数据可加）；
    ``blocked`` 是调用方传入的列表，用来收集"这个族为什么没产出提议"——
    数据不足时循环必须停下并说明原因，而不是退到"多训几轮"。
    """
    order = list(family_order or FAMILIES)
    ranked = sorted([b for b in buckets if b.labels_verified and b.rate],
                    key=lambda b: -float(b.rate))
    top = ranked[0].name if ranked else ""
    trials = {h.get("family") for h in (history or [])}
    out: list[Proposal] = []
    notes_extra: list = []
    steps_base = champion.get("steps")
    n_frames = int(dataset.get("n_train_frames") or 0)
    for fam in order:
        if fam in trials:
            continue
        # 路外候选占比过高同样是"场景配比"问题：模型在无铺装/土肩处画线，
        # 该加的是那些场景的数据，而不是先动训练轮次（实测：top 桶若是
        # candidate_off_road，旧写法会一路掉到最后的 epochs 族）。
        if fam == "scene_mix" and top in _DATA_SIDE_BUCKETS:
            # 只写真正改动的键：带空占位会让"一次只改一个因子"变成空话。
            # 场景配比族改的是**训练输入本身**（追加一组未入训的数据），
            # 这是训练器真能应用的键。
            dirs = [str(d) for d in (available_runs or [])]
            if not dirs:
                why = ("路外假线/候选不在漆线上占主导，该加的是那些场景的"
                       "数据；但没有任何可用的未入训数据组（--available-runs "
                       "为空）——需要先采集/复核新数据，不是改训练超参"
                       if available_runs is not None else
                       "路外假线/候选不在漆线上占主导，该族要加的是那些场景的"
                       "数据；调用方没有提供数据因子入口，本轮回退到训练轮次")
                if blocked is not None:
                    blocked.append({"family": fam, "why": why,
                                    "top_bucket": top})
                continue
            factor = {"add_runs": dirs[:1]}
            hyp = (f"路外假线/候选不在漆线上占主导：把未入训的场景组 "
                   f"{Path(dirs[0]).name} 加入训练输入（其余不变，等步数对照）")
        elif fam == "hard_negative_sampling" and top == "offroad_false_line":
            if blocked is not None:
                blocked.append({
                    "family": fam,
                    "why": ("按组采样权重（run_weights）训练器已实现，但提议器"
                            "还没有**按 run 的错误归因**来决定给哪个组加权："
                            "需要先有每开发路段的实测指标（下一步），再自动提议。"
                            "当前可由执行者显式给出 run_weights 因子"),
                    "top_bucket": top})
            continue
        elif fam == "loss_weights" and top == "missed_true_line":
            # 每族独立判断：旧写法在这里 continue 会连带跳过后面的 epochs 族
            factor = {"line_tversky_weight": 1.5}
            hyp = "漏真线为主：提高 line 通道 Tversky 权重（一次只改这一项）"
        elif fam == "epochs":
            if top in _DATA_SIDE_BUCKETS and available_runs is not None:
                if blocked is not None:
                    blocked.append({
                        "family": fam,
                        "why": ("错误集中在数据/场景（路外假线、候选不在漆线上），"
                                "延长训练不会解决；需要新数据或候选门改动"),
                        "top_bucket": top})
                continue
            factor = {"epochs": int((champion.get("epochs") or 3)) + 3}
            hyp = "以上因子都试过且仍有改善空间时，才延长训练轮次"
            if top in _DATA_SIDE_BUCKETS:
                notes_extra.append(
                    "错误集中在数据/场景：这是回退提议（调用方接上 "
                    "--available-runs 后可产出可执行的 add_runs）")
        else:
            continue
        not_applicable = [k for k in factor if k not in APPLICABLE_KEYS]
        if not_applicable:
            if blocked is not None:
                blocked.append({
                    "family": fam,
                    "why": (f"因子 {not_applicable} 训练器没有实现：产出一个"
                            f"应用不了的因子会让 rounds 拒绝整轮训练"),
                    "top_bucket": top})
            continue
        cid = _candidate_id(fam, factor)
        out.append(Proposal(
            candidate_id=cid, family=fam, factor=factor, hypothesis=hyp,
            expected_steps=_steps(steps_base, n_frames, factor),
            evidence=[b.name for b in ranked[:3]],
            notes=["single factor: every other setting is the champion's",
                   *notes_extra]))
        if len(out) >= int(max_proposals):
            break
    return out


def _steps(steps_base, n_frames: int, factor: dict) -> int | None:
    """估算总优化步数：数据臂变长时必须同时报告它（方案 §4）。"""
    if not n_frames:
        return None
    per_epoch = max(1, int(n_frames / 4))
    eps = int(factor.get("epochs") or 0)
    if eps and steps_base is not None:
        return int(steps_base) + per_epoch * max(0, eps - 3)
    return None


def _candidate_id(family: str, factor: dict) -> str:
    blob = json.dumps(factor, sort_keys=True, default=str)
    return f"{family}-{hashlib.sha256(blob.encode()).hexdigest()[:10]}"


def load_eval_artifacts(exp_dir: Path | str) -> dict:
    """读取 T13 风格的评估产物（评估矩阵 + identity probe + 事件）。"""
    d = Path(exp_dir)
    out: dict = {"pixel": None, "identity": None, "events": []}
    matrix = d / "06_eval_matrix.json"
    if matrix.exists():
        blob = json.loads(matrix.read_text(encoding="utf-8"))
        frozen = blob.get("frozen") or {}
        # 取最差的可信候选作为挖掘对象（champion 之外的基线也照此）
        rows = [(k, v) for k, v in frozen.items() if "error" not in v]
        if rows:
            out["pixel"] = min(rows, key=lambda kv: kv[1].get("line_iou") or 9)[1]
    for p in sorted(d.glob("ident_*.json")):
        blob = json.loads(p.read_text(encoding="utf-8"))
        s = blob.get("summary")
        if not s:
            continue
        if out["identity"] is None or \
                (s.get("candidates_off_road") or 0) > \
                (out["identity"].get("candidates_off_road") or 0):
            out["identity"] = {**s, "run": str(p.name)}
    ev = d / "events.jsonl"
    if ev.exists():
        out["events"] = [json.loads(ln) for ln in
                         ev.read_text(encoding="utf-8").splitlines()
                         if ln.strip()]
    return out
