"""选样评分与批次配额（方案 §7.5–7.7）。

方案原文：「选样评分结合场景缺口、未知程度、模型分歧和已验证的任务错误。无真值
区域只能进入复核队列，不能自动当 hard negative。」「限制同一地点连续帧和同一
相机的占比；保留无线负例、左右侧、铺装/土肩等覆盖。每次只增加预设小批场景，
经审计后才训练。」

本模块只做两件事，都是**纯函数**（输入是普通 dict，不读盘、不起进程）：

* :func:`frame_score`：给一帧打分，并把分数**拆成四个可查的分量**
  （未知程度 / 模型分歧 / 已验证错误 / 场景缺口），同时给出 ``route``：
  ``train`` 还是 ``review``；
* :func:`pick_batch`：按分数挑一小批，同时守住三条配额——同组帧数上限、
  同相机占比上限、覆盖种类（无线负例/左右侧/铺装土肩）不被冲掉——并如实
  报告丢了多少、为什么丢。

为什么"未复核只能进复核队列"要写死在代码里：未标注区域不等于"确定不是漆线"。
把没看过的帧当负例喂进去，等于用未知当答案（方案 §6.2、A2）。
"""

from __future__ import annotations

from pathlib import Path

from .labels import PAINT_SOURCE_RANK

#: 一批的预设规模（方案 §7.6「每次只增加预设小批场景」）
BATCH_SIZE = 20
#: 同一地点（组）在一批里最多几帧：相邻帧不是独立样本（§10.1 场景重采样纪律）
MAX_PER_GROUP = 6
#: 同一相机在一批里的占比上限
MAX_CAMERA_SHARE = 0.5
#: 评分权重（分数是"值得先看"的排序依据，不是任何模型指标）。
#: 已验证的任务错误权重最高：它直接指出模型现在错在哪。
WEIGHTS = {
    "verified_errors": 1.0,     # 每个已验证错误 +1.0
    "unknown_frac": 1.0,        # 未知像素占比（0..1）
    "disagreement": 1.0,        # 两臂预测分歧（0..1）
    "scene_gap": 0.5,           # 场景覆盖缺口（0..1，来自协议覆盖要求）
}


def _rank_of(label_source: str) -> str:
    return PAINT_SOURCE_RANK.get(str(label_source or ""), "absent")


def frame_score(rec: dict, *, gap_of=None) -> dict:
    """一帧的选样分数与去向。``gap_of(group) -> 0..1`` 可选（场景缺口）。"""
    rec = dict(rec or {})
    group = str(rec.get("group") or "")
    gap = float(rec.get("scene_gap") if rec.get("scene_gap") is not None
                else (gap_of(group) if gap_of else 0.0))
    parts = {
        "unknown_frac": round(float(rec.get("unknown_frac") or 0.0), 4),
        "disagreement": round(float(rec.get("disagreement") or 0.0), 4),
        "verified_errors": int(rec.get("verified_errors") or 0),
        "scene_gap": round(max(0.0, min(1.0, gap)), 4),
    }
    score = (parts["verified_errors"] * WEIGHTS["verified_errors"]
             + parts["unknown_frac"] * WEIGHTS["unknown_frac"]
             + parts["disagreement"] * WEIGHTS["disagreement"]
             + parts["scene_gap"] * WEIGHTS["scene_gap"])
    rank = _rank_of(rec.get("label_source"))
    verified = rank == "verified"
    # 三档去向（方案 §6.1/§6.2、A2）：
    #   train          —— 已复核真值：正例与**负例**都能教；
    #   train_research —— agent/弱监督：能训（研究），但**不能**教"这里不是线"，
    #                     也不能当评价参考；
    #   review         —— 来源不明/不可靠：先复核，别拿未标注区域当负例。
    if verified:
        route, reason = "train", "labels are verified (human/engine-verified)"
    elif rank in ("agent", "pseudo"):
        route, reason = (
            "train_research",
            f"rank {rank!r}: usable for research training, but unlabelled "
            "regions must not be taught as negatives and it cannot be an "
            "evaluation reference")
    else:
        route, reason = (
            "review",
            f"labels not verified (rank {rank!r}): an unlabelled region is not "
            "a confirmed negative, so this frame must be reviewed first")
    return {
        "frame": str(rec.get("frame") or ""),
        "group": group, "view": str(rec.get("view") or ""),
        "exposure": rec.get("exposure"),
        "kinds": list(rec.get("kinds") or []),
        "label_source": str(rec.get("label_source") or ""),
        "label_rank": rank,
        "score": round(float(score), 4),
        "parts": parts,
        "route": route,
        "route_reason": reason,
        #: 只有复核过的帧才允许贡献"这里不是漆线"的监督
        "may_teach_negative": bool(verified),
    }


def pick_batch(scored: list, *, limit: int = BATCH_SIZE,
               max_per_group: int = MAX_PER_GROUP,
               max_camera_share: float = MAX_CAMERA_SHARE,
               require_kinds=None) -> dict:
    """按分数挑一批，守住配额与覆盖；如实报告取舍。

    ``scored`` 是 :func:`frame_score` 的输出（或同形状 dict）。返回
    ``{items, n_pool, n_items, dropped_by_group_limit, dropped_by_camera_limit,
    coverage_filled, reasons}``。
    """
    items = [dict(s) for s in (scored or [])]
    limit = max(1, int(limit))
    ranked = sorted(items, key=lambda s: (-float(s.get("score") or 0.0),
                                          str(s.get("frame") or "")))
    # 同分帧按**相机轮转**排：否则同分的一批会按文件名顺序被同一路相机占满
    # （实测：640 帧池子里前 18 个全是 front_fisheye）。分数仍是主序，
    # 轮转只决定同分内部的顺序——不改变"高分优先"。
    bands: dict = {}
    for s in ranked:
        bands.setdefault(round(float(s.get("score") or 0.0), 6), []).append(s)
    spread: list = []
    for _score in sorted(bands, reverse=True):
        by_view: dict = {}
        for s in bands[_score]:
            by_view.setdefault(str(s.get("view") or ""), []).append(s)
        while any(by_view.values()):
            for v in list(by_view):
                if by_view[v]:
                    spread.append(by_view[v].pop(0))
    ranked = spread
    picked: list = []
    per_group: dict = {}
    dropped_group = 0
    dropped_camera = 0
    reasons: list = []
    # 第一遍：按分数取，只守**硬**配额——同组帧数上限。同组帧是同一地点的
    # 相邻帧，多取等于假装样本变多了，所以这条不让步。
    for s in ranked:
        if len(picked) >= limit:
            break
        g = str(s.get("group") or "")
        if per_group.get(g, 0) >= max_per_group:
            dropped_group += 1
            continue
        picked.append(s)
        per_group[g] = per_group.get(g, 0) + 1
    # 第二遍：相机占比是**可替换**的软约束。有别的相机可换就换掉超出的；
    # 换不到就如实报告"这个池子里满足不了"——不能因为池子只有一路相机，
    # 就把批次压成 1 帧（那是把配额当目的，反而没数据可训）。
    def _share(view: str) -> float:
        return (sum(1 for it in picked if str(it.get("view")) == view)
                / max(1, len(picked)))

    # 每路相机至少允许 1 帧：占比上限在"批次只有 1-2 帧"时数学上不可满足，
    # 硬套会把最高分帧反复换掉（实测：limit=1 时把分数 5.0 的帧换成了 0 分的）。
    _floor_per_view = 1
    for view in sorted({str(it.get("view")) for it in picked}):
        if not view or len(picked) < 2:
            continue
        _allowed = max(_floor_per_view,
                       int(max_camera_share * len(picked)))
        if sum(1 for it in picked if str(it.get("view")) == view) <= _allowed:
            continue
        others = [s for s in ranked
                  if str(s.get("view")) != view and s not in picked
                  and per_group.get(str(s.get("group") or ""), 0) < max_per_group]
        while (sum(1 for it in picked if str(it.get("view")) == view)
               > _allowed) and others:
            victim = sorted([it for it in picked
                             if str(it.get("view")) == view],
                            key=lambda s: (float(s.get("score") or 0.0),
                                           str(s.get("frame"))))[0]
            take = others.pop(0)
            picked.remove(victim)
            per_group[str(victim.get("group") or "")] -= 1
            picked.append(take)
            per_group[str(take.get("group") or "")] = (
                per_group.get(str(take.get("group") or ""), 0) + 1)
            dropped_camera += 1
        if _share(view) > max_camera_share:
            reasons.append(
                f"camera share for {view!r} is {_share(view):.0%} > "
                f"{max_camera_share:.0%}: the pool has no alternative camera to "
                "swap in, so this quota could not be satisfied (reported, not "
                "silently ignored)")

    # 覆盖兜底：要求的种类一个都没进批次时，用该种类里分最高的补进来
    # （宁可挤掉一个高分帧，也不能让"无线负例"这种覆盖整批缺失）
    coverage_filled: list = []
    for kind in (require_kinds or []):
        if any(kind in (it.get("kinds") or []) for it in picked):
            continue
        cands = [s for s in ranked
                 if kind in (s.get("kinds") or []) and s not in picked]
        if not cands:
            reasons.append(f"coverage kind {kind!r} requested but the pool has "
                           f"none: cannot be filled")
            continue
        take = cands[0]
        if len(picked) >= limit and picked:
            # 挤掉一个非兜底帧（优先挤掉同组重复度最高的）
            victim = sorted(picked, key=lambda s: (float(s.get("score") or 0.0),
                                                   str(s.get("frame"))))[0]
            picked.remove(victim)
            reasons.append(f"coverage: swapped out {victim.get('frame')!r} to "
                           f"keep kind {kind!r}")
        picked.append(take)
        coverage_filled.append(kind)
        reasons.append(f"coverage: forced in {take.get('frame')!r} for kind "
                       f"{kind!r}")
    if dropped_group:
        reasons.append(f"{dropped_group} frame(s) dropped by the per-group limit "
                       f"({max_per_group}/group): neighbouring frames are not "
                       "independent samples")
    if dropped_camera:
        reasons.append(f"{dropped_camera} frame(s) dropped by the camera-share "
                       f"limit ({max_camera_share:.0%})")
    n_review = sum(1 for s in picked if s.get("route") == "review")
    if n_review:
        reasons.append(f"{n_review} of {len(picked)} picked frame(s) can only go "
                       "to the review queue (labels not verified)")
    return {"items": picked, "n_pool": len(items), "n_items": len(picked),
            "dropped_by_group_limit": dropped_group,
            "dropped_by_camera_limit": dropped_camera,
            "coverage_filled": coverage_filled, "reasons": reasons,
            "n_review_only": n_review,
            "n_trainable": len(picked) - n_review}


def pool_from_dirs(dirs, *, root=None, kinds_of=None) -> list:
    """从数据目录建选样池：身份（map/source_id）、视角、曝光、标签来源。

    身份读 ``meta.json``（先本目录、再父目录），**不从目录名猜**；标签来源读
    标注凭证（``annotation.json`` / ``meta.json`` 的 ``label_source``）。
    读不到凭证 = 未复核（rank ``absent``）——选样时会路由到复核队列。
    """
    from .credentials import read_dir_credentials
    from .manifest import dir_group

    out: list = []
    for d in dirs or []:
        d = Path(d)
        meta = None
        for cand in (d / "meta.json", d.parent / "meta.json"):
            if cand.is_file():
                try:
                    import json as _json
                    meta = _json.loads(cand.read_text(encoding="utf-8"))
                except Exception:                          # noqa: BLE001
                    meta = None
                break
        cred = read_dir_credentials(d) or {}
        # 视角目录自己的 meta.json 会列出**全部视角**的帧，只按文件名匹配会
        # 张冠李戴（实测：640 帧池子里每一帧的 view 都变成了最后一个视角）。
        # 先按"视角目录/文件名"匹配，只有 meta 里确实只有一个视角时才退回文件名。
        meta_frames = list((meta or {}).get("frames") or [])
        n_views = len({str(f.get("view") or "") for f in meta_frames})
        by_rel = {str(f.get("path") or ""): f for f in meta_frames}
        by_name = ({str(f.get("path") or "").split("/")[-1]: f
                    for f in meta_frames} if n_views <= 1 else {})
        for f in sorted(d.glob("frame_*.npz")):
            rec = (by_rel.get(f"{d.name}/{f.name}") or by_rel.get(str(f))
                   or by_name.get(f.name) or {})
            out.append({
                "frame": str(f), "group": dir_group(d),
                "view": str(rec.get("view") or d.name),
                "exposure": rec.get("exposure"),
                "label_source": str(cred.get("label_source") or ""),
                "kinds": list((kinds_of(f) if kinds_of else []) or []),
            })
    return out
