"""训练 lineage / 已见组关系审计（方案 v2 S1 §4–5）。

S1 要回答两个**不能靠猜**的问题：

1. 评价帧是否落在某个候选/生产模型的**训练组**里（已见组）——组键用
   ``manifest.dir_group``，与整组隔离、数据集清单**同一个定义**（方案 W2）；
2. 该帧的**内容**是否出现在训练目录里（同图重采/复制，比组重叠更强的证据）。

证据只认 checkpoint 自己声明的 ``train_args.runs``（训练目录列表）：

* 读不到 ``train_args``、或 ``runs`` 为空 → **UNKNOWN**（``None``），
  ``lineage_unknown`` 记模型名。**绝不默认 False**——"查不到"与"查过没有"
  必须是两回事（方案 §5：剩余帧不能直接标 leak-free）；
* 正向命中（组或内容）是证明，不会被另一个 checkpoint 的 UNKNOWN 推翻，
  但那个 UNKNOWN 仍记在 ``lineage_unknown`` 里。

内容比对复用 ``manifest.content_sha16``（同一套哈希定义），只扫 checkpoint
声明的 ``runs`` 目录**本层**的 ``frame_*.npz``，不做宽范围递归扫描。

**实测 caveat（生产 best.pt）**：``train_args.runs`` 可读（13 个目录），但其中
多个目录没有 meta 身份，``dir_group`` 退成 ``dir/<名字>`` / ``dir/<绝对路径>``
——与带身份的评价组键（``map/source_id``）**不同键空间**，相等判定只会给
False。这是"组键定义下不同组"，不是"默认干净"：无身份的训练目录无法用组键
证明没练过，报告不得据此单独宣称 leak-free（方案 §5 要求继续查空间与内容）。
``runs`` 里的相对路径按**进程 cwd** 解析（与 ``dir_group`` 一致）：审计时
cwd 必须与当初算帧组键时相同（本仓库约定为仓库根）。

``parent`` 是被审 checkpoint 集合的训练父来源（``init_from`` / ``resume_from``
/ ``parent`` 任一）：多个 checkpoint 取值不一致、或有 checkpoint 读不到 /
没记 → 不点名（``None`` + ``parent_unknown=True``）。它是**模型级**事实，按
接口要求逐帧重复返回。

本模块只做纯计算 + 只读文件访问：不训练、不占 GPU、不改任何产物、不写盘。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from beamng_autopilot.experiments.manifest import content_sha16, dir_group

#: train_args 里可能记录"父 checkpoint"的键（按优先级取第一个有值的）。
PARENT_KEYS = ("init_from", "resume_from", "parent")


def _read_train_args(ckpt_path) -> dict:
    """读 checkpoint 的 ``train_args``；读不了/不是 dict 时返回 ``{}``。

    只用 ``torch.load(..., weights_only=False)``（旧 checkpoint 里可能含
    numpy/list 等非白名单对象，实测权重加载会失败）。
    """
    try:
        import torch
        blob = torch.load(str(ckpt_path), map_location="cpu",
                          weights_only=False)
    except Exception:                                    # noqa: BLE001
        return {}
    if not isinstance(blob, dict):
        return {}
    ta = blob.get("train_args")
    return dict(ta) if isinstance(ta, dict) else {}


def load_train_runs(ckpt_path) -> list[str] | None:
    """读 checkpoint 声明的训练目录（``train_args.runs``）。

    返回 ``None`` 表示 UNKNOWN：文件读不了 / 不是 dict / 没有 train_args /
    train_args 不是 dict / runs 缺失或不是列表 / 列表里没有有效路径。
    **不返回 ``[]``**——空列表会被下游当成"没有任何训练目录"，而事实是不知道。
    """
    runs = _read_train_args(ckpt_path).get("runs")
    if not isinstance(runs, (list, tuple)):
        return None
    out = [str(r) for r in runs if str(r).strip()]
    return out or None


def _parent_from(train_args: dict) -> str | None:
    """train_args 里的父来源。旧 checkpoint 的 ``resume_from`` 可能是 bool
    （"这一轮是不是续训"），不是路径，不能当父 checkpoint 名字——忽略。"""
    for k in PARENT_KEYS:
        v = train_args.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, Path):
            return str(v)
    return None


def _load_checkpoint(entry: dict) -> dict:
    """把一个 ``{"name", "path"}`` 变成审计缓存项（runs / groups / parent）。"""
    name = str(entry.get("name") or entry.get("path") or "")
    path = str(entry.get("path") or "")
    ta = _read_train_args(path)
    runs = ta.get("runs")
    runs = ([str(r) for r in runs if str(r).strip()]
            if isinstance(runs, (list, tuple)) else [])
    return {"name": name, "path": path,
            # runs=None 就是 UNKNOWN（train_args 读不到 / 没有 runs / runs 空）
            "runs": runs or None,
            "groups": {dir_group(r) for r in runs} if runs else set(),
            "parent": _parent_from(ta)}


def _scan_run_colours(run_dir: str) -> set[str] | None:
    """扫一个训练目录**本层**的 ``frame_*.npz``，返回 colour 的内容哈希集合。

    ``None`` = 目录不存在 / 一个可读 npz 都没有 → 内容关系**不可判**（不是
    "训练集里没有"）。只 ``glob("frame_*.npz")`` 不递归：递归会把别的实验
    目录也算进来，既慢又会把无关集合当证据。缺 colour 列的 npz 跳过。
    """
    d = Path(run_dir)
    if not d.is_dir():
        return None
    hashes: set[str] = set()
    for f in sorted(d.glob("frame_*.npz")):
        try:
            with np.load(f) as z:
                if "colour" not in z.files:
                    continue
                hashes.add(content_sha16(np.asarray(z["colour"], np.uint8)))
        except Exception:                                # noqa: BLE001
            continue
    return hashes or None


def _content_status(ck: dict, sha: str, cache: dict) -> bool | None:
    """该 checkpoint 声明的训练目录里有没有这个内容哈希。

    True = 至少一个目录命中；False = 所有目录都扫到且没有；None = 至少一个
    目录扫不了（不存在/不可读）→ 不可判。命中优先于不可判（命中是证据）。
    """
    hit = False
    unknown = False
    for r in ck["runs"]:
        if r not in cache:
            cache[r] = _scan_run_colours(r)
        got = cache[r]
        if got is None:
            unknown = True
        elif sha in got:
            hit = True
    if hit:
        return True
    return None if unknown else False


def frame_training_relation(frames, checkpoints) -> list[dict]:
    """逐帧判定"是否被某个 checkpoint 的训练见过"。

    ``frames``：``[{"path", "group", "content_sha16"}]``（``content_sha16``
    可为 ``None``，此时内容关系不判、字段为 ``None``——不猜）。

    ``checkpoints``：``[{"name", "path"}]``。

    返回每帧 ``{"path", "group", "train_group_overlap", "train_content_overlap",
    "seen_in", "lineage_unknown", "parent", "parent_unknown"}``：

    * ``train_group_overlap``：``dir_group(train_run) == frame["group"]``；
      训练目录无身份时键退成 ``dir/...``，与带身份的组键不同键空间——相等判定
      只给 False，报告不得把它单独当 leak-free 结论（见模块 docstring）；
    * ``train_content_overlap``：仅在该帧给了 ``content_sha16`` 时判定；与
      checkpoint 声明目录里的 ``frame_*.npz`` colour 哈希比对；
    * ``True > None > False``：只要有一个可读 checkpoint 命中就是 True；
      有 checkpoint 的 lineage 读不到（或帧自己没有组/没有内容哈希）→ None；
      全部可读且都没命中才是 False；
    * ``seen_in``：命中（组或内容）的模型名；``lineage_unknown``：runs 读不到
      的模型名；
    * ``parent`` / ``parent_unknown``：见模块 docstring（模型级，逐帧重复）。

    只读文件、纯函数：不训练、不占 GPU。
    """
    ck = [_load_checkpoint(c) for c in (checkpoints or [])]
    cache: dict = {}                       # 一次调用内每个 runs 目录只扫一遍
    parents = sorted({c["parent"] for c in ck
                      if c["runs"] is not None and c["parent"]})
    parents_complete = bool(ck) and all(
        c["runs"] is not None and c["parent"] for c in ck)
    if len(parents) == 1:
        parent, parent_unknown = parents[0], not parents_complete
    else:
        # 0 个（没记/读不到）或互相矛盾 → 不点名，记 UNKNOWN
        parent, parent_unknown = None, True

    out = []
    for f in frames:
        group = str(f.get("group") or "")
        sha = f.get("content_sha16")
        sha = str(sha) if sha else None
        # 帧自己没有组 / 没有内容哈希，或压根没给 checkpoint → 判不了
        g_unknown = (not group) or (not ck)
        c_unknown = (sha is None) or (not ck)
        g_hit = c_hit = False
        seen: list[str] = []
        unknown_models: list[str] = []
        for c in ck:
            if c["runs"] is None:
                unknown_models.append(c["name"])
                g_unknown = True
                if sha is not None:
                    c_unknown = True
                continue
            this_g = bool(group) and group in c["groups"]
            if this_g:
                g_hit = True
            this_c = False
            if sha is not None:
                st = _content_status(c, sha, cache)
                if st is None:
                    c_unknown = True
                elif st:
                    c_hit = True
                    this_c = True
            if this_g or this_c:
                seen.append(c["name"])
        if g_hit:
            group_rel: bool | None = True
        elif g_unknown:
            group_rel = None
        else:
            group_rel = False
        if sha is None:
            content_rel: bool | None = None       # 没给哈希 -> 不判（不猜）
        elif c_hit:
            content_rel = True
        elif c_unknown:
            content_rel = None
        else:
            content_rel = False
        out.append({"path": str(f.get("path") or ""), "group": group,
                    "train_group_overlap": group_rel,
                    "train_content_overlap": content_rel,
                    "seen_in": seen, "lineage_unknown": unknown_models,
                    "parent": parent, "parent_unknown": parent_unknown})
    return out


def summarize(relations) -> dict:
    """把逐帧关系汇总成报告口径：哪些帧是候选已见、哪些是未知。

    返回 ``{n_frames, n_group_overlap, n_content_overlap, n_lineage_unknown,
    n_parent_unknown, by_group, in_sample_frames}``。计数守恒：
    True / False / None 是 ``train_group_overlap`` 的完整划分，
    ``n_lineage_unknown`` 就是 **None 的帧数**（读不到 lineage，不是"没重叠"）；
    ``by_group`` 的每项与顶层同口径（键为组名，按组名排序）。

    ``in_sample_frames`` = 组重叠或内容重叠为 True 的帧路径（候选已见）；
    内容重叠为 True 的帧即使组不重叠也算已见（同图复制是最强证据）。
    """
    rels = list(relations)
    top = {"n_frames": 0, "n_group_overlap": 0, "n_content_overlap": 0,
           "n_lineage_unknown": 0, "n_parent_unknown": 0,
           "by_group": {}, "in_sample_frames": []}
    for r in rels:
        key = str(r.get("group") or "")
        grp = top["by_group"].setdefault(key, {
            "n_frames": 0, "n_group_overlap": 0, "n_content_overlap": 0,
            "n_lineage_unknown": 0, "n_parent_unknown": 0,
            "in_sample_frames": []})
        seen = (r.get("train_group_overlap") is True
                or r.get("train_content_overlap") is True)
        path = str(r.get("path") or "")
        for acc in (top, grp):
            acc["n_frames"] += 1
            acc["n_group_overlap"] += int(
                r.get("train_group_overlap") is True)
            acc["n_content_overlap"] += int(
                r.get("train_content_overlap") is True)
            acc["n_lineage_unknown"] += int(
                r.get("train_group_overlap") is None)
            acc["n_parent_unknown"] += int(bool(r.get("parent_unknown")))
            if seen:
                acc["in_sample_frames"].append(path)
    top["by_group"] = dict(sorted(top["by_group"].items()))
    return top
