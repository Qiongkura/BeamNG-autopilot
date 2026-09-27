"""训练采样 v1：全池按场景（run）/视角配额轮换（T16 冻结契约 §3.2）。

为什么需要它：旧协议用 ``--max-train-frames`` 把候选训练集**固定截**成 N 帧
来凑"等步数对照"，新增视角因此永远进不了损失（实测：候选臂 48 帧只练 12 帧）。
T16 把等步数交给优化步预算（``--total-steps``），数据侧改为**从完整合格池**
按 run/视角配额轮换采样，并如实记录到底有多少**不同帧**真正进入过损失
（``unique_seen``）——池清单/训练目录里的文件数不能代替这个计数（方案 §7：
不得把候选池 60 个帧文件显示成"60 帧已学习"）。

本模块是纯逻辑（只依赖 numpy），**不 import torch**：训练器、autoloop 与测试
共用同一份采样语义，避免出现第二套实现。契约要点：

* 逐帧身份 = 内容 hash（sha256 前 16 位；公式与
  ``beamng_autopilot.experiments.manifest.content_sha16`` 一致，于是同一帧在
  两个实验臂、两个进程、两个命名空间里都是同一个 hash）；
* 每轮（epoch）= 池的一次配额轮换：按各 run 帧数比例分配配额、run 内洗牌、
  一轮内不重复同一帧（除非某个 run 的配额大于它的帧数）；
* 跨轮重新洗牌，且相邻两轮的顺序**不相同**（有放回时无法保证时除外）；
* 状态是纯 JSON（rng 状态转成 list/int）：落盘后精确恢复，
  ``next_batch()`` 序列与不中断完全一致。
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

#: 采样器状态版本（结构变化时必须 +1，旧状态直接拒绝而不是猜）
STATE_VERSION = 1

#: ``view`` 缺失时的报告键（不是"未知视角"的猜测，只是分组用的占位）
VIEW_UNKNOWN = "unknown"


def frame_identity(frame) -> str:
    """一帧的内容身份：sha256 前 16 位。

    ``frame`` 是训练器 ``train_frames`` 的元素，结构是
    ``(colour: np.ndarray, label: np.ndarray)``（``load_frames`` 的产物）。
    这里只哈希 **colour 的字节**，公式与
    ``experiments.manifest.content_sha16`` 相同——理由有两个：

    * 跨进程/跨臂稳定：不依赖路径、不依赖 Python 的 ``hash()`` 随机化；
    * 与仓库既有的内容哈希（manifest 去重、lineage 的"训练是否见过这张图"）
      **同一个命名空间**，采样报告与血缘审计可以对得上账。

    同一张图被复制成两个 npz（同色同标签）算同一帧；标签不同但 colour 逐字节
    相同的两份文件也会算同一帧——这在项目里不发生，且对"多少**不同画面**真正
    进了损失"这个口径而言，"画面"才是身份。

    也接受 dict（``{"colour": ...}``）与 npz 路径（读 colour 列，读不出时退回
    文件字节哈希，加 ``file:`` 域前缀避免与数组哈希混淆）。
    """
    colour = None
    if isinstance(frame, dict):
        colour = frame.get("colour")
        if colour is None:
            for key in ("image", "rgb", "frame"):
                if key in frame:
                    colour = frame[key]
                    break
        if colour is None:
            raise ValueError("frame_identity: dict 里找不到 colour 列")
    elif isinstance(frame, (tuple, list)):
        if not frame:
            raise ValueError("frame_identity: 空元组")
        colour = frame[0]                      # (colour, label) -> colour
    elif isinstance(frame, (str, bytes)) or hasattr(frame, "__fspath__"):
        path = str(frame)
        try:
            with np.load(path) as z:
                colour = np.asarray(z["colour"], dtype=np.uint8)
        except Exception:                      # noqa: BLE001
            h = hashlib.sha256()
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            return h.hexdigest()[:16]
    else:
        colour = frame

    a = np.ascontiguousarray(np.asarray(colour, dtype=np.uint8))
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


def plan_epochs(total_steps: int, n_train: int, batch: int) -> dict:
    """把步预算换算成每轮步数与轮数（纯函数，训练器/autoloop/测试共用）。

    ``steps_per_epoch = ceil(n_train / batch)``：全池配额轮换一轮正好把池走一遍，
    批数就是这个值（不足一批的尾批也算一步）。``epochs`` 取**最少**能满足预算
    的轮数，即 ``ceil(total_steps / steps_per_epoch)``——它是循环上界，不是主
    停止条件（主条件是实际 optimizer step 数）。
    """
    total_steps = int(total_steps)
    n_train = int(n_train)
    batch = int(batch)
    if batch <= 0:
        raise ValueError("batch 必须 > 0")
    if total_steps <= 0:
        # 0 = 关闭步预算（历史 epoch 协议），不是"跑 0 步"
        raise ValueError("total_steps 必须 > 0；legacy 用 --epochs，不走本函数")
    steps_per_epoch = max(1, -(-max(0, n_train) // batch))
    epochs = max(1, -(-total_steps // steps_per_epoch))
    return {"steps_per_epoch": int(steps_per_epoch),
            "epochs": int(epochs),
            "total_steps": int(total_steps)}


def allocate_run_quotas(run_sizes, round_size: int) -> list[int]:
    """把 ``round_size`` 个槽位按各 run 帧数比例分配给各 run（最大余数法）。

    为什么按比例而不是"每 run 等量"：等量会把小 run 的帧**重复**放大，把大 run
    的帧丢出这一轮，"曝光均衡"就变成"改变数据配比"；按比例则每个 run 的曝光
    与其规模成正比，全部帧每轮都覆盖一次（方案 §3.2 的"配额轮换"）。

    两条保护规则（规格要求）：
    * 帧数 > 0 的 run，在 ``round_size >= run 数`` 时**至少 1 帧**——极小 run
      不会被比例取整饿死；
    * 配额超过该 run 帧数时不做静默截断（调用方据此决定"允许重复"）。
    """
    sizes = [max(0, int(s)) for s in run_sizes]
    round_size = max(0, int(round_size))
    if not sizes or round_size <= 0 or sum(sizes) <= 0:
        return [0] * len(sizes)
    total = sum(sizes)
    exact = [round_size * s / total for s in sizes]
    quota = [int(x) for x in exact]
    left = round_size - sum(quota)
    order = sorted(range(len(sizes)),
                   key=lambda i: (-(exact[i] - quota[i]), i))
    for i in order[:max(0, left)]:
        quota[i] += 1
    nonempty = [i for i, s in enumerate(sizes) if s > 0]
    if round_size >= len(nonempty):
        for i in nonempty:
            if quota[i] > 0:
                continue
            # 从配额最大的 run 借一个（并列时取下标最小，确定性）；借不到就
            # 保持比例结果——宁可少 1 帧，也不把顺序依赖藏起来
            donor = max(range(len(sizes)), key=lambda k: (quota[k], -k))
            if quota[donor] > 1:
                quota[donor] -= 1
                quota[i] = 1
    return quota


def _key_int(run_key: str) -> int:
    """把 run_key 变成稳定的整数熵（Python 的 hash() 有随机化，不能用）。"""
    return int.from_bytes(
        hashlib.sha256(str(run_key or "").encode("utf-8")).digest()[:8], "big")


def _plain(obj):
    """numpy 标量/数组 -> 纯 Python，保证 JSON 与 torch.load(weights_only) 都吃。"""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return [_plain(v) for v in obj.tolist()]
    return obj


def _rng_state_to_json(rng: np.random.Generator) -> dict:
    return _plain(rng.bit_generator.state)


def _rng_from_json(blob: dict) -> np.random.Generator:
    """按状态里的 ``bit_generator`` 名重建 generator（类型不符就报错，不猜）。"""
    name = str((blob or {}).get("bit_generator") or "")
    if name != "PCG64":
        raise ValueError(f"采样器状态里的 bit_generator={name!r} 不是 PCG64")
    rng = np.random.default_rng(0)
    rng.bit_generator.state = _plain(blob)
    return rng


def _canonical_json(obj) -> str:
    return json.dumps(_plain(obj), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


class QuotaSampler:
    """全池按 run/视角配额轮换的采样器（池内下标）。

    ``pool`` 的每一项：``{"frame", "run", "view", "hash", "has_line"}``
    （``frame`` 只用于搬运，采样器本身不碰它）。``seed`` 与 ``run_key`` 一起
    决定顺序：同 seed + 同池 + 同 run_key ⇒ 完全相同的采样序列与曝光计数。

    ``run_key`` 是**本训练任务**的稳定标识（训练器用 ``dataset_id``，不用
    run_id/candidate_id）：配对实验里两臂希望"同数据同 seed ⇒ 同顺序"，用
    candidate id 会把配对打散。
    """

    def __init__(self, pool, *, batch: int, seed: int, run_key: str = ""):
        pool = list(pool)
        if not pool:
            raise ValueError("QuotaSampler: 池为空（没有可采样帧）")
        self._batch = int(batch)
        if self._batch < 1:
            raise ValueError("QuotaSampler: batch 必须 >= 1")
        self._seed = int(seed)
        self._run_key = str(run_key or "")
        self._frame = [e.get("frame") for e in pool]
        self._hashes = [str(e.get("hash") or "") for e in pool]
        if any(not h for h in self._hashes):
            raise ValueError("QuotaSampler: 池里有条目缺 hash（身份必须显式给出）")
        self._run_of = [str(e.get("run") or "") for e in pool]
        self._view_of = [("" if e.get("view") is None else str(e["view"]))
                         for e in pool]
        self._has_line = [bool(e.get("has_line")) for e in pool]
        self._unique = set(self._hashes)
        # 同一 hash 出现在多个槽位（重复帧）时按"或"合并正/负例分类，避免
        # 同一 hash 在报告里同时算正例和负例
        self._hash_has_line: dict[str, bool] = {}
        for h, hl in zip(self._hashes, self._has_line):
            self._hash_has_line[h] = self._hash_has_line.get(h, False) or hl
        self._runs = sorted(set(self._run_of))
        self._indices_by_run = {
            r: [i for i, x in enumerate(self._run_of) if x == r]
            for r in self._runs}
        self._rng = np.random.default_rng([self._seed, _key_int(self._run_key)])
        self._order: list[int] | None = None
        self._last_order: list[int] = []
        self._cursor = 0
        self._round = 0
        self._exposures_total = 0
        self._exposures_by_index: dict[int, int] = {}
        self._seen: set[str] = set()
        self._pool_digest_cache = self._pool_digest()

    # ------------------------------------------------------------------ 池
    @property
    def pool(self) -> list:
        return list(self._frame)

    @property
    def runs(self) -> list[str]:
        return list(self._runs)

    @property
    def round(self) -> int:
        """已开始的轮数（第 1 轮在建第一份轮计划时为 1）。"""
        return int(self._round)

    @property
    def cursor(self) -> int:
        """当前轮里已经取走的槽位数（续训时用它定位断点）。"""
        return int(self._cursor)

    @property
    def steps_per_epoch(self) -> int:
        """一轮的批数 = ceil(池大小/batch)（尾批不足也算一步）。"""
        return max(1, -(-len(self._frame) // self._batch))

    def _pool_digest(self) -> str:
        h = hashlib.sha256()
        for sha, run, view in zip(self._hashes, self._run_of, self._view_of):
            h.update(f"{sha}|{run}|{view}\n".encode("utf-8"))
        return h.hexdigest()[:16]

    # ------------------------------------------------------------ 轮计划
    def _build_round(self) -> list[int]:
        """一轮的下标序列：run 配额 + 组间混排 + run 内洗牌。

        先按配额分配槽位（比例），再用 rng 洗牌"槽位所属的 run 标签"，最后按
        轮内洗好的顺序从各 run 队列取帧——这样每个 batch 混有多个 run/视角，
        而不是"先把 run A 走完再走 run B"。配额大于该 run 帧数时该 run 内部
        允许重复（规格里的唯一例外）。
        """
        sizes = [len(self._indices_by_run[r]) for r in self._runs]
        quotas = allocate_run_quotas(sizes, len(self._frame))
        labels = np.repeat(np.arange(len(self._runs), dtype=np.int64), quotas)
        self._rng.shuffle(labels)               # 组间混排（跨轮重新洗牌）
        queues: list[list[int]] = []
        for k, run in enumerate(self._runs):
            idx = np.asarray(self._indices_by_run[run], dtype=np.int64)
            want = int(quotas[k])
            if want <= len(idx):
                pick = idx[self._rng.permutation(len(idx))[:want]]
            else:                                # 配额 > 帧数：有放回（规格例外）
                pick = idx[self._rng.choice(len(idx), size=want, replace=True)]
            queues.append([int(v) for v in pick.tolist()])
        ptr = [0] * len(self._runs)
        order: list[int] = []
        for k in labels.tolist():
            order.append(queues[k][ptr[k]])
            ptr[k] += 1
        return self._avoid_repeat_order(order)

    def _avoid_repeat_order(self, order: list[int]) -> list[int]:
        """相邻两轮的顺序必须不同（"跨轮重新洗牌"的可检查形式）。

        洗牌本来就几乎不会重复；但池很小时（例如 2 帧 1 run）重复顺序完全可能，
        那样"跨轮重新洗牌"就名存实亡。这里交换前两个不同元素强行打破（全元素
        相同时做不到也不该硬造，例如池只有 1 帧）。
        """
        if not self._last_order or order != self._last_order:
            return order
        for j in range(1, len(order)):
            if order[j] != order[0]:
                order[0], order[j] = order[j], order[0]
                break
        return order

    # ------------------------------------------------------------- 取批
    def next_batch(self) -> list[int]:
        """下一批池内下标（一轮走完自动开新一轮）。"""
        if self._order is None or self._cursor >= len(self._order):
            self._last_order = list(self._order or [])
            self._order = self._build_round()
            self._cursor = 0
            self._round += 1
        out = self._order[self._cursor:self._cursor + self._batch]
        self._cursor += len(out)
        for j in out:
            self._exposures_by_index[j] = self._exposures_by_index.get(j, 0) + 1
            self._exposures_total += 1
            self._seen.add(self._hashes[j])      # 只记**真实进入 batch** 的 hash
        return [int(j) for j in out]

    # ------------------------------------------------------------- 报告
    def report(self) -> dict:
        """数据利用计数。``unique_seen`` 只来自真实采样（不是池清单）。"""
        by_run = {r: 0 for r in self._runs}
        # 视角分组只列池里真实存在的视角（缺 view 的记 unknown），不给不存在的
        # 视角造 0 键——报告里的键集合本身也会被读成"有哪些视角"
        by_view = {v: 0 for v in sorted({v or VIEW_UNKNOWN
                                         for v in self._view_of})}
        for i, n in self._exposures_by_index.items():
            by_run[self._run_of[i]] = by_run.get(self._run_of[i], 0) + int(n)
            v = self._view_of[i] or VIEW_UNKNOWN
            by_view[v] = by_view.get(v, 0) + int(n)
        with_line = sum(1 for h in self._seen if self._hash_has_line.get(h))
        return {
            # 池大小（去重后的不同内容数）：重复帧不虚增可用帧数
            "unique_available": int(len(self._unique)),
            "unique_seen": int(len(self._seen)),
            "exposures_total": int(self._exposures_total),
            "exposures_by_run": {k: int(v) for k, v in sorted(by_run.items())},
            "exposures_by_view": {k: int(v)
                                  for k, v in sorted(by_view.items())},
            "frames_seen_with_line": int(with_line),
            "frames_seen_without_line": int(len(self._seen) - with_line),
            # 冗余但有用：槽位数（含重复帧）与已开轮数
            "pool_size": int(len(self._frame)),
            "rounds_done": int(self._round),
        }

    # ------------------------------------------------------------- 状态
    def state(self) -> dict:
        """完整可序列化状态（JSON 安全；torch.load(weights_only=True) 也能读）。"""
        return {
            "version": int(STATE_VERSION),
            "batch": int(self._batch),
            "seed": int(self._seed),
            "run_key": self._run_key,
            "pool_digest": self._pool_digest_cache,
            "round": int(self._round),
            "order": [] if self._order is None else [int(j) for j in self._order],
            "last_order": [int(j) for j in self._last_order],
            "cursor": int(self._cursor),
            "rng": _rng_state_to_json(self._rng),
            "exposures_total": int(self._exposures_total),
            "exposures_by_index": {str(k): int(v)
                                   for k, v in sorted(
                                       self._exposures_by_index.items())},
            "seen": sorted(self._seen),
        }

    def state_digest(self) -> str:
        """状态的 sha256[:16]：checkpoint 里用它做一次廉价的一致性指纹。"""
        return hashlib.sha256(
            _canonical_json(self.state()).encode("utf-8")).hexdigest()[:16]

    @classmethod
    def from_state(cls, state: dict, pool, *, batch: int) -> "QuotaSampler":
        """从 ``state()`` 精确恢复；恢复后 ``next_batch()`` 与不中断逐位一致。

        池变了（内容/归属/顺序不同）或 batch 变了都直接报错：那已经不是同一次
        采样任务，静默继续会产出无法对账的"续训"。
        """
        if not isinstance(state, dict):
            raise ValueError("from_state: state 不是 dict")
        if int(state.get("version") or 0) != STATE_VERSION:
            raise ValueError(
                f"from_state: 状态版本 {state.get('version')!r} 不受支持"
                f"（当前 {STATE_VERSION}）")
        if int(state.get("batch") or 0) != int(batch):
            raise ValueError(
                f"from_state: 状态的 batch={state.get('batch')} 与当前 "
                f"{batch} 不一致（批边界会变，无法逐位复现）")
        if state.get("rng") is None:
            raise ValueError("from_state: 缺 rng 状态（不能假装恢复到同一序列）")
        obj = cls(pool, batch=int(batch), seed=int(state.get("seed") or 0),
                  run_key=str(state.get("run_key") or ""))
        if str(state.get("pool_digest") or "") != obj._pool_digest_cache:
            raise ValueError("from_state: 池与状态不匹配（内容/run/视角有变）")
        obj._rng = _rng_from_json(dict(state["rng"]))
        obj._order = [int(j) for j in (state.get("order") or [])]
        obj._last_order = [int(j) for j in (state.get("last_order") or [])]
        obj._cursor = int(state.get("cursor") or 0)
        obj._round = int(state.get("round") or 0)
        obj._exposures_total = int(state.get("exposures_total") or 0)
        obj._exposures_by_index = {
            int(k): int(v)
            for k, v in dict(state.get("exposures_by_index") or {}).items()}
        obj._seen = {str(h) for h in (state.get("seen") or [])}
        n_pool = len(obj._frame)
        for name in ("order", "last_order"):
            bad = [j for j in getattr(obj, f"_{name}") if not 0 <= j < n_pool]
            if bad:
                raise ValueError(f"from_state: {name} 里有池外下标 {bad[:5]}")
        if not 0 <= obj._cursor <= len(obj._order):
            raise ValueError(
                f"from_state: cursor={obj._cursor} 超出轮长 {len(obj._order)}")
        return obj
