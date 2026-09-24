r"""M5 路面/标线分割训练：轻量 UNet（beamng_autopilot.vision.segmentation）。

数据：scripts/m5_collect_seg.py 采集的 npz 帧（colour + label）。
划分：按帧序时间划分（前 80% 训练，后 20% 验证，与 M3 相同做法）。
      --split per-run 时改为"每个 run 各取时间尾部 20% 做验证"——
      避免多 run 拼接后全局尾部集中在最后几个 run（标线稀疏段），
      让验证集被某一类路段主导、line IoU 失真。
损失：交叉熵 + 中位频率类别加权（标线像素极少，不加权学不动）。
指标：mIoU + 各类 IoU + 像素准确率。自动保存最优模型与训练曲线。

--min-line-frac <x>：丢弃标线像素占比低于 x 的帧（x 常用 0.003）。
标线稀疏帧大多是无标线路段，混进训练只稀释 line 监督；held-out 评估
（run_navroute_town / run_bridge）显示密集 run 训练的模型泛化才有效。
--balance-runs：多 run 等量采样，每个 run 每 epoch 贡献相同帧数
（小 run 循环补齐），防止帧数多的 run 数值主导梯度——v9 教训：
标线稀疏 run 帧数多会淹没密集 run，模型被教成"标线很稀有"。

验证/评估指标口径：IoU 用全局累加 inter/union 计算（iou_from_accum），
未出现的类别不虚高为 1.0、不计入 mIoU。

用法:
    .venv\Scripts\python.exe scripts\m5_train_seg.py --runs logs\m5_seg\run_*
        --epochs 40 --out logs\m5_seg\seg_model
    中断后续训（每轮结束自动落盘 checkpoint_last.pt）:
    .venv\Scripts\python.exe scripts\m5_train_seg.py --runs logs\m5_seg\run_*
        --epochs 40 --out logs\m5_seg\seg_model --resume logs\m5_seg\seg_model\checkpoint_last.pt
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beamng_autopilot import config
from beamng_autopilot.vision.segmentation import (
    SegUNet, N_CLASSES, CLASS_NAMES, iou_from_accum,
)
from beamng_autopilot.experiments.checkpoint import checkpoint_extras
from beamng_autopilot.vision.dataset_split import (
    FrameRef, coverage_digest, cross_view_leak, frame_refs_from_meta,
    leak_check, select_weak_lines, split_audit, split_by_group,
)


#: 训练过程可视化的进程内状态（供异常处理器写"失败"任务记录）
_MON: dict = {}


def record_training_failure(exc: BaseException) -> None:
    """训练异常退出时补一条 ``failed`` 任务记录（图表保留失败前的记录）。

    方案要求"训练失败时保留此前图表，并显示错误摘要"；失败不写的话，看板
    只会停在最后一条 running 记录上，看起来像"还在跑"。
    """
    store = _MON.get("store")
    if store is None:
        return
    try:
        sampler = _MON.get("sampler")
        if sampler is not None:
            sampler.stop()
        from beamng_autopilot.experiments.metrics import task_record
        args = _MON.get("args")
        run_id = getattr(args, "metrics_run", "") or ""
        steps = _MON.get("step", 0)
        store.append(task_record(
            run_id, getattr(args, "task_name", "") or run_id, "failed",
            total_steps=_MON.get("total_steps"),
            current_step=steps, epoch=_MON.get("epoch"),
            started_at=(store.task() or {}).get("started_at"),
            error=f"{type(exc).__name__}: {exc}"[:400]))
        print(f"[train] 已记录失败状态 -> {store.path}", flush=True)
    except Exception:                        # noqa: BLE001
        pass                                 # 留痕失败不能掩盖原始异常


def _run_key(rd: Path) -> str:
    """A per-directory key that is unique across ring collections.

    ``<collection>/<view>`` dirs share the basename ``front_main``; keying
    ``per_run`` by the basename collapsed six collections into one entry
    (measured: --split by-map-scene then trained on 25 of 173 frames).  The
    key is the path relative to LOGS_DIR when possible, else the path
    itself, so every directory is its own split group.
    """
    p = Path(rd).resolve()
    try:
        base = Path(config.LOGS_DIR).resolve()
        rel = p.relative_to(base)
        return rel.as_posix()
    except Exception:                        # noqa: BLE001
        return p.as_posix()


def line_supervision_flags(ys, paint_source: str) -> list:
    """逐样本判断：这一帧的 **line 通道**有没有可信真值。

    返回 False 的帧，其 line 类会被整通道屏蔽：既不做正样本也不做负样本。
    判据来自仓库统一的标签审计 ``audit_label``，不是命令行声明——引擎标注的
    paint 类在这批地图上不可用（可见漆线常被标成沥青），所以会被判 False 并
    **不再产生错误监督**。审计抛异常时按"不可信"处理（拒绝监督是保守方向）。
    """
    from beamng_autopilot.experiments.labels import audit_label
    arr = np.asarray(ys.detach().cpu().numpy(), dtype=np.uint8)
    out = []
    for b in range(arr.shape[0]):
        try:
            out.append(bool(audit_label(arr[b],
                                        paint_source=paint_source).paint.valid))
        except Exception:                                 # noqa: BLE001
            out.append(False)
    return out


def load_frames(
    run_dirs: list[Path], min_line_frac: float = 0.0,
    line_only_dirs: set[str] | None = None,
    thin_line: int = 0,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict]:
    """读取所有 npz 帧，按文件名排序（时间序）。

    ``min_line_frac > 0`` 时丢弃标线像素占比低于阈值的帧。返回
    (frames, per_run)：per_run[run_name] 记录该 run 的帧总数/保留数/
    标线像素占比/在拼接列表中的 [start, end) 下标，供 per-run 验证划分。

    ``thin_line > 0`` 把标线目标侵蚀 N 个像素，教模型学**细**车道线：标注的
    line 类（与用户手绘 4-6 px 笔画）比"配对真正消费的那条细线"宽，而
    2026-09-11 的实验证明两者不是同一个目标（标注 line IoU 与配对率单调反
    相关）。``cv2.ximgproc`` 不可用，形态学侵蚀是最便宜的骨架化近似。全标注
    集里被侵蚀的边缘记作路面(1)；line-only 人工集里周围本就未知，记作忽略(255)。
    """
    import cv2

    frames: list[tuple[np.ndarray, np.ndarray]] = []
    per_run: dict[str, dict] = {}
    line_only_dirs = line_only_dirs or set()
    for rd in run_dirs:
        fs = sorted(glob.glob(str(rd / "frame_*.npz")))
        if not fs:
            raise SystemExit(f"没有找到数据: {rd}")
        # The KEY must be unique per directory, not per basename: ring
        # collections are passed as <collection>/<view>, so six collections
        # all named "front_main" collided and per_run kept only the last -
        # with --split by-map-scene that silently trained on 25 of the 173
        # frames (measured 2026-09-24).  The basename stays as a label.
        run_name = _run_key(rd)
        n_run = n_kept = 0
        n_pix = 0
        line_px = 0
        rec = {"frames": len(fs), "kept": 0, "dir_name": rd.name,
               "line_px_frac": 0.0, "start": len(frames), "end": len(frames)}
        for f in fs:
            d = np.load(f)
            colour, label = d["colour"], d["label"]
            colour = np.asarray(colour, dtype=np.uint8)
            label = np.asarray(label, dtype=np.uint8)
            _lab_vals = np.unique(label)
            _bad = [int(v) for v in _lab_vals if int(v) not in (0, 1, 2, 255)]
            if _bad:
                raise SystemExit(
                    f"标签污染：{f} 里出现未定义类别 {_bad}（只允许 "
                    f"0/1/2/255）——按 T14 §4 立即失败，不静默当背景")
            if rd.name in line_only_dirs:
                # hand annotations mark ONLY line pixels; every other
                # pixel is unknown, not background/road
                label = np.where(label == 2, 2, 255).astype(np.uint8)
            if thin_line > 0:
                line = (label == 2).astype(np.uint8)
                if line.any():
                    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    thin = cv2.erode(line, k, iterations=int(thin_line))
                    removed = (line > 0) & (thin == 0)
                    # Full-label scenes: the eroded border is road.  A
                    # line-only set has no information around the stroke, so
                    # there it stays unknown (already 255) rather than being
                    # taught as road.
                    if not (rd.name in line_only_dirs):
                        label = np.where(removed, 1, label).astype(np.uint8)
                    label = np.where(thin > 0, 2, label).astype(np.uint8)
            n_pix = colour.shape[0] * colour.shape[1]
            line_px += int((label == 2).sum())
            n_run += 1
            if min_line_frac > 0 and (label == 2).mean() < min_line_frac:
                continue
            frames.append((colour, label))
            # kept so the weak/faded-line band can be selected for real
            # (plan E7) instead of inferred from the run average
            rec.setdefault("fracs", []).append(float((label == 2).mean()))
            n_kept += 1
        rec["kept"] = n_kept
        rec["line_px_frac"] = line_px / max(1, n_run * n_pix)
        rec["end"] = len(frames)
        # Real identities (T10): the collector's meta carries map/episode,
        # wall clock and the exposure counter.  The run DIRECTORY name is
        # kept only as a fallback, and the fallback is reported below.
        # A ring collection keeps its meta at the collection root while the
        # frames live in <root>/<view>/, so it is read from there and
        # FILTERED to this view - passing all eight views' frames as this
        # run's identity would mis-state counts and exposures.
        meta, meta_level = None, None
        if (rd / "meta.json").exists():
            meta, meta_level = rd / "meta.json", "self"
        elif (rd.parent / "meta.json").exists():
            meta, meta_level = rd.parent / "meta.json", "parent"
        if meta is not None:
            try:
                blob = json.loads(meta.read_text(encoding="utf-8"))
                if meta_level == "parent":
                    blob = dict(blob, frames=[
                        f for f in (blob.get("frames") or [])
                        if str(f.get("view") or "").strip() == rd.name])
                rec["meta"] = blob
                rec["meta_level"] = meta_level
            except Exception as exc:         # noqa: BLE001
                rec["meta_error"] = f"{type(exc).__name__}: {exc}"
        per_run[run_name] = rec
    if not frames:
        raise SystemExit(f"过滤后没有数据（min_line_frac={min_line_frac}）")
    return frames, per_run


def median_freq_weights(labels: list[np.ndarray],
                       line_weight: float = 2.0) -> torch.Tensor:
    """Median frequency balancing：权重与类别频率成反比。"""
    hist = np.zeros(N_CLASSES, dtype=np.float64)
    for _, lab in labels:
        known = np.asarray(lab).ravel()
        known = known[known < N_CLASSES]
        hist += np.bincount(known, minlength=N_CLASSES)
    hist /= max(1.0, hist.sum())
    med = float(np.median(hist[hist > 0]))
    w = np.array([med / max(h, 1e-6) for h in hist])
    w = np.clip(w, 0.1, 20.0)
    w[2] *= line_weight  # line 类再放大：细线目标需要更强的监督
    return torch.tensor(w, dtype=torch.float32)


def split_frames(
    frames: list[tuple[np.ndarray, np.ndarray]], per_run: dict,
    split: str, val_frac: float,
    train_only_runs: set[str] | None = None,
) -> tuple[list[tuple[np.ndarray, np.ndarray]],
           list[tuple[np.ndarray, np.ndarray]]]:
    """把帧列表划分成训练/验证。

    ``tail``：全部 run 拼接后取全局尾部（历史行为）。
    ``per-run``：每个 run 各取时间尾部 val_frac 做验证——避免拼接后
    全局尾部集中在最后几个 run（往往是标线稀疏段），让验证集被某一类
    路段主导、line IoU 失真。
    """
    train_only_runs = train_only_runs or set()
    if split == "per-run":
        train_frames: list = []
        val_frames: list = []
        for name, rec in per_run.items():
            k = rec["kept"]
            if name in train_only_runs:
                train_frames.extend(frames[rec["start"]:rec["end"]])
                continue
            seg = frames[rec["start"]:rec["end"]]
            if k <= 1:
                train_frames.extend(seg)
                continue
            n_val = max(1, int(k * val_frac))
            train_frames.extend(seg[:k - n_val])
            val_frames.extend(seg[k - n_val:])
        return train_frames, val_frames
    n_val = max(1, int(len(frames) * val_frac))
    return frames[:len(frames) - n_val], frames[len(frames) - n_val:]


def train_run_bounds(
    frames: list[tuple[np.ndarray, np.ndarray]], per_run: dict,
    split: str, val_frac: float,
    train_only_runs: set[str] | None = None,
) -> list[tuple[int, int]]:
    """split_frames 划分后，各 run 在训练帧列表中的 [start, end) 下标。

    与 split_frames 保持同一套逻辑（per-run 取每 run 头部、tail 取
    全局头部），供 --balance-runs 做逐 run 等量采样。
    """
    bounds: list[tuple[int, int]] = []
    train_only_runs = train_only_runs or set()
    if split in ("per-run", "by-map-scene"):
        off = 0
        for name, rec in per_run.items():
            k = rec["kept"]
            if name in train_only_runs:
                bounds.append((off, off + k))
                off += k
                continue
            if k <= 1:
                bounds.append((off, off + k))
                off += k
                continue
            n_val = max(1, int(k * val_frac))
            bounds.append((off, off + (k - n_val)))
            off += k - n_val
        return bounds
    end = len(frames) - max(1, int(len(frames) * val_frac))
    for rec in per_run.values():
        s, e = rec["start"], rec["end"]
        bounds.append((min(s, end), min(e, end)))
    return bounds


def cap_train_frames(train_frames: list, bounds: list[tuple[int, int]],
                     n_max: int, *, seed: int = 0):
    """把训练帧**按各 run 的帧数配额**截到 ``n_max`` 帧（等步数对照用）。

    为什么不能直接取前 n_max 帧：等步数对照的候选臂通常是"多了一组场景"，
    取前面那一段切掉的正好是新增的那一组，因子等于没生效。按 run 配额取，
    每个 run 都有代表；run 内用固定种子抽样（可复现）。
    返回 ``(frames, bounds, note)``；``n_max<=0`` 或本来就够少时原样返回。
    """
    if n_max <= 0:
        return train_frames, bounds, {"applied": False, "why": "n_max<=0"}
    spans = [(s, e, max(0, e - s)) for s, e in bounds]
    total = sum(k for _s, _e, k in spans)
    if total <= n_max:
        return train_frames, bounds, {"applied": False,
                                      "why": f"训练帧 {total} <= {n_max}"}
    # 按长度比例分配配额，余数给小数部分最大的（并列时按下标，确定性）
    quota = []
    for i, (_s, _e, k) in enumerate(spans):
        exact = n_max * k / total
        quota.append([i, int(exact), exact - int(exact)])
    left = n_max - sum(q[1] for q in quota)
    for i, _q, _frac in sorted(quota, key=lambda x: (-x[2], x[0]))[:left]:
        quota[i][1] += 1
    rng = np.random.default_rng(int(seed))
    out: list = []
    new_bounds: list[tuple[int, int]] = []
    per_run: dict = {}
    off = 0
    for (s, e, k), (i, want, _frac) in zip(spans, quota):
        idx = list(range(s, e))
        if want < k:
            pick = np.sort(rng.choice(k, size=want, replace=False))
            idx = [idx[int(j)] for j in pick]
        out.extend(train_frames[j] for j in idx)
        new_bounds.append((off, off + len(idx)))
        off += len(idx)
        per_run[i] = [k, len(idx)]
    note = {"applied": True, "n_max": int(n_max), "kept": off, "total": total,
            "per_run": per_run}
    return out, new_bounds, note


def weighted_run_indices(run_bounds: list[tuple[int, int]],
                         weights: dict, rng: np.random.Generator):
    """按 run 权重采样**每轮总帧数不变**的训练帧（标准 oversampling）。

    用途是"困难样本采样"这一类数据因子：某个场景组太弱就把它的配额加权上去，
    **有放回**重复它的帧；被降权的 run 少采（无放回）。总样本数不变 ⇒ 批数与
    优化步数与对照臂一致，避免把"训练更久"混进结论。

    为什么不能"每 run 只采不重复的帧"：那样在总帧数固定时，每个 run 最终都会
    被采满自己的全部帧，权重完全不起作用（实测踩到：3 个 10 帧的 run 权重
    2/1/0.5 得到配额 [10,10,10]）。
    """
    spans = [(s, e) for s, e in run_bounds if e > s]
    if not spans:
        return np.array([], dtype=np.int64), {"applied": False}
    total = sum(e - s for s, e in spans)
    w = [max(0.0, float(weights.get(str(i), weights.get(i, 1.0))))
         for i in range(len(spans))]
    if sum(w) <= 0:
        w = [1.0] * len(spans)
    # 配额按权重比例分配，余数给小数部分最大的（确定性），和恰为 total
    exact = [total * wi / sum(w) for wi in w]
    quota = [int(x) for x in exact]
    left = total - sum(quota)
    order = sorted(range(len(spans)), key=lambda i: (-(exact[i] - quota[i]), i))
    for i in order[:left]:
        quota[i] += 1
    out: list = []
    per_run: list = []
    for i, (s, e) in enumerate(spans):
        room = e - s
        want = quota[i]
        idx = np.arange(s, e)
        if want <= 0:
            per_run.append({"run": i, "frames": room, "quota": 0,
                            "repeats": 0})
            continue
        replace = want > room                     # 超额 -> 有放回（重复采样）
        pick = rng.choice(room, size=want, replace=replace)
        out.extend(int(idx[j]) for j in pick)
        per_run.append({"run": i, "frames": room, "quota": int(want),
                        "repeats": int(max(0, want - room))})
    note = {"applied": True, "total": total, "quota": quota,
            "weights": {str(i): w[i] for i in range(len(w))},
            "per_run": per_run}
    return np.array(out, dtype=np.int64), note


def balanced_indices(run_bounds: list[tuple[int, int]],
                     rng: np.random.Generator) -> np.ndarray:
    """多 run 等量采样：每个 run 每 epoch 贡献相同帧数（小 run 循环补齐）。

    多 run 混训时帧数多的 run 会数值压制其他 run（v9 教训：标线稀疏
    run 帧数多，淹没密集 run，模型被教成"标线很稀有"，line 召回掉档）。
    等量采样让每个路段域对梯度贡献相同，稀疏 run 不再靠帧数主导。
    """
    nonempty = [(s, e) for s, e in run_bounds if e > s]
    if not nonempty:
        return np.array([], dtype=np.int64)
    n_max = max(e - s for s, e in nonempty)
    idx = []
    for s, e in nonempty:
        n = e - s
        perm = rng.permutation(n)
        reps = int(np.ceil(n_max / n))
        idx.append((np.tile(perm, reps)[:n_max] + s).astype(np.int64))
    out = np.concatenate(idx)
    rng.shuffle(out)
    return out


def _augment(frame, rng: np.random.Generator,
            line_morph: bool = False):
    """在线数据增强（colour/label 同步变换），提升路段泛化。

    关键：色相/饱和度扰动 + 模糊，逼模型学"标线结构"而不是记住
    特定路段的颜色纹理（旧模型过拟合训练路段：换路 recall 0%）。

    ``line_morph`` 开启标线形态扰动（dilate/erode/随机擦除）。实验
    （v7）证明它对标线 IoU 是负优化（0.33 -> 0.03，训练/留出集都是），
    默认关闭；仅作消融保留。
    """
    import cv2

    colour, label = frame
    if rng.random() < 0.5:                      # 水平翻转
        colour = np.fliplr(colour).copy()
        label = np.fliplr(label).copy()
    if rng.random() < 0.8:                      # 亮度/对比度扰动（光照鲁棒：
        a = float(0.6 + 0.8 * rng.random())     # 覆盖晨/午/昏/夜差异）
        b = float(-35.0 + 70.0 * rng.random())
        colour = np.clip(colour.astype(np.float32) * a + b,
                         0, 255).astype(np.uint8)
    if rng.random() < 0.5:                      # 色温扰动：R/B 通道独立增益
        rg = float(0.88 + 0.24 * rng.random())  # （模拟清晨偏红/黄昏偏橙）
        bg = float(0.88 + 0.24 * rng.random())
        c = colour.astype(np.float32)
        c[..., 0] *= rg
        c[..., 2] *= bg
        colour = np.clip(c, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:                      # 色相/饱和度扰动（换路面颜色）
        hsv = cv2.cvtColor(colour, cv2.COLOR_RGB2HSV).astype(np.int16)
        hsv[..., 0] = (hsv[..., 0] + int(rng.integers(-20, 21))) % 180
        s_gain = float(0.6 + 1.0 * rng.random())
        hsv[..., 1] = np.clip(hsv[..., 1] * s_gain, 0, 255)
        v_gain = float(0.85 + 0.3 * rng.random())
        hsv[..., 2] = np.clip(hsv[..., 2] * v_gain, 0, 255)
        colour = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    if rng.random() < 0.35:                     # 模糊（模拟行驶运动模糊）
        k = int(rng.integers(3, 8)) | 1
        colour = cv2.GaussianBlur(colour, (k, k), 0)
    if rng.random() < 0.7:                      # 随机裁剪后缩放回原尺寸
        h, w = colour.shape[:2]
        ch, cw = int(h * 0.85), int(w * 0.85)
        y0 = int(rng.integers(0, h - ch + 1))
        x0 = int(rng.integers(0, w - cw + 1))
        colour = cv2.resize(colour[y0:y0 + ch, x0:x0 + cw], (w, h),
                            interpolation=cv2.INTER_AREA)
        label = cv2.resize(label[y0:y0 + ch, x0:x0 + cw], (w, h),
                           interpolation=cv2.INTER_NEAREST)
    if line_morph and rng.random() < 0.45 and (label == 2).any():
        # 标线形态扰动：模拟远距离细线/断线/磨损，逼模型学标线结构
        # 而不是记住粗线。粗线->弥合（dilate），细线->收缩（erode），
        # 或随机擦除几段（虚线/磨损），三种都以真实物理为约束：
        # 擦除掉的标线底下是沥青路面（1），不是背景。
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        line = (label == 2).astype(np.uint8)
        r = rng.random()
        if r < 0.4:                             # 弥合断裂：标线变粗/连上
            line2 = cv2.dilate(line, k, iterations=1).astype(bool)
            label[line2] = 2
        elif r < 0.75:                          # 远处细线：标线收缩变细
            line2 = cv2.erode(line, k, iterations=1).astype(bool)
            label[line.astype(bool) & ~line2] = 1
        else:                                   # 随机擦除段：虚线/磨损
            erase = np.zeros_like(label, dtype=bool)
            for _ in range(int(rng.integers(1, 4))):
                ys = int(rng.integers(0, label.shape[0] - 6))
                xs = int(rng.integers(0, label.shape[1] - 6))
                hh = int(rng.integers(2, 12))
                ww = int(rng.integers(2, 12))
                erase[ys:ys + hh, xs:xs + ww] = True
            label[erase & line.astype(bool)] = 1
    return colour, label


def main() -> None:
    ap = argparse.ArgumentParser(description="分割训练")
    ap.add_argument("--runs", nargs="+", required=True,
                    help="数据目录（可多个，如 logs/m5_seg/run_*）")
    ap.add_argument("--line-only-runs", nargs="*", default=[],
                    help="只标了标线的人工标注目录；非 line 像素忽略(255)，不当作背景")
    ap.add_argument("--thin-line-labels", type=int, default=0, metavar="N",
                    help="把标线目标侵蚀 N 像素以教模型学「细」车道线"
                         "（0=关闭）。标注的 line 类比配对消费的细线宽，"
                         "见 2026-09-11 的 IoU 与配对率反相关结论；"
                         "cv2.ximgproc 不可用时这是最便宜的骨架化近似。")
    ap.add_argument("--train-only-runs", nargs="*", default=[],
                    help="只进训练、不切到验证集的人工标注目录")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--split-gap-s", type=float, default=0.5,
                    help="相邻帧判定的时间间隔（秒）；相差不超过它的两帧"
                         "不应被切到 train/val 两侧")
    ap.add_argument("--split",
                    choices=["tail", "per-run", "by-map-scene"],
                    default="tail",
                    help="验证划分：tail=全部 run 拼接后取全局尾部 "
                         "(默认，历史行为)；per-run=每个 run 各取尾部 "
                         "val_frac 做验证，避免稀疏 run 主导验证集")
    ap.add_argument("--weak-line-oversample", type=int, default=0,
                    help="把淡线/远场带（low..high 的 line 占比）的训练帧额外重复 "
                         "N 次（增补，不替换密集标线数据）；0=关闭")
    ap.add_argument("--weak-line-low", type=float, default=0.0005)
    ap.add_argument("--weak-line-high", type=float, default=0.01)
    ap.add_argument("--min-line-frac", type=float, default=0.0,
                    help="丢弃标线像素占比低于该值的帧（如 0.003），"
                         "防止无标线路段稀释 line 监督")
    ap.add_argument("--balance-runs", action="store_true",
                    help="多 run 等量采样：每个 run 每 epoch 贡献相同帧数"
                         "（小 run 循环补齐），防止帧数多的 run 数值主导"
                         "梯度（v9 教训：稀疏 run 淹没密集 run）")
    ap.add_argument("--line-weight", type=float, default=2.0,
                    help="extra multiplier on the line class loss weight")
    ap.add_argument("--run-weights", default="",
                    help="按 run 的采样权重，如 \"0=2.0,1=0.5\"（下标按 "
                         "训练目录顺序）。总帧数/批数不变 ⇒ 步数与对照臂相同；"
                         "用于困难样本/场景配比这类数据因子")
    ap.add_argument("--max-train-frames", type=int, default=0,
                    help="把训练帧按各 run 配额截到 N 帧（等步数对照用；0=不截）。"
                         "验证集不受影响")
    ap.add_argument("--ignore-line-class", action="store_true",
                    help="缺可信标线真值的帧：把 line 类从 softmax 分母里去掉"
                         "（既无正样本也无负样本），只训练路面/背景通道；"
                         "判据来自标签审计，不是命令行声明")
    ap.add_argument("--paint-source", default="engine_annotation",
                    choices=("engine_annotation", "human_revision", "pseudo"),
                    help="标记真值来源；只有审计判为可用时才监督 line 通道")
    ap.add_argument("--line-tversky-weight", type=float, default=1.0,
                    help="line-channel Tversky term weight (FN>FP, thin-line "
                         "recall); 0 disables and falls back to pure "
                         "weighted CE")
    ap.add_argument("--line-cldice-weight", type=float, default=1.0,
                    help="line-channel soft-clDice term weight (keeps the "
                         "predicted line connected); 0 disables")
    ap.add_argument("--line-tversky-alpha", type=float, default=0.3,
                    help="Tversky FP weight; alpha=beta=0.5 即无偏 Dice")
    ap.add_argument("--line-tversky-beta", type=float, default=0.7,
                    help="Tversky FN weight (beta>alpha 提细线召回,但实测"
                         "会把配对中心拉偏,见 logs/_task_v13_cldice.json)")
    # 任务指标早停：本栈上 val_mIoU 与成对率反相关，按 mIoU 选检查点会挑错
    # epoch（docs/lateral_reference_diag_20260911.md §15）。
    ap.add_argument("--task-eval-every", type=int, default=0, metavar="N",
                    help="每 N 轮用配对任务（scripts/m5_seg_task_eval.py）评估"
                         "当前权重，保留 best_task.pt；0=关（默认）")
    ap.add_argument("--task-episodes", type=int, default=4,
                    help="任务评估取最新 N 个影子 episode（默认 4）")
    ap.add_argument("--task-episode-names", nargs="*", default=None,
                    help="固定任务评估的 episode 文件名：影子集随每次实车增长，"
                         "不固定就无法跨训练比较")
    ap.add_argument("--task-min-in-lane", type=float, default=0.20,
                    metavar="F",
                    help="best_task.pt 的准入门槛：成对率再高，若该轮的 in-lane "
                         "低于 F 也不选（可用率与横向正确性在 epoch 间互相交换，"
                         "只按成对率会选中「配对多但中心错」的权重，默认 0.20）")
    ap.add_argument("--line-morph", action="store_true",
                    help="enable line morphology augmentation (default "
                         "off: v7 ablation showed it hurts line IoU)")
    ap.add_argument("--no-amp", action="store_true",
                    help="disable mixed-precision (AMP) training")
    ap.add_argument("--out", default=str(config.LOGS_DIR / "m5_seg" / "seg_model"))
    ap.add_argument("--resume", default=None, metavar="CHECKPOINT",
                    help="从 checkpoint_last.pt 续训（跳过已完成的 epoch）")
    ap.add_argument("--init", default=None, metavar="CHECKPOINT",
                    help="只加载模型权重作为初始化；优化器/学习率/epoch 从头开始")
    ap.add_argument("--seed", type=int, default=42)
    # --- T14 实验台账（全部可选：不传时行为与以前完全一致）--------------
    ap.add_argument("--dataset-id", default="", metavar="ID",
                    help="数据版本 id（由 experiments.manifest 生成），写进 "
                         "checkpoint 与事件，保证权重能追溯到一个不可变版本")
    ap.add_argument("--dataset-manifest", default=None, metavar="JSON",
                    help="清单文件：只用于记录与校验，不改变读取的 run 列表")
    ap.add_argument("--run-id", default="", metavar="ID",
                    help="实验 run id（事件流的第一个字段）")
    ap.add_argument("--candidate-id", default="", metavar="ID",
                    help="候选 id（一个因子一个候选）")
    ap.add_argument("--events", default=None, metavar="JSONL",
                    help="往该事件文件追加 epoch_end 事件（T14 事件协议）")
    # --- 训练过程可视化（可选；不传 --metrics-run 时不写任何指标）--------
    ap.add_argument("--metrics-run", default=None, metavar="RUN_ID",
                    help="逐 step 指标写进 logs/experiments/<RUN_ID>/metrics.jsonl，"
                         "供 scripts/m5_train_monitor.py 实时查看")
    ap.add_argument("--task-name", default="", metavar="NAME",
                    help="看板状态栏显示的任务名（默认用 run id）")
    ap.add_argument("--monitor-interval", type=float, default=2.0, metavar="S",
                    help="硬件采样周期（秒），默认 2 s：GPU 功耗/利用率/显存、"
                         "CPU 利用率、系统内存")
    ap.add_argument("--stop-after", type=int, default=None, metavar="N",
                    help="跑到第 N 轮就落盘退出（用于验证『中断续训≈未中断』："
                         "调度器仍按 --epochs 建，所以学习率计划与未中断时一致；"
                         "改 --epochs 会改变 LR 计划，那不是中断，是换配方）")
    ap.add_argument("--save-every-epoch", action="store_true",
                    help="每个 epoch 另存 epoch_XX.pt。方案 §4 要求『保存所有被任务"
                         "评估的 epoch』，便于按下游指标重选；也是定位"
                         "『续训≠未中断』差异在哪一步的唯一办法")
    ap.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto",
                    help="训练设备。auto=有 CUDA 就用 CUDA；cpu 用于确定性"
                         "核对（见 --deterministic：CUDA 上 nll_loss2d 没有确定性"
                         "实现，逐位比较只能在 CPU 上做）")
    ap.add_argument("--deterministic", action="store_true",
                    help="强制确定性内核（cudnn.deterministic + "
                         "use_deterministic_algorithms + CUBLAS workspace）："
                         "T14 要求『中断续训≈未中断』可验证；默认关闭，因为"
                         "它会让训练变慢，且部分算子没有确定性实现会直接报错")
    ap.add_argument("--vram-frac", type=float, default=0.6,
                    help="max fraction of GPU VRAM training may use; keeps "
                         "headroom for the running game so its rendering "
                         "never starves (white windows)")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    if args.deterministic:
        # 必须在 CUDA 初始化之前设：否则 cuBLAS 仍会用非确定性 GEMM
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
            print("[train] 确定性模式已开（cudnn.deterministic + "
                  "use_deterministic_algorithms + CUBLAS workspace）",
                  flush=True)
        except Exception as exc:             # noqa: BLE001
            print(f"[train] 确定性模式打开失败：{exc}", flush=True)
            raise
    torch.manual_seed(args.seed)
    if args.deterministic:
        import random as _random
        _random.seed(args.seed)              # python/numpy 也固定，方便比对
        np.random.seed(args.seed)
    if args.device == "cpu":
        device = "cpu"
    elif args.device == "cuda":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device}", flush=True)

    # --- T14 台账：事件流 + 候选身份（不传 --events 时全为空操作）---------
    events = None
    _event = None
    if args.events:
        from beamng_autopilot.experiments.events import Event, EventLog
        _log = EventLog(Path(args.events).parent)
        _log.path = Path(args.events)
        events = _log

        def _event(phase, status, **kw):            # noqa: ANN001
            return Event(run_id=args.run_id or Path(args.events).parent.name,
                         candidate_id=args.candidate_id or "adhoc",
                         dataset_id=args.dataset_id or "unversioned",
                         config_hash=f"{args.lr:g}/{args.batch}/"
                                     f"{args.line_tversky_weight:g}",
                         seed=int(args.seed), phase=phase, status=status,
                         **kw)

        # 断电重启后回到 training 阶段：上一次的事件里没有终态就补一条
        last = events.last()
        if last is None or last.phase != "training":
            events.append(_event("queued", "start",
                                 note=f"runs={len(args.runs)} "
                                      f"epochs={args.epochs}"))
            events.append(_event("auditing", "ok",
                                 note=f"dataset_id={args.dataset_id or '-'} "
                                      f"manifest={args.dataset_manifest or '-'}"))
            events.append(_event("training", "started",
                                 note=f"batch={args.batch} lr={args.lr:g}"))
    if device == "cuda":
        # Cap training VRAM so a concurrently running game keeps enough
        # memory to render.  Without this, batch 16 training filled the
        # whole 12 GB card and the game's window went white.
        torch.cuda.set_per_process_memory_fraction(
            max(0.1, min(1.0, args.vram_frac)))

    line_only_dirs = {Path(p).name for p in args.line_only_runs}
    train_only_runs = {Path(p).name for p in args.train_only_runs}
    frames, per_run = load_frames([Path(p) for p in args.runs],
                                  args.min_line_frac,
                                  line_only_dirs=line_only_dirs,
                                  thin_line=int(args.thin_line_labels or 0))
    n = len(frames)
    train_frames, val_frames = split_frames(
        frames, per_run, args.split if args.split != "by-map-scene" else "tail",
        args.val_frac, train_only_runs=train_only_runs)
    if args.split == "by-map-scene":
        # Plan E7: split by map/scene group (the run name carries the
        # scene - and the map when the run recorded one), never by
        # shuffling frames, and AUDIT the result: a group on both sides or
        # a map with no validation frames is reported, not trusted.
        refs = []
        meta_notes: list[str] = []
        for _name, _rec in per_run.items():
            _fracs = _rec.get("fracs") or []
            _meta = _rec.get("meta")
            if _meta:
                # identity comes from the recording; the INDEX comes from
                # the kept sequence, because min_line_frac filtering drops
                # frames and the meta's own counter knows nothing about it
                _ident, _n = frame_refs_from_meta(_meta, run=_name)
                meta_notes.extend(_n)
                if _rec.get("meta_error"):
                    meta_notes.append(f"{_name}: meta unreadable "
                                      f"({_rec['meta_error']})")
                if len(_ident) < _rec["kept"]:
                    meta_notes.append(
                        f"{_name}: meta has {len(_ident)} frames but "
                        f"{_rec['kept']} were kept; the extra frames use the "
                        f"run name only")
                for _i in range(_rec["kept"]):
                    _base = _ident[_i] if _i < len(_ident) else FrameRef()
                    refs.append(FrameRef(
                        index=_rec["start"] + _i, run=_name,
                        map_name=_base.map_name, source_id=_base.source_id,
                        t=_base.t, t_wall=_base.t_wall,
                        t_is_index=_base.t_is_index, exposure=_base.exposure,
                        view=_base.view,
                        line_frac=(_fracs[_i] if _i < len(_fracs) else 0.0)))
            else:
                meta_notes.append(f"{_name}: no meta.json; the group is the "
                                  f"directory name and t is a frame index")
                refs.extend(FrameRef(index=_rec["start"] + _i, run=_name,
                                     t=float(_i), t_is_index=True,
                                     line_frac=(_fracs[_i]
                                                if _i < len(_fracs)
                                                else 0.0))
                            for _i in range(_rec["kept"]))
        if meta_notes:
            print(f"[train] 身份回退 {len(meta_notes)} 条 -> "
                  f"{meta_notes[:4]}", flush=True)
        plan = split_by_group(refs, val_frac=args.val_frac)
        train_frames = [frames[r.index] for r in plan.train]
        val_frames = [frames[r.index] for r in plan.val]
        _leak = leak_check(plan)
        _cov = coverage_digest(plan)
        print(f"[train] split by-map-scene: {len(plan.groups_train)} 训练组 / "
              f"{len(plan.groups_val)} 验证组, "
              f"帧重叠={len(_leak['leaked_frames'])}, "
              f"共享组={len(_leak['leaked_groups'])}",
              flush=True)
        if _cov["maps_without_val"]:
            print(f"[train] 注意：以下组没有验证帧 -> "
                  f"{_cov['maps_without_val']}", flush=True)
        # Cross-view: the same exposure seen by two mounts is ONE instant,
        # so those frames must stay on one side (T10).  No exposure
        # counters means the check could not run - reported as such, never
        # as "no leak".
        _xv = cross_view_leak(plan, refs)
        if not _xv["checked"]:
            print(f"[train] 跨视角检查未执行：{_xv['reason']}", flush=True)
        elif _xv["leaked_groups"]:
            print(f"[train] 严重：同一曝光的多视角被切到两侧 "
                  f"{_xv['leaked_groups'][:6]}", flush=True)
        else:
            print(f"[train] 跨视角同曝光分组 "
                  f"{_xv['n_cross_view_groups']} 组，无跨侧泄漏", flush=True)
        # Duplicated samples and temporal adjacency, reported SEPARATELY
        # (plan T10): a copied frame inflates one side while looking like
        # two samples, and two frames a fraction of a second apart are the
        # same moment of the drive.  Each section says whether it could be
        # checked at all.
        _audit = split_audit(plan, refs, gap_s=float(args.split_gap_s))
        _dup = _audit["duplicates"]
        _adj = _audit["temporal_adjacency"]
        if not _dup["checked"]:
            print(f"[train] 复制样本检查未执行：{_dup['reason']}", flush=True)
        elif _dup["n_crossing"]:
            print(f"[train] 严重：疑似同一批样本被切到两侧 "
                  f"{_dup['detail']}", flush=True)
        else:
            print(f"[train] 复制样本分组 {_dup['n_groups']} 组，"
                  f"无跨侧", flush=True)
        if not _adj["checked"]:
            print(f"[train] 时间邻近检查未执行：{_adj['reason']}", flush=True)
        elif _adj["n_crossing"]:
            print(f"[train] 注意：{_adj['n_crossing']} 对相差 ≤"
                  f"{_adj['gap_s']}s 的相邻帧被切到两侧 -> "
                  f"{_adj['detail'][:3]}", flush=True)
        else:
            print(f"[train] 时间邻近 {_adj['n_pairs']} 对（≤{_adj['gap_s']}s），"
                  f"无跨侧", flush=True)
        # Two different statements (plan T10): a FRAME on both sides is a
        # hard leak; a GROUP on both sides is what the temporal-tail
        # protocol does by construction and must be named as such instead
        # of being reported as "no leak" (or lumped in with the former).
        if _leak["leaked_frames"]:
            print(f"[train] 严重：同一帧出现在两侧 "
                  f"{_leak['leaked_frames'][:8]}", flush=True)
        if _leak["leaked_groups"]:
            print(f"[train] 协议提醒：{len(_leak['leaked_groups'])} 个组被"
                  f"时间尾切分（这是 temporal-tail 开发验证，不是 episode/组"
                  f"隔离；要组隔离请用 holdout_groups）-> "
                  f"{_leak['leaked_groups'][:6]}", flush=True)
    if args.weak_line_oversample > 0 and train_frames:
        _refs = [FrameRef(index=i, run="", t=float(i),
                          line_frac=float((lb == 2).mean()))
                 for i, (_img, lb) in enumerate(train_frames)]
        _weak = select_weak_lines(_refs, low=args.weak_line_low,
                                  high=args.weak_line_high)
        if _weak:
            _pick = [train_frames[r.index] for r in _weak]
            for _ in range(int(args.weak_line_oversample)):
                train_frames.extend(_pick)
            print(f"[train] weak-line oversample: {len(_pick)} 帧 x"
                  f"{int(args.weak_line_oversample)} 已加入训练集",
                  flush=True)
        else:
            print(f"[train] weak-line band {args.weak_line_low}.."
                  f"{args.weak_line_high}: 没有符合条件的帧", flush=True)
    train_bounds = train_run_bounds(
        frames, per_run, args.split, args.val_frac,
        train_only_runs=train_only_runs)
    if args.max_train_frames:
        before = len(train_frames)
        train_frames, train_bounds, cap_note = cap_train_frames(
            train_frames, train_bounds, int(args.max_train_frames),
            seed=args.seed)
        print(f"[train] --max-train-frames={args.max_train_frames}: "
              f"训练帧 {before} -> {len(train_frames)}（按 run 配额，"
              f"验证集不变）", flush=True)
    print(f"[train] 共 {n} 帧: 训练 {len(train_frames)} / 验证 {len(val_frames)}",
          flush=True)
    if args.balance_runs:
        n_max = max((e - s for s, e in train_bounds if e > s), default=0)
        print(f"[train] balance_runs: 每轮每 run 等量采样 "
              f"(各 {n_max} 帧/run, 共 "
              f"{len(train_bounds) * n_max} 帧/轮)", flush=True)
    print(f"[train] 类别: {CLASS_NAMES}", flush=True)
    for name, rec in sorted(per_run.items()):
        drop = rec["frames"] - rec["kept"]
        print(f"[train] run {name:<24} 帧{rec['frames']:>4} "
              f"保留{rec['kept']:>4} 丢弃{drop:>4} "
              f"line_px_frac={rec['line_px_frac']:.5f}", flush=True)

    weights = median_freq_weights(train_frames,
                                   line_weight=args.line_weight)
    print(f"[train] 类别权重: {weights.tolist()}", flush=True)

    model = SegUNet().to(device)
    if args.init:
        init_ckpt = torch.load(args.init, map_location=device,
                               weights_only=False)
        model.load_state_dict(init_ckpt["state_dict"])
        print(f"[train] 初始化模型权重: {args.init}（优化器从头开始）",
              flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    # 加权 CE 之外对 line 通道再加两个区域项：Tversky(FN>FP, 细线召回)
    # + soft-clDice(连通性, 治断线帧)。权重 0 即回退历史纯 CE 行为。
    from beamng_autopilot.vision.seg_losses import LineSegLoss
    crit = LineSegLoss(weight=weights.to(device), ignore_index=255,
                       w_tversky=args.line_tversky_weight,
                       w_cldice=args.line_cldice_weight,
                       tversky_alpha=args.line_tversky_alpha,
                       tversky_beta=args.line_tversky_beta)
    if args.line_tversky_weight > 0 or args.line_cldice_weight > 0:
        print(f"[train] line region loss: tversky={args.line_tversky_weight} "
              f"cldice={args.line_cldice_weight}", flush=True)
    if args.ignore_line_class:
        print(f"[train] --ignore-line-class：paint_source={args.paint_source}；"
              f"审计判为不可信的帧其 line 类被整通道屏蔽（整批屏蔽时区域项"
              f"自动跳过，混批会报错要求拆批）", flush=True)
    use_amp = (device == "cuda") and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def to_tensor(frame, dev):
        colour, label = frame
        # All sources must share one batch shape: Tech captures are
        # 536x403 while the user's manual line frames are 400x300.
        # Resize RGB smoothly and labels with nearest-neighbour so class
        # ids (including ignore=255) remain intact.
        import cv2
        colour = np.asarray(colour, dtype=np.uint8)
        label = np.asarray(label, dtype=np.uint8)
        if colour.shape[:2] != (403, 536):
            colour = cv2.resize(colour, (536, 403),
                                interpolation=cv2.INTER_AREA)
        if label.shape[:2] != (403, 536):
            label = cv2.resize(label, (536, 403),
                               interpolation=cv2.INTER_NEAREST)
        x = torch.from_numpy(colour).permute(2, 0, 1).float() / 255.0
        y = torch.from_numpy(label).long()
        return x.to(dev), y.to(dev)

    # 训练过程可视化：逐 step 指标与硬件采样（mon["store"] 为 None 时全为空操作）
    mon: dict = {"store": None, "sampler": None, "step": 0,
                 "steps_per_epoch": 0}
    # 被整通道屏蔽的帧数（证据：这一轮到底有没有真的屏蔽掉 line 监督）
    _IGN = {"frames": 0}

    def monitor_step(batch_i: int, ep: int, *, loss_v: float, acc_v: float,
                     lr_v: float, step_s: float) -> None:
        """写一条 train 记录；梯度范数在 backward 之后单独算（见调用处）。"""
        store = mon["store"]
        if store is None:
            return
        mon["step"] += 1
        _MON["step"] = mon["step"]
        _MON["epoch"] = ep
        store.append({"kind": "train", "step": mon["step"], "epoch": ep,
                      "loss": float(loss_v), "acc": float(acc_v),
                      "grad_norm": mon.pop("pending_grad_norm", None),
                      "lr": float(lr_v), "step_s": float(step_s),
                      "run_id": args.metrics_run})

    run_weights: dict = {}
    for _part in str(args.run_weights or "").split(","):
        _part = _part.strip()
        if not _part or "=" not in _part:
            continue
        _k, _v = _part.split("=", 1)
        try:
            run_weights[_k.strip()] = float(_v)
        except ValueError:
            raise SystemExit(f"--run-weights 解析失败: {_part!r}")
    if run_weights:
        print(f"[train] run 采样权重: {run_weights}"
              + ("（balance_runs 先按等量采样，权重不再叠加）"
                 if args.balance_runs else "（总帧数不变）"), flush=True)

    def run_epoch(fr, train: bool, ep: int = 0, balance_runs: bool = False,
                  run_bounds: list[tuple[int, int]] | None = None):
        model.train(train)
        total_loss, correct, n_pix = 0.0, 0, 0
        inter = np.zeros(N_CLASSES)
        union = np.zeros(N_CLASSES)
        # 每轮独立随机种子：旧写法 seed 只看 len(fr) 恒定不变，导致
        # 每轮增强/洗牌序列完全一致，等于数据没有随机化。
        rng = np.random.default_rng(args.seed + ep * 1000003 + len(fr))
        if train and balance_runs and run_bounds:
            order = balanced_indices(run_bounds, rng)
        elif train and run_weights and run_bounds:
            order, _w_note = weighted_run_indices(run_bounds, run_weights, rng)
            if ep == 0:
                print(f"[train] 加权采样配额/run: {_w_note['quota']} "
                      f"(总 {_w_note['total']} 帧)", flush=True)
        elif train:
            order = rng.permutation(len(fr))
        else:
            order = np.arange(len(fr))
        for i in range(0, len(order), args.batch):
            _t_step0 = time.perf_counter()
            batch = [fr[int(j)] for j in order[i:i + args.batch]]
            if train:
                batch = [_augment(f, rng, line_morph=args.line_morph)
                         for f in batch]
            xs = torch.stack([to_tensor(f, device)[0] for f in batch])
            ys = torch.stack([to_tensor(f, device)[1] for f in batch])
            if train:
                opt.zero_grad()
            # Validation must not build autograd graphs: it halves the
            # peak VRAM and keeps a concurrently running game rendering.
            with torch.set_grad_enabled(train), \
                    torch.autocast("cuda", enabled=use_amp):
                logits = model(xs)
                ys_eval = ys
                cmask = None
                if args.ignore_line_class:
                    flags = line_supervision_flags(ys, args.paint_source)
                    if not all(flags):
                        ys_eval = ys.clone()
                        cmask = torch.ones((ys.shape[0], N_CLASSES),
                                           dtype=torch.bool, device=device)
                        for _b, _ok in enumerate(flags):
                            if _ok:
                                continue
                            # 没有可信 line 真值：目标里的 line 像素去掉（否则
                            # 损失里出现"被屏蔽的类当目标"），并把该类移出分母
                            ys_eval[_b][ys[_b] == 2] = 255
                            cmask[_b, 2] = False
                        _IGN["frames"] += sum(1 for _ok in flags if not _ok)
                loss = crit(logits, ys_eval, class_mask=cmask)
            if train:
                scaler.scale(loss).backward()
                # 梯度范数要在 unscale 之后算：AMP 下 .backward() 的梯度带了
                # scaler 缩放因子，直接统计会得到一个随缩放漂移的假数。
                if mon["store"] is not None:
                    if use_amp:
                        scaler.unscale_(opt)
                    _tot = 0.0
                    for _p in model.parameters():
                        if _p.grad is not None:
                            # float64 累计：fp32 下平方可能溢出成 inf，而
                            # 非有限值既不能进 JSON 也不是有效观测
                            _tot += float(
                                _p.grad.detach().double().pow(2).sum())
                    _gn = math.sqrt(_tot) if math.isfinite(_tot) else None
                    mon["pending_grad_norm"] = _gn
                scaler.step(opt)
                scaler.update()
                monitor_step(i, ep, loss_v=float(loss.item()),
                             acc_v=(float(((logits.argmax(dim=1) == ys_eval)
                                           & (ys_eval != 255)).sum())
                                    / max(1, int((ys_eval != 255).sum()))),
                             lr_v=float(sched.get_last_lr()[-1]),
                             step_s=time.perf_counter() - _t_step0)
            _loss_v = float(loss.item())
            if not math.isfinite(_loss_v):
                # 方案 §4 要求 NaN 立即失败：继续跑只会把非有限权重写进
                # checkpoint，而且"跑完 40 轮"看起来像成功。
                raise FloatingPointError(
                    f"非有限损失 loss={_loss_v}（epoch {ep}, step "
                    f"{mon['step'] + 1}, lr={sched.get_last_lr()[-1]:g}）"
                    f"：按 T14 §4 立即失败，已保留此前日志")
            total_loss += _loss_v * len(batch)
            pred = logits.argmax(dim=1)
            ys_eval = ys_eval if isinstance(ys_eval, torch.Tensor) else ys
            valid = ys_eval != 255
            correct += int(((pred == ys_eval) & valid).sum())
            n_pix += int(valid.sum())
            for c in range(N_CLASSES):
                p = (pred == c) & valid
                t = (ys_eval == c) & valid
                inter[c] += int((p & t).sum())
                union[c] += int((p | t).sum())
        ious = iou_from_accum(inter, union)
        present = union > 0
        return (total_loss / max(1, len(fr)),
                correct / max(1, n_pix), ious, present)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    start_ep = 0
    best_miou = -1.0
    hist = {"epoch": [], "train_loss": [], "val_miou": [], "val_acc": []}
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt and ckpt.get("scaler") is not None and use_amp:
            scaler.load_state_dict(ckpt["scaler"])
        start_ep = int(ckpt.get("next_epoch", args.epochs))
        best_miou = float(ckpt.get("best_miou", -1.0))
        hist = ckpt.get("hist", hist)
        # T14：随机状态与数据版本必须一起恢复，否则"续训"与"未中断"不等价
        from beamng_autopilot.experiments.checkpoint import (
            missing_extras, restore_rng)
        _missing = missing_extras(ckpt)
        if _missing:
            print(f"[train] 注意：该 checkpoint 缺恢复字段 {_missing}；"
                  f"续训将按当前进程状态继续（不声称与未中断逐位一致）",
                  flush=True)
        else:
            _ok = restore_rng(ckpt)
            print(f"[train] 随机状态与数据版本已恢复（dataset_id="
                  f"{ckpt.get('dataset_id')}, rng_ok={_ok}）", flush=True)
        print(f"[train] 从 {args.resume} 续训, 从 epoch {start_ep} 继续 "
              f"(共 {args.epochs})", flush=True)

    # 任务指标校验（opt-in）：本栈上 val_mIoU 与配对任务**反相关**
    # （`scripts/m5_seg_task_eval.py`、docs/lateral_reference_diag_20260911.md
    # §15），所以按 mIoU 选 best.pt 会挑到错误的 epoch。--task-eval-every N
    # 每 N 轮用同一个评估口径量一次成对率并保留 best_task.pt；episode 列表
    # 在训练前解析一次，训练中不漂移（影子集会随每次实车增长）。
    task_eps = None
    task_measure = None
    best_task = float(ckpt.get("best_task", -1.0)) if start_ep else -1.0
    if args.task_eval_every > 0:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from m5_seg_task_eval import _episodes as _task_eps
        from m5_seg_task_eval import measure as _task_measure
        task_measure = _task_measure
        task_eps = _task_eps(config.LOGS_DIR / "m5_e2e", "shadow_fsd_*.npz",
                             args.task_episode_names, args.task_episodes)
        hist.setdefault("task_paired", [])
        hist.setdefault("task_in_lane", [])
        print(f"[train] 任务评估每 {args.task_eval_every} 轮："
              f"{len(task_eps)} 个 episode"
              f"{'（固定）' if args.task_episode_names else '（最新 N）'}",
              flush=True)

    # --- 训练过程可视化：任务记录 + 硬件采样线程（未启用时全为空操作）-----
    if args.metrics_run:
        from beamng_autopilot.experiments.metrics import (
            MetricsStore, SystemSampler, task_record)
        _mstore = MetricsStore(config.LOGS_DIR / "experiments"
                               / args.metrics_run)
        mon["store"] = _mstore
        mon["steps_per_epoch"] = max(1, int(math.ceil(
            len(train_frames) / max(1, args.batch))))
        _total_steps = mon["steps_per_epoch"] * int(args.epochs)
        _name = args.task_name or args.metrics_run
        _mstore.append(task_record(
            args.metrics_run, _name, "running", total_steps=_total_steps,
            current_step=mon["steps_per_epoch"] * int(start_ep),
            epoch=start_ep, started_at=time.time(),
            extra={"batch": args.batch, "lr": args.lr, "seed": args.seed,
                   "epochs": args.epochs, "resume_from": bool(args.resume),
                   "steps_per_epoch": mon["steps_per_epoch"],
                   "n_train": len(train_frames), "n_val": len(val_frames)}))
        mon["sampler"] = SystemSampler(
            _mstore, interval_s=args.monitor_interval,
            run_id=args.metrics_run).start()
        _MON.update({"store": _mstore, "sampler": mon["sampler"],
                     "args": args, "step": 0,
                     "total_steps": _total_steps})
        print(f"[train] 指标 -> {_mstore.path}（逐 step；硬件每 "
              f"{args.monitor_interval:g}s 采样）", flush=True)

    for ep in range(start_ep, args.epochs):
        t0 = time.time()
        tr_loss, tr_acc, _, _ = run_epoch(
            train_frames, train=True, ep=ep,
            balance_runs=args.balance_runs, run_bounds=train_bounds)
        sched.step()
        va_loss, va_acc, va_ious, va_present = run_epoch(
            val_frames, train=False, ep=ep)
        m_iou = (float(va_ious[va_present].mean())
                 if va_present.any() else 0.0)
        # 目标指标单独入史：line 类 IoU 是本训练真正要抬的量,mIoU 被路面
        # 类主导,看不出 line 的起落。
        line_iou_ep = float(va_ious[2]) if va_present[2] else None
        if args.ignore_line_class and _IGN["frames"]:
            # 指标侧同样屏蔽：line 通道被整通道屏蔽时，这轮的"标线 IoU"是拿
            # 引擎参考算的（这批地图上引擎 line 类不可信，且左侧漆线常常没有
            # 标注）——按方案 §1"缺真值时标线指标必须停止或屏蔽"记为 None，
            # 不做成数字（否则"未测"会被读成"测到很差"）。
            line_iou_ep = None
        hist["epoch"].append(ep)
        hist["train_loss"].append(round(tr_loss, 4))
        hist["val_miou"].append(round(m_iou, 4))
        hist["val_acc"].append(round(va_acc, 4))
        if args.ignore_line_class:
            hist.setdefault("line_ignored_frames", []).append(
                int(_IGN["frames"]))
        hist.setdefault("val_line_iou", []).append(
            None if line_iou_ep is None else round(line_iou_ep, 4))
        if mon["store"] is not None:
            mon["store"].append({
                "kind": "epoch", "epoch": ep, "step": mon["step"],
                "run_id": args.metrics_run,
                "train_loss": float(tr_loss), "train_acc": float(tr_acc),
                "val_loss": float(va_loss), "val_acc": float(va_acc),
                "val_miou": float(m_iou),
                "val_line_iou": (None if line_iou_ep is None
                                 else float(line_iou_ep)),
                "not_applicable": ({"val_line_iou": (
                    "line 通道被整通道屏蔽（缺可信漆线真值），标线指标未测"
                    if (args.ignore_line_class and _IGN["frames"])
                    else "验证集没有 line 类像素")}
                    if line_iou_ep is None else None)})
        print(f"[train] ep {ep:02d}  loss={tr_loss:.4f} "
              f"val_acc={va_acc:.3f} val_mIoU={m_iou:.4f} "
              f"val_lineIoU={'n/a' if line_iou_ep is None else round(line_iou_ep, 4)} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if args.stop_after is not None and (ep + 1) >= int(args.stop_after):
            print(f"[train] --stop-after={args.stop_after}：在 epoch {ep} 后"
                  f"落盘并退出（模拟中断；--epochs 未变，LR 计划一致）",
                  flush=True)
            break
        if events is not None:
            # T14 事件协议：每个 epoch 一条 epoch_end，指标带单位与分子/分母；
            # 缺测写 missing 而不是 0（看板据此画断点而不是假 0）。
            from beamng_autopilot.experiments.events import metric as _metric
            events.append(_event(
                phase="training", status="epoch_end", epoch=ep,
                step=(ep + 1) * max(1, len(train_frames) // max(1, args.batch)),
                metrics={
                    "train_loss": _metric(tr_loss, "loss"),
                    "val_miou": _metric(m_iou, "miou"),
                    "val_acc": _metric(va_acc, "acc"),
                    "val_line_iou": _metric(
                        None if line_iou_ep is None else line_iou_ep, "iou",
                        missing="" if line_iou_ep is not None else
                        "line class absent from the validation slice"),
                    "lr": _metric(sched.get_last_lr()[-1], "1"),
                }))
        if m_iou > best_miou:
            best_miou = m_iou
            torch.save({
                "state_dict": model.state_dict(),
                "n_classes": N_CLASSES,
                "class_names": CLASS_NAMES,
                "val_miou": round(m_iou, 4),
                "val_ious": [round(float(v), 4) for v in va_ious],
                "val_acc": round(float(va_acc), 4),
                "weights": weights.tolist(),
                # 超参随模型落盘：复现/对比不同 --line-weight 轮次有据可查
                "train_args": {
                    "line_weight": args.line_weight,
                    "line_tversky_weight": args.line_tversky_weight,
                    "line_cldice_weight": args.line_cldice_weight,
                    "line_tversky_alpha": args.line_tversky_alpha,
                    "line_tversky_beta": args.line_tversky_beta,
                    "line_morph": args.line_morph,
                    "amp": use_amp,
                    "epochs": args.epochs,
                    "batch": args.batch,
                    "lr": args.lr,
                    "val_frac": args.val_frac,
                    "seed": args.seed,
                    "runs": [str(p) for p in args.runs],
                    "split": args.split,
                    "min_line_frac": args.min_line_frac,
                    "balance_runs": args.balance_runs,
                    "n_train": len(train_frames),
                    "n_val": len(val_frames),
                },
            }, out_dir / "best.pt")
            print(f"[train] 保存最优 mIoU={m_iou:.4f} -> "
                  f"{out_dir / 'best.pt'}", flush=True)
        if args.save_every_epoch:
            # 逐 epoch 快照：方案要求保存所有被评估过的 epoch（按下游指标重选），
            # 也是"续训到底哪一步开始不同"的定位手段
            torch.save({
                "state_dict": model.state_dict(),
                "next_epoch": ep + 1, "epoch": ep,
                "best_miou": best_miou, "hist": hist,
                "dataset_id": args.dataset_id or "unversioned",
                "train_args": {"epochs": args.epochs, "batch": args.batch,
                               "lr": args.lr, "seed": args.seed},
            }, out_dir / f"epoch_{ep:02d}.pt")
        ckpt = {
            "state_dict": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "scaler": scaler.state_dict() if use_amp else None,
            "next_epoch": ep + 1,
            "best_miou": best_miou,
            "best_task": best_task,
            "hist": hist,
            "train_args": {
                "line_weight": args.line_weight,
                "line_tversky_weight": args.line_tversky_weight,
                "line_cldice_weight": args.line_cldice_weight,
                "line_tversky_alpha": args.line_tversky_alpha,
                "line_tversky_beta": args.line_tversky_beta,
                "line_morph": args.line_morph,
                # 证据：这一轮是否真的屏蔽了 line 监督、依据哪个真值来源
                "ignore_line_class": bool(args.ignore_line_class),
                "paint_source": args.paint_source,
                "line_ignored_frames": int(_IGN["frames"]),
                "amp": use_amp,
                "epochs": args.epochs,
                "batch": args.batch,
                "lr": args.lr,
                "val_frac": args.val_frac,
                "seed": args.seed,
                "runs": [str(p) for p in args.runs],
                "split": args.split,
                "min_line_frac": args.min_line_frac,
                "balance_runs": args.balance_runs,
                "n_train": len(train_frames),
                "n_val": len(val_frames),
                "max_train_frames": int(args.max_train_frames),
                "run_weights": dict(run_weights),
            },
            # T14：恢复所需的随机状态/数据版本/git/环境，缺一项就不能声称
            # "中断续训 = 未中断"。旧 checkpoint 读起来仍然兼容。
            **checkpoint_extras(dataset_id=args.dataset_id or "unversioned",
                                candidate_id=args.candidate_id,
                                run_id=args.run_id),
        }
        torch.save(ckpt, out_dir / "checkpoint_last.pt")

        # 任务指标：按成对率（而非 mIoU）保留这一轮权重。评估吃刚落盘的
        # checkpoint_last.pt，因此与最终可复现的权重完全一致。
        t_paired = t_in_lane = None
        if task_eps and (ep + 1) % args.task_eval_every == 0:
            try:
                tv = task_measure(str(out_dir / "checkpoint_last.pt"), task_eps)
                t_paired = tv.get("paired_rate")
                t_in_lane = tv.get("in_lane_rate")
                print(f"[train]     task paired={t_paired:.1%} "
                      f"in_lane={t_in_lane:.1%} "
                      f"lat_p50={tv.get('lat_p50_m')}m", flush=True)
                # 每个已评估的 epoch 都留一份权重：可用率与横向正确性在 epoch
                # 之间互相交换，改选门槛后必须能离线重选，不必重训。
                torch.save({
                    "state_dict": model.state_dict(),
                    "n_classes": N_CLASSES,
                    "class_names": CLASS_NAMES,
                    "val_miou": round(m_iou, 4),
                    "task_paired_rate": t_paired,
                    "task_in_lane_rate": t_in_lane,
                    "task_lat_p50_m": tv.get("lat_p50_m"),
                    "task_episodes": [Path(e).name for e in task_eps],
                    "weights": weights.tolist(),
                    "train_args": ckpt["train_args"],
                }, out_dir / f"task_ep{ep:02d}.pt")
            except Exception as exc:
                print(f"[train] 任务评估失败: {exc}", flush=True)
        if task_eps:
            hist["task_paired"].append(t_paired)
            hist["task_in_lane"].append(t_in_lane)
        eligible = (t_paired is not None and t_in_lane is not None
                    and t_in_lane >= args.task_min_in_lane)
        if eligible and t_paired > best_task:
            best_task = float(t_paired)
            torch.save({
                "state_dict": model.state_dict(),
                "n_classes": N_CLASSES,
                "class_names": CLASS_NAMES,
                "val_miou": round(m_iou, 4),
                "task_paired_rate": t_paired,
                "task_in_lane_rate": t_in_lane,
                "task_episodes": [Path(e).name for e in task_eps],
                "weights": weights.tolist(),
                "train_args": ckpt["train_args"],
            }, out_dir / "best_task.pt")
            print(f"[train] 保存最优任务成对率={t_paired:.1%} -> "
                  f"{out_dir / 'best_task.pt'}", flush=True)

    (out_dir / "train_hist.json").write_text(
        json.dumps(hist, ensure_ascii=False, indent=1), encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot(hist["epoch"], hist["train_loss"], label="train loss")
        plt.plot(hist["epoch"], hist["val_miou"], label="val mIoU")
        plt.xlabel("epoch")
        plt.legend()
        plt.tight_layout()
        plt.savefig(str(out_dir / "curve.png"), dpi=110)
        print(f"[train] 曲线 -> {out_dir / 'curve.png'}", flush=True)
    except Exception as exc:
        print(f"[train] 曲线保存失败: {exc}", flush=True)

    if not val_frames:
        # With train-only manual runs there is no validation split; the
        # normal best_miou=0.0 comparison would leave best.pt at epoch 0
        # while checkpoint_last.pt contains the learned final weights.
        final_ckpt = torch.load(out_dir / "checkpoint_last.pt",
                                map_location=device, weights_only=False)
        torch.save({
            "state_dict": final_ckpt["state_dict"],
            "n_classes": N_CLASSES,
            "class_names": CLASS_NAMES,
            "val_miou": None, "val_ious": [], "val_acc": None,
            "weights": weights.tolist(),
            "train_args": {"line_weight": args.line_weight,
                           "line_tversky_weight": args.line_tversky_weight,
                           "line_cldice_weight": args.line_cldice_weight,
                           "line_tversky_alpha": args.line_tversky_alpha,
                           "line_tversky_beta": args.line_tversky_beta,
                           "line_morph": args.line_morph,
                           "amp": use_amp, "epochs": args.epochs,
                           "batch": args.batch, "lr": args.lr,
                           "val_frac": args.val_frac, "seed": args.seed,
                           "runs": [str(p) for p in args.runs],
                           "split": args.split,
                           "min_line_frac": args.min_line_frac,
                           "balance_runs": args.balance_runs,
                           "line_only_runs": list(args.line_only_runs),
                           "thin_line_labels": int(args.thin_line_labels or 0),
                           "train_only_runs": list(args.train_only_runs),
                           "n_train": len(train_frames), "n_val": 0},
        }, out_dir / "best.pt")
        print(f"[train] 无验证集，best.pt 使用最终 epoch 权重 -> "
              f"{out_dir / 'best.pt'}", flush=True)
    if mon["store"] is not None:
        mon["sampler"].stop()
        mon["store"].append(task_record(
            args.metrics_run, args.task_name or args.metrics_run, "completed",
            total_steps=mon["steps_per_epoch"] * int(args.epochs),
            current_step=mon["step"], epoch=int(args.epochs),
            started_at=(mon["store"].task() or {}).get("started_at"),
            extra={"best_val_miou": round(float(best_miou), 4),
                   "best_pt": str(out_dir / "best.pt")}))
        print(f"[train] 指标收尾：{mon['step']} 个优化步已记录 -> "
              f"{mon['store'].path}", flush=True)
    print(f"[train] 完成: 最优验证 mIoU={best_miou:.4f} "
          f"-> {out_dir / 'best.pt'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:             # 含 KeyboardInterrupt
        record_training_failure(exc)
        raise
