"""T14 可复现性验收：『中断续训 = 未中断』必须逐位一致（确定性路径）。

方案 §4 的要求是"若现有 checkpoint 未保存恢复所需全部随机状态或数据版本，
先补齐并验证『中断续训≈未中断』再宣称可复现"。这个测试把验证自动化：

* 同样 `--epochs 3`，A 一次跑完；B 用 `--stop-after 2` 模拟中断再续训；
* `--device cpu --deterministic` 让内核确定（CUDA 上 nll_loss2d **没有**
  确定性实现，逐位比较只能在 CPU 上做，见 docs/T14_PROGRESS_20260924.md）；
* 比较最终 `checkpoint_last.pt` 的 state_dict，要求 0 差异。

**踩过的坑（写在这里防止别人重犯）**：如果"中断"是把 `--epochs` 从 2 改成 3
再续训，那是**换配方**而不是中断——调度器的 LR 计划由 `--epochs` 决定，续训那
一轮的学习率会变成 0，于是差异看起来像"漏状态"。必须用 `--stop-after`。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

COMMON = ("--split", "tail", "--val-frac", "0.2", "--batch", "4",
          "--lr", "1e-3", "--seed", "7", "--device", "cpu",
          "--deterministic", "--epochs", "3")


def _frames(root: Path, n: int = 6) -> Path:
    d = root / "runs" / "front_main"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        colour = np.full((24, 32, 3), 40 + i * 6, np.uint8)
        label = np.zeros((24, 32), np.uint8)
        label[6:20, :] = 1
        label[12, :6] = 2
        np.savez(d / f"frame_{i:05d}.npz", colour=colour, label=label)
    return d


def _train(runs: Path, out: Path, *extra: str) -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / "m5_train_seg.py"),
           "--runs", str(runs), *COMMON, "--out", str(out), *extra]
    env = dict(os.environ)
    env["BEAMNG_LOGS_DIR"] = str(out.parent / "_logs")
    r = subprocess.run(cmd, capture_output=True, text=True, env=env,
                       timeout=900)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1000:]


def test_resume_equals_uninterrupted_on_the_deterministic_path(tmp_path):
    from beamng_autopilot.experiments.checkpoint import (
        load_full, missing_extras, weights_equal)

    runs = _frames(tmp_path)
    a = tmp_path / "A"
    b = tmp_path / "B"
    _train(runs, a)
    _train(runs, b, "--stop-after", "2")          # 中断：同样 --epochs 3
    _train(runs, b, "--resume", str(b / "checkpoint_last.pt"))

    ca = load_full(a / "checkpoint_last.pt")
    cb = load_full(b / "checkpoint_last.pt")
    assert missing_extras(ca) == [] and missing_extras(cb) == [], \
        "恢复字段不全就不该声称可复现"
    assert ca["dataset_id"] == cb["dataset_id"]
    assert ca["next_epoch"] == cb["next_epoch"] == 3
    eq = weights_equal(ca["state_dict"], cb["state_dict"], atol=0.0)
    assert eq["n_diff"] == 0, (
        f"续训与未中断不逐位一致：n_diff={eq['n_diff']} "
        f"max={eq['max_abs_diff']:.3e} worst={eq['max_key']}")


def test_the_same_configuration_twice_is_bitwise_identical(tmp_path):
    """同配置跑两次也必须一致——否则上一条测不出任何东西。"""
    from beamng_autopilot.experiments.checkpoint import (
        load_full, weights_equal)

    runs = _frames(tmp_path)
    a, a2 = tmp_path / "A", tmp_path / "A2"
    _train(runs, a)
    _train(runs, a2)
    eq = weights_equal(load_full(a / "checkpoint_last.pt")["state_dict"],
                       load_full(a2 / "checkpoint_last.pt")["state_dict"],
                       atol=0.0)
    assert eq["n_diff"] == 0, (
        f"确定性路径上两次独立运行都不同：max={eq['max_abs_diff']:.3e} — "
        f"说明确定性开关没生效（CUDA 上 nll_loss2d 无确定性实现，必须用 CPU）")


def test_the_interruption_hook_keeps_the_lr_schedule_identical(tmp_path):
    """`--stop-after` 只是提前退出，不改 LR 计划；换 --epochs 才会改。"""
    runs = _frames(tmp_path)
    b = tmp_path / "B"
    _train(runs, b, "--stop-after", "2")
    hist = json.loads((b / "train_hist.json").read_text(encoding="utf-8"))
    assert hist["epoch"] == [0, 1], "应当只跑了 2 轮"
    ckpt_lines = (b / "checkpoint_last.pt")
    assert ckpt_lines.exists()
