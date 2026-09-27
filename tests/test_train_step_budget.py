"""优化步预算（T16 §3.1）+ 全池采样接线：纯逻辑 + 小合成数据集成。

分工：

* 纯逻辑部分钉 ``validate_arg_combos``（互斥组合）、``build_sampler_pool``
  （内容身份 + run 归属 + 视角）、``run_view_map``；
* 集成部分用 6 帧合成 npz 在 CPU 上真跑训练器，断言
  ``steps_done == --total-steps``、``stopped_by == "step_budget"``、
  ``sampler_report.unique_seen`` **来自真实采样**且在 (0, unique_available]；
* 续训等价：``--stop-after`` 模拟中断后再 ``--resume``，与一次跑完**逐位一致**
  （CPU 上比较 state_dict，atol=0）。中断点必须是 epoch 边界——训练器只在
  每轮结束落盘，这是历史协议的一部分。

集成用例里 ``--width 0.25`` 只是让 CPU 跑得动（SegUNet 通道下限 8，宽度只影响
算力，不影响步预算/采样/恢复语义）。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from beamng_autopilot.vision.sampling import QuotaSampler, frame_identity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 训练器顶层 import torch：没有 torch 的环境直接跳过（纯逻辑部分也需要它，
# 因为 validate_arg_combos / build_sampler_pool 定义在训练器里）
pytest.importorskip("torch")

_TRAIN_PATH = ROOT / "scripts" / "m5_train_seg.py"
_spec = importlib.util.spec_from_file_location("m5_train_seg_budget", _TRAIN_PATH)
tr = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(tr)


# ------------------------------------------------------------------ 纯逻辑
def _args(**kw):
    base = dict(max_train_frames=0, total_steps=0, init=None, resume=None,
                sampler="auto", balance_runs=False, run_weights="")
    base.update(kw)
    return argparse.Namespace(**base)


def test_auto_sampler_follows_the_step_budget():
    assert tr.validate_arg_combos(_args()) == "legacy"
    assert tr.validate_arg_combos(_args(total_steps=72)) == "quota"
    assert tr.validate_arg_combos(_args(sampler="legacy", total_steps=72)) \
        == "legacy"
    assert tr.validate_arg_combos(_args(sampler="quota")) == "quota"


def test_refused_combinations_exit_with_a_reason():
    cases = [
        _args(max_train_frames=12, total_steps=72),      # 截帧 vs 全池
        _args(init="a.pt", resume="b.pt"),               # 初始化 vs 续训
        _args(sampler="quota", max_train_frames=12),
        _args(sampler="quota", balance_runs=True),
        _args(sampler="quota", run_weights="0=2.0"),
        _args(total_steps=-1),
    ]
    for case in cases:
        with pytest.raises(SystemExit):
            tr.validate_arg_combos(case)


def _np_frame(k: int):
    return (np.full((6, 6, 3), k, np.uint8), np.ones((6, 6), np.uint8))


def test_run_attribution_survives_a_reordered_split():
    """run 归属按内容反查：--split by-map-scene 会重排训练帧，下标区间不可信。"""
    all_frames = [_np_frame(k) for k in range(6)]
    per_run = {"a": {"start": 0, "end": 3}, "b": {"start": 3, "end": 6}}
    reordered = [all_frames[4], all_frames[1], all_frames[5]]   # 组划分的产物
    pool = tr.build_sampler_pool(all_frames, per_run, reordered)
    assert [e["run"] for e in pool] == ["b", "a", "b"]
    assert [e["hash"] for e in pool] == [tr.frame_identity(f) for f in reordered]


def test_duplicate_frames_inherit_the_run_they_copy():
    """--weak-line-oversample 追加的副本没有自己的区间，必须继承来源的 run。"""
    all_frames = [_np_frame(k) for k in range(4)]
    per_run = {"a": {"start": 0, "end": 2}, "b": {"start": 2, "end": 4}}
    pool = tr.build_sampler_pool(all_frames, per_run,
                                 all_frames + [all_frames[3]])
    assert [e["run"] for e in pool] == ["a", "a", "b", "b", "b"]
    s = QuotaSampler(pool, batch=5, seed=1, run_key="")
    s.next_batch()
    # 重复内容只算 1 个可用帧，但曝光按槽位计
    assert s.report()["unique_available"] == 4
    assert s.report()["exposures_by_run"]["b"] == 3


def test_sampler_pool_records_identity_and_line_presence():
    # 两帧内容必须不同：同内容在不同 run 里按"同一帧"处理（先出现的 run 胜）
    f_a = (np.full((8, 8, 3), 7, np.uint8), np.full((8, 8), 2, np.uint8))
    f_b = (np.full((8, 8, 3), 9, np.uint8), np.ones((8, 8), np.uint8))
    per_run = {"a": {"start": 0, "end": 1, "meta": {"frames": [{"view": "cam0"}]}},
               "b": {"start": 1, "end": 2}}
    pool = tr.build_sampler_pool([f_a, f_b], per_run, [f_a, f_b])
    assert pool[0]["has_line"] is True and pool[1]["has_line"] is False
    assert pool[0]["view"] == "cam0" and pool[1]["view"] is None
    assert pool[0]["run"] == "a" and pool[1]["run"] == "b"
    assert len(pool[0]["hash"]) == 16
    from beamng_autopilot.vision.sampling import frame_identity
    assert pool[0]["hash"] == frame_identity(f_a)
    assert tr.run_view_map(per_run) == {"a": "cam0", "b": None}


def test_cli_refuses_capping_together_with_a_step_budget(tmp_path):
    """端到端：拒绝发生在加载数据之前（不依赖 runs 目录存在）。"""
    r = subprocess.run(
        [sys.executable, str(_TRAIN_PATH), "--runs", str(tmp_path / "nope"),
         "--max-train-frames", "12", "--total-steps", "72",
         "--out", str(tmp_path / "out")],
        capture_output=True, text=True, timeout=120)
    assert r.returncode != 0
    assert "--max-train-frames" in (r.stdout + r.stderr)


# --------------------------------------------------------------- 集成夹具
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


def _train(runs: Path, out: Path, *extra: str, timeout: int = 600):
    cmd = [sys.executable, str(_TRAIN_PATH), "--runs", str(runs),
           "--split", "tail", "--val-frac", "0.2", "--seed", "7",
           "--device", "cpu", "--no-amp", "--out", str(out), *extra]
    env = dict(os.environ)
    env["BEAMNG_LOGS_DIR"] = str(runs.parent.parent / "logs")
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, env=env,
                       timeout=timeout)
    dt = time.time() - t0
    assert r.returncode == 0, (r.stdout[-1500:] + r.stderr[-1000:])
    assert dt < 120, f"这个集成用例应当很快，实测 {dt:.0f}s（命令：{cmd}）"
    return r


def _hist(out: Path) -> dict:
    return json.loads((out / "train_hist.json").read_text(encoding="utf-8"))


def _ckpt(out: Path) -> dict:
    import torch
    # weights_only=True：autoloop 就是这么读 train_args 的，新字段必须是纯类型
    return torch.load(str(out / "checkpoint_last.pt"), map_location="cpu",
                      weights_only=True)


def test_budget_mode_stops_at_the_step_budget_end_to_end(tmp_path):
    """规格要求的集成用例：6 帧、CPU、无 AMP、--total-steps 5。"""
    runs = _frames(tmp_path)
    out = tmp_path / "A"
    _train(runs, out, "--total-steps", "5", "--batch", "4")
    hist = _hist(out)
    assert hist["steps_done"] == 5
    assert hist["total_steps"] == 5
    assert hist["stopped_by"] == "step_budget"
    assert hist["sampler"] == "quota"
    rep = hist["sampler_report"]
    assert 0 < rep["unique_seen"] <= rep["unique_available"] == rep["pool_size"] == 5
    assert rep["exposures_total"] >= 5

    ta = _ckpt(out)["train_args"]
    assert ta["steps_done"] == 5 and ta["total_steps"] == 5
    assert ta["sampler"] == "quota"
    assert ta["sampling"] == "quota_full_pool"
    assert ta["stopped_by"] == "step_budget"
    assert ta["sampler_state_digest"] == hist["sampler_state_digest"]
    # LR 计划按步调度（T_max=N），epoch 数由步预算反推（ceil(5/2)=3）
    assert ta["scheduler"]["step_per"] == "step"
    assert ta["scheduler"]["T_max"] == 5
    assert len(hist["epoch"]) == 3
    # 采样器完整状态落盘，且是纯 JSON 类型
    state = _ckpt(out)["sampler_state"]
    json.dumps(state)
    assert state["version"] == 1 and state["batch"] == 4
    assert state["exposures_total"] == rep["exposures_total"]
    assert len(state["seen"]) == rep["unique_seen"]
    # unique_seen 的 hash 必须来自真实的 npz 内容（不是池清单/目录数派生）
    from beamng_autopilot.vision.sampling import frame_identity
    on_disk = set()
    for f in sorted(runs.glob("frame_*.npz")):
        with np.load(f) as z:
            on_disk.add(frame_identity((z["colour"], z["label"])))
    assert set(state["seen"]) <= on_disk, "报告里的 hash 不在训练帧内容里"


def test_resume_reproduces_an_uninterrupted_budget_run(tmp_path):
    """同 seed 同数据：中断续训与一次跑完**逐位一致**（含采样器状态）。"""
    from beamng_autopilot.experiments.checkpoint import load_full, weights_equal

    runs = _frames(tmp_path)
    a, b = tmp_path / "A", tmp_path / "B"
    common = ("--total-steps", "5", "--batch", "4", "--width", "0.25")
    _train(runs, a, *common)
    # --stop-after 2 在 epoch 1 结束处退出，落盘的是 epoch 0 末的 checkpoint
    # （训练器历史协议：只在每轮结束落盘）；budget 还剩 3 步没走
    _train(runs, b, *common, "--stop-after", "2")
    _train(runs, b, *common, "--resume", str(b / "checkpoint_last.pt"))

    ca, cb = load_full(a / "checkpoint_last.pt"), load_full(b / "checkpoint_last.pt")
    assert ca["train_args"]["steps_done"] == 5
    assert cb["train_args"]["steps_done"] == 5
    assert cb["train_args"]["stopped_by"] == "step_budget"
    eq = weights_equal(ca["state_dict"], cb["state_dict"], atol=0.0)
    assert eq["n_diff"] == 0, (f"续训与不中断不逐位一致: {eq['max_abs_diff']:.3e} "
                               f"worst={eq['max_key']}")
    assert (ca["train_args"]["sampler_state_digest"]
            == cb["train_args"]["sampler_state_digest"])
    assert _hist(a)["sampler_report"] == _hist(b)["sampler_report"]


def test_legacy_path_still_stops_by_epochs_and_claims_no_sampler_report(tmp_path):
    """不给 --total-steps：旧 epoch 协议不变，且不伪造采样报告。"""
    runs = _frames(tmp_path)
    out = tmp_path / "L"
    _train(runs, out, "--epochs", "1", "--batch", "4", "--width", "0.25")
    hist = _hist(out)
    assert hist["epoch"] == [0]
    assert hist["total_steps"] == 0 and hist["stopped_by"] == "epochs"
    assert hist["sampler"] == "legacy"
    assert hist["sampler_report"] is None, "legacy 不逐帧记账，不能拿池大小冒充"
    assert hist["init"]["init_source"] == "random"
    ck = _ckpt(out)
    assert ck["sampler_state"] is None
    assert ck["train_args"]["total_steps"] == 0
    assert ck["train_args"]["sampler_state_digest"] is None
    assert ck["train_args"]["scheduler"]["step_per"] == "epoch"
    assert ck["train_args"]["steps_done"] >= 1
