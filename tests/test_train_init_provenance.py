"""init v1 谱系记录（T16 §3.3）：来源档位、完整性、架构拒绝。

要点（都是判定层要读的证据，不能"看起来像"）：

* ``--init`` 只加载权重（优化器从头）——本文件钉住它记录的
  ``init_from/init_sha16/init_source/init_arch/parent_history/
  provenance_complete``；
* 父 ``train_args`` 缺 init 记录（旧 checkpoint）→ ``provenance_complete=False``，
  **不得**把缺记录当完整谱系；
* 链要能追到**显式**随机根才算完整；
* 父 width 与本次 ``--width`` 不一致 → 直接报错，不静默部分加载；
* 未给 ``--init`` → 显式 ``init_source="random"`` 且 ``provenance_complete=True``。

用 ``torch.save`` 伪造父 checkpoint（只有 train_args + 小 state_dict），
不跑真实训练（唯一例外是最后两个端到端拒绝用例，它们在加载数据后、训练前就退出）。
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")

_TRAIN_PATH = ROOT / "scripts" / "m5_train_seg.py"
_spec = importlib.util.spec_from_file_location("m5_train_seg_init", _TRAIN_PATH)
tr = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(tr)


def _save(path: Path, **fields) -> Path:
    torch.save(fields, path)
    return path


def _parent(path: Path, *, run_id="", dataset_id="ds_x", state_dict=None,
            **train_args) -> Path:
    return _save(path, state_dict=(state_dict if state_dict is not None
                                  else {"w": torch.zeros(2)}),
                 run_id=run_id, dataset_id=dataset_id, train_args=train_args)


def _prov(path: Path, *, width: float = 1.0) -> dict:
    return tr.init_provenance(path, torch.load(str(path), map_location="cpu",
                                               weights_only=False), width=width)


def test_a_random_root_parent_is_recorded_as_a_complete_lineage(tmp_path):
    """新训练器写出的随机根：有 init_* 记录且根为显式 random -> 完整。"""
    p = _parent(tmp_path / "random_root.pt", run_id="t14_baseline_20260927",
                dataset_id="ds_root", state_dict={"w": torch.ones(3)},
                total_steps=72, steps_done=72, steps_in_epoch=0,
                stopped_by="step_budget", sampler="quota",
                sampler_state_digest="abc123", init_from=None, init_sha16=None,
                init_source="random", init_arch=None, parent_history=None,
                provenance_complete=True)
    from beamng_autopilot.experiments.checkpoint import file_sha16
    got = _prov(p)
    assert got["init_from"] == str(p)
    assert got["init_sha16"] == file_sha16(p)
    assert got["init_source"] == "random"
    assert got["init_arch"] is None
    assert got["provenance_complete"] is True
    assert got["parent_history"] == {
        "run_id": "t14_baseline_20260927", "steps_done": 72,
        "total_steps": 72, "dataset_id": "ds_root"}


def test_a_legacy_parent_is_never_called_complete(tmp_path):
    """旧 checkpoint（没有 init_* 记录）：来源按随机根记，但谱系不完整。"""
    p = _save(tmp_path / "legacy.pt", state_dict={"w": torch.zeros(1)},
              train_args={"epochs": 40, "batch": 8, "lr": 1e-3, "seed": 42})
    got = _prov(p)
    assert got["init_source"] == "random"
    assert got["provenance_complete"] is False, "缺记录不能被当成完整谱系"
    assert got["parent_history"] == {"run_id": None, "steps_done": None,
                                     "total_steps": None, "dataset_id": None}
    assert got["init_arch"] is None, "父没记 width 时留 None，不编造"


def test_width_mismatch_is_refused_instead_of_partially_loading(tmp_path):
    p = _parent(tmp_path / "w1.pt", epochs=1, batch=4,
                arch_args={"width": 2.0}, init_from=None, init_sha16=None,
                init_source="random", provenance_complete=True)
    with pytest.raises(SystemExit) as exc:
        _prov(p, width=1.0)
    assert "width" in str(exc.value)
    # 一致时正常通过，并把父的宽度记下来
    ok = _prov(p, width=2.0)
    assert ok["init_arch"] == 2.0


def test_the_role_comes_from_the_parent_run_id(tmp_path):
    root = _parent(tmp_path / "root.pt", run_id="random_root", dataset_id="ds0",
                   init_from=None, init_sha16=None, init_source="random",
                   provenance_complete=True, steps_done=10, total_steps=10)
    champ = _parent(tmp_path / "champ.pt", run_id="champion_round2",
                    dataset_id="ds1", init_from=str(root),
                    init_sha16="deadbeef", init_source="random",
                    provenance_complete=True)
    prod = _parent(tmp_path / "prod.pt", run_id="production_v3",
                   dataset_id="ds2", init_from=str(root),
                   init_sha16="deadbeef", init_source="champion",
                   provenance_complete=True)
    assert _prov(champ)["init_source"] == "champion"
    assert _prov(champ)["provenance_complete"] is True
    assert _prov(prod)["init_source"] == "production"
    # 父没记 init_source 时，链上的角色字样也算证据（链判定）
    no_rec = _parent(tmp_path / "no_rec.pt", run_id="champion_later",
                     dataset_id="ds3", init_from=str(root),
                     init_sha16="deadbeef")
    got = _prov(no_rec)
    assert got["init_source"] == "champion"
    assert got["provenance_complete"] is False, "缺 init_source 记录 -> 不完整"


def test_a_break_in_the_chain_is_not_complete(tmp_path):
    legacy = _parent(tmp_path / "legacy_root.pt", epochs=2, batch=4)
    child = _parent(tmp_path / "child.pt", run_id="run_child", dataset_id="ds9",
                    init_from=str(legacy), init_sha16="cafebabe",
                    init_source="random", provenance_complete=False,
                    total_steps=5, steps_done=5)
    got = _prov(child)
    assert got["provenance_complete"] is False
    assert got["parent_history"]["steps_done"] == 5
    # 父链文件读不到（被删/换路径）同样不能算完整
    gone = _parent(tmp_path / "gone.pt", run_id="run_gone",
                   init_from=str(tmp_path / "missing.pt"),
                   init_sha16="cafebabe", init_source="random")
    assert _prov(gone)["provenance_complete"] is False


def test_without_init_the_random_root_is_explicit():
    assert tr.random_init_block() == {
        "init_from": None, "init_sha16": None, "init_source": "random",
        "init_arch": None, "parent_history": None,
        "provenance_complete": True}


# --------------------------------------------------------- 端到端（拒绝路径）
def _tiny_runs(root: Path) -> Path:
    d = root / "runs" / "front_main"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(2):
        np.savez(d / f"frame_{i:05d}.npz",
                 colour=np.full((16, 20, 3), 30 + i, np.uint8),
                 label=np.ones((16, 20), np.uint8))
    return d


def _cli(runs: Path, tmp_path: Path, *extra: str):
    env = dict(os.environ)
    env["BEAMNG_LOGS_DIR"] = str(tmp_path / "logs")
    return subprocess.run(
        [sys.executable, str(_TRAIN_PATH), "--runs", str(runs),
         "--out", str(tmp_path / "out"), *extra],
        capture_output=True, text=True, env=env, timeout=300)


def test_cli_refuses_init_together_with_resume(tmp_path):
    runs = _tiny_runs(tmp_path)
    parent = _parent(tmp_path / "p.pt", init_from=None, init_sha16=None,
                     init_source="random", provenance_complete=True)
    r = _cli(runs, tmp_path, "--init", str(parent), "--resume", str(parent))
    assert r.returncode != 0
    assert "--init" in (r.stdout + r.stderr) and "--resume" in r.stdout + r.stderr


def test_cli_refuses_a_width_mismatch_before_loading_any_weight(tmp_path):
    runs = _tiny_runs(tmp_path)
    parent = _parent(tmp_path / "w1.pt", arch_args={"width": 1.0},
                     init_from=None, init_sha16=None, init_source="random",
                     provenance_complete=True)
    r = _cli(runs, tmp_path, "--init", str(parent), "--width", "2.0",
             "--epochs", "1", "--device", "cpu")
    assert r.returncode != 0
    assert "width" in (r.stdout + r.stderr)
    # 拒绝发生在训练开始前：不能留下任何训练产物
    assert not (tmp_path / "out" / "train_hist.json").exists()


def test_cli_wires_init_into_the_history_and_the_checkpoint(tmp_path):
    """真实训练一次拿到父 checkpoint，再用它 --init：字段必须落到产物里。"""
    from beamng_autopilot.experiments.checkpoint import file_sha16

    runs = _tiny_runs(tmp_path)
    base = tmp_path / "base"
    r = _cli(runs, tmp_path, "--epochs", "1", "--batch", "4",
             "--width", "0.25", "--device", "cpu", "--no-amp",
             "--split", "tail", "--val-frac", "0.5", "--out", str(base))
    assert r.returncode == 0, r.stdout[-800:]
    parent = base / "checkpoint_last.pt"
    assert parent.exists()

    child = tmp_path / "child"
    r = _cli(runs, tmp_path, "--epochs", "1", "--batch", "4",
             "--width", "0.25", "--device", "cpu", "--no-amp",
             "--split", "tail", "--val-frac", "0.5", "--init", str(parent),
             "--out", str(child))
    assert r.returncode == 0, r.stdout[-800:]
    hist = json.loads((child / "train_hist.json").read_text(encoding="utf-8"))
    init = hist["init"]
    assert init["init_from"] == str(parent)
    assert init["init_sha16"] == file_sha16(parent)
    assert init["init_source"] == "random"          # 父是显式随机根
    assert init["init_arch"] == 0.25
    assert init["provenance_complete"] is True
    assert init["parent_history"]["steps_done"] >= 1
    assert init["parent_history"]["dataset_id"] == "unversioned"
    ck = torch.load(str(child / "checkpoint_last.pt"), map_location="cpu",
                    weights_only=True)
    assert ck["train_args"]["init_sha16"] == init["init_sha16"]
    assert ck["train_args"]["init_source"] == "random"
    assert ck["train_args"]["provenance_complete"] is True
