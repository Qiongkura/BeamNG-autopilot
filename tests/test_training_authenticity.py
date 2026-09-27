"""训练真伪证据工具的行为（方案 §S6："不能以生成了目录证明学习发生"）。

用合成 checkpoint 与两种写法的 train_hist 钉住：
两臂权重差异可测、列式/行式 loss 曲线都能读、缺失产物如实记 missing。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load():
    spec = importlib.util.spec_from_file_location(
        "m5_training_authenticity", ROOT / "scripts" / "m5_training_authenticity.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["m5_training_authenticity"] = mod
    spec.loader.exec_module(mod)
    return mod


def _ckpt(path: Path, value: float, *, n_train=20, epochs=24, batch=4):
    torch.save({"model": {"w": torch.full((4, 4), value)},
                "train_args": {"n_train": n_train, "epochs": epochs,
                               "batch": batch, "seed": 42}}, path)


def test_weight_diff_detects_a_real_change(tmp_path):
    mod = _load()
    a, b = tmp_path / "a.pt", tmp_path / "b.pt"
    _ckpt(a, 0.0)
    _ckpt(b, 0.5)
    d = mod.weight_diff(a, b)
    assert d["n_tensors"] == 1 and d["changed_tensors"] == 1
    assert d["mean_abs_diff"] == pytest.approx(0.5)


def test_identical_weights_are_not_counted_as_changed(tmp_path):
    mod = _load()
    a, b = tmp_path / "a.pt", tmp_path / "b.pt"
    _ckpt(a, 0.25)
    _ckpt(b, 0.25)
    d = mod.weight_diff(a, b)
    assert d["changed_tensors"] == 0 and d["mean_abs_diff"] == 0.0


def test_loss_curve_reads_both_shapes(tmp_path):
    mod = _load()
    col = tmp_path / "col.json"
    col.write_text(json.dumps({"epoch": [0, 1, 2],
                               "train_loss": [2.0, 1.5, 1.0]}), encoding="utf-8")
    row = tmp_path / "row.json"
    row.write_text(json.dumps([{"epoch": 0, "train_loss": 2.0},
                               {"epoch": 1, "train_loss": 1.0}]), encoding="utf-8")
    c = mod.loss_curve(col)
    r = mod.loss_curve(row)
    assert c["status"] == "ok" and c["drop_pct"] == 50.0
    assert r["status"] == "ok" and r["drop_pct"] == 50.0
    assert mod.loss_curve(tmp_path / "none.json")["status"] == "missing"


def test_audit_run_reports_steps_from_train_args(tmp_path):
    mod = _load()
    run = tmp_path / "run"
    (run / "baseline" / "seed42").mkdir(parents=True)
    (run / "round0" / "seed42").mkdir(parents=True)
    _ckpt(run / "baseline" / "seed42" / "checkpoint_last.pt", 0.0)
    _ckpt(run / "round0" / "seed42" / "checkpoint_last.pt", 0.3)
    (run / "baseline" / "seed42" / "train_hist.json").write_text(
        json.dumps({"train_loss": [2.0, 1.0]}), encoding="utf-8")
    rep = mod.audit_run(run, seeds=(42,))
    e = rep["seeds"]["42"]
    assert e["steps_expected"] == 120, "20 帧 / batch 4 × 24 epoch = 120 步"
    assert e["arms_weight_diff"]["changed_tensors"] == 1
    assert e["baseline_loss"]["drop_pct"] == 50.0
