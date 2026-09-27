"""运行产物保留策略（方案 v2 §S5）：中间产物可删，恢复/选择 checkpoint 与证据必留。

实测背景：4 小时耐久里逐 epoch 权重吃掉 ~25 GB，把磁盘压到资源门以下、后 9 轮
全被拦。但清理必须**只**作用中间产物——恢复 checkpoint、选择 checkpoint、判定/
日志/事件（证据）与所有者信息一律保留；含最终集封存的目录一律不碰。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load():
    spec = importlib.util.spec_from_file_location(
        "m5_prune_run_artifacts", ROOT / "scripts" / "m5_prune_run_artifacts.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["m5_prune_run_artifacts"] = mod
    spec.loader.exec_module(mod)
    return mod


def _fake_run(tmp_path: Path, name: str = "run1") -> Path:
    d = tmp_path / name
    (d / "round0" / "seed42").mkdir(parents=True)
    for i in range(3):
        (d / "round0" / "seed42" / f"epoch_{i:02d}.pt").write_bytes(b"x" * 100)
    (d / "round0" / "seed42" / "checkpoint_last.pt").write_bytes(b"y" * 10)
    (d / "round0" / "seed42" / "best.pt").write_bytes(b"z" * 10)
    (d / "round0" / "seed42" / "train_hist.json").write_text("{}", encoding="utf-8")
    (d / "decision_cand-r0.json").write_text("{}", encoding="utf-8")
    (d / "events.jsonl").write_text("{}\n", encoding="utf-8")
    (d / "run.log").write_text("log\n", encoding="utf-8")
    (d / "controller.lock").write_text("owner", encoding="utf-8")
    return d


def test_dry_run_plans_but_deletes_nothing(tmp_path, monkeypatch):
    mod = _load()
    monkeypatch.setattr(mod, "ALLOWED_ROOT", tmp_path)
    d = _fake_run(tmp_path)
    rep = mod.plan(d)
    assert len(rep["prune"]) == 3 and not rep["refused"]
    assert mod.main(["--run-dir", str(d)]) == 0
    assert len(list(d.rglob("epoch_*.pt"))) == 3, "dry-run 不许删任何东西"


def test_apply_deletes_only_intermediate_weights(tmp_path, monkeypatch, capsys):
    mod = _load()
    monkeypatch.setattr(mod, "ALLOWED_ROOT", tmp_path)
    d = _fake_run(tmp_path)
    assert mod.main(["--run-dir", str(d), "--apply"]) == 0
    assert not list(d.rglob("epoch_*.pt"))
    for keep in ("checkpoint_last.pt", "best.pt", "train_hist.json",
                 "decision_cand-r0.json", "events.jsonl", "run.log",
                 "controller.lock"):
        assert (d / "round0" / "seed42" / keep).exists() or (d / keep).exists(), keep
    reports = list(d.glob("prune_report_*.json"))
    assert reports, "删完必须留记录（含所有者/commit/清单）"
    rec = json.loads(reports[0].read_text(encoding="utf-8"))
    assert rec["deleted"] and rec["freed_bytes"] > 0 and rec["kept_n"] >= 7


def test_keep_epochs_keeps_the_newest_n(tmp_path, monkeypatch):
    mod = _load()
    monkeypatch.setattr(mod, "ALLOWED_ROOT", tmp_path)
    d = _fake_run(tmp_path)
    import os
    import time as _t
    for i, f in enumerate(sorted(d.rglob("epoch_*.pt"))):
        os.utime(f, (1_700_000_000 + i * 10, 1_700_000_000 + i * 10))
    rep = mod.plan(d, keep_epochs=1)
    assert len(rep["prune"]) == 2, rep["prune"]
    assert len(rep["keep"]) >= 8


def test_a_directory_outside_the_allowed_root_is_refused(tmp_path, monkeypatch):
    mod = _load()
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    monkeypatch.setattr(mod, "ALLOWED_ROOT", allowed)
    outside = _fake_run(tmp_path, "outside")
    rep = mod.plan(outside)
    assert rep["refused"] and not rep["prune"]


def test_a_sealed_final_set_is_never_pruned(tmp_path, monkeypatch):
    mod = _load()
    monkeypatch.setattr(mod, "ALLOWED_ROOT", tmp_path)
    d = _fake_run(tmp_path)
    (d / "final_set_seal.json").write_text("{}", encoding="utf-8")
    rep = mod.plan(d)
    assert rep["refused"] and not rep["prune"], "最终集目录一律不碰"
    assert mod.main(["--run-dir", str(d), "--apply"]) == 3
    assert len(list(d.rglob("epoch_*.pt"))) == 3
