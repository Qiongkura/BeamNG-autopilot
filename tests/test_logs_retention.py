"""logs 权重保留策略：只删权重、证据不动、默认 dry-run、路径不出 logs。

这些测试钉的是**删除边界**，不是删除速度——一旦边界破了，被删的是别人的
实验证据，找不回来。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "m5_logs_retention", ROOT / "scripts" / "m5_logs_retention.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["m5_logs_retention"] = mod
    spec.loader.exec_module(mod)
    return mod


def _tree(tmp_path: Path) -> Path:
    """造一棵有代表性的 logs 树：证据 + 权重 + 部署权重 + best 权重。"""
    logs = tmp_path / "logs"
    exp = logs / "experiments" / "run_a" / "seed42"
    exp.mkdir(parents=True)
    (exp / "checkpoint_last.pt").write_bytes(b"x" * 4096)
    (exp / "checkpoint_ep3.pt").write_bytes(b"x" * 2048)
    (exp / "best.pt").write_bytes(b"x" * 1024)
    (exp / "summary.json").write_text("{}", encoding="utf-8")
    (exp / "train.log").write_text("ok", encoding="utf-8")
    (exp / "curve.png").write_bytes(b"png")
    deployed = logs / "m5_seg" / "seg_model"
    deployed.mkdir(parents=True)
    (deployed / "best.pt").write_bytes(b"w" * 8192)
    (deployed / "checkpoint_last.pt").write_bytes(b"w" * 8192)
    other = logs / "m5_e2e"
    other.mkdir()
    (other / "best_temporal.pt").write_bytes(b"t" * 777)
    return logs


def test_dry_run_plans_the_weights_but_deletes_nothing(tmp_path: Path) -> None:
    mod = _load()
    logs = _tree(tmp_path)
    rc = mod.main(["--logs-dir", str(logs)])
    assert rc == 0
    # 两个非 best 权重被计划删除，文件仍在（dry-run）
    seed = logs / "experiments" / "run_a" / "seed42"
    assert (seed / "checkpoint_last.pt").is_file()
    assert (seed / "checkpoint_ep3.pt").is_file()
    manifests = list(logs.glob("logs_retention_*.json"))
    assert len(manifests) == 1
    blob = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert blob["dry_run"] is True
    names = sorted(Path(it["path"]).name for it in blob["deleted"])
    assert names == ["checkpoint_ep3.pt", "checkpoint_last.pt"]


def test_apply_deletes_only_weights_and_keeps_evidence(tmp_path: Path) -> None:
    mod = _load()
    logs = _tree(tmp_path)
    rc = mod.main(["--logs-dir", str(logs), "--apply"])
    assert rc == 0
    seed = logs / "experiments" / "run_a" / "seed42"
    assert not (seed / "checkpoint_last.pt").exists()
    assert not (seed / "checkpoint_ep3.pt").exists()
    # 证据与 best 权重原样保留
    assert (seed / "summary.json").is_file()
    assert (seed / "train.log").is_file()
    assert (seed / "curve.png").is_file()
    assert (seed / "best.pt").is_file()


def test_deployed_seg_model_and_e2e_weights_are_protected(tmp_path: Path) -> None:
    mod = _load()
    logs = _tree(tmp_path)
    mod.main(["--logs-dir", str(logs), "--apply"])
    # 部署目录不在默认扫描根内，即便显式扫到也必须受保护
    assert (logs / "m5_seg" / "seg_model" / "best.pt").is_file()
    assert (logs / "m5_seg" / "seg_model" / "checkpoint_last.pt").is_file()
    assert (logs / "m5_e2e" / "best_temporal.pt").is_file()
    candidates, protected = mod.scan(logs, ("m5_seg",))
    assert candidates == []
    assert {Path(it["path"]).name for it in protected} == {
        "best.pt", "checkpoint_last.pt"}


def test_path_outside_the_logs_root_is_refused(tmp_path: Path) -> None:
    mod = _load()
    logs = _tree(tmp_path)
    outside = tmp_path / "elsewhere.pt"
    outside.write_bytes(b"keep me")
    with pytest.raises(mod.RetentionError):
        mod.resolve_within(outside, logs)
    with pytest.raises(mod.RetentionError):
        mod.apply_plan(logs, [{"path": str(outside), "size": 7, "mtime": 0.0}],
                       min_free_gb=0.0, manifest_path=logs / "m.json",
                       dry_run=False)
    assert outside.is_file()


def test_min_free_target_stops_once_reached(tmp_path: Path, monkeypatch) -> None:
    mod = _load()
    logs = _tree(tmp_path)
    monkeypatch.setattr(mod, "free_gb", lambda _p: 0.0)
    big = logs / "experiments" / "big"
    big.mkdir()
    files = []
    for i in range(3):
        p = big / f"checkpoint_{i}.pt"
        p.write_bytes(b"z" * (2 * 1024 * 1024))
        files.append(p)
    candidates, _ = mod.scan(logs, ("experiments",))
    candidates = [{"path": str(p), "size": p.stat().st_size, "mtime": 0.0}
                  for p in files] + candidates
    summary = mod.apply_plan(logs, candidates, min_free_gb=0.003,
                             manifest_path=logs / "m.json", dry_run=False)
    assert len(summary["deleted"]) >= 1
    assert summary["skipped"], "达到目标后剩余候选必须跳过"
    assert all("目标" in it["reason"] for it in summary["skipped"])


def test_documented_init_weight_is_kept(tmp_path: Path) -> None:
    """文档点名当 --init 的权重不能被删（否则预注册协议无法原样重跑）。"""
    mod = _load()
    logs = _tree(tmp_path)
    shared = (logs / "experiments" / "t14_e1_promotable_20260927"
              / "baseline" / "seed42")
    shared.mkdir(parents=True)
    (shared / "checkpoint_last.pt").write_bytes(b"i" * 4096)
    delivered = (logs / "experiments" / "t16_negdose6x_20261001"
                 / "round0" / "seed42")
    delivered.mkdir(parents=True)
    (delivered / "checkpoint_last.pt").write_bytes(b"d" * 4096)
    mod.main(["--logs-dir", str(logs), "--apply"])
    assert (shared / "checkpoint_last.pt").is_file()
    assert (delivered / "checkpoint_last.pt").is_file()
    candidates, protected = mod.scan(logs, ("experiments",))
    reasons = {Path(it["path"]).parent.parent.name: it["reason"]
               for it in protected if "保留清单" in it["reason"]}
    assert reasons, protected


def test_extra_keep_glob_from_cli_and_file(tmp_path: Path) -> None:
    mod = _load()
    logs = _tree(tmp_path)
    keep_dir = logs / "experiments" / "my_negative_arm" / "seed7"
    keep_dir.mkdir(parents=True)
    (keep_dir / "checkpoint_last.pt").write_bytes(b"k" * 512)
    # --keep 生效
    mod.main(["--logs-dir", str(logs), "--apply",
              "--keep", "**/my_negative_arm/**"])
    assert (keep_dir / "checkpoint_last.pt").is_file()
    # 清单文件同样生效
    again = logs / "experiments" / "my_negative_arm" / "seed8"
    again.mkdir(parents=True)
    (again / "checkpoint_last.pt").write_bytes(b"k" * 512)
    (logs / mod.KEEP_FILE).write_text(
        "# 自己加的保留\n**/my_negative_arm/**\n", encoding="utf-8")
    mod.main(["--logs-dir", str(logs), "--apply"])
    assert (again / "checkpoint_last.pt").is_file()


def test_sealed_final_set_is_never_touched(tmp_path: Path) -> None:
    """封存最终集整棵树都不许动：它不是运行产物（与 prune 脚本同一底线）。"""
    mod = _load()
    logs = _tree(tmp_path)
    sealed = (logs / "experiments" / "final_set_v5_20261005" / "seal")
    sealed.mkdir(parents=True)
    (sealed / "final_set_seal.json").write_text("{}", encoding="utf-8")
    arm = logs / "experiments" / "final_set_v5_20261005" / "round0" / "seed42"
    arm.mkdir(parents=True)
    (arm / "checkpoint_last.pt").write_bytes(b"f" * 4096)
    mod.main(["--logs-dir", str(logs), "--apply"])
    assert (arm / "checkpoint_last.pt").is_file(), "封存集旁边的权重也不能删"
    candidates, protected = mod.scan(logs, ("experiments",))
    assert any("封存" in it["reason"] for it in protected), protected


def test_older_than_days_filters_the_candidates(tmp_path: Path) -> None:
    mod = _load()
    logs = _tree(tmp_path)
    seed = logs / "experiments" / "run_a" / "seed42"
    fresh, old = seed / "checkpoint_last.pt", seed / "checkpoint_ep3.pt"
    import os
    import time
    old_time = time.time() - 40 * 86400
    os.utime(fresh, (time.time(), time.time()))
    os.utime(old, (old_time, old_time))
    candidates, _ = mod.scan(logs, ("experiments",), older_than_days=30)
    assert [Path(it["path"]).name for it in candidates] == ["checkpoint_ep3.pt"]
