"""后台执行入口（`run --no-dry-run`）的接线：复用 rounds、预算、断点恢复、停因。

下一阶段方案第 3 项的验收：无人操作完成 ≥2 轮离线实验；一轮候选被实测淘汰；
中断后能恢复；标签不足/资源超限/连续无收益时自行停止；默认不启动游戏采集。
本文件只钉**接线**（不真训练）：把 `cmd_rounds` 换成记录器，检查入口给它的参数、
锁与资源门、GPU 分钟账本与停止原因是否都落到事件里。
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load():
    spec = importlib.util.spec_from_file_location(
        "m5_seg_autoloop_entry", ROOT / "scripts" / "m5_seg_autoloop.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["m5_seg_autoloop_entry"] = mod
    spec.loader.exec_module(mod)
    return mod


def _config(tmp_path: Path, **over) -> Path:
    blob = {
        "collect": "off", "daily_gpu_minutes": 120.0, "max_wall_minutes": 180.0,
        "window_start_hour": 0, "window_end_hour": 24,
        "pause_while_user_active": True, "min_free_vram_mb": 0.0,
        "min_free_disk_gb": 0.0, "max_candidates": 6,
        "max_rounds_without_gain": 2, "seeds": [42, 43, 44],
        "dry_run": True,
        "runs": ["logs/m5_seg/diverse_town_20260924/front_main"],
        "baseline_runs": ["logs/m5_seg/diverse_town_20260924/front_main"],
        "eval_runs": ["logs/m5_seg/diverse_wide_20260924/front_main"],
        "proposals": "", "rounds": 2, "epochs": 3, "batch": 4, "lr": 0.001,
        "allow_road_only": True, "equal_steps": True,
        "trainer_script": "m5_train_seg.py",
    }
    blob.update(over)
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(blob), encoding="utf-8")
    return p


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    """把 exp_dir 指到 tmp，并记录 cmd_rounds 的调用参数。"""
    loop = _load()
    monkeypatch.setattr(loop, "exp_dir", lambda rid: tmp_path / "exp" / rid)
    calls = []

    def fake_rounds(args):
        calls.append(args)
        # 模拟"一轮候选被实测淘汰"：写一份判定文件再返回 1
        d = loop.exp_dir(args.run_id)
        d.mkdir(parents=True, exist_ok=True)
        (d / "decision_cand-r0.json").write_text(json.dumps({
            "candidate_id": "cand-r0",
            "pairings": {"road_iou": {"metric": "road_iou", "n": 3,
                                      "mean_delta": 0.01,
                                      "verdict": "inconclusive"}},
            "hard_gate": {}, "decision": {"decision": "rejected",
                                          "reasons": ["no credible gain"]},
        }), encoding="utf-8")
        return 1

    monkeypatch.setattr(loop, "cmd_rounds", fake_rounds)
    return loop, tmp_path, calls


def test_run_no_dry_run_reuses_the_rounds_flow_and_records_the_budget(
        sandbox, tmp_path):
    loop, _root, calls = sandbox
    cfg = _config(tmp_path)
    rc = loop.main(["run", "--run-id", "e1", "--no-dry-run", "--config",
                    str(cfg)])
    assert calls, "非 dry-run 必须真的调用 rounds 流程"
    a = calls[0]
    assert a.rounds == 2 and a.allow_road_only is True
    assert a.equal_steps is True and a.resume is True
    assert a.eval_runs == ["logs/m5_seg/diverse_wide_20260924/front_main"]
    assert rc == 1, "候选被淘汰时入口要把 rounds 的退出码透传"
    ledger = json.loads((loop.exp_dir("e1") / "gpu_minutes.json")
                        .read_text(encoding="utf-8"))
    today = list(ledger)[0]
    assert ledger[today]["minutes"] >= 0.0, "GPU 分钟要落盘（跨进程累计）"
    phases = [json.loads(ln)["phase"] for ln in
              (loop.exp_dir("e1") / "events.jsonl").read_text(
                  encoding="utf-8").splitlines() if ln.strip()]
    assert "training" in phases and ("paused" in phases or "training" in phases)


def test_the_daily_budget_blocks_the_entry_point_before_training(
        sandbox, tmp_path):
    """预算耗尽：不训练、写明原因（验收里"资源超限时自行停止"）。"""
    loop, _root, calls = sandbox
    d = loop.exp_dir("e2")
    d.mkdir(parents=True, exist_ok=True)
    import time as _t
    (d / "gpu_minutes.json").write_text(json.dumps({
        _t.strftime("%Y-%m-%d"): {"minutes": 999.0}}), encoding="utf-8")
    cfg = _config(tmp_path, daily_gpu_minutes=120.0)
    rc = loop.main(["run", "--run-id", "e2", "--no-dry-run", "--config",
                    str(cfg)])
    assert not calls, "预算耗尽时不得调用 rounds"
    assert rc == 5
    evs = [json.loads(ln) for ln in
           (d / "events.jsonl").read_text(encoding="utf-8").splitlines()
           if ln.strip()]
    assert any(e.get("status") == "resource_blocked" for e in evs), evs


def test_resume_skips_rounds_that_already_have_a_decision(tmp_path,
                                                          monkeypatch):
    """断点恢复：已有判定的轮不重跑（中断后接着跑）。

    这一条用**真的** `cmd_rounds`（只替换 exp_dir 指到 tmp），并且 `--plan-only`
    保证不训练：resume 命中时那一轮直接跳过，不留任何训练产物。
    """
    loop = _load()
    monkeypatch.setattr(loop, "exp_dir", lambda rid: tmp_path / "exp" / rid)
    run_dir = tmp_path / "exp" / "e3"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "champion.json").write_text(json.dumps(
        {"by_seed": {"42": 0.4}, "source": "baseline_arm"}), encoding="utf-8")
    (run_dir / "decision_dup-r0.json").write_text(json.dumps(
        {"candidate_id": "dup-r0",
         "pairings": {"road_iou": {"metric": "road_iou", "n": 3}},
         "hard_gate": {}, "decision": {"decision": "rejected",
                                       "reasons": ["x"]}}), encoding="utf-8")
    props = tmp_path / "props.json"
    props.write_text(json.dumps({"proposals": [
        {"candidate_id": "dup", "factor": {"add_runs": ["x"]}}]}),
        encoding="utf-8")
    rc = loop.main(["rounds", "--run-id", "e3", "--rounds", "1", "--runs",
                    "logs/m5_seg/diverse_town_20260924/front_main",
                    "--eval-runs", "logs/m5_seg/diverse_wide_20260924/front_main",
                    "--seeds", "42", "--proposals", str(props), "--resume",
                    "--plan-only"])
    assert rc == 0
    assert not (run_dir / "round0").exists(),         "resume 跳过的轮不得留下训练产物"
