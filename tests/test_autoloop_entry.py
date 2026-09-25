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
import time
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


def test_the_wall_clock_limit_actually_stops_the_entry(sandbox, tmp_path):
    """G08：`max_wall_minutes` 真的能停（不是配置里的装饰）。

    起点落在 run 目录：这里预置一个"已经跑了 3 小时"的起点，配置上限 60 分钟
    -> 入口必须**在采集/训练之前**停下（rc=8）并写事件。
    """
    loop, root, calls = sandbox
    cfg = _config(tmp_path, max_wall_minutes=60.0, collect="off")
    d = loop.exp_dir("e20")
    d.mkdir(parents=True, exist_ok=True)
    (d / "run_state.json").write_text(json.dumps(
        {"started_at": time.time() - 3 * 3600}), encoding="utf-8")
    rc = loop.main(["run", "--run-id", "e20", "--no-dry-run", "--config",
                    str(cfg), "--user-inactive"])
    assert rc == 8, f"墙钟到顶应走停止路径（rc=8），实际 {rc}"
    assert not calls, "墙钟到顶时不得调用 rounds"
    evs = [json.loads(ln) for ln in (d / "events.jsonl").read_text(
        encoding="utf-8").splitlines() if ln.strip()]
    assert any("wall clock" in (e.get("note") or "") for e in evs), evs
    # 停止后开新一轮：起点被清掉（否则下次启动还会立刻停）
    assert not (d / "run_state.json").exists(), "停止后应重开墙钟"


def test_a_declared_active_user_pauses_before_any_work(sandbox, tmp_path):
    """G08：声明用户在用时，资源门在采集/训练之前拦下（rc=5）。"""
    loop, root, calls = sandbox
    cfg = _config(tmp_path, collect="off")
    rc = loop.main(["run", "--run-id", "e21", "--no-dry-run", "--config",
                    str(cfg), "--user-active"])
    assert rc == 5, f"用户在用应被资源门拦下，实际 {rc}"
    assert not calls
    evs = [json.loads(ln) for ln in (loop.exp_dir("e21") / "events.jsonl"
                                     ).read_text(encoding="utf-8").splitlines()
           if ln.strip()]
    assert any(e.get("status") == "resource_blocked" for e in evs), evs
    assert any("user is using the machine" in (e.get("note") or "")
               for e in evs), evs


def test_run_no_dry_run_reuses_the_rounds_flow_and_records_the_budget(
        sandbox, tmp_path):
    loop, _root, calls = sandbox
    cfg = _config(tmp_path)
    rc = loop.main(["run", "--run-id", "e1", "--no-dry-run", "--config",
                    str(cfg), "--user-inactive"])
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
                    str(cfg), "--user-inactive"])
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


def test_the_daily_gpu_cap_counts_the_whole_machine(sandbox, tmp_path,
                                                  monkeypatch):
    """每日上限是"这台机器今天跑了多少"，不是"这个 run 跑了多少"。

    实测缺口：只按 run 记账时，换一个 run-id 当天已用时长就归零，上限形同虚设
    （本轮实测：三次重跑同一个 run-id 累计 60.7 min，而别的 run-id看不到）。
    """
    loop, _root, calls = sandbox
    monkeypatch.setattr(loop, "machine_gpu_path",
                        lambda: tmp_path / "machine.json")
    import time as _t
    today = _t.strftime("%Y-%m-%d")
    (tmp_path / "machine.json").write_text(json.dumps({
        today: {"minutes": 130.0, "runs": {"something_else": 130.0}}}),
        encoding="utf-8")
    cfg = _config(tmp_path, daily_gpu_minutes=120.0)
    rc = loop.main(["run", "--run-id", "e9", "--no-dry-run", "--config",
                    str(cfg), "--user-inactive"])
    assert not calls, "机器合计超上限就不能再训练"
    assert rc == 5
    evs = [json.loads(ln) for ln in
           (loop.exp_dir("e9") / "events.jsonl").read_text(
               encoding="utf-8").splitlines() if ln.strip()]
    assert any(e.get("status") == "resource_blocked" for e in evs), evs


def test_gpu_minutes_sum_across_run_ids(sandbox, tmp_path, monkeypatch):
    """机器级账本把当天所有 run 加在一起；每个 run 的自己那份不变。"""
    loop, _root, _calls = sandbox
    monkeypatch.setattr(loop, "machine_gpu_path",
                        lambda: tmp_path / "machine.json")
    monkeypatch.setattr(loop, "gpu_minutes_path",
                        lambda rid: tmp_path / f"{rid}.json")
    loop.add_gpu_minutes("runA", 10.0)
    loop.add_gpu_minutes("runB", 5.5)
    assert loop.machine_gpu_minutes_today() == 15.5
    assert loop.gpu_minutes_today("runB") == 5.5
    blob = json.loads((tmp_path / "machine.json").read_text(encoding="utf-8"))
    runs = list(blob.values())[0]["runs"]
    assert runs == {"runA": 10.0, "runB": 5.5}, runs





def test_the_closing_event_respects_the_state_machine(sandbox, tmp_path,

                                                      monkeypatch):

    """整轮失败后的收尾事件也要合法（实测踩到：从 failed 只能到



    queued/paused，写 training 会抛 ValueError，整轮跑完却以 rc=1 结束）。

    """

    loop, _root, _calls = sandbox

    from beamng_autopilot.experiments.events import Event



    def failing_rounds(args):

        log = loop._log(args.run_id)

        log.append(loop._ev(args.run_id, "failed", "train_error",

                            note="boom"))

        return 1



    monkeypatch.setattr(loop, "cmd_rounds", failing_rounds)

    cfg = _config(tmp_path, rounds=1)

    rc = loop.main(["run", "--run-id", "e10", "--no-dry-run", "--config",

                    str(cfg), "--user-inactive"])

    assert rc == 1, "rounds 的退出码要透传"

    evs = [json.loads(ln) for ln in

           (loop.exp_dir("e10") / "events.jsonl").read_text(

               encoding="utf-8").splitlines() if ln.strip()]

    assert evs[-1]["phase"] == "paused", evs[-1]

    assert any(e["status"] == "train_error" for e in evs)

    assert Event is not None





def test_a_stop_condition_blocks_starting_the_round(sandbox, tmp_path,

                                                    monkeypatch):

    """G01 反例：连续无收益达到上限后，入口**不得再采集/训练**。



    方案 G01：原实现计算完 should_stop 只打印，随后仍可能进入采集；

    止损必须是**启动硬门**（采集前、训练前都检查，恢复历史后一样）。

    """

    loop, _root, calls = sandbox

    d = loop.exp_dir("e11")

    d.mkdir(parents=True, exist_ok=True)

    for i in range(2):                     # max_rounds_without_gain=2

        (d / f"decision_cand{i}-r{i}.json").write_text(json.dumps({

            "candidate_id": f"cand{i}", "pairings": {}, "hard_gate": {},

            "decision": {"decision": "rejected",

                         "reasons": ["no credible gain"]}}),

            encoding="utf-8")

    collected = []



    def fake_collect(args, cfg, *, log):

        collected.append(1)

        return None, 6



    monkeypatch.setattr(loop, "collect_once", fake_collect)

    cfg = _config(tmp_path, max_rounds_without_gain=2, collect="tech")

    rc = loop.main(["run", "--run-id", "e11", "--no-dry-run", "--config",

                    str(cfg), "--user-inactive"])

    assert not collected, "停止条件满足时不得启动采集"

    assert not calls, "停止条件满足时不得调用 rounds"

    assert rc == 8, f"停止应有专门的退出码，实际 {rc}"

    evs = [json.loads(ln) for ln in

           (d / "events.jsonl").read_text(encoding="utf-8").splitlines()

           if ln.strip()]

    assert any(e.get("status") == "stopped_before_start" for e in evs), evs





def test_the_machine_lease_blocks_a_second_run(sandbox, tmp_path,

                                               monkeypatch):

    """G02：不同 run 也不能同时做重型实验（机器级租约）。



    旧实现是 run 内锁 + 非原子写 + 6 小时无条件接管；现在是机器级租约，

    活着的持有者抢不走。测试注入探测函数，不需真进程。

    """

    loop, _root, calls = sandbox

    import time as _t

    lease_path = tmp_path / "machine_lease.json"

    monkeypatch.setattr(loop, "machine_lease", lambda: loop.MachineLease(

        lease_path, alive_fn=lambda p: True,

        created_fn=lambda p: "t-holder"))

    lease_path.write_text(json.dumps({

        "pid": 424242, "created": "t-holder", "t": _t.time(),

        "hb": _t.time(), "host": "OTHER"}), encoding="utf-8")

    cfg = _config(tmp_path)

    rc = loop.main(["run", "--run-id", "e12", "--no-dry-run", "--config",

                    str(cfg), "--user-inactive"])

    assert rc == 4, f"租约被占时应拒绝启动，实际 {rc}"

    assert not calls, "租约被占时不得调用 rounds"

    evs = [json.loads(ln) for ln in

           (loop.exp_dir("e12") / "events.jsonl").read_text(

               encoding="utf-8").splitlines() if ln.strip()]

    assert any(e.get("status") == "lease_blocked" for e in evs), evs

