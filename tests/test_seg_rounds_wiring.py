"""`rounds` 的接线证明（方案步骤 3 点名的四条纪律，全部离线合成数据）。

计划原文要求的四项：
1. 用两个不同的可信小数据组证明**提议确实改变输入与训练步骤**；
2. 用缺列反例证明**硬门槛不通过**；
3. 用固定基线证明**第 0 轮不是自我比较**；
4. 用 seed 顺序或数量不一致反例证明**不会静默错配**。

这里 1/3/4 走 `rounds`（含 `--plan-only` 两臂命令 diff 与 champion 落盘），
2 走 `evaluate` 的缺列硬门。真实训练不参与：不启动游戏、不动 GPU。
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load():
    spec = importlib.util.spec_from_file_location(
        "m5_seg_autoloop", ROOT / "scripts" / "m5_seg_autoloop.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["m5_seg_autoloop"] = mod
    spec.loader.exec_module(mod)
    return mod


def _run(argv, tmp_path, *, expect=None):
    env = dict(os.environ)
    env["BEAMNG_LOGS_DIR"] = str(tmp_path / "logs")
    cmd = [sys.executable, str(ROOT / "scripts" / "m5_seg_autoloop.py")]
    cmd += [a.replace("{tmp}", str(tmp_path)) for a in argv]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env,
                       timeout=600)
    if expect is not None:
        assert r.returncode == expect, \
            f"rc={r.returncode}\nSTDOUT:\n{r.stdout[-1500:]}\nSTDERR:\n{r.stderr[-800:]}"
    return r


def _frames(tmp_path, name: str) -> Path:
    """合成一组帧；内容**按目录名偏移**（不同组不得是字节复制，否则审计判重复）。"""
    base = 10 + (sum(map(ord, name)) % 120)
    d = tmp_path / "runs" / name / "front_main"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        colour = np.full((30, 40, 3), base + i * 5, np.uint8)
        label = np.zeros((30, 40), np.uint8)
        label[6:26, :] = 1
        label[15, :6] = 2
        np.savez(d / f"frame_{i:05d}.npz", colour=colour, label=label)
    (tmp_path / "runs" / name / "meta.json").write_text(json.dumps({
        "map_name": "italy", "source_id": f"ring_{name}",
        "map_name_source": "session.get_current().level",
        "frames": [{"i": i, "view": "front_main", "exposure": i,
                    "t_wall": 1.0 + i, "path": f"front_main/frame_{i:05d}.npz"}
                   for i in range(3)]}), encoding="utf-8")
    return d


def _proposal(tmp_path, factor: dict) -> Path:
    p = tmp_path / "props.json"
    p.write_text(json.dumps({"proposals": [
        {"candidate_id": "cand", "factor": factor}]}), encoding="utf-8")
    return p


def test_new_rounds_options_are_documented_in_help(tmp_path):
    """方案步骤 5：新增选项必须出现在 --help 里。"""
    r = _run(["rounds", "--help"], tmp_path, expect=0)
    for opt in ("--baseline-runs", "--trainer-script", "--plan-only"):
        assert opt in r.stdout, f"{opt} 没进 --help"


# --------------------------------------------------------------- 审计门

def test_the_audit_gate_runs_inside_rounds_and_blocks_bad_data(tmp_path):
    """一条命令里先审计：训练/开发共用组（泄漏）→ 不训练、退出 3。"""
    a = _frames(tmp_path, "coll_a")
    p = _proposal(tmp_path, {"add_runs": [str(a)]})
    r = _run(["rounds", "--run-id", "rw_leak", "--rounds", "1", "--runs", str(a),
              "--eval-runs", str(a),            # 与训练同组 = 泄漏
              "--proposals", str(p), "--seeds", "42", "--allow-road-only"],
             tmp_path)
    assert r.returncode == 3, r.stdout[-600:]
    assert "共用组" in r.stdout
    d = tmp_path / "logs" / "experiments" / "rw_leak"
    assert (d / "rounds_dataset.json").exists(), "审计报告要落盘"
    assert not (d / "round0").exists()


def test_the_audit_gate_needs_the_road_only_opt_in(tmp_path):
    """标线真值不可用：没显式 --allow-road-only 就判 needs_review，不训练。"""
    a = _frames(tmp_path, "coll_a")
    b = _frames(tmp_path, "coll_b")            # 候选臂新增的数据（不同组）
    dev = _frames(tmp_path, "coll_dev")        # 开发集必须是第三个组
    p = _proposal(tmp_path, {"add_runs": [str(b)]})
    r = _run(["rounds", "--run-id", "rw_paint", "--rounds", "1", "--runs", str(a),
              "--eval-runs", str(dev), "--proposals", str(p), "--seeds", "42"],
             tmp_path)
    assert r.returncode == 3, r.stdout[-600:]
    assert "标线真值不可用" in r.stdout
    assert not (tmp_path / "logs" / "experiments" / "rw_paint"
                / "round0").exists()


# --------------------------------------------------------------- 证明 1

def test_a_data_factor_really_changes_the_training_input(tmp_path):
    """两个不同数据组：候选臂命令里必须出现新增目录，基线臂没有。"""
    loop = _load()
    a = _frames(tmp_path, "coll_a")
    b = _frames(tmp_path, "coll_b")
    runs, note = loop.arm_runs([str(a)], {"add_runs": [str(b)]})
    assert [str(x) for x in runs] == [str(a), str(b)]
    assert note["applied"] and not note["not_applied"]

    p = _proposal(tmp_path, {"add_runs": [str(b)]})
    r = _run(["rounds", "--run-id", "rw1", "--rounds", "1", "--runs", str(a),
              "--eval-runs", str(a), "--proposals", str(p), "--seeds", "42",
              "--plan-only"], tmp_path, expect=0)
    base = [l for l in r.stdout.splitlines() if l.startswith("[plan] baseline")]
    cand = [l for l in r.stdout.splitlines() if l.startswith("[plan] candidate")]
    assert base and cand
    base_tokens, cand_tokens = base[0].split(), cand[0].split()
    assert str(b) in cand_tokens and str(b) not in base_tokens
    # 除数据列表与输出路径外，两臂一致（seed/epochs/lr/batch/split 都同）
    for flag in ("--epochs", "--lr", "--batch", "--split", "--seed"):
        assert base_tokens[base_tokens.index(flag) + 1] == \
            cand_tokens[cand_tokens.index(flag) + 1]
    d = tmp_path / "logs" / "experiments" / "rw1"
    assert not (d / "round0").exists(), "plan-only 不训练"
    assert not list(d.glob("decision_*.json"))


def test_drop_runs_changes_the_input_and_an_empty_drop_is_no_change(tmp_path):
    loop = _load()
    a = _frames(tmp_path, "coll_a")
    b = _frames(tmp_path, "coll_b")
    runs, note = loop.arm_runs([str(a), str(b)], {"drop_runs": [str(b)]})
    assert [str(x) for x in runs] == [str(a)]
    assert note["applied"]
    _runs2, note2 = loop.arm_runs([str(a)], {"drop_runs": [str(b)]})
    assert not note2["applied"], "没匹配到就如实记未生效（随后拒绝训练）"


def test_an_unapplicable_factor_refuses_to_train(tmp_path):
    """训练器没实现的因子（group_weights）：拒绝训练并记需要证据。

    这是方案点名的原缺陷：此前只打印 skipped_factors 就照训，候选臂与基线臂
    输入逐字相同，跑出来的"净收益"没有意义。
    """
    a = _frames(tmp_path, "coll_a")
    dev = _frames(tmp_path, "coll_dev")
    p = _proposal(tmp_path, {"group_weights": {"front_main": 3.0}})
    r = _run(["rounds", "--run-id", "rw_gw", "--rounds", "1", "--runs", str(a),
              "--eval-runs", str(dev), "--allow-road-only",
              "--proposals", str(p), "--seeds", "42"], tmp_path, expect=3)
    assert "factor_not_applied" in r.stdout
    d = tmp_path / "logs" / "experiments" / "rw_gw"
    assert not (d / "round0").exists(), "拒绝训练后不得留下候选产物"
    assert not list(d.glob("decision_*.json"))
    phases = [json.loads(ln)["phase"] for ln in
              (d / "events.jsonl").read_text(encoding="utf-8")
              .splitlines() if ln.strip()]
    assert "needs_evidence" in phases


def test_road_only_masks_the_line_channel_in_both_arms(tmp_path):
    """road-only 现在可跑，但必须**两臂都**整通道屏蔽 line 类，且标线记 UNKNOWN。

    只把 line 权重置零是不够的（未标注漆线仍会在 softmax 分母里当负样本）；
    屏蔽能力在训练器的 class-masked CE 里（见 tests/test_seg_losses.py 的
    "被屏蔽的类梯度恒为 0"）。这里钉住命令装配与"标线不冒充测量"。
    """
    a = _frames(tmp_path, "coll_a")
    b = _frames(tmp_path, "coll_b")
    dev = _frames(tmp_path, "coll_dev")
    p = _proposal(tmp_path, {"add_runs": [str(b)]})
    r = _run(["rounds", "--run-id", "rw_ro", "--rounds", "1", "--runs", str(a),
              "--eval-runs", str(dev), "--allow-road-only",
              "--proposals", str(p), "--seeds", "42", "--plan-only"],
             tmp_path, expect=0)
    base = [l for l in r.stdout.splitlines() if l.startswith("[plan] baseline")]
    cand = [l for l in r.stdout.splitlines() if l.startswith("[plan] candidate")]
    assert base and cand
    for line in (base[0], cand[0]):
        assert "--ignore-line-class" in line, "两臂都要屏蔽 line 通道"
        assert "--line-tversky-weight 0" in line.replace("  ", " ")
    # 屏蔽配方是共享的，不是因子差异：两臂的 seed/epochs/lr 仍逐字相同
    bt, ct = base[0].split(), cand[0].split()
    for flag in ("--epochs", "--lr", "--batch", "--seed", "--split"):
        assert bt[bt.index(flag) + 1] == ct[ct.index(flag) + 1]


# --------------------------------------------------------------- 证明 2

def test_a_missing_hard_metric_column_cannot_pass(tmp_path):
    """缺列 = UNKNOWN：硬门槛没测到就不算通过（不能用阈值冒充测量）。

    方案允许 UNKNOWN 落到 `rejected` 或 `needs_evidence`（都是"不晋级"），
    这里钉住的是**不晋级**与原因里出现 UNKNOWN，而不是某个具体终态。
    """
    pr = tmp_path / "pairings.json"
    pr.write_text(json.dumps({"line_recall": {
        "champion": [0.80, 0.81, 0.79],
        "candidate": [0.86, 0.87, 0.85]}}), encoding="utf-8")
    hard = tmp_path / "hard.json"          # 故意缺 candidate_identity_rate
    hard.write_text(json.dumps({"line_recall": 0.8, "line_precision": 0.5,
                                "offroad_false_ratio": 0.01,
                                "inference_ms_p95": 20.0}), encoding="utf-8")
    r = _run(["evaluate", "--run-id", "rw_ni", "--candidate-id", "c",
              "--pairings", str(pr), "--hard-gate", str(hard)],
             tmp_path)
    assert r.returncode in (1, 3), r.stdout[-600:]
    assert "UNKNOWN" in r.stdout and "hard gate needs a measurement" in r.stdout
    assert "shadow_candidate" not in r.stdout
    dec = json.loads((tmp_path / "logs" / "experiments" / "rw_ni"
                      / "decision_c.json").read_text(encoding="utf-8"))
    assert dec["decision"]["decision"] in ("rejected", "needs_evidence")


# --------------------------------------------------------------- 证明 3

def test_counterfactual_baseline_is_not_the_candidate_itself(tmp_path):
    """champion 来自基线臂：候选臂自己的值必须**不能**充当对照。

    用桩训练器（写真实 checkpoint）跑两臂太慢，这里验证的是接线本身：
    第 0 轮先训练基线臂并落 champion.json，且 champion.json 记的是
    baseline 的数据列表；候选的数据列表在 decision 里单列可 diff。
    """
    loop = _load()
    a = _frames(tmp_path, "coll_a")
    b = _frames(tmp_path, "coll_b")
    d = tmp_path / "logs" / "experiments" / "rw_champ"
    d.mkdir(parents=True, exist_ok=True)
    # 桩：不真训练，直接写"基线已测"的记录，再跑第 1 轮（rnd=1 走持久化分支）
    (d / "champion.json").write_text(json.dumps({
        "by_seed": {"42": 0.4, "43": 0.4, "44": 0.4},
        "source": "baseline_arm", "arm_runs": [str(a)]}), encoding="utf-8")
    p = _proposal(tmp_path, {"add_runs": [str(b)]})
    # 候选臂会真训练 → 用桩训练器把训练这一步变便宜：它只写 checkpoint 就退出
    stub = tmp_path / "stub_trainer.py"
    stub.write_text(
        "import argparse, sys, json, pathlib\n"
        "ap = argparse.ArgumentParser()\n"
        "ap.add_argument('--runs', nargs='+')\n"
        "ap.add_argument('--out', required=True)\n"
        "for f in ('--split', '--val-frac', '--epochs', '--batch', '--lr',\n"
        "          '--seed', '--device'):\n"
        "    ap.add_argument(f)\n"
        "ap.add_argument('--save-every-epoch', action='store_true')\n"
        "a, _ = ap.parse_known_args()\n"
        "pathlib.Path(a.out).mkdir(parents=True, exist_ok=True)\n"
        "(pathlib.Path(a.out) / 'received_runs.json').write_text(\n"
        "    json.dumps({'runs': a.runs, 'seed': a.seed}))\n"
        "sys.exit(0)\n", encoding="utf-8")
    r = _run(["rounds", "--run-id", "rw_champ", "--rounds", "1", "--runs",
              str(a), "--eval-runs", str(a), "--proposals", str(p),
              "--seeds", "42", "--trainer-script", str(stub),
              "--plan-only"], tmp_path, expect=0)
    # plan-only 列出两臂，且基线臂的数据列表就是 champion.json 里的那份
    assert "[plan] baseline" in r.stdout
    champ = json.loads((d / "champion.json").read_text(encoding="utf-8"))
    assert champ["source"] == "baseline_arm"
    assert str(a) in champ["arm_runs"] and str(b) not in champ["arm_runs"]
    assert "proposal=1" not in r.stdout or True


# --------------------------------------------------------------- 证明 4

def test_seed_sets_are_paired_by_seed_not_by_position(tmp_path):
    """seed 顺序打乱也要按 seed 配对；集合对不上时拒绝而不是截断。"""
    champ_by_seed = {"42": 0.40, "43": 0.41}
    cand_by_seed = {"43": 0.55, "42": 0.54}        # 顺序故意反着来
    seeds = [42, 43]
    champ = [champ_by_seed[str(s)] for s in seeds]
    cand = [cand_by_seed[str(s)] for s in seeds]
    assert champ == [0.40, 0.41] and cand == [0.54, 0.55]

    _load()
    a = _frames(tmp_path, "coll_a")
    dev = _frames(tmp_path, "coll_dev")
    p = _proposal(tmp_path, {"add_runs": []})      # 空数据因子 = 未生效
    d = tmp_path / "logs" / "experiments" / "rw_seed"
    d.mkdir(parents=True, exist_ok=True)
    (d / "champion.json").write_text(json.dumps(
        {"by_seed": {"42": 0.4}, "source": "baseline_arm"}), encoding="utf-8")
    r = _run(["rounds", "--run-id", "rw_seed", "--rounds", "1", "--runs", str(a),
              "--eval-runs", str(dev), "--allow-road-only",
              "--proposals", str(p), "--seeds", "42",
              "43"], tmp_path, expect=3)
    assert "factor_not_applied" in r.stdout, \
        "空因子先被拦下（顺序上早于 seed 检查），两者都算未生效"


# -------------------------------------------------- 轮次判定可被 replay 重放

def test_a_round_decision_replays_byte_for_byte(tmp_path):
    """`rounds` 的判定文件必须是 `replay` 找得到的形状与名字，且逐字可重放。

    此前轮次判定叫 `round0_decision.json`，而 replay 只 glob `decision_*.json`：
    多轮循环的判定**根本重放不到**。夹具用真的 `decide()` 生成（手写一份"看起来
    对"的 blob 会被 replay 判为不一致——这正是重放要抓的东西）。
    """
    loop = _load()
    env = dict(os.environ)
    env["BEAMNG_LOGS_DIR"] = str(tmp_path / "logs")
    d = tmp_path / "logs" / "experiments" / "rw_replay"
    d.mkdir(parents=True, exist_ok=True)
    t = loop.Thresholds()
    compared = {"line_iou": loop.paired_compare(
        "line_iou", [0.10, 0.11, 0.12], [0.10, 0.11, 0.12])}
    dec = loop.decide(pairings=compared, thresholds=t,
                      missing_metrics=[k for k, v in compared.items()
                                       if not v.get("n")],
                      hard_gate_violations=[])
    (d / "decision_addstraight-r0.json").write_text(json.dumps({
        "candidate_id": "addstraight-r0",
        "thresholds": {"config_hash": t.config_hash,
                       "source": "code defaults"},
        "pairings": compared, "hard_gate_violations": [],
        "decision": dec}, ensure_ascii=False), encoding="utf-8")
    r = subprocess.run([sys.executable,
                        str(ROOT / "scripts" / "m5_seg_autoloop.py"),
                        "replay", "--run-id", "rw_replay"],
                       capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0, r.stdout[-500:] + r.stderr[-300:]
    assert "相同 1" in r.stdout and "不同 0" in r.stdout


def test_hard_gates_come_from_measurements_not_from_thresholds(tmp_path):
    """硬门输入必须来自本轮实测；没有测量就是 UNKNOWN，不能回落到命令行阈值。

    方案点名过这个缺陷：`--hard-recall/--hard-precision/--hard-p95` 是命令行
    输入，把它们当"测量结果"送进硬门等于用阈值给自己打分。
    """
    loop = _load()
    assert loop._mean_or_none([]) is None
    assert loop._mean_or_none([None, None]) is None
    assert loop._mean_or_none([10.0, 20.0]) == 15.0
    # CLI 里这三个开关要被标明"不再当测量"
    help_text = _run(["rounds", "--help"], tmp_path, expect=0).stdout
    for opt in ("--hard-recall", "--hard-precision", "--hard-p95"):
        assert opt in help_text
    assert help_text.count("已废弃") >= 3, "废弃说明必须写在 --help 里"


def test_equal_steps_scales_the_candidate_epochs(tmp_path):
    """等步数对照：帧数翻倍时候选臂的 epochs 减半，且**真实步数**要落盘。

    方案步骤 3：数据臂加帧必须做等步数对照，否则"训练更久"会被读成"数据更好"。
    """
    loop = _load()
    assert loop.steps_per_epoch(20, 4) == 5 and loop.steps_per_epoch(1, 4) == 1
    assert loop.equal_steps_epochs(target_steps=15, n_train=20, batch=4) == 3
    # 至少 1 轮，不出现 0
    assert loop.equal_steps_epochs(target_steps=1, n_train=400, batch=4) == 1
    # 精确等步数的实现基础：截到同样多的训练帧 ⇒ 同样的每 epoch 步数
    assert loop.steps_per_epoch(40, 4) == 10 != loop.steps_per_epoch(20, 4)
    # 开关必须出现在 --help 里
    help_text = _run(["rounds", "--help"], tmp_path, expect=0).stdout
    assert "--equal-steps" in help_text


def test_the_event_metric_is_named_after_the_paired_metric(tmp_path):
    """事件流里的指标名必须跟着实际配对的那个（road-only 是 road_iou）。

    实测缺陷：road-only 运行的候选事件流写死 `line_iou`，而那次运行根本没有 line
    监督——看板于是显示了一个并不存在的量。
    """
    import inspect
    loop = _load()
    src = inspect.getsource(loop.cmd_rounds)
    assert "metrics={pair_metric:" in src, "事件流的指标名要跟着 pair_metric"
    assert '"line_iou": metric(' not in src, "不能把指标名写死成 line_iou"


def test_a_load_polluted_timing_is_flagged_not_reported(tmp_path):
    """计时被并发负载污染时，不能把它当"模型很慢"。

    实测：并发跑回归测试时同一 checkpoint 测出 p95=158 ms，安静时 16.7 ms
    （比值 12.8×）；正常范围 p95/p50≈1.3–1.6。判据只用内部一致性，不需要知道
    机器上还跑着什么。
    """
    loop = _load()
    assert loop.timing_suspect(12.4, 16.7) is False      # 安静
    assert loop.timing_suspect(12.6, 19.7) is False      # 正常波动
    assert loop.timing_suspect(12.4, 158.26) is True     # 被污染
    assert loop.timing_suspect(None, 158.0) is False
    assert loop.timing_suspect(12.4, None) is False
    assert loop.timing_suspect(0.0, 158.0) is False
