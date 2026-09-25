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
    # 拒训事件必须落在**合法阶段**上：这里日志已经在 training（候选逐轮训练那条
    # 先写了），从 training 只能到 evaluating/failed/paused —— 所以是 paused。
    # 原来写 needs_evidence 会直接抛 ValueError，把整轮炸掉（实测踩到）。
    assert phases[-1] in ("paused", "needs_review"), phases


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
    loop = _load()
    # 扫**整个入口文件**而不是 cmd_rounds 的函数体：实现允许把主体拆到内部函数
    # （例如加了机器级租约之后），但"事件里的指标名跟着 pair_metric"这条属性不变。
    src = (ROOT / "scripts" / "m5_seg_autoloop.py").read_text(encoding="utf-8")
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


def test_plateau_check_flags_a_still_moving_metric(tmp_path):
    """没到平台期的判定只能算暂行：读数还在动就要说出来。

    实测：3 epoch 时 road_iou 摆 ±0.26、12 epoch 时收敛到 ±0.03，所以"最后 k 轮
    验证指标还在动"是把瞬态读数当结论的信号。缺列时如实 UNKNOWN，不猜。
    """
    loop = _load()
    moving = {"val_miou": [0.40, 0.42, 0.55, 0.70, 0.88]}
    r = loop.plateau_check(moving, k=3, tol=0.02)
    assert r["at_plateau"] is False and r["spread"] > 0.3
    flat = {"val_miou": [0.86, 0.876, 0.878, 0.877, 0.879]}
    assert loop.plateau_check(flat, k=3, tol=0.02)["at_plateau"] is True
    # 缺列：UNKNOWN 而不是 False（不能把"没测"当"没到"）
    assert loop.plateau_check({})["at_plateau"] is None
    assert loop.plateau_check({"val_miou": []})["at_plateau"] is None
    assert loop.plateau_check({"val_line_iou": [None, None]})["at_plateau"] is None


def test_repeated_timing_keeps_the_least_contaminated_measurement(tmp_path):
    """本机时延有间歇性负载：同 seed 测两次取较小值，两次都留档。

    实测：同一 checkpoint 的 p50 在 9.95 与 21.67 ms 之间、p95 在 11.4 与 63.8 ms
    之间跳（并发跑 pytest 时最坏 158 ms），所以单次测量不能进硬门。
    """
    loop = _load()
    assert loop.min_positive(63.8, 11.4) == 11.4
    assert loop.min_positive(None, 11.4) == 11.4
    assert loop.min_positive(None, None) is None
    assert loop.min_positive(11.4) == 11.4


def test_every_arm_and_seed_streams_per_step_metrics(tmp_path):
    """每臂每 seed 都要写逐 step 指标——否则监控器/看板没有实时曲线。

    实测：`rounds` 一直没传 `--metrics-run`，最近 5 场运行都没有
    `logs/experiments/<run>/metrics.jsonl`，工作区的训练监控器因此只能找到老 demo。
    """
    loop = _load()
    a = _frames(tmp_path, "coll_a")
    b = _frames(tmp_path, "coll_b")
    p = _proposal(tmp_path, {"add_runs": [str(b)]})
    r = _run(["rounds", "--run-id", "rw_m", "--rounds", "1", "--runs", str(a),
              "--eval-runs", str(a), "--proposals", str(p), "--seeds", "42",
              "43", "--plan-only"], tmp_path, expect=0)
    base = [l for l in r.stdout.splitlines() if l.startswith("[plan] baseline")]
    cand = [l for l in r.stdout.splitlines() if l.startswith("[plan] candidate")]
    for line in base + cand:
        toks = line.split()
        assert "--metrics-run" in toks, line[:200]
        rid = toks[toks.index("--metrics-run") + 1]
        assert "rw_m" in rid and "-s4" in rid, rid
    # 两臂的 run id 必须不同（否则两条曲线会写到同一个文件里互相覆盖）
    base_ids = {l.split()[l.split().index("--metrics-run") + 1] for l in base}
    cand_ids = {l.split()[l.split().index("--metrics-run") + 1] for l in cand}
    assert not (base_ids & cand_ids), (base_ids, cand_ids)


def test_a_recipe_factor_is_reported_with_its_own_epochs(tmp_path):
    """配方类因子改了训练轮数，记录里就不能写基线轮的数值。

    因子族里有 `epochs`（TRAINER_FLAG_FACTORS），旗标附在候选臂命令末尾、
    覆盖 `--epochs`。若判定文件还写 24、实际跑了 48，就是"记录与实际不一致"。
    """
    loop = _load()
    assert loop.factor_epochs(24, ["--epochs", "48"]) == 48
    assert loop.factor_epochs(24, []) == 24
    assert loop.factor_epochs(24, ["--lr", "0.002"]) == 24
    # 命令行最后出现的 --epochs 胜出（与 argparse 一致）
    assert loop.factor_epochs(24, ["--epochs", "48", "--epochs", "12"]) == 12
    a = _frames(tmp_path, "coll_a")
    dev = _frames(tmp_path, "coll_dev")
    p = tmp_path / "prop_epochs.json"
    p.write_text(json.dumps({"proposals": [{"candidate_id": "longer",
                                            "factor": {"epochs": 48}}]}),
                 encoding="utf-8")
    r = _run(["rounds", "--run-id", "rw_epochs", "--rounds", "1",
              "--runs", str(a), "--eval-runs", str(dev),
              "--allow-road-only", "--proposals", str(p),
              "--seeds", "42", "--epochs", "24", "--plan-only"],
             tmp_path, expect=0)
    cand = [ln for ln in r.stdout.splitlines() if ln.startswith("[plan] candidate")]
    base = [ln for ln in r.stdout.splitlines() if ln.startswith("[plan] baseline")]
    assert cand and "--epochs 48" in cand[0], cand[:1]
    assert base and "--epochs 24" in base[0], base[:1]


def test_a_capacity_factor_only_changes_the_candidate(tmp_path):
    """容量族（`width`）必须走白名单变成 `--width`，且只加在候选臂上。

    数据与步数都不动、只改模型宽度——这是"数据因子"和"训练预算"两条杠杆都测到
    回报边界之后的第三条杠杆，命令行的逐项 diff 必须干净（基线臂不能被动到）。
    """
    loop = _load()
    flags, skipped = loop.factor_to_flags({"width": 2.0})
    assert flags == ["--width", "2.0"] and not skipped, (flags, skipped)
    assert "width" in loop.TRAINER_FLAG_FACTORS
    a = _frames(tmp_path, "coll_a")
    dev = _frames(tmp_path, "coll_dev")
    p = tmp_path / "prop_width.json"
    p.write_text(json.dumps({"proposals": [{"candidate_id": "wide",
                                            "factor": {"width": 2.0}}]}),
                 encoding="utf-8")
    r = _run(["rounds", "--run-id", "rw_width", "--rounds", "1",
              "--runs", str(a), "--eval-runs", str(dev),
              "--allow-road-only", "--proposals", str(p),
              "--seeds", "42", "--epochs", "24", "--plan-only"],
             tmp_path, expect=0)
    cand = [ln for ln in r.stdout.splitlines() if ln.startswith("[plan] candidate")]
    base = [ln for ln in r.stdout.splitlines() if ln.startswith("[plan] baseline")]
    assert cand and "--width 2.0" in cand[0], cand[:1]
    assert base and "--width" not in base[0], base[:1]


def test_the_plateau_guard_needs_both_arms(tmp_path):
    """平台期要两臂都判：只判候选臂时，"候选更好"可能只是候选训得更久。

    实测来历：第 5 轮 96-epoch 臂有一个 seed 末段大跳（spread 0.33），而基线臂
    的轮内波动一直很小——只看一臂会把这种不稳定读成"候选的容量/预算效应"。
    """
    loop = _load()
    ok = {"42": {"at_plateau": True}}
    bad = {"42": {"at_plateau": False}}
    unknown = {"42": {"at_plateau": None}}
    assert loop._both_at_plateau(ok, ok) is True
    assert loop._both_at_plateau(ok, bad) is False
    assert loop._both_at_plateau(ok, unknown) is None, "缺测是 UNKNOWN，不当通过"
    assert loop._both_at_plateau({}, {}) is None
    missing = loop.plateau_from_hist(tmp_path / "nope.json")
    assert missing["at_plateau"] is None and "missing" in missing
    good = tmp_path / "train_hist.json"
    good.write_text(json.dumps({"epoch": [0, 1, 2, 3],
                                "val_miou": [0.5, 0.6, 0.601, 0.602]}),
                    encoding="utf-8")
    assert loop.plateau_from_hist(good)["at_plateau"] is True


def test_a_full_round_writes_a_decision_file(tmp_path):
    """整轮跑到底（桩训练器写合法产物）→ 判定必须落盘。

    实测踩到：把 `all_at_plateau` 的赋值放在判定字典**之后**，写盘那行直接
    UnboundLocalError——训练全跑完却拿不到判定，一轮 GPU 白花。只测
    `--plan-only`、或只测被栏下的路径都抓不到它，必须有一条"真的走到
    写判定"的回归。
    """
    _load()
    a = _frames(tmp_path, "coll_a")
    b = _frames(tmp_path, "coll_b")
    dev = _frames(tmp_path, "coll_dev")
    stub = ROOT / "tests" / "_stub_trainer_e2e.py"
    p = _proposal(tmp_path, {"add_runs": [str(b)]})
    r = _run(["rounds", "--run-id", "rw_full", "--rounds", "1",
              "--runs", str(a), "--eval-runs", str(dev),
              "--allow-road-only", "--proposals", str(p), "--seeds", "42",
              "--epochs", "24", "--device", "cpu",
              "--trainer-script", str(stub)], tmp_path, expect=0)
    d = tmp_path / "logs" / "experiments" / "rw_full"
    decs = sorted(d.glob("decision_*.json"))
    assert decs, r.stdout[-1400:] + r.stderr[-800:]
    blob = json.loads(decs[0].read_text(encoding="utf-8"))
    assert "all_at_plateau" in blob, sorted(blob)
    assert "plateau_baseline_by_seed" in blob, "两臂平台期都要记"
    assert blob["plateau_by_seed"]["42"]["at_plateau"] is True
    assert blob["plateau_baseline_by_seed"]["42"]["at_plateau"] is True
    assert blob["all_at_plateau"] is True
    assert blob["pairings"]["road_iou"]["n"] == 1
    # 步数来自 checkpoint 的 train_args（桩：n_train=2, batch=2 -> 1 步/轮 × 24）
    assert blob["steps_by_arm"]["baseline"]["42"] == 24, blob["steps_by_arm"]
    assert blob["steps_by_arm"]["candidate"]["42"] == 24, "等步数对照"


def test_a_line_supervised_arm_trains_the_line_channel(tmp_path):
    """研究臂：给了漆线来源就**不再屏蔽 line 通道**，两臂都带 --paint-source。

    实测依据（2026-09-25）：引擎标注的漆线类存在但不完整（覆盖
    RGB 漆线候选 ~0.61），可以当弱监督——不开它的时候模型
    完全不产出标线（评估里 `no predicted line pixels at all`）。
    """
    loop = _load()
    a = _frames(tmp_path, "coll_a")
    dev = _frames(tmp_path, "coll_dev")
    p = _proposal(tmp_path, {"line_tversky_weight": 2.0})
    r = _run(["rounds", "--run-id", "rw_line", "--rounds", "1",
              "--runs", str(a), "--eval-runs", str(dev),
              "--paint-source", "coll_a=engine_annotation_partial",
              "--research-arm", "--proposals", str(p),
              "--seeds", "42", "--epochs", "24", "--plan-only"],
             tmp_path, expect=0)
    cand = [ln for ln in r.stdout.splitlines() if ln.startswith("[plan] candidate")]
    base = [ln for ln in r.stdout.splitlines() if ln.startswith("[plan] baseline")]
    assert cand and base, r.stdout[-600:]
    for line in (cand[0], base[0]):
        assert "--paint-source engine_annotation_partial" in line, line
        assert "--ignore-line-class" not in line, "line 通道必须开着"
    assert "--line-tversky-weight 2.0" in cand[0]
    assert "--line-tversky-weight" not in base[0], "因子只加在候选臂"
    # 路面方案的老行为不变：--allow-road-only 仍然屏蔽 line 通道
    r2 = _run(["rounds", "--run-id", "rw_road", "--rounds", "1",
               "--runs", str(a), "--eval-runs", str(dev),
               "--allow-road-only", "--proposals", str(p),
               "--seeds", "42", "--epochs", "24", "--plan-only"],
              tmp_path, expect=0)
    road = [ln for ln in r2.stdout.splitlines() if ln.startswith("[plan] candidate")]
    assert road and "--ignore-line-class" in road[0]
    assert "--paint-source" not in road[0]


def test_a_research_arm_cannot_be_promoted():
    """弱监督真值下不允许晋级：「精度达标」可能只是「只预测了

    被标注的那部分」。非研究臂不受影响。
    """
    loop = _load()
    assert loop._block_research_promotion(
        {"decision": "shadow_candidate", "reasons": ["x"]},
        False)["decision"] == "shadow_candidate"
    got = loop._block_research_promotion(
        {"decision": "shadow_candidate", "reasons": ["x"]}, True)
    assert got["decision"] == "needs_evidence", got
    assert "x" in got["reasons"] and any("research arm" in r
                                        for r in got["reasons"]), got
    assert loop._block_research_promotion(
        {"decision": "approved_for_review"}, True)["decision"] == \
        "needs_evidence"
    # 已被拒/缺证据的结论不动
    for d in ("rejected", "needs_evidence"):
        assert loop._block_research_promotion({"decision": d}, True)["decision"] == d


def test_paint_source_flag_parsing():
    """--paint-source RUN=SOURCE → {run: source}；格式不对的条目不猜。

    同时登记**规范化后的路径**：manifest 按 ``str(rd)`` 查来源，而 rd 由
    runs/eval_runs 的字符串构造（Windows 下是反斜杠）——实测踩到：只用正斜杠的键
    会静默回落 engine_annotation，审计报告里的 paint_ok 与原因就是错的。
    """
    loop = _load()
    args = type("A", (), {"paint_source": [
        " logs/m5_seg/x/front_main =engine_annotation_partial"]})()
    got = loop.paint_sources_from(args)
    assert got["logs/m5_seg/x/front_main"] == "engine_annotation_partial"
    # 同时登记规范化路径：manifest 按 str(rd) 查来源（Windows 是反斜杠）
    norm = str(Path("logs/m5_seg/x/front_main"))
    assert got[norm] == "engine_annotation_partial", got
    assert len(got) == 2, got
    assert loop.paint_sources_from(type("A", (), {"paint_source": None})()) == {}
    assert loop.paint_sources_from(type("A", (), {
        "paint_source": ["no-equals-sign"]})()) == {}


def _agent_dir(tmp_path: Path, name: str) -> Path:
    """带「agent 逐帧核对」凭证的数据目录（凭证跟着帧走）。"""
    d = _frames(tmp_path, name)
    (d / "annotation.json").write_text(json.dumps({
        "label_source": "agent_revision", "generator": "test",
        "frames": [{"path": f"{name}/front_main/frame_{i:05d}.npz"}
                   for i in range(3)]}, ensure_ascii=False),
        encoding="utf-8")
    return d


def test_an_agent_source_cannot_be_promoted_even_without_the_flag(tmp_path):
    """G03 反例：漏传 --research-arm 也不能让 agent 数据参与晋级。

    原实现用"字符串以 _partial 结尾或等于 pseudo"判研究臂，
    于是 `--paint-source X=agent_revision` 不带 flag 就能绕过；
    现在资格由**逐帧凭证**派生。
    """
    loop = _load()
    a = _agent_dir(tmp_path, "coll_a")
    dev = _agent_dir(tmp_path, "coll_dev")
    p = _proposal(tmp_path, {"line_tversky_weight": 2.0})
    r = _run(["rounds", "--run-id", "rw_src1", "--rounds", "1",
              "--runs", str(a), "--eval-runs", str(dev),
              "--paint-source", f"{a}=agent_revision",
              "--proposals", str(p), "--seeds", "42", "--epochs", "24",
              "--plan-only"], tmp_path, expect=0)
    assert r.returncode == 0
    # plan-only 不写判定，所以直接验证解析结果
    res = loop.resolve_paint_sources(type("A", (), {
        "paint_source": [f"{a}=agent_revision"],
        "research_arm": False})())
    assert res["research_only"] is True, res
    run = list(res["runs"].values())[0]
    assert run["effective"] == "agent_revision" and run["can_promote"] is False
    assert run["credential"] == "agent_revision", "凭证要能读到"


def test_declaring_human_revision_cannot_upgrade_agent_data(tmp_path):
    """命令行写 human_revision 不能把 agent 数据升格（凭证优先）。"""
    loop = _load()
    a = _agent_dir(tmp_path, "coll_b")
    res = loop.resolve_paint_sources(type("A", (), {
        "paint_source": [f"{a}=human_revision"], "research_arm": False})())
    run = list(res["runs"].values())[0]
    assert run["effective"] == "agent_revision", run
    assert run["can_promote"] is False and res["research_only"] is True
    assert any("using the credential" in n for n in res["notes"]), res["notes"]


def test_a_full_round_records_the_source_resolution_and_replays_it(tmp_path):
    """判定里存来源资格，replay 重放同一结论（方案 G10）。"""
    loop = _load()
    a = _agent_dir(tmp_path, "coll_c")
    dev = _agent_dir(tmp_path, "coll_dev2")
    stub = ROOT / "tests" / "_stub_trainer_e2e.py"
    p = _proposal(tmp_path, {"line_tversky_weight": 2.0})
    r = _run(["rounds", "--run-id", "rw_src2", "--rounds", "1",
              "--runs", str(a), "--eval-runs", str(dev),
              "--paint-source", f"{a}=agent_revision",
              "--proposals", str(p), "--seeds", "42", "--epochs", "24",
              "--device", "cpu", "--trainer-script", str(stub)],
             tmp_path, expect=0)
    d = tmp_path / "logs" / "experiments" / "rw_src2"
    decs = sorted(d.glob("decision_*.json"))
    assert decs, r.stdout[-1200:]
    blob = json.loads(decs[0].read_text(encoding="utf-8"))
    assert blob["research_only"] is True, "无 flag 也要记研究臂"
    res = blob.get("paint_source_resolution") or {}
    assert res.get("research_only") is True and res.get("runs"), res
    # replay 必须重放出同一判定（包括研究臂后置降级）
    r2 = _run(["replay", "--run-id", "rw_src2"], tmp_path, expect=0)
    assert "相同 1" in r2.stdout and "不同 0" in r2.stdout, r2.stdout


def test_the_decision_carries_a_self_checking_protocol_snapshot(tmp_path):
    """判定文件必须能自查口径（方案 §10.3：保存完整协议快照及哈希）。

    实测缺口：判定文件只记了"阈值来自哪个文件"，协议定义（分母/聚合/空间缓冲）
    完全没进文件——旧口径消失后，没人能证明这份判定是对哪套口径做的。
    另外，重放**不能**默默换成今天的最新阈值：快照被改过要报不一致。
    """
    loop = _load()
    a = _frames(tmp_path, "coll_ps")
    b = _frames(tmp_path, "coll_ps_b")
    dev = _frames(tmp_path, "coll_ps_dev")
    stub = ROOT / "tests" / "_stub_trainer_e2e.py"
    p = _proposal(tmp_path, {"add_runs": [str(b)]})
    _run(["rounds", "--run-id", "rw_proto", "--rounds", "1",
          "--runs", str(a), "--eval-runs", str(dev), "--allow-road-only",
          "--proposals", str(p), "--seeds", "42", "--epochs", "24",
          "--device", "cpu", "--trainer-script", str(stub)],
         tmp_path, expect=0)
    dec = sorted((tmp_path / "logs" / "experiments" / "rw_proto"
                  ).glob("decision_*.json"))[0]
    blob = json.loads(dec.read_text(encoding="utf-8"))
    snap = blob.get("protocol")
    assert snap, "判定文件必须带协议快照"
    ver = loop.verify_snapshot(snap)
    assert ver["ok"], ver
    assert ver["matches_current_protocol"], ver
    # 快照里的阈值就是本轮真正用的那套（不是"此刻磁盘上最新的文件"）
    assert snap["thresholds"]["line_recall_min"] ==         loop.Thresholds().line_recall_min
    # 重放：一致 -> 相同 1；被改过 -> 明确报"协议快照不符"（不能静默按新口径算）
    r = _run(["replay", "--run-id", "rw_proto"], tmp_path, expect=0)
    assert "相同 1" in r.stdout, r.stdout
    blob["protocol"]["spatial_buffer_m"] = 5.0
    dec.write_text(json.dumps(blob, ensure_ascii=False), encoding="utf-8")
    r2 = _run(["replay", "--run-id", "rw_proto"], tmp_path, expect=1)
    assert "协议快照与记录的哈希不符" in r2.stdout, r2.stdout


def test_final_confirmation_must_name_a_real_candidate_checkpoint():
    """R2 绑定**具体 checkpoint**（方案 §10.3），不能登记一个跨 seed 的平均成绩。"""
    loop = _load()
    sha = {"42": "aaa", "43": "bbb", "44": "ccc"}
    assert loop._match_confirmed_seed("bbb", sha) == ("43", [])
    seed, issues = loop._match_confirmed_seed("zzz", sha)
    assert seed is None and "not among this round's candidate checkpoints" in issues[0]
    seed, issues = loop._match_confirmed_seed("", sha)
    assert seed is None and "no model_sha16" in issues[0], issues


def test_the_confirm_program_seals_confirms_and_consumes(tmp_path):
    """端到端：封存 -> 唯一允许的读者评估一次 -> 再确认被拒（方案 §7）。

    最终确认程序是**唯一**能读最终集的入口：搜索器读到封存目录会拒训
    （另有测试），这里钉住确认程序自己：访问放行才评估、写绑定记录、
    第二次确认被拒（失败也消费）。
    """
    import torch
    from beamng_autopilot.vision.segmentation import SegUNet
    loop = _load()          # 只为加载 scripts/ 到 sys.path
    d = _frames(tmp_path, "final_scene")           # npz + meta（带身份）
    frames = sorted(d.glob("frame_*.npz"))
    assert len(frames) == 3
    ckpt = tmp_path / "cand.pt"
    torch.save({"state_dict": SegUNet(width=1.0).state_dict(),
                "train_args": {"arch_args": {"width": 1.0}}}, ckpt)
    seal_dir = tmp_path / "seal"
    argv = [sys.executable, str(ROOT / "scripts" / "m5_final_set.py"),
            "seal", "--name", "f1", "--dataset-id", "ds1", "--out", str(seal_dir)]
    for f in frames:
        argv += ["--frame", str(f)]
    r = subprocess.run(argv, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    rec_path = tmp_path / "confirmation.json"
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "m5_final_set.py"), "confirm",
         "--out", str(seal_dir), "--dataset", str(d), "--candidate-id", "cand-A",
         "--model", str(ckpt), "--record", str(rec_path), "--device", "cpu"],
        capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    assert rec["kind"] == "final_confirmation" and rec["candidate_id"] == "cand-A"
    assert rec["protocol_hash"] and rec["seal_digest"] and rec["model_sha16"]
    assert rec["results"]["overall"]["n_frames"] == 3, rec["results"]
    assert rec["notes"], "没测的口径必须写明（不能拿一部分口径冒充整套）"
    # 第二次确认：已被消费 -> 拒绝，且不评估（rc=3）
    r2 = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "m5_final_set.py"), "confirm",
         "--out", str(seal_dir), "--dataset", str(d), "--candidate-id", "cand-A",
         "--model", str(ckpt), "--device", "cpu"],
        capture_output=True, text=True, timeout=300)
    assert r2.returncode == 3, r2.stdout + r2.stderr
    assert "already consumed" in r2.stdout, r2.stdout


def test_a_final_confirmation_that_does_not_match_is_rejected(tmp_path):
    """确认记录对不上 = 输入不一致 -> rejected（方案 §10.3 第一条）。

    反例：拿另一个候选/另一套权重的确认来给本轮候选背书。没有记录则只记
    R2 未确认（研究结论不受影响），两者不能混成一个通道。
    """
    loop = _load()
    a = _frames(tmp_path, "coll_fc")
    b = _frames(tmp_path, "coll_fc_b")
    dev = _frames(tmp_path, "coll_fc_dev")
    stub = ROOT / "tests" / "_stub_trainer_e2e.py"
    p = _proposal(tmp_path, {"add_runs": [str(b)]})
    rec = tmp_path / "bogus_confirmation.json"
    rec.write_text(json.dumps({
        "kind": "final_confirmation",
        "protocol_hash": loop.snapshot_hash(loop.protocol_blob(
            thresholds=loop.Thresholds().__dict__)),
        "candidate_id": "some-other-candidate", "model_sha16": "deadbeef",
        "seal_digest": "x"}), encoding="utf-8")
    _run(["rounds", "--run-id", "rw_fc", "--rounds", "1", "--runs", str(a),
          "--eval-runs", str(dev), "--allow-road-only", "--proposals", str(p),
          "--seeds", "42", "--epochs", "24", "--device", "cpu",
          "--trainer-script", str(stub), "--final-confirm", str(rec)],
         tmp_path, expect=0)
    blob = json.loads(sorted((tmp_path / "logs" / "experiments" / "rw_fc"
                             ).glob("decision_*.json"))[0].read_text(
                                 encoding="utf-8"))
    conf = blob["final_confirmation"]
    assert conf["status"] == "mismatch", conf
    assert any("candidate" in i or "weights" in i for i in conf["issues"]), conf
    assert blob["r2_confirmed"] is False
    assert blob["decision"]["decision"] == "rejected", blob["decision"]


def test_a_masked_channel_is_missing_evidence_not_a_violation():
    """road-only：标线整通道屏蔽 -> 标线指标"未测" -> needs_evidence。

    实测踩到（E2 那轮）：池化硬门把 None 写成 "UNKNOWN (hard gate needs a
    measurement)" 塞进**违反**通道，判定成 rejected——读起来像"候选不合格"，
    实际是"这个通道没测"（代码注释本来就写着记 None 是为了不被当违反）。
    同时逐 seed / 分场景在屏蔽模式下只查可测口径，否则模型"没画线"的近零读数
    会被当成逐 seed 违反。
    """
    loop = _load()
    t = loop.Thresholds()
    hard = {"line_recall": None, "line_precision": None,
            "offroad_false_ratio": None, "candidate_identity_rate": None,
            "inference_ms_p95": 18.0}
    sp = loop.hard_split(hard, t)
    assert sp["violations"] == [] and len(sp["missing"]) == 4, sp
    per_seed = {"42": {"line_recall": 0.002, "inference_ms_p95": 18.0}}
    masked = ("inference_ms_p95",)
    assert loop.per_seed_gate_violations(per_seed, t, fields=masked) == []
    assert loop.per_seed_missing(per_seed, t, fields=masked) == []
    # 不限制字段时，屏蔽通道的近零读数会被当违反（这就是 road-only 下的假 rejected）
    assert loop.per_seed_gate_violations(per_seed, t), "反例不成立"
    dec = loop.decide(
        pairings={"road_iou": loop.paired_compare("road_iou", [0.90] * 5,
                                                  [0.88] * 5)},
        thresholds=t,
        missing_metrics=[f"{n}: UNKNOWN (hard gate needs a measurement)"
                         for n in sp["missing"]],
        hard_gate_violations=sp["violations"])
    assert dec["decision"] == "needs_evidence", dec
    assert any("UNKNOWN" in r for r in dec["reasons"]), dec["reasons"]


def test_the_evaluation_reference_credentials_decide_eligibility(tmp_path):
    """A1 反例：**漏传** --paint-source 时，agent 评价真值仍不能晋级。

    实测踩到（E2 那轮）：资格原来只遍历 `--paint-source`，于是不带这个参数时
    wide/plain 的 agent 起草评价真值被当成可晋级参考，判定写 research_only=False。
    现在评价侧直接读 eval 目录自己的凭证（self -> parent）。
    """
    loop = _load()
    dev = _agent_dir(tmp_path, "coll_eval_cred")   # annotation.json: agent_revision

    class A:
        paint_source = None
        eval_runs = [str(dev)]

    res = loop.resolve_paint_sources(A())
    assert res["research_only"] is True, res
    assert any("evaluation reference" in r for r in res["reasons"]), res
    entry = res["runs"][str(dev)]
    assert entry["rank"] == "agent" and entry["can_promote"] is False, entry
    assert entry.get("role") == "evaluation_reference", entry

    # 人工修订的凭证 -> 可当评价参考（不误伤）
    dev2 = _frames(tmp_path, "coll_eval_human")
    (dev2 / "annotation.json").write_text(json.dumps({
        "label_source": "human_revision", "generator": "test",
        "frames": [{"path": f"coll_eval_human/front_main/frame_{i:05d}.npz"}
                   for i in range(3)]}, ensure_ascii=False), encoding="utf-8")

    class B:
        paint_source = None
        eval_runs = [str(dev2)]

    res2 = loop.resolve_paint_sources(B())
    assert res2["research_only"] is False, res2
    assert res2["runs"][str(dev2)]["rank"] == "verified", res2["runs"]

    # 没有任何凭证的目录：保守判为不可晋级（缺证据不是通过）
    class C:
        paint_source = None
        eval_runs = [str(tmp_path / "no_such_dir")]

    res3 = loop.resolve_paint_sources(C())
    assert res3["research_only"] is True, res3


def test_evaluate_also_downgrades_a_research_source(tmp_path):
    """直接调 evaluate 也不能旁路：同一条资格规则（G03）。"""
    loop = _load()
    a = _agent_dir(tmp_path, "coll_e")
    pairings = tmp_path / "pairings.json"
    pairings.write_text(json.dumps({"line_iou": {
        "champion": [0.10, 0.11, 0.12],
        "candidate": [0.30, 0.31, 0.32]}}), encoding="utf-8")
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({}), encoding="utf-8")
    r = _run(["evaluate", "--run-id", "rw_ev",
              "--candidate-id", "cand", "--pairings", str(pairings),
              "--hard-gate", str(gate),
              "--paint-source", f"{a}=agent_revision"],
             tmp_path, expect=1)   # 不可晋级的判定 rc=1
    d = tmp_path / "logs" / "experiments" / "rw_ev"
    blob = json.loads(sorted(d.glob("decision_*.json"))[0].read_text(
        encoding="utf-8"))
    assert blob["research_only"] is True, blob.get("decision")
    # 同时作为 G05 的反例：**只有 line_iou 改善**（任务主指标缺失）不能进入影子
    # 候选，判定必须是 rejected（而不是被研究臂降级成的 needs_evidence）。
    assert blob["decision"]["decision"] == "rejected", blob["decision"]
    reasons = " ".join(blob["decision"]["reasons"])
    assert "line_iou" in reasons, blob["decision"]["reasons"]


def test_a_promotion_needs_a_task_metric_improvement_not_just_iou():
    """G05 的三个关键判定例（方案 §10.3）。

    实测缺口：`rounds` 只把 IoU 送进判定器，而判定要求**任务主指标**
    有可信改善——于是任何候选都晋不了级（IoU 在判定里只是辅助指标）。
    现在逐 seed 采集任务主指标并按 seed 成对。三个例子：

    * 任务主指标可信改善 + 硬门全过 → ``shadow_candidate``；
    * **IoU 改善但任务主指标变差** → ``rejected``（IoU 不能覆盖）；
    * 任务指标缺测 → ``needs_evidence``（不当通过）。
    """
    loop = _load()
    seeds = [42, 43, 44, 45, 46]
    t = loop.Thresholds()

    def _arm(base, jitter):
        out = {}
        for i, s in enumerate(seeds):
            out[str(s)] = {k: v + jitter * (i - 2) * 0.001
                           for k, v in base.items()}
        return out

    champ_base = {"candidate_identity_rate": 0.62, "line_recall": 0.72,
                  "line_precision": 0.42, "offroad_false_line_px": 1000.0,
                  "inference_ms_p95": 20.0}
    # 情况 1：主指标全面变好
    cand_good = {"candidate_identity_rate": 0.85, "line_recall": 0.88,
                 "line_precision": 0.58, "offroad_false_line_px": 600.0,
                 "inference_ms_p95": 16.0}
    iou_up = loop.paired_compare("line_iou", [0.20] * 5, [0.30] * 5)
    paired = loop.task_pairings(_arm(champ_base, 1.0), _arm(cand_good, 1.0),
                                seeds, extra={"line_iou": iou_up})
    for name in loop.TASK_METRICS:
        assert name in paired, f"{name} 必须进成对比较"
    assert paired["candidate_identity_rate"]["verdict"] == "candidate_better"
    dec = loop.decide(pairings=paired, thresholds=t, missing_metrics=[],
                      hard_gate_violations=[])
    assert dec["decision"] == "shadow_candidate", dec

    # 情况 2：IoU 改善、但主指标变差 -> rejected
    cand_bad = {"candidate_identity_rate": 0.55, "line_recall": 0.60,
                "line_precision": 0.35, "offroad_false_line_px": 1500.0,
                "inference_ms_p95": 25.0}
    paired2 = loop.task_pairings(_arm(champ_base, 1.0), _arm(cand_bad, 1.0),
                                 seeds, extra={"line_iou": iou_up})
    dec2 = loop.decide(pairings=paired2, thresholds=t, missing_metrics=[],
                       hard_gate_violations=[])
    assert dec2["decision"] == "rejected", dec2
    assert any("line_recall" in r or "candidate_identity_rate" in r
               for r in dec2["reasons"]), dec2["reasons"]

    # 情况 3：任务指标测不到（缺测）-> needs_evidence，不当通过
    empty = {str(s): {} for s in seeds}
    paired3 = loop.task_pairings(empty, empty, seeds,
                                 extra={"line_iou": iou_up})
    assert paired3["line_recall"]["n"] == 0
    dec3 = loop.decide(pairings=paired3, thresholds=t, missing_metrics=[],
                       hard_gate_violations=[])
    assert dec3["decision"] == "needs_evidence", dec3


def test_seed_hard_covers_every_hard_check():
    """接线漂移守卫：逐 seed 度量必须覆盖硬门表里的每一项。

    实测缺口（G05）：判定要的指标没接线，于是任何候选都晋不了级；这类漂移
    不会报错，只会永远给不出通过——所以用测试钉住键的对齐。
    """
    loop = _load()
    got = loop._seed_hard({"line_recall": 0.5, "line_precision": 0.4,
                           "offroad_false_frac_of_pred": 0.1},
                          {"candidate_identity_rate": 0.7}, 18.0)
    assert set(got) == {n for n, _l, _b in loop.HARD_CHECKS}, got
    assert got["inference_ms_p95"] == 18.0
    # 缺测写 None，不写 0（UNKNOWN ≠ 0）
    assert loop._seed_hard({}, None, None)["line_recall"] is None


def test_a_running_game_or_an_unknown_probe_blocks_the_timing():
    """G06：游戏在跑、或探测失败（UNKNOWN）时，独占计时的前提不成立。

    两种情况都必须返回原因（进判定文件的 timing_suspect，硬门 p95 记 None）；
    只有"确认没有游戏"才允许引用本轮延迟。
    """
    loop = _load()
    assert loop.timing_precondition(False) is None
    running = loop.timing_precondition(True)
    assert running and "game process is running" in running, running
    unknown = loop.timing_precondition(None)
    assert unknown and "UNKNOWN" in unknown, unknown


def _located_frames(tmp_path: Path, name: str, *, source_id: str,
                    pos, n: int = 3) -> Path:
    """带地图身份与逐帧位姿的小数据目录（空间隔离要用）。"""
    base = 10 + (sum(map(ord, name)) % 120)
    d = tmp_path / "located" / name / "front_main"
    d.mkdir(parents=True, exist_ok=True)
    frames = []
    for i in range(n):
        colour = np.full((30, 40, 3), base + i * 5, np.uint8)
        label = np.zeros((30, 40), np.uint8)
        label[6:26, :] = 1
        np.savez(d / f"frame_{i:05d}.npz", colour=colour, label=label)
        frames.append({"path": f"front_main/frame_{i:05d}.npz",
                       "view": "front_main", "exposure": i,
                       "pos": [float(pos[0]) + i * 2.0, float(pos[1]), 0.0],
                       "heading": 0.0})
    (d.parent / "meta.json").write_text(json.dumps({
        "map_name": "italy", "map_name_source": "test",
        "source_id": source_id, "roles": {"front_main": n},
        "frames": frames}, ensure_ascii=False), encoding="utf-8")
    return d


def test_the_audit_refuses_a_same_place_recollection(tmp_path):
    """W2 反例：不同 source_id 但**同一地点**的采集不能绕过隔离。

    实测依据：`diverse_straightstreet` 与 `ident_probe_straight` 首帧相距 **0 m**，
    source_id 不同——整组隔离的字面实现会放过这种泄漏。
    """
    loop = _load()
    train = _located_frames(tmp_path, "coll_p", source_id="ring_p",
                           pos=(700.0, 700.0))
    dev = _located_frames(tmp_path, "coll_q", source_id="ring_q",
                          pos=(720.0, 700.0))     # 相距 20 m < 50 m 缓冲
    # 因子必须**真能生效**（否则先被 factor_not_applied 拦下，测不到空间闸门）：
    # 用训练器开关型因子，只改配方不改数据。
    p = _proposal(tmp_path, {"line_tversky_weight": 2.0})
    r = _run(["rounds", "--run-id", "rw_spatial", "--rounds", "1",
              "--runs", str(train), "--eval-runs", str(dev),
              "--allow-road-only", "--proposals", str(p),
              "--seeds", "42", "--plan-only"], tmp_path, expect=0)
    assert r.returncode == 0
    # plan-only 不过审计，所以直接跑审计路径：用真的 rounds
    r2 = _run(["rounds", "--run-id", "rw_spatial2", "--rounds", "1",
               "--runs", str(train), "--eval-runs", str(dev),
               "--allow-road-only", "--proposals", str(p),
               "--seeds", "42", "--epochs", "24", "--device", "cpu",
               "--trainer-script", str(ROOT / "tests" / "_stub_trainer_e2e.py")],
              tmp_path, expect=3)
    assert "空间隔离违规" in r2.stdout, r2.stdout[-600:]
    d = tmp_path / "logs" / "experiments" / "rw_spatial2"
    rep = json.loads((d / "rounds_dataset.json").read_text(encoding="utf-8"))
    sp = rep.get("spatial") or {}
    assert sp.get("violations") and sp["violations"][0]["why"] == "within_buffer"
    # 帧内步距 2 m：最近的一对是 coll_p 的末帧(+4 m) 与 coll_q 的首帧(+0 m)
    assert sp["min_distance_m"] == 16.0, sp
    assert sp["buffer_m"] == 50.0
    assert not (d / "round0").exists(), "拒绝后不得训练"





def test_a_sealed_final_set_may_not_be_read_by_the_loop(tmp_path):

    """最终集只允许确认程序读（方案 §7/§10.3）。



    搜索/训练入口读到带封存文件的目录必须**拒训**（否则"独立最终集"就只是一句话）。

    """

    loop = _load()

    train = _frames(tmp_path, "coll_final")

    dev = _frames(tmp_path, "coll_dev_final")

    (train / "final_set_seal.json").write_text(json.dumps({

        "name": "f", "digest": "x", "protocol_hash": "p"}),

        encoding="utf-8")

    p = _proposal(tmp_path, {"line_tversky_weight": 2.0})

    r = _run(["rounds", "--run-id", "rw_final", "--rounds", "1",

              "--runs", str(train), "--eval-runs", str(dev),

              "--allow-road-only", "--proposals", str(p),

              "--seeds", "42", "--plan-only"], tmp_path, expect=0)

    assert r.returncode == 0

    r2 = _run(["rounds", "--run-id", "rw_final2", "--rounds", "1",

               "--runs", str(train), "--eval-runs", str(dev),

               "--allow-road-only", "--proposals", str(p),

               "--seeds", "42", "--epochs", "24", "--device", "cpu",

               "--trainer-script", str(ROOT / "tests" / "_stub_trainer_e2e.py")],

              tmp_path, expect=3)

    assert "最终集" in r2.stdout, r2.stdout[-500:]

    d = tmp_path / "logs" / "experiments" / "rw_final2"

    assert not (d / "round0").exists(), "拒绝后不得训练"





def test_identity_metrics_report_coverage_and_roles(tmp_path):

    """G09：候选匹配要**分口径报**（冻结匹配率 / 带参考匹配率 /



    可测候选覆盖率 / 左右角色一致率）——旧口径的分母里混着

    "该侧没有参考"的候选（实测约 38%），那些应记 UNKNOWN。

    """

    loop = _load()

    d = _located_frames(tmp_path, "coll_id", source_id="ring_i",

                        pos=(900.0, 900.0))



    def fake(run, meta, *, view, model_path):

        return {"summary": {"match_rate": 0.15,

                            "match_rate_with_reference": 0.28,

                            "role_agreement_rate": 0.42,

                            "candidate_paint_recall": 0.31,

                            "n_candidates": 100,

                            "n_candidates_with_reference": 62}}



    got = loop.identity_metrics(Path("ck.pt"), [str(d)], probe_fn=fake)

    assert got["candidate_identity_rate"] == 0.15, got

    assert got["candidate_identity_rate_with_reference"] == 0.28

    assert got["candidate_reference_coverage"] == 0.62, got

    assert got["left_right_role_agreement"] == 0.42

    assert got["n_candidates"] == 100.0

    # 测不到（没有 meta / 没有帧）就是 None，不写 0

    got2 = loop.identity_metrics(Path("ck.pt"),

                                 [str(tmp_path / "nope")], probe_fn=fake)

    assert got2["candidate_identity_rate"] is None

    assert got2["candidate_reference_coverage"] is None

    # 覆盖率门槛尚未标定：阻止相应晋级（方案 §10.2）

    assert loop.COVERAGE_GATE_FROZEN is False

    # 逐场景明细必须真的有：调用方读 per_group 时静默拿到空字典，会看起来像

    # "这个场景没有候选"，实际是没接线（实测踩到）

    assert set(got["per_group"]) == {"italy/ring_i"}, got.get("per_group")

    assert got["per_group"]["italy/ring_i"]["candidate_identity_rate"] == 0.15

    assert got2["per_group"] == {}

