"""S5 接线证明：无效因子不得占训练预算，停止条件只数"有意义的无收益"。

方案 v2 §S5 / T14 的实测反例：
* road-only（line 类整通道屏蔽）配方里提线损失因子——`--line-weight` 照样
  会变成训练器旗标，但没有任何监督，历史上有 5 轮就这么白跑了；
* 停止条件原来只数"连续 N 轮 rejected/needs_evidence"，于是无效因子、缺标注、
  资格失败都成了"模型没有提升空间"的证据。

这里用合成帧 + 桩训练器证明接线：真实训练不参与（不启动游戏、不动 GPU）。
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

STUB = ROOT / "tests" / "_stub_trainer_e2e.py"


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
            f"rc={r.returncode}\nSTDOUT:\n{r.stdout[-1500:]}\n" \
            f"STDERR:\n{r.stderr[-800:]}"
    return r


def _frames(tmp_path, name: str) -> Path:
    """合成一组帧；内容按目录名偏移（不同组不得是字节复制，否则审计判重复）。"""
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


def _events(run_dir: Path) -> list:
    return [json.loads(ln) for ln in
            (run_dir / "events.jsonl").read_text(encoding="utf-8")
            .splitlines() if ln.strip()]


# --------------------------------------------------- 无效因子拒训（真实运行）

def test_a_line_loss_factor_is_refused_when_line_supervision_is_masked(tmp_path):
    """road-only + 线损失因子：拒训、记账、不留候选产物。

    反例依据：`--line-weight` 在 `factor_to_flags` 里**会**变成旗标，
    旧检查（"至少有一个键变成旗标"）因此放行；但 line 类被
    `--ignore-line-class` 整通道屏蔽，这一轮两臂的实际监督完全相同。
    """
    _load()
    a = _frames(tmp_path, "coll_a")
    dev = _frames(tmp_path, "coll_dev")
    p = _proposal(tmp_path, {"line_weight": 2.0})
    r = _run(["rounds", "--run-id", "s5_line", "--rounds", "1",
              "--runs", str(a), "--eval-runs", str(dev),
              "--allow-road-only", "--proposals", str(p), "--seeds", "42",
              "--epochs", "24", "--device", "cpu",
              "--trainer-script", str(STUB)], tmp_path, expect=3)
    assert "factor_not_applied" in r.stdout, r.stdout[-800:]
    assert "line_supervision=False" in r.stdout, \
        "拒训原因必须写明是监督模式屏蔽（不是训练器没实现）"
    d = tmp_path / "logs" / "experiments" / "s5_line"
    assert not (d / "round0").exists(), "拒绝训练后不得留下候选产物"
    assert not list(d.glob("decision_*.json")), "没有训练就没有判定"
    phases = [e["phase"] for e in _events(d)]
    # 日志已经在 training（候选逐轮训练那条先写了）：从 training 只能到
    # evaluating/failed/paused，所以拒训事件落在 paused（needs_review 被
    # 状态机拒绝时的既有回退），status 仍是 factor_not_applied。
    assert phases[-1] in ("paused", "needs_review"), phases
    assert any(e.get("status") == "factor_not_applied" for e in _events(d)), \
        "拒训原因必须留在事件流里"
    # 拒训必须发生在训练之前：不允许出现"训练完成"事件（白跑一轮的证据）
    assert not [e for e in _events(d)
                if e["phase"] == "training" and e.get("status") == "done"], \
        "被拒的轮次不得留下训练完成事件"


def test_plan_only_warns_about_the_inactive_factor_but_still_prints_it(tmp_path):
    """`--plan-only` 是预览：照常打印两臂命令，但要标出该因子不会生效。"""
    _load()
    a = _frames(tmp_path, "coll_a")
    dev = _frames(tmp_path, "coll_dev")
    p = _proposal(tmp_path, {"line_weight": 2.0})
    r = _run(["rounds", "--run-id", "s5_plan", "--rounds", "1",
              "--runs", str(a), "--eval-runs", str(dev),
              "--allow-road-only", "--proposals", str(p), "--seeds", "42",
              "--plan-only"], tmp_path, expect=0)
    assert "inactive" in r.stdout, r.stdout[-800:]
    assert "--line-weight" in r.stdout, "预览仍要打印真实命令"
    d = tmp_path / "logs" / "experiments" / "s5_plan"
    assert not (d / "round0").exists()


def test_a_non_line_factor_still_trains_under_road_only(tmp_path):
    """对照组：road-only 只屏蔽线损失键；数据组成因子照常训练（不是一刀切）。"""
    _load()
    a = _frames(tmp_path, "coll_a")
    b = _frames(tmp_path, "coll_b")
    dev = _frames(tmp_path, "coll_dev")
    p = _proposal(tmp_path, {"add_runs": [str(b)]})
    r = _run(["rounds", "--run-id", "s5_data", "--rounds", "1",
              "--runs", str(a), "--eval-runs", str(dev),
              "--allow-road-only", "--proposals", str(p), "--seeds", "42",
              "--epochs", "24", "--device", "cpu",
              "--trainer-script", str(STUB)], tmp_path, expect=0)
    d = tmp_path / "logs" / "experiments" / "s5_data"
    decs = sorted(d.glob("decision_*.json"))
    assert decs, r.stdout[-1200:]
    blob = json.loads(decs[0].read_text(encoding="utf-8"))
    assert blob["factor_activity"]["active"] is True, blob["factor_activity"]
    assert blob["round_outcome"] in (
        "promoted", "meaningful_no_gain", "not_a_verdict",
        "qualification_failure", "missing_labels"), blob["round_outcome"]
    # 因子真生效，所以这一轮**不得**被归成 invalid_factor
    assert blob["round_outcome"] != "invalid_factor"
    # 归因与判定必须自洽：meaningful_no_gain 只能是 rejected/needs_evidence
    if blob["round_outcome"] == "meaningful_no_gain":
        assert blob["decision"]["decision"] in ("rejected", "needs_evidence")
    # 看板要的入口字段必须真的落盘（缺一个，面板就只能显示"无数据"）
    for _k in ("git_commit", "git_dirty", "device", "data_counts",
               "scene_applicability", "factor_activity", "round_outcome"):
        assert _k in blob, f"判定文件缺 {_k}：{sorted(blob)}"
    assert blob["device"] == "cpu", blob["device"]
    assert set(blob["data_counts"]) == {"generated", "reviewed", "trained",
                                        "evaluated"}, blob["data_counts"]
    assert blob["data_counts"]["trained"] > 0, blob["data_counts"]


def test_the_propose_cli_exposes_the_road_only_switch(tmp_path):
    """propose 子命令必须能拿到监督模式：否则提议阶段仍会产出线损失因子。"""
    r = _run(["propose", "--help"], tmp_path, expect=0)
    assert "--allow-road-only" in r.stdout, r.stdout


def test_the_fp_fn_direction_knob_reaches_the_training_command(tmp_path):
    """S6 E2 前置：`line_tversky_alpha` 必须能变成训练器旗标。

    实测依据（T15）：只有 `beta/alpha` 比值决定 FP/FN 方向，而 alpha 一直不在
    提议白名单里 -> 方向实验提议不出来。训练器本身早就有 `--line-tversky-alpha`。
    """
    loop = _load()
    flags, skipped = loop.factor_to_flags({"line_tversky_alpha": 0.5})
    assert flags == ["--line-tversky-alpha", "0.5"], flags
    assert not skipped, skipped
    # 方向旋钮属于线通道：road-only 下必须判 inactive（与其它线损失键一致）
    from beamng_autopilot.experiments.proposer import (
        LINE_LOSS_KEYS, factor_activity)
    assert "line_tversky_alpha" in LINE_LOSS_KEYS
    assert factor_activity({"line_tversky_alpha": 0.5},
                           line_supervision=False)["active"] is False
    assert factor_activity({"line_tversky_alpha": 0.5},
                           line_supervision=True)["active"] is True
