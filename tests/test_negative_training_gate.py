"""E1 前置：训练用"困难负例"目录的资格门（方案 §S6/E1 + §3.5/T10）。

实测背景（`logs/experiments/e1_readiness_20260926.json`）：仓库里没有同时满足
"评价集之外 + 有身份/位姿 + 人工确认无线"的负例目录——
* `dirt_road_*` 无地图身份（审计直接拒收）；
* 引擎采集（it2/wc）的 line 类"游戏不提供"，全零是**缺失**而非确认；
* 评价包里的 verified 负例是开发帧（E1 明文禁止入训）。

所以这条门的职责是：把"全零标线"目录分成 confirmed / weak / rejected 三类，
让"用弱负例冒充已确认负例"不可能悄悄发生。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beamng_autopilot.experiments.negative_scenes import (  # noqa: E402
    negative_training_eligibility,
)


def test_a_verified_all_zero_dir_is_a_confirmed_negative() -> None:
    out = negative_training_eligibility(
        [{"dir": "pack/dirt", "n_frames": 25, "n_line_frames": 0,
          "rank": "verified"}], research=False)
    assert [c["dir"] for c in out["confirmed"]] == ["pack/dirt"]
    assert not out["weak"] and not out["rejected"]
    assert "confirmed=1" in out["note"]


def test_a_non_verified_all_zero_dir_is_refused_in_a_promotion_run() -> None:
    """全零 + 非 verified：可晋级运行里必须拒训（不是"确认无线"）。"""
    out = negative_training_eligibility(
        [{"dir": "logs/m5_seg/collect_wc/front_main", "n_frames": 21,
          "n_line_frames": 0, "rank": "engine"}], research=False)
    assert [r["dir"] for r in out["rejected"]] == ["logs/m5_seg/collect_wc/front_main"]
    assert not out["confirmed"]
    assert "T10" in out["rejected"][0]["why"]


def test_the_same_dir_is_a_weak_negative_when_explicitly_research() -> None:
    out = negative_training_eligibility(
        [{"dir": "logs/m5_seg/collect_wc/front_main", "n_frames": 21,
          "n_line_frames": 0, "rank": "engine"}], research=True)
    assert not out["rejected"]
    assert [w["dir"] for w in out["weak"]] == ["logs/m5_seg/collect_wc/front_main"]
    assert "not a confirmed one" in out["weak"][0]["why"]


def test_a_mixed_dir_is_ordinary_training_data() -> None:
    """有正有负的目录是普通训练数据，不受本门约束（否则会把正常训练挡死）。"""
    out = negative_training_eligibility(
        [{"dir": "logs/m5_seg/line_truth_agent_full_20260925/town/front_main",
          "n_frames": 20, "n_line_frames": 20, "rank": "agent"},
         {"dir": "logs/m5_seg/collect_it2/front_main", "n_frames": 30,
          "n_line_frames": 12, "rank": "engine"}], research=False)
    assert not out["confirmed"] and not out["weak"] and not out["rejected"]


def test_the_gate_is_recorded_in_the_rounds_audit(tmp_path) -> None:
    """接线证明：真实 `rounds` 审计把弱负例逐条写进 rounds_dataset.json。"""
    import importlib.util
    import json
    import os
    import subprocess

    import numpy as np

    root = Path(__file__).resolve().parents[1]
    # 一个"有身份 + 引擎标签 + 全零标线"的训练目录（= 弱负例）
    neg = tmp_path / "runs" / "coll_neg" / "front_main"
    neg.mkdir(parents=True)
    for i in range(3):
        np.savez(neg / f"frame_{i:05d}.npz",
                 colour=np.full((30, 40, 3), 40 + i, np.uint8),
                 label=np.zeros((30, 40), np.uint8))
    (tmp_path / "runs" / "coll_neg" / "meta.json").write_text(json.dumps({
        "map_name": "italy", "source_id": "ring_neg",
        "map_name_source": "session.get_current().level",
        "frames": [{"i": i, "view": "front_main", "exposure": i,
                    "path": f"front_main/frame_{i:05d}.npz"}
                   for i in range(3)]}), encoding="utf-8")
    # 一个正常训练目录（有标线真值）与一个开发目录
    pos = tmp_path / "runs" / "coll_pos" / "front_main"
    pos.mkdir(parents=True)
    for i in range(3):
        lab = np.zeros((30, 40), np.uint8)
        lab[6:26, :] = 1
        lab[15, :6] = 2
        np.savez(pos / f"frame_{i:05d}.npz",
                 colour=np.full((30, 40, 3), 10 + i * 5, np.uint8), label=lab)
    (tmp_path / "runs" / "coll_pos" / "meta.json").write_text(json.dumps({
        "map_name": "italy", "source_id": "ring_pos",
        "map_name_source": "session.get_current().level",
        "frames": [{"i": i, "view": "front_main", "exposure": i,
                    "path": f"front_main/frame_{i:05d}.npz"}
                   for i in range(3)]}), encoding="utf-8")
    env = dict(os.environ)
    env["BEAMNG_LOGS_DIR"] = str(tmp_path / "logs")
    # 真跑一轮（桩训练器，不占 GPU）：`_rounds_audit` 才会被调用并落盘
    dev = tmp_path / "runs" / "coll_dev" / "front_main"
    dev.mkdir(parents=True)
    for i in range(2):
        lab = np.zeros((30, 40), np.uint8)
        lab[6:26, :] = 1
        np.savez(dev / f"frame_{i:05d}.npz",
                 colour=np.full((30, 40, 3), 200 + i, np.uint8), label=lab)
    (tmp_path / "runs" / "coll_dev" / "meta.json").write_text(json.dumps({
        "map_name": "italy", "source_id": "ring_dev",
        "map_name_source": "session.get_current().level",
        "frames": [{"i": i, "view": "front_main", "exposure": i,
                    "path": f"front_main/frame_{i:05d}.npz"}
                   for i in range(2)]}), encoding="utf-8")
    prop = tmp_path / "props.json"
    prop.write_text(json.dumps({"proposals": [
        {"candidate_id": "cand", "factor": {"lr": 0.0005}}]}),
        encoding="utf-8")
    r = subprocess.run(
        [sys.executable, str(root / "scripts" / "m5_seg_autoloop.py"), "rounds",
         "--run-id", "neg_gate", "--rounds", "1",
         "--runs", str(pos), str(neg), "--eval-runs", str(dev),
         "--allow-road-only", "--seeds", "42", "--epochs", "24",
         "--device", "cpu", "--proposals", str(prop),
         "--trainer-script", str(root / "tests" / "_stub_trainer_e2e.py")],
        capture_output=True, text=True, env=env, timeout=900)
    assert r.returncode == 0, r.stdout[-900:] + r.stderr[-400:]
    rep = json.loads((tmp_path / "logs" / "experiments" / "neg_gate"
                      / "rounds_dataset.json").read_text(encoding="utf-8"))
    gate = rep["negative_training"]
    assert [w["dir"].replace("\\", "/").endswith("coll_neg/front_main")
            for w in gate["weak"]] == [True], gate
    assert gate["note"].startswith("confirmed=0 weak=1"), gate["note"]
    assert "弱负例" in r.stdout, r.stdout[-500:]
