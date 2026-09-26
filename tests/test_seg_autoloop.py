"""阶段 D 离线循环的验收：审计门、单因子提议、成对淘汰、重放、dry-run。

方案对阶段 D 的完成条件是"连续至少 3 轮可无人值守运行；至少包含一次净负收益
自动淘汰；相同输入重放得到相同决策与淘汰理由"。这里用合成输入把这些条件测到，
真实数据上的那次淘汰见 docs/T14_PROGRESS_20260924.md。
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


def _load():
    spec = importlib.util.spec_from_file_location(
        "m5_seg_autoloop", ROOT / "scripts" / "m5_seg_autoloop.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["m5_seg_autoloop"] = mod
    spec.loader.exec_module(mod)
    return mod


def _run(loop, argv, tmp_path, *, expect=None):
    """把产物重定向到 tmp_path（不改真实 logs/）。"""
    env = dict(os.environ)
    env["BEAMNG_LOGS_DIR"] = str(tmp_path / "logs")
    cmd = [sys.executable, str(ROOT / "scripts" / "m5_seg_autoloop.py")]
    cmd += [a.replace("{tmp}", str(tmp_path)) for a in argv]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env,
                       timeout=600)
    if expect is not None:
        assert r.returncode == expect, \
            f"rc={r.returncode}\nSTDOUT:\n{r.stdout[-1500:]}\n{r.stderr[-800:]}"
    return r


def _frames(tmp_path, name: str, *, line_px: int) -> Path:
    d = tmp_path / "runs" / name / "front_main"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        colour = np.full((30, 40, 3), 50 + i * 5, np.uint8)
        label = np.zeros((30, 40), np.uint8)
        label[6:26, :] = 1                    # 20x40=800 px > 准入门槛 200
        if line_px:
            label[15, :line_px] = 2
        np.savez(d / f"frame_{i:05d}.npz", colour=colour, label=label)
    (tmp_path / "runs" / name / "meta.json").write_text(json.dumps({
        "map_name": "italy", "source_id": f"ring_{name}",
        "frames": [{"i": i, "view": "front_main", "exposure": i,
                    "t_wall": 1.0 + i, "path": f"front_main/frame_{i:05d}.npz"}
                   for i in range(3)]}), encoding="utf-8")
    return d


class TestAuditGate:
    def test_paint_truth_missing_goes_to_review_not_training(self, tmp_path):
        _load()
        _frames(tmp_path, "coll_a", line_px=6)
        r = _run(None, ["audit", "--run-id", "loop1", "--runs",
                        "{tmp}/runs/coll_a/front_main"], tmp_path, expect=3)
        assert "needs_review" in r.stdout
        log = (tmp_path / "logs" / "experiments" / "loop1" / "events.jsonl")
        phases = [json.loads(ln)["phase"] for ln in
                  log.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert phases == ["queued", "auditing", "needs_review"], \
            "必须走合法迁移，不能从 queued 直接跳 needs_review"

    def test_road_only_mode_can_reach_training(self, tmp_path):
        _load()
        _frames(tmp_path, "coll_a", line_px=6)
        _run(None, ["audit", "--run-id", "loop2", "--runs",
                    "{tmp}/runs/coll_a/front_main", "--allow-road-only"],
             tmp_path, expect=0)

    def test_no_trainable_data_stops_the_loop(self, tmp_path):
        _load()
        d = _frames(tmp_path, "coll_a", line_px=0)
        # 把标签全清成背景：没有任何通道有真值
        for f in d.glob("frame_*.npz"):
            z = np.load(f)
            np.savez(f, colour=z["colour"], label=np.zeros_like(z["label"]))
        r = _run(None, ["audit", "--run-id", "loop3", "--runs",
                        "{tmp}/runs/coll_a/front_main"], tmp_path, expect=2)
        assert "停止" in r.stdout


class TestProposeAndEliminate:
    def _eval_dir(self, tmp_path, *, iou: float) -> Path:
        d = tmp_path / "eval"
        d.mkdir(parents=True, exist_ok=True)
        (d / "06_eval_matrix.json").write_text(json.dumps({"frozen": {
            "armA_seed42/best.pt": {
                "line_iou": iou, "line_precision": 0.4, "line_recall": 0.8,
                "offroad_false_line_px": 9000, "pred_line_px": 30000,
                "gt_line_px": 20000, "missed_true_line_px": 1500,
                "model": "armA_seed42/best.pt"}}}), encoding="utf-8")
        (d / "ident_t13_testA2_armA42.json").write_text(json.dumps({
            "summary": {"candidates_total": 100, "candidates_off_road": 90,
                        "role_agreement_rate": 0.5, "match_rate": 0.2,
                        "candidate_paint_recall_p50": 0.001,
                        "run": "ident_t13_testA2_armA42.json"}}),
            encoding="utf-8")
        return d

    def test_one_factor_proposals_and_a_review_queue(self, tmp_path):
        _load()
        d = self._eval_dir(tmp_path, iou=0.4)
        # 场景配比族要的是"可用的未入训数据组"：给一个真实存在的目录，
        # 提议才会是可执行的 add_runs；不给则要报告"需要新数据"而不是硬提议
        _frames(tmp_path, "coll_new", line_px=6)
        r = _run(None, ["propose", "--run-id", "loop4", "--eval-dir",
                        str(d), "--n-train-frames", "173",
                        "--available-runs",
                        "{tmp}/runs/coll_new/front_main"],
                 tmp_path, expect=0)
        blob = json.loads((tmp_path / "logs" / "experiments" / "loop4"
                           / "proposals.json").read_text(encoding="utf-8"))
        assert blob["proposals"], "应当给出提议"
        for p in blob["proposals"]:
            assert len(p["factor"]) == 1, "一次实验只改一个因子"
        assert list(blob["proposals"][0]["factor"]) == ["add_runs"],             "场景配比族的因子必须是训练器能应用的 add_runs"
        assert blob["needs_review"], "缺可信漆线真值的错误必须进复核队列"
        assert "candidates_not_on_paint" in json.dumps(blob["needs_review"],
                                                       ensure_ascii=False)

    def _pairings(self, tmp_path, *, gain: float) -> Path:
        """主指标（line_recall）成对 + 辅助指标（IoU）各一份。

        方案把像素 IoU 定为**辅助**门槛，主指标是身份/漏线/精度——所以
        "IoU 涨了"不足以晋级，这个测试用 line_recall 才代表"可信改善"。
        """
        p = tmp_path / "pairings.json"
        p.write_text(json.dumps({
            "line_recall": {"champion": [0.80, 0.81, 0.79],
                            "candidate": [0.80 + gain, 0.81 + gain,
                                          0.79 + gain]},
            "line_iou": {"champion": [0.40, 0.41, 0.39],
                         "candidate": [0.40, 0.41, 0.39]}}),
            encoding="utf-8")
        return p

    def test_the_auxiliary_iou_alone_cannot_promote(self, tmp_path):
        """IoU 是辅助指标：只涨它不改主指标时，判定不能晋级。

        规则更新（2026-09-25，W5/G09）：这里**一个任务主指标都没测**，
        所以判定是 `needs_evidence`（"证据缺失"），而不是旧语义的 `rejected`
        （"测了但没改善"）——方案 §10.3 的判定顺序把这两种分开。
        """
        _load()
        p = tmp_path / "pairings.json"
        p.write_text(json.dumps({"line_iou": {
            "champion": [0.40, 0.41, 0.39],
            "candidate": [0.50, 0.51, 0.49]}}), encoding="utf-8")
        hard = tmp_path / "hard.json"
        hard.write_text(json.dumps({"line_recall": 0.8, "line_precision": 0.5,
                                    "candidate_identity_rate": 0.7,
                                    "candidate_reference_coverage": 0.92,
                                    "left_right_role_agreement": 0.75,
                                    "offroad_false_ratio": 0.01,
                                    "inference_ms_p95": 20.0}), encoding="utf-8")
        r = _run(None, ["evaluate", "--run-id", "loop_iou", "--candidate-id",
                        "iouonly", "--pairings", str(p), "--hard-gate",
                        str(hard)], tmp_path, expect=3)
        assert "no primary (task) metric was measured" in r.stdout, r.stdout

    def test_a_net_negative_round_is_eliminated_with_reasons(self, tmp_path):
        _load()
        p = self._pairings(tmp_path, gain=-0.05)
        hard = tmp_path / "hard.json"
        hard.write_text(json.dumps({"line_recall": 0.8, "line_precision": 0.5,
                                    "candidate_identity_rate": 0.7,
                                    "candidate_reference_coverage": 0.92,
                                    "left_right_role_agreement": 0.75,
                                    "offroad_false_ratio": 0.01,
                                    "inference_ms_p95": 20.0}), encoding="utf-8")
        r = _run(None, ["evaluate", "--run-id", "loop5", "--candidate-id",
                        "worse", "--pairings", str(p), "--hard-gate",
                        str(hard)], tmp_path, expect=1)
        assert "rejected" in r.stdout and "champion_better" in r.stdout
        dec = json.loads(Path(tmp_path / "logs" / "experiments" / "loop5"
                              / "decision_worse.json").read_text(
                                  encoding="utf-8"))
        assert dec["decision"]["decision"] == "rejected"
        assert any("champion_better" in x for x in dec["decision"]["reasons"])

    def test_a_credible_gain_is_only_a_shadow_candidate(self, tmp_path):
        _load()
        p = self._pairings(tmp_path, gain=0.05)
        hard = tmp_path / "hard.json"
        hard.write_text(json.dumps({"line_recall": 0.8, "line_precision": 0.5,
                                    "candidate_identity_rate": 0.7,
                                    "candidate_reference_coverage": 0.92,
                                    "left_right_role_agreement": 0.75,
                                    "offroad_false_ratio": 0.01,
                                    "inference_ms_p95": 20.0}), encoding="utf-8")
        r = _run(None, ["evaluate", "--run-id", "loop6", "--candidate-id",
                        "better", "--pairings", str(p), "--hard-gate",
                        str(hard)], tmp_path, expect=0)
        assert "shadow_candidate" in r.stdout
        assert "NOT replaced" in r.stdout or "不覆盖" in r.stdout or \
            "shadow only" in r.stdout

    def test_the_decision_replays_byte_for_byte(self, tmp_path):
        _load()
        p = self._pairings(tmp_path, gain=0.05)
        hard = tmp_path / "hard.json"
        hard.write_text(json.dumps({"line_recall": 0.8, "line_precision": 0.5,
                                    "candidate_identity_rate": 0.7,
                                    "candidate_reference_coverage": 0.92,
                                    "left_right_role_agreement": 0.75,
                                    "offroad_false_ratio": 0.01,
                                    "inference_ms_p95": 20.0}), encoding="utf-8")
        _run(None, ["evaluate", "--run-id", "loop7", "--candidate-id", "c1",
                    "--pairings", str(p), "--hard-gate", str(hard)],
             tmp_path, expect=0)
        r = _run(None, ["replay", "--run-id", "loop7"], tmp_path, expect=0)
        assert "不同 0" in r.stdout

    def test_evaluate_refuses_a_run_stuck_in_review(self, tmp_path):
        """needs_review 的 run 不能再评估：先补真值重新审计。"""
        _load()
        _frames(tmp_path, "coll_a", line_px=6)
        _run(None, ["audit", "--run-id", "loop8", "--runs",
                    "{tmp}/runs/coll_a/front_main"], tmp_path, expect=3)
        p = self._pairings(tmp_path, gain=0.05)
        r = _run(None, ["evaluate", "--run-id", "loop8", "--candidate-id", "c1",
                        "--pairings", str(p)], tmp_path, expect=2)
        assert "状态机拒绝" in r.stdout
        dec = list((tmp_path / "logs" / "experiments" / "loop8")
                   .glob("decision_*.json"))
        assert not dec, "被拒绝的评估不得留下判定文件（否则事件与产物分叉）"


class TestDryRun:
    def test_dry_run_prints_the_plan_and_executes_nothing(self, tmp_path):
        _load()
        r = _run(None, ["run", "--run-id", "loop9", "--candidate-id", "cand",
                        "--dataset-id", "ds1", "--train-args",
                        "--epochs 3 --batch 4"], tmp_path, expect=0)
        assert "dry-run：未执行任何训练/采集命令" in r.stdout
        assert "m5_train_seg.py" in r.stdout, "计划里要有训练命令"
        # 没有训练产物、没有指标文件：dry-run 不执行
        assert not (tmp_path / "logs" / "experiments" / "loop9"
                    / "candidates").exists()

    def test_the_plan_marks_tech_collection_as_requiring_authorisation(
            self, tmp_path):
        _load()
        cfg = tmp_path / "cfg.json"
        cfg.write_text(json.dumps({"collect": "tech", "dry_run": True,
                                   "experiments_root": str(
                                       tmp_path / "logs" / "experiments")}),
                       encoding="utf-8")
        r = _run(None, ["run", "--run-id", "loop10", "--candidate-id", "cand",
                        "--config", str(cfg)], tmp_path, expect=0)
        assert "需授权" in r.stdout


class TestNonInferiority:
    """次要指标只要**非劣**；主指标才要求可信改善（方案 §5）。"""

    def _hard(self, tmp_path):
        """硬门输入夹具。

        协议 v4（2026-09-26）起 `candidate_reference_coverage` 与
        `left_right_role_agreement` 是**已标定的硬门输入**（门槛 0.80 / 0.70，
        见 docs/CANDIDATE_GATE_CALIBRATION_20260926.md），所以夹具必须给出实测值：
        缺这两项会走缺测通道（needs_evidence），测试就不再是在测原意了。
        """
        p = tmp_path / "hard.json"
        p.write_text(json.dumps({"line_recall": 0.8, "line_precision": 0.5,
                                 "candidate_identity_rate": 0.7,
                                 "candidate_reference_coverage": 0.92,
                                 "left_right_role_agreement": 0.75,
                                 "offroad_false_ratio": 0.01,
                                 "inference_ms_p95": 20.0}), encoding="utf-8")
        return p

    def test_a_zero_delta_secondary_metric_is_non_inferior(self, tmp_path):
        _load()
        pr = tmp_path / "pairings.json"
        pr.write_text(json.dumps({
            "line_recall": {"champion": [0.80, 0.81, 0.79],
                            "candidate": [0.86, 0.87, 0.85]},
            "offroad_false_line_px": {"champion": [10000, 11000, 9000],
                                      "candidate": [10000, 11000, 9000],
                                      "lower_is_better": True}}),
            encoding="utf-8")
        r = _run(None, ["evaluate", "--run-id", "ni1", "--candidate-id", "c",
                        "--pairings", str(pr), "--hard-gate",
                        str(self._hard(tmp_path))], tmp_path, expect=0)
        assert "shadow_candidate" in r.stdout, \
            "次要指标 0 差异是非劣，不该挡晋级"

    def test_a_materially_worse_secondary_metric_blocks(self, tmp_path):
        _load()
        pr = tmp_path / "pairings.json"
        pr.write_text(json.dumps({
            "line_recall": {"champion": [0.80, 0.81, 0.79],
                            "candidate": [0.86, 0.87, 0.85]},
            "offroad_false_line_px": {"champion": [10000, 11000, 9000],
                                      "candidate": [16000, 17000, 15000],
                                      "lower_is_better": True}}),
            encoding="utf-8")
        r = _run(None, ["evaluate", "--run-id", "ni2", "--candidate-id", "c",
                        "--pairings", str(pr), "--hard-gate",
                        str(self._hard(tmp_path))], tmp_path, expect=3)
        assert "worse than the champion" in r.stdout
        assert "needs_evidence" in r.stdout,             "主指标变差 + 另一项改善 = 证据不足，不能当晋级(0)也不能当淘汰(1)"
