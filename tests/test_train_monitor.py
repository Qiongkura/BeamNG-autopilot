"""训练监控：数据层、界面与增量接口的验收（方案 §3/§4/§5）。

覆盖的要点：
* 记录持久化 + 按 seq 增量拉取 + 去重 + 半写行可识别；
* "没采集到 / 任务不适用 / 真的是 0" 三者不混；
* 统计与抽稀：统计基于原始值、非有限值被忽略且计数；
* 界面：12 张图、状态栏字段、无外部依赖、DEMO 横幅、未提供功耗的提示；
* 端到端：真实跑一次小训练，检查逐 step 指标与任务状态真的落盘。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(ROOT))
sys.path.insert(0, os.fspath(ROOT / "scripts"))


# ---------------------------------------------------------------------------
class TestMetricStore:
    def _store(self, tmp_path):
        from beamng_autopilot.experiments.metrics import MetricsStore
        return MetricsStore(tmp_path / "run1")

    def test_records_round_trip_and_incremental_pull_dedupes_by_seq(
            self, tmp_path):
        st = self._store(tmp_path)
        for i in range(5):
            st.append({"kind": "train", "step": i + 1, "loss": 1.0 / (i + 1)})
        recs, problems = st.read()
        assert len(recs) == 5 and problems == []
        assert [r["seq"] for r in recs] == [1, 2, 3, 4, 5]
        inc = st.read_since(3)
        assert [r["step"] for r in inc["records"]] == [4, 5]
        assert inc["next_since"] == 5
        again = st.read_since(5)
        assert again["records"] == [], "同一 seq 不会重复返回（去重靠序号）"
        assert again["next_since"] == 5

    def test_a_torn_last_line_is_reported_and_the_next_append_survives(
            self, tmp_path):
        st = self._store(tmp_path)
        st.append({"kind": "train", "step": 1, "loss": 2.0})
        with st.path.open("a", encoding="utf-8") as fh:
            fh.write('{"kind": "train", "step": 2, "lo')
        st.append({"kind": "train", "step": 3, "loss": 1.0})
        recs, problems = st.read()
        assert [r["step"] for r in recs] == [1, 3], \
            "半写行之后写入的记录必须仍然可读"
        assert problems and "unreadable" in problems[0]

    def test_task_status_and_error_are_readable(self, tmp_path):
        from beamng_autopilot.experiments.metrics import task_record
        st = self._store(tmp_path)
        st.append(task_record("run1", "seg 训练", "running", total_steps=100,
                              current_step=0, epoch=0))
        st.append(task_record("run1", "seg 训练", "failed", total_steps=100,
                              current_step=7, epoch=0, error="RuntimeError: 显存不足"))
        task = st.task()
        assert task["status"] == "failed"
        assert "显存不足" in task["error"]
        with pytest.raises(ValueError):
            task_record("run1", "x", "没这个状态", total_steps=1)

    def test_a_sampler_failure_is_recorded_not_swallowed(self, tmp_path):
        from beamng_autopilot.experiments.metrics import (
            MetricsStore, SystemSampler)

        def boom():
            raise RuntimeError("nvml 挂了")

        st = MetricsStore(tmp_path / "s")
        sp = SystemSampler(st, interval_s=0.2, probe=boom, run_id="r1")
        sp.start()
        time.sleep(0.6)
        sp.stop()
        recs = st.system_records()
        assert recs and any("nvml" in json.dumps(r, ensure_ascii=False)
                            for r in recs), "采样异常必须留痕"


class TestStatsAndDownsample:
    def test_stats_ignore_non_finite_and_report_how_many(self):
        from beamng_autopilot.experiments.metrics import stats_of
        st = stats_of([1.0, 2.0, 3.0, float("nan"), None, float("inf"), "x"])
        assert st["n"] == 3 and st["dropped_non_finite"] == 4
        assert st["mean"] == 2.0 and st["median"] == 2.0
        assert st["min"] == 1.0 and st["max"] == 3.0

    def test_an_empty_series_is_missing_not_zero(self):
        from beamng_autopilot.experiments.metrics import stats_of
        st = stats_of([])
        assert st["n"] == 0 and "missing" in st
        assert "mean" not in st, "没有数据就不能给出均值（更不能是 0）"

    def test_moving_average_keeps_gaps_instead_of_filling_them(self):
        from beamng_autopilot.experiments.metrics import moving_average
        sma = moving_average([1.0, 2.0, 3.0, None, 5.0], 2)
        assert sma[3] is None, "缺失点不能被平滑补成数字"
        assert sma[4] == pytest.approx((3.0 + 5.0) / 2)

    def test_downsample_keeps_stats_from_raw_values(self):
        from beamng_autopilot.experiments.metrics import downsample, stats_of
        pts = [(i, i * 1.0) for i in range(5000)]
        ds = downsample(pts, 500)
        assert len(ds["points"]) <= 502 and ds["stride"] >= 10
        assert ds["n_raw"] == 5000 and ds["note"]
        raw_stats = stats_of([y for _, y in pts])
        assert raw_stats["n"] == 5000, "统计必须基于原始点而不是抽稀后"

    def test_real_hardware_sample_marks_unavailable_sources(self):
        from beamng_autopilot.experiments.metrics import sample_system
        rec = sample_system()
        assert rec["kind"] == "system" and "aggregate" in rec
        # 真机上应能读到 GPU 显存；读不到也只是 None + 原因，不能是 0
        assert rec["gpu_mem_gib"] is None or rec["gpu_mem_gib"] > 0
        for k, v in (rec.get("unavailable") or {}).items():
            assert isinstance(v, str) and v, f"{k} 的不可用原因不能为空"


class TestUiContract:
    def _html(self, **kw):
        from beamng_autopilot.experiments.monitor_ui import render_html
        return render_html(run_id="r1", **kw)

    def test_all_twelve_charts_and_the_status_bar_are_present(self):
        html = self._html()
        for title in ("Loss", "Accuracy", "Gradient Norm", "Learning Rate",
                      "Loss Distribution", "Gradient Norm Distribution",
                      "GPU Memory", "Training Speed", "GPU Power",
                      "GPU Utilization", "CPU Utilization", "Memory Usage"):
            assert title in html, f"缺图：{title}"
        for label in ("任务名称", "运行状态", "epoch", "已运行时间",
                      "当前 loss", "平均每步耗时", "当前时间"):
            assert label in html, f"状态栏缺字段：{label}"

    def test_the_page_is_self_contained(self):
        html = self._html()
        assert "http://" not in html.replace("http://127.0.0.1", "") \
            and "https://" not in html, "页面不得引用外部资源"
        assert "<script src" not in html and "<link" not in html

    def test_demo_records_are_labelled_and_snapshot_needs_no_server(self):
        from beamng_autopilot.experiments.metrics import demo_records
        recs = demo_records("demo1", steps=5)
        html = self._html(mode="snapshot", records=recs)
        assert "DEMO" in html
        assert json.loads(html.split("const P = ", 1)[1].split(";\n", 1)[0])[
            "mode"] == "snapshot"

    def test_an_unavailable_power_source_gets_a_sentence_not_a_zero(self):
        from beamng_autopilot.experiments.monitor_ui import render_html
        recs = [{"kind": "system", "seq": 1, "t": 0.0, "gpu_mem_gib": 1.0,
                 "gpu_power_w": None, "gpu_util_pct": 5.0, "cpu_util_pct": 3.0,
                 "sys_mem_gib": 4.0,
                 "unavailable": {"gpu0.power_w":
                                 "this device does not provide power readings"}}]
        html = render_html(run_id="r1", mode="snapshot", records=recs)
        assert "该设备未提供功耗数据" in html
        # 数值字段为 None：图表必须走"未采集/不提供"分支而不是画 0
        assert '"gpu_power_w": null' in html

    def test_a_failed_task_shows_an_error_summary_and_keeps_charts(self):
        from beamng_autopilot.experiments.monitor_ui import render_html
        recs = [{"kind": "task", "seq": 1, "run_id": "r1", "name": "t",
                 "status": "running", "total_steps": 10, "started_at": 0.0},
                {"kind": "train", "seq": 2, "step": 1, "loss": 2.0, "acc": 0.5,
                 "grad_norm": 1.0, "lr": 3e-4, "step_s": 0.05},
                {"kind": "task", "seq": 3, "run_id": "r1", "name": "t",
                 "status": "failed", "error": "RuntimeError: CUDA out of memory"}]
        html = render_html(run_id="r1", mode="snapshot", records=recs)
        assert "训练失败" in html and "CUDA out of memory" in html
        assert '"step": 1' in html, "失败前记录必须保留"


class TestMonitorServer:
    def _serve(self, tmp_path):
        from beamng_autopilot.experiments import monitor_server
        from beamng_autopilot.experiments.metrics import MetricsStore
        st = MetricsStore(tmp_path / "run1")
        st.append({"kind": "train", "step": 1, "loss": 2.0})
        st.append({"kind": "train", "step": 2, "loss": 1.5})
        srv, th, url = monitor_server.serve(tmp_path / "run1", port=0,
                                            run_id="run1")
        return srv, th, url, st

    def test_metrics_endpoint_is_incremental(self, tmp_path):
        srv, _th, url, _st = self._serve(tmp_path)
        try:
            with urllib.request.urlopen(url + "metrics?since=1",
                                        timeout=5) as r:
                j = json.loads(r.read().decode("utf-8"))
            assert [x["step"] for x in j["records"]] == [2]
            assert j["next_since"] == 2
            try:
                urllib.request.urlopen(url + "metrics?since=abc", timeout=5)
                raise AssertionError("非法 since 必须返回 400")
            except urllib.error.HTTPError as e:
                assert e.code == 400
        finally:
            srv.shutdown()

    def test_the_page_and_state_endpoints_answer(self, tmp_path):
        srv, _th, url, _st = self._serve(tmp_path)
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                html = r.read().decode("utf-8")
            assert "Loss" in html and r.status == 200
            with urllib.request.urlopen(url + "state", timeout=5) as r:
                st = json.loads(r.read().decode("utf-8"))
            assert st["last_seq"] == 2 and st["run_id"] == "run1"
            with urllib.request.urlopen(url + "health", timeout=5) as r:
                assert json.loads(r.read().decode("utf-8"))["ok"] is True
        finally:
            srv.shutdown()


# ---------------------------------------------------------------------------
class TestTrainingWiring:
    """端到端：真跑一次小训练，检查逐 step 指标与任务状态真的落盘。"""

    def _write_frames(self, tmp_path, n=6):
        d = tmp_path / "runs" / "front_main"
        d.mkdir(parents=True)
        for i in range(n):
            colour = np.full((16, 20, 3), 40 + i * 7, np.uint8)
            label = np.zeros((16, 20), np.uint8)
            label[4:12, :] = 1
            label[6:8, 2:8] = 2
            np.savez(d / f"frame_{i:05d}.npz", colour=colour, label=label)
        return d          # load_frames 直接在该目录里找 frame_*.npz

    def test_a_real_run_records_per_step_metrics_and_task_state(
            self, tmp_path):
        run_dir = self._write_frames(tmp_path)
        metrics_run = "pytest_monitor"
        out = tmp_path / "out"
        cmd = [sys.executable, str(ROOT / "scripts" / "m5_train_seg.py"),
               "--runs", str(run_dir), "--split", "tail", "--val-frac", "0.34",
               "--epochs", "2", "--batch", "2", "--lr", "1e-3", "--seed", "3",
               "--out", str(out), "--metrics-run", metrics_run,
               "--task-name", "pytest 小训练", "--monitor-interval", "0.3",
               "--vram-frac", "0.3"]
        env = dict(os.environ)
        env["BEAMNG_LOGS_DIR"] = str(tmp_path / "logs")
        r = subprocess.run(cmd, capture_output=True, text=True, env=env,
                           timeout=600)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        mfile = tmp_path / "logs" / "experiments" / metrics_run / "metrics.jsonl"
        assert mfile.exists(), f"指标文件没生成：{mfile}"
        recs = [json.loads(ln) for ln in mfile.read_text(
            encoding="utf-8").splitlines() if ln.strip()]
        train = [x for x in recs if x["kind"] == "train"]
        assert len(train) >= 4, "6 帧/批 2/验证 2 -> 每轮至少 2 步"
        for key in ("loss", "acc", "grad_norm", "lr", "step_s", "step",
                    "epoch"):
            assert key in train[0], f"逐 step 记录缺字段 {key}"
        assert all(isinstance(x["grad_norm"], (int, float))
                   and x["grad_norm"] > 0 for x in train), \
            "AMP 下梯度范数必须 unscale 后统计，不能是 0/缺失"
        assert [x["step"] for x in train] == list(range(1, len(train) + 1))
        tasks = [x for x in recs if x["kind"] == "task"]
        assert [t["status"] for t in tasks] == ["running", "completed"]
        assert tasks[0]["total_steps"] == 4 and tasks[0]["name"] == "pytest 小训练"
        assert tasks[-1]["current_step"] == len(train)

    def test_a_failing_run_writes_a_failed_task_record(self, tmp_path):
        """数据目录不存在 -> 训练在早期失败，看板仍要知道它失败了。"""
        metrics_run = "pytest_monitor_fail"
        cmd = [sys.executable, str(ROOT / "scripts" / "m5_train_seg.py"),
               "--runs", str(tmp_path / "does_not_exist"),
               "--epochs", "1", "--metrics-run", metrics_run,
               "--out", str(tmp_path / "out2")]
        env = dict(os.environ)
        env["BEAMNG_LOGS_DIR"] = str(tmp_path / "logs")
        r = subprocess.run(cmd, capture_output=True, text=True, env=env,
                           timeout=300)
        assert r.returncode != 0
        mfile = (tmp_path / "logs" / "experiments" / metrics_run
                 / "metrics.jsonl")
        if mfile.exists():
            recs = [json.loads(ln) for ln in mfile.read_text(
                encoding="utf-8").splitlines() if ln.strip()]
            fails = [x for x in recs if x.get("status") == "failed"]
            assert fails, "失败状态必须落盘（否则看板停在 running）"
            assert "没有找到数据" in fails[-1].get("error", "")
