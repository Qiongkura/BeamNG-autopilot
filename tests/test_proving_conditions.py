# -*- coding: utf-8 -*-
"""试车场静止条件实验 runner：格名/门限/环境读回（fail-closed）。"""
import importlib.util
from pathlib import Path


def _mod():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "mpc_test", str(root / "scripts" / "m5_proving_conditions.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _report(ticks=12, true_n=12, cert=1.0):
    return {"summary": {
        "ticks": ticks, "true_pairs": true_n, "phantom_pairs": 0,
        "ambiguous_pairs": 0, "modes": {"pair": ticks},
        "cert_line_recall": {"both_hit_rate": cert},
    }}


def test_cell_names_and_grid_size():
    m = _mod()
    cells = list(m._cells())
    assert len(cells) == 12
    assert m._cell_name("night", "rain") == "gridmap_night_rain"


def test_score_pass(tmp_path):
    m = _mod()
    log = tmp_path / "x.run.log"
    log.write_text("[fsd-drive] environment -> tod=12:00 weather=- "
                   "applied_tod=12:00:00\n", encoding="utf-8")
    row = m._score_cell(_report(), log)
    assert row["status"] == "PASS", row
    assert row["tod_after"] == "12:00:00"


def test_score_fail_on_true_rate(tmp_path):
    m = _mod()
    log = tmp_path / "x.run.log"
    log.write_text("environment -> tod=12:00 weather=rainy applied_tod=12:00:00\n",
                   encoding="utf-8")
    row = m._score_cell(_report(ticks=12, true_n=5, cert=1.0), log)
    assert row["status"] == "FAIL", row


def test_score_unknown_without_readback(tmp_path):
    m = _mod()
    log = tmp_path / "x.run.log"
    log.write_text("[fsd-drive] no environment line\n", encoding="utf-8")
    row = m._score_cell(_report(), log)
    assert row["status"] == "UNKNOWN", row
    assert "environment_readback_missing" in row["reasons"]


def test_score_unknown_on_too_few_ticks(tmp_path):
    m = _mod()
    log = tmp_path / "x.run.log"
    log.write_text("environment -> tod=12:00 applied_tod=12:00:00\n",
                   encoding="utf-8")
    row = m._score_cell(_report(ticks=3, true_n=3), log)
    assert row["status"] == "UNKNOWN", row


def test_runner_keeps_drive_log_for_env_readback():
    """runner 的 stdout 必须写 .probe.log；.run.log 是探针保存的 drive 输出。"""
    src = (Path(__file__).resolve().parents[1]
           / "scripts" / "m5_proving_conditions.py").read_text("utf-8")
    assert '".probe.log"' in src
    assert '".run.log").write_text' not in src


def test_run_handles_cell_timeout_and_continues(tmp_path, monkeypatch, capsys):
    """单格挂死→整树清理、写 .probe.log、rc=-9，并继续下一格。"""
    import json

    m = _mod()
    monkeypatch.setattr(m, "OUT_DIR", tmp_path)
    calls = []

    def fake_cell(cmd, *, timeout_s):
        calls.append((cmd, timeout_s))
        if len(calls) == 1:
            return -9, "partial stdout", "partial stderr", True
        return 0, "ok out", "", False

    monkeypatch.setattr(m, "_run_cell_process", fake_cell)
    assert m.run("clear", cell_timeout_s=12.5) == 0
    assert len(calls) == 4  # dawn/noon/dusk/night × clear 全部跑完，未被第 1 格卡死
    assert all(t == 12.5 for _, t in calls)

    probe_log = (tmp_path / "gridmap_dawn_clear.probe.log").read_text("utf-8")
    assert "partial stdout" in probe_log
    assert "timeout (12.5s)" in probe_log

    out_lines = capsys.readouterr().out
    rows = json.loads(out_lines[out_lines.index("[\n"):])
    assert rows[0]["cell"] == "gridmap_dawn_clear"
    assert rows[0]["rc"] == -9 and rows[0]["timeout"] is True
    assert rows[1]["rc"] == 0


def test_timeout_path_kills_whole_process_tree(monkeypatch):
    """超时必须杀进程树（taskkill /T），否则 drive/引擎残留占端口。"""
    import subprocess
    from types import SimpleNamespace

    m = _mod()
    killed = []

    class FakeProc:
        pid = 4242
        returncode = None

        def communicate(self, timeout=None):
            if timeout is not None and not killed:
                raise subprocess.TimeoutExpired(cmd="x", timeout=timeout)
            return "partial", "err"

        def kill(self):
            pass

    monkeypatch.setattr(m.subprocess, "Popen", lambda *a, **k: FakeProc())
    monkeypatch.setattr(m, "_game_pids", lambda: set())
    monkeypatch.setattr(m, "_kill_tree", lambda pid: killed.append(pid))
    monkeypatch.setattr(m, "_kill_new_game_pids", lambda before: None)
    rc, _out, _err, timed_out = m._run_cell_process(["x"], timeout_s=0.01)
    assert timed_out is True and rc == -9
    assert killed == [4242]




def test_kill_tree_decodes_native_taskkill_output(monkeypatch):
    """taskkill 的 GBK 输出必须 replace 解码，不能抛 UnicodeDecodeError。"""
    m = _mod()
    calls = []
    monkeypatch.setattr(m.os, "name", "nt")
    monkeypatch.setattr(m.subprocess, "run",
                        lambda *a, **k: calls.append((a, k)))
    m._kill_tree(1234)
    assert calls and calls[0][1].get("encoding") == "utf-8"
    assert calls[0][1].get("errors") == "replace"
