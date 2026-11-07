# -*- coding: utf-8 -*-
"""试车场采集 runner：站点/条件格/命令/环境证据/attempt 门禁（预注册 §2-§3，
§6.2 门 1/2/4 闭环：环境读回核对、帧清单核对、按记录 PID 清理引擎、
attempt 隔离与可恢复）。"""
import importlib.util
import json
from pathlib import Path

import pytest


def _mod():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "mpc_capture", str(root / "scripts" / "m5_proving_capture.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_capture_cli_requires_switch_and_passes_reclaim(monkeypatch):
    """CLI 直通编排：--capture 需要 site+cell，--reclaim 透传。"""
    import sys
    m = _mod()
    seen = {}
    monkeypatch.setattr(m, "capture", lambda *a, **k: seen.update(a=a, k=k) or 0)
    monkeypatch.setattr(sys, "argv", ["capture", "--capture", "--site", "S1",
                                      "--cell", "noon_rain", "--reclaim"])
    assert m.main() == 0
    assert seen["a"] == ("S1", "noon_rain")
    assert seen["k"]["reclaim"] is True
    monkeypatch.setattr(sys, "argv", ["capture", "--capture", "--site", "S1"])
    with pytest.raises(SystemExit):
        m.main()


def test_capture_killswitch_blocks_everything(monkeypatch, tmp_path):
    m = _mod()
    monkeypatch.setattr(m, "CAP_DIR", tmp_path)
    monkeypatch.setattr(m, "_CAPTURE_ENABLED", False)
    monkeypatch.setattr(m.subprocess, "run", lambda *a, **k: pytest.fail(
        "kill-switch on: must not start or kill a process"))
    with pytest.raises(RuntimeError, match="BLOCKED"):
        m.capture("S1", "noon_clear")
    with pytest.raises(RuntimeError, match="BLOCKED"):
        m.run_site("S1")
    with pytest.raises(RuntimeError, match="BLOCKED"):
        m.run_all()
    with pytest.raises(RuntimeError, match="BLOCKED"):
        m._kill_games()
    assert list(tmp_path.iterdir()) == []


def test_capture_result_requires_all_independent_evidence():
    m = _mod()
    good = dict(returncode=0, timed_out=False, frame_count=20,
                environment={"tod_matches": True, "weather_matches": True},
                ownership_verified=True)
    result = m.assess_capture_result(**good)
    assert result == {"status": "EVIDENCE_READY", "reasons": [],
                      "training_qualified": False}
    for name, bad in (("minimum_frames", 0), ("minimum_frames", True),
                      ("returncode", 1), ("returncode", True),
                      ("timed_out", None), ("frame_count", 19),
                      ("frame_count", None), ("ownership_verified", None),
                      ("environment", {"tod_matches": True}),
                      ("environment", {"tod_matches": False,
                                       "weather_matches": True})):
        args = dict(good)
        args[name] = bad
        result = m.assess_capture_result(**args)
        assert result["status"] == "INCOMPLETE" and result["reasons"], (name, bad)
        assert result["training_qualified"] is False


def test_attempt_directory_is_unique_and_preserves_previous(tmp_path):
    import pytest
    m = _mod()
    previous = m.create_attempt_dir(tmp_path)
    (previous / "evidence.txt").write_text("keep", encoding="utf-8")
    second = m.create_attempt_dir(tmp_path)
    assert previous != second and previous.parent == second.parent == tmp_path
    assert (previous / "evidence.txt").read_text(encoding="utf-8") == "keep"
    assert list(second.iterdir()) == []
    with pytest.raises(ValueError):
        m.create_attempt_dir(tmp_path / "missing")


def test_attempt_manifest_exclusive_create_preserves_existing(tmp_path):
    import json
    import pytest
    m = _mod()
    path = m.write_attempt_manifest(tmp_path, {"status": "INCOMPLETE"})
    before = path.read_bytes()
    assert json.loads(before)["status"] == "INCOMPLETE"
    with pytest.raises(FileExistsError):
        m.write_attempt_manifest(tmp_path, {"status": "ok"})
    assert path.read_bytes() == before
    with pytest.raises(ValueError):
        m.write_attempt_manifest(tmp_path, {"bad": float("nan")})
    assert path.read_bytes() == before


def test_frame_manifest_checks_hash_membership_and_duplicates(tmp_path):
    import hashlib
    m = _mod()
    f = tmp_path / "frame_001.npz"
    f.write_bytes(b"original")
    entry = {"file": f.name, "sha256": hashlib.sha256(b"original").hexdigest()}
    assert m.verify_frame_manifest(tmp_path, [entry])["valid"] is True
    assert not m.verify_frame_manifest(tmp_path, [entry, entry])["valid"]
    f.write_bytes(b"changed")
    assert "digest_mismatch:frame_001.npz" in m.verify_frame_manifest(tmp_path, [entry])["errors"]
    f.write_bytes(b"original")
    (tmp_path / "old.npz").write_bytes(b"old")
    assert "unlisted:old.npz" in m.verify_frame_manifest(tmp_path, [entry])["errors"]
    assert not m.verify_frame_manifest(tmp_path, [{"file": "../outside.npz",
                                                    "sha256": "0" * 64}])["valid"]
    assert not m.verify_frame_manifest(tmp_path, [])["valid"]


def test_frame_manifest_rejects_windows_aliases_and_traversal(tmp_path):
    import hashlib
    m = _mod()
    (tmp_path / "Frame.npz").write_bytes(b"data")
    digest = hashlib.sha256(b"data").hexdigest()
    entries = [{"file": "Frame.npz", "sha256": digest},
               {"file": "frame.npz", "sha256": digest}]
    result = m.verify_frame_manifest(tmp_path, entries)
    assert not result["valid"] and "duplicate:frame.npz" in result["errors"]
    for name in (r"..\outside.npz", "../outside.npz", "C:outside.npz"):
        result = m.verify_frame_manifest(tmp_path, [{"file": name, "sha256": digest}])
        assert not result["valid"] and "invalid_filename" in result["errors"]


def test_frame_manifest_rejects_missing_and_malformed_digest(tmp_path):
    import hashlib
    m = _mod()
    (tmp_path / "present.npz").write_bytes(b"frame")
    good = hashlib.sha256(b"frame").hexdigest()
    result = m.verify_frame_manifest(tmp_path, [
        {"file": "present.npz", "sha256": good},
        {"file": "missing.npz", "sha256": good},
    ])
    assert result["valid"] is False
    assert any("missing.npz" in error for error in result["errors"])
    result = m.verify_frame_manifest(tmp_path, [{"file": "present.npz", "sha256": "x" * 64}])
    assert result["valid"] is False
    assert "invalid_digest:present.npz" in result["errors"]


def test_sites_exclude_eval_geometry():
    m = _mod()
    assert set(m.SITES) == {"S1", "S2", "S3", "P"}
    assert m.SITES["P"]["split"] == "holdout"
    # 评测站点 (500,500) 不得出现在采集站里
    assert all([500.0, 500.0, 100.0] != s["mid"] for s in m.SITES.values())


def test_site_doc_is_proving_lane():
    m = _mod()
    doc = m.site_doc("S1")
    assert doc["scene_type"] == "proving_lane"
    assert doc["capture"]["split"] == "train"
    assert len(doc["scene"]["lines"]) == 2


def test_cell_args_and_unknown():
    m = _mod()
    assert m.cell_args("dawn_rain") == ("06:30", "rainy")
    assert m.cell_args("noon_fog") == ("12:00", "foggy_morning")
    assert m.cell_args("night_clear") == ("21:30", None)
    try:
        m.cell_args("midnight_rain")
    except ValueError:
        pass
    else:
        raise AssertionError("未知格必须报错")


def test_capture_cmd_has_condition_and_frames(tmp_path, monkeypatch):
    m = _mod()
    monkeypatch.setattr(m, "SITES_DIR", tmp_path)
    (tmp_path / "S1.json").write_text("{}", encoding="utf-8")
    cmd = m.capture_cmd("S1", "dusk_fog", tmp_path / "probe.json", frames=30)
    assert "--weather-preset" in cmd
    assert cmd[cmd.index("--weather-preset") + 1] == "foggy_morning"
    assert cmd[cmd.index("--tod") + 1] == "18:30"
    assert cmd[cmd.index("--frames") + 1] == "30"
    assert "--save-frames" in cmd


def test_env_evidence_parses_readback(tmp_path):
    m = _mod()
    p = tmp_path / "x.run.log"
    p.write_text("[fsd-drive] environment -> tod=06:30 weather=rainy "
                 "applied_tod=06:29:59\n", encoding="utf-8")
    ev = m._env_evidence(p)
    assert ev["tod_after"] == "06:29:59"
    assert ev["weather"] == "rainy"
    assert m._env_evidence(tmp_path / "missing.log")["tod_after"] is None


def test_proving_env_is_diagnostic_config():
    m = _mod()
    assert m.PROVING_ENV == {"BEAMNG_CV_WHITE": "0",
                             "BEAMNG_LINE_MERGE": "0",
                             "BEAMNG_PAIR_WIDTH_NEAR": "1"}


def test_capture_cmd_uses_kinematic_mode(tmp_path, monkeypatch):
    """采集走运动学步进，不再依赖 FSD 放置/驾驶。

    2026-11-06：原实现传 --drive-seconds 让 fsd_drive 开车攒帧，但模型在
    低照度看不见线就永远出不了放置阶段（S1 四格 1848 帧、位移 <=0.0007 m）。
    现在改为 --kinematic-capture，与模型能力完全解耦。
    """
    m = _mod()
    monkeypatch.setattr(m, "SITES_DIR", tmp_path)
    (tmp_path / "S1.json").write_text("{}", encoding="utf-8")
    cmd = m.capture_cmd("S1", "noon_rain", tmp_path / "probe.json", frames=60)
    assert "--kinematic-capture" in cmd
    assert "--drive-seconds" not in cmd, "FSD 驾驶窗口已废弃"
    assert cmd[cmd.index("--kinematic-step-m") + 1] == "1.0"
    assert cmd[cmd.index("--kinematic-span-m") + 1] == "30.0"


def test_p_site_has_four_holdout_cells():
    m = _mod()
    assert len(m.P_CELLS) == 4
    assert set(m.P_CELLS) == {"noon_clear", "noon_rain",
                              "dusk_clear", "night_clear"}


def test_run_site_does_not_trust_historical_ok_and_continues(tmp_path, monkeypatch):
    m = _mod()
    monkeypatch.setattr(m, "_CAPTURE_ENABLED", True)
    monkeypatch.setattr(m, "CAP_DIR", tmp_path)
    monkeypatch.setattr(m, "wait_engine_gone", lambda *a, **k: True)
    called = []
    monkeypatch.setattr(m, "capture", lambda site, cell, **k: (called.append(cell), 0)[1])
    ok = tmp_path / "P" / "noon_clear"
    ok.mkdir(parents=True)
    (ok / "capture.json").write_text('{"status": "ok"}', encoding="utf-8")
    assert m.run_site("P") == 1
    assert (ok / "capture.json").read_text(encoding="utf-8") == '{"status": "ok"}'
    assert called == ["noon_rain", "dusk_clear", "night_clear"]


def test_run_site_propagates_failure_without_overwriting_evidence(tmp_path, monkeypatch):
    m = _mod()
    monkeypatch.setattr(m, "_CAPTURE_ENABLED", True)
    monkeypatch.setattr(m, "CAP_DIR", tmp_path)
    monkeypatch.setattr(m, "wait_engine_gone", lambda *a, **k: True)
    monkeypatch.setattr(m.time, "sleep", lambda _: None)
    man = tmp_path / "P" / "noon_clear" / "capture.json"
    man.parent.mkdir(parents=True)
    original = b'{"status":"INCOMPLETE", "evidence":"keep"}'
    man.write_bytes(original)

    def boom(site, cell, **k):
        raise RuntimeError("probe died")

    monkeypatch.setattr(m, "capture", boom)
    assert m.run_site("P", only=("noon_clear",)) == 1
    assert man.read_bytes() == original


def test_run_site_refuses_unknown_engine_and_propagates_nonzero(tmp_path, monkeypatch):
    m = _mod()
    monkeypatch.setattr(m, "_CAPTURE_ENABLED", True)
    monkeypatch.setattr(m, "CAP_DIR", tmp_path)
    monkeypatch.setattr(m.time, "sleep", lambda _: None)
    called = []
    monkeypatch.setattr(m, "capture", lambda *a, **k: called.append(a) or 2)
    monkeypatch.setattr(m, "wait_engine_gone", lambda: False)
    assert m.run_site("S1", only=("noon_clear",)) == 1
    assert called == []
    monkeypatch.setattr(m, "wait_engine_gone", lambda: True)
    assert m.run_site("S1", only=("noon_clear",)) == 1
    assert len(called) == 1


def test_wait_engine_gone_clean(monkeypatch):
    m = _mod()
    monkeypatch.setattr(m, "_game_pids", lambda: set())
    assert m.wait_engine_gone(0.1) is True


def test_wait_engine_unknown_or_live_is_not_gone(monkeypatch):
    m = _mod()
    monkeypatch.setattr(m, "_game_pids", lambda: None)
    assert m.wait_engine_gone(0) is False
    monkeypatch.setattr(m, "_game_pids", lambda: {123})
    assert m.wait_engine_gone(0) is False


def test_game_enumeration_failure_stays_unknown(monkeypatch):
    from beamng_autopilot.experiments import collection
    m = _mod()
    monkeypatch.setattr(collection, "game_pids", lambda: None)
    assert m._game_pids() is None


def test_run_all_covers_36_cells(monkeypatch):
    m = _mod()
    monkeypatch.setattr(m, "_CAPTURE_ENABLED", True)
    order = []
    monkeypatch.setattr(m, "run_site", lambda site, **k: (
        order.append(site), 1 if site == "S2" else 0)[1])
    assert m.run_all() == 1
    assert order == ["S1", "S2", "S3", "P"]


# ── 天气读回语义（§6.2-2：独立引擎侧证据，非请求回声）────────────────────
class _FakeBng:
    def __init__(self, tod_after="12:00:00", preset="nil"):
        self.env = self
        self.control = self
        self._tod_after = tod_after
        self._preset = preset  # 引擎侧状态：读回的唯一来源

    def set_weather_preset(self, preset):
        # 读回是引擎状态而非请求回声：set 不改变读回（过渡期语义由真机验证）
        pass

    def set_tod(self, tod):
        pass

    def get_tod(self):
        return {"timeStr": self._tod_after}

    def queue_lua_command(self, chunk, response=False):
        assert "getCurrentWeatherPreset" in chunk
        return self._preset


class _FakeConn:
    def __init__(self, bng):
        self.bng = bng


def _probe_mod():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "mpp_probe", str(root / "scripts" / "m5_pairing_probe.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _apply(conn, tod, weather):
    return _probe_mod()._apply_environment(conn, tod, weather)


def test_weather_readback_rainy_matches_only_engine_side():
    conn = _FakeConn(_FakeBng(tod_after="12:00:00", preset="rainy"))
    rec = _apply(conn, "12:00", "rainy")
    assert rec["tod_matches"] is True
    assert rec["weather_matches"] is True and rec["valid"] is True


def test_weather_readback_fresh_engine_fails_rainy_request():
    """全新引擎 preset=nil（未选预设）→ 请求 rainy 不能声称匹配。"""
    conn = _FakeConn(_FakeBng(tod_after="12:00:00", preset="nil"))
    rec = _apply(conn, "12:00", "rainy")
    assert rec["weather_matches"] is False and rec["valid"] is False


def test_weather_readback_clear_passes_on_nil_and_fails_on_leftover_rain():
    conn = _FakeConn(_FakeBng(tod_after="06:30:00", preset="nil"))
    rec = _apply(conn, "06:30", None)
    assert rec["weather_matches"] is True and rec["valid"] is True
    conn = _FakeConn(_FakeBng(tod_after="06:30:00", preset="rainy"))
    rec = _apply(conn, "06:30", None)
    assert rec["weather_matches"] is False and rec["valid"] is False


def test_weather_readback_tod_mismatch_blocks_even_with_weather():
    conn = _FakeConn(_FakeBng(tod_after="18:30:00", preset="rainy"))
    rec = _apply(conn, "12:00", "rainy")
    assert rec["tod_matches"] is False and rec["valid"] is False


# ── attempt 编排（§6.2 门 1/4）───────────────────────────────────────────
class _FakeProc:
    def __init__(self, rc=0, stdout=""):
        self.returncode = rc
        self.stdout = stdout
        self.stderr = ""


def _fake_probe_run_factory(tmp_path, *, rc=0, n_frames=25, weather="rainy",
                            tod="12:00:00", engine_pid=4242, report=None):
    """伪造探针子进程：在 attempt 目录里落盘 probe.json + 帧清单 + 帧。"""
    import hashlib

    state = {"killed_pids": []}

    def fake_run(cmd, **kwargs):
        out = Path(cmd[cmd.index("--out") + 1])
        frames_dir = out.with_suffix(".frames")
        frames_dir.mkdir(parents=True, exist_ok=True)
        files = []
        for i in range(n_frames):
            f = frames_dir / f"frame_{i:05d}.npz"
            f.write_bytes(b"frame-%04d" % i)
            files.append({"file": f.name,
                          "sha256": hashlib.sha256(f.read_bytes()).hexdigest()})
        if report is None:
            rep = {"schema": "pairing-probe-v2", "phase": "done",
                   "engine_pid": engine_pid,
                   "environment": {"tod_requested": tod,
                                   "tod_after": tod,
                                   "tod_matches": True,
                                   "weather_requested": weather,
                                   "weather_readback": weather,
                                   "weather_matches": True,
                                   "valid": True},
                   "kinematic": {"saved": n_frames, "files": files},
                   "frames": n_frames}
        else:
            rep = report
        out.write_text(json.dumps(rep, ensure_ascii=False), encoding="utf-8")
        return _FakeProc(rc=rc)

    return fake_run, state


def _wire_capture(tmp_path, monkeypatch, fake_run, state, engine_gone=True,
                  game_pids=None):
    """game_pids：每次 _game_pids() 调用的返回序列（最后一项重复）。
    None = 恒空集。差分归属测试用 [set(), {leaked...}] 模拟泄漏。"""
    m = _mod()
    seq = [set()] if game_pids is None else [set(s) for s in game_pids]
    calls = {"n": 0}

    def fake_pids():
        idx = min(calls["n"], len(seq) - 1)
        calls["n"] += 1
        return set(seq[idx])

    monkeypatch.setattr(m, "CAP_DIR", tmp_path / "cap")
    monkeypatch.setattr(m, "SITES_DIR", tmp_path / "sites")
    (tmp_path / "sites").mkdir(parents=True, exist_ok=True)
    (tmp_path / "sites" / "S1.json").write_text("{}", encoding="utf-8")
    def fake_probe(cmd, log_path, *, timeout_s=1200):
        rc = fake_run(cmd).returncode
        Path(log_path).write_text("fake probe\n", encoding="utf-8")
        return rc, False

    monkeypatch.setattr(m, "run_probe_attempt", fake_probe)
    monkeypatch.setattr(m, "wait_engine_gone", lambda *a, **k: engine_gone)
    monkeypatch.setattr(m, "_game_pids", fake_pids)
    monkeypatch.setattr(m, "_kill_pid_tree",
                        lambda pid: state["killed_pids"].append(pid) or True)
    monkeypatch.setattr(m.time, "sleep", lambda _: None)
    return m


def test_capture_attempt_flow_evidence_ready(tmp_path, monkeypatch):
    fake_run, state = _fake_probe_run_factory(tmp_path)
    m = _wire_capture(tmp_path, monkeypatch, fake_run, state)
    rc = m.capture("S1", "noon_rain", frames=60)
    assert rc == 0
    cell = tmp_path / "cap" / "S1" / "noon_rain"
    attempts = list(p for p in cell.iterdir() if p.is_dir())
    assert len(attempts) == 1
    att = attempts[0]
    result = json.loads((att / "attempt_result.json").read_text("utf-8"))
    assert result["status"] == "EVIDENCE_READY"
    assert result["engine_pid"] == 4242
    assert result["frame_verification"]["valid"] is True
    assert state["killed_pids"] == [4242]
    summary = json.loads((cell / "capture.json").read_text("utf-8"))
    assert summary["status"] == "ok" and summary["frames_saved"] == 25
    assert summary["env"]["tod_matches"] is True


def test_capture_tampered_frame_manifest_is_incomplete(tmp_path, monkeypatch):
    """磁盘帧被篡改/缺失 → 清单核对失败 → INCOMPLETE，不入池资格。"""
    fake_run, state = _fake_probe_run_factory(tmp_path)

    def tampering_run(cmd, **kwargs):
        rc = fake_run(cmd, **kwargs)
        out = Path(cmd[cmd.index("--out") + 1])
        victim = out.with_suffix(".frames") / "frame_00003.npz"
        victim.write_bytes(b"tampered")
        return rc

    m = _wire_capture(tmp_path, monkeypatch, tampering_run, state)
    rc = m.capture("S1", "noon_rain")
    assert rc == 1
    attempts = list((tmp_path / "cap" / "S1" / "noon_rain").glob("attempt_*"))
    result = json.loads((attempts[0] / "attempt_result.json").read_text("utf-8"))
    assert result["status"] == "INCOMPLETE"
    assert any(e.startswith("frame_manifest:") for e in result["reasons"])
    summary = json.loads(
        (tmp_path / "cap" / "S1" / "noon_rain" / "capture.json")
        .read_text("utf-8"))
    assert summary["status"] == "INCOMPLETE"


def test_capture_no_pid_leaked_engine_is_swept_by_diff(tmp_path, monkeypatch):
    """探针没记录 PID（连接阶段挂起）但引擎泄漏 → 差分清扫按 PID 清掉。"""
    fake_run, state = _fake_probe_run_factory(tmp_path, engine_pid=None)
    m = _wire_capture(tmp_path, monkeypatch, fake_run, state,
                      game_pids=[set(), {999, 1000}])
    assert m.capture("S1", "noon_rain") == 0
    attempts = list((tmp_path / "cap" / "S1" / "noon_rain").glob("attempt_*"))
    result = json.loads((attempts[0] / "attempt_result.json").read_text("utf-8"))
    assert result["status"] == "EVIDENCE_READY"
    assert result["engine_pid"] is None
    assert result["engine_sweep_ok"] is True
    assert sorted(state["killed_pids"]) == [999, 1000]


def test_capture_unknown_engine_enumeration_is_ownership_unverified(
        tmp_path, monkeypatch):
    """进程枚举未知（None）→ 不能建立差分基线 → 拒绝发起。"""
    fake_run, state = _fake_probe_run_factory(tmp_path)
    m = _mod()
    monkeypatch.setattr(m, "CAP_DIR", tmp_path / "cap")
    monkeypatch.setattr(m, "SITES_DIR", tmp_path / "sites")
    (tmp_path / "sites").mkdir(parents=True, exist_ok=True)
    (tmp_path / "sites" / "S1.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(m, "wait_engine_gone", lambda *a, **k: True)
    monkeypatch.setattr(m, "_game_pids", lambda: None)
    monkeypatch.setattr(m, "run_probe_attempt", lambda *a, **k: pytest.fail(
        "unknown enumeration must not launch a probe"))
    assert m.capture("S1", "noon_rain") == 1
    assert not list((tmp_path / "cap" / "S1" / "noon_rain").glob("attempt_*"))


def test_capture_incomplete_then_reclaim_supersedes_legacy(tmp_path, monkeypatch):
    """INCOMPLETE 保留 legacy；--reclaim 后 EVIDENCE_READY 才并入并取代。"""
    fake_fail, _ = _fake_probe_run_factory(tmp_path, rc=3, report={
        "schema": "pairing-probe-v2", "phase": "done", "engine_pid": None,
        "environment": {"valid": False}, "kinematic": {"saved": 0,
                                                       "files": []},
        "frames": 0})
    m = _wire_capture(tmp_path, monkeypatch, fake_fail, {"killed_pids": []})
    assert m.capture("S1", "noon_rain", reclaim=True) == 1
    legacy = json.loads(
        (tmp_path / "cap" / "S1" / "noon_rain" / "capture.json")
        .read_text("utf-8"))
    assert legacy["status"] == "INCOMPLETE"  # 未过门：legacy 保留

    fake_ok, state = _fake_probe_run_factory(tmp_path)
    m2 = _wire_capture(tmp_path, monkeypatch, fake_ok, state)
    assert m2.capture("S1", "noon_rain", frames=60, reclaim=True) == 0
    summary = json.loads(
        (tmp_path / "cap" / "S1" / "noon_rain" / "capture.json")
        .read_text("utf-8"))
    assert summary["status"] == "ok"
    assert summary["superseded_legacy"] is True
    assert summary["legacy"]["status"] == "INCOMPLETE"
    attempts = list(p for p in
                    (tmp_path / "cap" / "S1" / "noon_rain").iterdir()
                    if p.is_dir())
    assert len(attempts) == 2  # 两次尝试目录都保留（§6.2-4 可恢复）


def test_run_site_retries_incomplete_and_skips_ok(tmp_path, monkeypatch):
    m = _mod()
    monkeypatch.setattr(m, "_CAPTURE_ENABLED", True)
    monkeypatch.setattr(m, "CAP_DIR", tmp_path)
    monkeypatch.setattr(m, "wait_engine_gone", lambda *a, **k: True)
    called = []
    monkeypatch.setattr(m, "capture",
                        lambda site, cell, **k: called.append(cell) or 0)
    ok_cell = tmp_path / "P" / "noon_clear"
    ok_cell.mkdir(parents=True)
    (ok_cell / "capture.json").write_text('{"status": "ok"}', encoding="utf-8")
    inc_cell = tmp_path / "P" / "noon_rain"
    inc_cell.mkdir(parents=True)
    (inc_cell / "capture.json").write_text('{"status": "INCOMPLETE"}',
                                           encoding="utf-8")
    # ok 格默认跳过（审计标记 → 批次未完成 = 1）；INCOMPLETE 自动重试
    assert m.run_site("P") == 1
    assert called == ["noon_rain", "dusk_clear", "night_clear"]
    # reclaim=True 时 ok 格也重采，批次全绿
    called.clear()
    assert m.run_site("P", reclaim=True) == 0
    assert called == ["noon_clear", "noon_rain", "dusk_clear", "night_clear"]


def test_kill_pid_tree_uses_recorded_pid(tmp_path, monkeypatch):
    """_kill_pid_tree 按记录 PID 调 taskkill；GBK 输出 errors=replace。"""
    import subprocess as sp
    m = _mod()
    seen = {}

    def fake_run(cmd, **k):
        seen["cmd"] = cmd
        seen["errors"] = k.get("errors")
        return sp.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(sp, "run", fake_run)
    assert m._kill_pid_tree(4242) is True
    assert seen["cmd"] == ["taskkill", "/PID", "4242", "/T", "/F"]
    assert seen["errors"] == "replace"


def test_run_probe_attempt_survives_grandchild_holding_stdout(tmp_path):
    """回归：引擎（孙进程）继承探针 stdout 不得卡住编排侧。

    2026-11-07 实测：管道模式下探针 22:17 已写完报告，引擎仍持有管道
    写端，subprocess.run 在超时后无超时地二次 drain → 编排器 22:53
    仍挂死。落文件 + 独立 wait 后应立刻返回。
    """
    import subprocess as sp
    import sys
    import time
    m = _mod()
    pidfile = tmp_path / "grandchild.pid"
    code = (
        "import subprocess, sys\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)'])\n"
        f"open({str(pidfile)!r}, 'w').write(str(p.pid))\n"
        "print('probe done', flush=True)\n"
    )
    log = tmp_path / "probe.log"
    t0 = time.monotonic()
    rc, timed_out = m.run_probe_attempt([sys.executable, "-c", code], log,
                                        timeout_s=30)
    elapsed = time.monotonic() - t0
    assert rc == 0 and timed_out is False
    assert elapsed < 5.0, f"编排侧被孙进程句柄卡住 {elapsed:.1f}s"
    assert "probe done" in log.read_text(encoding="utf-8")
    # 清掉测试留下的孙进程（等它自然退出太慢）
    sp.run(["taskkill", "/PID", pidfile.read_text().strip(), "/T", "/F"],
           capture_output=True, text=True, errors="replace")


def test_run_probe_attempt_timeout_kills_tree(tmp_path):
    """超时按进程树强杀：探针 + 它启动的子进程都不留孤儿。"""
    import subprocess as sp
    import sys
    m = _mod()
    pidfile = tmp_path / "child.pid"
    code = (
        "import subprocess, sys, time\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open({str(pidfile)!r}, 'w').write(str(p.pid))\n"
        "time.sleep(60)\n"
    )
    log = tmp_path / "probe.log"
    rc, timed_out = m.run_probe_attempt([sys.executable, "-c", code], log,
                                        timeout_s=3)
    assert timed_out is True
    assert rc != 0
    assert "--- timeout (3s) ---" in log.read_text(encoding="utf-8")
    pid = int(pidfile.read_text())
    listing = sp.run(["tasklist", "/FI", f"PID eq {pid}"],
                     capture_output=True, text=True, errors="replace").stdout
    assert str(pid) not in listing
