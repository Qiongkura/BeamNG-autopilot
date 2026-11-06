# -*- coding: utf-8 -*-
"""试车场采集 runner：站点/条件格/命令/环境证据（预注册 §2-§3）。"""
import importlib.util
from pathlib import Path


def _mod():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "mpc_capture", str(root / "scripts" / "m5_proving_capture.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_capture_cli_is_blocked_before_launch(monkeypatch):
    import sys
    m = _mod()
    monkeypatch.setattr(m, "capture", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("blocked CLI must not launch")))
    monkeypatch.setattr(sys, "argv", ["capture", "--capture", "--site", "S1",
                                     "--cell", "noon_rain"])
    assert m.main() == 3


def test_direct_capture_and_cleanup_refuse_without_touching_resources(tmp_path, monkeypatch):
    import pytest
    m = _mod()
    monkeypatch.setattr(m, "CAP_DIR", tmp_path)
    monkeypatch.setattr(m.subprocess, "run", lambda *a, **k: pytest.fail("must not start or kill a process"))
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
    monkeypatch.setattr(m, "run_site", lambda site: (order.append(site),
                                                    1 if site == "S2" else 0)[1])
    assert m.run_all() == 1
    assert order == ["S1", "S2", "S3", "P"]
