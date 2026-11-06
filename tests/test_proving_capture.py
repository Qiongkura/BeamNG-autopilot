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


def test_run_site_skips_ok_and_continues(tmp_path, monkeypatch):
    m = _mod()
    monkeypatch.setattr(m, "_CAPTURE_ENABLED", True)
    monkeypatch.setattr(m, "CAP_DIR", tmp_path)
    monkeypatch.setattr(m, "wait_engine_gone", lambda *a, **k: True)
    called = []
    monkeypatch.setattr(m, "capture", lambda site, cell, **k: called.append(cell))
    ok = tmp_path / "P" / "noon_clear"
    ok.mkdir(parents=True)
    (ok / "capture.json").write_text('{"status": "ok"}', encoding="utf-8")
    m.run_site("P")
    assert called == ["noon_rain", "dusk_clear", "night_clear"]


def test_run_site_records_failure(tmp_path, monkeypatch):
    m = _mod()
    monkeypatch.setattr(m, "_CAPTURE_ENABLED", True)
    monkeypatch.setattr(m, "CAP_DIR", tmp_path)
    monkeypatch.setattr(m, "wait_engine_gone", lambda *a, **k: True)

    def boom(site, cell, **k):
        raise RuntimeError("probe died")

    monkeypatch.setattr(m, "capture", boom)
    m.run_site("P")
    man = tmp_path / "P" / "noon_clear" / "capture.json"
    data = __import__("json").loads(man.read_text(encoding="utf-8"))
    assert data["status"] == "INCOMPLETE"
    assert "probe died" in data["error"]


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
    monkeypatch.setattr(m, "run_site", lambda site: order.append(site))
    m.run_all()
    assert order == ["S1", "S2", "S3", "P"]
