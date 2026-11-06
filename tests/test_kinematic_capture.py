"""运动学步进采集：与模型能力解耦、位姿多样、帧格式与 FSD 采集一致。

背景（2026-11-06 实测）：原采集走 FSD 严格放置，模型在低照度看不见线就
永远出不了放置阶段——`S1/{dawn_clear,dawn_rain,dawn_fog,noon_clear}` 共
1848 帧、位移 ≤0.0007 m、独立位姿 1 个，全是废数据。运动学采集沿站点
中心线逐站传送取帧，哪怕全黑画面也能采满整条里程。

这些测试用假 conn / 假 ring 钉住纯逻辑，不启动游戏。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "m5_pairing_probe", ROOT / "scripts" / "m5_pairing_probe.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["m5_pairing_probe"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_collect_rejects_unverified_kinematic_before_connecting(monkeypatch):
    from types import SimpleNamespace
    from beamng_autopilot import connector

    def forbidden(*args, **kwargs):
        raise AssertionError("blocked collection must not connect")

    monkeypatch.setattr(connector, "BeamNGConnector", forbidden)
    assert _load().collect(SimpleNamespace(kinematic_capture=True)) == 3


def test_pairing_probe_does_not_kill_processes_by_pid_difference():
    import ast
    source = (ROOT / "scripts" / "m5_pairing_probe.py").read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    collect = next(n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name == "collect")
    forbidden = {"close_started_game", "taskkill", "kill"}
    calls = {n.func.id if isinstance(n.func, ast.Name) else n.func.attr
             for n in ast.walk(collect) if isinstance(n, ast.Call)
             and isinstance(n.func, (ast.Name, ast.Attribute))}
    assert not (calls & forbidden)


class _State:
    def __init__(self, pos, heading):
        self.pos = np.asarray(pos, float)
        self.heading = float(heading)


class _FakeConn:
    """记录每次 safe_teleport 的位姿；不依赖游戏。"""

    def __init__(self):
        self.io_lock = __import__("threading").RLock()
        self.calls = []
        self._state = _State([0.0, 0.0, 100.0], 0.0)

    def safe_teleport(self, x, y, heading_deg=None, settle=8, **_kw):
        self.calls.append((float(x), float(y), float(heading_deg)))
        self._state = _State([x, y, 100.0], np.radians(heading_deg or 0.0))
        return True

    def get_state(self):
        return self._state


class _FakeRing:
    def __init__(self, width=536, height=403, roles=None):
        self.width, self.height = width, height
        self.closed = False

    def grab_ring(self):
        return {"front_main": (np.zeros((self.height, self.width, 3),
                                        np.uint8), object())}

    def close(self):
        self.closed = True


def _scene(mid=(300.0, 700.0, 100.0), direction=(0.0, 1.0, 0.0)):
    return {"scene": {"mid": list(mid), "dir": list(direction),
                      "half_width_m": 3.5}}


@pytest.mark.parametrize("overrides", [
    {"step_m": 0}, {"step_m": -1}, {"step_m": float("nan")},
    {"span_m": float("inf")}, {"span_m": 3}, {"settle_s": -1},
    {"lateral_jitter": -1}, {"yaw_jitter_deg": float("inf")},
    {"frames_per_station": 0}, {"frames_per_station": True},
])
def test_invalid_kinematic_parameters_fail_before_resources(tmp_path, monkeypatch, overrides):
    mod = _load()
    conn = _FakeConn()
    def forbidden(*args, **kwargs):
        pytest.fail("camera must not open for invalid input")
    monkeypatch.setattr("beamng_autopilot_tech.providers.TechCameraRingProvider", forbidden)
    with pytest.raises(ValueError):
        mod.kinematic_capture(conn, _scene(), tmp_path / "new", **overrides)
    assert not (tmp_path / "new").exists()
    assert conn.calls == []


def test_invalid_kinematic_direction_rejected(tmp_path):
    with pytest.raises(ValueError, match="geometry"):
        _load().kinematic_capture(_FakeConn(), _scene(direction=(0, 0, 0)), tmp_path / "new")
    assert not (tmp_path / "new").exists()


def test_kinematic_capture_walks_the_whole_span(tmp_path, monkeypatch) -> None:
    mod = _load()
    conn = _FakeConn()
    ring = _FakeRing()
    monkeypatch.setattr(
        "beamng_autopilot_tech.providers.TechCameraRingProvider",
        lambda *a, **k: ring, raising=False)
    info = mod.kinematic_capture(
        conn, _scene(), tmp_path, step_m=5.0, span_m=20.0,
        lateral_jitter=0.0, yaw_jitter_deg=0.0, settle_s=0.0)
    # 4, 9, 14, 19 四站
    assert info["stations"] == 4, info
    assert info["saved"] == 4, info
    ys = [c[1] for c in conn.calls]
    assert ys == [704.0, 709.0, 714.0, 719.0], ys
    assert ring.closed is True


def test_lateral_and_yaw_jitter_stay_in_range(tmp_path, monkeypatch) -> None:
    mod = _load()
    conn = _FakeConn()
    monkeypatch.setattr(
        "beamng_autopilot_tech.providers.TechCameraRingProvider",
        lambda *a, **k: _FakeRing(), raising=False)
    mod.kinematic_capture(
        conn, _scene(), tmp_path, step_m=2.0, span_m=30.0,
        lateral_jitter=0.25, yaw_jitter_deg=2.0, settle_s=0.0)
    xs = np.array([c[0] for c in conn.calls])
    yaws = np.array([c[2] for c in conn.calls])
    assert np.all(np.abs(xs - 300.0) <= 0.25 + 1e-9), xs
    assert np.all(np.abs(yaws - 90.0) <= 2.0 + 1e-9), yaws
    # 抖动必须真的发生（不是全零）
    assert xs.std() > 0.0
    assert yaws.std() > 0.0


def test_frames_are_written_in_the_fsd_probe_format(tmp_path,
                                                   monkeypatch) -> None:
    mod = _load()
    monkeypatch.setattr(
        "beamng_autopilot_tech.providers.TechCameraRingProvider",
        lambda *a, **k: _FakeRing(), raising=False)
    mod.kinematic_capture(_FakeConn(), _scene(), tmp_path, step_m=10.0,
                          span_m=10.0, lateral_jitter=0.0,
                          yaw_jitter_deg=0.0, settle_s=0.0)
    files = sorted(tmp_path.glob("frame_*.npz"))
    assert len(files) == 1, files
    z = np.load(files[0], allow_pickle=True)
    assert set(z.keys()) == {"frame", "pos", "heading", "cam", "t"}, z.keys()
    assert z["frame"].shape == (403, 536, 3)


def test_capture_does_not_depend_on_the_model(tmp_path, monkeypatch) -> None:
    """全黑帧（模型必然 0 检出）也必须照常采满——这正是原采集的死锁点。"""
    mod = _load()

    class _BlackRing(_FakeRing):
        def grab_ring(self):
            return {"front_main": (np.zeros((403, 536, 3), np.uint8),
                                   object())}

    monkeypatch.setattr(
        "beamng_autopilot_tech.providers.TechCameraRingProvider",
        lambda *a, **k: _BlackRing(), raising=False)
    info = mod.kinematic_capture(_FakeConn(), _scene(), tmp_path,
                                 step_m=5.0, span_m=20.0, lateral_jitter=0.0,
                                 yaw_jitter_deg=0.0, settle_s=0.0)
    assert info["saved"] == 4, info


def test_missing_camera_writes_nothing_but_does_not_raise(tmp_path,
                                                         monkeypatch) -> None:
    mod = _load()

    class _DeadRing(_FakeRing):
        def grab_ring(self):
            raise RuntimeError("camera gone")

    monkeypatch.setattr(
        "beamng_autopilot_tech.providers.TechCameraRingProvider",
        lambda *a, **k: _DeadRing(), raising=False)
    info = mod.kinematic_capture(_FakeConn(), _scene(), tmp_path,
                                 step_m=5.0, span_m=20.0, lateral_jitter=0.0,
                                 yaw_jitter_deg=0.0, settle_s=0.0)
    assert info["saved"] == 0
    assert not list(tmp_path.glob("frame_*.npz"))


def test_tod_readback_matching_and_unknown():
    mod = _load()
    assert mod._tod_match("06:30", "06:29:59") is True
    assert mod._tod_match("00:00", "23:59:59") is True
    assert mod._tod_match("06:30", "12:00:00") is False
    assert mod._tod_match("06:30", None) is None
    assert mod._tod_match("25:00", "00:00:00") is None
    assert mod._tod_match("06:30", "not a time") is None


def test_invalid_environment_input_never_mutates_game():
    mod = _load()
    from types import SimpleNamespace
    class Env:
        def __init__(self):
            self.calls = []
        def set_weather_preset(self, value):
            self.calls.append(("weather", value))
        def set_tod(self, **kwargs):
            self.calls.append(("tod", kwargs))
    env = Env()
    conn = SimpleNamespace(bng=SimpleNamespace(env=env))
    for tod, weather in (("25:00", "rainy"), (None, "rainy"), ("06:30", "")):
        result = mod._apply_environment(conn, tod, weather)
        assert result["valid"] is False and result.get("error")
    assert env.calls == []


def test_environment_readback_requires_tod_and_rejects_unread_weather(monkeypatch):
    mod = _load()

    class Env:
        def set_tod(self, tod):
            self.tod = tod
        def set_weather_preset(self, value):
            self.weather = value
        def get_tod(self):
            return {"timeStr": "06:30:00"}

    class Bng:
        env = Env()

    class Conn:
        bng = Bng()

    clear = mod._apply_environment(Conn(), "06:30", None)
    assert clear["valid"] is False and clear["tod_after"] == "06:30:00"
    assert clear["tod_matches"] is True and clear["weather_readback"] == "UNKNOWN"
    rain = mod._apply_environment(Conn(), "06:30", "rainy")
    assert rain["valid"] is False
    assert rain["weather_readback"] == "UNKNOWN"
