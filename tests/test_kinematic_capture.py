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


def _load(*, enable_for_test=False):
    spec = importlib.util.spec_from_file_location(
        "m5_pairing_probe", ROOT / "scripts" / "m5_pairing_probe.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["m5_pairing_probe"] = mod
    spec.loader.exec_module(mod)
    if enable_for_test:
        mod._KINEMATIC_ENABLED = True
    return mod


def _load_capture():
    spec = importlib.util.spec_from_file_location(
        "mpc_capture_for_kinematic",
        ROOT / "scripts" / "m5_proving_capture.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_kinematic_eligibility_gate_is_orchestrator_side(monkeypatch, tmp_path):
    """3591c22 起资格门从探针搬到编排侧（探针只采集与落盘）。

    探针的运动学入口已解锁；"证据可用"必须由 m5_proving_capture 的
    attempt 流程判定——kill-switch、引擎归属/清理、环境读回、帧清单
    核对全部通过才写 EVIDENCE_READY，探针自身不提供任何旁路开关。
    """
    assert _load()._KINEMATIC_ENABLED is True
    cap = _load_capture()
    monkeypatch.setattr(cap, "_CAPTURE_ENABLED", False)
    monkeypatch.setattr(cap, "CAP_DIR", tmp_path)
    with pytest.raises(RuntimeError, match="BLOCKED"):
        cap.capture("S1", "noon_clear")


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


def test_kinematic_capture_has_no_verification_bypass_parameter():
    """资格判定不在探针：kinematic_capture 没有 verify/enabled 之类旁路
    参数，直接调用只产出帧；是否可作证据由编排侧 attempt 闭环决定。"""
    import inspect
    sig = inspect.signature(_load().kinematic_capture)
    assert not ({"enabled", "verified", "bypass", "allow"}
                & set(sig.parameters))


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
    mod = _load(enable_for_test=True)
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
        _load(enable_for_test=True).kinematic_capture(_FakeConn(), _scene(direction=(0, 0, 0)), tmp_path / "new")
    assert not (tmp_path / "new").exists()


def test_kinematic_capture_walks_the_whole_span(tmp_path, monkeypatch) -> None:
    mod = _load(enable_for_test=True)
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
    mod = _load(enable_for_test=True)
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
    mod = _load(enable_for_test=True)
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
    mod = _load(enable_for_test=True)

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
    mod = _load(enable_for_test=True)

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
    mod = _load(enable_for_test=True)
    assert mod._tod_match("06:30", "06:29:59") is True
    assert mod._tod_match("00:00", "23:59:59") is True
    assert mod._tod_match("06:30", "12:00:00") is False
    assert mod._tod_match("06:30", None) is None
    assert mod._tod_match("25:00", "00:00:00") is None
    assert mod._tod_match("06:30", "not a time") is None


def test_invalid_environment_input_never_mutates_game():
    mod = _load(enable_for_test=True)
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
    mod = _load(enable_for_test=True)

    class Env:
        def set_tod(self, tod):
            self.tod = tod
        def set_weather_preset(self, value):
            self.weather = value
        def get_tod(self):
            return {"timeStr": "06:30:00"}

    class Control:
        def __init__(self, resp=None, boom=False):
            self.resp, self.boom = resp, boom
        def queue_lua_command(self, *args, **kwargs):
            if self.boom:
                raise RuntimeError("rpc unavailable")
            return self.resp

    class Bng:
        def __init__(self, control):
            self.env, self.control = Env(), control

    class Conn:
        def __init__(self, control):
            self.bng = Bng(control)

    fast = {"settle_s": 0.01, "poll_s": 0.001}
    # 无预设 + 引擎报告 nil → clear 证据成立（全新引擎默认态即无预设）
    clear = mod._apply_environment(Conn(Control("nil")), "06:30", None, **fast)
    assert clear["tod_matches"] is True
    assert clear["weather_readback"] is None
    assert clear["weather_matches"] is True and clear["valid"] is True
    # 无预设但引擎残留 rainy → clear 不成立
    left = mod._apply_environment(Conn(Control("rainy")), "06:30", None, **fast)
    assert left["weather_matches"] is False and left["valid"] is False
    # 请求 rainy 但引擎仍 nil（未生效）→ 不成立
    rain = mod._apply_environment(Conn(Control("nil")), "06:30", "rainy", **fast)
    assert rain["weather_matches"] is False and rain["valid"] is False
    # 请求 rainy 且引擎读回 rainy → 成立
    ok = mod._apply_environment(Conn(Control("rainy")), "06:30", "rainy", **fast)
    assert ok["weather_matches"] is True and ok["valid"] is True
    # 读回本身失败（RPC 不可用）→ 不伪造 UNKNOWN 成功，整体 invalid
    unread = mod._apply_environment(Conn(Control(boom=True)), "06:30", None,
                                    **fast)
    assert unread["valid"] is False and "error" in unread


def test_environment_waits_for_weather_animation_before_setting_tod():
    """回归：天气动画回写 TOD 时，等动画落定再设 tod 并轮询到容差内。

    2026-11-07 实测：设完 rainy 立刻设 tod 再立刻读回，读到的是动画写的
    墙钟（14:2x）而不是请求值（06:30）→ 9/9 天气格判 INVALID。
    """
    mod = _load(enable_for_test=True)

    class Env:
        def __init__(self):
            self.weather = None
            self.weather_polls = 0
            self.tod = "14:22:47"
            self.tod_sets = 0
            self.animate = 3   # 前 3 次读回仍是动画态
        def set_weather_preset(self, value):
            self.weather = value
        def get_tod(self):
            # 动画未落定前，读回是墙钟；落定后反映最后一次 set_tod
            return {"timeStr": self.tod if self.weather_polls >= 3
                    else "14:22:47"}
        def set_tod(self, tod):
            self.tod_sets += 1
            self.tod = tod

    class Control:
        def __init__(self, env):
            self.env = env
        def queue_lua_command(self, *args, **kwargs):
            self.env.weather_polls += 1
            # 预设需要 3 次轮询才落定（模拟动画）
            return self.env.weather if self.env.weather_polls >= 3 else "nil"

    class Conn:
        def __init__(self, env):
            self.bng = type("B", (), {})()
            self.bng.env = env
            self.bng.control = Control(env)

    env = Env()
    rec = mod._apply_environment(Conn(env), "06:30", "rainy",
                                 settle_s=5.0, poll_s=0.001)
    assert rec["weather_matches"] is True
    assert rec["tod_matches"] is True and rec["valid"] is True
    assert rec["tod_after"] == "06:30:00"
    assert env.weather_polls >= 3          # 确实等到了动画落定
    assert env.tod_sets >= 1
