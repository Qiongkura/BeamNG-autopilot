"""FSD 入口的默认值必须合规：sensor + strict，旧路径要显式逃生口。

背景（2026-10-20 评审）：AGENTS.md 铁律禁止 FSD 栈及其入口
（``m5_fsd_drive`` / ``m5_shadow_drive``）用"导航线/地图线 + 偏移"做横向
参考，但两个入口的 ``--lane-mode`` 默认都是 ``map``、``--strict`` 默认关——
于是**每个不带参数的 FSD 运行都在走被禁的路径**，而且跑出来的数字看起来
和合规运行一模一样。这里把默认值与逃生口钉住。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_fsd_drive_defaults_to_sensor_and_strict() -> None:
    mod = _load("m5_fsd_drive")
    args = mod.build_parser().parse_args([])
    assert args.lane_mode == "sensor", "默认必须是感知主导"
    assert args.strict is True, "默认必须开 strict"
    assert args.allow_legacy_map_lane is False


def test_shadow_drive_defaults_to_sensor_and_strict() -> None:
    mod = _load("m5_shadow_drive")
    args = mod.build_parser().parse_args([])
    assert args.lane_mode == "sensor"
    assert args.strict is True
    assert args.allow_legacy_map_lane is False


@pytest.mark.parametrize("entry", ["m5_fsd_drive", "m5_shadow_drive"])
def test_legacy_lane_refuses_to_run_without_the_escape_hatch(
        entry, monkeypatch, capsys) -> None:
    """``--lane-mode map`` 不带逃生口 → 立刻 SystemExit(2)，不进驾驶循环。"""
    mod = _load(entry)

    def boom(_args):                       # 走到这里就说明守卫没拦住
        raise AssertionError("不合规配置竟然进了驾驶循环")

    if hasattr(mod, "run"):
        monkeypatch.setattr(mod, "run", boom)
    monkeypatch.setattr(sys, "argv", [entry, "--lane-mode", "map"])
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "AGENTS.md" in err and "allow-legacy-map-lane" in err


@pytest.mark.parametrize("entry", ["m5_fsd_drive", "m5_shadow_drive"])
def test_no_strict_also_needs_the_escape_hatch(entry) -> None:
    mod = _load(entry)
    parser = mod.build_parser()
    args = parser.parse_args(["--no-strict", "--lane-mode", "sensor"])
    from beamng_autopilot.fsd_realism import lane_mode_is_fsd_compliant
    assert not lane_mode_is_fsd_compliant(args.lane_mode, args.strict)


def test_the_escape_hatch_is_recorded_as_a_warning(monkeypatch, capsys) -> None:
    """带逃生口时允许跑，但必须打印"不得用于验收"的告警。"""
    mod = _load("m5_fsd_drive")
    seen = {}

    def fake_run(args):
        seen.update(vars(args))
        return 0

    monkeypatch.setattr(mod, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["m5_fsd_drive", "--lane-mode", "map",
                                      "--allow-legacy-map-lane"])
    assert mod.main() == 0
    assert seen["lane_mode"] == "map"
    assert "不得用于 FSD 验收" in capsys.readouterr().err


def test_benchmark_defaults_mirror_the_entry_point() -> None:
    """benchmark 的 ``_DRIVE_ARG_DEFAULTS`` 必须与入口 argparse 一致。

    2026-10-06 实测过同类漂移：manifest 记 `lane_mode=map/strict=false`，
    实际跑的是 sensor+strict。这里把两份默认值直接对账。
    """
    entry = _load("m5_fsd_drive")
    bench = _load("m5_fsd_benchmark")
    args = vars(entry.build_parser().parse_args([]))
    shared = ("runtime", "attach", "seconds", "speed", "steps", "cam_w",
              "cam_h", "seg_model", "line_seg_model", "teleport", "out",
              "lane_mode", "strict", "traffic", "goal", "no_signal",
              "no_shadow", "vis")
    drift = {k: (bench._DRIVE_ARG_DEFAULTS.get(k), args.get(k)) for k in shared
             if bench._DRIVE_ARG_DEFAULTS.get(k) != args.get(k)}
    assert not drift, f"默认值漂移（benchmark, entry）：{drift}"


def test_every_benchmark_scenario_is_fsd_compliant() -> None:
    """注册表里的每个场景都必须落在合规路径上（含"吃默认值"的 free）。"""
    bench = _load("m5_fsd_benchmark")
    from beamng_autopilot.fsd_realism import lane_mode_is_fsd_compliant

    for name in bench.SCENARIOS:
        ns = bench.scenario_args(name, {}, Path("_eff"))
        assert lane_mode_is_fsd_compliant(ns.lane_mode, ns.strict), (
            f"场景 {name} 不是合规路径：lane_mode={ns.lane_mode} "
            f"strict={ns.strict}")
