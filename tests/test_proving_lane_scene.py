# -*- coding: utf-8 -*-
"""试车场场景规格：proving_lane 必须先铺深色铺装再画线。

GridMap 地面是浅色混凝土，经典亮度臂会把整片地面当白漆，与学习掩码里的
两条完整线并成一个无效大块（2026-11-04 根因）。铺装 decal 让线成为画面
里唯一的亮细结构；本测试钉住规格与构建路径，防止回归成"只有线"。
"""
from pathlib import Path

import pytest


def _load_scenes_module():
    import importlib.util
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "mcs_test_proving", str(root / "scripts" / "m5_controlled_scenes.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_proving_lane_scene_has_pavement():
    mod = _load_scenes_module()
    scene = mod.SCENES["proving_lane"]
    assert float(scene["pave_width_m"]) >= 8.0
    assert float(scene["line_z_offset_m"]) > 0.0
    assert [ln[2] for ln in scene["lines"]] == ["left", "right"]
    src = (Path(__file__).resolve().parents[1]
           / "scripts" / "m5_controlled_scenes.py").read_text("utf-8")
    assert 'spec.get("pave_width_m")' in src
    assert "MAT_PAVE, _prid, _pnodes, _pw" in src
    assert 'spec.get("line_z_offset_m")' in src


def test_other_scenes_do_not_get_pavement():
    mod = _load_scenes_module()
    for name in ("known_line", "known_no_line", "occluded_line",
                 "slope_curve", "material_mix"):
        assert "pave_width_m" not in mod.SCENES[name], name
