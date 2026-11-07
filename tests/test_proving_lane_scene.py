# -*- coding: utf-8 -*-
"""试车场场景规格：proving_lane 只有线，默认不铺装。

2026-11-04 实测（GridMap）：
- 浅色地面 + 默认经典白漆臂：地面纹理被当白漆，学习掩码的两条 30m 线被
  并成碎段 -> 配对只能镜像；
- 加 20m 深色铺装：画面更像路，但纹理亮度让线碎成 3–9m（pair 6/12）；
- 纯浅色地面 + BEAMNG_CV_WHITE=0：两条 30m+ solid、12/12 双侧 pair。
所以默认试车场不铺装；`pave_width_m` 只是可选机制（本测试钉住它仍可用）。
"""
from pathlib import Path


def _load_scenes_module():
    import importlib.util
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "mcs_test_proving", str(root / "scripts" / "m5_controlled_scenes.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_proving_lane_has_two_lines_and_no_pavement():
    mod = _load_scenes_module()
    scene = mod.SCENES["proving_lane"]
    assert [ln[2] for ln in scene["lines"]] == ["left", "right"]
    assert "pave_width_m" not in scene
    src = (Path(__file__).resolve().parents[1]
           / "scripts" / "m5_controlled_scenes.py").read_text("utf-8")
    # 可选铺装机制仍在（默认不启用）
    assert 'spec.get("pave_width_m")' in src
    assert "MAT_PAVE, _prid, _pnodes, _pw" in src


def test_other_scenes_do_not_get_pavement():
    mod = _load_scenes_module()
    for name in ("known_line", "known_no_line", "occluded_line",
                 "slope_curve", "material_mix"):
        assert "pave_width_m" not in mod.SCENES[name], name


def test_proving_lane_left_is_single_left_line_without_pavement():
    """A1 对照实验①的规格：只左线，其余几何/约定与 proving_lane 一致。"""
    mod = _load_scenes_module()
    scene = mod.SCENES["proving_lane_left"]
    assert [ln[2] for ln in scene["lines"]] == ["left"]
    assert scene["lines"][0][0] == "white"
    assert scene["lines"][0][1] == +1
    assert "pave_width_m" not in scene
    assert "occluder" not in scene
