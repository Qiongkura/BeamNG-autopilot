"""`m5_controlled_scenes.py` 的纯逻辑测试（T16 §4.1/§4.2 路线 B）。

不连游戏：用假 conn 喂合成道路，验证
1. 五类场景规格齐备（有线/无线/遮挡/坡面曲线/材质混合）；
2. 线节点与真值点的几何（横向偏移方向、role、沿链等距）；
3. 站点规划沿道路前进且间隔正确；
4. `known_no_line` 不生成线、`occluded_line` 记录遮挡车、`material_mix` 有干扰带。
"""

from __future__ import annotations

import importlib.util
import sys
import threading
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load():
    spec = importlib.util.spec_from_file_location(
        "m5_controlled_scenes", ROOT / "scripts" / "m5_controlled_scenes.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["m5_controlled_scenes"] = mod
    spec.loader.exec_module(mod)
    return mod


class _State:
    def __init__(self, pos, heading):
        self.pos = np.asarray(pos, dtype=float)
        self.heading = float(heading)
        self.dir = np.array([np.cos(heading), np.sin(heading), 0.0])


class _Scenario:
    def __init__(self, roads):
        self._roads = roads

    def get_road_network(self, **_kw):
        return self._roads


class _Bng:
    def __init__(self, roads):
        self.scenario = _Scenario(roads)


class _Conn:
    def __init__(self, roads, state):
        self.bng = _Bng(roads)
        self._state = state
        self.io_lock = threading.Lock()

    def get_state(self):
        return self._state


def _road(n_rows=60, step=10.0, width=4.0):
    rows = [{"middle": [k * step, 0.0, 1.0], "left": [k * step, width],
             "right": [k * step, -width]} for k in range(n_rows)]
    return {"roadA": {"edges": rows, "lanesLeft": 1, "lanesRight": 1}}


def test_five_scene_classes_are_specified():
    m = _load()
    assert set(m.SCENES) == {"known_line", "known_no_line", "occluded_line",
                             "slope_curve", "material_mix"}
    assert m.SCENES["known_no_line"]["lines"] == []
    assert m.SCENES["occluded_line"].get("occluder") is True
    assert m.SCENES["material_mix"].get("gravel") is True
    assert len(m.SCENES["known_line"]["lines"]) == 2
    # 线材质必须是地图自己的贴花材质（路线 B：真实可见表面，不是自造方块）
    for spec in m.SCENES.values():
        for kind, _sign, _role in spec["lines"]:
            assert kind in m._MAT
            assert "italy_road_markings" in m._MAT[kind]


def test_line_nodes_follow_the_site_direction_and_lateral_sign():
    m = _load()
    site = {"mid": [0.0, 0.0, 1.0], "dir": [1.0, 0.0, 0.0],
            "half_width_m": 4.0}
    left = m._line_nodes(site, +m.LAT_LANE_HALF)      # 正 = 车体左
    assert left and all(n[0] > 1.0 for n in left), left   # 都在车前
    # dir=+x 时左向是 +y：正偏移 -> y 为正
    assert all(n[1] > 0 for n in left), left
    right = m._line_nodes(site, -m.LAT_LANE_HALF)
    assert all(n[1] < 0 for n in right), right
    # 沿链等距（1.0 m 步长：节点密到"节点连线 = 渲染贴花"）
    d = [abs(left[i + 1][0] - left[i][0]) for i in range(len(left) - 1)]
    assert all(abs(x - 1.0) < 1e-6 for x in d), d


def test_scene_roles_follow_the_lateral_sign_convention():
    """role 的横向符号：**正 = 左**。实测踩到反号 -> 翻转审计报 CAMERA_FLIP。"""
    m = _load()
    for name, spec in m.SCENES.items():
        for _kind, sign, role in spec["lines"]:
            assert (sign > 0) == (role == "left"), (name, sign, role)


def test_line_truth_points_are_class2_with_role():
    m = _load()
    nodes = [[0.0, 0.0, 1.0], [10.0, 0.0, 1.0]]
    pts = m._line_truth(nodes, "left", per_m=2.0)
    assert len(pts) == 5
    assert all(p["class"] == 2 and p["role"] == "left" for p in pts)
    assert abs(pts[0]["world"][0] - 0.0) < 1e-9
    assert abs(pts[-1]["world"][0] - 8.0) < 1e-9


def test_plan_sites_walks_along_the_road_with_spacing():
    m = _load()
    conn = _Conn(_road(), _State([0.0, 0.0, 1.0], 0.0))
    sites = m.plan_sites(conn, n_sites=5, spacing_m=120.0)
    assert len(sites) == 5
    xs = [s["mid"][0] for s in sites]
    assert xs == sorted(xs), xs
    assert abs(xs[0] - 0.0) < 1e-6
    gaps = [abs(xs[i + 1] - xs[i]) for i in range(4)]
    assert all(abs(g - 120.0) < 25.0 for g in gaps), gaps
    assert all(s["half_width_m"] > 0 and s["road_id"] == "roadA" for s in sites)


def test_plan_sites_raises_when_the_chain_is_too_short():
    m = _load()
    conn = _Conn(_road(n_rows=3, step=10.0), _State([0.0, 0.0, 1.0], 0.0))
    try:
        m.plan_sites(conn, n_sites=5, spacing_m=120.0)
    except RuntimeError as exc:
        assert "站点" in str(exc)
    else:                                                  # pragma: no cover
        raise AssertionError("道路链太短时必须报错，不能返回不足的站点")
