"""`m5_tech_truth_points.py` 的纯逻辑测试（T16 Order 2）。

重点覆盖两个实测踩到的坑：
1. **采样方向**：road network 的 edge 顺序可能与车头方向相反——按数组顺序走
   会把真值点采到车后（曾导致 18 个点全部投影到相机后方，看着像标签错误）。
2. **车道边界**：`lanesLeft/lanesRight` 等分路面，车在左/右车道时的边界不同。

用假 conn（鸭子类型）喂合成道路，不连游戏。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load():
    spec = importlib.util.spec_from_file_location(
        "m5_tech_truth_points", ROOT / "scripts" / "m5_tech_truth_points.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["m5_tech_truth_points"] = mod
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
    """最小假连接器：只提供 build_truth_points 用到的接口。"""

    def __init__(self, roads, state):
        self.bng = _Bng(roads)
        self._state = state
        self.io_lock = __import__("threading").Lock()

    def get_state(self):
        return self._state


def _straight_road(n_rows=8, step=10.0, width=4.0):
    """沿 +x 的直路；``edges`` 数组顺序与 +x 一致。"""
    rows = []
    for k in range(n_rows):
        x = k * step
        rows.append({"middle": [x, 0.0, 1.0], "left": [x, width],
                     "right": [x, -width]})
    return {"roadA": {"edges": rows, "lanesLeft": 1, "lanesRight": 1,
                      "name": "A"}}


def test_nearest_edge_picks_the_closest_segment():
    loop = _load()
    roads = _straight_road()
    rid, meta, i0, t0, edges = loop._nearest_edge(roads, [12.0, 0.5])
    assert rid == "roadA"
    assert i0 == 1 and abs(t0 - 0.2) < 1e-6


def test_lane_boundaries_split_the_road():
    loop = _load()
    ll, lr, note = loop._lane_boundaries(
        {"lanesLeft": 1, "lanesRight": 1}, 4.0, -4.0, 1.0)
    assert (round(ll, 3), round(lr, 3)) == (4.0, 0.0) and not note
    ll2, lr2, _ = loop._lane_boundaries(
        {"lanesLeft": 1, "lanesRight": 1}, 4.0, -4.0, -1.0)
    assert (round(ll2, 3), round(lr2, 3)) == (0.0, -4.0)


def test_lane_boundaries_without_lane_counts_use_whole_road():
    loop = _load()
    ll, lr, note = loop._lane_boundaries({}, 4.0, -4.0, 0.0)
    assert (ll, lr) == (4.0, -4.0) and "一条车道" in note


def test_world_at_interpolates_lateral_offset():
    loop = _load()
    rows = _straight_road()["roadA"]["edges"]
    mid = loop._world_at(rows, 1, 0.5, 0.0, lat_left=4.0, lat_right=-4.0)
    left = loop._world_at(rows, 1, 0.5, 4.0, lat_left=4.0, lat_right=-4.0)
    assert [round(v, 2) for v in mid] == [15.0, 0.0, 1.0]
    assert [round(v, 2) for v in left] == [15.0, 4.0, 1.0]


def test_points_are_sampled_in_front_of_the_car_even_when_edges_are_reversed():
    """edge 数组方向与车头相反时，真值点仍必须全在车前方（实测踩到的坑）。"""
    loop = _load()
    rows = _straight_road()["roadA"]["edges"][::-1]        # 数组顺序反了
    roads = {"roadA": {"edges": rows, "lanesLeft": 1, "lanesRight": 1}}
    conn = _Conn(roads, _State([30.0, 0.0, 1.0], 0.0))     # 朝 +x
    blob = loop.build_truth_points(conn, span_m=20.0, n_samples=5,
                                   forward_skip_m=4.0)
    assert blob.get("error") is None, blob
    pts = blob["truth_points"]
    assert pts, "至少要采到点"
    fwd = [float(p["forward_m"]) for p in pts]
    assert min(fwd) > 0.0, f"有真值点落在车后：{fwd}"
    assert len({p["class"] for p in pts}) == 2, "线点与路面点都要有"


def test_points_refuse_when_sampling_cannot_stay_in_front():
    """采样方向无法保持在车前时必须报错，而不是输出无效几何。"""
    loop = _load()
    rows = _straight_road(n_rows=2, step=5.0)["roadA"]["edges"]
    roads = {"roadA": {"edges": rows, "lanesLeft": 1, "lanesRight": 1}}
    conn = _Conn(roads, _State([100.0, 0.0, 1.0], 0.0))    # 车离路很远
    blob = loop.build_truth_points(conn, span_m=20.0, n_samples=5,
                                   forward_skip_m=4.0)
    # 要么采到车前点、要么明确失败；不允许"有输出但点在车后"
    for p in blob.get("truth_points") or []:
        assert float(p["forward_m"]) > 0.0
