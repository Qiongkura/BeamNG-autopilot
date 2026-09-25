"""空间隔离的判据（方案 W2 §7.3）：同地点重采与同曝光多视角都不能绕过隔离。

方案原文：「按空间路段和采集组隔离训练/开发/最终集。相邻位置的重复采集即使
source_id 不同，也不能绕过隔离；同曝光多视角必须同组。空间缓冲依据相机可见范围
制定并冻结，不能仅检查字节重复。」

本项目的实测依据（为什么必须做）：`diverse_straightstreet` 与 `ident_probe_straight`
首帧相距 **0 m**、`diverse_town` 与 `holdout_town` 相距 30 m —— 它们 source_id 不同，
整组隔离的字面实现会放过这种泄漏。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beamng_autopilot.experiments.manifest import FrameRecord  # noqa: E402
from beamng_autopilot.experiments.protocol import protocol_blob  # noqa: E402
from beamng_autopilot.experiments.spatial import (  # noqa: E402
    SPATIAL_BUFFER_M, distance_m, group_exposure_leak, spatial_conflicts,
)


def _rec(path, group, *, pos=(0.0, 0.0), exposure=None, view="front_main",
         source_id="") -> FrameRecord:
    return FrameRecord(path=path, run="r", view=view, group=group,
                       map_name="italy", source_id=source_id or group,
                       t_wall=None, exposure=exposure, content_sha16="x",
                       label_sha16="y", road_px=1, line_px=1, quality={},
                       pos=pos)


def test_the_buffer_is_frozen_in_the_protocol():
    """空间缓冲必须冻结并可复算：改它 = 改协议（哈希跟着变）。"""
    blob = protocol_blob()
    assert blob["spatial_buffer_m"] == SPATIAL_BUFFER_M == 50.0
    assert "camera" in blob["spatial_buffer_note"]


def test_same_place_from_different_source_ids_is_a_violation():
    """实测反例：不同 source_id 但相距 0–30 m 的两次采集不能算"隔离"。"""
    train = [_rec("t0", "italy/ring_a", pos=(100.0, 100.0))]
    dev = [_rec("d0", "italy/ring_b", pos=(100.0, 100.0))]     # 同一个点
    rep = spatial_conflicts(train, dev)
    assert [v["why"] for v in rep["violations"]] == ["within_buffer"], rep
    assert rep["violations"][0]["distance_m"] == 0.0
    # 30 m 也算（缓冲 50 m）
    rep2 = spatial_conflicts(train, [_rec("d1", "italy/ring_b",
                                          pos=(130.0, 100.0))])
    assert rep2["violations"] and rep2["violations"][0]["distance_m"] == 30.0
    # 超过缓冲就不算
    rep3 = spatial_conflicts(train, [_rec("d2", "italy/ring_b",
                                          pos=(100.0 + SPATIAL_BUFFER_M + 1,
                                               100.0))])
    assert rep3["violations"] == [], rep3
    assert rep3["min_distance_m"] == SPATIAL_BUFFER_M + 1


def test_same_group_is_reported_even_without_positions():
    train = [_rec("t0", "italy/ring_a", pos=None)]
    dev = [_rec("d0", "italy/ring_a", pos=None)]
    rep = spatial_conflicts(train, dev)
    assert rep["violations"][0]["why"] == "same_group", rep
    # 同组命中时直接记违规、不去算距离（缺位置计数因此为 0）；缺位置只在
    # 需要按距离判定时才计数（见下面那条测试）
    assert rep["n_missing_position"] == 0


def test_the_same_exposure_across_views_must_not_be_split():
    """同曝光多视角被拆到训练/开发 = 同一瞬间两边都有（方案点名）。"""
    # 同一次采集（同 source_id）的另一个视角被拆到"另一组"——典型成因是身份缺失
    # 导致组键退化成 dir/<名字>。此时同组检查抓不到，必须靠曝光检查兜住。
    train = [_rec("t0", "italy/ring_a", exposure=7, view="front_main",
                  source_id="ring_a", pos=(500.0, 500.0))]
    dev = [_rec("d0", "dir/front_main", exposure=7, view="pillar_left",
                source_id="ring_a", pos=(500.0, 500.0))]
    rep = spatial_conflicts(train, dev)
    whys = [v["why"] for v in rep["violations"]]
    assert whys == ["same_exposure_cross_view"], rep
    # 一对帧只报一条（先命中先算）：同曝光已经足以判泄漏，不必再报距离
    # 同一次采集里，同一 exposure 落在不同组也要报（同曝光必须同组）
    leak = group_exposure_leak([
        _rec("a", "italy/ring_a", exposure=7, view="front_main",
             source_id="ring_a"),
        _rec("b", "dir/pillar_left", exposure=7, view="pillar_left",
             source_id="ring_a")])
    assert leak == [{"source_id": "ring_a", "exposure": 7,
                     "groups": ["dir/pillar_left", "italy/ring_a"]}], leak
    # 同一组里的多视角是正常情况（必须同组 -> 不报）
    assert group_exposure_leak([
        _rec("a", "italy/ring_a", exposure=7, view="front_main",
             source_id="ring_a"),
        _rec("b", "italy/ring_a", exposure=7, view="pillar_left",
             source_id="ring_a")]) == []


def test_missing_positions_are_not_guessed():
    assert distance_m(None, (1.0, 2.0)) is None
    assert distance_m((1.0, 2.0), None) is None
    assert distance_m([1.0], [1.0, 2.0]) is None
    rep = spatial_conflicts([_rec("t", "italy/a", pos=None)],
                            [_rec("d", "italy/b", pos=None)])
    assert rep["violations"] == []
    assert rep["min_distance_m"] is None
    assert rep["n_missing_position"] == 1
    assert rep["n_pairs_checked"] == 1


def test_cross_map_pairs_are_not_compared_by_coordinates():
    """跨地图不比坐标（实测：两张地图的默认出生点都在原点附近）。

    west_coast_usa 首帧 (-0.03,-0.01)、italy gm_walk 首帧 (-0.03,0.003)——
    按距离算是"相距 0.016 m 的同地点重采"，纯属假冲突；而同一地图内的
    0 m 重采必须继续被抓到。
    """
    from beamng_autopilot.experiments.spatial import spatial_conflicts

    class R:
        def __init__(self, path, group, map_name, sid, pos):
            self.path, self.group, self.map_name = path, group, map_name
            self.source_id, self.exposure, self.view = sid, 0, "front_main"
            self.pos = pos

    tr = [R("a", "italy/ring_a", "italy", "ring_a", (0.0, 0.0))]
    dv = [R("b", "west_coast_usa/ring_b", "west_coast_usa", "ring_b",
            (0.0, 0.0))]
    got = spatial_conflicts(tr, dv)
    assert got["violations"] == [], got
    assert got["n_pairs_skipped_other_map"] == 1, got
    # 同一地图内的 0 m 重采仍然要报
    dv2 = [R("c", "italy/ring_c", "italy", "ring_c", (0.0, 0.0))]
    got2 = spatial_conflicts(tr, dv2)
    assert got2["violations"] and got2["violations"][0]["why"] == "within_buffer"


def test_group_spread_reveals_a_near_duplicate_collection():
    """覆盖范围：30 帧 × 2 m 步长却只走了 13 m 的采集是近重复集。

    实测依据：west_coast_usa 那次采集 x 跨度 6.5 m、y 跨度 12.3 m（车辆几乎没动），
    而身份/计数审计全过（120 帧、四个视角各 30）。帧数不等于独立样本。
    """
    from beamng_autopilot.experiments.spatial import group_spread

    class R:
        def __init__(self, path, group, pos):
            self.path, self.group, self.map_name = path, group, "italy"
            self.source_id, self.exposure, self.view = "ring_a", 0, "front_main"
            self.pos = pos

    near_dup = [R(f"f{i}", "italy/ring_a", (i * 0.43, 0.0)) for i in range(30)]
    got = group_spread(near_dup, expect_step_m=2.0)
    assert len(got) == 1 and got[0]["step_ratio"] < 0.5, got
    assert got[0]["extent_m"] < 15, got
    # 正常采集（间距接近期望步长）不该被判成近重复
    normal = [R(f"g{i}", "italy/ring_b", (i * 2.0, 0.0)) for i in range(30)]
    got2 = group_spread(normal, expect_step_m=2.0)
    assert 0.9 < got2[0]["step_ratio"] < 1.1, got2
    assert got2[0]["coverage_ratio"] > 0.9, got2
    # 间距正常但路径在小范围折返：覆盖比会很低（实测 west_coast_usa 那次：
    # 间距 1.997 m≈期望，跨度只有 13.9 m，覆盖比 0.058）——两件事分开报
    loop = [R(f"l{i}", "italy/ring_d",
              (float(i % 4) * 2.0, float(i // 4) * 2.0)) for i in range(30)]
    got4 = group_spread(loop, expect_step_m=2.0)
    assert got4[0]["step_ratio"] > 0.5, got4
    assert got4[0]["coverage_ratio"] < 0.3, got4
    # 位姿缺失不参与（不猜），且要能看出"没有可定位的帧"
    got3 = group_spread([R("x", "italy/ring_c", None)])
    assert got3[0]["extent_m"] is None and "unknown" in got3[0]["note"]
