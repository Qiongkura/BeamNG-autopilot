# -*- coding: utf-8 -*-
"""试车场导出：几何真值/漆色/栅格化门禁（预注册 §3.3）。"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _mod():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "mpc_export", str(root / "scripts" / "m5_proving_export.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _doc():
    return {"scene_type": "proving_lane",
            "capture": {"site": "S1", "edge_margin_m": 0.8},
            "scene": {"mid": [300.0, 700.0, 100.0],
                      "dir": [0.0, 1.0, 0.0],
                      "half_width_m": 3.5, "slope_pct": 0.0,
                      "curve_deg": 0.0}}


def test_site_geometry_has_two_parallel_rails():
    m = _mod()
    geom = m.site_geometry(_doc())
    assert [g["role"] for g in geom] == ["left", "right"]
    for g in geom:
        sep = np.linalg.norm(g["left"][0, :2] - g["right"][0, :2])
        assert abs(sep - m.LINE_WIDTH_M) < 1e-6
        # 线位：左线在道路中心左侧，右线在右侧
    assert geom[0]["center"][0, 0] < geom[1]["center"][0, 0]


def test_truth_points_are_class2_with_roles():
    m = _mod()
    pts = m.truth_points(m.site_geometry(_doc()))
    assert pts and all(p["class"] == 2 for p in pts)
    assert {p["role"] for p in pts} == {"left", "right"}


def test_paint_mask_accepts_orange_and_rejects_grey():
    m = _mod()
    img = np.zeros((20, 20, 3), np.uint8)
    img[5:10, 5:10] = [255, 128, 0]          # 橙
    img[12:16, 12:16] = [128, 128, 128]      # 灰
    mask = m.paint_mask(img)
    assert mask[7, 7] and not mask[13, 13]


def test_build_masks_with_fake_camera():
    m = _mod()
    geom = m.site_geometry(_doc())

    class FakeCam:
        def project(self, pts, pos, heading):
            pts = np.asarray(pts, float)
            u = 100.0 + (pts[:, 0] - pos[0]) * 30.0
            v = 120.0 - (pts[:, 1] - pos[1]) * 15.0
            return u, v, np.ones(len(pts), bool)

    rgb = np.zeros((240, 240, 3), np.uint8)
    line, road = m.build_masks(rgb, FakeCam(), np.array([300.0, 700.0, 100.0]),
                               0.0, geom)
    assert line.sum() > 0
    assert road.sum() >= line.sum()


def test_formal_export_is_blocked_before_touching_existing_data(tmp_path, monkeypatch):
    m = _mod()
    monkeypatch.setattr(m, "POOL_DIR", tmp_path)
    marker = tmp_path / "existing.npz"
    marker.write_bytes(b"evidence")
    with pytest.raises(RuntimeError, match="BLOCKED"):
        m.export_cell("S1", "noon_rain")
    assert marker.read_bytes() == b"evidence"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["existing.npz"]


def test_pose_duplicate_checks_all_accepted_and_wraps_angle():
    m = _mod()
    accepted = [([0.0, 0.0], np.pi - 0.005), ([3.0, 0.0], 0.0)]
    match = m.find_pose_duplicate([0.1, 0.0], -np.pi + 0.005, accepted)
    assert match is not None and match[0] == 0  # A -> B -> A, ±π
    assert match[1] == pytest.approx(0.1)
    assert match[2] == pytest.approx(0.01)
    assert m.find_pose_duplicate([1.0, 0.0], np.pi, accepted) is None
    assert m.find_pose_duplicate([0.0, 0.0], np.pi - 0.1, accepted) is None


@pytest.mark.parametrize("pos, heading", [([np.nan, 0], 0), ([0, np.inf], 0),
                                           ([0, 0], np.nan), ([1], 0)])
def test_pose_duplicate_rejects_invalid_input(pos, heading):
    with pytest.raises(ValueError):
        _mod().find_pose_duplicate(pos, heading, [])


def test_select_pose_candidates_audit_revisit_and_invalid():
    m = _mod()
    records = [
        {"source_id": "a", "pos": [0, 0], "heading": np.pi - .005},
        {"source_id": "b", "pos": [2, 0], "heading": 0},
        {"source_id": "revisit", "pos": [.1, 0], "heading": -np.pi + .005},
        {"source_id": "invalid", "pos": [np.nan, 0], "heading": 0},
        {"source_id": "rotation", "pos": [0, 0], "heading": 0},
    ]
    accepted, rejected = m.select_pose_candidates(records)
    assert [x["source_id"] for x in accepted] == ["a", "b", "rotation"]
    assert rejected[0]["matched_source_id"] == "a"
    assert rejected[0]["distance_m"] == pytest.approx(.1)
    assert rejected[0]["angle_rad"] == pytest.approx(.01)
    assert rejected[1] == {"source_id": "invalid", "reason": "invalid_pose"}
    with pytest.raises(ValueError, match="source_id"):
        m.select_pose_candidates([records[0], records[0]])


def test_pose_duplicate_threshold_boundaries_and_invalid_history():
    m = _mod()
    with pytest.raises(ValueError, match="threshold"):
        m.select_pose_candidates([], min_xy=-1)
    accepted = [([0, 0], 0)]
    assert m.find_pose_duplicate([0.5, 0], 0, accepted) is None
    assert m.find_pose_duplicate([0, 0], 0.03, accepted) is None
    with pytest.raises(ValueError):
        m.find_pose_duplicate([0, 0], 0, [([np.nan, 0], 0)])


def test_pose_repeat_thresholds_are_strict_enough_for_static_frames():
    """静止帧位姿差 0、雨滴帧位姿差 0：都必须落在阈值内被拒。

    2026-11-06 实测：S1/noon_clear 437 帧 437 个不同 SHA-1（光影/雨滴每帧
    微变），但独立位姿只有 1 个。只按哈希去重会把同一位置的静止画面当成
    30 个训练样本。
    """
    m = _mod()
    assert m.POSE_MIN_XY_M >= 0.5, m.POSE_MIN_XY_M
    assert m.POSE_MIN_HEADING_RAD >= 0.02, m.POSE_MIN_HEADING_RAD
    # 静止帧（0 m / 0 rad）必须被判为重复
    assert 0.0 < m.POSE_MIN_XY_M and 0.0 < m.POSE_MIN_HEADING_RAD
    # 运动学采集的默认步长 1.0 m 必须超过阈值（否则整批都被判重复）
    assert 1.0 > m.POSE_MIN_XY_M
