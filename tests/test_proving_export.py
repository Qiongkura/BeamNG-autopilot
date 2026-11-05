# -*- coding: utf-8 -*-
"""试车场导出：几何真值/漆色/栅格化门禁（预注册 §3.3）。"""
import importlib.util
from pathlib import Path

import numpy as np


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
