# -*- coding: utf-8 -*-
"""A1 分析脚本的离线数学验证：不依赖引擎，用真实 CameraModel + 合成渲染图。

场景几何与试车场一致：dir=+x（heading 0），横向左正（lat 正 = +y），
节点链 x=5..25、z=0。橙漆画在已知 lat 上，扫描应还原"渲染 lat - 节点 lat"。
"""
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, str(ROOT / rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


diag = _load("diag_a1_test_mod", "scripts/m5_proving_diag_a1.py")
exp = _load("export_a1_test_mod", "scripts/m5_proving_export.py")

FOV_DEG = 120
W, H = 1024, 800
ORANGE = (255, 128, 0)


def _camera():
    from beamng_autopilot.vision.projection import CameraModel
    return CameraModel(
        offset=np.array([0.0, 1.216, 1.386]),
        fwd_local=np.array([0.0, 0.99994, -0.0112]),
        up_local=np.array([0.0, 0.0112, 0.99994]),
        fov_deg=FOV_DEG, width=W, height=H)


def _nodes(lat_m: float) -> np.ndarray:
    xs = np.arange(5.0, 26.0, 1.0)
    return np.column_stack([xs, np.full(len(xs), lat_m), np.zeros(len(xs))])


def _paint_strip(img: np.ndarray, cam, lat_m: float, thickness: int = 5):
    """把一条橙漆带画进合成图（沿 x 5..25m、横向 lat_m）。"""
    pts = _nodes(lat_m)
    u, v, ok = cam.project(pts, np.array([0.0, 0.0, 0.0]), 0.0)
    for i in range(len(pts) - 1):
        if ok[i] and ok[i + 1] and np.isfinite(u[i]) and np.isfinite(u[i + 1]):
            cv2.line(img, (int(u[i]), int(v[i])), (int(u[i + 1]), int(v[i + 1])),
                     ORANGE, thickness)
    return img


def _paint_pts(img: np.ndarray) -> np.ndarray:
    mask = exp.paint_mask(img)
    ys, xs = np.nonzero(mask)
    return np.column_stack([xs, ys]).astype(float) if len(xs) else np.zeros((0, 2))


def test_scan_recovers_known_lateral_shift():
    """渲染线在 lat -3.5、节点在 -2.7 → 扫描应还原 -0.8m 偏移。"""
    cam = _camera()
    img = np.full((H, W, 3), 200, np.uint8)
    img = _paint_strip(img, cam, -3.5)
    offsets = diag.scan_line_offsets(
        cam, np.array([0.0, 0.0, 0.0]), 0.0, img.shape,
        _nodes(-2.7), _paint_pts(img))
    got = [o["offset_m"] for o in offsets if o["offset_m"] is not None]
    assert len(got) >= 10
    assert abs(float(np.median(got)) - (-0.8)) < 0.15


def test_scan_without_paint_reports_none_offsets():
    cam = _camera()
    img = np.full((H, W, 3), 200, np.uint8)
    offsets = diag.scan_line_offsets(
        cam, np.array([0.0, 0.0, 0.0]), 0.0, img.shape,
        _nodes(-2.7), _paint_pts(img))
    assert offsets
    assert all(o["offset_m"] is None and o["dist_px"] is None for o in offsets)


def test_phantom_frame_stats_silent_without_right_paint():
    """右线真值点投影：画面只有左线漆 → 覆盖率必须为 0（A1 对照①的判据）。"""
    cam = _camera()
    img = np.full((H, W, 3), 200, np.uint8)
    img = _paint_strip(img, cam, 0.9)  # 只画左线
    right_truth = [{"world": [float(x), -2.7, 0.0]}
                   for x in np.arange(5.0, 26.0, 1.7)]
    u, v = diag._phantom_project(cam, np.array([0.0, 0.0, 0.0]), 0.0,
                                 right_truth)
    rec = diag._phantom_frame_stats(u, v, img.shape, _paint_pts(img))
    assert rec["checked"] > 0
    assert rec["coverage"] == 0.0


def test_phantom_frame_stats_hits_when_right_painted():
    cam = _camera()
    img = np.full((H, W, 3), 200, np.uint8)
    img = _paint_strip(img, cam, 0.9)
    img = _paint_strip(img, cam, -2.7)  # 右线也在
    right_truth = [{"world": [float(x), -2.7, 0.0]}
                   for x in np.arange(5.0, 26.0, 1.7)]
    u, v = diag._phantom_project(cam, np.array([0.0, 0.0, 0.0]), 0.0,
                                 right_truth)
    rec = diag._phantom_frame_stats(u, v, img.shape, _paint_pts(img))
    assert rec["coverage"] >= 0.5


def test_line_lateral_dir_left_positive_for_plus_x_chain():
    nodes = _nodes(-2.7)
    lat2d = diag.line_lateral_dir(nodes)
    assert lat2d[1] > 0.9  # +x 方向链的横向（左正）应为 +y


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
