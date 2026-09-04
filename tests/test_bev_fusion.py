"""Offline tests for the multi-camera -> BEV feature fusion layer."""

from __future__ import annotations

import numpy as np
import pytest

from beamng_autopilot.bev_fusion import (
    BEVFeatureMap,
    CameraFeature,
    fuse_camera_features,
    fuse_front_frame_vector_space,
    project_mask_to_ego,
)
from beamng_autopilot.recording import FMAP_CHANNELS
from beamng_autopilot.vision.projection import CameraModel


def _cam() -> CameraModel:
    return CameraModel(np.array([0.0, 1.0, 1.4]),
                       np.array([0.0, 0.9999, -0.02]),
                       np.array([0.0, 0.02, 0.9999]), 65.0, 160, 120)


def test_bev_map_accumulates_and_probability() -> None:
    m = BEVFeatureMap(n=20, res=0.5)
    assert m.extent == 5.0
    feat = CameraFeature("front_main", "obstacle",
                         np.array([[2.0, 0.0, 0.0]]), confidence=0.9)
    m.accumulate(feat)
    p = m.get("obstacle")
    cell = m._cell(2.0, 0.0)
    assert p[cell] > 0.5
    assert m.sources["obstacle"][cell] >= 1


def test_bev_map_false_positive_fades() -> None:
    m = BEVFeatureMap(n=20, res=0.5)
    # a low-confidence vote should barely register
    m.accumulate(CameraFeature("r", "obstacle",
                               np.array([[1.0, 0.0, 0.0]]), confidence=0.4))
    cell = m._cell(1.0, 0.0)
    assert m.get("obstacle")[cell] < 0.5


def test_fuse_camera_features_two_cameras_agree() -> None:
    cam = _cam()
    # two cameras both see an obstacle at ego (3, 0)
    f1 = CameraFeature("front_main", "obstacle",
                       np.array([[3.0, 0.0, 0.0]]), 0.8)
    f2 = CameraFeature("pillar_left", "obstacle",
                       np.array([[3.0, 0.0, 0.0]]), 0.8)
    fmap = fuse_camera_features([f1, f2], n=20, res=0.5)
    cell = fmap._cell(3.0, 0.0)
    # two agreeing cameras push the probability well above either alone
    assert fmap.get("obstacle")[cell] > 0.7
    # a cell far away stays neutral (~0.5 or below)
    far = fmap._cell(-4.0, 0.0)
    assert fmap.get("obstacle")[far] <= 0.5


def test_fuse_attention_weights_scale_votes() -> None:
    fmap_w = fuse_camera_features(
        [CameraFeature("front_main", "obstacle",
                       np.array([[2.0, 0.0, 0.0]]), 0.8)],
        n=20, res=0.5, attention_weights={"front_main": 0.1})
    fmap = fuse_camera_features(
        [CameraFeature("front_main", "obstacle",
                       np.array([[2.0, 0.0, 0.0]]), 0.8)],
        n=20, res=0.5)
    cell = fmap._cell(2.0, 0.0)
    assert fmap.get("obstacle")[cell] >= fmap_w.get("obstacle")[cell]


def test_project_mask_returns_ego_points() -> None:
    cam = _cam()
    mask = np.zeros((120, 160), dtype=bool)
    mask[60:] = True
    points = project_mask_to_ego(mask, cam, np.array([0.0, 0.0, 0.0]),
                                 0.0, ground_z=0.0, step=8)
    assert len(points) == 1 and points[0].shape[1] == 3
    xs = points[0][:, 0]
    # projected road points should be ahead (positive x ego), not behind
    assert xs.min() > 0.0
    assert points[0][:, 2].min() >= 0.0


def test_fuse_front_frame_vector_space_channels() -> None:
    """Shadow-recorder vector space carries obstacle/drivable/lane/sign."""
    cam = _cam()
    h, w = cam.height, cam.width
    road = np.zeros((h, w), dtype=bool)
    road[h // 2:] = True
    line = np.zeros((h, w), dtype=bool)
    line[:, w // 2] = True
    hits = [type("Hit", (), {"x": 5.0, "y": 1.0})(),
            type("Hit", (), {"x": -3.0, "y": 2.0})()]
    fmap = fuse_front_frame_vector_space(
        {"road": road, "line": line}, cam,
        np.array([0.0, 0.0, 0.0]), 0.0, ground_z=0.0,
        obstacles=hits, n=60, res=0.5, step=8)
    assert fmap is not None
    # LiDAR hit -> obstacle evidence at its cell
    cell = fmap._cell(5.0, 1.0)
    assert fmap.get("obstacle")[cell] > 0.5
    # semantic road -> drivable evidence projected AHEAD, not behind
    drv = fmap.get("drivable")
    rows = np.argwhere(drv > 0.5)
    assert len(rows) > 0 and rows[:, 0].min() < 30
    # painted line -> lane evidence
    assert fmap.get("lane").max() > 0.5
    # no sign head yet: sign channel stays neutral
    assert float(np.abs(fmap.get("sign") - 0.5).max()) < 1e-5
    # recorder contract: (C, N, N) float32 channel stack
    stacked = np.stack([np.asarray(fmap.get(c), dtype=np.float32)
                        for c in FMAP_CHANNELS])
    assert stacked.shape == (4, 60, 60)
    assert stacked.dtype == np.float32


def test_fuse_front_frame_vector_space_none_when_empty() -> None:
    cam = _cam()
    fmap = fuse_front_frame_vector_space(
        {}, cam, np.array([0.0, 0.0, 0.0]), 0.0,
        ground_z=0.0, obstacles=(), n=60, res=0.5)
    assert fmap is None


def test_stamp_signal_bearing_places_lamp_by_pixel_bearing() -> None:
    from types import SimpleNamespace

    from beamng_autopilot.bev_fusion import stamp_signal_bearing

    cam = SimpleNamespace(fx=100.0, cx=80.0)
    # lamp RIGHT of the image centre (u = +0.4): ego y must be NEGATIVE
    fmap = BEVFeatureMap(n=40, res=0.5)   # extent 10 m
    stamp_signal_bearing(fmap, cam, (120.0, 20.0), confidence=0.9,
                         d_lo_m=6.0, d_hi_m=9.0, step_m=3.0)
    p = fmap.get("sign")
    right_cell = fmap._cell(6.0, -6.0 * 0.4)
    left_cell = fmap._cell(6.0, +6.0 * 0.4)
    assert right_cell is not None and p[right_cell] > 0.5
    assert left_cell is None or p[left_cell] <= 0.5
    # lamp straight ahead (u = 0): the band sits along the y=0 axis
    fmap2 = BEVFeatureMap(n=40, res=0.5)
    stamp_signal_bearing(fmap2, cam, (80.0, 20.0), confidence=0.9,
                         d_lo_m=6.0, d_hi_m=9.0, step_m=3.0)
    fwd_cell = fmap2._cell(6.0, 0.0)
    assert fwd_cell is not None and fmap2.get("sign")[fwd_cell] > 0.5
    # low confidence: the band must stay neutral (no dead-channel noise;
    # neutral probability is exactly sigmoid(0) = 0.5)
    fmap3 = BEVFeatureMap(n=40, res=0.5)
    stamp_signal_bearing(fmap3, cam, (120.0, 20.0), confidence=0.3,
                         d_lo_m=6.0, d_hi_m=9.0, step_m=3.0)
    assert float(fmap3.get("sign").max()) <= 0.5
