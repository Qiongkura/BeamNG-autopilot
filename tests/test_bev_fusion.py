"""Offline tests for the multi-camera -> BEV feature fusion layer."""

from __future__ import annotations

import numpy as np
import pytest

from beamng_autopilot.bev_fusion import (
    BEVFeatureMap,
    CameraFeature,
    fuse_camera_features,
    project_mask_to_ego,
)
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