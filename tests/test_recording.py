"""Offline tests for shadow-mode recording and the episode dataset."""

from __future__ import annotations

import json

import numpy as np
import pytest

from beamng_autopilot.recording import (
    EpisodeDataset,
    ShadowFrame,
    ShadowRecorder,
)


def test_recorder_roundtrip(tmp_path) -> None:
    rec = ShadowRecorder(tmp_path, "test_roundtrip")
    rgb = np.zeros((8, 16, 3), dtype=np.uint8)
    lab = np.zeros((8, 16), dtype=np.uint8)
    lab[2:6, 4:12] = 1
    lab[3:5, 6:10] = 2
    fmap = np.zeros((4, 60, 60), dtype=np.float32)
    fmap[0, 10:20, 20:30] = 1.0
    fmap[2, :, 30] = 0.8
    rec.add(ShadowFrame(x=1.0, y=2.0, heading=0.1, speed=5.0,
                        throttle=0.4, brake=0.0, steer=0.2,
                        bev_raster=np.zeros((60, 60), dtype=np.float32),
                        drivable=np.ones((60, 60), dtype=np.uint8),
                        fmap=fmap,
                        trajectory=np.array([[1.0, 2.0], [2.0, 2.0]]),
                        target_speed=8.0, lane_src="semantic",
                        cost=0.3, kind="arc", rgb=rgb, label=lab,
                        quality=0.9))
    rec.add(ShadowFrame(x=1.5, y=2.1, heading=0.11, speed=5.5,
                        throttle=0.35, brake=0.0, steer=0.1, quality=0.2))
    out = rec.save()
    assert out is not None and out.exists()
    with np.load(out, allow_pickle=True) as z:
        assert int(z["version"]) == 3
        assert z["t"].shape[0] == 2
        assert z["steer"][0] == 0.2
        assert z["lane_src"][0] == "semantic"
        assert bool(z["trajectory_ok"][0]) and not bool(z["trajectory_ok"][1])
        assert z["bev"].shape == (2, 60, 60)
        assert z["fmap"].shape == (2, 4, 60, 60)
        assert abs(float(z["fmap"][0, 2, :, 30].max()) - 0.8) < 1e-5
        assert z["rgb"].shape == (2, 8, 16, 3)
        assert z["label"].shape == (2, 8, 16)
        assert int((z["label"][0] == 2).sum()) == 8
        assert abs(float(z["quality"][0]) - 0.9) < 1e-5
        assert abs(float(z["quality"][1]) - 0.2) < 1e-5
        _meta = json.loads(np.asarray(z["meta"]).item().decode("utf-8"))
        assert _meta["fmap_channels"] == 4
        assert _meta["episode_version"] == 3


def test_recorder_empty_save_returns_none(tmp_path) -> None:
    rec = ShadowRecorder(tmp_path, "empty")
    assert rec.save() is None


def test_episode_dataset_iterates(tmp_path) -> None:
    rec = ShadowRecorder(tmp_path, "ds")
    for i in range(3):
        rec.add(ShadowFrame(x=i, y=0.0, steer=0.1 * i,
                            bev_raster=np.zeros((60, 60), np.float32)))
    out = rec.save()
    ds = EpisodeDataset([out])
    assert len(ds) == 3
    bev, action = ds[1]
    assert tuple(bev.shape) == (60, 60)
    assert abs(float(action[0]) - 0.1) < 1e-6


def test_episode_dataset_missing_file_ok(tmp_path) -> None:
    ds = EpisodeDataset([tmp_path / "nope.npz"])
    assert len(ds) == 0


def test_episode_dataset_multimodal(tmp_path) -> None:
    rec = ShadowRecorder(tmp_path, "mm")
    rgb = np.full((6, 10, 3), 7, dtype=np.uint8)
    lab = np.full((6, 10), 1, dtype=np.uint8)
    lab[:, 4:6] = 2
    for i in range(2):
        rec.add(ShadowFrame(x=i, y=0.0, steer=0.1 * i,
                            bev_raster=np.zeros((60, 60), np.float32),
                            rgb=rgb, label=lab, quality=0.8))
    out = rec.save()
    ds = EpisodeDataset([out], modalities=("bev", "rgb", "label"))
    assert len(ds) == 2
    (bev, img, seg), action = ds[1]
    assert tuple(bev.shape) == (60, 60)
    assert tuple(img.shape) == (6, 10, 3)
    assert int(img[0, 0, 0]) == 7
    assert tuple(seg.shape) == (6, 10)
    assert int((seg[0] == 2).sum()) == 2
    assert abs(float(action[0]) - 0.1) < 1e-6


def test_episode_dataset_quality_gate(tmp_path) -> None:
    rec = ShadowRecorder(tmp_path, "q")
    for i in range(4):
        rec.add(ShadowFrame(x=i, y=0.0, steer=0.1 * i,
                            bev_raster=np.zeros((60, 60), np.float32),
                            quality=0.9 if i % 2 == 0 else 0.1))
    out = rec.save()
    ds = EpisodeDataset([out], min_quality=0.5)
    assert len(ds) == 2
    bev, action = ds[1]
    assert abs(float(action[0]) - 0.2) < 1e-6
