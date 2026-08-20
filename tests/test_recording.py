"""Offline tests for shadow-mode recording and the episode dataset."""

from __future__ import annotations

import numpy as np
import pytest

from beamng_autopilot.recording import (
    EpisodeDataset,
    ShadowFrame,
    ShadowRecorder,
)


def test_recorder_roundtrip(tmp_path) -> None:
    rec = ShadowRecorder(tmp_path, "test_roundtrip")
    rec.add(ShadowFrame(x=1.0, y=2.0, heading=0.1, speed=5.0,
                        throttle=0.4, brake=0.0, steer=0.2,
                        bev_raster=np.zeros((60, 60), dtype=np.float32),
                        drivable=np.ones((60, 60), dtype=np.uint8),
                        trajectory=np.array([[1.0, 2.0], [2.0, 2.0]]),
                        target_speed=8.0, lane_src="semantic",
                        cost=0.3, kind="arc"))
    rec.add(ShadowFrame(x=1.5, y=2.1, heading=0.11, speed=5.5,
                        throttle=0.35, brake=0.0, steer=0.1))
    out = rec.save()
    assert out is not None and out.exists()
    with np.load(out, allow_pickle=True) as z:
        assert int(z["version"]) == 1
        assert z["t"].shape[0] == 2
        assert z["steer"][0] == 0.2
        assert z["lane_src"][0] == "semantic"
        assert bool(z["trajectory_ok"][0]) and not bool(z["trajectory_ok"][1])
        assert z["bev"].shape == (2, 60, 60)


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