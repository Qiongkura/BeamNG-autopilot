"""Deterministic evidence contracts; synthetic fixtures are not driving results."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from beamng_autopilot.experiments import range_replay as rr
from beamng_autopilot.perception import Obstacle
from beamng_autopilot.runtime import RangeSample
from beamng_autopilot_tech.providers import _RawRange, TechRangeProvider


@pytest.fixture()
def capture(tmp_path, monkeypatch):
    monkeypatch.setattr(rr, "source_snapshot", lambda root: {"complete": True, "sha256": "fixture-source"})
    box = Obstacle(10., 2., .5, .7, axis=np.array([1., 0.]),
                   velocity=np.array([3., 0.]), vehicle_id="fixture-car")
    payload = _RawRange(cloud=np.arange(24, dtype=np.float64).reshape(8, 3),
                        lua_obstacles=[box], ray_hits=[(9., .4)], heading=.1, radius=55.,
                        meta={"source_poll_started_t": 100., "source_errors": {},
                              "finite_cloud_points": np.int64(8)})
    directory = tmp_path / "capture"
    record = rr.write_replay(directory, payload, pos=[1., 2., 3.],
                              ego_half_len=2.2, ego_half_width=.9)
    return directory, payload, record


def test_capture_roundtrip_preserves_cloud_boxes_source_and_numpy_fields(capture):
    directory, payload, record = capture
    loaded, sample, manifest = rr.load_replay(directory)
    np.testing.assert_array_equal(loaded.cloud, payload.cloud)
    np.testing.assert_array_equal(loaded.lua_obstacles[0].axis, [1., 0.])
    np.testing.assert_array_equal(loaded.lua_obstacles[0].velocity, [3., 0.])
    assert loaded.lua_obstacles[0].vehicle_id == "fixture-car"
    assert loaded.ray_hits == payload.ray_hits
    assert loaded.meta == payload.meta and sample["pos"] == [1., 2., 3.]
    assert record["status"] == "captured" and manifest["complete"] is True
    assert record["manifest_sha256"] == rr._sha(directory / "manifest.json")


@pytest.mark.parametrize("file", ["cloud.npy", "sample.json"])
def test_corruption_rejects_before_use(capture, file):
    directory, _payload, _record = capture
    with (directory / file).open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="digest mismatch"):
        rr.load_replay(directory)


def test_existing_capture_cannot_be_overwritten(capture):
    directory, payload, record = capture
    with pytest.raises(FileExistsError):
        rr.write_replay(directory, payload, pos=[1, 2, 3], ego_half_len=2.2, ego_half_width=.9)
    assert rr._sha(directory / "manifest.json") == record["manifest_sha256"]


def test_incomplete_write_has_no_complete_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(rr, "source_snapshot", lambda root: {"complete": True, "sha256": "fixture"})
    def fail(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(rr.np, "save", fail)
    directory = tmp_path / "partial"
    with pytest.raises(OSError, match="disk full"):
        rr.write_replay(directory, _RawRange(cloud=np.zeros((4, 3)), radius=55.),
                         pos=[0, 0, 0], ego_half_len=2.2, ego_half_width=.9)
    assert directory.exists() and not (directory / "manifest.json").exists()
    with pytest.raises(FileNotFoundError):
        rr.load_replay(directory)


@pytest.mark.parametrize("mutation", ["cloud_nan", "meta_nan", "pos_shape", "zero_extent", "bad_hits"])
def test_invalid_capture_does_not_publish_manifest(capture, mutation):
    directory, payload, _record = capture
    target = directory.parent / "invalid"
    pos, half = [1, 2, 3], 2.2
    if mutation == "cloud_nan":
        payload.cloud[0, 0] = np.nan
    elif mutation == "meta_nan":
        payload.meta["bad"] = np.array([np.nan])
    elif mutation == "pos_shape":
        pos = [1, 2]
    elif mutation == "zero_extent":
        half = 0.
    elif mutation == "bad_hits":
        payload.ray_hits = [(1, 2, 3)]
    with pytest.raises(ValueError):
        rr.write_replay(target, payload, pos=pos, ego_half_len=half, ego_half_width=.9)
    assert not target.exists()


def _reference(tmp_path):
    path = tmp_path / "reference.py"
    path.write_text("class TechRangeProvider:\n"
                    "    def process(self, payload, pos):\n"
                    "        return RangeSample(obstacles=list(payload.lua_obstacles),\n"
                    "                           ray_hits=list(payload.ray_hits))\n", encoding="utf-8")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_comparison_checks_every_output_and_keeps_raw_round_samples(capture, tmp_path, monkeypatch):
    directory, _payload, _record = capture
    reference, sha = _reference(tmp_path)
    calls = []
    def process(self, payload, pos):
        assert not getattr(self._lidar_tracker, "_tracks", [])
        assert payload.lua_obstacles[0].x == 10.
        calls.append(pos)
        result = RangeSample(obstacles=list(payload.lua_obstacles), ray_hits=list(payload.ray_hits))
        payload.lua_obstacles = []  # Must not contaminate another call.
        return result
    monkeypatch.setattr(TechRangeProvider, "process", process)
    report = rr.compare_replay(directory, reference_source=reference, reference_sha256=sha,
                                rounds=2, iterations=2, warmup=1)
    assert report["outputs_equal"] and report["compared_calls"] == 12
    assert len(calls) == 6
    assert [r["order"] for r in report["rounds"]] == [["reference", "current"], ["current", "reference"]]
    for mode in ("reference", "current"):
        assert all(len(r["timings_ms"][mode]) == 2 for r in report["rounds"])
        assert report["best_p95_ms"][mode] == min(r["p95_ms"][mode] for r in report["rounds"])


def test_difference_or_unpinned_reference_cannot_be_called_equivalent(capture, tmp_path, monkeypatch):
    directory, _payload, _record = capture
    reference, sha = _reference(tmp_path)
    with pytest.raises(ValueError, match="reference source hash"):
        rr.compare_replay(directory, reference_source=reference, reference_sha256="wrong", iterations=1, warmup=0)
    monkeypatch.setattr(TechRangeProvider, "process", lambda *a, **k: RangeSample())
    with pytest.raises(ValueError, match="geometry differs"):
        rr.compare_replay(directory, reference_source=reference, reference_sha256=sha, iterations=1, warmup=0)


def test_replay_requires_current_source_and_rejects_source_change(capture, tmp_path, monkeypatch):
    directory, _payload, _record = capture
    reference, sha = _reference(tmp_path)
    monkeypatch.setattr(rr, "source_snapshot", lambda root: {"complete": True, "sha256": "other"})
    with pytest.raises(ValueError, match="capture/current source differs"):
        rr.compare_replay(directory, reference_source=reference, reference_sha256=sha)
    sources = iter([{"complete": True, "sha256": "fixture-source"},
                    {"complete": True, "sha256": "other"}])
    monkeypatch.setattr(rr, "source_snapshot", lambda root: next(sources))
    monkeypatch.setattr(TechRangeProvider, "process", lambda self, p, pos:
                        RangeSample(obstacles=p.lua_obstacles, ray_hits=p.ray_hits))
    with pytest.raises(ValueError, match="source changed during"):
        rr.compare_replay(directory, reference_source=reference, reference_sha256=sha, iterations=1, warmup=0)
