"""Training annotation provenance and collector isolation, without a game."""
from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from beamng_autopilot.labeling.tech_annotation import (
    LABEL_SOURCE, annotation_metadata, prepare_annotation_sample,
)
from beamng_autopilot_tech.annotations import to_label


def pair():
    rgb = np.full((4, 6, 4), 100, np.uint8)
    ann = np.zeros((4, 6, 4), np.uint8)
    ann[:, :, :3] = (128, 196, 255)
    ann[:, 2:4, :3] = (255, 196, 128)
    ann[:, :, 3] = 255
    return rgb, ann


def test_rgba_and_alternate_road_palette():
    _, ann = pair()
    lab = to_label(ann)
    assert np.all(lab[:, :2] == 1)
    assert np.all(lab[:, 2:4] == 2)


@pytest.mark.parametrize("ann", [None, np.zeros((2, 3)),
    np.zeros((0, 3, 3), np.uint8), np.zeros((2, 3, 3), np.float32)])
def test_invalid_annotation_fails_without_pseudo_labels(ann):
    with pytest.raises(ValueError):
        prepare_annotation_sample(pair()[0], ann, width=3, height=2)


def test_preserves_original_and_nearest_palette(tmp_path):
    rgb, ann = pair()
    sample, stats = prepare_annotation_sample(rgb, ann, width=3, height=2)
    np.testing.assert_array_equal(sample["annotation_raw"], ann)
    np.testing.assert_array_equal(sample["label"], [[1, 2, 1], [1, 2, 1]])
    assert sample["colour"].shape == (2, 3, 3)
    assert stats["line_pixels"] == 2
    assert not stats["needs_line_review"]
    assert annotation_metadata()["annotation_review_required"] is True
    np.savez_compressed(tmp_path / "sample.npz", **sample)
    with np.load(tmp_path / "sample.npz", allow_pickle=False) as saved:
        assert saved["label_source"].item() == LABEL_SOURCE
        np.testing.assert_array_equal(saved["annotation_raw"], ann)


def test_missing_lines_are_flagged_not_fabricated():
    rgb, ann = pair()
    ann[:, :, :3] = (12, 34, 56)
    sample, stats = prepare_annotation_sample(rgb, ann, width=3, height=2)
    assert not sample["label"].any()
    assert stats["needs_line_review"]
    assert stats["background_or_unmapped_pixels"] == 6


def test_mismatched_rgb_annotation_rejected():
    rgb, ann = pair()
    with pytest.raises(ValueError, match="matching"):
        prepare_annotation_sample(rgb[:2], ann, width=3, height=2)


@pytest.mark.parametrize("missing", [False, True])
def test_collector_records_annotation_and_cleans_up(tmp_path, monkeypatch, missing):
    from scripts import m5_collect_seg as collector
    import beamngpy.sensors
    events = []
    class AI:
        def set_mode(self, mode):
            events.append(mode)
        def set_speed(self, *args, **kwargs):
            pass
    class Connector:
        def __init__(self, **kwargs):
            self.io_lock = threading.RLock()
            self.bng = object()
            self.vehicle = SimpleNamespace(ai=AI())
        def open(self, **kwargs):
            pass
        def attach_vehicle(self, **kwargs):
            pass
        def get_state(self):
            return SimpleNamespace(pos=np.zeros(3), heading=0.)
        def close(self):
            events.append("closed")
    class Camera:
        def __init__(self, *args, **kwargs):
            assert kwargs["is_render_annotations"] is True
            assert kwargs["is_render_colours"] is True
        def poll(self):
            rgb, ann = pair()
            return {"colour": rgb, "annotation": None if missing else ann}
        def remove(self):
            events.append("removed")
    monkeypatch.setattr(collector, "BeamNGConnector", Connector)
    monkeypatch.setattr(beamngpy.sensors, "Camera", Camera)
    monkeypatch.setattr(collector.config, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(collector.sys, "argv", ["collect", "--runtime", "tech",
        "--frames", "1", "--run", "test", "--rate", "100000", "--no-step"])
    if missing:
        with pytest.raises(ValueError, match="missing"):
            collector.main()
    else:
        collector.main()
    assert events[-3:] == ["disabled", "removed", "closed"]
    output = tmp_path / "m5_seg" / "run_test"
    meta = json.loads((output / "meta.json").read_text(encoding="utf-8"))
    assert meta["label_source"] == LABEL_SOURCE
    assert meta["label_usage"] == "training_evaluation_only"
    assert len(meta["frames"]) == (0 if missing else 1)
    if not missing:
        with np.load(output / "frame_00000.npz") as d:
            assert d["label_source"].item() == LABEL_SOURCE
            assert "annotation_raw" in d.files
    # Collision must fail before connecting or touching the vehicle.
    events.clear()
    with pytest.raises(FileExistsError):
        collector.main()
    assert events == []
