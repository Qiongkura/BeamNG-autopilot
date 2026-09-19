"""Yellow paint candidates must be elongated paint ON the road mask.

Live east_coast 2026-09-19: the classic yellow detector lit up on
yellow-green dirt/grass (round blobs, off the pavement), a false blob
became the "centre line" and the centre-line policy walked the car off
the road (edge_over up to 5.9 m).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from beamng_autopilot.vision.segmentation import Segmenter


class _RoadSeg(Segmenter):
    """Segmenter stub: road = left half of the frame, no inference."""

    def __init__(self):
        self.route_is_dirt = False
        self.road = np.zeros((60, 80), dtype=bool)
        self.road[:, :40] = True

    def predict(self, frame_rgb):
        return self.road, np.zeros_like(self.road)


def _yellow_marking(pixels):
    xs = pixels[:, 0].astype(float) * 0.1
    ys = pixels[:, 1].astype(float) * 0.1
    return SimpleNamespace(
        world=np.column_stack([xs, ys, np.zeros(len(xs))]),
        pixels=pixels, kind="dashed", color="yellow", confidence=0.8)


def _patch_frame():
    frame = np.zeros((60, 80, 3), np.uint8)
    frame[5:15, 5:45] = (255, 200, 0)      # cv_yellow fires on the patch
    return frame


def _run(monkeypatch, markings):
    seg = _RoadSeg()
    monkeypatch.setattr(
        "beamng_autopilot.vision.lanes._mask_to_markings",
        lambda mask, color="white", *a, **k: (
            markings if color == "yellow" else []))
    out = seg.detect_lines(
        _patch_frame(), None, np.zeros(3), 0.0, ground_z=0.0,
        line_mask=np.zeros((60, 80), bool), road_mask=seg.road)
    return [m for m in out if getattr(m, "color", "") == "yellow"]


def test_round_yellow_blob_is_dropped(monkeypatch):
    """A round blob (dirt/grass in the frame) is not a paint line."""
    rng = np.random.default_rng(7)
    blob = _yellow_marking(np.column_stack([
        rng.integers(5, 25, 60), rng.integers(10, 30, 60)]))
    assert _run(monkeypatch, [blob]) == []


def test_elongated_but_off_road_yellow_is_dropped(monkeypatch):
    """An elongated stripe lying on non-road pixels is not paint."""
    stripe = _yellow_marking(np.column_stack([
        np.linspace(50, 70, 30).astype(int), np.full(30, 10)]))
    assert _run(monkeypatch, [stripe]) == []


def test_elongated_on_road_yellow_stripe_is_kept(monkeypatch):
    stripe = _yellow_marking(np.column_stack([
        np.linspace(5, 45, 30).astype(int), np.full(30, 10)]))
    kept = _run(monkeypatch, [stripe])
    assert len(kept) == 1 and kept[0].color == "yellow"
