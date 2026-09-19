"""Segmentation probability gate wiring (plan E1, opt-in in the head)."""

from __future__ import annotations

import numpy as np
import pytest

from beamng_autopilot.vision.heads import semantic as sem_mod
from beamng_autopilot.vision.heads.semantic import SemanticHead
from beamng_autopilot.vision.hydra import FrameContext
from beamng_autopilot.vision.seg_probs import SegProbabilityMaps

H, W = 40, 40


def _ctx(**kwargs) -> FrameContext:
    return FrameContext(frame_rgb=np.zeros((H, W, 3), np.uint8), cam=None,
                        pos=(0.0, 0.0, 0.0), heading=0.0, **kwargs)


class _ProbSegmenter:
    """Segmenter stub exposing both entry points."""

    def __init__(self, road_fp: bool = True) -> None:
        self.calls = 0
        self.prob_calls = 0
        self.road_fp = bool(road_fp)

    def _masks(self):
        road = np.ones((H, W), dtype=bool)
        line = np.zeros((H, W), dtype=bool)
        line[10:14, 10:12] = True          # a genuine marking
        if self.road_fp:
            line[20:24, 30:32] = True      # a false positive off the road
        return road, line

    def predict(self, frame):
        self.calls += 1
        return self._masks()

    def predict_with_probs(self, frame):
        self.prob_calls += 1
        road, line = self._masks()
        r = np.full((H, W), 0.9, dtype=np.float32)
        l = np.zeros((H, W), dtype=np.float32)
        l[10:14, 10:12] = 0.9              # the real marking: high prob
        if self.road_fp:
            # the false positive sits on a whole OFF-ROAD region: the
            # road-context max filter reaches 3 px, so a blob-sized dip
            # would be bridged by its own surroundings
            r[:, 24:] = 0.05
            l[20:24, 30:32] = 0.9
        maps = SegProbabilityMaps(line=l, road=r)
        return road, line, maps


class _PlainSegmenter:
    """A segmenter without the probability entry point (older stubs)."""

    def predict(self, frame):
        return (np.ones((H, W), dtype=bool),
                np.zeros((H, W), dtype=bool))


def _head(seg, monkeypatch, enabled: bool) -> SemanticHead:
    monkeypatch.setattr(sem_mod, "SEG_PROB_GATE_ENABLED", enabled)
    monkeypatch.setenv("BEAMNG_YELLOW_FUSION", "0")
    return SemanticHead(segmenter=seg, enable_evidence=False)


def test_gate_is_off_by_default() -> None:
    import importlib
    import os
    saved = os.environ.pop("BEAMNG_SEG_PROB_GATE", None)
    try:
        mod = importlib.reload(sem_mod)
        assert mod.SEG_PROB_GATE_ENABLED is False
    finally:
        if saved is not None:
            os.environ["BEAMNG_SEG_PROB_GATE"] = saved
        importlib.reload(sem_mod)      # leave the module as we found it


def test_disabled_gate_keeps_every_pixel(monkeypatch) -> None:
    seg = _ProbSegmenter()
    head = _head(seg, monkeypatch, enabled=False)
    out = head.run(_ctx())
    assert "seg_gate" not in out.meta
    assert seg.prob_calls == 0
    assert int(out.masks["line"].sum()) == 8 + 8      # both blobs survive


def test_enabled_gate_removes_the_off_road_false_positive(monkeypatch) -> None:
    seg = _ProbSegmenter()
    head = _head(seg, monkeypatch, enabled=True)
    out = head.run(_ctx())
    assert seg.prob_calls == 1
    assert seg.calls == 0, "one inference must serve both outputs"
    stats = out.meta["seg_gate"]
    assert stats["line_before"] == 16 and stats["line_road"] == 8
    line = out.masks["line"]
    assert int(line.sum()) == 8                        # only the real one
    assert line[10:14, 10:12].all()
    assert not line[20:24, 30:32].any()


def test_gate_only_ever_removes_pixels(monkeypatch) -> None:
    seg_off = _ProbSegmenter()
    off = _head(seg_off, monkeypatch, enabled=False).run(_ctx())
    seg_on = _ProbSegmenter()
    on = _head(seg_on, monkeypatch, enabled=True).run(_ctx())
    assert not (on.masks["line"] & ~off.masks["line"]).any()
    assert not (on.masks["road"] & ~off.masks["road"]).any()


def test_plain_segmenter_still_works_with_the_switch_on(monkeypatch) -> None:
    head = _head(_PlainSegmenter(), monkeypatch, enabled=True)
    out = head.run(_ctx())
    assert "seg_gate" not in out.meta      # nothing to gate with
    assert not out.masks["line"].any()


# ---------------------------------------------------------------------------
# E2: near/mid/far zoned thresholds in the head
# ---------------------------------------------------------------------------

class _ZonedSegmenter(_ProbSegmenter):
    """Line probability high everywhere; support only in the near band."""

    def predict_with_probs(self, frame):
        self.prob_calls += 1
        road, line = self._masks()
        r = np.full((H, W), 0.95, dtype=np.float32)
        l = np.zeros((H, W), dtype=np.float32)
        l[:, 10:12] = 0.95                 # a full-height marking
        maps = SegProbabilityMaps(line=l, road=r)
        return road, line, maps


def test_zoned_gate_is_off_unless_enabled(monkeypatch) -> None:
    seg = _ZonedSegmenter()
    head = _head(seg, monkeypatch, enabled=False)
    monkeypatch.setattr(sem_mod, "SEG_ZONES_ENABLED", False)
    out = head.run(_ctx())
    assert "seg_gate_zones" not in out.meta


def test_zoned_gate_reports_per_zone_stats(monkeypatch) -> None:
    seg = _ZonedSegmenter()
    head = _head(seg, monkeypatch, enabled=True)
    monkeypatch.setattr(sem_mod, "SEG_ZONES_ENABLED", True)
    out = head.run(_ctx())
    assert out.meta["seg_gate_mode"] == "zoned"
    zones = out.meta["seg_gate_zones"]["zones"]
    assert set(zones) == {"near", "mid", "far"}
    # the plan's intent shows through in the reported policy
    assert zones["near"]["line_min"] > zones["far"]["line_min"]
    assert zones["near"]["hold_s"] < zones["far"]["hold_s"]
    assert zones["far"]["history_required"] == 1
    assert out.meta["seg_gate_zones"]["line_after"] > 0


def test_zoned_gate_still_only_removes_pixels(monkeypatch) -> None:
    plain = _head(_ProbSegmenter(), monkeypatch, enabled=False).run(_ctx())
    monkeypatch.setattr(sem_mod, "SEG_ZONES_ENABLED", True)
    zoned = _head(_ZonedSegmenter(), monkeypatch, enabled=True).run(_ctx())
    assert not (zoned.masks["line"] & ~plain.masks["line"]).any()
