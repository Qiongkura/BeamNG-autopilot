"""One observed frame -> one segmentation -> one fused marking input."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from beamng_autopilot.vision.heads.semantic import SemanticHead
from beamng_autopilot.vision.hydra import FrameContext
from beamng_autopilot.vision.segmentation import Segmenter


def _ctx(**kwargs):
    return FrameContext(frame_rgb=np.zeros((40, 40, 3), np.uint8),
                        cam=None, pos=(0., 0., 0.), heading=0., **kwargs)


def _segmenter():
    seg = object.__new__(Segmenter)
    seg.calls = 0
    def predict(frame):
        seg.calls += 1
        return (np.ones(frame.shape[:2], bool),
                np.zeros(frame.shape[:2], bool))
    seg.predict = predict
    return seg


def test_semantic_infers_only_once():
    seg = _segmenter()
    out = SemanticHead(segmenter=seg, enable_evidence=False).run(_ctx())
    assert seg.calls == 1
    assert out.meta["markings"] == []


def test_fused_mask_reaches_both_marking_extractors(monkeypatch):
    from beamng_autopilot.vision import lanes
    seen = []
    def extract(mask, *args, **kwargs):
        seen.append(mask.copy())
        return []
    monkeypatch.setattr(lanes, "_mask_to_markings", extract)
    monkeypatch.setattr(lanes, "recover_dashed_boundaries", extract)
    monkeypatch.setenv("BEAMNG_DASHED_RECOVERY", "1")
    class Evidence:
        def fuse(self, line, *args, **kwargs):
            out = line.copy()
            out[10:25, 20:23] = True
            return out
    seg = _segmenter()
    head = SemanticHead(segmenter=seg)
    head._evidence = Evidence()
    out = head.run(_ctx())
    assert seg.calls == 1
    assert len(seen) == 2
    for mask in seen:
        np.testing.assert_array_equal(mask > 0, out.masks["line"])
    assert out.meta["line_pixels_raw"] == 0
    assert out.meta["line_pixels_added"] == 45


def test_explicit_empty_mask_does_not_reinfer():
    seg = _segmenter()
    ctx = _ctx()
    assert seg.detect_lines(ctx.frame_rgb, None, ctx.pos, ctx.heading,
                            line_mask=np.zeros((40, 40), bool)) == []
    assert seg.calls == 0


def test_legacy_detect_lines_still_infers_once():
    seg = _segmenter()
    ctx = _ctx()
    assert seg.detect_lines(ctx.frame_rgb, None, ctx.pos, ctx.heading) == []
    assert seg.calls == 1


def test_explicit_mask_must_match_image_shape():
    seg = _segmenter()
    ctx = _ctx()
    with pytest.raises(ValueError, match="shape"):
        seg.detect_lines(ctx.frame_rgb, None, ctx.pos, ctx.heading,
                         line_mask=np.zeros((20, 20), bool))


def test_head_uses_observation_timestamp_and_resets_history():
    class Evidence:
        def __init__(self):
            self.times = []
            self.reset_calls = 0
        def fuse(self, line, *args, now=None):
            self.times.append(now)
            return line
        def reset(self):
            self.reset_calls += 1
    seg = _segmenter()
    seg._prev_line = np.ones((40, 40), bool)
    head = SemanticHead(segmenter=seg)
    ev = head._evidence = Evidence()
    head.run(_ctx(timestamp=10.5))
    assert ev.times == [10.5]
    head.reset()
    assert ev.reset_calls == 1
    assert seg._prev_line is None


def test_failed_inference_cannot_republish_historical_line():
    seg = _segmenter()
    def fail(frame):
        raise RuntimeError("inference unavailable")
    seg.predict = fail
    def no_extract(*args, **kwargs):
        pytest.fail("failed inference must use the existing fallback only")
    seg.detect_lines = no_extract
    class Evidence:
        reset_calls = 0
        def fuse(self, line, *args, **kwargs):
            pytest.fail("failed inference must not revive history")
        def reset(self):
            self.reset_calls += 1
    class NoLanes:
        def detect(self, *args, **kwargs):
            return []
    head = SemanticHead(segmenter=seg, lane_detector=NoLanes())
    ev = head._evidence = Evidence()
    out = head.run(_ctx())
    assert not out.masks["line"].any()
    assert out.meta["markings"] == []
    assert ev.reset_calls == 1
    assert "inference unavailable" in out.meta["line_errors"]["predict"]


def test_nonfront_camera_does_not_extract_or_accumulate():
    seg = _segmenter()
    def fail(*args, **kwargs):
        pytest.fail("nonfront camera must not extract markings")
    seg.detect_lines = fail
    head = SemanticHead(segmenter=seg)
    out = head.run(_ctx(role="pillar_left"))
    assert seg.calls == 1
    assert out.meta["markings"] == []
    assert head._evidence is None


def test_evidence_failure_keeps_current_observation():
    seg = _segmenter()
    road = np.ones((40, 40), bool)
    raw = np.zeros((40, 40), bool)
    raw[10:25, 20:23] = True
    seg.predict = lambda frame: (road, raw)
    seg.detect_lines = lambda *args, **kwargs: []
    class BrokenEvidence:
        reset_calls = 0
        def fuse(self, *args, **kwargs):
            raise ValueError("bad pose")
        def reset(self):
            self.reset_calls += 1
    head = SemanticHead(segmenter=seg)
    ev = head._evidence = BrokenEvidence()
    out = head.run(_ctx())
    np.testing.assert_array_equal(out.masks["line"], raw)
    assert ev.reset_calls == 1
    assert out.meta["line_pixels_added"] == 0
    assert out.meta["line_errors"]["evidence"] == "bad pose"


def test_task_replay_resets_and_uses_recording_time(tmp_path, monkeypatch):
    import json
    from scripts import m5_seg_task_eval as replay
    from beamng_autopilot.vision.hydra import TaskOutput
    events = []
    class Head:
        def __init__(self, **kwargs):
            pass
        def reset(self):
            events.append("reset")
        def run(self, ctx):
            events.append(ctx.timestamp)
            return TaskOutput(meta={"markings": []})
    monkeypatch.setattr(replay, "Segmenter", lambda **kwargs: None)
    monkeypatch.setattr(replay, "SemanticHead", Head)
    episodes = []
    for n, times in enumerate(([10., 10.5], [1., 1.25])):
        path = tmp_path / f"episode_{n}.npz"
        np.savez(path, t=times, x=[0., 1.], y=[0., 0.],
                 heading=[0., 0.], rgb=np.zeros((2, 40, 40, 3), np.uint8),
                 meta=np.frombuffer(json.dumps({"cam_w": 40, "cam_h": 40})
                                    .encode(), dtype=np.uint8))
        episodes.append(path)
    report = replay.measure(None, episodes)
    assert report["frames"] == 4
    assert events == ["reset", 10., 10.5, "reset", 1., 1.25]
