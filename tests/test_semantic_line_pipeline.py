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


def test_nonempty_line_mask_uses_classic_fallback_when_extractor_empty():
    seg = _segmenter()
    seg.predict = lambda frame: (
        np.ones(frame.shape[:2], bool),
        np.ones(frame.shape[:2], bool),
    )
    seg.detect_lines = lambda *args, **kwargs: []

    class Classic:
        def detect(self, *args, **kwargs):
            return [("classic-short-paint", 1.0)]

    out = SemanticHead(segmenter=seg, lane_detector=Classic(),
                       enable_evidence=False).run(_ctx())
    assert out.meta["markings"] == [("classic-short-paint", 1.0)]


def test_classic_unknown_fragment_is_promoted_only_with_learned_line():
    from beamng_autopilot.vision.lanes import LaneMarking
    seg = _segmenter()
    seg.predict = lambda frame: (
        np.ones(frame.shape[:2], bool),
        np.ones(frame.shape[:2], bool),
    )
    seg.detect_lines = lambda *args, **kwargs: []
    world = np.column_stack([np.linspace(2.0, 7.0, 5),
                             np.full(5, 2.0)])
    marking = LaneMarking(world=world, pixels=world.copy(),
                          kind="unknown", confidence=0.7)

    class Classic:
        def detect(self, *args, **kwargs):
            return [marking]

    out = SemanticHead(segmenter=seg, lane_detector=Classic(),
                       enable_evidence=False).run(_ctx())
    assert out.meta["markings"][0].kind == "thin"


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


class _Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def _timed_segmenter(monkeypatch, clock):
    import torch
    from beamng_autopilot.vision import segmentation as module

    monkeypatch.setattr(module.time, "perf_counter", clock)
    original_resize = module.cv2.resize

    def resize(*args, **kwargs):
        clock.advance(0.001)
        return original_resize(*args, **kwargs)

    monkeypatch.setattr(module.cv2, "resize", resize)
    seg = object.__new__(Segmenter)
    seg.device, seg.half = "cpu", False
    seg._road_idx, seg._line_idx = 1, 2
    seg.calls = 0

    def model(x):
        seg.calls += 1
        clock.advance(0.005)
        return torch.zeros((1, 3, 4, 4))

    def postprocess(frame, road, line):
        clock.advance(0.003)
        return road, line

    seg.model = model
    seg._postprocess = postprocess
    return seg


def test_segmenter_times_one_inference_through_mask_materialization(monkeypatch):
    clock = _Clock()
    seg = _timed_segmenter(monkeypatch, clock)
    road, line = seg.predict(_ctx().frame_rgb)
    timing = dict(seg.last_timing_ms)
    assert seg.calls == 1
    assert not road.any() and not line.any()
    assert timing["preprocess"] == pytest.approx(1.0)
    assert timing["inference_decode"] == pytest.approx(6.0)
    assert timing["postprocess"] == pytest.approx(3.0)
    assert timing["probabilities"] is None
    assert timing["total"] == pytest.approx(10.0)
    seg.predict(_ctx().frame_rgb)
    assert timing == seg.last_timing_ms
    assert timing is not seg.last_timing_ms


def test_probability_timing_reuses_the_same_logits(monkeypatch):
    clock = _Clock()
    seg = _timed_segmenter(monkeypatch, clock)
    seen = []

    def probabilities(frame, *, _logits=None):
        assert _logits is not None
        seen.append(_logits)
        clock.advance(0.004)
        return "maps", None, None

    seg.predict_proba = probabilities
    road, line, maps = seg.predict_with_probs(_ctx().frame_rgb)
    assert seg.calls == 1 and len(seen) == 1 and maps == "maps"
    assert seg.last_timing_ms["probabilities"] == pytest.approx(4.0)
    assert seg.last_timing_ms["total"] == pytest.approx(14.0)


def test_failed_inference_retains_timings_without_old_postprocess(monkeypatch):
    clock = _Clock()
    seg = _timed_segmenter(monkeypatch, clock)
    seg.predict(_ctx().frame_rgb)

    def fail(x):
        clock.advance(0.008)
        raise RuntimeError("model failed")

    seg.model = fail
    with pytest.raises(RuntimeError, match="model failed"):
        seg.predict(_ctx().frame_rgb)
    assert seg.last_timing_ms["preprocess"] == pytest.approx(1.0)
    assert seg.last_timing_ms["inference_decode"] == pytest.approx(8.0)
    assert seg.last_timing_ms["postprocess"] is None
    assert seg.last_timing_ms["probabilities"] is None
    assert seg.last_timing_ms["total"] == pytest.approx(9.0)


def test_head_timing_keeps_skipped_stages_unknown_and_copies_per_call(monkeypatch):
    from beamng_autopilot.vision.heads import semantic as module

    clock = _Clock()
    seg = _timed_segmenter(monkeypatch, clock)
    monkeypatch.setattr(module, "SEG_PROB_GATE_ENABLED", False)
    monkeypatch.setattr(module, "SEG_ZONES_ENABLED", False)
    monkeypatch.setattr(module, "MARK_CLASS_ENABLED", False)
    monkeypatch.setenv("BEAMNG_YELLOW_FUSION", "0")
    head = SemanticHead(segmenter=seg, enable_evidence=False)
    out = head.run(_ctx(role="pillar_left"))
    timing = out.meta["semantic_ms"]
    assert seg.calls == 1
    assert timing["prediction"] == pytest.approx(10.0)
    assert timing["total"] == pytest.approx(10.0)
    for name in ("yellow", "probability_gate", "evidence", "markings", "classification"):
        assert timing[name] is None
    details = out.meta["segmentation_ms"]
    assert details["line"] is None
    assert details["road"] == seg.last_timing_ms
    assert details["road"] is not seg.last_timing_ms
    seg.last_timing_ms["total"] = -1.0
    assert details["road"]["total"] == pytest.approx(10.0)


def test_head_does_not_republish_old_segmenter_timing(monkeypatch):
    monkeypatch.setenv("BEAMNG_YELLOW_FUSION", "0")
    seg = _segmenter()
    seg.last_timing_ms = {"total": 999.0}
    out = SemanticHead(segmenter=seg, enable_evidence=False).run(
        _ctx(role="pillar_left"))
    assert out.meta["segmentation_ms"] == {"road": None, "line": None}


class TestLineCandidateGate:
    """T08: the union's classic-CV arm needs the gate the yellow arm has.

    The union ``line | cv_white`` recovers paint the model misses, but the
    classic arm is a brightness/contrast rule that fires on kerbs, seams
    and shadows: measured against the engine's own annotation, 40-85% of
    the union candidates lay off the pavement and none on labelled paint.
    A candidate the LEARNED mask supports is kept as is - applying the
    elongation rule to everything deleted all 100 mask-backed candidates
    on an urban junction (zebra crossing = wide blocks).
    """

    def _mk(self, pixels, **kw):
        from beamng_autopilot.vision.lanes import LaneMarking
        import numpy as np
        return LaneMarking(world=np.zeros((len(pixels), 2)),
                           pixels=np.asarray(pixels, dtype=float),
                           color=kw.get("color", "white"),
                           kind=kw.get("kind", "thin"),
                           confidence=1.0)

    def test_an_unsupported_blob_is_dropped_and_an_elongated_on_road_kept(self):
        import numpy as np
        from beamng_autopilot.vision.segmentation import gate_line_candidates
        h = w = 60
        line = np.zeros((h, w), dtype=bool)
        road = np.zeros((h, w), dtype=bool)
        road[30:, :] = True
        blob = self._mk([[20 + (i % 5), 40 + (i // 5)] for i in range(25)])
        stroke = self._mk([[10 + i, 45] for i in range(20)])      # 20x1
        kept, dropped, on = gate_line_candidates([blob, stroke], line, road)
        assert on is True
        assert kept == [stroke], "only the elongated, on-road one survives"
        assert dropped.get("blob") == 1
        assert kept[0].meta["learned_frac"] == 0.0
        assert kept[0].meta["on_road_frac"] == 1.0
        assert kept[0].meta["aspect"] >= 2.5

    def test_an_unsupported_stroke_off_the_pavement_is_dropped(self):
        import numpy as np
        from beamng_autopilot.vision.segmentation import gate_line_candidates
        line = np.zeros((40, 40), dtype=bool)
        road = np.zeros((40, 40), dtype=bool)
        road[30:, :] = True                     # pavement only at the bottom
        off = self._mk([[2 + i, 5] for i in range(15)])           # top-left
        kept, dropped, _ = gate_line_candidates([off], line, road)
        assert kept == [] and dropped.get("off_pavement") == 1

    def test_a_learned_backed_candidate_is_kept_whatever_its_shape(self):
        import numpy as np
        from beamng_autopilot.vision.segmentation import gate_line_candidates
        line = np.zeros((60, 60), dtype=bool)
        road = np.zeros((60, 60), dtype=bool)   # nothing on the road at all
        wide = self._mk([[20 + (i % 10), 20 + (i // 10)] for i in range(50)])
        for px, py in wide.pixels.astype(int):
            line[py, px] = True                 # the learned mask supports it
        kept, dropped, _ = gate_line_candidates([wide], line, road)
        assert kept == [wide] and dropped == {}
        assert kept[0].meta["learned_frac"] == 1.0

    def test_the_gate_can_be_disabled_for_an_ab(self):
        import numpy as np
        from beamng_autopilot.vision.segmentation import gate_line_candidates
        line = np.zeros((40, 40), dtype=bool)
        road = np.zeros((40, 40), dtype=bool)
        blob = self._mk([[2 + (i % 4), 2 + (i // 4)] for i in range(16)])
        kept, dropped, on = gate_line_candidates([blob], line, road,
                                                 gate_on=False)
        assert on is False and kept == [blob] and dropped == {}

    def test_a_candidate_without_pixels_is_counted_not_crashed(self):
        import numpy as np
        from beamng_autopilot.vision.lanes import LaneMarking
        from beamng_autopilot.vision.segmentation import gate_line_candidates
        empty = LaneMarking(world=np.zeros((0, 2)), pixels=np.zeros((0, 2)))
        kept, dropped, _ = gate_line_candidates(
            [empty], np.zeros((10, 10), dtype=bool), None)
        assert kept == [] and dropped.get("no_pixels") == 1


class TestLineMaskRefine:
    """T11: the polarity-agnostic outlier test, and its measured limits.

    Measured against the engine's annotation: the appearance test is a net
    win on one dev scene (precision 0.194 -> 0.243 at unchanged recall) and
    costs ~20 points of recall on another, so it ships as a switch, not a
    default.  What these tests pin is the SEMANTICS: either polarity passes,
    a pixel matching the local road statistics does not, and a pixel with no
    road reference is KEPT and counted as unknown rather than dropped.
    """

    def test_paint_darker_or_brighter_than_the_road_both_pass(self):
        import numpy as np
        from beamng_autopilot.vision.seg_probs import refine_line_mask
        grey_road = 180
        img = np.full((60, 60, 3), grey_road, dtype=np.uint8)
        road = np.ones((60, 60), dtype=bool)
        dark = np.zeros((60, 60), dtype=bool); dark[30, 5:25] = True
        bright = np.zeros((60, 60), dtype=bool); bright[40, 5:25] = True
        img[30, 5:25] = 120          # darker than the road
        img[40, 5:25] = 240          # brighter than the road
        line = dark | bright
        keep, stats = refine_line_mask(line, road, img, on_road_dilate_px=0,
                                       z_min=1.2)
        assert keep[30, 5:25].all() and keep[40, 5:25].all(), \
            "polarity must not matter"
        assert stats["kept"] == int(line.sum())

    def test_a_pixel_matching_the_local_road_is_removed(self):
        import numpy as np
        from beamng_autopilot.vision.seg_probs import refine_line_mask
        img = np.full((60, 60, 3), 180, dtype=np.uint8)
        img[30, 5:25] = 182          # indistinguishable from the road
        line = np.zeros((60, 60), dtype=bool); line[30, 5:25] = True
        road = np.ones((60, 60), dtype=bool)
        keep, stats = refine_line_mask(line, road, img, on_road_dilate_px=0)
        assert keep.sum() == 0 and stats["not_outlier"] == 20

    def test_without_a_road_reference_the_pixel_is_kept_as_unknown(self):
        import numpy as np
        from beamng_autopilot.vision.seg_probs import refine_line_mask
        img = np.full((60, 60, 3), 180, dtype=np.uint8)
        line = np.zeros((60, 60), dtype=bool); line[30, 5:25] = True
        road = np.zeros((60, 60), dtype=bool)     # nothing to compare against
        keep, stats = refine_line_mask(line, road, img,
                                       on_road_dilate_px=0)
        assert stats["unknown_kept"] == 20
        assert keep.sum() == 20, "an unverifiable pixel is not 'not a line'"

    def test_the_structural_test_removes_off_road_pixels(self):
        import numpy as np
        from beamng_autopilot.vision.seg_probs import refine_line_mask
        img = np.full((80, 80, 3), 180, dtype=np.uint8)
        img[10, 5:25] = 240          # a bright stroke on the building
        line = np.zeros((80, 80), dtype=bool); line[10, 5:25] = True
        road = np.zeros((80, 80), dtype=bool); road[50:, :] = True
        keep, stats = refine_line_mask(line, road, img, on_road_dilate_px=4)
        assert keep.sum() == 0 and stats["off_road"] == 20

    def test_a_shape_mismatch_is_refused(self):
        import numpy as np
        import pytest as _pytest
        from beamng_autopilot.vision.seg_probs import refine_line_mask
        with _pytest.raises(ValueError):
            refine_line_mask(np.zeros((10, 10), dtype=bool),
                             np.zeros((9, 10), dtype=bool),
                             np.zeros((10, 10, 3), dtype=np.uint8))
