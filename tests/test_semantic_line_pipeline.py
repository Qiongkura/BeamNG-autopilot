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


class TestOptionalSurfaceGate:
    """可选收紧：连 learned-backed 候选也要求落在**模型自己的路面掩码**上。

    默认**关闭**（`BEAMNG_LINE_CAND_SURFACE_GATE`）。注意判据**不能**写成
    "路面 ∪ 线掩码"：``learned`` 的定义就是"像素在 line 掩码里的占比"，所以那个
    并集对 learned-backed 候选恒真、等于没写（这版被本测试当场抓住）。可用的
    运行期判据是"是否落在模型的路面区域内"。
    """

    def _mk(self, pixels, learned=1.0):
        from beamng_autopilot.vision.lanes import LaneMarking

        class _M(LaneMarking):
            def __init__(self):
                super().__init__(world=np.zeros((len(pixels), 3)), pixels=np.asarray(pixels),
                                 color="white", kind="thin", confidence=0.9, meta={})
        return _M()

    def _masks(self, shape=(20, 40)):
        """线掩码在路面之外（模拟"模型画到了自己路面区域之外"）。"""
        import numpy as np
        line = np.zeros(shape, bool)
        line[2, 2:8] = True                 # 漆线在上方：既在 line 内、又在 road 外
        road = np.zeros(shape, bool)
        road[10:20, :] = True               # 路面在下半
        return line, road

    def test_off_by_default_keeps_learned_candidates_off_surface(self):
        from beamng_autopilot.vision.segmentation import gate_line_candidates
        line, road = self._masks()
        mk = self._mk([[3, 2], [4, 2], [5, 2]])    # 在 line 掩码内（learned=1）
        kept, dropped, _ = gate_line_candidates([mk], line, road)
        assert len(kept) == 1 and not dropped, "默认行为不得改变"
        assert mk.meta["learned_frac"] == 1.0, "该候选确实是 learned-backed"

    def test_enabled_drops_learned_candidates_off_the_road_region(self):
        from beamng_autopilot.vision.segmentation import gate_line_candidates
        line, road = self._masks()
        # 两个都 learned-backed：一个落在模型路面区域外，一个落在区域内
        off = self._mk([[3, 2], [4, 2], [5, 2]])
        inside = self._mk([[3, 12], [4, 12], [5, 12]])
        line[12, 2:6] = True                  # 让 inside 也 learned-backed
        kept, dropped, _ = gate_line_candidates([off, inside], line, road,
                                               surface_gate=True)
        ids = [id(m) for m in kept]
        assert id(off) not in ids, "模型路面区域外的 learned 候选应被丢"
        assert id(inside) in ids, "路面区域内的必须保留"
        assert dropped.get("learned_off_road") == 1
        # 注意：surface_frac（线∪路面）对 learned 候选恒为 1——它只是在 line 掩码里。
        # 这正是"并集判据"对 learned 候选无效的原因，有区分度的是 on_road_frac。
        assert off.meta["on_road_frac"] == 0.0 and off.meta["surface_frac"] == 1.0
        assert inside.meta["on_road_frac"] == 1.0

    def test_a_missing_road_mask_is_unknown_not_a_drop(self):
        from beamng_autopilot.vision.segmentation import gate_line_candidates
        line, _road = self._masks()
        mk = self._mk([[5, 2], [6, 2]])
        kept, dropped, _ = gate_line_candidates([mk], line, None,
                                                surface_gate=True)
        assert len(kept) == 1 and not dropped, "路面未知时不得据此丢弃"
        assert mk.meta["surface_frac"] is None


# ------------------------------------------------- 同侧近邻候选合并（路线 b）

def test_merge_close_candidates_same_side_only():
    """同侧 <1.5 m 合并成一条（并集），异侧/远距不动，可关。

    实测依据（T16 §18.5 路线 b）：未匹配候选里 15–23% 与**同侧 1.5 m 内的已匹配
    候选**成对出现（同一根线被切成两条），合并是**参考无关**的，离线账面身份率
    +0.040…+0.065 且已匹配候选一条不丢。真线间距 ≥2.5 m（车道宽），
    1.5 m 半径不会把两根真线并成一根。
    """
    import numpy as np
    from beamng_autopilot.vision.segmentation import merge_close_candidates
    from beamng_autopilot.vision.lanes import LaneMarking

    def mk(lat, n=5):
        return LaneMarking(world=np.array([[0.0, float(lat)]] * n),
                           pixels=np.zeros((n, 2)), color="white",
                           kind="thin", confidence=0.5)

    # 生产默认半径 1.0 m（实测选定，§20：6/6 arm 过角色门）：0.8 m 的两条并成一条
    out, info = merge_close_candidates([mk(1.7), mk(2.5)], pos=(0.0, 0.0, 0.0),
                                       heading=0.0)
    assert len(out) == 1 and info["merged"] == 1
    assert out[0].meta["merged_from"] == 2
    assert len(out[0].world) == 10          # 并集：信息只增不减
    # 同侧 1.2 m：超过默认半径 -> 不合并（真线近/远档可以只差 1.0–1.5 m）
    assert len(merge_close_candidates([mk(1.7), mk(2.9)], pos=(0.0, 0.0, 0.0),
                                      heading=0.0)[0]) == 2
    # 小半径（测量用，§19 的 0.5 m 档）：0.8 m 也不合并
    small, _ = merge_close_candidates([mk(1.7), mk(2.5)],
                                      pos=(0.0, 0.0, 0.0), heading=0.0,
                                      max_gap_m=0.5)
    assert len(small) == 2
    # 同侧 2.5 m（真线间距量级）：不合并
    assert len(merge_close_candidates([mk(1.7), mk(4.2)], pos=(0.0, 0.0, 0.0),
                                      heading=0.0)[0]) == 2
    # 异侧 1.2 m：不合并（横向符号不同）
    assert len(merge_close_candidates([mk(1.7), mk(-1.7)], pos=(0.0, 0.0, 0.0),
                                      heading=0.0)[0]) == 2
    # 关掉：原样返回
    off, info_off = merge_close_candidates([mk(1.7), mk(2.0)],
                                           pos=(0.0, 0.0, 0.0), heading=0.0,
                                           enable=False)
    assert len(off) == 2 and info_off["enabled"] is False
    # 缺 world 的候选（无法定位）：保留、不参与合并
    bad = LaneMarking(world=np.zeros((0, 2)), pixels=np.zeros((3, 2)))
    out2, _ = merge_close_candidates([mk(1.7), mk(2.0), bad],
                                     pos=(0.0, 0.0, 0.0), heading=0.0)
    assert len(out2) == 2


def test_scope_lateral_candidates_drops_only_beyond_limit():
    """横向口径门（单因子开关，默认关）：只丢 |lat| 超限的；world 缺失的保留
    （判不了不丢）；关掉/非正值是 no-op；debug 里能看到计数。

    实测依据（R3 第二轮 + 离线扫描，seed 42）：丢 |lat|>3 m 的候选，
    dev 身份 0.4396→0.7129（M 160→72）、R3 0.3989→0.7576（M 恒 75）——
    代价在召回，改默认必须先过像素级门。
    """
    import numpy as np
    from beamng_autopilot.vision.segmentation import (
        scope_lateral_candidates)
    from beamng_autopilot.vision.lanes import LaneMarking

    def mk(lat, n=5):
        return LaneMarking(world=np.array([[0.0, float(lat)]] * n),
                           pixels=np.zeros((n, 2)), color="white",
                           kind="thin", confidence=0.5)

    debug = {}
    out, info = scope_lateral_candidates(
        [mk(1.7), mk(-6.0), mk(7.5), mk(-2.0)], pos=(0.0, 0.0, 0.0),
        heading=0.0, max_lat_m=3.0, enable=True, debug=debug)
    assert sorted(float(np.median(m.world[:, 1])) for m in out) == [-2.0, 1.7]
    assert info["dropped"] == 2 and info["unknown_kept"] == 0
    assert debug["line_candidate_lat_scope"]["dropped"] == 2
    # world 缺失：保留（判不了不丢）
    bad = LaneMarking(world=None, pixels=np.zeros((3, 2)), color="white",
                      kind="thin", confidence=0.5)
    out, info = scope_lateral_candidates([bad, mk(9.0)], pos=(0.0, 0.0, 0.0),
                                         heading=0.0, max_lat_m=3.0,
                                         enable=True)
    assert len(out) == 1 and info["unknown_kept"] == 1 and info["dropped"] == 1
    # 默认关 / 非正值：no-op（候选集属冻结口径，改默认要走协议新版）
    for kw in ({"enable": False, "max_lat_m": 3.0},
               {"enable": True, "max_lat_m": 0.0}):
        out, info = scope_lateral_candidates([mk(1.7), mk(9.0)],
                                             pos=(0.0, 0.0, 0.0), heading=0.0,
                                             **kw)
        assert len(out) == 2 and info["dropped"] == 0

def test_head_final_candidate_scope_call_sites():
    """head 的最终候选集处理：合并补齐与横向口径门必须都在**写 meta 之前**，
    且默认关、debug 键不互相覆盖（两处都实测踩过）。

    实测（2026-10-04）：
    * 把门接在 detect_lines 里只覆盖一个贡献者（门只丢 30/227，离线扫描应丢
      100/227）——黄臂补候选/经典回退发生在 detect_lines 之后；
    * 最终合并若复用 `line_candidate_merge` 键会覆盖提取器那次的 in/out。
    """
    from pathlib import Path as _P
    src = (_P(__file__).resolve().parents[1] / "beamng_autopilot" / "vision"
           / "heads" / "semantic.py").read_text(encoding="utf-8")
    # 三项开关收在 line_scope（协议级）；head/segmentation 只问它，不自己读 env
    assert "line_scope.merge_final_enabled()" in src, "合并补齐必须走协议开关"
    assert "line_candidate_merge_final" in src, "最终合并要单独记，不覆盖提取器那次"
    assert "scope_lateral_candidates(" in src
    i_merge = src.index("merge_close_candidates(")
    i_gate = src.index("scope_lateral_candidates(")
    i_store = src.index('out.meta["markings"] = markings')
    assert i_merge < i_gate < i_store, "顺序必须是：合并补齐 -> 横向门 -> 写 meta"
    seg = (_P(__file__).resolve().parents[1] / "beamng_autopilot" / "vision"
           / "segmentation.py").read_text(encoding="utf-8")
    assert "line_scope.lat_max_m()" in seg, "横向门默认值必须来自协议开关"
    assert "line_scope.appearance_gate_enabled()" in seg, "外观门必须走协议开关"

def test_paint_like_mask_and_appearance_gate(monkeypatch):
    """外观判据：白/暖白/黄像漆，灰/彩色不像；形状错抛错（不静默全 False）。

    掩码门是**单调**的：`line & 像漆` 只删不加；env 默认关（掩码口径属冻结
    范围）。实测依据见 `docs/T16_NEXT_FACTOR_FAR_OFFROAD_PROPOSAL_20261004.md` §9。
    """
    import numpy as np
    from beamng_autopilot.vision.paint_appearance import paint_like_mask
    from beamng_autopilot.vision.segmentation import Segmenter

    rgb = np.zeros((3, 4, 3), np.uint8)
    rgb[0, 0] = (230, 232, 228)      # 白漆：亮、无彩
    rgb[0, 1] = (200, 180, 150)      # 暖白：r-b = 50
    rgb[0, 2] = (200, 190, 60)       # 黄漆
    rgb[1, 0] = (120, 120, 120)      # 中灰（沥青/阴影边界）不像漆
    rgb[1, 1] = (150, 60, 40)        # 橙棕不像漆
    rgb[1, 2] = (100, 180, 60)       # 饱和绿（植被）不像漆：r < 135 且不亮
    m = paint_like_mask(rgb)
    assert m[0, 0] and m[0, 1] and m[0, 2]
    assert not m[1, 0] and not m[1, 1] and not m[1, 2]
    # 已知的**宽松**处（实测记录，别当 bug 修）：黄绿植被 (200,240,60) 会命中
    # 黄漆规则——这正是"外观判据不能当真值判据"的原因（负例上 13k–68k px）。
    # 作门用没问题（单调过滤，只是不删这些像素）。
    yg = np.zeros((1, 1, 3), np.uint8)
    yg[0, 0] = (200, 240, 60)
    assert paint_like_mask(yg)[0, 0], "黄绿命中黄漆规则：文档化的宽松边界"
    try:
        paint_like_mask(np.zeros((3, 4), np.uint8))
        raise AssertionError("形状不对必须抛错")
    except ValueError:
        pass
    # 门：默认关 -> 原样；开着 -> 只删不加
    seg = Segmenter.__new__(Segmenter)
    line = np.zeros((3, 4), bool)
    line[0, 0] = line[1, 0] = True    # 一个像漆、一个不像
    monkeypatch.delenv("BEAMNG_LINE_APPEARANCE_GATE", raising=False)
    out = seg._appearance_gate_line(line, rgb)
    assert int(out.sum()) == 2, "默认关必须原样返回"
    monkeypatch.setenv("BEAMNG_LINE_APPEARANCE_GATE", "1")
    out = seg._appearance_gate_line(line, rgb)
    assert out[0, 0] and not out[1, 0], "开了只保留像漆的像素"
    assert int(out.sum()) == 1
    assert not seg._appearance_gate_line(line, rgb)[1, 0], "只删不加（单调）"

def test_line_scope_protocol_switch(monkeypatch):
    """协议开关：v7 默认三项全关；BEAMNG_PROTOCOL=v8 三项全开且 L=5.5（=配对可达
    边界）；单项 env 优先于协议默认（供单因子测量/消融）。

    为什么要它：三项变更散在三个模块，采纳时"切默认"靠手改容易只改一半
    （本项目踩过"改了一半的口径"的坑）。这里把开关收成一处并锁住语义。
    """
    from beamng_autopilot.lane.constants import LANE_PAIR_NEAR_MAX_M
    from beamng_autopilot.vision import line_scope
    for k in ("BEAMNG_PROTOCOL", "BEAMNG_LINE_LAT_MAX_M",
              "BEAMNG_LINE_MERGE_FINAL", "BEAMNG_LINE_APPEARANCE_GATE"):
        monkeypatch.delenv(k, raising=False)
    # 默认 v7：三项全关
    assert line_scope.protocol() == "v7"
    assert line_scope.lat_max_m() == 0.0
    assert not line_scope.merge_final_enabled()
    assert not line_scope.appearance_gate_enabled()
    # v8：三项全开，L 取配对可达边界
    monkeypatch.setenv("BEAMNG_PROTOCOL", "v8")
    assert line_scope.lat_max_m() == float(LANE_PAIR_NEAR_MAX_M) == 5.5
    assert line_scope.merge_final_enabled()
    assert line_scope.appearance_gate_enabled()
    # 单项 env 优先（消融：v8 下单独关掉外观门）
    monkeypatch.setenv("BEAMNG_LINE_APPEARANCE_GATE", "0")
    assert not line_scope.appearance_gate_enabled()
    assert line_scope.lat_max_m() == 5.5
    # 未知协议值按默认（v7）处理，不猜
    monkeypatch.setenv("BEAMNG_PROTOCOL", "v9")
    assert line_scope.protocol() == "v7" and line_scope.lat_max_m() == 0.0
    # 生效版本号跟着开关走（记录必须反映实际口径）
    from beamng_autopilot.experiments.protocol import (
        PROTOCOL_VERSION, PROTOCOL_VERSION_V8, active_protocol_version)
    assert active_protocol_version() == PROTOCOL_VERSION
    monkeypatch.setenv("BEAMNG_PROTOCOL", "v8")
    assert active_protocol_version() == PROTOCOL_VERSION_V8


def test_lateral_scope_honours_protocol_switch(monkeypatch):
    """横向门在 v8 下默认生效（L=5.5）：5.0 m 留、6.0 m 丢。"""
    import numpy as np
    from beamng_autopilot.vision.segmentation import scope_lateral_candidates
    from beamng_autopilot.vision.lanes import LaneMarking

    def mk(lat):
        return LaneMarking(world=np.array([[0.0, float(lat)]] * 3),
                           pixels=np.zeros((3, 2)), color="white",
                           kind="thin", confidence=0.5)

    monkeypatch.delenv("BEAMNG_LINE_LAT_MAX_M", raising=False)
    monkeypatch.setenv("BEAMNG_PROTOCOL", "v8")
    out, info = scope_lateral_candidates([mk(5.0), mk(6.0)],
                                         pos=(0.0, 0.0, 0.0), heading=0.0)
    assert len(out) == 1 and info["dropped"] == 1 and info["max_lat_m"] == 5.5
