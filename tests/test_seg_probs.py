"""Segmentation probabilities and dual gating (plan phase E1)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from beamng_autopilot.vision.seg_probs import (
    SEG_LINE_PROB_MIN,
    SEG_ROAD_PROB_MIN,
    SegProbabilityMaps,
    gate_masks,
    softmax_maps,
)

H, W = 20, 24


def _maps(road=0.9, line=0.0, shape=(H, W)):
    r = np.full(shape, float(road), dtype=np.float32)
    l = np.full(shape, float(line), dtype=np.float32)
    return SegProbabilityMaps(line=l, road=r)


# ---------------------------------------------------------------------------
# probability maps
# ---------------------------------------------------------------------------

def test_softmax_maps_shapes_and_normalisation() -> None:
    logits = np.zeros((3, 4, 5), dtype=np.float32)
    logits[2] = 4.0                       # line channel wins everywhere
    m = softmax_maps(logits)
    assert m.line.shape == (4, 5) and m.road.shape == (4, 5)
    total = m.line + m.road + m.background
    assert np.allclose(total, 1.0, atol=1e-5)
    assert (m.line > m.road).all()


def test_softmax_maps_accepts_a_batch_axis_and_passes_probabilities() -> None:
    probs = np.zeros((1, 3, 2, 2), dtype=np.float32)
    probs[0, 1] = 0.7
    probs[0, 0] = 0.3
    m = softmax_maps(probs)
    assert np.allclose(m.road, 0.7, atol=1e-6)
    assert np.allclose(m.background, 0.3, atol=1e-6)


def test_softmax_maps_survives_non_finite_logits() -> None:
    logits = np.full((3, 3, 3), np.nan, dtype=np.float32)
    logits[2] = -np.inf
    m = softmax_maps(logits)
    assert np.isfinite(m.line).all() and np.isfinite(m.road).all()


def test_softmax_maps_rejects_a_multi_sample_batch() -> None:
    with pytest.raises(ValueError):
        softmax_maps(np.zeros((2, 3, 4, 4), dtype=np.float32))


def test_named_channels_report_absent_classes_as_none() -> None:
    m = _maps()
    named = m.named()
    assert named["road_probability"] is not None
    assert named["line_probability"] is not None
    # the 3-class model has no white/yellow channel: None, not a fake zero
    assert named["white_probability"] is None
    assert named["yellow_probability"] is None


# ---------------------------------------------------------------------------
# dual gating
# ---------------------------------------------------------------------------

def test_bright_line_on_road_passes_both_gates() -> None:
    m = _maps(road=0.95, line=0.0)
    m.line[8:12, 10:12] = 0.9
    road, line, stats = gate_masks(m)
    assert road.any() and line.any()
    assert stats.line_before == stats.line_after_road == 8
    assert stats.mean_line_prob > 0.8


def test_weak_line_probability_is_gated_out() -> None:
    m = _maps(road=0.95, line=0.0)
    m.line[8:12, 10:12] = SEG_LINE_PROB_MIN - 0.1
    _road, line, stats = gate_masks(m)
    assert not line.any()
    assert stats.line_before == 0


def test_line_off_road_context_is_gated_out() -> None:
    """A line-like stroke with no road around it is not a lane marking."""
    m = _maps(road=0.05, line=0.0)          # everything is "grass"
    m.line[8:12, 10:12] = 0.95              # white post / wall edge
    _road, line, stats = gate_masks(m)
    assert not line.any(), "off-road white must not become a lane line"
    assert stats.line_before == 8 and stats.line_after_road == 0


def test_paint_on_road_passes_even_though_paint_is_not_road() -> None:
    """The gate must be the road CONTEXT, not the probability at the pixel.

    White paint is not asphalt, so the model legitimately reads a low
    road probability exactly where a line is - an at-the-pixel gate would
    erase every real marking.
    """
    m = _maps(road=0.95, line=0.0)
    m.road[8:12, 10:12] = 0.05              # the paint stroke itself
    m.line[8:12, 10:12] = 0.95
    _road, line, stats = gate_masks(m)
    assert line.any(), "the road-context gate must look past the stroke"
    assert int(np.count_nonzero(line)) == 8


def test_road_mask_is_a_soft_threshold_not_argmax() -> None:
    m = _maps(road=0.0)
    m.road[:, :10] = 0.6                    # road
    m.road[:, 10:] = 0.199                  # below the soft floor
    road, _line, stats = gate_masks(m)
    assert int(road[:, :10].sum()) == H * 10
    assert not road[:, 10:].any()
    assert stats.road_before == stats.road_after == H * 10


def test_extra_line_source_still_needs_the_road_context() -> None:
    """The HSV yellow prior cannot smuggle an off-road match in."""
    m = _maps(road=0.05, line=0.0)
    extra = np.zeros((H, W), dtype=bool)
    extra[4:6, 4:6] = True
    _road, line, _stats = gate_masks(m, extra_line=extra)
    assert not line.any()
    m2 = _maps(road=0.9, line=0.0)
    _road2, line2, stats2 = gate_masks(m2, extra_line=extra)
    assert int(np.count_nonzero(line2)) == 4
    assert stats2.line_before == 4


def test_context_radius_is_configurable() -> None:
    m = _maps(road=0.0, line=0.0)
    m.line[10, 10] = 0.9
    m.road[10, 14] = 0.9                    # road 4 px away
    _r, near, _s = gate_masks(m, context_px=1)
    assert not near.any()
    _r2, far, _s2 = gate_masks(m, context_px=4)
    assert far[10, 10], "a wider context window must reach the road"


def test_gate_stats_digest_is_json_safe() -> None:
    m = _maps(road=0.9, line=0.0)
    m.line[3:5, 3:5] = 0.8
    _road, _line, stats = gate_masks(m)
    text = json.dumps(stats.digest())
    assert "line_road" in text and "nan" not in text.lower()


def test_mismatched_shapes_are_rejected() -> None:
    m = SegProbabilityMaps(line=np.zeros((4, 4), dtype=np.float32),
                           road=np.zeros((5, 5), dtype=np.float32))
    with pytest.raises(ValueError):
        gate_masks(m)


def test_defaults_are_the_documented_thresholds() -> None:
    assert 0.0 < SEG_ROAD_PROB_MIN < SEG_LINE_PROB_MIN <= 1.0


# ---------------------------------------------------------------------------
# Segmenter.predict_proba (keeps the probabilities the argmax threw away)
# ---------------------------------------------------------------------------

def test_predict_proba_returns_normalised_maps_and_raw_argmax(tmp_path) -> None:
    """The probability API must agree with the argmax it is built on."""
    torch = pytest.importorskip("torch")
    from beamng_autopilot.vision.segmentation import Segmenter, SegUNet
    ckpt = tmp_path / "best.pt"
    model = SegUNet(n_classes=3)
    torch.manual_seed(0)
    torch.save({"state_dict": model.state_dict(), "n_classes": 3,
                "class_names": ["background", "asphalt", "line"]}, ckpt)
    seg = Segmenter(model_path=ckpt, device="cpu", use_half=False)
    frame = np.random.default_rng(3).integers(
        0, 255, (48, 64, 3), dtype=np.uint8)
    logits = seg._infer_logits(frame)
    maps, raw_road, raw_line = seg.predict_proba(frame, _logits=logits)
    assert maps.line.shape == (48, 64)
    assert maps.road.shape == (48, 64)
    total = maps.line + maps.road + maps.background
    assert np.allclose(total, 1.0, atol=1e-4)
    assert maps.line.min() >= 0.0 and maps.line.max() <= 1.0
    # the raw masks are exactly the argmax decision, channel by channel
    road, line = seg._argmax_masks(logits, frame)
    assert np.array_equal(raw_road, road)
    assert np.array_equal(raw_line, line)


@pytest.mark.parametrize("road_index,line_index,winner", [(1, 2, "road"), (2, 1, "line")])
def test_raw_ties_are_exclusive_and_follow_native_channel_order(road_index, line_index, winner):
    torch = pytest.importorskip("torch")
    from beamng_autopilot.vision.segmentation import Segmenter
    seg = object.__new__(Segmenter)
    seg._road_idx, seg._line_idx, seg._bg_idx = road_index, line_index, 0
    logits = torch.full((1, 3, 1, 2), 3.0)
    logits[:, 0] = -10
    frame = np.zeros((2, 4, 3), np.uint8)
    _maps_out, road, line = seg.predict_proba(frame, _logits=logits)
    assert not (road & line).any()
    assert road.all() == (winner == "road")
    assert line.all() == (winner == "line")
    logits.zero_()  # Background wins a three-way tie.
    _maps_out, road, line = seg.predict_proba(frame, _logits=logits)
    assert not road.any() and not line.any()


def test_raw_resize_precedes_probability_interpolation_decisions():
    torch = pytest.importorskip("torch")
    from beamng_autopilot.vision.segmentation import Segmenter
    seg = object.__new__(Segmenter)
    seg._road_idx, seg._line_idx, seg._bg_idx = 1, 2, 0
    logits = torch.tensor([[[[-100., -100.]],
                            [[np.log(.51), np.log(.01)]],
                            [[np.log(.49), np.log(.99)]]]])
    maps, road, line = seg.predict_proba(np.zeros((1, 4, 3), np.uint8), _logits=logits)
    np.testing.assert_array_equal(road, [[True, True, False, False]])
    np.testing.assert_array_equal(line, [[False, False, True, True]])
    assert maps.line[0, 1] > maps.road[0, 1]  # Linear probabilities differ here.
    assert maps.road[0, 1] == pytest.approx(.385, abs=1e-5)


@pytest.mark.parametrize("missing,index", [("road", None), ("line", None), ("line", 4)])
def test_probability_api_rejects_missing_channels_explicitly(missing, index):
    torch = pytest.importorskip("torch")
    from beamng_autopilot.vision.segmentation import Segmenter
    seg = object.__new__(Segmenter)
    seg._road_idx, seg._line_idx, seg._bg_idx = 1, 2, 0
    setattr(seg, f"_{missing}_idx", index)
    with pytest.raises(ValueError, match="requires road and line"):
        seg.predict_proba(np.zeros((2, 4, 3), np.uint8), _logits=torch.zeros(1, 3, 1, 2))


def test_predict_and_predict_with_probs_agree(tmp_path) -> None:
    """One inference must serve both APIs with identical masks."""
    torch = pytest.importorskip("torch")
    from beamng_autopilot.vision.segmentation import Segmenter, SegUNet
    ckpt = tmp_path / "best.pt"
    torch.manual_seed(1)
    model = SegUNet(n_classes=3)
    torch.save({"state_dict": model.state_dict(), "n_classes": 3,
                "class_names": ["background", "asphalt", "line"]}, ckpt)
    seg = Segmenter(model_path=ckpt, device="cpu", use_half=False)
    frame = np.random.default_rng(7).integers(0, 255, (48, 64, 3),
                                              dtype=np.uint8)
    road_a, line_a = seg.predict(frame)
    seg2 = Segmenter(model_path=ckpt, device="cpu", use_half=False)
    road_b, line_b, maps = seg2.predict_with_probs(frame)
    assert np.array_equal(road_a, road_b)
    assert np.array_equal(line_a, line_b)
    assert maps.line.shape == frame.shape[:2]
