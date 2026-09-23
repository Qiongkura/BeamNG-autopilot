"""Class probabilities and dual gating for segmentation (plan phase E1).

``Segmenter.predict`` throws the class probabilities away with an argmax
and then has to defend the resulting binary masks with morphology,
temporal hysteresis and shape heuristics.  Plan phase E1 asks for the
probabilities to be KEPT and used as gates instead:

* a ``line`` pixel must clear ``line_min`` on the line channel, AND
* it must sit in a ROAD CONTEXT - because white paint is not asphalt, the
  road probability AT a painted pixel is legitimately low, so the gate is
  a max-filtered road probability around the pixel, not the value at the
  pixel itself;
* the ``road`` decision itself is a soft threshold on the road channel
  rather than an argmax, so a 50/50 boundary pixel no longer flips the
  drivable mask.

Everything here is pure numpy so the gates are unit-testable without a
model, a GPU or a frame: the caller converts whatever its network emits
into ``(C, H, W)`` logits (or probabilities) and gets maps and masks
back, plus a digest that says how many pixels each gate removed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# A line pixel must clear this probability on the line channel.
SEG_LINE_PROB_MIN = 0.5
# The road CONTEXT around a line pixel must reach this (max-filtered).
SEG_ROAD_PROB_MIN = 0.35
# The road mask itself is a soft threshold, not an argmax.
SEG_ROAD_SOFT_MIN = 0.20
# Radius (pixels) of the road-context max filter: wide enough to see past
# the paint stroke or a curb line, small enough not to bridge to the next
# street.
SEG_ROAD_CONTEXT_PX = 3


@dataclass
class SegProbabilityMaps:
    """Per-class probability maps at one resolution, values in [0, 1]."""

    line: np.ndarray
    road: np.ndarray
    background: np.ndarray | None = None
    white: np.ndarray | None = None
    yellow: np.ndarray | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.line.shape[0]), int(self.line.shape[1]))

    def named(self) -> dict:
        """The plan's four channels plus background, when available.

        A channel the model does not have is reported as None rather
        than fabricated - the current UNet is 3-class
        (background/road/line) and the yellow/white evidence comes from
        other sources when a caller supplies it.
        """
        return {"road_probability": self.road,
                "line_probability": self.line,
                "background_probability": self.background,
                "white_probability": self.white,
                "yellow_probability": self.yellow}


@dataclass
class GateStats:
    """How many pixels each gate removed (telemetry + tests)."""

    line_before: int = 0
    line_after_prob: int = 0
    line_after_road: int = 0
    road_before: int = 0
    road_after: int = 0
    mean_line_prob: float = 0.0
    mean_road_context: float = 0.0
    extra: dict = field(default_factory=dict)

    def digest(self) -> dict:
        def _r(v: float) -> float | None:
            return None if not math.isfinite(float(v)) else round(float(v), 4)
        return {
            "line_before": int(self.line_before),
            "line_prob": int(self.line_after_prob),
            "line_road": int(self.line_after_road),
            "road_before": int(self.road_before),
            "road_after": int(self.road_after),
            "mean_line_p": _r(self.mean_line_prob),
            "mean_road_ctx": _r(self.mean_road_context),
        }


def softmax_maps(logits: np.ndarray, *, line_index: int = 2,
                 road_index: int = 1, background_index: int = 0,
                 white_index: int | None = None,
                 yellow_index: int | None = None
                 ) -> SegProbabilityMaps:
    """``(C, H, W)`` logits (or probabilities) -> per-class maps.

    Accepts a leading batch axis of 1.  Inputs that already look like
    probabilities (all finite, non-negative, summing to 1 across the
    channels) are passed through, so a caller with a softmaxed tensor
    does not pay for a second one.
    """
    arr = np.asarray(logits, dtype=np.float32)
    if arr.ndim == 4:
        if arr.shape[0] != 1:
            raise ValueError(f"expected a single sample, got {arr.shape}")
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"expected (C, H, W), got {arr.shape}")
    finite = np.isfinite(arr)
    looks_like_probs = (finite.all() and (arr >= 0.0).all()
                        and float(np.abs(arr.sum(axis=0) - 1.0).max()) < 1e-3)
    if looks_like_probs:
        probs = arr
    else:
        shifted = np.where(finite, arr, -np.inf)
        mx = shifted.max(axis=0, keepdims=True)
        # an all -inf column (every channel non-finite) would otherwise
        # subtract -inf from -inf and emit a NaN warning; zeroing the max
        # leaves exp(-inf) = 0 and the total>0 guard turns that into 0.0
        mx = np.where(np.isfinite(mx), mx, 0.0)
        shifted = shifted - mx
        exp = np.exp(np.where(np.isfinite(shifted), shifted, -np.inf))
        total = exp.sum(axis=0, keepdims=True)
        probs = np.where(total > 0.0, exp / np.maximum(total, 1e-12), 0.0)
    probs = np.clip(probs, 0.0, 1.0).astype(np.float32)

    def _pick(idx: int | None):
        if idx is None or not (0 <= int(idx) < probs.shape[0]):
            return None
        return probs[int(idx)]

    return SegProbabilityMaps(
        line=_pick(line_index),
        road=_pick(road_index),
        background=_pick(background_index),
        white=_pick(white_index),
        yellow=_pick(yellow_index))


def _max_filter(m: np.ndarray, radius: int) -> np.ndarray:
    """Sliding max over a square window (no SciPy dependency)."""
    r = max(0, int(radius))
    if r == 0:
        return m
    out = m
    for _ in range(r):
        padded = np.pad(out, 1, mode="edge")
        out = np.maximum.reduce([
            padded[:-2, :-2], padded[:-2, 1:-1], padded[:-2, 2:],
            padded[1:-1, :-2], padded[1:-1, 1:-1], padded[1:-1, 2:],
            padded[2:, :-2], padded[2:, 1:-1], padded[2:, 2:],
        ])
    return out


def gate_masks(maps: SegProbabilityMaps, *,
               line_min: float = SEG_LINE_PROB_MIN,
               road_min: float = SEG_ROAD_PROB_MIN,
               road_soft_min: float = SEG_ROAD_SOFT_MIN,
               context_px: int = SEG_ROAD_CONTEXT_PX,
               extra_line: np.ndarray | None = None,
               ) -> tuple[np.ndarray, np.ndarray, GateStats]:
    """``(road_mask, line_mask, stats)`` from probability maps.

    The line gate is the plan's dual gate: line probability above
    ``line_min`` AND road context (max-filtered road probability) above
    ``road_min``.  ``extra_line`` is an optional independent line source
    (the HSV yellow prior, say); it still has to pass the road-context
    gate, so an off-road colour match cannot enter the mask.
    """
    if maps.line is None or maps.road is None:
        raise ValueError("line and road probability maps are required")
    if maps.line.shape != maps.road.shape:
        raise ValueError("line and road maps must share a shape")
    stats = GateStats()
    road_ctx = _max_filter(maps.road, int(context_px))

    raw_line = maps.line >= float(line_min)
    if extra_line is not None:
        extra = np.asarray(extra_line, dtype=bool)
        if extra.shape == raw_line.shape:
            raw_line = raw_line | extra
    stats.line_before = int(np.count_nonzero(raw_line))
    stats.line_after_prob = stats.line_before

    road_mask = maps.road >= float(road_soft_min)
    stats.road_before = int(np.count_nonzero(road_mask))
    stats.road_after = stats.road_before

    context_ok = road_ctx >= float(road_min)
    line_mask = raw_line & context_ok
    stats.line_after_road = int(np.count_nonzero(line_mask))
    if stats.line_before:
        stats.mean_line_prob = float(
            maps.line[raw_line].mean()) if raw_line.any() else 0.0
    if stats.line_after_road:
        stats.mean_road_context = float(road_ctx[line_mask].mean())
    return road_mask, line_mask, stats

# ---------------------------------------------------------------------------
# Appearance + structure refinement of the line mask (T11/T08).
#
# Measured against the engine's own annotation (20 real frames, urban
# junction): the line mask's precision was 0.130 at 0.927 recall - 7x over
# marking - and the false positives were NOT sky (0.1%) but man-made hard
# surfaces: BUILDINGS 62.8% of the left-half FPs (11.6% of all building
# pixels called line), ASPHALT 24-54%, SIDEWALK 12-13% (74.5% of sidewalk
# pixels on one side!).  47-86% of those components are LINE-SHAPED, so a
# shape gate cannot remove them.  Brightness is INVERTED on that scene: the
# real paint is darker (153.9) than the FPs (185.0) and than the pavement
# (186.1), so no fixed-polarity brightness rule works.
#
# The two tests here are therefore:
#   structure - the pixel must lie on (or within a few px of) the published
#               road surface; buildings, sidewalk and sky are off it;
#   appearance- the pixel must be an OUTLIER against the LOCAL ROAD
#               appearance in EITHER direction, so paint darker or brighter
#               than the pavement both pass, while texture shadows/joints
#               that match the local road statistics do not.
# When the local road reference is unavailable the pixel is KEPT and counted
# as unknown: an unverifiable pixel must not silently become "not a line".
# ---------------------------------------------------------------------------
SEG_LINE_ON_ROAD_DILATE_PX = 6
SEG_LINE_OUTLIER_Z_MIN = 1.2
SEG_LINE_REF_WINDOW_PX = 31
SEG_LINE_REF_MIN_ROAD_PX = 40
SEG_LINE_REF_SCALE_FLOOR = 6.0


def refine_line_mask(line, road, frame_rgb, *,
                     on_road_dilate_px: int = SEG_LINE_ON_ROAD_DILATE_PX,
                     z_min: float = SEG_LINE_OUTLIER_Z_MIN,
                     window_px: int = SEG_LINE_REF_WINDOW_PX,
                     min_ref_px: int = SEG_LINE_REF_MIN_ROAD_PX,
                     scale_floor: float = SEG_LINE_REF_SCALE_FLOOR):
    """Refine a line mask with a structure test and a polarity-free outlier.

    Returns ``(refined_mask, stats)``; ``stats`` separates WHY pixels went
    (off_road / not_outlier), how many could not be judged (unknown_kept),
    and keeps the input counts so a report can compute both precision and
    recall without trusting the function's own arithmetic.
    """
    import cv2

    line = np.asarray(line, dtype=bool)
    road = np.asarray(road, dtype=bool)
    if line.shape != road.shape:
        raise ValueError("line and road masks must share a shape")
    rgb = np.asarray(frame_rgb)
    grey = (rgb[..., :3].astype(np.float32).mean(axis=2) if rgb.ndim == 3
            else rgb.astype(np.float32))
    stats = {"in": int(line.sum()), "road_px": int(road.sum())}
    if stats["in"] == 0:
        stats.update({"kept": 0, "off_road": 0, "not_outlier": 0,
                      "unknown_kept": 0})
        return line.copy(), stats
    k = max(1, int(on_road_dilate_px))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    # A missing road reference is UNKNOWN, not "off the road": with no road
    # to judge against, deleting line evidence would turn a perception
    # outage into a confident "no lines" (and the plan's rule is that
    # unknown must not read as a negative).
    road_missing = stats["road_px"] < int(min_ref_px)
    stats["road_reference"] = "missing" if road_missing else "ok"
    road_near = (np.ones_like(road) if road_missing else
                 cv2.dilate(road.astype(np.uint8), kernel).astype(bool))
    struct_ok = line & road_near
    stats["off_road"] = int((line & ~road_near).sum())
    if not struct_ok.any():
        stats.update({"kept": 0, "not_outlier": 0, "unknown_kept": 0})
        return np.zeros_like(line), stats
    # LOCAL road appearance: masked box statistics inside the window
    w = max(3, int(window_px) | 1)
    road_f = road.astype(np.float32)
    cnt = cv2.boxFilter(road_f, -1, (w, w), normalize=False)
    s1 = cv2.boxFilter(grey * road_f, -1, (w, w), normalize=False)
    s2 = cv2.boxFilter(grey * grey * road_f, -1, (w, w), normalize=False)
    ref_ok = cnt >= float(min_ref_px)
    mean = np.where(ref_ok, s1 / np.maximum(cnt, 1.0), 0.0)
    var = np.where(ref_ok, np.maximum(s2 / np.maximum(cnt, 1.0) - mean * mean,
                                      0.0), 0.0)
    scale = np.maximum(np.sqrt(var), float(scale_floor))
    z = np.abs(grey - mean) / scale
    outlier_ok = (z >= float(z_min)) | ~ref_ok
    keep = struct_ok & outlier_ok
    stats["unknown_kept"] = int((struct_ok & ~ref_ok).sum())
    stats["not_outlier"] = int((struct_ok & ref_ok & ~outlier_ok).sum())
    stats["kept"] = int(keep.sum())
    stats["z_p50_kept"] = (None if stats["kept"] == 0 else
                           round(float(np.median(z[keep])), 3))
    return keep, stats
