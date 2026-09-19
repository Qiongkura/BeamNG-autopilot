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
