"""Traffic-signal vision head (FSD-style light/color understanding).

FSD's HydraNet has a dedicated traffic-light head that reads the light
colour from the camera so the planner reacts to it.  This head gives the
project the same *shape*: it scans a frame's top region for signal
colours (red / yellow / green blobs, an extremely reliable cue in
simulation) and emits a ``TrafficVisionOutput`` suggestion that a
planner layer can fuse with the game's authoritative ``SignalRule``.

Two deliberate layers:

* ``suggest_signal_state`` - pure colour -> state logic (unit-testable,
  game-free): given an RGB frame and an optional lower-row ROI it returns
  "red" / "yellow" / "green" / "none".
* ``TrafficSignalHead`` - a HydraNet head wrapping that logic into the
  shared ``TaskOutput`` contract (its ``meta`` carries ``signal_state``
  and ``signal_conf``).

The output convention matches ``signal_action_label`` from ``traffic.py``
("red" / "yellow" / "green") so the two sources are directly comparable.
"""

from __future__ import annotations

import numpy as np

from ..hydra import FrameContext, TaskOutput

# Signal colour thresholds (HSV, tuned for the italy daytime palette).
RED_HUE = ((0, 10), (170, 180))
YELLOW_HUE = (12, 40)
GREEN_HUE = (45, 90)
MIN_SAT = 90
MIN_VAL = 90
MIN_BLOB_PX = 6


def _color_mask(hsv: np.ndarray) -> dict[str, np.ndarray]:
    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)
    sat_ok = s >= MIN_SAT
    val_ok = v >= MIN_VAL
    masks = {}
    red = np.zeros(h.shape, dtype=bool)
    for lo, hi in RED_HUE:
        red |= (h >= lo) & (h <= hi)
    masks["red"] = red & sat_ok & val_ok
    masks["yellow"] = ((h >= YELLOW_HUE[0]) & (h <= YELLOW_HUE[1])
                       & sat_ok & val_ok)
    masks["green"] = ((h >= GREEN_HUE[0]) & (h <= GREEN_HUE[1])
                      & sat_ok & val_ok)
    return masks


def suggest_signal_state_px(frame_rgb: np.ndarray,
                            bottom_share: float = 0.30
                            ) -> tuple[str, float, tuple[float, float] | None]:
    """Return ``(state, confidence, centroid_px)`` for the visible light.

    Same scan as :func:`suggest_signal_state` but also returns the
    winning colour blob's centroid ``(u, v)`` in pixel coordinates
    (None when no lamp is found) so callers can place the signal in
    space (the BEV ``sign`` channel stamps the lamp's bearing).
    """
    import cv2
    if frame_rgb is None or frame_rgb.size == 0:
        return "none", 0.0
    h, w = frame_rgb.shape[:2]
    roi = frame_rgb[: int(h * (1.0 - bottom_share)), :, :]
    if roi.size == 0:
        return "none", 0.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    masks = _color_mask(hsv)

    # A signal lamp is a compact blob, not a full-frame colour cast.  A
    # dusky/amber scene makes the whole frame yellow-ish, so a naive
    # "most pixels wins" would flag "yellow" on every overcast frame.
    # Require (a) the colour to be spatially compact (a lamp occupies a
    # small fraction of the upper frame; a global cast covers far more),
    # and (b) one colour to dominate the others (a real light is one
    # colour).
    frame_px = float(roi.shape[0] * roi.shape[1])
    max_lamp_share = 0.06   # a real lamp < ~6% of the upper frame
    best_state, best_conf = "none", 0.0
    counts = {s: float(mask.sum()) for s, mask in masks.items()}
    total = sum(counts.values())
    for state, mask in masks.items():
        px = int(mask.sum())
        if px < MIN_BLOB_PX:
            continue
        # (a) compactness: the colour covers a small share of the frame.
        if px / frame_px > max_lamp_share:
            continue
        # (b) dominance: this colour clearly beats the next one.
        share = px / max(1.0, total)
        if share < 0.6:
            continue
        # confidence from blob size + dominance
        conf = min(1.0, 0.4 + 0.4 * share)
        if conf > best_conf:
            best_state, best_conf = state, conf
    centroid = None
    if best_state != "none":
        ys, xs = np.nonzero(masks[best_state])
        if len(xs):
            centroid = (float(xs.mean()), float(ys.mean()))
    return best_state, best_conf, centroid


def suggest_signal_state(frame_rgb: np.ndarray,
                         bottom_share: float = 0.30) -> tuple[str, float]:
    """Return ``(state, confidence)`` for the light visible in the frame.

    Scans the frame for the largest colour blob; ``bottom_share`` limits
    the scan to the upper part of the frame (lights hang overhead) to
    avoid asphalt/brake-light contamination.  state is one of
    "red"/"yellow"/"green"/"none".
    """
    state, conf, _ = suggest_signal_state_px(frame_rgb, bottom_share)
    return state, conf


class TrafficSignalHead:
    """HydraNet head emitting a traffic-light state suggestion."""

    name = "traffic"

    def __init__(self, bottom_share: float = 0.30):
        self.bottom_share = float(bottom_share)

    def run(self, ctx: FrameContext) -> TaskOutput:
        out = TaskOutput()
        state, conf, px = suggest_signal_state_px(
            ctx.frame_rgb, bottom_share=self.bottom_share)
        out.meta["signal_state"] = state
        out.meta["signal_conf"] = conf
        out.meta["signal_px"] = px
        return out


def merge_signal_vision(rule_state: str | None,
                        vision_state: str,
                        vision_conf: float,
                        trust_vision_conf: float = 0.75
                        ) -> tuple[str, str]:
    """Fuse the game signal state with the vision suggestion.

    Returns ``(final_state, source)`` where source is "rule" (game),
    "vision", or "vision+rule" when they agree.  A confident visual
    reading overrides a missing/unknown rule; when the rule knows the
    state it wins (the game is authoritative - this is the FSD
    "sensor + map fusion" split, where the map is ground truth and
    sensor fills the gaps).
    """
    rule_state = (rule_state or "").lower()
    vision_state = (vision_state or "").lower()
    if rule_state in ("red", "yellow", "green"):
        return rule_state, ("vision+rule" if rule_state == vision_state
                            else "rule")
    if vision_conf >= trust_vision_conf and vision_state in (
            "red", "yellow", "green"):
        return vision_state, "vision"
    return "none", "none"