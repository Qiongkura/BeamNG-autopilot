"""Offline tests for the traffic-signal vision head."""

from __future__ import annotations

import numpy as np
import pytest

from beamng_autopilot.vision.heads.traffic import (
    TrafficSignalHead,
    merge_signal_vision,
    suggest_signal_state,
)
from beamng_autopilot.vision.hydra import FrameContext


def _signal_frame(hue: int | None, w=80, h=60) -> np.ndarray:
    """All-black frame with a coloured blob in the top-right corner."""
    import cv2
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    if hue is not None:
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 0] = int(hue)
        hsv[..., 1] = 255
        hsv[..., 2] = 255
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        frame[5:15, w - 20:w - 5] = rgb[5:15, w - 20:w - 5]
    return frame


def test_suggest_red_yellow_green() -> None:
    assert suggest_signal_state(_signal_frame(0))[0] == "red"
    assert suggest_signal_state(_signal_frame(15))[0] == "yellow"
    assert suggest_signal_state(_signal_frame(50))[0] == "green"


def test_suggest_none_on_plain_frame() -> None:
    frame = np.zeros((60, 80, 3), dtype=np.uint8)
    state, conf = suggest_signal_state(frame)
    assert state == "none"
    assert conf == 0.0


def test_signal_head_meta() -> None:
    from beamng_autopilot.vision.projection import CameraModel
    head = TrafficSignalHead()
    ctx = FrameContext(frame_rgb=_signal_frame(0),
                       cam=CameraModel(np.zeros(3), np.array([0., 1., 0.]),
                                       np.array([0., 0., 1.]), 65., 80, 60),
                       pos=np.array([0., 0., 0.]), heading=0.0)
    out = head.run(ctx)
    assert out.meta["signal_state"] == "red"
    assert out.meta["signal_conf"] > 0.5


def test_merge_rule_authoritative() -> None:
    assert merge_signal_vision("red", "green", 0.9)[0] == "red"
    assert merge_signal_vision("red", "red", 0.8)[1] == "vision+rule"
    assert merge_signal_vision("red", "red", 0.8)[0] == "red"


def test_merge_vision_fills_unknown() -> None:
    state, src = merge_signal_vision(None, "green", 0.85)
    assert state == "green" and src == "vision"
    state, src = merge_signal_vision("", "green", 0.5)
    assert state == "none"  # low confidence vision ignored