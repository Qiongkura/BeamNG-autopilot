"""Yellow paint unions into SemanticHead LINE channel."""

from __future__ import annotations

import numpy as np

from beamng_autopilot.vision.heads.semantic import SemanticHead
from beamng_autopilot.vision.hydra import FrameContext


class _SegEmpty:
    def predict(self, frame):
        return (np.ones(frame.shape[:2], bool),
                np.zeros(frame.shape[:2], bool))


def _yellow_frame(h=120, w=160):
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:] = (80, 80, 80)
    rgb[h // 2:] = (200, 180, 20)  # HSV yellow band
    return rgb


def test_yellow_union_adds_line_pixels():
    rgb = _yellow_frame()
    ctx = FrameContext(frame_rgb=rgb, cam=None, pos=(0.0, 0.0, 0.0),
                       heading=0.0, role="front_main")
    out = SemanticHead(segmenter=_SegEmpty(), enable_evidence=False).run(ctx)
    assert out.meta["line_pixels_yellow"] > 0
    assert np.count_nonzero(out.masks["line"]) > np.count_nonzero(
        out.meta["line_pixels_raw"] and 0 or 0)
    assert np.any(out.masks["line"])
