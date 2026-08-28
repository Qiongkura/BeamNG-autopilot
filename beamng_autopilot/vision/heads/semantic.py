"""Semantic (road / lane-line) task head.

Wraps the learned ``Segmenter`` (road + line masks) and the classic
``LaneDetector`` lane pipeline into one HydraNet head: given one frame
it returns the road / line / offroad masks plus the world-space lane
markings the planner consumes.
"""

from __future__ import annotations

import numpy as np

from ..hydra import FrameContext, TaskOutput


class SemanticHead:
    """Road / painted-line segmentation + lane markings task head."""

    name = "semantic"

    def __init__(self, segmenter=None, lane_detector=None):
        # Either may be None: lazily import/construct only when actually
        # used so the head stays usable in offline unit tests without a
        # trained model or a game.
        self.segmenter = segmenter
        self.lane_detector = lane_detector
        self._lane_fallback = None

    def _get_segmenter(self):
        if self.segmenter is None:
            from ..segmentation import Segmenter
            self.segmenter = Segmenter()
        return self.segmenter

    def _get_lanes(self):
        if self.lane_detector is None:
            from ..lanes import LaneDetector
            self.lane_detector = LaneDetector()
        return self.lane_detector

    def run(self, ctx: FrameContext) -> TaskOutput:
        out = TaskOutput()
        h, w = ctx.frame_rgb.shape[:2]
        road = np.ones((h, w), dtype=bool)
        line = np.zeros((h, w), dtype=bool)
        markings: list = []
        try:
            seg = self._get_segmenter()
            road, line = seg.predict(ctx.frame_rgb)
        except Exception:
            # No trained model / inference error: the planner simply
            # has no sensor lane this frame (existing fallback).
            road = np.ones((h, w), dtype=bool)
            line = np.zeros((h, w), dtype=bool)
        out.masks["road"] = road
        out.masks["line"] = line
        out.masks["offroad"] = ~road
        # Lane markings in world space (only meaningful on the front
        # camera; other roles leave this empty).
        if ctx.role == "front_main":
            try:
                seg = self._get_segmenter()
                markings = seg.detect_lines(
                    ctx.frame_rgb, ctx.cam, ctx.pos, ctx.heading,
                    ground_z=ctx.ground_z)
            except Exception:
                markings = []
            if not markings and not line.any():
                # Model unavailable or predicted no paint: fall back to the
                # classic-CV colour-threshold detector so the planner still
                # receives lane markings instead of an empty sensor lane.
                try:
                    markings = self._get_lanes().detect(
                        ctx.frame_rgb, ctx.cam, ctx.pos, ctx.heading,
                        ground_z=ctx.ground_z)
                except Exception:
                    markings = []
        out.meta["markings"] = markings
        return out