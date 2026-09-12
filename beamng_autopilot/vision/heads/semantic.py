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

    def __init__(self, segmenter=None, lane_detector=None,
                 enable_evidence: bool = True):
        # Either may be None: lazily import/construct only when actually
        # used so the head stays usable in offline unit tests without a
        # trained model or a game.
        self.segmenter = segmenter
        self.lane_detector = lane_detector
        self._lane_fallback = None
        # Multi-frame line evidence (world-space accumulation); front
        # camera only.  Fills single-frame line dropouts from recent
        # sightings without propagating unconfirmed single-frame noise.
        self.enable_evidence = bool(enable_evidence)
        self._evidence = None

    def reset(self) -> None:
        """Drop accumulated line evidence (location-bound; teleport)."""
        if self._evidence is not None:
            self._evidence.reset()
        if self.segmenter is not None and hasattr(self.segmenter, "reset"):
            self.segmenter.reset()

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
        prediction_ok = False
        try:
            seg = self._get_segmenter()
            road, line = seg.predict(ctx.frame_rgb)
            prediction_ok = True
        except Exception as exc:
            self.reset()
            out.meta.setdefault("line_errors", {})["predict"] = str(exc)
            # No trained model / inference error: the planner simply
            # has no sensor lane this frame (existing fallback).
            road = np.ones((h, w), dtype=bool)
            line = np.zeros((h, w), dtype=bool)
        raw_line = line
        out.meta["line_pixels_raw"] = int(np.count_nonzero(raw_line))
        if prediction_ok and self.enable_evidence and ctx.role == "front_main":
            try:
                if self._evidence is None:
                    from ..line_evidence import LineEvidenceAccumulator
                    self._evidence = LineEvidenceAccumulator()
                line = self._evidence.fuse(
                    line, ctx.cam, ctx.pos, ctx.heading, ctx.ground_z,
                    now=ctx.timestamp)
            except Exception as exc:
                if self._evidence is not None:
                    self._evidence.reset()
                out.meta.setdefault("line_errors", {})["evidence"] = str(exc)
        out.meta["line_pixels_added"] = int(np.count_nonzero(line & ~raw_line))
        out.masks["road"] = road
        out.masks["line"] = line
        out.masks["offroad"] = ~road
        # Lane markings in world space (only meaningful on the front
        # camera; other roles leave this empty).
        if ctx.role == "front_main":
            if prediction_ok:
                try:
                    markings = seg.detect_lines(
                        ctx.frame_rgb, ctx.cam, ctx.pos, ctx.heading,
                        ground_z=ctx.ground_z, line_mask=line)
                except Exception as exc:
                    out.meta.setdefault("line_errors", {})["markings"] = str(exc)
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