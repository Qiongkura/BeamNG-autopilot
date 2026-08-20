"""Object (vehicles / pedestrians / obstacles) task head.

Wraps the existing YOLO ``VisionDetector`` into a HydraNet head: given
one frame it returns the pixel-space boxes plus world-space obstacles.
On the ring's non-front cameras the detector still runs (FSD detects
lane-change / blind-spot targets from the side cameras too); the caller
decides which roles to enable.
"""

from __future__ import annotations

from ..hydra import FrameContext, TaskOutput


class ObjectHead:
    """Vehicle / pedestrian / obstacle detection task head."""

    name = "object"

    def __init__(self, detector=None, enabled_roles=("front_main",)):
        self.detector = detector
        self.enabled_roles = frozenset(enabled_roles)

    def _get_detector(self):
        if self.detector is None:
            from ..detection import VisionDetector
            self.detector = VisionDetector()
        return self.detector

    def run(self, ctx: FrameContext) -> TaskOutput:
        out = TaskOutput()
        if ctx.role not in self.enabled_roles:
            return out
        try:
            det = self._get_detector()
            obstacles, boxes = det.detect(
                ctx.frame_rgb, ctx.cam, ctx.pos, ctx.heading)
            out.obstacles = obstacles
            out.boxes = boxes
        except Exception:
            out.obstacles = []
            out.boxes = []
        return out