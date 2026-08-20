"""FSD-style shared backbone + HydraNet multi-task heads.

Tesla FSD (AI Day 2021/2022) runs *one* shared vision backbone over the
camera ring and attaches several task heads to its features - a
HydraNet - so every task shares early features instead of each algorithm
re-reading the frame.  This module gives the project the same *shape*:

* ``TaskOutput``: the uniform result container every head returns
  (frame-relative masks/boxes plus world-space obstacles a planner can
  consume).  The planner never sees raw pixels - only these ready
  outputs.
* ``HydraHead``: the interface contract a task head implements.
  ``run(frame_ctx) -> TaskOutput``.  ``frame_ctx`` is a small context
  carrying the frame, its CameraModel and the vehicle pose so a head can
  stay self-contained.
* The existing perception components (``Segmenter`` semantics, YOLO
  ``VisionDetector`` objects, lane detector) are *wrapped* as heads
  rather than rewritten, so current behaviour and the 94.6% driving
  result are preserved.
* ``HydraNet``: the shared-backbone registry.  It owns the *frame* and
  hands it to every registered head once per call, emulating "features
  computed once, many task heads" - when the heads later move to a real
  shared CNN the registry contract stays identical.

Anything in here is game-free and unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class TaskOutput:
    """Uniform output of one HydraNet task head.

    ``masks`` is a dict of frame-resolution boolean masks indexed by
    semantic name (e.g. "road", "line", "offroad").  ``boxes`` are
    pixel-space detections ``(x1, y1, x2, y2, label, conf)``.
    ``obstacles`` are world-space ``perception.Obstacle`` list.
    ``meta`` holds head-specific extras (lane markings, confidences...).
    """

    masks: dict[str, np.ndarray] = field(default_factory=dict)
    boxes: list[tuple] = field(default_factory=list)
    obstacles: list = field(default_factory=list)
    # Extra structured data, e.g. {"markings": [...]}.
    meta: dict = field(default_factory=dict)

    def mask(self, name: str, h: int | None = None, w: int | None = None
             ) -> np.ndarray | None:
        m = self.masks.get(name)
        if m is None or h is None or w is None:
            return m
        if m.shape[:2] == (h, w):
            return m
        import cv2
        return cv2.resize(m.astype(np.uint8), (w, h),
                          interpolation=cv2.INTER_NEAREST).astype(bool)


@dataclass
class FrameContext:
    """Everything a head needs about one camera frame.

    ``pos`` / ``heading`` are the world pose at capture time; heads that
    back-project to the ground use ``cam`` + these.  ``role`` names the
    ring camera the frame came from (front_main / pillar_left / ...).
    """

    frame_rgb: np.ndarray
    cam: object  # CameraModel
    pos: np.ndarray | tuple
    heading: float
    rotation: tuple | None = None
    ground_z: float = 0.0
    role: str = "front_main"


class HydraHead(Protocol):
    """Contract every task head implements."""

    name: str

    def run(self, ctx: FrameContext) -> TaskOutput: ...


class HydraNet:
    """Shared-backbone multi-task registry.

    ``run(ctx)`` forwards the one frame to every registered head in
    registration order and returns ``{head.name: TaskOutput}``, matching
    the HydraNet "shared features, many heads" data flow.  A head that
    fails is recorded in ``errors`` instead of taking the whole pipeline
    down, exactly as a real multi-task net stays alive on a failing head.
    """

    def __init__(self, heads=None):
        self._heads: dict[str, HydraHead] = {}
        self.errors: dict[str, str] = {}

    def add(self, head: HydraHead) -> None:
        self._heads[head.name] = head
        self.errors.pop(head.name, None)

    def __contains__(self, name: str) -> bool:
        return name in self._heads

    def names(self) -> tuple[str, ...]:
        return tuple(self._heads.keys())

    def run(self, ctx: FrameContext) -> dict[str, TaskOutput]:
        out: dict[str, TaskOutput] = {}
        for name, head in self._heads.items():
            try:
                out[name] = head.run(ctx)
            except Exception as exc:
                self.errors[name] = str(exc)
        return out