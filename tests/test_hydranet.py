"""Offline tests for the HydraNet shared-backbone multi-task structure."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from beamng_autopilot.vision.hydra import (
    FrameContext,
    HydraNet,
    TaskOutput,
)


def _ctx(role="front_main", h=64, w=96):
    rng = np.random.default_rng(0)
    from beamng_autopilot.vision.projection import CameraModel
    return FrameContext(
        frame_rgb=rng.integers(0, 255, (h, w, 3), dtype=np.uint8),
        cam=CameraModel(np.array([0.0, 1.0, 1.4]),
                        np.array([0.0, 1.0, 0.0]),
                        np.array([0.0, 0.0, 1.0]), 65.0, w, h),
        pos=np.array([0.0, 0.0, 0.0]),
        heading=0.0,
        role=role,
    )


def test_task_output_mask_resize() -> None:
    out = TaskOutput()
    out.masks["road"] = np.ones((32, 48), dtype=bool)
    m = out.mask("road", 64, 96)
    assert m is not None and m.shape == (64, 96)
    assert m.all()
    assert out.mask("nope") is None


def test_hydranet_records_all_head_outputs() -> None:
    class AHead:
        name = "a"

        def run(self, ctx):
            return TaskOutput(meta={"marks": [1, 2]})

    class BHead:
        name = "b"

        def run(self, ctx):
            return TaskOutput(obstacles=[object()])

    net = HydraNet()
    net.add(AHead())
    net.add(BHead())
    assert net.names() == ("a", "b")
    out = net.run(_ctx())
    assert set(out.keys()) == {"a", "b"}
    assert out["a"].meta["marks"] == [1, 2]
    assert len(out["b"].obstacles) == 1
    assert net.errors == {}


def test_hydranet_head_failure_isolated() -> None:
    class BadHead:
        name = "bad"

        def run(self, ctx):
            raise RuntimeError("boom")

    net = HydraNet()

    class GoodHead:
        name = "good"

        def run(self, ctx):
            return TaskOutput()

    net.add(BadHead())
    net.add(GoodHead())
    out = net.run(_ctx())
    assert "bad" in net.errors and "boom" in net.errors["bad"]
    assert "good" in out and net.errors.get("good") is None


def test_semantic_head_masks_and_markings() -> None:
    from beamng_autopilot.vision.heads.semantic import SemanticHead

    class FakeSeg:
        def predict(self, frame):
            h, w = frame.shape[:2]
            road = np.ones((h, w), dtype=bool)
            road[: h // 2] = False
            line = np.zeros((h, w), dtype=bool)
            line[h // 2 :, w // 2 - 2 : w // 2 + 2] = True
            return road, line

        def detect_lines(self, *a, **k):
            return [("mark", 1.0)]

    head = SemanticHead(segmenter=FakeSeg())
    out = head.run(_ctx())
    assert "road" in out.masks and "line" in out.masks
    assert "offroad" in out.masks
    assert out.meta["markings"] == [("mark", 1.0)]


def test_semantic_head_absent_model_falls_back() -> None:
    """Without a trained model the semantic head must not crash: it returns
    an all-road mask and falls back to classic-CV lane detection."""
    from beamng_autopilot.vision.heads.semantic import SemanticHead

    head = SemanticHead(segmenter="missing")  # never used; forces fallback
    out = head.run(_ctx())
    assert out.masks["road"].dtype == bool
    assert out.masks["road"].all()
    assert isinstance(out.meta["markings"], list)


def test_semantic_head_fallback_uses_classic_lanes() -> None:
    """When the segmentation model is absent the head must call the
    classic-CV LaneDetector and surface its markings to the planner."""
    from beamng_autopilot.vision.heads.semantic import SemanticHead

    class FakeLanes:
        def detect(self, *a, **k):
            return [("cv-mark", 2.0)]

    head = SemanticHead(segmenter="missing", lane_detector=FakeLanes())
    out = head.run(_ctx())
    assert out.meta["markings"] == [("cv-mark", 2.0)]


def test_object_head_skips_disabled_roles() -> None:
    from beamng_autopilot.vision.heads.object import ObjectHead

    class FakeDet:
        def detect(self, *a, **k):
            return [object()], [(0, 0, 1, 1, "car", 0.9)]

    head = ObjectHead(detector=FakeDet(), enabled_roles=("front_main",))
    out = head.run(_ctx(role="rear"))
    assert out.obstacles == [] and out.boxes == []


def test_object_head_runs_on_enabled_role() -> None:
    from beamng_autopilot.vision.heads.object import ObjectHead

    class FakeDet:
        def detect(self, *a, **k):
            return [object()], [(0, 0, 1, 1, "car", 0.9)]

    head = ObjectHead(detector=FakeDet(), enabled_roles=("front_main",))
    out = head.run(_ctx(role="front_main"))
    assert len(out.obstacles) == 1 and len(out.boxes) == 1