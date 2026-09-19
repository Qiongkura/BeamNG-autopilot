"""Fine-grained marking classes (plan E5)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from beamng_autopilot.lane.marking_class import (
    MARK_CURB,
    MARK_DASHED_WHITE,
    MARK_EDGE_MARGIN_M,
    MARK_NOT_PAINT,
    MARK_PAINT,
    MARK_ROADSIDE_ARTIFACT,
    MARK_SOLID_WHITE,
    MARK_UNKNOWN,
    MARK_WHITE_EDGE,
    MARK_YELLOW_CENTER,
    classify_marking,
    classify_marking_object,
)


# ---------------------------------------------------------------------------
# the taxonomy's contract
# ---------------------------------------------------------------------------

def test_paint_and_non_paint_partition_the_classes() -> None:
    assert set(MARK_PAINT).isdisjoint(MARK_NOT_PAINT)
    assert MARK_UNKNOWN not in MARK_PAINT + MARK_NOT_PAINT


def test_only_paint_may_bound_a_lane_as_a_marking() -> None:
    from beamng_autopilot.lane.marking_class import MarkingClassResult
    assert MarkingClassResult(MARK_SOLID_WHITE).is_paint
    assert not MarkingClassResult(MARK_CURB).is_paint
    assert MarkingClassResult(MARK_CURB).usable_as_lane_boundary
    assert not MarkingClassResult(MARK_ROADSIDE_ARTIFACT) \
        .usable_as_lane_boundary
    assert not MarkingClassResult(MARK_UNKNOWN).usable_as_lane_boundary


# ---------------------------------------------------------------------------
# paint classes
# ---------------------------------------------------------------------------

def test_solid_white_from_a_continuous_stroke() -> None:
    r = classify_marking(color="white", kind="solid", span_m=8.0,
                         gap_ratio=0.02)
    assert r.mark_class == MARK_SOLID_WHITE
    assert r.is_paint and r.confidence > 0.7


def test_dashed_white_from_the_gap_ratio() -> None:
    r = classify_marking(color="white", kind="solid", span_m=12.0,
                         gap_ratio=0.4)
    assert r.mark_class == MARK_DASHED_WHITE
    assert "gappy_stroke" in r.reasons


def test_long_solid_line_with_one_dropout_stays_solid() -> None:
    """The threshold must not turn a solid line into a dashed one."""
    r = classify_marking(color="white", kind="solid", span_m=20.0,
                         gap_ratio=0.1)
    assert r.mark_class == MARK_SOLID_WHITE


def test_dashed_white_from_the_extractor_kind() -> None:
    r = classify_marking(color="white", kind="dashed", span_m=10.0)
    assert r.mark_class == MARK_DASHED_WHITE


def test_yellow_is_the_centre_line() -> None:
    r = classify_marking(color="yellow", kind="solid", span_m=9.0)
    assert r.mark_class == MARK_YELLOW_CENTER
    assert r.is_paint


def test_white_edge_needs_the_own_lane_width() -> None:
    """Beyond the own-lane half width the paint is the lane EDGE."""
    r = classify_marking(color="white", kind="solid", span_m=14.0,
                         lateral_m=2.4, lane_half_m=1.75)
    assert r.mark_class == MARK_WHITE_EDGE
    assert "outside_own_lane_half" in r.reasons
    # inside the lane it is ordinary paint
    r2 = classify_marking(color="white", kind="solid", span_m=14.0,
                          lateral_m=1.0, lane_half_m=1.75)
    assert r2.mark_class == MARK_SOLID_WHITE
    # exactly at the margin is still inside (>= is what promotes it)
    r3 = classify_marking(color="white", kind="solid", span_m=14.0,
                          lateral_m=1.75 + MARK_EDGE_MARGIN_M - 0.01,
                          lane_half_m=1.75)
    assert r3.mark_class == MARK_SOLID_WHITE


def test_yellow_outranks_the_edge_rule() -> None:
    """A yellow line far out is still centre paint (colour beats offset)."""
    r = classify_marking(color="yellow", kind="solid", span_m=9.0,
                         lateral_m=3.0, lane_half_m=1.75)
    assert r.mark_class == MARK_YELLOW_CENTER


# ---------------------------------------------------------------------------
# non-paint classes
# ---------------------------------------------------------------------------

def test_off_pavement_far_out_is_a_roadside_artifact() -> None:
    r = classify_marking(color="white", kind="dashed", span_m=6.0,
                         off_pavement_m=2.5, on_drivable=False)
    assert r.mark_class == MARK_ROADSIDE_ARTIFACT
    assert not r.is_paint
    assert not r.usable_as_lane_boundary


def test_a_fragment_off_pavement_is_an_artifact_not_a_lane_line() -> None:
    """The red-white reflector post case: gaps must not make it dashed."""
    r = classify_marking(color="white", kind="dashed", span_m=1.0,
                         gap_ratio=0.5, off_pavement_m=0.3,
                         on_drivable=False)
    assert r.mark_class == MARK_ROADSIDE_ARTIFACT
    assert "fragment_off_road" in r.reasons


def test_a_kerb_just_off_the_pavement_is_a_curb() -> None:
    r = classify_marking(color="white", kind="solid", span_m=9.0,
                         off_pavement_m=0.3, on_drivable=False,
                         raised_height_m=0.12)
    assert r.mark_class == MARK_CURB
    assert r.usable_as_lane_boundary and not r.is_paint


def test_raise_alone_off_road_is_a_curb() -> None:
    r = classify_marking(raised_height_m=0.1, on_drivable=False)
    assert r.mark_class == MARK_CURB
    assert "raised_edge" in r.reasons


def test_on_road_marks_are_never_curbs() -> None:
    """A raised thing ON the drivable surface is not an edge treatment."""
    r = classify_marking(color="white", kind="solid", span_m=9.0,
                         raised_height_m=0.1, on_drivable=True)
    assert r.mark_class == MARK_SOLID_WHITE


# ---------------------------------------------------------------------------
# honest unknowns and digests
# ---------------------------------------------------------------------------

def test_no_evidence_is_unknown_not_a_default_class() -> None:
    r = classify_marking()
    assert r.mark_class == MARK_UNKNOWN
    assert not r.usable_as_lane_boundary
    r2 = classify_marking(color="unknown", kind="unknown")
    assert r2.mark_class == MARK_UNKNOWN


def test_a_solid_extractor_read_without_colour_still_classifies() -> None:
    r = classify_marking(kind="solid", span_m=7.0)
    assert r.mark_class == MARK_SOLID_WHITE
    assert r.confidence < 0.6, "it must not claim colour-level certainty"


def test_digest_is_json_safe() -> None:
    r = classify_marking(color="white", kind="solid", span_m=8.0,
                         gap_ratio=0.02, lateral_m=None)
    text = json.dumps(r.digest())
    assert "solid_white" in text and "nan" not in text.lower()


def test_nan_evidence_never_promotes_a_class() -> None:
    r = classify_marking(color="white", kind="solid", span_m=float("nan"),
                         gap_ratio=float("nan"), lateral_m=float("nan"),
                         lane_half_m=float("nan"))
    assert r.mark_class == MARK_SOLID_WHITE      # the coarse kind still says so
    r2 = classify_marking(span_m=float("nan"), gap_ratio=float("nan"))
    assert r2.mark_class == MARK_UNKNOWN


# ---------------------------------------------------------------------------
# object adapter
# ---------------------------------------------------------------------------

class _Marking:
    def __init__(self, world, color="white", kind="solid"):
        self.world = np.asarray(world, dtype=float)
        self.color = color
        self.kind = kind


def test_object_adapter_computes_the_span() -> None:
    mk = _Marking(np.column_stack([np.linspace(0.0, 10.0, 21),
                                   np.zeros(21)]))
    r = classify_marking_object(mk, gap_ratio=0.0)
    assert r.mark_class == MARK_SOLID_WHITE
    assert r.metrics["span_m"] == pytest.approx(10.0, abs=0.1)


def test_object_adapter_survives_a_missing_world() -> None:
    class _Bare:
        color = "yellow"
        kind = "solid"
    r = classify_marking_object(_Bare())
    assert r.mark_class == MARK_YELLOW_CENTER


# ---------------------------------------------------------------------------
# wiring: the semantic head refines and gates the extracted markings
# ---------------------------------------------------------------------------

def _head_with_markings(monkeypatch, markings, enabled: bool,
                        road_value: bool = True, cam=None):
    """Run the head with stubbed markings and a chosen road mask.

    ``cam=None`` means "no drivable evidence at all" (the classifier must
    then treat the markings as unknown, not as contradicted);
    ``road_value=False`` with a camera means "the road mask says this is
    not drivable".
    """
    import beamng_autopilot.vision.heads.semantic as sem_mod
    from beamng_autopilot.vision.heads.semantic import SemanticHead
    from beamng_autopilot.vision.hydra import FrameContext

    class _Seg:
        def predict(self, frame):
            h, w = frame.shape[:2]
            return (np.full((h, w), bool(road_value), dtype=bool),
                    np.zeros((h, w), dtype=bool))

        def detect_lines(self, *a, **kw):
            return list(markings)

    monkeypatch.setattr(sem_mod, "MARK_CLASS_ENABLED", enabled)
    monkeypatch.setenv("BEAMNG_YELLOW_FUSION", "0")
    monkeypatch.setenv("BEAMNG_DASHED_RECOVERY", "0")
    head = SemanticHead(segmenter=_Seg(), enable_evidence=False)
    ctx = FrameContext(frame_rgb=np.zeros((40, 40, 3), np.uint8), cam=cam,
                       pos=(0.0, 0.0, 0.0), heading=0.0)
    return head.run(ctx)


def _camera():
    from beamng_autopilot.vision.projection import CameraModel
    return CameraModel(offset=np.array([0.0, 1.2, 1.4]),
                       fwd_local=np.array([0.0, 1.0, 0.0]),
                       up_local=np.array([0.0, 0.0, 1.0]),
                       fov_deg=65.0, width=40, height=40)


def test_wiring_is_off_by_default(monkeypatch) -> None:
    mk = _Marking(np.column_stack([np.linspace(0.0, 8.0, 9), np.zeros(9)]))
    out = _head_with_markings(monkeypatch, [mk], enabled=False)
    assert "mark_class" not in out.meta
    assert len(out.meta["markings"]) == 1
    assert not hasattr(mk, "mark_class")


def test_wiring_labels_every_marking(monkeypatch) -> None:
    solid = _Marking(np.column_stack([np.linspace(0.0, 8.0, 9), np.zeros(9)]))
    yellow = _Marking(np.column_stack([np.linspace(0.0, 9.0, 10),
                                       np.full(10, 1.0)]), color="yellow")
    out = _head_with_markings(monkeypatch, [solid, yellow], enabled=True)
    # the wiring must not be hiding behind its own error channel: a
    # swallowed ImportError here looked exactly like "the switch is off"
    assert "mark_class" not in (out.meta.get("line_errors") or {}),         out.meta.get("line_errors")
    counts = out.meta["mark_class"]
    assert counts.get(MARK_SOLID_WHITE) == 1
    assert counts.get(MARK_YELLOW_CENTER) == 1
    assert solid.mark_class == MARK_SOLID_WHITE
    assert yellow.mark_class == MARK_YELLOW_CENTER
    assert len(out.meta["markings"]) == 2      # both are paint: both kept


def test_wiring_without_drivable_evidence_drops_nothing(monkeypatch) -> None:
    """Unknown evidence is not contradicted evidence (no camera: keep)."""
    mk = _Marking(np.column_stack([np.linspace(0.0, 8.0, 9), np.zeros(9)]))
    out = _head_with_markings(monkeypatch, [mk], enabled=True, cam=None)
    assert len(out.meta["markings"]) == 1
    assert mk.mark_class == MARK_SOLID_WHITE


def test_wiring_drops_marks_the_road_mask_refuses(monkeypatch) -> None:
    """With drivable evidence saying "not on the road", nothing is paint.

    This is the fail-closed side of E5: when the road mask refuses the
    marking's own footprint, it must not become lane geometry - and the
    telemetry still counts what was classified.
    """
    mk = _Marking(np.column_stack([np.linspace(2.0, 8.0, 7), np.zeros(7)]))
    out = _head_with_markings(monkeypatch, [mk], enabled=True,
                              road_value=False, cam=_camera())
    assert "mark_class" not in (out.meta.get("line_errors") or {}),         out.meta.get("line_errors")
    assert out.meta["markings"] == []
    assert out.meta["mark_class"].get(MARK_ROADSIDE_ARTIFACT, 0) >= 1


def test_off_drivable_alone_is_already_a_refusal() -> None:
    """The evidence the head can actually produce must be sufficient.

    A marking whose own points project outside the drivable surface is
    refused even without a measured distance off the pavement: the
    distance only separates a plausible kerb from clutter, and a kerb
    needs the raised / adjacency evidence on top.
    """
    r = classify_marking(color="white", kind="dashed", span_m=6.0,
                         on_drivable=False)
    assert r.mark_class == MARK_ROADSIDE_ARTIFACT
    assert "off_drivable" in r.reasons
    r2 = classify_marking(color="white", kind="solid", span_m=9.0,
                          on_drivable=False)
    assert r2.mark_class == MARK_ROADSIDE_ARTIFACT
