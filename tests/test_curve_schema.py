"""T10: the annotation schema must be able to say what pixels cannot.

The plan's T10 requires curve identity, left/right/divider role, visible
segments, occlusion, endpoints, attributes and explicit UNKNOWN - and the
separation of pixel paint, centre lines and INFERRED extensions.  These
tests pin both the accepted shape and the refusals, because a schema that
silently accepts a merged inferred curve is how an interpolation becomes
"truth".
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from beamng_autopilot.labeling import curve_schema as cs


def _measured(frame: int = 3, curve_id: str = "c1", role: str = "left"):
    return cs.Curve(
        curve_id=curve_id, role=role,
        attributes={"colour": "white", "style": "solid"},
        segments=[cs.CurveSegment(
            frame_start=frame, frame_end=frame + 4,
            source=cs.SRC_PIXEL_PAINT, visible=True,
            points=[cs.CurvePoint(u=100.0, v=180.0, frame=frame),
                    cs.CurvePoint(u=110.0, v=175.0, frame=frame + 1)])])


def _ann(*curves, **kw):
    return cs.FrameAnnotation(frame_index=kw.get("frame_index", 3),
                              t=kw.get("t", 12.5), image_w=320, image_h=240,
                              curves=list(curves),
                              run=kw.get("run", "run_7"),
                              map_name=kw.get("map_name", "italy"),
                              episode=kw.get("episode", "shadow_ep1"),
                              source_id=kw.get("source_id", "front_main"))


class TestAcceptedShape:
    def test_a_measured_curve_round_trips_through_jsonl(self, tmp_path):
        ann = _ann(_measured(), _measured(curve_id="c2", role="right"))
        p = tmp_path / "labels.jsonl"
        assert cs.dump_jsonl(p, [ann]) == 1
        back = cs.load_jsonl(p)
        assert len(back) == 1
        assert cs.validate(back[0]) == []
        assert cs.to_dict(back[0]) == cs.to_dict(ann)
        assert [c.role for c in back[0].curves] == ["left", "right"]
        assert back[0].map_name == "italy" and back[0].source_id == "front_main"

    def test_an_inferred_extension_is_accepted_when_it_names_its_source(self):
        curve = _measured()
        curve.segments.append(cs.CurveSegment(
            frame_start=8, frame_end=10, source=cs.SRC_INFERRED,
            visible=False, derived_from=["c1@frame3", "c1@frame7"]))
        assert cs.validate(_ann(curve)) == []

    def test_an_occluded_span_is_a_visible_false_segment(self):
        curve = _measured()
        curve.segments.append(cs.CurveSegment(
            frame_start=8, frame_end=10, source=cs.SRC_PIXEL_PAINT,
            visible=False, occlusion=cs.Occlusion(by="vehicle",
                                                  frame_start=8,
                                                  frame_end=10)))
        assert cs.validate(_ann(curve)) == []

    def test_unknown_fields_are_explicit(self):
        curve = _measured()
        curve.role = "unknown"
        curve.unknown = ["role", "style"]
        assert cs.validate(_ann(curve)) == []

    def test_the_annotator_roles_cover_the_plan(self):
        assert set(cs.ROLES) == {"left", "right", "divider", "centre",
                                 "unknown"}
        assert set(cs.SOURCES) == {"pixel_paint", "centre_line",
                                   "inferred_extension"}


class TestRefusals:
    def test_an_inferred_segment_without_provenance_is_rejected(self):
        curve = _measured()
        curve.segments.append(cs.CurveSegment(
            frame_start=8, frame_end=10, source=cs.SRC_INFERRED))
        errs = cs.validate(_ann(curve))
        assert any("derived from" in e for e in errs)

    def test_a_measured_segment_may_not_claim_to_be_derived(self):
        curve = _measured()
        curve.segments[0].derived_from = ["c9@frame1"]
        assert any("measured segment" in e
                   for e in cs.validate(_ann(curve)))

    def test_an_occluded_span_cannot_also_be_visible(self):
        curve = _measured()
        curve.segments.append(cs.CurveSegment(
            frame_start=8, frame_end=10, visible=True,
            occlusion=cs.Occlusion(by="wall", frame_start=8, frame_end=10)))
        assert any("occluded span must be stored as" in e
                   for e in cs.validate(_ann(curve)))

    def test_an_occlusion_without_a_cause_or_span_is_rejected(self):
        curve = _measured()
        curve.segments.append(cs.CurveSegment(
            frame_start=8, frame_end=10, visible=False,
            occlusion=cs.Occlusion(by="  ", frame_start=10, frame_end=8)))
        errs = cs.validate(_ann(curve))
        assert any("frame_end < frame_start" in e for e in errs)
        assert any("without a cause" in e for e in errs)

    def test_unknown_role_attribute_and_field_are_rejected(self):
        curve = _measured()
        curve.role = "middle"
        curve.attributes["thickness"] = 0.2
        curve.unknown = ["eyes"]
        errs = cs.validate(_ann(curve))
        assert any("role 'middle'" in e for e in errs)
        assert any("attribute 'thickness'" in e for e in errs)
        assert any("unknown field 'eyes'" in e for e in errs)

    def test_duplicate_identity_and_missing_segments_are_rejected(self):
        errs = cs.validate(_ann(_measured(), _measured()))
        assert any("duplicate curve_id 'c1'" in e for e in errs)
        empty = cs.Curve(curve_id="c3", role="right")
        assert any("no segments" in e for e in cs.validate(_ann(empty)))

    def test_a_point_outside_its_segment_span_is_rejected(self):
        curve = _measured()
        curve.segments[0].points.append(cs.CurvePoint(u=1.0, v=1.0, frame=99))
        assert any("outside the segment span" in e
                   for e in cs.validate(_ann(curve)))

    def test_a_wrong_version_is_rejected(self):
        ann = _ann(_measured())
        ann.version = 99
        assert any("version 99" in e for e in cs.validate(ann))

    def test_dump_refuses_an_invalid_record_and_load_refuses_a_bad_line(
            self, tmp_path):
        bad = _ann(_measured())
        bad.curves[0].role = "middle"
        p = tmp_path / "bad.jsonl"
        with pytest.raises(ValueError):
            cs.dump_jsonl(p, [bad])
        good = cs.to_dict(_ann(_measured()))
        good["curves"][0]["role"] = "middle"
        p.write_text(json.dumps(good) + "\n", encoding="utf-8")
        with pytest.raises(ValueError):
            cs.load_jsonl(p)

    def test_an_unknown_kwarg_is_a_type_error_not_a_silent_default(
            self, tmp_path):
        payload = cs.to_dict(_ann(_measured()))
        payload["lane_width_m"] = 3.5
        with pytest.raises(TypeError):
            cs.from_dict(payload)


class TestPixelClasses:
    def test_the_classes_are_pinned(self):
        assert (cs.CLS_BACKGROUND, cs.CLS_ROAD, cs.CLS_LINE,
                cs.CLS_IGNORE) == (0, 1, 2, 255)
        assert cs.PIXEL_CLASS_NAMES[255] == "ignore"

    def test_ignore_is_not_counted_as_background(self):
        mask = np.array([[0, 1, 2], [255, 255, 1]], dtype=np.uint8)
        rep = cs.pixel_value_report(mask)
        assert rep["counts"] == {"background": 1, "road": 2, "line": 1,
                                 "ignore": 2}
        assert rep["unknown_values"] == []
        assert rep["n"] == 6

    def test_an_undefined_value_is_reported_not_relabelled(self):
        mask = np.array([[2, 7], [255, 3]], dtype=np.uint8)
        rep = cs.pixel_value_report(mask)
        assert rep["unknown_values"] == [3, 7]
        assert rep["counts"]["line"] == 1

    def test_an_empty_mask_reports_nothing_rather_than_zero_line(self):
        rep = cs.pixel_value_report(np.zeros((0, 0), dtype=np.uint8))
        assert rep == {"n": 0, "counts": {}, "unknown_values": []}
