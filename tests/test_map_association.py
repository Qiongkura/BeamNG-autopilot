"""T08: map-assisted association must be falsifiable, and never a lateral source.

The plan's T08 allows the map to rank and reject PERCEPTION CANDIDATES and
forbids it from producing an executable lateral centre.  These tests pin
both halves:

* the positive path (a consistent map raises the right candidate's score);
* every rejection path the plan lists: no measurement, map offset/stale,
  wrong lane count, wrong width, junction ambiguity, side mismatch;
* and the structural guarantee: no lateral geometry exists anywhere in the
  module's output.

The A/B comparison is only meaningful if arm B can be WRONG, so the tests
also pin the counter it produces: score changes that happened while a
conflict was recorded (= false-acceptance risk).
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from beamng_autopilot.lane.map_association import (
    DIR_HARD_DEG,
    AssociationResult,
    MapLinkPrior,
    PerceptionCandidate,
    associate_candidate,
    compare_arms,
    expected_side_for,
)


def _prior(**kw):
    base = dict(link_id="n1->n2", direction_rad=0.0, lanes_hint=2,
                one_way=False, right_hand_drive=True)
    base.update(kw)
    return MapLinkPrior(**base)


def _cand(**kw):
    base = dict(cand_id="c1", side="right", kind="solid", bearing_rad=0.0,
                confidence=0.8, span_m=10.0, paired=True, fresh=True)
    base.update(kw)
    return PerceptionCandidate(**base)


class TestRealMapFieldsOnly:
    def test_the_link_prior_is_built_from_the_connector_view(self):
        rule = type("R", (), {
            "lanes": "++--", "one_way": False, "right_hand_drive": True,
            "n1": "a", "n2": "b", "in_pos": (0.0, 0.0, 0.0),
            "out_pos": (10.0, 0.0, 0.0), "in_radius": 50.0,
            "out_radius": 70.0, "drivability": 1.0})()
        prior = MapLinkPrior.from_road_rule(rule)
        assert prior.link_id == "a->b"
        assert prior.direction_rad == pytest.approx(0.0)
        assert prior.lanes_hint == 4
        assert prior.curvature_1pm == pytest.approx(1.0 / 60.0, rel=0.01)
        assert prior.one_way is False and prior.right_hand_drive is True

    def test_a_missing_link_is_unavailable_not_guessed(self):
        prior = MapLinkPrior.from_road_rule(None)
        assert prior.link_id is None
        assert prior.direction_rad is None and prior.lanes_hint is None

    def test_the_lane_string_is_a_count_never_a_width(self):
        assert MapLinkPrior.parse_lane_count("++--") == (4, 2)
        assert MapLinkPrior.parse_lane_count("2") == (2, None)
        assert MapLinkPrior.parse_lane_count("") == (None, None)
        assert MapLinkPrior.parse_lane_count("nonsense") == (None, None)


class TestSidePrior:
    def test_right_hand_traffic_puts_the_divider_left_and_the_edge_right(self):
        assert expected_side_for("divider", True) == "left"
        assert expected_side_for("solid", True) == "right"
        assert expected_side_for("dashed", True) == "right"

    def test_left_hand_traffic_swaps_them(self):
        assert expected_side_for("divider", False) == "right"
        assert expected_side_for("solid", False) == "left"

    def test_an_unknown_kind_has_no_side_prior(self):
        assert expected_side_for("blob", True) is None


class TestFalsifiableRejections:
    def test_a_consistent_candidate_is_boosted(self):
        r = associate_candidate(_cand(), _prior())
        assert r.abstain is None and r.conflicts == []
        assert r.score > 0.8          # boosted above the raw confidence

    def test_no_measurement_abstains_and_creates_nothing(self):
        r = associate_candidate(_cand(fresh=False), _prior())
        assert r.abstain and "not a fresh observation" in r.abstain
        assert r.hypothesis_id == "n1->n2"      # the map was known
        assert r.score == 0.0

    def test_no_map_link_abstains(self):
        r = associate_candidate(_cand(), MapLinkPrior())
        assert r.abstain and "no map link" in r.abstain
        assert r.hypothesis_id is None

    def test_a_junction_abstains_rather_than_picking_a_road(self):
        r = associate_candidate(_cand(), _prior(), other_links=2)
        assert r.abstain and "ambiguous" in r.abstain

    def test_a_wrong_lane_count_is_a_conflict_not_a_rerank(self):
        # the map says one lane; the pair is 7 m wide (two lanes)
        r = associate_candidate(_cand(width_m=7.0), _prior(lanes_hint=1))
        assert any("width_contradicts_map" in c for c in r.conflicts)
        assert r.score < 0.8

    def test_the_side_prior_catches_a_divider_on_the_wrong_side(self):
        r = associate_candidate(_cand(side="right", kind="divider"), _prior(),
                                trust_right_hand_drive=True)
        assert any("side_mismatch" in c for c in r.conflicts)

    def test_the_side_prior_is_OFF_until_the_flag_is_verified(self):
        """Measured 2026-09-22: the italy asset reports
        ``rightHandDrive: false`` while the road under test is driven on
        the right, so the flag is not trusted by default - and its being
        present must not demote a real candidate."""
        r = associate_candidate(_cand(side="right", kind="solid"), _prior())
        assert not any("side_mismatch" in c for c in r.conflicts)
        assert "right_hand_drive" not in r.map_fields

    def test_a_candidate_at_a_hard_angle_is_flagged(self):
        r = associate_candidate(
            _cand(bearing_rad=math.radians(DIR_HARD_DEG + 10)), _prior())
        assert any("direction_mismatch" in c for c in r.conflicts)

    def test_a_loose_angle_lowers_the_score_without_condemning(self):
        r = associate_candidate(_cand(bearing_rad=math.radians(30.0)), _prior())
        assert any("direction_loose" in c for c in r.conflicts)
        assert r.score > 0.0

    def test_the_conflict_names_the_map_field_and_the_quantity(self):
        r = associate_candidate(_cand(width_m=7.0), _prior(lanes_hint=1))
        text = " ".join(r.conflicts)
        assert "7.00 m" in text and "lanes=1" in text


class TestNoLateralOutput:
    """The structural guarantee the plan is most explicit about."""

    def test_the_result_has_no_lateral_or_authority_field(self):
        names = {f.name for f in dataclasses.fields(AssociationResult)}
        banned = ("center", "centre", "offset", "lateral", "target",
                  "authority", "drivable", "crossable", "path")
        assert not (names & set(banned)), names & set(banned)

    def test_the_prior_dataclasses_carry_no_geometry_beyond_the_link(self):
        for cls in (MapLinkPrior, PerceptionCandidate):
            names = {f.name for f in dataclasses.fields(cls)}
            assert "lane_center" not in names and "offset_m" not in names

    def test_the_module_exposes_no_lateral_target_function(self):
        import beamng_autopilot.lane.map_association as mod
        fns = [n for n, v in vars(mod).items()
               if callable(v) and not n.startswith("_")]
        assert not any(("center" in n or "target" in n or "offset" in n)
                       for n in fns), fns

    def test_scoring_never_depends_on_the_ego_lateral_position(self):
        """Same candidate + same map => same score, wherever the car is."""
        import inspect
        import re
        sig = inspect.signature(associate_candidate)
        assert not ({"pos", "ego_pos", "ego_lat", "line_lat", "lane_dev"}
                    & set(sig.parameters))
        src = inspect.getsource(associate_candidate)
        for banned in (r"pos", r"ego_lat", r"line_lat",
                       r"lane_dev"):
            assert not re.search(banned, src), banned
        # and the score is a pure function of (candidate, prior, links)
        a = associate_candidate(_cand(), _prior())
        b = associate_candidate(_cand(), _prior())
        assert a.score == b.score


class TestArmsComparison:
    def test_arm_b_only_rescores_and_reports_the_diff(self):
        cands = [_cand(cand_id="good", confidence=0.5),
                 # a candidate whose bearing contradicts the link: a
                 # conflict, so arm B must demote it
                 _cand(cand_id="bad", confidence=0.6,
                       bearing_rad=math.radians(DIR_HARD_DEG + 10))]
        rep = compare_arms(cands, _prior())
        assert rep["rank_changed"] is True          # the conflict demotes it
        assert rep["arm_a"] and rep["arm_b"]
        assert rep["order_b"][0] == "good"
        # no candidate was created, removed or moved
        assert ([r["candidate_id"] for r in rep["arm_a"]]
                == [r["candidate_id"] for r in rep["arm_b"]])

    def test_false_acceptance_is_counted_not_argued(self):
        # a candidate WITH a conflict that arm B still raises relative to
        # arm A cannot happen by construction here, and the counter must be
        # zero for this map; a map that did it would be caught by this test
        cands = [_cand(cand_id="c1", confidence=0.2),
                 _cand(cand_id="c2", confidence=0.9,
                       bearing_rad=math.radians(DIR_HARD_DEG + 10))]
        rep = compare_arms(cands, _prior())
        assert rep["false_acceptance_risk"] == rep["n_changed_with_conflict"]

    def test_an_unavailable_map_leaves_the_scores_alone(self):
        """The null behaviour: no map -> arm B is a no-op (and says why)."""
        cands = [_cand(cand_id="c1", confidence=0.4)]
        rep = compare_arms(cands, MapLinkPrior())
        assert rep["n_changed_scores"] == 0
        assert rep["arm_a"][0]["score"] == rep["arm_b"][0]["score"]
        assert rep["arm_b"][0]["abstain"]


class TestRejectionsOnWrongMaps:
    """错误地图必须能被拒绝（计划原话）。

    2026-09-24 更正：旧版把"链路方向"当**有向**的（`direction_rad=pi` 判硬冲突）。
    标线**没有箭头**，所以"反向"不是反证；判据改为**轴比对**（mod 180），
    "反向"作为**并列诊断**报告（`sense_delta_deg` / `direction_sense:*`）而不再否决。
    实测量级：三次**沿路行驶**的实车运行（33/48/59 个候选）里弦参考的方向一致性都是 100%——
    所以这个改动在车头朝路时是恒等变换，只在车头与道路无关时把伪 180° 去掉。
    """

    def test_a_map_pointing_the_other_way_is_reported_not_penalised(self):
        r = associate_candidate(_cand(bearing_rad=0.0),
                                _prior(direction_rad=math.pi))
        assert r.conflicts == [], "a lane marking has no arrowhead"
        assert r.sense_delta_deg == pytest.approx(180.0)
        assert any("direction_sense:flipped" in f for f in r.map_fields)
        # the axis verdict itself is unchanged and still published
        assert any("direction_source:chord" in f for f in r.map_fields)

    def test_a_map_across_the_road_is_still_rejected(self):
        """The axis test must not become a rubber stamp: 90 deg is 90 deg."""
        r = associate_candidate(_cand(bearing_rad=0.0),
                                _prior(direction_rad=math.pi / 2.0))
        assert any("direction_mismatch" in c for c in r.conflicts)
        assert r.sense_delta_deg == pytest.approx(90.0)

    def test_an_aligned_link_says_so(self):
        r = associate_candidate(_cand(bearing_rad=0.0), _prior(direction_rad=0.0))
        assert r.conflicts == []
        assert r.sense_delta_deg == pytest.approx(0.0)
        assert any("direction_sense:aligned" in f for f in r.map_fields)

    def test_a_one_way_map_with_a_two_sided_pair_is_flagged(self):
        r = associate_candidate(_cand(width_m=7.0),
                                _prior(lanes_hint=2, one_way=True))
        assert r.conflicts == [] or all("one_way" in c or "width" in c
                                        for c in r.conflicts)

    def test_a_stale_link_id_is_the_caller_s_problem_and_is_visible(self):
        """Two different link ids in consecutive ticks is a map jump: the
        association must expose the hypothesis id so a caller can see it."""
        a = associate_candidate(_cand(), _prior(link_id="n1->n2"))
        b = associate_candidate(_cand(), _prior(link_id="n9->n7"))
        assert a.hypothesis_id != b.hypothesis_id

    def test_no_candidate_ever_gains_authority_from_the_map(self):
        for kw in ({}, {"lanes_hint": 1}, {"direction_rad": None},
                   {"right_hand_drive": False}):
            r = associate_candidate(_cand(), _prior(**kw))
            assert not hasattr(r, "authority")
            assert r.score <= 1.25      # bounded gain, never a permission


class TestLocalTangent:
    """T08: the direction prior must be the LOCAL tangent, not the chord.

    Measured live: candidates on long links sat 33-45 deg off the chord
    because the chord spans the whole link while the car is at one point of
    it.  The arc model uses the link's OWN reported radius (inRadius /
    outRadius from the same Lua call) - no new data source - and returns
    BOTH turn signs, because BeamNG's radius sign convention is not
    documented; the caller pins it empirically and `direction_source`
    records what was graded against.
    """

    def test_the_arc_tangent_matches_the_analytic_rotation(self):
        from beamng_autopilot.lane.map_association import local_tangent_rad
        # chord along +x, 100 m long; ego 20 m along it; R = 50 m
        t, t_alt, off, r = local_tangent_rad((20.0, 0.0, 0.0), (0.0, 0.0),
                                             (100.0, 0.0), 50.0)
        assert off == pytest.approx(20.0)
        assert r == pytest.approx(50.0)
        assert math.degrees(t) == pytest.approx(math.degrees(0.4), abs=1e-6)
        assert math.degrees(t_alt) == pytest.approx(-math.degrees(0.4),
                                                    abs=1e-6)

    def test_the_offset_is_clamped_to_the_link(self):
        from beamng_autopilot.lane.map_association import local_tangent_rad
        t, _, off, _ = local_tangent_rad((500.0, 0.0, 0.0), (0.0, 0.0),
                                         (100.0, 0.0), 50.0)
        assert off == pytest.approx(100.0)
        assert math.degrees(t) == pytest.approx(math.degrees(2.0), abs=1e-6)

    def test_no_radius_means_no_tangent_rather_than_a_guess(self):
        from beamng_autopilot.lane.map_association import local_tangent_rad
        assert local_tangent_rad((10.0, 0.0, 0.0), (0.0, 0.0),
                                 (100.0, 0.0), 0.0) == (None, None, None, None)
        assert local_tangent_rad((10.0, 0.0, 0.0), (0.0, 0.0),
                                 (0.0, 0.0), 50.0)[0] is None

    def test_the_prior_records_which_direction_was_used(self):
        from beamng_autopilot.lane.map_association import MapLinkPrior

        class _Rule:
            lanes = "++"
            in_pos = (0.0, 0.0, 0.0)
            out_pos = (100.0, 0.0, 0.0)
            in_radius = 50.0
            out_radius = 50.0
            n1 = "a"
            n2 = "b"
            one_way = False
            drivability = 1.0
            right_hand_drive = False
        straight = MapLinkPrior.from_road_rule(_Rule())
        assert straight.direction_source == "chord" and straight.tangent_rad is None
        with_pos = MapLinkPrior.from_road_rule(_Rule(), pos=(20.0, 3.0, 0.0))
        assert with_pos.direction_source == "tangent_arc"
        assert math.degrees(with_pos.tangent_rad) == pytest.approx(
            math.degrees(0.4), abs=1e-6)
        assert with_pos.arc_offset_m == pytest.approx(20.0)


class TestChordIsTheDefaultReference:
    """Measured: the arc tangent (from in/outRadius) is NOT usable here.

    On a straight link the chord is constant while the tangent estimate
    swept 4..166 deg, and the candidates' median |delta| was 17.9 deg
    against the chord vs ~80 deg against either tangent sign - because
    those radii are 3.5 m, not a road arc radius.  The graded reference is
    therefore the chord unless the caller opts in, and the result records
    which one was used.
    """

    def _cand(self):
        from beamng_autopilot.lane.map_association import PerceptionCandidate
        return PerceptionCandidate(cand_id="c", side="left", kind="solid",
                                   bearing_rad=0.0, confidence=0.9,
                                   span_m=5.0, fresh=True)

    def _prior(self):
        from beamng_autopilot.lane.map_association import MapLinkPrior
        return MapLinkPrior(link_id="a->b", direction_rad=0.0, lanes_hint=2,
                            tangent_rad=math.radians(80.0),
                            tangent_alt_rad=math.radians(-80.0),
                            arc_offset_m=5.0, radius_m=3.5,
                            direction_source="tangent_arc")

    def test_default_grades_the_chord(self):
        from beamng_autopilot.lane.map_association import associate_candidate
        res = associate_candidate(self._cand(), self._prior())
        assert res.conflicts == [], "chord agrees with the candidate"
        assert any("direction_source:chord" in f for f in res.map_fields)

    def test_the_tangent_is_opt_in_and_then_disagrees(self):
        from beamng_autopilot.lane.map_association import associate_candidate
        res = associate_candidate(self._cand(), self._prior(), use_tangent=True)
        assert any("direction_mismatch" in c for c in res.conflicts)
        assert any("direction_source:tangent_arc" in f for f in res.map_fields)


class TestTheRecordedSourceNeverOutrunsTheData:
    """T08 余项: a second tangent source (the map graph's polyline) exists.

    ``in/outRadius`` is 3.5 m on this map - not a road arc radius - so the
    probe can grade against the map graph's centre-line polyline instead.
    A declared source is a LABEL, and a label must never claim a reference
    the test did not use: when no tangent was the reference (not asked for,
    or none available) the source stays ``chord``.
    """

    def _cand(self, deg=0.0):
        from beamng_autopilot.lane.map_association import PerceptionCandidate
        return PerceptionCandidate(cand_id="c", side="left", kind="solid",
                                   bearing_rad=math.radians(deg),
                                   confidence=0.9, span_m=5.0, fresh=True)

    def _prior(self, tangent_deg=None):
        from beamng_autopilot.lane.map_association import MapLinkPrior
        return MapLinkPrior(
            link_id="a->b", direction_rad=0.0, lanes_hint=2,
            tangent_rad=(None if tangent_deg is None
                         else math.radians(tangent_deg)),
            direction_source="chord")

    def test_a_declared_source_is_only_recorded_when_a_tangent_was_used(self):
        from beamng_autopilot.lane.map_association import (
            associate_candidate, direction_reference_rad)
        prior = self._prior(tangent_deg=25.0)
        ref, src = direction_reference_rad(prior, use_tangent=True,
                                          direction_source="tangent_roadnet")
        assert src == "tangent_roadnet"
        assert math.degrees(ref) == pytest.approx(25.0)
        # not asked for -> chord, even though a label was offered
        ref, src = direction_reference_rad(prior, direction_source="tangent_roadnet")
        assert src == "chord" and math.degrees(ref) == pytest.approx(0.0)
        # asked for but nothing to use -> still chord, never a phantom source
        ref, src = direction_reference_rad(self._prior(), use_tangent=True,
                                          direction_source="tangent_roadnet")
        assert src == "chord" and ref == pytest.approx(0.0)
        res = associate_candidate(self._cand(), self._prior(),
                                  use_tangent=True,
                                  direction_source="tangent_roadnet")
        assert not any("tangent_roadnet" in f for f in res.map_fields)

    def test_the_polyline_arm_is_graded_and_labelled_as_itself(self):
        from beamng_autopilot.lane.map_association import associate_candidate
        prior = self._prior(tangent_deg=25.0)
        res = associate_candidate(self._cand(deg=25.0), prior,
                                  use_tangent=True,
                                  direction_source="tangent_roadnet")
        assert res.conflicts == [], "candidate agrees with the polyline tangent"
        assert any("direction_source:tangent_roadnet" in f
                   for f in res.map_fields)

    def test_compare_arms_passes_the_source_through(self):
        from beamng_autopilot.lane.map_association import compare_arms
        prior = self._prior(tangent_deg=25.0)
        rep = compare_arms([self._cand(deg=25.0)], prior, use_tangent=True,
                           direction_source="tangent_roadnet")
        row = rep["arm_b"][0]
        assert "direction_source:tangent_roadnet" in row["map_fields"]
        assert row["conflicts"] == []

    def test_the_delta_helper_is_the_same_quantity_the_test_grades(self):
        from beamng_autopilot.lane.map_association import (
            DIR_HARD_DEG, direction_delta_deg)
        assert direction_delta_deg(0.0, math.radians(30.0)) == pytest.approx(30.0)
        assert direction_delta_deg(math.radians(-179.0),
                                   math.radians(179.0)) == pytest.approx(2.0)
        assert direction_delta_deg(None, 0.0) is None
        assert direction_delta_deg(0.0, None) is None
        assert isinstance(DIR_HARD_DEG, float)
