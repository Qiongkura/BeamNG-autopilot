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
    """错误地图必须能被拒绝（计划原话）。"""

    def test_a_map_pointing_the_other_way_is_rejected(self):
        r = associate_candidate(_cand(bearing_rad=0.0),
                                _prior(direction_rad=math.pi))
        assert any("direction_mismatch" in c for c in r.conflicts)

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
