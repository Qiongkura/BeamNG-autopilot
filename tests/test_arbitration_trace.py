"""The arbitration chain must say what it did NOT evaluate.

`_evaluate_core` returns on the first rule that fires.  A verdict whose
reason says "scattered obstacle" therefore also says nothing at all about
the road-surface rule and the body-cross rule, both of which stop the car
and both of which sit further down the chain.  Reading that reason as
"the monitor looked and found nothing worse" is the failure mode these
tests exist to prevent.
"""

from beamng_autopilot.safety_monitor import (
    ARBITRATION_RULES,
    RULE_WORST_LEVEL,
    SafetyMonitor,
    arbitration_outcome,
    rule_for_reason,
)


class TestRuleForReason:
    """Every branch's reason must map back to its rule."""

    def test_a_plain_reason_maps_to_its_rule(self):
        assert rule_for_reason("scattered obstacle") == "scattered_obstacle"

    def test_the_path_hold_reason_carries_its_phase(self):
        # f"path hold ({hold_phase})" - the phase is not part of the id.
        assert rule_for_reason("path hold (grace)") == "path_hold"
        assert rule_for_reason("path hold (creep)") == "path_hold"

    def test_no_reason_means_no_rule_fired(self):
        assert rule_for_reason(None) is None
        assert rule_for_reason("") is None

    def test_an_unknown_reason_is_none_not_a_guess(self):
        # A new branch that forgets the table must read as unknown, not as
        # "no rule fired", which would claim every rule was evaluated.
        assert rule_for_reason("something new") is None

    def test_every_rule_is_reachable_from_a_reason(self):
        reasons = {
            "stale sensor": "stale_sensor_planner",
            "stale planner": "stale_sensor_planner",
            "path hold (grace)": "path_hold",
            "no drivable path": "no_drivable_path",
            "perception lane unavailable": "perception_lane_unavailable",
            "perceived road surface lost": "road_surface",
            "off perceived road surface": "road_surface",
            "path blocked by obstacle": "path_blocked",
            "scattered obstacle": "scattered_obstacle",
            "path grazes obstacle": "path_grazes",
            "lane boundary recovery": "lane_boundary_recovery",
            "planned boundary crossing ahead": "planned_boundary_crossing",
            "current vehicle body crosses lane boundary": "body_crosses_boundary",
            "path off-lane": "path_off_lane",
            "path near lane edge": "path_near_lane_edge",
            "obstacle very close": "obstacle_very_close",
        }
        covered = {rule_for_reason(r) for r in reasons}
        assert covered == set(ARBITRATION_RULES)


class TestArbitrationOutcome:
    def test_no_rule_fired_means_everything_was_evaluated(self):
        oc = arbitration_outcome(None)
        assert oc["effective"] is None
        assert oc["evaluated"] == list(ARBITRATION_RULES)
        assert oc["unevaluated"] == []
        assert oc["masked_hard"] == []

    def test_the_winner_and_everything_before_it_ran(self):
        oc = arbitration_outcome("road_surface")
        assert oc["evaluated"] == list(ARBITRATION_RULES[:5])

    def test_everything_after_the_winner_never_ran(self):
        oc = arbitration_outcome("road_surface")
        assert oc["unevaluated"] == list(ARBITRATION_RULES[5:])
        assert "body_crosses_boundary" in oc["unevaluated"]

    def test_a_soft_winner_masks_the_hard_rules_below_it(self):
        """The case the review asked for.

        "scattered obstacle" is a slowdown that returns before both rules
        that stop the car.  A report reading only the reason would
        conclude the monitor found nothing worse.
        """
        oc = arbitration_outcome("scattered_obstacle")
        assert RULE_WORST_LEVEL["scattered_obstacle"] == "degraded"
        assert oc["masked_hard"] == [
            r for r in ARBITRATION_RULES[7:]
            if RULE_WORST_LEVEL[r] == "minimal_risk"]

    def test_a_hard_winner_masks_nothing(self):
        # A stop is the worst thing the chain can do, so nothing below it
        # could have been stricter.
        oc = arbitration_outcome("path_blocked")
        assert oc["masked_hard"] == []

    def test_the_last_rule_leaves_nothing_unevaluated(self):
        oc = arbitration_outcome("obstacle_very_close")
        assert oc["unevaluated"] == []

    def test_an_unknown_rule_claims_no_coverage(self):
        """Position in the chain is unknown, so nothing downstream may be
        called evaluated - and nothing may be called masked either."""
        oc = arbitration_outcome("rule_from_a_newer_branch")
        assert oc["evaluated"] == []
        assert oc["unevaluated"] == []
        assert oc["masked_hard"] == []

    def test_every_rule_has_a_worst_level(self):
        assert set(RULE_WORST_LEVEL) == set(ARBITRATION_RULES)


class _Scene:
    def __init__(self, grid):
        self.grid = grid


class TestRoadSurfaceCheckHonesty:
    """A gate that did not run must not publish "checked, no answer"."""

    def test_a_scene_with_no_grid_was_not_checked(self):
        mon = SafetyMonitor()
        state, lost_s, checked = mon._road_surface_gate(_Scene(None), 10.0)
        assert state == "unknown"
        assert lost_s == 0.0
        assert checked is False

    def test_a_grid_the_reader_cannot_read_was_checked(self):
        """Grid present, no road layer: the reader ran and has no answer.

        That is a different statement from the case above, and the whole
        point of `road_checked` is to keep them apart.
        """
        mon = SafetyMonitor()
        state, lost_s, checked = mon._road_surface_gate(_Scene(object()), 10.0)
        assert state == "unknown"
        assert checked is True

    def test_a_gridless_scene_does_not_accumulate_loss_time(self):
        # Intermittent grids must not be read as "the road came back", so
        # the timer stays cleared - the pipeline staleness rules own this.
        mon = SafetyMonitor()
        for t in (10.0, 11.0, 12.0):
            _, lost_s, checked = mon._road_surface_gate(_Scene(None), t)
            assert lost_s == 0.0
            assert checked is False

    def test_a_readable_grid_accumulates_loss_time(self):
        mon = SafetyMonitor()
        mon._road_surface_gate(_Scene(object()), 10.0)
        _, lost_s, checked = mon._road_surface_gate(_Scene(object()), 13.0)
        assert checked is True
        assert lost_s == 3.0

    def test_the_verdict_carries_the_flag_not_a_constant_true(self):
        # The old code set road_checked = True on every path that reached
        # the gate, including the grid-less one.
        from beamng_autopilot.safety_monitor import SafetyVerdict
        v = SafetyVerdict()
        assert v.road_checked is False
        assert v.effective_rule is None
        assert v.rules_unevaluated == []
        assert v.masked_hard_rules == []
