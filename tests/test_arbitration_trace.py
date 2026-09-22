"""Safety traces report actual predicate visits and the final winner."""

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
            "car body observed off the pavement": "body_off_pavement",
            "car body observed off the pavement (creep)":
                "body_off_pavement",
            "path blocked by obstacle": "path_blocked",
            "scattered obstacle": "scattered_obstacle",
            "path grazes obstacle": "path_grazes",
            "lane boundary recovery": "lane_boundary_recovery",
            "planned boundary crossing ahead": "planned_boundary_crossing",
            "current vehicle body crosses lane boundary": "body_crosses_boundary",
            "planned vehicle body crosses lane boundary": "body_crosses_boundary",
            "path off-lane": "path_off_lane",
            "path near lane edge": "path_near_lane_edge",
            "obstacle very close": "obstacle_very_close",
            "obstacle contact risk": "obstacle_risk",
            "obstacle stopping distance": "obstacle_risk",
        }
        covered = {rule_for_reason(r) for r in reasons}
        assert covered == set(ARBITRATION_RULES)


class TestArbitrationOutcome:
    def test_absent_trace_never_claims_full_coverage(self):
        for fired in (None, "scattered_obstacle", "rule_from_a_newer_branch"):
            oc = arbitration_outcome(fired)
            assert oc["evaluated"] == []
            assert oc["unevaluated"] == []
            assert oc["masked_hard"] == []

    def test_explicit_visits_do_not_stop_at_the_soft_winner(self):
        oc = arbitration_outcome("scattered_obstacle",
                                 evaluated=ARBITRATION_RULES,
                                 level="degraded")
        assert oc["effective"] == "scattered_obstacle"
        assert oc["evaluated"] == list(ARBITRATION_RULES)
        assert oc["unevaluated"] == []
        assert oc["masked_hard"] == []

    def test_road_rule_uses_actual_not_worst_level(self):
        visits = list(ARBITRATION_RULES[:5])
        soft = arbitration_outcome("road_surface", evaluated=visits,
                                   level="degraded")
        assert "body_crosses_boundary" in soft["masked_hard"]
        hard = arbitration_outcome("road_surface", evaluated=visits,
                                   level="minimal_risk")
        assert hard["masked_hard"] == []
        assert hard["unevaluated"] == list(ARBITRATION_RULES[5:])

    def test_every_rule_has_a_worst_level(self):
        assert set(RULE_WORST_LEVEL) == set(ARBITRATION_RULES)

    def test_unknown_reason_does_not_become_an_all_checked_verdict(self):
        from beamng_autopilot.safety_monitor import SafetyVerdict
        v = SafetyMonitor()._finish(SafetyVerdict(reason="something new"))
        assert v.rules_evaluated == []
        assert v.rules_unevaluated == list(ARBITRATION_RULES)


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
