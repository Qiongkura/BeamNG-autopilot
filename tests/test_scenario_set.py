"""P5: a scenario set that can actually fail, and runs that can be paired."""

import pytest

from scripts.m5_scenario_set import (
    ALL_SCENARIOS,
    SCENARIOS,
    VALID_EXCLUSIONS,
    gate,
    main,
    measurement_coverage,
    pair_order,
    pairwise_deltas,
    summarise_effect,
)


class TestScenarioSet:
    def test_all_eight_are_defined(self):
        assert ALL_SCENARIOS == ("A", "B", "C", "D", "E", "F", "G", "H")

    def test_each_names_a_question_and_a_release(self):
        # A scenario without a question is a drive, not a test.
        for sid in ALL_SCENARIOS:
            assert SCENARIOS[sid]["question"]
            assert SCENARIOS[sid]["requires"]
            assert SCENARIOS[sid]["release"]

    def test_the_collision_precursor_needs_the_pedal_not_just_the_target(self):
        """D: comparing mon_target is not comparing what the car did."""
        assert "throttle" in SCENARIOS["D"]["requires"]
        assert "brake" in SCENARIOS["D"]["requires"]

    def test_the_unreachable_gap_scenario_requires_the_answer_not_the_bool(self):
        assert "corridor_state" in SCENARIOS["C"]["requires"]
        assert "corridor_reason" in SCENARIOS["C"]["requires"]

    def test_the_fault_scenarios_require_the_new_trace_fields(self):
        assert "head_sched" in SCENARIOS["E"]["requires"]
        # G is the scheduling/watchdog one: tick cost and command gap.
        assert "watchdog" in SCENARIOS["G"]["requires"]
        assert "cmd_gap_s" in SCENARIOS["G"]["requires"]


class TestPairOrder:
    def test_it_is_balanced(self):
        o = pair_order(8, seed=1)
        assert o.count("AB") == 4 and o.count("BA") == 4

    def test_the_same_seed_gives_the_same_order(self):
        assert pair_order(6, seed=7) == pair_order(6, seed=7)

    def test_a_different_seed_gives_a_different_order(self):
        assert pair_order(12, seed=1) != pair_order(12, seed=2)

    def test_an_odd_count_still_balances_as_far_as_it_can(self):
        o = pair_order(5, seed=3)
        assert len(o) == 5
        assert abs(o.count("AB") - o.count("BA")) <= 1

    def test_zero_pairs_is_empty(self):
        assert pair_order(0) == []


class TestPairwise:
    def test_the_delta_is_within_the_pair(self):
        assert pairwise_deltas([{"a": 10.0, "b": 4.0}]) == [6.0]

    def test_an_incomplete_pair_contributes_nothing(self):
        """A missing side is not a zero - that would invent an agreement
        that was never measured."""
        assert pairwise_deltas([{"a": 10.0, "b": None}]) == []

    def test_effect_smaller_than_the_spread_is_noise(self):
        pairs = [{"a": 10.0, "b": 9.0}, {"a": 5.0, "b": 8.0}]
        s = summarise_effect(pairs)
        assert s["n_pairs"] == 2
        assert s["spread"] > abs(s["mean_delta"])
        assert "noise" in s["note"]

    def test_a_consistent_sign_is_reported(self):
        s = summarise_effect([{"a": 10.0, "b": 5.0}, {"a": 9.0, "b": 4.0}])
        assert s["sign_consistent"] is True

    def test_an_inconsistent_sign_is_not(self):
        s = summarise_effect([{"a": 10.0, "b": 5.0}, {"a": 1.0, "b": 9.0}])
        assert s["sign_consistent"] is False

    def test_no_pairs_is_not_a_zero_effect(self):
        s = summarise_effect([])
        assert s["mean_delta"] is None and s["n_pairs"] == 0


class TestMeasurementCoverage:
    def test_a_run_missing_a_required_column_is_incomplete(self):
        runs = [{"frames": [{"damage_total": 0.0}]}]
        c = measurement_coverage(runs, "A")
        assert c["complete"] is False
        assert "cmd_gap_s" in c["missing"]

    def test_a_complete_run_is_complete(self):
        runs = [{"frames": [{"damage_total": 0.0, "reason": "",
                             "cmd_seq": 1, "cmd_gap_s": 0.1}]}]
        assert measurement_coverage(runs, "A")["complete"] is True

    def test_an_unknown_scenario_requires_nothing(self):
        assert measurement_coverage([], "Z")["required"] == []


class TestGate:
    def _full(self, **over):
        res = {}
        for sid in ALL_SCENARIOS:
            metrics = {k: (True if isinstance(v, bool) else v)
                       for k, v in SCENARIOS[sid]["release"].items()}
            res[sid] = [{"metrics": metrics}]
        res.update(over)
        return res

    def test_a_complete_passing_set_releases(self):
        g = gate(self._full())
        assert g["released"] is True

    def test_a_scenario_with_no_valid_runs_is_unknown_and_does_not_release(self):
        res = self._full()
        res["C"] = [{"excluded": True,
                     "exclusion_reason": "scenario_setup_failed"}]
        g = gate(res)
        assert g["scenarios"]["C"]["state"] == "UNKNOWN"
        assert g["released"] is False

    def test_a_collision_blocks_release(self):
        res = self._full()
        res["A"] = [{"metrics": {"collisions": 1, "unjustified_stops": 0}}]
        g = gate(res)
        assert g["scenarios"]["A"]["state"] == "FAIL"
        assert g["released"] is False

    def test_an_unmeasured_criterion_fails_not_passes(self):
        res = self._full()
        res["H"] = [{"metrics": {}}]
        g = gate(res)
        assert "not measured" in g["scenarios"]["H"]["reason"]

    def test_an_undeclared_exclusion_is_a_fail(self):
        res = self._full()
        res["B"] = [{"excluded": True, "exclusion_reason": "looked bad"}]
        g = gate(res)
        assert g["scenarios"]["B"]["state"] == "FAIL"

    def test_a_declared_exclusion_is_accepted(self):
        assert "contaminated" in VALID_EXCLUSIONS
        res = self._full()
        res["B"] = [{"excluded": True, "exclusion_reason": "contaminated"},
                    {"metrics": {k: (True if isinstance(v, bool) else v)
                                 for k, v in SCENARIOS["B"]["release"].items()}}]
        assert gate(res)["scenarios"]["B"]["state"] == "PASS"

    def test_zero_collisions_is_scoped_to_the_set(self):
        assert "THIS SET" in gate(self._full())["note"]


class TestCli:
    def test_it_prints_the_set(self, capsys):
        main([])
        out = capsys.readouterr().out
        assert "Minimum closed-loop scenario set" in out
        assert "the run" in out

    def test_it_generates_an_order(self, capsys):
        assert main(["--pairs", "4", "--seed", "5"]) == 0
        assert len(capsys.readouterr().out.split()) == 4

    def test_gating_a_file(self, tmp_path, capsys):
        import json
        p = tmp_path / "r.json"
        res = {}
        for sid in ALL_SCENARIOS:
            res[sid] = [{"metrics": {k: (True if isinstance(v, bool) else v)
                                     for k, v in SCENARIOS[sid]["release"].items()}}]
        p.write_text(json.dumps(res), encoding="utf-8")
        assert main(["--results", str(p)]) == 0
        assert "released: True" in capsys.readouterr().out
