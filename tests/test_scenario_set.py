"""P5: a scenario set that can actually fail, and runs that can be paired."""

from __future__ import annotations

import json

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


def _run(sid):
    """Measured fixture using fsd_drive's scalar and nested JSON encodings."""
    values = {
        "damage_total": 0.0, "reason": "", "cmd_seq": 1, "cmd_gap_s": 0.1,
        "corridor_state": "infeasible", "corridor_reason": "unreachable",
        "body_cross_current": 0, "road_surface": "on_road", "road_checked": 1,
        "mon_target": 0.0, "closest_obs_m": 2.0, "target_sm": 0.0,
        "throttle": 0.0, "brake": 1.0, "cmd_t": 100.0,
        "head_sched": {"object": {"state": "ran", "compute_ms": 1.0}},
        "head_age_s": {"object": 0.1},
        "range_sched": {"state": "scan", "compute_ms": None},
        "consumed": {"object": {"result_seq": 1, "source_seq": 1,
                                  "age_s": 0.1}},
        "road_lost_s": 0.0, "lat_left": -2.0, "lat_right": 2.0,
        "tick_ms": {"total": 10.0}, "budget_s": 0.1, "watchdog": "brake",
        "risk_closest_m": 2.0, "min_ttc": 1.0,
    }
    frame = {col: values[col] for col in SCENARIOS[sid]["requires"]
             if col != "min_ttc_s"}
    if "min_ttc_s" in SCENARIOS[sid]["requires"]:
        frame["min_ttc"] = values["min_ttc"]
    metrics = dict(SCENARIOS[sid]["release"])
    return {"frames": [frame], "metrics": metrics}


def _full(**over):
    res = {sid: [_run(sid)] for sid in ALL_SCENARIOS}
    res.update(over)
    return res


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

    def test_no_runs_do_not_supply_required_measurements(self):
        assert measurement_coverage([], "A")["complete"] is False

    @pytest.mark.parametrize("sid", ALL_SCENARIOS)
    def test_every_scenario_requires_usable_values_not_just_column_names(
            self, sid):
        for col in SCENARIOS[sid]["requires"]:
            for remove in (False, True):
                run = _run(sid)
                field = "min_ttc" if col == "min_ttc_s" else col
                if remove:
                    del run["frames"][0][field]
                else:
                    run["frames"][0][field] = None
                c = measurement_coverage([run], sid)
                assert c["complete"] is False, (sid, col, remove)
                assert col in c["missing"], (sid, col, remove)

    def test_one_good_frame_does_not_hide_a_partial_column(self):
        run = _run("A")
        run["frames"].append(dict(run["frames"][0]))
        del run["frames"][1]["cmd_gap_s"]
        c = measurement_coverage([run], "A")
        assert c["complete"] is False
        assert "cmd_gap_s" in c["missing"]

    @pytest.mark.parametrize("frames", [None, [], {}, [None], ["cmd_gap_s"]])
    def test_unusable_frame_containers_are_incomplete(self, frames):
        assert measurement_coverage([{"frames": frames}], "A")["complete"] is False

    @pytest.mark.parametrize("value", [
        float("nan"), float("inf"), -float("inf"), "0.1", False, {}, [],
    ])
    def test_numeric_evidence_must_be_finite_and_numeric(self, value):
        run = _run("A")
        run["frames"][0]["cmd_gap_s"] = value
        assert measurement_coverage([run], "A")["complete"] is False

    @pytest.mark.parametrize("sid,col,value", [
        ("B", "road_checked", "1"),
        ("B", "body_cross_current", "0"),
        ("B", "body_cross_current", 2),
        ("B", "road_surface", "not_a_state"),
        ("C", "corridor_state", ""),
        ("D", "head_sched", {}),
        ("D", "head_sched", {"object": None}),
        ("D", "head_sched", {"object": {"state": None}}),
        ("E", "head_age_s", {"object": float("inf")}),
        ("E", "head_age_s", {"object": 0.1, "semantic": None}),
        ("E", "range_sched", {}),
        ("E", "consumed", {}),
        ("E", "consumed", {"object": {"result_seq": None}}),
        ("G", "tick_ms", {"ring": 1.0, "total": float("nan")}),
        ("G", "tick_ms", {"ring": 1.0}),
    ])
    def test_structured_evidence_cannot_be_an_empty_or_invalid_placeholder(
            self, sid, col, value):
        run = _run(sid)
        run["frames"][0][col] = value
        assert measurement_coverage([run], sid)["complete"] is False

    @pytest.mark.parametrize("sid", ["B", "F"])
    def test_unchecked_road_defaults_are_not_measurements(self, sid):
        run = _run(sid)
        run["frames"][0]["road_checked"] = 0
        c = measurement_coverage([run], sid)
        assert c["complete"] is False
        assert "road_surface" in c["missing"]
        if sid == "F":
            assert "road_lost_s" in c["missing"]

    def test_h_accepts_the_actual_ttc_name_and_the_existing_protocol_name(self):
        run = _run("H")
        assert measurement_coverage([run], "H")["complete"] is True
        run["frames"][0]["min_ttc_s"] = run["frames"][0].pop("min_ttc")
        assert measurement_coverage([run], "H")["complete"] is True

    def test_a_present_but_invalid_ttc_is_not_hidden_by_an_alias(self):
        run = _run("H")
        run["frames"][0]["min_ttc_s"] = None
        assert measurement_coverage([run], "H")["complete"] is False

    def test_expected_boundary_loss_is_recorded_but_is_not_a_reading(self):
        run = _run("F")
        run["frames"][0].update(lat_left=None, lat_right=None,
                                lane_sel="perception-unavailable")
        c = measurement_coverage([run], "F")
        assert c["complete"] is True
        assert c["columns"]["lat_left"]["measured_frames"] == 0
        assert c["columns"]["lat_left"]["unavailable_frames"] == 1
        assert c["columns"]["lat_left"]["unknown_frames"] == 0
        assert gate(_full(F=[run]))["released"] is True
        del run["frames"][0]["lane_sel"]
        assert gate(_full(F=[run]))["scenarios"]["F"]["state"] == "UNKNOWN"

    @pytest.mark.parametrize("availability", [False, 0])
    def test_cold_start_has_no_numeric_age_or_consumed_result(self, availability):
        run = _run("E")
        run["frames"][0].update(
            head_sched={"object": {"state": "async_in_flight",
                                   "result_seq": None,
                                   "result_available": availability}},
            head_age_s={"object": None},
            consumed={"object": {"result_seq": None, "source_seq": 1,
                                  "age_s": None}})
        c = measurement_coverage([run], "E")
        assert c["complete"] is True
        assert c["columns"]["head_age_s"]["unavailable_frames"] == 1
        assert c["columns"]["consumed"]["measured_frames"] == 0
        assert gate(_full(E=[run]))["released"] is True
        run["frames"][0]["head_sched"]["object"]["result_available"] = "False"
        assert gate(_full(E=[run]))["scenarios"]["E"]["state"] == "UNKNOWN"

    def test_finite_age_cannot_hide_an_explicitly_unavailable_result(self):
        run = _run("E")
        run["frames"][0]["head_sched"]["object"].update(
            result_seq=None, result_available=False)
        assert measurement_coverage([run], "E")["complete"] is False

    def test_every_scheduled_head_needs_an_age_and_consumption_record(self):
        run = _run("E")
        run["frames"][0]["head_sched"]["semantic"] = {"state": "ran"}
        c = measurement_coverage([run], "E")
        assert "head_age_s" in c["missing"]
        assert "consumed" in c["missing"]

    def test_partial_coverage_reports_unknown_frames_not_just_runs(self):
        run = _run("A")
        run["frames"].append({**run["frames"][0], "damage_total": None})
        run["frames"].append(dict(run["frames"][0]))
        del run["frames"][2]["damage_total"]
        c = measurement_coverage([run], "A")
        d = c["columns"]["damage_total"]
        assert d["frames"] == 3
        assert d["measured_frames"] == 1
        assert d["missing_column"] == 1
        assert d["unknown_frames"] == 2
        assert d["coverage"] == pytest.approx(1.0 / 3.0)


class TestGate:
    def test_a_complete_passing_set_releases(self):
        g = gate(_full())
        assert g["released"] is True
        assert all(c["state"] == "PASS" for c in g["scenarios"].values())

    def test_a_scenario_with_no_valid_runs_is_unknown_and_does_not_release(self):
        res = _full()
        res["C"] = [{"excluded": True,
                     "exclusion_reason": "scenario_setup_failed"}]
        g = gate(res)
        assert g["scenarios"]["C"]["state"] == "UNKNOWN"
        assert g["released"] is False

    def test_a_collision_blocks_release_even_with_missing_evidence(self):
        # Match eval.score_run: a measured violation takes precedence over
        # UNKNOWN, but an unmeasured metric alone is not a measured FAIL.
        res = _full(A=[{"metrics": {"collisions": 1}}])
        g = gate(res)
        assert g["scenarios"]["A"]["state"] == "FAIL"
        assert g["released"] is False

    def test_an_unmeasured_criterion_is_unknown_not_a_measured_failure(self):
        res = _full()
        del res["H"][0]["metrics"]["collisions"]
        g = gate(res)
        assert g["scenarios"]["H"]["state"] == "UNKNOWN"
        assert "not measured" in g["scenarios"]["H"]["reason"]
        assert g["released"] is False

    def test_an_undeclared_exclusion_is_a_fail(self):
        res = _full()
        res["B"] = [{"excluded": True, "exclusion_reason": "looked bad"}]
        g = gate(res)
        assert g["scenarios"]["B"]["state"] == "FAIL"

    def test_a_declared_exclusion_is_accepted(self):
        assert "contaminated" in VALID_EXCLUSIONS
        res = _full()
        res["B"].append({"excluded": True, "exclusion_reason": "contaminated"})
        assert gate(res)["scenarios"]["B"]["state"] == "PASS"

    @pytest.mark.parametrize("value", ["false", "true", "1", 2, float("nan")])
    def test_an_invalid_exclusion_flag_cannot_hide_a_run(self, value):
        res = _full()
        res["B"].append({"excluded": value, "exclusion_reason": "contaminated"})
        g = gate(res)
        assert g["scenarios"]["B"]["state"] == "UNKNOWN"
        assert g["released"] is False

    def test_zero_collisions_is_scoped_to_the_set(self):
        assert "THIS SET" in gate(_full())["note"]

    def test_metrics_without_any_frame_evidence_cannot_release(self):
        res = _full()
        for runs in res.values():
            del runs[0]["frames"]
        g = gate(res)
        assert all(c["state"] == "UNKNOWN" for c in g["scenarios"].values())
        assert g["released"] is False

    @pytest.mark.parametrize("sid", ALL_SCENARIOS)
    def test_every_scenario_gates_on_each_required_column(self, sid):
        for col in SCENARIOS[sid]["requires"]:
            res = _full()
            field = "min_ttc" if col == "min_ttc_s" else col
            del res[sid][0]["frames"][0][field]
            g = gate(res)
            assert g["scenarios"][sid]["state"] == "UNKNOWN", (sid, col)
            assert col in g["scenarios"][sid]["reason"]
            assert g["released"] is False

    def test_an_incomplete_run_is_not_hidden_by_a_complete_run(self):
        res = _full()
        partial = _run("A")
        del partial["metrics"]["unjustified_stops"]
        res["A"].append(partial)
        assert gate(res)["scenarios"]["A"]["state"] == "UNKNOWN"

    @pytest.mark.parametrize("metrics", [None, [], "0", {}])
    def test_missing_or_invalid_metric_containers_are_unknown(self, metrics):
        res = _full()
        res["A"][0]["metrics"] = metrics
        g = gate(res)
        assert g["scenarios"]["A"]["state"] == "UNKNOWN"
        assert g["released"] is False

    @pytest.mark.parametrize("value", [
        None, "0", False, float("nan"), float("inf"), -float("inf"), {}, [],
    ])
    def test_invalid_numeric_release_metrics_are_unknown(self, value):
        res = _full()
        res["H"][0]["metrics"]["collisions"] = value
        g = gate(res)
        assert g["scenarios"]["H"]["state"] == "UNKNOWN"
        assert g["released"] is False

    @pytest.mark.parametrize("value", [
        None, "true", "false", "1", 2, -1, 0.5,
        float("nan"), float("inf"), -float("inf"), {}, [],
    ])
    def test_invalid_boolean_release_metrics_are_unknown(self, value):
        res = _full()
        res["G"][0]["metrics"]["watchdog_triggered"] = value
        g = gate(res)
        assert g["scenarios"]["G"]["state"] == "UNKNOWN"
        assert g["released"] is False

    @pytest.mark.parametrize("value,expected", [
        (True, "PASS"), (1, "PASS"), (1.0, "PASS"),
        (False, "FAIL"), (0, "FAIL"), (0.0, "FAIL"),
    ])
    def test_measured_boolean_criteria_accept_only_boolean_or_binary_scalars(
            self, value, expected):
        res = _full()
        res["G"][0]["metrics"]["watchdog_triggered"] = value
        g = gate(res)
        assert g["scenarios"]["G"]["state"] == expected
        assert g["released"] is (expected == "PASS")

    def test_raw_damage_does_not_invent_a_missing_collision_metric(self):
        res = _full()
        run = res["H"][0]
        run["frames"].append({**run["frames"][0], "damage_total": 10.0})
        del run["metrics"]["collisions"]
        assert gate(res)["scenarios"]["H"]["state"] == "UNKNOWN"


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
        p = tmp_path / "r.json"
        p.write_text(json.dumps(_full()), encoding="utf-8")
        assert main(["--results", str(p)]) == 0
        assert "released: True" in capsys.readouterr().out

    def test_missing_evidence_exits_nonzero_and_prints_unknown(self, tmp_path,
                                                              capsys):
        res = _full()
        del res["D"][0]["frames"][0]["throttle"]
        p = tmp_path / "incomplete.json"
        p.write_text(json.dumps(res), encoding="utf-8")
        assert main(["--results", str(p)]) == 1
        out = capsys.readouterr().out
        assert "UNKNOWN" in out and "throttle" in out
        assert "released: False" in out
