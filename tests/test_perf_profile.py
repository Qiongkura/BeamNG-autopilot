"""The profile must be able to say "not met" and "not measured" (P3)."""

from scripts.m5_perf_profile import (
    TARGET_TICK_P95_MS,
    collect,
    control_intervals,
    describe,
    main,
    pct,
)


def _f(**kw):
    base = {"t": 0.0,
            "tick_ms": {"ring": 100.0, "range": 50.0, "plan": 10.0,
                        "total": 160.0},
            "frame_ms": {"local": 1.0, "tick": 160.0, "grid_mon": 2.0,
                         "rest": 3.0},
            "budget_s": 0.45,
            "head_sched": {"semantic": {"compute_ms": 90.0},
                           "object": {"compute_ms": 5.0}}}
    base.update(kw)
    return base


class TestPercentiles:
    def test_a_single_value_is_that_value(self):
        assert pct([7.0], 95) == 7.0

    def test_no_values_is_none_not_zero(self):
        assert pct([], 95) is None

    def test_p95_is_above_p50(self):
        vals = list(range(1, 101))
        assert pct(vals, 95) > pct(vals, 50)

    def test_p100_equivalent_is_the_max(self):
        assert pct([1.0, 5.0, 3.0], 100) == 5.0


class TestDescribe:
    def test_it_reports_n_and_the_spread(self):
        d = describe([1.0, 2.0, 3.0])
        assert d["n"] == 3
        assert d["max"] == 3.0
        assert d["mean"] == 2.0

    def test_empty_is_none_everywhere(self):
        d = describe([])
        assert d["n"] == 0
        assert d["p50"] is None and d["max"] is None and d["mean"] is None

    def test_over_limit_is_counted(self):
        d = describe([10.0, 200.0, 300.0], over_ms=150.0)
        assert d["over_limit_count"] == 2


class TestCollect:
    def test_a_missing_stage_is_not_a_zero_stage(self):
        f = _f()
        del f["tick_ms"]["plan"]
        prof = collect([f])
        assert prof["tick_ms"]["plan"]["n"] == 0
        assert prof["tick_ms"]["plan"]["p50"] is None

    def test_heads_are_only_counted_when_they_ran(self):
        f = _f(head_sched={"semantic": {"compute_ms": 90.0},
                           "object": {"compute_ms": None}})
        prof = collect([f])
        assert prof["head_compute_ms"]["semantic"]["n"] == 1
        assert "object" not in prof["head_compute_ms"]

    def test_over_budget_frames_are_counted(self):
        prof = collect([_f(tick_ms={"total": 500.0}),
                        _f(tick_ms={"total": 100.0})])
        assert prof["budget_s"]["over_budget_frames"] == 1

    def test_a_frame_with_no_budget_is_not_over_budget(self):
        f = _f()
        del f["budget_s"]
        assert collect([f])["budget_s"]["over_budget_frames"] == 0

    def test_legacy_ring_is_not_inferred_to_be_camera_time(self):
        prof = collect([_f()])
        acquire = prof["perception_ms"]["camera_acquire"]
        assert acquire["n"] == 0 and acquire["p50"] is None
        assert acquire["unknown"] == 1 and acquire["coverage"] == 0.0
        assert prof["camera_internal_ms"] is None

    def test_new_stages_report_measurement_coverage_without_double_counting(self):
        frames = [_f(perception_ms={"camera_acquire": 10.0, "heads": 90.0,
                                    "heads_sync": 85.0},
                     semantic_ms={"prediction": 60.0, "markings": 20.0,
                                  "total": 85.0},
                     segmentation_ms={"road": {"preprocess": 4.0,
                                                 "inference_decode": 50.0,
                                                 "postprocess": 6.0,
                                                 "total": 60.0},
                                      "line": None}), _f()]
        prof = collect(frames)
        assert prof["tick_ms"]["total"]["p50"] == 160.0
        acquired = prof["perception_ms"]["camera_acquire"]
        assert acquired["p50"] == 10.0
        assert acquired["n"] == 1 and acquired["unknown"] == 1
        assert acquired["coverage"] == 0.5
        assert prof["semantic_ms"]["prediction"]["n"] == 1
        assert prof["segmentation_ms"]["road"]["inference_decode"]["p50"] == 50.0
        assert prof["segmentation_ms"]["line"]["total"]["n"] == 0

    def test_invalid_stage_values_remain_unknown(self):
        frames = [_f(perception_ms={"camera_acquire": value})
                  for value in (None, -1.0, float("nan"), float("inf"), "bad")]
        result = collect(frames)["perception_ms"]["camera_acquire"]
        assert result["n"] == 0 and result["unknown"] == 5
        assert result["p50"] is None


class TestControlInterval:
    def test_the_interval_is_measured_not_assumed(self):
        frames = [_f(t=0.0), _f(t=0.5), _f(t=1.0)]
        assert control_intervals(frames) == [0.5, 0.5]

    def test_a_missing_timestamp_breaks_the_interval_not_forces_zero(self):
        frames = [_f(t=0.0), _f(t=None), _f(t=1.0)]
        assert control_intervals(frames) == []

    def test_a_negative_step_is_dropped(self):
        # Non-monotonic time: a negative interval is a logging artefact,
        # not a control rate.
        assert control_intervals([_f(t=1.0), _f(t=0.0)]) == []

    def test_command_receipts_take_precedence_over_frame_timestamps(self):
        frames = [_f(t=0.0, cmd_gap_s=0.1,
                     substep_commands=[{"cmd_gap_s": 0.05}]),
                  _f(t=5.0, cmd_gap_s=0.2,
                     substep_commands=[{"cmd_gap_s": 0.06}])]
        assert control_intervals(frames) == [0.1, 0.05, 0.2, 0.06]
        assert collect(frames)["control_interval_source"] == "command_receipts"

    def test_protective_send_retains_the_long_gap_before_a_short_main_send(self):
        frames = [_f(cmd_gap_s=0.2, protective_commands=[
            {"cmd_gap_s": 2.0, "watchdog": "brake", "watchdog_braked": True}])]
        assert control_intervals(frames) == [2.0, 0.2]
        assert collect(frames)["control_interval_s"]["max"] == 2.0

    def test_partial_command_trace_never_falls_back_to_frame_gaps(self):
        frames = [_f(t=0.0, cmd_gap_s=None), _f(t=5.0)]
        assert control_intervals(frames) == []
        assert collect(frames)["control_interval_s"]["n"] == 0

    def test_legacy_frame_intervals_are_labelled_as_proxy(self):
        frames = [_f(t=0.0), _f(t=0.5)]
        assert collect(frames)["control_interval_source"] == "frame_timestamps_proxy"

    def test_invalid_command_gaps_do_not_pollute_percentiles(self):
        frames = [_f(cmd_gap_s=float("nan"), substep_commands=[
            {"cmd_gap_s": -1.0}, {"cmd_gap_s": float("inf")},
            {"cmd_gap_s": 0.1}, "bad"])]
        assert control_intervals(frames) == [0.1]


class TestCli:
    def test_it_says_the_target_is_not_met(self, tmp_path, capsys):
        """P3: a target that is not met stays not met."""
        p = tmp_path / "run.json"
        p.write_text("[]", encoding="utf-8")
        import json
        p.write_text(json.dumps([_f()]), encoding="utf-8")
        main([str(p)])
        out = capsys.readouterr().out
        assert "NOT MET" in out
        assert "cannot be named" in out      # the ring caveat is printed

    def test_a_faster_run_can_meet_it(self, tmp_path, capsys):
        import json
        p = tmp_path / "fast.json"
        p.write_text(json.dumps([_f(tick_ms={"total": 40.0})]),
                     encoding="utf-8")
        main([str(p)])
        assert "MEETS" in capsys.readouterr().out
        assert TARGET_TICK_P95_MS == 150.0
