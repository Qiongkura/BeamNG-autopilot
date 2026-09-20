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


class TestControlInterval:
    def test_the_interval_is_measured_not_assumed(self):
        frames = [_f(t=0.0), _f(t=0.5), _f(t=1.0)]
        assert control_intervals(frames) == [0.5, 0.5]

    def test_a_missing_timestamp_breaks_the_interval_not_forces_zero(self):
        frames = [_f(t=0.0), _f(t=None), _f(t=1.0)]
        assert control_intervals(frames) == [1.0]

    def test_a_negative_step_is_dropped(self):
        # Non-monotonic time: a negative interval is a logging artefact,
        # not a control rate.
        assert control_intervals([_f(t=1.0), _f(t=0.0)]) == []


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
