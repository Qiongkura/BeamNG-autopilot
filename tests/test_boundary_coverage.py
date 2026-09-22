"""P4: a lateral safety claim needs a boundary that was actually seen."""

from __future__ import annotations

import json

import pytest

from scripts.m5_boundary_coverage import (
    MIN_COVERAGE_FOR_A_CLAIM,
    coverage,
    longest_gap_runs,
    main,
)


def _f(**kw):
    base = {"t": 0.0, "speed": 1.0, "lat_left": 2.0, "lat_right": 2.0,
            "road_surface": "on_road", "road_checked": 1}
    base.update(kw)
    return base


class TestGaps:
    def test_a_full_run_of_readings_has_no_gap(self):
        assert longest_gap_runs([_f()] * 5, "lat_left") == []

    def test_a_gap_is_counted_in_frames(self):
        frames = [_f(lat_left=None)] * 3 + [_f()]
        assert longest_gap_runs(frames, "lat_left") == [3]

    def test_several_gaps_are_listed_separately(self):
        frames = [_f(lat_left=None), _f(), _f(lat_left=None),
                  _f(lat_left=None)]
        assert longest_gap_runs(frames, "lat_left") == [1, 2]

    def test_a_missing_column_is_a_gap_not_a_reading(self):
        f = _f()
        del f["lat_left"]
        assert longest_gap_runs([f], "lat_left") == [1]


class TestCoverage:
    def test_full_coverage_supports_a_claim(self):
        c = coverage([_f()] * 4)
        assert c["lat_left"]["coverage"] == 1.0
        assert c["verdict"]["supports_a_lateral_claim"] is True

    def test_the_2026_09_20_shape_does_not(self):
        # 82-99% of town frames had no boundary reading at all.
        frames = [_f()] + [_f(lat_left=None, lat_right=None)] * 19
        c = coverage(frames)
        assert c["lat_left"]["coverage"] == 0.05
        assert c["verdict"]["supports_a_lateral_claim"] is False

    def test_the_worse_channel_decides(self):
        frames = [_f(lat_right=None)]
        c = coverage(frames)
        assert c["verdict"]["min_boundary_coverage"] == 0.0

    def test_checked_is_separate_from_reported(self):
        """A run that reports on_road but never checked cannot support a
        road claim, however good the states look."""
        frames = [_f(road_surface="on_road", road_checked=False)] * 3
        c = coverage(frames)
        assert c["road_surface"]["states"]["on_road"] == 3
        assert c["road_surface"]["checked_coverage"] == 0.0

    def test_a_run_with_no_road_column_at_all(self):
        frames = [{"t": 0.0}]
        c = coverage(frames)
        assert c["road_surface"]["states"]["None"] == 1
        assert c["road_surface"]["checked_coverage"] == 0.0

    def test_the_gap_is_reported_as_a_distance(self):
        # A duration hides the cost; P4 asks for both.
        frames = [_f(lat_left=None, lat_right=None, t=float(i), speed=5.0)
                  for i in range(5)]
        c = coverage(frames)
        assert c["no_boundary_gap"]["longest_frames"] == 5
        assert c["no_boundary_gap"]["distance_m"] > 0.0

    def test_no_speed_means_no_claimed_distance(self):
        frames = [_f(lat_left=None, lat_right=None, speed=None)]
        c = coverage(frames)
        assert c["no_boundary_gap"]["distance_m"] is None

    def test_an_empty_run_is_not_a_zero_coverage_run(self):
        c = coverage([])
        assert c["frames"] == 0
        assert c["lat_left"]["coverage"] is None
        assert c["verdict"]["supports_a_lateral_claim"] is None

    def test_real_serialized_road_checked_counts_integer_one(self):
        frames = json.loads(json.dumps([
            _f(road_checked=int(True)), _f(road_checked=int(False)),
            _f(road_checked=True), _f(road_checked=False),
        ]))
        c = coverage(frames)
        assert c["road_surface"]["checked_frames"] == 2
        assert c["road_surface"]["checked_coverage"] == 0.5

    @pytest.mark.parametrize("flag", [
        "1", "true", "False", None, 2, -1, 0.5,
        float("nan"), float("inf"), -float("inf"), {}, [],
    ])
    def test_road_checked_does_not_coerce_invalid_flags(self, flag):
        c = coverage([_f(road_checked=flag)])
        assert c["road_surface"]["checked_frames"] == 0

    @pytest.mark.parametrize("value", [
        float("nan"), float("inf"), -float("inf"), "2.0", True, {}, [],
    ])
    def test_invalid_boundary_values_are_gaps_not_coverage(self, value):
        c = coverage([_f(lat_left=value, lat_right=value)])
        assert c["lat_left"]["frames_with_reading"] == 0
        assert c["lat_right"]["longest_gap_frames"] == 1
        assert c["verdict"]["supports_a_lateral_claim"] is False

    def test_single_sided_gaps_are_not_a_double_boundary_gap(self):
        frames = [_f(lat_left=None, t=float(i)) for i in range(4)]
        frames += [_f(lat_right=None, t=float(i)) for i in range(4, 8)]
        c = coverage(frames)
        assert c["lat_left"]["longest_gap_frames"] == 4
        assert c["lat_right"]["longest_gap_frames"] == 4
        assert c["no_boundary_gap"]["longest_frames"] == 0
        assert c["no_boundary_gap"]["duration_s"] is None
        assert c["no_boundary_gap"]["distance_m"] is None

    def test_longest_double_gap_uses_its_own_intervals_and_speeds(self):
        frames = [
            _f(t=0.0, speed=80.0, lat_left=None, lat_right=None),
            _f(t=0.5, speed=80.0),
            _f(t=1.0, speed=2.0, lat_left=None, lat_right=None),
            _f(t=2.0, speed=4.0, lat_left=None, lat_right=None),
            _f(t=4.0, speed=6.0, lat_left=None, lat_right=None),
            _f(t=7.0, speed=80.0),
        ]
        gap = coverage(frames)["no_boundary_gap"]
        assert gap["longest_frames"] == 3
        assert gap["duration_s"] == 6.0
        assert gap["distance_m"] == 28.0  # 2*1 + 4*2 + 6*3
        assert gap["mean_speed_mps"] == pytest.approx(28.0 / 6.0)

    def test_a_trailing_gap_does_not_extrapolate_past_the_last_frame(self):
        frames = [_f(t=0.0, speed=90.0)]
        frames += [_f(t=t, speed=2.0, lat_left=None, lat_right=None)
                   for t in (1.0, 2.0, 4.0)]
        gap = coverage(frames)["no_boundary_gap"]
        assert gap["longest_frames"] == 3
        assert gap["duration_s"] == 3.0
        assert gap["distance_m"] == 6.0

    @pytest.mark.parametrize("value", [
        None, float("nan"), float("inf"), -float("inf"), "1.0", True,
    ])
    def test_missing_or_invalid_gap_speed_does_not_borrow_other_samples(
            self, value):
        frames = [
            _f(t=0.0, speed=50.0),
            _f(t=1.0, speed=value, lat_left=None, lat_right=None),
            _f(t=2.0, speed=2.0, lat_left=None, lat_right=None),
            _f(t=3.0, speed=50.0),
        ]
        gap = coverage(frames)["no_boundary_gap"]
        assert gap["duration_s"] == 2.0
        assert gap["distance_m"] is None
        assert gap["mean_speed_mps"] is None

    @pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -1.0])
    def test_invalid_internal_gap_timestamp_makes_duration_unknown(self, value):
        frames = [_f(t=t, lat_left=None, lat_right=None)
                  for t in (0.0, value, 2.0)]
        gap = coverage(frames)["no_boundary_gap"]
        assert gap["duration_s"] is None
        assert gap["distance_m"] is None


class TestCli:
    def test_it_names_the_blocker(self, tmp_path, capsys):
        import json
        p = tmp_path / "run.json"
        p.write_text(json.dumps([_f(lat_left=None, lat_right=None)] * 10),
                     encoding="utf-8")
        main([str(p)])
        out = capsys.readouterr().out
        assert "NOT SUPPORTED" in out
        assert "sensor-coverage blocker" in out

    def test_the_threshold_is_the_declared_one(self):
        assert MIN_COVERAGE_FOR_A_CLAIM == 0.50

    def test_a_missing_file_is_not_a_crash(self, tmp_path):
        assert main([str(tmp_path / "nope.json")]) == 1

    def test_cli_counts_integer_flags_in_existing_json_output(self, tmp_path,
                                                             capsys):
        p = tmp_path / "run.json"
        p.write_text(json.dumps([_f(road_checked=1), _f(road_checked=0)]),
                     encoding="utf-8")
        out_path = tmp_path / "coverage.json"
        assert main([str(p), "--json", str(out_path)]) == 0
        result = json.loads(out_path.read_text(encoding="utf-8"))[0]
        assert result["road_surface"]["checked_frames"] == 1
        assert "(50.0%)" in capsys.readouterr().out
