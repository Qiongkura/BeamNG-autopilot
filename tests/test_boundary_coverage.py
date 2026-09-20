"""P4: a lateral safety claim needs a boundary that was actually seen."""

from scripts.m5_boundary_coverage import (
    MIN_COVERAGE_FOR_A_CLAIM,
    coverage,
    longest_gap_runs,
    main,
)


def _f(**kw):
    base = {"t": 0.0, "speed": 1.0, "lat_left": 2.0, "lat_right": 2.0,
            "road_surface": "on_road", "road_checked": True}
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
