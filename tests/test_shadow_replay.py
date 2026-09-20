"""The shadow replay has to refuse to invent what the log lacks (P2.2)."""

import json

import pytest

from scripts.m5_shadow_replay import (
    NO_OBSTACLE_SENTINEL,
    closest_of,
    damage_onsets,
    main,
    nearer_than,
    open_frames,
    summarise,
)


def _f(**kw):
    base = {"t": 0.0, "corridor_open": False, "closest_obs_m": 50.0,
            "mon_target": 6.0, "speed": 0.0, "damage_total": 0.0,
            "reason": "safe"}
    base.update(kw)
    return base


class TestClosest:
    def test_a_real_distance_is_a_distance(self):
        assert closest_of(_f(closest_obs_m=3.2)) == 3.2

    def test_the_sentinel_is_not_a_distance(self):
        # 999 means "nothing detected".  Left in, it becomes the median.
        assert closest_of(_f(closest_obs_m=NO_OBSTACLE_SENTINEL)) is None

    def test_a_missing_column_is_not_zero(self):
        f = _f()
        del f["closest_obs_m"]
        assert closest_of(f) is None

    def test_zero_is_a_real_distance(self):
        assert closest_of(_f(closest_obs_m=0.0)) == 0.0


class TestBuckets:
    def test_only_open_frames_can_be_counted(self):
        frames = [_f(corridor_open=True, closest_obs_m=1.0),
                  _f(corridor_open=False, closest_obs_m=1.0)]
        assert len(open_frames(frames)) == 1

    def test_nearer_than_uses_the_real_distance(self):
        frames = [_f(corridor_open=True, closest_obs_m=1.5),
                  _f(corridor_open=True, closest_obs_m=9.0)]
        assert len(nearer_than(open_frames(frames), 2.0)) == 1

    def test_the_sentinel_is_never_counted_as_near(self):
        frames = [_f(corridor_open=True,
                     closest_obs_m=NO_OBSTACLE_SENTINEL)]
        assert nearer_than(open_frames(frames), 2.0) == []

    def test_missing_columns_are_reported_not_swallowed(self):
        frames = [_f(corridor_open=True), _f(corridor_open=True)]
        for f in frames:
            del f["closest_obs_m"]
        res = summarise(frames, "x")
        assert res["closest_missing_column"] == 2
        assert res["open_closest_min_m"] is None   # not 0.0

    def test_a_missing_corridor_open_column_is_reported(self):
        frames = [_f()]
        del frames[0]["corridor_open"]
        res = summarise(frames, "x")
        assert res["corridor_open_missing_column"] == 1
        assert res["corridor_open_frames"] == 0


class TestDamage:
    def test_an_increase_is_an_onset(self):
        frames = [_f(damage_total=0.0), _f(damage_total=0.0),
                  _f(damage_total=1.5)]
        assert damage_onsets(frames) == [2]

    def test_a_flat_run_has_no_onsets(self):
        assert damage_onsets([_f(damage_total=2.0)] * 5) == []

    def test_a_missing_channel_is_not_an_onset(self):
        frames = [_f(), _f()]
        for f in frames:
            del f["damage_total"]
        assert damage_onsets(frames) == []


class TestCli:
    def test_a_real_run_file_is_read(self, tmp_path, capsys):
        frames = [_f(corridor_open=True, closest_obs_m=1.2,
                     damage_total=0.0),
                  _f(corridor_open=True, closest_obs_m=1.2,
                     damage_total=2.0)]
        p = tmp_path / "run.json"
        p.write_text(json.dumps(frames), encoding="utf-8")
        assert main([str(p)]) == 0
        out = capsys.readouterr().out
        assert "corridor_open == True" in out
        assert "damage onsets at frame index : [1]" in out

    def test_a_missing_file_is_not_a_crash(self, tmp_path, capsys):
        assert main([str(tmp_path / "nope.json")]) == 1

    def test_an_unusable_set_exits_nonzero(self, tmp_path, capsys):
        p = tmp_path / "empty.json"
        p.write_text("[]", encoding="utf-8")
        assert main([str(p)]) == 1

    def test_it_says_the_grid_is_absent(self, tmp_path, capsys):
        """The run has no grid; the script must not imply otherwise."""
        p = tmp_path / "run.json"
        p.write_text(json.dumps([_f()]), encoding="utf-8")
        main([str(p)])
        assert "no grid" in capsys.readouterr().out
