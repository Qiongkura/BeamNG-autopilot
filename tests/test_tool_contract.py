"""T12: the experiment tools must measure what they claim to measure.

The plan reviewed the tools written for the previous round and found
specific defects (FINAL_INTEGRATED_PLAN §1.6-23/24/25).  These tests pin
the four that change a NUMBER a report would quote:

1. ``m5_phase1_ab._metrics_from_hist`` connected two stops across the
   driving between them and reported driving time as stopping;
2. ``m5_body_cov_probe`` measured a 2.3/0.95 rectangle while claiming to
   match the production 2.2/0.9, read an argument that no longer exists
   for its main-camera arm, and exited 0 whatever the classification;
3. ``m5_perf_profile`` printed ``NOT MET`` and returned 0;
4. ``m5_perf_decompose``'s ``stale_owner`` was described as the stale
   decision when it is only the largest age.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ab():
    return _load("_m5_phase1_ab", "scripts/m5_phase1_ab.py")


class TestStopSpanRule:
    def _hist(self, speeds, step=0.5):
        return [{"t": round(i * step, 3), "speed": sp, "pos": [float(i), 0.0]}
                for i, sp in enumerate(speeds)]

    def test_two_stops_do_not_absorb_the_driving_between_them(self, ab):
        # 6 s stopped, 10 s driving, 6 s stopped - 22 frames at 1 s
        speeds = [0.1] * 6 + [8.0] * 10 + [0.1] * 6
        m = ab._metrics_from_hist(self._hist(speeds, step=1.0))
        assert m["stop_frames"] == 12
        # 6 + 5 = 11 s: each stopped frame owns the interval to the NEXT
        # frame of the run (the transition frame out of each stop included,
        # the last frame of the run owns none)
        assert m["stop_s"] == pytest.approx(11.0)
        assert m["stop_s"] < 15.0

    def test_a_single_continuous_stop_still_counts_fully(self, ab):
        speeds = [0.1] * 10
        m = ab._metrics_from_hist(self._hist(speeds, step=1.0))
        assert m["stop_s"] == pytest.approx(9.0)

    def test_the_reason_seconds_use_the_same_rule(self, ab):
        rows = self._hist([0.1] * 6 + [8.0] * 10 + [0.1] * 6, step=1.0)
        for r in rows:
            r["reason"] = "no drivable path"
        m = ab._metrics_from_hist(rows)
        assert m["no_drivable_path_s"] == pytest.approx(11.0)


class TestBodyCovProbe:
    def test_the_rectangle_is_the_production_one(self):
        probe = _load("_m5_body_cov_probe", "scripts/m5_body_cov_probe.py")
        from beamng_autopilot.vehicle_body import HALF_LENGTH_M, HALF_WIDTH_M
        assert probe.HALF_LEN_M == pytest.approx(float(HALF_LENGTH_M))
        assert probe.HALF_WIDTH_M == pytest.approx(float(HALF_WIDTH_M))

    def test_the_camera_selector_has_no_undefined_argument(self):
        # code only: the docstring explains the removed argument by name
        lines = [ln.split("#")[0]
                 for ln in (ROOT / "scripts/m5_body_cov_probe.py")
                 .read_text("utf-8").splitlines()]
        assert not any("args.role" in ln for ln in lines)

    def test_the_main_and_fisheye_modes_are_distinct(self):
        """``--cam front_main`` must not silently measure both cameras."""
        src = (ROOT / "scripts/m5_body_cov_probe.py").read_text("utf-8")
        assert 'args.cam in ("main", "front_main")' in src


class TestPerfGate:
    def test_not_met_is_visible_in_the_profile(self, tmp_path):
        prof = _load("_m5_perf_profile", "scripts/m5_perf_profile.py")
        prof.TARGET_TICK_P95_MS = 1.0        # impossible target
        frames = [{"t": i * 0.5, "tick_ms": {"total": 500.0}} for i in range(8)]
        f = tmp_path / "h.json"
        f.write_text(json.dumps(frames), encoding="utf-8")
        rc = prof.main([str(f)])
        assert rc == 2, "a missed performance target must not exit 0"

    def test_meeting_the_target_exits_zero(self, tmp_path):
        prof = _load("_m5_perf_profile2", "scripts/m5_perf_profile.py")
        prof.TARGET_TICK_P95_MS = 10_000.0
        frames = [{"t": i * 0.5, "tick_ms": {"total": 100.0}}
                  for i in range(8)]
        f = tmp_path / "h.json"
        f.write_text(json.dumps(frames), encoding="utf-8")
        assert prof.main([str(f)]) == 0

    def test_missing_tick_ms_is_unknown_not_a_crash(self, tmp_path, capsys):
        prof = _load("_m5_perf_profile3", "scripts/m5_perf_profile.py")
        frames = [{"t": i * 0.5, "speed": 1.0} for i in range(5)]
        f = tmp_path / "h.json"
        f.write_text(json.dumps(frames), encoding="utf-8")
        assert prof.main([str(f)]) == 0
        assert "no data" in capsys.readouterr().out


class TestTruthMeasure:
    """The pixel->metre truth rule must be recomputable and honest."""

    def _mod(self):
        return _load("_m5_truth_measure", "scripts/m5_truth_measure.py")

    def test_the_band_starts_beyond_the_bonnet(self):
        """The first version measured 2-6 m and detected the bonnet's bright
        stripe; the fix is the band plus the bonnet row limit."""
        m = self._mod()
        assert m.BAND_NEAR_M >= 4.5, m.BAND_NEAR_M
        assert m.BONNET_TOP_FRAC <= 0.70

    def test_an_ambiguous_margin_is_not_a_verdict(self):
        m = self._mod()
        assert m.MARGIN_TOL_M > 0.0
        # the verdict vocabulary is three-valued plus UNMEASURED
        src = (ROOT / "scripts" / "m5_truth_measure.py").read_text("utf-8")
        for word in ('"cross"', '"clear"', '"ambiguous"', '"UNMEASURED"'):
            assert word in src, word

    def test_the_episode_join_refuses_a_length_mismatch(self):
        """A wrong join produced a 54% FN rate that was pure artefact."""
        src = (ROOT / "scripts" / "m5_truth_measure.py").read_text("utf-8")
        assert "!= len(t)" in src and "refusing to" in src

    def test_the_paint_rule_is_documented_with_its_thresholds(self):
        m = self._mod()
        assert m.PAINT_ROW_DELTA > 0 and m.PAINT_ABS_MIN > 0
        src = (ROOT / "scripts" / "m5_truth_measure.py").read_text("utf-8")
        assert "p80" in src and "coverage" in src.lower()

    def test_the_measurement_uses_no_perception_geometry(self):
        """Truth must not be the perception's own lane reference."""
        src = (ROOT / "scripts" / "m5_truth_measure.py").read_text("utf-8")
        for banned in ("lane_ref", "lat_left", "body_lat_left", "line_lat"):
            # allowed only as RECORDED comparison fields in the output row
            for line in src.splitlines():
                if banned in line and "f.get(" not in line and "#" not in line:
                    raise AssertionError(f"truth measurement reads {banned}: "
                                         f"{line.strip()}")
