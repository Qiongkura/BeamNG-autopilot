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


class TestFreezeHoldout:
    """T10: a locked holdout must be identifiable, and must refuse dev runs.

    The first two freezes written by ``m5_freeze_holdout.py`` produced the
    SAME digest for two different collections, because the identity was the
    role directory name (``front_main``) plus a frame counter.  A manifest
    that cannot tell two collections apart cannot lock anything.
    """

    def _collection(self, tmp_path, name: str, frames: int = 3):
        d = tmp_path / name / "front_main"
        d.mkdir(parents=True, exist_ok=True)
        for i in range(frames):
            (d / f"frame_{i:05d}.npz").write_bytes(b"")
        return d

    def test_two_collections_of_the_same_view_do_not_collide(self, tmp_path):
        from beamng_autopilot.vision.dataset_split import (SplitPlan,
                                                           freeze_testset)
        fh = _load("m5_freeze_holdout", "scripts/m5_freeze_holdout.py")
        a = self._collection(tmp_path, "coll_a")
        b = self._collection(tmp_path, "coll_b")
        ra, _ = fh._refs_for_run(a, 0)
        rb, _ = fh._refs_for_run(b, 0)
        assert {r.run for r in ra} == {"coll_a/front_main"}
        assert {r.run for r in rb} == {"coll_b/front_main"}
        da = freeze_testset(SplitPlan(val=ra), tmp_path / "a.json")["digest"]
        db = freeze_testset(SplitPlan(val=rb), tmp_path / "b.json")["digest"]
        assert da != db, "two collections must not share a digest"

    def test_a_development_run_is_refused(self, tmp_path, capsys):
        fh = _load("m5_freeze_holdout", "scripts/m5_freeze_holdout.py")
        d = self._collection(tmp_path, "coll_a")
        rc = fh.main(["--runs", str(d), "--dev-runs", "front_main",
                      "--out", str(tmp_path / "frozen.json")])
        assert rc == 3
        assert "REFUSING" in capsys.readouterr().out
        assert not (tmp_path / "frozen.json").exists()

    def test_holdout_all_freezes_every_frame_and_checks_back(self, tmp_path):
        from beamng_autopilot.vision.dataset_split import check_frozen_testset
        fh = _load("m5_freeze_holdout", "scripts/m5_freeze_holdout.py")
        d = self._collection(tmp_path, "coll_c", frames=4)
        out = tmp_path / "frozen.json"
        rc = fh.main(["--runs", str(d), "--holdout-all", "--min-frames", "4",
                      "--out", str(out), "--json", str(tmp_path / "rep.json")])
        assert rc == 0
        rep = json.loads((tmp_path / "rep.json").read_text(encoding="utf-8"))
        assert rep["n_val"] == 4 and rep["n_train"] == 0
        assert rep["check"]["ok"] is True
        from beamng_autopilot.vision.dataset_split import SplitPlan
        refs, _ = fh._refs_for_run(d, 0)
        assert check_frozen_testset(out, SplitPlan(val=refs))["ok"] is True

    def test_an_existing_manifest_is_not_silently_replaced(self, tmp_path,
                                                           capsys):
        fh = _load("m5_freeze_holdout", "scripts/m5_freeze_holdout.py")
        d = self._collection(tmp_path, "coll_d")
        out = tmp_path / "frozen.json"
        assert fh.main(["--runs", str(d), "--holdout-all", "--min-frames", "1",
                        "--out", str(out)]) == 0
        capsys.readouterr()
        assert fh.main(["--runs", str(d), "--holdout-all", "--min-frames", "1",
                        "--out", str(out)]) == 4
        assert "--force" in capsys.readouterr().out


class TestMarkingIdentityProbe:
    """T08 prerequisite: candidates confirmed against an INDEPENDENT source.

    The engine's annotation is rendered from its own materials, so it can
    confirm or deny a perception candidate without touching the model.  The
    maths must refuse to score where the comparison is undefined: a frame
    with no engine line pixels has no recall, and reporting 1.0 there would
    be "no data reads as pass".
    """

    def test_mask_comparison_reports_none_where_undefined(self):
        import numpy as np
        probe = _load("m5_marking_identity_probe",
                      "scripts/m5_marking_identity_probe.py")
        empty = np.zeros((4, 4), dtype=bool)
        full = np.ones((4, 4), dtype=bool)
        r = probe.compare_masks(empty, empty)
        assert r["precision"] is None and r["recall"] is None
        assert r["iou"] is None
        r2 = probe.compare_masks(full, empty)
        # precision HAS a denominator here, so 0.0 is a real measurement
        # ("none of what the model calls a line is confirmed"); recall has
        # none, so it stays None instead of becoming 1.0
        assert r2["precision"] == 0.0 and r2["recall"] is None

    def test_mask_comparison_math_and_sides(self):
        import numpy as np
        probe = _load("m5_marking_identity_probe",
                      "scripts/m5_marking_identity_probe.py")
        perc = np.zeros((4, 4), dtype=bool)
        eng = np.zeros((4, 4), dtype=bool)
        perc[1, 0:2] = True          # left half
        eng[1, 0:2] = True           # engine agrees
        eng[2, 3] = True             # engine has a line the model missed
        r = probe.compare_masks(perc, eng)
        assert r["n_perception_px"] == 2 and r["n_engine_px"] == 3
        assert r["n_intersection"] == 2
        assert r["precision"] == pytest.approx(1.0)
        assert r["recall"] == pytest.approx(2 / 3, abs=1e-3)
        assert r["recall_left"] == pytest.approx(1.0)
        assert r["engine_px_right"] == 1
        assert r["recall_right"] == pytest.approx(0.0)
        assert r["engine_sides"] == 2 and r["perception_sides"] == 1

    def test_a_shape_mismatch_is_reported_not_smoothed(self):
        import numpy as np
        probe = _load("m5_marking_identity_probe",
                      "scripts/m5_marking_identity_probe.py")
        r = probe.compare_masks(np.zeros((4, 4), dtype=bool),
                                np.zeros((3, 4), dtype=bool))
        assert "reason" in r

    def test_a_run_without_labels_is_refused(self, tmp_path):
        import numpy as np
        probe = _load("m5_marking_identity_probe",
                      "scripts/m5_marking_identity_probe.py")
        d = tmp_path / "front_main"
        d.mkdir()
        np.savez_compressed(d / "frame_00000.npz",
                            colour=np.zeros((8, 8, 3), dtype=np.uint8))
        res = probe.probe(d)
        assert "no label array" in res.get("reason", "")

    def test_an_empty_run_is_refused(self, tmp_path):
        probe = _load("m5_marking_identity_probe",
                      "scripts/m5_marking_identity_probe.py")
        d = tmp_path / "front_main"
        d.mkdir()
        res = probe.probe(d)
        assert "no frame_*.npz" in res.get("reason", "")

    def test_the_camera_model_is_read_or_reported_missing(self):
        probe = _load("m5_marking_identity_probe",
                      "scripts/m5_marking_identity_probe.py")
        assert probe.camera_from_meta({}, "front_main") is None
        assert probe.camera_from_meta({"cameras": {"front_main": {}}},
                                      "front_main") is None
        meta = {"cameras": {"front_main": {
            "offset": [0.0, 1.5, 1.4], "fwd": [0.0, 1.0, 0.0],
            "up": [0.0, 0.0, 1.0], "fov_deg": 65.0, "width": 536,
            "height": 403}}}
        cam = probe.camera_from_meta(meta, "front_main")
        assert cam is not None and cam.width == 536


class TestCandidateIdentity:
    """T08: identity from the engine's labels, in the VEHICLE frame.

    The engine's annotation is independent of the perception model, so it
    can assign a candidate its role (which side of the car, and whether it
    bounds the ego's lane).  These tests pin the pure parts: the role
    vocabulary, the near/far ranking, the three-valued label breakdown and
    the match/null-shift arithmetic.
    """

    def _probe(self):
        return _load("m5_marking_identity_probe",
                     "scripts/m5_marking_identity_probe.py")

    def test_role_comes_from_the_lateral_offset_sign(self):
        p = self._probe()
        assert p.role_of(2.0) == "left"
        assert p.role_of(-2.0) == "right"
        assert p.role_of(0.2) == "straddled"
        assert p.role_of(None) == "unknown"

    def test_the_nearest_line_on_a_side_bounds_the_lane(self):
        p = self._probe()
        lines = [{"lat_m": 5.0, "role": "left"}, {"lat_m": 1.8, "role": "left"},
                 {"lat_m": -3.0, "role": "right"}]
        out = p.assign_roles(lines)
        roles = {round(ln["lat_m"], 1): ln["role"] for ln in out}
        assert roles[1.8] == "near_left" and roles[5.0] == "far_left"
        assert roles[-3.0] == "near_right", \
            "the only line on a side is the near one"
        # the input list is not mutated: callers keep their own view
        assert lines[1]["role"] == "left"

    def test_the_label_breakdown_separates_paint_road_and_offroad(self):
        import numpy as np
        p = self._probe()
        label = np.zeros((4, 4), dtype=np.uint8)
        label[0, :] = 2          # paint
        label[1, :] = 1          # road
        b = p.candidate_label_breakdown(np.array([[0.0, 0.0], [1.0, 1.0]]),
                                        label)
        assert b["n_px"] == 2
        assert b["on_line_frac"] == 0.5 and b["on_road_frac"] == 0.5
        assert b["off_road_frac"] == pytest.approx(0.0)
        # a pixel outside the image is clamped, never dropped silently
        b2 = p.candidate_label_breakdown(np.array([[99.0, 99.0]]), label)
        assert b2["n_px"] == 1

    def test_matching_uses_the_lateral_tolerance(self):
        p = self._probe()
        engine = [{"lat_m": -1.3, "role": "near_right"},
                  {"lat_m": 5.0, "role": "near_left"}]
        assert p.match_candidate(-1.7, engine) is not None
        assert p.match_candidate(-2.5, engine) is None
        # the CLOSEST line wins, not the first one found
        engine2 = [{"lat_m": 5.0, "role": "near_left"},
                   {"lat_m": 5.2, "role": "far_left"}]
        assert p.match_candidate(5.15, engine2)["lat_m"] == 5.2

    def test_the_projection_respects_its_range_window(self):
        import json as _json
        import numpy as np
        p = self._probe()
        meta = _json.loads((ROOT / "logs" / "m5_seg" /
                            "ident_probe_straight_20260923" /
                            "meta.json").read_text(encoding="utf-8"))
        cam = p.camera_from_meta(meta, "front_main")
        if cam is None:
            pytest.skip("no recorded camera model")
        # a horizon pixel has no ground hit; a near-road pixel does
        horizon = p.project_pixels([cam.cx], [10], cam)
        assert len(horizon) == 0
        near = p.project_pixels([cam.cx], [int(cam.height * 0.85)], cam)
        assert len(near) == 1 and 0.0 < float(near[0, 0]) < 30.0


class TestFpMaterialDecomp:
    """T11 baseline: where the line channel's false positives come from.

    The decomposition is only as good as its reference: the engine's own
    palette.  A run without a saved palette frame, or without a palette at
    all, must say so instead of reporting an empty table as a clean result.
    """

    def _tool(self):
        return _load("m5_fp_material_decomp",
                     "scripts/m5_fp_material_decomp.py")

    def test_the_material_lut_skips_malformed_entries(self):
        tool = self._tool()
        lut = tool.material_lut({"ASPHALT": [28, 28, 28], "BROKEN": None,
                                 "SHORT": [1, 2], "GRASS": [10, 200, 10]})
        assert lut[(28, 28, 28)] == "ASPHALT"
        assert lut[(10, 200, 10)] == "GRASS"
        assert len(lut) == 2

    def test_a_run_without_palette_frames_is_refused(self, tmp_path):
        import numpy as np
        tool = self._tool()
        d = tmp_path / "run" / "front_main"
        d.mkdir(parents=True)
        np.savez_compressed(d / "frame_00000.npz",
                            colour=np.zeros((8, 8, 3), dtype=np.uint8),
                            label=np.zeros((8, 8), dtype=np.uint8))
        res = tool.decompose(tmp_path / "run",
                             {"ASPHALT": [28, 28, 28]})
        assert "annotation_raw" in res.get("reason", "")

    def test_an_empty_palette_is_refused(self, tmp_path):
        import numpy as np
        tool = self._tool()
        d = tmp_path / "run" / "front_main"
        d.mkdir(parents=True)
        np.savez_compressed(d / "frame_00000.npz",
                            colour=np.zeros((8, 8, 3), dtype=np.uint8),
                            label=np.zeros((8, 8), dtype=np.uint8),
                            annotation_raw=np.zeros((8, 8, 3),
                                                    dtype=np.uint8))
        res = tool.decompose(tmp_path / "run", {})
        assert "no palette" in res.get("reason", "")

    def test_a_missing_run_directory_is_refused(self, tmp_path):
        tool = self._tool()
        res = tool.decompose(tmp_path / "nope", {"ASPHALT": [28, 28, 28]})
        assert "no frames" in res.get("reason", "")

    def test_the_palette_sidecar_is_read_without_the_simulator(self, tmp_path):
        tool = self._tool()
        (tmp_path / "palette_classes.json").write_text(
            json.dumps({"ASPHALT": [28, 28, 28]}), encoding="utf-8")
        pal = tool.load_palette(tmp_path, fetch_live=True)
        assert pal == {"ASPHALT": [28, 28, 28]}, \
            "a saved sidecar must win, so no simulator is touched"


class TestRingCollectorStepM:
    """A stationary collector run is 20 copies of one frame, not a sequence.

    Measured on a flat stretch: with only ``--step`` the car never moved and
    the first and last poses were identical - a duplicate sample set that a
    freeze would have locked as an "evaluation sequence".  ``--step-m``
    places the car forward before each grab, the same semantics the boundary
    capture tool uses, and the help text says why.
    """

    def test_the_collector_exposes_step_m_with_its_reason(self):
        src = (ROOT / "scripts" / "m5_collect_seg_ring.py").read_text(
            encoding="utf-8")
        assert '"--step-m"' in src
        assert "identical frames" in src, \
            "the flag must carry the trap it was added for"
        assert "safe_teleport" in src


class TestCollectedMapIdentity:
    """The recorded map must be the SESSION's, not the argument's.

    Measured defect: the collector wrote the literal ``"italy"`` while the
    running session had ``east_coast_usa`` / ``gridmap_v2`` loaded, so two
    ring collections carry the wrong map and the training entry's map
    identity is wrong for them.  ``--map`` is only what the connector was
    told to load, and on ``--attach`` that says nothing about the session.
    """

    def _tool(self):
        return _load("m5_collect_seg_ring",
                     "scripts/m5_collect_seg_ring.py")

    class _Scenario:
        def __init__(self, level):
            self._level = level

        def get_current(self):
            if self._level is None:
                raise RuntimeError("scenario not started")
            return type("S", (), {"level": self._level})()

    def test_the_session_map_wins_over_the_argument(self):
        tool = self._tool()
        bng = type("B", (), {"scenario": self._Scenario("east_coast_usa")})()
        name, source = tool.session_map_name(bng, "italy")
        assert name == "east_coast_usa" and source != "argument-fallback"

    def test_a_session_that_cannot_be_asked_falls_back_and_says_so(self):
        tool = self._tool()
        name, source = tool.session_map_name(self._Scenario(None), "italy")
        assert name == "italy" and source == "argument-fallback"
        name, source = tool.session_map_name(object(), "gridmap_v2")
        assert name == "gridmap_v2" and source == "argument-fallback"

    def test_the_hardcoded_map_is_gone_from_the_meta(self):
        src = (ROOT / "scripts" / "m5_collect_seg_ring.py").read_text(
            encoding="utf-8")
        assert '"map_name": "italy"' not in src, \
            "the literal made two collections claim the wrong map"
        assert '"map_name_source"' in src


class TestFollowRoadPlacement:
    """T10/T11: staying ON the roadnet between placements.

    Measured need: a straight placed walk leaves a curving street within
    metres (and on east_coast_usa the 8-view ring plus map-wide teleport
    sampling killed the simulator session).  The roadnet-guided step keeps
    successive poses on the network; and when the car is OFF it - the
    east_coast spawn sat at ``road_px=22``, and the first version then
    reported "no forward neighbour" and captured nothing - the step SNAPS
    to the nearest node instead of failing.
    """

    def _tool(self):
        return _load("m5_collect_seg_ring",
                     "scripts/m5_collect_seg_ring.py")

    def _rn(self):
        import numpy as np

        class _RN:
            ready = True
            nodes = np.array([[0.0, 0.0], [10.0, 0.0], [30.0, 0.0]])

            def __init__(self):
                self.adj = {0: [(1, 10.0)], 1: [(0, 10.0), (2, 20.0)],
                            2: [(1, 20.0)]}

            def _nearest(self, p):
                import numpy as _np
                return int(_np.argmin(_np.linalg.norm(
                    self.nodes - _np.asarray(p, float)[:2], axis=1)))
        return _RN()

    def test_a_far_off_network_pose_is_snapped_onto_the_road(self):
        import numpy as np
        tool = self._tool()
        tgt, hdg, why = tool.follow_road_step(
            self._rn(), np.array([40.0, 30.0]), 0.0, 2.0)
        assert tgt is not None and "snapped" in why
        assert tgt[0] == 30.0 and tgt[1] == 0.0, "snapped to the node itself"

    def test_a_forward_pose_steps_along_the_edge(self):
        import numpy as np
        tool = self._tool()
        tgt, hdg, why = tool.follow_road_step(
            self._rn(), np.array([1.0, 0.0]), 0.0, 2.0)
        assert why == "" and tgt is not None
        assert tgt[0] > 1.0 and abs(tgt[1]) < 1e-6

    def test_a_backward_heading_stops_instead_of_reversing(self):
        import numpy as np
        tool = self._tool()
        tgt, _, why = tool.follow_road_step(
            self._rn(), np.array([1.0, 0.0]), np.pi, 2.0)
        assert tgt is None and "no forward neighbour" in why

    def test_an_unready_roadnet_is_reported(self):
        import numpy as np
        tool = self._tool()
        tgt, _, why = tool.follow_road_step(None, np.array([0.0, 0.0]), 0.0)
        assert tgt is None and "not ready" in why

    def test_the_collector_exposes_follow_road_with_its_reason(self):
        src = (ROOT / "scripts" / "m5_collect_seg_ring.py").read_text(
            encoding="utf-8")
        assert '"--follow-road"' in src and "ROADNET" in src


class TestFreezeHoldoutSizeGuard:
    """A 1-frame "holdout" is an artefact, not a test set.

    Measured: a walk that stopped after one placement still produced a run
    directory, and the freeze tool happily wrote a 1-frame manifest.  The
    guard refuses a set below ``--min-frames`` unless ``--force`` is given,
    so a failed collection cannot masquerade as an evaluation baseline.
    """

    def _tool(self):
        return _load("m5_freeze_holdout", "scripts/m5_freeze_holdout.py")

    def _run(self, tmp_path, frames: int):
        d = tmp_path / "coll" / "front_main"
        d.mkdir(parents=True, exist_ok=True)
        for i in range(frames):
            (d / f"frame_{i:05d}.npz").write_bytes(b"")
        return d

    def test_a_one_frame_set_is_refused(self, tmp_path, capsys):
        tool = self._tool()
        d = self._run(tmp_path, 1)
        rc = tool.main(["--runs", str(d), "--holdout-all",
                        "--out", str(tmp_path / "f.json")])
        assert rc == 5
        assert not (tmp_path / "f.json").exists()
        assert "artefact" in capsys.readouterr().out

    def test_force_still_allows_a_deliberate_tiny_set(self, tmp_path):
        tool = self._tool()
        d = self._run(tmp_path, 2)
        rc = tool.main(["--runs", str(d), "--holdout-all", "--min-frames", "2",
                        "--out", str(tmp_path / "f.json")])
        assert rc == 0 and (tmp_path / "f.json").is_file()

    def test_a_normal_set_passes_the_default_guard(self, tmp_path):
        tool = self._tool()
        d = self._run(tmp_path, 6)
        rc = tool.main(["--runs", str(d), "--holdout-all",
                        "--out", str(tmp_path / "f.json")])
        assert rc == 0


class TestMapAssocProbeModelFlag:
    """The map-prior A/B must be runnable with the SAME checkpoint as the
    identity work, or the two halves of T08 describe different models."""

    def test_the_probe_can_select_a_checkpoint(self):
        src = (ROOT / "scripts" / "m5_map_assoc_probe.py").read_text(
            encoding="utf-8")
        assert '"--model"' in src
        assert "Segmenter(" in src and "model_path=args.model" in src


class TestMapAssocPairedArms:
    """T08 余项: all three references are graded on ONE candidate set.

    ``in/outRadius`` are 3.5 m on this map, so the tangent has to come from
    the map graph's own centre-line polyline.  Two separate runs are NOT a
    controlled comparison (measured 32 vs 43 candidates between runs), so
    the chord / arc-tangent / roadnet-polyline arms are built per frame in
    one process; each arm reports its own direction deviation, its own
    paint confirmation and its own joint gate.
    """

    def _probe(self):
        return _load("m5_map_assoc_probe", "scripts/m5_map_assoc_probe.py")

    def _cand(self, probe, cand_id, bearing_deg, *, on_line=None,
              on_road=None, fresh=True):
        import math as _math
        c = probe.PerceptionCandidate(
            cand_id=cand_id, side="left", kind="solid",
            bearing_rad=(None if bearing_deg is None
                         else _math.radians(bearing_deg)),
            confidence=0.9, span_m=5.0, fresh=fresh)
        if on_line is not None:
            c.__dict__["on_line_frac"] = on_line
        if on_road is not None:
            c.__dict__["on_road_frac"] = on_road
        return c

    def _prior(self, probe, tangent_deg=None):
        import math as _math
        return probe.MapLinkPrior(
            link_id="a->b", direction_rad=0.0, lanes_hint=2,
            tangent_rad=(None if tangent_deg is None
                         else _math.radians(tangent_deg)),
            direction_source="chord")

    def test_the_direction_deviation_is_measured_per_reference(self):
        import math as _math
        probe = self._probe()
        cands = [self._cand(probe, "mk0", 0.0)]
        prior = self._prior(probe, tangent_deg=20.0)
        chord = probe._direction_error(cands, prior)
        arc = probe._direction_error(cands, prior, use_tangent=True,
                                     direction_source="tangent_arc")
        road = probe._direction_error(
            cands, probe.MapLinkPrior(link_id="a->b", direction_rad=0.0,
                                      tangent_rad=_math.radians(30.0)),
            use_tangent=True, direction_source="tangent_roadnet")
        assert chord["reference_source"] == "chord"
        assert chord["median_deg"] == 0.0 and chord["n"] == 1
        assert arc["median_deg"] == 20.0 and arc["within_hard_frac"] == 1.0
        assert road["median_deg"] == 30.0
        assert road["reference_source"] == "tangent_roadnet"

    def test_a_polyline_that_disagrees_is_reported_as_a_hard_deviation(self):
        import math as _math
        probe = self._probe()
        road = probe._direction_error(
            [self._cand(probe, "mk0", 0.0)],
            probe.MapLinkPrior(link_id="a->b", direction_rad=0.0,
                               tangent_rad=_math.radians(60.0)),
            use_tangent=True, direction_source="tangent_roadnet")
        assert road["median_deg"] == 60.0
        assert road["within_hard_frac"] == 0.0

    def test_an_unmeasurable_bearing_is_an_empty_block_not_a_zero(self):
        probe = self._probe()
        cand = self._cand(probe, "mk0", None)
        blk = probe._direction_error([cand], self._prior(probe))
        assert blk["n"] == 0 and "median_deg" not in blk
        assert blk["delta_deg"] == []

    def test_the_aggregate_keeps_one_block_per_arm(self):
        """Same frame, two arms, different verdicts - a paired comparison."""
        probe = self._probe()
        painted = self._cand(probe, "mk0", 0.0, on_line=0.8, on_road=0.8)
        unpainted = self._cand(probe, "mk1", 0.0, on_line=0.0, on_road=1.0)
        cands = [dict(c.__dict__) for c in (painted, unpainted)]
        prior = self._prior(probe, tangent_deg=60.0)
        arm_priors = {
            "chord": (prior, False, "chord"),
            "tangent_roadnet": (prior, True, "tangent_roadnet"),
        }
        rows, arms_dir = {}, {}
        for arm, (pr, use, src) in arm_priors.items():
            rep = probe.compare_arms([painted, unpainted], pr,
                                     use_tangent=use, direction_source=src)
            rows[arm] = rep["arm_b"]
            arms_dir[arm] = probe._direction_error([painted, unpainted], pr,
                                                   use_tangent=use,
                                                   direction_source=src)
        fr = {"frame": 0, "link_id": "a->b", "direction_deg": 0.0,
              "tangent_deg": 60.0, "tangent_alt_deg": None,
              "direction_source": "chord", "arc_offset_m": None,
              "radius_m": None, "n_candidates": 2, "rows": rows["chord"],
              "rank_changed": False, "arms": rows,
              "arms_direction": arms_dir,
              "arms_rank_changed": {a: False for a in arm_priors},
              "candidates": cands}
        agg = probe._aggregate([fr])
        assert agg["arm_order"] == ["chord", "tangent_roadnet"]
        chord, road = agg["arms"]["chord"], agg["arms"]["tangent_roadnet"]
        # the chord arm accepts both; only one of them is on engine paint
        assert (chord["accepted"], chord["hard"]) == (2, 0)
        assert chord["paint"]["false_acceptance"] == 1
        assert chord["paint"]["judged"] == 2
        assert chord["joint_gate"]["accepted"] == 1
        assert chord["joint_gate"]["dropped_road_only"] == 1
        # the 60 deg polyline tangent rejects both, so no false acceptance
        assert (road["accepted"], road["hard"]) == (0, 2)
        assert road["paint"]["false_acceptance"] == 0
        assert road["paint"]["false_rejection"] == 1
        assert road["joint_gate"]["dropped_paint_confirmed"] == 1
        assert road["direction"]["median_deg"] == 60.0
        assert chord["direction"]["median_deg"] == 0.0
        # top level stays the PRIMARY arm's view (old evidence comparable)
        assert agg["accepted"] == 2 and agg["arms"]["chord"]["accepted"] == 2
        assert agg["paint"]["false_acceptance"] == 1

    def test_the_joint_buckets_sum_to_the_judged_rows(self):
        """Every judged candidate lands in exactly one joint-gate bucket."""
        probe = self._probe()
        cands = [self._cand(probe, "p0", 0.0, on_line=0.9, on_road=0.9),
                 self._cand(probe, "p1", 0.0, on_line=0.0, on_road=0.9),
                 self._cand(probe, "p2", 0.0, on_line=0.0, on_road=0.0),
                 self._cand(probe, "p3", 60.0, on_line=0.9, on_road=0.9),
                 self._cand(probe, "p4", 0.0, on_line=None, on_road=0.9)]
        rows = [{"candidate_id": c.cand_id, "conflicts": []} for c in cands]
        rows[3] = {"candidate_id": "p3", "conflicts": [
            "direction_mismatch: candidate 60 deg off the map link direction"]}
        fr = {"frame": 0, "link_id": "a->b", "direction_deg": 0.0,
              "tangent_deg": None, "tangent_alt_deg": None,
              "direction_source": "chord", "arc_offset_m": None,
              "radius_m": None, "n_candidates": len(rows), "rows": rows,
              "rank_changed": False,
              "candidates": [dict(c.__dict__) for c in cands]}
        agg = probe._aggregate([fr])
        j = agg["joint_gate"]
        assert j["judged"] == 4, "p4 has no paint fraction and is not judged"
        assert j["accepted"] == 1
        assert j["dropped_paint_confirmed"] == 1
        assert j["dropped_road_only"] == 1
        assert j["dropped_off_road"] == 1
        assert j["buckets_sum_to_judged"] is True
        po = agg["paint_only_gate"]
        assert (po["accepted"], po["direction_conflicted"]) == (2, 1), \
            "paint alone keeps the direction-condemned candidate"

    def test_an_abstained_candidate_is_not_a_false_acceptance(self):
        probe = self._probe()
        cand = self._cand(probe, "mk0", 0.0, on_line=0.0, on_road=1.0,
                          fresh=False)
        rows = [{"candidate_id": "mk0", "conflicts": [],
                 "abstain": "candidate is not a fresh observation"}]
        fr = {"frame": 0, "link_id": "a->b", "direction_deg": 0.0,
              "tangent_deg": None, "tangent_alt_deg": None,
              "direction_source": "chord", "arc_offset_m": None,
              "radius_m": None, "n_candidates": 1, "rows": rows,
              "rank_changed": False,
              "candidates": [dict(cand.__dict__)]}
        agg = probe._aggregate([fr])
        assert agg["abstain"] == 1 and agg["accepted"] == 0
        p = agg["paint"]
        assert p["measured"] == 1 and p["judged"] == 0
        assert p["false_acceptance"] == 0
        assert p["false_acceptance_rate"] is None, \
            "a refusal has no rate - it is not a denominator"
        assert agg["joint_gate"]["accepted"] == 0

    def test_the_aggregate_reports_the_pooled_per_frame_medians(self):
        probe = self._probe()
        cands = [self._cand(probe, "mk0", 0.0)]
        out = []
        for k in range(2):
            out.append({"frame": k, "link_id": "a->b", "direction_deg": 0.0,
                        "tangent_deg": None, "tangent_alt_deg": None,
                        "direction_source": "chord", "arc_offset_m": None,
                        "radius_m": None, "n_candidates": 1, "rows": [],
                        "rank_changed": False, "candidates": [],
                        "arms": {"chord": []},
                        "arms_direction": {"chord": probe._direction_error(
                            cands, self._prior(probe))},
                        "arms_rank_changed": {"chord": False}})
        agg = probe._aggregate(out)
        d = agg["arms"]["chord"]["direction"]
        assert d["n"] == 2 and d["median_deg"] == 0.0
        assert d["frames_measured"] == 2
        assert d["per_frame_median_deg"] == [0.0, 0.0]

    def test_the_probe_grades_axes_and_counts_the_sense_flips(self):
        """A marking running the other way round is the SAME axis.

        Measured: on three driving-along runs (33/48/59 candidates) the chord
        agreed in sense every time, so the axis reduction is a no-op there
        and only removes the ~180 deg artefact elsewhere - but it must be
        reported when it happens, not silently dropped.
        """
        probe = self._probe()
        cands = [self._cand(probe, "mk0", 180.0)]
        blk = probe._direction_error(cands, self._prior(probe))
        assert blk["median_deg"] == 0.0, "an axis test is orientation-free"
        rows = probe.compare_arms(cands, self._prior(probe))["arm_b"]
        assert rows[0]["sense_delta_deg"] == pytest.approx(180.0)
        fr = {"frame": 0, "link_id": "a->b", "direction_deg": 0.0,
              "tangent_deg": None, "tangent_alt_deg": None,
              "direction_source": "chord", "arc_offset_m": None,
              "radius_m": None, "n_candidates": 1, "rows": rows,
              "rank_changed": False,
              "candidates": [dict(cands[0].__dict__)]}
        agg = probe._aggregate([fr])
        ds = agg["direction_sense"]
        assert ds["measured"] == 1 and ds["flipped"] == 1
        assert ds["flipped_frac"] == 1.0
        assert agg["accepted"] == 1, "a flipped sense is not a conflict"

    def test_the_cli_can_skip_the_map_graph(self):
        src = (ROOT / "scripts" / "m5_map_assoc_probe.py").read_text(
            encoding="utf-8")
        assert '"--no-roadnet"' in src


class TestCandidateBearingOrientation:
    """A marking stored backwards must not read as a 180-degree conflict.

    Measured live on two painted stretches: with the raw
    ``world[-1] - world[0]`` bearing, BOTH candidates on a straight link
    came back as "179 deg" / "154 deg" direction conflicts.  The extractor
    gives no near->far ordering guarantee, so the probe now orients the
    bearing away from the car using the ego heading ONLY (no map, no
    lateral geometry).
    """

    def _probe(self):
        return _load("m5_map_assoc_probe", "scripts/m5_map_assoc_probe.py")

    def _marking(self, world):
        import numpy as np

        class _M:
            kind = "solid"
            conf = 0.9
        m = _M()
        m.world = np.asarray(world, dtype=float)
        return m

    def test_a_backwards_polyline_gives_the_same_bearing(self):
        import numpy as np
        probe = self._probe()
        fwd_world = [[2.0, 0.0], [6.0, 0.0], [10.0, 0.0]]
        back_world = list(reversed(fwd_world))
        pos = np.zeros(3)
        a = probe._candidates_from_markings(
            [self._marking(fwd_world)], pos, 0.0)[0]
        b = probe._candidates_from_markings(
            [self._marking(back_world)], pos, 0.0)[0]
        assert a.bearing_rad == pytest.approx(0.0, abs=1e-9)
        assert b.bearing_rad == pytest.approx(a.bearing_rad, abs=1e-9), \
            "the stored order must not flip the reported direction"

    def test_the_bearing_always_points_away_from_the_car(self):
        import math
        import numpy as np
        probe = self._probe()
        # a line beside the car, stored from far to near
        world = [[10.0, 3.0], [6.0, 3.0], [2.0, 3.0]]
        c = probe._candidates_from_markings([self._marking(world)],
                                            np.zeros(3), 0.0)[0]
        assert abs(c.bearing_rad) < math.radians(45), \
            "sideways-but-forward, never backwards"
        assert c.side == "left"


class TestMapAssocFalseAcceptance:
    """T08's required quantity: accepted by the prior, not paint-confirmed.

    Measured on 12 real frames of a painted stretch (chord reference):
    49 candidates, 19 accepted, only 2 sitting on engine paint -> the
    strict pixel criterion puts the false-acceptance upper bound at 75%.
    The aggregator is a pure function, so the arithmetic is pinned here
    and an UNMEASURED paint check stays empty instead of reading as zero.
    """

    def _probe(self):
        return _load("m5_map_assoc_probe", "scripts/m5_map_assoc_probe.py")

    def _frame(self, rows, cands, source="chord", rank=False):
        return {"frame": 0, "link_id": "a->b", "direction_deg": 0.0,
                "tangent_deg": None, "tangent_alt_deg": None,
                "direction_source": source, "arc_offset_m": None,
                "radius_m": None, "n_candidates": len(rows), "rows": rows,
                "rank_changed": rank, "candidates": cands}

    def test_the_verdict_and_paint_counts(self):
        probe = self._probe()
        rows = [{"candidate_id": "c0", "conflicts": []},
                {"candidate_id": "c1",
                 "conflicts": ["direction_mismatch: 60 deg off (hard 45)"]},
                {"candidate_id": "c2",
                 "conflicts": ["direction_loose: 30 deg off"]}]
        cands = [{"cand_id": "c0", "on_line_frac": 0.0, "on_road_frac": 1.0},
                 {"cand_id": "c1", "on_line_frac": 1.0, "on_road_frac": 0.0},
                 {"cand_id": "c2", "on_line_frac": 0.0, "on_road_frac": 0.0}]
        agg = probe._aggregate([self._frame(rows, cands, rank=True)])
        assert (agg["accepted"], agg["hard"], agg["soft"]) == (1, 1, 1)
        p = agg["paint"]
        assert (p["on_paint"], p["on_road_only"], p["off_road"]) == (1, 1, 1)
        # "accepted" here means the prior did NOT reject it: a soft conflict
        # has a penalty but still passes, so it counts as a false acceptance
        # when the paint check does not confirm it (c0 and c2).
        assert p["false_acceptance"] == 2, "not-rejected but not paint-confirmed"
        assert p["false_rejection"] == 1, "hard-conflicted but paint-confirmed"
        assert agg["rank_changed_frames"] == 1
        assert agg["direction_sources"] == {"chord": 1}

    def test_without_the_paint_check_the_block_stays_empty(self):
        probe = self._probe()
        rows = [{"candidate_id": "c0", "conflicts": []}]
        agg = probe._aggregate([self._frame(rows, [{"cand_id": "c0"}])])
        assert agg["accepted"] == 1
        assert agg["paint"] == {}, "an unmeasured quantity is not zero"

    def test_the_cli_exposes_frames_and_direction(self):
        src = (ROOT / "scripts" / "m5_map_assoc_probe.py").read_text(
            encoding="utf-8")
        assert '"--frames"' in src and '"--direction"' in src
        assert '"--annotations"' in src


class TestRoadnetPolylineTangent:
    """T08: the tangent comes from the map GRAPH, not the 3.5 m radii.

    ``inRadius/outRadius`` are not road arc radii on this map (measured),
    so the tangent arm now uses ``RoadNetwork.nearby_polylines`` - the same
    node graph the routing uses.  A +-10 m window on NODE arc positions can
    contain a single node (measured with a 10 m spacing), so the immediate
    neighbours are the fallback; a missing graph returns None rather than a
    guessed direction.
    """

    def _tool(self):
        return _load("m5_map_assoc_probe", "scripts/m5_map_assoc_probe.py")

    class _RN:
        ready = True

        def nearby_polylines(self, xy, radius=60.0):
            import numpy as _np
            return [_np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0],
                               [30.0, 3.0], [40.0, 6.0]])]

    def test_a_straight_run_gives_the_run_direction(self):
        import math
        import numpy as np
        tool = self._tool()
        t, span, off = tool.roadnet_tangent_rad(self._RN(),
                                                np.array([10.0, 0.0, 0.0]), 0.0)
        assert math.degrees(t) == pytest.approx(0.0, abs=1e-6)
        assert span == pytest.approx(20.0)
        assert off == pytest.approx(0.0)

    def test_a_bend_gives_the_local_direction_not_the_far_end(self):
        import math
        import numpy as np
        tool = self._tool()
        t, span, off = tool.roadnet_tangent_rad(
            self._RN(), np.array([33.0, 3.5, 0.0]), math.radians(15.0))
        assert t is not None, "the neighbour fallback must beat the window"
        deg = math.degrees(t)
        assert 10.0 < deg < 25.0, f"local bend direction, got {deg:.1f}"
        assert span is not None and span > 0.0

    def test_a_missing_graph_is_not_a_guessed_direction(self):
        import numpy as np
        tool = self._tool()
        assert tool.roadnet_tangent_rad(None, np.zeros(3), 0.0) == (None, None,
                                                                    None)
