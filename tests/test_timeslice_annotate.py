"""T10: the time-slice annotator must assist, and must say what it is.

The plan proposes this tool (`scripts/m5_timeslice_annotate.py`), states
that it only applies to a continuous trackable identity, and requires the
result to be an ASSISTED label rather than reliable dense truth.  These
tests pin the arithmetic, the mandatory breaks, the schema compliance and
the claim that every produced label is an inferred extension - never
measured paint.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from beamng_autopilot.labeling import curve_schema as cs
from scripts import m5_timeslice_annotate as ts


def _frames(n: int = 12, u_stroke: int = 200, h: int = 240, w: int = 320):
    out = []
    for _ in range(n):
        img = np.tile(np.arange(w, dtype=np.uint8), (h, 1)).astype(np.float32)
        img = (img / 3.0).astype(np.uint8)
        img[:, u_stroke - 5:u_stroke + 5] = 255
        out.append(img)
    return out


class TestInterpolation:
    def test_linear_between_two_clicks(self):
        rows = ts.interpolate_rows([[0, 180], [10, 130]],
                                   frame_start=0, frame_end=10)
        assert rows[5] == pytest.approx(155.0)
        assert sorted(rows) == list(range(11))

    def test_nothing_is_extrapolated_beyond_the_clicks(self):
        rows = ts.interpolate_rows([[3, 180], [6, 150]],
                                   frame_start=0, frame_end=9)
        assert sorted(rows) == [3, 4, 5, 6]

    def test_a_break_is_not_bridged(self):
        rows = ts.interpolate_rows([[0, 180], [10, 130]],
                                   frame_start=0, frame_end=10, breaks=[5])
        assert sorted(rows) == [0, 10], "frames across a break are UNKNOWN"

    def test_a_click_on_the_break_frame_is_kept(self):
        rows = ts.interpolate_rows([[0, 180], [5, 160], [10, 130]],
                                   frame_start=0, frame_end=10, breaks=[5])
        assert rows[5] == 160.0
        # the break separates frame 5 from what came BEFORE it; the frames
        # after the click are still one continuous stretch
        assert 4 not in rows
        assert sorted(rows) == [0, 5, 6, 7, 8, 9, 10]


def _slanted_frames(n: int = 12, rows=(100, 150, 200), h: int = 240,
                    w: int = 320, drift_px_per_frame: float = 3.0):
    """Frames where the stroke's column depends on the ROW (a real line).

    ``drift_px_per_frame`` moves the stroke sideways over time, which is
    what a time slice at a fixed column window is supposed to show.
    """
    out = []
    for j in range(n):
        img = (np.tile(np.arange(w, dtype=np.uint8), (h, 1)) / 3.0).astype(
            np.uint8)
        for r in rows:
            u = int(200 - 0.5 * (r - 150) + drift_px_per_frame * j)
            if not (0 < u - 5 and u + 5 < w):
                continue
            img[r - 2:r + 3, u - 5:u + 5] = 255
        out.append(img)
    return out


class TestTimeSlice:
    def test_the_slice_is_rows_by_frames(self):
        img = ts.build_timeslice(_frames(7), rows=[100, 150, 200])
        assert img.shape == (3, 7)
        assert float(img[1].std()) < 1e-6      # a constant row over flat art

    def test_the_slice_shows_the_stroke_crossing_the_column_window(self):
        # the stroke drifts 3 px/frame through a 20 px window centred on 200,
        # so the band is bright while it is inside the window and flat after
        img = ts.build_timeslice(_slanted_frames(12, rows=(150,)), rows=[150],
                                 u0=190, u1=210)
        assert img.shape == (1, 12)
        band = img[0]
        top = float(band.max())
        assert top > 60.0, "the stroke is detectable in the slice"
        bright = np.nonzero(band >= 0.5 * top)[0]
        assert bright.size >= 3 and int(bright.max()) <= 5, \
            "the band is lit while the stroke is inside 190..210"
        assert float(band[-1]) < 0.25 * top, \
            "and is flat once the stroke has left the window"

    def test_the_mean_statistic_still_exists_for_an_explicit_choice(self):
        img = ts.build_timeslice(_slanted_frames(4), rows=[150], stat="mean")
        assert float(img[0].mean()) > 40.0

    def test_the_slice_respects_the_column_window(self):
        img = ts.build_timeslice(_frames(4, u_stroke=40), rows=[150],
                                 u0=100, u1=200)
        assert float(img.max()) < 60.0, \
            "the stroke is outside the window, so nothing stands out"

    def test_an_empty_or_flat_window_reports_no_contrast(self):
        flat = [np.full((20, 40), 90, dtype=np.uint8) for _ in range(3)]
        img = ts.build_timeslice(flat, rows=[10])
        assert float(np.abs(img).max()) < 5.0


class TestSnap:
    def test_the_bright_stroke_is_found(self):
        hit = ts.snap_to_stroke(_frames(1)[0], 150.0)
        assert hit["u"] == pytest.approx(200, abs=4)
        assert hit["v"] == 150.0 and hit["reason"] == ""

    def test_a_window_without_the_stroke_reports_rather_than_guesses(self):
        hit = ts.snap_to_stroke(_frames(1, u_stroke=40)[0], 150.0, u0=100,
                                u1=200)
        assert hit["u"] is not None
        assert hit["score"] is not None      # it answers, the caller judges

    def test_an_empty_window_is_reported(self):
        hit = ts.snap_to_stroke(_frames(1)[0], 150.0, u0=200, u1=200)
        assert hit["u"] is None and "empty" in hit["reason"]

    def test_avoid_inverts_the_search_for_a_dark_stroke(self):
        img = _frames(1)[0].copy()
        img[:, 195:205] = 0
        bright = ts.snap_to_stroke(img, 150.0)
        dark = ts.snap_to_stroke(img, 150.0, avoid=True)
        assert dark["score"] < bright["score"]


class TestProducedLabels:
    def test_labels_are_inferred_with_provenance_and_schema_valid(self):
        anns, rep = ts.timeslice_to_annotations(
            _frames(12), [[0, 180], [11, 140]], rows=[160, 180, 200],
            role="left", curve_id="c1", run="run_9", map_name="italy",
            episode="ep.npz", source_id="front_main")
        assert rep["annotated_frames"] == 12 and rep["runs"] == 1
        for ann in anns:
            assert cs.validate(ann) == []
            for curve in ann.curves:
                assert curve.role == "left"
                for seg in curve.segments:
                    assert seg.source == cs.SRC_INFERRED
                    assert seg.derived_from, "an inferred span must cite its clicks"
                    assert len(seg.points) == 12
            assert ann.episode == "ep.npz" and ann.source_id == "front_main"

    def test_a_break_splits_the_record_into_two_runs(self):
        # a break needs a control point on BOTH sides: the frames between
        # two clicks are not interpolated across it
        anns, rep = ts.timeslice_to_annotations(
            _frames(12), [[0, 180], [5, 160], [6, 158], [11, 140]],
            rows=[160, 180, 200], breaks=[6])
        assert rep["runs"] == 2
        spans = sorted((seg.frame_start, seg.frame_end)
                       for a in anns for c in a.curves for seg in c.segments)
        assert spans == [(0, 5), (6, 11)]
        for ann in anns:
            assert cs.validate(ann) == []

    def test_the_labels_round_trip_through_the_schema_writer(self, tmp_path):
        anns, _ = ts.timeslice_to_annotations(_frames(5), [[0, 180], [4, 150]],
                                              rows=[160])
        p = tmp_path / "labels.jsonl"
        assert cs.dump_jsonl(p, anns) == len(anns)
        back = cs.load_jsonl(p)
        assert any("inferred" in n for a in back for n in a.notes)


class TestSpotCheck:
    def test_error_statistics_and_unknowns(self):
        rep = ts.spot_check({0: 180.0, 1: 178.0, 2: 176.0},
                            {0: 180.0, 1: 175.0, 3: 170.0}, tol_px=3.0)
        assert rep["n"] == 2
        assert rep["err_p50_px"] == pytest.approx(1.5)
        assert rep["err_max_px"] == pytest.approx(3.0)
        assert rep["within_tol_frac"] == pytest.approx(1.0)
        assert rep["unknown"] == [2, 3], "one-sided frames are UNKNOWN"

    def test_a_miss_is_counted_as_a_miss(self):
        rep = ts.spot_check({0: 180.0}, {0: 160.0}, tol_px=3.0)
        assert rep["within_tol_frac"] == 0.0 and rep["err_max_px"] == 20.0

    def test_no_overlap_is_not_a_perfect_score(self):
        rep = ts.spot_check({0: 1.0}, {9: 1.0})
        assert rep["n"] == 0 and "reason" in rep


class TestCli:
    def test_an_episode_without_rgb_is_refused(self, tmp_path):
        p = tmp_path / "ep.npz"
        np.savez(p, x=np.zeros(3))
        clicks = tmp_path / "clicks.json"
        clicks.write_text("[[0, 180], [1, 170]]", encoding="utf-8")
        rc = ts.main(["--episode", str(p), "--rows", "160",
                      "--clicks", str(clicks)])
        assert rc == 2

    def test_the_cli_writes_labels_and_a_report(self, tmp_path):
        p = tmp_path / "ep.npz"
        np.savez(p, rgb=np.stack(_frames(6)))
        clicks = tmp_path / "clicks.json"
        clicks.write_text(json.dumps([[0, 180], [5, 150]]), encoding="utf-8")
        spot = tmp_path / "spot.json"
        spot.write_text(json.dumps([[0, 180], [5, 152]]), encoding="utf-8")
        out = tmp_path / "labels.jsonl"
        rep = tmp_path / "report.json"
        rc = ts.main(["--episode", str(p), "--rows", "160", "180",
                      "--clicks", str(clicks), "--spot-check", str(spot),
                      "--out", str(out), "--json", str(rep), "--role", "right"])
        assert rc == 0
        assert out.exists() and rep.exists()
        report = json.loads(rep.read_text(encoding="utf-8"))
        assert report["annotated_frames"] == 6
        assert report["spot_check"]["n"] == 2
        assert report["limits"], "the tool must publish its own limits"
        back = cs.load_jsonl(out)
        assert [c.role for a in back for c in a.curves] == ["right"]
