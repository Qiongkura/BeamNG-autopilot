"""P1-1: near-field drivable evidence — measurement and input modes.

The 0-4 m band in front of the car is where "no drivable path" comes from:
the front camera's nearest visible ground is ~3.5 m.  These tests pin the
measurement (coverage, UNKNOWN when nothing was observed), the injected
upper-bound band (which must be labelled as an injection), and the fact
that the default input mode still polls one camera only.
"""

from __future__ import annotations

import numpy as np
import pytest

from beamng_autopilot import fsd_stack as fs
from beamng_autopilot.occupancy import OccupancyGrid, nearfield_coverage


def _grid():
    g = OccupancyGrid(60, 60, 0.5)
    return g


def test_coverage_is_unknown_not_zero_when_nothing_was_observed():
    c = nearfield_coverage(_grid(), np.zeros(3), 0.0, ahead_m=4.0)
    assert c["observed_cells"] == 0
    assert c["drivable_cells"] == 0
    assert c["drivable_frac"] is None          # UNKNOWN, never 0
    assert c["observed_frac"] == 0.0           # the band exists, nothing seen
    assert c["band_cells"] > 0
    assert c["ahead_m"] == 4.0


def test_coverage_counts_the_band_ahead_of_the_car_only():
    g = _grid()
    g.observed[:] = 1
    # paint one 4 m x 4 m patch straddling the car's position
    rows = int(4.0 / g.res)
    mid = g.n_rows // 2
    # ahead = SMALLER row indices
    g.drivable[mid - rows:mid, mid - 4:mid + 4] = 1
    # and one patch behind the car, which must not be counted
    g.drivable[mid + 2:mid + 2 + rows, mid - 4:mid + 4] = 1
    c = nearfield_coverage(g, np.zeros(3), 0.0, ahead_m=4.0)
    # the whole 0-4 m strip is "observed" (every lateral cell), but only the
    # painted patch is drivable, and the patch BEHIND the car is not counted
    assert c["band_cells"] == (rows + 1) * g.n_cols
    assert c["observed_cells"] == c["band_cells"]
    assert c["observed_frac"] == pytest.approx(1.0)
    assert c["drivable_cells"] == 8 * rows
    assert c["drivable_frac"] == pytest.approx(8 * rows / ((rows + 1) * g.n_cols),
                                               abs=0.002)


def test_the_default_mode_polls_one_camera_only():
    assert fs.NEARFIELD_CAM in ("off", "0", "none", "")


def test_the_injected_band_is_labelled_as_an_injection(monkeypatch):
    """Handoff P1-1 allows an UPPER-BOUND injection; it must not hide."""
    monkeypatch.setattr(fs, "NEARFIELD_CAM", "band")
    g = _grid()
    meta = fs.nearfield_step(None, None, {}, None, g, np.zeros(3), 0.0, 1)
    assert meta is not None
    assert meta["nearfield_mode"] == "band"
    assert meta["nearfield_injected_band_m"] == fs.NEARFIELD_BAND_AHEAD_M
    mid = g.n_rows // 2
    assert (g.drivable[mid - 8:mid] > 0).any()
    assert (g.observed[mid - 8:mid] > 0).any()


def test_off_mode_does_nothing(monkeypatch):
    monkeypatch.setattr(fs, "NEARFIELD_CAM", "off")
    g = _grid()
    assert fs.nearfield_step(None, None, {}, None, g, np.zeros(3), 0.0, 1) \
        is None
    assert not (g.drivable > 0).any()


def test_a_missing_fisheye_degrades_to_a_recorded_skip(monkeypatch):
    monkeypatch.setattr(fs, "NEARFIELD_CAM", "fisheye")
    g = _grid()
    meta = fs.nearfield_step(None, None, {}, None, g, np.zeros(3), 0.0, 1)
    assert meta["nearfield_skipped"] == "no fisheye frame this tick"
    assert not (g.drivable > 0).any()


def test_the_fisheye_pass_projects_its_road_mask_with_its_own_model(
        monkeypatch):
    """The fisheye's road pixels must land in the grid via ITS camera.

    A pinhole model built from the fisheye mount is used (the Tech fisheye
    render is rectilinear, measured: the horizon and the guardrail stay
    straight in the frame), and the projection plane is the ROAD, not the
    ego origin - 0.17 m of height bias matters at 2 m.
    """
    monkeypatch.setattr(fs, "NEARFIELD_CAM", "fisheye")
    calls = {}

    class _Seg:
        def predict(self, rgb):
            road = np.zeros(rgb.shape[:2], dtype=bool)
            road[rgb.shape[0] // 2:, :] = True      # lower half = road
            return road, np.zeros_like(road)

    class _Head:
        def _get_segmenter(self):
            return _Seg()

    stack = type("S", (), {"hydra": type("H", (), {
        "_heads": {"semantic": _Head()}})()})()

    def fake_project(grid, mask, cam, pos, heading, **kw):
        calls["mask_px"] = int(np.count_nonzero(mask))
        calls["cam"] = cam
        calls["ground_z"] = kw.get("ground_z")
        calls["max_ahead"] = kw.get("max_ahead_m")

    monkeypatch.setattr(fs, "project_road_mask_to_grid", fake_project)
    g = _grid()
    rgb = np.zeros((20, 20, 3), dtype=np.uint8)
    meta = fs.nearfield_step(stack, None, {"front_fisheye": (rgb, "FISH_CAM")},
                             None, g, np.zeros(3), 0.0, 1)
    assert calls["cam"] == "FISH_CAM"
    assert calls["mask_px"] == 200
    assert calls["max_ahead"] == fs.NEARFIELD_MAX_AHEAD_M
    assert calls["ground_z"] is not None          # the road plane, not pos[2]
    assert meta["nearfield_road_px"] == 200
    assert "nearfield_ms" in meta


def test_projection_accepts_an_explicit_ground_plane():
    import inspect

    from beamng_autopilot.occupancy import project_road_mask_to_grid
    sig = inspect.signature(project_road_mask_to_grid)
    assert "ground_z" in sig.parameters
    assert sig.parameters["ground_z"].default is None


# --------------------------------------------------------------------------
# T05: naming, physical blind zone, resolution budget
# --------------------------------------------------------------------------
def test_fisheye_and_fuse_are_one_code_path_and_say_so(monkeypatch):
    """``fuse`` is an ALIAS, not a second algorithm.

    The main view's road is already in the grid when the near-field pass
    runs, so adding the fisheye IS the fusion.  The telemetry must not let
    a reader believe two methods were compared (plan T05).
    """
    import beamng_autopilot.fsd_stack as fs
    seen = {}

    def _fake_project(grid, mask, cam, pos, heading, **kw):
        seen["called"] = True
        seen["kw"] = kw

    monkeypatch.setattr(fs, "project_road_mask_to_grid", _fake_project)
    calls = []

    class _Head:
        def _get_segmenter(self):
            class _Seg:
                def predict(self, rgb):
                    import numpy as np
                    return np.ones(rgb.shape[:2], dtype=bool), None
            return _Seg()

    class _Stack:
        hydra = type("H", (), {"_heads": {"semantic": _Head()}})()

    import numpy as np
    from beamng_autopilot.occupancy import OccupancyGrid
    pos = np.array([0.0, 0.0, 0.17])
    for asked in ("fisheye", "fuse"):
        monkeypatch.setattr(fs, "NEARFIELD_CAM", asked)
        grid = OccupancyGrid(60, 60, 0.5, origin=(0.0, 0.0), heading=0.0)
        cam = object()
        meta = fs.nearfield_step(
            _Stack(), type("O", (), {"meta": {}})(),
            {"front_fisheye": (np.zeros((4, 4, 3), dtype=np.uint8), cam)},
            None, grid, pos, 0.0, tick_num=1)
        calls.append((asked, dict(meta)))
    assert calls[0][1]["nearfield_mode"] == "fisheye"
    assert "nearfield_alias" not in calls[0][1]
    assert calls[1][1]["nearfield_mode"] == "fisheye"
    assert calls[1][1]["nearfield_alias"] == "fuse"
    assert "one code path" in calls[1][1]["nearfield_alias_note"]


def test_the_fisheye_blind_zone_is_physical_not_a_training_gap():
    """No fine-tune can see ground the lens never images."""
    from beamng_autopilot import geometry as G
    from beamng_autopilot.vision.ring import camera_ring_models
    ring = camera_ring_models(320, 240)
    pos = np.array([0.0, 0.0, G.EGO_GROUND_GAP_M])
    near = G.nearest_ground_distance_m(ring["front_fisheye"], pos)
    assert near is not None and near > 2.0, near
    # and it is CLOSER than the main camera's, which is why it exists
    main = G.nearest_ground_distance_m(ring["front_main"], pos)
    assert main is not None and near < main
    # resolution: a 0.1 m marking is resolvable far less far than the
    # main view's, so the fisheye buys near field, not far field
    assert G.resolution_distance_m(ring["front_fisheye"].fx, 0.1,
                                   2.0) < G.resolution_distance_m(
        ring["front_main"].fx, 0.1, 2.0)
