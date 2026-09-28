"""`m5_lane_geometry_measure.py` 的往返测试（T16 §4.4 车道约定实测）。

反投影必须是 `CameraModel.project` 的**逆**：把已知地面点投到像素、再反投影
回来，横向/前向都要能复原。测不到这一点，量出来的"车道宽度"就不可信。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beamng_autopilot import config  # noqa: E402
from beamng_autopilot.vision.ring import camera_ring_models  # noqa: E402


def _load():
    spec = importlib.util.spec_from_file_location(
        "m5_lane_geometry_measure",
        ROOT / "scripts" / "m5_lane_geometry_measure.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["m5_lane_geometry_measure"] = mod
    spec.loader.exec_module(mod)
    return mod


def _ground_point(pos, heading, *, forward_m, lateral_m):
    h = float(heading)
    fwd = np.array([np.cos(h), np.sin(h), 0.0])
    left = np.array([-np.sin(h), np.cos(h), 0.0])
    p = np.asarray(pos, dtype=float) + forward_m * fwd + lateral_m * left
    return np.array([p[0], p[1],
                     float(np.asarray(pos, dtype=float)[2])
                     - float(config.EGO_ORIGIN_GROUND_GAP_M)])


def test_back_project_round_trips_projected_ground_points():
    m = _load()
    cam = camera_ring_models(536, 403)["front_main"]
    pos = [613.9, 563.6, 175.48]
    heading = -2.55445
    for forward_m, lateral_m in ((6.0, 1.5), (10.0, -1.8), (20.0, 3.2)):
        pt = _ground_point(pos, heading, forward_m=forward_m, lateral_m=lateral_m)
        u, v, valid = cam.project([pt], pos, heading)
        assert bool(valid[0]), (forward_m, lateral_m)
        back = m.back_project_ground(u[0], v[0], cam, pos, heading,
                                     ground_z=pt[2])
        assert back is not None
        lat = m.lateral_of(back, pos, heading)
        assert abs(lat - lateral_m) < 1e-6, (lat, lateral_m)
        # 前向距离也要复原
        h = float(heading)
        fwd = np.array([np.cos(h), np.sin(h), 0.0])
        fwd_back = float((np.asarray(back)[:2]
                          - np.asarray(pos, dtype=float)[:2]) @ fwd[:2])
        assert abs(fwd_back - forward_m) < 1e-6, (fwd_back, forward_m)


def test_measure_frame_finds_lateral_offsets():
    m = _load()
    cam = camera_ring_models(536, 403)["front_main"]
    pos = [613.9, 563.6, 175.48]
    heading = -2.55445
    # 造一帧：把两条地面线（±1.6 m，6..20 m）投到像素上，写成 label
    lab = np.zeros((403, 536), np.uint8)
    for lat in (-1.6, 1.6):
        for fm in np.arange(6.0, 20.1, 1.0):
            pt = _ground_point(pos, heading, forward_m=float(fm),
                               lateral_m=lat)
            u, v, ok = cam.project([pt], pos, heading)
            if ok[0] and np.isfinite(u[0]) and np.isfinite(v[0]):
                lab[int(round(v[0])), int(round(u[0]))] = 2
    assert int((lab == 2).sum()) > 5
    vals = m.measure_frame(lab, cam, pos, heading, ground_z=175.48 - 0.17,
                           min_row_frac=0.55)
    assert vals, "近场线像素应能量出横向"
    a = np.abs(np.asarray(vals))
    assert abs(float(np.median(a)) - 1.6) < 0.15, np.median(a)


def test_summarize_reports_histogram_and_percentiles():
    m = _load()
    s = m.summarize([-1.6, -1.55, 1.6, 1.65, 0.1])
    assert s["n"] == 5 and "hist" in s and s["abs_p50"] is not None
    assert m.summarize([]) == {"n": 0}
