"""Unit tests for the live lane-recognition overlay renderer."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from beamng_autopilot.vision.live_vis import render_lane_vis


class _Cam:
    """Minimal camera stub: every world point projects to a fixed pixel."""

    def project(self, world, pos, heading):
        n = len(np.asarray(world))
        return (np.full(n, 50.0), np.full(n, 50.0), np.ones(n, dtype=bool))


def _tick():
    # Real-proportion stub (the BEV panel is 240x240 and pastes top-right;
    # a 60x80 frame would be fully covered by it).
    h, w = 300, 400
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    road = np.zeros((h, w), dtype=bool)
    road[250:, :] = True
    line = np.zeros((h, w), dtype=bool)
    line[250:252, 10:200] = True
    sem = SimpleNamespace(
        masks={"road": road, "line": line},
        meta={"markings": [SimpleNamespace(
            world=np.array([[5.0, 1.5, 0.0], [10.0, 1.5, 0.0],
                            [15.0, 1.5, 0.0]]),
            kind="solid", color="yellow", confidence=0.8)]})
    drivable = np.zeros((8, 8), dtype=np.uint8)
    drivable[4:, 3:6] = 1
    return SimpleNamespace(
        frame=frame, cam=_Cam(), head_outputs={"semantic": sem},
        meta={"lane_src_sel": "sensor", "lane_paired": 1},
        drivable=drivable, bev=np.zeros((8, 8)),
        feature_map=None, best_path=None, lane_ref=None,
        snapshot=SimpleNamespace(pos=np.array([0.0, 0.0, 0.0]),
                                 heading=0.0))


def test_render_returns_bgr_image_with_hud():
    img = render_lane_vis(_tick(), np.zeros(3), 0.0, line_lat=1.2)
    assert img.ndim == 3 and img.shape[2] == 3
    assert img.shape[0] == 300 and img.shape[1] == 400
    # HUD band is dark with text drawn on it
    assert img[:34].std() > 0.0


def test_render_tints_masks_and_draws_markings():
    tick = _tick()
    img = render_lane_vis(tick, np.zeros(3), 0.0)
    # road rows picked up the green tint (BGR green channel dominant),
    # sampled OUTSIDE the top-right BEV panel region (cols >= 160 in a
    # 400-wide stub frame)
    assert img[280, 40, 1] > img[200, 40, 1]


def test_render_survives_missing_pieces():
    tick = _tick()
    tick.head_outputs = {}
    tick.drivable = None
    tick.bev = None
    tick.snapshot = None
    tick.cam = None
    img = render_lane_vis(tick, np.zeros(3), 0.0)
    assert img.ndim == 3
