"""Unit tests for the paved-boundary lateral reference.

``lane.pavement.paved_edge_lane_center`` answers "where should the car
sit when the paved road carries no marking at all": half a lane width to
the LEFT of the observed paved right edge, clamped inside the pavement,
with the two pavement edges published as the tick's hard boundaries
(AGENTS.md「驾驶约束」: marking >> pavement edge, and never drive onto the
soil shoulder).

The tests render a synthetic pavement into a real ``CameraModel`` (the
front ring camera) by forward-projecting ground points, so the mask and
the back-projection are a genuine round trip - no hand-written pixel
bookkeeping.
"""

from __future__ import annotations

import numpy as np
import pytest

from beamng_autopilot.lane.pavement import (
    PAVED_EDGE_MARGIN_M,
    PAVED_EDGE_TOTAL_MAX_M,
    PAVED_LANE_HALF_M,
    paved_edge_lane_center,
)
from beamng_autopilot.vision.ring import camera_ring_models

W, H = 536, 403
POS = (10.0, 20.0, 1.5)
HEADING = 0.0
GROUND_Z = 1.33                      # POS[2] - EGO_ORIGIN_GROUND_GAP_M
CAM = camera_ring_models(W, H)["front_main"]


def _world(lon, lat):
    """Ego-frame (forward, left) metres -> world xy on the ground plane."""
    ch, sh = np.cos(HEADING), np.sin(HEADING)
    x = POS[0] + lon * ch - lat * sh
    y = POS[1] + lon * sh + lat * ch
    return x, y


def _pavement_mask(spans, w=W, h=H):
    """Rasterise pavement rectangles ``(lon0, lon1, lat0, lat1)``."""
    mask = np.zeros((h, w), dtype=bool)
    for lon0, lon1, lat0, lat1 in spans:
        lon = np.arange(lon0, lon1 + 1e-9, 0.25)
        lat = np.arange(lat0, lat1 + 1e-9, 0.05)
        LON, LAT = np.meshgrid(lon, lat)
        x, y = _world(LON.ravel(), LAT.ravel())
        z = np.full(x.shape, GROUND_Z)
        u, v, ok = CAM.project(np.column_stack([x, y, z]), POS, HEADING)
        sel = ok & np.isfinite(u) & np.isfinite(v)
        ui = np.round(u[sel]).astype(int)
        vi = np.round(v[sel]).astype(int)
        inb = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        mask[vi[inb], ui[inb]] = True
    return mask


def _ref(spans, **kw):
    dbg: dict = {}
    mask = _pavement_mask(spans)
    ref = paved_edge_lane_center(mask, CAM, POS, HEADING,
                                ground_z=GROUND_Z, debug=dbg, **kw)
    return ref, dbg


def _lat_of(world_pt):
    """Lateral (left = +) of a world point relative to the ego."""
    return float(world_pt[1] - POS[1])           # heading 0 -> y is left


# The keep-right target is half a lane left of the right edge, biased
# PAVED_EDGE_MARGIN_M further in (away from the shoulder).
_TARGET_OFFSET = PAVED_LANE_HALF_M + PAVED_EDGE_MARGIN_M


def test_straight_road_keeps_half_a_lane_from_the_right_edge():
    ref, dbg = _ref([(0.0, 26.0, -3.0, 3.0)])
    assert dbg["mode"] == "ok", dbg
    assert ref is not None
    # The own-lane target sits one lane width in from a right edge at
    # -3.0 m (plus the outward-safety margin), and the read never errs
    # OUTWARD of that - the asymmetric bound is the safety property.
    nominal = -3.0 + _TARGET_OFFSET
    lat = _lat_of(ref.center[1])
    assert nominal - 0.10 <= lat <= nominal + 0.50, (lat, nominal)
    assert abs(ref.span_m - 6.0) < 0.45
    assert abs(_lat_of(ref.right[0]) - (-3.0)) < 0.30
    assert abs(_lat_of(ref.left[0]) - 3.0) < 0.30
    # The reference begins near the car so the planner can act on it.
    assert ref.first_lon_m <= 12.0


def test_the_target_never_leaves_the_pavement():
    """A 2.8 m track: half-lane-right would cross the left edge -> clamp."""
    ref, dbg = _ref([(0.0, 26.0, -1.4, 1.4)])
    assert ref is not None, dbg
    for p in ref.center[1:]:
        assert abs(_lat_of(p)) <= 1.4


def test_soil_beyond_the_pavement_is_not_used_as_road():
    """The pavement is 5 m wide; the soil shoulder beside it must not
    move the reference: the right edge stays the PAVED edge."""
    ref, dbg = _ref([(0.0, 26.0, -2.5, 2.5)])
    assert ref is not None, dbg
    assert abs(_lat_of(ref.right[0]) - (-2.5)) < 0.30
    assert _lat_of(ref.center[1]) >= (-2.5 + _TARGET_OFFSET) - 0.10


def test_pavement_either_side_of_the_car_is_not_enough():
    """Pavement to both sides with the car in a hole between them must
    NOT count as "the car is on the pavement": on the live east_coast
    frames 70/100 that exact shape was a car standing in the vegetation
    with the classifier's patches on either side, and it is
    indistinguishable from the car's own hood hiding the near field."""
    ref, dbg = _ref([(0.0, 26.0, -3.5, -0.9),
                     (0.0, 26.0, 0.9, 3.5)])
    assert ref is None
    assert dbg["mode"] == "ego_not_on_pavement", dbg


def test_an_outward_jump_is_not_followed():
    """The pavement suddenly reads 2.5 m wider from 12 m on: that is the
    mask spilling onto the shoulder, not a road that doubles in width in
    two bands.  The read may not follow it outward band by band."""
    ref, dbg = _ref([(0.0, 12.0, -2.5, 2.5),
                     (12.0, 26.0, -5.0, 2.5)])
    assert ref is not None, dbg
    lats = [_lat_of(p) for p in ref.right]
    assert min(lats) >= max(lats) - PAVED_EDGE_TOTAL_MAX_M - 0.20, lats
    # and the near part of the reference still sits on the true edge
    assert _lat_of(ref.right[0]) >= -2.5 - PAVED_EDGE_TOTAL_MAX_M - 0.20
    """A pavement wider than the view: the "edge" would be the image
    border, so the read must abstain instead of inventing a boundary."""
    ref, dbg = _ref([(0.0, 26.0, -9.0, 9.0)])
    assert ref is None
    assert dbg["mode"] in ("too_few_bands", "too_far_or_short"), dbg


def test_ego_on_soil_abstains():
    """Pavement beside the car (not under it): the car is off the paved
    surface - the answer is fail-closed, not a new lateral target."""
    ref, dbg = _ref([(0.0, 26.0, 2.0, 6.0)])
    assert ref is None
    assert dbg["mode"] == "ego_not_on_pavement", dbg


def test_a_plaza_sized_pavement_is_not_a_lane():
    """Once the read is plaza / junction-mouth sized the "right edge"
    stops being a road boundary and the candidate abstains (the target
    definition itself does not care how wide the road is - measured on
    the east_coast unmarked stretch, where 10.2 m reads were being
    rejected by the borrowed 10 m corridor gate)."""
    ref, dbg = _ref([(0.0, 26.0, -10.0, 10.0)])
    assert ref is None
    assert dbg["mode"] in ("too_few_bands", "too_far_or_short"), dbg


def test_a_read_that_wide_is_a_spilled_mask_not_a_road():
    """A 12 m read means the mask swallowed the shoulder: the real
    pavement on that stretch is ~6 m (hand labels).  The candidate must
    abstain - live 2026-09-18 a run that accepted these reads drove on
    the shoulder and hit the guardrail."""
    ref, dbg = _ref([(0.0, 26.0, -6.0, 6.0)])
    assert ref is None
    assert dbg["mode"] in ("too_few_bands", "too_far_or_short"), dbg


def test_no_pavement_returns_none():
    dbg: dict = {}
    assert paved_edge_lane_center(np.zeros((H, W), dtype=bool), CAM, POS,
                                  HEADING, ground_z=GROUND_Z,
                                  debug=dbg) is None
    assert dbg["mode"] == "empty_mask"


def test_corridor_tracking_does_not_jump_to_a_neighbouring_patch():
    """The road ends at 9 m; a driveway sits 5 m to the right beyond it.
    The read must not continue onto the driveway."""
    ref, dbg = _ref([(0.0, 9.0, -2.0, 2.0),
                     (12.0, 26.0, -7.0, -3.0)])
    assert dbg["usable_bands"] < 3, dbg
    assert ref is None


def test_a_short_patch_ahead_is_not_enough():
    """Only one usable band of pavement (~3 m) is not a steerable
    reference."""
    ref, dbg = _ref([(0.0, 7.0, -2.5, 2.5)])
    assert ref is None
    assert dbg["mode"] in ("too_few_bands", "too_far_or_short"), dbg


def test_target_sits_right_of_the_ego():
    """Right-hand traffic: the keep-right target is on the ego's right."""
    ref, dbg = _ref([(0.0, 30.0, -3.0, 3.0)])
    assert ref is not None, dbg
    lats = [_lat_of(p) for p in ref.center[1:]]
    assert lats[0] < 0.0


@pytest.mark.parametrize("span", [(2.8, 3.0), (4.0, 3.0)])
def test_narrow_road_target_stays_inside(span):
    """On a single-track paved road half-lane-right would hang over the
    left edge, so the target clamps to the middle instead."""
    half = 0.5 * span[0]
    ref, dbg = _ref([(0.0, 26.0, -half, half)])
    assert ref is not None, dbg
    for p in ref.center[1:]:
        assert abs(_lat_of(p)) <= half
    assert max(abs(_lat_of(p)) for p in ref.center[1:]) <= half - 0.5
