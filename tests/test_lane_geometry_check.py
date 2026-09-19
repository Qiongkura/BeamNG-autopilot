"""Lane geometry consistency checks (plan phase E4)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from beamng_autopilot.lane.geometry_check import (
    LANE_CORRIDOR_MAX_DEV_M,
    LANE_MAX_CURVATURE_PER_M,
    LANE_MAX_WIDTH_RATE_M_PER_M,
    LANE_MIN_VANISH_M,
    LaneGeometryLimits,
    check_lane_geometry,
    drivable_overlap,
    polyline_deviation_m,
    resample_polyline,
    signed_curvature,
    vanishing_distance_m,
    width_profile,
)


def _straight(y: float, x0: float = 0.0, x1: float = 40.0, n: int = 41):
    return np.column_stack([np.linspace(x0, x1, n), np.full(n, float(y))])


def _arc(radius: float, heading_span_deg: float, y_off: float = 0.0,
         n: int = 60, offset: float = 0.0):
    """An arc of the given radius, shifted laterally by ``offset``."""
    ang = np.linspace(0.0, math.radians(heading_span_deg), n)
    x = radius * np.sin(ang)
    y = radius * (1.0 - np.cos(ang))
    return np.column_stack([x, y + y_off + offset])


class _Grid:
    """Minimal BEV grid: everything drivable, or nothing."""

    def __init__(self, value: float = 1.0, n: int = 60, res: float = 0.5,
                 origin=(0.0, 0.0)):
        self.drivable = np.full((n, n), float(value), dtype=np.float32)
        self.origin = origin
        self.res = float(res)
        self.n = int(n)

    def world_to_cell(self, x, y):
        c = int((float(x) - self.origin[0]) / self.res) + self.n // 2
        r = int((float(y) - self.origin[1]) / self.res) + self.n // 2
        return r, c


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def test_resample_and_curvature_of_a_known_arc() -> None:
    r = 25.0
    pts, arc = resample_polyline(_arc(r, 60.0), 1.0)
    assert len(pts) == len(arc) >= 5
    k = signed_curvature(pts)
    assert float(np.median(np.abs(k))) == pytest.approx(1.0 / r, rel=0.15)


def test_straight_line_has_no_curvature() -> None:
    pts, _ = resample_polyline(_straight(0.0), 1.0)
    assert float(np.max(np.abs(signed_curvature(pts)))) < 1e-6


def test_width_profile_of_a_constant_lane() -> None:
    arc, w = width_profile(_straight(0.0), _straight(3.5))
    assert len(w) >= 5
    assert float(np.median(w)) == pytest.approx(3.5, abs=0.02)


def test_vanishing_distance_of_a_wedge() -> None:
    """Two boundaries that meet ahead give a distance; parallel gives None."""
    left = np.array([[0.5 * i, 0.0] for i in range(21)])
    right = np.array([[0.5 * i, 3.5 - 0.175 * i] for i in range(21)])
    d = vanishing_distance_m(left, right)
    # the pair closes at x = 10 m (3.5 m of lateral closing at 0.35 m/m)
    assert d is not None and d == pytest.approx(10.0, abs=3.0)
    assert vanishing_distance_m(_straight(0.0), _straight(3.5)) is None


def test_drivable_overlap_and_missing_evidence() -> None:
    assert drivable_overlap(_straight(0.0), None) is None
    assert drivable_overlap(_straight(0.0), _Grid(1.0)) == pytest.approx(1.0)
    assert drivable_overlap(_straight(0.0), _Grid(0.0)) == pytest.approx(0.0)


def test_polyline_deviation() -> None:
    assert polyline_deviation_m(_straight(0.0), None) is None
    assert polyline_deviation_m(_straight(0.0), _straight(0.0)) == \
        pytest.approx(0.0, abs=0.05)
    assert polyline_deviation_m(_straight(0.0), _straight(1.2)) == \
        pytest.approx(1.2, abs=0.05)


# ---------------------------------------------------------------------------
# the checks
# ---------------------------------------------------------------------------

def test_a_clean_pair_passes_every_check() -> None:
    rep = check_lane_geometry(center=_straight(1.75), left=_straight(0.0),
                              right=_straight(3.5), drivable=_Grid(1.0),
                              corridor=_straight(1.7), prev_ref=_straight(1.8))
    assert rep.ok, rep.reasons
    assert rep.metrics["width_m"] == pytest.approx(3.5, abs=0.05)
    assert rep.metrics["drivable_frac"] == pytest.approx(1.0)
    assert rep.metrics["corridor_dev_m"] < 0.2
    assert rep.metrics["ref_dev_m"] < 0.2


def test_width_band_is_enforced() -> None:
    # 7.0 m is outside the band (LANE_WIDTH_MAX_M = 6.5)
    rep = check_lane_geometry(center=_straight(3.5), left=_straight(0.0),
                              right=_straight(7.0))
    assert not rep.ok and "width_band" in rep.reasons


def test_sudden_width_change_is_rejected() -> None:
    """The plan's named case: a pairing whose width swings along the lane.

    Every single sample stays inside the width band - only the RATE gives
    the mis-pairing away.
    """
    x = np.linspace(0.0, 40.0, 41)
    width = np.where(x < 20.0, 3.0, 4.6)      # a 1.6 m step
    right = np.column_stack([x, width])
    rep = check_lane_geometry(center=_straight(2.0), left=_straight(0.0),
                              right=right)
    assert rep.metrics["width_min"] >= 2.2 and rep.metrics["width_max"] <= 6.5
    assert "width_rate" in rep.reasons and not rep.ok


def test_a_too_tight_boundary_is_rejected() -> None:
    tight = _arc(6.0, 60.0)                  # 6 m radius: not a road edge
    rep = check_lane_geometry(center=_arc(6.0, 60.0, offset=1.75),
                              left=tight)
    assert not rep.ok and "left_curve" in rep.reasons
    assert rep.metrics["left_curv"] > LANE_MAX_CURVATURE_PER_M


def test_a_window_that_pinches_shut_is_rejected() -> None:
    left = _straight(0.0)
    right = _straight(3.5)
    right[25:, 1] = left[25:, 1] + 0.05      # the pair closes to a point
    rep = check_lane_geometry(center=_straight(1.0), left=left, right=right)
    assert not rep.ok
    # the pinch is caught - by the width RATE (a step, not a vanishing
    # point) or by the vanishing distance, depending on the shape
    assert "vanish" in rep.reasons or "width_rate" in rep.reasons         or "width_band" in rep.reasons


def test_lane_centre_off_the_drivable_area_is_rejected() -> None:
    rep = check_lane_geometry(center=_straight(1.75), left=_straight(0.0),
                              right=_straight(3.5), drivable=_Grid(0.0))
    assert not rep.ok and "drivable" in rep.reasons


def test_lane_centre_far_from_the_lidar_corridor_is_rejected() -> None:
    rep = check_lane_geometry(center=_straight(1.75), left=_straight(0.0),
                              right=_straight(3.5),
                              corridor=_straight(1.75 + 3.0))
    assert not rep.ok and "corridor" in rep.reasons
    assert rep.metrics["corridor_dev_m"] > LANE_CORRIDOR_MAX_DEV_M


def test_a_reference_jump_is_rejected() -> None:
    rep = check_lane_geometry(center=_straight(1.75), left=_straight(0.0),
                              right=_straight(3.5), prev_ref=_straight(3.0))
    assert not rep.ok and "ref_jump" in rep.reasons


def test_missing_context_is_not_an_error() -> None:
    """Absent evidence must not fail a geometrically clean lane."""
    rep = check_lane_geometry(center=_straight(1.75), left=_straight(0.0),
                              right=_straight(3.5))
    assert rep.ok
    assert "drivable_frac" not in rep.metrics


def test_degenerate_inputs_are_safe() -> None:
    for kwargs in (dict(center=None), dict(center=_straight(0.0)[:1]),
                   dict(center=_straight(1.75), left=_straight(0.0)[:1]),
                   dict(center=_straight(1.75), left=None, right=None)):
        rep = check_lane_geometry(**kwargs)
        assert isinstance(rep.ok, bool)
        json.dumps(rep.digest())


def test_limits_are_configurable() -> None:
    strict = LaneGeometryLimits(width_min_m=3.0, width_max_m=4.0,
                                max_width_rate=0.01)
    rep = check_lane_geometry(center=_straight(1.75), left=_straight(0.0),
                              right=_straight(5.0), limits=strict)
    assert not rep.ok and "width_band" in rep.reasons
    rep2 = check_lane_geometry(
        center=_straight(1.75), left=_straight(0.0), right=_straight(5.0),
        limits=LaneGeometryLimits(width_min_m=1.0, width_max_m=8.0))
    assert "width_band" not in rep2.reasons


def test_digest_is_json_safe() -> None:
    rep = check_lane_geometry(center=_straight(1.75), left=_straight(0.0),
                              right=_straight(3.5))
    text = json.dumps(rep.digest())
    assert "width_m" in text and "nan" not in text.lower()


def test_width_rate_limit_is_the_documented_one() -> None:
    assert 0.0 < LANE_MAX_WIDTH_RATE_M_PER_M < 1.0
    assert LANE_MIN_VANISH_M > 0.0


# ---------------------------------------------------------------------------
# wiring: select_lane_reference withdraws a geometrically wrong sensor lane
# ---------------------------------------------------------------------------

LANE_HALF_M = 1.75
LANE_W_M = 3.5


def _route(n: int = 41) -> np.ndarray:
    xs = np.linspace(0.0, 40.0, n)
    return np.column_stack([xs, np.zeros_like(xs)])


def _line(y, xs=None):
    xs = np.linspace(0.0, 30.0, 31) if xs is None else xs
    return np.column_stack([xs, np.full_like(xs, float(y))])


def _sensor_lane(center_y: float = -LANE_HALF_M, left_y=None, right_y=None):
    """A real LaneFrame shaped like the vision pairing output.

    The boundaries sit half a lane either side of the centre (a lane is
    not twice its own width): left = centre + w/2, right = centre - w/2.
    """
    from beamng_autopilot.lane.tracking import LaneFrame
    left_y = center_y + 0.5 * LANE_W_M if left_y is None else left_y
    right_y = center_y - 0.5 * LANE_W_M if right_y is None else right_y
    return LaneFrame(center=_line(center_y), left=_line(left_y),
                     right=_line(right_y), width=LANE_W_M, confidence=0.8,
                     span_m=30.0, sources=("vision",), paired=True)


def _select(frame, monkeypatch, enabled: bool, **kwargs):
    from beamng_autopilot.lane import reference as ref_mod
    monkeypatch.setattr(ref_mod, "LANE_GEOM_ENABLED", enabled)
    return ref_mod.select_lane_reference(
        lane_frame=frame, pos=np.zeros(3), heading=0.0,
        route_ref=_route(), has_nav_route=True, lane_mode="map", **kwargs)


def test_wiring_keeps_a_clean_pair(monkeypatch) -> None:
    rep = _select(_sensor_lane(), monkeypatch, enabled=True)
    assert rep.src == "sensor"
    assert rep.center is not None
    assert rep.meta["lane_geom"]["ok"] == 1
    assert rep.meta["lane_geom"]["m"]["width_m"] == pytest.approx(
        LANE_W_M, abs=0.2)


def test_wiring_withdraws_a_width_swinging_pair(monkeypatch) -> None:
    """The plan's named case, through the real selection path."""
    xs = np.linspace(0.0, 30.0, 31)
    # 3.0 m of lane, then a 1.8 m step to 4.8 m: every sample inside the
    # width band, only the RATE gives the mis-pairing away
    swing = np.where(xs < 15.0, -LANE_HALF_M - 3.0, -LANE_HALF_M - 4.8)
    frame = _sensor_lane()
    frame.right = np.column_stack([xs, swing])
    rep = _select(frame, monkeypatch, enabled=True)
    assert rep.center is None, "a wrong-shaped pair must not steer the car"
    assert rep.src == "perception-unavailable"
    assert rep.meta["lane_geom"]["ok"] == 0
    assert "geom:" in rep.meta.get("lane_reject_reason", "")


def test_wiring_is_off_by_default(monkeypatch) -> None:
    xs = np.linspace(0.0, 30.0, 31)
    swing = np.where(xs < 15.0, -LANE_HALF_M - 3.0, -LANE_HALF_M - 4.8)
    frame = _sensor_lane()
    frame.right = np.column_stack([xs, swing])
    rep = _select(frame, monkeypatch, enabled=False)
    assert rep.src == "sensor" and rep.center is not None
    assert "lane_geom" not in rep.meta


def test_wiring_uses_the_corridor_and_previous_reference(monkeypatch) -> None:
    rep = _select(_sensor_lane(), monkeypatch, enabled=True,
                  corridor=_line(-LANE_HALF_M - 4.0))
    assert rep.center is None
    assert "corridor" in rep.meta.get("lane_reject_reason", "")
    rep2 = _select(_sensor_lane(), monkeypatch, enabled=True,
                   prev_ref=_line(-LANE_HALF_M - 3.0))
    assert rep2.center is None
    assert "ref_jump" in rep2.meta.get("lane_reject_reason", "")


def test_wiring_never_touches_a_map_lane(monkeypatch) -> None:
    """A map-derived reference is not a perception claim to check."""
    rep = _select(None, monkeypatch, enabled=True)
    assert rep.src == "map"
    assert rep.center is not None
    assert "lane_geom" not in rep.meta


# ---------------------------------------------------------------------------
# stack plumbing: the E4 context actually reaches the check
# ---------------------------------------------------------------------------

def test_stack_corridor_is_only_built_when_the_gate_is_on(monkeypatch) -> None:
    """The E4 context costs work, so it must not run with the switch off."""
    from beamng_autopilot.fsd_stack import FSDStack
    from beamng_autopilot.lane import reference as ref_mod
    st = FSDStack.__new__(FSDStack)
    hits = [(6.0, -1.0), (6.0, 1.0), (9.0, -0.5), (9.0, 0.5)]
    out = type("O", (), {"ray_hits": hits})()
    monkeypatch.setattr(ref_mod, "LANE_GEOM_ENABLED", False)
    assert st._lane_geom_corridor(out, np.zeros(3), 0.0) is None
    monkeypatch.setattr(ref_mod, "LANE_GEOM_ENABLED", True)
    got = st._lane_geom_corridor(out, np.zeros(3), 0.0)
    assert got is None or (np.asarray(got).ndim == 2
                           and np.asarray(got).shape[1] == 2)
    # no ray hits at all is "no corridor evidence", not a crash
    assert st._lane_geom_corridor(
        type("O", (), {"ray_hits": []})(), np.zeros(3), 0.0) is None


def test_stack_passes_the_previous_reference_into_the_gate() -> None:
    """The slew state must reach the E4 check, or the jump rule is dead.

    Pinned at the source level: the call site is inside a 600-line tick,
    and a regression there would silently disable "与上一时刻 reference
    的偏差" while every unit test still passed.
    """
    import inspect
    from beamng_autopilot.fsd_stack import FSDStack
    text = inspect.getsource(FSDStack.tick)
    assert 'prev_ref=getattr(self, "_lane_ref_prev", None)' in text
    assert "corridor=self._lane_geom_corridor(" in text
