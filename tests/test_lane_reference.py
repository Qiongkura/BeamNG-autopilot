"""Offline tests for the single owner of "which lane geometry may steer".

``lane/reference.py`` decides what enters the planner's Scene; the
consumer side (which Scene field may steer) is ``planning/lateral_ref.py``
and has its own tests.  Here we pin the producer's contract:

* trust order - paired sensor lane > trusted single painted boundary >
  map prior (legacy mode only) > BEV free-space centre (no nav route);
* strict perception never builds map lane geometry at all, even when a
  map override is handed in;
* strict mode fails closed (no centre) when perception cannot supply a
  lane, so the caller can only degrade;
* the BEV whole-road centre may never reach the planner's Scene.
"""

from __future__ import annotations

import numpy as np
import pytest

from beamng_autopilot.fsd_realism import (
    SRC_BEV_ROUTE,
    SRC_MAP,
    SRC_PAVED,
    SRC_SENSOR,
    SRC_UNAVAILABLE,
)
from beamng_autopilot.lane import (
    LaneFrame,
    LaneReference,
    bev_drivable_center,
    select_lane_reference,
)
from beamng_autopilot.occupancy import OccupancyGrid

LANE_HALF_M = 1.8
LANE_W_M = 3.6


def _route(length_m: float = 40.0, last: tuple[float, float] = (0.0, 0.0),
           n: int = 41) -> np.ndarray:
    """Straight nav route ahead of the ego, road centreline at ``y=0``."""
    xs = np.linspace(0.0, length_m, n)
    return np.column_stack([xs, np.full_like(xs, last[1])])


def _sensor_lane(*, side: str = "own", heading_axis: str = "fwd",
                 paired: bool = True, confidence: float = 0.8,
                 sources: tuple[str, ...] = ("vision",)) -> LaneFrame:
    """A LaneFrame shaped like the vision pairing output.

    ``side="own"`` puts the centre in the ego lane (right of the route
    centreline, minding left-handed +y), ``side="oncoming"`` on the left
    of it.  ``heading_axis="back"`` reverses it to test the bearing gate.
    """
    y_c = -LANE_HALF_M if side == "own" else LANE_HALF_M
    ys = LANE_HALF_M if side == "own" else -LANE_HALF_M
    span = 30.0
    if heading_axis == "fwd":
        xs = np.linspace(0.0, span, 31)
    else:
        xs = np.linspace(0.0, -span, 31)
    def line(y: float) -> np.ndarray:
        return np.column_stack([xs, np.full_like(xs, y)])
    return LaneFrame(
        center=line(y_c),
        left=line(y_c + ys),
        right=line(y_c - ys),
        width=LANE_W_M,
        confidence=confidence,
        span_m=span,
        sources=sources,
        paired=paired,
    )


def _corridor_grid() -> OccupancyGrid:
    """A 60x60 grid whose whole drivable band sits on the road centre."""
    grid = OccupancyGrid(n_rows=60, n_cols=60, res=0.5)
    grid.drivable[:, 28:32] = 1.0
    return grid


class _PavedRef:
    """The ``lane.pavement.PavedLane`` geometry the stack hands over.

    Only the fields ``select_lane_reference`` consumes: a keep-right
    centre polyline on the right half of the road, the two observed
    pavement edges and the measured span.
    """

    def __init__(self, half_m: float = 3.5) -> None:
        xs = np.linspace(0.0, 24.0, 9)
        self.center = np.column_stack([xs, np.full_like(xs, -1.9)])
        self.left = np.column_stack([xs, np.full_like(xs, half_m)])
        self.right = np.column_stack([xs, np.full_like(xs, -half_m)])
        self.span_m = 2.0 * half_m


# --------------------------------------------------------------------------
# trust order
# --------------------------------------------------------------------------
def test_paired_sensor_lane_leads_over_map_prior() -> None:
    ref = select_lane_reference(
        lane_frame=_sensor_lane(),
        pos=np.zeros(3), heading=0.0,
        route_ref=_route(), has_nav_route=True,
        lane_mode="map",
    )
    assert ref.src == SRC_SENSOR
    assert ref.center is not None
    assert float(np.median(ref.center[:, 1])) == pytest.approx(-LANE_HALF_M)
    assert ref.strict is False
    assert ref.scene_ref is not None


def test_legacy_map_prior_is_the_reference_without_a_sensor_lane() -> None:
    ref = select_lane_reference(
        lane_frame=None,
        pos=np.zeros(3), heading=0.0,
        route_ref=_route(), has_nav_route=True,
        lane_mode="map",
    )
    assert ref.src == SRC_MAP
    assert ref.map_lane is not None
    assert ref.center is not None
    # the map-prior own lane sits right of the road centreline
    assert float(np.median(ref.center[:, 1])) < 0.0
    assert ref.boundaries is True


# --------------------------------------------------------------------------
# gates: a sensor lane that is not the own lane must not lead
# --------------------------------------------------------------------------
def test_wrong_way_sensor_lane_falls_back_to_map_prior() -> None:
    ref = select_lane_reference(
        lane_frame=_sensor_lane(heading_axis="back"),
        pos=np.zeros(3), heading=0.0,
        route_ref=_route(), has_nav_route=True,
        lane_mode="map",
    )
    assert ref.src == SRC_MAP
    assert ref.rejected is True
    assert ref.meta["lane_reject_reason"] == "heading"


def test_oncoming_lane_is_rejected_by_the_side_gate() -> None:
    ref = select_lane_reference(
        lane_frame=_sensor_lane(side="oncoming"),
        pos=np.zeros(3), heading=0.0,
        route_ref=_route(), has_nav_route=True,
        lane_mode="map",
    )
    assert ref.src == SRC_MAP
    assert ref.rejected is True
    assert ref.meta["lane_reject_reason"] == "side"


# --------------------------------------------------------------------------
# strict perception: no map geometry, ever
# --------------------------------------------------------------------------
def test_strict_never_builds_map_geometry_even_with_an_override() -> None:
    override = (np.zeros((5, 2)), np.zeros((5, 2)), np.zeros((5, 2)))
    ref = select_lane_reference(
        lane_frame=None,
        pos=np.zeros(3), heading=0.0,
        route_ref=_route(), has_nav_route=True,
        map_lane_override=override,
        lane_mode="sensor", strict_sensor=True,
    )
    assert ref.src == SRC_UNAVAILABLE
    assert ref.center is None
    assert ref.map_lane is None
    assert ref.strict is True


def test_strict_paired_lane_leads_and_clears_the_map_prior() -> None:
    ref = select_lane_reference(
        lane_frame=_sensor_lane(),
        pos=np.zeros(3), heading=0.0,
        route_ref=_route(), has_nav_route=True,
        lane_mode="sensor", strict_sensor=True,
    )
    assert ref.src == SRC_SENSOR
    assert ref.center is not None
    assert ref.map_lane is None
    assert ref.boundaries is True
    assert ref.left is not None and ref.right is not None


def test_strict_low_confidence_single_edge_fails_closed() -> None:
    """An unpaired, low-confidence read is not a perception lane."""
    ref = select_lane_reference(
        lane_frame=_sensor_lane(paired=False, confidence=0.2),
        pos=np.zeros(3), heading=0.0,
        route_ref=_route(), has_nav_route=True,
        lane_mode="sensor", strict_sensor=True,
    )
    assert ref.src == SRC_UNAVAILABLE
    assert ref.center is None


def test_strict_trusted_single_painted_boundary_publishes_the_lane() -> None:
    """A confident single painted edge is a perception lane (width inferred).

    The missing side is mirrored at the lane-width contract, so the lane
    CENTRE may steer - but the mirrored edge is not a physical boundary
    and must never become the no-cross rule (``boundaries`` stays
    False); only a real two-sided detection or the map prior may arm it.
    """
    ref = select_lane_reference(
        lane_frame=_sensor_lane(paired=False, confidence=0.8),
        pos=np.zeros(3), heading=0.0,
        route_ref=_route(), has_nav_route=True,
        lane_mode="sensor", strict_sensor=True,
    )
    assert ref.src == SRC_SENSOR
    assert ref.center is not None
    assert ref.map_lane is None
    assert ref.width == pytest.approx(LANE_W_M)
    assert ref.boundaries is False


def test_strict_explicit_single_paint_at_medium_confidence_publishes():
    lane = _sensor_lane(paired=False, confidence=0.4)
    lane.right_kind = "dashed"
    ref = select_lane_reference(
        lane_frame=lane,
        pos=np.zeros(3), heading=0.0,
        route_ref=_route(), has_nav_route=True,
        lane_mode="sensor", strict_sensor=True,
    )
    assert ref.src == SRC_SENSOR
    assert ref.center is not None


def test_strict_rejected_lane_fails_closed_rather_than_single_edging() -> None:
    """The single-edge fallback must not resurrect a lane the gates dropped."""
    ref = select_lane_reference(
        lane_frame=_sensor_lane(heading_axis="back", confidence=0.9),
        pos=np.zeros(3), heading=0.0,
        route_ref=_route(), has_nav_route=True,
        lane_mode="sensor", strict_sensor=True,
    )
    assert ref.src == SRC_UNAVAILABLE
    assert ref.center is None


# --------------------------------------------------------------------------
# paved boundary: the no-marking case on a paved road
# --------------------------------------------------------------------------
def test_strict_no_marking_uses_the_paved_boundary() -> None:
    """No marking + observed pavement -> the pavement boundary leads, and
    its edges are the hard boundaries (AGENTS.md「驾驶约束」).

    OPT-IN: the default is still fail-closed (see the test below), since
    a pavement edge the model reads 0.25 m past the true pavement sends
    the car onto the shoulder.
    """
    ref = select_lane_reference(
        lane_frame=None,
        pos=np.zeros(3), heading=0.0,
        route_ref=_route(), has_nav_route=True,
        lane_mode="sensor", strict_sensor=True,
        paved_ref=_PavedRef(), paved_fallback=True,
    )
    assert ref.src == SRC_PAVED
    assert ref.center is not None
    assert ref.map_lane is None
    assert ref.boundaries is True
    assert ref.left is not None and ref.right is not None
    assert ref.width == pytest.approx(7.0)
    assert ref.meta["lane_src"] == "paved"


def test_strict_painted_marking_still_beats_the_paved_boundary() -> None:
    """标线 >> 路面边界: a usable single painted edge wins."""
    ref = select_lane_reference(
        lane_frame=_sensor_lane(paired=False, confidence=0.8),
        pos=np.zeros(3), heading=0.0,
        route_ref=_route(), has_nav_route=True,
        lane_mode="sensor", strict_sensor=True,
        paved_ref=_PavedRef(), paved_fallback=True,
    )
    assert ref.src == SRC_SENSOR


def test_strict_paved_boundary_is_off_by_default() -> None:
    """A paved candidate alone does NOT unlock strict motion: the live
    2026-09-18 run that enabled it rode the lane line and hit the
    guardrail, so the default stays fail-closed until the model's
    pavement edge is trustworthy."""
    ref = select_lane_reference(
        lane_frame=None,
        pos=np.zeros(3), heading=0.0,
        route_ref=_route(), has_nav_route=True,
        lane_mode="sensor", strict_sensor=True,
        paved_ref=_PavedRef(),
    )
    assert ref.src == SRC_UNAVAILABLE
    assert ref.center is None


def test_paved_boundary_never_replaces_the_map_prior_legacy_mode() -> None:
    """Legacy (non-strict) mode keeps its map-prior own lane: the paved
    candidate is a STRICT-mode perception fallback, not a new default."""
    ref = select_lane_reference(
        lane_frame=None,
        pos=np.zeros(3), heading=0.0,
        route_ref=_route(), has_nav_route=True,
        lane_mode="map", strict_sensor=False,
        paved_ref=_PavedRef(),
    )
    assert ref.src == SRC_MAP


# --------------------------------------------------------------------------
# BEV free-space centre: probes only, never the planner's lane
# --------------------------------------------------------------------------
def test_bev_centre_is_available_without_a_nav_route() -> None:
    grid = _corridor_grid()
    ref = select_lane_reference(
        lane_frame=None,
        pos=np.zeros(3), heading=0.0,
        route_ref=None, has_nav_route=False,
        grid=grid, lane_mode="map",
    )
    assert ref.center is not None
    # the whole-road centre is not a lane: it must never reach the Scene
    assert ref.scene_ref is None
    assert ref.map_lane is None


def test_strict_never_uses_the_bev_whole_road_centre() -> None:
    ref = select_lane_reference(
        lane_frame=None,
        pos=np.zeros(3), heading=0.0,
        route_ref=None, has_nav_route=False,
        grid=_corridor_grid(), lane_mode="sensor", strict_sensor=True,
    )
    assert ref.center is None
    assert ref.src == SRC_UNAVAILABLE


def test_bev_drivable_center_tracks_the_free_corridor() -> None:
    grid = _corridor_grid()
    ref = bev_drivable_center(grid, np.zeros(3), 0.0)
    assert ref is not None
    assert len(ref) >= 3
    # the corridor is centred on the ego's forward axis
    assert float(np.median(ref[:, 1])) == pytest.approx(0.0, abs=0.6)
    # and ordered near -> far from the ego
    dists = np.linalg.norm(ref - np.zeros(2), axis=1)
    assert dists[0] <= dists[-1]


def test_bev_drivable_center_needs_drivable_cells() -> None:
    assert bev_drivable_center(OccupancyGrid(60, 60, 0.5),
                               np.zeros(3), 0.0) is None


def test_lane_reference_dataclass_defaults_are_empty() -> None:
    ref = LaneReference()
    assert ref.center is None and ref.scene_ref is None
    assert ref.src == "" and ref.strict is False


# --------------------------------------------------------------------------
# labels: one decision, one source of truth
# --------------------------------------------------------------------------
def test_lane_src_label_is_a_pure_function_of_the_source() -> None:
    """The telemetry label must never disagree with the gating source."""
    labels = {
        SRC_SENSOR: "sensor",
        SRC_MAP: "map_lane",
        SRC_UNAVAILABLE: "perception-unavailable",
    }
    cases = [
        dict(lane_frame=_sensor_lane(), lane_mode="map"),
        dict(lane_frame=None, lane_mode="map"),
        dict(lane_frame=_sensor_lane(), lane_mode="sensor", strict_sensor=True),
        dict(lane_frame=None, lane_mode="sensor", strict_sensor=True),
    ]
    for case in cases:
        ref = select_lane_reference(pos=np.zeros(3), heading=0.0,
                                    route_ref=_route(), has_nav_route=True,
                                    **case)
        expected = labels.get(ref.src, SRC_BEV_ROUTE)
        assert ref.meta["lane_src"] == expected, (case, ref.src, ref.meta)


def test_downgraded_sensor_lane_is_labelled_as_map() -> None:
    """A sensor lane the consistency check drops must not still log sensor.

    This is the drift the single owner removed: the old label was rebuilt
    from "is there a lane frame", so a frame the map prior had taken over
    still reported ``lane_src=sensor``.
    """
    far = np.column_stack([np.linspace(0.0, 30.0, 31), np.full(31, -8.0)])
    ref = select_lane_reference(
        lane_frame=_sensor_lane(),
        pos=np.zeros(3), heading=0.0,
        route_ref=_route(), has_nav_route=True,
        map_lane_override=(far, far, far),
        lane_mode="sensor",
    )
    assert ref.src == SRC_MAP
    assert ref.meta["lane_src"] == "map_lane"


def test_bev_fallback_is_labelled_bev_route() -> None:
    ref = select_lane_reference(
        lane_frame=None, pos=np.zeros(3), heading=0.0,
        route_ref=None, has_nav_route=False,
        grid=_corridor_grid(), lane_mode="map",
    )
    assert ref.src == ""
    assert ref.meta["lane_src"] == SRC_BEV_ROUTE


# --------------------------------------------------------------------------
# strict perception SIDE gate: the road centre line is never the
# own lane.  Regression for live town run 2026-09-20 where --strict
# bypassed this gate and the car sat on the road centre line at
# mean=+1.18 m painted-line lateral (p50=+1.04 m).
# --------------------------------------------------------------------------
def _sensor_lane_at(y_center: float, *, paired: bool = True,
                    confidence: float = 0.8,
                    width: float | None = None) -> LaneFrame:
    """LaneFrame with the centre pinned at an arbitrary lateral y.

    The width defaults to the lane-width contract so the helper matches
    what select_lane_reference consumes from a real perception read; pass
    ``width`` for the measured over-wide reads (a lane + shoulder).
    """
    xs = np.linspace(0.0, 30.0, 31)
    w = LANE_W_M if width is None else float(width)
    return LaneFrame(
        center=np.column_stack([xs, np.full_like(xs, y_center)]),
        left=np.column_stack([xs, np.full_like(xs, y_center + w / 2)]),
        right=np.column_stack([xs, np.full_like(xs, y_center - w / 2)]),
        width=w,
        confidence=confidence,
        span_m=30.0,
        sources=("vision",),
        paired=paired,
    )


def test_strict_rejects_lane_sitting_on_the_route_centerline() -> None:
    """Strict perception must NOT trust a centre pinned at y=0.

    A perception lane whose centre coincides with the road centre line
    is the WHOLE-ROAD free corridor, not the ego lane - trusting it
    rides the divider end to end (live town run 2026-09-20, gate_on_3).
    """
    ref = select_lane_reference(
        lane_frame=_sensor_lane_at(0.0),
        pos=np.zeros(3), heading=0.0,
        route_ref=_route(), has_nav_route=True,
        lane_mode="sensor", strict_sensor=True,
    )
    assert ref.src == SRC_UNAVAILABLE
    assert ref.center is None
    assert ref.meta["lane_reject_reason"] == "side"


def test_strict_rejects_lane_inside_the_centreline_band() -> None:
    """Strict perception requires >= 0.2 m RIGHT of the route.

    Anything inside the [-0.2, +inf] m band is still effectively the
    road centre line: 0.1 m right of the route is straddling, not
    driving.  Only outside the band may the centre steer.
    """
    ref = select_lane_reference(
        lane_frame=_sensor_lane_at(-0.1),
        pos=np.zeros(3), heading=0.0,
        route_ref=_route(), has_nav_route=True,
        lane_mode="sensor", strict_sensor=True,
    )
    assert ref.src == SRC_UNAVAILABLE
    assert ref.center is None
    assert ref.meta["lane_reject_reason"] == "side"


def test_non_strict_sensor_lane_on_centreline_is_still_published() -> None:
    """The relaxed tolerance for perception-led modes is preserved.

    Non-strict ``sensor`` mode allows the corner-apex oncoming-side
    read, so a centre pinned at y=0 stays publishable there.  This is
    the only mode where the centreline-riding lane can survive, and
    the downstream planner is responsible for the harder no-cross
    rule.
    """
    ref = select_lane_reference(
        lane_frame=_sensor_lane_at(0.0),
        pos=np.zeros(3), heading=0.0,
        route_ref=_route(), has_nav_route=True,
        lane_mode="sensor", strict_sensor=False,
    )
    assert ref.src == SRC_SENSOR
    assert ref.center is not None


def test_a_refusal_publishes_the_offset_and_limit_that_caused_it() -> None:
    """A side-gate refusal must be auditable from telemetry alone.

    Without the measured offset and the threshold, "the gate said no"
    cannot be told apart from a pairing error - the two need opposite
    fixes.  The published numbers must be the ones the gate used.
    """
    ref = select_lane_reference(
        lane_frame=_sensor_lane_at(-0.1),
        pos=np.zeros(3), heading=0.0,
        route_ref=_route(), has_nav_route=True,
        lane_mode="sensor", strict_sensor=True,
    )
    assert ref.meta["lane_reject_reason"] == "side"
    assert ref.meta["lane_side_off_m"] == pytest.approx(-0.1, abs=1e-6)
    assert ref.meta["lane_side_limit_m"] == pytest.approx(-0.2)
    assert ref.meta["pair_paired"] == 1
    assert ref.meta["pair_width_m"] == pytest.approx(LANE_W_M, abs=1e-6)
    assert ref.meta["pair_span_m"] == pytest.approx(30.0)
    assert ref.meta["pair_sources"] == ["vision"]


def test_an_accepted_lane_publishes_the_same_measurements() -> None:
    ref = select_lane_reference(
        lane_frame=_sensor_lane_at(-1.75),
        pos=np.zeros(3), heading=0.0,
        route_ref=_route(), has_nav_route=True,
        lane_mode="sensor", strict_sensor=True,
    )
    assert ref.src == SRC_SENSOR
    assert ref.meta["lane_side_off_m"] == pytest.approx(-1.75, abs=1e-6)
    assert ref.meta["lane_side_limit_m"] == pytest.approx(-0.2)


def test_unmeasurable_offset_is_reported_as_unknown_not_zero() -> None:
    """A lane with no measurable overlap must not publish a 0.0 offset.

    0.0 is the most dangerous value here (it reads as "on the centre
    line"), so it must not be manufactured when nothing was measured.
    """
    ref = select_lane_reference(
        lane_frame=LaneFrame(
            center=np.array([[500.0, 500.0], [501.0, 500.0], [502.0, 500.0]]),
            left=None, right=None, width=0.0, confidence=0.9,
            span_m=3.0, sources=("vision",), paired=False),
        pos=np.zeros(3), heading=0.0,
        route_ref=_route(), has_nav_route=True,
        lane_mode="sensor", strict_sensor=True,
    )
    assert ref.meta["lane_side_off_m"] is None
    assert ref.meta["lane_side_limit_m"] == pytest.approx(-0.2)


class _Marking:
    """Minimal LaneMarking stand-in for the divider fallback."""

    def __init__(self, world, kind="solid", color="white", confidence=0.8):
        self.world = np.asarray(world, dtype=float)
        self.kind, self.color, self.confidence = kind, color, confidence


def _divider_marks(y: float, *, kind: str = "solid", length_m: float = 20.0,
                   conf: float = 0.8, n: int = 41):
    xs = np.linspace(0.0, length_m, n)
    return [_Marking(np.column_stack([xs, np.full_like(xs, y)]),
                     kind=kind, confidence=conf)]


def _grid_full(n: int = 60, res: float = 0.5, *, drivable_y=(-6.0, 6.0),
               observed: bool = True):
    """Grid with a drivable/observed band.

    ``grid.world_to_cell`` maps car-left (+y) to a SMALLER column, so a
    cell's car-frame lateral is ``extent - (col + 0.5) * res``.  The band
    ``[y_lo, y_hi]`` (left-positive) therefore spans the columns below.
    """
    grid = OccupancyGrid(n, n, float(res))
    if drivable_y is not None:
        extent = n * res / 2.0

        def col(y: float) -> int:
            return int(round((extent - float(y)) / res - 0.5))
        lo, hi = col(max(drivable_y)), col(min(drivable_y))
        grid.drivable[:, lo:hi + 1] = 1
    if observed:
        grid.observed[:] = 1
    return grid


def test_divider_beside_the_car_yields_the_own_lane_on_its_right() -> None:
    """Stuck ON the centreline the whole-road reads are refused; the divider
    is not.  A long solid marking lying on the route (car at y=0, paint at
    y=-0.37) puts the own lane half a lane to its right."""
    grid = _grid_full()
    ref = select_lane_reference(
        lane_frame=None,
        pos=np.zeros(3), heading=0.0,
        route_ref=_route(), has_nav_route=True,
        grid=grid, lane_mode="sensor", strict_sensor=True,
        markings=_divider_marks(-0.37),
    )
    assert ref.src == SRC_SENSOR
    assert ref.center is not None
    assert float(np.median(ref.center[:, 1])) == pytest.approx(
        -0.37 - LANE_W_M / 2, abs=0.1)
    assert ref.meta["lane_from"] == "divider_right_shift"
    assert ref.boundaries is True          # the paint is a hard no-cross edge
    assert ref.left is not None and len(ref.left) >= 2


def test_a_marking_off_the_route_is_not_a_divider() -> None:
    """A lane edge 2.5 m off the road centre is not the divider: shifting
    the lane half a width off it would invent a lane on the shoulder."""
    ref = select_lane_reference(
        lane_frame=None,
        pos=np.zeros(3), heading=0.0,
        route_ref=_route(), has_nav_route=True,
        grid=_grid_full(), lane_mode="sensor", strict_sensor=True,
        markings=_divider_marks(-2.5),
    )
    assert ref.src == SRC_UNAVAILABLE
    assert ref.center is None
    assert ref.meta["lane_divider"]["reason"] == "no_divider_line"


def test_short_paint_fragment_is_not_a_divider() -> None:
    ref = select_lane_reference(
        lane_frame=None,
        pos=np.zeros(3), heading=0.0,
        route_ref=_route(), has_nav_route=True,
        grid=_grid_full(), lane_mode="sensor", strict_sensor=True,
        markings=_divider_marks(-0.3, length_m=2.0, n=5),
    )
    assert ref.src == SRC_UNAVAILABLE
    assert ref.meta["lane_divider"]["reason"] == "no_divider_line"


def test_divider_lane_must_land_on_observed_pavement() -> None:
    """The shift may not put the car where the sensors never saw road."""
    grid = _grid_full(drivable_y=(0.5, 6.0))     # pavement starts LEFT of 0.5
    ref = select_lane_reference(
        lane_frame=None,
        pos=np.zeros(3), heading=0.0,
        route_ref=_route(), has_nav_route=True,
        grid=grid, lane_mode="sensor", strict_sensor=True,
        markings=_divider_marks(-0.37),
    )
    assert ref.src == SRC_UNAVAILABLE
    assert ref.center is None
    assert "off_drivable" in ref.meta["lane_divider"]["reason"]


def test_divider_lane_needs_the_route_as_its_side_veto() -> None:
    """No route -> no way to tell which side the own lane is on: refuse."""
    ref = select_lane_reference(
        lane_frame=None,
        pos=np.zeros(3), heading=0.0,
        route_ref=None, has_nav_route=False,
        grid=_grid_full(), lane_mode="sensor", strict_sensor=True,
        markings=_divider_marks(-0.37),
    )
    assert ref.src == SRC_UNAVAILABLE
    assert ref.center is None


def test_an_accepted_pair_off_the_observed_pavement_is_withdrawn() -> None:
    """A 5.24 m "lane" 4.2 m right of the road centre sits past the right
    edge of a ~7 m road.  The side gate passes it (it IS on the right
    side); only the pavement check catches it."""
    grid = _grid_full(drivable_y=(-3.2, 3.8))
    frame = _sensor_lane_at(-4.2, width=5.24)
    ref = select_lane_reference(
        lane_frame=frame,
        pos=np.zeros(3), heading=0.0,
        route_ref=_route(), has_nav_route=True,
        grid=grid, lane_mode="sensor", strict_sensor=True,
    )
    assert ref.src == SRC_UNAVAILABLE
    assert ref.center is None
    assert ref.meta["lane_drivable"]["reason"].startswith("centre off")


def test_an_accepted_pair_on_the_observed_pavement_survives() -> None:
    grid = _grid_full(drivable_y=(-3.2, 3.8))
    frame = _sensor_lane_at(-LANE_HALF_M)
    ref = select_lane_reference(
        lane_frame=frame,
        pos=np.zeros(3), heading=0.0,
        route_ref=_route(), has_nav_route=True,
        grid=grid, lane_mode="sensor", strict_sensor=True,
    )
    assert ref.src == SRC_SENSOR
    assert ref.meta["lane_drivable"]["frac"] >= 0.6


def test_no_drivable_evidence_abstains_and_says_so() -> None:
    """No grid must not silently read as "on pavement" nor as "off"."""
    frame = _sensor_lane_at(-LANE_HALF_M)
    ref = select_lane_reference(
        lane_frame=frame,
        pos=np.zeros(3), heading=0.0,
        route_ref=_route(), has_nav_route=True,
        grid=None, lane_mode="sensor", strict_sensor=True,
    )
    assert ref.src == SRC_SENSOR
    assert ref.meta["lane_drivable"]["frac"] is None
    assert ref.meta["lane_drivable"]["reason"] == "not enough observed samples"


def test_an_accepted_pair_still_wins_over_the_divider_fallback() -> None:
    """The fallback only fills a gap - it never overrides an accepted lane."""
    frame = _sensor_lane_at(-LANE_HALF_M)
    ref = select_lane_reference(
        lane_frame=frame,
        pos=np.zeros(3), heading=0.0,
        route_ref=_route(), has_nav_route=True,
        grid=_grid_full(), lane_mode="sensor", strict_sensor=True,
        markings=_divider_marks(-0.37),
    )
    assert ref.src == SRC_SENSOR
    assert "lane_from" not in ref.meta
    assert float(np.median(ref.center[:, 1])) == pytest.approx(
        -LANE_HALF_M, abs=1e-6)


def test_divider_needs_a_real_paint_kind_not_a_thin_skeleton() -> None:
    ref = select_lane_reference(
        lane_frame=None,
        pos=np.zeros(3), heading=0.0,
        route_ref=_route(), has_nav_route=True,
        grid=_grid_full(), lane_mode="sensor", strict_sensor=True,
        markings=_divider_marks(-0.37, kind="thin"),
    )
    assert ref.src == SRC_UNAVAILABLE


def test_map_mode_keeps_its_legacy_minus_0p4_tolerance() -> None:
    """The legacy map-mode side gate (left_max_m = -0.4) is unchanged."""
    # y = -0.3 is 0.3 m right of the route - inside the legacy band,
    # published.
    ref_ok = select_lane_reference(
        lane_frame=_sensor_lane_at(-0.3),
        pos=np.zeros(3), heading=0.0,
        route_ref=_route(), has_nav_route=True,
        lane_mode="map",
    )
    assert ref_ok.src in (SRC_SENSOR, SRC_MAP)
    assert ref_ok.center is not None
    # y = +0.3 is left of the route - rejected.
    ref_no = select_lane_reference(
        lane_frame=_sensor_lane_at(+0.3),
        pos=np.zeros(3), heading=0.0,
        route_ref=_route(), has_nav_route=True,
        lane_mode="map",
    )
    assert ref_no.rejected is True
    assert ref_no.meta["lane_reject_reason"] == "side"


# --------------------------------------------------------------------------
# the three gates, as one testable verdict (2026-10-20 extraction)
# --------------------------------------------------------------------------
def _verdict(**kwargs):
    from beamng_autopilot.lane.reference import _lane_gate_verdict
    base = dict(lane_frame=_sensor_lane(), lane_ref=_sensor_lane().center,
                route_ref=_route(), pos=np.zeros(3), heading=0.0,
                lane_mode="map", strict_lane=False)
    base.update(kwargs)
    return _lane_gate_verdict(**base)


def test_gate_verdict_accepts_the_own_lane_and_reports_the_numbers():
    v = _verdict()
    assert v["rejected"] is False and v["reason"] is None
    meta = v["meta"]
    assert meta["lane_side_limit_m"] == -0.4          # map 模式的来袭侧容差
    assert meta["pair_paired"] == 1
    assert meta["pair_width_m"] == 3.6
    assert isinstance(meta["lane_side_off_m"], float)


def test_gate_verdict_rejects_a_wrong_way_lane_on_heading():
    v = _verdict(lane_frame=_sensor_lane(heading_axis="back"),
                 lane_ref=_sensor_lane(heading_axis="back").center)
    assert v["rejected"] is True and v["reason"] == "heading"


def test_gate_verdict_rejects_the_oncoming_lane_on_side():
    oncoming = _sensor_lane(side="oncoming")
    v = _verdict(lane_frame=oncoming, lane_ref=oncoming.center,
                 strict_lane=True)
    assert v["rejected"] is True and v["reason"] == "side"
    assert v["meta"]["lane_side_limit_m"] == -0.2     # strict 收紧


def test_side_limit_follows_the_lane_mode():
    assert _verdict(lane_mode="sensor")["meta"]["lane_side_limit_m"] == 0.5
    assert _verdict(lane_mode="auto")["meta"]["lane_side_limit_m"] == 0.5
    assert _verdict(strict_lane=True)["meta"]["lane_side_limit_m"] == -0.2


def test_corner_gate_only_applies_outside_sensor_mode(monkeypatch):
    """转角门是"地图模式"的保守门；感知主导模式要让车道自己过弯。"""
    from beamng_autopilot.planning import arbiter

    monkeypatch.setattr(arbiter, "lane_route_turn_ok",
                        lambda *a, **k: False, raising=True)
    assert _verdict(lane_mode="map")["reason"] == "corner"
    assert _verdict(lane_mode="sensor")["rejected"] is False


def test_map_consistency_needs_enough_near_field_points():
    """近场点不足 3 个时**不下结论**（None），调用方维持原状、不误降级。"""
    from beamng_autopilot.lane.reference import _sensor_lane_within_map

    short = np.array([[0.2, -1.8], [0.4, -1.8]])
    assert _sensor_lane_within_map(short, _route(), np.zeros(3),
                                   max_center_m=1.5) is None


def test_map_consistency_compares_the_near_field_median_gap():
    from beamng_autopilot.lane.reference import _sensor_lane_within_map

    center = _route()                       # 路线中心在 y=0
    close = np.column_stack([np.linspace(1.0, 8.0, 8),
                             np.full(8, -0.3)])
    far = np.column_stack([np.linspace(1.0, 8.0, 8),
                           np.full(8, -3.0)])
    assert _sensor_lane_within_map(close, center, np.zeros(3),
                                   max_center_m=1.5) is True
    assert _sensor_lane_within_map(far, center, np.zeros(3),
                                   max_center_m=1.5) is False
    # sensor 模式的 2.5 m 容忍：3 m 仍算不一致，但 2 m 就算一致
    assert _sensor_lane_within_map(far, center, np.zeros(3),
                                   max_center_m=2.5) is False
    near = np.column_stack([np.linspace(1.0, 8.0, 8), np.full(8, -2.0)])
    assert _sensor_lane_within_map(near, center, np.zeros(3),
                                   max_center_m=2.5) is True


def test_far_sensor_lane_is_downgraded_to_the_map_reference():
    """端到端：偏离地图本车道 3 m 的配对车道在 sensor 模式下降级为 map。"""
    lane = _sensor_lane()
    lane.center = lane.center.copy()
    lane.center[:, 1] += -3.0
    ref = select_lane_reference(
        lane_frame=lane, pos=np.zeros(3), heading=0.0,
        route_ref=_route(), has_nav_route=True,
        lane_mode="sensor", strict_sensor=False,
    )
    assert ref.src == SRC_MAP, "偏离 3 m 超过 sensor 模式 2.5 m 容忍，必须降级为地图本车道"


def test_map_edge_width_uses_nearest_points_not_index_pairing():
    """两条边线的弧长参数化可以不同；逐元素配对会在弯道上读出偏大宽度。"""
    from beamng_autopilot.lane.reference import _lane_width_from_map_edges

    xs = np.linspace(0.0, 40.0, 41)
    left = np.column_stack([xs, np.full_like(xs, 1.75)])
    right = np.column_stack([xs, np.full_like(xs, -1.75)])
    assert _lane_width_from_map_edges(left, right, default_m=3.0) == 3.5

    # 右侧边线用两倍密度 + 半格错位：最近点距离仍是 3.5，逐元素会随时间偏移
    xs2 = np.linspace(0.0, 40.0, 81)
    right2 = np.column_stack([xs2, np.full_like(xs2, -1.75)])
    got = _lane_width_from_map_edges(left, right2, default_m=3.0)
    assert abs(got - 3.5) < 0.05, got


def test_map_edge_width_falls_back_to_the_contract_width():
    from beamng_autopilot.lane.reference import (
        _lane_width_from_map_edges, LANE_WIDTH_DEFAULT_M)

    assert _lane_width_from_map_edges([], [], default_m=3.2) == 3.2
    assert _lane_width_from_map_edges([], [], default_m=0.0) == \
        LANE_WIDTH_DEFAULT_M


def test_map_edge_width_warns_and_falls_back_on_a_bad_read():
    warnings = []

    def warn(key, msg):
        warnings.append((key, msg))

    from beamng_autopilot.lane.reference import _lane_width_from_map_edges

    got = _lane_width_from_map_edges(None, None, default_m=3.1, warn=warn)
    assert got == 3.1
    assert warnings and warnings[0][0] == "lane_width", warnings


def test_trusted_single_paint_needs_vision_and_a_typed_marking():
    from beamng_autopilot.lane.reference import _trusted_single_paint

    frame = _sensor_lane(paired=False, confidence=0.8)
    frame.right_kind = "solid"
    ref = frame.center
    assert _trusted_single_paint(frame, ref, sensor_paired=False) is True

    # 明确画线类型门槛 0.35：0.36 过、0.34 不过
    frame.confidence = 0.36
    assert _trusted_single_paint(frame, ref, sensor_paired=False) is True
    frame.confidence = 0.34
    assert _trusted_single_paint(frame, ref, sensor_paired=False) is False

    # 未知类型要 0.50
    frame.right_kind = "unknown"
    frame.confidence = 0.45
    assert _trusted_single_paint(frame, ref, sensor_paired=False) is False
    frame.confidence = 0.55
    assert _trusted_single_paint(frame, ref, sensor_paired=False) is True

    # 已配对的车道走自己的路径；LiDAR 单边推不出车道；没有中心折线
    paired = _sensor_lane(paired=True, confidence=0.9)
    assert _trusted_single_paint(paired, paired.center,
                                 sensor_paired=True) is False
    lidar = _sensor_lane(paired=False, confidence=0.9,
                         sources=("lidar",))
    lidar.right_kind = "solid"
    assert _trusted_single_paint(lidar, lidar.center,
                                 sensor_paired=False) is False
    assert _trusted_single_paint(frame, None, sensor_paired=False) is False
    assert _trusted_single_paint(None, ref, sensor_paired=False) is False


def test_reference_meta_assembles_labels_and_refusals():
    from beamng_autopilot.lane.reference import _reference_meta

    gate = {"lane_side_off_m": -1.2}
    meta = _reference_meta(gate, divider_dbg=None, divider=None,
                           rejected=False, reject_reason=None,
                           src_published=True, lane_src_sel=SRC_SENSOR,
                           strict_lane=True)
    assert meta["lane_src_sel"] == SRC_SENSOR
    assert meta["lane_src"] != SRC_BEV_ROUTE, "已发布来源不能落回 bev/route 标签"
    assert meta["lane_strict"] == 1
    assert meta["lane_side_off_m"] == -1.2
    assert gate == {"lane_side_off_m": -1.2}, "入参不得被改写"

    refused = _reference_meta(gate, divider_dbg=None, divider=None,
                              rejected=True, reject_reason="side",
                              src_published=True, lane_src_sel=SRC_MAP,
                              strict_lane=False)
    assert refused["lane_reject_reason"] == "side"
    assert "lane_strict" not in refused

    divider = _reference_meta(gate, divider_dbg={"reason": ""},
                              divider=("center", "left"), rejected=False,
                              reject_reason=None, src_published=True,
                              lane_src_sel=SRC_SENSOR, strict_lane=False)
    assert divider["lane_divider"] == {"reason": ""}
    assert divider["lane_from"] == "divider_right_shift"

    unpublished = _reference_meta({}, divider_dbg=None, divider=None,
                                  rejected=False, reject_reason=None,
                                  src_published=False, lane_src_sel="",
                                  strict_lane=False)
    assert "lane_src_sel" not in unpublished
    assert unpublished["lane_src"] == SRC_BEV_ROUTE


def test_bev_fallback_is_off_in_strict_and_needs_enough_points():
    from beamng_autopilot.lane.reference import _bev_whole_road_center

    grid = _corridor_grid()
    ref = _bev_whole_road_center(grid, np.zeros(3), 0.0, strict_lane=False)
    assert ref is not None and len(ref) >= 4
    assert _bev_whole_road_center(grid, np.zeros(3), 0.0,
                                  strict_lane=True) is None
    assert _bev_whole_road_center(None, np.zeros(3), 0.0,
                                  strict_lane=False) is None
    empty = OccupancyGrid(n_rows=60, n_cols=60, res=0.5)   # 无可行驶 cell
    assert _bev_whole_road_center(empty, np.zeros(3), 0.0,
                                  strict_lane=False) is None


def test_paved_authority_needs_the_flag_and_a_real_centre():
    """铺装边界只在"无标线 + 有铺装 + 中心够长"时接手；否则让位给 fail-closed。"""
    from beamng_autopilot.lane.reference import _paved_boundary_usable

    ref = _PavedRef()
    assert _paved_boundary_usable(True, ref) is True
    assert _paved_boundary_usable(False, ref) is False, "默认关，不得自行启用"

    class NoCentre:
        center = None

    class ShortCentre:
        center = np.zeros((2, 2))

    assert _paved_boundary_usable(True, None) is False
    assert _paved_boundary_usable(True, NoCentre()) is False
    assert _paved_boundary_usable(True, ShortCentre()) is False


def test_paved_geometry_is_2d_and_uses_the_measured_span():
    from beamng_autopilot.lane.reference import _paved_geometry

    center, left, right, width = _paved_geometry(_PavedRef(half_m=3.5))
    assert center.shape[1] == 2 and left.shape[1] == 2 and right.shape[1] == 2
    assert abs(width - 7.0) < 1e-9
    assert left[:, 1].max() > 0 > right[:, 1].min(), "左正右负（+y 为左）"

    class NoSpan:
        center = np.zeros((5, 3))       # 带 z 的折线
        left = np.zeros((5, 3))
        right = np.zeros((5, 3))

    c2, _, _, w2 = _paved_geometry(NoSpan())
    assert c2.shape == (5, 2), "只取前两列"
    assert w2 == 0.0


def test_divider_lane_finds_the_paint_and_reports_debug():
    """分隔线兜底：找线 + 右移半车道，并把诊断留在 debug 里。"""
    from beamng_autopilot.lane.reference import _divider_lane

    dbg = {}
    divider, dbg = _divider_lane(_divider_marks(-0.37), np.zeros(3), 0.0,
                                 _route(), _grid_full(), debug=dbg)
    assert divider is not None, dbg
    center, left = divider[0], divider[1]
    assert float(np.median(center[:, 1])) == pytest.approx(
        -0.37 - LANE_W_M / 2, abs=0.1)
    assert left is not None

    # 2.5 m 外的那条不是分隔线（会凭空造出一条肩上车道）
    off, _ = _divider_lane(_divider_marks(-2.5), np.zeros(3), 0.0,
                           _route(), _grid_full(), debug={})
    assert off is None


def test_divider_lane_never_raises_and_keeps_the_reason(monkeypatch):
    from beamng_autopilot.lane import reference as ref_mod

    # 没有标记 -> 找不到分隔线，但**不算失败**：debug 原样保留
    divider, dbg = ref_mod._divider_lane(None, np.zeros(3), 0.0, _route(),
                                        None, debug={"stale": 1})
    assert divider is None
    assert dbg == {"stale": 1}

    # 真的抛异常 -> 不冒泡，debug 被换成只含失败原因的新 dict（与搬移前一致）
    def boom(*_a, **_k):
        raise RuntimeError("divider exploded")

    monkeypatch.setattr(ref_mod, "own_lane_beside_divider", boom)
    divider, dbg = ref_mod._divider_lane(None, np.zeros(3), 0.0, _route(),
                                        None, debug={"stale": 1})
    assert divider is None
    assert "reason" in dbg and "exploded" in dbg["reason"]
    assert "stale" not in dbg


def test_fail_closed_backstop_clears_half_a_decision():
    """末道闸：sensor 标签 + 无几何 -> unavailable，且几何一起清掉。"""
    from beamng_autopilot.fsd_realism import SRC_UNAVAILABLE as UNAVAIL
    from beamng_autopilot.lane.reference import _enforce_publishable

    bad = LaneReference(center=None, src=SRC_SENSOR, src_published=True,
                        left=np.zeros((3, 2)), right=np.zeros((3, 2)),
                        width=3.5, boundaries=True, meta={"lane_src": "sensor"})
    out = _enforce_publishable(bad)
    assert out.src == UNAVAIL
    assert out.center is None and out.left is None and out.right is None
    assert out.width == 0.0 and out.boundaries is False
    assert out.meta["lane_ref_invariant"].startswith("sensor source")
    assert out.meta["lane_src"] != "sensor"

    good = LaneReference(center=np.column_stack([np.linspace(0, 10, 6),
                                                 np.full(6, -1.8)]),
                         src=SRC_SENSOR, meta={})
    assert _enforce_publishable(good) is good
    assert good.src == SRC_SENSOR


# --------------------------------------------------------------------------
# geometric consistency gate (plan E4, opt-in; default OFF)
# --------------------------------------------------------------------------
def test_geometry_gate_withdraws_a_geometrically_bad_sensor_lane(monkeypatch):
    """形状不合格 -> 撤销为 unavailable，reason 以 ``geom:`` 开头。

    先钉住"撤销"这条行为，再动它的实现（本轮把它抽成 _apply_geometry_gate）。
    """
    from beamng_autopilot.lane import reference as ref_mod

    class BadReport:
        ok = False
        reasons = ["vanish"]

        @staticmethod
        def digest():
            return {"ok": False}

    monkeypatch.setattr(ref_mod, "LANE_GEOM_ENABLED", True)
    import beamng_autopilot.lane.geometry_check as gc
    monkeypatch.setattr(gc, "check_lane_geometry",
                        lambda **kw: BadReport())
    ref = select_lane_reference(
        lane_frame=_sensor_lane(), pos=np.zeros(3), heading=0.0,
        route_ref=_route(), has_nav_route=True,
        lane_mode="sensor", strict_sensor=False,
    )
    assert ref.src == SRC_UNAVAILABLE
    assert ref.center is None
    # 几何撤销写在 meta 里（dataclass 的 reject_reason 只反映三维门），
    # 这是当前契约——先钉住，抽实现时不许改。
    reason = str(ref.meta.get("lane_reject_reason"))
    assert reason.startswith("geom:"), reason
    assert "vanish" in reason
    assert ref.meta["lane_src"] != SRC_SENSOR


def test_geometry_gate_error_leaves_the_decision_unchanged(monkeypatch):
    """检查器自己抛异常 -> 只记 lane_geom_error，决策不变（不能因工具坏了就撤）。"""
    from beamng_autopilot.lane import reference as ref_mod

    def boom(**kw):
        raise RuntimeError("geometry exploded")

    monkeypatch.setattr(ref_mod, "LANE_GEOM_ENABLED", True)
    import beamng_autopilot.lane.geometry_check as gc
    monkeypatch.setattr(gc, "check_lane_geometry", boom)
    ref = select_lane_reference(
        lane_frame=_sensor_lane(), pos=np.zeros(3), heading=0.0,
        route_ref=_route(), has_nav_route=True,
        lane_mode="sensor", strict_sensor=False,
    )
    assert "geometry exploded" in ref.meta.get("lane_geom_error", "")
    assert ref.center is not None and len(ref.center) >= 3


def test_geometry_gate_is_a_no_op_when_disabled():
    from beamng_autopilot.lane.reference import _apply_geometry_gate

    center = np.column_stack([np.linspace(0, 10, 6), np.full(6, -1.8)])
    meta = {}
    out_center, out_src, revoked = _apply_geometry_gate(
        enabled=False, center=center, lane_frame=None, lane_src_sel=SRC_SENSOR,
        grid=None, corridor=None, prev_ref=None, meta=meta)
    assert out_center is center and out_src == SRC_SENSOR and revoked is False
    assert meta == {}, "默认关时不得写任何 meta 字段"


def test_hard_boundaries_need_a_real_perceived_edge():
    """no-cross 权威只给真被感知到的边：配对、铺装、分隔线、或地图先验本身。"""
    from beamng_autopilot.lane.reference import _publishes_hard_boundaries

    paired = _sensor_lane(paired=True)
    single = _sensor_lane(paired=False)
    assert _publishes_hard_boundaries(paired, SRC_SENSOR, None, None) is True
    assert _publishes_hard_boundaries(single, SRC_SENSOR, None, None) is False
    assert _publishes_hard_boundaries(None, SRC_PAVED, None, None) is True
    assert _publishes_hard_boundaries(None, SRC_MAP, ("mc", "ml", "mr"),
                                      None) is True
    assert _publishes_hard_boundaries(None, SRC_SENSOR, None,
                                      ("center", "left")) is True
    assert _publishes_hard_boundaries(None, SRC_SENSOR, None, None) is False
