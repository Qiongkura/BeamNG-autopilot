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
