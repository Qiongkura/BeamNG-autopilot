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
