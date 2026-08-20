"""Lane topology head (FSD lane-network graph shape).

FSD's lane neural network doesn't just see lines - it outputs a *lane
graph*: which lane the ego is in, whether adjacent lanes exist, and how
they connect, so the planner can decide "can I change left / right" and
"where is the drivable corridor".  This head gives the project that
``LaneGraph`` structure on top of the existing ``LaneFrame`` (a two-sided
frame already carries the left/right boundaries and the lane centre).

* ``LaneEdge`` / ``LaneGraph``: the graph shape - a string id, the
  ego's lane with its bounds, and the left/right neighbours plus their
  exist / crossable flags.
* ``build_lane_graph``: from a ``LaneFrame`` (+ lidar corridor) derive
  the graph.  Pure + unit-testable.
* ``LaneTopologyHead``: a HydraNet head that consumes a FrameContext and
  a precomputed sensor lane, emitting the graph in ``TaskOutput.meta``.

The graph's "crossable" semantics reuse the single-edge trust from the
planner: a boundary that the ego could legally cross (overtaking /
lane change) is a real painted line; a solid wall / guardrail is not a
lane boundary at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..hydra import FrameContext, TaskOutput

# How much lateral room a neighbour needs before it "exists" (m).
NEIGHBOUR_MIN_M = 0.8


@dataclass
class LaneEdge:
    """One edge of the lane graph (a lane boundary or a facet)."""

    kind: str                # "left" | "right"
    exists: bool = True
    crossable: bool = True   # painted line (could legally change lane)
    offset_m: float = 0.0    # lateral distance from ego to the boundary


@dataclass
class LaneGraph:
    """The ego lane and its neighbours - FSD lane-graph shape."""

    has_lane: bool = False
    centre: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    width_m: float = 0.0
    left: LaneEdge = field(default_factory=lambda: LaneEdge("left"))
    right: LaneEdge = field(default_factory=lambda: LaneEdge("right"))
    # summary for the planner / telemetry
    labels: dict[str, bool] = field(default_factory=dict)

    def to_meta(self) -> dict:
        return {
            "has_lane": self.has_lane,
            "width": round(self.width_m, 2),
            "left_exists": self.left.exists,
            "left_crossable": self.left.crossable,
            "right_exists": self.right.exists,
            "right_crossable": self.right.crossable,
        }


def build_lane_graph(lane_frame, width_default: float = 3.5,
                     solid_kinds=("solid", "wall", "guardrail")
                     ) -> LaneGraph:
    """Derive a ``LaneGraph`` from a sensor ``LaneFrame``.

    Uses the frame's explicit left/right boundaries when present; a
    single-sided frame mirrors the assumed width to estimate the far
    side.  A boundary whose kind is a physical no-cross (solid wall /
    guardrail / solid paint) is marked not crossable - the planner must
    not change lanes through it.
    """
    graph = LaneGraph()
    if lane_frame is None:
        graph.labels = {"lane_ok": False,
                        "change_left_ok": False,
                        "change_right_ok": False}
        return graph
    center = getattr(lane_frame, "center", None)
    centre = np.asarray(center[:, :2], dtype=float) if center is not None \
        and len(center) else np.zeros((0, 2))
    graph.centre = centre
    width = float(getattr(lane_frame, "width", 0.0) or 0.0)
    if width <= 0.0:
        width = width_default
    graph.width_m = width
    graph.has_lane = len(centre) >= 2

    left = getattr(lane_frame, "left", None)
    right = getattr(lane_frame, "right", None)

    if left is not None and len(left):
        offset = _edge_offset(centre, np.asarray(left[:, :2], dtype=float))
        left_edge = LaneEdge(
            "left", exists=True,
            crossable=_crossable_kind(str(getattr(lane_frame, "left_kind",
                                                  "")), solid_kinds),
            offset_m=offset)
    else:
        # assume a mirrored left boundary
        left_edge = LaneEdge("left", exists=width > NEIGHBOUR_MIN_M,
                             crossable=True, offset_m=width / 2.0)
    if right is not None and len(right):
        offset = _edge_offset(centre, np.asarray(right[:, :2], dtype=float))
        right_edge = LaneEdge(
            "right", exists=True,
            crossable=_crossable_kind(str(getattr(lane_frame, "right_kind",
                                                  "")), solid_kinds),
            offset_m=offset)
    else:
        right_edge = LaneEdge("right", exists=width > NEIGHBOUR_MIN_M,
                              crossable=True, offset_m=width / 2.0)

    graph.left = left_edge
    graph.right = right_edge
    graph.labels = {
        "lane_ok": graph.has_lane,
        "change_left_ok": left_edge.exists and left_edge.crossable,
        "change_right_ok": right_edge.exists and right_edge.crossable,
    }
    return graph


def _crossable_kind(kind: str, solid_kinds) -> bool:
    kind = (kind or "").lower()
    return not (kind in solid_kinds)


def _edge_offset(centre, edge_pts) -> float:
    """Median lateral distance from the lane centre to a boundary."""
    if len(centre) < 1 or len(edge_pts) < 1:
        return 0.0
    # average distance from centre points to the nearest edge point
    offs = []
    for c in centre[:6]:
        d = np.linalg.norm(edge_pts - c, axis=1)
        offs.append(float(d.min()))
    return float(np.median(offs)) if offs else 0.0


class LaneTopologyHead:
    """HydraNet head: sensor lane -> lane graph."""

    name = "topology"

    def __init__(self, solid_kinds=("solid", "wall", "guardrail"),
                 width_default: float = 3.5):
        self.solid_kinds = tuple(solid_kinds)
        self.width_default = float(width_default)

    def run(self, ctx: FrameContext,
            sensor_lane=None) -> TaskOutput:
        out = TaskOutput()
        graph = build_lane_graph(
            sensor_lane, width_default=self.width_default,
            solid_kinds=self.solid_kinds)
        out.meta["lane_graph"] = graph.to_meta()
        out.meta["change_left"] = graph.labels["change_left_ok"]
        out.meta["change_right"] = graph.labels["change_right_ok"]
        return out