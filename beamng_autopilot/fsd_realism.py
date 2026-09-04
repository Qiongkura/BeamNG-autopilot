"""FSD realism invariants - the hard bottom-layer requirements.

This module makes the project's "same logic as a real FSD" requirement
machine-checkable instead of prose.  The invariants are documented in
``docs/fsd_realism.md``; the helpers here let tests (and runtime gates)
assert that the lane-keep / road-boundary logic stays PERCEPTION-only.

Key contract
------------
* lane source ``"sensor"``              - FSD: perception lane leads.
* lane source ``"perception-unavailable"`` - FSD degradation: no lane,
  safety decides (never map lane geometry).
* lane source ``"map"``                 - NON-FSD fallback (old rule
  compatibility).  Allowed only when ``strict`` is False.
"""

from __future__ import annotations

from pathlib import Path

# Lane sources the stack can report for its lane-keep reference.
SRC_SENSOR = "sensor"
SRC_UNAVAILABLE = "perception-unavailable"
SRC_MAP = "map"
SRC_BEV_ROUTE = "bev/route"

FSD_LANE_SOURCES = (SRC_SENSOR, SRC_UNAVAILABLE)
NON_FSD_LANE_SOURCES = (SRC_MAP, SRC_BEV_ROUTE)

# Machine-checkable invariant registry: (id, rule, enforced_by).
FSD_INVARIANTS = (
    ("lane-perception-only",
     "lane centre/boundaries must come from perception (semantic + LiDAR), "
     "never from map/nav lane geometry",
     "lane/source + perception_lateral_guard + strict_sensor"),
    ("map-intent-only",
     "nav route is destination intent only; it must not drive lateral "
     "lane-keeping",
     "fsd_stack plan_route/lane_ref split"),
    ("perception-unavailable-degrades",
     "no paired sensor lane -> no-lane degradation, never a map-lane lead",
     "fsd_stack strict_sensor=True"),
    ("no-simulator-privilege-in-inference",
     "Lua ground truth / annotations must not enter the inference path",
     "vision/ + runtime providers"),
    ("shadow-loop",
     "training data = real executed control aligned with shadow "
     "predictions, bad frames dropped",
     "recording.py + dataset wedge/quality gates"),
)

# Module source files that must NEVER import the road graph / map lane
# geometry (the perception-only rule).  Checked by test_fsd_realism.
# Every entry is a perception / safety / planning / control pure-logic
# module (no RoadNetwork, DecalRoad, map lane or nav polyline reference).
# fsd_stack.py is deliberately NOT listed: it is the integration point
# that legitimately consumes the nav route as destination intent.
NO_MAP_GUARDED_FILES = (
    # lane perception
    "beamng_autopilot/lane/perception_guard.py",
    "beamng_autopilot/lane/constants.py",
    "beamng_autopilot/lane/fusion.py",
    "beamng_autopilot/lane/lidar.py",
    "beamng_autopilot/lane/pairing.py",
    "beamng_autopilot/lane/tracking.py",
    # camera-ring vision perception
    "beamng_autopilot/vision/ring.py",
    "beamng_autopilot/vision/lanes.py",
    "beamng_autopilot/vision/hydra.py",
    "beamng_autopilot/vision/segmentation.py",
    "beamng_autopilot/vision/band.py",
    "beamng_autopilot/vision/detection.py",
    # BEV / temporal / safety
    "beamng_autopilot/occupancy.py",
    "beamng_autopilot/bev_fusion.py",
    "beamng_autopilot/temporal.py",
    "beamng_autopilot/safety_monitor.py",
    # planning pure logic (sensor-space planning, not map polylines)
    "beamng_autopilot/planning/arbiter.py",
    "beamng_autopilot/planning/constraints.py",
    "beamng_autopilot/planning/geometry.py",
    "beamng_autopilot/planning/intent.py",
    "beamng_autopilot/planning/scene.py",
    "beamng_autopilot/planning/selector.py",
    "beamng_autopilot/planning/speed_profile.py",
    "beamng_autopilot/planning/trajectory.py",
    # control pure logic (tracking / gearbox / anti-reverse / speed)
    "beamng_autopilot/control/gearbox.py",
    "beamng_autopilot/control/handover.py",
    "beamng_autopilot/control/pid.py",
    "beamng_autopilot/control/pure_pursuit.py",
    "beamng_autopilot/control/reverse_guard.py",
    "beamng_autopilot/control/reverse_maneuver.py",
    "beamng_autopilot/control/speed.py",
)


def lane_source_ok(src: str | None, strict: bool = False) -> bool:
    """True when a lane source satisfies the given realism level."""
    if src is None:
        return False
    if strict:
        return src in FSD_LANE_SOURCES
    return src in FSD_LANE_SOURCES or src in NON_FSD_LANE_SOURCES


def assert_realistic_lane(src: str | None, strict: bool = False) -> None:
    """Raise ``ValueError`` when a strict run is about to use a map lane."""
    if strict and src in NON_FSD_LANE_SOURCES:
        raise ValueError(
            f"FSD strict mode refuses non-perception lane source: {src!r} "
            "(docs/fsd_realism.md §4)")


def check_no_map_imports() -> list[str]:
    """List guarded files that (accidentally) use road-graph/map code.

    Parses the module with ``ast`` so docstrings that merely *say* "no
    RoadNetwork / no nav route" are not flagged - only real imports,
    names and attribute accesses are.
    """
    import ast

    root = Path(__file__).resolve().parents[1]
    bad = []
    for rel in NO_MAP_GUARDED_FILES:
        text = (root / rel).read_text(encoding="utf-8")
        tree = ast.parse(text)
        names: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for a in node.names:
                    names.append(a.name.split(".")[0])
            elif isinstance(node, ast.Name):
                names.append(node.id)
            elif isinstance(node, ast.Attribute):
                names.append(node.attr)
        lowered = " ".join(names).lower()
        if any(tok in lowered for tok in ("roadnetwork", "road_graph",
                                          "roadgraph", "decalroad",
                                          "map_lane", "nav_route")):
            bad.append(rel)
    return bad
