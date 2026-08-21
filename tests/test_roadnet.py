"""Offline tests for RoadNetwork intersection stitching + A* route."""
from __future__ import annotations

import numpy as np

from beamng_autopilot.roadnet import RoadNetwork


def _two_road_junction():
    """Two road polylines meeting at (5,0); separate road components."""
    rn = RoadNetwork()
    pts = []
    for i in range(11):          # horizontal road A: x 0..10 at y 0
        pts.append((float(i), 0.0))
    for i in range(1, 11):       # vertical road B: x 5, y 1..10
        pts.append((5.0, float(i)))
    rn.nodes = np.asarray(pts, dtype=float)
    rn.adj = {i: [] for i in range(len(pts))}
    for i in range(10):
        rn.adj[i].append((i + 1, 1.0))
        rn.adj[i + 1].append((i, 1.0))
    for i in range(11, 20):
        rn.adj[i].append((i + 1, 1.0))
        rn.adj[i + 1].append((i, 1.0))
    rn.ready = True
    return rn


def test_stitch_connects_intersection() -> None:
    rn = _two_road_junction()
    added = rn._stitch_intersections(join_m=2.0)
    assert added > 0
    # the two road components must now be one connected graph: BFS from
    # road A node 0 must reach road B nodes (indices >= 11)
    seen = {0}
    stack = [0]
    while stack:
        cur = stack.pop()
        for nxt, _ in rn.adj[cur]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    assert any(b >= 11 for b in seen)


def test_route_crosses_junction_after_stitch() -> None:
    rn = _two_road_junction()
    rn._stitch_intersections(join_m=2.0)
    route = rn.route((0.0, 0.0), (5.0, 10.0), step=1.0)
    assert route is not None and len(route) >= 12
    assert abs(float(route[0, 0])) < 1e-6 and abs(float(route[0, 1])) < 1e-6
    assert abs(float(route[-1, 0]) - 5.0) < 1e-6
    assert abs(float(route[-1, 1]) - 10.0) < 1e-6


def test_no_route_before_stitch() -> None:
    rn = _two_road_junction()
    # without stitching the two components are disconnected -> no route
    route = rn.route((0.0, 0.0), (5.0, 10.0), step=1.0)
    assert route is None
