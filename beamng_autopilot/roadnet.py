"""Road-network route generation from the running scenario.

Queries DecalRoad data through beamngpy and builds a graph from the road
centre lines, then runs A* to generate a drivable route between two points
the user marks in the map.  Falls back to a straight interpolated path only
when the map has no DecalRoads at all (free-roam maps); when roads exist,
A* must find a real route and returns None on failure.
"""

from __future__ import annotations

import heapq
import logging
import math

import numpy as np

logger = logging.getLogger(__name__)


class RoadNetwork:
    def __init__(self):
        self.nodes: np.ndarray | None = None  # Nx2 (x, y)
        self.heights: np.ndarray | None = None  # N, road-surface z per node
        self.adj: dict[int, list[tuple[int, float]]] = {}
        self.ready = False
        self.info = "no road data"

    @property
    def node_count(self) -> int:
        return 0 if self.nodes is None else len(self.nodes)

    def build(self, bng) -> bool:
        """Query the running scenario and build the graph."""
        try:
            # Prefer the scenario API: it works on BeamNG.tech, where the
            # legacy get_roads() command can hang forever on a started map.
            road_network = getattr(getattr(bng, "scenario", None),
                                   "get_road_network", None)
            if road_network is not None:
                roads = road_network(
                    include_edges=True, drivable_only=True)
            else:
                roads = bng.get_road_network(
                    include_edges=True, drivable_only=True)
        except Exception:
            # NOTE: bare except kept — scenario API can fail with any
            # transport error; fall back to legacy get_roads().
            try:
                roads = bng.get_roads()
            except Exception as exc:
                self.info = f"road query failed: {exc}"
                return False
        if not roads:
            self.info = "map has no DecalRoad data"
            return False

        pts: list[np.ndarray] = []
        heights: list[float] = []
        edges: list[tuple[int, int]] = []
        n_used = 0
        for road_id, meta in roads.items():
            name = road_id
            edge_rows = None
            if isinstance(meta, dict):
                name = meta.get("name") or meta.get("mid") or road_id
                if isinstance(meta.get("edges"), list):
                    edge_rows = meta["edges"]
            if not edge_rows:
                try:
                    edge_rows = bng.get_road_edges(str(name))
                except Exception:
                    # NOTE: bare except kept — Lua command can fail with
                    # any transport error; skip this road segment.
                    continue
            if not edge_rows:
                continue
            row_pts = []
            for row in edge_rows:
                m = row.get("middle")
                if m is None:
                    continue
                m3 = np.asarray(m, dtype=float)
                row_pts.append(m3[:2])
                heights.append(float(m3[2]) if m3.size >= 3 else math.nan)
            if len(row_pts) >= 2:
                base = len(pts)
                pts.extend(row_pts)
                for i in range(len(row_pts) - 1):
                    edges.append((base + i, base + i + 1))
                n_used += 1

        if len(pts) < 2:
            self.info = "no usable road centre lines found"
            return False

        self.nodes = np.asarray(pts, dtype=float)
        self.heights = None
        if len(heights) == len(self.nodes) and all(
                math.isfinite(h) for h in heights):
            self.heights = np.asarray(heights, dtype=float)
        self.adj = {i: [] for i in range(len(self.nodes))}
        for a, b in edges:
            d = float(np.linalg.norm(self.nodes[a] - self.nodes[b]))
            self.adj[a].append((b, d))
            self.adj[b].append((a, d))
        self.ready = True
        self.info = f"{len(self.nodes)} nodes / {len(edges)} edges / {n_used} roads"
        return True

    def _nearest(self, xy) -> int:
        d = np.linalg.norm(self.nodes - np.asarray(xy[:2], dtype=float), axis=1)
        return int(np.argmin(d))

    def nearby_polylines(self, xy, radius: float = 90.0):
        """Chain the road graph near ``xy`` into centre-line polylines.

        Returns a list of (N, 2) arrays (world x/y), each a connected run of
        road nodes whose nodes all lie within ``radius`` metres of ``xy``.
        Used by the EID to draw the surrounding road network (lane lines)
        without shipping the whole graph to the GUI.
        """
        if not self.ready or self.nodes is None:
            return []
        center = np.asarray(xy[:2], dtype=float)
        d = np.linalg.norm(self.nodes[:, :2] - center, axis=1)
        near = set(np.where(d <= radius)[0].tolist())
        if not near:
            return []

        used: set[int] = set()
        chains: list[np.ndarray] = []
        for start in sorted(near):
            if start in used:
                continue
            chain = [start]
            used.add(start)
            # Extend forward then backward through unvisited near nodes.
            while True:
                cur = chain[-1]
                nxt = [j for j, _ in self.adj.get(cur, [])
                       if j in near and j not in used]
                if not nxt:
                    break
                used.add(nxt[0])
                chain.append(nxt[0])
            while True:
                cur = chain[0]
                prv = [j for j, _ in self.adj.get(cur, [])
                       if j in near and j not in used]
                if not prv:
                    break
                used.add(prv[0])
                chain.insert(0, prv[0])
            if len(chain) >= 2:
                chains.append(np.asarray(self.nodes[chain, :2], dtype=float))
        return chains

    def nearest_node_xyz(self, xy):
        """World (x, y, z) of the nearest road node, or None when no height
        data is available (z is the road-surface height)."""
        if not self.ready or self.heights is None:
            return None
        i = self._nearest(xy)
        return (float(self.nodes[i, 0]), float(self.nodes[i, 1]),
                float(self.heights[i]))

    def road_heading_at(self, xy):
        """Heading (radians, atan2 convention) of the road axis at the node
        nearest to xy, aligned along the longest continuous road chain.
        Returns None when the node has no neighbours."""
        if not self.ready:
            return None
        i = self._nearest(xy)
        nbrs = self.adj.get(i, [])
        if not nbrs:
            return None
        best_j, best_len = -1, 0.0
        for j, _ in nbrs:
            prev, cur = i, j
            length = 0.0
            for _ in range(200):
                length += float(np.linalg.norm(self.nodes[cur] - self.nodes[prev]))
                nxt = [k for k, _ in self.adj.get(cur, []) if k != prev]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]
            if length > best_len:
                best_len, best_j = length, j
        if best_j < 0:
            return None
        d = self.nodes[best_j] - self.nodes[i]
        return float(np.arctan2(d[1], d[0]))

    def goal_along_route(self, start_xy, dist: float):
        """Pick a road node roughly `dist` metres along the road graph from
        the node nearest to start_xy (used for scripted e2e goals)."""
        if not self.ready:
            return None
        s = self._nearest(start_xy)
        best = (0.0, s)
        far = (0.0, s)
        visited = {s: 0.0}
        heap = [(0.0, s)]
        while heap:
            d, cur = heapq.heappop(heap)
            if d != visited.get(cur):
                continue
            if d > far[0]:
                far = (d, cur)
            if abs(d - dist) < abs(best[0] - dist):
                best = (d, cur)
            for nxt, w in self.adj.get(cur, []):
                nd = d + w
                if nd < visited.get(nxt, float("inf")):
                    visited[nxt] = nd
                    heapq.heappush(heap, (nd, nxt))
        if best[0] < dist * 0.5:
            best = far
        i = best[1]
        return (float(self.nodes[i, 0]), float(self.nodes[i, 1]))

    def route(self, start_xy, goal_xy, step: float = 1.5):
        """A* along the road graph; returns Nx2 waypoints (start..goal)."""
        if not self.ready:
            return None
        s = self._nearest(start_xy)
        g = self._nearest(goal_xy)
        if s == g:
            return self._interpolate(start_xy, goal_xy, step)

        open_heap: list[tuple[float, int]] = [(0.0, s)]
        came: dict[int, int] = {}
        gcost = {s: 0.0}
        closed = set()
        while open_heap:
            f, cur = heapq.heappop(open_heap)
            if cur == g:
                break
            if cur in closed:
                continue
            closed.add(cur)
            for nxt, w in self.adj.get(cur, []):
                if nxt in closed:
                    continue
                ng = gcost[cur] + w
                if ng < gcost.get(nxt, float("inf")):
                    gcost[nxt] = ng
                    came[nxt] = cur
                    h = float(np.linalg.norm(self.nodes[nxt] - self.nodes[g]))
                    heapq.heappush(open_heap, (ng + h, nxt))
        if g not in gcost:
            self.info = "A* found no path on road graph"
            return None

        idx_path = [g]
        cur = g
        while cur != s:
            cur = came[cur]
            idx_path.append(cur)
        idx_path.reverse()
        chain = [np.asarray(start_xy[:2], dtype=float)]
        chain.extend(self.nodes[i] for i in idx_path)
        chain.append(np.asarray(goal_xy[:2], dtype=float))
        return self._interpolate_chain(chain, step)

    @staticmethod
    def _interpolate_chain(chain, step: float) -> np.ndarray:
        out = []
        for a, b in zip(chain, chain[1:]):
            dist = float(np.linalg.norm(b - a))
            if dist < 1e-6:
                continue
            n = max(1, int(np.ceil(dist / step)))
            ts = np.linspace(0.0, 1.0, n, endpoint=False)
            for t in ts:
                out.append(a + t * (b - a))
        if out:
            out.append(chain[-1])
        return np.asarray(out, dtype=float)

    @classmethod
    def _interpolate(cls, start_xy, goal_xy, step: float) -> np.ndarray:
        a = np.asarray(start_xy[:2], dtype=float)
        b = np.asarray(goal_xy[:2], dtype=float)
        return cls._interpolate_chain([a, b], step)
