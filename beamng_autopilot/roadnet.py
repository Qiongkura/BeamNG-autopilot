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
    def __init__(self, intersection_join_m: float = 2.5):
        self.nodes: np.ndarray | None = None  # Nx2 (x, y)
        self.heights: np.ndarray | None = None  # N, road-surface z per node
        self.lefts: np.ndarray | None = None  # Nx2 road LEFT edge (world)
        self.rights: np.ndarray | None = None  # Nx2 road RIGHT edge (world)
        self.adj: dict[int, list[tuple[int, float]]] = {}
        self.ready = False
        self.info = "no road data"
        # Road centre-lines arrive per DecalRoad segment; two roads meeting
        # at an intersection keep separate endpoint nodes that the raw edge
        # list never links.  Stitch their near nodes into one graph so A*
        # can route ACROSS junctions (town -> goal probe: the car's start
        # component was only 23 nodes and A* returned "no path").
        self.intersection_join_m = float(intersection_join_m)
        self.stitch_edges = 0

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
        lefts: list[np.ndarray] = []
        rights: list[np.ndarray] = []
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
            row_heights = []
            row_lefts = []
            row_rights = []
            for row in edge_rows:
                m = row.get("middle")
                if m is None:
                    continue
                m3 = np.asarray(m, dtype=float)
                row_pts.append(m3[:2])
                row_heights.append(float(m3[2]) if m3.size >= 3 else math.nan)
                _edge2 = lambda k: (np.asarray(row.get(k), dtype=float)[:2]
                                    if row.get(k) is not None
                                    and len(np.asarray(row.get(k), dtype=float)) >= 2
                                    else np.array([np.nan, np.nan]))
                row_lefts.append(_edge2("left"))
                row_rights.append(_edge2("right"))
            # gate BEFORE appending: a road with a single valid row would
            # push 1 edge row but 0 nodes, desyncing len(lefts) from
            # len(nodes) and discarding the WHOLE map's edge data
            if len(row_pts) >= 2:
                base = len(pts)
                pts.extend(row_pts)
                heights.extend(row_heights)
                lefts.extend(row_lefts)
                rights.extend(row_rights)
                for i in range(len(row_pts) - 1):
                    edges.append((base + i, base + i + 1))
                n_used += 1

        if len(pts) < 2:
            self.info = "no usable road centre lines found"
            return False

        self.nodes = np.asarray(pts, dtype=float)
        self.lefts = (np.asarray(lefts, dtype=float)
                      if len(lefts) == len(self.nodes) else None)
        self.rights = (np.asarray(rights, dtype=float)
                       if len(rights) == len(self.nodes) else None)
        self.heights = None
        if len(heights) == len(self.nodes) and all(
                math.isfinite(h) for h in heights):
            self.heights = np.asarray(heights, dtype=float)
        self.adj = {i: [] for i in range(len(self.nodes))}
        for a, b in edges:
            d = float(np.linalg.norm(self.nodes[a] - self.nodes[b]))
            self.adj[a].append((b, d))
            self.adj[b].append((a, d))
        # Stitch DecalRoad endpoints that meet near an intersection so the
        # graph becomes one connected navigable network (A* otherwise can
        # find no path across a junction).
        try:
            self.stitch_edges = self._stitch_intersections(
                join_m=self.intersection_join_m)
        except Exception as exc:  # scipy unavailable should not kill build
            logger.warning("[roadnet] intersection stitch failed: %s", exc)
            self.stitch_edges = 0
        self.ready = True
        self.info = (f"{len(self.nodes)} nodes / {len(edges)} edges / "
                     f"+{self.stitch_edges} join edges / {n_used} roads")
        return True

    def _stitch_intersections(self, join_m: float) -> int:
        """Bridge near DecalRoad nodes so A* can cross junctions.

        Each road's centre line is stored as one polyline; its nodes are
        only linked to the node before/after on the SAME road.  At a
        junction (including where a road continues straight through while
        another one ends into it) the endpoint nodes of the meeting roads
        lie within a couple of metres but are never connected, so the graph
        is disconnected and ``route()`` returns None for any start/goal on
        different roads.

        We add an undirected edge between every node pair that is closer
        than ``join_m`` and not already adjacent.  The radius is deliberately
        small so we only stitch genuine junction geography and never shortcut
        across a wide median / block between parallel carriageways.
        """
        if self.nodes is None or len(self.nodes) < 3:
            return 0
        try:
            from scipy.spatial import cKDTree
        except Exception:
            return 0
        tree = cKDTree(self.nodes)
        pairs = tree.query_pairs(r=float(join_m), output_type="ndarray")
        existing = set()
        for a, bs in self.adj.items():
            for b, _ in bs:
                existing.add((a, b))
        added = 0
        for a, b in pairs:
            ia, ib = int(a), int(b)
            if ia == ib or (ia, ib) in existing or (ib, ia) in existing:
                continue
            d = float(np.linalg.norm(self.nodes[ia] - self.nodes[ib]))
            if d <= 1e-9 or d > float(join_m):
                continue
            self.adj[ia].append((ib, d))
            self.adj[ib].append((ia, d))
            existing.add((ia, ib))
            existing.add((ib, ia))
            added += 1
        return added

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

    def _astar_idx(self, start_xy, goal_xy):
        """A* over the road graph; returns the node-index path or None."""
        if not self.ready:
            return None
        s = self._nearest(start_xy)
        g = self._nearest(goal_xy)
        if s == g:
            return [s]
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
        return idx_path

    def route(self, start_xy, goal_xy, step: float = 1.5):
        """A* along the road graph; returns Nx2 waypoints (start..goal)."""
        idx_path = self._astar_idx(start_xy, goal_xy)
        if idx_path is None:
            return None
        chain = [np.asarray(start_xy[:2], dtype=float)]
        chain.extend(self.nodes[i] for i in idx_path)
        chain.append(np.asarray(goal_xy[:2], dtype=float))
        return self._interpolate_chain(chain, step)

    def route_with_edges(self, start_xy, goal_xy, step: float = 1.5):
        """A* route plus the road's real LEFT/RIGHT edge polylines.

        Returns ``(route, left, right)`` - all Nx2 world polylines
        sampled at the SAME arc positions (same ``step`` interpolation as
        ``route()``), so ``right[k]`` is the road's right edge beside
        ``route[k]``.  Edge rows can be missing for a node (NaN rows are
        skipped); the caller falls back to synthetic offsets when a
        polyline has too few usable points.
        """
        idx_path = self._astar_idx(start_xy, goal_xy)
        if idx_path is None:
            return None, None, None
        if self.lefts is None or self.rights is None:
            route = self.route(start_xy, goal_xy, step=step)
            return route, None, None
        route_pts: list[np.ndarray] = []
        left_pts: list[np.ndarray] = []
        right_pts: list[np.ndarray] = []
        route_pts.append(np.asarray(start_xy[:2], dtype=float))
        left_pts.append(np.array([np.nan, np.nan]))
        right_pts.append(np.array([np.nan, np.nan]))
        nodes = self.nodes
        lefts = self.lefts
        rights = self.rights
        for i in idx_path:
            route_pts.append(nodes[i])
            left_pts.append(lefts[i])
            right_pts.append(rights[i])
        route_pts.append(np.asarray(goal_xy[:2], dtype=float))
        left_pts.append(np.array([np.nan, np.nan]))
        right_pts.append(np.array([np.nan, np.nan]))

        # Interpolate edges with the SAME t fractions as the route: walk
        # route_pts pairs and, for each route sample, the edge pair index
        # matches the route pair index (start/goal appended NaN rows are
        # skipped by holding the previous valid edge point).
        def _interp_edge(edge_pts, route_out):
            out: list[np.ndarray] = []
            k = 0
            prev = None
            for a, b in zip(route_pts, route_pts[1:]):
                dist = float(np.linalg.norm(b - a))
                if dist < 1e-6:
                    # Zero-length route pair (e.g. the caller passed the
                    # nearest road NODE as start, so start == nodes[0]):
                    # the route interpolation skips it AND its edge pair
                    # must be skipped too - otherwise the edge index lags
                    # one node behind and the whole first segment holds
                    # the node-0 edge constant (own-lane boundary 5 m
                    # wrong at the start -> every candidate rejected).
                    k += 1
                    continue
                n = max(1, int(np.ceil(dist / step)))
                ts = np.linspace(0.0, 1.0, n, endpoint=False)
                ea = edge_pts[k]
                eb = edge_pts[k + 1]
                k += 1
                if (ea is None or not np.all(np.isfinite(ea))):
                    ea = prev if prev is not None else eb
                if (eb is None or not np.all(np.isfinite(eb))):
                    eb = ea
                if np.all(np.isfinite(ea)):
                    prev = ea
                if np.all(np.isfinite(ea)) and np.all(np.isfinite(eb)):
                    for t in ts:
                        out.append(ea + t * (eb - ea))
                else:
                    for _ in ts:
                        out.append(np.array([np.nan, np.nan]))
            if len(out) < len(route_out):
                last = prev if prev is not None else np.array([np.nan, np.nan])
                out.append(last)
            return np.asarray(out, dtype=float)

        route_out = self._interpolate_chain(route_pts, step)
        left_out = _interp_edge(left_pts, route_out)
        right_out = _interp_edge(right_pts, route_out)
        # Trim NaN padding at the head/tail using a COMMON window so the
        # left/right polylines keep the same index as the route (trimming
        # each side independently would shift them relative to each other).
        good = np.flatnonzero(
            np.all(np.isfinite(left_out), axis=1)
            & np.all(np.isfinite(right_out), axis=1))
        if len(good) < 4:
            return route_out, None, None
        s = int(good[0])
        e = int(good[-1]) + 1
        route_out = route_out[s:e]
        left_out = left_out[s:e]
        right_out = right_out[s:e]
        n = len(route_out)
        # Orient the stored left/right edges to the ROUTE travel direction:
        # DecalRoad rows store left/right in the road's own direction, which
        # can be the OPPOSITE of the A* path on a segment (hairpin rows flip
        # the pair at the apex).  The planner needs the edge on the route's
        # RIGHT side - pick per vertex by the sign of the lateral offset.
        for k in range(1, n - 1):
            tv = route_out[k + 1] - route_out[k - 1]
            L = float(np.linalg.norm(tv))
            if L < 1e-9:
                continue
            rn = np.array([tv[1] / L, -tv[0] / L])
            if np.all(np.isfinite(left_out[k])) \
                    and np.all(np.isfinite(right_out[k])):
                dl = float(np.dot(left_out[k] - route_out[k], rn))
                dr = float(np.dot(right_out[k] - route_out[k], rn))
                if dr < 0.0 and dl > 0.0:
                    lk = left_out[k].copy()
                    rk = right_out[k].copy()
                    left_out[k] = rk
                    right_out[k] = lk
        return route_out, left_out, right_out

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
