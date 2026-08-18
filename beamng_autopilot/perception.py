"""Environment perception: nearby obstacles from the running scenario.

Two complementary sources, both driven through the beamngpy Lua bridge so
they work on the Steam edition without a tech license:

* ``scan_obstacles``: the game's object database (``scenario_objects``).
  Every spawned object (other vehicles, barriers, cones, trees, ...) is
  reported as an axis-aligned box in world space.  On free-roam maps this
  can be empty because most static scenery is not registered as a
  scenario object.

* ``scan_obstacles_raycast``: a synthetic 2D lidar built from the game's
  physics ray ``castRayStatic(origin, dir, dist)`` - the same call the
  game's own AI traffic uses for obstacle detection.  It returns a plain
  hit distance, so rays are cast in two horizontal circles around the
  ego vehicle: a mid fan at 1 m (walls, tree trunks, poles, barriers)
  and a low fan at 0.45 m over the near field (rocks, stumps, bushes,
  curbs).  Dynamic vehicles are NOT hit by this physics ray, so they are
  picked up by ``scan_obstacles_vehicles`` instead; ``scan_obstacles_all``
  merges both sources.

``scan_obstacles_all`` merges both sources into a single obstacle list.
The local planner turns those boxes into an avoidance path and a speed
limit.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

_LUA_CHUNK = r"""
local ego = %(ego)s
local radius = %(radius)s
local ex, ey = %(ex)s, %(ey)s
local out = {}
local objs = scenario_objects and scenario_objects.getObjects()
if objs then
  for i = 1, #objs do
    local o = objs[i]
    if o and o.id ~= ego and o.pos and o.size then
      local dx = o.pos.x - ex
      local dy = o.pos.y - ey
      if dx * dx + dy * dy < radius * radius then
        out[#out + 1] = {
          x = o.pos.x, y = o.pos.y, z = o.pos.z,
          sx = o.size.x, sy = o.size.y, sz = o.size.z,
        }
      end
    end
  end
end
return jsonEncode(out)
"""

_RAYCAST_LUA = r"""
local ox, oy, oz = %(ox)s, %(oy)s, %(oz)s
local radius = %(radius)s
local n = %(n)s
local rise_h = %(rise_h)s
local rise_low = %(rise_low)s
local low_h = %(low_h)s
local low_r = %(low_r)s
local low_n = %(low_n)s
local out = {}
local two_pi = 2.0 * math.pi

local function scan_engine(h, r, cnt, tag)
  local rh = rise_h
  if tag == "low" then rh = rise_low end
  for i = 0, cnt - 1 do
    local a = (i / cnt) * two_pi
    local c, s = math.cos(a), math.sin(a)
    -- Engine.castRay(origin, target, includeTerrain, renderGeometry):
    -- absolute-target form returns a table with a world-space hit point.
    -- It also sees TSStatic objects that the global castRayStatic helper
    -- misses (newly spawned walls / barriers), while still ignoring
    -- dynamic vehicles (those come from the vehicles source).
    local o = vec3(ox, oy, oz + h)
    local t = vec3(ox + c * r, oy + s * r, oz + h)
    local res = Engine.castRay(o, t, true, false)
    if res and res.pt then
      local dist = math.sqrt((res.pt.x - ox)^2 + (res.pt.y - oy)^2)
      if dist < r then
        -- A second ray a few metres higher in the same direction tells a
        -- ground / slope hit (which it simply clears) apart from a real
        -- vertical obstacle such as a wall, pole or tree trunk (which it
        -- hits at roughly the same distance).
        local uo = vec3(ox, oy, oz + h + rh)
        local ut = vec3(ox + c * r, oy + s * r, oz + h + rh)
        local ur = Engine.castRay(uo, ut, true, false)
        local up = r + 1
        if ur and ur.pt then
          up = math.sqrt((ur.pt.x - ox)^2 + (ur.pt.y - oy)^2)
        end
        out[#out + 1] = { x = res.pt.x, y = res.pt.y,
                          z = res.pt.z, d = dist, up = up, fan = tag }
      end
    end
  end
end

-- Mid fan at obstacle mid-height over the full radius: walls, trunks,
-- poles, barriers.  Engine.castRay is used here because castRayStatic
-- does not reliably collide with TSStatic objects spawned at runtime.
scan_engine(1.0, radius, n, "mid")
-- Low fan just above the grass over the near field: rocks, stumps,
-- bushes and curbs that the mid fan passes right over.
scan_engine(low_h, low_r, low_n, "low")
return jsonEncode(out)
"""

_VEHICLES_LUA = r"""
local ego = %(ego)s
local ex, ey = %(ex)s, %(ey)s
local radius = %(radius)s
local out = {}
invalidateVehicleCache()
for _, veh in ipairs(getAllVehicles()) do
  local ok, id = pcall(function() return veh:getId() end)
  if ok and id ~= nil and tostring(id) ~= ego then
    local okp, pos = pcall(function() return veh:getPosition() end)
    if okp and pos then
      local dx = pos.x - ex
      local dy = pos.y - ey
      if dx * dx + dy * dy < radius * radius then
        -- Vehicle heading so the planner gets a correctly oriented
        -- footprint: a car parked along the road is a slim box, not a
        -- 9 m wide wall that blocks every detour.  getHeadingVector is
        -- not present on this Tech build's Vehicle objects, so the
        -- direction vector is used (same atan2 world-heading convention).
        local yaw = 0.0
        local okh, hv = pcall(function() return veh:getDirectionVector() end)
        if okh and hv then
          yaw = math.atan2(hv.y, hv.x)
        end
        -- Velocity for ACC / overtaking: the world-space speed of the
        -- other vehicle lets us keep a time gap instead of treating every
        -- car as a static obstacle.
        local vx, vy = 0.0, 0.0
        local okv, vel = pcall(function() return veh:getVelocity() end)
        if okv and vel then
          vx = vel.x or 0.0
          vy = vel.y or 0.0
        end
        out[#out + 1] = { x = pos.x, y = pos.y, yaw = yaw,
                          id = tostring(id), vx = vx, vy = vy }
      end
    end
  end
end
return jsonEncode(out)
"""

# Health flags for the UI: one entry per sensor source, set to a message when
# that source's last scan failed (Lua error / bad response / comms problem).
# ``None`` per source means that source's last scan was clean.  The autopilot
# loop surfaces this so the user can tell "nothing detected" apart from "the
# sensor itself is down".  "vision" is informational (the YOLO channel is
# optional); it is not part of errors_active() because the core three sources
# already keep the car safe when the camera channel is unavailable.
last_error: dict[str, str | None] = {
    "scenario": None,
    "vehicles": None,
    "raycast": None,
    "vision": None,
}


def errors_active() -> bool:
    """True when any sensor source currently reports a problem."""
    return any(msg is not None for msg in last_error.values())


def errors_summary() -> str:
    """Human-readable list of the failing sensor sources, "" when clean."""
    parts = [f"{src}: {msg}" for src, msg in last_error.items()
             if msg is not None]
    return "; ".join(parts)


@dataclass
class Obstacle:
    """Axis-aligned obstacle box in world coordinates (x, y plane)."""

    x: float
    y: float
    half_w: float  # half bbox width along world x
    half_h: float  # half bbox height along world y
    category: str = "object"
    label: str = ""  # human-readable kind (e.g. "car") for the HUD
    # Optional oriented footprint for raycast walls.  ``axis`` points along
    # the obstacle's principal axis; ``half_len`` / ``half_thick`` are the
    # half extents in that frame.  When set, the planner uses this footprint
    # instead of the world-aligned box, which is what prevents a diagonal
    # roadside wall from looking like a huge square that blocks the road.
    axis: np.ndarray | None = None
    half_len: float = 0.0
    half_thick: float = 0.0
    # Optional dynamic-vehicle state for ACC / overtaking.  ``velocity`` is
    # the world velocity (m/s), ``heading`` the yaw (radians) and
    # ``vehicle_id`` a stable id for tracking across frames.
    velocity: np.ndarray | None = None
    heading: float | None = None
    vehicle_id: str | None = None

    @property
    def center(self) -> np.ndarray:
        return np.array([self.x, self.y])


def _lua_str(value: object) -> str:
    s = str(value).replace("\\", "\\\\").replace("'", "\\'")
    return "'" + s + "'"


def _oriented_dims(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Full extent of a point set along its principal axes.

    Walls are often rotated relative to the world axes; judging their
    thickness from the world-aligned bounding box makes every diagonal
    wall look like a wide area cloud and gets it split into fragments.
    This returns (long axis, short axis) lengths instead.
    """
    if not points:
        return 0.0, 0.0
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return 0.0, 0.0
    center = pts.mean(axis=0)
    cov = np.cov(pts, rowvar=False, bias=True)
    if cov.shape == (2, 2):
        vals, vecs = np.linalg.eigh(cov)
        axis = vecs[:, int(np.argmax(vals))]
    else:
        axis = np.array([1.0, 0.0])
    along = np.dot(pts - center, axis)
    across = np.dot(pts - center, [-axis[1], axis[0]])
    major = float(np.max(along) - np.min(along))
    minor = float(np.max(across) - np.min(across))
    return (major, minor) if major >= minor else (minor, major)


def _principal_axis(points: list[tuple[float, float]]) -> np.ndarray:
    """Unit vector along the longest axis of a 2D point set."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return np.array([1.0, 0.0])
    center = pts.mean(axis=0)
    cov = np.cov(pts, rowvar=False, bias=True)
    if cov.shape == (2, 2):
        vals, vecs = np.linalg.eigh(cov)
        return vecs[:, int(np.argmax(vals))]
    return np.array([1.0, 0.0])


def _cluster_points(pts, cell: float = 2.0, min_size: float = 0.9,
                    max_dim: float = 6.0, max_len: float = 14.0,
                    thin: float = 2.5, _depth: int = 0,
                    split_walls: bool = False, category: str = "raycast"):
    """Cluster 2D hit points into connected components -> obstacle boxes.

    Points are quantized onto a ``cell``-meter grid and connected with an
    8-neighbourhood flood fill, so a continuous wall or building side
    becomes one elongated obstacle instead of a cloud of points.

    A component whose bounding box grows large in *both* axes is a chain
    of scattered hits (a row of trees / poles), not a solid wall: two
    nearby trunks merge, then the merged box touches the next one, and so
    on until one giant box covers a whole grove and falsely blocks the
    road.  Such area clouds are re-clustered with a finer grid so each
    real object keeps its own small box; long thin walls stay intact.
    """
    grid: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for x, y in pts:
        key = (int(math.floor(x / cell)), int(math.floor(y / cell)))
        grid.setdefault(key, []).append((float(x), float(y)))
    seen: set[tuple[int, int]] = set()
    out: list[Obstacle] = []
    for key in grid:
        if key in seen:
            continue
        stack = [key]
        seen.add(key)
        comp: list[tuple[float, float]] = []
        while stack:
            k = stack.pop()
            comp.extend(grid[k])
            kx, ky = k
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nk = (kx + dx, ky + dy)
                    if nk in grid and nk not in seen:
                        seen.add(nk)
                        stack.append(nk)
        xs = [p[0] for p in comp]
        ys = [p[1] for p in comp]
        hw = max(min_size, (max(xs) - min(xs)) / 2.0)
        hh = max(min_size, (max(ys) - min(ys)) / 2.0)
        major, minor = _oriented_dims(comp)
        area_cloud = major > 2.0 * max_dim and minor > 2.0 * max_dim
        sparse_chain = False
        if major > max_len and minor < thin:
            # Long and thin: a real wall is hit every 0.7-2 m (dense), a
            # row of scattered trees has 3 m+ gaps between hits.  Only
            # the sparse chain is a clustering artefact worth splitting.
            along = sorted(xs) if hw >= hh else sorted(ys)
            gaps = [b - a for a, b in zip(along, along[1:])]
            mean_gap = (sum(gaps) / len(gaps)) if gaps else 0.0
            sparse_chain = mean_gap > 2.0
        if (area_cloud or sparse_chain) and _depth < 4 and cell > 0.5:
            # Scattered cloud / sparse chain: split it with a finer grid,
            # each sub-cluster passes the same size check (recursively).
            out.extend(_cluster_points(
                comp, cell=max(0.5, cell / 2.0), min_size=min_size,
                max_dim=max_dim, max_len=max_len, thin=thin,
                _depth=_depth + 1, split_walls=split_walls,
                category=category))
            continue
        cx = (min(xs) + max(xs)) / 2.0
        cy = (min(ys) + max(ys)) / 2.0
        label = ""
        if major > 4.0 and minor < 2.5:
            label = "wall"
        if (split_walls and _depth < 6 and major > 4.0 and minor < 12.0):
            # A wall seen around a bend is still a single angular sector,
            # but its world-axis-aligned box can balloon across the road.
            # Split it along its own principal axis into short dense
            # segments; each segment keeps the wall label while the box no
            # longer pretends a curved wall fills the whole bend.
            axis = _principal_axis(comp)
            proj = [(float((p[0] - cx) * axis[0] + (p[1] - cy) * axis[1]),
                     (float(p[0]), float(p[1]))) for p in comp]
            proj.sort(key=lambda item: item[0])
            chunks: list[list[tuple[float, float]]] = []
            cur: list[tuple[float, float]] = []
            start_proj: float | None = None
            for pr, point in proj:
                if cur and pr - start_proj > 7.0:
                    chunks.append(cur)
                    cur = []
                    start_proj = None
                if not cur:
                    start_proj = pr
                cur.append(point)
            if cur:
                chunks.append(cur)
            if len(chunks) > 1:
                for chunk in chunks:
                    out.extend(_cluster_points(
                        chunk, cell=cell, min_size=min_size,
                        max_dim=max_dim, max_len=max_len, thin=thin,
                        _depth=_depth + 1, split_walls=split_walls,
                        category=category))
                continue
            # A bend can also make one component wide instead of long:
            # two nearly parallel wall faces get fused into a short fat
            # box whose inner edge sits closer to the route than either
            # real wall.  Split the component at the point furthest from
            # its principal axis; the two halves become separate thin
            # wall segments that match the actual surfaces.
            if minor >= 2.5 and len(comp) >= 4:
                pairs = sorted(
                    (float((p[0] - cx) * axis[0] + (p[1] - cy) * axis[1]),
                     float((p[0] - cx) * -axis[1] + (p[1] - cy) * axis[0]),
                     (float(p[0]), float(p[1])))
                    for p in comp)
                max_i = max(range(len(pairs)),
                            key=lambda i: abs(pairs[i][1]))
                if abs(pairs[max_i][1]) >= 2.0 \
                        and 1 <= max_i <= len(pairs) - 2:
                    left = [p for _, _, p in pairs[:max_i + 1]]
                    right = [p for _, _, p in pairs[max_i + 1:]]
                    if len(left) >= 2 and len(right) >= 2:
                        for chunk in (left, right):
                            out.extend(_cluster_points(
                                chunk, cell=cell, min_size=min_size,
                                max_dim=max_dim, max_len=max_len,
                                thin=thin, _depth=_depth + 1,
                                split_walls=split_walls,
                                category=category))
                        continue
        out.append(Obstacle(x=cx, y=cy, half_w=hw, half_h=hh,
                            category=category, label=label,
                            axis=_principal_axis(comp),
                            half_len=max(0.0, major / 2.0),
                            half_thick=max(0.0, minor / 2.0)))
    return out


def _split_raycast_sectors(pts, origin, max_gap_deg: float = 30.0):
    """Split lidar hits into angular sectors seen from the sensor origin.

    A full 360-degree fan may hit several different walls around the ego.
    Flood-filling those hits on one world-space grid can chain them into a
    single giant box that appears to cross the road, so each raycast scan
    is first split at large angular gaps.  A continuous wall produces a
    dense sequence of neighbouring rays and stays in one sector.
    """
    ox, oy = float(origin[0]), float(origin[1])
    if len(pts) <= 2:
        return [list(pts)]
    items = []
    for x, y in pts:
        ang = math.degrees(math.atan2(y - oy, x - ox)) % 360.0
        items.append((ang, (float(x), float(y))))
    items.sort(key=lambda item: item[0])
    n = len(items)
    gaps = [
        items[(i + 1) % n][0] - items[i][0] if i < n - 1
        else items[0][0] + 360.0 - items[i][0]
        for i in range(n)
    ]
    # Rotate the ring so the largest gap sits between the last and first
    # point; every remaining gap is then a normal forward interval.
    largest = max(range(n), key=lambda i: gaps[i])
    start = (largest + 1) % n
    rotated = items[start:] + items[:start]
    sectors: list[list[tuple[float, float]]] = []
    cur = [rotated[0][1]]
    for i in range(1, n):
        if rotated[i][0] - rotated[i - 1][0] > max_gap_deg:
            sectors.append(cur)
            cur = []
        cur.append(rotated[i][1])
    sectors.append(cur)
    return sectors


# ---- Tech LiDAR obstacle pipeline -------------------------------------
#
# The BeamNG.tech LiDAR is a dense 360-degree cloud (100k+ hits per poll)
# that includes the road surface, roadside terrain and the ego vehicle
# itself.  It cannot be clustered as-is: ground/terrain points chain into
# one giant false wall around the car.  The pipeline below is therefore:
#
#   finite -> range window -> voxel downsample -> self-footprint removal
#   -> local ground removal (per angular sector x range ring, so slopes
#      keep working) -> angular sector split -> flood-fill clustering.
#
# The result are ``category="lidar"`` boxes that merge with the Lua
# scenario/vehicle/raycast sources (and vision) through merge_obstacles().

LIDAR_MIN_DIST_M = 2.5
LIDAR_VOXEL_M = 0.5
LIDAR_MAX_POINTS = 6000
LIDAR_OBSTACLE_VOXEL_M = 0.75
LIDAR_OBSTACLE_MAX_POINTS = 4000
LIDAR_GROUND_CLEARANCE_M = 0.35
LIDAR_MAX_HEIGHT_M = 4.5
LIDAR_GROUND_ANG_CELLS = 72      # angular sectors for the local ground ref
LIDAR_GROUND_RING_M = 5.0        # range rings for the local ground ref
LIDAR_Z_SPAN_M = 15.0            # drop absurd outliers before anything else


def downsample_cloud(points: np.ndarray,
                     max_points: int = LIDAR_MAX_POINTS,
                     voxel: float = LIDAR_VOXEL_M) -> np.ndarray:
    """Cap a dense 360 LiDAR cloud with a voxel grid + deterministic stride.

    BeamNG.tech can return 150k+ hits in one poll; feeding all of them to
    the 2 m flood-fill turns roadside walls around the ego into one giant
    connected box.  A ``voxel``-meter 2D grid keeps one representative hit
    per cell, then a deterministic stride enforces the final cap.  The
    grid keys are packed into one int64 (32 bits per axis) so the unique
    pass stays fast on 200k+ point clouds.
    """
    if len(points) <= max_points:
        return points
    kx = np.floor(points[:, 0] / voxel).astype(np.int64)
    ky = np.floor(points[:, 1] / voxel).astype(np.int64)
    keys = (kx << 32) | (ky & 0xFFFFFFFF)
    _, idx = np.unique(keys, return_index=True)
    sampled = points[idx]
    if len(sampled) > max_points:
        step = int(np.ceil(len(sampled) / max_points))
        sampled = sampled[::step][:max_points]
    return sampled


def _local_ground_z(cloud: np.ndarray, ox: float, oy: float,
                    n_ang: int = LIDAR_GROUND_ANG_CELLS,
                    ring: float = LIDAR_GROUND_RING_M) -> np.ndarray:
    """Per-point local ground reference from sector x range ring percentiles.

    A single global ground height fails on slopes (the road ahead can be
    several meters below the ego).  Points are binned by direction (72
    cells) and range (5 m rings); the 5th percentile of z inside each bin
    is that bin's ground reference, which follows hills and valleys.
    Returns one ground-z value per input point.
    """
    ang = np.degrees(np.arctan2(cloud[:, 1] - oy, cloud[:, 0] - ox)) % 360.0
    ai = np.clip((ang / (360.0 / n_ang)).astype(np.int64), 0, n_ang - 1)
    dist = np.hypot(cloud[:, 0] - ox, cloud[:, 1] - oy)
    ri = np.clip((dist / ring).astype(np.int64), 0, 100000)
    key = ai * 1000003 + ri
    order = np.argsort(key, kind="stable")
    ks = key[order]
    zs = cloud[order, 2]
    starts = np.r_[0, np.flatnonzero(np.diff(ks)) + 1]
    ends = np.r_[starts[1:], len(ks)]
    ground = np.empty(len(ks), dtype=float)
    for s, e in zip(starts, ends):
        ground[s:e] = np.percentile(zs[s:e], 5.0)
    out = np.empty(len(cloud), dtype=float)
    out[order] = ground
    return out


def lidar_obstacles(cloud: np.ndarray, pos, radius: float = 45.0,
                    self_rect: tuple[float, float, float] | None = None,
                    ground_clearance: float = LIDAR_GROUND_CLEARANCE_M,
                    max_height: float = LIDAR_MAX_HEIGHT_M,
                    max_points: int = LIDAR_OBSTACLE_MAX_POINTS,
                    voxel: float = LIDAR_OBSTACLE_VOXEL_M) -> list[Obstacle]:
    """Cluster a 360 LiDAR cloud into ``category="lidar"`` obstacle boxes.

    ``pos`` is the ego world position; ``self_rect`` is
    ``(half_len, half_w, heading)`` of the ego footprint (with margin) used
    to remove self-hits.  Returns obstacle boxes in world coordinates; the
    caller merges them with the Lua scenario/vehicle/raycast sources.
    """
    pts = np.asarray(cloud, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 3 or len(pts) < 4:
        return []
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 4:
        return []
    ox, oy, oz = float(pos[0]), float(pos[1]), float(pos[2])
    dist = np.hypot(pts[:, 0] - ox, pts[:, 1] - oy)
    keep = ((dist >= LIDAR_MIN_DIST_M) & (dist <= radius)
            & (np.abs(pts[:, 2] - oz) <= LIDAR_Z_SPAN_M))
    pts = pts[keep]
    if len(pts) < 4:
        return []
    pts = downsample_cloud(pts, max_points=max_points, voxel=voxel)
    if self_rect is not None:
        half_len, half_w, heading = self_rect
        uf = np.array([np.cos(heading), np.sin(heading)])
        ur = np.array([-uf[1], uf[0]])
        local = pts[:, :2] - np.array([ox, oy])
        on_car = ((np.abs(local @ uf) <= half_len)
                  & (np.abs(local @ ur) <= half_w))
        pts = pts[~on_car]
    if len(pts) < 4:
        return []
    ground = _local_ground_z(pts, ox, oy)
    keep = ((pts[:, 2] - ground >= ground_clearance)
            & (pts[:, 2] - ground <= max_height))
    pts2d = [(float(x), float(y)) for x, y, _ in pts[keep]]
    if len(pts2d) < 2:
        return []
    out: list[Obstacle] = []
    for sector in _split_raycast_sectors(pts2d, (ox, oy)):
        # Coarser flood-fill cell than the raycast fan: the lidar cloud is
        # dense, so neighbouring poles/trunks/fence posts merge into one
        # box instead of dozens of 0.9 m specks that clutter the planner.
        out.extend(_cluster_points(sector, cell=3.0, split_walls=True,
                                   category="lidar"))
    return out


class LidarClusterTracker:
    """Tracks lidar cluster centroids across polls to estimate velocity.

    Cluster centroids jitter every poll (voxel resampling shifts which
    points land in each cell), so a velocity is only reported once the
    cluster has been matched for ``min_matches`` polls, the smoothed speed
    stays below ``max_speed`` (a "teleporting" centroid is a new object,
    not a 60 m/s vehicle) and above ``min_speed`` (real traffic moves at
    least a few m/s; anything slower is centroid jitter on a static
    object).  Static clusters therefore keep ``velocity=None``; unmatched
    / stale tracks are dropped.
    """

    def __init__(self, match_m: float = 3.0, ttl_s: float = 2.0,
                 min_matches: int = 2, max_speed: float = 35.0,
                 min_speed: float = 2.0) -> None:
        self.match_m = match_m
        self.ttl_s = ttl_s
        self.min_matches = min_matches
        self.max_speed = max_speed
        self.min_speed = min_speed
        self._tracks: list[dict] = []

    def update(self, boxes: list[Obstacle], t: float) -> None:
        """Attach ``velocity`` / ``heading`` / ``vehicle_id`` to boxes."""
        if not boxes:
            self._tracks = []
            return
        for tr in self._tracks:
            tr["matched"] = False
        for i, ob in enumerate(boxes):
            best = None
            best_d = self.match_m
            for tr in self._tracks:
                d = math.hypot(ob.x - tr["x"], ob.y - tr["y"])
                if d < best_d:
                    best_d = d
                    best = tr
            if best is not None and not best["matched"]:
                dt = t - best["t"]
                best["matched"] = True
                best["t"] = t
                # epsilon: 100.1 - 100.0 is 0.0999... in binary floats
                if 0.1 - 1e-6 <= dt <= 1.0:
                    vx = (ob.x - best["x"]) / dt
                    vy = (ob.y - best["y"]) / dt
                    k = 0.5
                    best["vx"] = (1.0 - k) * best["vx"] + k * vx
                    best["vy"] = (1.0 - k) * best["vy"] + k * vy
                best["x"], best["y"] = ob.x, ob.y
                best["matches"] += 1
                if (best["matches"] >= self.min_matches
                        and self.min_speed
                        <= math.hypot(best["vx"], best["vy"])
                        <= self.max_speed):
                    ob.velocity = np.array([best["vx"], best["vy"]],
                                           dtype=float)
                    ob.heading = float(math.atan2(best["vy"], best["vx"]))
                    ob.vehicle_id = f"lidar-{best['id']}"
            else:
                self._tracks.append({
                    "id": len(self._tracks),
                    "x": ob.x, "y": ob.y, "t": t,
                    "vx": 0.0, "vy": 0.0, "matches": 1, "matched": True,
                })
        self._tracks = [tr for tr in self._tracks
                        if tr["matched"] and t - tr["t"] <= self.ttl_s]


def scan_obstacles_raycast(
    bng,
    pos,
    radius: float = 55.0,
    rays: int = 90,
    hit_height: float = 1.0,
    min_dist: float = 2.5,
    z_tol: float = 4.0,
    cluster_cell: float = 2.0,
    rise_probe: float = 1.8,
    rise_low: float = 0.45,
    rise_tol: float = 2.0,
    near_keep_mid: float = 10.0,
    low_height: float = 0.45,
    low_radius: float = 35.0,
    low_rays: int = 90,
    low_near_keep: float = 20.0,
    return_hits: bool = False,
) -> list[Obstacle]:
    """Synthetic 2D lidar: horizontal castRayStatic fans around the ego.

    Two concentric fans share one Lua round-trip: a mid fan at
    ``hit_height`` (1 m) over the full radius catches walls, tree trunks,
    poles and barriers, and a low fan at ``low_height`` (0.45 m) over the
    near field catches rocks, stumps, bushes and curbs that the mid fan
    passes right over.  Flat ground is not hit by a horizontal ray;
    terrain rises are recognised by a second, higher probe ray, while a
    real obstacle is hit at roughly the same distance by both rays.
    Dynamic vehicles are not returned by the physics ray; use
    ``scan_obstacles_vehicles`` for those.  Hits are filtered (own car,
    implausible heights), clustered into connected components and
    returned as obstacle boxes in world coordinates.

    With ``return_hits=True`` the raw filtered (x, y) hit points are also
    returned as ``(obstacles, hits)`` so callers can build a free-space
    corridor from the same raycast fan without issuing a second scan.
    """
    global last_error
    p = np.asarray(pos, dtype=float)
    chunk = _RAYCAST_LUA % {
        "ox": float(p[0]),
        "oy": float(p[1]),
        "oz": float(p[2]),
        "radius": float(radius),
        "n": int(rays),
        "rise_h": float(rise_probe),
        "rise_low": float(rise_low),
        "low_h": float(low_height),
        "low_r": float(low_radius),
        "low_n": int(low_rays),
    }
    try:
        resp = bng.queue_lua_command(chunk, response=True)
    except Exception as exc:
        # NOTE: bare except kept — Lua command can fail with any
        # transport error; we return empty results and record the error.
        last_error["raycast"] = str(exc)
        logger.warning("[perception] raycast scan failed: %s", exc)
        return [] if not return_hits else ([], [])
    if not resp:
        last_error["raycast"] = "empty response"
        return [] if not return_hits else ([], [])
    try:
        data = json.loads(str(resp))
    except (ValueError, TypeError):
        last_error["raycast"] = "bad response"
        logger.warning("[perception] raycast scan: bad response")
        return [] if not return_hits else ([], [])
    last_error["raycast"] = None
    pts: list[tuple[float, float]] = []
    for entry in data or []:
        try:
            x = float(entry["x"])
            y = float(entry["y"])
            z = float(entry["z"])
        except (KeyError, TypeError, ValueError):
            continue
        if not all(np.isfinite([x, y, z])):
            continue
        hdist = math.hypot(x - p[0], y - p[1])
        if hdist < min_dist:
            continue  # own car / very close clutter
        if abs(z - p[2]) > z_tol:
            continue  # bridges far above or tunnels below
        fan = str(entry.get("fan") or "mid")
        fan_r = float(low_radius if fan == "low" else radius)
        near_keep = low_near_keep if fan == "low" else near_keep_mid
        # Terrain slopes and ground rises are not obstacles: the probe ray
        # a few metres higher clears them, while a real wall / trunk / pole
        # is hit at about the same distance.  A probe that clears the hit
        # (no contact or contact far beyond) means the first ray hit the
        # ground, so the point is dropped - except close in, where a low
        # rock / stump / curb lets the probe pass right over it and the
        # hit is exactly what we want to keep.
        try:
            up = None if entry.get("up") is None else float(entry["up"])
        except (TypeError, ValueError):
            up = None
        if up is None or up > fan_r:
            if hdist > near_keep:
                continue
        elif up > hdist + float(rise_tol):
            continue
        pts.append((x, y))
    if not pts:
        return [] if not return_hits else ([], [])
    out: list[Obstacle] = []
    for sector in _split_raycast_sectors(pts, p[:2]):
        out.extend(_cluster_points(sector, cell=cluster_cell,
                                   split_walls=True))
    if return_hits:
        return out, pts
    return out


def scan_obstacles_all(
    bng,
    ego_vid: str | None,
    pos,
    radius: float = 55.0,
    use_raycast: bool = True,
    use_scenario: bool = True,
    use_vehicles: bool = True,
    return_hits: bool = False,
) -> list[Obstacle]:
    """Merge scenario-object boxes, live vehicles and raycast hits.

    With ``return_hits=True`` the raw raycast hit points are returned as
    ``(obstacles, hits)``; the default behaviour is unchanged.
    """
    out: list[Obstacle] = []
    hits: list[tuple[float, float]] = []
    if use_scenario:
        out.extend(scan_obstacles(bng, ego_vid, pos, radius=radius))
    if use_vehicles:
        out.extend(scan_obstacles_vehicles(bng, ego_vid, pos, radius=radius))
    if use_raycast:
        if return_hits:
            ray_obs, ray_hits = scan_obstacles_raycast(
                bng, pos, radius=radius, return_hits=True)
            out.extend(ray_obs)
            hits.extend(ray_hits)
        else:
            out.extend(scan_obstacles_raycast(bng, pos, radius=radius))
    if return_hits:
        return out, hits
    return out


def merge_obstacles(obstacles, merge_dist: float = 2.5) -> list[Obstacle]:
    """Merge boxes from different sensors that cover the same spot.

    The same vehicle is typically reported by both ``getAllVehicles()`` and
    the vision detector / LiDAR clusterer at nearly the same position;
    without merging the planner would see a double box.  Only compact box
    sources (scenario / vehicle / vision / lidar) are merged with each
    other.  Raycast walls keep their footprint so a car standing next to a
    wall does not grow the wall box - but a LiDAR wall that sees the *same*
    wall as the raycast fan (both ``label="wall"``) is merged so the two
    channels do not double the wall.
    """
    compact = {"scenario", "vehicle", "vision", "lidar"}
    merged: list[Obstacle] = []
    for ob in obstacles:
        target = None
        for prev in merged:
            same_wall = (prev.label == "wall" and ob.label == "wall")
            if (prev.category not in compact
                    or ob.category not in compact) and not same_wall:
                continue
            if math.hypot(ob.x - prev.x, ob.y - prev.y) <= merge_dist:
                target = prev
                break
        if target is None:
            merged.append(ob)
            continue
        x0 = min(target.x - target.half_w, ob.x - ob.half_w)
        x1 = max(target.x + target.half_w, ob.x + ob.half_w)
        y0 = min(target.y - target.half_h, ob.y - ob.half_h)
        y1 = max(target.y + target.half_h, ob.y + ob.half_h)
        target.x = (x0 + x1) / 2.0
        target.y = (y0 + y1) / 2.0
        target.half_w = (x1 - x0) / 2.0
        target.half_h = (y1 - y0) / 2.0
        if not target.label and ob.label:
            target.label = ob.label
        # Keep the dynamic-vehicle state (velocity / heading / stable id)
        # when a richer source (scene vehicle scan) merges into a
        # vision-only box, so ACC still sees a moving lead after merging.
        if target.velocity is None and ob.velocity is not None:
            target.velocity = ob.velocity
            target.heading = ob.heading
            target.vehicle_id = ob.vehicle_id
        # Keep the oriented footprint (axis + extents) when a scene vehicle
        # merges into a lidar/vision box: without it the planner sees the
        # world-aligned AABB of a diagonally parked car, which can span the
        # whole road and block every detour.
        if target.axis is None and ob.axis is not None:
            target.axis = ob.axis
            target.half_len = ob.half_len
            target.half_thick = ob.half_thick
    return merged


def filter_self_overlap(obstacles, pos, margin: float = 0.5,
                        categories=("vision",)) -> list[Obstacle]:
    """Drop obstacles whose footprint contains the ego position.

    The scenario and vehicle sources exclude the own car by id, but the
    vision channel cannot: a chase cam that shows the player's car gets
    back-projected right onto the ego itself, so the planner thinks the
    lane is blocked by a wall sitting on top of the car.  Any obstacle
    whose box covers the ego centre (within ``margin``) is therefore
    treated as a self-detection ghost and removed.  Only boxes that
    actually sit on top of the car are dropped: a real vehicle that is
    merely close (even 1-2 m away) is kept, because a car right in front
    is exactly what the planner must react to.
    """
    px, py = float(pos[0]), float(pos[1])
    out: list[Obstacle] = []
    for ob in obstacles:
        if ob.category in categories and (
                abs(ob.x - px) <= ob.half_w + margin
                and abs(ob.y - py) <= ob.half_h + margin):
            continue
        out.append(ob)
    return out


def drop_vision_waypoint_ghosts(obstacles, anchors, margin: float = 2.0):
    """Drop vision detections sitting on the route start/goal markers.

    The yellow start waypoint sphere and the red goal sphere are drawn in
    the 3D world as navigation cues; YOLO often reads them as "person" and
    back-projects a blocker right onto an empty road, so the planner pins
    the speed and the car stutters.  Only ``vision`` boxes whose centre is
    within ``margin`` of a route marker (start, current, goal) are removed;
    everything else is untouched.
    """
    pts = []
    for a in anchors or []:
        try:
            v = np.asarray(a, dtype=float)[:2]
        except (TypeError, ValueError):
            continue
        if len(v) == 2 and all(np.isfinite(v)):
            pts.append(v)
    if not pts:
        return obstacles
    out = []
    for ob in obstacles:
        if ob.category == "vision" and any(
                math.hypot(ob.x - ax, ob.y - ay) <= margin
                for ax, ay in pts):
            continue
        out.append(ob)
    return out


def scan_obstacles_vehicles(
    bng,
    ego_vid: str | None,
    pos,
    radius: float = 60.0,
    min_dist: float = 3.0,
    half_w: float = 2.2,
    half_h: float = 4.6,
) -> list[Obstacle]:
    """Enumerate live vehicles (parked or driving) as obstacle boxes.

    ``scenario_objects`` does not list vehicles on free-roam maps and the
    raycast fan can miss low or unevenly parked cars, so the vehicle
    registry is queried through the Lua bridge (``getAllVehicles()`` -
    works on the Steam build without a tech license, unlike beamngpy's
    tech-only ``UpdateScenario`` message).  The ego vehicle is skipped and
    everything else within ``radius`` becomes a box around its world
    position.  The vehicle heading is read through ``getHeadingVector()`` so
    the box follows the vehicle's true orientation (a parked car along the
    road stays a slim 2.2 x 4.6 m box instead of a huge axis-aligned block).
    """
    global last_error
    p = np.asarray(pos, dtype=float)
    chunk = _VEHICLES_LUA % {
        "ego": _lua_str(ego_vid or ""),
        "radius": float(radius),
        "ex": float(p[0]),
        "ey": float(p[1]),
    }
    try:
        resp = bng.queue_lua_command(chunk, response=True)
    except Exception as exc:
        # NOTE: bare except kept — Lua command can fail with any
        # transport error; we return empty results and record the error.
        last_error["vehicles"] = str(exc)
        logger.warning("[perception] vehicle scan failed: %s", exc)
        return []
    if not resp:
        last_error["vehicles"] = "empty response"
        return []
    try:
        data = json.loads(str(resp))
    except (ValueError, TypeError):
        last_error["vehicles"] = "bad response"
        logger.warning("[perception] vehicle scan: bad response")
        return []
    last_error["vehicles"] = None
    out: list[Obstacle] = []
    for entry in data or []:
        try:
            x = float(entry["x"])
            y = float(entry["y"])
        except (KeyError, TypeError, ValueError):
            continue
        if not all(np.isfinite([x, y])):
            continue
        dist = math.hypot(x - p[0], y - p[1])
        if dist < min_dist or dist > radius:
            continue
        try:
            yaw = float(entry.get("yaw", 0.0))
        except (TypeError, ValueError):
            yaw = 0.0
        try:
            vx = float(entry.get("vx", 0.0))
            vy = float(entry.get("vy", 0.0))
        except (TypeError, ValueError):
            vx = vy = 0.0
        vid = entry.get("id")
        vid = str(vid) if vid is not None else None
        ca, sa = abs(math.cos(yaw)), abs(math.sin(yaw))
        # World-axis-aligned bounding box of the oriented rectangle: the
        # vehicle's long side (2*half_h) points along its heading, the
        # short side (2*half_w) across it.
        hw = ca * half_h + sa * half_w
        hh = sa * half_h + ca * half_w
        vel = (np.array([vx, vy], dtype=float)
               if np.isfinite(vx) and np.isfinite(vy) else None)
        out.append(Obstacle(x=x, y=y, half_w=hw, half_h=hh,
                            category="vehicle", heading=yaw,
                            velocity=vel, vehicle_id=vid,
                            axis=np.array([math.cos(yaw), math.sin(yaw)]),
                            half_len=half_h / 2.0, half_thick=half_w / 2.0))
    return out


def scan_obstacles(
    bng,
    ego_vid: str | None,
    pos,
    radius: float = 60.0,
    min_footprint: float = 0.6,
    min_height: float = 0.5,
    min_dist: float = 1.5,
) -> list[Obstacle]:
    """Query nearby scenario objects and return them as obstacle boxes.

    ``ego_vid`` is the id of the ego vehicle (skipped in the query).
    Returns an empty list when the map has no object data, so autopilot
    degrades gracefully to pure route following.
    """
    global last_error
    p = np.asarray(pos, dtype=float)
    chunk = _LUA_CHUNK % {
        "ego": _lua_str(ego_vid or ""),
        "radius": float(radius),
        "ex": float(p[0]),
        "ey": float(p[1]),
    }
    try:
        resp = bng.queue_lua_command(chunk, response=True)
    except Exception as exc:
        # NOTE: bare except kept — Lua command can fail with any
        # transport error; we return empty results and record the error.
        last_error["scenario"] = str(exc)
        logger.warning("[perception] obstacle scan failed: %s", exc)
        return []
    if not resp:
        last_error["scenario"] = "empty response"
        return []
    try:
        data = json.loads(str(resp))
    except (ValueError, TypeError):
        last_error["scenario"] = "bad response"
        logger.warning("[perception] obstacle scan: bad response")
        return []
    last_error["scenario"] = None
    out: list[Obstacle] = []
    for entry in data or []:
        try:
            x = float(entry["x"])
            y = float(entry["y"])
            sx = float(entry["sx"])
            sy = float(entry["sy"])
            sz = float(entry["sz"])
        except (KeyError, TypeError, ValueError):
            continue
        if not all(np.isfinite([x, y, sx, sy, sz])):
            continue
        # Ignore objects sitting right under / on the ego (its own body on
        # maps that also register it as a scenario object, invisible ground
        # props): a box at the car's position would always look blocked.
        if math.hypot(x - p[0], y - p[1]) < min_dist:
            continue
        # Ignore flat ground props and dust; keep things we can actually hit.
        if max(sx, sy) < min_footprint or sz < min_height:
            continue
        out.append(Obstacle(x=x, y=y,
                            half_w=max(0.05, sx / 2.0),
                            half_h=max(0.05, sy / 2.0)))
    return out
