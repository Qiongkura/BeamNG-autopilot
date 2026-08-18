"""Camera overlay helpers for road-network vs painted lane diagnostics.

Renders a front-camera frame with the ego body extents, the nearest
road-network lane boundaries / lane centre, vision-detected markings, a
numeric summary panel and a compact lateral cross-section diagram.  Used
by the offline annotator and the live lane-state viewer.
"""

from __future__ import annotations

import logging
import math

import cv2
import numpy as np

logger = logging.getLogger(__name__)

BODY_COLOR = (255, 80, 255)
ROAD_COLOR = (0, 200, 255)
LANE_COLOR = (0, 255, 120)
CENTER_COLOR = (255, 255, 0)
MARK_WHITE = (230, 230, 230)
MARK_YELLOW = (0, 255, 255)
ROADNET_LANE_COLOR = (150, 150, 160)
LANE_FRAME_COLOR = (255, 140, 60)   # 感知配对车道边界（pair_lane_markings）
PANEL_BG = (0, 0, 0)
PANEL_TEXT = (255, 255, 255)
LAT_BG = (16, 18, 24)
LAT_AXIS = (110, 112, 120)

PAVEMENT_EDGE_CONF_MIN = 0.45
PAVEMENT_SIDE_CONF_MIN = 0.35
PAVEMENT_OFF_FRACTION = 0.5
PAVEMENT_OFF_CONFIRM = 2
PAVEMENT_MIN_EDGE_LAT_M = 0.9
PAVEMENT_CHROMA_THRESHOLD = 26.0
PAVEMENT_MAX_LAT_M = 7.0
PAVEMENT_LAT_STEP_M = 0.1
PAVEMENT_STATIONS_M = (2.5, 3.5, 5.0, 6.5, 8.0, 10.0, 12.5)
PAVEMENT_NEAR_MAX_M = 6.5
PAVEMENT_MIN_NEAR_SIDES = 3
PAVEMENT_MIN_WIDTH_M = 2.0
PAVEMENT_MAX_WIDTH_M = 7.5


def unit_fwd(state) -> np.ndarray:
    """Forward unit vector of a vehicle state in world x/y."""
    fwd = np.asarray(state.dir[:2], dtype=float)
    n = float(np.linalg.norm(fwd))
    if n > 1e-9:
        return fwd / n
    return np.array([math.cos(state.heading), math.sin(state.heading)])


def _lat_of(world_xy, pos, left) -> float:
    p = np.asarray(world_xy, dtype=float)[:2]
    return float((p - pos) @ left)


def _interp_edge(a, b, t):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return a + t * (b - a)


def road_lane_geometry(conn, pos, fwd) -> dict | None:
    """Nearest road-network lane geometry in the ego car frame.

    All distances are measured from the ego centre in metres; positive
    lateral offset is left of the vehicle.  The nearest DecalRoad edge
    pair is interpolated at the ego position and split into lanes using
    the road's lane counts.
    """
    pos = np.asarray(pos, dtype=float)[:2]
    with conn.io_lock:
        roads = conn.bng.scenario.get_road_network(
            include_edges=True, drivable_only=True)

    candidates: list[dict] = []
    for rid, meta in roads.items():
        if not isinstance(meta, dict):
            continue
        edges = meta.get("edges")
        if not isinstance(edges, list) or len(edges) < 2:
            continue
        mids = []
        for row in edges:
            m = row.get("middle")
            mids.append(None if m is None else np.asarray(m, dtype=float))
        candidates.append((rid, meta, mids, edges))

    best = None
    for rid, meta, mids, edges in candidates:
        for i in range(len(mids) - 1):
            a, b = mids[i], mids[i + 1]
            if a is None or b is None:
                continue
            ab = b[:2] - a[:2]
            d2 = float(ab @ ab)
            if d2 < 1e-9:
                continue
            t = float((pos - a[:2]) @ ab / d2)
            t = min(1.0, max(0.0, t))
            p = a[:2] + t * ab
            dist = float(np.hypot(*(p - pos)))
            if best is None or dist < best[0]:
                best = (dist, rid, meta, i, t, edges, mids)
    if best is None:
        return None

    _, rid, meta, i, t, edges, mids = best
    # 方向一致性检查：车头必须大致沿着最近路段方向行驶，否则"车辆前向
    # 的横向坐标"没有意义（路口中心/斜停时会把左边界算到车身上、车道
    # 宽变成 0.5m 之类的荒谬值）。不一致时返回 None，调用方显示
    # "off-direction" 而不是错误数值。
    a0, b0 = mids[i], mids[i + 1]
    edge_dir = (b0[:2] - a0[:2])[:2]
    en = float(np.linalg.norm(edge_dir))
    if en > 1e-9:
        edge_dir = edge_dir / en
        fwd2 = np.asarray(fwd[:2], dtype=float)
        fn = float(np.linalg.norm(fwd2))
        if fn > 1e-9:
            align = abs(float(edge_dir @ (fwd2 / fn)))
            if align < 0.5:  # 车头与道路方向夹角 > 60°
                return None
    row_a, row_b = edges[i], edges[i + 1]
    left_pt = _interp_edge(row_a["left"], row_b["left"], t)
    mid_pt = _interp_edge(row_a["middle"], row_b["middle"], t)
    right_pt = _interp_edge(row_a["right"], row_b["right"], t)

    left = np.array([-fwd[1], fwd[0]])
    left_lat = _lat_of(left_pt, pos, left)
    mid_lat = _lat_of(mid_pt, pos, left)
    right_lat = _lat_of(right_pt, pos, left)
    if left_lat < right_lat:
        left_lat, right_lat = right_lat, left_lat

    width = float(left_lat - right_lat)
    half = width / 2.0
    n_left = int(meta.get("lanesLeft") or 0)
    n_right = int(meta.get("lanesRight") or 0)
    car_rel = float(-mid_lat)

    if n_left <= 0 and n_right <= 0:
        total = 1
        lane_w = width
    elif n_left <= 0 or n_right <= 0:
        total = max(1, n_left + n_right)
        lane_w = width / total
    else:
        total = 1
        lane_w = width

    if n_left <= 0 or n_right <= 0:
        d_from_left = float(left_lat)
        k = int(math.floor(max(0.0, d_from_left) / lane_w))
        k = min(max(0, k), total - 1)
        lane_left_lat = left_lat - k * lane_w
        lane_right_lat = left_lat - (k + 1) * lane_w
    elif car_rel < 0.0:
        lane_w = half / n_right
        k = int(math.floor(-car_rel / lane_w)) if lane_w > 0 else 0
        k = min(max(0, k), n_right - 1)
        lane_left_lat = mid_lat - half + (k + 1) * lane_w
        lane_right_lat = mid_lat - half + k * lane_w
    else:
        lane_w = half / n_left
        k = int(math.floor(car_rel / lane_w)) if lane_w > 0 else 0
        k = min(max(0, k), n_left - 1)
        lane_left_lat = mid_lat + half - k * lane_w
        lane_right_lat = mid_lat + half - (k + 1) * lane_w

    lane_center_lat = 0.5 * (lane_left_lat + lane_right_lat)
    return {
        "road_id": rid,
        "lanes_left": n_left,
        "lanes_right": n_right,
        "road_width": round(width, 3),
        "road_center_lat": round(mid_lat, 3),
        "left_edge_lat": round(left_lat, 3),
        "right_edge_lat": round(right_lat, 3),
        "lane_left_lat": round(lane_left_lat, 3),
        "lane_right_lat": round(lane_right_lat, 3),
        "lane_center_lat": round(lane_center_lat, 3),
        "center_offset": round(-lane_center_lat, 3),
        "left_dist": round(lane_left_lat, 3),
        "right_dist": round(-lane_right_lat, 3),
        "lane_width": round(abs(lane_right_lat - lane_left_lat), 3),
    }


def ego_extents(conn) -> tuple[float, float]:
    """Approximate ego half-length / half-width from the BeamNG bbox."""
    try:
        with conn.io_lock:
            bbox = conn.vehicle.get_bbox()
        fl = np.asarray(bbox["front_bottom_left"], dtype=float)[:2]
        fr = np.asarray(bbox["front_bottom_right"], dtype=float)[:2]
        rl = np.asarray(bbox["rear_bottom_left"], dtype=float)[:2]
        rr = np.asarray(bbox["rear_bottom_right"], dtype=float)[:2]
        length = float(np.linalg.norm((fl + fr) / 2.0 - (rl + rr) / 2.0))
        width = float(np.linalg.norm((fr + rr) / 2.0 - (fl + rl) / 2.0))
        return max(0.5, length / 2.0), max(0.5, width / 2.0)
    except Exception:
        # NOTE: bare except kept — get_bbox can fail with any transport
        # error or missing keys; return conservative default extents.
        logger.debug("[lane_overlay] ego bbox query failed; using defaults")
        return 2.4, 1.0


def world_lat_line(pos, fwd, lat, s0: float, s1: float) -> np.ndarray:
    left = np.array([-fwd[1], fwd[0]])
    ss = np.linspace(s0, s1, 12)
    return pos + ss[:, None] * fwd + lat * left


def _road_chroma_center(frame_rgb) -> tuple[float, float]:
    """Median (R-G, G-B) chroma of the road ahead of the ego."""
    h, w = frame_rgb.shape[:2]
    y0, y1 = int(h * 0.35), int(h * 0.72)
    x0, x1 = int(w * 0.32), int(w * 0.68)
    if y1 <= y0 or x1 <= x0:
        return 0.0, 0.0
    roi = frame_rgb[y0:y1, x0:x1].astype(np.float32)
    r, g, b = roi[..., 0], roi[..., 1], roi[..., 2]
    return float(np.median(r - g)), float(np.median(g - b))


def _offroad_mask(frame_rgb) -> np.ndarray:
    """Boolean mask of grass / dirt / vegetation outside the paved edge.

    Asphalt stays close to gray (R=G=B), so its chroma distance from the
    road colour model is small even under shadows.  Grass and dirt have a
    strong green/brown chroma and are the only things promoted to off-road.
    """
    import cv2

    img = frame_rgb.astype(np.float32)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    c0, c1 = _road_chroma_center(frame_rgb)
    chroma = np.hypot((r - g) - c0, (g - b) - c1)
    off = chroma > PAVEMENT_CHROMA_THRESHOLD

    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    h = hsv[..., 0].astype(np.int16)
    s = hsv[..., 1].astype(np.int16)
    v = hsv[..., 2].astype(np.int16)
    grass = ((h >= 25) & (h <= 100) & (s >= 45) & (v >= 28) & (v <= 230))
    dirt = ((h >= 5) & (h <= 45) & (s >= 50) & (v >= 28) & (v <= 210))
    off |= grass | dirt

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    off = cv2.morphologyEx(off.astype(np.uint8), cv2.MORPH_CLOSE,
                           kernel).astype(bool)
    return off


def _pavement_off_fraction(mask, cam, pos, heading, fwd, left,
                           s_m: float, lat_m: float,
                           ground_z: float) -> float | None:
    """Off-road fraction of the image patch under a ground-plane sample."""
    p3 = np.array([
        pos[0] + s_m * fwd[0] + lat_m * left[0],
        pos[1] + s_m * fwd[1] + lat_m * left[1],
        ground_z,
    ])
    u, v, ok = cam.project(np.array([p3]), pos, heading)
    if not ok[0] or not np.isfinite(u[0]) or not np.isfinite(v[0]):
        return None
    uu = int(round(float(u[0])))
    vv = int(round(float(v[0])))
    h, w = mask.shape[:2]
    if uu < 5 or vv < 5 or uu >= w - 5 or vv >= h - 5:
        return None
    patch = mask[vv - 5:vv + 6, uu - 5:uu + 6]
    if patch.size == 0:
        return None
    return float(patch.mean())


def _side_pavement_lat(mask, cam, pos, heading, fwd, left,
                       s_m: float, side: float,
                       ground_z: float) -> float | None:
    """First stable pavement-to-off-road lateral transition on one side.

    A transition only counts when there are at least two road samples,
    then an off-road sample, and at least two off-road samples within the
    next few samples.  Isolated coloured patches (signs, car bodies,
    overlay marks) are ignored.
    """
    n = int(round(PAVEMENT_MAX_LAT_M / PAVEMENT_LAT_STEP_M))
    road_streak = 0
    last_road = None
    pending_off = None
    pending_i = -10
    for k in range(1, n + 1):
        lat = side * k * PAVEMENT_LAT_STEP_M
        frac = _pavement_off_fraction(
            mask, cam, pos, heading, fwd, left, s_m, lat, ground_z)
        if frac is None:
            break
        if frac < PAVEMENT_OFF_FRACTION:
            road_streak += 1
            last_road = lat
            pending_off = None
            pending_i = -10
        elif pending_off is None:
            pending_off = lat
            pending_i = k
        elif (road_streak >= 2 and last_road is not None
              and k - pending_i <= 4):
            after = 1
            for kk in range(k + 1, min(k + 4, n + 1)):
                lat2 = side * kk * PAVEMENT_LAT_STEP_M
                frac2 = _pavement_off_fraction(
                    mask, cam, pos, heading, fwd, left, s_m, lat2,
                    ground_z)
                if frac2 is None:
                    break
                if frac2 >= PAVEMENT_OFF_FRACTION:
                    after += 1
            if after >= PAVEMENT_OFF_CONFIRM:
                edge = 0.5 * (last_road + pending_off)
                if abs(edge) >= PAVEMENT_MIN_EDGE_LAT_M:
                    return float(edge)
            pending_off = lat
            pending_i = k
    return None


def estimate_pavement_edges(frame_rgb, cam, pos, heading,
                            ground_z: float | None = None,
                            offroad_mask: np.ndarray | None = None) -> dict | None:
    """Vision-only paved edge from the pavement / dirt / grass boundary.

    The returned dict uses the same lateral-field names as
    :func:`road_lane_geometry` (left is + in the ego frame), plus world
    polylines that follow the detected edge over the visible range.  A
    side is reported independently, so a one-sided boundary still works
    when the other edge is outside the camera frustum; ``left_lat`` and
    ``right_lat`` are ``None`` for sides that could not be found.

    ``offroad_mask`` optionally replaces the classic-CV chroma classifier:
    pass the learned segmentation's off-road mask (True = not asphalt) to
    reuse this exact edge-extraction geometry with a neural input.
    """
    if frame_rgb is None or cam is None:
        return None
    pos = np.asarray(pos, dtype=float)
    if ground_z is None:
        ground_z = float(pos[2]) if pos.size > 2 else 0.0
    heading = float(heading)
    fwd = np.array([math.cos(heading), math.sin(heading)])
    left = np.array([-fwd[1], fwd[0]])
    if offroad_mask is not None:
        mask = offroad_mask
    else:
        mask = _offroad_mask(frame_rgb)

    left_lats: list[float | None] = []
    right_lats: list[float | None] = []
    for s_m in PAVEMENT_STATIONS_M:
        left_lats.append(_side_pavement_lat(
            mask, cam, pos, heading, fwd, left, s_m, 1.0, ground_z))
        right_lats.append(_side_pavement_lat(
            mask, cam, pos, heading, fwd, left, s_m, -1.0, ground_z))

    def _side_state(valid: list[tuple[float, float]]):
        if len(valid) < 2:
            return None
        near = valid[0]
        near_lats = [lat for s_m, lat in valid if s_m <= PAVEMENT_NEAR_MAX_M]
        if len(near_lats) >= 2:
            spread = float(np.std(near_lats))
        else:
            spread = float(np.std([lat for _s, lat in valid[:2]]))
        consistency = 1.0 / (1.0 + spread / 0.35)
        coverage = len(valid) / len(PAVEMENT_STATIONS_M)
        confidence = 0.6 * coverage + 0.4 * consistency
        if confidence < PAVEMENT_SIDE_CONF_MIN:
            return None
        return {
            "near": float(near[1]),
            "lats": [float(lat) for _s, lat in valid],
            "confidence": round(confidence, 3),
        }

    left_valid = [
        (s_m, l) for s_m, l in zip(PAVEMENT_STATIONS_M, left_lats)
        if l is not None
    ]
    right_valid = [
        (s_m, r) for s_m, r in zip(PAVEMENT_STATIONS_M, right_lats)
        if r is not None
    ]
    left_state = _side_state(left_valid)
    right_state = _side_state(right_valid)
    if left_state is None and right_state is None:
        return None

    lane_left = left_state["near"] if left_state is not None else None
    lane_right = right_state["near"] if right_state is not None else None
    lane_width = (
        lane_left - lane_right
        if lane_left is not None and lane_right is not None else None)
    lane_center = (
        0.5 * (lane_left + lane_right)
        if lane_left is not None and lane_right is not None else None)

    left_world: list[tuple[float, float]] = []
    right_world: list[tuple[float, float]] = []
    for s_m, l, r in zip(PAVEMENT_STATIONS_M, left_lats, right_lats):
        if l is not None:
            left_world.append((
                pos[0] + s_m * fwd[0] + l * left[0],
                pos[1] + s_m * fwd[1] + l * left[1],
            ))
        if r is not None:
            right_world.append((
                pos[0] + s_m * fwd[0] + r * left[0],
                pos[1] + s_m * fwd[1] + r * left[1],
            ))

    confidence = max(
        [s["confidence"] for s in (left_state, right_state)
         if s is not None] or [0.0])
    if lane_width is not None and not (
            PAVEMENT_MIN_WIDTH_M <= lane_width <= PAVEMENT_MAX_WIDTH_M):
        confidence *= 0.7
    if lane_center is not None and abs(lane_center) > 2.5:
        confidence *= 0.7

    return {
        "source": "vision",
        "confidence": round(confidence, 3),
        "left_lat": None if lane_left is None else round(lane_left, 3),
        "right_lat": None if lane_right is None else round(lane_right, 3),
        "lane_left_lat": None if lane_left is None else round(lane_left, 3),
        "lane_right_lat": None if lane_right is None else round(lane_right, 3),
        "lane_center_lat": None if lane_center is None
        else round(lane_center, 3),
        "left_dist": None if lane_left is None else round(lane_left, 3),
        "right_dist": None if lane_right is None else round(-lane_right, 3),
        "lane_width": None if lane_width is None else round(lane_width, 3),
        "center_offset": None if lane_center is None
        else round(-lane_center, 3),
        "left_world": np.asarray(left_world, dtype=float),
        "right_world": np.asarray(right_world, dtype=float),
        "stations": list(PAVEMENT_STATIONS_M),
        "left_lats": left_lats,
        "right_lats": right_lats,
        "left_confidence": None if left_state is None
        else left_state["confidence"],
        "right_confidence": None if right_state is None
        else right_state["confidence"],
    }


def merge_boundary_geometry(roadnet: dict | None,
                            vision: dict | None) -> dict | None:
    """Merge roadnet and vision pavement boundaries into one overlay dict.

    The lane boundaries are the ROADNET lane split (the map knows the lane
    count and widths); the vision pavement edges are the *road* edge
    (grass/asphalt transition), not lane lines - letting them override the
    map lane split is exactly what made the "lane centre" sit on the road
    centre.  Vision only fills in a side when the roadnet data is missing,
    and the source of each side is kept for rendering and labels.
    """
    if roadnet is None and vision is None:
        return None

    lane_left = None
    lane_right = None
    source_left = None
    source_right = None
    if roadnet is not None:
        lane_left = float(roadnet["lane_left_lat"])
        source_left = "roadnet"
        lane_right = float(roadnet["lane_right_lat"])
        source_right = "roadnet"
    # 只有 roadnet 缺失的侧才用 vision 路面边缘补位（语义不同：那是
    # 道路物理边缘，不是车道分割线）
    if lane_left is None and vision is not None \
            and vision.get("left_lat") is not None:
        lane_left = float(vision["left_lat"])
        source_left = "vision"
    if lane_right is None and vision is not None \
            and vision.get("right_lat") is not None:
        lane_right = float(vision["right_lat"])
        source_right = "vision"
    if lane_left is None or lane_right is None:
        return None

    vision_both = (source_left == "vision" and source_right == "vision")
    if vision_both:
        lane_center = 0.5 * (lane_left + lane_right)
        lane_width = lane_left - lane_right
        center_source = "vision"
        width_source = "vision"
    elif roadnet is not None:
        lane_center = float(roadnet["lane_center_lat"])
        lane_width = float(roadnet["lane_width"])
        center_source = "roadnet"
        width_source = "roadnet"
    else:
        lane_center = None
        lane_width = None
        center_source = None
        width_source = None
    merged = {
        "source": "mixed",
        "source_left": source_left,
        "source_right": source_right,
        "lane_left_lat": round(lane_left, 3),
        "lane_right_lat": round(lane_right, 3),
        "lane_center_lat": None if lane_center is None
        else round(lane_center, 3),
        "left_dist": round(lane_left, 3),
        "right_dist": round(-lane_right, 3),
        "lane_width": None if lane_width is None else round(lane_width, 3),
        "center_offset": None if lane_center is None
        else round(-lane_center, 3),
        "boundary_span": round(abs(lane_left - lane_right), 3),
        "width_source": width_source,
        "center_source": center_source,
        "confidence": float(vision["confidence"]) if vision is not None
        else 0.0,
        "left_confidence": None if vision is None
        else vision.get("left_confidence"),
        "right_confidence": None if vision is None
        else vision.get("right_confidence"),
    }
    if vision is not None:
        merged["left_world"] = vision.get("left_world")
        merged["right_world"] = vision.get("right_world")
        merged["stations"] = vision.get("stations")
        merged["left_lats"] = vision.get("left_lats")
        merged["right_lats"] = vision.get("right_lats")
    if roadnet is not None:
        merged["road_id"] = roadnet.get("road_id")
        merged["road_width"] = roadnet.get("road_width")
        merged["road_center_lat"] = roadnet.get("road_center_lat")
        merged["left_edge_lat"] = roadnet.get("left_edge_lat")
        merged["right_edge_lat"] = roadnet.get("right_edge_lat")
    return merged


def draw_lane_frame(img, frame, cam, st, heading) -> None:
    """Draw the perceived lane frame (pair_lane_markings result).

    This is the real perception output - the lane boundaries paired from
    detected markings - as opposed to the map/roadnet reference lines.
    ``frame.left`` / ``frame.right`` are world polylines of the detected
    lane edges; ``frame.center`` is their midpoint.
    """
    if frame is None or cam is None:
        return
    pos = np.asarray(st.pos, dtype=float)
    for side, poly, tag in (
            ("left", getattr(frame, "left", None), "lane left (vision)"),
            ("right", getattr(frame, "right", None), "lane right (vision)")):
        if poly is None or len(poly) < 2:
            continue
        pts = project_poly(cam, np.asarray(poly, dtype=float), pos, heading)
        if pts is not None:
            draw_poly(img, pts, LANE_FRAME_COLOR, 2, tag)
    center = getattr(frame, "center", None)
    if center is not None and len(center) >= 2:
        pts = project_poly(cam, np.asarray(center, dtype=float), pos, heading)
        if pts is not None:
            draw_poly(img, pts, CENTER_COLOR, 2, "lane center (vision)")


def project_poly(cam, wpts, pos3, heading, ground_z=None) -> np.ndarray | None:
    """Project world points to image pixels, dropping invalid samples."""
    w = np.asarray(wpts, dtype=float)
    if w.ndim != 2 or len(w) < 2:
        return None
    if w.shape[1] == 2:
        z = float(pos3[2]) if ground_z is None else float(ground_z)
        w = np.column_stack([w, np.full(len(w), z)])
    u, v, valid = cam.project(w, pos3, heading)
    pix = np.column_stack([u, v])
    pix = pix[valid]
    pix = pix[np.isfinite(pix).all(axis=1)]
    if len(pix) < 2:
        return None
    return pix.astype(np.int32)


def draw_poly(img, pix, color, thickness: int, label=None) -> None:
    cv2.polylines(img, [pix], False, color, thickness, cv2.LINE_AA)
    if label and len(pix):
        x = int(np.clip(pix[0, 0], 4, img.shape[1] - 90))
        y = int(np.clip(pix[0, 1] - 4, 14, img.shape[0] - 4))
        cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, color, 2, cv2.LINE_AA)


def _draw_lateral_bar(img, geometry, half_w: float, pad: int = 16,
                      height: int = 132) -> None:
    """Bottom-right cross-section: car between lane boundaries, in metres."""
    hh, ww = img.shape[:2]
    if geometry is None:
        return
    lane_l = geometry.get("lane_left_lat")
    lane_r = geometry.get("lane_right_lat")
    if lane_l is None and lane_r is None:
        return
    lane_l = float(lane_l) if lane_l is not None else None
    lane_r = float(lane_r) if lane_r is not None else None
    span = max(2.2, half_w * 2.0 + 0.8)
    if lane_l is not None and lane_r is not None:
        span = max(span, abs(lane_l - lane_r))
    scale = (ww - 2 * pad) / span
    bar_w = ww - 2 * pad
    x0 = pad
    y0 = hh - height - pad

    cv2.rectangle(img, (x0, y0), (x0 + bar_w, y0 + height), LAT_BG, -1)
    cv2.rectangle(img, (x0, y0), (x0 + bar_w, y0 + height), LAT_AXIS, 1)
    cx = x0 + bar_w / 2.0
    cy_mid = y0 + height - 42

    def x_of(lat: float) -> int:
        return int(round(cx + lat * scale))

    left_color = (LANE_COLOR if geometry.get("source_left") == "vision"
                  else ROADNET_LANE_COLOR)
    right_color = (LANE_COLOR if geometry.get("source_right") == "vision"
                   else ROADNET_LANE_COLOR)
    for lat, color in ((lane_l, left_color), (lane_r, right_color)):
        if lat is None:
            continue
        xx = x_of(lat)
        cv2.line(img, (xx, y0 + 22), (xx, y0 + height - 10), color, 2)
    center_lat = geometry.get("lane_center_lat")
    if center_lat is not None:
        center_lat = float(center_lat)
        cv2.line(img, (x_of(center_lat), y0 + 30),
                 (x_of(center_lat), y0 + height - 10), CENTER_COLOR, 1)

    bl = x_of(-half_w)
    br = x_of(half_w)
    cv2.rectangle(img, (bl, cy_mid - 20), (br, cy_mid + 20), BODY_COLOR, 2)
    cv2.line(img, (bl, cy_mid - 24), (bl, cy_mid + 24), BODY_COLOR, 1)
    cv2.line(img, (br, cy_mid - 24), (br, cy_mid + 24), BODY_COLOR, 1)

    cv2.putText(img, "LATERAL (m, left +)",
                (x0 + 8, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                PANEL_TEXT, 1, cv2.LINE_AA)
    if lane_l is not None:
        cv2.putText(img, f"L {lane_l:+.2f}", (x0 + 8, y0 + height - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, left_color, 1,
                    cv2.LINE_AA)
    if lane_r is not None:
        cv2.putText(img, f"R {lane_r:+.2f}",
                    (x0 + bar_w - 74, y0 + height - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, right_color, 1,
                    cv2.LINE_AA)
    if lane_l is not None:
        gap_l = lane_l - half_w
        cv2.putText(img, f"gap {gap_l:.2f}",
                    (x_of(half_w) + 6, cy_mid - 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, BODY_COLOR, 1,
                    cv2.LINE_AA)
    if lane_r is not None:
        gap_r = -lane_r - half_w
        cv2.putText(img, f"gap {gap_r:.2f}",
                    (x_of(-half_w) - 96, cy_mid - 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, BODY_COLOR, 1,
                    cv2.LINE_AA)


def render_lane_overlay(img, st, geometry, markings, cam, half_w, *,
                        vision_text: str | None = None,
                        lateral: bool = True) -> np.ndarray:
    """Draw the diagnostic overlay on one RGB front-camera frame."""
    overlay = img.copy()
    hh, ww = overlay.shape[:2]
    fwd = unit_fwd(st)
    pos = np.asarray(st.pos[:2], dtype=float)
    heading = float(st.heading)
    left = np.array([-fwd[1], fwd[0]])

    for lat, color, tag in (
            (half_w, BODY_COLOR, f"body left +{half_w:.2f}"),
            (-half_w, BODY_COLOR, f"body right -{half_w:.2f}")):
        pts = project_poly(
            cam, world_lat_line(pos, fwd, lat, 1.5, 12.0),
            st.pos, heading)
        if pts is not None:
            draw_poly(overlay, pts, color, 2, tag)
            p3 = np.array([
                pos[0] + 2.0 * fwd[0] + lat * left[0],
                pos[1] + 2.0 * fwd[1] + lat * left[1],
                st.pos[2]])
            u, v, ok = cam.project(np.array([p3]), st.pos, heading)
            if ok[0]:
                x0 = int(np.clip(u[0] - 30, 0, ww - 1))
                y0 = int(np.clip(v[0], 0, hh - 1))
                cv2.arrowedLine(overlay, (x0, hh - 1), (x0, y0),
                                color, 2, cv2.LINE_AA, tipLength=0.25)

    if geometry is not None:
        for key, color, tag in (
                ("left_edge_lat", ROAD_COLOR,
                 f"road left +{geometry['left_edge_lat']:.2f}"),
                ("right_edge_lat", ROAD_COLOR,
                 f"road right {geometry['right_edge_lat']:.2f}")):
            pts = project_poly(
                cam, world_lat_line(pos, fwd, float(geometry[key]),
                                    1.5, 22.0),
                st.pos, heading)
            if pts is not None:
                draw_poly(overlay, pts, color, 2, tag)

        for side, key in (("left", "lane_left_lat"),
                          ("right", "lane_right_lat")):
            if geometry.get(key) is None:
                continue
            # 外侧车道边界 = 道路边缘（最外侧车道的物理事实）：该侧由
            # 橙色 road edge 线表示，灰色/绿色 boundary 线跳过，避免两线
            # 重合让人误以为"车道线画在了道路边缘上"。
            edge_key = ("left_edge_lat" if side == "left"
                        else "right_edge_lat")
            if geometry.get(edge_key) is not None and abs(
                    float(geometry[key]) - float(geometry[edge_key])) < 0.3:
                continue
            if geometry.get(f"source_{side}") == "vision":
                color = LANE_COLOR
                tag = f"boundary {side} vision pavement"
            else:
                color = ROADNET_LANE_COLOR
                tag = f"boundary {side} roadnet"
            world = geometry.get(f"{side}_world")
            if world is not None and len(world) >= 2:
                pts = project_poly(cam, world, st.pos, heading)
            else:
                pts = project_poly(
                    cam, world_lat_line(pos, fwd, float(geometry[key]),
                                        1.5, 22.0),
                    st.pos, heading)
            if pts is not None:
                draw_poly(overlay, pts, color, 2, tag)

        if geometry.get("lane_center_lat") is not None:
            pts = project_poly(
                cam, world_lat_line(pos, fwd,
                                    float(geometry["lane_center_lat"]),
                                    1.5, 22.0),
                st.pos, heading)
            if pts is not None:
                draw_poly(overlay, pts, CENTER_COLOR, 1,
                          f"center {geometry['lane_center_lat']:+.2f}")

    for mk in markings:
        pts = np.asarray(mk.pixels, dtype=np.int32)
        color = MARK_YELLOW if mk.color == "yellow" else MARK_WHITE
        cv2.polylines(overlay, [pts], False, color,
                      3 if mk.kind == "solid" else 1, cv2.LINE_AA)
        if len(pts):
            rel = np.asarray(mk.world, dtype=float)[:, :2] - pos
            s = rel @ fwd
            lat = rel @ left
            label = f"{mk.kind} s={np.median(s):.1f} lat={np.median(lat):+.2f}"
            x = int(np.clip(pts[0, 0], 4, ww - 220))
            y = int(np.clip(pts[0, 1] - 3, 12, hh - 4))
            cv2.putText(overlay, label, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1,
                        cv2.LINE_AA)

    lines = [
        f"heading={math.degrees(heading):+.1f}deg speed={st.speed:.2f} m/s",
    ]
    if geometry is not None:
        src_l = geometry.get("source_left") or "none"
        src_r = geometry.get("source_right") or "none"
        lines.append(
            f"boundary: left {geometry['left_dist']:.2f} m {src_l} | "
            f"right {geometry['right_dist']:.2f} m {src_r}")
        width_src = geometry.get("width_source") or "mixed"
        center_src = geometry.get("center_source") or "mixed"
        if geometry.get("lane_width") is not None:
            lines.append(
                f"lane width {geometry['lane_width']:.2f} m {width_src} | "
                f"center offset {geometry['center_offset']:+.2f} m "
                f"{center_src}")
        else:
            lines.append(
                f"boundary span {geometry['boundary_span']:.2f} m mixed")
        if geometry.get("road_center_lat") is not None:
            lines.append(
                f"roadnet ref: center lat {geometry['road_center_lat']:+.2f} "
                f"| road width {geometry['road_width']:.2f} m")
    lines.append(f"ego body half-width {half_w:.2f} m (bbox)")
    if vision_text:
        lines.append(vision_text)
    for i, text in enumerate(lines):
        cv2.rectangle(overlay, (8, 8 + i * 26),
                      (ww - 8, 30 + i * 26), PANEL_BG, -1)
        cv2.putText(overlay, text, (14, 24 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, PANEL_TEXT, 1,
                    cv2.LINE_AA)

    if lateral:
        _draw_lateral_bar(overlay, geometry, half_w)
    return overlay
