"""Front-camera road-marking detection.

The Steam build has no camera sensor, so the same grabbed game frame that
feeds YOLO is processed with classic CV masks (white / yellow pixels),
connected components and ground-plane back-projection.  Each detected
marking becomes a world-space polyline plus its original pixel polyline,
so the autopilot can both show it in the HUD and constrain the planned
path to the right side of a solid line.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np


LANE_LINE_MAX_PERP_SPAN_M = 0.9
LANE_LINE_MAX_PERP_FRAC = 0.04
THIN_LINE_MIN_SPAN_M = 1.5
THIN_LINE_MIN_LEN_M = 1.5
THIN_LINE_MAX_PIXEL_W_MED = 40.0
THIN_LINE_MAX_PIXEL_W_P90 = 70.0
WHITE_GRAY_MIN = 150
WHITE_SAT_MAX = 140
SMOOTHER_MATCH_MAX_M = 0.8
SMOOTHER_BUFFER_FRAMES = 8
SMOOTHER_MIN_FRAMES = 4
SMOOTHER_STALE_S = 5.0
SMOOTHER_STALE_STOP_S = 20.0
SMOOTHER_STALE_SPEED_MPS = 2.0
SMOOTHER_STATIONS = 24
SMOOTHER_MAX_OUTPUT = 6
SMOOTHER_MAX_TRACKS = 32
SMOOTHER_DEDUPE_LAT_M = 0.6
_REAL_KINDS = frozenset(("solid", "dashed", "thin"))
_KIND_RANK = {"thin": 1, "dashed": 2, "solid": 3}


def marking_is_zigzag(span: float, world_len: float) -> bool:
    """True when a back-projected polyline is mostly jitter, not a line.

    ``span`` is the net extent along the marking's principal axis and
    ``world_len`` the summed segment length.  A real road line keeps a
    ratio close to 1; camera jitter and pavement texture can double it.
    """
    return world_len > span * 2.2 + 1.0


def _kind_for(span: float, world_len: float, solid_len: float) -> str:
    """Classify a marking from its net extent and polyline length.

    ``span`` is the extent along the marking's principal axis.  A
    back-projected blob often zig-zags: its summed polyline length can
    look like a long solid line while the net span is only a metre or
    two.  Only a marking with real net extent is allowed to become
    solid/dashed; everything else is treated as unknown noise.
    """
    zigzag = marking_is_zigzag(span, world_len)
    if world_len >= solid_len and span >= max(1.5, solid_len * 0.4) \
            and not zigzag:
        return "solid"
    if world_len >= THIN_LINE_MIN_LEN_M and span >= THIN_LINE_MIN_SPAN_M \
            and not zigzag:
        return "dashed"
    return "unknown"


def _row_core_points(ys, xs) -> tuple[np.ndarray, np.ndarray]:
    """Collapse each image row to its median x and row pixel width.

    A painted line viewed from the car is a narrow band: the median-x
    skeleton follows the line while a wide pavement patch / shadow keeps
    a wide row spread and is rejected by the pixel-width checks.
    """
    rows = np.unique(ys)
    core = np.empty((len(rows), 2), dtype=float)
    widths = np.empty(len(rows), dtype=float)
    for j, y in enumerate(rows):
        xrow = xs[ys == y]
        core[j, 0] = float(np.median(xrow))
        core[j, 1] = float(y)
        widths[j] = float(np.max(xrow) - np.min(xrow) + 1)
    return core, widths


def _order_world_polyline(wpts: np.ndarray, ppts: np.ndarray):
    """Sort world points along their principal axis and return metrics."""
    center = wpts.mean(axis=0)
    d = wpts - center
    cov = d.T @ d
    evals, evecs = np.linalg.eigh(cov)
    axis = evecs[:, int(np.argmax(evals))]
    proj = wpts @ axis
    order2 = np.argsort(proj)
    wpts = wpts[order2]
    ppts = ppts[order2]
    span = float(np.max(proj[order2]) - np.min(proj[order2]))
    world_len = float(np.sum(
        np.linalg.norm(np.diff(wpts, axis=0), axis=1)))
    minor = evecs[:, int(np.argmin(evals))]
    perp = wpts @ minor
    perp_span = float(np.max(perp) - np.min(perp))
    return wpts, ppts, span, world_len, perp_span


def _thin_line_ok(core_span: float, core_len: float,
                  core_perp: float, row_w_med: float,
                  row_w_p90: float) -> bool:
    """True when a core skeleton looks like a real thin road line.

    ``kind`` stays ``"thin"`` rather than ``"solid"`` so the lane
    matcher can use it as a centre-line boundary without ever turning a
    pavement patch into a no-cross solid line.
    """
    if core_span < THIN_LINE_MIN_SPAN_M or core_len < THIN_LINE_MIN_LEN_M:
        return False
    if marking_is_zigzag(core_span, core_len):
        return False
    # A painted line may curve or drift a little in world space while its
    # pixel skeleton stays narrow; only a blob that is genuinely wide
    # relative to its own length is rejected here.  The pixel row-width
    # checks below are the main protection against pavement patches.
    if core_perp > max(1.0, 0.18 * core_span):
        return False
    if row_w_med > THIN_LINE_MAX_PIXEL_W_MED \
            or row_w_p90 > THIN_LINE_MAX_PIXEL_W_P90:
        return False
    return True


@dataclass
class LaneMarking:
    """A detected road marking in world and image space."""

    world: np.ndarray      # (N, 2) ground-plane points
    pixels: np.ndarray     # (N, 2) image (u, v) points
    color: str = "white"   # "white", "yellow" or "unknown"
    kind: str = "unknown"  # "solid", "dashed", "thin" or "unknown"
    confidence: float = 0.0


def _color_masks(frame_rgb):
    """Return [(color, uint8 mask)] for white and yellow road markings."""
    import cv2

    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    white = ((gray >= WHITE_GRAY_MIN) & (sat <= WHITE_SAT_MAX)) \
        .astype(np.uint8) * 255
    yellow = (((hsv[:, :, 0] >= 15) & (hsv[:, :, 0] <= 40)
               & (hsv[:, :, 1] >= 70) & (hsv[:, :, 2] >= 110))
              .astype(np.uint8) * 255)
    return [("white", white), ("yellow", yellow)]


class LaneDetector:
    """Classic-CV lane-marking detector with ground-plane back-projection."""

    def __init__(self, min_area: int = 30, min_height: int = 18,
                 max_dist: float = 45.0, solid_len: float = 6.0):
        self.min_area = min_area
        self.min_height = min_height
        self.max_dist = max_dist
        self.solid_len = solid_len

    def detect(self, frame_rgb, cam_model, pos, heading,
               ground_z: float | None = None) -> list[LaneMarking]:
        """Detect road markings in ``frame_rgb`` (RGB uint8)."""
        import cv2

        from beamng_autopilot.vision.detection import back_project

        if frame_rgb is None or cam_model is None:
            return []
        p = np.asarray(pos, dtype=float)
        if ground_z is None:
            ground_z = float(p[2]) if p.size > 2 else 0.0
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        markings: list[LaneMarking] = []

        for color, mask0 in _color_masks(frame_rgb):
            mask = cv2.morphologyEx(mask0, cv2.MORPH_OPEN, kernel)
            mask = cv2.dilate(mask, kernel, iterations=1)
            n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
            for i in range(1, n):
                x, y, w, h, area = stats[i]
                if area < self.min_area or h < self.min_height or w < 3:
                    continue
                if w > max(8, h * 2.5):
                    continue
                if w * h <= 0 or area / (w * h) < 0.10:
                    continue
                ys, xs = np.where(labels == i)
                order = np.argsort(ys)
                xs = xs[order].astype(float)
                ys = ys[order].astype(float)
                step = max(1, int(len(xs) // 36))
                world: list[tuple[float, float]] = []
                pixels: list[tuple[float, float]] = []
                for u, v in zip(xs[::step], ys[::step]):
                    wp = back_project(float(u), float(v), cam_model,
                                      pos, heading, ground_z=ground_z)
                    if wp is None:
                        continue
                    dist = math.hypot(wp[0] - p[0], wp[1] - p[1])
                    if dist < 2.0 or dist > self.max_dist:
                        continue
                    world.append(wp)
                    pixels.append((float(u), float(v)))
                if len(world) < 4:
                    continue
                wpts = np.asarray(world, dtype=float)
                ppts = np.asarray(pixels, dtype=float)
                # Sort along the dominant world direction so a diagonal
                # line becomes a clean polyline instead of a zig-zag.
                wpts, ppts, span, world_len, perp_span = \
                    _order_world_polyline(wpts, ppts)
                kind = _kind_for(span, world_len, self.solid_len)
                if perp_span > max(LANE_LINE_MAX_PERP_SPAN_M,
                                   LANE_LINE_MAX_PERP_FRAC * span):
                    # A real painted line is thin in world space even when
                    # its back-projected polyline is long.  A dark pavement
                    # patch / repair scar is a wide blob and must never be
                    # promoted to a solid no-cross boundary.
                    kind = "unknown"
                if kind == "unknown" and span >= THIN_LINE_MIN_SPAN_M:
                    # The raw component is too wide in world space, but the
                    # median-x skeleton may still be a real painted line.
                    core, widths = _row_core_points(ys, xs)
                    core_world: list[tuple[float, float]] = []
                    core_pixels: list[tuple[float, float]] = []
                    for cu, cv in core:
                        wp = back_project(float(cu), float(cv), cam_model,
                                          pos, heading, ground_z=ground_z)
                        if wp is None:
                            continue
                        dist = math.hypot(wp[0] - p[0], wp[1] - p[1])
                        if dist < 2.0 or dist > self.max_dist:
                            continue
                        core_world.append(wp)
                        core_pixels.append((float(cu), float(cv)))
                    if len(core_world) >= 4:
                        c_wpts, c_ppts, c_span, c_len, c_perp = \
                            _order_world_polyline(
                                np.asarray(core_world, dtype=float),
                                np.asarray(core_pixels, dtype=float))
                        row_w_med = (float(np.median(widths))
                                     if len(widths) else 0.0)
                        row_w_p90 = (float(np.percentile(widths, 90.0))
                                     if len(widths) else 0.0)
                        if _thin_line_ok(c_span, c_len, c_perp,
                                         row_w_med, row_w_p90):
                            kind = "thin"
                            wpts, ppts = c_wpts, c_ppts
                            span, world_len, perp_span = \
                                c_span, c_len, c_perp
                conf = min(1.0, 0.35 + 0.25 * (area / 1500.0)
                           + 0.4 * min(1.0, span / 40.0))
                markings.append(LaneMarking(
                    world=wpts, pixels=ppts, color=color, kind=kind,
                    confidence=float(conf)))
        return markings


def _resample_polyline(wpts: np.ndarray, n: int) -> np.ndarray:
    """Evenly resample a world polyline by cumulative arc length."""
    pts = np.asarray(wpts, dtype=float)
    if len(pts) == 0:
        return pts
    if len(pts) == 1 or n <= 1:
        return np.repeat(pts[:1], max(1, n), axis=0)
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(d)])
    total = float(cum[-1])
    if total <= 1e-9:
        return np.repeat(pts[:1], n, axis=0)
    stations = np.linspace(0.0, total, n)
    return np.column_stack([
        np.interp(stations, cum, pts[:, 0]),
        np.interp(stations, cum, pts[:, 1]),
    ])


def _nearest_mean(a: np.ndarray, b: np.ndarray) -> float:
    """Mean distance from each point of ``a`` to its nearest ``b``."""
    diff = a[:, None, :] - b[None, :, :]
    return float(np.mean(np.min(np.hypot(diff[:, :, 0], diff[:, :, 1]),
                                axis=1)))


def _polyline_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric mean nearest-point distance between two polylines."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return math.inf
    sa = _resample_polyline(a, 12)
    sb = _resample_polyline(b, 12)
    return 0.5 * (_nearest_mean(sa, sb) + _nearest_mean(sb, sa))


class MarkingSmoother:
    """Short-history temporal filter for per-frame lane markings.

    The raw detector runs independently every frame, so painted lines
    flicker as masks / connected components change.  Each new marking is
    matched to a track by colour and world-space distance; a track only
    becomes visible after a few matches, stays drawn across dropouts, and
    its output polyline is the average of the recent frames re-projected
    through the current camera model.  Only tracks whose history agrees on
    a real line type (solid / dashed / thin) are returned; single-frame
    fragments stay hidden.
    """

    def __init__(self, match_max_m: float = SMOOTHER_MATCH_MAX_M,
                 buffer: int = SMOOTHER_BUFFER_FRAMES,
                 min_frames: int = SMOOTHER_MIN_FRAMES,
                 stale_s: float = SMOOTHER_STALE_S,
                 stale_stop_s: float = SMOOTHER_STALE_STOP_S,
                 stale_speed_mps: float = SMOOTHER_STALE_SPEED_MPS,
                 stations: int = SMOOTHER_STATIONS,
                 max_output: int = SMOOTHER_MAX_OUTPUT,
                 max_tracks: int = SMOOTHER_MAX_TRACKS,
                 dedupe_lat_m: float = SMOOTHER_DEDUPE_LAT_M):
        self.match_max_m = float(match_max_m)
        self.buffer = int(buffer)
        self.min_frames = int(min_frames)
        self.stale_s = float(stale_s)
        self.stale_stop_s = float(stale_stop_s)
        self.stale_speed_mps = float(stale_speed_mps)
        self.stations = int(stations)
        self.max_output = int(max_output)
        self.max_tracks = int(max_tracks)
        self.dedupe_lat_m = float(dedupe_lat_m)
        self.tracks: list[dict] = []

    def reset(self) -> None:
        self.tracks.clear()

    def update(self, markings, cam_model, pos, heading,
               ground_z: float = 0.0, warmup: bool = False,
               speed: float = 0.0,
               now: float | None = None) -> list[LaneMarking]:
        """Match ``markings`` to tracks and return stabilised markings."""
        now = time.time() if now is None else float(now)
        pos = np.asarray(pos, dtype=float)
        fwd = np.array([math.cos(heading), math.sin(heading)])
        speed = float(speed)
        if speed >= self.stale_speed_mps:
            stale_s = self.stale_s
        elif speed <= 0.1:
            stale_s = self.stale_stop_s
        else:
            t = min(1.0, speed / self.stale_speed_mps)
            stale_s = self.stale_stop_s + t * (
                self.stale_s - self.stale_stop_s)
        self.tracks = [tr for tr in self.tracks
                       if now - tr["last_seen"] <= stale_s]

        base = len(self.tracks)
        used: set[int] = set()
        for mk in markings:
            wpts = np.asarray(mk.world, dtype=float)
            if wpts.ndim != 2 or len(wpts) < 2:
                continue
            best_i = -1
            best_d = self.match_max_m
            for i in range(base):
                tr = self.tracks[i]
                if i in used or tr["color"] != mk.color:
                    continue
                d = _polyline_distance(tr["wpts"][-1][0], wpts)
                if d < best_d:
                    best_d = d
                    best_i = i
            if best_i >= 0:
                tr = self.tracks[best_i]
                used.add(best_i)
            elif mk.kind in _REAL_KINDS:
                tr = {
                    "color": mk.color,
                    "wpts": [],
                    "conf": [],
                    "kind": mk.kind,
                    "kinds": [mk.kind] if mk.kind in _REAL_KINDS else [],
                    "best_kind": (mk.kind if mk.kind in _REAL_KINDS
                                  else "unknown"),
                    "count": 0,
                    "last_seen": now,
                }
                self.tracks.append(tr)
            else:
                continue
            tr["wpts"].append((wpts, fwd.copy()))
            if len(tr["wpts"]) > self.buffer:
                del tr["wpts"][:-self.buffer]
            tr["conf"].append(float(mk.confidence))
            if len(tr["conf"]) > self.buffer:
                del tr["conf"][:-self.buffer]
            tr["kind"] = mk.kind
            if mk.kind in _REAL_KINDS:
                tr["kinds"].append(mk.kind)
                if len(tr["kinds"]) > self.buffer:
                    del tr["kinds"][:-self.buffer]
                if (tr["best_kind"] not in _REAL_KINDS
                        or _KIND_RANK[mk.kind]
                        > _KIND_RANK[tr["best_kind"]]):
                    tr["best_kind"] = mk.kind
            tr["last_seen"] = now
            tr["count"] += 1

        if len(self.tracks) > self.max_tracks:
            ordered = sorted(
                self.tracks,
                key=lambda tr: (
                    tr["best_kind"] in _REAL_KINDS,
                    tr["count"],
                    tr["last_seen"]),
                reverse=True)
            self.tracks = ordered[:self.max_tracks]

        scored: list[tuple[float, LaneMarking]] = []
        for tr in self.tracks:
            if not warmup and (tr["count"] < self.min_frames
                               or not tr["kinds"]):
                continue
            kind = self._kind_for(tr)
            wpts = self._average(tr)
            if wpts is None and warmup and len(tr["wpts"]) >= 1:
                wpts = tr["wpts"][-1][0]
            if wpts is None or len(wpts) < 2:
                continue
            pixels = self._project(wpts, cam_model, pos, heading, ground_z)
            if pixels is None or len(pixels) < 2:
                continue
            conf = float(np.mean(tr["conf"])) if tr["conf"] else 0.0
            span = float(np.sum(np.linalg.norm(
                np.diff(wpts, axis=0), axis=1))) if len(wpts) > 1 else 0.0
            mk = LaneMarking(
                world=wpts, pixels=pixels, color=tr["color"],
                kind=kind, confidence=conf)
            kind_w = {"solid": 1.0, "dashed": 0.9, "thin": 0.7}.get(
                kind, 0.5)
            score = float(conf) * kind_w * min(1.0, span / 12.0)
            scored.append((score, mk))
        scored.sort(key=lambda item: item[0], reverse=True)
        left = np.array([-fwd[1], fwd[0]])
        accepted: list[tuple[LaneMarking, float]] = []
        for _score, mk in scored:
            rel = np.asarray(mk.world, dtype=float)[:, :2] - pos[:2]
            lat = float(np.median(rel @ left))
            if any(mk.color == old.color
                   and abs(lat - old_lat) <= self.dedupe_lat_m
                   for old, old_lat in accepted):
                continue
            accepted.append((mk, lat))
        out = [mk for mk, _lat in accepted[:self.max_output]]
        return out

    @staticmethod
    def _kind_for(tr: dict) -> str:
        """Most reliable real line type seen by this track, solid first."""
        kind = tr.get("best_kind")
        if kind in _REAL_KINDS:
            return kind
        return "unknown"

    def _average(self, tr: dict) -> np.ndarray | None:
        resampled: list[np.ndarray] = []
        for wpts, fwd in tr["wpts"]:
            if len(wpts) < 2:
                continue
            pts = _resample_polyline(wpts, self.stations)
            # The detector's principal-axis sign can flip between frames;
            # keep every history polyline pointing the same way.
            if float((pts[-1] - pts[0]) @ fwd) < 0.0:
                pts = pts[::-1].copy()
            resampled.append(pts)
        if len(resampled) < 2:
            return None
        return np.mean(np.asarray(resampled, dtype=float), axis=0)

    def _project(self, wpts, cam_model, pos, heading,
                 ground_z: float) -> np.ndarray | None:
        pos3 = np.asarray(pos, dtype=float)
        if pos3.size < 3:
            pos3 = np.append(pos3, [float(ground_z)] * (3 - pos3.size))
        world = np.column_stack([
            wpts[:, 0], wpts[:, 1],
            np.full(len(wpts), float(ground_z))])
        u, v, valid = cam_model.project(world, pos3, float(heading))
        valid = np.asarray(valid, dtype=bool)
        u = np.asarray(u, dtype=float)[valid]
        v = np.asarray(v, dtype=float)[valid]
        keep = np.isfinite(u) & np.isfinite(v)
        u, v = u[keep], v[keep]
        if len(u) < 2:
            return None
        return np.column_stack([u, v]).astype(np.int32)
