"""Live lane-recognition overlay: what the stack sees, one frame at a time.

``render_lane_vis`` turns one FSD tick into a single BGR diagnostic image
so a human can watch the recognition chain work in real time:

* top-left  - the camera frame with the ROAD mask (green tint) and the
  LINE mask (magenta tint) actually produced this tick;
* world markings reprojected into the image (each semantic-head candidate,
  colour = paint colour, thin = low confidence);
* the accepted lane reference (cyan) and its published hard boundaries
  (left yellow / right red) reprojected where they have image support;
* top-right - the BEV grid: drivable (green), obstacle (red), lane channel
  (magenta), the chosen trajectory (cyan) and the lane reference (yellow);
* a HUD line with the numbers that matter: speed, lane source, paired
  flag, confidence, span, pairing mode / reject reason, painted-line
  lateral, planned deviation, and the safety level/reason.

Pure rendering - it consumes the tick, changes no decisions.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


def _tint(img: np.ndarray, mask: np.ndarray, color: tuple[int, int, int],
          alpha: float = 0.55) -> None:
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return
    overlay = img.copy()
    overlay[m] = color
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, dst=img)


def _poly_points(world: np.ndarray, cam, pos, heading,
                 ground_z: float = 0.0) -> np.ndarray:
    """Image points (int, Nx2) for a world polyline (front-camera only)."""
    pts = np.asarray(world, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 2 or len(pts) < 2:
        return np.empty((0, 2), dtype=int)
    if pts.shape[1] == 2:
        # CameraModel.project expects world xyz; the lane/markings
        # polylines are ground-plane (N, 2).
        pts = np.column_stack([pts, np.full(len(pts), float(ground_z))])
    u, v, ok = cam.project(pts, pos, heading)
    out = np.stack([u, v], axis=1)
    return out[np.isfinite(out).all(axis=1) & ok].astype(int)


def _draw_poly(img: np.ndarray, pts: np.ndarray, color, thick: int = 2,
               dotted: bool = False) -> None:
    if len(pts) < 2:
        return
    if not dotted:
        cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, color, thick,
                      cv2.LINE_AA)
        return
    for a, b in zip(pts[:-1], pts[1:]):
        if int(a[0] + a[1]) % 2 == 0:
            continue
        cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                 color, thick, cv2.LINE_AA)


def _bev_panel(out, n: int = 60) -> np.ndarray:
    """BEV raster: drivable green / obstacle red / lane magenta + paths."""
    size = 240
    grid = np.zeros((n, n, 3), dtype=np.uint8)
    panel = np.zeros((size, size, 3), dtype=np.uint8)
    drv = getattr(out, "drivable", None)
    bev = getattr(out, "bev", None)
    if drv is not None and drv.shape == (n, n):
        grid[drv > 0] = (0, 110, 0)
    if bev is not None and bev.shape == (n, n):
        grid[bev >= 0.6] = (0, 0, 220)
    sem = (getattr(out, "head_outputs", None) or {}).get("semantic")
    if sem is not None and "line" in getattr(sem, "masks", {}):
        pass  # lane channel comes from fmap below; mask has no BEV form
    # NOTE: the fmap lane channel is deliberately NOT drawn - the ground-
    # plane back-projection of a thin paint line smears across the far
    # grid (a 2 px line at the horizon fans out over tens of metres), so
    # the channel reads road-wide magenta and tells the viewer nothing.
    # The camera-view reprojection above shows the real line evidence.
    panel = cv2.resize(grid, (size, size), interpolation=cv2.INTER_NEAREST)
    best = getattr(out, "best_path", None)
    lane_ref = getattr(out, "lane_ref", None)
    for poly, color in ((lane_ref, (0, 255, 255)), (best, (255, 255, 0))):
        if poly is None:
            continue
        pts = np.asarray(poly, dtype=float)[:, :2]
        if len(pts) < 2:
            continue
        # world -> ego grid cell (grid centred on the ego, +x fwd, +y left)
        pos = out.snapshot.pos if getattr(out, "snapshot", None) is not None \
            else None
        if pos is None:
            continue
        h = float(out.snapshot.heading)
        rel = pts - np.asarray(pos[:2], dtype=float)[None, :]
        ch, sh = math.cos(h), math.sin(h)
        ex = rel[:, 0] * ch + rel[:, 1] * sh
        ey = -rel[:, 0] * sh + rel[:, 1] * ch
        extent = n * 0.5 * 0.5
        for k in range(len(pts)):
            if abs(ex[k]) >= extent or abs(ey[k]) >= extent:
                continue
            row = int((extent - ex[k]) / (2 * extent) * size)
            col = int((extent - ey[k]) / (2 * extent) * size)
            cv2.circle(panel, (col, row), 2, color, -1)
    cv2.putText(panel, "BEV", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255, 255, 255), 1)
    return panel


def render_lane_vis(out, pos, heading: float, *,
                    line_lat=None, extra_lines=()) -> np.ndarray:
    """Render one tick as a BGR diagnostic image (see module doc)."""
    frame = getattr(out, "frame", None)
    if frame is None:
        return np.zeros((240, 320, 3), dtype=np.uint8)
    rgb = np.asarray(frame, dtype=np.uint8)
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    cam = getattr(out, "cam", None)
    heads = getattr(out, "head_outputs", None) or {}
    sem = heads.get("semantic")
    if sem is not None and cam is not None:
        masks = getattr(sem, "masks", {})
        if "road" in masks:
            _tint(img, masks["road"], (0, 160, 0), 0.35)
        if "line" in masks:
            _tint(img, masks["line"], (255, 0, 255), 0.55)
        for m in getattr(sem, "meta", {}).get("markings", []) or []:
            pts = _poly_points(np.asarray(m.world, dtype=float), cam, pos,
                               heading, float(pos[2]) if len(pos) > 2 else 0.0)
            color = (0, 255, 255) if getattr(m, "color", "") == "yellow" \
                else (255, 255, 255)
            _draw_poly(img, pts, color, 2,
                       dotted=getattr(m, "kind", "") == "dashed")
    lane_ref = getattr(out, "lane_ref", None)
    for poly, color, thick in (
            (lane_ref, (255, 200, 0), 2),
            (getattr(out, "lane_left", None), (0, 255, 255), 2),
            (getattr(out, "lane_right", None), (0, 0, 255), 2)):
        if poly is None or cam is None:
            continue
        _draw_poly(img, _poly_points(np.asarray(poly, dtype=float)[:, :2],
                                     cam, pos, heading,
                                     float(pos[2]) if len(pos) > 2 else 0.0),
                   color, thick)
    for poly in extra_lines:
        if poly is None or cam is None:
            continue
        _draw_poly(img, _poly_points(np.asarray(poly, dtype=float)[:, :2],
                                     cam, pos, heading), (0, 140, 255), 1)

    # BEV panel pastes FIRST (top-right); the HUD draws over everything
    # and must come last or the paste erases it (caught by unit test).
    panel = _bev_panel(out)
    h0, w0 = img.shape[:2]
    # The frame may be smaller than the panel (low-res test stubs): paste
    # whatever fits, anchored at the top-right corner.
    ph = min(panel.shape[0], h0)
    pw = min(panel.shape[1], w0)
    img[0:ph, w0 - pw:] = panel[:ph, :pw]

    meta = getattr(out, "meta", {}) or {}
    fusion = meta.get("lane_fusion_debug") or {}
    pair = meta.get("lane_pair_debug") or {}
    chosen = fusion.get("chosen") or {}
    hud1 = (f"v={float(getattr(out, 'best_speed', 0) or 0):4.1f} "
            f"src={meta.get('lane_src_sel', '?')} "
            f"paired={meta.get('lane_paired', '?')} "
            f"conf={chosen.get('confidence', '-')} "
            f"span={chosen.get('span_m', '-')}")
    hud2 = (f"pair={pair.get('mode', fusion.get('mode', '-'))} "
            f"rej={sum((pair.get('pair_rejects') or {}).values())} "
            f"line_lat={line_lat} "
            f"dev={meta.get('lane_dev_m', '-')}")
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(img, hud1, (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, hud2, (6, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (0, 255, 255), 1, cv2.LINE_AA)
    return img
