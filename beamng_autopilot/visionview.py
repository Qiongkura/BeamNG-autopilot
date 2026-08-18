"""Tesla-style vision overlay: planned route / waypoints / goal rendered
directly in the 3D world (BeamNG debug primitives) plus a bird-view map and
a front-camera projection overlay in the HUD.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _to_xy(points):
    """Accept (N,2) or (N,3) point arrays; return float (N,2)."""
    if points is None:
        return None
    arr = np.asarray(points, dtype=float)
    if arr.ndim == 2 and arr.shape[1] >= 3:
        arr = arr[:, :2]
    return arr


ROUTE_COLOR = (0.2, 0.9, 0.3, 0.9)       # green: planned route
WP_COLOR = (0.9, 0.9, 0.2, 1.0)          # yellow: user waypoints
GOAL_COLOR = (0.95, 0.3, 0.2, 1.0)       # red: goal
STATUS_COLOR = (0.3, 0.8, 1.0, 1.0)      # blue: status text
PRED_COLOR = (0.2, 0.7, 1.0, 1.0)        # projected route in camera view
OBS_COLOR = (0.95, 0.25, 0.2, 1.0)       # red: detected obstacles
DET_COLOR = (0.1, 0.85, 1.0, 1.0)        # cyan: raw YOLO boxes in camera view


class WorldOverlay:
    """Owns BeamNG debug primitives and refreshes them on demand."""

    def __init__(self, bng):
        self.bng = bng
        self._polyline_ids: list[int] = []
        self._sphere_ids: list[int] = []
        self._text_ids: list[int] = []
        self._rect_ids: list[int] = []

    def _raw_connection(self):
        """Return the beamngpy connection, or None when unavailable."""
        conn = getattr(self.bng, "connection", None)
        if conn is None or not callable(getattr(conn, "send", None)):
            return None
        return conn

    def _batch(self, messages, expected):
        """Send a list of protocol messages and wait for all responses.

        The normal debug API performs one blocking send/recv per primitive,
        which costs about one round trip each (~16 ms) and made a full
        overlay refresh take ~750 ms.  Sending every message first and only
        then draining the responses collapses the refresh into ~2 round
        trips.  Returns a list aligned with ``messages`` (None entries mark
        failed sends/responses) or None when raw batching is unavailable.
        """
        conn = self._raw_connection()
        if conn is None:
            return None
        sent: list = []
        for msg in messages:
            try:
                sent.append(conn.send(msg))
            except Exception:
                # NOTE: bare except kept — debug primitive send can fail
                # with any transport error; mark as failed.
                sent.append(None)
        out: list = []
        for msg, resp in zip(messages, sent):
            if resp is None:
                out.append(None)
                continue
            want = expected.get(msg["type"])
            try:
                out.append(resp.recv(want))
            except Exception:
                # NOTE: bare except kept — debug primitive recv can fail
                # with any transport error; mark as failed.
                out.append(None)
        return out

    def _clear_type(self, obj_type: str, ids) -> None:
        """Best-effort fallback that removes one object type individually."""
        for oid in ids:
            try:
                if obj_type == "polylines":
                    self.bng.remove_debug_polyline(oid)
                elif obj_type == "spheres":
                    self.bng.remove_debug_spheres(oid)
                elif obj_type == "text":
                    self.bng.remove_debug_text(oid)
                elif obj_type == "rectangles":
                    self.bng.remove_debug_rectangle(oid)
            except Exception:
                # NOTE: bare except kept — best-effort removal of
                # individual debug primitives; ignore failures.
                pass

    def _clear(self):
        grouped = {
            "polylines": list(self._polyline_ids),
            "spheres": list(self._sphere_ids),
            "text": list(self._text_ids),
            "rectangles": list(self._rect_ids),
        }
        messages = [
            dict(type="RemoveDebugObjects", objType=obj_type, objIDs=ids)
            for obj_type, ids in grouped.items() if ids
        ]
        results = self._batch(
            messages, {"RemoveDebugObjects": "DebugObjectsRemoved"})
        if results is not None and not (results and all(
                r is None for r in results)):
            for msg, res in zip(messages, results):
                if res is None:
                    self._clear_type(msg["objType"], grouped[msg["objType"]])
            self._polyline_ids.clear()
            self._sphere_ids.clear()
            self._text_ids.clear()
            self._rect_ids.clear()
            return
        for obj_type, ids in grouped.items():
            self._clear_type(obj_type, ids)
        self._polyline_ids.clear()
        self._sphere_ids.clear()
        self._text_ids.clear()
        self._rect_ids.clear()

    def _draw_batched(self, route_xy, waypoints, goal_xy, obstacles,
                      status_text, status_pos, z: float, markers: bool) -> bool:
        """Add all debug primitives in one batched round-trip."""
        messages: list[dict] = []
        if route_xy is not None and len(route_xy) >= 2:
            messages.append(dict(
                type="AddDebugPolyline",
                coordinates=[(float(x), float(y), z) for x, y in route_xy],
                color=ROUTE_COLOR, cling=False, offset=0.0))
        if markers and waypoints is not None and len(waypoints) > 0:
            messages.append(dict(
                type="AddDebugSpheres",
                coordinates=[(float(x), float(y), z) for x, y in waypoints],
                radii=[0.8] * len(waypoints),
                colors=[WP_COLOR] * len(waypoints),
                cling=True, offset=0.0))
        if markers and goal_xy is not None:
            messages.append(dict(
                type="AddDebugSpheres",
                coordinates=[(float(goal_xy[0]), float(goal_xy[1]), z)],
                radii=[1.4], colors=[GOAL_COLOR], cling=True, offset=0.0))
            messages.append(dict(
                type="AddDebugText",
                origin=(float(goal_xy[0]), float(goal_xy[1]), z + 1.5),
                content="GOAL", color=GOAL_COLOR, cling=True, offset=0.0))
        if obstacles:
            for ob in obstacles:
                hw, hh = float(ob.half_w), float(ob.half_h)
                messages.append(dict(
                    type="AddDebugRectangle",
                    vertices=[
                        (float(ob.x + hw), float(ob.y + hh), z),
                        (float(ob.x - hw), float(ob.y + hh), z),
                        (float(ob.x - hw), float(ob.y - hh), z),
                        (float(ob.x + hw), float(ob.y - hh), z),
                    ],
                    color=OBS_COLOR, cling=True, offset=0.1))
        if status_text and status_pos is not None:
            messages.append(dict(
                type="AddDebugText",
                origin=(float(status_pos[0]), float(status_pos[1]),
                        float(status_pos[2]) + 2.0),
                content=status_text, color=STATUS_COLOR,
                cling=True, offset=0.0))
        if not messages:
            return True
        expected = {
            "AddDebugPolyline": "DebugPolylineAdded",
            "AddDebugSpheres": "DebugSphereAdded",
            "AddDebugRectangle": "DebugRectangleAdded",
            "AddDebugText": "DebugTextAdded",
        }
        results = self._batch(messages, expected)
        if results is None or (results and all(
                r is None for r in results)):
            return False
        for msg, res in zip(messages, results):
            if res is None:
                continue
            try:
                if msg["type"] == "AddDebugPolyline":
                    self._polyline_ids.append(int(res["lineID"]))
                elif msg["type"] == "AddDebugSpheres":
                    self._sphere_ids.extend(
                        int(s) for s in res["sphereIDs"])
                elif msg["type"] == "AddDebugRectangle":
                    self._rect_ids.append(int(res["rectangleID"]))
                elif msg["type"] == "AddDebugText":
                    self._text_ids.append(int(res["textID"]))
            except Exception:
                pass
        return True

    def update(self, route_xy=None, waypoints=None, goal_xy=None,
               obstacles=None, status_text=None, status_pos=None,
               z: float = 0.5,
               enabled: bool = True,
               markers: bool = True) -> None:
        """Re-render all overlay primitives (call at a low rate, ~2-4 Hz)."""
        self._clear()
        if not enabled:
            return
        route_xy = _to_xy(route_xy)
        waypoints = _to_xy(waypoints)
        if goal_xy is not None:
            g = np.asarray(goal_xy, dtype=float)
            goal_xy = g[:2] if g.ndim == 1 and len(g) >= 2 else g
        if self._draw_batched(
                route_xy, waypoints, goal_xy, obstacles,
                status_text, status_pos, z, markers):
            return
        if route_xy is not None and len(route_xy) >= 2:
            coords = [(float(x), float(y), z) for x, y in route_xy]
            try:
                pid = self.bng.add_debug_polyline(coords, ROUTE_COLOR)
                self._polyline_ids.append(pid)
            except Exception:
                pass
        if markers and waypoints is not None and len(waypoints) > 0:
            coords = [(float(x), float(y), z) for x, y in waypoints]
            radii = [0.8] * len(coords)
            try:
                ids = self.bng.add_debug_spheres(
                    coords, radii, WP_COLOR, cling=True)
                self._sphere_ids.extend(ids)
            except Exception:
                pass
        if markers and goal_xy is not None:
            try:
                ids = self.bng.add_debug_spheres(
                    [(float(goal_xy[0]), float(goal_xy[1]), z)],
                    [1.4], GOAL_COLOR, cling=True)
                self._sphere_ids.extend(ids)
                tid = self.bng.add_debug_text(
                    (float(goal_xy[0]), float(goal_xy[1]), z + 1.5),
                    "GOAL", GOAL_COLOR, cling=True)
                self._text_ids.append(tid)
            except Exception:
                pass
        if obstacles:
            for ob in obstacles:
                hw, hh = float(ob.half_w), float(ob.half_h)
                verts = [
                    (float(ob.x + hw), float(ob.y + hh), z),
                    (float(ob.x - hw), float(ob.y + hh), z),
                    (float(ob.x - hw), float(ob.y - hh), z),
                    (float(ob.x + hw), float(ob.y - hh), z),
                ]
                try:
                    rid = self.bng.add_debug_rectangle(
                        verts, OBS_COLOR, cling=True, offset=0.1)
                    self._rect_ids.append(rid)
                except Exception:
                    pass
        if status_text and status_pos is not None:
            try:
                tid = self.bng.add_debug_text(
                    (float(status_pos[0]), float(status_pos[1]),
                     float(status_pos[2]) + 2.0),
                    status_text, STATUS_COLOR, cling=True)
                self._text_ids.append(tid)
            except Exception:
                pass

    def close(self) -> None:
        self._clear()


def render_birdview(canvas, route_xy=None, waypoints=None, goal_xy=None,
                    obstacles=None, pos=None, heading: float = 0.0,
                    radius_m: float = 60.0, lane_markings=None):
    """Draw a vehicle-centric bird-view map into `canvas` (BGR uint8)."""
    import cv2

    h, w = canvas.shape[:2]
    cx, cy = w // 2, h // 2
    scale = min(w, h) / (2.0 * radius_m)
    ch = np.cos(-heading)
    sh = np.sin(-heading)

    def to_canvas(x, y):
        dx = x - pos[0]
        dy = y - pos[1]
        rx = dx * ch - dy * sh
        ry = dx * sh + dy * ch
        return int(cx - rx * scale), int(cy - ry * scale)

    route_xy = _to_xy(route_xy)
    waypoints = _to_xy(waypoints)
    if goal_xy is not None:
        g = np.asarray(goal_xy, dtype=float)
        goal_xy = g[:2] if g.ndim == 1 and len(g) >= 2 else g

    cv2.circle(canvas, (cx, cy), int(radius_m * scale), (90, 96, 110), 1)
    cv2.circle(canvas, (cx, cy), int(radius_m * scale * 0.5),
               (60, 66, 80), 1)
    cv2.line(canvas, (cx - int(radius_m * scale), cy),
             (cx + int(radius_m * scale), cy), (45, 50, 62), 1)
    cv2.line(canvas, (cx, cy - int(radius_m * scale)),
             (cx, cy + int(radius_m * scale)), (45, 50, 62), 1)

    def clip_pt(x, y):
        return max(0, min(w - 1, x)), max(0, min(h - 1, y))

    if route_xy is not None and len(route_xy) >= 2:
        pts = []
        for x, y in route_xy:
            px, py = to_canvas(x, y)
            pts.append(clip_pt(px, py))
        for a, b in zip(pts, pts[1:]):
            cv2.line(canvas, a, b, (70, 200, 90), 2)

    if lane_markings:
        for mk in lane_markings:
            world = np.asarray(getattr(mk, "world", None), dtype=float)
            if world.ndim != 2 or world.shape[1] < 2 or len(world) < 2:
                continue
            color = ((0, 255, 255) if getattr(mk, "color", "white") == "yellow"
                     else (255, 255, 255))
            pts = [clip_pt(*to_canvas(float(x), float(y)))
                   for x, y in world[:, :2]]
            poly = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [poly], False, color, 2)

    if waypoints is not None:
        for x, y in waypoints:
            px, py = clip_pt(*to_canvas(x, y))
            cv2.circle(canvas, (px, py), 5, (60, 200, 230), -1)

    if goal_xy is not None:
        px, py = clip_pt(*to_canvas(goal_xy[0], goal_xy[1]))
        cv2.circle(canvas, (px, py), 8, (40, 80, 240), -1)
        cv2.putText(canvas, "GOAL", (px + 10, py + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 80, 240), 2)

    if obstacles:
        for ob in obstacles:
            hw, hh = float(ob.half_w), float(ob.half_h)
            corners = [
                (ob.x + hw, ob.y + hh), (ob.x - hw, ob.y + hh),
                (ob.x - hw, ob.y - hh), (ob.x + hw, ob.y - hh),
            ]
            pts = [clip_pt(*to_canvas(x, y)) for x, y in corners]
            poly = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(canvas, [poly], (30, 45, 130))
            cv2.polylines(canvas, [poly], True, (60, 80, 235), 2)

    # vehicle arrow
    fx = int(cx + 14 * np.cos(heading) * 0)
    a = (cx, cy)
    b = (int(cx + 16 * np.cos(-heading + 0.0)), int(cy - 16))
    cv2.arrowedLine(canvas, a, b, (235, 235, 235), 3, tipLength=0.45)
    return canvas


def render_camera_overlay(img, route_xy, pos, heading, cam_model,
                          obstacles=None, color=PRED_COLOR,
                          max_ahead_m: float = 40.0, det_boxes=None,
                          lane_markings=None):
    """Project the planned route into the front-camera frame.

    ``det_boxes`` is an optional list of ``(x1, y1, x2, y2, label, conf)``
    in the image's pixel space - raw YOLO detections drawn as cyan boxes so
    the user can compare what the detector sees with the world-projected
    obstacle boxes.
    """
    import cv2

    route_xy = _to_xy(route_xy)
    if route_xy is not None and len(route_xy) >= 2:
        pts = np.asarray(route_xy, dtype=float)
        d = np.linalg.norm(pts - np.asarray(pos[:2]), axis=1)
        pts = pts[d <= max_ahead_m]
        if len(pts) >= 2:
            world = np.column_stack([pts, np.zeros(len(pts))])
            u, v, valid = cam_model.project(world, pos, heading)
            valid = valid & ~np.isnan(u) & ~np.isnan(v)
            screen = []
            for i in range(len(pts)):
                if valid[i]:
                    screen.append((int(u[i]), int(v[i])))
            if len(screen) >= 2:
                for a, b in zip(screen, screen[1:]):
                    cv2.line(img, a, b,
                             (int(color[0] * 255), int(color[1] * 255),
                              int(color[2] * 255)), 3)

    if lane_markings:
        for mk in lane_markings:
            world = np.asarray(getattr(mk, "world", None), dtype=float)
            if world.ndim != 2 or world.shape[1] < 2 or len(world) < 2:
                continue
            pts3 = np.column_stack([world[:, :2], np.zeros(len(world))])
            u, v, valid = cam_model.project(pts3, pos, heading)
            valid = valid & ~np.isnan(u) & ~np.isnan(v)
            screen = [(int(u[i]), int(v[i]))
                      for i in range(len(world)) if valid[i]]
            if len(screen) < 2:
                continue
            poly = np.array(screen, dtype=np.int32).reshape((-1, 1, 2))
            bgr = ((0, 255, 255)
                   if getattr(mk, "color", "white") == "yellow"
                   else (255, 255, 255))
            thickness = 4 if getattr(mk, "kind", "") == "solid" else 3
            cv2.polylines(img, [poly], False, bgr, thickness,
                          cv2.LINE_AA)

    if obstacles:
        for ob in obstacles:
            hw, hh = float(ob.half_w), float(ob.half_h)
            corners = np.asarray([
                (ob.x + hw, ob.y + hh, 0.0), (ob.x - hw, ob.y + hh, 0.0),
                (ob.x - hw, ob.y - hh, 0.0), (ob.x + hw, ob.y - hh, 0.0),
            ], dtype=float)
            u, v, valid = cam_model.project(corners, pos, heading)
            if not np.all(valid & ~np.isnan(u) & ~np.isnan(v)):
                continue
            pts = np.array(
                [(int(u[k]), int(v[k])) for k in range(4)],
                dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts], True, (60, 80, 235), 3)
            label = ob.label or ob.category
            if label:
                xs = [p[0][0] for p in pts]
                ys = [p[0][1] for p in pts]
                cv2.putText(img, label, (min(xs), max(ys) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 80, 235),
                            2)
    if det_boxes:
        bgr = (int(DET_COLOR[0] * 255), int(DET_COLOR[1] * 255),
               int(DET_COLOR[2] * 255))
        for x1, y1, x2, y2, label, conf in det_boxes:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)),
                          bgr, 2)
            cv2.putText(img, f"{label} {conf:.2f}",
                        (int(x1), max(16, int(y1) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2)
    return img
