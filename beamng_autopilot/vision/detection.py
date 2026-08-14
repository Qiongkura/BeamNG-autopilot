"""Vision obstacle detection: YOLOv8n on the grabbed game frame.

The Steam edition has no camera sensor API, so the game window is captured
with PrintWindow (``connector.grab_screen`` -> RGB (H, W, 3)) and a small
COCO detector finds vehicles / pedestrians in the image.  Each detection
box's bottom edge is back-projected through the live in-game camera pose
onto the ground plane, producing a world-space :class:`Obstacle` the
planner can route around.  This gives a third, forward-looking obstacle
source on top of the raycast fan and the ``getAllVehicles`` registry, and
it is the channel that will later feed end-to-end driving.

The detector is loaded lazily so importing this module stays cheap; the
first :meth:`VisionDetector.detect` call pays the ultralytics / CUDA load.
"""

from __future__ import annotations

import math
import os
import threading
from pathlib import Path

import numpy as np

# Ultralytics wants to write settings into the user profile; this sandboxed
# account cannot create AppData dirs, so redirect config to a project-local
# folder before ultralytics is imported.
_PROJECT = Path(__file__).resolve().parents[2]
os.environ.setdefault("YOLO_CONFIG_DIR", str(_PROJECT / ".yolo"))

WEIGHTS = _PROJECT / "weights" / "yolov8n.pt"

# COCO class ids that matter for road obstacle avoidance.
COCO_CLASSES = {
    0: "person",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

_MODEL = None
_MODEL_LOCK = threading.Lock()


def load_model(weights=None):
    """Load (once) and return the shared YOLOv8n model."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL is None:
            from ultralytics import YOLO

            _MODEL = YOLO(str(weights or WEIGHTS))
    return _MODEL


def back_project(u, v, cam_model, pos, heading, ground_z: float = 0.0):
    """Ground-plane world point under image pixel (u, v).

    The ray from the camera centre through the pixel is intersected with
    the horizontal plane at ``ground_z`` (default: world z = 0).  Returns
    ``(x, y)`` or ``None`` when the ray does not hit that plane in front of
    the camera (sky, horizon, camera behind the plane).
    """
    C, r, f, u_axis = cam_model.camera_pose(pos, heading)
    d = (r * ((u - cam_model.cx) / cam_model.fx)
         + f + u_axis * ((cam_model.cy - v) / cam_model.fy))
    if d[2] >= -1e-9:
        return None  # ray points up or parallel to the ground
    t = (ground_z - C[2]) / d[2]
    if t <= 0.05:
        return None
    p = C + t * d
    return float(p[0]), float(p[1])


_LUA_CAMERA = (
    "local p=getCameraPosition(); local f=getCameraForward(); "
    "local u=getCameraUp(); "
    "return string.format('%f,%f,%f,%f,%f,%f,%f,%f,%f,%f', "
    "p.x,p.y,p.z,f.x,f.y,f.z,u.x,u.y,u.z,getCameraFovDeg())"
)


def live_camera_model(bng, width, height, pos, heading, fallback=None):
    """Build a CameraModel matching the live in-game camera pose.

    Queries ``getCameraPosition/Forward/Up/FovDeg`` through the Lua bridge
    (works on the Steam edition) and converts the world-space pose into the
    vehicle-local frame the CameraModel expects, so back-projection follows
    whatever view the player is in (hood cam, chase cam, ...).  When the
    query fails, ``fallback`` is returned if given, otherwise a calibrated
    default model scaled to ``(width, height)``.
    """
    try:
        resp = bng.queue_lua_command(_LUA_CAMERA, response=True)
        vals = [float(v) for v in str(resp).split(",")]
        if len(vals) != 10:
            raise ValueError(f"unexpected camera response: {resp!r}")
    except Exception:
        if fallback is not None:
            return fallback
        from beamng_autopilot.vision.projection import default_camera

        return default_camera(int(width), int(height))

    p, f, u, fov = vals[:3], vals[3:6], vals[6:9], vals[9]
    h = float(heading)
    right = np.array([math.sin(h), -math.cos(h), 0.0])
    fwd = np.array([math.cos(h), math.sin(h), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    R = np.column_stack([right, fwd, up])
    pos = np.asarray(pos, dtype=float)
    offset = R.T @ (np.asarray(p) - pos)
    f_local = R.T @ np.asarray(f)
    u_local = R.T @ np.asarray(u)
    from beamng_autopilot.vision.projection import CameraModel

    return CameraModel(
        offset=offset,
        fwd_local=f_local,
        up_local=u_local,
        fov_deg=float(fov),
        width=int(width),
        height=int(height),
    )


class VisionDetector:
    """YOLOv8n front-camera detector producing world-space obstacles."""

    def __init__(self, weights=None, conf: float = 0.35,
                 max_dist: float = 55.0, device=None):
        # ``None`` lets ultralytics pick the best available device itself
        # (CUDA when present); this version rejects the explicit "auto".
        self._weights = weights
        self.conf = conf
        self.max_dist = max_dist
        self.device = device
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            self._model = load_model(self._weights)
        return self._model

    def detect(self, frame_rgb, cam_model, pos, heading):
        """Detect road obstacles in a grabbed game frame.

        ``frame_rgb`` is the RGB (H, W, 3) uint8 array from
        ``connector.grab_screen``.  Returns ``(obstacles, boxes)`` where
        ``obstacles`` is a list of world-space ``perception.Obstacle`` and
        ``boxes`` is a list of ``(x1, y1, x2, y2, label, conf)`` in the
        frame's pixel space (for HUD drawing).
        """
        import cv2

        from beamng_autopilot.perception import Obstacle, filter_self_overlap

        if cam_model is None or frame_rgb is None:
            return [], []
        h_img, w_img = frame_rgb.shape[:2]
        ground_z = float(pos[2]) if len(np.asarray(pos)) > 2 else 0.0
        model = self._ensure_model()
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        predict_kwargs = {"conf": self.conf, "verbose": False}
        if self.device:
            predict_kwargs["device"] = self.device
        results = model.predict(bgr, **predict_kwargs)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return [], []
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        obstacles: list[Obstacle] = []
        boxes_out: list[tuple] = []
        for (x1, y1, x2, y2), c, cf in zip(xyxy, cls, confs):
            if int(c) not in COCO_CLASSES:
                continue
            bw = float(x2 - x1)
            bh = float(y2 - y1)
            if bw < 8 or bh < 8:
                continue
            u_c = (x1 + x2) / 2.0
            v_b = float(y2)
            # Bottom edge must be in the lower half of the frame: something
            # actually on the road, not sky / billboards / trees.
            if v_b < h_img * 0.45:
                continue
            ground = back_project(u_c, v_b, cam_model, pos, heading,
                                  ground_z=ground_z)
            if ground is None:
                continue
            gx, gy = ground
            dist = math.hypot(gx - pos[0], gy - pos[1])
            if dist > self.max_dist or dist < 2.0:
                continue
            # Rough width from the bottom edge spread on the ground plane.
            gl = back_project(x1, v_b, cam_model, pos, heading,
                              ground_z=ground_z)
            gr = back_project(x2, v_b, cam_model, pos, heading,
                              ground_z=ground_z)
            width = 2.0
            if gl is not None and gr is not None:
                width = math.hypot(gr[0] - gl[0], gr[1] - gl[1])
            width = max(0.6, min(width, 6.0))
            label = COCO_CLASSES.get(int(c), "object")
            if label == "person":
                half_w, half_h = 0.4, 0.4
            else:
                half_w = width / 2.0
                half_h = width * (1.15 if label in ("bus", "truck") else 0.95)
            obstacles.append(Obstacle(
                x=gx, y=gy, half_w=half_w, half_h=half_h,
                category="vision", label=label))
            boxes_out.append((int(x1), int(y1), int(x2), int(y2),
                              label, float(cf)))
        # A chase cam that shows the player's own car back-projects a
        # "bus"/"car" box right on top of the ego; drop any vision box
        # whose footprint contains the ego so it can never block the lane.
        return filter_self_overlap(obstacles, pos), boxes_out
