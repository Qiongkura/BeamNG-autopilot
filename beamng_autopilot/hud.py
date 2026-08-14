"""In-process live HUD for the autopilot driving loops (M1 / M3 / M4).

Renders a dashboard frame (speed / throttle / brake / steer / G-forces /
heading / lap) with an optional front-camera preview and shows it in a cv2
window from the same process that drives the car, so you can watch telemetry
while the autopilot runs.

Usage inside a driving script::

    hud = LiveHUD()
    ...
    if not hud.update(data, cam=frame):   # False when q / ESC pressed
        break
    ...
    hud.close()

Keys: q or ESC quits the window and stops the driving loop.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

W, H = 1280, 720
BG = (22, 24, 30)
GRAY = (110, 112, 120)
WHITE = (235, 235, 235)
GREEN = (70, 200, 90)
RED = (40, 80, 235)
BLUE = (240, 180, 40)
CYAN = (220, 200, 60)
GCOL = (255, 190, 80)
FONT = cv2.FONT_HERSHEY_SIMPLEX

CAM_W, CAM_H = 790, 440
PANEL_W = 420


def _vbar(img, x, y, w, h, value, color):
    v = float(np.clip(value, 0.0, 1.0))
    cv2.rectangle(img, (x, y), (x + w, y + h), GRAY, 1)
    fh = int(round(h * v))
    if fh > 0:
        cv2.rectangle(img, (x + 2, y + h - fh), (x + w - 2, y + h - 2), color, -1)


def _hbar(img, x, y, w, h, value, color):
    v = float(np.clip(value, 0.0, 1.0))
    cv2.rectangle(img, (x, y), (x + w, y + h), GRAY, 1)
    fw = int(round(w * v))
    if fw > 0:
        cv2.rectangle(img, (x + 2, y + 2), (x + fw - 2, y + h - 2), color, -1)


def _steer_bar(img, x, y, w, h, value):
    cv2.rectangle(img, (x, y), (x + w, y + h), GRAY, 1)
    cv2.line(img, (x + w // 2, y + 2), (x + w // 2, y + h - 2), GRAY, 1)
    off = int(round(np.clip(value, -1.0, 1.0) * (w // 2 - 6)))
    px = x + w // 2 + off
    cv2.rectangle(img, (px - 4, y + 2), (px + 4, y + h - 2), BLUE, -1)


def _g_meter(img, cx, cy, scale, g_lat, g_lon):
    for g, col in ((0.5, GRAY), (1.0, (90, 100, 120)), (2.0, (45, 52, 68))):
        r = int(scale * g)
        cv2.circle(img, (cx, cy), r, col, 1)
    cv2.line(img, (cx - int(scale * 2.2), cy), (cx + int(scale * 2.2), cy), GRAY, 1)
    cv2.line(img, (cx, cy - int(scale * 2.2)), (cx, cy + int(scale * 2.2)), GRAY, 1)
    dx = int(np.clip(g_lat, -2.0, 2.0) * scale)
    dy = int(np.clip(-g_lon, -2.0, 2.0) * scale)  # +lon (braking) -> up
    px, py = cx + dx, cy + dy
    cv2.circle(img, (px, py), 9, GCOL, -1)
    cv2.circle(img, (px, py), 9, (10, 10, 14), 1)
    cv2.putText(img, "G", (cx + int(scale * 2.2) - 14, cy + int(scale * 2.2) - 10),
                FONT, 0.5, GRAY, 1)


def render_hud(data, cam=None, fps: float = 0.0, stale: bool | None = None) -> np.ndarray:
    """Render one HUD frame from a telemetry dict (and optional camera image)."""
    img = np.full((H, W, 3), BG, np.uint8)

    cv2.putText(img, "BEAMNG AUTOPILOT - LIVE TELEMETRY", (20, 36), FONT, 0.7, WHITE, 2)
    cv2.putText(img, f"HUD {fps:.0f} fps", (W - 220, 36), FONT, 0.55, GRAY, 1)

    if cam is not None:
        prev = cv2.resize(cam, (CAM_W, CAM_H), interpolation=cv2.INTER_AREA)
        img[80:80 + CAM_H, 20:20 + CAM_W] = prev
        cv2.rectangle(img, (20, 80), (20 + CAM_W, 80 + CAM_H), (70, 76, 88), 1)
        cv2.putText(img, "FRONT CAM", (30, 106), FONT, 0.6, CYAN, 2)
        panel_x = 840
    else:
        panel_x = (W - PANEL_W) // 2
    cx = panel_x + PANEL_W // 2

    if stale is None:
        stale = data is None
    status, scol = ("WAITING FOR TELEMETRY...", RED) if stale else ("LIVE", GREEN)
    cv2.putText(img, status, (W - 220, 60), FONT, 0.55, scol, 2)

    if data is None:
        cv2.putText(img, "Start a driving / collect script to publish data.",
                    (W // 2 - 300, H // 2), FONT, 0.7, GRAY, 2)
        return img

    speed_kph = data["speed"] * 3.6

    # big speed number
    cv2.putText(img, "SPEED", (cx - 32, 112), FONT, 0.6, GRAY, 1)
    sp_text = f"{speed_kph:3.0f}"
    (tw, _), _ = cv2.getTextSize(sp_text, cv2.FONT_HERSHEY_DUPLEX, 2.6, 6)
    cv2.putText(img, sp_text, (cx - tw // 2, 222), cv2.FONT_HERSHEY_DUPLEX, 2.6, WHITE, 6)
    cv2.putText(img, "km/h", (cx + tw // 2 + 10, 206), FONT, 0.7, GRAY, 2)

    # speed bar (0-180 km/h)
    _hbar(img, cx - 150, 252, 300, 12, speed_kph / 180.0, CYAN)

    # throttle / brake
    _vbar(img, cx - 108, 294, 32, 92, data["throttle"], GREEN)
    _vbar(img, cx + 76, 294, 32, 92, data["brake"], RED)
    cv2.putText(img, f"THROTTLE {data['throttle']:.2f}", (cx - 108, 400), FONT, 0.5, WHITE, 1)
    cv2.putText(img, f"BRAKE {data['brake']:.2f}", (cx + 40, 400), FONT, 0.5, WHITE, 1)

    # steering
    cv2.putText(img, "STEER", (cx - 150, 432), FONT, 0.55, GRAY, 1)
    _steer_bar(img, cx - 150, 440, 300, 18, data["steer"])
    cv2.putText(img, f"{data['steer']:+.2f}", (cx + 160, 455), FONT, 0.55, BLUE, 1)

    # G-force meter
    _g_meter(img, cx, 565, 40, data.get("g_lat", 0.0), data.get("g_lon", 0.0))

    # footer
    heading = data.get("heading")
    htxt = f"{np.degrees(heading):.0f}" if heading is not None else "-"
    footer = (f"t +{data['t']:.1f}s   HEADING {htxt} deg   "
              f"LAP {data.get('lap', '-')}   NEAREST {data.get('nearest', '-')}   "
              f"LAT {data.get('g_lat', 0.0):+.2f} g   LON {data.get('g_lon', 0.0):+.2f} g")
    cv2.putText(img, footer, (20, 692), FONT, 0.55, GRAY, 1)

    extra = data.get("extra")
    if extra:
        ex = "   ".join(f"{k.upper()} {v}" for k, v in extra.items())
        cv2.putText(img, ex, (20, 668), FONT, 0.55, CYAN, 1)
    return img


class LiveHUD:
    """In-process dashboard window; call update() once per control loop."""

    def __init__(self, name: str = "BeamNG Autopilot - Telemetry", show_camera: bool = True):
        self.name = name
        self.show_camera = show_camera
        self._t0 = time.time()
        self._frames = 0
        self._last_ok = 0.0

    def update(self, data, cam=None) -> bool:
        """Render and show one frame. Returns False when q / ESC is pressed."""
        self._frames += 1
        fps = self._frames / max(time.time() - self._t0, 1e-6)
        if data is not None:
            self._last_ok = time.time()
        stale = time.time() - self._last_ok > 3.0
        img = render_hud(data, cam if self.show_camera else None, fps, stale)
        cv2.imshow(self.name, img)
        key = cv2.waitKey(1) & 0xFF
        return key not in (ord("q"), 27)

    def close(self) -> None:
        try:
            cv2.destroyWindow(self.name)
        except cv2.error:
            pass
