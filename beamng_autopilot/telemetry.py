"""Live telemetry broadcast for the realtime dashboard (m4_dashboard.py).

Driving / collect scripts call TelemetryBroadcaster.publish() once per control
loop; a background thread writes the latest snapshot to
logs/telemetry/live.json so a separate dashboard process can render it without
blocking the control loop.  G-forces are derived from the velocity vector and
low-pass filtered.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np

from . import config

G = 9.81
SMOOTH = 0.35  # EMA alpha for acceleration smoothing at ~15 Hz updates


def _json_vec(v) -> list | None:
    """Convert a 3D numpy vector to a JSON-safe rounded list."""
    if v is None:
        return None
    try:
        arr = np.asarray(v, dtype=float)
        if arr.ndim != 1 or arr.size < 3:
            return None
        return [round(float(x), 3) for x in arr[:3]]
    except Exception:
        return None


def read_live(path=None) -> dict | None:
    """Read the latest telemetry snapshot written by a broadcaster."""
    p = Path(path or (config.LOGS_DIR / "telemetry" / "live.json"))
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


class GForceFilter:
    """EMA low-pass filter converting velocity deltas into lateral/longitudinal G."""

    def __init__(self, smooth: float = SMOOTH):
        self.smooth = smooth
        self._prev_vel = None
        self._prev_t = None
        self.g_lat = 0.0
        self.g_lon = 0.0

    def update(self, t, vel, dir_vec, up_vec):
        """Feed one sample; returns the smoothed (g_lat, g_lon)."""
        if vel is not None and dir_vec is not None and up_vec is not None:
            if self._prev_vel is not None and self._prev_t is not None and t - self._prev_t > 1e-4:
                acc = (np.asarray(vel, float) - np.asarray(self._prev_vel, float)) / (t - self._prev_t)
                fwd = np.asarray(dir_vec, float)
                right = np.cross(fwd, np.asarray(up_vec, float))
                lon = float(np.dot(acc, fwd)) / G
                lat = float(np.dot(acc, right)) / G
                self.g_lat = self.smooth * lat + (1.0 - self.smooth) * self.g_lat
                self.g_lon = self.smooth * lon + (1.0 - self.smooth) * self.g_lon
        self._prev_vel = None if vel is None else np.asarray(vel, float)
        self._prev_t = t
        return self.g_lat, self.g_lon


class TelemetryBroadcaster:
    def __init__(self, path=None):
        self.path = Path(path or (config.LOGS_DIR / "telemetry" / "live.json"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._latest = None
        self._g = GForceFilter()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._writer, daemon=True)
        self._thread.start()

    @property
    def latest(self) -> dict | None:
        """Most recently published snapshot (for in-process HUD readers)."""
        with self._lock:
            return self._latest

    def _writer(self) -> None:
        while not self._stop.wait(0.1):
            with self._lock:
                data = self._latest
            if data is None:
                continue
            tmp = self.path.with_suffix(".tmp")
            try:
                tmp.write_text(json.dumps(data), encoding="utf-8")
                tmp.replace(self.path)
            except Exception:
                pass  # dashboard survives a failed write

    def publish(
        self,
        *,
        t: float,
        speed: float,
        throttle: float = 0.0,
        brake: float = 0.0,
        steer: float = 0.0,
        vel=None,
        dir_vec=None,
        up_vec=None,
        pos=None,
        heading: float | None = None,
        lap: int | None = None,
        nearest: int | None = None,
        extra: dict | None = None,
    ) -> None:
        with self._lock:
            g_lat, g_lon = self._g.update(t, vel, dir_vec, up_vec)

        data = {
            "t": round(float(t), 3),
            "speed": round(float(speed), 3),
            "throttle": round(float(throttle), 4),
            "brake": round(float(brake), 4),
            "steer": round(float(steer), 4),
            "g_lat": round(float(g_lat), 3),
            "g_lon": round(float(g_lon), 3),
            "heading": None if heading is None else round(float(heading), 4),
            "lap": lap,
            "nearest": nearest,
            "pos": [round(float(v), 3) for v in pos] if pos is not None else None,
            "vel": _json_vec(vel),
            "dir_vec": _json_vec(dir_vec),
            "up_vec": _json_vec(up_vec),
        }
        if extra:
            data["extra"] = extra
        with self._lock:
            self._latest = data

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        # Flush the latest snapshot synchronously so a caller that stops
        # immediately after the final publish still records that frame.
        with self._lock:
            data = self._latest
        if data is not None:
            tmp = self.path.with_suffix(".tmp")
            try:
                tmp.write_text(json.dumps(data), encoding="utf-8")
                tmp.replace(self.path)
            except Exception:
                pass
