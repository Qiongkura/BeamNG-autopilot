"""Runtime-neutral sensor providers for Steam and BeamNG.tech builds."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config
from .perception import Obstacle, scan_obstacles_all
from .vision.detection import live_camera_model
from .vision.projection import CameraModel


@dataclass
class RangeSample:
    """Obstacles plus the raw 2D ray/lidar hits from one range scan."""

    obstacles: list[Obstacle] = field(default_factory=list)
    ray_hits: list[tuple[float, float]] = field(default_factory=list)


class CameraProvider:
    """Returns front-camera frames and the matching CameraModel."""

    def grab(self) -> np.ndarray:
        raise NotImplementedError

    def camera_model(self, pos, heading, width, height,
                     fallback: CameraModel | None = None,
                     rotation=None) -> CameraModel:
        """Camera model for the latest frame.

        ``rotation`` is the optional (x, y, z, w) vehicle quaternion: when
        given, the calibrated extrinsics are rotated by the full 6-DOF
        pose (real-vehicle sensor-fusion logic); without it the provider
        falls back to its own pose source (yaw-only or live camera query).
        """
        raise NotImplementedError

    def close(self) -> None:
        return None


class RangeProvider:
    """Returns merged obstacle boxes and raw ray hits for one scan."""

    def scan(self, pos, ego_vid=None, radius: float = 55.0) -> RangeSample:
        raise NotImplementedError

    def close(self) -> None:
        return None


class SteamCameraProvider(CameraProvider):
    """Steam runtime: existing window/Lua screen capture plus live camera."""

    def __init__(self, conn) -> None:
        self.conn = conn

    def grab(self) -> np.ndarray:
        return self.conn.grab_screen()

    def camera_model(self, pos, heading, width, height,
                     fallback: CameraModel | None = None,
                     rotation=None) -> CameraModel:
        # Steam path: the frame shows whatever the player's camera sees
        # (free-look included), so the live Lua camera query stays the
        # pose source; ``rotation`` is ignored here on purpose.
        with self.conn.io_lock:
            return live_camera_model(
                self.conn.bng, int(width), int(height), pos, heading,
                fallback=fallback)

    def close(self) -> None:
        return None


class SteamRangeProvider(RangeProvider):
    """Steam runtime: Lua raycast plus scenario/vehicle obstacle sources."""

    def __init__(self, conn) -> None:
        self.conn = conn

    def scan(self, pos, ego_vid=None, radius: float = 55.0) -> RangeSample:
        with self.conn.io_lock:
            obstacles, hits = scan_obstacles_all(
                self.conn.bng, ego_vid, pos, radius=radius, return_hits=True)
        return RangeSample(obstacles=obstacles, ray_hits=hits)

    def close(self) -> None:
        return None


def resolve_runtime(conn, mode: str | None = None) -> str:
    """Resolve auto to steam/tech after the BeamNGpy connection exists."""
    mode = (mode or config.RUNTIME_MODE).lower()
    if mode != "auto":
        return mode
    if getattr(conn.bng, "tech_enabled", lambda: None)() is True:
        return "tech"
    return "steam"


def build_camera_provider(conn, mode, width: int = 1076, height: int = 806,
                          annotations: bool = False):
    """Return (CameraProvider, resolved runtime) for the connected session.

    ``annotations`` enables Tech annotation rendering (pixel truth) on the
    camera; it is ignored on Steam.
    """
    mode = resolve_runtime(conn, mode)
    if mode == "tech":
        from beamng_autopilot_tech.providers import TechCameraProvider

        return TechCameraProvider(conn, width, height,
                                  annotations=annotations), mode
    return SteamCameraProvider(conn), mode


def build_range_provider(conn, mode):
    """Return (RangeProvider, resolved runtime) for the connected session."""
    mode = resolve_runtime(conn, mode)
    if mode == "tech":
        from beamng_autopilot_tech.providers import TechRangeProvider

        return TechRangeProvider(conn), mode
    return SteamRangeProvider(conn), mode


def build_camera_ring_provider(conn, mode, width: int = 1076,
                               height: int = 806):
    """Return a multi-camera ring provider, or None on Steam.

    The FSD-style surround camera ring is a Tech capability (beamngpy
    Camera sensors); on Steam only the front camera exists, so callers
    fall back to ``build_camera_provider``.
    """
    mode = resolve_runtime(conn, mode)
    if mode == "tech":
        from beamng_autopilot_tech.providers import TechCameraRingProvider

        return TechCameraRingProvider(conn, width, height), mode
    return None, mode
