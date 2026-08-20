"""Multi-camera surround ring, aligned with the Tesla FSD camera layout.

FSD (AI Day 2021) perceives with eight cameras around the car: a front
main/narrow pair looking far ahead, a front fisheye covering the side
blind zones, left/right B-pillar cameras, left/right rear-wing mirrors,
and a rear camera.  Every camera is a fixed extrinsic calibration (its
pose relative to the vehicle) - at runtime only the vehicle's full
6-DoF pose is fed in.

This module is the pure, game-free definition of that ring:

* ``CameraMount``: one camera's calibration (offset + local axes + FOV),
  which is exactly what ``CameraModel`` needs.
* ``CAMERA_RING``: the default eight-mount layout (FSD-style roles).
* ``camera_ring_models()``: the matching ``CameraModel`` list/registry.

The Steam front-only path stays the fallback; this ring is the Tech-side
perception source.  Nothing here touches the game.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .projection import CameraModel

# Camera roles (FSD nomenclature, ordered around the car).
FRONT_MAIN = "front_main"
FRONT_NARROW = "front_narrow"
FRONT_FISHEYE = "front_fisheye"
PILLAR_LEFT = "pillar_left"
PILLAR_RIGHT = "pillar_right"
REAR_LEFT = "rear_left"
REAR_RIGHT = "rear_right"
REAR = "rear"

RING_ROLES = (
    FRONT_MAIN,
    FRONT_NARROW,
    FRONT_FISHEYE,
    PILLAR_LEFT,
    PILLAR_RIGHT,
    REAR_LEFT,
    REAR_RIGHT,
    REAR,
)

RING_DEF = RING_ROLES


@dataclass(frozen=True)
class CameraMount:
    """One camera in the ring.

    ``offset`` / ``fwd_local`` / ``up_local`` are in the vehicle's local
    frame using the (right, forward, up) axis convention of
    ``CameraModel`` (forward = the direction the front camera looks),
    matching the calibrated front-camera extrinsics so slopes and yaw
    are handled the same way.
    """

    role: str
    # Pan angle around the vertical axis, 0 = vehicle forward, positive to
    # the right, degrees.  Pure documentation/telemetry; the actual pointing
    # comes from ``fwd_local``.
    pan_deg: float
    # Vertical FOV in degrees.
    fov_deg: float
    offset: np.ndarray
    fwd_local: np.ndarray
    up_local: np.ndarray

    def camera_model(self, width: int = 1076, height: int = 806) -> CameraModel:
        return CameraModel(
            offset=np.asarray(self.offset, dtype=float),
            fwd_local=np.asarray(self.fwd_local, dtype=float),
            up_local=np.asarray(self.up_local, dtype=float),
            fov_deg=self.fov_deg,
            width=width,
            height=height,
        )


def _fwd(pan_deg: float, pitch_deg: float = 0.0) -> np.ndarray:
    """Local forward vector from a horizontal pan and a vertical pitch.

    pan 0 -> straight ahead, +90 -> the vehicle's right (the +x local
    axis), +180 -> straight back, matching the CameraModel axis
    convention used by ``project``.
    """
    r = math.radians(pan_deg)
    p = math.radians(pitch_deg)
    return np.array([
        math.sin(r) * math.cos(p),
        math.cos(r) * math.cos(p),
        math.sin(p),
    ])


def _mount(role: str, pan_deg: float, fov_deg: float,
           fx_offset: float, fy_offset: float, z: float,
           pitch_deg: float = 0.0) -> CameraMount:
    """Build one mount with a level ``up_local`` axis."""
    return CameraMount(
        role=role,
        pan_deg=pan_deg,
        fov_deg=fov_deg,
        offset=np.array([fx_offset, fy_offset, z], dtype=float),
        fwd_local=_fwd(pan_deg, pitch_deg),
        up_local=np.array([0.0, 0.0, 1.0], dtype=float),
    )


# Default eight-mount ring.  Footprints are those of an ETK800-class sedan
# (about 1.9 m wide, 4.8 m long); offsets are rough but the pan/FOV are the
# calibrated part - the exact mm placement has negligible effect at this
# scale, exactly as with real sensor calibration.
CAMERA_RING: tuple[CameraMount, ...] = (
    # Front, looking far ahead at two magnifications (main + narrow) and a
    # fisheye that sees the sides of the car / corner blind zones.
    _mount(FRONT_MAIN, 0.0, 65.0, 0.0, 1.5, 1.4, pitch_deg=-2.0),
    _mount(FRONT_NARROW, 0.0, 35.0, 0.0, 1.5, 1.4, pitch_deg=-2.0),
    _mount(FRONT_FISHEYE, 0.0, 120.0, 0.0, 1.5, 1.35, pitch_deg=-2.0),
    # B-pillar / side cameras: see the lane to the side and slightly ahead.
    # pan is positive to the right, so the left camera has negative pan.
    _mount(PILLAR_LEFT, -118.0, 90.0, -0.9, 0.4, 1.45),
    _mount(PILLAR_RIGHT, 118.0, 90.0, 0.9, 0.4, 1.45),
    # Rear wing mirrors: cover the side-rear blind zones (outward + back).
    _mount(REAR_LEFT, -155.0, 90.0, -0.95, -1.4, 1.3),
    _mount(REAR_RIGHT, 155.0, 90.0, 0.95, -1.4, 1.3),
    # Rear window: straight backward.
    _mount(REAR, 180.0, 65.0, 0.0, -2.0, 1.35, pitch_deg=-2.0),
)


def camera_ring_models(width: int = 1076, height: int = 806) -> dict[str, CameraModel]:
    """Registry of ``{role: CameraModel}`` for the default ring."""
    return {m.role: m.camera_model(width, height) for m in CAMERA_RING}


def pan_deg_of(cam: CameraModel) -> float:
    """Horizontal pan of a camera model relative to vehicle forward.

    Computed from the local forward vector so tests and telemetry can
    tell which way each ring camera actually points (0 = forward,
    +90 = right, 180 = back, -90 = left).  Returns None when the forward
    vector has no horizontal component.
    """
    f = np.asarray(cam.fwd_local, dtype=float)[:2]
    n = float(np.linalg.norm(f))
    if n < 1e-9:
        return None
    return math.degrees(math.atan2(f[0], f[1]))