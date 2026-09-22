"""One geometric baseline for every perception and telemetry consumer.

The round-5 plan (T05, §2.4) found the project measuring in three
different vertical references at once:

* the semantic/road BEV lift used the EGO ORIGIN plane (``pos.z``), which
  sits ``EGO_ORIGIN_GROUND_GAP_M`` above the road surface;
* the fisheye near-field, the pavement-edge candidate and ``line_lat``
  already used the ROAD plane (``pos.z - gap``);
* the pose fed to the camera models was yaw-only, so a pitched/rolled
  vehicle still projected as if it were level.

This module is the single definition of that baseline.  It is deliberately
tiny and dependency-free: the point is that ONE function answers "where is
the ground under the car" and ONE label says which attitude model produced
a number, so a disagreement between consumers is a code error rather than
a hidden constant difference.

Scope (and honesty): this is the FLAT-GROUND baseline.  The ground plane
is horizontal and the road surface is assumed to be that plane.  A slope
or a crest is NOT modelled here - see ``docs/GEOMETRY_BASELINE.md`` for the
error budget and the ODD statement; anything that relies on the lateral
reference on a slope must be validated separately.

Axis and unit conventions (project-wide, world frame):
``x`` east, ``y`` north, ``z`` UP, all metres; heading is the world
``atan2(y, x)`` direction of the vehicle's forward axis in radians.  The
camera frame used by ``CameraModel`` is (u = right, v = down, forward)
with ``fx == fy`` in pixels.
"""

from __future__ import annotations

import os

import numpy as np

from beamng_autopilot.config import (
    EGO_HALF_LENGTH_M,
    EGO_HALF_WIDTH_M,
    EGO_ORIGIN_GROUND_GAP_M,
)

#: Distance from the vehicle ORIGIN (what the simulator reports as ``pos``)
#: down to the road surface.  A vehicle constant, not a terrain model.
EGO_GROUND_GAP_M = float(EGO_ORIGIN_GROUND_GAP_M)

#: Vehicle footprint half-extents, metres (the shape every body check uses).
FOOTPRINT_HALF_LENGTH_M = float(EGO_HALF_LENGTH_M)
FOOTPRINT_HALF_WIDTH_M = float(EGO_HALF_WIDTH_M)

#: Name of the ground model every measurement in this baseline assumes.
GROUND_MODEL_FLAT = "flat_road_plane"

#: Use the unified road plane for ALL consumers (main view included).
#:
#: DEFAULT OFF, deliberately.  The road plane is the physically right one
#: (the ego origin sits ``EGO_GROUND_GAP_M`` above the road), but switching
#: the main-view lift to it changes which cells count as observed drivable
#: pavement, and that feeds the reference's on-pavement gate.  Measured on
#: one live pair (2026-09-22, same start pose, 14 s each): the lane
#: reference's drivable fraction p50 moved 0.75 -> 0.50 and one run left
#: two frames with no recorded fraction at all - a permission-relevant
#: change that a single pair cannot validate (plan §6/§8.2: any change to
#: reference authority needs a paired, interleaved A/B).  So the DEFINITION
#: is unified (every consumer calls :func:`projection_ground_z`) while the
#: VALUE stays legacy until that A/B is run; ``BEAMNG_GEOM_GROUND_PLANE=1``
#: enables the fix for it.
GROUND_PLANE_UNIFIED = os.environ.get("BEAMNG_GEOM_GROUND_PLANE", "0") != "0"

#: Feed the vehicle attitude (quaternion) into the camera pose instead of
#: a yaw-only rotation.  On flat ground with a level camera this is
#: numerically a no-op; on a slope or under body roll it is the difference
#: between the calibrated extrinsics and a lie.
POSE_ROTATION_ENABLED = os.environ.get("BEAMNG_GEOM_ROTATION", "1") != "0"


def ego_ground_z(pos) -> float:
    """World z of the ROAD PLANE under the vehicle.

    ``pos`` is the vehicle origin in world coordinates (x, y, z).  On flat
    ground the road surface is exactly ``EGO_GROUND_GAP_M`` below it.  On a
    slope this is still the horizontal plane through the contact patch
    directly under the origin - the flat-ground model does not tilt with
    the vehicle.
    """
    p = np.asarray(pos, dtype=float)
    if p.size < 3 or not np.isfinite(p[2]):
        return 0.0
    return float(p[2]) - EGO_GROUND_GAP_M


def projection_ground_z(pos, *, unified: bool | None = None) -> float:
    """The ground plane a projection call should use.

    ``unified=True`` (default, from ``GROUND_PLANE_UNIFIED``) returns the
    road plane for every camera; ``unified=False`` reproduces the legacy
    main-view behaviour (the ego-origin plane) so the two can be compared
    without editing call sites.
    """
    if unified is None:
        unified = GROUND_PLANE_UNIFIED
    if unified:
        return ego_ground_z(pos)
    p = np.asarray(pos, dtype=float)
    return float(p[2]) if p.size > 2 else 0.0


def pose_label(rotation, *, enabled: bool | None = None) -> str:
    """Which attitude model a pose came from (for telemetry).

    ``yaw_only`` means the camera was placed level: valid on flat ground,
    optimistic under pitch/roll.  ``quat_6dof`` means the vehicle
    quaternion rotated the calibrated extrinsics.  A label that does not
    match the numbers is exactly the kind of silent inconsistency T05
    exists to remove, so every tick publishes one.
    """
    if enabled is None:
        enabled = POSE_ROTATION_ENABLED
    if not enabled or rotation is None:
        return "yaw_only"
    try:
        q = np.asarray(rotation, dtype=float)
    except Exception:
        return "yaw_only"
    if q.shape != (4,) or not np.isfinite(q).all():
        return "yaw_only"
    return "quat_6dof"


def resolution_distance_m(fx_px: float, mark_width_m: float,
                          min_px: float = 2.0) -> float:
    """Furthest distance at which a marking still spans ``min_px`` pixels.

    A pinhole maps a lateral extent ``w`` at depth ``d`` to
    ``px = fx * w / d``, so inverting it gives the detection-distance
    budget the survey (L39: ``Np = C·d/w``) asks for.  ``min_px = 2`` is
    the smallest span that can carry a direction at all; a real detector
    needs substantially more, which is why the numbers here are an UPPER
    bound on usefulness, not a guarantee.
    """
    fx = float(fx_px)
    w = float(mark_width_m)
    m = float(min_px)
    if not (np.isfinite(fx) and np.isfinite(w) and np.isfinite(m)):
        return float("nan")
    if fx <= 0.0 or w <= 0.0 or m <= 0.0:
        return float("nan")
    return fx * w / m


def nearest_ground_distance_m(cam_model, pos, pitch_rad: float = 0.0,
                              ground_z: float | None = None) -> float | None:
    """Distance to the ground hit by the BOTTOM image row (blind zone).

    The forward camera cannot see closer than this: every pixel row above
    the last one maps further away.  Computed from the same camera model
    the pipeline uses, but analytically (ray -> horizontal plane), so it is
    independent of the pixel-sampling code path.  ``None`` when the bottom
    ray does not reach the ground plane (camera pointing up, or the plane
    is above the camera).
    """
    if cam_model is None:
        return None
    p = np.asarray(pos, dtype=float)
    gz = ego_ground_z(p) if ground_z is None else float(ground_z)
    C, r, f, u_axis = cam_model.camera_pose(p, 0.0)
    # Vehicle pitch: rotate the camera frame about its right axis.  A
    # positive pitch tilts the optical axis DOWN (towards the road), which
    # shortens the blind zone.
    pt = float(pitch_rad)
    if abs(pt) > 1e-12:
        cp, sp = np.cos(pt), np.sin(pt)
        f2 = f * cp + u_axis * (-sp)
        u2 = u_axis * cp + f * sp
        f, u_axis = f2 / np.linalg.norm(f2), u2 / np.linalg.norm(u2)
    v_bottom = float(cam_model.height) - 1.0
    d = (r * ((0.0 - cam_model.cx) / cam_model.fx)
         + f + u_axis * ((cam_model.cy - v_bottom) / cam_model.fy))
    dz = float(d[2])
    if dz >= -1e-9:
        return None
    t = (gz - float(C[2])) / dz
    if t <= 0.0:
        return None
    dx, dy = float(C[0] + t * d[0]) - float(p[0]), float(C[1] + t * d[1]) - float(p[1])
    return float(np.hypot(dx, dy))
