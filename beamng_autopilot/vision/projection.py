"""Pinhole camera model calibrated for the front camera (Steam / Tech).

The calibration run (scripts/m2_calibrate_camera.py) queried the in-game
camera pose/FOV via Lua while the vehicle sat at known positions:
  * camera local offset: (0, 1.216, 1.386) in (right, forward, up)
  * camera looks forward with a slight downward pitch (~-0.011 on z)
  * vertical FOV 65 deg

Real-vehicle logic (Tesla / Xpeng style): the sensor extrinsics are fixed
by calibration, and at runtime only the vehicle's full 6-DOF pose (the
rotation quaternion) is fed in.  ``CameraModel.camera_pose`` therefore
accepts an optional ``rotation`` quaternion; when it is absent the legacy
yaw-only path is used so old callers keep working unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def quat_to_rot(q) -> np.ndarray:
    """(x, y, z, w) quaternion -> 3x3 world rotation matrix.

    Columns of the returned matrix are the local (right, fwd, up) axes
    expressed in world coordinates, matching the R built from yaw in the
    legacy path.  BeamNG reports vehicle rotation as (x, y, z, w) with w
    the real part.
    """
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),
         2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z),
         2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w),
         1.0 - 2.0 * (x * x + y * y)],
    ])


@dataclass
class CameraModel:
    offset: np.ndarray  # camera position in vehicle local frame (right, fwd, up)
    fwd_local: np.ndarray  # camera forward in vehicle local frame
    up_local: np.ndarray  # camera up in vehicle local frame
    fov_deg: float
    width: int
    height: int

    @property
    def fx(self) -> float:
        return (self.height / 2.0) / np.tan(np.deg2rad(self.fov_deg) / 2.0)

    @property
    def fy(self) -> float:
        return self.fx

    @property
    def cx(self) -> float:
        return self.width / 2.0

    @property
    def cy(self) -> float:
        return self.height / 2.0

    def camera_pose(self, pos, heading: float, rotation=None):
        """World-space camera pose from vehicle state.

        Returns (C, right, fwd, up) where right/fwd/up are the camera axes.

        ``rotation`` is an optional (x, y, z, w) vehicle quaternion.  When
        given, the full 6-DOF pose rotates the calibrated extrinsics, so
        slopes (pitch/roll) are handled correctly; without it the legacy
        yaw-only path is used (flat-ground assumption).
        """
        p = np.asarray(pos, dtype=float)
        if rotation is None:
            h = float(heading)
            right = np.array([np.sin(h), -np.cos(h), 0.0])
            fwd = np.array([np.cos(h), np.sin(h), 0.0])
            up = np.array([0.0, 0.0, 1.0])
            R = np.column_stack([right, fwd, up])
        else:
            # BeamNG rotation quaternions (x, y, z, w) follow the inverse
            # (conjugate) convention and map the vehicle-local y axis to
            # the *backward* direction.  Verified against live vehicle
            # state: the CameraModel rotation is
            #   quat_to_rot(x, y, z, -w) @ diag(1, -1, 1)
            # whose columns are the world (right, fwd, up) axes.
            q = np.asarray(rotation, dtype=float)
            R = (quat_to_rot(np.array([q[0], q[1], q[2], -q[3]]))
                 @ np.diag([1.0, -1.0, 1.0]))
        C = p + R @ np.asarray(self.offset, dtype=float)
        f = R @ np.asarray(self.fwd_local, dtype=float)
        u = R @ np.asarray(self.up_local, dtype=float)
        f = f / np.linalg.norm(f)
        u = u / np.linalg.norm(u)
        r = np.cross(f, u)
        r = r / np.linalg.norm(r)
        return C, r, f, u

    def project(self, world_points, pos, heading: float):
        """Project world points into the image. Returns (u, v, valid)."""
        C, r, f, u_axis = self.camera_pose(pos, heading)
        pts = np.atleast_2d(np.asarray(world_points, dtype=float))
        d = pts - C
        x = d @ r
        depth = d @ f
        z = d @ u_axis
        valid = depth > 0.1
        u = np.full(len(pts), np.nan)
        v = np.full(len(pts), np.nan)
        u[valid] = self.cx + self.fx * x[valid] / depth[valid]
        v[valid] = self.cy - self.fy * z[valid] / depth[valid]
        return u, v, valid

    def ground_row(self, ahead_m: float, pos, heading: float) -> float:
        """Image row where the ground at `ahead_m` metres in front of the
        vehicle appears (z = 0 plane)."""
        h = float(heading)
        fwd = np.array([np.cos(h), np.sin(h), 0.0])
        point = np.asarray(pos, dtype=float) + ahead_m * fwd
        _, _, v, _ = self._project_one(point, pos, heading)
        return float(v)

    def _project_one(self, world_point, pos, heading: float):
        u, v, valid = self.project(np.asarray([world_point]), pos, heading)
        return u[0], v[0], valid[0]


def default_camera(width: int = 1076, height: int = 806) -> CameraModel:
    return CameraModel(
        offset=np.array([0.0, 1.216, 1.386]),
        fwd_local=np.array([0.0, 0.99994, -0.0112]),
        up_local=np.array([0.0, 0.0112, 0.99994]),
        fov_deg=65.0,
        width=width,
        height=height,
    )
