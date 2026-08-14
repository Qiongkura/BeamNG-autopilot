"""Pinhole camera model calibrated for the Steam-edition front camera.

The calibration run (scripts/m2_calibrate_camera.py) queried the in-game
camera pose/FOV via Lua while the vehicle sat at known positions:
  * camera local offset: (0, 1.216, 1.386) in (right, forward, up)
  * camera looks forward with a slight downward pitch (~-0.011 on z)
  * vertical FOV 65 deg
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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

    def camera_pose(self, pos, heading: float):
        """World-space camera pose from vehicle state.

        Returns (C, right, fwd, up) where right/fwd/up are the camera axes.
        """
        h = float(heading)
        right = np.array([np.sin(h), -np.cos(h), 0.0])
        fwd = np.array([np.cos(h), np.sin(h), 0.0])
        up = np.array([0.0, 0.0, 1.0])
        p = np.asarray(pos, dtype=float)
        C = p + self.offset[0] * right + self.offset[1] * fwd + self.offset[2] * up
        f = self.fwd_local[0] * right + self.fwd_local[1] * fwd + self.fwd_local[2] * up
        u = self.up_local[0] * right + self.up_local[1] * fwd + self.up_local[2] * up
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
