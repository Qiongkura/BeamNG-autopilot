"""Multi-camera -> BEV feature fusion (FSD vector-space construction).

Tesla FSD lifts each camera's 2D features into a shared bird's-eye view
with a learned BEV transformer / coordinate grid - every camera votes
into the same ego-centred cells, so the network can reason about the
world spatially no matter which camera saw a given object.  This module
provides the same *shape* on top of the geometric infrastructure:

* ``CameraFeature``: one camera's contribution - its detections/masks
  already back-projected into ego frame as sparse world points with a
  per-point confidence.
* ``BEVFeatureMap``: the fused ego-centred map.  Each cell accumulates
  evidence from every camera (log-odds style), keeping per-source
  channels (obstacle, drivable, lane, sign) like the FSD vector space's
  semantic channel stack.
* ``fuse_camera_features``: the fusion primitive - projects each
  camera's ego points into cells and accumulates; later this is the
  slot where a learned cross-attention would live.

Everything is game-free + unit-testable; the weights/attention args are
there so the call contract matches a future learned fusion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class CameraFeature:
    """One camera's back-projected contribution to vector space.

    ``points`` is (M, 3) ego-frame (x forward, y left, z up) hits from
    that camera (detection corners, mask samples, ...).  ``channel`` is
    the semantic kind ("obstacle" / "drivable" / "lane" / "sign").
    ``confidence`` 0..1.
    """

    role: str
    channel: str
    points: np.ndarray
    confidence: float = 0.8

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=float).reshape(-1, 3)
        self.confidence = float(np.clip(self.confidence, 0.0, 1.0))


class BEVFeatureMap:
    """Ego-centred fused feature map (the vector space).

    Cells accumulate log-odds evidence per channel.  ``get(channel)``
    returns the current probability raster for a semantic channel so a
    planner or a teaching signal can consume the fused space directly.
    """

    CHANNELS = ("obstacle", "drivable", "lane", "sign")

    def __init__(self, n: int = 60, res: float = 0.5):
        self.n = int(n)
        self.res = float(res)
        self.extent = 0.5 * n * res
        # log-odds accumulators per channel
        self.logodds: dict[str, np.ndarray] = {
            ch: np.zeros((n, n), dtype=np.float32) for ch in self.CHANNELS
        }
        self.sources: dict[str, np.ndarray] = {
            ch: np.zeros((n, n), dtype=np.uint16) for ch in self.CHANNELS
        }

    # ------------------------------------------------------------------
    def _cell(self, ex: float, ey: float) -> tuple[int, int] | None:
        if abs(ex) >= self.extent or abs(ey) >= self.extent:
            return None
        x = float(ex)
        y = float(ey)
        r = int((self.extent - x) / self.res)
        c = int((self.extent - y) / self.res)
        if not (0 <= r < self.n and 0 <= c < self.n):
            return None
        return r, c

    def accumulate(self, feature: CameraFeature) -> None:
        """Add one camera's votes into the map (log-odds accumulation)."""
        ch = feature.channel if feature.channel in self.CHANNELS \
            else "obstacle"
        lodds = self.logodds[ch]
        src = self.sources[ch]
        for px, py, _ in feature.points:
            cell = self._cell(float(px), float(py))
            if cell is None:
                continue
            r, c = cell
            # log-odds update: p/(1-p); a hit adds confidence^...; keep a
            # bounded running log-odds so a lone false positive fades.
            lodds[r, c] = max(-8.0, min(8.0, lodds[r, c] + 2.0 *
                                        (feature.confidence - 0.5)))
            src[r, c] = np.uint16(min(65535, int(src[r, c]) + 1))

    def get(self, channel: str) -> np.ndarray:
        """Probability 0..1 raster for a channel (sigmoid of log-odds)."""
        if channel not in self.logodds:
            return np.zeros((self.n, self.n), dtype=np.float32)
        return 1.0 / (1.0 + np.exp(-self.logodds[channel]))

    def raster(self) -> np.ndarray:
        """Combined obstacle evidence (the occupancy input a planner uses)."""
        return self.get("obstacle")

    # ------------------------------------------------------------------
    def clear(self) -> None:
        for ch in self.CHANNELS:
            self.logodds[ch].fill(0.0)
            self.sources[ch].fill(0)


def fuse_camera_features(features, n: int = 60, res: float = 0.5,
                         attention_weights=None) -> BEVFeatureMap:
    """Fuse a list of ``CameraFeature`` into one ``BEVFeatureMap``.

    ``attention_weights`` is accepted for interface parity with a future
    learned cross-attention layer (a ``{role: weight}`` dict that scales
    each camera's confidence); when None all cameras get equal weight.
    Returns the fused map.
    """
    fmap = BEVFeatureMap(n=n, res=res)
    for feat in features:
        w = 1.0
        if attention_weights is not None:
            w = float(attention_weights.get(feat.role, 1.0))
        feat.confidence *= max(0.0, min(1.0, w))
        fmap.accumulate(feat)
    return fmap


def project_mask_to_ego(mask, cam, pos, heading, ground_z: float = 0.0,
                        channel: str = "obstacle",
                        step: int = 6, max_ahead_m: float = 45.0,
                        ) -> list[np.ndarray]:
    """Back-project a 2D bool mask into ego-frame 3D points.

    A convenience that mirrors ``project_road_mask_to_grid`` but returns
    ego coords so many cameras can be fused by ``fuse_camera_features``.
    Returns a list of (M, 3) arrays - one per role, ready to be wrapped
    in a ``CameraFeature``.
    """
    import numpy as _np
    h, w = mask.shape[:2]
    C, r_vec, f_vec, u_vec = cam.camera_pose(pos, heading)
    samples = []
    fx, fy = cam.fx, cam.fy
    cx, cy = cam.cx, cam.cy
    ch = math.cos(float(heading))
    sh = math.sin(float(heading))
    for v in range(0, h, step):
        for u in range(0, w, step):
            if not mask[v, u]:
                continue
            x_cam = (u - cx) / fx
            y_cam = (v - cy) / fy
            D = (x_cam * r_vec + y_cam * -u_vec + 1.0 * f_vec)
            if D[2] >= -1e-6:
                continue
            t = (float(ground_z) - float(C[2])) / float(D[2])
            if not (0.0 < t <= max_ahead_m):
                continue
            wx = float(C[0]) + t * float(D[0])
            wy = float(C[1]) + t * float(D[1])
            ex = (wx - float(pos[0])) * ch + (wy - float(pos[1])) * sh
            ey = -(wx - float(pos[0])) * sh + (wy - float(pos[1])) * ch
            if ex * ex + ey * ey > max_ahead_m * max_ahead_m:
                continue
            samples.append((ex, ey, 0.0))
    if not samples:
        return []
    return [np.asarray(samples, dtype=float).reshape(-1, 3)]