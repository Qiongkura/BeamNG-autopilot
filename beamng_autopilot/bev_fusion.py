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
        """Add one camera's votes into the map (log-odds accumulation).

        Vectorised: cells for all points are computed at once and hit
        counts are folded in per cell.  Sequential clamped updates with a
        CONSTANT per-feature delta equal one clipped linear step per cell
        (the accumulators are always kept inside [-8, 8], so intermediate
        clamps can never bind), which is what the scalar loop produced.
        """
        ch = feature.channel if feature.channel in self.CHANNELS \
            else "obstacle"
        pts = np.asarray(feature.points, dtype=float).reshape(-1, 3)
        if not len(pts):
            return
        ex = pts[:, 0]
        ey = pts[:, 1]
        inside = (np.abs(ex) < self.extent) & (np.abs(ey) < self.extent)
        if not inside.any():
            return
        r = ((self.extent - ex[inside]) / self.res).astype(np.int64)
        c = ((self.extent - ey[inside]) / self.res).astype(np.int64)
        ok = (r >= 0) & (r < self.n) & (c >= 0) & (c < self.n)
        if not ok.any():
            return
        flat = r[ok] * self.n + c[ok]
        hits = np.bincount(
            flat, minlength=self.n * self.n).reshape(self.n, self.n)
        lodds = self.logodds[ch]
        src = self.sources[ch]
        hit = hits > 0
        # log-odds update: p/(1-p); a hit adds confidence^...; keep a
        # bounded running log-odds so a lone false positive fades.
        delta = 2.0 * (feature.confidence - 0.5)
        lodds[hit] = np.clip(lodds[hit] + delta * hits[hit], -8.0, 8.0)
        src[hit] = np.minimum(
            65535, src[hit].astype(np.int64) + hits[hit]).astype(np.uint16)

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

    Fully vectorised (meshgrid over the sampled mask pixels); the
    per-pixel ray math is identical to the original scalar loop.
    """
    h, w = mask.shape[:2]
    C, r_vec, f_vec, u_vec = cam.camera_pose(pos, heading)
    fx, fy = cam.fx, cam.fy
    cx, cy = cam.cx, cam.cy
    ch = math.cos(float(heading))
    sh = math.sin(float(heading))
    us = np.arange(0, w, step)
    vs = np.arange(0, h, step)
    uu, vv = np.meshgrid(us, vs)
    ok = np.asarray(mask[vv, uu], dtype=bool)
    xc = (uu - cx) / fx
    yc = (vv - cy) / fy
    D = xc[..., None] * r_vec - yc[..., None] * u_vec + f_vec
    Dz = D[..., 2]
    ok &= Dz < -1e-6
    t = np.zeros_like(Dz)
    np.divide(ground_z - C[2], Dz, out=t, where=ok)
    good = ok & (t > 0.0) & (t <= max_ahead_m)
    wx = C[0] + t * D[..., 0]
    wy = C[1] + t * D[..., 1]
    ex = (wx - float(pos[0])) * ch + (wy - float(pos[1])) * sh
    ey = -(wx - float(pos[0])) * sh + (wy - float(pos[1])) * ch
    good &= ex * ex + ey * ey <= max_ahead_m * max_ahead_m
    if not good.any():
        return []
    pts = np.column_stack([ex[good], ey[good], np.zeros(int(good.sum()))])
    return [pts]


def stamp_signal_bearing(fmap: BEVFeatureMap, cam, px, confidence: float,
                         role: str = "front_main",
                         d_lo_m: float = 6.0, d_hi_m: float = 24.0,
                         step_m: float = 2.0) -> None:
    """Stamp a detected traffic lamp into the ``sign`` channel.

    A signal lamp hangs OVERHEAD, so a ground-plane back-projection of
    its pixel is meaningless (the ray lands tens of metres past the
    pole).  The honest spatial read a single front camera can give is
    the lamp's BEARING (from the pixel column through the pinhole) plus
    the fact that a signal exists somewhere ahead of it: the lamp is
    stamped as a distance band along that bearing, so vector space
    carries "traffic control ahead at this bearing" for the planner and
    the E2E input instead of a dead channel.
    """
    u = (float(px[0]) - float(cam.cx)) / float(cam.fx)
    ds = np.arange(float(d_lo_m), float(d_hi_m) + 0.5 * float(step_m),
                   float(step_m))
    # ego frame: x forward, y LEFT positive - a lamp right of the image
    # centre (u > 0) must land on negative-y cells.
    pts = np.column_stack([ds, -ds * u, np.zeros_like(ds)])
    fmap.accumulate(CameraFeature(role, "sign", pts, confidence=confidence))


def fuse_front_frame_vector_space(
        masks, cam, pos, heading, ground_z: float = 0.0,
        obstacles=(), n: int = 60, res: float = 0.5,
        step: int = 6, max_ahead_m: float = 40.0) -> BEVFeatureMap | None:
    """Fuse one front-camera semantic frame + LiDAR into a vector-space map.

    Mirrors the ``FSDStack.tick`` fusion block so shadow-mode recording
    produces the same (obstacle / drivable / lane / sign) channel stack
    the live FSD drive records from ``out.feature_map``: the semantic
    ROAD mask votes into ``drivable``, the painted LINE mask into
    ``lane`` (both back-projected to ego via ``project_mask_to_ego``),
    and the LiDAR obstacle points into ``obstacle``.  ``sign`` stays
    neutral here (the shadow recorder has no sign head yet; the live
    FSDStack.tick stamps it via ``stamp_signal_bearing``).

    ``masks`` is a dict-like of semantic boolean masks (``road`` /
    ``line``); ``obstacles`` an iterable of objects with ``.x`` / ``.y``
    (range-provider hits).  Returns a fresh ego-centred map, or None when
    nothing was perceived this frame.
    """
    fmap = BEVFeatureMap(n=n, res=res)
    seen = False
    if masks:
        if "road" in masks:
            pts = project_mask_to_ego(
                masks["road"], cam, pos, heading, ground_z=ground_z,
                channel="drivable", step=step, max_ahead_m=max_ahead_m)
            for p in pts:
                fmap.accumulate(CameraFeature(
                    "front_main", "drivable", p, confidence=0.7))
            if pts:
                seen = True
        if "line" in masks:
            pts = project_mask_to_ego(
                masks["line"], cam, pos, heading, ground_z=ground_z,
                channel="lane", step=step, max_ahead_m=max_ahead_m)
            for p in pts:
                fmap.accumulate(CameraFeature(
                    "front_main", "lane", p, confidence=0.75))
            if pts:
                seen = True
    obs = [(float(o.x), float(o.y), 0.0) for o in (obstacles or ())]
    if obs:
        fmap.accumulate(CameraFeature(
            "sensor_fusion", "obstacle",
            np.asarray(obs, dtype=float).reshape(-1, 3), confidence=0.85))
        seen = True
    return fmap if seen else None
