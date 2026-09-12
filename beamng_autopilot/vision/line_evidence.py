"""Multi-frame line-mask evidence accumulation in world space.

Adapts the spatio-temporal matching-fusion idea from 王俊《无人驾驶车辆
环境感知系统关键技术研究》ch3: back-project each frame's line mask onto
the ground plane, accumulate per-cell evidence across frames with ego
pose compensation, then re-project the accumulated evidence into the
current view and union it with the current mask.  A real marking seen on
>=2 frames survives brief per-frame dropouts; a single-frame false
positive (hits < HIT_MIN) is never propagated, so the fused mask is both
more complete and less jittery than any single frame - which is exactly
what the painted-lane lateral reference downstream needs
(docs/lateral_reference_diag_20260911.md: 5.3% of ticks jump >1.0 m).
"""

from __future__ import annotations

import time

import numpy as np

from .lanes import _back_project_many

CELL_M = 0.15        # world grid quantum; near-field px is 0.05-0.3 m
HIT_ADD = 1.0        # per-sighting evidence
HIT_CAP = 2.0
HIT_MIN = 1.5        # need ~2 sightings to propagate into later frames
DECAY_PER_S = 0.25   # linear decay; a 2-hit cell survives ~2 s unseen
MAX_AGE_S = 3.0
MAX_RANGE_M = 70.0
MAX_POINTS = 24000   # per-update back-projection budget


class LineEvidenceAccumulator:
    """World-space hit grid for line pixels, fused back into each frame."""

    def __init__(self, cell_m: float = CELL_M) -> None:
        self.cell_m = float(cell_m)
        # (ix, iy) -> [hits, last_seen_seconds]
        self._cells: dict[tuple[int, int], list] = {}
        self._last_t: float | None = None

    def reset(self) -> None:
        self._cells.clear()
        self._last_t = None

    def _key(self, x: float, y: float) -> tuple[int, int]:
        return (int(np.floor(x / self.cell_m)),
                int(np.floor(y / self.cell_m)))

    def _decay_and_prune(self, now: float, ego) -> None:
        if self._last_t is None:
            self._last_t = now
            return
        dt = float(now - self._last_t)
        self._last_t = now
        if dt <= 0.0:
            return
        ex, ey = float(ego[0]), float(ego[1])
        dead = []
        for k, rec in self._cells.items():
            rec[0] -= DECAY_PER_S * dt
            if (rec[0] <= 0.0 or now - rec[1] > MAX_AGE_S
                    or abs(k[0] * self.cell_m - ex) > MAX_RANGE_M
                    or abs(k[1] * self.cell_m - ey) > MAX_RANGE_M):
                dead.append(k)
        for k in dead:
            del self._cells[k]

    def update(self, line_mask: np.ndarray, cam_model, pos, heading: float,
               ground_z: float = 0.0, now: float | None = None) -> None:
        """Add at most one vote per cell per observation timestamp.

        Replay callers supply capture time; live callers default to the
        monotonic clock. A backwards timestamp starts fresh history.
        """
        now = time.monotonic() if now is None else float(now)
        if not np.isfinite(now):
            raise ValueError("timestamp must be finite")
        if self._last_t is not None:
            if now < self._last_t:
                self.reset()
            elif now == self._last_t:
                return  # Reprocessing one observation is not a new vote.
        self._decay_and_prune(now, pos)
        mask = np.asarray(line_mask, dtype=bool)
        if not mask.any():
            return
        ys, xs = np.nonzero(mask)
        if len(ys) > MAX_POINTS:
            sel = np.linspace(0, len(ys) - 1, MAX_POINTS).astype(np.int64)
            ys, xs = ys[sel], xs[sel]
        pts, ok = _back_project_many(xs.astype(float), ys.astype(float),
                                     cam_model, pos, heading, ground_z)
        if not ok.any():
            return
        points = pts[ok]
        points = points[np.isfinite(points).all(axis=1)]
        # Pixel density must not turn a single sighting into confirmation.
        keys = {self._key(x, y) for x, y in points}
        for key in keys:
            rec = self._cells.get(key)
            if rec is None:
                if len(self._cells) >= 4 * MAX_POINTS:
                    continue
                rec = [0.0, now]
                self._cells[key] = rec
            rec[0] = min(HIT_CAP, rec[0] + HIT_ADD)
            rec[1] = now

    def fused_support(self) -> np.ndarray:
        """World cells with enough evidence, as (N, 2) world points."""
        return np.array([[k[0] * self.cell_m, k[1] * self.cell_m]
                        for k, rec in self._cells.items()
                        if rec[0] >= HIT_MIN], dtype=float).reshape(-1, 2)

    def fuse(self, line_mask: np.ndarray, cam_model, pos, heading: float,
             ground_z: float = 0.0, now: float | None = None) -> np.ndarray:
        """Update with the current mask, then union re-projected evidence.

        Returns a bool mask of the same shape as ``line_mask``; when no
        evidence has accumulated yet it is the input mask unchanged.
        """
        self.update(line_mask, cam_model, pos, heading, ground_z, now)
        mask = np.asarray(line_mask, dtype=bool).copy()
        pts = self.fused_support()
        if len(pts) == 0:
            return mask
        pos3 = np.asarray(pos, dtype=float)
        if pos3.size < 3:
            pos3 = np.append(pos3, [float(ground_z)] * (3 - pos3.size))
        world = np.column_stack([
            pts[:, 0], pts[:, 1],
            np.full(len(pts), float(ground_z))])
        u, v, valid = cam_model.project(world, pos3, float(heading))
        valid = np.asarray(valid, dtype=bool)
        u = np.asarray(u, dtype=float)[valid]
        v = np.asarray(v, dtype=float)[valid]
        h, w = mask.shape
        inb = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        mask[v[inb].astype(int), u[inb].astype(int)] = True
        return mask
