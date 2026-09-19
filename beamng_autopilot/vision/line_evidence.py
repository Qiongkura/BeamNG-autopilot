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
HIT_MIN = 1.25       # accept first re-sight of a cell after one prior hit
DECAY_PER_S = 0.12   # yellow fragments drop out often; keep evidence ~7 s
MAX_AGE_S = 6.0
MAX_RANGE_M = 70.0
MAX_POINTS = 24000   # per-update back-projection budget


class LineEvidenceAccumulator:
    """World-space hit grid for line pixels, fused back into each frame."""

    def __init__(self, cell_m: float = CELL_M) -> None:
        self.cell_m = float(cell_m)
        # (ix, iy) -> [hits, last_seen_seconds]
        self._cells: dict[tuple[int, int], list] = {}
        self._last_t: float | None = None
        # Last frame that actually OBSERVED line pixels.  Distinct from
        # ``_last_t`` (every update call): "连续丢线超过阈值" must be
        # measured from the last real sighting, not from the last call.
        self._last_observation_t: float | None = None

    def reset(self) -> None:
        self._cells.clear()
        self._last_t = None
        self._last_observation_t = None

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
        if keys:
            self._last_observation_t = now
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

    def confidence(self) -> dict:
        """Dual confidence of the accumulated evidence (plan phase E3).

        Kept SEPARATE on purpose - a fused line pixel is not the same
        evidence when it was just observed and when it is being held:

        * ``current_confidence``  - share of the supported cells whose
          last sighting is the most recent observation (fresh evidence);
        * ``history_confidence``  - mean per-cell strength scaled by
          recency over the supported cells (how much the HELD evidence
          still deserves to be trusted; it decays to 0);
        * ``since_observation_s`` - how long no line pixel has been seen,
          and ``expired`` once that passes ``MAX_AGE_S`` (continuous
          loss invalidates history - the plan's explicit expiry rule).

        Everything is derived from the accumulator's own bookkeeping; no
        confidence is invented for evidence that does not exist.
        """
        now = self._last_t if self._last_t is not None else time.monotonic()
        since = (None if self._last_observation_t is None
                 else max(0.0, float(now) - float(self._last_observation_t)))
        support = [(k, rec) for k, rec in self._cells.items()
                   if rec[0] >= HIT_MIN]
        if not support:
            return {
                "current_confidence": 0.0,
                "history_confidence": 0.0,
                "n_supported": 0, "n_current": 0, "n_history": 0,
                "since_observation_s": (None if since is None
                                        else round(since, 3)),
                "expired": bool(since is not None and since > MAX_AGE_S),
            }
        last = self._last_t
        n_current = 0
        strengths = []
        for _k, rec in support:
            if last is not None and rec[1] >= float(last) - 1e-9:
                n_current += 1
            strength = min(1.0, float(rec[0]) / HIT_CAP)
            age = max(0.0, float(now) - float(rec[1]))
            strengths.append(strength * max(0.0, 1.0 - age / MAX_AGE_S))
        n_supported = len(support)
        return {
            "current_confidence": round(n_current / float(n_supported), 4),
            "history_confidence": round(float(np.mean(strengths)), 4),
            "n_supported": n_supported,
            "n_current": n_current,
            "n_history": n_supported - n_current,
            "since_observation_s": (None if since is None
                                    else round(since, 3)),
            "expired": bool(since is not None and since > MAX_AGE_S),
        }

    def fuse_with_confidence(self, line_mask: np.ndarray, cam_model, pos,
                             heading: float, ground_z: float = 0.0,
                             now: float | None = None
                             ) -> tuple[np.ndarray, dict]:
        """``fuse()`` plus the dual-confidence provenance (plan E3)."""
        raw = np.asarray(line_mask, dtype=bool)
        mask = self.fuse(line_mask, cam_model, pos, heading, ground_z, now)
        info = self.confidence()
        info["added_pixels"] = int(np.count_nonzero(mask & ~raw))
        return mask, info

    def support_mask(self, shape, cam_model, pos, heading: float,
                     ground_z: float = 0.0) -> np.ndarray:
        """The accumulated evidence re-projected into an IMAGE mask.

        Reads only: it does not add a vote (that is :meth:`update`), so a
        caller can ask "does history support this pixel?" BEFORE deciding
        what to fuse - which is what the far-field zone rule needs (plan
        E2: a loose far candidate must be temporally confirmed).
        """
        h, w = int(shape[0]), int(shape[1])
        mask = np.zeros((h, w), dtype=bool)
        pts = self.fused_support()
        if len(pts) == 0 or cam_model is None:
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
        inb = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        mask[v[inb].astype(int), u[inb].astype(int)] = True
        return mask

    def fuse(self, line_mask: np.ndarray, cam_model, pos, heading: float,
             ground_z: float = 0.0, now: float | None = None) -> np.ndarray:
        """Update with the current mask, then union re-projected evidence.

        Returns a bool mask of the same shape as ``line_mask``; when no
        evidence has accumulated yet it is the input mask unchanged.
        """
        self.update(line_mask, cam_model, pos, heading, ground_z, now)
        mask = np.asarray(line_mask, dtype=bool).copy()
        mask |= self.support_mask(mask.shape, cam_model, pos, heading,
                                  ground_z)
        return mask
