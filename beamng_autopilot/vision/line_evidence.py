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
# A backwards sequence jump larger than this is treated as a counter WRAP
# (new epoch), not as an out-of-order frame.
SEQ_WRAP_GUARD = 1 << 20
# Local bands (metres ahead of the car) for the per-band evidence report.
LOCAL_BAND_EDGES_M = (0.0, 10.0, 25.0)
# Added pixels whose supporting cells were stamped within this many seconds
# count as "current" evidence; anything older is "history".
ADDED_FRESH_S = 1.0


class LineEvidenceAccumulator:
    """World-space hit grid for line pixels, fused back into each frame."""

    def __init__(self, cell_m: float = CELL_M) -> None:
        self.cell_m = float(cell_m)
        # (ix, iy) -> [hits, last_seen_seconds, last_observation_key]
        self._cells: dict[tuple[int, int], list] = {}
        self._last_t: float | None = None
        # Last frame that actually OBSERVED line pixels.  Distinct from
        # ``_last_t`` (every update call): "连续丢线超过阈值" must be
        # measured from the last real sighting, not from the last call.
        self._last_observation_t: float | None = None
        # --- source identity (plan §3.2 / T03) -------------------------
        # Per-source sequence watermark: which real capture a source has
        # already contributed.  A replayed or out-of-order sequence must
        # not add a vote, and a counter wrap starts a new epoch instead of
        # throwing history away.
        self._last_seq: dict[str, int] = {}
        self._epoch = 0
        self._events: dict[str, int] = {}
        self._source_obs: dict[str, int] = {}

    def reset(self) -> None:
        self._cells.clear()
        self._last_t = None
        self._last_observation_t = None
        self._last_seq.clear()
        self._events.clear()
        self._source_obs.clear()

    def _bump(self, name: str, by: int = 1) -> None:
        self._events[name] = int(self._events.get(name, 0)) + int(by)

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
               ground_z: float = 0.0, now: float | None = None,
               source_id: str = "front_main",
               source_seq: int | None = None,
               capture_t: float | None = None) -> None:
        """Add at most one vote per cell per real OBSERVATION.

        The vote key is the observation, not the call: a cell that was
        already stamped by this capture (``capture_t`` + source sequence)
        does not get a second vote, however many times the caller
        reprocesses it, and however the processing time moved on.  Two
        genuine captures of identical content are two observations and do
        count twice (plan T03 反例).

        ``source_seq`` is a per-source monotonic counter: a repeat is
        rejected as a duplicate, a small backwards step as out-of-order,
        and a large one as a counter wrap (new epoch - history is kept,
        because nothing about the world changed).
        """
        now = time.monotonic() if now is None else float(now)
        if not np.isfinite(now):
            raise ValueError("timestamp must be finite")
        src = str(source_id or "unknown")
        # --- source identity gate ---------------------------------------
        if source_seq is not None:
            try:
                seq = int(source_seq)
            except (TypeError, ValueError):
                seq = None
            if seq is not None:
                last = self._last_seq.get(src)
                if last is not None:
                    if seq == last:
                        self._bump("rejected_duplicate")
                        return
                    if seq < last:
                        if last - seq > SEQ_WRAP_GUARD:
                            self._epoch += 1
                            self._bump("epoch_wrap")
                            self._last_seq[src] = seq
                        else:
                            self._bump("rejected_out_of_order")
                            return
                    else:
                        self._last_seq[src] = seq
                else:
                    self._last_seq[src] = seq
        if self._last_t is not None:
            if now < self._last_t:
                self.reset()
            elif now == self._last_t:
                self._bump("rejected_same_processing_time")
                return  # Reprocessing one observation is not a new vote.
        self._decay_and_prune(now, pos)
        mask = np.asarray(line_mask, dtype=bool)
        if not mask.any():
            self._bump("empty_observations")
            self._source_obs[src] = int(self._source_obs.get(src, 0)) + 1
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
        if not keys:
            return
        # The observation key: capture time when known (the physical
        # instant), else the processing time.  Cells already stamped by
        # this key are not re-voted - this is what stops two cameras (or
        # one camera processed twice) from double-counting one sighting.
        obs_key = (int(self._epoch),
                   float(capture_t) if capture_t is not None else float(now))
        voted = 0
        for key in keys:
            rec = self._cells.get(key)
            if rec is None:
                if len(self._cells) >= 4 * MAX_POINTS:
                    continue
                rec = [0.0, now, obs_key]
                self._cells[key] = rec
            elif len(rec) > 2 and rec[2] == obs_key:
                self._bump("votes_suppressed_same_capture")
                continue
            rec[0] = min(HIT_CAP, rec[0] + HIT_ADD)
            rec[1] = now
            if len(rec) > 2:
                rec[2] = obs_key
            voted += 1
        self._last_observation_t = now
        self._bump("accepted_observations")
        self._bump("accepted_cell_votes", voted)
        self._source_obs[src] = int(self._source_obs.get(src, 0)) + 1

    def fused_support(self) -> np.ndarray:
        """World cells with enough evidence, as (N, 2) world points."""
        return np.array([[k[0] * self.cell_m, k[1] * self.cell_m]
                        for k, rec in self._cells.items()
                        if rec[0] >= HIT_MIN], dtype=float).reshape(-1, 2)

    def source_events(self) -> dict:
        """Refresh requests vs. what actually became NEW evidence (T03).

        ``accepted_observations`` counts captures that added votes;
        ``accepted_cell_votes`` is the number of cells they stamped.  The
        rejected counters say WHY a call added nothing (duplicate
        sequence, out-of-order, same processing time, same capture instant
        as another source) - a caller must be able to tell "we asked for a
        refresh" from "new source evidence arrived" (plan §3.2/T03).
        """
        out: dict = {"epoch": int(self._epoch)}
        for k in ("accepted_observations", "accepted_cell_votes",
                  "rejected_duplicate", "rejected_out_of_order",
                  "rejected_same_processing_time", "epoch_wrap",
                  "votes_suppressed_same_capture", "empty_observations"):
            out[k] = int(self._events.get(k, 0))
        out["by_source"] = dict(sorted(self._source_obs.items()))
        out["seq_watermarks"] = dict(sorted(self._last_seq.items()))
        return out

    def support_digest(self, points, now: float | None = None,
                       role: str = "candidate") -> dict | None:
        """How much of THIS candidate's geometry is fresh evidence (T03).

        For a candidate polyline (world points), reports the fraction of
        its length whose cells are supported at all, and of those, how
        much was stamped by the most recent observation versus only held
        from earlier ones.  That is what "selected boundary: current
        support vs history-only support" needs, per candidate, instead of
        one global ratio over the whole cache.
        """
        pts = np.asarray(points, dtype=float) if points is not None else None
        if pts is None or pts.ndim != 2 or len(pts) == 0:
            return None
        now_t = float(now) if now is not None else (
            float(self._last_t) if self._last_t is not None else None)
        last = self._last_t
        n_sup = n_cur = 0
        ages: list[float] = []
        for x, y in pts[:, :2]:
            rec = self._cells.get(self._key(float(x), float(y)))
            if rec is None or rec[0] < HIT_MIN:
                continue
            n_sup += 1
            if last is not None and rec[1] >= float(last) - 1e-9:
                n_cur += 1
            if now_t is not None:
                ages.append(max(0.0, float(now_t) - float(rec[1])))
        n = int(len(pts))
        return {
            # Self-describing: a lane CENTRE sits half a lane from the paint
            # cells by construction, so its own support is expected to be
            # near zero - reporting the number without the role invites
            # reading it as "the reference has no evidence".
            "role": str(role),
            "n_points": n,
            "n_supported": n_sup,
            "supported_frac": round(n_sup / n, 3),
            "n_current": n_cur,
            "current_frac": round(n_cur / n, 3),
            "history_only_frac": round((n_sup - n_cur) / n, 3),
            "min_age_s": (round(min(ages), 3) if ages else None),
            "max_age_s": (round(max(ages), 3) if ages else None),
        }

    def local_bands(self, pos, heading: float,
                    edges=LOCAL_BAND_EDGES_M,
                    now: float | None = None) -> list[dict]:
        """Per-band supported/current counts and ages along the car's axis.

        The plan's 反例 "远处持续刷新、近处过期" is invisible in one global
        ratio: a far-field refresh keeps it high while the near field has
        already expired.  Bands are measured in the CAR frame so normal
        motion does not move a cell between bands.
        """
        now_t = float(now) if now is not None else (
            float(self._last_t) if self._last_t is not None else None)
        p = np.asarray(pos, dtype=float)[:2]
        fwd = np.array([np.cos(float(heading)), np.sin(float(heading))])
        last = self._last_t
        bands: list[dict] = []
        for i in range(len(edges) - 1):
            lo, hi = float(edges[i]), float(edges[i + 1])
            n_sup = n_cur = 0
            ages: list[float] = []
            for k, rec in self._cells.items():
                if rec[0] < HIT_MIN:
                    continue
                wx, wy = k[0] * self.cell_m, k[1] * self.cell_m
                d = (np.array([wx, wy]) - p) @ fwd
                if not (lo <= d < hi):
                    continue
                n_sup += 1
                if last is not None and rec[1] >= float(last) - 1e-9:
                    n_cur += 1
                if now_t is not None:
                    ages.append(max(0.0, float(now_t) - float(rec[1])))
            bands.append({
                "from_m": lo, "to_m": hi, "supported": n_sup,
                "current": n_cur,
                "current_frac": (round(n_cur / n_sup, 3) if n_sup else None),
                "oldest_age_s": (round(max(ages), 3) if ages else None),
                "min_age_s": (round(min(ages), 3) if ages else None),
            })
        return bands

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
                             now: float | None = None,
                             source_id: str = "front_main",
                             source_seq: int | None = None,
                             capture_t: float | None = None,
                             yellow_mask=None
                             ) -> tuple[np.ndarray, dict]:
        """``fuse()`` plus the provenance breakdown (plan E3/T03).

        ``added_pixels`` is split by the AGE of the evidence behind it:
        ``added_pixels_current`` are pixels the accumulator supplied from
        cells stamped within ``ADDED_FRESH_S`` (the car is still tracking
        that paint), ``added_pixels_history`` the rest - and the yellow
        prior is reported on its own, because it is a colour heuristic, not
        a model detection, and the plan asks for the two to be published
        separately.
        """
        raw = np.asarray(line_mask, dtype=bool)
        mask = self.fuse(line_mask, cam_model, pos, heading, ground_z, now,
                         source_id=source_id, source_seq=source_seq,
                         capture_t=capture_t)
        info = self.confidence()
        now_t = float(now) if now is not None else (
            float(self._last_t) if self._last_t is not None else None)
        added = mask & ~raw
        fresh = np.zeros(added.shape, dtype=bool)
        old = np.zeros(added.shape, dtype=bool)
        if added.any() and now_t is not None:
            # Attribute each added pixel to the age of its supporting cell.
            h, w = added.shape[:2]
            ys, xs = np.nonzero(added)
            pts, ok = _back_project_many(xs.astype(float), ys.astype(float),
                                         cam_model, pos, heading, ground_z)
            if ok.any():
                pairs = np.column_stack([pts[ok], ys[ok], xs[ok]])
                for wx, wy, vv, uu in pairs:
                    rec = self._cells.get(self._key(float(wx), float(wy)))
                    if rec is None:
                        continue
                    age = max(0.0, now_t - float(rec[1]))
                    if age <= ADDED_FRESH_S:
                        fresh[int(vv), int(uu)] = True
                    else:
                        old[int(vv), int(uu)] = True
        info["added_pixels"] = int(np.count_nonzero(added))
        info["added_pixels_current"] = int(np.count_nonzero(fresh))
        info["added_pixels_history"] = int(np.count_nonzero(old))
        info["added_fresh_s"] = float(ADDED_FRESH_S)
        if yellow_mask is not None:
            ym = np.asarray(yellow_mask, dtype=bool)
            info["yellow_pixels_in_line"] = int(np.count_nonzero(
                np.asarray(line_mask, dtype=bool) & ym))
            info["added_pixels_yellow"] = int(np.count_nonzero(added & ym))
        else:
            info["yellow_pixels_in_line"] = 0
            info["added_pixels_yellow"] = 0
        info["source_events"] = self.source_events()
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
             ground_z: float = 0.0, now: float | None = None,
             source_id: str = "front_main", source_seq: int | None = None,
             capture_t: float | None = None) -> np.ndarray:
        """Update with the current mask, then union re-projected evidence.

        Returns a bool mask of the same shape as ``line_mask``; when no
        evidence has accumulated yet it is the input mask unchanged.
        """
        self.update(line_mask, cam_model, pos, heading, ground_z, now,
                    source_id=source_id, source_seq=source_seq,
                    capture_t=capture_t)
        mask = np.asarray(line_mask, dtype=bool).copy()
        mask |= self.support_mask(mask.shape, cam_model, pos, heading,
                                  ground_z)
        return mask
