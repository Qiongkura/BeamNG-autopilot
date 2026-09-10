"""Runtime acceptance contract for learned trajectory candidates.

The learned planners (E2E trajectory regressor, DAVE-2 arc rollout) can
emit finite-looking but physically absurd paths when the live input
distribution shifts.  Every learned path passes through this one pure
validator before it can become an arbitration candidate: finite values,
forward progress, bounded extent/lateral excursion and a bounded
curvature.  Keeping the contract here means training diagnostics, the
runtime boundary and the tests all speak about the same path geometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

MIN_EXTENT_M = 1.0
MAX_EXTENT_M = 80.0
MAX_LATERAL_M = 25.0
MAX_BACKSTEP_M = 1.5
MAX_CURVATURE = 0.35       # 1/m; ~2.9 m minimum radius
_MIN_SEGMENT_M = 0.05


@dataclass(frozen=True)
class PathValidation:
    """Result of :func:`validate_learned_path` (metrics for telemetry)."""

    ok: bool
    reason: str
    extent_m: float
    lateral_m: float
    backstep_m: float
    max_curvature: float


def _invalid(reason: str) -> PathValidation:
    return PathValidation(False, reason, 0.0, 0.0, 0.0, 0.0)


def _wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def validate_learned_path(
        path,
        origin=None,
        forward=None,
        *,
        min_extent_m: float = MIN_EXTENT_M,
        max_extent_m: float = MAX_EXTENT_M,
        max_lateral_m: float = MAX_LATERAL_M,
        max_backstep_m: float = MAX_BACKSTEP_M,
        max_curvature: float = MAX_CURVATURE) -> PathValidation:
    """Validate one learned path in a known frame.

    ``origin`` / ``forward`` define the ego frame: the trajectory must
    advance along ``forward`` and stay within ``max_lateral_m`` of that
    axis.  Ego-relative E2E output uses ``origin=(0, 0)`` and
    ``forward=(1, 0)``; the BC arc uses the ego pose.  Deep-learning
    garbage is rejected with a stable reason string for telemetry.
    """
    try:
        arr = np.asarray(path, dtype=float)
    except (TypeError, ValueError):
        return _invalid("shape")
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        return _invalid("shape")
    arr = arr[:, :2]
    if not np.isfinite(arr).all():
        return _invalid("nonfinite")
    o = np.zeros(2, dtype=float) if origin is None else np.asarray(
        origin, dtype=float).reshape(-1)[:2]
    if o.shape != (2,) or not np.isfinite(o).all():
        return _invalid("shape")
    if forward is None:
        f = np.array([1.0, 0.0], dtype=float)
    else:
        f = np.asarray(forward, dtype=float).reshape(-1)[:2]
    if f.shape != (2,) or not np.isfinite(f).all():
        return _invalid("shape")
    fn = float(np.linalg.norm(f))
    if fn < 1e-9:
        return _invalid("shape")
    f = f / fn

    rel = arr - o[None, :]
    d = np.diff(rel, axis=0)
    seg = np.linalg.norm(d, axis=1)
    extent = float(np.linalg.norm(rel[-1]))
    lateral = float(np.max(np.abs(rel[:, 0] * f[1] - rel[:, 1] * f[0])))
    fwd_step = d @ f
    backstep = float(max(0.0, -float(np.min(fwd_step))))

    max_k = 0.0
    for i in range(1, len(d)):
        ds = 0.5 * (float(seg[i - 1]) + float(seg[i]))
        if ds < _MIN_SEGMENT_M:
            continue
        th0 = math.atan2(float(d[i - 1, 1]), float(d[i - 1, 0]))
        th1 = math.atan2(float(d[i, 1]), float(d[i, 0]))
        max_k = max(max_k, abs(_wrap_angle(th1 - th0)) / ds)

    if extent < float(min_extent_m):
        return PathValidation(False, "extent_low", extent, lateral,
                              backstep, max_k)
    if extent > float(max_extent_m):
        return PathValidation(False, "extent_high", extent, lateral,
                              backstep, max_k)
    if lateral > float(max_lateral_m):
        return PathValidation(False, "lateral", extent, lateral,
                              backstep, max_k)
    if backstep > float(max_backstep_m):
        return PathValidation(False, "backstep", extent, lateral,
                              backstep, max_k)
    if max_k > float(max_curvature):
        return PathValidation(False, "curvature", extent, lateral,
                              backstep, max_k)
    return PathValidation(True, "", extent, lateral, backstep, max_k)


__all__ = ["PathValidation", "validate_learned_path"]
