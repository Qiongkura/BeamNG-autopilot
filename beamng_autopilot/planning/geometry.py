"""Vectorised geometry helpers for planning / safety (pure numpy).

The planning and safety layers repeatedly measure how far path points sit
from a reference polyline (lane centre / nav route).  The first versions
used nested Python loops (every path point against every route segment),
which cost hundreds of milliseconds per FSD tick with a dense 80+ point
route and ~18 candidate paths.  These helpers broadcast the same
point-to-segment distance in numpy so the per-tick cost drops to a few
milliseconds.
"""

from __future__ import annotations

import numpy as np


def polyline_point_distances(points, polyline) -> np.ndarray:
    """Minimum distance from each point to any segment of a polyline.

    ``points`` is (N, 2), ``polyline`` is (M, 2).  Returns (N,) with the
    Euclidean distance to the nearest polyline segment (degenerate
    segments collapse to their start vertex, matching the scalar
    implementation used before this helper).
    """
    pts = np.asarray(points, dtype=float)
    poly = np.asarray(polyline, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 2 or len(pts) == 0:
        return np.zeros((0,), dtype=float)
    if poly is None or len(poly) < 2:
        return np.full(len(pts), np.inf, dtype=float)
    a = poly[:-1]
    b = poly[1:]
    ab = b - a
    l2 = np.einsum("ij,ij->i", ab, ab)
    rel = pts[:, None, :] - a[None, :, :]
    t = np.clip(np.einsum("ijk,jk->ij", rel, ab)
                / np.maximum(l2[None, :], 1e-12), 0.0, 1.0)
    proj = a[None, :, :] + t[..., None] * ab[None, :, :]
    return np.linalg.norm(pts[:, None, :] - proj, axis=2).min(axis=1)
