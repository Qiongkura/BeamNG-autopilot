"""Live DAVE-2 BC steering model for the FSD arbitration chain.

Wraps a trained M3 behaviour-cloning checkpoint (``beamng_autopilot.bc
.Dave2``: 200x66 RGB -> normalized steering, BatchNorm variant) so the
drive loop can rank the imitation-learned steering as a neural candidate
the same way the E2E planner is ranked (neural above the rule backup,
gated by the safety monitor).  The BC net predicts a STEERING value, not
a trajectory: :func:`steer_to_path` rolls the kinematic bicycle model
forward into a world-space arc so the safety monitor and the arbitration
chain can treat it exactly like any other candidate path.

``steer_to_path`` is a pure function so tests can verify the geometry
without a model or torch weights.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from ..bc import Dave2, conv_feature_size, preprocess_frame

DEFAULT_BC_WEIGHTS = "logs/m3_bc/bc_tech_smallgrid.pt"


def steer_to_path(steer: float, pos, heading: float,
                  length_m: float = 18.0, n: int = 13,
                  wheelbase: float = 2.9,
                  steer_ratio: float = 0.6) -> np.ndarray | None:
    """Roll a constant BC steering input into a world-space arc.

    ``steer`` uses the project's normalized convention (negative = LEFT,
    matching the PurePursuit / BC label contract); ``steer_ratio`` is the
    rad-per-normalized-input road-wheel ratio the feed-forward path
    curvature uses (``fsd_drive._path_curvature_ff``).  Kinematic bicycle:
    ``kappa = tan(delta) / wheelbase`` with ``delta = -steer *
    steer_ratio`` (the minus flips normalized-left into positive road
    curvature).  Returns an ``(n, 2)`` world polyline starting at
    ``pos`` ( Exclusive of the start point), or None on malformed input.
    """
    s = float(steer)
    if not math.isfinite(s):
        return None
    s = max(-1.0, min(1.0, s))
    p = np.asarray(pos, dtype=float).ravel()
    if p.size < 2 or not np.isfinite(p[:2]).all():
        return None
    delta = -s * float(steer_ratio)
    kappa = math.tan(delta) / float(wheelbase)
    c, sn = math.cos(float(heading)), math.sin(float(heading))
    x0, y0 = float(p[0]), float(p[1])
    pts = []
    for i in range(1, max(2, int(n)) + 1):
        d = length_m * i / max(2, int(n))
        if abs(kappa) < 1e-6:                 # straight
            ex, ey = d, 0.0
        else:
            th = d * kappa                    # signed arc angle
            ex = math.sin(th) / kappa
            ey = (1.0 - math.cos(th)) / kappa
        pts.append((x0 + ex * c - ey * sn,
                    y0 + ex * sn + ey * c))
    return np.asarray(pts, dtype=float)


class BCRuntime:
    """Load + serve one trained DAVE-2 BC checkpoint for live driving."""

    def __init__(self, weights=None, device: str | None = None) -> None:
        self.weights = Path(weights) if weights else None
        self.device = device or "cpu"
        self.net = None
        self.ckpt: dict | None = None
        self.img_w = 200
        self.img_h = 66
        self._err: str | None = None
        if self.weights is not None and self.weights.exists():
            try:
                self._load()
            except Exception as exc:            # missing/broken weights
                self._err = str(exc)
                self.net = None

    @property
    def loaded(self) -> bool:
        return self.net is not None

    @property
    def error(self) -> str | None:
        return self._err

    def _load(self) -> None:
        import torch
        ckpt = torch.load(self.weights, map_location="cpu",
                          weights_only=False)
        w, h = (int(x) for x in (ckpt.get("resize") or (200, 66)))
        self.img_w, self.img_h = w, h
        net = Dave2(feat_in=conv_feature_size(h, w))
        net.load_state_dict(ckpt["state_dict"])
        net.to(self.device)
        net.eval()
        self.net = net
        self.ckpt = ckpt

    def predict_steer(self, frame) -> tuple[float, float]:
        """One RGB frame -> ``(normalized steer, inference ms)``.

        Follows the training contract exactly (RGB uint8 -> resize to the
        checkpoint shape -> scale to [-1, 1]).  An unusable frame returns
        ``(0.0, 0.0)`` - the drive loop then simply has no BC candidate.
        """
        if self.net is None or frame is None:
            return 0.0, 0.0
        rgb = np.asarray(frame, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] < 3 or rgb.shape[0] < 8:
            return 0.0, 0.0
        t0 = time.time()
        x = preprocess_frame(rgb, self.img_w, self.img_h).to(self.device)
        import torch
        with torch.no_grad():
            steer = float(self.net(x).item())
        ms = (time.time() - t0) * 1000.0
        if not math.isfinite(steer):
            return 0.0, ms
        return max(-1.0, min(1.0, steer)), ms
