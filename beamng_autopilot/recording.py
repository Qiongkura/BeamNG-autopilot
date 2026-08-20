"""Shadow-mode data recording - the FSD-style data-fusion collector.

Tesla FSD continuously records shadow trips: while the production stack
(his rule planner today, a future learned planner later) drives, the
perception-pipeline runs the FSD-style stack (ring -> HydraNets -> BEV
occupancy -> layered planner) *in the background* and logs what it
*would* have done next to the actual executed control.  Those aligned
(sensor, truth, prediction) samples are the training data for an
end-to-end stack - exactly the structure FSD uses.

This module gives that shape without writing to the game:

* ``ShadowFrame``: one timestamped snapshot - the ego state, the
  executed control, plus the shadow stack's predictions (BEV occupancy
  raster, chosen trajectory, chosen speed).
* ``ShadowRecorder``: buffers frames and writes a versioned .npz
  episode; the recorded tensors are exactly what a future network uses
  as (observation, action, label).
* ``EpisodeDataset``: a torch ``Dataset`` over recorded .npz episodes
  that yields ``(bev_raster, action)`` pairs - a plausible input/action
  contract for the end-to-end skeleton (goal 6).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

EPISODE_VERSION = 1


@dataclass
class ShadowFrame:
    """One shadow-mode sample (everything is loggable / consumable)."""

    t: float = 0.0
    # ego state + executed control (the "truth")
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    speed: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0
    steer: float = 0.0
    # shadow stack predictions
    bev_raster: np.ndarray | None = None      # (N, N) occupancy 0..1
    drivable: np.ndarray | None = None        # (N, N) free-space flag
    trajectory: np.ndarray | None = None      # (M, 2) chosen world path
    target_speed: float = 0.0
    lane_src: str = ""                         # which head produced the lane
    cost: float = -1.0
    kind: str = ""                             # which trajectory was chosen


class ShadowRecorder:
    """Accumulate ``ShadowFrame``s and write one episode .npz."""

    def __init__(self, log_dir, sequence: str):
        self.log_dir = Path(log_dir)
        self.sequence = sequence
        self.frames: list[ShadowFrame] = []
        self._t0 = time.time()

    def add(self, frame: ShadowFrame) -> None:
        frame.t = float(time.time() - self._t0)
        self.frames.append(frame)

    def __len__(self) -> int:
        return len(self.frames)

    def save(self, prefix: str | None = None) -> Path | None:
        """Write the episode as ``<log>/shadow_<seq>_<stamp>.npz``.

        Returns the file path or None when nothing was recorded.
        """
        if not self.frames:
            return None
        stamp = time.strftime("%Y%m%d_%H%M%S")
        name = f"shadow_{self.sequence}_{stamp}.npz"
        if prefix:
            name = f"{prefix}_{name}"
        out = self.log_dir / name
        out.parent.mkdir(parents=True, exist_ok=True)

        t = np.array([f.t for f in self.frames], dtype=np.float64)
        meta = np.array([f.x for f in self.frames], dtype=np.float64)
        # pack variable-size trajectory/bev arrays with a mask
        n = len(self.frames)
        max_traj = max((len(f.trajectory) for f in self.frames
                        if f.trajectory is not None), default=0)
        traj = np.full((n, max(2, max_traj), 2), np.nan, dtype=np.float64)
        traj_ok = np.zeros(n, dtype=bool)
        bev = np.zeros((n, 60, 60), dtype=np.float32)  # fixed-size grid
        drv = np.zeros((n, 60, 60), dtype=np.uint8)
        for i, f in enumerate(self.frames):
            if f.trajectory is not None and len(f.trajectory):
                m = min(len(f.trajectory), max(2, max_traj))
                traj[i, :m] = np.asarray(f.trajectory[:m], dtype=float)
                traj_ok[i] = True
            if f.bev_raster is not None:
                bev[i] = np.asarray(f.bev_raster, dtype=np.float32)
            if f.drivable is not None:
                drv[i] = np.asarray(f.drivable, dtype=np.uint8)

        np.savez_compressed(
            out,
            version=np.int64(EPISODE_VERSION),
            t=t,
            x=meta,
            y=np.array([f.y for f in self.frames]),
            heading=np.array([f.heading for f in self.frames]),
            speed=np.array([f.speed for f in self.frames]),
            throttle=np.array([f.throttle for f in self.frames]),
            brake=np.array([f.brake for f in self.frames]),
            steer=np.array([f.steer for f in self.frames]),
            bev=bev,
            drivable=drv,
            trajectory=traj,
            trajectory_ok=traj_ok,
            target_speed=np.array([f.target_speed for f in self.frames]),
            lane_src=np.array([f.lane_src for f in self.frames],
                              dtype=object),
            cost=np.array([f.cost for f in self.frames]),
            kind=np.array([f.kind for f in self.frames], dtype=object),
            meta=json.dumps({
                "sequence": self.sequence,
                "frames": n,
            }).encode("utf-8"),
        )
        return out


class EpisodeDataset:
    """torch Dataset over recorded shadow episodes.

    Yields ``(bev_raster, action)`` where ``action`` is
    ``(steer, throttle)`` - the contract the end-to-end skeleton trains
    against: observation = fused vector space, action = executed control.
    """

    def __init__(self, ep_files):
        import torch  # noqa: F401  (lazy import: only needed to train)

        if isinstance(ep_files, (str, Path)):
            ep_files = [Path(ep_files)]
        self.files = [Path(p) for p in ep_files if Path(p).exists()]
        self._n = 0
        for p in self.files:
            with np.load(p, allow_pickle=True) as z:
                self._n += int(z["t"].shape[0])

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int):
        import torch
        for p in self.files:
            with np.load(p, allow_pickle=True) as z:
                n = int(z["t"].shape[0])
                if idx < n:
                    bev = torch.from_numpy(
                        np.asarray(z["bev"][idx], dtype=np.float32))
                    act = torch.tensor(
                        [float(z["steer"][idx]),
                         float(z["throttle"][idx])],
                        dtype=torch.float32)
                    return bev, act
                idx -= n
        raise IndexError(idx)