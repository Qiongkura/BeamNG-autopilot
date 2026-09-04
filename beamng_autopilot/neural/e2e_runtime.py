"""Live E2E neural planner for the FSD real-driving loop.

Wraps a trained ``E2ENetTorch`` checkpoint so the real-driving entry
point (``scripts/m5_fsd_drive.py``) can rank the learned planner in the
arbitration chain the way FSD ranks its neural planner above the
kinematic/rule backup.  The wrapper owns everything the drive loop needs
to stay close to the training contract:

* resize / scale the front RGB, the segmentation label and the BEV
  raster to the checkpoint's input shapes;
* keep a rolling ``history+1`` frame buffer (temporal conv), padding
  the first frames with the earliest one exactly like the offline probe;
* run the no-grad forward pass on the selected device;
* inverse-transform the predicted ego-relative trajectory back to world
  coordinates (the inverse of ``dataset.ShadowMultimodalDataset._traj_ego``).

``ego_path_to_world`` is a pure function so tests can verify the
transform without a model.
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .e2e_torch import E2ENetTorch

DEFAULT_E2E_WEIGHTS = "logs/m5_e2e/best_temporal.pt"


def ego_path_to_world(traj_ego, pos, heading) -> np.ndarray | None:
    """Ego-relative trajectory -> world frame (inverse of the dataset).

    The training target keeps the recorded world trajectory in the ego
    frame: forward ``= dx*cos + dy*sin``, left ``= -dx*sin + dy*cos``.
    Given the ego pose this inverts cleanly::

        dx = x_ego*cos - y_ego*sin
        dy = x_ego*sin + y_ego*cos

    Returns ``(N, 2)`` or None for malformed input.
    """
    t = np.asarray(traj_ego, dtype=float)
    if t.ndim != 2 or t.shape[1] < 2:
        return None
    c, s = float(np.cos(heading)), float(np.sin(heading))
    x, y = float(pos[0]), float(pos[1])
    return np.stack([x + t[:, 0] * c - t[:, 1] * s,
                     y + t[:, 0] * s + t[:, 1] * c], axis=1)


class E2ERuntime:
    """Load + serve one trained E2E checkpoint for live driving."""

    def __init__(self, weights=None, device: str | None = None) -> None:
        self.weights = Path(weights) if weights else None
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.net: E2ENetTorch | None = None
        self.ckpt: dict | None = None
        self.img_h = 120
        self.img_w = 160
        self.grid_n = 60
        self._buf: deque | None = None
        if self.weights is not None and self.weights.exists():
            self._load()

    # ------------------------------------------------------------------
    @property
    def loaded(self) -> bool:
        return self.net is not None

    @property
    def history(self) -> int:
        return self.net.history if self.net is not None else 0

    @property
    def n_waypoints(self) -> int:
        return self.net.n_waypoints if self.net is not None else 16

    def reset(self) -> None:
        """Drop the temporal buffer (teleport / run restart)."""
        if self._buf is not None:
            self._buf.clear()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        ckpt = torch.load(self.weights, map_location="cpu")
        net = E2ENetTorch(
            grid_n=int(ckpt.get("grid_n", 60)),
            n_waypoints=int(ckpt.get("n_waypoints", 16)),
            history=int(ckpt.get("history", 0)))
        net.load_state_dict(ckpt["model"])
        net.to(self.device)
        net.eval()
        self.net = net
        self.ckpt = ckpt
        self.img_h = int(ckpt.get("img_h", 120))
        self.img_w = int(ckpt.get("img_w", 160))
        self.grid_n = int(ckpt.get("grid_n", 60))
        self._buf = deque(maxlen=self.history + 1)

    @staticmethod
    def _label_from_outputs(head_outputs, shape) -> np.ndarray | None:
        """Synthesize the 3-class label (0 bg / 1 road / 2 line)."""
        heads = head_outputs or {}
        sem = heads.get("semantic")
        if sem is None:
            return None
        masks = getattr(sem, "masks", None) or {}
        road = masks.get("road")
        if road is None:
            return None
        h, w = shape
        lab = np.zeros((h, w), dtype=np.uint8)

        def _mask2d(m, size):
            m = np.asarray(m, dtype=bool).reshape(-1)
            t = torch.from_numpy(np.asarray(m, dtype=np.uint8))
            t = F.interpolate(
                t.reshape(1, 1, 1, -1).float(),
                size=(size[0] * size[1],), mode="nearest")
            return t.reshape(size).numpy() > 0

        r = np.asarray(road, dtype=bool)
        if r.shape == (h, w):
            lab[r] = 1
        else:
            lab[:] = np.where(_mask2d(r, (h, w)), 1, lab)
        line = masks.get("line")
        if line is not None:
            l = np.asarray(line, dtype=bool)
            if l.shape == (h, w):
                lab[l] = 2
            else:
                lab[:] = np.where(_mask2d(l, (h, w)), 2, lab)
        return lab

    def _prep(self, rgb, label, bev):
        """One frame -> (t_rgb, t_label, t_bev) at the checkpoint shapes."""
        t_rgb = torch.from_numpy(
            np.ascontiguousarray(rgb, dtype=np.uint8).astype(
                np.float32)).permute(2, 0, 1)[None]
        t_rgb = F.interpolate(t_rgb, size=(self.img_h, self.img_w),
                              mode="bilinear", align_corners=False)[0]
        # Same uint8 rounding as the dataset so live input matches the
        # training distribution.
        t_rgb = t_rgb.clamp(0.0, 255.0).round().to(torch.uint8).float() / 255.0
        if label is None:
            t_label = torch.zeros(1, self.img_h, self.img_w,
                                  dtype=torch.float32)
        else:
            t_label = torch.from_numpy(label.astype(np.float32))[None, None]
            t_label = F.interpolate(t_label, size=(self.img_h, self.img_w),
                                    mode="nearest")[0]
        bev = np.asarray(bev, dtype=np.float32)
        if bev.ndim == 3:
            bev = bev[0] if bev.shape[0] == 1 else bev.mean(axis=0)
        t_bev = torch.from_numpy(bev)[None]
        if t_bev.shape != (1, self.grid_n, self.grid_n):
            t_bev = F.interpolate(t_bev[None],
                                  size=(self.grid_n, self.grid_n),
                                  mode="nearest")[0]
        return t_rgb, t_label, t_bev

    # ------------------------------------------------------------------
    def step(self, out, pos, heading, speed):
        """One prediction over an ``FSDTick``-like output.

        Returns ``(world_path (N,2) | None, action (2,) | None, ms)``.
        ``out`` needs ``.frame``, ``.bev`` and ``.head_outputs`` (the
        FSD stack tick provides all three).  An unusable frame returns
        (None, None, 0.0) so the drive loop degrades silently.
        """
        if self.net is None:
            return None, None, 0.0
        frame = getattr(out, "frame", None)
        bev = getattr(out, "bev", None)
        if frame is None or bev is None:
            return None, None, 0.0
        rgb = np.asarray(frame, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            return None, None, 0.0
        label = self._label_from_outputs(
            getattr(out, "head_outputs", None), rgb.shape[:2])
        t_rgb, t_label, t_bev = self._prep(rgb, label, bev)
        assert self._buf is not None
        self._buf.append((t_rgb, t_label, t_bev))
        need = self.history + 1
        frames = list(self._buf)
        if len(frames) < need:   # pad the early temporal window
            frames = [frames[0]] * (need - len(frames)) + frames
        t0 = time.time()
        with torch.no_grad():
            traj, action = self.net(
                torch.stack([f[0] for f in frames])[None].to(self.device),
                torch.stack([f[1] for f in frames])[None].to(self.device),
                torch.stack([f[2] for f in frames])[None].to(self.device),
                torch.tensor([[float(speed)]], dtype=torch.float32,
                             device=self.device))
        ms = (time.time() - t0) * 1000.0
        world = ego_path_to_world(traj[0].cpu().numpy(), pos, heading)
        if world is None:
            return None, None, ms
        return world, action[0].cpu().numpy(), ms
