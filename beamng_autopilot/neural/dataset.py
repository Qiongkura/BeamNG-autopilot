"""Dataset over recorded shadow episodes for the real E2E training stack.

``ShadowMultimodalDataset`` turns the multi-modal .npz episodes produced
by ``ShadowRecorder`` (version 2: rgb + segmentation label + BEV raster
+ quality gate) into ``(rgb, label, bev) -> (trajectory_ego, action)``
training pairs:

* the segmentation label and the BEV raster are the two "perception"
  views of the stack (semantic head and vector-space occupancy);
* the RGB frame is the raw-image view (DAVE-2 / image end-to-end);
* the trajectory is re-expressed in the ego frame using the recorded
  pose so the regression target is well-posed (a single image cannot
  infer the absolute world position);
* frames whose shadow prediction was gated out (``quality`` below
  ``min_quality``) or had no feasible trajectory are dropped at index
  build time so bad samples never poison training.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

GRID_N = 60
N_WAYPOINTS = 16
N_ACTION = 2


def _load_frames(ep_files, min_quality: float, min_speed: float):
    """Load (and quality/speed-filter) one episode into frame dicts."""
    out = []
    for p in ep_files:
        with np.load(p, allow_pickle=True) as z:
            n = int(z["t"].shape[0])
            q = np.asarray(z["quality"], dtype=np.float32) \
                if "quality" in z else np.ones(n, dtype=np.float32)
            ok = np.asarray(z["trajectory_ok"], dtype=bool) \
                if "trajectory_ok" in z else np.ones(n, dtype=bool)
            spd = np.asarray(z["speed"], dtype=np.float64)
            for i in range(n):
                if float(q[i]) < min_quality or not bool(ok[i]):
                    continue
                if float(spd[i]) < min_speed:
                    continue
                traj = np.asarray(z["trajectory"][i], dtype=np.float64)
                if traj.ndim != 2 or traj.shape[0] < 2:
                    continue
                rgb = np.asarray(z["rgb"][i], dtype=np.uint8) \
                    if "rgb" in z else None
                label = np.asarray(z["label"][i], dtype=np.uint8) \
                    if "label" in z else None
                bev = np.asarray(z["bev"][i], dtype=np.float32) \
                    if "bev" in z else None
                if rgb is None and label is None and bev is None:
                    continue
                out.append({
                    "x": float(z["x"][i]),
                    "y": float(z["y"][i]),
                    "heading": float(z["heading"][i]),
                    "rgb": rgb,
                    "label": label,
                    "bev": bev,
                    "traj_world": traj,
                    "steer": float(z["steer"][i]),
                    "throttle": float(z["throttle"][i]),
                })
    return out


class ShadowMultimodalDataset(torch.utils.data.Dataset):
    """Multimodal (rgb, label, bev) -> (trajectory_ego, action)."""

    def __init__(self, ep_files, min_quality: float = 0.5,
                 min_speed: float = 0.5,
                 img_h: int = 120, img_w: int = 160,
                 n_waypoints: int = N_WAYPOINTS,
                 augment: bool = False, seed: int = 0) -> None:
        if isinstance(ep_files, (str, Path)):
            ep_files = [Path(ep_files)]
        self.files = [Path(p) for p in ep_files if Path(p).exists()]
        self.min_quality = float(min_quality)
        self.min_speed = float(min_speed)
        self.img_h = int(img_h)
        self.img_w = int(img_w)
        self.n_waypoints = int(n_waypoints)
        self.augment = bool(augment)
        self.frames = _load_frames(self.files, self.min_quality,
                                   self.min_speed)
        if self.augment:
            self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.frames)

    def _traj_ego(self, f: dict) -> torch.Tensor:
        """World trajectory -> ego frame, trimmed/padded to N waypoints.

        Recorded trajectories are variable length and NaN-padded to the
        archive's max width; only the finite prefix is used, then the
        last valid point is repeated to reach ``n_waypoints``.
        """
        h = float(f["heading"])
        c, s = np.cos(h), np.sin(h)
        rel = np.asarray(f["traj_world"], dtype=np.float32) - \
            np.array([f["x"], f["y"]], dtype=np.float32)
        finite = np.isfinite(rel).all(axis=1)
        rel = rel[finite]
        if len(rel) == 0:
            rel = np.zeros((1, 2), dtype=np.float32)
        ego = np.stack([rel[:, 0] * c + rel[:, 1] * s,
                        -rel[:, 0] * s + rel[:, 1] * c], axis=1)
        n = self.n_waypoints
        if len(ego) >= n:
            ego = ego[:n]
        else:
            pad = np.tile(ego[-1:], (n - len(ego), 1))
            ego = np.concatenate([ego, pad], axis=0)
        return torch.from_numpy(ego)

    def __getitem__(self, idx: int):
        f = self.frames[idx]
        # Fixed modality order (rgb, label, bev); a missing modality is
        # zero-filled so the batch collate never misaligns tensors.
        if f["rgb"] is not None:
            t_rgb = torch.from_numpy(
                np.asarray(f["rgb"], dtype=np.float32)).permute(2, 0, 1)
            t_rgb = F.interpolate(
                t_rgb[None], size=(self.img_h, self.img_w),
                mode="bilinear", align_corners=False)[0] / 255.0
        else:
            t_rgb = torch.zeros(3, self.img_h, self.img_w,
                                dtype=torch.float32)
        if f["label"] is not None:
            t_label = torch.from_numpy(
                np.asarray(f["label"], dtype=np.float32))[None]
            t_label = F.interpolate(
                t_label[None], size=(self.img_h, self.img_w),
                mode="nearest")[0]
        else:
            t_label = torch.zeros(1, self.img_h, self.img_w,
                                  dtype=torch.float32)
        if f["bev"] is not None:
            t_bev = torch.from_numpy(
                np.asarray(f["bev"], dtype=np.float32))[None]
        else:
            t_bev = torch.zeros(1, GRID_N, GRID_N, dtype=torch.float32)
        mods = [t_rgb, t_label, t_bev]

        if self.augment and self.rng.random() < 0.5:
            # horizontal flip: mirror every modality; steer and the ego-x
            # trajectory axis flip sign, throttle stays.
            mods = [torch.flip(m, dims=[-1]) if m.ndim == 3 else m
                    for m in mods]
            flip = True
        else:
            flip = False

        traj = self._traj_ego(f)
        steer = float(f["steer"])
        if flip:
            steer = -steer
            traj = traj * torch.tensor([-1.0, 1.0])
        action = torch.tensor([steer, float(f["throttle"])],
                              dtype=torch.float32)
        return tuple(mods), traj, action

    @staticmethod
    def collate(batch):
        """Collate (rgb, label, bev) batches into stacked tensors."""
        import torch
        bs = len(batch)
        n_mod = len(batch[0][0])
        out = []
        for m in range(n_mod):
            shape = batch[0][0][m].shape
            stacked = torch.stack([b[0][m] for b in batch])
            out.append(stacked)
        trajs = torch.stack([b[1] for b in batch])
        acts = torch.stack([b[2] for b in batch])
        return tuple(out), trajs, acts
