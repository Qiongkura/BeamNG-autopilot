"""Dataset over recorded shadow episodes for the real E2E training stack.

``ShadowMultimodalDataset`` turns the multi-modal .npz episodes produced
by ``ShadowRecorder`` (version 2: rgb + segmentation label + BEV raster
+ quality gate) into ``(rgb, label, bev) -> (trajectory_ego, mask,
action)`` training pairs:

* the segmentation label and the vector-space grid are the two
  "perception" views of the stack (semantic head and fused BEV feature
  map); the grid carries the FSD-style channel stack - obstacle /
  drivable / lane / sign in recorded v3 episodes, synthesised from the
  legacy occupancy + drivable raster when an older episode has no fmap;
* the RGB frame is the raw-image view (DAVE-2 / image end-to-end);
* the trajectory is re-expressed in the ego frame using the recorded
  pose so the regression target is well-posed (a single image cannot
  infer the absolute world position); ``mask`` marks the valid
  waypoints (recorded trajectories are NaN-padded, and frames without a
  feasible shadow trajectory are kept for action-only supervision);
* every frame with a sane ego state is kept - the executed
  ``(steer, throttle)`` is ground truth regardless of whether the shadow
  planner found a feasible trajectory, so dropping those frames would
  throw away the bulk of each episode.  ``min_quality`` / ``min_speed``
  remain available for strict trajectory-only or static-frame removal.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

GRID_N = 60
N_WAYPOINTS = 16
N_ACTION = 2


def _has_wedge_restart(spd: np.ndarray, min_static: int = 4,
                      min_after: int = 3) -> bool:
    """True when an episode has a mid-run stop followed by more driving.

    The recorder teleports back to the route start when the rule driver
    wedges into a wall; those episodes contain bad labels (full-lock
    steer into the obstacle).  A stop at the END of the route (the
    end-zone brake) has no driving after it and must NOT be treated as a
    wedge.
    """
    v = np.asarray(spd, dtype=float) < 0.3
    n = len(v)
    i = 0
    while i < n:
        if not v[i]:
            i += 1
            continue
        j = i
        while j < n and v[j]:
            j += 1
        if j - i >= min_static and j + min_after <= n and \
                bool(np.any(~v[j:j + min_after])):
            return True
        i = j
    return False


def _build_index(ep_files, min_quality: float, min_speed: float,
                 drop_wedge_episodes: bool, history: int = 0,
                 dedup: bool = False):
    """Index (file, frame) pairs from the lightweight scalar fields only.

    The camera arrays are megabytes per frame and reading them all up
    front blew past memory once recordings grew (ArrayMemoryError on
    ~1200 frames); the dataset now loads each frame lazily in
    ``__getitem__``.  ``history`` reserves that many earlier frames in
    the same episode so temporal windows never cross episodes.

    ``dedup`` skips near-duplicate consecutive frames (speed / steer /
    throttle almost unchanged): shadow runs contain long static stretches
    whose repeated frames only slow training and over-fit the net to the
    same image.
    """
    idx = []
    for fi, p in enumerate(ep_files):
        with np.load(p, allow_pickle=True) as z:
            n = int(z["t"].shape[0])
            q = np.asarray(z["quality"], dtype=np.float32) \
                if "quality" in z else np.ones(n, dtype=np.float32)
            ok = np.asarray(z["trajectory_ok"], dtype=bool) \
                if "trajectory_ok" in z else np.ones(n, dtype=bool)
            spd = np.asarray(z["speed"], dtype=np.float64)
            steer = np.asarray(z["steer"], dtype=np.float64) \
                if "steer" in z else np.zeros(n, dtype=np.float64)
            throttle = np.asarray(z["throttle"], dtype=np.float64) \
                if "throttle" in z else np.zeros(n, dtype=np.float64)
            has_rgb = "rgb" in z
            has_label = "label" in z
            has_bev = ("bev" in z) or ("fmap" in z)
            if drop_wedge_episodes and _has_wedge_restart(spd):
                continue
            last = None  # (speed, steer, throttle) of the last kept frame
            for i in range(n):
                if i < history:
                    continue
                if float(q[i]) < min_quality:
                    continue
                if float(spd[i]) < min_speed:
                    continue
                if not (has_rgb or has_label or has_bev):
                    continue
                if dedup and last is not None and \
                        abs(float(spd[i]) - last[0]) < 0.15 and \
                        abs(float(steer[i]) - last[1]) < 0.02 and \
                        abs(float(throttle[i]) - last[2]) < 0.02:
                    continue
                last = (float(spd[i]), float(steer[i]),
                        float(throttle[i]))
                idx.append((fi, i, bool(ok[i])))
    return idx


class ShadowMultimodalDataset(torch.utils.data.Dataset):
    """Multimodal (rgb, label, bev) -> (trajectory_ego, mask, action)."""

    def __init__(self, ep_files, min_quality: float = 0.0,
                 min_speed: float = 0.5,
                 drop_wedge_episodes: bool = True,
                 history: int = 0,
                 img_h: int = 120, img_w: int = 160,
                 n_waypoints: int = N_WAYPOINTS,
                 augment: bool = False, seed: int = 0,
                 dedup: bool = False) -> None:
        if isinstance(ep_files, (str, Path)):
            ep_files = [Path(ep_files)]
        self.files = [Path(p) for p in ep_files if Path(p).exists()]
        self.min_quality = float(min_quality)
        self.min_speed = float(min_speed)
        self.img_h = int(img_h)
        self.img_w = int(img_w)
        self.n_waypoints = int(n_waypoints)
        self.history = int(history)
        self.augment = bool(augment)
        self.index = _build_index(self.files, self.min_quality,
                                  self.min_speed,
                                  drop_wedge_episodes,
                                  self.history,
                                  dedup=bool(dedup))
        self._cache: dict[int, list[dict]] = {}
        if self.augment:
            self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.index)

    def _traj_ego(self, x: float, y: float, heading: float,
                 traj_world: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """World trajectory -> ego frame + validity mask.

        Recorded trajectories are variable length and NaN-padded to the
        archive's max width; only the finite prefix is used, the last
        valid point is repeated to reach ``n_waypoints``, and ``mask``
        marks the real prefix (1) vs the padded tail (0).  A frame with
        no feasible trajectory returns an all-zero mask (action-only
        supervision).
        """
        h = float(heading)
        c, s = np.cos(h), np.sin(h)
        rel = np.asarray(traj_world, dtype=np.float32) - \
            np.array([x, y], dtype=np.float32)
        finite = np.isfinite(rel).all(axis=1)
        rel = rel[finite]
        valid = int(finite.sum())
        if valid == 0:
            rel = np.zeros((1, 2), dtype=np.float32)
        ego = np.stack([rel[:, 0] * c + rel[:, 1] * s,
                        -rel[:, 0] * s + rel[:, 1] * c], axis=1)
        n = self.n_waypoints
        valid = min(valid, n)
        if len(ego) >= n:
            ego = ego[:n]
        else:
            pad = np.tile(ego[-1:], (n - len(ego), 1))
            ego = np.concatenate([ego, pad], axis=0)
        mask = np.zeros(n, dtype=np.float32)
        mask[:valid] = 1.0
        return (torch.from_numpy(ego),
                torch.from_numpy(mask))

    def _episode(self, fi: int) -> list[dict]:
        """Process one episode once and cache the small tensors.

        The raw camera arrays are resized to ``(img_h, img_w)`` at load
        time and stored as float tensors, so the whole corpus fits in a
        few hundred MB and every epoch reads plain in-memory tensors
        instead of decompressing .npz per frame (which took ~20 min per
        training run).
        """
        ep = self._cache.get(fi)
        if ep is not None:
            return ep
        z = np.load(self.files[fi], allow_pickle=True)
        n = int(z["t"].shape[0])
        # Read each sensor array ONCE (compressed npz decompresses the
        # whole key per access, so per-frame z["rgb"][i] is quadratic -
        # 88 frames took ~13 s); process from the in-memory arrays.
        rgb = np.asarray(z["rgb"], dtype=np.uint8) if "rgb" in z else None
        label = np.asarray(z["label"], dtype=np.uint8) \
            if "label" in z else None
        bev = np.asarray(z["bev"], dtype=np.float32) \
            if "bev" in z else None
        fmap = np.asarray(z["fmap"], dtype=np.float32) \
            if "fmap" in z else None
        drv = np.asarray(z["drivable"], dtype=np.uint8) \
            if "drivable" in z else None
        traj = np.asarray(z["trajectory"], dtype=np.float64)
        xs = np.asarray(z["x"], dtype=np.float64)
        ys = np.asarray(z["y"], dtype=np.float64)
        hdgs = np.asarray(z["heading"], dtype=np.float64)
        steers = np.asarray(z["steer"], dtype=np.float64)
        thr = np.asarray(z["throttle"], dtype=np.float64)
        spds = np.asarray(z["speed"], dtype=np.float64)
        frames = []
        for i in range(n):
            if rgb is not None:
                t = torch.from_numpy(rgb[i].astype(np.float32)).permute(2, 0, 1)
                # Store resized uint8 (19 KB/frame) so the whole corpus
                # fits in ~100 MB; float conversion happens per item.
                t_rgb = F.interpolate(
                    t[None], size=(self.img_h, self.img_w),
                    mode="bilinear", align_corners=False)[0]
                t_rgb = t_rgb.clamp(0.0, 255.0).round().to(torch.uint8)
            else:
                t_rgb = torch.zeros(3, self.img_h, self.img_w,
                                    dtype=torch.uint8)
            if label is not None:
                t = torch.from_numpy(label[i].astype(np.float32))[None]
                t_label = F.interpolate(
                    t[None], size=(self.img_h, self.img_w),
                    mode="nearest")[0].to(torch.uint8)
            else:
                t_label = torch.zeros(1, self.img_h, self.img_w,
                                      dtype=torch.uint8)
            if bev is not None:
                t_bev = torch.from_numpy(bev[i])[None]
            else:
                t_bev = torch.zeros(1, GRID_N, GRID_N, dtype=torch.float32)
            if fmap is not None and fmap.shape[1] > 0:
                t_fmap = torch.from_numpy(fmap[i])          # (C, N, N)
                if t_fmap.shape[1:] != (GRID_N, GRID_N):
                    t_fmap = F.interpolate(
                        t_fmap[None], size=(GRID_N, GRID_N),
                        mode="nearest")[0]
            else:
                # Legacy episode: synthesise the vector-space channels
                # from the single-channel occupancy + drivable raster.
                t_fmap = torch.zeros(4, GRID_N, GRID_N,
                                     dtype=torch.float32)
                t_fmap[0] = t_bev[0]
                if drv is not None:
                    t_fmap[1] = torch.from_numpy(
                        drv[i].astype(np.float32))
            traj_ego, mask = self._traj_ego(
                float(xs[i]), float(ys[i]), float(hdgs[i]), traj[i])
            frames.append({
                "mods": [t_rgb, t_label, t_fmap],
                "traj": traj_ego,
                "mask": mask,
                "steer": float(steers[i]),
                "throttle": float(thr[i]),
                "speed": float(spds[i]),
            })
        z.close()
        self._cache[fi] = frames
        return frames

    def close(self) -> None:
        self._cache = {}

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __getitem__(self, idx: int):
        fi, i, _ok = self.index[idx]
        ep = self._episode(fi)
        f = ep[i]
        if self.history > 0:
            # Temporal window: stack the last ``history+1`` frames of the
            # same episode along a new time axis (T, C, H, W).
            mods = [torch.stack([ep[j]["mods"][m]
                                 for j in range(i - self.history, i + 1)],
                                dim=0)
                    for m in range(3)]
            speed = torch.tensor([f["speed"]], dtype=torch.float32)
        else:
            mods = list(f["mods"])
            speed = torch.tensor([f["speed"]], dtype=torch.float32)
        traj = f["traj"]
        mask = f["mask"]

        flip = False
        # uint8 -> float for the network (rgb 0..1, label 0..2)
        mods[0] = mods[0].float() / 255.0
        mods[1] = mods[1].float()

        if self.augment:
            if self.rng.random() < 0.5:
                # horizontal flip: mirror every modality; steer and the
                # ego-x trajectory axis flip sign, throttle stays.
                mods = [torch.flip(m, dims=[-1]) for m in mods]
                flip = True
            # mild brightness/contrast jitter on the raw image only
            if self.rng.random() < 0.5:
                gain = float(self.rng.uniform(0.85, 1.15))
                mods[0] = torch.clamp(mods[0] * gain, 0.0, 1.0)

        steer = float(f["steer"])
        if flip:
            steer = -steer
            traj = traj * torch.tensor([-1.0, 1.0])
        action = torch.tensor([steer, float(f["throttle"])],
                              dtype=torch.float32)
        return tuple(mods), traj, mask, action, speed

    @staticmethod
    def collate(batch):
        """Collate (rgb, label, bev) batches into stacked tensors."""
        bs = len(batch)
        n_mod = len(batch[0][0])
        out = [torch.stack([b[0][m] for b in batch]) for m in range(n_mod)]
        trajs = torch.stack([b[1] for b in batch])
        masks = torch.stack([b[2] for b in batch])
        acts = torch.stack([b[3] for b in batch])
        speeds = torch.stack([b[4] for b in batch])
        return tuple(out), trajs, masks, acts, speeds
