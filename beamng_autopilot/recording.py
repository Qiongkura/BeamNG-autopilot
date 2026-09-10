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
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EPISODE_VERSION = 3
EPISODE_SCHEMA = "fsd_shadow_episode"

# Channel order of the fused vector-space feature map (mirrors
# ``BEVFeatureMap.CHANNELS``): obstacle / drivable / lane / sign.
FMAP_CHANNELS = ("obstacle", "drivable", "lane", "sign")


def shadow_expert_sample_ok(*, strict_perception: bool, lane_ref,
                            trajectory) -> tuple[bool, str]:
    """Validate one shadow sample before it becomes an expert demonstration.

    Strict FSD recordings must never learn lateral behaviour from a
    trajectory that was produced without a perception lane reference.
    The kinematic arc fan can still be feasible from occupancy alone, so
    checking only ``trajectory is not None`` is not enough: the sample
    contract must require the same lateral perception contract as the
    runtime planner.
    """
    if trajectory is None:
        return False, "no shadow trajectory"
    try:
        traj = np.asarray(trajectory, dtype=float)
    except (TypeError, ValueError):
        return False, "invalid shadow trajectory"
    if traj.ndim != 2 or traj.shape[0] < 2 or traj.shape[1] < 2 \
            or not np.isfinite(traj[:, :2]).all():
        return False, "invalid shadow trajectory"
    if not strict_perception:
        return True, ""
    if lane_ref is None:
        return False, "perception lane unavailable"
    try:
        lane = np.asarray(lane_ref, dtype=float)
    except (TypeError, ValueError):
        return False, "invalid perception lane"
    if lane.ndim != 2 or lane.shape[0] < 4 or lane.shape[1] < 2 \
            or not np.isfinite(lane[:, :2]).all():
        return False, "perception lane unavailable"
    return True, ""


def git_revision() -> str:
    """Best-effort code revision for episode provenance."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


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
    fmap: np.ndarray | None = None            # (C, N, N) fused vector space
    trajectory: np.ndarray | None = None      # (M, 2) chosen world path
    target_speed: float = 0.0
    lane_src: str = ""                         # which head produced the lane
    cost: float = -1.0
    kind: str = ""                             # which trajectory was chosen
    # camera observations (image end-to-end / multi-modal training)
    rgb: np.ndarray | None = None              # (H, W, 3) uint8 front camera
    label: np.ndarray | None = None            # (H, W) uint8 0=bg 1=road 2=line
    quality: float = 1.0                       # 0..1 shadow-prediction gate


class ShadowRecorder:
    """Accumulate ``ShadowFrame``s and write one episode .npz."""

    def __init__(self, log_dir, sequence: str, provenance=None):
        self.log_dir = Path(log_dir)
        self.sequence = sequence
        self.provenance = dict(provenance or {})
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
        # Fused vector-space feature map (C, N, N) channel-first; older
        # episodes have no fmap and stay empty (the dataset synthesises
        # the channels from bev + drivable).
        fmap_c = fmap_n = 0
        for f in self.frames:
            if f.fmap is not None:
                fmap_c = int(np.asarray(f.fmap).shape[0])
                fmap_n = int(np.asarray(f.fmap).shape[1])
                break
        fmap = np.zeros((n, fmap_c, max(0, fmap_n), max(0, fmap_n)),
                        dtype=np.float32)
        # Camera observations are variable-size: pack into one fixed
        # (n, H, W, 3) / (n, H, W) array using the first frame's size.
        cam_h = cam_w = 0
        for f in self.frames:
            if f.rgb is not None:
                cam_h, cam_w = f.rgb.shape[:2]
                break
        rgb = np.zeros((n, cam_h, cam_w, 3), dtype=np.uint8)
        label = np.zeros((n, cam_h, cam_w), dtype=np.uint8)
        for i, f in enumerate(self.frames):
            if f.trajectory is not None and len(f.trajectory):
                m = min(len(f.trajectory), max(2, max_traj))
                traj[i, :m] = np.asarray(f.trajectory[:m], dtype=float)
                traj_ok[i] = True
            if f.bev_raster is not None:
                bev[i] = np.asarray(f.bev_raster, dtype=np.float32)
            if f.drivable is not None:
                drv[i] = np.asarray(f.drivable, dtype=np.uint8)
            if f.fmap is not None:
                _fm = np.asarray(f.fmap, dtype=np.float32)
                if _fm.shape[:2] == (fmap_c, fmap_n):
                    fmap[i] = _fm
            if f.rgb is not None and cam_h and cam_w:
                rgb[i, :cam_h, :cam_w] = np.asarray(
                    f.rgb[:cam_h, :cam_w], dtype=np.uint8)
            if f.label is not None and cam_h and cam_w:
                label[i, :cam_h, :cam_w] = np.asarray(
                    f.label[:cam_h, :cam_w], dtype=np.uint8)

        tags = {
            "has_rgb": any(f.rgb is not None for f in self.frames),
            "has_label": any(f.label is not None for f in self.frames),
            "has_bev": any(f.bev_raster is not None for f in self.frames),
            "has_fmap": any(f.fmap is not None for f in self.frames),
            "has_trajectory": any(f.trajectory is not None
                                  for f in self.frames),
        }
        provenance = {
            "git_commit": git_revision(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        }
        provenance.update(self.provenance)
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
            fmap=fmap,
            trajectory=traj,
            trajectory_ok=traj_ok,
            target_speed=np.array([f.target_speed for f in self.frames]),
            lane_src=np.array([f.lane_src for f in self.frames],
                              dtype=object),
            cost=np.array([f.cost for f in self.frames]),
            kind=np.array([f.kind for f in self.frames], dtype=object),
            rgb=rgb,
            label=label,
            quality=np.array([f.quality for f in self.frames],
                             dtype=np.float32),
            meta=json.dumps({
                "schema": EPISODE_SCHEMA,
                "sequence": self.sequence,
                "frames": n,
                "cam_h": int(cam_h),
                "cam_w": int(cam_w),
                "fmap_channels": int(fmap_c),
                "episode_version": int(EPISODE_VERSION),
                "tags": tags,
                "provenance": provenance,
            }).encode("utf-8"),
        )
        return out


class EpisodeDataset:
    """torch Dataset over recorded shadow episodes.

    ``modalities`` selects the observation contract:
      * ``"bev"``   - 60x60 occupancy raster (vector-space end-to-end)
      * ``"rgb"``   - front camera image (image end-to-end, DAVE-2 style)
      * ``"label"`` - 3-class segmentation label (0=bg 1=road 2=line)
    ``action`` is always ``(steer, throttle)`` - the executed control.
    ``min_quality`` drops shadow frames whose prediction was gated out
    at recording time, so bad shadow samples never poison training.
    """

    def __init__(self, ep_files, modalities=("bev",), min_quality: float = 0.0):
        if isinstance(ep_files, (str, Path)):
            ep_files = [Path(ep_files)]
        self.files = [Path(p) for p in ep_files if Path(p).exists()]
        self.modalities = tuple(modalities)
        self.meta: list[dict] = []
        self.schemas: list[str] = []
        self.episode_versions: list[int | None] = []
        self.provenance: list[dict] = []
        for m in self.modalities:
            if m not in ("bev", "rgb", "label", "drivable"):
                raise KeyError(f"unknown modality: {m}")
        # Pre-filter by the recorded quality gate so the index stays
        # stable across iterations.
        self._idx: list[tuple[int, int]] = []
        for fi, p in enumerate(self.files):
            with np.load(p, allow_pickle=True) as z:
                meta = {}
                if "meta" in z:
                    try:
                        meta = json.loads(
                            np.asarray(z["meta"]).item().decode("utf-8"))
                    except Exception:
                        meta = {}
                schema = str(meta.get("schema", "") or "")
                if schema and schema != EPISODE_SCHEMA:
                    raise ValueError(
                        f"unknown episode schema {schema!r} in {p}")
                self.meta.append(meta)
                self.schemas.append(schema or "legacy")
                try:
                    version = (
                        int(meta["episode_version"])
                        if "episode_version" in meta else None)
                except (TypeError, ValueError):
                    version = None
                self.episode_versions.append(version)
                self.provenance.append(
                    dict(meta.get("provenance") or {}))
                n = int(z["t"].shape[0])
                q = np.asarray(z["quality"], dtype=np.float32) \
                    if "quality" in z else np.ones(n, dtype=np.float32)
                for i in range(n):
                    if float(q[i]) >= min_quality:
                        self._idx.append((fi, i))

    def __len__(self) -> int:
        return len(self._idx)

    def __getitem__(self, idx: int):
        import torch
        fi, i = self._idx[idx]
        with np.load(self.files[fi], allow_pickle=True) as z:
            obs = []
            for m in self.modalities:
                if m == "bev":
                    obs.append(torch.from_numpy(
                        np.asarray(z["bev"][i], dtype=np.float32)))
                elif m == "drivable":
                    obs.append(torch.from_numpy(
                        np.asarray(z["drivable"][i], dtype=np.uint8)))
                elif m == "rgb":
                    obs.append(torch.from_numpy(
                        np.asarray(z["rgb"][i], dtype=np.uint8)))
                elif m == "label":
                    obs.append(torch.from_numpy(
                        np.asarray(z["label"][i], dtype=np.uint8)))
            act = torch.tensor(
                [float(z["steer"][i]), float(z["throttle"][i])],
                dtype=torch.float32)
            if len(obs) == 1:
                return obs[0], act
            return tuple(obs), act
