"""Real end-to-end driving network (FSD v12-style, torch CNN).

The numpy ``E2ENet`` in ``neural/__init__.py`` stays as the offline
contract skeleton; this module is the trainable stack.  It consumes the
multimodal shadow observations recorded by ``ShadowRecorder`` - front
RGB, the segmentation label (0=bg 1=road 2=line) and the BEV occupancy
raster - through three compact convolutional encoders, fuses the
features, and directly regresses a trajectory (ego-relative waypoints)
plus the action (steer, throttle), the way FSD v12's "planning is a
neural network" stacks do.

``E2ENetTorch`` replaces the internals of the skeleton without changing
the call contract: ``forward(rgb, label, bev) -> (trajectory, action)``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

GRID_N = 60          # BEV occupancy resolution (n x n)
N_WAYPOINTS = 16     # trajectory length the net emits
N_ACTION = 2         # (steer, throttle)


class _ConvBlock(nn.Module):
    """conv -> group-norm -> relu -> max-pool.

    GroupNorm (not BatchNorm) so training is stable on the small shadow
    datasets (tens to a few hundred frames per episode): BatchNorm with
    tiny batches makes the first Adam steps blow up because the batch
    statistics are noisy and the normalized output amplifies any weight
    shift.
    """

    def __init__(self, cin: int, cout: int, pool: bool = True) -> None:
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1)
        self.norm = nn.GroupNorm(min(8, cout), cout)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.act(self.norm(self.conv(x))))


class E2ENetTorch(nn.Module):
    """Multimodal CNN: (rgb, label, bev) -> (trajectory, action).

    * ``rgb_enc``   - front camera, 3 channels (240x320 at record time)
    * ``label_enc`` - segmentation label, 1 channel
    * ``bev_enc``   - occupancy raster, 1 channel (60x60)

    Each encoder ends with adaptive average pooling to a fixed 4x4 grid
    so the head is resolution-independent; the head is a small MLP that
    emits ``n_waypoints * 2 + 2`` values.
    """

    def __init__(self, grid_n: int = GRID_N, n_waypoints: int = N_WAYPOINTS,
                 latent: int = 256, history: int = 0) -> None:
        super().__init__()
        self.grid_n = int(grid_n)
        self.n_waypoints = int(n_waypoints)
        self.history = int(history)
        self.rgb_enc = nn.Sequential(
            _ConvBlock(3, 16), _ConvBlock(16, 32), _ConvBlock(32, 64))
        self.label_enc = nn.Sequential(
            _ConvBlock(1, 8), _ConvBlock(8, 16))
        self.bev_enc = nn.Sequential(
            _ConvBlock(1, 32), _ConvBlock(32, 64), _ConvBlock(64, 64))
        self.rgb_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.label_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.bev_pool = nn.AdaptiveAvgPool2d((4, 4))
        in_dim = 64 * 16 + 16 * 16 + 64 * 16
        # Temporal encoder: fuses ``history+1`` consecutive frames with a
        # 1-D conv over time; the current speed is appended afterwards so
        # the net can condition braking/acceleration on ego velocity.
        self.temporal: nn.Module | None = None
        head_in = in_dim
        if self.history > 0:
            self.temporal = nn.Conv1d(in_dim, in_dim // 2,
                                      kernel_size=self.history + 1)
            self.temporal_norm = nn.LayerNorm(in_dim // 2)
            head_in = in_dim // 2 + 1
        else:
            self.temporal_norm = None
        self.head = nn.Sequential(
            nn.Linear(head_in, latent),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(latent, latent // 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent // 2, self.n_waypoints * 2 + N_ACTION),
        )

    def forward(self, rgb: torch.Tensor, label: torch.Tensor | None = None,
                bev: torch.Tensor | None = None,
                speed: torch.Tensor | None = None,
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (trajectory (B, N, 2), action (B, 2)).

        Single-frame mode: ``rgb`` is ``(B, 3, H, W)`` float 0..1,
        ``label`` ``(B, 1, H, W)``, ``bev`` ``(B, 1, grid_n, grid_n)``.
        Temporal mode (``history > 0``): each modality is
        ``(B, T, C, H, W)`` with T = history+1 consecutive frames, and
        ``speed`` is ``(B,)`` or ``(B, 1)`` ego velocity.  Missing
        optional modalities are zero-filled so subsets still run.
        """
        b = int(rgb.shape[0])
        dev = rgb.device
        if rgb.dim() == 4:
            rgb = rgb.unsqueeze(1)
        if label is None:
            label = torch.zeros(b, 1, 1, *rgb.shape[3:], device=dev,
                                dtype=rgb.dtype)
        elif label.dim() == 4:
            label = label.unsqueeze(1)
        if bev is None:
            bev = torch.zeros(b, 1, 1, self.grid_n, self.grid_n,
                              device=dev, dtype=rgb.dtype)
        elif bev.dim() == 4:
            bev = bev.unsqueeze(1)
        t = int(rgb.shape[1])
        btc = (b * t,)
        xr = self.rgb_pool(self.rgb_enc(
            rgb.reshape(*btc, *rgb.shape[2:]))).flatten(1)
        xl = self.label_pool(self.label_enc(
            label.reshape(*btc, *label.shape[2:]))).flatten(1)
        xb = self.bev_pool(self.bev_enc(
            bev.reshape(*btc, *bev.shape[2:]))).flatten(1)
        x = torch.cat([xr, xl, xb], dim=1).reshape(b, t, -1)
        if self.temporal is not None:
            x = self.temporal(x.permute(0, 2, 1)).squeeze(-1)
            x = self.temporal_norm(x)
            if speed is None:
                speed = torch.zeros(b, 1, device=dev, dtype=x.dtype)
            else:
                speed = speed.reshape(b, 1).to(x.dtype)
            # ego velocity in m/s: normalise to ~0..0.7 so the raw value
            # never dominates the fused features (throttle blew up to 6.8
            # when a 0-7 m/s input hit the head directly).
            x = torch.cat([x, speed / 10.0], dim=1)
        else:
            x = x.reshape(b, -1)
        out = self.head(x)
        traj = out[:, : self.n_waypoints * 2].reshape(
            -1, self.n_waypoints, 2)
        action = out[:, self.n_waypoints * 2:]
        return traj, action

    def predict_numpy(self, rgb: np.ndarray, label: np.ndarray | None = None,
                      bev: np.ndarray | None = None,
                      device: str = "cpu",
                      ) -> tuple[np.ndarray, np.ndarray]:
        """NumPy convenience wrapper for probes: (traj (N,2), action (2,))."""
        self.eval()
        with torch.no_grad():
            t_rgb = torch.from_numpy(
                np.asarray(rgb, dtype=np.float32)[None] / 255.0)
            t_label = None
            if label is not None:
                t_label = torch.from_numpy(
                    np.asarray(label, dtype=np.float32)[None, None])
            t_bev = None
            if bev is not None:
                t_bev = torch.from_numpy(
                    np.asarray(bev, dtype=np.float32)[None, None])
            traj, action = self.forward(
                t_rgb.to(device), t_label.to(device) if t_label is not None
                else None,
                t_bev.to(device) if t_bev is not None else None)
            return (traj[0].cpu().numpy(), action[0].cpu().numpy())
