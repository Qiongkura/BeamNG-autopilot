"""Shared behavioural-cloning model + preprocessing helpers (M3)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class Dave2(nn.Module):
    """NVIDIA DAVE-2 style CNN with BatchNorm; linear scalar output.

    BatchNorm after every conv prevents the feature-collapse (constant
    output) failure mode of the plain DAVE-2 stack.  The head is linear
    instead of tanh because the steering targets live in a narrow range
    and tanh only amplifies collapsed activations.
    """

    def __init__(self, in_channels: int = 3, feat_in: int = 1152):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 24, 5, stride=2), nn.BatchNorm2d(24), nn.ReLU(inplace=True),
            nn.Conv2d(24, 36, 5, stride=2), nn.BatchNorm2d(36), nn.ReLU(inplace=True),
            nn.Conv2d(36, 48, 5, stride=2), nn.BatchNorm2d(48), nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, 3), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_in, 100), nn.ReLU(inplace=True),
            nn.Linear(100, 50), nn.ReLU(inplace=True),
            nn.Linear(50, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features(x))


def conv_feature_size(h: int, w: int, in_channels: int = 3) -> int:
    """Flattened size of the conv stack for a given input HxW."""
    with torch.no_grad():
        dummy = torch.zeros(1, in_channels, h, w)
        dummy = Dave2(in_channels=in_channels).features(dummy)
        return int(dummy.numel())


def preprocess_frame(img: np.ndarray, w: int, h: int) -> torch.Tensor:
    """RGB uint8 frame -> normalized (1, 3, H, W) float tensor in [-1, 1]."""
    small = cv2_resize_rgb(img, w, h)
    arr = small.astype(np.float32).transpose(2, 0, 1) / 255.0 * 2.0 - 1.0
    return torch.from_numpy(arr).unsqueeze(0)


def cv2_resize_rgb(img: np.ndarray, w: int, h: int) -> np.ndarray:
    import cv2
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
