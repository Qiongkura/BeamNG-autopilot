"""Learning-based road/lane segmentation (inference + model definition).

Replaces the classic-CV colour thresholds of ``LaneDetector`` and
``estimate_pavement_edges`` with a small UNet trained on BeamNG.tech
annotation ground truth.  The model runs on plain RGB frames, so both the
Steam (window capture) and Tech (camera sensor) runtimes share it; when no
model file is present the caller falls back to the classic-CV pipeline.

Output masks are consumed by the existing geometry pipeline:
  * ``line_mask`` -> :func:`beamng_autopilot.vision.lanes._mask_to_markings`
    (connected components + ground back-projection + kind classification)
  * ``road_mask`` -> the off-road mask input of ``estimate_pavement_edges``
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from beamng_autopilot import config

N_CLASSES = 3
CLASS_NAMES = ["background", "asphalt", "line"]
_INFER_W, _INFER_H = 536, 403  # 训练分辨率


class SegUNet(nn.Module):
    """Lightweight UNet: 3 encoder blocks + skip connections (~1.3M params)."""

    def __init__(self, in_channels: int = 3, n_classes: int = N_CLASSES):
        super().__init__()

        def _blk(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True))

        self.e1 = _blk(in_channels, 32)
        self.e2 = _blk(32, 64)
        self.e3 = _blk(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.mid = _blk(128, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.d2 = _blk(64 + 128, 64)   # up2(64) + skip e3(128)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.d1 = _blk(32 + 64, 32)    # up1(32) + skip e2(64)
        self.up0 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.d0 = _blk(16 + 32, 32)    # up0(16) + skip e1(32)
        self.head = nn.Conv2d(32, n_classes, 1)

    def forward(self, x):
        x1 = self.e1(x)
        x2 = self.e2(self.pool(x1))
        x3 = self.e3(self.pool(x2))
        m = self.mid(self.pool(x3))
        # 上采样结果对齐 skip 连接尺寸（输入奇数尺寸时池化会差 1px）
        u2 = F.interpolate(self.up2(m), size=x3.shape[-2:],
                           mode="bilinear", align_corners=False)
        d2 = self.d2(torch.cat([u2, x3], dim=1))
        u1 = F.interpolate(self.up1(d2), size=x2.shape[-2:],
                           mode="bilinear", align_corners=False)
        d1 = self.d1(torch.cat([u1, x2], dim=1))
        u0 = F.interpolate(self.up0(d1), size=x1.shape[-2:],
                           mode="bilinear", align_corners=False)
        d0 = self.d0(torch.cat([u0, x1], dim=1))
        return self.head(d0)


def default_model_path() -> Path | None:
    """logs/m5_seg/seg_model/best.pt when it exists, else None."""
    p = config.LOGS_DIR / "m5_seg" / "seg_model" / "best.pt"
    return p if p.is_file() else None


class Segmenter:
    """UNet segmentation over an RGB frame, with mask post-processing."""

    def __init__(self, model_path=None, device=None):
        path = Path(model_path) if model_path else default_model_path()
        if path is None:
            raise FileNotFoundError(
                "分割模型不存在；先运行 scripts/m5_train_seg.py 训练，"
                f"或传入 model_path（默认 {config.LOGS_DIR}/m5_seg/"
                "seg_model/best.pt）")
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(path, map_location=self.device)
        self.model = SegUNet(
            n_classes=int(ckpt.get("n_classes", N_CLASSES)))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()
        self.class_names = list(ckpt.get(
            "class_names", CLASS_NAMES))
        self._line_idx = self.class_names.index("line") \
            if "line" in self.class_names else 2
        self._road_idx = self.class_names.index("asphalt") \
            if "asphalt" in self.class_names else 1

    def predict(self, frame_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (road_mask, line_mask) at the input frame resolution."""
        h, w = frame_rgb.shape[:2]
        small = cv2.resize(frame_rgb, (_INFER_W, _INFER_H),
                           interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(small).permute(2, 0, 1).float().div_(255.0)
        x = x.unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
        pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
        road = pred == self._road_idx
        line = pred == self._line_idx
        # 标线掩码形态学清理：去掉孤立噪点、弥合小断裂
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        line = cv2.morphologyEx(line.astype(np.uint8), cv2.MORPH_CLOSE,
                                k).astype(bool)
        return road, line

    def detect_lines(self, frame_rgb, cam_model, pos, heading,
                     ground_z: float | None = None) -> list:
        """Line mask -> LaneMarking list (reuses the classic pipeline)."""
        from beamng_autopilot.vision.lanes import _mask_to_markings

        _, line = self.predict(frame_rgb)
        if not line.any():
            return []
        return _mask_to_markings(
            line.astype(np.uint8) * 255, "white", cam_model, pos, heading,
            ground_z=ground_z)

    def offroad_mask(self, frame_rgb: np.ndarray) -> np.ndarray:
        """True where the frame is NOT asphalt (feeds edge extraction)."""
        road, _ = self.predict(frame_rgb)
        return ~road