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
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from beamng_autopilot import config

N_CLASSES = 3
CLASS_NAMES = ["background", "asphalt", "line"]
_INFER_W, _INFER_H = 536, 403  # 训练分辨率
# 标线连通域最小面积：小于它的块视为路面纹理/阴影噪点（真实标线
# 在 536x403 上至少 ~150 px；远距离细线也能到几十 px，但形态细长，
# 由 max(cw, ch) >= 24 分支保留）。
_LINE_MIN_AREA_PX = 150
# Soil / gravel colour window (OpenCV HSV): warm hue, saturated enough and
# not in shadow.  Used only to keep the car on the PAVED surface - see
# ``strip_soil_from_road``.
_SOIL_HUE_MIN, _SOIL_HUE_MAX = 8, 35
_SOIL_SAT_MIN = 45
_SOIL_VAL_MIN = 60
# Strip soil from the drivable mask only when what is left is still a real
# paved surface.  Measured paved-share (fraction of the road mask that is
# NOT soil-coloured), 20 frames each:
#   paved road  : min 0.74, p10 0.87, p50 0.99
#   all-dirt    : p10 0.36, p50 0.66   (p90 1.00 = the soil detector found
#                 nothing to remove, so stripping is a no-op on those)
# 0.70 separates the two populations with margin, and it keeps the worst
# measured paved frame (0.752, a 14k px shoulder patch) strippable - a 0.80
# floor silently kept exactly the frames this exists for.
_SOIL_STRIP_MIN_PAVED_FRAC = 0.70


def snow_or_soil_mask(frame_rgb: np.ndarray) -> np.ndarray:
    """Warm, saturated ground (soil / gravel / sand) in the frame."""
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]
    return ((H >= _SOIL_HUE_MIN) & (H <= _SOIL_HUE_MAX)
            & (S > _SOIL_SAT_MIN) & (V > _SOIL_VAL_MIN))


def strip_soil_from_road(road: np.ndarray,
                         frame_rgb: np.ndarray,
                         route_is_dirt: bool = False) -> np.ndarray:
    """Keep the paved surface only; dirt is road on dirt ROUTES only.

    The model calls the dirt shoulder beside a paved road "asphalt"
    (measured on the 2026-09-18 east_coast run: up to 24.8% of the road
    mask was soil, a single 14k px patch joined to the road core).  The
    car must stay on the PAVED surface, so soil-coloured pixels are
    removed from the drivable mask.

    Whether dirt counts as road is a property of the ROUTE
    (AGENTS.md「驾驶约束」3), not of one frame's pixel share.  The old
    per-frame escape hatch (skip stripping when the paved share fell
    below 0.70) was measured live and removed: on the east_coast bend the
    mask cut off mid-road, the paved share dropped to 0.40, the strip was
    skipped, and the drivable layer - dirt included - walked the car off
    the pavement while ``lane=sensor`` (2026-09-19 14:31, 40 frames off
    the road).  ``route_is_dirt=True`` (the explicit --dirt-route opt-in
    for a genuine dirt route) returns the mask untouched; the default
    strips unconditionally.
    """
    m = np.asarray(road, dtype=bool)
    if not m.any() or route_is_dirt:
        return m
    soil = snow_or_soil_mask(frame_rgb)
    paved = m & ~soil
    if not paved.any():
        # Nothing paved survived: on a paved route this is a mask
        # failure, and returning the soil-included mask would let the
        # drivable layer claim the dirt is road.  Return an EMPTY mask
        # so the strict fail-closed path (no road evidence -> no
        # candidate) owns it instead.
        return paved
    return paved


def iou_from_accum(inter: np.ndarray, union: np.ndarray) -> np.ndarray:
    """由全局累加的 inter/union 计算各类 IoU，未出现的类别记 0。

    旧写法逐帧 ``inter / union if union else 1.0`` 再平均：对稀疏类
    （标线）几乎每帧 union==0，被记成满分，line IoU 被显著虚高。
    全局累加只对真实存在的类别给分：``IoU = total_inter /
    total_union``，全程没有该类别则记 0。
    """
    inter = np.asarray(inter, dtype=np.float64)
    union = np.asarray(union, dtype=np.float64)
    present = union > 0
    ious = np.zeros_like(inter)
    ious[present] = inter[present] / union[present]
    return ious


def fill_interior_holes(mask: np.ndarray) -> np.ndarray:
    """Fill every background region fully enclosed by ``mask``.

    A background region touching the frame border is outside the mask; a
    region that never reaches the border is a hole in it.
    """
    m = np.asarray(mask, dtype=np.uint8)
    h, w = m.shape
    padded = np.zeros((h + 2, w + 2), np.uint8)
    padded[1:-1, 1:-1] = m
    ff = padded.copy()
    cv2.floodFill(ff, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 2)
    holes = ff[1:-1, 1:-1] == 0
    return np.maximum(m, holes.astype(np.uint8)).astype(bool)


def constrain_line_to_road(line: np.ndarray, road: np.ndarray,
                           ksize: int = 7) -> np.ndarray:
    """Keep the line pixels that lie on the road surface.

    The model calls bright paint NOT asphalt, so the road mask carries
    holes exactly along the markings (measured on the east_coast 300 s
    run: 97.6% of the road-mask hole pixels sit on the line class).
    Intersecting the line mask with the dilated road mask therefore ate
    the marking's core and left a hollow outline - 56.8% of every marking
    went missing (2965 px -> 1281 px per frame), which then fragmented
    into the short thin strokes the pairing gates reject.

    Paint enclosed by road IS on the road, so the road mask's interior
    holes are filled before the containment test.  The false lines this
    constraint exists for - grass edges, walls, stones - sit OUTSIDE the
    road region and are still rejected.
    """
    m = np.asarray(line, dtype=bool)
    if not m.any():
        return m
    rd = fill_interior_holes(np.asarray(road, dtype=bool)).astype(np.uint8)
    rd = cv2.dilate(rd, cv2.getStructuringElement(
        cv2.MORPH_RECT, (int(ksize), int(ksize))))
    return m & rd.astype(bool)


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


_ACTIVE_MAP: str | None = None


def set_active_map(name: str | None) -> None:
    """登记当前关卡，default_model_path 据此优先选按地图专家模型。

    布局：logs/m5_seg/seg_model/by_map/<地图名>/best.pt（不存在则回退
    基础 best.pt）。必须在 Segmenter 初始化之前调用（connector 构造
    时自动登记）。
    """
    global _ACTIVE_MAP
    _ACTIVE_MAP = name


def default_model_path() -> Path | None:
    """按地图专家模型优先（by_map/<map>/best.pt），回退基础 best.pt。"""
    base = config.LOGS_DIR / "m5_seg" / "seg_model"
    if _ACTIVE_MAP:
        by_map = base / "by_map" / _ACTIVE_MAP / "best.pt"
        if by_map.is_file():
            return by_map
    p = base / "best.pt"
    return p if p.is_file() else None


class Segmenter:
    """UNet segmentation over an RGB frame, with mask post-processing."""

    def __init__(self, model_path=None, device=None, use_half: bool = True,
                 temporal_smooth: bool = False):
        path = Path(model_path) if model_path else default_model_path()
        if path is None:
            raise FileNotFoundError(
                "分割模型不存在；先运行 scripts/m5_train_seg.py 训练，"
                f"或传入 model_path（默认 {config.LOGS_DIR}/m5_seg/"
                "seg_model/best.pt）")
        # Which checkpoint is loaded decides how every map behaves, so it
        # is kept on the instance: an unpinned run must be attributable
        # instead of silently using whatever happens to be deployed.
        self.model_path = path
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(path, map_location=self.device)
        self.model = SegUNet(
            n_classes=int(ckpt.get("n_classes", N_CLASSES)))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()
        # GPU 上半精度推理：显存/延迟都减半，训练好的 BN 运行统计在
        # eval 模式下不受影响。CPU 保持 fp32。
        self.half = bool(use_half) and self.device == "cuda"
        if self.half:
            self.model.half()
            # 预热：第一次 CUDA 推理会触发 kernel 编译/显存分配，放在
            # init 里做掉，避免驾驶循环首帧卡顿。
            with torch.no_grad():
                self.model(torch.zeros(
                    1, 3, _INFER_H, _INFER_W, device=self.device,
                    dtype=torch.float16))
        # 标线时序一致性（组件滞回，默认关闭）：消融显示它对像素 line
        # IoU 略负（近场 0.333 -> 0.323），但对车道配对的稳定价值无法
        # 离线验证，保留为 opt-in 选项。
        self.temporal_smooth = bool(temporal_smooth)
        self._prev_line: np.ndarray | None = None
        # Route surface context (AGENTS.md「驾驶约束」3): dirt counts as
        # road only on an explicitly dirt ROUTE (--dirt-route).  Default
        # paved -> soil is stripped from the road mask unconditionally.
        self.route_is_dirt = False
        self.class_names = list(ckpt.get(
            "class_names", CLASS_NAMES))
        self._line_idx = self.class_names.index("line") \
            if "line" in self.class_names else 2
        self._road_idx = self.class_names.index("asphalt") \
            if "asphalt" in self.class_names else 1
        self._bg_idx = self.class_names.index("background") \
            if "background" in self.class_names else 0

    def _infer_logits(self, frame_rgb: np.ndarray):
        """One forward pass: the logits tensor for ``frame_rgb``."""
        small = cv2.resize(frame_rgb, (_INFER_W, _INFER_H),
                           interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(small).permute(2, 0, 1).float().div_(255.0)
        x = x.unsqueeze(0).to(self.device)
        if self.half:
            x = x.half()
        with torch.no_grad():
            return self.model(x)

    def _argmax_masks(self, logits, frame_rgb: np.ndarray):
        """The plain argmax road/line masks at the frame resolution."""
        h, w = frame_rgb.shape[:2]
        pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
        return pred == self._road_idx, pred == self._line_idx

    def predict(self, frame_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (road_mask, line_mask) at the input frame resolution."""
        road, line = self._argmax_masks(self._infer_logits(frame_rgb),
                                        frame_rgb)
        return self._postprocess(frame_rgb, road, line)

    def _postprocess(self, frame_rgb: np.ndarray, road, line):
        """The existing mask pipeline: soil, morphology, hysteresis, gates."""
        # 铺装路面约束：模型会把铺装路两侧的土肩也判成路面（实测最坏一帧
        # 掩码里 24.8% 是土、单块 1.4 万像素且与主路面连通）。车必须待在
        # 铺装面上，所以在"确实是铺装路"时把土色像素从可行驶掩码里去掉；
        # 全土路（剥完不足 80%）保持原样 —— 那里土就是路面。见
        # ``strip_soil_from_road``。
        road = strip_soil_from_road(
            road, frame_rgb,
            route_is_dirt=self.route_is_dirt)
        # 标线掩码形态学清理：去掉孤立噪点、弥合小断裂
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        line = cv2.morphologyEx(line.astype(np.uint8), cv2.MORPH_CLOSE,
                                k).astype(bool)
        # 时序组件滞回：只保留与上一帧标线（小膨胀容忍车体移动）重叠的
        # 连通域。单帧闪现的假线（草边/阴影/石头）会被丢弃；整帧都无
        # 稳定组件时（快速转弯/急变场景）保留原结果，避免误清空。这样
        # 只删不增，不会像补间隙那样把线加粗。
        if self.temporal_smooth and self._prev_line is not None and line.any() \
                and self._prev_line.shape == line.shape:
            k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            prev_d = cv2.dilate(
                self._prev_line.astype(np.uint8), k3).astype(bool)
            n, labels, stats, _ = cv2.connectedComponentsWithStats(
                line.astype(np.uint8), 8)
            keep = np.zeros_like(line)
            for i in range(1, n):
                comp = labels == i
                if float(comp[prev_d].mean()) >= 0.3:
                    keep[comp] = True
            if keep.any():
                line = keep
        self._prev_line = line.copy()
        # 物理约束：标线必须位于路面上。石头/护墙/草地边缘与标线视觉
        # 特征相似，模型常把它们误检为线；这些物体不在沥青路面上，用
        # 膨胀后的路面掩码约束即可滤掉（标线紧贴路面，边缘容忍 ~3px）。
        # 路面掩码在标线处是破洞的（模型把白漆判为非路面），直接相与
        # 会把标线芯挖空成轮廓，故先补内部空洞 —— 见
        # ``constrain_line_to_road``。
        #
        # When the model over-predicts the road (bridge shadows, walls and
        # sky merged into "asphalt"), a dilated road mask covers the whole
        # frame and lets every false line through.  Lines are then filtered
        # by their own shape instead: a real painted line is a long thin
        # stroke; texture / shadow specks are small blobs.  A component is
        # kept when it is elongated (major axis much longer than minor) or
        # big enough to be a real marking; scattered specks are dropped.
        line = constrain_line_to_road(line, road)
        if line.any():
            n, labels, stats, _ = cv2.connectedComponentsWithStats(
                line.astype(np.uint8), 8)
            keep = np.zeros_like(line)
            for i in range(1, n):
                x, y, cw, ch, area = stats[i]
                # 细长判据：连通域包围盒的长边 vs 短边。真实标线（含
                # 虚线片段）长宽比通常 >= 3；噪点块接近方形。
                long_side = max(cw, ch)
                short_side = min(cw, ch)
                if area >= _LINE_MIN_AREA_PX or (
                        long_side >= 20 and short_side >= 2
                        and long_side >= 2.5 * short_side):
                    keep[labels == i] = True
            line = keep
        return road, line

    def predict_proba(self, frame_rgb: np.ndarray, *, _logits=None):
        """Class probability maps at the input frame resolution (plan E1).

        Returns ``(maps, raw_road, raw_line)``: per-class probabilities
        (resized in probability space) plus the plain argmax decisions
        taken BEFORE the post-processing pipeline, so a caller can gate
        on the probabilities itself (``vision.seg_probs.gate_masks``)
        instead of inheriting a decision that threw the confidence away.

        ``_logits`` lets a caller that already ran the forward pass hand
        the tensor in, so one inference can serve both this API and the
        mask pipeline (see :meth:`predict_with_probs`).
        """
        from beamng_autopilot.vision.seg_probs import softmax_maps
        h, w = frame_rgb.shape[:2]
        logits = self._infer_logits(frame_rgb) if _logits is None else _logits
        probs = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()
        maps_small = softmax_maps(probs, line_index=self._line_idx,
                                  road_index=self._road_idx,
                                  background_index=self._bg_idx)
        out = {}
        for name in ("line", "road", "background"):
            m = getattr(maps_small, name)
            if m is None:
                out[name] = None
                continue
            out[name] = cv2.resize(np.asarray(m, dtype=np.float32), (w, h),
                                   interpolation=cv2.INTER_LINEAR)
        maps = type(maps_small)(
            line=out["line"], road=out["road"], background=out["background"])
        raw_road = maps.road >= maps.line
        raw_line = maps.line >= maps.road
        if maps.background is not None:
            raw_road = raw_road & (maps.road >= maps.background)
            raw_line = raw_line & (maps.line >= maps.background)
        return maps, raw_road, raw_line

    def predict_with_probs(self, frame_rgb: np.ndarray):
        """One inference -> ``(road_mask, line_mask, probability maps)``.

        Same pipeline output as :meth:`predict` (pinned by a test) plus
        the class probabilities of the SAME forward pass, so a caller can
        gate on them (plan phase E1) without paying for a second
        inference.
        """
        logits = self._infer_logits(frame_rgb)
        road, line = self._argmax_masks(logits, frame_rgb)
        road, line = self._postprocess(frame_rgb, road, line)
        maps, _raw_road, _raw_line = self.predict_proba(
            frame_rgb, _logits=logits)
        return road, line, maps

    def reset(self) -> None:
        """Clear image-space hysteresis after a discontinuity."""
        self._prev_line = None

    def detect_lines(self, frame_rgb, cam_model, pos, heading,
                     ground_z: float | None = None, *,
                     line_mask: np.ndarray | None = None,
                     road_mask: np.ndarray | None = None) -> list:
        """Line mask -> LaneMarking list (reuses the classic pipeline).

        The learned line mask is fused with a classic-CV bright-stroke
        mask: the UNet may miss the markings on an unseen scene (bridge /
        coastal roads), while painted lines are physically brighter than
        the asphalt, so a brightness threshold reliably recovers them.
        Both masks share the same ground-plane back-projection pipeline.
        """
        from beamng_autopilot.vision.lanes import (
            _mask_to_markings, WHITE_SAT_MAX, recover_dashed_boundaries)

        # An explicit (possibly empty) mask is authoritative: do not run
        # inference again or discard the head's temporal fusion.
        if line_mask is None:
            _, line = self.predict(frame_rgb)
        else:
            line = np.asarray(line_mask, dtype=bool)
            if line.shape != frame_rgb.shape[:2]:
                raise ValueError("line_mask shape must match image shape")
        # Classic-CV bright-stroke recovery, contrast-based: on a light
        # concrete road the absolute brightness of the paint and the
        # pavement are both high, so a global threshold fires on the whole
        # frame.  The paint is still BRIGHTER than the pavement around it,
        # so compare each pixel against its local neighbourhood mean.
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1]
        hue = hsv[:, :, 0]
        val = hsv[:, :, 2]
        local = cv2.blur(gray, (9, 9))
        bright = gray > local + 12
        # White paint: brighter than the neighbourhood and low saturation.
        cv_white = bright & (sat <= WHITE_SAT_MAX)
        # Yellow paint (left edge / centre line): yellowish hue, not too
        # saturated (sun-bleached paint reads as low-sat yellow).
        cv_yellow = ((hue >= 12) & (hue <= 45) & (sat >= 30)
                     & (val >= 100) & bright)
        # The HSV yellow prior (looser than cv_yellow - it catches the
        # bleached centre paint the learned mask absorbed).  Its pixels
        # must NOT stay in the white mask, or the centre paint is
        # re-labelled white and the RHT centre-line policy never fires
        # (live east_coast 2026-09-19: 'dashed/white' centre, mirror won).
        from beamng_autopilot.vision.yellow_line_mask import yellow_line_mask
        ym = yellow_line_mask(frame_rgb)
        cv_yellow = cv_yellow | ym
        cv_white = cv_white & ~ym
        out: list = []
        white_mask = (line | cv_white).astype(np.uint8) * 255
        if white_mask.any():
            out.extend(_mask_to_markings(
                white_mask, "white", cam_model, pos, heading,
                ground_z=ground_z))
            # The shape gates above keep only long strokes, but the town
            # ``line`` class is mostly short blocks (median 17 components
            # per frame, median height 7 px), so a dashed lane line leaves
            # no usable boundary behind.  Group the discarded collinear
            # fragments back into one long boundary so the own lane can
            # have two detected edges again.
            #
            # This runs on the LEARNED mask, not on ``white_mask``: the
            # classic-CV bright-stroke union above exists to recover paint
            # the model misses, but its texture edges are not paint and
            # they pollute the grouping (measured: grouping the union
            # yields a 23.9% paired rate, the learned mask alone 43.0%).
            #
            # BEAMNG_DASHED_RECOVERY=0 disables the stage (default ON).
            #
            # It was default-OFF because it cost ~70 ms/frame, about half
            # of the whole segmentation stage, and a live A/B showed it
            # losing lane continuity (lane_sel=sensor ~0-3% with it on,
            # 17.7% off) - a stage that doubles the perception cost turns
            # into MISSING perception because the FSD tick defers heads on
            # overrun.  The cost is now fixed: the camera pose is built once
            # per frame instead of once per sampled pixel, and the label
            # image is scanned once instead of once per component, which
            # takes `_mask_fragment_polylines` from 53.7 ms to 3.2 ms
            # (whole stage ~70 ms -> ~3.6 ms).  On a same-episode A/B the
            # benefit is intact: paired own-lane 13.0% -> 28.2% and
            # lane_sel=sensor 27.7% -> 59.7%.
            if os.environ.get("BEAMNG_DASHED_RECOVERY", "1") != "0":
                out.extend(recover_dashed_boundaries(
                    np.asarray(line, dtype=np.uint8) * 255,
                    cam_model, pos, heading, ground_z=ground_z,
                    yellow_mask=ym))
        if cv_yellow.any():
            for _ymk in _mask_to_markings(
                    cv_yellow.astype(np.uint8) * 255, "yellow",
                    cam_model, pos, heading, ground_z=ground_z):
                # Yellow-paint candidates must be PAINT: linearly
                # elongated, and lying ON the published road mask.  Live
                # east_coast 2026-09-19: the classic yellow detector lit
                # up on yellow-green dirt/grass (round blobs, off the
                # pavement), a false blob became the "centre line" and
                # the centre-line policy walked the car off the road.
                _pix = np.asarray(getattr(_ymk, "pixels", None),
                                  dtype=float)
                if _pix.ndim != 2 or len(_pix) < 6:
                    continue
                _bw = _pix.max(axis=0) - _pix.min(axis=0)
                _long = max(float(_bw[0]), float(_bw[1]))
                _short = max(1.0, min(float(_bw[0]), float(_bw[1])))
                if _long / _short < 2.5:
                    continue                     # a blob, not a line
                if road_mask is not None:
                    _rh, _rw = road_mask.shape[:2]
                    _yy = np.clip(_pix[:, 1].astype(int), 0, _rh - 1)
                    _xx = np.clip(_pix[:, 0].astype(int), 0, _rw - 1)
                    if float(np.count_nonzero(
                            road_mask[_yy, _xx])) / len(_pix) < 0.5:
                        continue                 # off the pavement
                out.append(_ymk)
        return out

    def offroad_mask(self, frame_rgb: np.ndarray) -> np.ndarray:
        """True where the frame is NOT asphalt (feeds edge extraction)."""
        road, _ = self.predict(frame_rgb)
        return ~road
