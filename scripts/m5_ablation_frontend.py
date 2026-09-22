"""T11 front-end ablation: classic CV candidates vs the learned mask.

One variable moves at a time - the LINE CANDIDATE FRONT-END - while the
post-processing chain, the data and the metric stay fixed.  The plan's
acceptance list for T11 asks for exactly this shape of report:

* ``raw`` -> ``final`` full chain IoU (front-end alone, then after the
  production post-process), so a good front-end that the post-process
  ruins cannot hide;
* **deleted true-line pixels** and **kept false-line pixels** counted
  SEPARATELY (the plan: "把误删真线和保留假线分别计量" - one number cannot
  express both failure modes);
* visible-segment geometry (component count and the longest component's
  extent) instead of IoU alone;
* latency per stage.

Arms (classic CV candidates named in the literature the project reviewed):

    model        the learned segmentation's line mask (what we ship today)
    abs          absolute colour thresholds - the pre-existing front-end
    tophat_otsu  grey Top-Hat + per-channel Otsu (L01/L07 family)
    ycbcr_pct    Y high-percentile + Cb low-percentile (L32/L17 family)

The percentile values are the literature's; they are NOT calibrated on
this project's data, and the script says so in its output rather than
pretending the numbers transfer.  No training, no threshold tuning, no
production switch changes: this measures, it does not select.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beamng_autopilot.vision.segmentation import (  # noqa: E402
    Segmenter,
    constrain_line_to_road,
    filter_line_shape,
    strip_soil_from_road,
)

#: The literature's percentiles (Son L32: Y top 3%, Cb bottom 1%; the
#: 鱼兆伟 L17 variant: 0.98 / 0.02).  Published, NOT calibrated here.
Y_PCT = 0.97
CB_PCT = 0.02
#: Top-Hat structuring element, pixels.  Line width on our frames is 4-6 px,
#: so the kernel is chosen slightly wider - and that choice is recorded.
TOPHAT_KERNEL = 9
ARMS = ("model", "abs", "tophat_otsu", "ycbcr_pct")
#: Training contract (m5_train_seg / m5_seg_stage_eval): 0 bg, 1 road,
#: 2 line, 255 ignore.  Scoring ignores the 255 area - an unlabelled pixel
#: is not a background pixel, and the same value is used by the T01-fixed
#: stage evaluator so the two reports are comparable.
IGNORE_VALUE = 255


def _load_frames(run_dirs, limit: int | None = None):
    frames = []
    for rd in run_dirs:
        for f in sorted(glob.glob(str(Path(rd) / "frame_*.npz"))):
            z = np.load(f)
            if "colour" not in z.files or "label" not in z.files:
                continue
            frames.append((np.asarray(z["colour"], dtype=np.uint8),
                           np.asarray(z["label"], dtype=np.uint8)))
            if limit is not None and len(frames) >= limit:
                return frames
    return frames


def arm_abs(rgb) -> np.ndarray:
    """The pre-existing absolute-threshold colour front-end."""
    from beamng_autopilot.vision.lanes import _color_masks
    masks = _color_masks(rgb)
    out = np.zeros(rgb.shape[:2], dtype=bool)
    for _name, m in masks:
        out |= (m > 0)
    return out


def arm_tophat_otsu(rgb) -> np.ndarray:
    """Grey Top-Hat + Otsu - the classic bright-stroke detector."""
    import cv2
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (TOPHAT_KERNEL,
                                                   TOPHAT_KERNEL))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)
    if not tophat.any():
        return np.zeros(gray.shape, dtype=bool)
    thr, _ = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return tophat >= max(1.0, float(thr))


def arm_ycbcr_pct(rgb) -> np.ndarray:
    """Y high percentile OR Cb low percentile (white / yellow paint)."""
    import cv2
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    y = ycrcb[:, :, 0].astype(np.float32)
    cb = ycrcb[:, :, 2].astype(np.float32)
    y_thr = float(np.percentile(y, 100.0 * Y_PCT))
    cb_thr = float(np.percentile(cb, 100.0 * CB_PCT))
    return (y >= y_thr) | (cb <= cb_thr)


def arm_model(rgb, seg: Segmenter) -> np.ndarray:
    road, line = seg.predict(rgb)
    arm_model.last_road = road
    return np.asarray(line, dtype=bool)


def _final_chain(raw: np.ndarray, road: np.ndarray, seg: Segmenter,
                 rgb) -> np.ndarray:
    """The SAME production post-process for every arm (one variable only)."""
    m = seg._morph_close_line(np.asarray(raw, dtype=bool))
    m = constrain_line_to_road(m, road)
    return filter_line_shape(m)


def _longest_component_extent(mask) -> int:
    """Longest bounding-box side of the longest component, in pixels.

    The LONGER side, not the width: a lane marking is a vertical stroke in
    the image, so measuring the width would report 4-6 px for every marking
    and hide exactly the visible-segment geometry this metric exists for.
    """
    import cv2
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(
        np.asarray(mask, dtype=np.uint8), 8)
    if n <= 1:
        return 0
    return int(max(max(int(stats[i][2]), int(stats[i][3]))
                   for i in range(1, n)))


def _components(mask) -> int:
    import cv2
    n, _l, _s, _ = cv2.connectedComponentsWithStats(
        np.asarray(mask, dtype=np.uint8), 8)
    return int(n - 1)


def _iou(pred, gt, valid) -> float | None:
    p = np.asarray(pred, dtype=bool) & valid
    g = np.asarray(gt, dtype=bool) & valid
    union = int(np.logical_or(p, g).sum())
    if union == 0:
        return None
    return float(np.logical_and(p, g).sum()) / union


def main() -> int:
    ap = argparse.ArgumentParser(description="T11 front-end ablation")
    ap.add_argument("--run", action="append", default=[],
                    help="labelled run dir(s) with frame_*.npz")
    ap.add_argument("--limit", type=int, default=30,
                    help="frames per arm (fixed data, all arms identical)")
    ap.add_argument("--model", type=str, default=None,
                    help="checkpoint (default: the production resolver)")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    runs = args.run or [str(ROOT / "logs" / "m5_seg" / "run_20260815_010127")]
    frames = _load_frames(runs, args.limit)
    if not frames:
        print(f"[abl] no labelled frames under {runs}")
        return 2
    print(f"[abl] data={runs} frames={len(frames)} "
          f"(identical for every arm; post-process and metric fixed)")

    seg = Segmenter(model_path=args.model) if args.model else Segmenter()
    report: dict = {"frames": len(frames), "arms": {},
                    "percentiles_from_literature": {"Y": Y_PCT, "Cb": CB_PCT},
                    "tophat_kernel_px": TOPHAT_KERNEL,
                    "note": ("percentiles are the literature's values, not "
                             "calibrated on this project's data; no training "
                             "and no threshold tuning happened here")}
    road_cache: list[np.ndarray] = []
    for rgb, _label in frames:
        road_cache.append(None)
    arms_out: dict[str, dict] = {
        a: {"raw_iou": [], "final_iou": [], "deleted_true_px": [],
            "false_line_px": [], "raw_components": [], "final_components": [],
            "raw_longest_px": [], "final_longest_px": [],
            "ms_raw": [], "ms_final": []} for a in ARMS}

    for idx, (rgb, label) in enumerate(frames):
        gt = (label == 2)
        valid = (label != IGNORE_VALUE)
        # the ROAD mask for the post-process: from the model, identical for
        # every arm (the post-process is the fixed part of this experiment)
        _road = None
        try:
            _road, _line = seg.predict(rgb)
        except Exception:
            _road = None
        if _road is None:
            _road = np.zeros(rgb.shape[:2], dtype=bool)
        _road = strip_soil_from_road(np.asarray(_road, dtype=bool), rgb,
                                     route_is_dirt=getattr(seg, "route_is_dirt",
                                                           False))
        for arm in ARMS:
            t0 = time.perf_counter()
            try:
                if arm == "model":
                    rawm = arm_model(rgb, seg)
                elif arm == "abs":
                    rawm = arm_abs(rgb)
                elif arm == "tophat_otsu":
                    rawm = arm_tophat_otsu(rgb)
                else:
                    rawm = arm_ycbcr_pct(rgb)
            except Exception as exc:
                arms_out[arm].setdefault("errors", []).append(str(exc))
                continue
            t1 = time.perf_counter()
            final = _final_chain(rawm, _road, seg, rgb)
            t2 = time.perf_counter()
            slot = arms_out[arm]
            ri = _iou(rawm, gt, valid)
            fi = _iou(final, gt, valid)
            if ri is not None:
                slot["raw_iou"].append(ri)
            if fi is not None:
                slot["final_iou"].append(fi)
            # the two failure modes, counted separately and only on the
            # KNOWN area (255 is unlabelled, not background)
            lost = gt & valid & ~final
            slot["deleted_true_px"].append(int(np.count_nonzero(lost)))
            false_pos = final & valid & ~gt
            slot["false_line_px"].append(int(np.count_nonzero(false_pos)))
            slot["raw_components"].append(_components(rawm))
            slot["final_components"].append(_components(final))
            slot["raw_longest_px"].append(_longest_component_extent(rawm))
            slot["final_longest_px"].append(_longest_component_extent(final))
            slot["ms_raw"].append((t1 - t0) * 1000.0)
            slot["ms_final"].append((t2 - t1) * 1000.0)

    def _stat(vals):
        v = [x for x in vals if x is not None]
        if not v:
            return None
        return {"n": len(v), "p50": round(float(statistics.median(v)), 4),
                "mean": round(float(sum(v) / len(v)), 4),
                "max": round(float(max(v)), 4)}

    print(f"[abl] {'arm':12s} {'raw_iou':>8s} {'final_iou':>9s} "
          f"{'del_true':>9s} {'false_px':>9s} {'comps':>6s} "
          f"{'longest':>8s} {'ms_raw':>7s} {'ms_post':>8s}")
    for arm in ARMS:
        slot = arms_out[arm]
        rep = {
            "raw_iou_p50": (_stat(slot["raw_iou"]) or {}).get("p50"),
            "final_iou_p50": (_stat(slot["final_iou"]) or {}).get("p50"),
            "deleted_true_line_px_p50": (_stat(
                slot["deleted_true_px"]) or {}).get("p50"),
            "kept_false_line_px_p50": (_stat(
                slot["false_line_px"]) or {}).get("p50"),
            "raw_components_p50": (_stat(slot["raw_components"]) or {}).get(
                "p50"),
            "final_components_p50": (_stat(
                slot["final_components"]) or {}).get("p50"),
            "raw_longest_extent_px_p50": (_stat(
                slot["raw_longest_px"]) or {}).get("p50"),
            "final_longest_extent_px_p50": (_stat(
                slot["final_longest_px"]) or {}).get("p50"),
            "front_end_ms_p50": (_stat(slot["ms_raw"]) or {}).get("p50"),
            "postprocess_ms_p50": (_stat(slot["ms_final"]) or {}).get("p50"),
            "errors": slot.get("errors", [])[:3],
        }
        report["arms"][arm] = rep
        print(f"[abl] {arm:12s} {str(rep['raw_iou_p50']):>8s} "
              f"{str(rep['final_iou_p50']):>9s} "
              f"{str(rep['deleted_true_line_px_p50']):>9s} "
              f"{str(rep['kept_false_line_px_p50']):>9s} "
              f"{str(rep['final_components_p50']):>6s} "
              f"{str(rep['final_longest_extent_px_p50']):>8s} "
              f"{str(rep['front_end_ms_p50']):>7s} "
              f"{str(rep['postprocess_ms_p50']):>8s}")
    print("[abl] one variable (the front-end); data, post-process and metric "
          "are identical across arms. No arm is promoted by this run.")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2,
                                              ensure_ascii=False),
                                   encoding="utf-8")
        print(f"[abl] -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
