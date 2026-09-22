"""Per-stage line metrics, so no number is quoted without its stage (P0-3).

The round-5 report mixed numbers from different stages of the same
pipeline - network raw argmax, after morphology, after the road
constraint, after shape filtering, and the full ``Segmenter.predict()`` -
and they differ by more than 2x.  That made "the post-processing scan
improved IoU to 0.358" look like a claim about the deployed model, when
the deployed pipeline scored 0.2186 on the same frames.

This script removes the ambiguity: every stage is computed with the SAME
functions the driving pipeline calls (``Segmenter._morph_close_line``,
``vision.segmentation.constrain_line_to_road``, ``filter_line_shape``),
and the final stage is cross-checked against ``Segmenter.predict()`` so a
mismatch fails loudly instead of being reported as a number.

Reported per stage: global line IoU, mean-frame line IoU, line pixel
count, component count - plus the dataset, frame count, class mapping and
ignore value, and the command that produced the JSON.

Usage::

    .venv\\Scripts\\python.exe scripts\\m5_seg_stage_eval.py \\
        --runs logs\\m5_seg\\manual_mountain_labeled logs\\m5_seg\\manual_review_batch_labeled \\
        --model logs\\m5_seg\\seg_model_hand\\best_task.pt \\
        --json logs\\goal_20260921\\seg_stages.json

    # the four constraint arms of handoff P1-4, same RGB / same checkpoint
    .venv\\Scripts\\python.exe scripts\\m5_seg_stage_eval.py --runs ... --model ... --arms
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from beamng_autopilot.vision.segmentation import (
    Segmenter,
    constrain_line_to_road,
    filter_line_shape,
    strip_soil_from_road,
)

# Training contract (m5_train_seg.py): 0 background, 1 road, 2 line, 255 ignore.
CLASS_MAPPING = {0: "background", 1: "road", 2: "line"}
IGNORE_VALUE = 255
STAGES = ("raw_argmax", "after_morph_close", "after_road_constraint",
          "after_shape_filter", "full_predict")


def _iou(pred, gt, valid=None) -> float | None:
    """Line IoU, restricted to pixels that carry a real label.

    The training contract marks unlabelled pixels 255 (``IGNORE_VALUE``).
    Scoring them as background punishes a correct prediction on an unknown
    pixel and rewards a model that predicts nothing there, so the valid
    area is applied before the intersection and the union (plan T01).
    """
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    if valid is not None:
        v = np.asarray(valid, dtype=bool)
        pred = pred & v
        gt = gt & v
    union = int(np.logical_or(pred, gt).sum())
    if union == 0:
        return None                     # nothing predicted, nothing true
    return float(np.logical_and(pred, gt).sum()) / union


def _components(mask) -> int:
    import cv2
    n, _labels, _stats, _ = cv2.connectedComponentsWithStats(
        np.asarray(mask, dtype=np.uint8), 8)
    return int(n - 1)


def legacy_pixel_road_constraint(line, road, ksize: int = 7):
    """The PRE-fix production constraint, reimplemented exactly as at HEAD.

    ``git show HEAD:beamng_autopilot/vision/segmentation.py`` shows it as
    ``fill_interior_holes(road) -> dilate(ksize) -> line & road`` with no
    per-component test and no minimal-road-mask guard.  The four-arm report
    needs THIS function as its "old" arm: a bare ``line & road`` is a
    different algorithm and its numbers cannot be used to describe the old
    pipeline (plan T01).
    """
    import cv2
    from beamng_autopilot.vision.segmentation import fill_interior_holes

    m = np.asarray(line, dtype=bool)
    if not m.any():
        return m
    rd = fill_interior_holes(np.asarray(road, dtype=bool)).astype(np.uint8)
    rd = cv2.dilate(rd, cv2.getStructuringElement(
        cv2.MORPH_RECT, (int(ksize), int(ksize)))).astype(bool)
    return m & rd


def load_frames(run_dirs):
    """``[(colour, label)]`` from ``frame_*.npz`` (training contract)."""
    frames = []
    per_run: dict[str, int] = {}
    for rd in run_dirs:
        d = Path(rd)
        fs = sorted(glob.glob(str(d / "frame_*.npz")))
        if not fs:
            print(f"[stages] no frames in {d}", flush=True)
            continue
        per_run[str(d)] = len(fs)
        for f in fs:
            z = np.load(f)
            if "colour" not in z.files or "label" not in z.files:
                continue
            frames.append((np.asarray(z["colour"], dtype=np.uint8),
                           np.asarray(z["label"], dtype=np.uint8)))
    return frames, per_run


def stage_report(frames, segmenter, constrain_mode: str) -> dict:
    """Per-stage line metrics over ``frames`` for one constrain variant.

    Also counts the line pixels that fall OUTSIDE the (dilated) road mask:
    that is the quantity the road constraint exists to remove, and it must
    be reported separately - a constraint that also deletes true paint can
    look identical on a "false pixels removed" column alone.
    """
    sums = {s: {"iou": 0.0, "n": 0, "px": 0, "comp": 0} for s in STAGES}
    off_road = {s: 0 for s in STAGES}
    raw_px = raw_comp = 0
    print_mismatch = 0
    for colour, label in frames:
        gt_line = (label == 2)
        valid_area = (np.asarray(label) != IGNORE_VALUE)
        road, line = segmenter._argmax_masks(
            segmenter._infer_logits(colour), colour)
        road = strip_soil_from_road(road, colour,
                                    route_is_dirt=segmenter.route_is_dirt)
        stages = {}
        stages["raw_argmax"] = np.asarray(line, dtype=bool)
        m = segmenter._morph_close_line(stages["raw_argmax"])
        stages["after_morph_close"] = m
        if constrain_mode == "none":
            c = m
        elif constrain_mode == "pixel":
            # The LEGACY production constraint, reimplemented exactly as it
            # was at HEAD: fill the road mask's interior holes, dilate by
            # ksize, then intersect PIXEL-WISE.  The previous version of
            # this arm did a bare ``m & road``, which is not the same
            # function, so the "the old constraint deleted almost all line
            # pixels" reading was not an equivalent reproduction (plan T01).
            c = legacy_pixel_road_constraint(m, road)
        elif constrain_mode == "component":
            # No elongated-stroke second tier: the explicit "no fallback"
            # variant.  (``elongated_frac=0.0`` did NOT mean that - it let
            # every component through the second tier.)
            c = constrain_line_to_road(m, road, elongated_frac=None)
        else:                            # "tiered" = production default
            c = constrain_line_to_road(m, road)
        stages["after_road_constraint"] = c
        stages["after_shape_filter"] = filter_line_shape(c)
        prod_road, prod_line = segmenter.predict(colour)
        stages["full_predict"] = np.asarray(prod_line, dtype=bool)
        if not np.array_equal(stages["full_predict"],
                              stages["after_shape_filter"]):
            print_mismatch += 1
        rd = np.asarray(road, dtype=bool)
        if rd.any():
            import cv2 as _cv2
            rd = _cv2.dilate(
                _cv2.morphologyEx(rd.astype(np.uint8), _cv2.MORPH_CLOSE,
                                  _cv2.getStructuringElement(
                                      _cv2.MORPH_RECT, (15, 15))),
                _cv2.getStructuringElement(_cv2.MORPH_RECT, (7, 7))).astype(bool)
        else:
            rd = np.ones_like(gt_line)
        for name, mask in stages.items():
            v = _iou(mask, gt_line, valid_area)
            if v is not None:
                sums[name]["iou"] += v
                sums[name]["n"] += 1
            sums[name]["px"] += int(mask.sum())
            sums[name]["comp"] += _components(mask)
            off_road[name] += int(np.count_nonzero(mask & ~rd))
        raw_px += int(gt_line.sum())
        raw_comp += _components(gt_line)
    frames_n = max(1, len(frames))
    out = {
        "constrain_mode": constrain_mode,
        "frames": len(frames),
        "gt_line_px_total": int(raw_px),
        "gt_line_components_total": int(raw_comp),
        "stages": {},
        "production_mismatch_frames": print_mismatch,
    }
    for name, acc in sums.items():
        out["stages"][name] = {
            "mean_frame_line_iou": (round(acc["iou"] / acc["n"], 4)
                                    if acc["n"] else None),
            "frames_scored": acc["n"],
            "line_px_total": acc["px"],
            "line_px_mean": int(round(acc["px"] / frames_n)),
            "components_total": acc["comp"],
            "off_road_line_px_total": int(off_road[name]),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run dirs with frame_*.npz (colour + label)")
    ap.add_argument("--model", required=True, help="segmentation checkpoint")
    ap.add_argument("--arms", action="store_true",
                    help="evaluate the four constraint arms (P1-4 offline)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    frames, per_run = load_frames(args.runs)
    if not frames:
        print("[stages] no usable frames")
        return 1
    seg = Segmenter(model_path=args.model)
    cmd = ("python scripts/m5_seg_stage_eval.py --runs "
           + " ".join(args.runs) + f" --model {args.model}"
           + (" --arms" if args.arms else ""))
    report = {
        "command": cmd,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": str(args.model),
        "keep_checkpoint_best_note": (
            "Segmenter loads the given file; 'best.pt' vs "
            "'checkpoint_last.pt' are different checkpoints and must be "
            "named explicitly in any report"),
        "dataset": {"runs": per_run, "frames": len(frames)},
        "contract": {"class_mapping": CLASS_MAPPING,
                     "ignore_value": IGNORE_VALUE,
                     "metric": "line IoU vs label==2, mean over frames; "
                               "frames with no line in either mask are "
                               "excluded from the mean (recorded as "
                               "frames_scored)"},
    }
    modes = (["none", "pixel", "component", "tiered"] if args.arms
             else ["tiered"])
    report["variants"] = {}
    for mode in modes:
        report["variants"][mode] = stage_report(frames, seg, mode)
        r = report["variants"][mode]
        print(f"--- constrain={mode}: frames={r['frames']} "
              f"gt_line_px={r['gt_line_px_total']} "
              f"production_mismatch={r['production_mismatch_frames']} ---")
        for name in STAGES:
            s = r["stages"][name]
            print(f"  {name:20s} IoU={s['mean_frame_line_iou']} "
                  f"px_mean={s['line_px_mean']:6d} "
                  f"comps={s['components_total']:5d} "
                  f"off_road_px={s['off_road_line_px_total']:6d} "
                  f"scored={s['frames_scored']}")
    prod = report["variants"].get("tiered")
    if prod is not None and prod["production_mismatch_frames"]:
        print("[stages] !! stage 5 (full Segmenter.predict) differs from "
              "stage 4: the stage definitions have drifted from the "
              "production pipeline - fix before quoting any number")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(
            json.dumps(report, ensure_ascii=False, indent=1),
            encoding="utf-8")
        print(f"wrote {args.json}")
    # A drifted stage definition invalidates every number in the report,
    # so it must not exit 0 (plan T01: "生产一致性失败必须非零退出").
    if prod is not None and prod["production_mismatch_frames"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
