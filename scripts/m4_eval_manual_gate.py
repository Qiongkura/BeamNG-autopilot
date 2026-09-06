"""Acceptance gate: v11 line recall on the USER'S manual annotations.

The 10 hand-annotated town frames (scripts/m5_annotate_manual.py output)
are the acceptance set for a candidate segmentation model.  The gate:
line recall on the user's strokes >= the threshold (default 0.80) - a
model that still misses the left boundary line or the far segments of
the markings fails, exactly the failure the user photographed.

    .venv\Scripts\python.exe scripts\m4_eval_manual_gate.py \
        --model logs/m5_seg/seg_model_v11/best.pt
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from beamng_autopilot import config
from beamng_autopilot.vision.segmentation import Segmenter


def main() -> int:
    ap = argparse.ArgumentParser(description="manual-annotation gate")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--frames", type=str, default=None,
                    help="annotated frames dir (default: newest manual_*)")
    ap.add_argument("--threshold", type=float, default=0.80,
                    help="required line recall on the user's strokes")
    ap.add_argument("--min-px", type=int, default=30)
    args = ap.parse_args()

    base = (Path(args.frames) if args.frames
            else Path(max(glob.glob(str(config.LOGS_DIR / "m5_seg"
                                     / "manual_*")),
                          key=os.path.getmtime)))
    files = sorted(glob.glob(str(base / "frame_*.npz")))
    if not files:
        print(f"[gate] no annotated frames in {base}")
        return 2
    print(f"[gate] model={args.model}")
    print(f"[gate] frames={base} ({len(files)})")

    seg = Segmenter(model_path=args.model)
    tot_u = tot_hit = 0
    per_frame = []
    for f in files:
        d = np.load(f)
        rgb, user = d["colour"], d["label"]
        user_line = user == 2
        if user_line.sum() < args.min_px:
            continue
        _, model_line = seg.predict(rgb)
        m = np.asarray(model_line) > 0
        if m.shape != user_line.shape:
            m = cv2.resize(m.astype(np.uint8),
                           (user_line.shape[1], user_line.shape[0]),
                           interpolation=cv2.INTER_NEAREST) > 0
        # tolerance-based recall: the model emits a THIN mask while the
        # hand strokes are 4-6 px wide - raw pixel IoU would cap recall
        # at ~1/3 for a PERFECT detection.  Dilate the model mask by
        # 3 px so coverage within +-3 px of the paint counts as hit.
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        m = cv2.dilate(m.astype(np.uint8), kern, iterations=1) > 0
        hit = int((user_line & m).sum())
        u = int(user_line.sum())
        tot_hit += hit
        tot_u += u
        rec = hit / max(1, u)
        per_frame.append((os.path.basename(f), u, hit, rec))
        print(f"  {os.path.basename(f)}: user={u:5d} hit={hit:5d} "
              f"recall={rec:.0%}")
    recall = tot_hit / max(1, tot_u)
    passed = recall >= args.threshold
    print(f"[gate] TOTAL line recall = {recall:.1%} "
          f"(threshold {args.threshold:.0%}) -> "
          f"{'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
