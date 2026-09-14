"""Inject HSV yellow paint into existing seg run line labels (pseudo).

For each frame npz with colour + label: line = label==2 | yellow_line_mask(colour).
Writes a new run directory with the same frame files.

Usage:
    .venv\\Scripts\\python.exe scripts/m5_inject_yellow_labels.py \\
        --src logs/m5_seg/run_xxx --out logs/m5_seg/run_xxx_yellow
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beamng_autopilot.vision.yellow_line_mask import yellow_line_mask


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    added = 0
    for fp in sorted(src.glob("frame_*.npz")):
        with np.load(fp, allow_pickle=True) as z:
            colour = np.asarray(z["colour"], dtype=np.uint8)
            label = np.asarray(z["label"]).copy()
        y = yellow_line_mask(colour)
        if y.shape == label.shape[:2]:
            before = int(np.count_nonzero(label == 2))
            label[y] = 2
            after = int(np.count_nonzero(label == 2))
            added += after - before
        np.savez_compressed(
            out / fp.name, colour=colour, label=label.astype(np.uint8))
        n += 1
    meta = src / "meta.json"
    if meta.is_file():
        shutil.copy2(meta, out / "meta.json")
    print(f"injected {n} frames, +{added} yellow line px -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
