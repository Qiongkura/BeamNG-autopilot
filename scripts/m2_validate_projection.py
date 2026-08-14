"""Offline validation of the calibrated camera model: project the recorded
track centreline ahead of the vehicle into the captured frames and print an
ASCII overlay so we can confirm the projection lands on the road."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beamng_autopilot.track import load_track
from beamng_autopilot.vision.projection import default_camera


def nearest_idx(pts, p):
    return int(np.argmin(np.linalg.norm(pts[:, :2] - p[:2], axis=1)))


def main() -> None:
    _root = Path(__file__).resolve().parent.parent
    run_dir = _root / "logs" / "m2_capture" / "20260811_172839"
    meta = json.load(open(run_dir / "meta.json", encoding="utf-8"))
    track, _ = load_track(_root / "data" / "track_smallgrid.npz")
    n = len(track)

    for idx in [0, 60, 100, 150, 200]:
        img = cv2.imread(str(run_dir / f"frame_{idx:05d}.jpg"))
        if img is None:
            continue
        h, w = img.shape[:2]
        cam = default_camera(w, h)
        m = meta[idx]
        pos = np.array(m["pos"]); hd = m["heading"]
        k = nearest_idx(track, pos)
        # sample track points ahead of the car, up to 60 m
        fwd = np.array([np.cos(hd), np.sin(hd), 0.0])
        ahead = []
        i = k
        traveled = 0.0
        while traveled < 60.0:
            j = (i + 1) % n
            seg = track[j] - track[i]
            L = float(np.linalg.norm(seg))
            t = 1.0
            if traveled + L > 60.0:
                t = (60.0 - traveled) / max(L, 1e-6)
            ahead.append(track[i] + t * seg)
            traveled += L
            i = j
        ahead = np.array(ahead)
        # only keep points that are in front of the car
        rel = ahead - pos
        in_front = rel @ fwd > 0.0
        ahead = ahead[in_front]
        u, v, valid = cam.project(ahead, pos, hd)
        pts = np.column_stack([u, v])[valid]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        small_h, small_w = 36, 96
        small = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_AREA)
        chars = " .:-=+*#%@"
        grid = np.array([[min(9, int(val / 26)) for val in row] for row in small])
        # overlay projected track centreline
        sx = pts[:, 0] * (small_w / w)
        sy = pts[:, 1] * (small_h / h)
        for xi, yi in zip(sx, sy):
            if 0 <= yi < small_h and 0 <= xi < small_w:
                grid[int(yi), int(xi)] = 10  # marker 'X'
        print(f"=== frame {idx} pos=({m['pos'][0]:.0f},{m['pos'][1]:.0f}) "
              f"h={hd:.2f} nearest={k} ===")
        glyphs = " .:-=+*#%@X"
        for row in grid:
            print("".join(glyphs[v] for v in row))


if __name__ == "__main__":
    main()
