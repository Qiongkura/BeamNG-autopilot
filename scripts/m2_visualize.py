"""M2 visualization: overlay detected path on sample frames.

Reads the vision-only detections (CSV) and the map truth, draws both on a
grid of frames and saves a montage PNG.  Red dots = vision-detected band
midpoints (the autonomous signal), green dots = map-projected centreline
(reference only, not used by the detector).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beamng_autopilot.track import load_track
from beamng_autopilot.vision.projection import default_camera


_ROOT = Path(__file__).resolve().parent.parent
RUN = _ROOT / "logs" / "m2_capture" / "20260811_172839"
DISTANCES = (4.0, 6.0, 8.0, 10.0, 12.0, 16.0)
FRAMES = (0, 50, 88, 100, 124, 190)


def load_csv(name):
    rows = {}
    for r in csv.DictReader(open(RUN / name, encoding="utf-8")):
        rows[int(r["idx"])] = {
            k: (float(r[k]) if r[k] not in ("", "nan") else np.nan) for k in r
        }
    return rows


def main() -> None:
    meta = json.load(open(RUN / "meta.json", encoding="utf-8"))
    track, _ = load_track(_ROOT / "data" / "track_smallgrid.npz")
    n = len(track)
    vis = load_csv("vision_only_features.csv")
    probe = cv2.imread(str(RUN / "frame_00000.jpg"))
    h, w = probe.shape[:2]
    cam = default_camera(w, h)
    fx = cam.fx

    def nearest_idx(p):
        return int(np.argmin(np.linalg.norm(track[:, :2] - np.asarray(p)[:2], axis=1)))

    def ahead_point(k, dist):
        i = k
        traveled = 0.0
        while traveled < dist:
            j = (i + 1) % n
            seg = track[j] - track[i]
            L = float(np.linalg.norm(seg))
            if traveled + L > dist:
                return track[i] + ((dist - traveled) / max(L, 1e-6)) * seg
            traveled += L
            i = j
        return track[i]

    tiles = []
    for idx in FRAMES:
        img = cv2.imread(str(RUN / f"frame_{idx:05d}.jpg"))
        m = meta[idx]
        pos = np.array(m["pos"])
        hd = float(m["heading"])
        k = nearest_idx(pos)
        rec = vis.get(idx, {})
        vpts = []
        tpts = []
        for d in DISTANCES:
            ld = rec.get(f"lat{d:.0f}", np.nan)
            if not np.isnan(ld):
                u = w / 2.0 + ld * fx / d
                _, v, _ = cam.project(np.asarray([[d, 0.0, 0.0]]), np.zeros(3), 0.0)
                vpts.append((int(round(u)), int(round(float(v[0])))))
            ap = ahead_point(k, d)
            u, v, ok = cam.project([ap], pos, hd)
            u = float(u[0])
            v = float(v[0])
            if bool(ok[0]) and 0 <= u < w and 0 <= v < h:
                tpts.append((int(round(u)), int(round(v))))
        for i in range(len(vpts) - 1):
            cv2.line(img, vpts[i], vpts[i + 1], (0, 255, 255), 2)
        for (u, v) in tpts:
            cv2.circle(img, (u, v), 5, (0, 255, 0), -1)
            cv2.circle(img, (u, v), 5, (0, 0, 0), 1)
        for (u, v) in vpts:
            cv2.circle(img, (u, v), 7, (0, 0, 255), -1)
            cv2.circle(img, (u, v), 7, (255, 255, 255), 1)
        cv2.putText(img, f"#{idx} steer={m['steer']:+.2f} v={m['speed']:.1f} m/s",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img, "red=vision  green=truth", (12, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        tiles.append(img)

    cols = 3
    rows_n = int(np.ceil(len(tiles) / cols))
    th, tw = tiles[0].shape[:2]
    canvas = np.zeros((rows_n * th, cols * tw, 3), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        canvas[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = t
    out = RUN / "vision_only" / "montage.png"
    cv2.imwrite(str(out), canvas)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
