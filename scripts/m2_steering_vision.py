"""M2 vision-only steering evaluation (no map prior).

The map-seeded run (m2_steering_signal.py) proves the rubber-band detector
tracks the true centreline, but it searches around the projected track point.
Here we remove the map entirely:
  * seed mode "center" - search a wide window around the image centre;
  * seed mode "track"  - follow the previous frame's detection (temporal
                         tracking), re-initialising from the image centre
                         when the lock is lost for a few frames.
The camera is only used to pick the ground row for each look-ahead distance
(fixed geometry, no track).  Outputs per-frame lateral offsets, correlation
statistics vs the applied steering command, an overlay montage and a video.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beamng_autopilot.vision.band import detect_pair_center
from beamng_autopilot.vision.projection import default_camera


DISTANCES = (4.0, 6.0, 8.0, 10.0, 12.0, 16.0)
TIRE_SPACING_M = 1.4

_ROOT = Path(__file__).resolve().parent.parent
_M2_RUN = str(_ROOT / "logs" / "m2_capture" / "20260811_172839")


def wrap_angle(a):
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def corr(a, b):
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 10:
        return float("nan"), 0
    return float(np.corrcoef(a[mask], b[mask])[0, 1]), int(mask.sum())


def median_smooth(x, k=3):
    x = np.asarray(x, dtype=float)
    out = np.copy(x)
    half = k // 2
    for i in range(len(x)):
        lo = max(0, i - half)
        hi = min(len(x), i + half + 1)
        vals = x[lo:hi]
        vals = vals[~np.isnan(vals)]
        if len(vals):
            out[i] = float(np.median(vals))
        else:
            out[i] = np.nan
    return out


def detect_at_distance(gray, cam, d, center, window_px):
    """Detect the band midpoint at distance d on the ground row `v`.

    Returns (offset_px, pair_w_px, conf, n_bands) or (None, ...) on failure.
    The ground row comes from camera geometry only (no map).
    """
    h, w = gray.shape[:2]
    fy = cam.fy
    # exact ground row: project the point d metres straight ahead of the
    # vehicle with the calibrated fixed camera pose (vehicle frame forward
    # is +x at heading 0; heading/pos only move the view, they do not change
    # where a straight-ahead ground point lands in the image)
    u, v, valid = cam.project(np.asarray([[d, 0.0, 0.0]]), np.zeros(3), 0.0)
    if not bool(valid[0]):
        return None, 0.0, 0.0, 0
    v = float(v[0])
    v = int(round(v))
    if not (2 <= v < h - 2):
        return None, 0.0, 0.0, 0
    expected_px = TIRE_SPACING_M * fy / max(d, 1e-3)
    # average a few adjacent rows for stability
    row = gray[max(0, v - 2):v + 3].mean(axis=0)
    off, pair_w, conf, n_bands = detect_pair_center(
        row, center, expected_px=expected_px, window=int(window_px)
    )
    return off, pair_w, conf, n_bands


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=_M2_RUN)
    ap.add_argument("--csv", default=_M2_RUN + r"\vision_only_features.csv")
    ap.add_argument("--video", action="store_true", default=True)
    args = ap.parse_args()

    run_dir = Path(args.run)
    meta = json.load(open(run_dir / "meta.json", encoding="utf-8"))
    out_dir = run_dir / "vision_only"
    out_dir.mkdir(exist_ok=True)

    probe = cv2.imread(str(run_dir / "frame_00000.jpg"))
    if probe is None:
        raise SystemExit(f"cannot read first frame in {run_dir}")
    h, w = probe.shape[:2]
    cam = default_camera(w, h)
    fx = cam.fx

    rows = []
    # temporal state per distance: last detected pixel centre
    last_center = {d: w / 2.0 for d in DISTANCES}
    lost = {d: 0 for d in DISTANCES}

    for idx, m in enumerate(meta):
        img = cv2.imread(str(run_dir / f"frame_{idx:05d}.jpg"))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rec = {
            "idx": idx,
            "t": m["t"],
            "speed": m["speed"],
            "steer": m["steer"],
            "heading": m["heading"],
        }
        for d in DISTANCES:
            rec[f"lat{d:.0f}"] = np.nan
            rec[f"nb{d:.0f}"] = 0
        for d in DISTANCES:
            expected_px = TIRE_SPACING_M * fx / max(d, 1e-3)
            center_win = max(180.0, 1.5 * expected_px)
            track_win = max(90.0, 0.6 * expected_px)
            seed = w / 2.0
            win = center_win
            if lost[d] <= 6:
                seed = float(np.clip(last_center[d], center_win, w - center_win))
                win = track_win
            off, _pw, conf, n_bands = detect_at_distance(gray, cam, d, seed, win)
            if off is None:
                # retry from the image centre with the wide window
                off, _pw, conf, n_bands = detect_at_distance(gray, cam, d, w / 2.0, center_win)
            if off is None or not (-w < off < w):
                lost[d] += 1
                continue
            u_det = seed + off
            rec[f"lat{d:.0f}"] = (u_det - w / 2.0) * d / fx
            rec[f"nb{d:.0f}"] = n_bands
            last_center[d] = u_det
            lost[d] = 0
        rows.append(rec)

    lat = {d: np.array([r[f"lat{d:.0f}"] for r in rows], dtype=float) for d in DISTANCES}
    steer = np.array([r["steer"] for r in rows], dtype=float)

    print(f"frames: {len(rows)}")
    for mode_name, mode_lat in (("raw", lat), ("med3", {d: median_smooth(lat[d]) for d in DISTANCES})):
        print(f"-- {mode_name} --")
        for d in DISTANCES:
            n_ok = int(np.sum(~np.isnan(mode_lat[d])))
            print(f"  d={d:>2.0f}m  detect={n_ok}/{len(rows)}")
        for d in (4.0, 12.0, 16.0):
            steer_ang = np.arctan2(np.where(np.isnan(mode_lat[d]), 0.0, mode_lat[d]), d)
            c, cnt = corr(steer_ang, steer)
            print(f"  steer_ang(lat{d:>2.0f}) vs steer: r={c:+.3f} (n={cnt})")
        drift = mode_lat[12.0] - mode_lat[4.0]
        c, cnt = corr(drift, steer)
        print(f"  drift(12-4)            vs steer: r={c:+.3f} (n={cnt})")

    # ---- overlays ----
    if args.video:
        video_path = out_dir / "vision_track.mp4"
        vw = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (w, h)
        )
    for idx, rec in enumerate(rows):
        img = cv2.imread(str(run_dir / f"frame_{rec['idx']:05d}.jpg"))
        if img is None:
            continue
        hh, ww = img.shape[:2]
        pts = []
        for d in DISTANCES:
            ld = rec[f"lat{d:.0f}"]
            if np.isnan(ld):
                continue
            u = ww / 2.0 + ld * fx / d
            _, v, _ = cam.project(np.asarray([[d, 0.0, 0.0]]), np.zeros(3), 0.0)
            v = float(v[0])
            pts.append((int(round(u)), int(round(v))))
        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i + 1], (0, 255, 255), 2)
        for (u, v) in pts:
            cv2.circle(img, (u, v), 7, (0, 0, 255), -1)
            cv2.circle(img, (u, v), 7, (255, 255, 255), 1)
        cv2.putText(img, f"#{rec['idx']} steer={rec['steer']:+.2f}", (16, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        if args.video:
            vw.write(img)
    if args.video:
        vw.release()
        print(f"saved {video_path}")

    with open(args.csv, "w", newline="") as fh:
        import csv
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {args.csv}")


if __name__ == "__main__":
    main()
