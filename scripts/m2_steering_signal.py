"""M2 offline evaluation: turn camera frames into a vision steering signal.

The captured frames (logs/m2_capture/<run>) contain the vehicle pose and the
projected track centreline is known, so we can measure how well a purely
visual detector tracks the road.  On this map the drivable band shows up as a
dark rubber/wear strip roughly on the centreline, so the detector looks for
the darkest narrow band around the projected centreline at several distances
ahead and reports its lateral offset.

Signals tested here:
  * lat(d):  detected band lateral offset (m, right-positive) at distance d
  * steer_ang: atan2(lat(12), 12)  - direction to the vision path point
  * steer_drift: lat(13) - lat(4)  - near/far band drift
Ground truth:
  * signed_turn: wrap(atan2(track[+12m] - pos) - heading)
  * meta steer:  the pure-pursuit steering command that was applied
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beamng_autopilot.track import load_track
from beamng_autopilot.vision.band import detect_pair_center
from beamng_autopilot.vision.projection import default_camera


DISTANCES = (4.0, 6.0, 8.0, 10.0, 12.0, 13.0, 16.0)

_ROOT = Path(__file__).resolve().parent.parent
_M2_RUN = str(_ROOT / "logs" / "m2_capture" / "20260811_172839")


def wrap_angle(a):
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def nearest_idx(pts, p):
    return int(np.argmin(np.linalg.norm(pts[:, :2] - np.asarray(p)[:2], axis=1)))


def ahead_point(track, k, dist):
    n = len(track)
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=_M2_RUN)
    ap.add_argument("--track", default=str(_ROOT / "data" / "track_smallgrid.npz"))
    ap.add_argument("--csv", default=_M2_RUN + r"\vision_features.csv")
    args = ap.parse_args()

    run_dir = Path(args.run)
    meta = json.load(open(run_dir / "meta.json", encoding="utf-8"))
    track, _ = load_track(args.track)
    n = len(track)

    rows = []
    for idx, m in enumerate(meta):
        img = cv2.imread(str(run_dir / f"frame_{idx:05d}.jpg"))
        if img is None:
            continue
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cam = default_camera(w, h)
        fx = cam.fx
        pos = np.array(m["pos"])
        hd = float(m["heading"])
        k = nearest_idx(track, pos)

        ap_gt = ahead_point(track, k, 12.0)
        signed_turn = wrap_angle(np.arctan2(ap_gt[1] - pos[1], ap_gt[0] - pos[0]) - hd)

        # true centreline position in the car frame (right-positive metres)
        rec = {
            "idx": idx,
            "t": m["t"],
            "heading": hd,
            "speed": m["speed"],
            "steer": m["steer"],
            "signed_turn": signed_turn,
            "nearest": m["nearest"],
        }
        for d in DISTANCES:
            rec[f"nb{d:.0f}"] = 0
            rec[f"pw{d:.0f}"] = np.nan
        confs = []
        for d in DISTANCES:
            ap = ahead_point(track, k, d)
            u, v, ok = cam.project([ap], pos, hd)
            u, v, ok = float(u[0]), float(v[0]), bool(ok[0])
            if not ok or not (0 <= u < w) or not (0 <= v < h):
                rec[f"lat{d:.0f}"] = np.nan
                rec[f"truth{d:.0f}"] = np.nan
                rec[f"cf{d:.0f}"] = 0.0
                continue
            expected_px = 1.4 * fx / max(d, 1e-3)
            off, pair_w, conf, n_bands = detect_pair_center(gray[int(v)], u, expected_px=expected_px)
            rec[f"nb{d:.0f}"] = n_bands
            rec[f"pw{d:.0f}"] = pair_w
            # absolute band position in the car frame (right-positive)
            rec[f"lat{d:.0f}"] = np.nan if off is None else (u + off - w / 2.0) * d / fx
            rec[f"truth{d:.0f}"] = (u - w / 2.0) * d / fx
            rec[f"cf{d:.0f}"] = conf
            confs.append(conf)
        rec["conf_mean"] = float(np.mean(confs)) if confs else 0.0
        rows.append(rec)

    # ---- aggregate signals ----
    lat = {d: np.array([r[f"lat{d:.0f}"] for r in rows], dtype=float) for d in DISTANCES}
    truth = {d: np.array([r[f"truth{d:.0f}"] for r in rows], dtype=float) for d in DISTANCES}
    steer_gt = np.array([r["steer"] for r in rows], dtype=float)
    turn_gt = np.array([r["signed_turn"] for r in rows], dtype=float)

    def corr(a, b):
        mask = ~(np.isnan(a) | np.isnan(b))
        if mask.sum() < 10:
            return float("nan"), 0
        c = float(np.corrcoef(a[mask], b[mask])[0, 1])
        return c, int(mask.sum())

    print(f"frames: {len(rows)}  (track n={n})")
    print("-- detector fidelity: vision band position vs true centreline (car frame) --")
    for d in DISTANCES:
        c_fid, cnt = corr(lat[d], truth[d])
        err = np.nanmean(np.abs(lat[d] - truth[d]))
        bias = np.nanmean(lat[d] - truth[d])
        n2 = sum(1 for r in rows if r[f"nb{d:.0f}"] == 2)
        print(f"  d={d:>2.0f}m  fidelity r={c_fid:+.3f} (n={cnt:3d})  mean|err|={err:.3f} m  bias={bias:+.3f} m  pair2={n2}/{len(rows)}")

    print("-- vision steering signals vs ground truth --")
    for d in (10.0, 12.0, 13.0, 16.0):
        steer_ang = np.arctan2(np.where(np.isnan(lat[d]), 0.0, lat[d]), d)
        c_turn, cnt = corr(steer_ang, turn_gt)
        c_steer, _ = corr(steer_ang, steer_gt)
        print(f"  steer_ang(lat{d:>2.0f}) vs signed_turn: r={c_turn:+.3f} (n={cnt:3d})  vs steer: r={c_steer:+.3f}")

    drift = lat[12.0] - lat[4.0]
    c_turn, cnt = corr(drift, turn_gt)
    c_steer, _ = corr(drift, steer_gt)
    print(f"  drift(12-4)      vs signed_turn: r={c_turn:+.3f} (n={cnt:3d})  vs steer: r={c_steer:+.3f}")

    # decompose: how much of the drift signal comes from the map prior vs the
    # pure visual band deviation from the projected centreline?
    off = {d: lat[d] - truth[d] for d in DISTANCES}
    prior_drift = truth[12.0] - truth[4.0]
    vis_drift = off[12.0] - off[4.0]
    c_turn, cnt = corr(prior_drift, turn_gt)
    print(f"  PRIOR drift(truth12-truth4)  vs signed_turn: r={c_turn:+.3f} (n={cnt:3d})")
    c_turn, cnt = corr(vis_drift, turn_gt)
    c_steer, _ = corr(vis_drift, steer_gt)
    print(f"  VISION drift(off12-off4)     vs signed_turn: r={c_turn:+.3f} (n={cnt:3d})  vs steer: r={c_steer:+.3f}")

    for idx in (0, 30, 80, 100, 130, 190, 200):
        if idx < len(rows):
            r = rows[idx]
            lats = " ".join(
                f"{d:.0f}:{r[f'lat{d:.0f}']:+.2f}" if not np.isnan(r[f"lat{d:.0f}"]) else f"{d:.0f}:-- "
                for d in DISTANCES
            )
            truths = " ".join(
                f"{d:.0f}:{r[f'truth{d:.0f}']:+.2f}" if not np.isnan(r[f"truth{d:.0f}"]) else f"{d:.0f}:-- "
                for d in DISTANCES
            )
            print(f"  frame {idx:3d} turn={r['signed_turn']:+.2f} steer={r['steer']:+.2f}")
            print(f"      vis  : {lats}")
            print(f"      truth: {truths}")

    with open(args.csv, "w", newline="") as fh:
        import csv

        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {args.csv}")


if __name__ == "__main__":
    main()
