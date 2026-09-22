"""Pixel->metre truth measurement for the crossing claims (stage C).

Turns the visual FP rate of ``docs/STAGE_C_TRUTH.md`` into a RECOMPUTABLE
one.  For every tick of a recorded drive it measures, in the CAR frame and
in metres:

* the lateral position of the paint's INNER edge, from the camera image
  (a documented brightness rule over a row band whose ground distance is
  known from the T05 geometry), and
* the lateral position of the vehicle's LEFT FRONT CORNER, computed from
  the recorded pose and the production footprint,

and reports the signed metric margin between them:

    margin = paint_inner_lat - corner_lat      (+ = left)

``margin <= 0`` means the body reaches the paint's inner edge -> the frame
IS a crossing by the same corner criterion the perception flag uses.
``margin > 0`` means the body is clear.  Comparing that verdict with the
recorded flag gives a false-positive / false-negative count over the frames
where the measurement is valid, instead of a visual impression.

Honest limits, printed with every run:

* the paint rule is a brightness heuristic (paint is brighter than asphalt
  on these frames); its coverage per frame is reported so a reader can see
  when it failed rather than trusting a number it could not produce;
* the pose is the recorded one, so a bad pose mis-places the corner;
* the measurement is per FRAME and independent of the perception's own
  lane geometry (it uses the image pixels and the pose only).

Usage::

    .venv\\Scripts\\python.exe scripts/m5_truth_measure.py \\
        --shadow logs/m5_e2e/shadow_fsd_....npz --telemetry tr2_B.json \\
        --json logs/goal_20260921/truth_measure.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beamng_autopilot import geometry as G  # noqa: E402
from beamng_autopilot.vehicle_body import (  # noqa: E402
    HALF_LENGTH_M,
    HALF_WIDTH_M,
    footprint_corners,
)
from beamng_autopilot.vision.lanes import _back_project_many  # noqa: E402
from beamng_autopilot.vision.ring import (  # noqa: E402
    FRONT_MAIN,
    camera_ring_models,
)

#: Row band (metres ahead of the ego) where the ROAD PAINT is visible.
#:
#: It must start BEYOND the ego's own bonnet: the front camera looks over
#: the hood, which fills the bottom of the frame, and rows that project to
#: 2-4 m on the ground plane land on the bonnet in the image.  The first
#: version of this tool used 2-6 m and its "paint detections" were the
#: bonnet's bright edge (measured 2026-09-22 by rendering the detection
#: overlay: the hits formed a line along the hood, not along the paint).
#: The T05 blind-zone measurement says the nearest VISIBLE ground is
#: ~4.3 m, so the band starts past that.
BAND_NEAR_M = 5.0
BAND_FAR_M = 9.0
#: Paint rule: a pixel is paint when it is brighter than its row's p80 by
#: this much AND above an absolute floor (asphalt in these frames sits well
#: below it, painted lines well above).
PAINT_ROW_DELTA = 25.0
PAINT_ABS_MIN = 120.0
#: The front camera looks over the hood, and the bonnet carries a BRIGHT
#: stripe; rows below this fraction of the image height are bonnet, never
#: road (measured on this camera: the dark band starts at ~0.66 of the
#: height, and the previous version's detections there were the stripe).
BONNET_TOP_FRAC = 0.66
#: The corner criterion needs a tolerance: the measurement itself scatters
#: row to row, so only a margin beyond this counts as "clear" or "overlap".
MARGIN_TOL_M = 0.10


def debug_frame(img, pos, heading, *, side: str = "left"):
    """Render the frame with the paint rule's detections marked.

    Without this the rule is unverifiable: on three consecutive runs it
    reported "every frame clear", "14 crossings" and "5 crossings", which
    cannot all describe the same road - the detector, not the car, was
    changing.  The overlay shows the row band and every accepted pixel so
    the rule can be judged by eye.
    """
    import cv2
    cam = camera_ring_models(int(img.shape[1]), int(img.shape[0]))[FRONT_MAIN]
    gz = G.ego_ground_z(pos)
    vis = cv2.cvtColor(np.asarray(img, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    h, w = vis.shape[:2]
    grey = cv2.cvtColor(np.asarray(img, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    hits = []
    for v in range(int(h * 0.45), min(h, int(h * BONNET_TOP_FRAC))):
        pts, ok = _back_project_many(np.array([float(w) / 2.0]),
                                     np.array([float(v)]), cam, pos, heading,
                                     gz)
        if not bool(ok[0]):
            continue
        d = float(np.hypot(pts[0][0] - float(pos[0]),
                           pts[0][1] - float(pos[1])))
        if not (BAND_NEAR_M <= d <= BAND_FAR_M):
            continue
        row = grey[v]
        thr = max(PAINT_ABS_MIN,
                  float(np.percentile(row, 80)) + PAINT_ROW_DELTA)
        cols = np.nonzero(row >= thr)[0]
        cols = cols[cols < w // 2] if side == "left" else cols[cols >= w // 2]
        if len(cols):
            c = int(cols.max() if side == "left" else cols.min())
            hits.append((v, c, float(thr)))
            cv2.circle(vis, (c, v), 2, (0, 0, 255), -1)
        cv2.line(vis, (0, v), (w - 1, v), (60, 60, 60), 1)
    cv2.putText(vis, f"thr_rule p80+{PAINT_ROW_DELTA:.0f}/{PAINT_ABS_MIN:.0f} "
                     f"hits={len(hits)}", (4, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    return vis, hits


def _paint_inner_lat(img, cam, pos, heading, ground_z, *,
                     side: str = "left") -> dict:
    """Lateral offset of the paint's inner edge, from the IMAGE pixels.

    ``side="left"`` searches the left half of the image (the case the
    crossing flag is about).  Returns ``p50/p10/p90`` of the per-row
    lateral offsets plus how many rows actually yielded a detection - a
    rule that fires on one row out of forty is not a measurement.
    """
    import cv2
    h, w = img.shape[:2]
    grey = cv2.cvtColor(np.asarray(img, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    # candidate rows: those whose ground distance lands inside the band
    rows = []
    for v in range(int(h * 0.45), min(h, int(h * BONNET_TOP_FRAC))):
        pts, ok = _back_project_many(np.array([float(w) / 2.0]),
                                     np.array([float(v)]), cam, pos, heading,
                                     ground_z)
        if not bool(ok[0]):
            continue
        d = float(np.hypot(pts[0][0] - float(pos[0]),
                           pts[0][1] - float(pos[1])))
        if BAND_NEAR_M <= d <= BAND_FAR_M:
            rows.append(v)
    lats = []
    for v in rows:
        row = grey[v]
        thr = max(PAINT_ABS_MIN, float(np.percentile(row, 80)) + PAINT_ROW_DELTA)
        cols = np.nonzero(row >= thr)[0]
        if side == "left":
            cols = cols[cols < w // 2]
            if not len(cols):
                continue
            col = float(cols.max())          # inner edge = nearest the car
        else:
            cols = cols[cols >= w // 2]
            if not len(cols):
                continue
            col = float(cols.min())
        pts, ok = _back_project_many(np.array([col]), np.array([float(v)]),
                                     cam, pos, heading, ground_z)
        if not bool(ok[0]):
            continue
        rel = np.array([float(pts[0][0]) - float(pos[0]),
                        float(pts[0][1]) - float(pos[1])])
        fwd = np.array([math.cos(float(heading)), math.sin(float(heading))])
        left = np.array([-fwd[1], fwd[0]])
        lats.append(float(rel @ left))
    if not lats:
        return {"ok": False, "rows": len(rows), "detected": 0}
    a = np.asarray(lats)
    return {"ok": True, "rows": len(rows), "detected": int(len(a)),
            "p50": float(np.median(a)), "p10": float(np.percentile(a, 10)),
            "p90": float(np.percentile(a, 90)),
            "sign_agree": float(np.mean(np.sign(a) == np.sign(np.median(a))))}


def measure_frame(img, pos, heading, *,
                  side: str = "left") -> dict:
    """One frame's metric margin between the paint and the body corner."""
    cam = camera_ring_models(int(img.shape[1]), int(img.shape[0]))[FRONT_MAIN]
    gz = G.ego_ground_z(pos)
    paint = _paint_inner_lat(img, cam, pos, heading, gz, side=side)
    # the body's left-front corner, from the recorded pose
    corners = footprint_corners(np.asarray(pos, dtype=float)[:2],
                               float(heading))
    fwd = np.array([math.cos(float(heading)), math.sin(float(heading))])
    left = np.array([-fwd[1], fwd[0]])
    rel = np.asarray(corners, dtype=float) - np.asarray(pos, dtype=float)[:2]
    lat = rel @ left
    fwd_d = rel @ fwd
    if side == "left":
        # the corner furthest to the left (the flag's worst corner)
        j = int(np.argmax(lat))
    else:
        j = int(np.argmin(lat))
    corner_lat = float(lat[j])
    out = {"paint": paint, "corner_lat_m": round(corner_lat, 3),
           "corner_fwd_m": round(float(fwd_d[j]), 2),
           "half_width_m": float(HALF_WIDTH_M),
           "half_length_m": float(HALF_LENGTH_M), "side": side}
    if not paint.get("ok"):
        out["verdict"] = "UNMEASURED"
        out["reason"] = "no paint detected in the band"
        return out
    margin = float(paint["p50"]) - corner_lat
    out["margin_m"] = round(margin, 3)
    out["margin_p10_m"] = round(float(paint["p10"]) - corner_lat, 3)
    if margin <= -float(MARGIN_TOL_M):
        out["verdict"] = "cross"       # the body reaches past the inner edge
    elif margin >= float(MARGIN_TOL_M):
        out["verdict"] = "clear"
    else:
        out["verdict"] = "ambiguous"   # inside the measurement scatter
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="pixel->metre truth measurement")
    ap.add_argument("--shadow", required=True, help="shadow_fsd_*.npz")
    ap.add_argument("--telemetry", default=None,
                    help="the run's telemetry JSON (for the perception flags)")
    ap.add_argument("--json", default=None)
    ap.add_argument("--align", choices=("index", "xy"), default="index",
                    help="how to pair episode frames with telemetry rows: "
                         "index (equal lengths, newest runs) or xy (nearest "
                         "position, for older runs whose counts differ)")
    ap.add_argument("--align-tol-m", type=float, default=1.0,
                    help="max position distance for --align xy")
    ap.add_argument("--debug-frame", type=int, default=None,
                    help="also render this frame's detections to --debug-out")
    ap.add_argument("--debug-out", type=str, default=None)
    ap.add_argument("--max-frames", type=int, default=0,
                    help="0 = every frame of the episode")
    args = ap.parse_args()

    z = np.load(args.shadow, allow_pickle=True)
    t = np.asarray(z["t"], dtype=float)
    xs = np.asarray(z["x"], dtype=float)
    ys = np.asarray(z["y"], dtype=float)
    hd = np.asarray(z["heading"], dtype=float)
    rgb = np.asarray(z["rgb"])
    n = len(t)
    if args.max_frames:
        n = min(n, int(args.max_frames))
    flags = {}
    if args.telemetry:
        hist = json.loads(Path(args.telemetry).read_text(encoding="utf-8"))
        # INDEX mapping, not time mapping.  The shadow episode's ``t`` is a
        # different clock (measured 2026-09-22: episode t starts at 11.36 s
        # while the telemetry's starts at 0.71 s), so joining on time
        # silently paired each frame with the WRONG tick and produced a
        # 54% "false negative" rate that was an artefact.  Both files
        # record one row per tick, so equal lengths are the contract.
        if args.align == "index":
            if len(hist) != len(t):
                print(f"[truth] !! telemetry rows ({len(hist)}) != episode "
                      f"frames ({len(t)}): the index mapping is not exact - "
                      f"refusing to attribute flags to frames (use "
                      f"--align xy for older runs)")
                return 2
            for i in range(n):
                flags[i] = hist[i]
        else:
            # position-based alignment: the car's (x, y) is a physical key
            # that does not drift the way a clock or a row counter can.
            # Uniqueness is required, or the pairing is a guess.
            hx = np.array([float(r.get("pos", [np.nan, np.nan])[0])
                           for r in hist])
            hy = np.array([float(r.get("pos", [np.nan, np.nan])[1])
                           for r in hist])
            paired = 0
            for i in range(n):
                d = np.hypot(hx - xs[i], hy - ys[i])
                if not np.isfinite(d).any():
                    continue
                j = int(np.nanargmin(d))
                order = np.sort(d[np.isfinite(d)])
                if order.size > 1 and order[1] < 1.5 * order[0]:
                    continue          # ambiguous: two rows equally close
                if order[0] > float(args.align_tol_m):
                    continue          # too far to be the same instant
                flags[i] = hist[j]
                paired += 1
            print(f"[truth] --align xy paired {paired}/{n} episode frames "
                  f"(tol {args.align_tol_m} m, uniqueness 1.5x)")

    if args.debug_frame is not None:
        i = int(args.debug_frame)
        img = rgb[i]
        pos = np.array([xs[i], ys[i], 0.0])
        vis, hits = debug_frame(img, pos, float(hd[i]))
        out_png = args.debug_out or "logs/goal_20260921/truth_debug.png"
        import cv2
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(out_png, vis)
        print(f"[truth] debug frame {i}: hits={len(hits)} "
              f"cols={[c for _v, c, _t in hits][:12]} -> {out_png}")
        return 0

    rows = []
    for i in range(n):
        pos = np.array([xs[i], ys[i], 0.0])
        # the episode stores x/y/heading only; the ground plane comes from
        # the vehicle origin, which the drive's telemetry carries.  Where it
        # is missing the measurement uses the episode's own z-less pose and
        # SAYS SO in the record.
        m = measure_frame(rgb[i], pos, float(hd[i]))
        f = flags.get(i) or {}
        m.update({"i": i, "t": round(float(t[i]), 3),
                  "perception_paired": int(i in flags),
                  "perception_cross": int(bool(f.get("body_cross_l")
                                               or f.get("body_cross_r"))),
                  "perception_off": (None if f.get("road_off") is None
                                     else float(f.get("road_off"))),
                  "lat_left": f.get("lat_left"), "line_lat": f.get("line_lat"),
                  "boundary_published": int(f.get("lat_left") is not None
                                            or f.get("lat_right") is not None)})
        rows.append(m)

    tally = {"cross": 0, "clear": 0, "ambiguous": 0, "UNMEASURED": 0}
    fp = fn = tp = tn = 0
    unpaired = 0
    for m in rows:
        tally[m["verdict"]] = tally.get(m["verdict"], 0) + 1
        # A frame with no telemetry row has NO CLAIM to compare against:
        # treating the missing claim as "not claimed" would fabricate a
        # false negative.  Measured 2026-09-22: a mismatched pair left 160
        # of 167 frames unpaired and reported 6 phantom FNs.
        if not m.get("perception_paired"):
            unpaired += 1
            continue
        claimed = bool(m["perception_cross"]) and bool(m["boundary_published"])
        if m["verdict"] == "UNMEASURED" or m["verdict"] == "ambiguous":
            continue
        truth_cross = (m["verdict"] == "cross")
        if claimed and not truth_cross:
            fp += 1
        elif truth_cross and not claimed:
            fn += 1
        elif claimed and truth_cross:
            tp += 1
        else:
            tn += 1
    denom_fp = fp + tn + tp
    denom_fn = fn + tp + tn
    print(f"[truth] shadow={Path(args.shadow).name} frames={len(rows)} "
          f"verdicts={tally} unpaired={unpaired}")
    print(f"[truth] measured frames: cross={tp + fn} clear={fp + tn} | "
          f"TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"[truth] false-positive rate (claimed crossing, measured clear) = "
          f"{fp}/{denom_fp} = "
          f"{(fp / denom_fp if denom_fp else float('nan')):.3f}")
    print(f"[truth] false-negative rate (measured crossing, not claimed) = "
          f"{fn}/{denom_fn} = "
          f"{(fn / denom_fn if denom_fn else float('nan')):.3f}")
    print("[truth] paint rule = row p80+25 and >=120 grey, rows 2-6 m ahead; "
          "per-frame coverage is in the JSON")
    print("[truth] UNMEASURED/ambiguous frames AND unpaired frames are "
          "excluded from the rates and counted - never treated as clean")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(
            {"shadow": args.shadow, "telemetry": args.telemetry,
             "verdicts": tally, "unpaired": unpaired,
             "tp": tp, "fp": fp, "fn": fn, "tn": tn,
             "fp_rate": (fp / denom_fp if denom_fp else None),
             "fn_rate": (fn / denom_fn if denom_fn else None),
             "paint_rule": {"row_delta": PAINT_ROW_DELTA,
                            "abs_min": PAINT_ABS_MIN,
                            "band_m": [BAND_NEAR_M, BAND_FAR_M]},
             "margin_tol_m": MARGIN_TOL_M, "frames": rows},
            indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"[truth] -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
