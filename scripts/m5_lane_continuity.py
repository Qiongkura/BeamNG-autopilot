"""Offline measurement of perception lane continuity - no game needed.

The remaining blocker in the strict FSD runs is not control or safety: the
car gets a *paired* perception lane in only a small fraction of frames, so
the stack spends the rest of the run in its fail-closed degradation and
creeps.  ``lane_sel=perception-unavailable`` for 193 of 217 frames in the
2026-09-07 town run.

This replays recorded shadow episodes through the SAME perception path
the live stack uses, so that number can be reproduced and iterated on
without launching the game:

    front camera model -> SemanticHead (learned line mask + classic-CV
    fallback) -> world-space LaneMarkings -> pair_lane_markings

For every frame it records whether an own-lane frame could be paired, and
splits the misses into the two causes that need different fixes:

* ``model_miss`` - the recorded ground-truth label HAS near-field line
  pixels but the model produced no markings at all (segmentation recall);
* ``pair_miss``  - the model did produce markings but they could not be
  paired into an own-lane frame (pairing / gate logic).

It also runs the same ticks through ``lane/reference.select_lane_reference``
in strict perception mode, which is what publishes the live
``lane_sel`` - so the offline verdict is directly comparable to the
telemetry field.

Usage:

    .venv\\Scripts\\python.exe scripts\\m5_lane_continuity.py --frames 40
    .venv\\Scripts\\python.exe scripts\\m5_lane_continuity.py --episodes 3 --all-frames
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from beamng_autopilot import config
from beamng_autopilot.fsd_realism import SRC_SENSOR, SRC_UNAVAILABLE
from beamng_autopilot.lane import pair_lane_markings, select_lane_reference
from beamng_autopilot.vision.heads.semantic import SemanticHead
from beamng_autopilot.vision.hydra import FrameContext
from beamng_autopilot.vision.lanes import _mask_to_markings
from beamng_autopilot.vision.ring import CAMERA_RING, FRONT_MAIN

# Semantic GT class for painted lines (recording.py writes 0=bg 1=road 2=line).
LABEL_LINE = 2
# A frame counts as "line present" only when the GT has enough near-field
# paint to be pairable at all - a couple of far pixels is below the
# pairing contract's own near-field requirement, so it must not be
# counted as a model miss.
GT_NEAR_MIN_PX = 60
# Lower part of the image = near field (the preview the own lane lives in).
GT_NEAR_ROW_FRAC = 0.55
# Raw marking side read: forward window used to decide which side a
# marking sits on (matches the pairing module's near-field band).
SIDE_READ_AHEAD_M = 6.0
# The pairing module classifies a candidate as left/right only outside
# this lateral band (``_cand_side``).
SIDE_BAND_M = 0.08
# Near-field paint pixels on one image half before that side counts as
# "painted" (a real line contributes hundreds; speckle does not).
SIDE_PAINT_MIN_PX = 40
# Fraction of a dropped component's / rejected marking's own pixels that
# must sit on annotated paint before it counts as a REAL lane line rather
# than noise (judged against the recorded GT label).
GT_REAL_LINE_FRAC = 0.5


def _episodes(data_dir: Path, pattern: str) -> list[str]:
    return sorted(glob.glob(str(data_dir / pattern)), key=os.path.getmtime)


def _near_gt_line_px(label: np.ndarray) -> int:
    if label is None:
        return 0
    h = label.shape[0]
    near = np.asarray(label, dtype=np.uint8)[int(h * GT_NEAR_ROW_FRAC):]
    return int((near == LABEL_LINE).sum())


# Distance bands for the GT audit, as fractions of image height measured
# from the top: the lane paint lives in the lower half and the far/mid/near
# split is what tells "cannot see far" apart from "cannot see at all".
GT_BANDS = (("far", 0.45, 0.62), ("mid", 0.62, 0.78), ("near", 0.78, 1.0))


def _audit_frame(mask, label, cx: int, acc: dict) -> None:
    """Add one frame's per-side, per-band GT paint vs model-mask counts.

    ``acc`` is keyed ``"<L|R>:<band>"`` -> ``[gt_px, pred_px, hit_px]``.
    A low recall means the model cannot see paint that is provably there
    (a capability gap); ``gt_px == 0`` means there was nothing to see
    (a scene / data-coverage fact, not a model failure).
    """
    if mask is None or label is None:
        return
    m = np.asarray(mask, dtype=bool)
    g = np.asarray(label) == LABEL_LINE
    if m.shape != g.shape:
        return
    h, w = m.shape
    for name, lo, hi in GT_BANDS:
        r0, r1 = int(h * lo), int(h * hi)
        for side, c0, c1 in (("L", 0, cx), ("R", cx, w)):
            gm = g[r0:r1, c0:c1]
            pm = m[r0:r1, c0:c1]
            slot = acc.setdefault(f"{side}:{name}", [0, 0, 0])
            slot[0] += int(gm.sum())
            slot[1] += int(pm.sum())
            slot[2] += int(np.logical_and(gm, pm).sum())


def _marking_sides(markings, pos, heading: float) -> list[float]:
    """Near-field lateral offset of every detected marking (left = +).

    This is the raw detection read, BEFORE ``_collect_candidates`` drops
    anything, so it can tell "the other side was never seen" apart from
    "it was seen and then filtered out".
    """
    fwd = np.array([np.cos(float(heading)), np.sin(float(heading))])
    left = np.array([-fwd[1], fwd[0]])
    p = np.asarray(pos[:2], dtype=float)
    out: list[float] = []
    for mk in markings:
        w = np.asarray(getattr(mk, "world", None), dtype=float)
        if w.ndim != 2 or w.shape[1] < 2 or len(w) < 2:
            continue
        rel = w[:, :2] - p
        lon = rel @ fwd
        lat = rel @ left
        near = lat[(lon >= 0.0) & (lon <= SIDE_READ_AHEAD_M)]
        out.append(float(np.median(near)) if len(near) >= 2
                   else float(np.median(lat)))
    return out


def _side_counts(sides: list[float]) -> tuple[int, int]:
    """(left, right) counts using the pairing module's own side band."""
    return (sum(1 for v in sides if v > SIDE_BAND_M),
            sum(1 for v in sides if v < -SIDE_BAND_M))


def _near_paint_px_by_half(mask, cx: int) -> tuple[int, int]:
    """Near-field paint pixels left / right of the principal point."""
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2 or not m.any():
        return 0, 0
    near = m[int(m.shape[0] * GT_NEAR_ROW_FRAC):]
    return int(near[:, :cx].sum()), int(near[:, cx:].sum())


def _refine_missing_side(markings, pos, heading: float, line_mask,
                         label, cx: int) -> str:
    """Split "one side not detected" into where the paint was lost.

    The image half of a lane side is what the forward camera sees, so a
    near-field paint count per half says whether the MISSING side was
    never painted (a dashed gap / an unmarked edge), was painted but the
    model's line mask missed it, or was in the mask and the marking
    extractor dropped it.  Those three need completely different fixes.
    """
    sides = _marking_sides(markings, pos, heading)
    left_n, right_n = _side_counts(sides)
    if left_n == 0 and right_n > 0:
        half, name = 0, "left"
    elif right_n == 0 and left_n > 0:
        half, name = 1, "right"
    else:
        return "one_side_not_detected"
    if line_mask is not None:
        ml, mr = _near_paint_px_by_half(line_mask, cx)
        if (ml if half == 0 else mr) >= SIDE_PAINT_MIN_PX:
            return f"missing_{name}__mask_has_paint_marking_dropped"
    if label is not None:
        gl, gr = _near_paint_px_by_half(
            np.asarray(label) == LABEL_LINE, cx)
        if (gl if half == 0 else gr) >= SIDE_PAINT_MIN_PX:
            return f"missing_{name}__paint_visible_model_missed"
        return f"missing_{name}__no_paint_on_that_side"
    return f"missing_{name}__unknown"


def _single_cause(markings, pos, heading: float, dbg: dict) -> str:
    """Why a frame that DID detect paint produced no paired lane frame.

    The candidates are read with the same near-field side rule the pairing
    uses (``debug['cands'][*]['near_med']``), so a bend whose far arc
    swings into the car frame is not misclassified.
    """
    mode = str(dbg.get("mode", "") or "")
    cands = dbg.get("cands") or []
    if mode == "none":
        return "all_markings_filtered"          # collector dropped every one
    if not mode and not cands:
        return "all_markings_filtered"

    def _side(c: dict) -> float:
        v = c.get("near_med")
        return float(c.get("med_lat", 0.0)) if v is None else float(v)

    cl, cr = _side_counts([_side(c) for c in cands])
    rl, rr = _side_counts(_marking_sides(markings, pos, heading))
    if rl == 0 or rr == 0:
        return "one_side_not_detected"
    if cl == 0 or cr == 0:
        return "one_side_filtered_by_candidate_gate"
    if not mode:
        return "pair_rejected_mirror_refused"
    return "both_sides_but_no_pair"


def measure_episode(ep: str, *, frames: int, sem: SemanticHead,
                    verbose: bool, diagnose: bool = False,
                    gt_audit: bool = False) -> dict:
    sem.reset()
    d = np.load(ep, allow_pickle=True)
    meta = json.loads(bytes(d["meta"]).decode("utf-8")) if "meta" in d.files \
        else {}
    xs, ys, hds = d["x"], d["y"], d["heading"]
    rgbs = d["rgb"]
    labels = d["label"] if "label" in d.files else None
    rec_src = d["lane_src"] if "lane_src" in d.files else None
    t = d["t"]

    h, w = np.asarray(rgbs[0]).shape[:2]
    cam_w = int(meta.get("cam_w", w) or w)
    cam_h = int(meta.get("cam_h", h) or h)
    mount = next(m for m in CAMERA_RING if m.role == FRONT_MAIN)
    cam = mount.camera_model(cam_w, cam_h)

    n = len(xs)
    idxs = (np.linspace(0, n - 1, frames).astype(int) if frames and frames < n
            else np.arange(n))

    counts = {
        "frames": 0,
        "paired": 0,
        "pair_failed": 0,     # markings produced, pair_lane_markings -> None
        "single_edge": 0,     # a frame exists but only one side was real
        "no_markings": 0,
        "model_miss": 0,      # GT has near paint, no markings produced
        "gt_no_paint": 0,     # nothing to detect at all
        "strict_sensor": 0,
        "strict_unavailable": 0,
    }
    rec_hist: dict[str, int] = {}
    off_hist: dict[str, int] = {}
    cause_hist: dict[str, int] = {}
    drop_hist: dict[str, int] = {}
    kind_hist: dict[str, int] = {}
    drop_stats: list[tuple] = []
    gate_hist: dict[str, int] = {}
    gate_stats: list[tuple] = []
    gt_acc: dict[str, list[int]] = {}
    n_marks: list[int] = []
    lat_paired: list[float] = []
    flags: list[int] = []
    ts: list[float] = []

    for i in idxs:
        pos = np.array([float(xs[i]), float(ys[i]), 0.0])
        heading = float(hds[i])
        rgb = np.asarray(rgbs[i], dtype=np.uint8)
        ctx = FrameContext(frame_rgb=rgb, cam=cam, pos=pos,
                           heading=heading, ground_z=0.0, role="front_main",
                           timestamp=float(t[i]))
        out = sem.run(ctx)
        markings = list(out.meta.get("markings") or [])
        n_marks.append(len(markings))

        dbg: dict = {}
        _gt = ((labels[i] == LABEL_LINE) if labels is not None else None)
        if _gt is not None:
            dbg["gt"] = _gt
        frame = None
        if markings:
            frame = pair_lane_markings(markings, pos, heading, debug=dbg)
        paired = bool(frame is not None and getattr(frame, "paired", False))
        flags.append(1 if paired else 0)
        ts.append(float(t[i]))
        if gt_audit:
            _audit_frame(out.masks.get("line"),
                         labels[i] if labels is not None else None,
                         int(cam.cx), gt_acc)

        gt_px = _near_gt_line_px(
            labels[i] if labels is not None else None)
        counts["frames"] += 1
        if paired:
            counts["paired"] += 1
            try:
                p = np.asarray(pos[:2], dtype=float)
                fwd = np.array([np.cos(heading), np.sin(heading)])
                left = np.array([-fwd[1], fwd[0]])
                c = np.asarray(frame.center, dtype=float)[:, :2]
                near = c[np.linalg.norm(c - p, axis=1) < 12.0]
                if len(near):
                    lat_paired.append(float(((near - p) @ left).mean()))
            except Exception:
                pass
        elif markings:
            if frame is None:
                counts["pair_failed"] += 1
            else:
                counts["single_edge"] += 1
        else:
            counts["no_markings"] += 1
        if diagnose and markings and not paired:
            cause = _single_cause(markings, pos, heading, dbg)
            if cause == "one_side_not_detected":
                cause = _refine_missing_side(
                    markings, pos, heading,
                    out.masks.get("line"),
                    labels[i] if labels is not None else None,
                    int(cam.cx))
            cause_hist[cause] = cause_hist.get(cause, 0) + 1
            if cause == "one_side_filtered_by_candidate_gate":
                # A marking WAS produced and the candidate gate dropped it:
                # record which gate and how far the reject was from passing.
                for k, v in (dbg.get("collect_drops") or {}).items():
                    gate_hist[k] = gate_hist.get(k, 0) + int(v)
                for row in (dbg.get("collect_stats") or []):
                    gate_stats.append(row)
            if cause.endswith("mask_has_paint_marking_dropped"):
                # Ask the extractor itself why the paint it can see did not
                # become a marking.
                md: dict = {}
                if _gt is not None:
                    md["gt"] = _gt
                _mask_to_markings(
                    np.asarray(out.masks.get("line"), dtype=np.uint8) * 255,
                    "white", cam, pos, heading, ground_z=0.0, debug=md)
                for k, v in (md.get("drops") or {}).items():
                    drop_hist[k] = drop_hist.get(k, 0) + int(v)
                for k, v in (md.get("kinds") or {}).items():
                    kind_hist[k] = kind_hist.get(k, 0) + int(v)
                for row in (md.get("drop_stats") or []):
                    drop_stats.append(row)
        if not markings and gt_px >= GT_NEAR_MIN_PX:
            counts["model_miss"] += 1
        if gt_px < GT_NEAR_MIN_PX:
            counts["gt_no_paint"] += 1

        # The strict-mode verdict is what the live telemetry calls lane_sel.
        ref = select_lane_reference(
            lane_frame=frame, pos=pos, heading=heading,
            route_ref=None, has_nav_route=False,
            lane_mode="sensor", strict_sensor=True,
        )
        if ref.src == SRC_SENSOR:
            counts["strict_sensor"] += 1
        else:
            counts["strict_unavailable"] += 1
        off_hist[ref.src] = off_hist.get(ref.src, 0) + 1
        if rec_src is not None:
            k = str(rec_src[i])
            rec_hist[k] = rec_hist.get(k, 0) + 1

        if verbose and counts["frames"] % 20 == 0:
            print(f"    frame {counts['frames']:4d}/{len(idxs)} "
                  f"t={float(t[i]):6.1f}s marks={len(markings):2d} "
                  f"paired={int(paired)} gt_px={gt_px:4d}")

    fr = max(1, counts["frames"])
    res = {
        "episode": os.path.basename(ep),
        "cam": [cam_w, cam_h],
        "frames": counts["frames"],
        "paired": counts["paired"],
        "paired_rate": round(counts["paired"] / fr, 4),
        "strict_sensor_rate": round(counts["strict_sensor"] / fr, 4),
        "pair_failed": counts["pair_failed"],
        "single_edge": counts["single_edge"],
        "no_markings": counts["no_markings"],
        "model_miss": counts["model_miss"],
        "gt_no_paint": counts["gt_no_paint"],
        "mean_markings": round(float(np.mean(n_marks)), 2) if n_marks else 0.0,
        "offline_src_hist": off_hist,
        "recorded_lane_src_hist": rec_hist,
    }
    if cause_hist:
        res["single_causes"] = cause_hist
    if drop_hist:
        res["extractor_drops"] = drop_hist
    if kind_hist:
        res["extractor_kinds"] = kind_hist
    if gate_hist:
        res["gate_drops"] = gate_hist
    if gate_stats:
        by_gate: dict[str, list] = {}
        for row in gate_stats:
            key, span, align, side = row[0], row[1], row[2], row[3]
            n_px = int(row[4]) if len(row) > 4 else 0
            gt_px = int(row[5]) if len(row) > 5 else 0
            by_gate.setdefault(key, []).append((span, align, side,
                                                n_px, gt_px))

        def _p50(rows, i):
            vals = [r[i] for r in rows if r[i] == r[i]]
            return round(float(np.median(vals)), 2) if vals else None

        res["gate_by_reason"] = {}
        for k, v in by_gate.items():
            n_px = sum(r[3] for r in v)
            gt_px = sum(r[4] for r in v)
            real = sum(1 for r in v
                       if r[3] and r[4] / r[3] >= GT_REAL_LINE_FRAC)
            res["gate_by_reason"][k] = {
                "n": len(v), "span_p50": _p50(v, 0),
                "align_p50": _p50(v, 1), "side_p50": _p50(v, 2),
                "gt_px_frac": round(gt_px / n_px, 3) if n_px else None,
                "n_real_line": real}
    if drop_stats:
        arr = np.asarray([(r[2], r[3], r[4]) for r in drop_stats],
                         dtype=float)
        res["drop_stats_n"] = int(len(arr))
        res["drop_wh_p50"] = [round(float(np.median(arr[:, 0])), 1),
                              round(float(np.median(arr[:, 1])), 1)]
        res["drop_wh_p90"] = [round(float(np.percentile(arr[:, 0], 90)), 1),
                              round(float(np.percentile(arr[:, 1], 90)), 1)]
        by_reason: dict[str, list[tuple[float, float, float, int, int]]] = {}
        for row in drop_stats:
            # key by "<side>:<reason>" so it lines up with extractor_drops
            by_reason.setdefault(f"{row[0]}:{row[1]}", []).append(
                (float(row[2]), float(row[3]), float(row[4]),
                 int(row[5]), int(row[6])))
        res["drop_by_reason"] = {}
        for k, v in by_reason.items():
            n_px = sum(r[3] for r in v)
            gt_px = sum(r[4] for r in v)
            real = sum(1 for r in v
                       if r[3] and r[4] / r[3] >= GT_REAL_LINE_FRAC)
            res["drop_by_reason"][k] = {
                "n": len(v),
                "w_p50": round(float(np.median([r[0] for r in v])), 1),
                "h_p50": round(float(np.median([r[1] for r in v])), 1),
                "area_p50": round(float(np.median([r[2] for r in v])), 1),
                "area_p90": round(float(np.percentile(
                    [r[2] for r in v], 90)), 1),
                "gt_px_frac": round(gt_px / n_px, 3) if n_px else None,
                "n_real_line": real}
    if lat_paired:
        res["lat_paired_mean_m"] = round(float(np.mean(lat_paired)), 3)
    if gt_acc:
        res["gt_audit"] = {
            k: {"gt_px": v[0], "model_px": v[1], "hit_px": v[2],
                "recall": round(v[2] / v[0], 3) if v[0] else None}
            for k, v in gt_acc.items()}
    # Dropout run lengths: a bounded temporal hold can only bridge SHORT
    # gaps, so how long the unpaired stretches last decides whether the
    # fix is temporal continuity or better per-frame detection.
    runs: list[tuple[int, float]] = []
    i = 0
    while i < len(flags):
        if flags[i] == 0:
            j = i
            while j < len(flags) and flags[j] == 0:
                j += 1
            dur = ts[min(j, len(ts) - 1)] - ts[i]
            runs.append((j - i, round(float(dur), 2)))
            i = j
        else:
            i += 1
    if runs:
        res["dropout_runs"] = len(runs)
        res["dropout_frames"] = sum(r[0] for r in runs)
        res["dropout_s_total"] = round(sum(r[1] for r in runs), 1)
        for bound in (0.5, 1.0, 2.0, 5.0):
            covered = sum(r[1] for r in runs if r[1] <= bound)
            res[f"dropout_s_within_{bound}s"] = round(covered, 1)
        res["dropout_longest_s"] = max(r[1] for r in runs)
        res["dropout_run_len_hist"] = {
            str(k): sum(1 for r in runs if r[0] == k)
            for k in sorted({r[0] for r in runs})[:12]}
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="perception lane continuity (offline)")
    ap.add_argument("--data", type=str, default=None,
                    help="episode directory (default logs/m5_e2e)")
    ap.add_argument("--pattern", type=str, default="shadow_fsd_*.npz")
    ap.add_argument("--episodes", type=int, default=1,
                    help="how many NEWEST episodes to measure")
    ap.add_argument("--episode-names", type=str, default=None,
                    help="comma-separated pinned episode file names "
                         "(overrides --episodes; shadow sets grow on every "
                         "live drive, so cross-time comparisons must pin)")
    ap.add_argument("--frames", type=int, default=40,
                    help="frames sampled per episode (0 = all)")
    ap.add_argument("--all-frames", action="store_true",
                    help="sample every frame of the episode")
    ap.add_argument("--seg-model", type=str, default=None,
                    help="segmentation checkpoint (default: deployed best.pt)")
    ap.add_argument("--out", type=str, default=None,
                    help="write the report JSON here")
    ap.add_argument("--diagnose", action="store_true",
                    help="classify WHY each unpaired frame missed its pair")
    ap.add_argument("--gt-audit", action="store_true",
                    help="per side/band GT paint vs model line mask")
    args = ap.parse_args()

    data_dir = Path(args.data) if args.data else config.LOGS_DIR / "m5_e2e"
    eps = _episodes(data_dir, args.pattern)
    if not eps:
        print(f"no episodes matching {args.pattern} in {data_dir}")
        return 1
    if args.episode_names:
        eps = [str(data_dir / n.strip())
               for n in args.episode_names.split(",") if n.strip()]
    else:
        eps = eps[-max(1, args.episodes):]

    from beamng_autopilot.vision.segmentation import Segmenter
    try:
        segmenter = Segmenter(model_path=args.seg_model)
    except FileNotFoundError as exc:
        print(f"segmentation model unavailable: {exc}")
        return 2
    sem = SemanticHead(segmenter=segmenter)
    print(f"seg model: {args.seg_model or 'deployed best.pt'} "
          f"(device={segmenter.device})")

    frames = 0 if args.all_frames else args.frames
    reports = []
    for ep in eps:
        r = measure_episode(ep, frames=frames, sem=sem,
                            verbose=os.environ.get("LANE_CONT_VERBOSE") == "1",
                            diagnose=args.diagnose,
                            gt_audit=args.gt_audit)
        reports.append(r)
        print(f"\n=== {r['episode']} (cam {r['cam'][0]}x{r['cam'][1]}) ===")
        print(f"  frames measured      : {r['frames']}")
        print(f"  PAIRED own-lane frame: {r['paired']:4d}  "
              f"({r['paired_rate']:.1%})")
        print(f"  strict lane_sel=sensor: {r['strict_sensor_rate']:.1%}  "
              f"(rest perception-unavailable)")
        print(f"  markings, pair FAILED: {r['pair_failed']}")
        print(f"  markings, single edge: {r['single_edge']}")
        print(f"  no markings at all   : {r['no_markings']}")
        print(f"    of which model miss: {r['model_miss']} "
              f"(GT near-field paint present)")
        print(f"  frames with no GT paint: {r['gt_no_paint']}")
        print(f"  mean markings/frame  : {r['mean_markings']}")
        if "lat_paired_mean_m" in r:
            print(f"  own-lane centre lat  : {r['lat_paired_mean_m']:+.3f} m")
        print(f"  offline src hist     : {r['offline_src_hist']}")
        print(f"  recorded lane_src    : {r['recorded_lane_src_hist']}")
        if r.get("single_causes"):
            print("  why unpaired frames missed a pair:")
            for k, v in sorted(r["single_causes"].items(),
                               key=lambda kv: -kv[1]):
                print(f"      {v:4d}  {k}")
        if r.get("extractor_drops"):
            print("  extractor losses (L/R:reason, gt = share of the "
                  "dropped pixels that are annotated paint):")
            for k, v in sorted(r["extractor_drops"].items(),
                               key=lambda kv: -kv[1])[:8]:
                extra = (r.get("drop_by_reason") or {}).get(k, {})
                gf = extra.get("gt_px_frac")
                print(f"      {v:4d}  {k:20s} gt="
                      f"{'-' if gf is None else f'{gf:.0%}':>5} "
                      f"real_line={extra.get('n_real_line')}")
        if r.get("extractor_kinds"):
            print(f"  extractor kept kinds : {r['extractor_kinds']}")
        if r.get("gate_drops"):
            print("  candidate gate dropped these markings "
                  "(gt = share of the marking's own pixels on annotated paint):")
            for k, v in sorted(r["gate_drops"].items(),
                               key=lambda kv: -kv[1])[:8]:
                extra = (r.get("gate_by_reason") or {}).get(k, {})
                gf = extra.get("gt_px_frac")
                print(f"      {v:4d}  {k:20s} gt="
                      f"{'-' if gf is None else f'{gf:.0%}':>5} "
                      f"real_line={extra.get('n_real_line')} "
                      f"span_p50={extra.get('span_p50')} "
                      f"side_p50={extra.get('side_p50')}")
        if r.get("dropout_runs"):
            print(f"  dropout runs         : {r['dropout_runs']} "
                  f"({r['dropout_frames']} frames, "
                  f"{r['dropout_s_total']}s total, longest "
                  f"{r['dropout_longest_s']}s)")
            print(f"  dropout seconds <=0.5/1/2/5s: "
                  f"{r.get('dropout_s_within_0.5s')}/"
                  f"{r.get('dropout_s_within_1.0s')}/"
                  f"{r.get('dropout_s_within_2.0s')}/"
                  f"{r.get('dropout_s_within_5.0s')}")
        if r.get("gt_audit"):
            print("  GT paint vs model line mask (recall = hit/GT px):")
            print("      side/band     GT_px   model_px  hit_px   recall")
            for k in ("L:far", "L:mid", "L:near", "R:far", "R:mid", "R:near"):
                v = r["gt_audit"].get(k)
                if not v:
                    continue
                rec = v["recall"]
                print(f"      {k:10s} {v['gt_px']:8d} {v['model_px']:9d} "
                      f"{v['hit_px']:7d}   "
                      f"{'-' if rec is None else f'{rec:.1%}':>7}")
        if r.get("drop_stats_n"):
            print(f"  dropped comps n={r['drop_stats_n']} "
                  f"(w,h) p50={r['drop_wh_p50']} p90={r['drop_wh_p90']}")

    if len(reports) > 1:
        tot = sum(r["frames"] for r in reports)
        pr = sum(r["paired"] for r in reports) / max(1, tot)
        ss = sum(round(r["strict_sensor_rate"] * r["frames"]) for r in reports)
        print(f"\n=== ALL {len(reports)} EPISODES: frames={tot} "
              f"paired={pr:.1%} strict_sensor={ss / max(1, tot):.1%} ===")

    out = (Path(args.out) if args.out
           else config.LOGS_DIR / "m5_lane_continuity.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reports, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print(f"report -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
