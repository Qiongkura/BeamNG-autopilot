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


def _episodes(data_dir: Path, pattern: str) -> list[str]:
    return sorted(glob.glob(str(data_dir / pattern)), key=os.path.getmtime)


def _near_gt_line_px(label: np.ndarray) -> int:
    if label is None:
        return 0
    h = label.shape[0]
    near = np.asarray(label, dtype=np.uint8)[int(h * GT_NEAR_ROW_FRAC):]
    return int((near == LABEL_LINE).sum())


def measure_episode(ep: str, *, frames: int, sem: SemanticHead,
                    verbose: bool) -> dict:
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
    n_marks: list[int] = []
    lat_paired: list[float] = []

    for i in idxs:
        pos = np.array([float(xs[i]), float(ys[i]), 0.0])
        heading = float(hds[i])
        rgb = np.asarray(rgbs[i], dtype=np.uint8)
        ctx = FrameContext(frame_rgb=rgb, cam=cam, pos=pos,
                           heading=heading, ground_z=0.0, role="front_main")
        out = sem.run(ctx)
        markings = list(out.meta.get("markings") or [])
        n_marks.append(len(markings))

        frame = None
        if markings:
            frame = pair_lane_markings(markings, pos, heading)
        paired = bool(frame is not None and getattr(frame, "paired", False))

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
    if lat_paired:
        res["lat_paired_mean_m"] = round(float(np.mean(lat_paired)), 3)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="perception lane continuity (offline)")
    ap.add_argument("--data", type=str, default=None,
                    help="episode directory (default logs/m5_e2e)")
    ap.add_argument("--pattern", type=str, default="shadow_fsd_*.npz")
    ap.add_argument("--episodes", type=int, default=1,
                    help="how many NEWEST episodes to measure")
    ap.add_argument("--frames", type=int, default=40,
                    help="frames sampled per episode (0 = all)")
    ap.add_argument("--all-frames", action="store_true",
                    help="sample every frame of the episode")
    ap.add_argument("--seg-model", type=str, default=None,
                    help="segmentation checkpoint (default: deployed best.pt)")
    ap.add_argument("--out", type=str, default=None,
                    help="write the report JSON here")
    args = ap.parse_args()

    data_dir = Path(args.data) if args.data else config.LOGS_DIR / "m5_e2e"
    eps = _episodes(data_dir, args.pattern)
    if not eps:
        print(f"no episodes matching {args.pattern} in {data_dir}")
        return 1
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
                            verbose=os.environ.get("LANE_CONT_VERBOSE") == "1")
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
