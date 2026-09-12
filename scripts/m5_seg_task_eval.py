"""Segmentation checkpoint validation by the PAIRING task, not by mask IoU.

The 2026-09-11 experiments settled that neither mask metric can select a
checkpoint for this stack:

* annotation-GT line IoU is monotonically ANTI-correlated with the paired
  rate (v13b 0.0109 -> 26.0% paired; v8 0.0922 -> 5.3% paired), so training
  or selecting on it optimises away the property the drive loop needs;
* the manual stroke gate measures recall only (no precision penalty) and
  rated a near-blind-on-mountain checkpoint 72.9%.

What the drive loop actually needs is the number this script reports: how
often the own lane can be PAIRED (two-sided) and whether the paired centre
lands in the ego lane.  Use it to compare checkpoints and to early-stop
training, e.g.

    .venv\\Scripts\\python.exe scripts\\m5_seg_task_eval.py ^
        --model logs\\m5_seg\\seg_model\\best.pt

The episode set is printed for every run on purpose: shadow episodes are
written by every live drive, so a "newest N" selection drifts and two runs
at different times are not comparable unless the same names are reused.
Pass ``--episode-names`` to pin them.
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
from beamng_autopilot.lane import pair_lane_markings
from beamng_autopilot.vision.heads.semantic import SemanticHead
from beamng_autopilot.vision.hydra import FrameContext
from beamng_autopilot.vision.ring import CAMERA_RING, FRONT_MAIN
from beamng_autopilot.vision.segmentation import Segmenter

# The paired centre is the MIDPOINT of the two detected lane boundaries, so
# when the car sits in its own lane it reads ~0 m in the car frame (left
# positive).  CORRECTION 2026-09-11: this used to be -1.75, which is the
# "centred" value of a different quantity - the offset from a single line
# (``line_lat``; fsd_drive.py's corrector comment says "+1.75 centred").
# Using -1.75 for the paired centre mis-measured it by a full half lane:
# on town episodes the GT paint midpoint and the recorded plan both give
# ~0 (GT p50 +0.33/+0.73, trajectory p50 +0.18/+0.07), and the wrong target
# flipped the model ranking (v13b in-lane 19% vs hand 18% under -1.75;
# 92% vs 52% under 0).  Re-run model selection with this value before
# trusting any checkpoint pin.
EGO_LANE_CENTRE_M = 0.0
IN_LANE_TOL_M = 1.2


def _episodes(data_dir: Path, pattern: str, names, count: int) -> list[Path]:
    if names:
        return [data_dir / n for n in names]
    fs = sorted(glob.glob(str(data_dir / pattern)), key=os.path.getmtime)
    return [Path(f) for f in fs[-max(1, count):]]


def measure(model: str | None, eps: list[Path]) -> dict:
    seg = Segmenter(model_path=model) if model else Segmenter()
    sem = SemanticHead(segmenter=seg)
    mount = next(m for m in CAMERA_RING if m.role == FRONT_MAIN)
    frames = paired = 0
    lats: list[float] = []
    per_ep: dict[str, dict] = {}
    for ep in eps:
        if not Path(ep).is_file():
            continue
        sem.reset()
        d = np.load(ep, allow_pickle=True)
        meta = json.loads(bytes(d["meta"]).decode())
        cam = mount.camera_model(int(meta["cam_w"]), int(meta["cam_h"]))
        xs, ys, hds, rgbs = d["x"], d["y"], d["heading"], d["rgb"]
        ts = d["t"]
        ep_frames = ep_paired = 0
        for i in range(len(xs)):
            pos = np.array([float(xs[i]), float(ys[i]), 0.0])
            heading = float(hds[i])
            ctx = FrameContext(frame_rgb=np.asarray(rgbs[i], np.uint8), cam=cam,
                               pos=pos, heading=heading, ground_z=0.0,
                               role="front_main", timestamp=float(ts[i]))
            out = sem.run(ctx)
            marks = list(out.meta.get("markings") or [])
            frame = pair_lane_markings(marks, pos, heading) if marks else None
            frames += 1
            ep_frames += 1
            if frame is not None and getattr(frame, "paired", False):
                paired += 1
                ep_paired += 1
                c = np.asarray(frame.center, dtype=float)[:, :2]
                p = pos[:2]
                fwd = np.array([np.cos(heading), np.sin(heading)])
                left = np.array([-fwd[1], fwd[0]])
                near = c[np.linalg.norm(c - p, axis=1) <= 12.0]
                if len(near):
                    lats.append(float(((near - p) @ left).mean()))
        per_ep[Path(ep).name] = {
            "frames": ep_frames, "paired": ep_paired,
            "paired_rate": (round(ep_paired / ep_frames, 4)
                            if ep_frames else 0.0)}
    v = np.asarray(lats, dtype=float)
    return {
        "model": model or "deployed default",
        "episodes": [Path(e).name for e in eps],
        "frames": frames,
        "paired": paired,
        "paired_rate": round(paired / frames, 4) if frames else 0.0,
        "lat_mean_m": round(float(v.mean()), 3) if len(v) else None,
        "lat_p50_m": round(float(np.median(v)), 3) if len(v) else None,
        "in_lane_rate": (round(float((np.abs(v - EGO_LANE_CENTRE_M)
                                     < IN_LANE_TOL_M).mean()), 4)
                         if len(v) else 0.0),
        "per_episode": per_ep,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="validate a seg checkpoint by the pairing task")
    ap.add_argument("--model", default=None,
                    help="checkpoint (default: the deployed model)")
    ap.add_argument("--data", default=str(config.LOGS_DIR / "m5_e2e"))
    ap.add_argument("--pattern", default="shadow_fsd_*.npz")
    ap.add_argument("--episodes", type=int, default=4,
                    help="how many NEWEST episodes (ignored with --episode-names)")
    ap.add_argument("--episode-names", nargs="*", default=None,
                    help="pin exact episode filenames for a comparable result")
    ap.add_argument("--json", default=None, help="write the report here")
    args = ap.parse_args()

    eps = _episodes(Path(args.data), args.pattern, args.episode_names,
                    args.episodes)
    if not eps:
        print(f"no episodes in {args.data}")
        return 1
    r = measure(args.model, eps)
    print(f"model        : {r['model']}")
    print(f"episodes     : {len(r['episodes'])} "
          f"(pinned={bool(args.episode_names)})")
    print(f"frames       : {r['frames']}")
    print(f"PAIRED rate  : {r['paired_rate']:.1%}  ({r['paired']} frames)")
    print(f"centre lat   : mean={r['lat_mean_m']} p50={r['lat_p50_m']} m "
          f"(ego lane {EGO_LANE_CENTRE_M} m)")
    print(f"IN-LANE rate : {r['in_lane_rate']:.1%}  "
          f"(|lat - {EGO_LANE_CENTRE_M}| < {IN_LANE_TOL_M} m)")
    out = (Path(args.json) if args.json
           else config.LOGS_DIR / "seg_task_eval.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print(f"report       : {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
