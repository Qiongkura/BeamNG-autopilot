"""T07 offline arm comparison for boundary evidence (no control change).

Runs the three publishers over recorded data and reports what each arm
finds, WITHOUT claiming accuracy: there is no independent boundary truth
in the recordings, so the honest outputs are the arm's own counts, the
agreement between arms derived from DIFFERENT sensors (LiDAR step vs
semantic mask edge), and the thresholds each arm used.  A map edge may be
included as an auxiliary reference and is labelled as such (plan §7.1:
"地图 edge_over 是辅助度量，不自动等于真实铺装/漆线真值").

Two evidence levels are mixed on purpose and labelled:

* ``--cloud`` / ``--mask`` npz captured live (class 2: recorded sensor
  replay) - the arms run on real sensor data;
* synthetic fallbacks when nothing is recorded, so the script still runs
  and says so.

Usage::

    .venv\\Scripts\\python.exe scripts\\m5_boundary_arms.py \\
        --cloud logs/goal_20260921/boundary_capture.npz --json logs/arms.json
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

from beamng_autopilot.lane.boundary_evidence import (  # noqa: E402
    CURB_HEIGHT_M,
    adaptive_step_threshold,
    associate,
    beam_spacing_m,
    curb_candidates,
    obstacle_entities,
)


def _fixed_threshold_arm(points, ground_z: float, step_m: float = 0.20):
    """The pre-T07 behaviour: one hand-tuned metre constant everywhere."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or len(pts) < 6:
        return {"kind": "curb", "n_points": 0, "threshold_m": step_m}
    p2, z = pts[:, :2], pts[:, 2]
    ang = np.arctan2(p2[:, 1], p2[:, 0])
    rng = np.linalg.norm(p2, axis=1)
    rb = np.floor(rng / 0.25).astype(int)
    ab = np.floor((ang + math.pi) / math.radians(2.0)).astype(int)
    cells: dict[tuple[int, int], list[int]] = {}
    for k in range(len(pts)):
        cells.setdefault((int(rb[k]), int(ab[k])), []).append(k)
    n = 0
    for (r_, a_), ids in cells.items():
        nbr = cells.get((r_, a_ + 1))
        if not nbr:
            continue
        if abs(float(np.median(z[nbr])) - float(np.median(z[ids]))) > step_m:
            n += 2
    return {"kind": "curb", "n_points": int(n), "threshold_m": float(step_m)}


def _run_arms(cloud, ground_z: float, mask=None, cam=None, pos=None,
              heading: float = 0.0, obstacles=()) -> dict:
    arms: dict = {}
    if cloud is not None and len(cloud):
        arms["curb_fixed_0p20"] = _fixed_threshold_arm(cloud, ground_z)
        ad = curb_candidates(cloud, ground_z=ground_z,
                             pos=(pos if pos is not None else np.zeros(3)))
        arms["curb_adaptive"] = {
            "kind": "curb", "n_points": int(len(ad.points)),
            "threshold_p50_m": ad.as_dict()["meta"].get("threshold_p50_m"),
            "local_gap_p50_m": ad.as_dict()["meta"].get("local_gap_p50_m")}
        arms["curb_candidates_sample"] = (
            ad.points[:20].round(3).tolist() if len(ad.points) else [])
        arms["_curb_evidence"] = ad
    else:
        arms["curb_fixed_0p20"] = {"kind": "curb", "n_points": None,
                                   "note": "no cloud recorded"}
        arms["curb_adaptive"] = {"kind": "curb", "n_points": None,
                                 "note": "no cloud recorded"}
    obs = obstacle_entities(obstacles)
    arms["obstacle_entities"] = obs.as_dict()
    arms["_obs_evidence"] = obs
    if mask is not None and cam is not None and pos is not None:
        from beamng_autopilot.lane.boundary_evidence import pavement_edges
        pe = pavement_edges(mask, cam, pos, heading, ground_z=ground_z)
        arms["pavement_edge"] = pe.as_dict()
        arms["_edge_evidence"] = pe
    else:
        arms["pavement_edge"] = {"kind": "pavement_edge", "n_points": None,
                                 "note": "no mask/camera recorded"}
    ev = [arms[k] for k in ("_curb_evidence", "_edge_evidence",
                            "_obs_evidence") if k in arms]
    arms["association"] = associate(ev)
    for key in ("_curb_evidence", "_edge_evidence", "_obs_evidence"):
        arms.pop(key, None)
    return arms


def main() -> int:
    ap = argparse.ArgumentParser(description="T07 boundary arm comparison")
    ap.add_argument("--cloud", type=str, default=None,
                    help="npz from scripts/m5_capture_boundary.py (points, "
                         "ground_z, optional mask/cam/pos)")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    payload: dict = {"evidence_level": "synthetic" if not args.cloud
                     else "recorded_sensor_replay",
                     "curb_height_m": CURB_HEIGHT_M,
                     "beam_spacing_m": {str(d): round(beam_spacing_m(d), 3)
                                        for d in (5, 10, 20, 40)},
                     "adaptive_threshold_m": {
                         str(d): round(adaptive_step_threshold(d * math.tan(
                             math.radians(2.0 * 26.9 / 16))), 3)
                         for d in (5, 10, 20, 40)},
                     "arms": {}, "notes": []}
    cloud = mask = cam = pos = None
    obstacles: list = []
    ground_z = 0.0
    heading = 0.0
    if args.cloud:
        z = np.load(args.cloud, allow_pickle=True)
        if "points" in z.files:
            cloud = np.asarray(z["points"], dtype=float)
        if "ground_z" in z.files:
            ground_z = float(np.asarray(z["ground_z"]).ravel()[0])
        if "mask" in z.files:
            mask = np.asarray(z["mask"], dtype=bool)
        if "pos" in z.files:
            pos = np.asarray(z["pos"], dtype=float).ravel()[:3]
        if "heading" in z.files:
            heading = float(np.asarray(z["heading"]).ravel()[0])
        if "cam_offset" in z.files and "cam_fwd" in z.files and pos is not None:
            from beamng_autopilot.vision.projection import CameraModel
            cam = CameraModel(
                offset=np.asarray(z["cam_offset"], dtype=float).ravel(),
                fwd_local=np.asarray(z["cam_fwd"], dtype=float).ravel(),
                up_local=np.asarray(z["cam_up"], dtype=float).ravel()
                if "cam_up" in z.files else np.array([0.0, 0.0, 1.0]),
                fov_deg=float(np.asarray(z["cam_fov"]).ravel()[0])
                if "cam_fov" in z.files else 65.0,
                width=int(np.asarray(z["cam_w"]).ravel()[0])
                if "cam_w" in z.files else 320,
                height=int(np.asarray(z["cam_h"]).ravel()[0])
                if "cam_h" in z.files else 240)
        print(f"[arms] loaded {args.cloud}: points={0 if cloud is None else len(cloud)} "
              f"mask={'yes' if mask is not None else 'no'} "
              f"cam={'yes' if cam is not None else 'no'}")
    if cloud is None and mask is None:
        payload["notes"].append(
            "nothing recorded: running a SYNTHETIC profile so the arms are "
            "exercised; these numbers are not evidence about the road")
        rng = np.random.default_rng(0)
        xs = np.arange(2.0, 30.0, 0.05)
        zs = np.zeros_like(xs)
        zs[xs >= 12.0] = CURB_HEIGHT_M
        cloud = np.column_stack([xs, rng.normal(0.0, 0.01, xs.size), zs])
        pos = np.zeros(3)

    payload["arms"] = _run_arms(cloud, ground_z, mask=mask, cam=cam, pos=pos,
                                heading=heading, obstacles=obstacles)
    for name, arm in payload["arms"].items():
        if not isinstance(arm, dict):
            continue
        if name in ("association",):
            print(f"[arms] association: same={len(arm['same_object'])} "
                  f"distinct={len(arm['distinct'])} merged={arm['merged']}")
            continue
        print(f"[arms] {name:18s} kind={arm.get('kind')} "
              f"n={arm.get('n_points')} "
              f"thr={arm.get('threshold_p50_m', arm.get('threshold_m'))}")
    print("[arms] no independent boundary truth in these recordings: the "
          "numbers above are COUNTS and cross-arm agreement, not accuracy")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(payload, indent=2,
                                              ensure_ascii=False),
                                   encoding="utf-8")
        print(f"[arms] -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
