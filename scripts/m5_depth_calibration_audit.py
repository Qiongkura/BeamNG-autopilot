"""Audit BeamNG non-shared float-depth encoding against generated truth points.

This is a read-only calibration witness. It never changes the frozen projection
or admission gates. It consumes ``*_depthcal.npz`` plus the matching
``scene_*.json`` truth chain, samples a 3x3 median at projected truth points,
and fits a monotone piecewise mapping from the raw float buffer to true optical
world depth. The holdout split is every fifth point; bins require at least two
holdout points and p95 absolute residual <= 1 m to be called supported.

The effective range is therefore a measured validity statement, not a claim
that the raw buffer is OpenGL NDC or that depth can certify a line projection.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beamng_autopilot.experiments.auto_truth import project_point  # noqa: E402

BIN_EDGES = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, float("inf"))
HOLDOUT_MOD = 5
MIN_HOLDOUT = 2
MAX_P95_M = 1.0


def _pava(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return monotone increasing block knots (raw value, depth metres)."""
    order = np.argsort(x, kind="mergesort")
    blocks: list[list[float]] = []
    for xi, yi in zip(x[order], y[order]):
        blocks.append([float(xi), float(xi), float(yi), 1.0])
        while len(blocks) >= 2 and blocks[-2][2] > blocks[-1][2]:
            a, b = blocks[-2], blocks[-1]
            n = a[3] + b[3]
            blocks[-2] = [a[0], b[1],
                          (a[2] * a[3] + b[2] * b[3]) / n, n]
            blocks.pop()
    # np.interp requires non-decreasing x; duplicate raw values collapse here.
    out: list[tuple[float, float]] = []
    for lo, hi, val, _n in blocks:
        if out and abs(out[-1][0] - lo) < 1e-12:
            out[-1] = (lo, val)
        else:
            out.append((lo, val))
    return np.asarray(out, dtype=float)


def _predict(model: np.ndarray, raw: np.ndarray) -> np.ndarray:
    return np.interp(raw, model[:, 0], model[:, 1])


def _load_samples(depthcal: Path, scene: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    d = np.load(depthcal, allow_pickle=False)
    raw = np.asarray(d["depth_ndc"], dtype=np.float32)
    cam = json.loads(str(d["camera_json"]))
    blob = json.loads(scene.read_text(encoding="utf-8"))
    points = [p for line in (blob.get("scene", {}).get("lines") or [])
              for p in (line.get("truth_points") or [])]
    camera_pos = np.asarray(cam["pose"]["pos"], dtype=float)
    camera_fwd = np.asarray(cam["pose"]["basis"]["fwd"], dtype=float)
    samples: list[tuple[float, float]] = []
    skipped = {"behind": 0, "outside": 0, "nonfinite": 0}
    h, w = raw.shape[:2]
    for p in points:
        u, v = project_point(p["world"], cam)
        if not np.isfinite(u + v):
            skipped["nonfinite"] += 1
            continue
        x, y = int(round(float(u))), int(round(float(v)))
        if not (1 <= x < w - 1 and 1 <= y < h - 1):
            skipped["outside"] += 1
            continue
        patch = raw[y - 1:y + 2, x - 1:x + 2]
        val = float(np.median(patch))
        depth = float((np.asarray(p["world"], dtype=float) - camera_pos)
                      @ camera_fwd)
        if not (math.isfinite(val) and math.isfinite(depth) and depth > 0):
            skipped["nonfinite"] += 1
            continue
        samples.append((val, depth))
    return (np.asarray([s[0] for s in samples], dtype=float),
            np.asarray([s[1] for s in samples], dtype=float),
            {"truth_points": len(points), "skipped": skipped,
             "raw_min": float(np.nanmin(raw)),
             "raw_max": float(np.nanmax(raw)),
             "raw_p99": float(np.nanpercentile(raw, 99))})


def _bin_report(y: np.ndarray, err: np.ndarray) -> list[dict]:
    rows: list[dict] = []
    for lo, hi in zip(BIN_EDGES, BIN_EDGES[1:]):
        mask = (y >= lo) & (y < hi)
        n = int(mask.sum())
        if not n:
            rows.append({"lo_m": lo, "hi_m": hi, "n": 0,
                         "supported": False, "reason": "no points"})
            continue
        ae = np.abs(err[mask])
        p95 = float(np.percentile(ae, 95))
        rows.append({"lo_m": lo, "hi_m": hi, "n": n,
                     "rmse_m": float(np.sqrt(np.mean(err[mask] ** 2))),
                     "p95_abs_m": p95,
                     "supported": bool(n >= MIN_HOLDOUT and p95 <= MAX_P95_M),
                     "reason": ("" if n >= MIN_HOLDOUT and p95 <= MAX_P95_M
                                else "insufficient holdout or p95 residual > 1m")})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True,
                    help="collection root containing frames/ and scene_*.json")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    depth_files = sorted((args.root / "frames").glob("*_depthcal.npz"))
    scenes = sorted(p for p in args.root.glob("scene_*.json")
                    if p.name != "scene_report.json")
    result = {"schema": "beamng-depth-calibration-v1",
              "source_root": str(args.root),
              "holdout": {"rule": "every fifth projected truth point",
                          "mod": HOLDOUT_MOD, "min_n": MIN_HOLDOUT,
                          "max_p95_m": MAX_P95_M},
              "scenes": [], "missing_depthcal_scenes": [],
              "conclusion": {"raw_buffer_is_verified_ndc": False}}
    for scene in scenes:
        name = scene.stem.removeprefix("scene_")
        depth_path = args.root / "frames" / f"{name}_depthcal.npz"
        if not depth_path.exists():
            result["missing_depthcal_scenes"].append(name)
            continue
        raw, truth, meta = _load_samples(depth_path, scene)
        if len(raw) < 4:
            result["scenes"].append({"scene": name, "status": "insufficient",
                                     **meta, "n_samples": int(len(raw))})
            continue
        hold = np.arange(len(raw)) % HOLDOUT_MOD == 0
        model = _pava(raw[~hold], truth[~hold])
        pred = _predict(model, raw[hold])
        err = pred - truth[hold]
        bins = _bin_report(truth[hold], err)
        supported = [b for b in bins if b.get("supported")]
        result["scenes"].append({
            "scene": name, "status": "measured", **meta,
            "n_samples": int(len(raw)), "n_holdout": int(hold.sum()),
            "raw_range_at_truth": [float(raw.min()), float(raw.max())],
            "truth_depth_range_m": [float(truth.min()), float(truth.max())],
            "holdout_rmse_m": float(np.sqrt(np.mean(err ** 2))),
            "holdout_p95_abs_m": float(np.percentile(abs(err), 95)),
            "bins": bins,
            "supported_bins": [[b["lo_m"], b["hi_m"]] for b in supported],
            "calibration_knots": model.tolist(),
        })
    measured = [s for s in result["scenes"] if s.get("status") == "measured"]
    result["conclusion"].update({
        "measured_scenes": len(measured),
        "missing_depthcal_scenes": len(result["missing_depthcal_scenes"]),
        "effective_range_is_not_global": True,
        "decision": ("depth usable only in explicitly supported bins; reject it "
                     "for line-separation decisions outside those bins"),
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print(json.dumps({"measured": len(measured),
                      "missing": result["missing_depthcal_scenes"],
                      "out": str(args.out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
