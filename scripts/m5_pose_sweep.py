"""Thin, Tech-only entry for audited pose sweeps; see the PowerShell wrapper."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beamng_autopilot.experiments.pose_sweep import (
    PoseRequest, machine_lease, pose_matrix, run_probe,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", required=True)
    ap.add_argument("--anchor", type=float, nargs=3, required=True)
    ap.add_argument("--goal", type=float, nargs=2, required=True)
    ap.add_argument("--lateral", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    ap.add_argument("--yaw", type=float, nargs="+", default=[0.0, 5.0, 10.0])
    ap.add_argument("--seg-model", required=True)
    ap.add_argument("--seconds", type=float, default=25.0)
    ap.add_argument("--max-wall-s", type=float, default=150.0)
    ap.add_argument("--speed", type=float, default=6.0)
    ap.add_argument("--port", type=int, default=64257)
    ap.add_argument("--pose-contract", choices=("production_alignment", "fixed_actual_pose"),
                    default="production_alignment")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--probe-geometry", action="store_true")
    ap.add_argument("--capture-range-replay", action="store_true")
    args = ap.parse_args()
    requests = [PoseRequest(args.map, tuple(p), tuple(args.goal),
                            str(Path(args.seg_model).resolve()), args.seconds,
                            args.speed, args.max_wall_s, args.pose_contract,
                            probe_geometry=args.probe_geometry,
                            capture_range_replay=args.capture_range_replay)
                for p in pose_matrix(tuple(args.anchor), args.lateral, args.yaw)]
    if args.plan_only:
        from dataclasses import asdict
        print(json.dumps([asdict(r) for r in requests], ensure_ascii=False, indent=2))
        return 0
    if args.out.exists():
        ap.error("output directory already exists; use a new run id to preserve evidence")
    if not Path(args.seg_model).is_file():
        ap.error("segmentation checkpoint does not exist")
    lease = machine_lease()
    held = lease.acquire()
    if not held["acquired"]:
        print(held["reason"])
        return 5
    results = []
    try:
        for i, request in enumerate(requests):
            result = run_probe(request, args.out / f"pose_{i:02d}",
                               port=args.port + i, lease=lease)
            results.append(result)
            print(f"pose {i}: {result['status']} {result.get('reasons', [])}", flush=True)
            if result["status"] == "blocked":
                break
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "summary.json").write_text(json.dumps(
            {"requested": len(requests), "attempted": len(results),
             "qualified": sum(r["qualified"] for r in results), "results": results},
            ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        lease.release()
    return 0 if len(results) == len(requests) and all(r["qualified"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
