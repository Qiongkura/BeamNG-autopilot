"""One-command FSD driving evaluation: run N live drives and report the
driving-quality metrics (lane crossings / reversing / off-road / stalls
/ speed smoothness / final stop) for every run plus an aggregate.

Two modes::

    # analyse existing telemetry exports
    python scripts/m5_fsd_eval.py --json logs/a.json logs/b.json \\
        --speed 6 --goal 572 533.5

    # run the car N times, then analyse (needs BeamNG.tech running)
    python scripts/m5_fsd_eval.py --runs 3 --attach --runtime tech \\
        --seconds 90 --speed 6 --steps 2 --cam-w 400 --cam-h 300 \\
        --teleport 729.6 763.9 45 --goal 572 533.5 --out logs/fsd_eval.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beamng_autopilot.config import LOGS_DIR, PROJECT_ROOT
from beamng_autopilot.eval import assess_many

DRIVE_SCRIPT = PROJECT_ROOT / "scripts" / "m5_fsd_drive.py"


def _run_drive(i: int, args) -> Path:
    """Launch one m5_fsd_drive run; returns the telemetry JSON path."""
    out = LOGS_DIR / f"fsd_eval_run{i}.json"
    cmd = [
        sys.executable, str(DRIVE_SCRIPT),
        "--runtime", args.runtime,
        "--seconds", str(args.seconds),
        "--speed", str(args.speed),
        "--steps", str(args.steps),
        "--cam-w", str(args.cam_w),
        "--cam-h", str(args.cam_h),
        "--out", str(out),
        "--lane-mode", str(args.lane_mode),
    ]
    if args.attach:
        cmd.append("--attach")
    if args.teleport is not None:
        cmd += ["--teleport"] + [str(x) for x in args.teleport]
    if args.goal is not None:
        cmd += ["--goal"] + [str(x) for x in args.goal]
    print(f"[fsd-eval] run {i + 1}: {' '.join(cmd)}")
    t0 = time.time()
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False,
                   timeout=max(60.0, float(args.seconds) + 300.0))
    print(f"[fsd-eval] run {i + 1} finished in {time.time() - t0:.1f}s")
    return out


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _line(name, r: dict) -> str:
    src = ",".join(f"{k}={v}" for k, v in r.get("source", {}).items())
    lvl = ",".join(f"{k}={v}" for k, v in r.get("level", {}).items())
    return (
        f"{name}: {r['frames']}f {r.get('duration_s', 0)}s "
        f"src[{src}] lvl[{lvl}] "
        f"rev={r.get('reversing_frames', 0)} "
        f"cross={r.get('cross_centre_frames', 0)}/"
        f"{r.get('cross_right_frames', 0)} "
        f"off={r.get('off_road_frames', 0)} "
        f"stall={r.get('stall_frames', 0)} "
        f"v={r.get('speed_min')}/{r.get('speed_med')}/{r.get('speed_max')} "
        f"thrflip={r.get('throttle_flips', 0)} "
        f"brkflip={r.get('brake_flips', 0)} "
        f"trav={r.get('travelled_m', 0)}m "
        f"final={r.get('final_pos')} rem={r.get('final_rem_end')} "
        + (f"goal={r.get('goal_dist_m')}m" if "goal_dist_m" in r else "")
        + (f" creep={r.get('creep_frac')}" if "creep_frac" in r else "")
    )


def _summary(rs) -> dict:
    tot = {
        "runs": len(rs),
        "reversing_frames": sum(r.get("reversing_frames", 0) for r in rs),
        "cross_centre_frames": sum(r.get("cross_centre_frames", 0) for r in rs),
        "cross_right_frames": sum(r.get("cross_right_frames", 0) for r in rs),
        "off_road_frames": sum(r.get("off_road_frames", 0) for r in rs),
        "stall_frames": sum(r.get("stall_frames", 0) for r in rs),
        "emergency_frames": sum(r.get("emergency_frames", 0) for r in rs),
        "minimal_risk_frames": sum(
            r.get("level", {}).get("minimal_risk", 0) for r in rs),
        "throttle_flips": sum(r.get("throttle_flips", 0) for r in rs),
        "brake_flips": sum(r.get("brake_flips", 0) for r in rs),
        "travelled_m": round(sum(r.get("travelled_m", 0) for r in rs), 1),
    }
    return tot


def main() -> int:
    ap = argparse.ArgumentParser(description="FSD driving evaluation")
    ap.add_argument("--json", nargs="*", default=None,
                    help="existing telemetry JSON files to analyse")
    ap.add_argument("--runs", type=int, default=0,
                    help="number of live drives to run (needs Tech)")
    ap.add_argument("--attach", action="store_true")
    ap.add_argument("--runtime", choices=("auto", "steam", "tech"),
                    default="tech")
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--speed", type=float, default=6.0)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--cam-w", type=int, default=400)
    ap.add_argument("--cam-h", type=int, default=300)
    ap.add_argument("--teleport", nargs=3, type=float, default=None,
                    metavar=("X", "Y", "YAW_DEG"))
    ap.add_argument("--goal", nargs=2, type=float, default=None,
                    metavar=("X", "Y"))
    ap.add_argument("--out", type=str, default=None,
                    help="path for the aggregate JSON report")
    ap.add_argument("--lane-mode", choices=("map", "auto", "sensor"),
                    default="map",
                    help="lane-keep reference policy passed to the drive")
    args = ap.parse_args()

    paths: list[Path] = []
    if args.runs > 0:
        for i in range(args.runs):
            paths.append(_run_drive(i, args))
    if args.json:
        paths += [Path(p) for p in args.json]
    if not paths:
        ap.error("give --runs N or --json file... (or both)")

    loaded = []
    for p in paths:
        try:
            loaded.append(_load(p))
        except Exception as e:
            print(f"[fsd-eval] cannot read {p}: {e}")
    if not loaded:
        print("[fsd-eval] no telemetry loaded")
        return 1

    rs = assess_many(loaded, goal=args.goal, cruise=args.speed)
    print("\n[fsd-eval] per-run:")
    for i, r in enumerate(rs):
        print("  " + _line(f"run{i + 1}", r))
    s = _summary(rs)
    print("\n[fsd-eval] aggregate:")
    for k, v in s.items():
        print(f"  {k}: {v}")

    if args.out:
        report = {"runs": rs, "summary": s}
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                         encoding="utf-8")
        print(f"[fsd-eval] report -> {out_p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
