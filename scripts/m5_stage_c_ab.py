"""Stage-C single-factor A/B: one switch, >=5 interleaved Tech runs per arm.

The plan's stage C allows exactly one already-shadow-validated change to be
wired and compared, under a declared ODD, the same route/goal, a controlled
start pose and a full record.  This harness does that for ONE environment
factor and nothing else:

* arms are taken round-robin (A,B,B,A,...) so session drift is shared;
* every run is one ``m5_fsd_drive`` from the same teleport, with the same
  ``--lane-mode sensor --strict`` baseline, pinned duration/speed;
* the metric set is the plan's §7 minimum: stop seconds and longest stop
  (end-zone split out), off-pavement, lane-crossing frames with their
  coverage, damage, distance, lane-source rate, drivable-gate seconds, plus
  the UNKNOWN counts so "not measured" can never be read as "clean";
* the record carries what §8.4 asks for: the switch state for both arms,
  the git commit/dirty flag, the drive command per run, and the per-run
  telemetry path.

Usage::

    .venv\\Scripts\\python.exe scripts\\m5_stage_c_ab.py --attach \\
        --factor BEAMNG_GEOM_GROUND_PLANE --arms 5 --seconds 20 \\
        --goal 800 735 --map italy --teleport 779.5 734.63 -13.2 \\
        --out-dir logs/goal_20260921/stageC_plane
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PYEXE = ROOT / ".venv" / "Scripts" / "python.exe"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: Switches fixed for BOTH arms (the values every earlier measurement used);
#: the factor itself is the only thing that moves.
PIN: dict[str, str] = {}
BASE_ARGS = ("--lane-mode", "sensor", "--strict")


def _code_state() -> dict:
    def _git(*args):
        try:
            out = subprocess.run(["git", *args], cwd=str(ROOT),
                                 capture_output=True, text=True)
            return out.stdout.strip()
        except Exception:
            return ""
    dirty = _git("status", "--porcelain")
    return {"commit": _git("rev-parse", "HEAD"),
            "dirty_files": len([ln for ln in dirty.splitlines() if ln.strip()]),
            "head_subject": _git("log", "-1", "--pretty=%s")}


def _canonical_digest(hist_path: Path) -> dict:
    proc = subprocess.run(
        [str(PYEXE), "scripts/m5_run_metrics.py", "--hist", str(hist_path)],
        cwd=str(ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        return {"status": "UNKNOWN",
                "reason": (proc.stderr or proc.stdout).strip()[-200:]}
    try:
        return list(json.loads(proc.stdout).values())[0]
    except Exception as exc:
        return {"status": "UNKNOWN", "reason": f"digest unreadable: {exc}"}


def run_metrics(hist: list[dict]) -> dict:
    """The plan's §7 metric set, read from one run's telemetry."""
    n = len(hist)
    if not n:
        return {"frames": 0, "status": "UNKNOWN"}
    t = [float(f.get("t") or 0.0) for f in hist]

    def _span(idx):
        tot = 0.0
        for i in idx:
            if i + 1 < n:
                tot += max(0.0, t[i + 1] - t[i])
        return round(tot, 2)

    def _num(v):
        return isinstance(v, (int, float)) and not isinstance(v, bool)

    stops = [i for i in range(n)
             if _num(hist[i].get("speed")) and float(hist[i]["speed"]) < 0.3]
    nodrv = [i for i in stops
             if str(hist[i].get("reason", "")).startswith("no drivable path")]
    lat = [f for f in hist if f.get("line_lat") is not None]
    lat_l = [f for f in hist if f.get("lat_left") is not None]
    lat_r = [f for f in hist if f.get("lat_right") is not None]
    # A crossing / off-pavement CLAIM is only a measurement when a
    # boundary was actually published this tick.  Hand-checking the
    # 2026-09-22 long runs showed the flags are set on frames with
    # ``lat_left = None`` too, where the field contract says the crossing
    # is UNKNOWN - counting those as "crossings" mixes a measurement with
    # a boundary-less claim, so they are counted separately.
    def _has_boundary(f):
        # ONLY the published boundary fields count.  ``body_lat_*`` is a
        # DERIVED corner distance (measured: it equals ``road_off`` on
        # every flagged frame), so accepting it as "a boundary was
        # published" made the gate pass on exactly the frames it exists to
        # separate - caught by re-running the check on the recorded runs.
        return (f.get("lat_left") is not None
                or f.get("lat_right") is not None)

    _cross = [f for f in hist
              if f.get("body_cross_l") or f.get("body_cross_r")]
    _off = [f for f in hist if (f.get("road_off") or 0.0) > 1e-9
            or (f.get("edge_over") or 0.0) > 1e-9
            or str(f.get("body_cov_status") or "") == "off_road"]
    bcross = [f for f in _cross if _has_boundary(f)]
    off = [f for f in _off if _has_boundary(f)]
    bcross_unmeasured = [f for f in _cross if not _has_boundary(f)]
    off_unmeasured = [f for f in _off if not _has_boundary(f)]
    dmg = [f.get("damage_total") for f in hist if _num(f.get("damage_total"))]
    dist = 0.0
    for a, b in zip(hist, hist[1:]):
        pa, pb = a.get("pos"), b.get("pos")
        if isinstance(pa, list) and isinstance(pb, list) and len(pa) > 1:
            dist += float(np.hypot(pb[0] - pa[0], pb[1] - pa[1]))
    lane = [str(f.get("lane_src") or "") for f in hist]
    drv = [f.get("lane_drivable") for f in hist
           if isinstance(f.get("lane_drivable"), dict)]
    drv_frac = [float(d["frac"]) for d in drv
                if _num(d.get("frac"))]
    unknown = collections.Counter()
    for f in hist:
        for k in ("line_lat", "lat_left", "lat_right"):
            if f.get(k) is None:
                unknown[k] += 1
        if f.get("reason") == "stale sensor":
            unknown["stale_sensor"] += 1
    risk_unknown = sum(1 for f in hist
                       if isinstance(f.get("lateral_risk"), dict)
                       and f["lateral_risk"].get("unknown"))
    return {
        "frames": n,
        "duration_s": round(t[-1] - t[0], 2) if n > 1 else 0.0,
        "distance_m": round(dist, 2),
        "stop_s": _span(stops),
        "stop_frames": len(stops),
        "no_drivable_path_s": _span(nodrv),
        "off_pavement_frames": len(off),
        "off_pavement_s": _span([i for i in range(n) if hist[i] in off]),
        "body_cross_frames": len(bcross),
        "body_cross_unmeasured_frames": len(bcross_unmeasured),
        "off_pavement_unmeasured_frames": len(off_unmeasured),
        "lat_frames": len(lat),
        "lat_left_frames": len(lat_l),
        "lat_right_frames": len(lat_r),
        "damage": (round(max(dmg), 3) if dmg else None),
        "lane_sensor_rate": round(
            sum(1 for s in lane if s == "sensor") / n, 3),
        "lane_drivable_frac_p50": (round(float(statistics.median(drv_frac)), 3)
                                   if drv_frac else None),
        "lane_drivable_checked_frames": len(drv_frac),
        "risk_unknown_frames": risk_unknown,
        "unknown_frames": dict(unknown),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="stage-C single-factor A/B")
    ap.add_argument("--factor", required=True,
                    help="env switch compared at 0 vs 1")
    ap.add_argument("--arms", type=int, default=5, help="runs PER ARM")
    ap.add_argument("--attach", action="store_true")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--speed", type=float, default=6.0)
    ap.add_argument("--goal", nargs=2, type=float, default=None)
    ap.add_argument("--map", type=str, default="italy")
    ap.add_argument("--seg-model", type=str, default=None)
    ap.add_argument("--teleport", nargs=3, type=float, default=None,
                    metavar=("X", "Y", "YAW_DEG"))
    ap.add_argument("--vis", type=int, default=0,
                    help="render an overlay frame every N ticks (0 = off); "
                         "each run writes its own --vis-dir")
    ap.add_argument("--order", default="ABBA",
                    help="arm order inside one round (default ABBA, so each "
                         "round is balanced and drift is shared)")
    ap.add_argument("--out-dir", type=str,
                    default="logs/goal_20260921/stageC")
    args = ap.parse_args()
    if args.arms < 5:
        print(f"[stageC] !! the plan requires >= 5 runs per arm; "
              f"--arms {args.arms} cannot support a claim")
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    order = [a for a in args.order.upper() if a in ("A", "B")]
    if not order:
        print("[stageC] --order must contain A and/or B")
        return 2
    code = _code_state()
    print(f"[stageC] factor={args.factor} arms/arm={args.arms} "
          f"order={order} seconds={args.seconds}")
    print(f"[stageC] code={code}")

    runs: list[dict] = []
    _seq = 0
    for i in range(args.arms):
        for arm in order:
            _seq += 1
            env = dict(os.environ)
            for k, v in PIN.items():
                env[k] = v
            env[args.factor] = "1" if arm == "B" else "0"
            # The tag carries the global sequence number as well as the
            # round: a balanced order like ABBA visits the same arm twice
            # per round, and naming by (arm, round) alone made the second
            # run overwrite the first one's telemetry (measured 2026-09-22).
            tag = f"{arm}{i}_{_seq:02d}"
            hist_path = out_dir / f"{tag}.json"
            cmd = [str(PYEXE), "scripts/m5_fsd_drive.py",
                   "--runtime", "tech", "--seconds", str(args.seconds),
                   "--speed", str(args.speed), "--out", str(hist_path),
                   *BASE_ARGS]
            if args.attach:
                cmd.append("--attach")
            if args.map:
                cmd += ["--map", str(args.map)]
            if args.goal is not None:
                cmd += ["--goal", str(args.goal[0]), str(args.goal[1])]
            if args.seg_model:
                cmd += ["--seg-model", str(args.seg_model)]
            if args.teleport is not None:
                cmd += ["--teleport", *[str(v) for v in args.teleport]]
            if args.vis:
                vis_dir = out_dir / f"vis_{tag}"
                cmd += ["--vis", str(args.vis), "--vis-dir", str(vis_dir)]
                rec_vis = str(vis_dir)
            else:
                rec_vis = None
            t0 = time.time()
            proc = subprocess.run(cmd, cwd=str(ROOT), env=env,
                                  capture_output=True, text=True)
            if not hist_path.is_file():
                tail = " | ".join((proc.stderr or proc.stdout or "")
                                  .strip().splitlines()[-2:])
                print(f"[stageC] {tag}: NO TELEMETRY - {tail}")
                continue
            hist = json.loads(hist_path.read_text(encoding="utf-8"))
            m = run_metrics(hist)
            rec = {"arm": arm, "round": i, "factor_value": env[args.factor],
                   "hist": str(hist_path), "vis_dir": rec_vis,
                   "cmd": " ".join(cmd),
                   "seconds_elapsed": round(time.time() - t0, 1),
                   "metrics": m, "digest": _canonical_digest(hist_path)}
            runs.append(rec)
            print(f"[stageC] {tag} {args.factor}={rec['factor_value']} "
                  f"dist={m['distance_m']:6.2f}m stop={m['stop_s']:6.2f}s "
                  f"nodrv={m['no_drivable_path_s']:6.2f}s "
                  f"off={m['off_pavement_frames']:2d} "
                  f"bcross={m['body_cross_frames']:2d} "
                  f"lat={m['lat_frames']}/{m['frames']} "
                  f"dmg={m['damage']} "
                  f"drvfrac={m['lane_drivable_frac_p50']} "
                  f"({rec['seconds_elapsed']}s)")
        (out_dir / "stageC_summary.json").write_text(
            json.dumps({"factor": args.factor, "order": order,
                        "arms_per_arm": args.arms, "code": code,
                        "pin": PIN, "base_args": list(BASE_ARGS),
                        "runs": runs}, indent=2, ensure_ascii=False),
            encoding="utf-8")

    if not runs:
        print("[stageC] no run produced telemetry")
        return 2
    print("\n[stageC] === per arm: lists, never a single median ===")
    for arm in ("A", "B"):
        sel = [r for r in runs if r["arm"] == arm]
        if not sel:
            continue
        def col(key):
            return [r["metrics"][key] for r in sel]
        print(f"[stageC] {arm} ({args.factor}="
              f"{'1' if arm == 'B' else '0'}) n={len(sel)}")
        for key in ("distance_m", "stop_s", "no_drivable_path_s",
                    "off_pavement_frames", "body_cross_frames",
                    "lat_frames", "damage", "lane_sensor_rate",
                    "lane_drivable_frac_p50", "risk_unknown_frames"):
            vals = col(key)
            num = [v for v in vals if isinstance(v, (int, float))]
            p50 = round(float(statistics.median(num)), 3) if num else None
            print(f"[stageC]   {key:24s} p50={p50} runs={vals}")
        unk = sum((r["metrics"]["unknown_frames"].get("lat_left", 0)
                   for r in sel), 0)
        print(f"[stageC]   lat_left UNKNOWN frames (sum): {unk}")
    print("[stageC] read the LISTS and the coverage, not one median; "
          "UNKNOWN rows are not clean rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
