"""Phase-1 four-arm A/B: near-field input x reference stability (handoff §3).

The round-5 handoff defines four arms and requires each one to be measured
at least five times, interleaved, before any of them is called better:

    A  front_main only,    stability off   (the 2026-09-20 configuration)
    B  front_fisheye added, stability off   (P1-1 input)
    C  front_main only,    stability on    (P1-2 gate)
    D  both                (the intended combination)

Every arm is one ``m5_fsd_drive`` run from the same recorded start pose on
the same map, with the same seconds/speed and the same pinned switches;
only the two factors move.  Runs go round-robin A,B,C,D,A,B,C,D..., so
session drift (engine warm-up, physics state) is shared instead of landing
on one arm - the failure mode that made a 2026-09-11 read "identical on
every metric" while the game was already closed.

Per run it records the raw telemetry path and the canonical digests
(``m5_run_metrics.digest_from_hist``: stop seconds by cause, lateral field
contract coverage, reference stability, near-field coverage, reject
attribution) plus the manifest switch state, and writes one JSON with
PER-ARM LISTS - medians are printed alongside, never instead.

Usage::
    .venv\\Scripts\\python.exe scripts/m5_phase1_ab.py --attach \\
        --arms 5 --seconds 16 \\
        --teleport 779.5 734.63 -13.2 \\
        --out-dir logs/goal_20260921/phase1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYEXE = ROOT / ".venv" / "Scripts" / "python.exe"

# Files whose change invalidates a run set: a mid-A/B edit must be visible.
WATCH = (
    "beamng_autopilot/fsd_stack.py",
    "beamng_autopilot/fsd_drive.py",
    "beamng_autopilot/safety_monitor.py",
    "beamng_autopilot/occupancy.py",
    "beamng_autopilot/planning/constraints.py",
    "beamng_autopilot/planning/selector.py",
    "beamng_autopilot/lane/stability.py",
    "beamng_autopilot/vision/segmentation.py",
    "beamng_autopilot/vision/lanes.py",
    "beamng_autopilot/planning/lateral_ref.py",
)

# The two factors.  ``None`` means "leave the switch unset" (= module
# default, which is what the recorded 2026-09-20 runs used for A).
ARMS: dict[str, tuple[str | None, str | None]] = {
    "A": (None, None),
    "B": ("fisheye", None),
    "C": (None, "1"),
    "D": ("fisheye", "1"),
}

# Switches pinned for EVERY arm.  EMPTY on purpose: the recorded
# 2026-09-20 baseline (and every n=2 measurement in §4c/§4d) ran with the
# module defaults - their manifests show every switch as None - so pinning
# anything here would introduce a THIRD factor.  A first attempt pinned
# BEAMNG_ASYNC_HEADS=1 and silently changed the runs (head age p50 0.5 s ->
# 1.7 s, every tick stale, both fail-closed gates reading "not checked"),
# which is exactly the confound the handoff forbids.  Factors are only
# BEAMNG_NEARFIELD_CAM and BEAMNG_REF_STABILITY, both below.
PIN: dict[str, str] = {}

# Fixed CLI for every arm, matching the recorded baseline runs: strict FSD
# realism with the PERCEPTION-led lane reference.  The drive CLI defaults to
# ``--lane-mode map``, and a first attempt of this harness inherited that
# default - the arms then ran on the map/nav lane (forbidden as a lateral
# reference here) and matched no previous measurement: ``lane_src`` came out
# ``map_lane`` on half the frames and ``no_drivable_path`` disappeared from
# the telemetry entirely.
BASE_ARGS = ("--lane-mode", "sensor", "--strict")


def _hashes() -> dict[str, str]:
    out = {}
    for rel in WATCH:
        p = ROOT / rel
        out[rel] = (hashlib.md5(p.read_bytes()).hexdigest()[:8]
                    if p.is_file() else "absent")
    return out


def _digest(hist_path: Path) -> dict:
    """Canonical digests from the shared CLI (one implementation only)."""
    proc = subprocess.run(
        [str(PYEXE), "scripts/m5_run_metrics.py", "--hist", str(hist_path)],
        cwd=str(ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        return {"status": "UNKNOWN",
                "reason": (proc.stderr or proc.stdout or "").strip()[-300:]}
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return {"status": "UNKNOWN", "reason": f"bad digest json: {exc}"}


def _summarise(runs: list[dict]) -> dict:
    """Per-arm lists plus the counts the §6 thresholds are stated in."""
    out: dict = {}
    for arm in ARMS:
        sel = [r for r in runs if r["arm"] == arm]
        if not sel:
            continue
        def _lst(key):
            return [r["metrics"].get(key) for r in sel]
        out[arm] = {
            "n": len(sel),
            "hist": [r["hist"] for r in sel],
            "damage": _lst("damage"),
            "distance_m": _lst("distance_m"),
            "stop_s": _lst("stop_s"),
            "stall_s": _lst("stall_s"),
            "no_drivable_path_s": _lst("no_drivable_path_s"),
            "lat_frames": _lst("lat_frames"),
            "body_cross_frames": _lst("body_cross_frames"),
            "off_road_s": _lst("off_road_s"),
            "lane_sensor_rate": _lst("lane_sensor_rate"),
            "nearfield_obs_cells": _lst("nearfield_obs_cells"),
            "ref_side_flips": _lst("ref_side_flips"),
            "body_cov_status": _lst("body_cov_status"),
            "plan_rejects": _lst("plan_rejects"),
        }
        for key in ("stop_s", "stall_s", "distance_m", "no_drivable_path_s"):
            vals = [v for v in out[arm][key] if isinstance(v, (int, float))]
            out[arm][f"{key}_p50"] = (round(statistics.median(vals), 2)
                                      if vals else None)
    return out


def _metrics_from_hist(hist: list[dict]) -> dict:
    """The §6 numbers, read from the telemetry itself."""
    import collections

    n = len(hist)
    if not n:
        return {"frames": 0}
    t = [float(f.get("t") or 0.0) for f in hist]

    def _span(idx, *, rule: str = "stop") -> float:
        """Duration owned by the flagged frames, using the CANONICAL rule.

        ``eval.STOP_INTERVAL_RULE``: each flagged frame owns the interval to
        the NEXT FRAME OF THE RUN (the last one owns none).  The previous
        version summed ``t[next_flagged] - t[flagged]``, which connects two
        stops across the driving between them and reports the driving time
        as stopping - a 6 s stop followed by 20 s of driving followed by a
        6 s stop read as 32 s of stopping (plan T12).
        """
        tot = 0.0
        for i in idx:
            if i + 1 < n:
                tot += max(0.0, t[i + 1] - t[i])
        return round(tot, 2)

    stops = [i for i in range(n)
             if isinstance(hist[i].get("speed"), (int, float))
             and float(hist[i]["speed"]) < 0.3]
    nodrv = [i for i in stops
             if str(hist[i].get("reason", "")).startswith("no drivable path")]
    lat_cov = [f for f in hist if f.get("line_lat") is not None]
    body_cross = [f for f in hist
                  if f.get("body_cross_l") or f.get("body_cross_r")]
    off_any = [f for f in hist
               if (f.get("road_off") or 0.0) > 1e-9
               or (f.get("edge_over") or 0.0) > 1e-9
               or str(f.get("body_cov_status") or "") == "off_road"]
    dmg = [f.get("damage_total") for f in hist
           if isinstance(f.get("damage_total"), (int, float))]
    dist = 0.0
    for a, b in zip(hist, hist[1:]):
        pa, pb = a.get("pos"), b.get("pos")
        if isinstance(pa, list) and isinstance(pb, list) and len(pa) > 1:
            dist += ((pb[0] - pa[0]) ** 2 + (pb[1] - pa[1]) ** 2) ** 0.5
    nf = collections.Counter()
    for f in hist:
        cov = f.get("nearfield_cov")
        if isinstance(cov, dict):
            nf["obs"] += int(cov.get("observed_cells") or 0)
            nf["n"] += 1
    lane = [str(f.get("lane_src") or "") for f in hist]
    # Comparability checks: an arm whose ticks were all stale, or whose
    # fail-closed gates never ran, is not the same experiment as one whose
    # gates measured - the first n=2 set and the first phase-1 attempt
    # differed exactly here (head age p50 0.5 s vs 1.7 s).
    head_age = [float(f["freshness"]["head_max_s"]) for f in hist
                if isinstance(f.get("freshness"), dict)
                and isinstance(f["freshness"].get("head_max_s"),
                               (int, float))]
    stale_frames = sum(1 for f in hist
                       if str(f.get("reason") or "") == "stale sensor")
    body_checked = sum(1 for f in hist if f.get("body_cov_checked"))
    rej = collections.Counter()
    for f in hist:
        r = f.get("plan_rejects")
        if isinstance(r, dict):
            for k, v in r.items():
                rej[k] += int(v)
    return {
        "frames": n,
        "duration_s": round(t[-1] - t[0], 2) if n > 1 else 0.0,
        "distance_m": round(dist, 2),
        "stop_frames": len(stops),
        "stop_s": _span(stops),
        "no_drivable_path_s": _span(nodrv),
        "lat_frames": len(lat_cov),
        "lat_cov_frac": round(len(lat_cov) / n, 3),
        "body_cross_frames": len(body_cross),
        "off_pavement_frames": len(off_any),
        "off_road_s": _span([i for i in range(n)
                             if hist[i] in off_any]),
        "damage": round(max(dmg), 3) if dmg else None,
        "lane_sensor_rate": round(
            sum(1 for s in lane if s == "sensor") / n, 3),
        "nearfield_obs_cells": (round(nf["obs"] / nf["n"], 1)
                                if nf["n"] else None),
        "ref_side_flips": sum(1 for f in hist if f.get("ref_flip")),
        "body_cov_status": collections.Counter(
            str(f.get("body_cov_status") or "")
            for f in hist).most_common(1)[0][0],
        "body_cov_checked_frac": round(body_checked / n, 3),
        "stale_frames": stale_frames,
        "head_max_s_p50": (round(statistics.median(head_age), 3)
                           if head_age else None),
        "plan_rejects": dict(rej.most_common(3)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="phase-1 four-arm A/B")
    ap.add_argument("--arms", type=int, default=5,
                    help="runs PER ARM (handoff requires >= 5)")
    ap.add_argument("--attach", action="store_true")
    ap.add_argument("--seconds", type=float, default=16.0)
    ap.add_argument("--speed", type=float, default=6.0)
    ap.add_argument("--teleport", nargs=3, type=float, default=None,
                    metavar=("X", "Y", "YAW_DEG"))
    ap.add_argument("--order", default="ABCD",
                    help="arm order inside one round (default ABCD)")
    # The plan requires the run record to name the FULL configuration, not
    # just the two factors: goal, map and model are passed through to the
    # drive entry point and recorded per run (plan T12/§8.2).
    ap.add_argument("--goal", nargs=2, type=float, default=None,
                    metavar=("X", "Y"))
    ap.add_argument("--map", type=str, default=None)
    ap.add_argument("--seg-model", type=str, default=None)
    ap.add_argument("--out-dir", type=str,
                    default="logs/goal_20260921/phase1")
    args = ap.parse_args()

    order = [a for a in args.order.upper() if a in ARMS]
    if sorted(order) != sorted(ARMS):
        print(f"[p1ab] --order must contain each of {sorted(ARMS)} exactly "
              f"once")
        return 2
    if args.arms < 5:
        print("[p1ab] !! handoff §6 requires >= 5 runs per arm; "
              f"--arms {args.arms} cannot support a claim")
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    base = _hashes()
    print(f"[p1ab] arms={order} runs/arm={args.arms} seconds={args.seconds} "
          f"speed={args.speed} teleport={args.teleport}")
    print(f"[p1ab] pinned={PIN}")
    print(f"[p1ab] code={base}")

    runs: list[dict] = []
    for i in range(args.arms):
        for arm in order:
            nf, stab = ARMS[arm]
            env = dict(os.environ)
            for name, val in PIN.items():
                env[name] = val
            env.pop("BEAMNG_NEARFIELD_CAM", None)
            env.pop("BEAMNG_REF_STABILITY", None)
            env.pop("BEAMNG_BODY_COVERAGE_GATE", None)
            if nf is not None:
                env["BEAMNG_NEARFIELD_CAM"] = nf
            if stab is not None:
                env["BEAMNG_REF_STABILITY"] = stab
            tag = f"{arm}{i}"
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
            t0 = time.time()
            proc = subprocess.run(cmd, cwd=str(ROOT), env=env,
                                  capture_output=True, text=True)
            changed = _hashes() != base
            base = _hashes()
            if changed:
                print("[p1ab] !! watched code changed during this arm - "
                      "run set invalidated")
            if not hist_path.is_file():
                tail = "\n".join((proc.stderr or proc.stdout or "")
                                 .strip().splitlines()[-3:])
                print(f"[p1ab] {tag}: NO TELEMETRY "
                      f"({time.time() - t0:.0f}s) - {tail}")
                continue
            hist = json.loads(hist_path.read_text(encoding="utf-8"))
            metrics = _metrics_from_hist(hist)
            digest = _digest(hist_path)
            rec = {"arm": arm, "round": i, "nearfield": nf,
                   "ref_stability": stab, "hist": str(hist_path),
                   "cmd": " ".join(cmd),
                   "env_overrides": {k: v for k, v in env.items()
                                     if k.startswith("BEAMNG_")},
                   "seconds": round(time.time() - t0, 1),
                   "metrics": metrics,
                   "stop_digest": list(digest.values())[0].get("stop_digest")
                   if digest else None}
            runs.append(rec)
            print(f"[p1ab] {tag} nf={nf or 'off':8s} stab={stab or 'off':3s} "
                  f"dist={metrics['distance_m']:6.2f}m "
                  f"stop={metrics['stop_s']:5.2f}s "
                  f"nodrv={metrics['no_drivable_path_s']:5.2f}s "
                  f"lat={metrics['lat_frames']}/{metrics['frames']} "
                  f"bodycross={metrics['body_cross_frames']} "
                  f"offpav={metrics['off_pavement_frames']} "
                  f"dmg={metrics['damage']} "
                  f"nfobs={metrics['nearfield_obs_cells']} "
                  f"flips={metrics['ref_side_flips']} "
                  f"stale={metrics['stale_frames']} "
                  f"headage={metrics['head_max_s_p50']} "
                  f"bodychk={metrics['body_cov_checked_frac']} "
                  f"({rec['seconds']}s)")
        # per-round checkpoint, so a crash mid-set does not lose the set
        (out_dir / "phase1_summary.json").write_text(
            json.dumps({"arms": ARMS, "pin": PIN, "order": order,
                        "base_args": list(BASE_ARGS),
                        "runs_requested": args.arms, "runs": runs,
                        "summary": _summarise(runs)},
                       indent=2, ensure_ascii=False), encoding="utf-8")

    if not runs:
        print("[p1ab] no arm produced telemetry - is BeamNG.tech running?")
        return 2
    summary = _summarise(runs)
    print("\n[p1ab] === per arm (lists, not just medians) ===")
    for arm, s in summary.items():
        print(f"[p1ab] {arm} n={s['n']} dist={s['distance_m']} "
              f"stop_s={s['stop_s']} nodrv_s={s['no_drivable_path_s']}")
        print(f"[p1ab]     lat_frames={s['lat_frames']} "
              f"body_cross={s['body_cross_frames']} "
              f"off_road_s={s['off_road_s']} damage={s['damage']}")
        print(f"[p1ab]     lane_sensor={s['lane_sensor_rate']} "
              f"nf_obs={s['nearfield_obs_cells']} "
              f"flips={s['ref_side_flips']} "
              f"body_cov={s['body_cov_status']}")
    print("[p1ab] crossing / off-pavement / damage are per-arm LISTS: "
          "compare the lists.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
