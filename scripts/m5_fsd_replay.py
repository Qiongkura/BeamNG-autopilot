# Offline replay of the FSD layered planner over recorded shadow episodes.
# Loads the recorded vector-space snapshot (BEV occupancy + drivable mask,
# ego pose per frame) from shadow .npz episodes, re-runs the planner's
# candidate fan + selector for every frame, and reports how often a
# feasible trajectory exists and whether the replay agrees with what the
# recorder actually chose.  No game needed - planning-layer companion to
# m5_e2e_probe.py (action-level replay).
#
# Usage:
#   .venv\Scripts\python.exe scripts\m5_fsd_replay.py --data logs/live_runs/shadow_episodes

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beamng_autopilot.occupancy import OccupancyGrid
from beamng_autopilot.planning import (
    Constraints,
    Scene,
    sample_arc,
    select_trajectory,
)

GRID_N = 60
GRID_RES = 0.5


def _replay_frame(x, y, heading, speed, bev, drivable, target_speed):
    # Return (best_path, meta) for one recorded frame.
    pos = np.array([float(x), float(y)], dtype=float)
    grid = OccupancyGrid(GRID_N, GRID_N, GRID_RES,
                         origin=(float(x), float(y)),
                         heading=float(heading))
    grid.occupancy = np.asarray(bev, dtype=np.float32).copy()
    grid.obstacle = (np.asarray(bev, dtype=np.float32) > 0.5).astype(np.uint8)
    if drivable is not None:
        grid.drivable = np.asarray(drivable, dtype=np.float32).copy()
    scene = Scene(pos=pos, heading=float(heading), grid=grid,
                  route=None, lane_ref=None,
                  target_speed=float(target_speed))
    fans = sample_arc(pos, float(heading),
                      speed=max(2.0, float(speed)),
                      max_steer=0.5, n_curv=13, max_curv=0.25)
    return select_trajectory(scene, fans, Constraints())


def _evaluate_episode(ep: Path) -> dict:
    with np.load(ep, allow_pickle=True) as z:
        n = int(z["t"].shape[0])
        x = np.asarray(z["x"], dtype=np.float64)
        y = np.asarray(z["y"], dtype=np.float64)
        hdg = np.asarray(z["heading"], dtype=np.float64)
        spd = np.asarray(z["speed"], dtype=np.float64)
        bev = np.asarray(z["bev"], dtype=np.float32)
        drv = np.asarray(z["drivable"], dtype=np.uint8) \
            if "drivable" in z else None
        traj_ok = np.asarray(z["trajectory_ok"], dtype=bool) \
            if "trajectory_ok" in z else np.ones(n, dtype=bool)
        kind = np.asarray(z["kind"], dtype=object) if "kind" in z else None
        tgt = np.asarray(z["target_speed"], dtype=np.float64) \
            if "target_speed" in z else np.full(n, 6.0)
        plan_ok = 0
        agree = 0
        kind_match = 0
        no_cand = 0
        repl_costs: list[float] = []
        rec_costs: list[float] = []
        for i in range(n):
            best, meta = _replay_frame(
                float(x[i]), float(y[i]), float(hdg[i]), float(spd[i]),
                bev[i], None if drv is None else drv[i], float(tgt[i]))
            ok = best is not None and len(best) >= 2
            plan_ok += int(ok)
            agree += int(ok == bool(traj_ok[i]))
            if kind is not None and ok and str(kind[i]) == \
                    str(meta.get("kind", "")):
                kind_match += 1
            if not ok:
                no_cand += 1
            if ok and meta.get("cost") is not None:
                repl_costs.append(float(meta["cost"]))
                if "cost" in z:
                    rec_costs.append(float(z["cost"][i]))
    n = max(1, n)
    return {
        "episode": ep.name,
        "frames": int(n),
        "plan_ok_rate": plan_ok / n,
        "agreement_rate": agree / n,
        "kind_match_rate": kind_match / max(1, plan_ok),
        "no_candidate_rate": no_cand / n,
        "replay_cost_mean": float(np.mean(repl_costs)) if repl_costs else None,
        "recorded_cost_mean": float(np.mean(rec_costs)) if rec_costs else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="FSD planner offline replay")
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--report", type=str,
                    default="logs/m5_e2e/fsd_replay_report.json")
    args = ap.parse_args()
    data = Path(args.data)
    eps = sorted(data.glob("*.npz"), key=lambda p: p.stat().st_mtime)
    if not eps:
        print(f"[fsd-replay] no episodes under {data}")
        return 1
    per = []
    t0 = time.time()
    for ep in eps:
        r = _evaluate_episode(ep)
        per.append(r)
        print(f"[fsd-replay] {r['episode']}: n={r['frames']} "
              f"plan_ok={r['plan_ok_rate'] * 100:.0f}% "
              f"agree={r['agreement_rate'] * 100:.0f}% "
              f"kind_match={r['kind_match_rate'] * 100:.0f}%")
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "aggregate": {
            "episodes": len(per),
            "frames": int(sum(r["frames"] for r in per)),
            "plan_ok_rate": float(np.mean([r["plan_ok_rate"]
                                           for r in per])),
            "agreement_rate": float(np.mean([r["agreement_rate"]
                                             for r in per])),
            "kind_match_rate": float(np.mean(
                [r["kind_match_rate"] for r in per])),
        },
        "episodes": per,
    }
    rp = Path(args.report)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                  encoding="utf-8")
    a = report["aggregate"]
    print(f"[fsd-replay] {a['episodes']} eps / {a['frames']} frames: "
          f"plan_ok={a['plan_ok_rate'] * 100:.1f}% "
          f"agree={a['agreement_rate'] * 100:.1f}% "
          f"kind_match={a['kind_match_rate'] * 100:.1f}% "
          f"({time.time() - t0:.0f}s) -> {rp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
