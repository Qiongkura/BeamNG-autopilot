"""One-shot debug: replay the last telemetry frame through the planner
and print exactly why desired_speed collapses to 0.
"""
import json
import sys
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from beamng_autopilot.perception import Obstacle
from beamng_autopilot.planner import LocalPlanner, corner_speed


def load_last_frame():
    lines = (PROJ / "logs" / "telemetry" / "live.json").read_text(
        encoding="utf-8").strip().splitlines()
    raw = [json.loads(l) for l in lines if l.strip()]
    return raw[-1]


def main():
    fr = load_last_frame()
    pos = np.array(fr["pos"][:2], dtype=float)
    heading = float(fr["heading"])
    nearest_full = int(fr["nearest"])
    extra = fr["extra"]
    cruise = 15.0  # m/s proxy; actual arg default is checked below

    # Telemetry route is downsampled every `rstep` point; rstep is derived
    # from the real length.  Reconstruct a 2 m-spaced approximation by
    # linear interpolation to match what the live script feeds the planner.
    rte = np.array(extra["rte"], dtype=float)
    step = max(1, round((len(rte) - 1) / 200) * 2)
    # rte holds every `step`th route point -> spacing factor
    seg = np.linalg.norm(np.diff(rte, axis=0), axis=1)
    approx_d = float(np.median(seg))
    n_route = (len(rte) - 1) * step + 1
    route = np.zeros((n_route, 2), dtype=float)
    for i in range(len(rte) - 1):
        a, b = rte[i], rte[i + 1]
        for k in range(step):
            t = k / step
            route[i * step + k] = a + t * (b - a)
    route[-1] = rte[-1]

    obstacles = [
        Obstacle(x=float(o[0]), y=float(o[1]),
                 half_w=float(o[2]), half_h=float(o[3]),
                 category=str(o[4]))
        for o in extra["boxes"]
    ]

    print(f"frame t={fr['t']:.3f} pos={pos.tolist()} heading={heading:.4f}")
    print(f"nearest(full)={nearest_full} n_route={n_route} "
          f"route_d={approx_d:.2f}m/idx  cruise={cruise}")
    print(f"obstacles={[(round(o.x,1), round(o.y,1), o.category) for o in obstacles]}")

    plan = LocalPlanner()
    drive_route, blocked = plan.plan(route, obstacles, pos, heading, nearest_full)
    drive_route = np.asarray(drive_route, dtype=float)
    print(f"plan: mode={plan.last_mode} blocked={blocked} "
          f"n={len(drive_route)} first={drive_route[0].tolist()}")

    if len(drive_route) >= 2:
        d0 = np.linalg.norm(drive_route[:, :2] - pos, axis=1)
        start_i = int(np.argmin(d0))
        if start_i > 0 and len(drive_route) - start_i >= 2:
            drive_route = drive_route[start_i:]
    print(f"trim: n={len(drive_route)} first={drive_route[0].tolist()} "
          f"(car {pos.tolist()})")

    # ---- corner_speed dissection --------------------------------------
    pts = drive_route
    n = len(pts)
    i0 = max(0, 0 - 10)
    i1 = min(n - 1, 0 + 24)
    sub = pts[i0:i1 + 1]
    d = np.diff(sub, axis=0)
    seglen = np.linalg.norm(d, axis=1)
    total_len = float(seglen.sum())
    ang = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    total_da = abs(float(ang[-1] - ang[0]))
    print(f"corner_speed window [{i0},{i1}] len={len(sub)} "
          f"total_len={total_len:.2f}m total_da={total_da:.4f}rad")
    if total_da < 1e-6:
        print("corner_speed -> base (straight)")
    else:
        radius = total_len / total_da
        v = min(cruise, float(np.sqrt(6.5 * radius)))
        print(f"corner_speed radius={radius:.2f}m v={v:.2f} m/s")

    v, obs_d = plan.speed(drive_route, obstacles, pos, heading, 0, cruise)
    print(f"speed() -> v={v:.3f} m/s obs_d={obs_d:.1f}m "
          f"(cruise={cruise}) mode={plan.last_mode}")

    # Steer cap: what steer angle would be needed to cut speed to v?
    # v_cap = sqrt(7 * 2.9/tan(steer_angle)); solve for steer_angle.
    if v < cruise - 1e-9 and v > 0:
        sa = np.arctan(7.0 * 2.9 / (v * v))
        print(f"  (if the steer cap were active, steer_angle would be "
              f"{sa:.3f} rad = {np.degrees(sa):.1f} deg)")


if __name__ == "__main__":
    main()
