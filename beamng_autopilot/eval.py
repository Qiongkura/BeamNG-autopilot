"""Offline assessment of FSD-drive telemetry runs.

``m5_fsd_drive.py --out`` writes one JSON list of per-frame snapshots.
This module turns one or many of those runs into the metrics that
matter for driving quality: lane-line / centre-line crossings, off-road
frames, reversing, minimal-risk stops, stalls, speed smoothness and the
final stop position.  Pure and game-free so it can be unit-tested and
reused by ``scripts/m5_fsd_eval.py`` (one-command run + report).
"""

from __future__ import annotations

import math
from typing import Iterable


# --- thresholds -------------------------------------------------------
# lat_left is the signed lateral distance to the LEFT (centre) boundary:
# positive = the ego is inside the oncoming lane (crossed the centre
# line).  lat_right is the signed distance to the RIGHT (road-edge)
# boundary: negative = the ego is off the road edge.
# Small epsilon so numeric noise at spawn (lat_left 0.004-0.007 in
# opt21 frame 0/1) does not count as a crossing; a real crossing puts
# the line under the car body (half-width ~0.9 m), so 0.1 m is still
# a strict "nose past the line" test.
CROSS_CENTRE_M = 0.1
CROSS_RIGHT_M = -0.1
NEAR_LINE_M = 0.25          # "on the line" band (report-only)
OFF_ROAD_M = 0.05           # road_off > 0 means outside a DecalRoad edge
STALL_SPEED_MPS = 0.5
STALL_REM_END_M = 8.0       # only count stalls away from the end zone

# --- benchmark hard targets -------------------------------------------
# The FSD realism bar (README "FSD 结构栈现状"): a benchmark scenario
# passes only with ZERO crossings, ZERO off-road frames, ZERO reversing
# and ZERO stalls; a goal scenario additionally must actually reach the
# goal (same tolerance as the rule autopilot's GOAL_RADIUS_M).
BENCH_MAX_REVERSING_FRAMES = 0
BENCH_MAX_CROSS_CENTRE = 0
BENCH_MAX_CROSS_RIGHT = 0
BENCH_MAX_OFF_ROAD_FRAMES = 0
BENCH_MAX_STALL_FRAMES = 0
BENCH_GOAL_TOL_M = 8.0


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _f(hist, key, i, default=None):
    """Field accessor with None / missing protection (no type filter)."""
    try:
        v = hist[i].get(key, default)
    except Exception:
        return default
    return default if v is None else v


def assess_run(hist: list[dict], goal=None, cruise: float | None = None,
               ) -> dict:
    """Compute driving-quality metrics for one telemetry run."""
    n = len(hist)
    out: dict = {
        "frames": n,
        "duration_s": round(float(_f(hist, "t", n - 1, 0.0) or 0.0), 2),
    }
    if n == 0:
        return out

    # sources / safety levels
    src: dict = {}
    lvl: dict = {}
    for i in range(n):
        s = str(_f(hist, "source", i, "?"))
        src[s] = src.get(s, 0) + 1
        lv = str(_f(hist, "level", i, "?"))
        lvl[lv] = lvl.get(lv, 0) + 1
    out["source"] = src
    out["level"] = lvl
    out["reversing_frames"] = int(sum(
        int(bool(_f(hist, "reversing", i, 0))) for i in range(n)))
    out["stuck_frames"] = int(sum(
        int(bool(_f(hist, "stuck", i, 0))) for i in range(n)))
    out["emergency_frames"] = int(sum(
        int(bool(_f(hist, "emergency", i, 0))) for i in range(n)))
    out["stopps"] = int(sum(
        1 for i in range(n)
        if (_f(hist, "stuck", i, 0) or _f(hist, "emergency", i, 0))))

    # lane discipline
    ll = [_f(hist, "lat_left", i) for i in range(n)]
    lr = [_f(hist, "lat_right", i) for i in range(n)]
    ll_v = [v for v in ll if _num(v)]
    lr_v = [v for v in lr if _num(v)]
    crossed_centre = sum(1 for v in ll_v if v > CROSS_CENTRE_M)
    crossed_right = sum(1 for v in lr_v if v < CROSS_RIGHT_M)
    near_centre = sum(1 for v in ll_v if -NEAR_LINE_M <= v <= NEAR_LINE_M)
    near_right = sum(1 for v in lr_v if -NEAR_LINE_M <= v <= NEAR_LINE_M)
    out["cross_centre_frames"] = crossed_centre
    out["cross_right_frames"] = crossed_right
    out["near_centre_frames"] = near_centre
    out["near_right_frames"] = near_right
    out["max_cross_centre_m"] = round(max(ll_v), 3) if ll_v else 0.0
    out["max_cross_right_m"] = round(min(lr_v), 3) if lr_v else 0.0
    out["lat_frames"] = len(ll_v)

    # off-road
    ro = [_f(hist, "road_off", i) for i in range(n)]
    ro_v = [v for v in ro if _num(v)]
    out["off_road_frames"] = sum(1 for v in ro_v if v > OFF_ROAD_M)
    out["max_road_off_m"] = round(max(ro_v), 3) if ro_v else 0.0

    # speed profile / smoothness
    v = [_f(hist, "speed", i, 0.0) for i in range(n)]
    v_v = [x for x in v if _num(x)]
    if v_v:
        out["speed_min"] = round(min(v_v), 2)
        out["speed_med"] = round(sorted(v_v)[len(v_v) // 2], 2)
        out["speed_max"] = round(max(v_v), 2)
    else:
        out["speed_min"] = out["speed_med"] = out["speed_max"] = 0.0
    if cruise:
        creep = [x for x in v_v if x < 0.3 * float(cruise)]
        out["creep_frac"] = round(len(creep) / len(v_v), 3) if v_v else 0.0

    thr = [int(bool(_f(hist, "throttle", i, 0) and
                     _f(hist, "throttle", i, 0) > 0.02)) for i in range(n)]
    brk = [int(bool(_f(hist, "brake", i, 0) and
                     _f(hist, "brake", i, 0) > 0.02)) for i in range(n)]
    out["throttle_flips"] = sum(1 for i in range(1, n)
                                if thr[i] != thr[i - 1])
    out["brake_flips"] = sum(1 for i in range(1, n)
                             if brk[i] != brk[i - 1])
    out["pedal_opposite_flips"] = sum(
        1 for i in range(1, n)
        if thr[i] and brk[i - 1] and not thr[i - 1])

    # stalls (away from the end-zone stop)
    stalls = 0
    for i in range(n):
        rem = _f(hist, "rem_end", i)
        rem = rem if _num(rem) else None
        if v_v[i] < STALL_SPEED_MPS and (rem is None or rem > STALL_REM_END_M):
            stalls += 1
    out["stall_frames"] = stalls

    # movement / final stop
    dist = 0.0
    for i in range(1, n):
        a = _f(hist, "pos", i - 1)
        b = _f(hist, "pos", i)
        if (isinstance(a, (list, tuple)) and isinstance(b, (list, tuple))
                and len(a) >= 2 and len(b) >= 2):
            dist += math.hypot(b[0] - a[0], b[1] - a[1])
    out["travelled_m"] = round(dist, 1)
    last = hist[-1]
    out["final_pos"] = [round(float(x), 2) for x in last.get("pos", [])[:2]]
    out["final_speed"] = round(float(_f(hist, "speed", n - 1, 0.0) or 0.0), 2)
    out["final_rem_end"] = (_f(hist, "rem_end", n - 1)
                            if _num(_f(hist, "rem_end", n - 1)) else None)
    out["final_lat_left"] = (round(float(ll[-1]), 3)
                             if _num(ll[-1]) else None)
    out["final_lat_right"] = (round(float(lr[-1]), 3)
                              if _num(lr[-1]) else None)
    if goal is not None:
        gx, gy = float(goal[0]), float(goal[1])
        out["goal_dist_m"] = round(math.hypot(
            float(last["pos"][0]) - gx,
            float(last["pos"][1]) - gy), 2)
    return out


def assess_many(runs: Iterable[list[dict]], goal=None,
                cruise: float | None = None) -> list[dict]:
    """Assess several runs; returns a list of result dicts."""
    return [assess_run(h, goal=goal, cruise=cruise) for h in runs]


def score_run(assessed: dict, require_goal: bool = False) -> dict:
    """Verdict on one assessed run against the benchmark hard targets.

    ``assessed`` is an :func:`assess_run` result.  Returns
    ``{"checks": {name: bool}, "pass": bool}`` — ``pass`` is True only
    when every hard target holds (and, for goal scenarios, the goal was
    reached within :data:`BENCH_GOAL_TOL_M`).  A run with no frames can
    never pass.
    """
    checks = {
        "has_frames": int(assessed.get("frames", 0) or 0) > 0,
        "no_reversing": int(assessed.get("reversing_frames", 0) or 0)
        <= BENCH_MAX_REVERSING_FRAMES,
        "no_centre_crossing": int(assessed.get("cross_centre_frames", 0) or 0)
        <= BENCH_MAX_CROSS_CENTRE,
        "no_edge_crossing": int(assessed.get("cross_right_frames", 0) or 0)
        <= BENCH_MAX_CROSS_RIGHT,
        "on_road": int(assessed.get("off_road_frames", 0) or 0)
        <= BENCH_MAX_OFF_ROAD_FRAMES,
        "no_stall": int(assessed.get("stall_frames", 0) or 0)
        <= BENCH_MAX_STALL_FRAMES,
    }
    if require_goal:
        gd = assessed.get("goal_dist_m")
        checks["reached_goal"] = (
            _num(gd) and float(gd) <= BENCH_GOAL_TOL_M)
    return {"checks": checks, "pass": all(bool(v) for v in checks.values())}


def score_many(assessed: list[dict], require_goal: bool = False) -> dict:
    """Aggregate benchmark verdicts over several assessed runs.

    Returns ``{"runs": [verdict...], "pass": bool, "n_pass": int}`` —
    the aggregate passes only when EVERY run passes.
    """
    verdicts = [score_run(a, require_goal=require_goal) for a in assessed]
    n_pass = sum(1 for v in verdicts if v["pass"])
    return {"runs": verdicts, "n_pass": n_pass,
            "pass": bool(verdicts) and n_pass == len(verdicts)}
