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

from beamng_autopilot.fsd_realism import SRC_SENSOR


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
# Off-pavement verdict from the ROUTE distance (the honest metric when
# the run carries route_dist): half of a ~7 m two-lane road plus a small
# tolerance - beyond this the car is on the shoulder, whatever the
# perception boundaries claim.
ROAD_HALF_WIDTH_M = 3.0
# Metres beyond the road edge polyline before the run counts as
# off-pavement (small tolerance for mask/GPS-level noise in the edges).
EDGE_OVER_M = 0.3
STALL_SPEED_MPS = 0.5
STALL_REM_END_M = 8.0       # only count stalls away from the end zone

# --- canonical stop / creep definition (review handoff P0-1) -----------
# One definition, used by every report and script, because the same run
# could previously be quoted as "57 stops", "53 stops" or "62 stops"
# depending on whether the reader counted speed < 0.3, speed <= 0.1,
# final_stop, emergency or "reason == no drivable path".  Everything a
# report quotes must come from :func:`stop_digest`, which states its own
# spec and returns ``None`` (UNKNOWN) for any column the run lacks.
STOP_SPEED_MPS = 0.3        # a "stop" frame: speed < this
CREEP_SPEED_MPS = 1.0       # "creep" band: STOP_SPEED_MPS <= speed < this
STOP_INTERVAL_RULE = (
    "each settled frame owns the interval to the next FRAME of the run; "
    "the last settled frame owns none")
STOP_HARD_RULE = ("final_stop flag, or emergency == 1, "
                  "or level == 'minimal_risk'")

# --- benchmark hard targets -------------------------------------------
# The FSD realism bar (README "FSD 结构栈现状"): a benchmark scenario
# passes only with ZERO crossings, ZERO off-road frames, ZERO reversing
# and ZERO stalls; a goal scenario additionally must actually reach the
# goal (same tolerance as the rule autopilot's GOAL_RADIUS_M).
# ego body footprint halves (etk800): used for BODY-aware line-crossing
# detection - a yawed car crosses the line with its body while the ego
# CENTRE point still reads in-lane (fsd_benchmark town 2026-09-06: the
# user photographed left wheels ON the line while crossC reported 0;
# body-left was over the line by 0.14-0.22 m for 3 frames).
BODY_HALF_W_M = 0.9
BODY_HALF_L_M = 2.2
BENCH_MAX_REVERSING_FRAMES = 0
BENCH_MAX_CROSS_CENTRE = 0
BENCH_MAX_CROSS_RIGHT = 0
BENCH_MAX_OFF_ROAD_FRAMES = 0
BENCH_MAX_STALL_FRAMES = 0
BENCH_GOAL_TOL_M = 8.0
BENCH_MAX_COLLISIONS = 0

# Three-state run verdict (plan P0-3).  ``UNKNOWN`` is a first-class
# outcome, not a soft FAIL: a run whose collision channel was never
# sampled must not read as "no collisions" (the §12 gate is
# ``collision_count = 0``, and a missing measurement cannot satisfy it),
# while a run that DID measure and saw nothing is a genuine PASS.
STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_UNKNOWN = "UNKNOWN"


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _f(hist, key, i, default=None):
    """Field accessor with None / missing protection (no type filter)."""
    try:
        v = hist[i].get(key, default)
    except Exception:
        return default
    return default if v is None else v


def _episodes(flags: list[bool],
              t: list[float]) -> tuple[float, float, int]:
    """(total_s, longest_s, episode_count) for a boolean mask.

    Each frame owns the interval to the NEXT frame; the LAST frame owns
    nothing, because the sampled window is ``[t[0], t[-1]]`` and no
    evidence exists past the final sample.  A mask covering the whole run
    therefore sums to exactly the run's sampled duration (rather than one
    frame short, or one frame long as a "reuse the previous gap" rule
    would make it).

    Frame COUNTS cannot be compared across runs of different cadence -
    60 frames is 36 s at 1.65 Hz and 6 s at 10 Hz - which is why the plan
    asks for duration, not counts.
    """
    n = len(flags)
    if n == 0:
        return 0.0, 0.0, 0
    dts = [(float(t[i + 1]) - float(t[i])) if i + 1 < n else 0.0
           for i in range(n)]
    total = 0.0
    longest = 0.0
    cur = 0.0
    episodes = 0
    inside = False
    for i in range(n):
        if flags[i]:
            if not inside:
                episodes += 1
                inside = True
                cur = 0.0
            step = max(0.0, dts[i])
            cur += step
            total += step
            if cur > longest:
                longest = cur
        else:
            inside = False
            cur = 0.0
    return total, longest, episodes


def stop_digest(hist: list[dict], *, settle_s: float = 0.0,
                end_zone_m: float = STALL_REM_END_M) -> dict:
    """One canonical stop/creep digest for a run (review handoff P0-1).

    Every number a report may quote about stopping comes from here, and
    the returned ``spec`` states exactly what each number means:

    * ``settle_s`` - frames with ``t < settle_s`` are excluded from every
      count (the post-teleport settling window); ``raw_frames`` still
      counts them;
    * ``interval_rule`` - durations use :data:`STOP_INTERVAL_RULE`;
    * ``stop`` / ``creep`` - :data:`STOP_SPEED_MPS` / :data:`CREEP_SPEED_MPS`;
    * ``hard_stop`` - :data:`STOP_HARD_RULE`, reported separately from the
      speed-based stop so "commanded stop" and "slow" are never conflated;
    * ``excl_end_zone`` - the same stop numbers with frames whose
      ``rem_end <= end_zone_m`` removed (a stop at the goal is intended);
    * ``by_reason`` - seconds/frames/longest per ``effective_rule`` (or
      ``reason`` when that is empty), which is what "classify the stall by
      cause" needs;
    * ``unknown`` - a field whose source column is missing from every
      settled frame is ``None`` and listed here.  It is never 0: a run
      without a ``speed`` column has NOT measured "no stopping".
    """
    n = len(hist)
    t = [_f(hist, "t", i, 0.0) or 0.0 for i in range(n)]
    settled = [i for i in range(n) if float(t[i]) >= float(settle_s)]
    unknown: list[str] = []

    def _col(key: str, kind: str = "num"):
        """Per-settled-frame values, or None when the column is absent.

        ``kind`` matters: a flag column read as numeric turns every
        ``True`` into "missing" (bools are not numbers here), and a text
        column read as numeric turns every reason into "missing" - both
        silently produced 0 counts in the first version of this function,
        which is exactly the failure this digest exists to remove.
        """
        raw = [_f(hist, key, i) for i in settled]
        if not raw:
            unknown.append(key)
            return None
        vals: list = []
        for v in raw:
            if kind == "flag":
                if isinstance(v, bool):
                    vals.append(bool(v))
                elif _num(v) and float(v) in (0.0, 1.0):
                    vals.append(bool(v))
                else:
                    vals.append(None)
            elif kind == "str":
                vals.append(str(v) if isinstance(v, str) and v.strip()
                            else None)
            else:
                vals.append(v if _num(v) else None)
        if all(v is None for v in vals):
            unknown.append(key)
            return None
        return vals

    def _block(mask, idx) -> dict:
        total, longest, eps = _episodes(mask, [t[i] for i in idx])
        return {"frames": int(sum(1 for m in mask if m)),
                "s": round(total, 2), "longest_s": round(longest, 2),
                "episodes": int(eps)}

    out: dict = {
        "spec": {
            "settle_s": float(settle_s),
            "settle_rule": "frames with t < settle_s are excluded from "
                           "every count; raw_frames keeps them",
            "interval_rule": STOP_INTERVAL_RULE,
            "stop": f"speed < {STOP_SPEED_MPS} m/s",
            "creep": f"{STOP_SPEED_MPS} <= speed < {CREEP_SPEED_MPS} m/s",
            "hard_stop": STOP_HARD_RULE,
            "by_reason": "effective_rule when non-empty, else reason",
            "by_reason_fields": (
                "frames = settled frames carrying the reason; s/active_s = "
                "seconds the reason was ACTIVE (not necessarily stopped); "
                "stop_frames/stop_s = the subset that was also a stop; "
                "longest_s = longest continuous stretch"),
            "unknown_rule": "a column absent from every settled frame "
                            "yields None and is listed in unknown[]",
        },
        "raw_frames": n,
        "settled_frames": len(settled),
    }

    speed = _col("speed")
    for thr, name in ((0.3, "speed_lt_0_3_frames"),
                      (0.1, "speed_lt_0_1_frames")):
        if speed is None:
            out[name] = None
        else:
            out[name] = int(sum(1 for v in speed if v is not None and v < thr))
    if speed is None:
        out.update({"stop_frames": None, "stop_s": None,
                    "stop_longest_s": None, "stop_episodes": None,
                    "creep_frames": None, "creep_s": None})
    else:
        out.update(_block([v is not None and v < STOP_SPEED_MPS
                           for v in speed], settled))
        creep = [v is not None and STOP_SPEED_MPS <= v < CREEP_SPEED_MPS
                 for v in speed]
        _c = _block(creep, settled)
        out.update({"creep_frames": _c["frames"], "creep_s": _c["s"],
                    "stop_frames": out["frames"], "stop_s": out["s"],
                    "stop_longest_s": out["longest_s"],
                    "stop_episodes": out["episodes"]})
        out.pop("frames", None)
        out.pop("s", None)
        out.pop("longest_s", None)
        out.pop("episodes", None)

    fs = _col("final_stop", "flag")
    em = _col("emergency", "flag")
    lv = _col("level", "str")
    hard = None
    if fs is not None or em is not None or lv is not None:
        hard = []
        for k in range(len(settled)):
            hit = bool((fs is not None and fs[k]) or (em is not None and em[k])
                       or (lv is not None and str(lv[k]) == "minimal_risk"))
            hard.append(hit)
    if hard is None:
        out.update({"final_stop_frames": None, "final_stop_s": None,
                    "emergency_frames": None, "emergency_s": None,
                    "hard_stop_frames": None, "hard_stop_s": None})
    else:
        _h = _block(hard, settled)
        out.update({"hard_stop_frames": _h["frames"],
                    "hard_stop_s": _h["s"]})
        if fs is None:
            out.update({"final_stop_frames": None, "final_stop_s": None})
        else:
            _f_ = _block([bool(v) for v in fs], settled)
            out.update({"final_stop_frames": _f_["frames"],
                        "final_stop_s": _f_["s"]})
        if em is None:
            out.update({"emergency_frames": None, "emergency_s": None})
        else:
            _e = _block([bool(v) for v in em], settled)
            out.update({"emergency_frames": _e["frames"],
                        "emergency_s": _e["s"]})

    rem = _col("rem_end")
    if speed is None or rem is None:
        out["excl_end_zone"] = None
    else:
        mask = [v is not None and v < STOP_SPEED_MPS
                and (rem[k] is None or float(rem[k]) > float(end_zone_m))
                for k, v in enumerate(speed)]
        out["excl_end_zone"] = dict(
            _block(mask, settled), end_zone_m=float(end_zone_m))

    reason_keys = ("effective_rule", "reason")
    cols = {k: _col(k, "str") for k in reason_keys}
    if cols["effective_rule"] is None and cols["reason"] is None:
        out["by_reason"] = None
    else:
        by_reason: dict = {}
        stop_mask = ([v is not None and v < STOP_SPEED_MPS for v in speed]
                     if speed is not None else None)
        _rules: list[str] = []
        for k, idx in enumerate(settled):
            rule = ""
            for key in reason_keys:
                if cols[key] is not None:
                    val = cols[key][k]
                    if val is not None and str(val).strip():
                        rule = str(val)
                        break
            _rules.append(rule)
            if not rule:
                continue
            slot = by_reason.setdefault(rule, {"frames": 0, "s": 0.0})
            slot["frames"] += 1
            # ``s`` is how long this reason was ACTIVE, which is not how
            # long the car was stopped for it: a rule can be in force while
            # the car is creeping or even driving.  Both numbers are kept
            # and named, because reading ``s`` as "seconds lost to this
            # rule" overstates the stall (measured 2026-09-21: 25.89 s
            # active vs 22.92 s actually stopped on the same run).
            slot["active_s"] = slot.get("active_s", 0.0)
            if k + 1 < len(settled):
                _dt_reason = max(0.0, float(t[settled[k + 1]]) - float(t[idx]))
                slot["s"] += _dt_reason
                slot["active_s"] += _dt_reason
            if stop_mask is not None and stop_mask[k]:
                slot["stop_frames"] = slot.get("stop_frames", 0) + 1
                slot.setdefault("stop_s", 0.0)
                if k + 1 < len(settled):
                    slot["stop_s"] += _dt_reason        # Longest CONTINUOUS stretch per reason, and (from the same walk)
        # the longest continuous stop for each: one 20 s stall and twenty
        # 1 s stalls are the same total and different driving failures.
        for reason in by_reason:
            _best = 0.0
            _best_stop = 0.0
            _run = 0.0
            _run_stop = 0.0
            for k in range(len(settled)):
                if _rules[k] != reason:
                    _run = 0.0
                    _run_stop = 0.0
                    continue
                step = 0.0
                if k + 1 < len(settled):
                    step = max(0.0, float(t[settled[k + 1]])
                               - float(t[settled[k]]))
                _run += step
                _best = max(_best, _run)
                if stop_mask is not None and stop_mask[k]:
                    _run_stop += step
                    _best_stop = max(_best_stop, _run_stop)
                else:
                    _run_stop = 0.0
            by_reason[reason]["longest_s"] = round(_best, 2)
            by_reason[reason]["stop_longest_s"] = (
                round(_best_stop, 2) if stop_mask is not None else None)
        for slot in by_reason.values():
            slot["s"] = round(slot["s"], 2)
            slot["active_s"] = round(slot["active_s"], 2)
            # 0.0 when the stop channel EXISTS and this reason never
            # coincided with a stop ("measured: it caused no stopping");
            # None only when the run has no stop measurement at all.  The
            # two are different statements and this project has already
            # been burned by collapsing them.
            if stop_mask is not None:
                slot["stop_s"] = round(float(slot.get("stop_s") or 0.0), 2)
                slot["stop_frames"] = int(slot.get("stop_frames", 0))
            else:
                slot["stop_s"] = None
                slot["stop_frames"] = None
        out["by_reason"] = by_reason

    if lv is not None:
        by_level: dict = {}
        for k, idx in enumerate(settled):
            key = str(lv[k]) if lv[k] is not None else "?"
            slot = by_level.setdefault(key, {"frames": 0, "s": 0.0})
            slot["frames"] += 1
            if k + 1 < len(settled):
                slot["s"] += max(0.0, float(t[settled[k + 1]]) - float(t[idx]))
        for slot in by_level.values():
            slot["s"] = round(slot["s"], 2)
        out["by_level"] = by_level
    else:
        out["by_level"] = None

    out["unknown"] = sorted(set(unknown))
    return out


def assess_run(hist: list[dict], goal=None, cruise: float | None = None,
               settle_s: float = 0.0) -> dict:
    """Compute driving-quality metrics for one telemetry run.

    ``settle_s`` excludes the first ``settle_s`` seconds from the
    DISCIPLINE counts (crossings / off-road / reversing / stalls).  A
    teleport spawn needs a couple of seconds to settle - the semantic
    head warms up and the perception placement converges onto the own
    lane - so the very first frames read as boundary noise, not driving
    (fsd_benchmark mountain run 4: lat_right -0.26 m at t=2.4 with the
    car ON the road, gone by t=3.5).  Default 0 keeps the historical
    (no-settle) numbers; the benchmark passes 3.0.  All other metrics
    (frames, duration, speed, travel) always cover the whole run.
    """
    n = len(hist)
    out: dict = {
        "frames": n,
        "duration_s": round(float(_f(hist, "t", n - 1, 0.0) or 0.0), 2),
    }
    if n == 0:
        return out
    t = [_f(hist, "t", i, 0.0) or 0.0 for i in range(n)]
    settled = [i for i in range(n) if float(t[i]) >= float(settle_s)]
    ns = len(settled)
    out["settled_frames"] = ns

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

    # perception-lane continuity: the leading indicator behind every
    # discipline metric below.  A strict run gets a lane reference only
    # while perception supplies one (``lane_sel``); once the pairing
    # drops the stack fails closed and creeps, so the fraction of frames
    # with a perception lane predicts stalls and progress long before
    # they show up as speed (town 2026-09-07: lane_sel=sensor in 24 of
    # 217 frames, median speed 0.57 m/s).  ``lane_paired`` is the
    # stricter subset where BOTH lane sides were real detections.
    lane_hist: dict = {}
    lane_sensor = 0
    lane_paired = 0
    for i in settled:
        sel = _f(hist, "lane_sel", i)
        if sel in (None, ""):
            sel = _f(hist, "lane_src", i)
        sel = str(sel) if sel not in (None, "") else "?"
        lane_hist[sel] = lane_hist.get(sel, 0) + 1
        if sel == SRC_SENSOR:
            lane_sensor += 1
        if int(bool(_f(hist, "lane_paired", i, 0))):
            lane_paired += 1
    out["lane_src_hist"] = lane_hist
    out["lane_sensor_frames"] = lane_sensor
    out["lane_sensor_rate"] = round(lane_sensor / ns, 3) if ns else 0.0
    out["lane_paired_frames"] = lane_paired
    out["lane_paired_rate"] = round(lane_paired / ns, 3) if ns else 0.0
    out["reversing_frames"] = int(sum(
        int(bool(_f(hist, "reversing", i, 0))) for i in settled))
    out["stuck_frames"] = int(sum(
        int(bool(_f(hist, "stuck", i, 0))) for i in settled))
    out["emergency_frames"] = int(sum(
        int(bool(_f(hist, "emergency", i, 0))) for i in settled))
    out["stopps"] = int(sum(
        1 for i in settled
        if (_f(hist, "stuck", i, 0) or _f(hist, "emergency", i, 0))))

    # lane discipline
    ll = [_f(hist, "lat_left", i) for i in settled]
    lr = [_f(hist, "lat_right", i) for i in settled]
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
    # A run with no lateral samples at all is UNMEASURED, not "perfectly
    # centred": the old 0.0 fallback read as a clean run (plan P0-5).
    out["max_cross_centre_m"] = round(max(ll_v), 3) if ll_v else None
    out["max_cross_right_m"] = round(min(lr_v), 3) if lr_v else None
    out["lat_frames"] = len(ll_v)
    # Per-side coverage (plan T01): the two crossing checks are separate
    # measurements and one side can be missing while the other is fine.
    # A frame with only ``lat_right`` published cannot say anything about
    # the centre/oncoming side, and the old single counter let that frame
    # read as "no crossing" on both sides.
    out["lat_left_frames"] = len(ll_v)
    out["lat_right_frames"] = len(lr_v)
    out["lat_left_coverage"] = (round(len(ll_v) / len(settled), 3)
                                if settled else None)
    out["lat_right_coverage"] = (round(len(lr_v) / len(settled), 3)
                                 if settled else None)

    # BODY-aware crossing: the car's yawed footprint extends its lateral
    # reach by half_w*|cos(dyaw)| + half_len*|sin(dyaw)|; the centre
    # point alone is blind to a yawed car crossing with its body.
    hd = [_f(hist, "heading", i) for i in settled]
    rb = [_f(hist, "route_bear", i) for i in settled]
    body_cross_centre = 0
    body_cross_right = 0
    max_body_left = None
    max_body_right = None
    for k, i in enumerate(settled):
        if not (_num(hd[k]) and _num(rb[k])):
            continue
        dyaw = math.radians((float(hd[k]) - float(rb[k]) + 180.0)
                            % 360.0 - 180.0)
        ext = (BODY_HALF_W_M * abs(math.cos(dyaw))
               + BODY_HALF_L_M * abs(math.sin(dyaw)))
        if _num(ll[k]):
            body_l = float(ll[k]) + ext
            max_body_left = (max(max_body_left, body_l)
                             if max_body_left is not None else body_l)
            if body_l > CROSS_CENTRE_M:
                body_cross_centre += 1
        if _num(lr[k]):
            body_r = float(lr[k]) - ext
            max_body_right = (min(max_body_right, body_r)
                              if max_body_right is not None else body_r)
            if body_r < CROSS_RIGHT_M:
                body_cross_right += 1
    out["body_cross_centre_frames"] = body_cross_centre
    out["body_cross_right_frames"] = body_cross_right
    out["max_body_left_m"] = (round(max_body_left, 3)
                              if max_body_left is not None else None)
    out["max_body_right_m"] = (round(max_body_right, 3)
                               if max_body_right is not None else None)

    # off-road.  Two sources, and they are NOT equivalent: ``road_off``
    # is PERCEPTION-based (distance past a detected boundary; blind in
    # strict mode when no boundary is published) while ``route_dist`` is
    # the distance from the nav-route road CENTRELINE - the honest
    # off-pavement metric.  When the run carries route_dist, off-road is
    # judged on it (ROAD_HALF_WIDTH_M); the perception value stays in
    # the report for diagnosis.
    rd = [_f(hist, "route_dist", i) for i in settled]
    rd_v = [v for v in rd if _num(v)]
    ro = [_f(hist, "road_off", i) for i in settled]
    ro_v = [v for v in ro if _num(v)]
    # The precise verdict: metres beyond the road's own edge polylines
    # (edge_over, > EDGE_OVER_M = off the pavement).  Fallbacks:
    # route_dist (centre distance vs half width), then the perception
    # road_off (blind in strict mode - kept for old telemetry).
    eo = [_f(hist, "edge_over", i) for i in settled]
    eo_v = [v for v in eo if _num(v)]
    # Frames are reported as DURATION and MAGNITUDE too (plan P0-4):
    # "60 frames off the pavement" is 36 s at 1.65 Hz and 6 s at 10 Hz,
    # and 0.60 m past the edge is a different event from 6.96 m past it.
    t_s = [float(t[i]) for i in settled]
    if eo_v:
        src, limit = "edge_over", EDGE_OVER_M
        mask = [_num(v) and float(v) > limit for v in eo]
        mags = [float(v) for v in eo if _num(v)]
        out["off_road_frames"] = sum(1 for m in mask if m)
        out["max_edge_over_m"] = round(max(eo_v), 3)
    elif rd_v:
        src, limit = "route_dist", ROAD_HALF_WIDTH_M
        mask = [_num(v) and float(v) > limit for v in rd]
        mags = [float(v) for v in rd if _num(v)]
        out["off_road_frames"] = sum(1 for m in mask if m)
        out["max_route_dist_m"] = round(max(rd_v), 3)
    elif ro_v:
        src, limit = "road_off", OFF_ROAD_M
        mask = [_num(v) and float(v) > limit for v in ro]
        mags = [float(v) for v in ro if _num(v)]
        out["off_road_frames"] = sum(1 for m in mask if m)
    else:
        # No off-road source at all: report MISSING, never 0 frames.
        src, limit = None, None
        mask = []
        mags = []
        out["off_road_frames"] = None
    out["off_road_measured"] = src is not None
    out["off_road_src"] = src
    if src is not None:
        total_s, longest_s, eps = _episodes(mask, t_s)
        out["off_road_s"] = round(total_s, 2)
        out["off_road_longest_s"] = round(longest_s, 2)
        out["off_road_episodes"] = eps
        off_mags = [m for m, ok in zip(mags, mask) if ok]
        out["off_road_max_m"] = (round(max(off_mags), 3)
                                 if off_mags else 0.0)
    else:
        out["off_road_s"] = None
        out["off_road_longest_s"] = None
        out["off_road_episodes"] = None
        out["off_road_max_m"] = None
    out["max_road_off_m"] = round(max(ro_v), 3) if ro_v else None

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

    # stalls (away from the end-zone stop).  NB: the speeds are read
    # PER settled frame - indexing v_v (a filtered list) by the raw
    # frame index desyncs the moment any speed is non-numeric.
    stalls = 0
    stall_mask: list[bool] = []
    for i in settled:
        rem = _f(hist, "rem_end", i)
        rem = rem if _num(rem) else None
        spd = _f(hist, "speed", i, 0.0)
        spd = spd if _num(spd) else 0.0
        hit = bool(spd < STALL_SPEED_MPS
                   and (rem is None or rem > STALL_REM_END_M))
        stall_mask.append(hit)
        if hit:
            stalls += 1
    out["stall_frames"] = stalls
    # Duration and episode count, not only frames (plan P0-4): one 40 s
    # stall and forty 1 s stalls are the same frame count and completely
    # different driving failures.
    st_total, st_longest, st_eps = _episodes(stall_mask, t_s)
    out["stall_s"] = round(st_total, 2)
    out["stall_longest_s"] = round(st_longest, 2)
    out["stall_events"] = st_eps
    out["settled_duration_s"] = round(
        max(0.0, (t_s[-1] - t_s[0]) if len(t_s) > 1 else 0.0), 2)
    # Exposure-normalised rates, so runs of different length compare.
    denom = out["settled_duration_s"] or 0.0
    if denom > 0.0:
        out["stall_frac"] = round(st_total / denom, 3)
        out["off_road_frac"] = (round(out["off_road_s"] / denom, 3)
                                if out["off_road_s"] is not None else None)
    else:
        out["stall_frac"] = None
        out["off_road_frac"] = None

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
                             if ll and _num(ll[-1]) else None)
    out["final_lat_right"] = (round(float(lr[-1]), 3)
                              if lr and _num(lr[-1]) else None)
    if goal is not None:
        gx, gy = float(goal[0]), float(goal[1])
        out["goal_dist_m"] = round(math.hypot(
            float(last["pos"][0]) - gx,
            float(last["pos"][1]) - gy), 2)
    out.update(collision_events(hist))
    return out


def collision_events(hist: list[dict], *,
                     min_delta: float = 0.01,
                     merge_s: float = 1.0) -> dict:
    """Count collision events from the damage channel of a run.

    The plan's §12 hard gate starts with ``collision_count = 0``, and the
    evaluator had no way to say it: crossings and off-road frames are
    proxies, not collisions.  A run that recorded the vehicle's damage
    (``damage_total`` per frame, from the beamngpy Damage sensor) can be
    counted honestly - a collision is a frame where the damage value
    INCREASED by at least ``min_delta``.

    Two counts, because they answer different questions (plan P0-4):

    * ``collision_count`` - frames where the damage rose.  One impact
      spread over several frames (the Damage sensor updates per tick
      while a contact persists) counts more than once.
    * ``collision_episodes`` - the same increases grouped by time: rises
      less than ``merge_s`` apart belong to ONE event.  This is the
      deduplicated event count, and the one to quote as "how many
      collisions happened".

    ``collided`` is the 0/1 reading - whether this run collided at all -
    which is what a run-level pass/fail gate actually consumes.

    Returns ``{"collision_count": int | None, "collision_episodes":
    int | None, "collided": bool | None, "damage_frames": int,
    "damage_total": float | None, "first_collision_t": float | None}``.
    ``collision_count`` is None when the run carries no damage samples at
    all: "not measured" must never read as "no collisions" - that is
    exactly the mistake this metric exists to prevent.
    """
    rows = [h for h in (hist or ()) if isinstance(h, dict)]
    vals: list[tuple[float, float]] = []
    for h in rows:
        v = h.get("damage_total")
        if not _num(v):
            continue
        # read the row directly: ``_f`` indexes hist[index][key], it is not
        # a per-row accessor (using it here silently produced t=0.0)
        t_raw = h.get("t")
        vals.append((float(t_raw) if _num(t_raw) else 0.0, float(v)))
    if not vals:
        return {"collision_count": None, "collision_episodes": None,
                "collided": None, "damage_frames": 0,
                "damage_total": None, "first_collision_t": None}
    events = 0
    first_t = None
    last_rise_t: float | None = None
    episodes = 0
    prev = vals[0][1]
    for t, v in vals[1:]:
        if v - prev >= float(min_delta):
            events += 1
            if first_t is None:
                first_t = t
            if last_rise_t is None or (t - last_rise_t) > float(merge_s):
                episodes += 1
            last_rise_t = t
        prev = v
    return {"collision_count": int(events),
            "collision_episodes": int(episodes),
            "collided": bool(events > 0),
            "damage_frames": len(vals),
            "damage_total": round(float(vals[-1][1]), 4),
            "first_collision_t": first_t}


def assess_many(runs: Iterable[list[dict]], goal=None,
                cruise: float | None = None,
                settle_s: float = 0.0) -> list[dict]:
    """Assess several runs; returns a list of result dicts."""
    return [assess_run(h, goal=goal, cruise=cruise, settle_s=settle_s)
            for h in runs]


def score_run(assessed: dict, require_goal: bool = False) -> dict:
    """Verdict on one assessed run against the benchmark hard targets.

    ``assessed`` is an :func:`assess_run` result.  Returns
    ``{"checks": {name: bool}, "unknown": [name], "status": str,
    "pass": bool}``.

    ``status`` is PASS / FAIL / UNKNOWN (plan P0-3):

    * FAIL - a measured target was violated.
    * UNKNOWN - nothing was violated, but at least one target could not be
      MEASURED, so it cannot be cleared either.  A run with no damage
      channel cannot demonstrate ``collision_count = 0``; the old verdict
      silently omitted the collision gate entirely, which is how a run
      with no collision evidence passed.
    * PASS - every target was measured and held.

    ``pass`` is True only for PASS, so the release gate never clears an
    UNKNOWN.  ``checks`` keeps its historical bool shape (an unmeasured
    target is False there, because it did not hold); ``unknown`` says
    whether that False means "violated" or "not measured".
    """
    checks: dict[str, bool] = {
        "has_frames": (
            int(assessed.get("frames", 0) or 0) > 0
            and int(assessed.get("settled_frames",
                                 assessed.get("frames", 0)) or 0) > 0),
        "no_reversing": int(assessed.get("reversing_frames", 0) or 0)
        <= BENCH_MAX_REVERSING_FRAMES,
        "no_centre_crossing": (
            int(assessed.get("cross_centre_frames", 0) or 0)
            <= BENCH_MAX_CROSS_CENTRE
            and int(assessed.get("body_cross_centre_frames", 0) or 0)
            <= BENCH_MAX_CROSS_CENTRE),
        "no_edge_crossing": (
            int(assessed.get("cross_right_frames", 0) or 0)
            <= BENCH_MAX_CROSS_RIGHT
            and int(assessed.get("body_cross_right_frames", 0) or 0)
            <= BENCH_MAX_CROSS_RIGHT),
        "no_stall": int(assessed.get("stall_frames", 0) or 0)
        <= BENCH_MAX_STALL_FRAMES,
    }
    unknown: list[str] = []
    # Lateral absence is UNMEASURED, not "no crossing" (plan T01/§1.4-15).
    # Each side is its own target: ``lat_left`` missing means the
    # centre/oncoming side was never measured, ``lat_right`` missing means
    # the pavement-edge side was not, and either one is UNKNOWN - a run
    # cannot clear a target whose column was empty for the whole run.
    if int(assessed.get("lat_left_frames", 0) or 0) <= 0:
        checks["no_centre_crossing"] = False
        unknown.append("no_centre_crossing")
    if int(assessed.get("lat_right_frames", 0) or 0) <= 0:
        checks["no_edge_crossing"] = False
        unknown.append("no_edge_crossing")
    # ``off_road_frames is None`` means no off-road source existed in the
    # run; that is unmeasured, not "on road".
    orf = assessed.get("off_road_frames")
    if orf is None:
        checks["on_road"] = False
        unknown.append("on_road")
    else:
        checks["on_road"] = int(orf) <= BENCH_MAX_OFF_ROAD_FRAMES
    # The collision gate (§12 #1).  ``collision_count is None`` = the
    # damage channel was never sampled.
    cc = assessed.get("collision_count")
    if cc is None:
        checks["no_collision"] = False
        unknown.append("no_collision")
    else:
        checks["no_collision"] = int(cc) <= BENCH_MAX_COLLISIONS
    if require_goal:
        gd = assessed.get("goal_dist_m")
        if not _num(gd):
            checks["reached_goal"] = False
            unknown.append("reached_goal")
        else:
            checks["reached_goal"] = float(gd) <= BENCH_GOAL_TOL_M
    failed = [k for k, ok in checks.items() if not ok and k not in unknown]
    if failed:
        status = STATUS_FAIL
    elif unknown:
        status = STATUS_UNKNOWN
    else:
        status = STATUS_PASS
    return {"checks": checks, "unknown": unknown, "status": status,
            "pass": status == STATUS_PASS}


def score_many(assessed: list[dict], require_goal: bool = False) -> dict:
    """Aggregate benchmark verdicts over several assessed runs.

    Returns ``{"runs": [...], "status": str, "pass": bool, "n_pass": int,
    "n_unknown": int, "n_failed": int, "n_collided": int | None,
    "collision_run_ratio": float | None}`` - the aggregate passes only
    when EVERY run passes, and is UNKNOWN (never PASS) when any run is
    UNKNOWN.  ``collision_run_ratio`` is the share of runs that collided,
    which is the honest reading of a 0/1 per-run collision flag; it is
    None when no run measured damage.
    """
    verdicts = [score_run(a, require_goal=require_goal) for a in assessed]
    n_pass = sum(1 for v in verdicts if v["status"] == STATUS_PASS)
    n_unknown = sum(1 for v in verdicts if v["status"] == STATUS_UNKNOWN)
    n_failed = sum(1 for v in verdicts if v["status"] == STATUS_FAIL)
    if n_failed:
        status = STATUS_FAIL
    elif n_unknown:
        status = STATUS_UNKNOWN
    elif verdicts:
        status = STATUS_PASS
    else:
        status = STATUS_UNKNOWN
    collided = [a.get("collided") for a in assessed]
    measured = [c for c in collided if c is not None]
    return {"runs": verdicts, "status": status,
            "pass": status == STATUS_PASS,
            "n_pass": n_pass, "n_unknown": n_unknown, "n_failed": n_failed,
            "n_collided": (sum(1 for c in measured if c)
                           if measured else None),
            "collision_run_ratio": (round(sum(1 for c in measured if c)
                                          / len(measured), 3)
                                    if measured else None)}
