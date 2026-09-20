"""调度饥饿指标：一次运行 / 一个 A/B 臂的一行汇总。

2026-09-20 的结论是「stale 不是感知崩溃，而是 tick 预算把 range / object
饿死」，所以判断一次改动有没有用，看的不再只是 ``collision_count`` 和
``stall_frames``，而是：

* ``range_skip_rate``   - 有多少帧的 LiDAR 是被预算跳过的（被复用）；
* ``range_forced_rate`` - 有多少帧是保底（keep-alive）把它强行拉回来的；
* ``range_age_p95``     - 复用到底把证据拖到多旧（STALE_RANGE_S = 2.0 是线）；
* ``object_skip_rate``  - YOLO 被饿死的比例；
* ``degraded_rate``     - 安全层被打到 degraded 的比例；
* ``tick_ms_p95``       - 代价：保底不是免费的。

运行级指标（碰撞 / 停帧 / 里程 / 压线）沿用 ``eval.assess_run``，不在这里
重算——一个指标只能有一个定义。

用法::

    .venv\\Scripts\\python.exe scripts\\m5_sched_metrics.py \\
        off=a.json,b.json,c.json  on=d.json,e.json,f.json

臂名可以随便起；每个臂给多个文件就是多轮，脚本打印每轮明细 + 臂均值。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beamng_autopilot.eval import assess_run, score_run   # noqa: E402

# town 场景（scripts/m5_fsd_benchmark.py 的注册项）：默认值只是省打字，
# 换场景请显式传 --goal。
DEFAULT_GOAL = (868.3, 744.9)
# 与 benchmark 一致：teleport 出生后的前几秒不算纪律（见 eval.assess_run）
DEFAULT_SETTLE_S = 8.0

# (key, header, width, kind)  kind: int | f3 | pct
_COLUMNS: tuple[tuple[str, str, int, str], ...] = (
    ("frames", "frames", 7, "int"),
    ("range_skip_rate", "range_skip%", 12, "pct"),
    ("range_forced_rate", "forced%", 9, "pct"),
    ("range_age_p50", "rage_p50", 10, "f3"),
    ("range_age_p95", "rage_p95", 10, "f3"),
    ("range_age_max", "rage_max", 10, "f3"),
    ("object_skip_rate", "obj_skip%", 11, "pct"),
    ("degraded_rate", "degraded%", 11, "pct"),
    ("tick_ms_p95", "tick_p95", 10, "f3"),
    ("ring_ms_p50", "ring_p50", 10, "f3"),
    ("speed_p50", "speed_p50", 11, "f3"),
    ("stall_frames", "stalls", 8, "int"),
    ("travelled_m", "travel_m", 10, "f3"),
    ("off_road_frames", "off_road", 10, "int"),
    ("collision_count", "coll", 6, "int"),
)


def _pct(vals: list[float], q: float) -> float | None:
    """Nearest-rank percentile (no interpolation, no numpy needed)."""
    if not vals:
        return None
    s = sorted(float(v) for v in vals)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return round(s[idx], 3)


def _mean(vals: list) -> float | None:
    vals = [float(v) for v in vals if v is not None]
    return round(statistics.fmean(vals), 3) if vals else None


def load_hist(path: str | Path) -> list[dict]:
    """Telemetry frames from a run file (a bare list, or {"hist": [...]})."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("hist") or []
    return list(data or [])


def sched_metrics(hist: list[dict]) -> dict:
    """Scheduler/starvation metrics for one run (pure, testable)."""
    n = len(hist)
    out: dict = {"frames": n}
    if n == 0:
        return out

    def _state(frame: dict) -> str:
        return str((frame.get("range_sched") or {}).get("state") or "")

    def _skips(frame: dict) -> list[str]:
        return [str(s) for s in (frame.get("budget_skips") or [])]

    # ``range_sched`` is the primary evidence; ``budget_skips`` is the older
    # column and still covers runs recorded before the scheduler record
    # existed.
    r_defer = sum(1 for f in hist
                  if _state(f) == "budget_deferred" or "range" in _skips(f))
    r_forced = sum(1 for f in hist if _state(f) == "keepalive_forced")
    r_scan = sum(1 for f in hist
                 if _state(f) in ("scanned", "keepalive_forced"))
    o_defer = sum(1 for f in hist if "object" in _skips(f))
    head_defer: dict[str, int] = {}
    for f in hist:
        for name, rec in (f.get("head_sched") or {}).items():
            if str((rec or {}).get("state")) == "budget_deferred":
                head_defer[name] = head_defer.get(name, 0) + 1

    ages: list[float] = []
    ticks: list[float] = []
    rings: list[float] = []
    ranges: list[float] = []
    for f in hist:
        age = (f.get("freshness") or {}).get("range_s")
        if age is not None:
            ages.append(float(age))
        tm = f.get("tick_ms") or {}
        for key, bucket in (("total", ticks), ("ring", rings),
                            ("range", ranges)):
            if tm.get(key) is not None:
                bucket.append(float(tm[key]))
    speeds = [float(f.get("speed") or 0.0) for f in hist]
    levels: dict[str, int] = {}
    for f in hist:
        lv = str(f.get("level") or "?")
        levels[lv] = levels.get(lv, 0) + 1

    out.update({
        "range_skip_frames": r_defer,
        "range_skip_rate": round(r_defer / n, 3),
        "range_forced_frames": r_forced,
        "range_forced_rate": round(r_forced / n, 3),
        "range_scan_frames": r_scan,
        "object_skip_frames": o_defer,
        "object_skip_rate": round(o_defer / n, 3),
        "head_defer_frames": head_defer,
        "range_age_p50": _pct(ages, 0.50),
        "range_age_p95": _pct(ages, 0.95),
        "range_age_max": (round(max(ages), 3) if ages else None),
        "tick_ms_p50": _pct(ticks, 0.50),
        "tick_ms_p95": _pct(ticks, 0.95),
        "ring_ms_p50": _pct(rings, 0.50),
        "range_ms_p50": _pct(ranges, 0.50),
        "speed_p50": round(statistics.median(speeds), 3) if speeds else None,
        "degraded_rate": round((n - levels.get("safe", 0)) / n, 3),
        "levels": levels,
    })
    return out


def run_metrics(hist: list[dict], goal=None,
                settle_s: float = DEFAULT_SETTLE_S) -> dict:
    """Scheduler metrics + the canonical run assessment in one dict."""
    assessed = assess_run(hist, goal=goal, settle_s=settle_s)
    merged = dict(sched_metrics(hist))
    for key in ("stall_frames", "travelled_m", "off_road_frames",
                "cross_centre_frames", "cross_right_frames",
                "goal_dist_m", "speed_med", "collision_count",
                "damage_total", "duration_s"):
        if key in assessed:
            merged[key] = assessed[key]
    merged["verdict"] = score_run(assessed, require_goal=bool(goal))
    return merged


def _parse_arms(items: list[str]) -> list[tuple[str, list[str]]]:
    arms = []
    for item in items:
        if "=" not in item:
            arms.append((Path(item).stem, [item]))
            continue
        name, files = item.split("=", 1)
        paths = [p for p in files.split(",") if p.strip()]
        arms.append((name.strip() or "arm", paths))
    return arms


def _fmt(value, kind: str, width: int) -> str:
    if value is None:
        text = "n/a"
    elif kind == "pct":
        text = f"{float(value) * 100.0:.1f}"
    elif kind == "f3":
        text = f"{float(value):.1f}"
    else:
        text = str(value)
    return f"{text:>{width}}"


def _row(label: str, m: dict) -> str:
    return (f"{label:<22}"
            + "".join(_fmt(m.get(k), kind, w) for k, _, w, kind in _COLUMNS))


def _print_detail(m: dict) -> None:
    """One line per round, indented under its arm."""
    print("  " + _row(str(m.get("file", "?")), m))


def main() -> int:
    ap = argparse.ArgumentParser(description="调度饥饿 / 保活 A/B 指标")
    ap.add_argument("arms", nargs="+",
                    help="arm=file1,file2,... （多个文件 = 多轮）")
    ap.add_argument("--goal", nargs=2, type=float, default=list(DEFAULT_GOAL),
                    help="终点坐标；不要 goal 判据时用 --no-goal")
    ap.add_argument("--no-goal", action="store_true")
    ap.add_argument("--settle-s", type=float, default=DEFAULT_SETTLE_S)
    ap.add_argument("--json", default=None, help="把每个臂的汇总写成 JSON")
    args = ap.parse_args()

    goal = None if args.no_goal else tuple(args.goal)
    report: dict = {}
    print(f"settle_s={args.settle_s}  goal={goal}")
    print(f"{'arm / run':<22}"
          + "".join(f"{h:>{w}}" for _, h, w, _ in _COLUMNS))
    for name, paths in _parse_arms(args.arms):
        rounds = []
        for path in paths:
            hist = load_hist(path)
            m = run_metrics(hist, goal=goal, settle_s=args.settle_s)
            m["file"] = Path(path).name
            rounds.append(m)
            _print_detail(m)
        agg: dict = {}
        for key, _, _, _ in _COLUMNS:
            vals = [r.get(key) for r in rounds]
            if vals and all(isinstance(v, (int, float)) for v in vals):
                agg[key] = _mean(vals)
        agg["n"] = len(rounds)
        agg["verdicts"] = [r.get("verdict") for r in rounds]
        report[name] = {"rounds": rounds, "mean": agg}
        print(_row(f"{name} MEAN", agg))
        print()

    if args.json:
        Path(args.json).write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"written -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
