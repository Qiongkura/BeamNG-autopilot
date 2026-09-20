"""调度饥饿指标（scripts/m5_sched_metrics.py）的纯函数测试。

这个脚本是 A/B 的读数工具，读错了会把结论读反，所以指标定义本身要被
钉住：跳过率来自 ``range_sched``（新）或 ``budget_skips``（旧），
range_age 只认 ``freshness.range_s``，degraded 只认 ``level != safe``。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "m5_sched_metrics", ROOT / "scripts" / "m5_sched_metrics.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sched = _load()


def _frame(t: float, *, level: str = "safe", speed: float = 3.0,
           range_s: float | None = 0.1, state: str | None = "scanned",
           skips=None, tick_ms: float | None = 600.0,
           head_sched: dict | None = None) -> dict:
    f: dict = {"t": t, "level": level, "speed": speed,
               "freshness": ({"range_s": range_s} if range_s is not None
                             else {}),
               "tick_ms": ({"total": tick_ms, "ring": 400.0, "range": 240.0}
                           if tick_ms is not None else {}),
               "budget_skips": list(skips or [])}
    if state is not None:
        f["range_sched"] = {"state": state, "age_s": range_s,
                            "compute_ms": 240.0}
    if head_sched:
        f["head_sched"] = head_sched
    return f


def test_empty_history() -> None:
    assert sched.sched_metrics([]) == {"frames": 0}


def test_skip_and_forced_rates_come_from_the_scheduler_state() -> None:
    hist = [
        _frame(0.0, state="scanned"),
        _frame(1.0, state="budget_deferred", skips=["range", "object"]),
        _frame(2.0, state="keepalive_forced"),
        _frame(3.0, state="budget_deferred", skips=["range"]),
    ]
    m = sched.sched_metrics(hist)
    assert m["frames"] == 4
    assert m["range_skip_frames"] == 2
    assert m["range_skip_rate"] == 0.5
    assert m["range_forced_frames"] == 1
    assert m["range_forced_rate"] == 0.25
    # scanned + forced are both real scans
    assert m["range_scan_frames"] == 2
    assert m["object_skip_frames"] == 1
    assert m["object_skip_rate"] == 0.25


def test_old_runs_fall_back_to_budget_skips() -> None:
    """Runs recorded before range_sched existed still count."""
    hist = [_frame(float(i), state=None, skips=["range"]) for i in range(3)]
    m = sched.sched_metrics(hist)
    assert m["range_skip_frames"] == 3
    assert m["range_forced_frames"] == 0
    assert m["range_scan_frames"] == 0


def test_range_age_uses_the_freshness_contract() -> None:
    hist = [_frame(0.0, range_s=0.5), _frame(1.0, range_s=2.5),
            _frame(2.0, range_s=61.9), _frame(3.0, range_s=None)]
    m = sched.sched_metrics(hist)
    assert m["range_age_max"] == 61.9
    # three samples, nearest-rank: p50 -> 2.5, p95 -> 61.9
    assert m["range_age_p50"] == 2.5
    assert m["range_age_p95"] == 61.9


def test_degraded_rate_is_everything_that_is_not_safe() -> None:
    hist = [_frame(0.0), _frame(1.0, level="degraded"),
            _frame(2.0, level="minimal_risk"), _frame(3.0, level="safe")]
    m = sched.sched_metrics(hist)
    # 4 frames, two of them "safe" (the default level is safe)
    assert m["degraded_rate"] == 0.5
    assert m["levels"] == {"safe": 2, "degraded": 1, "minimal_risk": 1}


def test_head_deferrals_are_attributed_per_head() -> None:
    hist = [
        _frame(0.0, head_sched={"object": {"state": "budget_deferred"},
                                "semantic": {"state": "ran"}}),
        _frame(1.0, head_sched={"object": {"state": "budget_deferred"}}),
        _frame(2.0, head_sched={"object": {"state": "ran"}}),
    ]
    m = sched.sched_metrics(hist)
    assert m["head_defer_frames"] == {"object": 2}


def test_tick_percentiles_are_nearest_rank() -> None:
    """No interpolation: the reported p95 is a tick that actually ran."""
    hist = [_frame(float(i), tick_ms=float(100 * i)) for i in range(1, 11)]
    m = sched.sched_metrics(hist)
    assert m["tick_ms_p50"] == 500.0
    assert m["tick_ms_p95"] == 1000.0
    assert m["ring_ms_p50"] == 400.0
    assert m["range_ms_p50"] == 240.0


def test_parse_arms_accepts_names_and_bare_files() -> None:
    arms = sched._parse_arms(["off=a.json,b.json", "c.json"])
    assert arms[0] == ("off", ["a.json", "b.json"])
    assert arms[1] == ("c", ["c.json"])


def test_load_hist_accepts_both_shapes(tmp_path) -> None:
    bare = tmp_path / "bare.json"
    bare.write_text('[{"t": 0.0}]', encoding="utf-8")
    assert sched.load_hist(bare) == [{"t": 0.0}]
    wrapped = tmp_path / "wrapped.json"
    wrapped.write_text('{"hist": [{"t": 1.0}]}', encoding="utf-8")
    assert sched.load_hist(wrapped) == [{"t": 1.0}]


@pytest.mark.parametrize("bad", ["[]", "{}"])
def test_load_hist_tolerates_empty_files(tmp_path, bad) -> None:
    p = tmp_path / "empty.json"
    p.write_text(bad, encoding="utf-8")
    assert sched.load_hist(p) == []
