"""Scheduler keep-alive (2026-09-20): a budget deferral may not starve a
safety modality.

The tick-budget governor compares the WHOLE tick's elapsed time against
the budget, and the camera ring alone costs more than that budget on the
live town runs - so ``range`` was deferred on 151/151 and 120/120 frames,
``range_age`` reached p50 61.9 s, the freshness contract read "stale
sensor" on every frame and the car crawled (speed p50 0.00 m/s).  The
same shape had already appeared once with ``range_every_n=3`` (4-6 s
range age); setting the throttle to 1 only handed the starvation to the
budget gate, because both reuse paths were missing one rule:

    a reused modality may not be pushed past its keep-alive bound.

These tests pin that rule, and pin that it is a FLOOR rather than an
exemption: inside the bound the budget still defers.
"""

from __future__ import annotations

import numpy as np
import pytest

import beamng_autopilot.fsd_stack as fs
from beamng_autopilot.fsd_stack import (
    FSDStack,
    OBJECT_KEEPALIVE_S,
    RANGE_KEEPALIVE_S,
    _budget_defers,
    _keepalive_expired,
    _keepalive_s,
)


# ---------------------------------------------------------------------------
# stubs (same shapes as tests/test_fsd_stack.py, kept local so this file
# reads as one contract)
# ---------------------------------------------------------------------------

class _Clock:
    """Clock that advances a fixed step on every read.

    Ages and elapsed times become deterministic and large regardless of
    the Windows timer resolution, so "the tick is over budget" is a fact
    of the test rather than a race.
    """

    def __init__(self, step: float = 1.0, t0: float = 1000.0) -> None:
        self.step = float(step)
        self.t = float(t0)

    def __call__(self) -> float:
        self.t += self.step
        return self.t


@pytest.fixture
def fast_clock(monkeypatch):
    """One tick spans more than any keep-alive bound."""
    clock = _Clock(step=1.0)
    monkeypatch.setattr(fs.time, "time", clock)
    return clock


class _StubRange:
    """Range stub that counts how many real scans happened."""

    def __init__(self) -> None:
        self.calls = 0

    def scan(self, pos):
        self.calls += 1
        from beamng_autopilot.perception import Obstacle
        from beamng_autopilot.runtime import RangeSample
        return RangeSample(
            obstacles=[Obstacle(x=6.0, y=0.0, half_w=1.0, half_h=1.0,
                                category="lidar")],
            ray_hits=[(6.0, -1.0), (6.0, 1.0)])


class _StubRing:
    role = "front_main"

    def grab_ring(self):
        from beamng_autopilot.vision.projection import CameraModel
        model = CameraModel(np.array([0.0, 1.0, 1.4]),
                            np.array([0.0, 0.9999, -0.02]),
                            np.array([0.0, 0.02, 0.9999]), 65.0, 160, 120)
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        return {"front_main": (frame, model)}

    def close(self):
        return None


class _StubConn:
    def get_state(self):
        class S:
            pos = np.array([0.0, 0.0, 0.0])
            heading = 0.0
            speed = 5.0
        return S()


class _CountingHead:
    """Head that counts how many times it actually ran."""

    def __init__(self, name: str) -> None:
        self.name = str(name)
        self.calls = 0

    def run(self, ctx):
        self.calls += 1
        from beamng_autopilot.vision.hydra import TaskOutput
        h, w = ctx.frame_rgb.shape[:2]
        road = np.zeros((h, w), dtype=bool)
        road[h // 2:] = True
        return TaskOutput(masks={"road": road}, meta={"markings": [1, 2]})


def _stack(*heads) -> FSDStack:
    st = FSDStack.__new__(FSDStack)
    st.conn = _StubConn()
    st.ring = _StubRing()
    st.mode = "tech-stub"
    st.range_prov = _StubRange()
    from beamng_autopilot.vision.hydra import HydraNet
    st.hydra = HydraNet()
    for head in heads:
        st.hydra.add(head)
    from beamng_autopilot.planning import Constraints
    st.constraints = Constraints(w_collision=5.0, w_curvature=0.5,
                                 w_lane_align=1.0)
    st.grid_n, st.grid_res = 60, 0.5
    # range state a __new__-built stub does not get from __init__
    st._last_range = None
    st._last_range_t = 0.0
    st._range_skip = 0
    return st


# ---------------------------------------------------------------------------
# the rule itself
# ---------------------------------------------------------------------------

def test_keepalive_is_off_by_default(monkeypatch) -> None:
    """Default OFF, like every behaviour-changing switch in this repo."""
    monkeypatch.setattr(fs, "SCHED_KEEPALIVE_ENABLED", False)
    assert _keepalive_s("range") is None
    assert _keepalive_s("object") is None
    # no floor -> the old unconditional deferral, however old the output
    assert _budget_defers("range", strict=True, every_n=2, budget=0.45,
                          elapsed=9.0, age_s=120.0)


def test_keepalive_bounds_stay_inside_the_stale_contract(monkeypatch) -> None:
    """The floor must fire before the safety monitor calls it stale.

    ``compensate_range_motion`` clamps its prediction at
    RANGE_REUSE_MAX_DT_S and the stale verdict is STALE_RANGE_S, so a
    keep-alive at or above either bound would protect nothing.
    """
    from beamng_autopilot.safety_monitor import STALE_RANGE_S
    assert RANGE_KEEPALIVE_S < STALE_RANGE_S
    assert RANGE_KEEPALIVE_S < fs.RANGE_REUSE_MAX_DT_S
    assert 0.0 < OBJECT_KEEPALIVE_S


def test_a_fresh_output_is_still_deferred(monkeypatch) -> None:
    """The floor is a floor, not an exemption: inside the bound the budget
    still owns smoothness."""
    monkeypatch.setattr(fs, "SCHED_KEEPALIVE_ENABLED", True)
    assert _budget_defers("range", strict=True, every_n=2, budget=0.45,
                          elapsed=9.0, age_s=RANGE_KEEPALIVE_S * 0.5)
    assert not _budget_defers("range", strict=True, every_n=2, budget=0.45,
                              elapsed=9.0, age_s=RANGE_KEEPALIVE_S)


def test_a_modality_with_no_output_is_never_deferred(monkeypatch) -> None:
    """Nothing to reuse: deferring would not serve a cached output, it
    would leave the modality absent for the whole run."""
    monkeypatch.setattr(fs, "SCHED_KEEPALIVE_ENABLED", True)
    assert not _budget_defers("object", strict=True, every_n=2, budget=0.45,
                              elapsed=9.0, age_s=None)


def test_semantic_keeps_its_strict_exception(monkeypatch) -> None:
    """The pre-existing strict rule is untouched by the floor."""
    monkeypatch.setattr(fs, "SCHED_KEEPALIVE_ENABLED", False)
    assert not _budget_defers("semantic", strict=True, every_n=2,
                              budget=0.45, elapsed=9.0, age_s=0.0)
    assert _budget_defers("semantic", strict=False, every_n=2,
                          budget=0.45, elapsed=9.0, age_s=0.0)


def test_expiry_is_exactly_the_bound(monkeypatch) -> None:
    monkeypatch.setattr(fs, "SCHED_KEEPALIVE_ENABLED", True)
    assert not _keepalive_expired("range", None)
    assert not _keepalive_expired("range", RANGE_KEEPALIVE_S - 1e-6)
    assert _keepalive_expired("range", RANGE_KEEPALIVE_S)
    assert not _keepalive_expired("semantic", 1e9)     # no floor: never


# ---------------------------------------------------------------------------
# the live shape: budget starvation, and the floor ending it
# ---------------------------------------------------------------------------

def test_budget_starvation_is_what_the_default_does(
        monkeypatch, fast_clock) -> None:
    """Reproduce the live failure offline: over budget every tick, the
    scan is reused for ever and its age walks away."""
    monkeypatch.setattr(fs, "SCHED_KEEPALIVE_ENABLED", False)
    st = _stack(_CountingHead("semantic"))
    out = None
    for _ in range(4):
        out = st.tick(time_budget_s=1e-12)
    assert st.range_prov.calls == 1, "one scan, then pure reuse"
    assert out.meta["range_sched"]["state"] == "budget_deferred"
    assert out.meta["range_sched"]["age_s"] > RANGE_KEEPALIVE_S
    assert "range" in (out.meta.get("tick_budget_skips") or [])


def test_keepalive_forces_the_scan_back(monkeypatch, fast_clock) -> None:
    """Same run, floor on: the reused scan cannot pass its bound."""
    monkeypatch.setattr(fs, "SCHED_KEEPALIVE_ENABLED", True)
    st = _stack(_CountingHead("semantic"))
    st.tick(time_budget_s=1e-12)            # first tick: nothing to reuse
    assert st.range_prov.calls == 1
    out = st.tick(time_budget_s=1e-12)      # age now past the bound
    assert out.meta["range_sched"]["state"] == "keepalive_forced"
    assert st.range_prov.calls == 2
    # a forced scan is not a budget skip: it is a starvation escape
    assert "range" not in (out.meta.get("tick_budget_skips") or [])


def test_keepalive_forces_the_object_head_past_its_bound(
        monkeypatch, fast_clock) -> None:
    """object_every_n=2 + a blown budget used to mean "never runs": the
    live baseline had n_object_obstacles = 0 on 151/151 frames."""
    monkeypatch.setattr(fs, "SCHED_KEEPALIVE_ENABLED", True)
    obj = _CountingHead("object")
    st = _stack(_CountingHead("semantic"), obj)
    st.object_every_n = 2
    out1 = st.tick(time_budget_s=1e-12)
    assert obj.calls == 1, "no output yet -> must run"
    assert out1.meta["head_sched"]["object"]["state"] == "ran"
    st.tick(time_budget_s=1e-12)            # not due this tick
    out3 = st.tick(time_budget_s=1e-12)     # due again, output now old
    assert obj.calls == 2
    assert out3.meta["head_sched"]["object"]["state"] == "keepalive_forced"


def test_the_every_n_throttle_obeys_the_same_bound(
        monkeypatch, fast_clock) -> None:
    """``range_every_n>1`` is the path that produced 4-6 s range age
    before it was forced to 1 in strict mode: fixing it alone only moved
    the starvation to the budget gate, so the bound covers it too."""
    monkeypatch.setattr(fs, "SCHED_KEEPALIVE_ENABLED", True)
    st = _stack(_CountingHead("semantic"))
    st.range_every_n = 3
    st.tick(time_budget_s=None)             # no budget: normal cadence
    assert st.range_prov.calls == 1
    out = st.tick(time_budget_s=None)       # throttle tick, age past bound
    assert out.meta["range_sched"]["state"] == "keepalive_forced"
    assert st.range_prov.calls == 2


# ---------------------------------------------------------------------------
# the scheduler record
# ---------------------------------------------------------------------------

def test_head_sched_says_why_a_head_did_not_run(
        monkeypatch, fast_clock) -> None:
    """The age says a head is old; head_sched says whether the cadence,
    the budget or the worker is the reason."""
    monkeypatch.setattr(fs, "SCHED_KEEPALIVE_ENABLED", False)
    sem = _CountingHead("semantic")
    st = _stack(sem)
    st.semantic_every_n = 2
    out1 = st.tick(time_budget_s=1e-12)     # due, over budget -> deferred
    rec = out1.meta["head_sched"]["semantic"]
    assert rec["state"] == "budget_deferred"
    assert "budget" in rec["reason"]
    assert rec["compute_ms"] is None
    assert sem.calls == 0
    # A deferred head stays due (``_head_retry``), so it is deferred
    # again rather than falling back to the cadence - the loop that
    # starves a head for a whole run.  The record must say so.
    out2 = st.tick(time_budget_s=1e-12)
    assert out2.meta["head_sched"]["semantic"]["state"] == "budget_deferred"

    # With no budget the head runs, and the NEXT tick is a cadence skip.
    st2 = _stack(_CountingHead("semantic"))
    st2.semantic_every_n = 2
    assert st2.tick(time_budget_s=None).meta["head_sched"]["semantic"][
        "state"] == "ran"
    out4 = st2.tick(time_budget_s=None)
    assert out4.meta["head_sched"]["semantic"]["state"] == "not_due"
    assert "every_n" in out4.meta["head_sched"]["semantic"]["reason"]


def test_head_sched_reports_the_compute_time(monkeypatch, fast_clock) -> None:
    st = _stack(_CountingHead("semantic"))
    out = st.tick(time_budget_s=None)
    rec = out.meta["head_sched"]["semantic"]
    assert rec["state"] == "ran"
    assert rec["compute_ms"] is not None and rec["compute_ms"] >= 0.0
    assert rec["age_s"] is None or rec["age_s"] >= 0.0
