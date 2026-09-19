"""Async heavy heads (plan phase A4): perception overruns must not block.

The load-bearing contract: a slow semantic/YOLO head costs FRESHNESS, not
control cadence - ``FSDStack.tick`` never waits for a head, serves the
newest available output, and reports its honest age.  Strict mode keeps
the semantic lane synchronous, because a stale semantic fail-closes the
car there (the same rule the tick-budget governor already enforces).
"""

from __future__ import annotations

import time

import numpy as np
import pytest

import beamng_autopilot.fsd_stack as fs
from beamng_autopilot.fsd_stack import (
    ASYNC_HEAD_TIMEOUT_S,
    FSDStack,
    FSDTick,
    _async_allowed,
)


@pytest.fixture(autouse=True)
def _enable_async(monkeypatch):
    """The async worker path is opt-in; these tests exercise it."""
    monkeypatch.setattr(fs, "ASYNC_HEADS_ENABLED", True)
from beamng_autopilot.vision.hydra import HydraNet, TaskOutput


class _StubRange:
    def scan(self, pos):
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


class _SlowHead:
    """A head that takes far longer than a control tick (like UNet/YOLO)."""

    def __init__(self, name: str, seconds: float, boom: bool = False):
        self.name = name
        self.seconds = float(seconds)
        self.boom = bool(boom)
        self.runs = 0

    def run(self, ctx):
        self.runs += 1
        time.sleep(self.seconds)
        if self.boom:
            raise RuntimeError("head exploded")
        h, w = ctx.frame_rgb.shape[:2]
        return TaskOutput(masks={"road": np.ones((h, w), dtype=bool)},
                          meta={"markings": [1], "slow_head": self.name})


def _stack(*heads) -> FSDStack:
    st = FSDStack.__new__(FSDStack)
    st.conn = _StubConn()
    st.ring = _StubRing()
    st.mode = "tech-stub"
    st.range_prov = _StubRange()
    st.hydra = HydraNet()
    for head in heads:
        st.hydra.add(head)
    from beamng_autopilot.planning import Constraints
    st.constraints = Constraints(w_collision=5.0, w_curvature=0.5,
                                 w_lane_align=1.0)
    st.grid_n, st.grid_res = 60, 0.5
    st._head_workers = {}
    st._head_async = set()
    # reset_temporal() touches these on a __new__-built stub
    st._lane_fusion_state = {}
    st.tracker = None
    st.fmap = None
    st._last_range = None
    st._last_range_t = 0.0
    st._range_skip = 0
    st._lane_ref_prev = None
    st._lane_ref_hold_t = 0.0
    return st


# ---------------------------------------------------------------------------
# eligibility
# ---------------------------------------------------------------------------

def test_object_head_is_async_in_both_modes() -> None:
    assert _async_allowed("object", strict=True)
    assert _async_allowed("object", strict=False)


def test_semantic_head_is_synchronous_in_strict_mode() -> None:
    """Strict mode's lateral input may never be served from a worker.

    A stale semantic lane reads as "stale sensor" and fail-closes the
    car, which is exactly what the budget governor refuses to cause.
    """
    assert not _async_allowed("semantic", strict=True)
    assert _async_allowed("semantic", strict=False)


def test_other_heads_stay_synchronous() -> None:
    assert not _async_allowed("traffic", strict=False)
    assert not _async_allowed("topology", strict=False)


def test_async_is_off_by_default(monkeypatch) -> None:
    """Default OFF: a behaviour-changing default needs live evidence."""
    monkeypatch.setattr(fs, "ASYNC_HEADS_ENABLED", False)
    assert not _async_allowed("object", strict=False)
    assert not _async_allowed("semantic", strict=False)


# ---------------------------------------------------------------------------
# the tick never waits
# ---------------------------------------------------------------------------

def test_slow_head_does_not_block_the_tick() -> None:
    slow = _SlowHead("object", 0.4)
    st = _stack(slow)
    t0 = time.perf_counter()
    out1 = st.tick()
    first = time.perf_counter() - t0
    assert isinstance(out1, FSDTick)
    assert first < 0.25, f"tick waited for the slow head: {first:.3f}s"
    # first tick has no output yet - honest "unknown", not a fake age
    assert out1.meta["head_age_s"]["object"] is None
    # a second tick still does not wait, and now reports the age
    t1 = time.perf_counter()
    out2 = st.tick()
    assert time.perf_counter() - t1 < 0.25
    assert "object" in out2.meta.get("head_async", [])
    # once the job lands, a later tick adopts it
    assert st._head_workers["object"].join(timeout=3.0)
    out3 = st.tick()
    assert "object" in out3.head_outputs
    assert out3.meta["head_age_s"]["object"] is not None


def test_async_head_failure_degrades_without_stopping_the_tick() -> None:
    boom = _SlowHead("object", 0.0, boom=True)
    st = _stack(boom)
    st.tick()
    assert st._head_workers["object"].join(timeout=3.0)
    out = st.tick()
    assert isinstance(out, FSDTick)
    assert "head exploded" in str(st.hydra.errors.get("object", ""))
    # the head stays due each tick, so it is re-submitted and fails again
    assert st._head_workers["object"].errors >= 1


def test_worker_health_is_published_for_telemetry() -> None:
    slow = _SlowHead("object", 0.05)
    st = _stack(slow)
    st.tick()
    assert st._head_workers["object"].join(timeout=3.0)
    out = st.tick()
    digest = out.meta.get("head_worker", {}).get("object")
    assert digest is not None
    assert digest["submitted"] >= 1
    assert digest["completed"] >= 1
    assert ASYNC_HEAD_TIMEOUT_S > 0.0


def test_strict_mode_keeps_semantic_synchronous_and_creates_no_worker() -> None:
    sem = _SlowHead("semantic", 0.0)
    st = _stack(sem, _SlowHead("object", 0.0))
    st.strict_sensor = True
    st.tick()
    assert "semantic" not in st._head_workers, \
        "strict semantic lane must stay synchronous"
    assert "object" in st._head_workers
    st._head_workers["object"].join(timeout=3.0)


def test_teleport_drops_a_pre_teleport_async_result() -> None:
    """A result computed at the old pose must never be adopted."""
    slow = _SlowHead("object", 0.2)
    st = _stack(slow)
    st.tick()
    assert st._head_workers["object"].join(timeout=3.0)
    st.reset_temporal()
    out = st.tick()
    # the pre-teleport result was drained, so this tick serves none of it
    assert out.meta["head_age_s"]["object"] is None


# ---------------------------------------------------------------------------
# range split (LiDAR fetch/cluster decoupling, plan A4)
# ---------------------------------------------------------------------------

class _SplitRange:
    """RangeProvider stub with the two-phase split and a slow CPU half."""

    range_split = True

    def __init__(self, cpu_seconds: float = 0.4):
        self.cpu_seconds = float(cpu_seconds)
        self.fetches = 0
        self.processes = 0

    def _points(self):
        from beamng_autopilot.perception import Obstacle
        from beamng_autopilot.runtime import RangeSample
        return RangeSample(
            obstacles=[Obstacle(x=6.0, y=0.0, half_w=1.0, half_h=1.0,
                                category="lidar")],
            ray_hits=[(6.0, -1.0), (6.0, 1.0)])

    def fetch(self, pos, ego_vid=None, radius: float = 55.0):
        self.fetches += 1
        return {"pos": np.asarray(pos, dtype=float).copy()}

    def process(self, payload, pos, ego_vid=None,
                radius: float = 55.0):
        self.processes += 1
        time.sleep(self.cpu_seconds)
        return self._points()

    def scan(self, pos, ego_vid=None, radius: float = 55.0):
        return self._points()


def test_async_range_does_not_block_the_tick() -> None:
    """A 400 ms clustering must not cost the tick 400 ms."""
    prov = _SplitRange(cpu_seconds=0.4)
    st = _stack(_SlowHead("object", 0.0))
    st.range_prov = prov
    t0 = time.perf_counter()
    out = st.tick()
    took = time.perf_counter() - t0
    assert took < 0.3, f"tick waited for the clustering: {took:.3f}s"
    assert out.meta.get("range_async") == 1
    assert prov.fetches == 1, "the locked fetch must happen on the tick"
    assert prov.processes == 1, "the CPU half goes to the worker"
    # the worker result lands and a later tick adopts it
    assert st._range_worker.join(timeout=5.0)
    out2 = st.tick()
    assert out2.ray_hits == [(6.0, -1.0), (6.0, 1.0)]
    assert out2.meta["range_age_s"] is not None


def test_async_range_serves_the_cached_scan_while_clustering() -> None:
    """Between scans the tick keeps a usable range sample (no starvation)."""
    prov = _SplitRange(cpu_seconds=0.05)
    st = _stack(_SlowHead("object", 0.0))
    st.range_prov = prov
    st.tick()
    assert st._range_worker.join(timeout=5.0)
    out = st.tick()                     # adopts the finished scan
    assert out.ray_hits
    out_next = st.tick()                # next job in flight -> cached
    assert out_next.ray_hits is not None


def test_async_range_worker_is_drained_on_teleport() -> None:
    prov = _SplitRange(cpu_seconds=0.2)
    st = _stack(_SlowHead("object", 0.0))
    st.range_prov = prov
    st.tick()
    assert st._range_worker.join(timeout=5.0)
    st.reset_temporal()
    out = st.tick()
    # the pre-teleport cloud was dropped; this tick has no cached sample
    assert out.meta["range_age_s"] is None


def test_async_range_is_off_by_default(monkeypatch) -> None:
    monkeypatch.setattr(fs, "ASYNC_HEADS_ENABLED", False)
    prov = _SplitRange(cpu_seconds=0.0)
    st = _stack(_SlowHead("object", 0.0))
    st.range_prov = prov
    out = st.tick()
    assert out.meta.get("range_async") is None
    assert prov.fetches == 0
    assert not hasattr(st, "_range_worker")
