"""Tech LiDAR range split: locked fetch, unlocked clustering (plan A4).

The contract that lets the heavy half run on a worker thread: ``fetch``
touches the connection (under ``io_lock``), ``process`` never does, and
``scan`` stays exactly ``process(fetch(...))`` so the threading boundary
is not a behaviour change.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest

import beamng_autopilot_tech.providers as prov
from beamng_autopilot.perception import LidarClusterTracker, Obstacle
from beamng_autopilot_tech.providers import TechRangeProvider


class _FakeLidar:
    """Duck-typed LiDAR sensor: ``poll`` returns a synthetic cloud."""

    def __init__(self, cloud: np.ndarray) -> None:
        self.cloud = cloud
        self.polls = 0

    def poll(self):
        self.polls += 1
        return {"pointCloud": self.cloud}

    def remove(self):
        return None


def _cloud() -> np.ndarray:
    """Points exercising every branch: far, near, on-car, and a NaN row."""
    pts = [
        (10.0, 0.5, 0.0),      # corridor hit
        (12.0, -1.0, 0.4),     # corridor hit
        (20.0, 8.0, 0.1),      # outside the 55 m/4 m window? no: inside
        (1.5, 0.2, 0.3),       # near field (1-2.5 m)
        (0.5, 0.0, 0.2),       # on the car footprint
        (30.0, 0.0, 0.0),
        (31.0, 0.6, 0.0),
        (32.0, -0.6, 0.0),
        (np.nan, 0.0, 0.0),    # dropped by the finite filter in fetch
        (33.0, 1.2, 0.0),
    ]
    return np.asarray(pts, dtype=float)


def _provider(cloud: np.ndarray | None = None) -> TechRangeProvider:
    p = TechRangeProvider.__new__(TechRangeProvider)
    p.conn = SimpleNamespace(io_lock=threading.Lock(), bng=None,
                             vehicle=SimpleNamespace(state={"dir": [1.0, 0.0, 0.0]}))
    p.lidar = _FakeLidar(_cloud() if cloud is None else cloud)
    p._ego_half_len, p._ego_half_w = 2.2, 0.9
    p._lidar_tracker = LidarClusterTracker()
    p.name = "test_lidar"
    return p


@pytest.fixture(autouse=True)
def _stub_lua(monkeypatch):
    """Stub the Lua/scenario fan and record the lock state it saw."""
    seen: dict = {}

    def _stub(bng, ego_vid, pos, radius=55.0, **kwargs):
        seen["locked_during_lua"] = p_lock.locked()
        return [], [(9.0, 0.4), (9.0, -0.4)]

    p_lock = threading.Lock()
    monkeypatch.setattr(prov, "scan_obstacles_all", _stub)
    return seen


def test_fetch_reads_under_the_lock_and_lua_sees_it(monkeypatch) -> None:
    seen: dict = {}

    def _stub(bng, ego_vid, pos, radius=55.0, **kwargs):
        seen["locked"] = prov_lock.locked()
        return [], [(9.0, 0.4)]

    prov_lock = threading.Lock()
    monkeypatch.setattr(prov, "scan_obstacles_all", _stub)
    p = _provider()
    p.conn.io_lock = prov_lock
    payload = p.fetch((0.0, 0.0, 0.0))
    assert seen["locked"] is True, "connection reads must hold io_lock"
    assert prov_lock.locked() is False, "fetch must release the lock"
    assert payload.cloud.shape[1] == 3
    assert np.isfinite(payload.cloud).all(), "fetch drops NaN rows"


def test_process_never_holds_the_lock(monkeypatch) -> None:
    """The CPU half must run with the connector lock FREE.

    Proved by taking the lock for the whole ``process`` call from another
    thread: a locked acquisition inside process would block forever, so
    the test would time out instead of passing.
    """
    p = _provider()
    payload = p.fetch((0.0, 0.0, 0.0))
    holder = threading.Thread(
        target=lambda: p.conn.io_lock.acquire(), daemon=True)
    holder.start()
    holder.join(timeout=1.0)
    assert p.conn.io_lock.locked() is True      # held by the other thread
    done: dict = {}

    def _run():
        done["sample"] = p.process(payload, (0.0, 0.0, 0.0))
    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(timeout=5.0)
    assert not worker.is_alive(), "process blocked on the connection lock"
    assert done["sample"] is not None
    p.conn.io_lock.release()


def test_clustering_happens_only_in_process(monkeypatch) -> None:
    calls: list = []
    real = prov.lidar_obstacles

    def _spy(*args, **kwargs):
        calls.append(len(args[0]))
        return real(*args, **kwargs)

    monkeypatch.setattr(prov, "lidar_obstacles", _spy)
    p = _provider()
    payload = p.fetch((0.0, 0.0, 0.0))
    assert calls == [], "fetch must not cluster"
    p.process(payload, (0.0, 0.0, 0.0))
    assert calls and calls[0] >= 4, "process clusters the cloud"


def test_scan_equals_process_of_fetch() -> None:
    """The split is a threading boundary, not a behaviour change."""
    a = _provider()
    b = _provider()
    sync = a.scan((0.0, 0.0, 0.0))
    split = b.process(b.fetch((0.0, 0.0, 0.0)), (0.0, 0.0, 0.0))
    assert sync.ray_hits == split.ray_hits
    assert len(sync.obstacles) == len(split.obstacles)
    for ob_a, ob_b in zip(sync.obstacles, split.obstacles):
        assert (ob_a.x, ob_a.y) == pytest.approx((ob_b.x, ob_b.y))
        assert ob_a.category == ob_b.category


def test_fetch_failure_yields_a_usable_empty_payload(monkeypatch) -> None:
    class _Boom:
        def poll(self):
            raise RuntimeError("lidar gone")

    p = _provider()
    p.lidar = _Boom()
    payload = p.fetch((0.0, 0.0, 0.0))
    assert payload is not None
    assert len(payload.cloud) == 0
    sample = p.process(payload, (0.0, 0.0, 0.0))
    assert sample is not None            # Lua hits still flow through


def test_base_provider_declares_no_split() -> None:
    from beamng_autopilot.runtime import RangeProvider
    assert RangeProvider.range_split is False
    assert TechRangeProvider.range_split is True


def test_sample_timing_separates_io_cpu_and_retains_each_source_error(monkeypatch):
    class Clock:
        value = 100.0

        def read(self):
            return self.value

    clock = Clock()
    monkeypatch.setattr(prov.time, "time", clock.read)
    monkeypatch.setattr(prov.time, "perf_counter", clock.read)
    p = _provider()

    def poll():
        clock.value += 0.2
        return {"pointCloud": _cloud()}

    def lua(*args, **kwargs):
        clock.value += 0.3
        return [], [(9.0, 0.4)]

    def cluster(*args, **kwargs):
        clock.value += 0.4
        return []

    p.lidar.poll = poll
    monkeypatch.setattr(prov, "scan_obstacles_all", lua)
    monkeypatch.setattr(prov, "lidar_obstacles", cluster)
    payload = p.fetch((0, 0, 0))
    clock.value += 0.6  # Worker queue delay is not poll/CPU compute.
    sample = p.process(payload, (0, 0, 0))
    assert sample.meta["source_poll_started_t"] == 100.0
    assert sample.meta["sensor_capture_t"] is None
    assert sample.meta["age_basis"] == "poll_start_proxy"
    assert sample.meta["lidar_poll_ms"] == pytest.approx(200)
    assert sample.meta["lua_poll_ms"] == pytest.approx(300)
    assert sample.meta["fetch_ms"] == pytest.approx(500)
    assert sample.meta["process_ms"] == pytest.approx(400)
    assert sample.meta["process_started_t"] - sample.meta["fetch_finished_t"] == pytest.approx(0.6)
    assert "process_ms" not in payload.meta  # Payload diagnostics are immutable.
    sample.meta["source_errors"]["later"] = "not shared"
    assert not payload.meta["source_errors"]

    def bad_poll():
        raise RuntimeError("lidar gone")

    p.lidar.poll = bad_poll
    failed = p.scan((0, 0, 0))
    assert failed.meta["source_errors"] == {"lidar": "lidar gone"}
    assert failed.ray_hits == [(9.0, 0.4)]  # Successful Lua must not erase error.


@pytest.mark.parametrize("has_hits", [False, True])
def test_existing_corridor_hits_skip_far_downsample_but_keep_near_obstacles(monkeypatch, has_hits):
    cloud = np.asarray([(1.2, 0, 0), (1.3, 0, .7), (1.4, 0, .8), (1.5, 0, .9),
                        (10, 2, 0), (11, 2, 0), (12, 2, 0), (13, 2, 0)], float)
    p = _provider(cloud)
    p._ego_half_len = p._ego_half_w = .2
    payload = p.fetch((0, 0, 0))
    payload.ray_hits = [(9., .4)] if has_hits else []
    seen = []
    def downsample(points):
        seen.append(points.copy())
        return points
    monkeypatch.setattr(prov, "downsample_cloud", downsample)
    monkeypatch.setattr(prov, "_local_ground_z", lambda points, *args: np.zeros(len(points)))
    monkeypatch.setattr(prov, "lidar_obstacles", lambda *a, **k: [Obstacle(10., 2., .5, .5)])
    sample = p.process(payload, (0, 0, 0))
    assert len(seen) == (1 if has_hits else 2)
    np.testing.assert_array_equal(seen[-1], cloud[:4])
    expected_far = [(9., .4)] if has_hits else [(float(x), float(y)) for x, y, _ in cloud[4:]]
    assert sample.ray_hits == expected_far + [(1.3, 0.), (1.4, 0.), (1.5, 0.)]
    assert sample.obstacles[0].x == 10 and sample.obstacles[0].y == 2
    assert sample.meta["corridor_downsampled"] is (not has_hits)


@pytest.mark.parametrize("source,expected", [(100., 100.), (None, 103.),
                                            (float("nan"), 103.), (104., 103.), (True, 103.)])
def test_tracker_timestamp_has_explicit_source_or_fallback(monkeypatch, source, expected):
    p = _provider()
    payload = p.fetch((0, 0, 0))
    payload.meta["source_poll_started_t"] = source
    times = []
    p._lidar_tracker = SimpleNamespace(update=lambda boxes, t: times.append(t))
    monkeypatch.setattr(prov.time, "time", lambda: 103.)
    monkeypatch.setattr(prov, "lidar_obstacles", lambda *a, **k: [])
    result = p.process(payload, (0, 0, 0))
    assert times == [expected]
    assert result.meta["tracker_timestamp_t"] == expected
    assert result.meta["tracker_time_basis"] == (
        "poll_start_proxy" if source == 100. else "processing_time_fallback")


def test_tracker_velocity_uses_poll_interval_despite_changing_processing_delay(monkeypatch):
    p = _provider()
    clock = [100.2]
    x = [10.]
    monkeypatch.setattr(prov.time, "time", lambda: clock[0])
    monkeypatch.setattr(prov, "lidar_obstacles", lambda *a, **k: [Obstacle(x[0], 5., .5, .5)])
    payload = p.fetch((0, 0, 0))
    payload.meta["source_poll_started_t"] = 100.
    first = p.process(payload, (0, 0, 0))
    assert first.obstacles[0].velocity is None
    clock[0], x[0] = 101.4, 12.
    payload.meta["source_poll_started_t"] = 100.5
    second = p.process(payload, (0, 0, 0))
    # Two metres / .5s, with the existing .5 EMA gain; completion dt=1.2s
    # would miss the tracker's <=1s velocity update window entirely.
    np.testing.assert_allclose(second.obstacles[0].velocity, [2., 0.])


def test_opt_in_capture_happens_once_and_disarm_prevents_later_fetch(monkeypatch, tmp_path):
    from beamng_autopilot.experiments import range_replay
    p = _provider()
    calls = []
    def save(directory, payload, **kwargs):
        calls.append((directory, payload, kwargs))
        assert not p.conn.io_lock.locked()
        return {"status": "captured", "points": len(payload.cloud)}
    monkeypatch.setattr(range_replay, "write_replay", save)
    p.fetch((0, 0, 0))
    assert not calls  # Normal driving defaults off.
    p.arm_replay_capture(tmp_path / "capture")
    p.fetch((0, 0, 0))
    p.fetch((0, 0, 0))
    assert len(calls) == 1 and p.disarm_replay_capture()["status"] == "captured"
    p.arm_replay_capture(tmp_path / "pending")
    assert p.disarm_replay_capture() == {"status": "pending"}
    p.fetch((0, 0, 0))
    assert len(calls) == 1


def test_failed_capture_is_not_retried_or_reported_success(monkeypatch, tmp_path):
    from beamng_autopilot.experiments import range_replay
    p = _provider()
    calls = []
    def fail(*args, **kwargs):
        calls.append(1)
        raise OSError("disk full")
    monkeypatch.setattr(range_replay, "write_replay", fail)
    p.arm_replay_capture(tmp_path / "capture")
    p.fetch((0, 0, 0))
    p.fetch((0, 0, 0))
    assert calls == [1]
    assert p.disarm_replay_capture() == {"status": "failed", "reason": "disk full"}


def test_disarm_waits_for_async_diagnostic_write(monkeypatch, tmp_path):
    from beamng_autopilot.experiments import range_replay
    p = _provider()
    writing, release, finished = threading.Event(), threading.Event(), threading.Event()
    def save(*args, **kwargs):
        writing.set()
        assert release.wait(5.)
        return {"status": "captured"}
    monkeypatch.setattr(range_replay, "write_replay", save)
    p.arm_replay_capture(tmp_path / "capture")
    worker = threading.Thread(target=lambda: p.fetch((0, 0, 0)))
    worker.start()
    assert writing.wait(5.)
    result = []
    def disarm():
        result.append(p.disarm_replay_capture())
        finished.set()
    stopper = threading.Thread(target=disarm)
    stopper.start()
    assert not finished.is_set()
    release.set()
    worker.join(5.)
    stopper.join(5.)
    assert not worker.is_alive() and not stopper.is_alive()
    assert result == [{"status": "captured"}]


@pytest.mark.parametrize("empty", [True, False])
def test_capture_stays_pending_for_empty_cloud_or_source_error(monkeypatch, tmp_path, empty):
    from beamng_autopilot.experiments import range_replay
    p = _provider(np.empty((0, 3))) if empty else _provider()
    if not empty:
        def fail(*args, **kwargs):
            raise RuntimeError("lua unavailable")
        monkeypatch.setattr(prov, "scan_obstacles_all", fail)
    monkeypatch.setattr(range_replay, "write_replay", lambda *a, **k: pytest.fail("invalid capture"))
    p.arm_replay_capture(tmp_path / "capture")
    p.fetch((0, 0, 0))
    assert p.disarm_replay_capture() == {"status": "pending"}
