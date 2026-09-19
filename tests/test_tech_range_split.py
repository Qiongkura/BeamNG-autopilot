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
from beamng_autopilot.perception import LidarClusterTracker
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
