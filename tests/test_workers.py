"""Latest-wins perception worker contract (plan phase A4).

The load-bearing property: a slow (or hung) heavy-perception job must
never block, delay or crash the caller - the control loop keeps its own
cadence and reads whatever result is newest.
"""

from __future__ import annotations

import threading
import time

from beamng_autopilot.workers import LatestJobRunner


def _slow(value, seconds: float, out: dict | None = None):
    def _job():
        time.sleep(seconds)
        if out is not None:
            out["ran"] = out.get("ran", 0) + 1
        return value
    return _job


def test_perception_overrun_never_blocks_the_caller() -> None:
    """The core A4 contract, measured in the caller's own clock.

    A 400 ms job must not cost the caller 400 ms: submit returns at once
    and a poll loop keeps running while the head is still working.
    """
    runner = LatestJobRunner("semantic", timeout_s=0.2)
    runner.submit(_slow("frame-a", 0.4))
    t0 = time.perf_counter()
    elapsed = 0.0
    polls = 0
    while elapsed < 0.2:
        assert runner.poll() is None        # nothing finished yet
        polls += 1
        elapsed = time.perf_counter() - t0
    assert polls > 100, f"poll loop starved: {polls} polls"
    assert elapsed < 0.35, f"caller blocked for {elapsed:.3f}s"
    assert runner.busy is True
    runner.join(timeout=2.0)


def test_result_is_delivered_once_then_consumed() -> None:
    runner = LatestJobRunner("object")
    runner.submit(_slow("boxes", 0.01))
    assert runner.join(timeout=2.0)
    res = runner.poll()
    assert res is not None and res.ok and res.value == "boxes"
    assert runner.poll() is None            # consumed exactly once


def test_newer_job_replaces_a_pending_one() -> None:
    """Latest wins: stale work must not queue up behind a slow job."""
    ran: dict = {}
    runner = LatestJobRunner("object")
    runner.submit(_slow("first", 0.25, ran))
    # while the first runs, two newer submissions arrive
    runner.submit(_slow("second", 0.0, ran))
    runner.submit(_slow("third", 0.0, ran))
    assert runner.replaced >= 1
    assert runner.join(timeout=2.0)
    values = []
    for _ in range(20):
        res = runner.poll()
        if res is not None:
            values.append(res.value)
        if not runner.busy and not runner.has_pending:
            break
        time.sleep(0.02)
    # the caller always reads the NEWEST result (the single result slot
    # is by design - an older finished result may be superseded)
    assert values and values[-1] == "third"
    # the replaced middle job never ran; the newest one did
    assert ran.get("ran") == 2, ran


def test_job_exception_is_captured_not_raised() -> None:
    def _boom():
        raise RuntimeError("head exploded")

    runner = LatestJobRunner("semantic")
    runner.submit(_boom)
    assert runner.join(timeout=2.0)
    res = runner.poll()
    assert res is not None
    assert res.ok is False
    assert "head exploded" in (res.error or "")
    assert runner.errors == 1


def test_overrun_is_counted_not_waited_for() -> None:
    runner = LatestJobRunner("semantic", timeout_s=0.05)
    runner.submit(_slow("late", 0.15))
    assert runner.busy is True
    assert runner.in_flight_s() > 0.0
    assert runner.join(timeout=2.0)
    runner.poll()
    assert runner.timeouts == 1
    assert runner.completed == 1


def test_idle_runner_costs_nothing() -> None:
    runner = LatestJobRunner("object")
    t0 = time.perf_counter()
    for _ in range(1000):
        assert runner.poll() is None
    assert time.perf_counter() - t0 < 0.1
    assert runner.digest()["busy"] == 0
    assert runner.in_flight_s() == 0.0


def test_digest_is_json_safe() -> None:
    import json
    runner = LatestJobRunner("object")
    runner.submit(_slow("x", 0.01))
    runner.join(timeout=2.0)
    runner.poll()
    text = json.dumps(runner.digest())
    assert "replaced" in text


def test_caller_thread_is_never_the_worker_thread() -> None:
    """Jobs run off the caller's thread - the whole point of the module."""
    seen: dict = {}
    here = threading.current_thread().name

    def _job():
        seen["thread"] = threading.current_thread().name
        return 1

    runner = LatestJobRunner("object")
    runner.submit(_job)
    runner.join(timeout=2.0)
    assert seen["thread"] != here
