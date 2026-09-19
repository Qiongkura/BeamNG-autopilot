"""Latest-wins background jobs for heavy perception heads (plan A4).

Phase A4 of the improvement plan: heavy perception must not block the
control loop, the control loop must always read the newest AVAILABLE
result, and a perception worker may only ever keep the newest task or
result - a pile-up of stale work is worse than no work.

``LatestJobRunner`` gives exactly that, as pure logic around one daemon
thread:

* ``submit()`` never blocks and never waits for the job already running;
  while a job runs, a newer submission REPLACES the pending one (the
  replacement is counted as ``replaced``), so a slow consumer cannot
  make stale work queue up;
* ``poll()`` never blocks - it returns the last FINISHED result once, or
  None.  A caller therefore keeps its own cadence no matter how slow the
  job is, and decides what to do about the age with its own freshness
  policy (the safety monitor's staleness contract is unchanged and stays
  the authority);
* a job that runs longer than ``timeout_s`` is not waited for either -
  the overrun is counted in ``timeouts`` for telemetry, and
  ``in_flight_s()`` reports how long the current job has been running;
* an exception inside a job is captured on its result, never raised into
  the caller (a crashed head must degrade, not take the loop down).

Thread-safety note for this repo: only jobs whose body is pure CPU may
run here.  Anything that touches the BeamNGpy connection (camera grab,
LiDAR poll, control) stays on the caller's thread - the connector's
``io_lock`` serialises that I/O, and moving it into a worker thread would
put socket access on two threads at once.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class JobResult:
    """One finished job: its value, or the error that replaced it.

    ``started_at`` / ``finished_at`` are MONOTONIC seconds
    (``time.perf_counter``) so ``duration_s`` is exact regardless of the
    platform clock resolution; a consumer that needs a wall-clock
    freshness stamp takes it from its own frame/tick clock.
    """

    value: Any = None
    error: str | None = None
    token: int = 0
    started_at: float = 0.0
    finished_at: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def duration_s(self) -> float:
        return max(0.0, float(self.finished_at) - float(self.started_at))


class LatestJobRunner:
    """Run at most one job at a time; the newest submission wins."""

    def __init__(self, name: str, timeout_s: float = 1.5) -> None:
        self.name = str(name)
        self.timeout_s = float(timeout_s)
        self._lock = threading.Lock()
        self._pending: tuple[Callable, tuple, dict, int] | None = None
        self._running = False
        self._done: JobResult | None = None
        self._token = 0
        self._started_at = 0.0
        self._thread: threading.Thread | None = None
        # counters (telemetry / tests)
        self.submitted = 0
        self.replaced = 0
        self.completed = 0
        self.timeouts = 0
        self.errors = 0

    # ------------------------------------------------------------------
    def submit(self, fn: Callable, *args, **kwargs) -> int:
        """Queue ``fn`` as the newest job and return its token.

        Returns immediately; running jobs are never waited for, and a
        job that has not started yet is replaced (nothing queues up).
        """
        with self._lock:
            self._token += 1
            token = self._token
            self.submitted += 1
            if self._running and self._pending is not None:
                self.replaced += 1
            self._pending = (fn, args, kwargs, token)
            if not self._running:
                self._start_locked()
            return token

    def _start_locked(self) -> None:
        job = self._pending
        if job is None:
            return
        self._pending = None
        self._running = True
        self._started_at = time.perf_counter()
        thread = threading.Thread(target=self._run, args=job,
                                  name=f"latest-job-{self.name}",
                                  daemon=True)
        self._thread = thread
        thread.start()

    def _run(self, fn: Callable, args: tuple, kwargs: dict,
             token: int) -> None:
        started = time.perf_counter()
        try:
            value = fn(*args, **kwargs)
            result = JobResult(value=value, token=token,
                               started_at=started,
                               finished_at=time.perf_counter())
        except Exception as exc:
            result = JobResult(error=str(exc) or exc.__class__.__name__,
                               token=token, started_at=started,
                               finished_at=time.perf_counter())
        with self._lock:
            self._done = result
            self._running = False
            self.completed += 1
            if not result.ok:
                self.errors += 1
            if result.duration_s > self.timeout_s:
                self.timeouts += 1
            self._start_locked()

    # ------------------------------------------------------------------
    def poll(self) -> JobResult | None:
        """The last finished result, consumed once.  Never blocks."""
        with self._lock:
            result = self._done
            self._done = None
        return result

    @property
    def busy(self) -> bool:
        with self._lock:
            return bool(self._running)

    @property
    def has_pending(self) -> bool:
        with self._lock:
            return self._pending is not None

    def in_flight_s(self, now: float | None = None) -> float:
        """How long the running job has been running (0 when idle)."""
        with self._lock:
            if not self._running:
                return 0.0
            started = self._started_at
        return max(0.0, (time.perf_counter() if now is None
                         else float(now)) - started)

    def join(self, timeout: float | None = None) -> bool:
        """Wait for the current job (TESTS / shutdown only).

        The live control path must never call this - waiting is exactly
        what the runner exists to avoid.
        """
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout=timeout)
        return not thread.is_alive()

    def digest(self, now: float | None = None) -> dict:
        """JSON-safe health summary for telemetry."""
        return {
            "busy": int(self.busy),
            "in_flight_s": round(self.in_flight_s(now), 3),
            "submitted": int(self.submitted),
            "replaced": int(self.replaced),
            "completed": int(self.completed),
            "timeouts": int(self.timeouts),
            "errors": int(self.errors),
        }
