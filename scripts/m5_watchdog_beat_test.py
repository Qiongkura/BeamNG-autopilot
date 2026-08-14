"""Regression test for the F9 watchdog freeze / handbrake-takeover bug.

The autopilot's game-side watchdog times out after 2.5 s.  F9 gearbox setup
(brake to a standstill + forward-gear probes) can take several seconds, and
the main loop cannot beat while it is synchronously inside that setup.  The
fix adds a daemon that beats while the setup phase is active - but only if
the long operation releases ``conn.io_lock`` per socket call instead of
holding it across the whole operation.

This test reproduces the lock pattern with a fake connector and asserts:
* per-call locking (the fixed path) lets the daemon beat repeatedly while
  the main thread is inside a 5 s blocking phase, with every beat gap
  below the game watchdog's 2.5 s timeout;
* holding the lock for the whole phase (the old bug) starves the daemon.

No BeamNG instance is required.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class _FakeConnector:
    def __init__(self) -> None:
        self.io_lock = threading.RLock()
        self.heartbeats = 0
        self.beat_times: list[float] = []

    def _beat(self) -> None:
        self.heartbeats += 1
        self.beat_times.append(time.time())


def _daemon(fake: _FakeConnector, stop: threading.Event,
            state: dict) -> None:
    """Same shape/timing as m5_autopilot's _wd_beat_daemon (0.5 s)."""
    while not stop.wait(0.5):
        if not state["active"]:
            continue
        if time.time() - state["since"] > 20.0:
            continue
        try:
            with fake.io_lock:
                # Re-check after acquiring: in the outer-lock case the daemon
                # can otherwise win the teardown race and beat once after the
                # phase has already ended.
                if not state["active"]:
                    continue
                fake._beat()
        except Exception:
            pass


def _long_blocking_op(fake: _FakeConnector, hold_outer: bool,
                      ticks: int, step_s: float) -> tuple[int, float]:
    """Simulate forward_gear_input(); returns (beats, max beat gap)."""
    state = {"active": True, "since": time.time()}
    stop = threading.Event()
    thread = threading.Thread(target=_daemon, args=(fake, stop, state),
                              daemon=True)
    thread.start()
    try:
        if hold_outer:
            with fake.io_lock:
                for _ in range(ticks):
                    time.sleep(step_s)
        else:
            for _ in range(ticks):
                with fake.io_lock:
                    time.sleep(step_s)
    finally:
        state["active"] = False
        stop.set()
        thread.join(timeout=1.0)
    times = fake.beat_times
    gaps = [b - a for a, b in zip(times, times[1:])] if len(times) > 1 else []
    max_gap = max(gaps) if gaps else 0.0
    return fake.heartbeats, max_gap


def _find_window_class_lookup() -> bool:
    """Window discovery must not take io_lock or read every window title."""
    from beamng_autopilot import connector as conn_mod

    class _FakeUser32:
        def __init__(self) -> None:
            self.conn = None
            self.find = True
            self.enum_calls = 0

        def FindWindowW(self, cls, _title):
            if self.conn is not None:
                got = self.conn.io_lock.acquire(blocking=False)
                if not got:
                    raise AssertionError("io_lock held during window lookup")
                self.conn.io_lock.release()
            return 0x1234 if self.find else 0

        def IsWindowVisible(self, _hwnd):
            return True

        def GetClassNameW(self, hwnd, buf, _n):
            buf.value = "GameEngineMainWindow"
            return len(buf.value)

        def GetWindowRect(self, _hwnd, rect):
            rect = getattr(rect, "_obj", rect)
            rect.left, rect.top = 10, 20
            rect.right, rect.bottom = 810, 620
            return True

        def EnumWindows(self, cb, _):
            self.enum_calls += 1
            cb(0x1234, 0)
            return True

        def GetWindowTextLengthW(self, *_args):
            raise AssertionError("window title read during discovery")

    fake = _FakeUser32()
    orig = conn_mod.ctypes.windll.user32
    conn_mod.ctypes.windll.user32 = fake
    try:
        conn = conn_mod.BeamNGConnector()
        fake.conn = conn
        hwnd, rect = conn._find_window(use_cache=False)
        exact = hwnd == 0x1234 and rect == (10, 20, 810, 620)

        conn._window_cache = (0x1234, (10, 20, 810, 620), time.time())
        hwnd2, rect2 = conn._find_window()
        cached = hwnd2 == 0x1234 and rect2 == (10, 20, 810, 620)

        fake.find = False
        hwnd3, _rect3 = conn._find_window(use_cache=False)
        fallback = hwnd3 == 0x1234 and fake.enum_calls == 1
        got_lock = conn.io_lock.acquire(blocking=False)
        lock_free = got_lock
        if got_lock:
            conn.io_lock.release()
    finally:
        conn_mod.ctypes.windll.user32 = orig
    return exact and cached and fallback and lock_free


def main() -> int:
    per_call = _long_blocking_op(_FakeConnector(), hold_outer=False,
                                 ticks=100, step_s=0.05)
    outer_hold = _long_blocking_op(_FakeConnector(), hold_outer=True,
                                   ticks=100, step_s=0.05)
    per_call, per_call_gap = per_call
    outer_hold, outer_gap = outer_hold
    # 5 s blocking phase at a 0.5 s daemon interval: the watchdog (2.5 s)
    # must never see a heartbeat gap near its timeout.
    ok = per_call >= 6
    ok = ok and per_call_gap <= 2.4
    ok = ok and outer_hold == 0
    ok = ok and outer_gap == 0.0
    window_ok = _find_window_class_lookup()
    ok = ok and window_ok
    print(f"[watchdog-beat] per-call lock beats={per_call} "
          f"(expect >= 6) max_gap={per_call_gap:.2f}s (expect <= 2.4s)")
    print(f"[watchdog-beat] outer lock beats={outer_hold} "
          f"(expect 0)")
    print(f"[window-finder] class lookup, cache, no io_lock, "
          f"no title reads -> {'PASS' if window_ok else 'FAIL'}")
    print("RESULT: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
