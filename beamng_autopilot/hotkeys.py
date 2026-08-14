"""Global hotkeys for the in-game autopilot UI (Windows, no third-party deps).

Uses RegisterHotKey + a dedicated background thread with a blocking message
loop, so F-key presses are queued instantly even while the driving loop is
busy inside slow calls (frame grab, simulation step, chart window, ...).
The game window does not need focus for these to fire.

Some games (and fullscreen exclusive modes) swallow F-key presses before
they reach the normal hotkey pipeline, and RegisterHotKey itself can fail
when another app already owns a key.  A second thread polls
GetAsyncKeyState for every bound key as a fallback (edge-triggered with a
debounce), so F9/F10 keep working even when the message-based path does
not deliver.

Bindings (F-keys):
    F8  toggle vision overlay
    F9  toggle autopilot
    F10 grab the in-game navigation route (big map destination)
    F11 clear route
    F12 quit

If a key cannot be registered (e.g. already taken by another app), the
listener falls back to alternates supplied per action, so quitting always
works.
"""

from __future__ import annotations

import ctypes
import queue
import threading
import time
from ctypes import wintypes

USER32 = ctypes.windll.user32
KERNEL32 = ctypes.windll.kernel32

WM_HOTKEY = 0x0312
WM_QUIT = 0x0012
MOD_NOREPEAT = 0x4000
MOD_CONTROL = 0x0002
MOD_ALT = 0x0001
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008

# RegisterHotKey modifier flag -> the virtual key to poll for it.
_MOD_VK = {MOD_ALT: 0x12, MOD_CONTROL: 0x11, MOD_SHIFT: 0x10, MOD_WIN: 0x5B}

VK_F8 = 0x77
VK_F9 = 0x78
VK_F10 = 0x79
VK_F11 = 0x7A
VK_F12 = 0x7B
VK_ESCAPE = 0x1B
VK_Q = 0x51


class MSG(ctypes.Structure):
    _fields_ = [
        ("hwnd", wintypes.HWND),
        ("message", wintypes.UINT),
        ("wParam", wintypes.WPARAM),
        ("lParam", wintypes.LPARAM),
        ("time", wintypes.DWORD),
        ("pt", wintypes.POINT),
    ]


class HotkeyListener:
    """Registers global F-key hotkeys and queues presses from a background thread."""

    def __init__(self, bindings: dict[int, str] | None = None,
                 alternates: dict[int, str] | None = None,
                 modifier_alternates: dict[tuple[int, int], str] | None = None):
        """Start the hotkey listener.

        `bindings`: primary key -> action name.
        `alternates`: extra (key, action) pairs tried when the primary key
            for that action is already taken.
        `modifier_alternates`: extra ((modifiers, key), action) pairs tried
            as a last resort (e.g. Ctrl+Q for quit).
        """
        self.bindings = bindings or {
            VK_F8: "vision",
            VK_F9: "autopilot",
            VK_F10: "navroute",
            VK_F11: "clear",
            VK_F12: "quit",
        }
        self.alternates = alternates or {}
        self.modifier_alternates = modifier_alternates or {}
        self._id_to_name: dict[int, str] = {}
        self._registered: list[int] = []
        self._fired: queue.Queue[str] = queue.Queue()
        self._last_emit: dict[str, float] = {}
        self._emit_lock = threading.Lock()
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._error: Exception | None = None
        self._thread_id: int | None = None
        self._thread = threading.Thread(
            target=self._run, name="hotkeys", daemon=True)
        self._poller = threading.Thread(
            target=self._poll_loop, name="hotkeys-poll", daemon=True)
        self._thread.start()
        self._poller.start()
        if not self._ready.wait(timeout=2.0):
            print("[hotkeys] listener thread did not become ready in time")
        if self._error is not None:
            raise self._error

    # ---- background thread -------------------------------------------------

    def _run(self) -> None:
        self._thread_id = int(KERNEL32.GetCurrentThreadId())
        try:
            self._register_all()
        except Exception as exc:
            self._error = exc
            self._ready.set()
            return
        self._ready.set()
        msg = MSG()
        while not self._stop.is_set():
            # Blocking call: WM_HOTKEY is posted to this thread's queue because
            # RegisterHotKey ran on this thread.
            r = USER32.GetMessageW(ctypes.byref(msg), None, 0, 0)
            if r <= 0:  # -1 = error, 0 = WM_QUIT
                break
            if msg.message == WM_HOTKEY:
                name = self._id_to_name.get(int(msg.wParam))
                if name is not None:
                    self._emit(name)
        # UnregisterHotKey must run on the thread that called RegisterHotKey.
        for hotkey_id in self._registered:
            USER32.UnregisterHotKey(None, hotkey_id)
        self._registered.clear()
        self._id_to_name.clear()

    def _register_all(self) -> None:
        failed: set[str] = set()
        # Primary keys first; an action's alternates are only tried when the
        # primary key for that action is already taken by another app.
        candidates: list[tuple[int, int, str]] = [
            (MOD_NOREPEAT, vk, name) for vk, name in self.bindings.items()
        ]
        for mods, vk, name in candidates:
            hotkey_id = 0x7000 + vk
            if hotkey_id in self._id_to_name:
                continue  # same key already bound to an action
            ok = USER32.RegisterHotKey(None, hotkey_id, mods, vk)
            if ok:
                self._registered.append(hotkey_id)
                self._id_to_name[hotkey_id] = name
            else:
                print(f"[hotkeys] failed to register VK=0x{vk:X} "
                      f"(action '{name}')")
                failed.add(name)
        for vk, name in self.alternates.items():
            if name not in failed:
                continue
            hotkey_id = 0x7000 + vk
            if hotkey_id in self._id_to_name:
                continue
            if USER32.RegisterHotKey(None, hotkey_id, MOD_NOREPEAT, vk):
                self._registered.append(hotkey_id)
                self._id_to_name[hotkey_id] = name
        for (mods, vk), name in self.modifier_alternates.items():
            if name not in failed:
                continue
            hotkey_id = 0x7000 + vk
            if hotkey_id in self._id_to_name:
                continue
            if USER32.RegisterHotKey(None, hotkey_id,
                                     MOD_NOREPEAT | mods, vk):
                self._registered.append(hotkey_id)
                self._id_to_name[hotkey_id] = name
        if not self._registered:
            raise RuntimeError("no hotkeys could be registered")

    def _poll_loop(self) -> None:
        """GetAsyncKeyState fallback: edge-triggered, debounced polling."""
        keys: list[tuple[int, int, str]] = []
        seen: set[int] = set()
        for vk, name in self.bindings.items():
            if vk not in seen:
                keys.append((0, vk, name))
                seen.add(vk)
        for vk, name in self.alternates.items():
            if vk not in seen:
                keys.append((0, vk, name))
                seen.add(vk)
        for (mods, vk), name in self.modifier_alternates.items():
            if vk not in seen:
                keys.append((mods, vk, name))
                seen.add(vk)
        if not keys:
            return
        down: dict[int, bool] = {vk: False for _, vk, _ in keys}
        while not self._stop.is_set():
            time.sleep(0.015)
            for mods, vk, name in keys:
                pressed = bool(USER32.GetAsyncKeyState(vk) & 0x8000)
                if pressed and mods:
                    for flag, mod_vk in _MOD_VK.items():
                        if flag & mods and not (
                                USER32.GetAsyncKeyState(mod_vk) & 0x8000):
                            pressed = False
                            break
                if pressed and not down[vk]:
                    self._emit(name)
                down[vk] = pressed

    def _emit(self, name: str) -> None:
        """Queue an action, deduplicating near-simultaneous detections."""
        now = time.time()
        with self._emit_lock:
            last = self._last_emit.get(name, 0.0)
            # The message loop and the poller can both detect the same
            # physical press within ~15-30 ms of each other; dedupe only
            # that window so a fast double-tap (e.g. F9 twice to toggle on
            # and off) is not swallowed.
            if now - last < 0.06:
                return
            self._last_emit[name] = now
        self._fired.put(name)

    # ---- public API ---------------------------------------------------------

    def pump(self) -> list[str]:
        """Return the names of all hotkeys pressed since the last call."""
        fired: list[str] = []
        while True:
            try:
                fired.append(self._fired.get_nowait())
            except queue.Empty:
                break
        return fired

    def close(self) -> None:
        self._stop.set()
        if self._thread_id is not None:
            USER32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)
        for thread in (self._thread, self._poller):
            if thread.is_alive():
                thread.join(timeout=2.0)
        self._registered.clear()
