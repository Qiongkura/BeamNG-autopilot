"""BeamNG.drive 会话封装：连接、场景加载、车辆状态与控制。"""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass
from typing import Optional

import ctypes
import ctypes.wintypes
import numpy as np
from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.misc.quat import angle_to_quat
from pathlib import Path

from . import config
from .traffic import RoadRuleView, SignalRule


_WINDOW_CLASS_NAMES = (
    "GameEngineMainWindow",
    "BeamNG.drive",
    "BeamNG.drive.x64",
)
_WINDOW_CACHE_TTL = 2.0


@dataclass
class VehicleState:
    pos: np.ndarray  # (3,) 世界坐标
    dir: np.ndarray  # (3,) 前向单位向量
    up: np.ndarray  # (3,)
    vel: np.ndarray  # (3,) 速度
    rotation: np.ndarray  # (4,) 四元数 xyzw

    @property
    def heading(self) -> float:
        """世界系航向角（弧度）。"""
        return float(np.arctan2(self.dir[1], self.dir[0]))

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.vel))


class BeamNGConnector:
    """管理一次与 BeamNG.drive 的完整会话（启动 -> 建场景 -> 控制 -> 退出）。"""

    def __init__(
        self,
        map_name: str = config.DEFAULT_MAP,
        vehicle_model: str = config.DEFAULT_VEHICLE,
        host: str = config.HOST,
        port: int = config.PORT,
        home=config.BEAMNG_HOME,
        user=None,
        steps_per_second: int = 60,
    ):
        self.map_name = map_name
        self.vehicle_model = vehicle_model
        self.sps = steps_per_second
        if user is None:
            user = config.runtime_user(
                "tech" if Path(home).resolve()
                == config.BEAMNG_TECH_HOME.resolve() else "steam")
        self.bng = BeamNGpy(host, port, home=str(home), user=str(user))
        self.user_dir = Path(user)
        self.scenario: Optional[Scenario] = None
        self.vehicle: Optional[Vehicle] = None
        # beamngpy's Connection is not safe for concurrent request/response
        # cycles: two threads can interleave a send with the other's recv and
        # both block forever.  Every socket call in the connector and every
        # direct bng/vehicle call in the autopilot uses this lock.
        self.io_lock = threading.RLock()
        self._window_cache: tuple | None = None

    def __enter__(self):
        return self.open(launch=True)

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def open(self, launch: bool = True):
        """Launch (or attach to) the simulator and prepare the connection."""
        with self.io_lock:
            if launch and self._port_is_open(self.bng.host, self.bng.port):
                # A game is already listening on the port.  beamngpy's
                # open(launch=True) treats ONE failed hello as "no instance"
                # and spawns a second game, which then cannot bind the port
                # and the whole session dies.  Attach to the running
                # instance instead, retrying the handshake a few times (the
                # game can drop a hello while busy); only a genuinely closed
                # port goes down the launch path.
                last_exc: Exception | None = None
                for _ in range(3):
                    try:
                        self.bng.open(launch=False)
                        last_exc = None
                        break
                    except Exception as exc:
                        last_exc = exc
                        time.sleep(1.0)
                if last_exc is not None:
                    raise RuntimeError(
                        f"game at {self.bng.host}:{self.bng.port} is "
                        "listening but not answering the RPC handshake "
                        "(stale session or mid-load); restart the game "
                        "and retry") from last_exc
            else:
                self.bng.open(launch=launch)
            self.bng.set_steps_per_second(self.sps)
        return self

    @staticmethod
    def _port_is_open(host: str, port: int, timeout: float = 0.5) -> bool:
        """True when something is listening on the game port."""
        import socket

        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            return False

    def close(self) -> None:
        try:
            if self.bng is not None:
                # bng.close() sends quit_beamng() even in attach mode (the
                # process handle is None, so it takes the else branch and
                # quits the user's game). disconnect() only drops the socket
                # and leaves the game running.
                with self.io_lock:
                    self.bng.disconnect()
        except Exception as exc:  # 关闭阶段异常只记录，不阻断
            print(f"[close] {exc}")

    def load_scenario(self, spawn_pos=None, spawn_heading: Optional[float] = None):
        """加载地图并生成车辆，等待场景正式开始。

        未指定出生点时使用地图默认出生点；目前 italy 使用
        ``spawn_crossroads``（十字路口）。
        """
        if spawn_pos is None:
            if self.map_name == "italy":
                spawn_pos = config.ITALY_SPAWN_CROSSROADS_POS
                if spawn_heading is None:
                    spawn_heading = config.ITALY_SPAWN_CROSSROADS_HEADING
            else:
                spawn_pos = (0.0, 0.0, 0.0)
        if spawn_heading is None:
            spawn_heading = 0.0
        with self.io_lock:
            scenario = Scenario(self.map_name, "autopilot_m1")
            vehicle = Vehicle("ego", model=self.vehicle_model, color="Red")
        # 世界系航向角 h（atan2(dy,dx) 约定）与 BeamNG 的 yaw 换算（实测
        # 于 2026-08-15 十字路口实验，四个方向全部验证）：
        # teleport/场景里 yaw 按顺时针应用，yaw(deg) = -degrees(h) - 90。
        yaw_deg = -math.degrees(float(spawn_heading)) - 90.0
        quat = angle_to_quat((0.0, 0.0, yaw_deg))
        scenario.add_vehicle(vehicle, pos=spawn_pos, rot_quat=quat, cling=True)
        with self.io_lock:
            scenario.make(self.bng)
            self.bng.scenario.load(scenario)
            self.bng.scenario.start()
        self.scenario = scenario
        self.vehicle = vehicle
        return vehicle

    def attach_vehicle(self, vid: Optional[str] = None,
                       already_open: bool = False):
        """Attach to a vehicle already present in a running BeamNG session
        (the user entered a map manually and placed a vehicle).

        Connects without launching the simulator and picks the first active
        vehicle unless a specific vid is requested. Pass ``already_open=True``
        when the connection was probed beforehand to avoid re-opening it.
        """
        with self.io_lock:
            if not already_open:
                self.bng.open(launch=False)
                self.bng.set_steps_per_second(self.sps)
            vehicles = self.bng.get_current_vehicles()
            if not vehicles:
                raise RuntimeError("No vehicles in the current scenario; "
                                   "place a vehicle in the map first")
            if vid is not None and vid not in vehicles:
                raise RuntimeError(
                    f"Vehicle '{vid}' not found; available: {sorted(vehicles)}")
            target_vid = vid if vid is not None else next(iter(vehicles))
            vehicle = vehicles[target_vid]
            vehicle.connect(self.bng)
            self.vehicle = vehicle
            self.scenario = None
        print(f"[attach] vehicle '{target_vid}' attached")
        return vehicle

    def current_env(self) -> dict:
        """Return the actual map / vehicle of the running session.

        Queries the live scenario for the level (map) name and the ego
        vehicle model so the GUI settings can auto-fetch them instead of
        trusting the launch args (attach mode always uses the running map).
        Falls back to the launch args when the query fails.
        """
        with self.io_lock:
            env = {"map": self.map_name, "vehicle": self.vehicle_model}
            try:
                scenario = self.bng.scenario.get_current(connect=False)
                lvl = getattr(scenario, "level", None)
                if hasattr(lvl, "name"):
                    env["map"] = lvl.name
                elif isinstance(lvl, str) and lvl:
                    env["map"] = lvl
                elif getattr(scenario, "name", None):
                    env["map"] = scenario.name
                self._env_fail_printed = False
            except Exception as exc:
                if not getattr(self, "_env_fail_printed", False):
                    self._env_fail_printed = True
                    print(f"[env] current map query failed: {exc}")
            if self.vehicle is not None:
                model = getattr(self.vehicle, "model", None)
                if model:
                    env["vehicle"] = model
        return env

    def reposition_on_road(self, roadnet, lift: float = 0.6,
                           settle: int = 30) -> bool:
        """Teleport the ego vehicle onto the nearest road-network node.

        Scenarios spawned at (0, 0, 0) can leave the car below terrain when
        the map origin is not on a road (e.g. hirochi_raceway); snapping the
        car to a road node fixes that so autopilot routes stay on the roads.
        """
        if not roadnet.ready:
            print("[reposition] roadnet not ready; vehicle left in place")
            return False
        if roadnet.node_count < config.ROADNET_REPOSITION_MIN_NODES:
            print("[reposition] road network still loading "
                  f"({roadnet.node_count} nodes); vehicle left in place")
            return False
        try:
            st = self.get_state()
        except Exception as exc:
            print(f"[reposition] state read failed: {exc}")
            return False
        xyz = roadnet.nearest_node_xyz(st.pos[:2])
        if xyz is None:
            print("[reposition] no road height data; vehicle left in place")
            return False
        dist = math.hypot(st.pos[0] - xyz[0], st.pos[1] - xyz[1])
        if dist <= config.ROADNET_REPOSITION_NEAR_M:
            print(f"[reposition] vehicle already on road "
                  f"({dist:.1f}m); left in place")
            return False
        # Align the car with the road axis so it doesn't spawn pointing into
        # a wall/barrier on maps whose roads are not axis-aligned.
        heading = roadnet.road_heading_at(st.pos[:2])
        rot_quat = None
        if heading is not None:
            yaw_deg = -math.degrees(float(heading)) - 90.0
            rot_quat = angle_to_quat((0.0, 0.0, yaw_deg))
        pos = (float(xyz[0]), float(xyz[1]), float(xyz[2]) + lift)
        try:
            with self.io_lock:
                self.vehicle.teleport(pos, rot_quat=rot_quat)
                self.step(settle)
                after = self.get_state()
            print(f"[reposition] vehicle -> ({pos[0]:.1f}, {pos[1]:.1f}, "
                  f"{after.pos[2]:.1f}) heading={math.degrees(after.heading):.0f}deg")
            return True
        except Exception as exc:
            print(f"[reposition] failed: {exc}")
            return False

    def set_front_camera(self, pos=(0.0, 1.2, 1.4), direction=(0.0, -1.0, 0.0)):
        """Switch the in-game camera to a fixed point ahead of the ego vehicle.
        Works without a BeamNG.tech license (Steam edition)."""
        with self.io_lock:
            self.bng.set_relative_camera(pos=tuple(pos), dir=tuple(direction))

    def _find_window(self, use_cache: bool = True):
        """Find the BeamNG game window. Returns (hwnd, rect) or (None, None).

        The game window is looked up by class name first (a single Win32
        call), with an EnumWindows fallback that only inspects class names.
        No per-window GetWindowText round-trips are made, and no io_lock is
        taken, so a slow/hung desktop window can never stall the socket
        heartbeat.  The handle is cached briefly and the rect is refreshed
        cheaply on each call.
        """
        now = time.time()
        user32 = ctypes.windll.user32
        if use_cache and self._window_cache is not None:
            hwnd, _, ts = self._window_cache
            if now - ts < _WINDOW_CACHE_TTL:
                rect = self._window_rect(user32, hwnd)
                if rect is not None:
                    self._window_cache = (hwnd, rect, now)
                    return hwnd, rect
                self._window_cache = None

        best_hwnd = None
        best_rect = None
        best_area = 0

        def consider(hwnd) -> None:
            nonlocal best_hwnd, best_rect, best_area
            if not user32.IsWindowVisible(hwnd):
                return
            buf = ctypes.create_unicode_buffer(256)
            n = user32.GetClassNameW(hwnd, buf, 256)
            if n == 0 or buf.value not in _WINDOW_CLASS_NAMES:
                return
            rect = self._window_rect(user32, hwnd)
            if rect is None:
                return
            area = (rect[2] - rect[0]) * (rect[3] - rect[1])
            if area > best_area:
                best_hwnd = hwnd
                best_rect = rect
                best_area = area

        for cls in _WINDOW_CLASS_NAMES:
            hwnd = user32.FindWindowW(cls, None)
            if hwnd:
                consider(hwnd)
                if best_hwnd is not None:
                    break

        if best_hwnd is None:
            @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p,
                                ctypes.c_void_p)
            def enum_callback(hwnd, _):
                consider(hwnd)
                return True

            user32.EnumWindows(enum_callback, 0)

        if best_hwnd is not None:
            self._window_cache = (best_hwnd, best_rect, time.time())
        return best_hwnd, best_rect

    @staticmethod
    def _window_rect(user32, hwnd):
        """Return (left, top, right, bottom) for a valid game window."""
        rect = ctypes.wintypes.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return None
        w = rect.right - rect.left
        h = rect.bottom - rect.top
        if w < 100 or h < 100:
            return None
        return rect.left, rect.top, rect.right, rect.bottom

    def grab_screen(self) -> np.ndarray:
        """Grab a front-view frame from BeamNG.

        Uses PrintWindow when a game window exists (fast, works when
        occluded); otherwise asks the game's built-in screenshot extension
        to write a PNG and reads it back.  Returns an RGB (H, W, 3) uint8
        array.
        """
        hwnd, rect = self._find_window()
        if hwnd is None:
            return self._grab_lua_screenshot()
        user32 = ctypes.windll.user32
        gdi32 = ctypes.windll.gdi32

        left, top, right, bottom = rect
        w = right - left
        h = bottom - top

        hwnd_dc = user32.GetWindowDC(hwnd)
        mem_dc = gdi32.CreateCompatibleDC(hwnd_dc)
        bmp = gdi32.CreateCompatibleBitmap(hwnd_dc, w, h)
        old = gdi32.SelectObject(mem_dc, bmp)

        try:
            # PW_RENDERFULLCONTENT = 0x00000002
            user32.PrintWindow(hwnd, mem_dc, 0x00000002)

            class BITMAPINFOHEADER(ctypes.Structure):
                _fields_ = [
                    ("biSize", ctypes.c_uint32),
                    ("biWidth", ctypes.c_int32),
                    ("biHeight", ctypes.c_int32),
                    ("biPlanes", ctypes.c_uint16),
                    ("biBitCount", ctypes.c_uint16),
                    ("biCompression", ctypes.c_uint32),
                    ("biSizeImage", ctypes.c_uint32),
                    ("biXPelsPerMeter", ctypes.c_int32),
                    ("biYPelsPerMeter", ctypes.c_int32),
                    ("biClrUsed", ctypes.c_uint32),
                    ("biClrImportant", ctypes.c_uint32),
                ]

            bmi = BITMAPINFOHEADER()
            bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
            bmi.biWidth = w
            bmi.biHeight = -h  # top-down
            bmi.biPlanes = 1
            bmi.biBitCount = 32
            bmi.biCompression = 0  # BI_RGB

            buf = ctypes.create_string_buffer(w * h * 4)
            got = gdi32.GetDIBits(
                mem_dc, bmp, 0, h, buf, ctypes.byref(bmi), 0  # DIB_RGB_COLORS
            )
            if got == 0:
                raise RuntimeError("GetDIBits failed")
        finally:
            gdi32.SelectObject(mem_dc, old)
            gdi32.DeleteObject(bmp)
            gdi32.DeleteDC(mem_dc)
            user32.ReleaseDC(hwnd, hwnd_dc)

        arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
        return np.ascontiguousarray(arr[:, :, 2::-1])  # BGRA -> RGB

    def _grab_lua_screenshot(self, timeout: float = 15.0) -> np.ndarray:
        """Fallback screen capture via BeamNG's built-in screenshot Lua
        extension.  Used on headless / hidden-window sessions where
        PrintWindow has no window handle to copy."""
        import time

        import cv2

        if self.bng is None:
            raise RuntimeError("BeamNG connection unavailable for screenshot")
        rel = f"logs/codex_screen_{int(time.time() * 1000)}"
        try:
            with self.io_lock:
                self.bng.queue_lua_command(
                    "extensions.load('ge/screenshot')\n"
                    f"extensions.ge_screenshot.doScreenshot("
                    f"nil, nil, '{rel}', 'png')\n"
                    "return 'queued'", response=True)
        except Exception as exc:
            raise RuntimeError(f"Lua screenshot queue failed: {exc}") from exc

        candidates = [
            self.user_dir / "current" / f"{rel}.png",
            self.user_dir / f"{rel}.png",
        ]
        deadline = time.time() + timeout
        last_size = -1
        last_seen = 0.0
        path = None
        while time.time() < deadline:
            for cand in candidates:
                try:
                    if cand.is_file() and cand.stat().st_size > 10_000:
                        path = cand
                        break
                except OSError:
                    continue
            if path is not None:
                try:
                    size = path.stat().st_size
                    if size == last_size and time.time() - last_seen >= 0.4:
                        break
                    if size != last_size:
                        last_size = size
                        last_seen = time.time()
                except OSError:
                    pass
            time.sleep(0.2)

        if path is None:
            raise RuntimeError(
                "Lua screenshot not found in "
                + ", ".join(str(c) for c in candidates))
        img = cv2.imread(str(path))
        if img is None or img.size == 0:
            raise RuntimeError(f"Lua screenshot unreadable: {path}")
        return np.ascontiguousarray(img[:, :, ::-1])  # BGR -> RGB

    def get_state(self) -> VehicleState:
        with self.io_lock:
            self.vehicle.sensors.poll()
            st = self.vehicle.state
            return VehicleState(
                pos=np.asarray(st["pos"], dtype=float),
                dir=np.asarray(st["dir"], dtype=float),
                up=np.asarray(st["up"], dtype=float),
                vel=np.asarray(st["vel"], dtype=float),
                rotation=np.asarray(st["rotation"], dtype=float),
            )

    def get_wheel_speed(self):
        """Best-effort average wheel speed (m/s) via the Lua bridge.

        ``electrics.values.wheelspeed`` is a plain game-side value, so it
        works on the Steam build without a tech license.  The chunk runs
        on the vehicle-scoped Lua channel (``QueueLuaCommandVE``), where
        the vehicle's own ``electrics`` table is in scope; the engine
        channel no longer exposes ``getVehicleByID()`` in current builds.
        Returns None when the value is unavailable so callers degrade
        gracefully.
        """
        with self.io_lock:
            if self.vehicle is None:
                return None
            try:
                resp = self.vehicle.queue_lua_command(
                    "local ws = electrics.values.wheelspeed\n"
                    "if ws == nil then return -1 end\n"
                    "return ws", response=True)
                val = float(resp)
                return val if math.isfinite(val) and val >= 0.0 else None
            except Exception:
                # Legacy builds without the vehicle-scoped channel:
                # engine-scope lookup, guarded so the removed global never
                # raises.
                try:
                    vid = self.vehicle.vid
                    resp = self.bng.queue_lua_command(
                        "local f = getVehicleByID\n"
                        "if type(f) == 'function' then\n"
                        "  local v = f('%s')\n"
                        "  if v and v.electrics and v.electrics.values then\n"
                        "    return v.electrics.values.wheelspeed\n"
                        "  end\n"
                        "end\n"
                        "return -1" % vid, response=True)
                    val = float(resp)
                    return val if math.isfinite(val) and val >= 0.0 else None
                except Exception:
                    return None

    def control(
        self,
        throttle: float = 0.0,
        steering: float = 0.0,
        brake: float = 0.0,
        gear=None,
        parkingbrake=None,
        clutch=None,
    ) -> None:
        with self.io_lock:
            kwargs = dict(throttle=float(throttle), steering=float(steering),
                          brake=float(brake))
            if gear is not None:
                kwargs["gear"] = int(gear)
            if parkingbrake is not None:
                kwargs["parkingbrake"] = float(parkingbrake)
            if clutch is not None:
                kwargs["clutch"] = float(clutch)
            self.vehicle.control(**kwargs)

    def step(self, count: int = 1, wait: bool = True) -> None:
        with self.io_lock:
            self.bng.step(count, wait=wait)

    def ai_set_line(self, points, speed: float, cling: bool = True) -> None:
        """让游戏内 AI 沿折线行驶（用于录制参考轨迹）。"""
        line = [{"pos": (float(x), float(y), float(z)), "speed": float(speed)} for x, y, z in points]
        with self.io_lock:
            self.vehicle.ai.set_line(line, cling=cling)

    def ai_set_speed(self, speed: float, mode: str = "limit") -> None:
        with self.io_lock:
            self.vehicle.ai.set_speed(speed, mode=mode)

    def ai_disable(self) -> None:
        with self.io_lock:
            self.vehicle.ai.set_mode("disabled")

    def read_navigation_route(self) -> Optional[np.ndarray]:
        """Read the navigation route set on the in-game big map.

        BeamNG exposes the active navigation route (blue arrows / ground
        markers) through ``core_groundMarkers.routePlanner.path``: a list of
        world-space points ordered from the vehicle to the destination. This
        runs a tiny Lua chunk over the comms channel and returns an (N, 3)
        float array, or None when no navigation route is currently active.
        """
        with self.io_lock:
            chunk = (
                "local pts = {}\n"
                "local rp = core_groundMarkers and "
                "core_groundMarkers.routePlanner\n"
                "if rp and rp.path then\n"
                "  for i, e in ipairs(rp.path) do\n"
                "    if e.pos then "
                "pts[#pts + 1] = {e.pos.x, e.pos.y, e.pos.z} end\n"
                "  end\n"
                "end\n"
                "return jsonEncode(pts)"
            )
            try:
                resp = self.bng.control.queue_lua_command(
                    chunk, response=True)
            except Exception as exc:
                print(f"[lua] read_navigation_route failed: {exc}")
                return None
        if not resp:
            return None
        try:
            data = json.loads(str(resp))
        except (ValueError, TypeError):
            print(f"[lua] read_navigation_route: bad response {resp!r}")
            return None
        if not data or len(data) < 2:
            return None
        arr = np.asarray(data, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 3:
            return None
        # The game's route tracker can corrupt path points when the vehicle
        # ends up far off the route line (observed: start z dragged to
        # -57000). Reject implausible routes instead of autopiloting into
        # them; the user can just pick the destination again on the map.
        if not np.all(np.isfinite(arr)):
            print("[lua] read_navigation_route: non-finite points; ignored")
            return None
        if float(np.max(np.abs(arr))) > 20000.0:
            print("[lua] read_navigation_route: coordinates out of map "
                  "bounds; ignored")
            return None
        if len(arr) > 1:
            seg = np.linalg.norm(np.diff(arr[:, :3], axis=0), axis=1)
            if float(np.max(seg)) > 5000.0:
                print("[lua] read_navigation_route: implausible segment "
                      "length; ignored")
                return None
        # The big map's route can be sparse (road-node spaced, ~15-20 m
        # apart), which makes pure-pursuit steering jaggy. Densify to a
        # ~2 m polyline spacing; keep the original when it is already
        # dense enough so no shape detail is lost.
        return self._densify_route(arr, spacing=2.0)

    def read_nav_world_visible(self) -> bool:
        """Return whether in-world navigation rendering is enabled."""
        with self.io_lock:
            chunk = (
                "local flags = {}\n"
                "if settings then\n"
                "  flags[1] = settings.getValue("
                "'showNavigationGroundmarkers')\n"
                "  flags[2] = settings.getValue('showNavigationArrows')\n"
                "end\n"
                "return jsonEncode(flags)"
            )
            try:
                resp = self.bng.control.queue_lua_command(
                    chunk, response=True)
            except Exception as exc:
                print(f"[lua] read_nav_world_visible failed: {exc}")
                return True
        try:
            flags = json.loads(str(resp))
            return bool(flags and (flags[0] or flags[1]))
        except (ValueError, TypeError, IndexError):
            print(f"[lua] read_nav_world_visible: bad response {resp!r}")
            return True

    def read_current_road_rule(self, pos, dir_vec) -> RoadRuleView | None:
        """Read the road link under the ego plus global road rules.

        The Lua chunk normalizes ``n1/n2`` to the link's ``inNode -> outNode``
        order so the ``lanes`` string has the same meaning on both Steam and
        BeamNG.tech.  Returns None when no map/road data is available.
        """
        x, y, z = (float(pos[0]), float(pos[1]), float(pos[2]))
        dx, dy, dz = (float(dir_vec[0]), float(dir_vec[1]),
                      float(dir_vec[2]))
        chunk = (
            "local n1, n2 = map.findBestRoad("
            f"vec3({x:.4f}, {y:.4f}, {z:.4f}), "
            f"vec3({dx:.4f}, {dy:.4f}, {dz:.4f}))\n"
            "if not n1 or not n2 then return 'nil' end\n"
            "local nodes = map.getMap().nodes\n"
            "local link = nodes[n1].links[n2] or nodes[n2].links[n1]\n"
            "if not link then return 'nil' end\n"
            "local n1o, n2o = n1, n2\n"
            "if link.inNode and link.outNode then\n"
            "  n1o, n2o = link.inNode, link.outNode\n"
            "else\n"
            "  local p1, p2 = nodes[n1].pos, nodes[n2].pos\n"
            "  if dx * (p2.x - p1.x) + dy * (p2.y - p1.y) < 0 then\n"
            "    n1o, n2o = n2, n1\n"
            "  end\n"
            "end\n"
            "local p1 = link.inPos or nodes[n1o].pos\n"
            "local p2 = link.outPos or nodes[n2o].pos\n"
            "local edgeDir = p2 - p1\n"
            "local edgeLen = edgeDir:length()\n"
            "local rightVec = vec3()\n"
            "if edgeLen > 1e-9 then\n"
            "  edgeDir:setScaled(1.0 / edgeLen)\n"
            "  rightVec = edgeDir:cross(nodes[link.inNode or n1o].normal)\n"
            "end\n"
            "local rules = {}\n"
            "if map.getRoadRules then rules = map.getRoadRules() end\n"
            "return jsonEncode({n1=n1o, n2=n2o, "
            "speedLimit=link.speedLimit, oneWay=link.oneWay, "
            "lanes=link.lanes, drivability=link.drivability, "
            "type=link.type, rightHandDrive=rules.rightHandDrive, "
            "turnOnRed=rules.turnOnRed, "
            "inPos={p1.x, p1.y, p1.z}, "
            "outPos={p2.x, p2.y, p2.z}, "
            "inRadius=link.inRadius, outRadius=link.outRadius, "
            "rightVec={rightVec.x, rightVec.y, rightVec.z}})\n"
        )
        with self.io_lock:
            try:
                resp = self.bng.control.queue_lua_command(
                    chunk, response=True)
            except Exception as exc:
                if not getattr(self, "_road_rule_warned", False):
                    self._road_rule_warned = True
                    print(f"[lua] read_current_road_rule failed: {exc}")
                return None
        if resp is None or str(resp).strip() == "nil":
            return None
        try:
            data = json.loads(str(resp))
        except (ValueError, TypeError):
            if not getattr(self, "_road_rule_warned", False):
                self._road_rule_warned = True
                print(f"[lua] read_current_road_rule: bad response {resp!r}")
            return None
        rule = RoadRuleView.from_lua_dict(data)
        if rule is not None:
            self._road_rule_warned = False
        return rule

    def read_signal_snapshot(self, vid: str, n1: str, n2: str
                             ) -> list[SignalRule]:
        """Return signals attached to the current road link.

        Uses the Tech 0.38-compatible ``core_trafficSignals`` map-node API.
        Steam builds expose the same functions, so this remains portable;
        the newer Steam-only ``getBestSignal`` shortcut is deliberately not
        used.  Missing signals / disabled extension return an empty list.
        """
        chunk = (
            "local out = {}\n"
            "local ts = core_trafficSignals\n"
            "if not ts or not ts.getMapNodeSignals or "
            "not ts.getSignalByName then return jsonEncode(out) end\n"
            "local mns = ts.getMapNodeSignals()\n"
            "local lists = {}\n"
            "if mns[n1] and mns[n1][n2] then lists[#lists + 1] = "
            "mns[n1][n2] end\n"
            "if mns[n2] and mns[n2][n1] then lists[#lists + 1] = "
            "mns[n2][n1] end\n"
            "for _, arr in ipairs(lists) do\n"
            "  for _, s in ipairs(arr) do\n"
            "    local inst = ts.getSignalByName(s.instance)\n"
            "    local d = {instance=s.instance, action=s.action or 0, "
            "state=s.state, useLane=s.useLane}\n"
            "    if s.pos then d.pos = {s.pos.x, s.pos.y, s.pos.z} end\n"
            "    if inst and inst.getVehPlacement then\n"
            "      local place = inst:getVehPlacement('" + vid + "')\n"
            "      if place then\n"
            "        d.dot = place.dot\n"
            "        d.dist = place.dist\n"
            "        d.relDist = place.relDist\n"
            "      end\n"
            "    end\n"
            "    out[#out + 1] = d\n"
            "  end\n"
            "end\n"
            "return jsonEncode(out)\n"
        )
        with self.io_lock:
            try:
                resp = self.bng.control.queue_lua_command(
                    chunk, response=True)
            except Exception as exc:
                if not getattr(self, "_signal_rule_warned", False):
                    self._signal_rule_warned = True
                    print(f"[lua] read_signal_snapshot failed: {exc}")
                return []
        if resp is None or str(resp).strip() in ("", "nil"):
            return []
        try:
            data = json.loads(str(resp))
        except (ValueError, TypeError):
            if not getattr(self, "_signal_rule_warned", False):
                self._signal_rule_warned = True
                print(f"[lua] read_signal_snapshot: bad response {resp!r}")
            return []
        if not isinstance(data, list):
            return []
        signals = [s for s in (SignalRule.from_lua_dict(x) for x in data)
                   if s is not None]
        if signals or not getattr(self, "_signal_rule_warned", False):
            self._signal_rule_warned = False
        return signals

    def set_nav_world_visible(self, visible: bool,
                              persist: bool = False) -> bool:
        """Show/hide the game's in-world navigation line and arrows.

        The route itself stays active in ``core_groundMarkers.routePlanner``
        and keeps appearing on the map, so ``read_navigation_route()`` and
        autopilot following are unaffected.  With ``persist=True`` the
        choice is written to the game's cloud settings so it also applies
        to later manual game sessions.
        """
        value = "true" if visible else "false"
        persist_arg = ", true" if persist else ""
        with self.io_lock:
            chunk = (
                "if not settings then return 'ok' end\n"
                f"settings.setValue("
                f"'showNavigationGroundmarkers', {value}{persist_arg})\n"
                f"settings.setValue('showNavigationArrows', "
                f"{value}{persist_arg})\n"
                "return 'ok'"
            )
            try:
                resp = self.bng.control.queue_lua_command(
                    chunk, response=True)
            except Exception as exc:
                print(f"[lua] set_nav_world_visible failed: {exc}")
                return False
        return str(resp).strip() == "ok"

    @staticmethod
    def _densify_route(arr: np.ndarray, spacing: float = 2.0) -> np.ndarray:
        """Resample a polyline to roughly ``spacing``-meter segments."""
        if arr.shape[0] < 2:
            return arr
        seg = np.linalg.norm(np.diff(arr[:, :3], axis=0), axis=1)
        total = float(seg.sum())
        if total <= 0.0:
            return arr
        n = max(2, int(round(total / spacing)) + 1)
        if n <= arr.shape[0]:
            return arr
        cum = np.concatenate(([0.0], np.cumsum(seg)))
        s = np.linspace(0.0, total, n)
        out = np.empty((n, 3), dtype=float)
        for k in range(3):
            out[:, k] = np.interp(s, cum, arr[:, k])
        return out
