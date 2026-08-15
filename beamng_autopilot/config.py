"""全局配置：BeamNG 路径、默认地图/车型、项目目录。"""

import math
import os
from pathlib import Path

# 项目内部目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"

# BeamNG.drive 安装目录与用户目录。
#
# 发布版本不硬编码任何机器路径：
# - BEAMNG_HOME 优先取环境变量，其次自动探测 Steam 库（注册表 +
#   libraryfolders.vdf + 常见安装路径），探测失败才回退到旧默认值并
#   打印警告（本机零配置照常工作）。
# - BEAMNG_USER 优先取环境变量，否则用当前用户目录（通用，不再包含
#   机器用户名）。
# - BEAMNG_TECH_HOME / BEAMNG_TECH_USER 已支持环境变量，保留原默认值
#   作为开发者机器上的探测失败回退。

def _probe_steam_beamng_home() -> Path | None:
    """Locate a Steam-installed BeamNG.drive without any hardcoded paths.

    Checks, in order: the Steam registry key (``SteamPath``), every library
    folder listed in ``steamapps/libraryfolders.vdf`` (the file Steam uses
    for multiple game libraries), and a few common install locations.
    Returns None when no candidate contains a BeamNG.drive build.
    """
    candidates: list[Path] = []
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                            r"Software\Valve\Steam") as key:
            steam_path = Path(str(winreg.QueryValueEx(key, "SteamPath")[0]))
        candidates.append(steam_path / "steamapps" / "common" / "BeamNG.drive")
        vdf = steam_path / "steamapps" / "libraryfolders.vdf"
        if vdf.is_file():
            text = vdf.read_text(encoding="utf-8", errors="ignore")
            for m in __import__("re").finditer(
                    r'"path"\s+"([^"]+)"', text):
                lib = Path(m.group(1))
                candidates.append(lib / "steamapps" / "common" / "BeamNG.drive")
    except Exception:
        pass
    candidates.extend([
        Path("C:/Program Files (x86)/Steam/steamapps/common/BeamNG.drive"),
        Path("D:/SteamLibrary/steamapps/common/BeamNG.drive"),
    ])
    for cand in candidates:
        if (cand / "Bin64" / "BeamNG.drive.x64.exe").exists():
            return cand
    return None


_LEGACY_BEAMNG_HOME = Path(r"G:\SteamLibrary\steamapps\common\BeamNG.drive")
_env_home = os.environ.get("BEAMNG_HOME")
if _env_home:
    BEAMNG_HOME = Path(_env_home)
else:
    _probed_home = _probe_steam_beamng_home()
    if _probed_home is not None:
        BEAMNG_HOME = _probed_home
    else:
        BEAMNG_HOME = _LEGACY_BEAMNG_HOME
        print("[config] WARNING: BeamNG.drive not found via Steam registry; "
              f"falling back to {BEAMNG_HOME}. Set the BEAMNG_HOME "
              "environment variable to the actual install directory.")

_env_user = os.environ.get("BEAMNG_USER")
BEAMNG_USER = Path(
    _env_user if _env_user
    else Path.home() / "AppData" / "Local" / "BeamNG.drive" / "0.39")
BEAMNG_TECH_USER = Path(
    os.environ.get(
        "BEAMNG_TECH_USER",
        str(BEAMNG_USER.parent / "0.38")))
BEAMNG_TECH_HOME = Path(
    os.environ.get("BEAMNG_TECH_HOME", r"G:\BeamNG.tech.v0.38.5.0"))

# Runtime selection: "auto" (detect after connecting), "steam" (default
# Steam-compatible path) or "tech" (BeamNG.tech extension path).
RUNTIME_MODE = os.environ.get("BEAMNG_RUNTIME", "auto").lower()

# Process names shared by the launcher and the autopilot's running-game probe.
BEAMNG_PROCESS_NAMES = {
    "BeamNG.drive",
    "BeamNG.drive.x64",
    "BeamNG.drive.x64.exe",
    "BeamNG.tech",
    "BeamNG.tech.x64",
    "BeamNG.tech.x64.exe",
}

# BeamNGpy 通信：端口与运行时绑定，Steam / Tech 可同时跑。
# - Steam 用 PORT（默认 64256，可用 BEAMNG_PORT 覆盖）
# - Tech 用 TECH_PORT（默认 64257，可用 BEAMNG_TECH_PORT 覆盖）
# 用 config.runtime_port(mode) 取端口，不要直接用 PORT 连 Tech。
HOST = "127.0.0.1"
PORT = int(os.environ.get("BEAMNG_PORT", "64256"))
TECH_PORT = int(os.environ.get("BEAMNG_TECH_PORT", "64257"))

# Road-network guards used before teleporting a freshly spawned car:
# only reposition once the road graph is dense, and only when the car is
# not already sitting on a road node (sparse early data used to move the
# car into walls/trees on BeamNG.tech).
ROADNET_REPOSITION_MIN_NODES = 2000
ROADNET_REPOSITION_NEAR_M = 25.0

# 默认场景
DEFAULT_MAP = "italy"
DEFAULT_VEHICLE = "etk800"

# 意大利地图默认出生点：spawn_crossroads（十字路口）。
# 2026-08-15 实测定点：以 tech 自动驾驶测试默认位置为准（车停在路面
# 上，含当前朝向；z 为车辆 origin 高度，spawn 时 cling 会贴地）。
ITALY_SPAWN_CROSSROADS_POS = (729.634694, 763.914991, 177.954548)
ITALY_SPAWN_CROSSROADS_HEADING = 0.445966302

# 车体 origin 离路面高度（etk800，2026-08-15 实测：st.pos[2] - 最近路网
# 节点 z = 0.17m）。地面反投影必须用路面高度；用车辆 origin 的 z 会让
# 投影出的标线/边界在 5m 处偏约 0.5m、20m 处偏约 2m。
EGO_ORIGIN_GROUND_GAP_M = 0.17


def runtime_home(mode: str | None = None) -> Path:
    """Return the game install directory for the requested runtime."""
    if resolve_launch_runtime(mode) == "tech":
        return BEAMNG_TECH_HOME
    return BEAMNG_HOME


def runtime_user(mode: str | None = None) -> Path:
    """Return the game user directory for the requested runtime."""
    if resolve_launch_runtime(mode) == "tech":
        return BEAMNG_TECH_USER
    return BEAMNG_USER


def resolve_launch_runtime(mode: str | None = None) -> str:
    """Resolve the runtime used to launch/attach before a connection exists.

    ``auto`` prefers BeamNG.tech when a Tech install is present (the
    developer machine), otherwise falls back to the Steam build so ordinary
    open-source users keep working with no Tech setup.
    """
    mode = (mode or RUNTIME_MODE).lower()
    if mode != "auto":
        return mode
    return "tech" if BEAMNG_TECH_HOME.is_dir() else "steam"


def runtime_port(mode: str | None = None) -> int:
    """Comms port bound to a runtime: Steam -> PORT, Tech -> TECH_PORT.

    Pass this instead of ``PORT`` whenever the target runtime is known, so
    the Steam and Tech instances can run side by side on fixed ports.
    """
    if resolve_launch_runtime(mode) == "tech":
        return TECH_PORT
    return PORT
