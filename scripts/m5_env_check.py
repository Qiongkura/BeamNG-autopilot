"""M5 环境自检：一键检查依赖、游戏路径、资源与运行时状态。

普通用户拿到仓库后先运行本脚本，输出一份可读的环境清单：
哪些正常、哪些缺失、如何修复。退出码 0 = 无错误，1 = 有错误。

用法:
    .venv\Scripts\python.exe scripts\m5_env_check.py
"""

from __future__ import annotations

import importlib.util
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beamng_autopilot import config

# 控制台编码：保证 ✅/❌/⚠️ 与中文在任意终端下不崩溃
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# (status, label, detail)；status: "ok" | "warn" | "err" | "info"
RESULTS: list[tuple[str, str, str]] = []


def ok(label: str, detail: str = "") -> None:
    RESULTS.append(("ok", label, detail))


def info(label: str, detail: str = "") -> None:
    RESULTS.append(("info", label, detail))


def warn(label: str, detail: str = "") -> None:
    RESULTS.append(("warn", label, detail))


def err(label: str, detail: str = "") -> None:
    RESULTS.append(("err", label, detail))


def _module_installed(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def check_python() -> None:
    v = sys.version_info
    if v >= (3, 10):
        ok(f"Python {v.major}.{v.minor}.{v.micro}", "需要 3.10+")
    else:
        err(f"Python {v.major}.{v.minor}.{v.micro}", "需要 3.10+，请升级 Python 后重建 venv")


def check_dependencies() -> None:
    """核心 / 视觉 / 可选三档依赖检查。"""
    for mod, pkg in (("numpy", "numpy"), ("cv2", "opencv-python"),
                     ("beamngpy", "beamngpy==1.35.1"),
                     ("matplotlib", "matplotlib")):
        if _module_installed(mod):
            ok(f"依赖 {mod}", "")
        else:
            err(f"依赖 {mod} 缺失", f"请安装: pip install {pkg}")

    for mod, pkg in (("torch", "torch"), ("ultralytics", "ultralytics")):
        if _module_installed(mod):
            ok(f"依赖 {mod}", "")
        else:
            warn(f"依赖 {mod} 缺失",
                 f"视觉避障/车道线/BC 训练不可用；请安装: pip install {pkg}")

    try:
        import torch

        if torch.cuda.is_available():
            ok("CUDA 可用", torch.cuda.get_device_name(0))
        else:
            warn("CUDA 不可用", "YOLO 与训练将使用 CPU（明显变慢）；"
                 "安装 CUDA 版 torch 可加速")
    except Exception:
        pass

    for mod, pkg, desc in (
        ("psutil", "psutil", "游戏进程检测"),
        ("gymnasium", "gymnasium", "M4 决策层"),
        ("stable_baselines3", "stable-baselines3", "M4 决策层"),
        ("dxcam", "dxcam", "抓帧加速探针（可选）"),
    ):
        if _module_installed(mod):
            continue
        if mod == "psutil":
            warn(f"可选依赖 {mod} 缺失",
                 f"进程检测不可用；请安装: pip install {pkg}")
        else:
            info(f"可选依赖 {mod} 未安装", f"{desc}；需要时: pip install {pkg}")


def check_assets() -> None:
    """项目数据 / 模型资源。"""
    track = config.DATA_DIR / "track_smallgrid.npz"
    if track.is_file():
        ok("循迹样例数据", "data/track_smallgrid.npz")
    else:
        warn("循迹样例数据缺失", "data/track_smallgrid.npz 不存在")

    w = config.PROJECT_ROOT / "weights" / "yolov8n.pt"
    if w.is_file():
        ok("YOLO 权重", f"{w.name} ({w.stat().st_size / 1e6:.1f} MB)")
    else:
        warn("YOLO 权重缺失",
             "首次启用视觉避障时会自动下载 ~6.5MB 到 weights/yolov8n.pt")

    yolo_cfg = config.PROJECT_ROOT / ".yolo"
    if yolo_cfg.is_dir():
        ok("YOLO 配置目录", ".yolo/")
    else:
        info("YOLO 配置目录未创建", "运行时自动创建（.yolo/，避免写入 AppData）")


def check_beamng() -> None:
    """游戏安装 / 用户目录 / 双运行时。"""
    exe = config.BEAMNG_HOME / "Bin64" / "BeamNG.drive.x64.exe"
    if exe.is_file():
        ok("BeamNG.drive 安装", str(config.BEAMNG_HOME))
    else:
        err("BeamNG.drive 未找到",
            f"探测结果: {config.BEAMNG_HOME}；请安装 Steam 版，或设置 "
            "BEAMNG_HOME 环境变量指向安装目录")

    info("BeamNG 用户目录", str(config.BEAMNG_USER))

    if config.BEAMNG_TECH_HOME.is_dir():
        ok("BeamNG.tech 通道", str(config.BEAMNG_TECH_HOME))
    else:
        info("BeamNG.tech 未安装",
             "仅使用 Steam 版功能（截屏 + Lua 感知），"
             "功能不受影响；Tech 通道可选")

    info("运行时模式", f"BEAMNG_RUNTIME={config.RUNTIME_MODE} "
                       f"(auto 优先 Tech，其次 Steam)")


def check_runtime_state() -> None:
    """游戏进程与通信端口（仅诊断，不是错误）。"""
    try:
        import psutil

        names = config.BEAMNG_PROCESS_NAMES
        running = [p.info["name"] for p in psutil.process_iter(["name"])
                   if p.info["name"] in names]
    except Exception:
        running = []

    if not running:
        info("游戏未运行", "由 launch_game.py / m5_launcher.py 启动时会自动拉起")
        return

    info("游戏正在运行", ", ".join(sorted(set(running))))
    try:
        with socket.create_connection((config.HOST, config.PORT), timeout=1.0):
            ok("通信端口", f"{config.HOST}:{config.PORT} 可连接")
    except OSError:
        err("通信端口不通", f"{config.HOST}:{config.PORT} 未监听。"
            "游戏需带通信参数启动：用 scripts/launch_game.py 启动，"
            "或在 Steam 启动选项加 -tcom -tport 64256")


def main() -> int:
    print("=" * 60)
    print(" BeamNG Autopilot 环境自检")
    print("=" * 60)
    check_python()
    check_dependencies()
    check_assets()
    check_beamng()
    check_runtime_state()

    print()
    print("-" * 60)
    n_ok = n_warn = n_err = n_info = 0
    for status, label, detail in RESULTS:
        if status == "ok":
            mark, n_ok = "  [OK]  ", n_ok + 1
        elif status == "warn":
            mark, n_warn = " [WARN] ", n_warn + 1
        elif status == "err":
            mark, n_err = " [FAIL] ", n_err + 1
        else:
            mark, n_info = " [INFO] ", n_info + 1
        line = f"{mark}{label}"
        if detail:
            line += f" — {detail}"
        print(line)

    print("-" * 60)
    print(f"结果: {n_ok} 正常 / {n_warn} 警告 / {n_err} 错误 / {n_info} 信息")
    if n_err:
        print("存在错误：请按上方 [FAIL] 提示修复后重试。")
        return 1
    if n_warn:
        print("有警告：不影响循迹，但部分功能会降级。")
        return 0
    print("环境就绪，可以运行 scripts/m5_launcher.py 或 scripts/m5_autopilot.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())