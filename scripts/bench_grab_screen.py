"""抓帧性能探针：PrintWindow (GDI) vs Windows Graphics Capture (dxcam)。

Steam 通道的视觉感知（YOLO / 车道线）帧率受抓帧速度限制。本探针在
游戏窗口上分别用两种方式抓取 N 帧，统计单帧耗时（中位数 / P95）与
可达帧率，用来判断是否值得把 connector.grab_screen 换成 DXGI 抓帧。

用法（先进游戏、进地图、让游戏窗口可见）:
    .venv\Scripts\python.exe scripts\bench_grab_screen.py --frames 100

不需要 beamngpy 连接，只需要游戏窗口存在。
dxcam 未安装时只跑 PrintWindow 基线：
    .venv\Scripts\python.exe -m pip install dxcam
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beamng_autopilot.connector import BeamNGConnector

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _stats(times_ms: list[float], fails: int) -> dict:
    a = np.asarray(times_ms, dtype=float)
    return {
        "n": len(a),
        "median_ms": float(np.median(a)),
        "p95_ms": float(np.percentile(a, 95)),
        "min_ms": float(np.min(a)),
        "max_ms": float(np.max(a)),
        "fps": 1000.0 / float(np.median(a)) if len(a) else 0.0,
        "fails": fails,
    }


def bench_printwindow(conn, frames: int, warmup: int) -> dict:
    """现有 grab_screen 路径：PrintWindow + GetDIBits（含窗口查找缓存）。"""
    print("[*] PrintWindow 基线 ...")
    for _ in range(warmup):
        conn.grab_screen()
    times: list[float] = []
    fails = 0
    for i in range(frames):
        t0 = time.perf_counter()
        img = conn.grab_screen()
        dt = (time.perf_counter() - t0) * 1000.0
        if img is None or img.size == 0:
            fails += 1
            continue
        times.append(dt)
        if (i + 1) % 25 == 0:
            print(f"    {i + 1}/{frames} 帧 (中位 {np.median(times):.1f} ms)")
    return _stats(times, fails)


def bench_dxcam(rect, frames: int, warmup: int) -> dict:
    """Windows Graphics Capture (dxcam)：抓窗口矩形对应的屏幕区域。

    注意：抓的是屏幕区域而非窗口内容——窗口被其他窗口遮挡 / 最小化时
    抓到的是遮挡内容，这是与 PrintWindow 的一个关键差异。dxcam 只能抓
    主屏坐标 (0,0)-(cx,cy) 范围内的区域，副屏（负坐标）窗口无法抓取。
    """
    import ctypes

    import dxcam

    camera = dxcam.create(output_color="RGB")
    if camera is None:
        return {"error": "dxcam.create() 失败（无可用 DXGI 输出）"}
    left, top, right, bottom = (int(v) for v in rect)
    cx = ctypes.windll.user32.GetSystemMetrics(0)  # SM_CXSCREEN
    cy = ctypes.windll.user32.GetSystemMetrics(1)  # SM_CYSCREEN
    if right <= 0 or bottom <= 0 or left >= cx or top >= cy:
        return {"error": f"窗口完全在主屏 ({cx}x{cy}) 之外"
                         f"（窗口在 {left},{top}）；"
                         "请把游戏窗口拖到主屏后重测"}
    region = (left, top, right, bottom)
    print(f"[*] dxcam 区域 ({left},{top})-({right},{bottom}) ...")
    try:
        for _ in range(warmup):
            camera.grab(region=region)
        times: list[float] = []
        fails = 0
        for i in range(frames):
            t0 = time.perf_counter()
            frame = camera.grab(region=region)
            dt = (time.perf_counter() - t0) * 1000.0
            if frame is None or frame.size == 0:
                fails += 1
                continue
            times.append(dt)
            if (i + 1) % 25 == 0:
                print(f"    {i + 1}/{frames} 帧 (中位 {np.median(times):.1f} ms)")
        return _stats(times, fails)
    finally:
        try:
            camera.stop()
        except Exception:
            pass


def main() -> int:
    ap = argparse.ArgumentParser(description="抓帧性能对比探针")
    ap.add_argument("--frames", type=int, default=100,
                    help="每种方式抓取的帧数（默认 100）")
    ap.add_argument("--warmup", type=int, default=5,
                    help="预热帧数（默认 5）")
    ap.add_argument("--no-dxcam", action="store_true",
                    help="跳过 dxcam 对比，只跑 PrintWindow 基线")
    args = ap.parse_args()

    if args.frames < 10:
        print("[!] --frames 至少 10，统计才有意义")
        return 2

    conn = BeamNGConnector()  # 只用于窗口查找与 PrintWindow 抓帧，不连接游戏
    hwnd, rect = conn._find_window()
    if hwnd is None or rect is None:
        print("[!] 未找到 BeamNG 游戏窗口。")
        print("    请先启动游戏并进入地图，保持窗口可见后重试。")
        return 2
    w = rect[2] - rect[0]
    h = rect[3] - rect[1]
    print(f"[*] 游戏窗口 {w}x{h} @ ({rect[0]},{rect[1]})")
    print("    提示：窗口若被遮挡/最小化，dxcam 抓到的会是遮挡内容；")
    print("    建议把游戏窗口置于前台再测。")
    print()

    results: dict[str, dict] = {}
    results["printwindow"] = bench_printwindow(conn, args.frames, args.warmup)
    if args.no_dxcam:
        results["dxcam"] = {"skipped": True}
    else:
        try:
            results["dxcam"] = bench_dxcam(rect, args.frames, args.warmup)
        except ImportError:
            results["dxcam"] = {"error": "dxcam 未安装；pip install dxcam 后可对比"}
        except Exception as exc:
            results["dxcam"] = {"error": str(exc)}

    print()
    print("=" * 64)
    print(f"{'方式':<14}{'中位 ms':>10}{'P95 ms':>10}{'帧率 fps':>10}"
          f"{'失败帧':>8}")
    print("-" * 64)
    for name in ("printwindow", "dxcam"):
        r = results[name]
        if "error" in r:
            print(f"{name:<14}{'—':>10}{'—':>10}{'—':>10}{'—':>8}"
                  f"  {r['error']}")
            continue
        if r.get("skipped"):
            print(f"{name:<14}{'—':>10}{'—':>10}{'—':>10}{'—':>8}  跳过")
            continue
        print(f"{name:<14}{r['median_ms']:>10.1f}{r['p95_ms']:>10.1f}"
              f"{r['fps']:>10.1f}{r['fails']:>8}")
    print("=" * 64)

    pw = results["printwindow"]
    dx = results.get("dxcam", {})
    if "error" not in dx and not dx.get("skipped") and pw["n"]:
        speedup = pw["median_ms"] / max(dx["median_ms"], 1e-6)
        print(f"结论: dxcam 中位耗时是 PrintWindow 的 {speedup:.1f} 倍速度"
              f"（{pw['median_ms']:.0f}ms -> {dx['median_ms']:.1f}ms）。")
        if speedup >= 1.5:
            print("    值得把 connector.grab_screen 换成 DXGI 抓帧"
                  "（注意窗口遮挡语义差异）。")
        else:
            print("    差距不大，保持 PrintWindow 即可。")
    return 0


if __name__ == "__main__":
    sys.exit(main())