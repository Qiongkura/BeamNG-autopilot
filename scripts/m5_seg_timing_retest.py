"""按"受控复测"协议重测推理延迟：静默机器 + 同一 checkpoint 重复测量取较小值。

为什么需要单独一个入口：本机时延有**间歇性负载污染**（实测：并发跑 pytest 时
同一 checkpoint 测出 p95=158 ms、安静时 16.7 ms；第 3 轮 5 个 seed 的 p95 也曾
出现 10.6–80.2 ms 的 7.6 倍漂移）。`rounds` 内部虽然做了"同 seed 立即再测一次取
较小值"，但实验期间机器上还有别的东西在跑时，那两次也可能都被污染——第 3 轮那批
延迟数字因此被判定**不可引用**，要在静默机器上按本协议重测。

协议（与 `rounds` 的硬门口径一致，不另立一套）：

1. **静默前置检查**：GPU 利用率低，且没有正在吃 CPU 的其它 python 进程
   （两次采样比 CPU 时间，避免只看进程名把空闲的看板服务也算成"在忙"）；
   不安静就**拒绝出数**，除非显式 ``--force``。
2. 每个 checkpoint 重复 ``--repeats`` 次，逐次记录 p50/p95，并给
   ``suspect``（p95/p50 > 4 = 本次被污染，与 `rounds.timing_suspect` 同一判据）。
3. **报告取"没被污染的重复里 p95 最小的一次"**；全部重复都留在 JSON 里可复核。
   没有一次干净就报"未测"，不报"很快"。

用法::

    python scripts/m5_seg_timing_retest.py \\
        --model s42=logs/experiments/<run>/round0/seed42/checkpoint_last.pt \\
        --dev-runs logs/m5_seg/diverse_wide_20260924/front_main \\
                   logs/m5_seg/diverse_plain_20260924/front_main \\
        --repeats 3 --out logs/experiments/<run>/timing_retest.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

#: 出现在机器上就说明"不安静"的游戏进程（训练/评估是我们自己的 python，另算）
GAME_IMAGES = ("BeamNG.tech.x64.exe", "BeamNG.drive.x64.exe",
               "BeamNG.tech.exe")

#: 两次采样之间某个 python 进程的 CPU 时间涨超过这么多秒，就算它在干活
BUSY_CPU_DELTA_S = 0.5


def _cpu_times(names=("python", "pythonw")) -> dict | None:
    """``{pid: CPU 秒}``；任何探测失败返回 ``None``（不确定，不当作安静）。

    用 PowerShell 7 取 ``Get-Process`` 的 ``CPU``（累计处理器秒）。原来用
    ``wmic``，实测在这台 Windows 11 上已经不可用（进程探测直接失败），于是
    "静默检查"永远拿不到结论、每次都以"不安静"拒绝出数——探测手段本身不能用，
    和"机器在忙"必须分开。
    """
    shell = shutil.which("pwsh") or shutil.which("powershell")
    if not shell:
        return None
    # 一次取全部进程、在 Python 里按名字过滤：`-Name python,pythonw` 这种写法在
    # 某个名字不存在时会整体失败（实测：pythonw 不在 -> 探测返回 None -> 永远
    # 判"机器不安静"）。CPU 用整数毫秒输出，避开中文区域的小数点变成逗号。
    script = ("Get-Process | ForEach-Object { $_.ProcessName + ',' + "
              "$_.Id.ToString() + ',' + "
              "[string]([int]([math]::Round($_.CPU * 1000))) }")
    want = {str(n).lower().replace(".exe", "") for n in names}
    try:
        r = subprocess.run([shell, "-NoProfile", "-Command", script],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=60)
    except Exception:                                     # noqa: BLE001
        return None
    if r.returncode != 0:
        return None
    out: dict = {}
    for line in (r.stdout or "").splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 3 or not parts[1].isdigit():
            continue
        if parts[0].lower() not in want:
            continue
        try:
            out[int(parts[1])] = float(parts[2]) / 1000.0     # 毫秒 -> 秒
        except ValueError:
            continue
    return out


def game_running(images=GAME_IMAGES) -> bool | None:
    for img in images:
        try:
            r = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {img}", "/NH"],
                               capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=20)
        except Exception:                                 # noqa: BLE001
            return None
        if r.returncode != 0:
            return None
        if img.lower() in (r.stdout or "").lower():
            return True
    return False


def busy_pythons(*, interval_s: float = 2.0) -> list | None:
    """除自己之外**正在吃 CPU** 的 python 进程；探测失败返回 ``None``。"""
    a = _cpu_times()
    if a is None:
        return None
    time.sleep(max(0.2, float(interval_s)))
    b = _cpu_times()
    if b is None:
        return None
    me = os.getpid()
    busy = []
    for pid, cpu_b in b.items():
        if pid == me:
            continue
        delta = cpu_b - float(a.get(pid, cpu_b))
        if delta > BUSY_CPU_DELTA_S:
            busy.append({"pid": pid, "cpu_delta_s": round(delta, 2)})
    return busy


def gpu_util() -> float | None:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20)
        if r.returncode != 0:
            return None
        return float((r.stdout or "0").strip().splitlines()[0])
    except Exception:                                     # noqa: BLE001
        return None


def is_suspect(p50, p95, ratio: float = 4.0) -> bool:
    """``rounds.timing_suspect`` 的同一判据；缺测按"不可信"处理，不按"很快"。"""
    try:
        p50, p95 = float(p50), float(p95)
    except (TypeError, ValueError):
        return True
    return p50 <= 0 or (p95 / p50) > float(ratio)


def measure(model, dev_runs, *, device: str = "cuda", repeats: int = 3,
            evaluate=None, load_frames=None) -> dict:
    """重复测量一个 checkpoint（``evaluate``/``load_frames`` 可注入，便于测试）。"""
    if load_frames is None or evaluate is None:
        import m5_seg_eval_matrix as em
        load_frames = load_frames or em.load_frames
        evaluate = evaluate or em.evaluate_model
    frames = load_frames([Path(r) for r in dev_runs])
    runs: list = []
    for i in range(max(1, int(repeats))):
        t0 = time.time()
        m = evaluate(Path(model), frames, device=device) or {}
        p50, p95 = m.get("inference_ms_p50"), m.get("inference_ms_p95")
        runs.append({"i": i, "p50": p50, "p95": p95,
                     "wall_s": round(time.time() - t0, 2),
                     "road_iou": m.get("road_iou"),
                     "suspect": is_suspect(p50, p95)})
    clean = [r for r in runs if not r["suspect"] and r["p95"] is not None]
    pick = min(clean, key=lambda r: r["p95"]) if clean else None
    p95s = [r["p95"] for r in runs if r["p95"] is not None]
    return {"model": str(model), "repeats": runs, "chosen": pick,
            "p95_min": min(p95s) if p95s else None,
            "p95_median": sorted(p95s)[len(p95s) // 2] if p95s else None,
            "p95_max": max(p95s) if p95s else None,
            "clean_repeats": len(clean), "any_suspect": len(clean) < len(runs)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True,
                    metavar="NAME=PATH", help="可重复；每个 checkpoint 一项")
    ap.add_argument("--dev-runs", nargs="+", required=True)
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default=None, help="结果 JSON")
    ap.add_argument("--force", action="store_true",
                    help="机器不安静时也照测（结果会标注不静默）")
    args = ap.parse_args(argv)

    busy = busy_pythons()
    game = game_running()
    util = gpu_util()
    quiet = bool(busy == [] and game is False
                 and (util is None or util <= 20.0))
    print(f"[timing] 静默检查：吃 CPU 的 python={busy} 游戏={game} "
          f"GPU 利用率={util} -> quiet={quiet}")
    if not quiet and not args.force:
        print("[timing] 机器不安静：按协议拒绝出数（先停掉其它训练/评估，"
              "或用 --force 明确接受污染）")
        return 3

    out = {"quiet": quiet, "busy_pythons": busy, "game_running": game,
           "gpu_util": util, "device": args.device,
           "repeats": int(args.repeats),
           "started_iso": time.strftime("%Y-%m-%dT%H:%M:%S"), "models": []}
    rc = 0
    for spec in args.model:
        if "=" not in spec:
            print(f"[timing] 参数格式应为 NAME=PATH：{spec}")
            return 2
        name, path = spec.split("=", 1)
        p = Path(path)
        if not p.exists():
            print(f"[timing] 缺 {name}：{p}")
            rc = 2
            continue
        got = measure(p, args.dev_runs, device=args.device,
                      repeats=args.repeats)
        got["name"] = name
        out["models"].append(got)
        chosen = got["chosen"]["p95"] if got["chosen"] else "未测"
        print(f"[timing] {name}: 重复 p95={[r['p95'] for r in got['repeats']]} "
              f"→ 取 {chosen} ms（min/median/max = {got['p95_min']}/"
              f"{got['p95_median']}/{got['p95_max']}，干净 {got['clean_repeats']}/"
              f"{len(got['repeats'])} 次）")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                  encoding="utf-8")
        print(f"[timing] -> {args.out}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
