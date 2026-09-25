"""训练看板 / 实时监控的双击入口：挑最近一轮 → 渲染或起服务 → 打开浏览器。

根目录的两个 VBS（``启动训练看板.vbs``、``启动训练监控.vbs``）只做三件事：
找到 venv 解释器、跑本脚本、失败时把退出码弹出来。**逻辑都在这里**，所以
双击路径能不能用由本脚本决定，VBS 里不写任何视图或路径规则。

两个模式各自复用已有实现，不复制视图代码：

* ``dashboard``：把最近一轮的 ``events.jsonl`` 交给 ``m5_seg_dashboard.py``
  的渲染器，出一张自包含 HTML 再用默认浏览器打开；
* ``monitor``：把最近一轮带逐 step 指标的 run 交给 ``monitor_server``
  起只读服务，再用默认浏览器打开；
* ``history``：全仓训练台账（``m5_training_history.py``）——**不挑某一轮**，
  把 ``logs/`` 下所有训练产物扫成一张历史表；它只列事实、不做排行。

"最近一轮" = ``logs/experiments/<run_id>/`` 里标记文件 mtime 最新的那个
（看板认 ``events.jsonl``，监控认 ``metrics.jsonl``）；``--run-id`` 可显式
指定。没有可用数据时**不猜、不造样本**：打印候选与原因，退出码 2。

退出码：``0`` 正常；``2`` 没有可用数据（原因与候选一起打印）；``3`` 端口
连系统分配都起不来（正常情况下见不到）。

端口规则：``--port`` 上如果已经在看**同一个 run**，就直接开浏览器（双击两
次不会起第二个服务）；如果跑的是**别的 run** 或端口被别的程序占着，绝不顺手
打开（那是别的实验的曲线），而是说明原因后改用系统分配的端口，并把新地址
打印出来。

只读纪律：本脚本不训练、不接触游戏、不写任何 run 目录（``monitor`` 的
HTTP 服务也只读指标文件）。默认也**不**顺手带上 T13 历史或别的实验的评估
矩阵——那些是别的实验的数字，混进本页会让"这一轮"看起来比实际好；要带
就显式 ``--t13-import``。per-run 的 ``probes/`` 目录属于本 run，存在时会
自动带上。

用法::

    pwsh> .venv\\Scripts\\python.exe scripts\\m5_training_view.py dashboard
    pwsh> .venv\\Scripts\\python.exe scripts\\m5_training_view.py monitor --port 8760
    pwsh> .venv\\Scripts\\python.exe scripts\\m5_training_view.py list
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from beamng_autopilot import config  # noqa: E402
from beamng_autopilot.experiments import monitor_server  # noqa: E402
from beamng_autopilot.experiments.metrics import MetricsStore  # noqa: E402
import m5_seg_dashboard  # noqa: E402  （同目录脚本，复用它的渲染器）
import m5_training_history  # noqa: E402  （台账渲染器）

#: 看板认事件流，监控认逐 step 指标——两个标记文件决定"哪一轮"。
MARKER_DASHBOARD = "events.jsonl"
MARKER_MONITOR = "metrics.jsonl"
DEFAULT_DASHBOARD_OUT = "dashboard_latest.html"
DEFAULT_HISTORY_OUT = "training_history.html"

EXIT_NO_DATA = 2
EXIT_PORT_BUSY = 3


def experiments_root() -> Path:
    return Path(config.LOGS_DIR) / "experiments"


def candidates(root: Path, marker: str) -> list[tuple[float, str, Path]]:
    """按标记文件 mtime 倒序列出候选 run（只扫 ``experiments/`` 下一层）。"""
    out: list[tuple[float, str, Path]] = []
    if not root.is_dir():
        return out
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        m = d / marker
        if m.is_file():
            out.append((m.stat().st_mtime, d.name, d))
    out.sort(key=lambda t: t[0], reverse=True)
    return out


def resolve_run(root: Path, marker: str, run_id: str | None
                ) -> tuple[tuple[str, Path] | None, str]:
    """返回 ``((run_id, run_dir), "")`` 或 ``(None, 原因)``。"""
    if run_id:
        d = root / run_id
        if not d.is_dir():
            return None, f"{d} 不存在"
        if not (d / marker).is_file():
            return None, f"{d} 里没有 {marker}"
        return (d.name, d), ""
    found = candidates(root, marker)
    if not found:
        return None, f"{root} 下没有任何带 {marker} 的 run"
    return (found[0][1], found[0][2]), ""


def _print_candidates(root: Path, marker: str) -> None:
    found = candidates(root, marker)
    print(f"[view] 候选（按 {marker} 的 mtime 倒序，最多 10 个）：")
    if not found:
        print("  （无）")
        return
    for ts, name, _d in found[:10]:
        stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))
        print(f"  {stamp}  {name}")


def _no_data(mode: str, root: Path, marker: str, reason: str,
             hint: str) -> int:
    print(f"[view] {mode}：没有可用数据 — {reason}")
    _print_candidates(root, marker)
    print(f"[view] {hint}")
    return EXIT_NO_DATA


def open_target(target: str, *, no_open: bool) -> None:
    """打开浏览器；打不开只提示，不当失败（渲染/服务本身是成功的）。"""
    if no_open:
        print(f"[view] --no-open：请自己打开 {target}")
        return
    if not webbrowser.open(target):
        print(f"[view] 打不开默认浏览器，请手动打开 {target}")


def _serving(url: str) -> str | None:
    """该端口上监控服务正在看的 ``run_id``；不是我们的服务/没人监听则 None。

    只认 ``/state`` 且必须同时带 ``run_id`` 与 ``status``（我们的服务一定
    有这两项）：否则"某个别的程序占着这个端口"会被误判成"已在看这一轮"。
    """
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/state",
                                    timeout=1.5) as resp:
            if resp.status != 200:
                return None
            blob = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return None
    if not isinstance(blob, dict) or "status" not in blob:
        return None
    rid = blob.get("run_id")
    return str(rid) if rid else None


# ---------------------------------------------------------------------------
# dashboard：渲染最近一轮的学习看板
# ---------------------------------------------------------------------------
def cmd_dashboard(args) -> int:
    root = experiments_root()
    run, reason = resolve_run(root, MARKER_DASHBOARD, args.run_id)
    if run is None:
        return _no_data(
            "看板", root, MARKER_DASHBOARD, reason,
            "事件流由实验循环写入：先跑 m5_seg_autoloop.py rounds "
            "（或训练时带 --metrics-run）再看。")
    run_id, run_dir = run
    out = Path(args.out) if args.out else root / DEFAULT_DASHBOARD_OUT

    argv = ["render", "--out", str(out),
            "--events", str(run_dir / MARKER_DASHBOARD)]
    # probes/ 是本 run 自己的产物，存在就带上；别的实验的矩阵不自动带。
    probes = run_dir / "probes"
    if probes.is_dir():
        argv += ["--probes", str(probes)]
    if args.t13_import:
        argv.append("--t13-import")

    print(f"[view] 看板 run={run_id} → {out}")
    rc = m5_seg_dashboard.main(argv)
    if rc != 0:
        print(f"[view] 渲染失败（退出码 {rc}）")
        return rc
    open_target(out.resolve().as_uri(), no_open=args.no_open)
    return 0


# ---------------------------------------------------------------------------
# monitor：给最近一轮带逐 step 指标的 run 起实时服务
# ---------------------------------------------------------------------------
def cmd_monitor(args) -> int:
    root = experiments_root()
    run, reason = resolve_run(root, MARKER_MONITOR, args.run_id)
    if run is None:
        return _no_data(
            "监控", root, MARKER_MONITOR, reason,
            "逐 step 指标只在训练时带 --metrics-run 才写："
            "m5_train_seg.py --metrics-run <run_id>。没有它的 run 只能看"
            "看板（events.jsonl 是另一条流）。")
    run_id, run_dir = run
    url = f"http://{args.host}:{args.port}/"

    already = _serving(url)
    if already == run_id:
        print(f"[view] {url} 上已经在看 {run_id}，直接打开（不重复起）。")
        open_target(url, no_open=args.no_open)
        return 0

    def start(port: int):
        return monitor_server.serve(
            run_dir, host=args.host, port=port, run_id=run_id,
            title=f"训练监控 · {run_id}", poll_ms=args.poll_ms)

    conflict = ""
    if already:
        # 端口上跑的是**别的 run**：不能顺手打开（那是别的实验的曲线），
        # 也不硬抢端口——直接用系统分配端口，并把冲突原因说出来。
        conflict = f"端口 {args.port} 上跑的是另一个 run（{already}）"
        want = 0
    else:
        want = args.port

    try:
        srv, _th, url = start(want)
    except OSError as exc:
        conflict = conflict or f"端口 {args.port} 起不来：{exc}"
        if want == 0:
            print(f"[view] {conflict}")
            print(f"[view] 系统分配端口也起不来：{exc}")
            return EXIT_PORT_BUSY
        try:
            srv, _th, url = start(0)
        except OSError as exc2:
            print(f"[view] {conflict}")
            print(f"[view] 退回系统分配端口也失败：{exc2}")
            return EXIT_PORT_BUSY
    if conflict:
        print(f"[view] {conflict} → 本轮改用 {url}")

    store = MetricsStore(run_dir)
    task = store.task() or {}
    print(f"[view] run={run_id} 指标文件 {store.path}")
    print(f"[view] 任务状态 {task.get('status', 'waiting')} · "
          f"{len(store.read()[0])} 条记录")
    print(f"[view] 页面 {url}（Ctrl+C 退出；关掉页面不影响训练）")
    open_target(url, no_open=args.no_open)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[view] 退出")
    finally:
        srv.shutdown()
    return 0


# ---------------------------------------------------------------------------
# list：只列候选，不渲染不起服务
# ---------------------------------------------------------------------------
# history：全仓训练台账（不属于某一轮，所以没有 run-id）
# ---------------------------------------------------------------------------
def cmd_history(args) -> int:
    out = Path(args.out) if args.out else \
        experiments_root() / DEFAULT_HISTORY_OUT
    argv = ["render", "--out", str(out)]
    if args.json:
        argv += ["--json", str(args.json)]
    print(f"[view] 训练台账 → {out}")
    rc = m5_training_history.main(argv)
    if rc != 0:
        print(f"[view] 台账渲染失败（退出码 {rc}）")
        return rc
    open_target(out.resolve().as_uri(), no_open=args.no_open)
    return 0


# ---------------------------------------------------------------------------
def cmd_list(args) -> int:
    root = experiments_root()
    print(f"[view] experiments 根目录 {root}")
    for mode, marker in (("看板", MARKER_DASHBOARD),
                         ("监控", MARKER_MONITOR)):
        print(f"\n== {mode}（{marker}）==")
        _print_candidates(root, marker)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="训练看板/监控的双击入口（只读：不训练、不碰游戏、"
                    "不写 run 目录）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dashboard", help="渲染最近一轮的学习看板并打开浏览器")
    d.add_argument("--run-id", default=None,
                   help="指定 run（默认取 events.jsonl 最新的那个）")
    d.add_argument("--out", default=None,
                   help=f"HTML 输出路径（默认 experiments/{DEFAULT_DASHBOARD_OUT}）")
    d.add_argument("--t13-import", action="store_true",
                   help="额外导入 T13 六个历史 run 的曲线（默认不带："
                        "那是别的实验的数字，不该混进本页）")
    d.add_argument("--no-open", action="store_true",
                   help="只渲染，不开浏览器")
    d.set_defaults(func=cmd_dashboard)

    m = sub.add_parser("monitor",
                       help="给最近一轮带逐 step 指标的 run 起实时监控")
    m.add_argument("--run-id", default=None,
                   help="指定 run（默认取 metrics.jsonl 最新的那个）")
    m.add_argument("--host", default="127.0.0.1")
    m.add_argument("--port", type=int, default=8760)
    m.add_argument("--poll-ms", type=int, default=2000)
    m.add_argument("--no-open", action="store_true",
                   help="只起服务，不开浏览器")
    m.set_defaults(func=cmd_monitor)

    l = sub.add_parser("list", help="列出候选 run（不渲染、不起服务）")
    l.set_defaults(func=cmd_list)

    h = sub.add_parser("history",
                       help="全仓训练台账：所有轮一起看（不做排行）")
    h.add_argument("--out", default=None,
                   help=f"HTML 输出路径（默认 experiments/{DEFAULT_HISTORY_OUT}）")
    h.add_argument("--json", default=None, help="额外导出 JSON 台账")
    h.add_argument("--no-open", action="store_true",
                   help="只渲染，不开浏览器")
    h.set_defaults(func=cmd_history)

    args = ap.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
