"""训练监控入口：看板服务 / 离线快照 / 演示数据（薄入口，逻辑在库里）。

用法::

    # 实时看板（训练进行中刷新浏览器即可；默认只绑回环地址）
    pwsh> .venv\\Scripts\\python.exe scripts\\m5_train_monitor.py serve --run-id t14_monitor_demo --port 8760

    # 训练结束后回看（生成自包含 HTML，不需要服务）
    pwsh> .venv\\Scripts\\python.exe scripts\\m5_train_monitor.py snapshot `
              --run-id t14_monitor_demo --out logs\\experiments\\monitor_snapshot.html

    # 演示数据（只用于界面联调，页面会显示 DEMO 横幅）
    pwsh> .venv\\Scripts\\python.exe scripts\\m5_train_monitor.py demo `
              --run-id demo1 --steps 400 --with-unavailable-power

训练进程写指标的方式见 ``scripts/m5_train_seg.py --metrics-run <run_id>``：
每个优化步一行 train 记录，硬件按 ``--monitor-interval``（默认 2 s）采样。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beamng_autopilot import config  # noqa: E402
from beamng_autopilot.experiments import monitor_server  # noqa: E402
from beamng_autopilot.experiments.metrics import (  # noqa: E402
    MetricsStore, demo_records, sample_system, task_record,
)
from beamng_autopilot.experiments.monitor_ui import write_snapshot  # noqa: E402


def run_dir(run_id: str) -> Path:
    return Path(config.LOGS_DIR) / "experiments" / run_id


def cmd_serve(args) -> int:
    d = run_dir(args.run_id)
    d.mkdir(parents=True, exist_ok=True)
    _srv, _th, url = monitor_server.serve(
        d, host=args.host, port=args.port, run_id=args.run_id,
        title=args.title or f"训练监控 · {args.run_id}", poll_ms=args.poll_ms)
    store = MetricsStore(d)
    task = store.task()
    print(f"[monitor] 指标文件 {store.path}")
    print(f"[monitor] 任务状态 {(task or {}).get('status', 'waiting')} · "
          f"{len(store.read()[0])} 条记录")
    print(f"[monitor] 页面 {url}  （Ctrl+C 退出；关闭页面不影响训练）")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[monitor] 退出")
    return 0


def cmd_snapshot(args) -> int:
    store = MetricsStore(run_dir(args.run_id))
    recs, problems = store.read()
    if not recs:
        print(f"[monitor] {store.path} 没有记录：快照会是一张空页"
              f"（不编造数据）")
    p = write_snapshot(args.out, run_id=args.run_id, records=recs,
                       title=args.title or f"训练监控快照 · {args.run_id}")
    task = store.task() or {}
    print(f"[monitor] {len(recs)} 条记录 -> {p}（状态 "
          f"{task.get('status', 'waiting')}，不可读行 {len(problems)}）")
    return 0


def cmd_demo(args) -> int:
    store = MetricsStore(run_dir(args.run_id))
    un = ({"gpu0.power_w": "this device does not provide power readings"}
          if args.with_unavailable_power else None)
    recs = demo_records(args.run_id, steps=args.steps,
                        with_hardware=not args.no_hardware, unavailable=un)
    for r in recs:
        store.append(r)
    print(f"[monitor] DEMO：写入 {len(recs)} 条演示记录 -> {store.path}")
    if un:
        print("[monitor] 已注入『设备未提供功耗』场景（界面应显示提示而不是 0）")
    print(f"[monitor] 看它：serve --run-id {args.run_id}")
    return 0


def cmd_probe(args) -> int:
    """打印一次真实硬件采样（排查"为什么图是空的"用）。"""
    rec = sample_system()
    print(json.dumps(rec, indent=1, ensure_ascii=False))
    return 0


def cmd_status(args) -> int:
    store = MetricsStore(run_dir(args.run_id))
    recs, problems = store.read()
    task = store.task() or {}
    print(f"[monitor] run={args.run_id} 记录 {len(recs)} 条 "
          f"last_seq={store.last_seq()} 状态={task.get('status', 'waiting')}")
    if task.get("error"):
        print(f"[monitor] 失败原因：{task['error']}")
    if problems:
        print(f"[monitor] 不可读行 {len(problems)}：{problems[:3]}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="训练过程可视化（实时/快照/演示）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="起只读监控服务")
    s.add_argument("--run-id", required=True)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8760)
    s.add_argument("--title", default="")
    s.add_argument("--poll-ms", type=int, default=2000)
    s.set_defaults(func=cmd_serve)

    s = sub.add_parser("snapshot", help="把已有记录渲染成自包含 HTML")
    s.add_argument("--run-id", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--title", default="")
    s.set_defaults(func=cmd_snapshot)

    s = sub.add_parser("demo", help="写入演示记录（界面会标 DEMO）")
    s.add_argument("--run-id", default="demo1")
    s.add_argument("--steps", type=int, default=300)
    s.add_argument("--no-hardware", action="store_true")
    s.add_argument("--with-unavailable-power", action="store_true")
    s.set_defaults(func=cmd_demo)

    s = sub.add_parser("probe", help="打印一次真实硬件采样")
    s.set_defaults(func=cmd_probe)

    s = sub.add_parser("status", help="打印指标文件状态")
    s.add_argument("--run-id", required=True)
    s.set_defaults(func=cmd_status)

    args = ap.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
