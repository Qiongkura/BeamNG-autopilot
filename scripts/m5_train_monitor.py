"""训练监控入口：看板服务 / 离线快照 / 演示数据（薄入口，逻辑在库里）。

用法::

    # 实时看板：不给 --run-id 就**跟随最新一轮**（开了新训练不用重启监控）
    pwsh> .venv\\Scripts\\python.exe scripts\\m5_train_monitor.py serve --port 8760

    # 钉住某一轮（不跟随）
    pwsh> .venv\\Scripts\\python.exe scripts\\m5_train_monitor.py serve --run-id t14_monitor_demo

    # 训练结束后回看（生成自包含 HTML，不需要服务；不给 --run-id 取最新一轮）
    pwsh> .venv\\Scripts\\python.exe scripts\\m5_train_monitor.py snapshot `
              --run-id t14_monitor_demo --out logs\\experiments\\monitor_snapshot.html

    # 演示数据（只用于界面联调，页面会显示 DEMO 横幅）
    pwsh> .venv\\Scripts\\python.exe scripts\\m5_train_monitor.py demo `
              --run-id demo1 --steps 400 --with-unavailable-power

训练进程写指标的方式见 ``scripts/m5_train_seg.py --metrics-run <run_id>``：
每个优化步一行 train 记录，硬件按 ``--monitor-interval``（默认 2 s）采样。
**跟随模式只认写了这个文件的轮次**：没带 ``--metrics-run`` 的训练不会出现，
页面显示"等待中"，不编造轮次。
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


def _newest_or_die(args) -> str:
    """``--run-id`` 没给时取最新一轮；没有任何一轮就报原因退出（不猜）。"""
    if args.run_id:
        return args.run_id
    d = monitor_server.newest_run_dir(args.root)
    if d is None:
        root = Path(args.root) if args.root else monitor_server.experiments_root()
        print(f"[monitor] {root} 下没有任何带 "
              f"{monitor_server.MARKER} 的轮次：逐 step 指标只在训练带 "
              f"--metrics-run 时才写。用 --run-id 指定其它轮次。")
        raise SystemExit(2)
    print(f"[monitor] 未指定 --run-id：取最新一轮 {d.name}")
    return d.name


def cmd_serve(args) -> int:
    """起服务。给了 ``--run-id`` 就钉住那一轮；没给就跟随最新一轮。"""
    follow = args.run_id is None
    target = monitor_server.FollowTarget(
        None if follow else run_dir(args.run_id),
        run_id="" if follow else args.run_id,
        title=args.title, poll_ms=args.poll_ms, follow=follow,
        root=args.root, interval=args.follow_interval)
    cur = target.current()
    if follow:
        print(f"[monitor] 跟随最新一轮：根目录 {target.root}，每 "
              f"{args.follow_interval:g}s 检查一次，出现更新的一轮自动切换"
              f"（页面自己清空重画，不用重启）")
    if cur["run_id"]:
        print(f"[monitor] 当前轮次 {cur['run_id']} · 指标文件 "
              f"{cur['store'].path}")
    else:
        print("[monitor] 还没有可跟随的轮次：页面显示『等待中』，等训练带上 "
              "--metrics-run 开始写指标就会自动接上")

    _srv, _th, url = monitor_server.serve(target=target, host=args.host,
                                          port=args.port, poll_ms=args.poll_ms)
    store = cur["store"]
    task = store.task()
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
    run_id = _newest_or_die(args)
    store = MetricsStore(run_dir(run_id))
    recs, problems = store.read()
    if not recs:
        print(f"[monitor] {store.path} 没有记录：快照会是一张空页"
              f"（不编造数据）")
    p = write_snapshot(args.out, run_id=run_id, records=recs,
                       title=args.title or f"训练监控快照 · {run_id}")
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
    run_id = _newest_or_die(args)
    store = MetricsStore(run_dir(run_id))
    recs, problems = store.read()
    task = store.task() or {}
    print(f"[monitor] run={run_id} 记录 {len(recs)} 条 "
          f"last_seq={store.last_seq()} 状态={task.get('status', 'waiting')}")
    if task.get("error"):
        print(f"[monitor] 失败原因：{task['error']}")
    if problems:
        print(f"[monitor] 不可读行 {len(problems)}：{problems[:3]}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="训练过程可视化（实时/快照/演示）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="起只读监控服务（不给 --run-id 就跟随最新一轮）")
    s.add_argument("--run-id", default=None,
                   help="钉住某一轮；不给则跟随根目录下最新的一轮")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8760)
    s.add_argument("--title", default="")
    s.add_argument("--poll-ms", type=int, default=2000)
    s.add_argument("--root", default=None,
                   help="跟随哪个目录（默认 logs/experiments）")
    s.add_argument("--follow-interval", type=float,
                   default=monitor_server.FOLLOW_INTERVAL_S,
                   help=f"跟随模式多久检查一次新轮次（秒，默认 "
                        f"{monitor_server.FOLLOW_INTERVAL_S:g}）")
    s.set_defaults(func=cmd_serve)

    s = sub.add_parser("snapshot", help="把已有记录渲染成自包含 HTML")
    s.add_argument("--run-id", default=None, help="不给则取最新一轮")
    s.add_argument("--out", required=True)
    s.add_argument("--title", default="")
    s.add_argument("--root", default=None)
    s.set_defaults(func=cmd_snapshot)

    s = sub.add_parser("demo", help="写入演示记录（界面会标 DEMO）")
    s.add_argument("--run-id", default="demo1")
    s.add_argument("--steps", type=int, default=300)
    s.add_argument("--no-hardware", action="store_true")
    s.add_argument("--with-unavailable-power", action="store_true")
    s.set_defaults(func=cmd_demo)

    s = sub.add_parser("probe", help="打印一次真实硬件采样")
    s.set_defaults(func=cmd_probe)

    s = sub.add_parser("status", help="打印指标文件状态（不给 --run-id 取最新一轮）")
    s.add_argument("--run-id", default=None)
    s.add_argument("--root", default=None)
    s.set_defaults(func=cmd_status)

    args = ap.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
