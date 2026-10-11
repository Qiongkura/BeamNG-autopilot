"""监控服务：只读 HTTP 接口 + 增量拉取（stdlib，无额外依赖）。

两种"看哪一轮"的模式：

* **固定**：``serve(<run_dir>)`` —— 只看那一轮（给 ``--run-id`` 时是它）；
* **跟随**：``serve(follow=True, root=<experiments 目录>)`` —— 始终看
  ``root/*/metrics.jsonl`` 里 mtime 最新的那一轮；出现更新的一轮就切过去
  （只在**更新**时切，所以正在写的那一轮不会被旧文件抢走）。开了新训练不用
  重启监控，页面自己会清空重画：``/metrics`` 每次都带回当前的 ``run_id``，
  前端据此判断"换轮次了"。

接口（全部只读，不向 autopilot 发控制命令）：

* ``GET /``                    —— 监控页面（由 ``monitor_ui.render_html`` 生成）
* ``GET /metrics?since=<seq>`` —— 增量：只返回 ``seq > since`` 的记录，带
  ``next_since``、``run_id`` 与不可读行清单；前端据此去重、断线重连后接着拉，
  并在 ``run_id`` 变化时清空重画。
* ``GET /state``               —— 轻量状态（任务状态/最新 seq/记录数/是否跟随），
  给脚本与命令行探活。
* ``GET /health``              —— 200 + 记录文件是否可读。

为什么用 stdlib 而不是 WebSocket：项目里没有前端框架也没有推送依赖，
``/metrics?since=`` 已经满足方案要求的"增量 + 去重 + 断线自动恢复"，而且
服务可以被一条命令启停、不占训练 GPU。页面自身用轮询（默认 2 s）。
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from beamng_autopilot import config
from beamng_autopilot.experiments.metrics import MetricsStore
from beamng_autopilot.experiments.monitor_ui import render_html

#: 一轮的标记文件：有它就说明这一轮写过逐 step 指标。
MARKER = "metrics.jsonl"
#: 跟随模式重新解析"哪一轮"的最小间隔（秒）。目录扫描很便宜，但没必要每次
#: 请求都扫；取值与页面轮询节奏一致。
FOLLOW_INTERVAL_S = 2.0


def experiments_root() -> Path:
    """默认跟随的根目录（``logs/experiments``，可用 ``BEAMNG_LOGS_DIR`` 重定向）。"""
    return Path(config.LOGS_DIR) / "experiments"


def newest_run_dir(root: Path | str | None) -> Path | None:
    """``root/<run_id>/metrics.jsonl`` 里 mtime 最新的 run 目录；没有则 ``None``。

    只扫一层、只认标记文件，不做递归：跟随的目标是"正在训练的那一轮"，而
    机器级台账、探针目录等旁路产物没有这个文件。
    """
    base = Path(root) if root else experiments_root()
    if not base.is_dir():
        return None
    best: tuple[float, Path] | None = None
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        m = d / MARKER
        try:
            if not m.is_file():
                continue
            mt = m.stat().st_mtime
        except OSError:
            continue
        if best is None or mt > best[0]:
            best = (mt, d)
    return best[1] if best else None


def _marker_mtime(run_dir: Path | None) -> float:
    if not run_dir:
        return -1.0
    try:
        return (Path(run_dir) / MARKER).stat().st_mtime
    except OSError:
        return -1.0


class FollowTarget:
    """解析"现在服务哪一轮"：固定一轮，或跟随 ``root`` 下最新的一轮。

    ``current()`` 返回 ``{"store", "run_dir", "run_id", "title", "poll_ms"}``；
    跟随模式还没出现任何一轮时 ``run_id`` 是空串、``store`` 指向一个不存在的
    文件（页面显示"等待中"，不编造轮次名）。
    """

    def __init__(self, run_dir: Path | str | None = None, *, run_id: str = "",
                 title: str = "", poll_ms: int = 2000, follow: bool = False,
                 root: Path | str | None = None,
                 interval: float = FOLLOW_INTERVAL_S) -> None:
        self.follow = bool(follow)
        self.root = Path(root) if root else experiments_root()
        self.interval = max(0.2, float(interval))
        self._next_check = 0.0
        self._cur = self._make(run_dir, run_id, title, poll_ms)
        if self.follow and run_dir is None:
            self.current()                       # 立刻解析一次，别等第一次请求
        self._cur["poll_ms"] = int(poll_ms)

    def _make(self, run_dir, run_id: str, title: str, poll_ms: int) -> dict:
        if run_dir is None:
            # 跟随模式尚未出现任何一轮：给一个空 store，页面照常渲染"等待中"。
            return {"store": MetricsStore(self.root / "_no_run_yet_"),
                    "run_dir": None, "run_id": "",
                    "title": title or "训练监控（跟随最新一轮）",
                    "poll_ms": int(poll_ms)}
        d = Path(run_dir)
        rid = run_id or d.name
        return {"store": MetricsStore(d), "run_dir": d, "run_id": rid,
                "title": title or f"训练监控 · {rid}", "poll_ms": int(poll_ms)}

    def current(self) -> dict:
        if not self.follow:
            return self._cur
        now = time.monotonic()
        if self._cur["run_dir"] is not None and now < self._next_check:
            return self._cur
        self._next_check = now + self.interval
        newest = newest_run_dir(self.root)
        if newest is None:
            return self._cur
        cur_dir = self._cur["run_dir"]
        if cur_dir is None or Path(newest) != Path(cur_dir):
            # 只在真的更新时切：正在写的那一轮 mtime 一直在涨，不会被抢走。
            if _marker_mtime(newest) > _marker_mtime(cur_dir):
                if cur_dir is not None:
                    print(f"[monitor] 跟随切换到 {Path(newest).name}")
                poll_ms = self._cur["poll_ms"]
                self._cur = self._make(newest, "", "", poll_ms)
        return self._cur


def metrics_query(store: MetricsStore, since: int, run_id: str = "") -> dict:
    """``/metrics`` 的纯逻辑（单测直接调用，不必起端口）。

    ``run_id`` 是这一份响应对应的轮次：前端靠它发现服务端跟随到了新的一轮，
    然后清空重画（否则新轮次的 seq 从 0 开始，旧 ``since`` 会让页面永远拉不到
    新数据）。
    """
    out = store.read_since(int(since))
    task = store.task()
    out["task"] = task
    out["run_id"] = run_id
    from beamng_autopilot.experiments.learning_status import read_learning_status
    out["pipeline"] = read_learning_status(Path(store.run_dir))
    return out



class MonitorHandler(BaseHTTPRequestHandler):
    target: FollowTarget = None         # type: ignore[assignment]

    def log_message(self, fmt, *a):     # noqa: A003 - 静默，别刷训练日志
        return

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):                   # noqa: N802 - BaseHTTPRequestHandler 接口
        cur = self.target.current()      # 跟随模式下每次请求都问一次
        store, run_id = cur["store"], cur["run_id"]
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            html = render_html(run_id=run_id, title=cur["title"],
                               mode="live", poll_ms=cur["poll_ms"])
            self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
            return
        if u.path == "/metrics":
            q = parse_qs(u.query)
            try:
                since = int((q.get("since") or ["0"])[0])
            except ValueError:
                self._send(400, b'{"error":"since must be an integer"}',
                           "application/json")
                return
            body = json.dumps(metrics_query(store, since, run_id),
                              ensure_ascii=False, default=str,
                              allow_nan=False).encode("utf-8")
            self._send(200, body, "application/json; charset=utf-8")
            return
        if u.path == "/state":
            task = store.task()
            body = json.dumps({"run_id": run_id, "follow": self.target.follow,
                               "status": (task or {}).get("status", "waiting"),
                               "last_seq": store.last_seq(),
                               "n_records": len(store.read()[0])},
                              ensure_ascii=False).encode("utf-8")
            self._send(200, body, "application/json; charset=utf-8")
            return
        if u.path == "/health":
            recs, problems = store.read()
            self._send(200, json.dumps({"ok": True, "n": len(recs),
                                        "problems": problems}).encode("utf-8"),
                       "application/json")
            return
        self._send(404, b'{"error":"not found"}', "application/json")


def serve(run_dir: Path | str | None = None, *, host: str = "127.0.0.1",
          port: int = 8760, run_id: str = "", title: str = "",
          poll_ms: int = 2000, follow: bool = False,
          root: Path | str | None = None,
          follow_interval: float = FOLLOW_INTERVAL_S,
          target: FollowTarget | None = None) -> tuple:
    """起服务；``port=0`` 让系统分配。返回 ``(server, thread, url)``。

    ``run_dir`` 给定时看那一轮；``follow=True`` 时改看 ``root``（默认
    ``logs/experiments``）下 ``metrics.jsonl`` 最新的那一轮，并在出现更新的
    一轮时自动切过去——开了新训练不用重启监控。``target`` 可直接传入
    （调用方要在起服务前就知道"当前是哪一轮"时用）。

    只绑定回环地址：这是本机监控页，不需要对局域网暴露训练进程的文件。
    """
    tgt = target if target is not None else FollowTarget(
        run_dir, run_id=run_id, title=title, poll_ms=poll_ms,
        follow=follow, root=root, interval=follow_interval)
    handler = type("_H", (MonitorHandler,), {"target": tgt})
    srv = ThreadingHTTPServer((host, int(port)), handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True,
                          name="monitor-server")
    th.start()
    url = f"http://{host}:{srv.server_address[1]}/"
    return srv, th, url
