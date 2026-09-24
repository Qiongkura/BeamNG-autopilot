"""监控服务：只读 HTTP 接口 + 增量拉取（stdlib，无额外依赖）。

接口（全部只读，不向 autopilot 发控制命令）：

* ``GET /``                    —— 监控页面（由 ``monitor_ui.render_html`` 生成）
* ``GET /metrics?since=<seq>`` —— 增量：只返回 ``seq > since`` 的记录，带
  ``next_since`` 与不可读行清单；前端据此去重、断线重连后接着拉。
* ``GET /state``               —— 轻量状态（任务状态/最新 seq/记录数），
  给脚本与命令行探活。
* ``GET /health``              —— 200 + 记录文件是否可读。

为什么用 stdlib 而不是 WebSocket：项目里没有前端框架也没有推送依赖，
``/metrics?since=`` 已经满足方案要求的"增量 + 去重 + 断线自动恢复"，而且
服务可以被一条命令启停、不占训练 GPU。页面自身用轮询（默认 2 s）。
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from beamng_autopilot.experiments.metrics import MetricsStore
from beamng_autopilot.experiments.monitor_ui import render_html


def metrics_query(store: MetricsStore, since: int) -> dict:
    """``/metrics`` 的纯逻辑（单测直接调用，不必起端口）。"""
    out = store.read_since(int(since))
    task = store.task()
    out["task"] = task
    return out


class MonitorHandler(BaseHTTPRequestHandler):
    store: MetricsStore = None          # type: ignore[assignment]
    run_id: str = ""
    title: str = ""
    poll_ms: int = 2000

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
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            html = render_html(run_id=self.run_id, title=self.title,
                               mode="live", poll_ms=self.poll_ms)
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
            body = json.dumps(metrics_query(self.store, since),
                              ensure_ascii=False, default=str,
                              allow_nan=False).encode("utf-8")
            self._send(200, body, "application/json; charset=utf-8")
            return
        if u.path == "/state":
            task = self.store.task()
            body = json.dumps({"run_id": self.run_id, "status":
                               (task or {}).get("status", "waiting"),
                               "last_seq": self.store.last_seq(),
                               "n_records": len(self.store.read()[0])},
                              ensure_ascii=False).encode("utf-8")
            self._send(200, body, "application/json; charset=utf-8")
            return
        if u.path == "/health":
            recs, problems = self.store.read()
            self._send(200, json.dumps({"ok": True, "n": len(recs),
                                        "problems": problems}).encode("utf-8"),
                       "application/json")
            return
        self._send(404, b'{"error":"not found"}', "application/json")


def serve(run_dir: Path | str, *, host: str = "127.0.0.1", port: int = 8760,
          run_id: str = "", title: str = "", poll_ms: int = 2000) -> tuple:
    """起服务；``port=0`` 让系统分配。返回 ``(server, thread, url)``。

    只绑定回环地址：这是本机监控页，不需要对局域网暴露训练进程的文件。
    """
    store = MetricsStore(Path(run_dir))
    handler = type("_H", (MonitorHandler,), {
        "store": store, "run_id": run_id or Path(run_dir).name,
        "title": title or f"训练监控 · {Path(run_dir).name}",
        "poll_ms": int(poll_ms)})
    srv = ThreadingHTTPServer((host, int(port)), handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True,
                          name="monitor-server")
    th.start()
    url = f"http://{host}:{srv.server_address[1]}/"
    return srv, th, url
