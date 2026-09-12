"""reCAPTCHA 式标线图片人工标注服务.

浏览器 3x3 宫格：人工点击所有含漆画标线（车道线/边线/箭头/斑马线）的图片，
提交后逐图追加 JSONL 记录（has_line 0/1）。已标注帧自动跳过，可断点续标；
``--strategy score`` 配合模型不确定度分数（JSON: {帧名: 分数}）优先挑高分帧
（带 epsilon 探索），即人工反馈驱动的主动学习闭环：标注结果可直接作为
标线存在性分类器的训练集，或作为强化学习的奖励/筛选信号。
"""

from __future__ import annotations

import html
import json
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

IMAGE_EXTS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
CONTENT_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                 ".png": "image/png", ".bmp": "image/bmp",
                 ".webp": "image/webp"}
DEFAULT_TASK = "选择所有含有道路标线的图片"
TASK_HINT = ("点击含道路标线的图片（车道线/边线/箭头/斑马线等漆画标线），"
             "再点一次取消；「全选」一键勾选/取消整组；一组标完点「提交」"
             "自动换下一组，整组都没有标线就点「全部无标线」。")
SCORE_EPSILON = 0.15     # score 策略下每个名额的随机探索概率


def scan_images(root: Path, recursive: bool = False) -> list[Path]:
    """收集 root 下的图片文件，按相对路径排序（保证批次可复现）。"""
    root = Path(root)
    found = root.rglob("*") if recursive else root.glob("*")
    return sorted((p for p in found
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTS),
                  key=lambda p: p.relative_to(root).as_posix())


class LabelStore:
    """JSONL 追加式标签存储；重载时同一路径后写覆盖先写（断点续标）。"""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self.labeled: dict[str, int] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            key, val = rec.get("path"), rec.get("has_line")
            if key is None or val is None:
                continue
            self.labeled[key] = int(val)

    def append(self, records: list[dict]) -> None:
        if not records:
            return
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                for rec in records:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    self.labeled[rec["path"]] = int(rec["has_line"])


class BatchPicker:
    """从待标池挑下一批：random 均匀抽样；score 按分数优先（带探索）。"""

    def __init__(self, scores: dict | None = None,
                 epsilon: float = SCORE_EPSILON,
                 rng: random.Random | None = None):
        self.scores = scores or {}
        self.epsilon = float(epsilon)
        self.rng = rng or random.Random()

    def _score(self, name: str) -> float:
        if name in self.scores:
            return float(self.scores[name])
        base = name.rsplit("/", 1)[-1]
        if base in self.scores:
            return float(self.scores[base])
        return float("-inf")     # 没有分数的帧排在最后

    def pick(self, pending: list[str], k: int) -> list[str]:
        if k >= len(pending):
            return list(pending)
        if not self.scores:
            return self.rng.sample(pending, k)
        pool = sorted(pending, key=self._score, reverse=True)
        out: list[str] = []
        while pool and len(out) < k:
            if (self.epsilon and len(pool) > 1
                    and self.rng.random() < self.epsilon):
                out.append(pool.pop(self.rng.randrange(len(pool))))
            else:
                out.append(pool.pop(0))
        return out


class GridLabelApp:
    """宫格标注状态：待标池、批次生成与提交落盘（线程安全）。"""

    def __init__(self, root: Path, paths: list[Path], store: LabelStore,
                 task: str = DEFAULT_TASK, batch_size: int = 9, cols: int = 3,
                 strategy: str = "random", scores: dict | None = None,
                 include_labeled: bool = False, seed: int | None = None):
        self.root = Path(root)
        self.store = store
        self.task = task
        self.batch_size = max(1, int(batch_size))
        self.cols = max(1, int(cols))
        self.strategy = strategy
        self.rng = random.Random(seed)
        self.picker = BatchPicker(scores=scores, rng=self.rng)
        self._lock = threading.Lock()
        self._batch_seq = 0
        self.by_name: dict[str, Path] = {
            p.relative_to(self.root).as_posix(): p for p in paths}
        if include_labeled:
            self.pending = list(self.by_name)
        else:
            self.pending = [n for n in self.by_name
                            if str(self.by_name[n]) not in store.labeled]

    def progress(self) -> dict:
        with self._lock:
            return {"labeled": len(self.by_name) - len(self.pending),
                    "total": len(self.by_name)}

    def next_batch(self) -> dict:
        with self._lock:
            batch = self.picker.pick(self.pending, self.batch_size)
            self._batch_seq += 1
            images = [{"name": n, "url": "/img?name=" + quote(n, safe="")}
                      for n in batch]
            return {"batch_id": self._batch_seq, "images": images,
                    "labeled": len(self.by_name) - len(self.pending),
                    "total": len(self.by_name)}

    def submit(self, labels: dict) -> dict:
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        records: list[dict] = []
        with self._lock:
            for name, val in labels.items():
                if name not in self.by_name or name not in self.pending:
                    continue
                records.append({"ts": ts, "image": name,
                                "path": str(self.by_name[name]),
                                "has_line": 1 if int(val) else 0,
                                "strategy": self.strategy})
            if records:
                self.store.append(records)
                done_names = {r["image"] for r in records}
                self.pending = [n for n in self.pending
                                if n not in done_names]
            return {"ok": True, "accepted": len(records),
                    "done": not self.pending,
                    "labeled": len(self.by_name) - len(self.pending),
                    "total": len(self.by_name)}


PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>__TASK__</title>
<style>
  body{margin:0;min-height:100vh;display:flex;align-items:center;
       justify-content:center;background:rgba(0,0,0,.55);
       font-family:"Microsoft YaHei",Arial,sans-serif}
  .card{background:#fff;border-radius:8px;box-shadow:0 2px 12px rgba(0,0,0,.4);
        padding:24px;width:520px}
  h1{font-size:18px;margin:0 0 6px;color:#202124}
  .hint{color:#5f6368;font-size:12px;line-height:1.6;margin-bottom:12px}
  .grid{display:grid;grid-template-columns:repeat(__COLS__,1fr);gap:6px}
  .cell{position:relative;aspect-ratio:1/1;overflow:hidden;border:3px solid
        transparent;border-radius:4px;cursor:pointer;background:#e8eaed}
  .cell img{width:100%;height:100%;object-fit:cover;display:block}
  .cell.sel{border-color:#1a73e8}
  .cell.sel::after{content:"\\2713";position:absolute;right:4px;top:4px;
        color:#fff;background:#1a73e8;width:20px;height:20px;border-radius:50%;
        text-align:center;line-height:20px;font-size:13px}
  .bar{display:flex;align-items:center;justify-content:space-between;
       margin-top:14px}
  .btn{background:#1a73e8;color:#fff;border:none;border-radius:4px;
       padding:10px 22px;font-size:14px;cursor:pointer}
  .btn:disabled{background:#9db8d9;cursor:default}
  .btn.ghost{background:#fff;color:#1a73e8;border:1px solid #dadce0}
  .prog{color:#5f6368;font-size:12px}
  .done{grid-column:1/-1;text-align:center;padding:40px 0;font-size:16px;
        color:#188038}
</style>
</head>
<body>
<div class="card">
  <h1>__TASK__</h1>
  <div class="hint">__HINT__</div>
  <div class="grid" id="grid"></div>
  <div class="bar">
    <span class="prog" id="prog"></span>
    <span>
      <button class="btn ghost" id="all">全选</button>
      <button class="btn ghost" id="none">全部无标线</button>
      <button class="btn" id="submit">提交</button>
    </span>
  </div>
</div>
<script>
let cur = null, sel = new Set();
const grid = document.getElementById('grid');
const prog = document.getElementById('prog');
async function loadBatch() {
  const d = await (await fetch('/api/batch')).json();
  sel = new Set();
  grid.innerHTML = '';
  if (!d.images.length) {
    grid.innerHTML = '<div class="done">\\u2705 全部标注完成，可关闭页面</div>';
    document.getElementById('submit').disabled = true;
    document.getElementById('none').disabled = true;
    prog.textContent = `已标注 ${d.labeled} / ${d.total}`;
    return;
  }
  for (const im of d.images) {
    const cell = document.createElement('div');
    cell.className = 'cell';
    const img = document.createElement('img');
    img.src = im.url;
    cell.appendChild(img);
    cell.onclick = () => {
      if (sel.has(im.name)) { sel.delete(im.name); }
      else { sel.add(im.name); }
      cell.classList.toggle('sel');
    };
    grid.appendChild(cell);
  }
  prog.textContent = `已标注 ${d.labeled} / ${d.total}`;
  cur = d;
}
async function submit(allZero) {
  if (!cur) return;
  const labels = {};
  for (const im of cur.images) {
    labels[im.name] = (!allZero && sel.has(im.name)) ? 1 : 0;
  }
  await fetch('/api/submit', {method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({labels: labels})});
  loadBatch();
}
document.getElementById('submit').onclick = () => submit(false);
document.getElementById('none').onclick = () => submit(true);
document.getElementById('all').onclick = () => {
  const on = sel.size !== cur.images.length;   // 已全选则一键取消
  sel = new Set();
  grid.querySelectorAll('.cell').forEach((c, i) => {
    c.classList.toggle('sel', on);
    if (on) sel.add(cur.images[i].name);
  });
};
document.addEventListener('keydown', e => { if (e.key === 'Enter') submit(false); });
loadBatch();
</script>
</body>
</html>
"""


def render_page(task: str, cols: int) -> str:
    return (PAGE.replace("__TASK__", html.escape(task))
                .replace("__HINT__", html.escape(TASK_HINT))
                .replace("__COLS__", str(max(1, int(cols)))))


def make_handler(app: GridLabelApp) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):   # 刷图频繁，不刷控制台
            pass

        def _send(self, code: int, ctype: str, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, "application/json; charset=utf-8",
                       json.dumps(obj, ensure_ascii=False).encode("utf-8"))

        def do_GET(self) -> None:
            u = urlparse(self.path)
            if u.path == "/":
                self._send(200, "text/html; charset=utf-8",
                           render_page(app.task, app.cols).encode("utf-8"))
            elif u.path == "/api/batch":
                self._json(app.next_batch())
            elif u.path == "/img":
                name = (parse_qs(u.query).get("name") or [""])[0]
                path = app.by_name.get(name)
                if path is None or not path.is_file():
                    self.send_error(404)
                    return
                ctype = CONTENT_TYPES.get(path.suffix.lower(),
                                          "application/octet-stream")
                self._send(200, ctype, path.read_bytes())
            else:
                self.send_error(404)

        def do_POST(self) -> None:
            u = urlparse(self.path)
            if u.path != "/api/submit":
                self.send_error(404)
                return
            try:
                n = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(n) or b"{}")
                labels = payload.get("labels")
                if not isinstance(labels, dict):
                    raise ValueError("labels must be an object")
            except (ValueError, json.JSONDecodeError):
                self._json({"ok": False, "error": "bad payload"}, 400)
                return
            self._json(app.submit({str(k): v for k, v in labels.items()}))

    return Handler


def serve(app: GridLabelApp, host: str = "127.0.0.1",
          port: int = 8787) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(app))
