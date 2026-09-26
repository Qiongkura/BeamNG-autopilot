"""T14 训练台账：把全仓训练产物扫成一张可查的历史表（只读，一张自包含 HTML）。

和 ``m5_seg_dashboard.py`` 的分工：看板是**一轮一页**（带证据等级、成对比较、
硬门判定），台账是**所有轮一页**（谁在什么时候训了什么、训了多少、末轮数字
是多少、决策结果是什么）。两者扫描范围、更新频率、可比性口径都不同，所以
不塞进看板。

它专门避免三个已经出过错的陷阱：

1. **跨实验排行**：297 个 run 的数据集、验证划分、epoch 数都不一样，
   "line IoU 最高的模型"这种排行是在拿不可比的东西比。台账只列事实，
   并在页头写死"本页不做排行，要比较请看同一轮的看板"；每列都标证据等级
   （本页全部是 ``开发集``，因为来源只有 ``train_hist.json`` 的验证划分）。
2. **缺列被画成 0**：缺键、``None``、非有限值一律渲染成"缺列"，统计里单独
   计数（``缺列 N 个``）；只有真记录到的 0 才写 0。
3. **状态靠猜**：run 的完成状态只看产物（有 ``best.pt`` / 只有
   ``checkpoint_last.pt`` / 都没有），不看 mtime 也不假设"跑过就是完成"。

只读纪律：不训练、不接触游戏、不写 run 目录；除 ``--out`` / ``--json``
指定的文件外不落盘。

用法::

    pwsh> .venv\\Scripts\\python.exe scripts\\m5_training_history.py render `
              --out logs\\experiments\\training_history.html
    pwsh> .venv\\Scripts\\python.exe scripts\\m5_training_history.py list --limit 20
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from beamng_autopilot import config  # noqa: E402

#: 训练产物只在这两个命名子树下扫（不做全仓递归扫描）。
SCAN_ROOTS = ("m5_seg", "experiments")
HIST_NAME = "train_hist.json"
DECISION_NAME = "decision_"          # decision_<candidate>-r<N>.json
FINAL_LEDGER = "final_set_ledger.jsonl"

#: ``train_hist.json`` 的逐 epoch 键。缺键 = 缺列，不写 0。
HIST_KEYS = ("train_loss", "val_acc", "val_miou", "val_line_iou")
MAIN_DEV_KEY = "val_line_iou"        # 台账里"最好 epoch"按它取（项目主指标）

#: 本页所有指标的唯一来源与等级：各 run 自己的验证划分。
LEVEL_DEV = "开发集"
MISSING = "缺列"


# ---------------------------------------------------------------------------
# 读产物
# ---------------------------------------------------------------------------
def _finite(v) -> float | None:
    """只接受有限数值；``None``/``NaN``/``inf``/非数值都是"缺列"。"""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    f = float(v)
    return f if f == f and abs(f) != float("inf") else None


def _series(blob: dict, key: str) -> list:
    v = blob.get(key)
    return list(v) if isinstance(v, list) else []


@dataclass(frozen=True)
class RunRow:
    """一次训练（一个 ``train_hist.json``）。"""

    rel: str                 # logs 下的相对路径（台账里的唯一标识）
    root: str                # 顶层：m5_seg / experiments
    family: str              # 分组键（相对顶层的第一个目录）
    sub: str                 # family 内部的子路径（'' 表示 family 自己就是 run）
    mtime: float
    epochs: int
    last: dict               # 末轮 4 个键 → float | None
    best_dev: float | None   # 最佳 val_line_iou（缺列则 None）
    best_dev_epoch: int | None
    missing_keys: tuple      # 缺列的键名
    best_ckpt: bool
    last_ckpt: bool
    drill_run_id: str        # 可下钻的 run-id（experiments 顶层的 run 目录名）
    line_ignored: int | None = None   # line_ignored_frames 末值（产物自报）

    @property
    def status(self) -> str:
        if self.epochs == 0:
            return "空曲线"
        if self.best_ckpt:
            return "完成（有 best.pt）"
        return "中断（只到 last）" if self.last_ckpt else "无权重产物"

    @property
    def label(self) -> str:
        return f"{self.family}/{self.sub}" if self.sub else self.family

    @property
    def missing_reason(self) -> str:
        """``val_line_iou`` 缺列的详细原因——只读产物自报的字段，不推测。

        ``line_ignored_frames`` 是训练器在 line 通道被屏蔽时逐 epoch 记的
        帧数（该工作线同批把 ``val_line_iou`` 记成 None）；没有这一列的老
        run 早于标线通道本身。
        """
        if not any("val_line_iou" in m for m in self.missing_keys):
            return ""
        if any("无该列" in m for m in self.missing_keys):
            return "产物没有 val_line_iou 列（早于标线通道）"
        if self.line_ignored:
            return f"line 通道被屏蔽（line_ignored_frames 末值 {self.line_ignored}）"
        return "末轮 val_line_iou 非数值"

    @property
    def missing_kind(self) -> str:
        """缺列的短类别（汇总用；细节在 ``missing_reason``）。"""
        r = self.missing_reason
        if not r:
            return ""
        return r.split("（")[0]


def scan_runs(logs_dir: Path) -> list[RunRow]:
    """扫 ``<logs>/m5_seg`` 与 ``<logs>/experiments`` 下的全部训练产物。"""
    rows: list[RunRow] = []
    for top in SCAN_ROOTS:
        base = logs_dir / top
        if not base.is_dir():
            continue
        for hist in sorted(base.rglob(HIST_NAME)):
            parts = hist.relative_to(base).parts[:-1]
            if not parts:
                continue
            family, sub = parts[0], "/".join(parts[1:])
            try:
                blob = json.loads(hist.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                blob = {}
            if not isinstance(blob, dict):
                blob = {}
            last, missing, n_ep = {}, [], 0
            for key in HIST_KEYS:
                series = _series(blob, key)
                n_ep = max(n_ep, len(series))
                if not series:
                    last[key] = None
                    missing.append(f"{key}（无该列）")
                    continue
                val = _finite(series[-1])
                last[key] = val
                if val is None:
                    missing.append(f"{key}（末轮非数值）")
            dev = [_finite(v) for v in _series(blob, MAIN_DEV_KEY)]
            pairs = [(v, i) for i, v in enumerate(dev) if v is not None]
            # 取最佳值；同值取更后的 epoch（max 对 (值, 下标) 元组就是这个语义）。
            best_dev, best_ep = max(pairs) if pairs else (None, None)
            ignored = [v for v in (_finite(x)
                                   for x in _series(blob, "line_ignored_frames"))
                       if v is not None]
            d = hist.parent
            rows.append(RunRow(
                rel=str(hist.relative_to(logs_dir)), root=top, family=family,
                sub=sub, mtime=hist.stat().st_mtime, epochs=n_ep, last=last,
                best_dev=best_dev, best_dev_epoch=best_ep,
                missing_keys=tuple(missing),
                best_ckpt=(d / "best.pt").is_file(),
                last_ckpt=(d / "checkpoint_last.pt").is_file(),
                drill_run_id=family if top == "experiments" else "",
                line_ignored=int(ignored[-1]) if ignored else None,
            ))
    rows.sort(key=lambda r: (-r.mtime, r.rel))
    return rows


@dataclass(frozen=True)
class DecisionRow:
    """一条硬门判定（``experiments/<run>/decision_<candidate>-r<N>.json``）。"""

    rel: str
    run: str
    candidate: str
    decision: str
    reasons: tuple
    gates: tuple
    pairings: tuple          # ((metric, mean_delta, verdict, n), ...)
    research_only: bool
    mtime: float


def scan_decisions(logs_dir: Path) -> list[DecisionRow]:
    base = logs_dir / "experiments"
    rows: list[DecisionRow] = []
    if not base.is_dir():
        return rows
    for p in sorted(base.rglob(f"{DECISION_NAME}*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(d, dict):
            continue
        dd = d.get("decision") or {}
        pairs = []
        for name, pr in (d.get("pairings") or {}).items():
            if isinstance(pr, dict):
                pairs.append((name, _finite(pr.get("mean_delta")),
                              str(pr.get("verdict", "")), pr.get("n")))
        rows.append(DecisionRow(
            rel=str(p.relative_to(logs_dir)),
            run=p.relative_to(base).parts[0],
            candidate=str(d.get("candidate_id", "?")),
            decision=str(dd.get("decision", "?")),
            reasons=tuple(str(r) for r in (dd.get("reasons") or [])),
            gates=tuple(str(g) for g in (d.get("hard_gate_violations") or [])),
            pairings=tuple(pairs),
            research_only=bool(d.get("research_only")),
            mtime=p.stat().st_mtime,
        ))
    rows.sort(key=lambda r: (-r.mtime, r.rel))
    return rows


def production_model() -> dict:
    """生产链当前会加载的权重（``default_model_path`` 是权威解析，含地图专家）。

    该解析函数在 ``vision.segmentation`` 里，导入会拉起 torch（约 2 s），
    所以只在渲染这一节时惰性导入；``list`` 模式不付这个代价。
    """
    out = {"path": "", "sha16": "", "mtime": "", "size_mb": None, "error": ""}
    try:
        from beamng_autopilot.vision.segmentation import default_model_path
        p = default_model_path()
    except Exception as exc:                      # pragma: no cover - 环境问题
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out
    if not p:
        out["error"] = "default_model_path() 返回 None（生产权重不存在）"
        return out
    out["path"] = str(p)
    st = p.stat()
    out["sha16"] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    out["mtime"] = time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime))
    out["size_mb"] = round(st.st_size / 1e6, 1)
    experts = sorted(p.parent.glob("by_map/*/best.pt"))
    out["experts"] = [str(e) for e in experts]
    return out


def final_set_state(logs_dir: Path) -> dict:
    """最终集确认：只读封存账本，逐条列出来，不替它下结论。

    账本里既有访问控制记录（allowed/reasons）也有确认结果（``result``）；
    所以判定只认"有结果的条目"，没结果就写"未测"，不把访问记录读成确认。
    """
    out: dict = {"ledgers": [], "n_entries": 0, "n_with_result": 0, "error": ""}
    for p in sorted(logs_dir.rglob(FINAL_LEDGER)):
        try:
            lines = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()
                     if x.strip()]
        except (OSError, ValueError) as exc:
            out["error"] = f"{p}: {exc}"
            continue
        rows = []
        for e in lines:
            if not isinstance(e, dict):
                continue
            rows.append({
                "t": str(e.get("t", "")),
                "candidate": str(e.get("candidate_id", "")),
                "caller": str(e.get("caller", "")),
                "purpose": str(e.get("purpose", "")),
                "allowed": bool(e.get("allowed")),
                "result": str(e.get("result") or ""),
                "reasons": [str(x) for x in (e.get("reasons") or [])],
            })
        out["ledgers"].append({"rel": str(p.relative_to(logs_dir)),
                               "n": len(rows), "entries": rows})
        out["n_entries"] += len(rows)
        out["n_with_result"] += sum(1 for r in rows if r["result"])
    return out


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
def families(rows: list[RunRow]) -> list[dict]:
    """按 family 汇总（只做计数与时间跨度，**不排名**）。"""
    groups: dict[tuple, list[RunRow]] = {}
    for r in rows:
        groups.setdefault((r.root, r.family), []).append(r)
    out = []
    for (root, fam), rs in groups.items():
        dev = [r.best_dev for r in rs if r.best_dev is not None]
        out.append({
            "root": root, "family": fam, "n_runs": len(rs),
            "epochs": sum(r.epochs for r in rs),
            "first": min(r.mtime for r in rs),
            "last": max(r.mtime for r in rs),
            "n_with_dev": len(dev),
            "dev_range": (min(dev), max(dev)) if dev else None,
            "n_best_ckpt": sum(1 for r in rs if r.best_ckpt),
        })
    out.sort(key=lambda g: -g["last"])
    return out


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------
def _fmt(v, width: int = 0) -> str:
    if v is None:
        return MISSING
    s = f"{v:.4f}"
    return s.rjust(width) if width else s


def _stamp(ts: float) -> str:
    return time.strftime("%m-%d %H:%M", time.localtime(ts))


CSS = """
body{font:13px/1.55 "Segoe UI","Microsoft YaHei",sans-serif;margin:0;padding:18px 22px;
     background:#f6f7f9;color:#1d2329}
h1{font-size:20px;margin:0 0 4px}
h2{font-size:15px;margin:22px 0 8px;padding-bottom:5px;border-bottom:1px solid #d8dee6}
.sub{color:#5b6774;margin:2px 0}
.warn{background:#fff6e5;border-left:4px solid #e0a03a;padding:9px 12px;margin:10px 0;
      border-radius:3px}
.ok{color:#1f7a3f}.bad{color:#b3261e}
table{border-collapse:collapse;width:100%;background:#fff;font-size:12px}
th,td{border:1px solid #e2e7ee;padding:3px 6px;text-align:left;white-space:nowrap}
th{background:#eef2f7;position:sticky;top:0;font-weight:600}
td.num,th.num{text-align:right}
tr:nth-child(even) td{background:#fbfcfd}
.mut{color:#7b8794}
.tag{display:inline-block;padding:0 5px;border-radius:8px;font-size:11px}
.tag.dev{background:#e6f0ff;color:#1c4f9c}
.tag.miss{background:#f0f0f0;color:#5b6774}
.wrap{max-height:70vh;overflow:auto;border:1px solid #d8dee6;border-radius:4px}
#q{width:260px;padding:4px 7px;border:1px solid #c8d0da;border-radius:3px;margin:6px 0}
code{background:#eef2f7;padding:1px 4px;border-radius:3px}
"""

JS = """
function filt(){
  var q=document.getElementById('q').value.toLowerCase();
  var rows=document.querySelectorAll('#runs tbody tr');
  for(var i=0;i<rows.length;i++){
    rows[i].style.display = rows[i].getAttribute('data-k').indexOf(q)>=0 ? '' : 'none';
  }
  document.getElementById('n').textContent =
    document.querySelectorAll('#runs tbody tr:not([style*="none"])').length;
}
"""


def _decision_cell(dr: DecisionRow) -> str:
    parts = []
    for name, delta, verdict, n in dr.pairings:
        parts.append(f"{html.escape(name)} Δ{_fmt(delta)} → {html.escape(verdict)}"
                     f"（n={n}）")
    txt = "；".join(parts) or MISSING
    if dr.reasons:
        txt += "<br><span class='mut'>" + html.escape(dr.reasons[0][:120]) + "</span>"
    return txt


def render_html(ctx: dict) -> str:
    prod = ctx["production"]
    fs = ctx["final_set"]
    runs = ctx["runs"]
    dec = ctx["decisions"]
    fams = ctx["families"]

    parts = [
        "<!DOCTYPE html>", '<html lang="zh-CN">', "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>T14 训练台账</title>",
        f"<style>{CSS}</style>", f"<script>{JS}</script>", "</head>", "<body>",
        "<h1>T14 训练台账（只读）</h1>",
        f'<p class="sub">生成时间（本地）: '
        f'{time.strftime("%Y-%m-%d %H:%M:%S")} · 扫描 '
        f'<code>{html.escape(ctx["scan"]["logs_dir"])}</code> 下的 '
        f'{"、".join(html.escape(r) for r in SCAN_ROOTS)}</p>',
        '<p class="sub">只读：不训练、不接触游戏、不写任何 run 目录；'
        '缺键与非有限值一律写"缺列"，只有真记录到的 0 才写 0。</p>',
        f'<div class="warn"><b>本页不做排行。</b>全部指标都是各 run '
        f'<b>{LEVEL_DEV}</b>（自己的验证划分）的末轮/最佳值：数据集、划分、'
        f'epoch 数都不同，跨 family 比大小没有意义。要比较请看同一轮的看板'
        f'（成对比较 + 硬门判定），或 eval matrix（同一批输入）。'
        f'台账的用途是"谁在什么时候训了什么、结果是什么"。</div>',
    ]

    s = ctx["scan"]
    parts.append("<h2>扫描范围与计数</h2><table>")
    parts.append(f'<tr><th>训练 run（{HIST_NAME}）</th><td class="num">{s["n_runs"]}</td>'
                 f'<td class="mut">m5_seg {s["n_by_root"].get("m5_seg", 0)} · '
                 f'experiments {s["n_by_root"].get("experiments", 0)}</td></tr>')
    parts.append(f'<tr><th>合计 epoch</th><td class="num">{s["epochs"]}</td>'
                 f'<td class="mut">按各 run 自报的 epoch 数相加</td></tr>')
    parts.append(f'<tr><th>家族数</th><td class="num">{s["n_families"]}</td>'
                 f'<td class="mut">分组键 = 顶层下的第一个目录</td></tr>')
    parts.append(f'<tr><th>硬门判定</th><td class="num">{s["n_decisions"]}</td>'
                 f'<td class="mut">'
                 + html.escape("、".join(f"{k} ×{v}" for k, v in s["decisions"].items())
                               or "无")
                 + "</td></tr>")
    parts.append(f'<tr><th>带缺列的 run</th><td class="num">{s["n_missing"]}</td>'
                 f'<td class="mut">只缺 <code>{MAIN_DEV_KEY}</code> 一列；'
                 "缺键与末轮非数值都写&quot;缺列&quot;，不写 0。原因按产物自报："
                 + html.escape("；".join(f"{k} ×{v}"
                                        for k, v in s["missing_by_reason"].items())
                               or "无")
                 + "</td></tr>")
    parts.append("</table>")

    parts.append("<h2>生产权重（驾驶链当前会加载的）</h2><table>")
    if prod.get("path"):
        parts.append(f'<tr><th>路径</th><td><code>{html.escape(prod["path"])}</code></td></tr>')
        parts.append(f'<tr><th>sha256[:16]</th><td><code>{prod["sha16"]}</code></td></tr>')
        parts.append(f'<tr><th>写于</th><td>{prod["mtime"]} · {prod["size_mb"]} MB</td></tr>')
        exp = prod.get("experts") or []
        parts.append('<tr><th>地图专家</th><td>'
                     + (html.escape("、".join(Path(e).parent.name for e in exp))
                        if exp else '<span class="mut">无 by_map 专家</span>')
                     + "</td></tr>")
    else:
        parts.append(f'<tr><th>状态</th><td class="bad">未测：'
                     f'{html.escape(prod.get("error") or "不可用")}</td></tr>')
    parts.append("</table>")

    parts.append("<h2>最终集一次确认</h2>")
    if not fs["ledgers"]:
        parts.append(f'<table><tr><th>状态</th><td class="bad">未测：没有任何 '
                     f'<code>{FINAL_LEDGER}</code>，即至今没有一次最终集确认'
                     f'（搜索结果全部停留在 {LEVEL_DEV}）</td></tr></table>')
    else:
        if fs["n_with_result"]:
            parts.append(f'<p class="ok">账本里有 {fs["n_with_result"]} 条带结果的确认。'
                         f'</p>')
        else:
            parts.append(f'<div class="warn"><b>未测</b>：账本里共 {fs["n_entries"]} '
                         f'条记录，'
                         f'<b>没有一条带确认结果</b>（都是访问控制记录）——'
                         f'即至今没有一次完成的最终集确认。不要把下面这些 '
                         f'<code>allowed</code> 记录读成"确认通过"。</div>')
        parts.append("<div class='wrap'><table><thead><tr>"
                     "<th>账本</th><th>时间</th><th>候选</th><th>调用者</th>"
                     "<th>用途</th><th>允许</th><th>结果 / 理由</th>"
                     "</tr></thead><tbody>")
        for led in fs["ledgers"]:
            for e in led["entries"]:
                raw = e["result"] or "；".join(x[:90] for x in e["reasons"])
                why = html.escape(raw) if raw else '<span class="mut">—</span>'
                allow = ('<span class="ok">是</span>' if e["allowed"]
                         else '<span class="bad">否</span>')
                parts.append(
                    f'<tr><td><code>{html.escape(led["rel"])}</code></td>'
                    f'<td>{html.escape(e["t"])}</td>'
                    f'<td>{html.escape(e["candidate"])}</td>'
                    f'<td>{html.escape(e["caller"])}</td>'
                    f'<td>{html.escape(e["purpose"])}</td>'
                    f'<td>{allow}</td><td>{why}</td></tr>')
        parts.append("</tbody></table></div>")
        if fs.get("error"):
            parts.append(f'<p class="mut">读账本出错：{html.escape(fs["error"])}</p>')

    parts.append(f"<h2>家族汇总（{len(fams)} 个，按最近活动倒序）</h2><div class='wrap'><table>")
    parts.append("<thead><tr><th>家族</th><th>顶层</th><th class='num'>run</th>"
                 "<th class='num'>epoch</th><th>首次 → 最近</th>"
                 f"<th>{MAIN_DEV_KEY} 区间〔{LEVEL_DEV}〕</th>"
                 "<th class='num'>有 best.pt</th></tr></thead><tbody>")
    for g in fams:
        rng = (f"{_fmt(g['dev_range'][0])} ~ {_fmt(g['dev_range'][1])}"
               f"（{g['n_with_dev']}/{g['n_runs']} 有值）"
               if g["dev_range"] else f'<span class="tag miss">{MISSING}</span>')
        parts.append(
            f"<tr><td>{html.escape(g['family'])}</td><td>{g['root']}</td>"
            f"<td class='num'>{g['n_runs']}</td><td class='num'>{g['epochs']}</td>"
            f"<td>{_stamp(g['first'])} → {_stamp(g['last'])}</td>"
            f"<td>{rng}</td><td class='num'>{g['n_best_ckpt']}</td></tr>")
    parts.append("</tbody></table></div>")

    parts.append(f"<h2>训练 run 台账（{len(runs)} 行，按时间倒序）</h2>")
    parts.append('<input id="q" oninput="filt()" '
                 'placeholder="过滤：家族 / run / 顶层…">'
                 f'<span class="mut">显示 <b id="n">{len(runs)}</b> / {len(runs)} 行'
                 f'（全部指标为 {LEVEL_DEV}）</span>')
    parts.append("<div class='wrap'><table id='runs'><thead><tr>"
                 "<th>时间</th><th>顶层</th><th>家族</th><th>run</th>"
                 "<th class='num'>epoch</th><th class='num'>train_loss</th>"
                 "<th class='num'>val_miou</th><th class='num'>val_line_iou</th>"
                 "<th class='num'>最佳 line_iou@ep</th><th>状态</th>"
                 "<th>下钻 run-id</th></tr></thead><tbody>")
    for r in runs:
        best = (f"{_fmt(r.best_dev)}@{r.best_dev_epoch}"
                if r.best_dev is not None else f'<span class="tag miss">{MISSING}</span>')
        drill = (f"<code>{html.escape(r.drill_run_id)}</code>"
                 if r.drill_run_id else '<span class="mut">—</span>')
        sub = html.escape(r.sub) or '<span class="mut">（家族本身）</span>'
        line_cell = (_fmt(r.last["val_line_iou"]) if r.last["val_line_iou"] is not None
                     else f'<span class="tag miss" title="'
                          f'{html.escape(r.missing_reason, quote=True)}">'
                          f'{MISSING}</span>')
        key = html.escape(f"{r.root} {r.family} {r.sub} {r.rel}".lower(),
                          quote=True)
        parts.append(
            f"<tr data-k=\"{key}\"><td>{_stamp(r.mtime)}</td>"
            f"<td class='mut'>{r.root}</td><td>{html.escape(r.family)}</td>"
            f"<td>{sub}</td>"
            f"<td class='num'>{r.epochs}</td>"
            f"<td class='num'>{_fmt(r.last['train_loss'])}</td>"
            f"<td class='num'>{_fmt(r.last['val_miou'])}</td>"
            f"<td class='num'>{line_cell}</td>"
            f"<td class='num'>{best}</td>"
            f"<td>{html.escape(r.status)}</td><td>{drill}</td></tr>")
    parts.append("</tbody></table></div>")

    parts.append(f"<h2>硬门判定台账（{len(dec)} 条，按时间倒序）</h2>")
    if not dec:
        parts.append(f'<p class="mut">{MISSING}：没有 {DECISION_NAME}*.json</p>')
    else:
        parts.append("<div class='wrap'><table><thead><tr>"
                     "<th>时间</th><th>run</th><th>候选</th><th>判定</th>"
                     "<th class='num'>硬门违反</th><th>配对比较〔" + LEVEL_DEV + "〕</th>"
                     "</tr></thead><tbody>")
        for d in dec:
            tag = "rejected" if d.decision == "rejected" else d.decision
            extra = ' <span class="tag miss">research_only</span>' if d.research_only else ""
            parts.append(
                f"<tr><td>{_stamp(d.mtime)}</td><td>{html.escape(d.run)}</td>"
                f"<td>{html.escape(d.candidate)}</td>"
                f"<td class='bad'>{html.escape(tag)}</td>"
                f"<td class='num'>{len(d.gates)}</td>"
                f"<td>{_decision_cell(d)}</td></tr>")
        parts.append("</tbody></table></div>")

    parts.append("<h2>缺口（本页不覆盖的）</h2><table>")
    parts.append("<tr><th>Tech 闭环</th><td class='bad'>未测：台账只扫离线训练产物，"
                 "没有任何驾驶/闭环数据</td></tr>")
    parts.append("<tr><th>推理性能</th><td class='mut'>不在此页（见 eval matrix 与监控页）"
                 "</td></tr>")
    parts.append("<tr><th>跨 family 比较</th><td class='mut'>本页刻意不做（见页头警告）"
                 "</td></tr>")
    parts.append("</table>")

    parts.append("</body></html>")
    return "".join(parts)


def build_context(logs_dir: Path) -> dict:
    runs = scan_runs(logs_dir)
    dec = scan_decisions(logs_dir)
    counts: dict[str, int] = {}
    for d in dec:
        counts[d.decision] = counts.get(d.decision, 0) + 1
    by_root: dict[str, int] = {}
    for r in runs:
        by_root[r.root] = by_root.get(r.root, 0) + 1
    by_reason: dict[str, int] = {}
    for r in runs:
        if r.missing_kind:
            by_reason[r.missing_kind] = by_reason.get(r.missing_kind, 0) + 1
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "scan": {
            "logs_dir": str(logs_dir),
            "n_runs": len(runs), "n_families": len({(r.root, r.family) for r in runs}),
            "n_decisions": len(dec), "n_missing": sum(1 for r in runs if r.missing_keys),
            "epochs": sum(r.epochs for r in runs),
            "n_by_root": by_root, "decisions": counts,
            "missing_by_reason": by_reason,
        },
        "production": production_model(),
        "final_set": final_set_state(logs_dir),
        "families": families(runs),
        # 渲染直接吃 dataclass（属性访问）；导 JSON 时才转 dict。
        "runs": runs,
        "decisions": dec,
    }


def _jsonable(ctx: dict) -> dict:
    out = dict(ctx)
    out["runs"] = [dict(asdict(r), status=r.status, label=r.label)
                   for r in ctx["runs"]]
    out["decisions"] = [asdict(d) for d in ctx["decisions"]]
    return out


def _render_once(args) -> int:
    logs_dir = Path(args.logs_dir) if args.logs_dir else Path(config.LOGS_DIR)
    ctx = build_context(logs_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(ctx), encoding="utf-8")
    s = ctx["scan"]
    print(f"[history] {time.strftime('%Y-%m-%d %H:%M:%S')} wrote {out} "
          f"({out.stat().st_size} bytes)")
    print(f"[history] run {s['n_runs']} 个（{s['n_families']} 家族，"
          f"{s['epochs']} epoch）· 判定 {s['n_decisions']} 条"
          f"（{s['decisions'] or '无'}）· 带缺列 {s['n_missing']} 个")
    if args.json:
        jp = Path(args.json)
        jp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text(json.dumps(_jsonable(ctx), ensure_ascii=False, indent=1),
                      encoding="utf-8")
        print(f"[history] json -> {jp}")
    return 0


def _cmd_list(args) -> int:
    logs_dir = Path(args.logs_dir) if args.logs_dir else Path(config.LOGS_DIR)
    runs = scan_runs(logs_dir)
    print(f"[history] {logs_dir}：{len(runs)} 个训练 run"
          f"（全部指标为 {LEVEL_DEV}）")
    print(f"{'时间':11s} {'顶层':12s} {'家族':30s} {'epoch':>5s} "
          f"{'loss':>8s} {'miou':>8s} {'lineIoU':>8s}  状态")
    for r in runs[: args.limit]:
        print(f"{_stamp(r.mtime):11s} {r.root:12s} {r.label[:30]:30s} {r.epochs:5d} "
              f"{_fmt(r.last['train_loss']):>8s} {_fmt(r.last['val_miou']):>8s} "
              f"{_fmt(r.last['val_line_iou']):>8s}  {r.status}")
    if len(runs) > args.limit:
        print(f"… 其余 {len(runs) - args.limit} 行用 render 看（--limit 可调）")
    dec = scan_decisions(logs_dir)
    counts: dict[str, int] = {}
    for d in dec:
        counts[d.decision] = counts.get(d.decision, 0) + 1
    print(f"[history] 硬门判定 {len(dec)} 条：{counts or '无'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="T14 训练台账（只读：扫训练产物出历史表，不训练、不写 run 目录）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render", help="渲染一张自包含 HTML 台账")
    r.add_argument("--out", required=True, help="生成的 HTML 路径")
    r.add_argument("--json", default=None, help="额外导出 JSON（同一个 context）")
    r.add_argument("--logs-dir", default=None,
                   help=f"产物根目录（默认 {config.LOGS_DIR}）")
    r.set_defaults(func=_render_once)
    l = sub.add_parser("list", help="在终端列出 run（不写文件）")
    l.add_argument("--limit", type=int, default=30)
    l.add_argument("--logs-dir", default=None)
    l.set_defaults(func=_cmd_list)
    args = ap.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
