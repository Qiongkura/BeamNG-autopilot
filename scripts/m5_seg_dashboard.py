"""T14 只读学习看板：事件流 + 数据清单 + 评估矩阵 + 固定探针 → 一张自包含 HTML。

方案（`docs/T14_BACKGROUND_MODEL_ITERATION_PLAN_20260924.md`）要求"学习过程
可视化"是**必交付**项，本脚本是它的实现。它专门避免三个已经在别处出过错、
后果又很贵的陷阱：

1. **开发曲线被当成安全成绩**：每个数字都挂一个证据等级（``训练中`` /
   ``开发集`` / ``最终集一次确认`` / ``Tech 闭环未测``），标签由渲染器统一
   产生，视图代码想"顺手写个数字"也带不上等级；训练 loss 下降、开发集
   line IoU 上升都只是优化过程，页面里用固定文案说明最终结论只能来自冻结
   确认与 Tech 闭环。
2. **缺测被画成 0**：``events.metric()`` 的缺测记录带 ``missing`` 原因，
   而它允许 value 与 missing 同时存在（"曾经算过但不可信"）。渲染与导出都
   以 ``missing`` 优先：写"未测（UNKNOWN: 原因）"，绝不写那个残留数字。
   泄漏审计的 ``checked: false`` 渲染成"未检查"，于是 0 不会被读成"查过了、
   没问题"；闭环字段没有事件就写"未测 no Tech closed loop in this run"，
   不会显示 0 事故。
3. **探针图被当成证据**：只有 sidecar（帧哈希 / 权重哈希 / 标签来源 / 后处理
   / 生成时间）齐全、且图片相对 HTML 文件真实存在时才输出 ``<img>``；否则只
   写文字（``missing: <path>`` / "无法验证"），因为一张没有版本锁定的图看起来
   和有证据的图一模一样。

只读纪律：本脚本**不训练、不启动或连接游戏、不调用 ``ControlBridge``、不写
任何候选目录**；除 ``--out`` 与 ``--pack`` 指定的文件外不落盘，运行期间不加
锁、不起后台进程，因此 ``watch`` 开着或关掉都不影响训练。

用法::

    .venv\\Scripts\\python.exe scripts\\m5_seg_dashboard.py render ^
        --out logs\\experiments\\dashboard.html ^
        --events logs\\experiments\\<run_id>\\events.jsonl ^
        --eval logs\\m5_seg\\seg_t13_data_20260924\\06_eval_matrix.json ^
        --ident logs\\m5_seg\\seg_t13_data_20260924 --t13-import ^
        --pack logs\\experiments\\dashboard_evidence.json

    .venv\\Scripts\\python.exe scripts\\m5_seg_dashboard.py watch ^
        --events logs\\experiments\\<run_id>\\events.jsonl ^
        --out logs\\experiments\\dashboard.html --every 15 [--once]

除方案给出的选项外还有两个只增不减的输入（不改变上面 CLI 的含义）：

* ``--tasks PATH``：候选任务结果（文件或目录）。每个 JSON 是一次"真实
  T13 相关帧/人物"的任务结果；没有它时该视图渲染 missing data 表，而不是
  编一个候选身份。
* ``--champion NAME``：成对比较里的 champion 臂名（默认取共同 seed 最多的
  两臂中排序靠前的那个，T13 即 armA=基线配方）。
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import sys
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beamng_autopilot import config
from beamng_autopilot.experiments.events import TRANSITIONS, EventLog

#: T13 历史导入的默认落点（方案阶段 C：先复用 train_hist.json 当验收样例）。
T13_DEFAULT_ROOT = config.LOGS_DIR / "m5_seg" / "seg_t13_data_20260924"

# ---------------------------------------------------------------------------
# 证据等级：任何数字都必须挂其中一个标签
# ---------------------------------------------------------------------------
LEVEL_TRAIN = "training"
LEVEL_DEV = "dev"
LEVEL_FINAL = "final"
LEVEL_UNMEASURED = "unmeasured"
LEVELS = (LEVEL_TRAIN, LEVEL_DEV, LEVEL_FINAL, LEVEL_UNMEASURED)
LEVEL_LABELS = {
    LEVEL_TRAIN: "训练中",
    LEVEL_DEV: "开发集",
    LEVEL_FINAL: "最终集一次确认",
    LEVEL_UNMEASURED: "Tech 闭环未测",
}

#: 没有 Tech 闭环事件时闭环字段的固定原因（方案：不能写 0 事故）。
CLOSED_LOOP_REASON = "no Tech closed loop in this run"

#: 闭环字段 → 可能承载它的指标名。只有事件流里真的出现这些指标才算"测到"。
CLOSED_LOOP_FIELDS = (
    ("碰撞", ("collisions", "collision_n", "damage_events")),
    ("压线", ("lane_crossings", "line_crossing_n")),
    ("出铺装", ("off_pavement", "off_pavement_n")),
    ("停车", ("stop_seconds", "stops_by_reason")),
    ("deadline", ("tick_deadline_violations", "deadline_violations")),
    ("新源消费", ("new_source_consumed", "source_refresh_consumed")),
)
CLOSED_LOOP_KEYS = tuple(k for _, keys in CLOSED_LOOP_FIELDS for k in keys)

#: 缺测（UNKNOWN）在列表里的统一前缀，测试与人都按它找问题。
UNKNOWN_PREFIX = "未测（UNKNOWN: "

SPLIT_LEVEL = {"dev": LEVEL_DEV, "frozen": LEVEL_FINAL, "final": LEVEL_FINAL,
               "train": LEVEL_TRAIN}
SPLIT_LABEL = {"dev": "开发集", "frozen": "冻结集", "final": "最终集",
               "train": "训练集", "none": "未划分"}

#: T13/生产模型名 → 可配对的臂（armA_seed42/best.pt 这类）。
ARM_RE = re.compile(r"^(arm[A-Za-z]*?)[_-]?seed(\d+)$")

CSS = """
:root { color-scheme: dark; }
body { margin: 0; padding: 0 0 40px; background: #0b0d12; color: #e6e8ee;
  font: 13px/1.5 "Segoe UI", "Microsoft YaHei", system-ui, sans-serif; }
header { padding: 18px 22px 6px; }
h1 { font-size: 20px; margin: 0 0 6px; }
h2 { font-size: 16px; margin: 0 0 10px; padding: 6px 0 6px 8px;
  border-left: 4px solid #4c8dff; background: #11141c; }
h3 { font-size: 14px; margin: 16px 0 6px; color: #cdd3e0; }
p.sub { color: #8b93a7; margin: 3px 0; font-size: 12px; }
section { margin: 16px 22px; padding: 12px 14px; background: #0f1219;
  border: 1px solid #232838; border-radius: 6px; }
section.sub { background: #0c0f15; border-style: dashed; margin: 12px 0; }
table { border-collapse: collapse; width: 100%; margin: 6px 0 10px; }
th, td { border: 1px solid #232838; padding: 4px 7px; text-align: left;
  vertical-align: top; font-size: 12px; }
th { background: #161a24; color: #aab2c5; font-weight: 600; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.lvl { display: inline-block; margin-left: 6px; padding: 0 5px; border-radius: 3px;
  font-size: 11px; border: 1px solid #3a4156; color: #c8cede; white-space: nowrap; }
.lvl.training { background: #1d2a44; border-color: #35507f; }
.lvl.dev { background: #24301c; border-color: #4a6b2f; }
.lvl.final { background: #3a2418; border-color: #8a5527; }
.lvl.unmeasured { background: #2c1a1e; border-color: #7d3742; }
.miss { color: #ff9b9b; }
.unknown { color: #ffd479; }
.ok { color: #7bd88f; }
.hint { color: #8b93a7; font-size: 12px; }
.mono { font-family: Consolas, "Cascadia Mono", monospace; font-size: 12px; }
figure { display: inline-block; margin: 8px 14px 8px 0; vertical-align: top; }
figcaption { color: #9aa3b8; font-size: 12px; margin-top: 2px; }
svg { background: #0d1117; border: 1px solid #232838; border-radius: 4px; }
svg text.ax { fill: #8b93a7; font-size: 10px; text-anchor: end; }
svg text.axlbl { fill: #aab2c5; font-size: 10px; text-anchor: middle; }
svg text.legend { fill: #c8cede; font-size: 11px; }
svg text.val { fill: #dfe4f0; font-size: 10px; }
svg text.misspt { fill: #ff9b9b; font-size: 9px; text-anchor: middle; }
svg text.xtick { fill: #8b93a7; font-size: 10px; text-anchor: middle; }
ul.reasons { margin: 4px 0 4px 18px; padding: 0; }
img.probe { max-width: 420px; border: 1px solid #232838; border-radius: 4px; }
footer { margin: 18px 22px; color: #7d859a; font-size: 12px; }
"""


# ---------------------------------------------------------------------------
# 渲染原语：证据等级 + 缺测
# ---------------------------------------------------------------------------
def _esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def _badge(level: str) -> str:
    lvl = level if level in LEVELS else LEVEL_UNMEASURED
    return f'<span class="lvl {lvl}">{LEVEL_LABELS[lvl]}</span>'


def _unknown(reason: str, *, level: str = LEVEL_UNMEASURED) -> str:
    """缺测渲染：只写"未测 + 原因"，不写数字（方案硬要求）。"""
    why = reason or "not measured"
    return f'<span class="miss">{UNKNOWN_PREFIX}{_esc(why)}）</span>{_badge(level)}'


def _fmt_value(value: float, unit: str, digits: int = 4) -> str:
    if unit in ("ratio", "frac", "%"):
        return f"{float(value) * 100:.1f}%"
    if unit:
        return f"{float(value):.{digits}g} {unit}"
    return f"{float(value):.{digits}g}"


@dataclass
class Evidence:
    """一个可渲染的数字：值 + 单位 + 分子/分母 + 证据等级；缺测只带原因。

    ``missing`` 非空即视为缺测——即使记录里残留 ``value``（``events.metric``
    允许两者共存，含义是"算过但不可信"）也不渲染那个数，避免缺测被读成漂亮
    的成绩。
    """

    name: str
    level: str
    value: float | None = None
    unit: str = ""
    numerator: int | None = None
    denominator: int | None = None
    missing: str = ""
    digits: int = 4

    @property
    def measured(self) -> bool:
        if self.missing:
            return False
        return self.value is not None and math.isfinite(float(self.value))

    def cell(self) -> str:
        if not self.measured:
            return _unknown(self.missing or "not measured", level=self.level)
        text = _esc(_fmt_value(float(self.value), self.unit, self.digits))
        if self.numerator is not None and self.denominator is not None:
            text += (f' <span class="hint">({int(self.numerator)}/'
                     f'{int(self.denominator)})</span>')
        return f'<span class="mono">{text}</span>{_badge(self.level)}'

    def as_dict(self) -> dict:
        """导出用：只有真测到才给数值，否则只给状态与原因。"""
        if not self.measured:
            return {"name": self.name, "status": "未测",
                    "reason": self.missing or "not measured",
                    "level": LEVEL_LABELS.get(self.level, self.level)}
        out = {"name": self.name, "value": float(self.value), "unit": self.unit,
               "level": LEVEL_LABELS.get(self.level, self.level)}
        if self.numerator is not None and self.denominator is not None:
            out["numerator"] = int(self.numerator)
            out["denominator"] = int(self.denominator)
        return out


def _metric_evidence(rec: object, name: str, level: str) -> Evidence:
    """事件/样例里的指标记录 → Evidence。缺测记录连 value 一起忽略。"""
    if isinstance(rec, dict):
        return Evidence(
            name=name, level=level,
            value=(None if rec.get("value") is None else rec.get("value")),
            unit=str(rec.get("unit") or ""),
            numerator=rec.get("numerator"), denominator=rec.get("denominator"),
            missing=str(rec.get("missing") or ""))
    if isinstance(rec, (int, float)):
        return Evidence(name=name, level=level, value=float(rec))
    return Evidence(name=name, level=level, missing="not a metric record")


# ---------------------------------------------------------------------------
# 学习曲线：每条序列一个 <svg>，各自的纵轴、缺测留空
# ---------------------------------------------------------------------------
@dataclass
class Series:
    name: str                       # 含 seed，图例里必须能区分
    metric: str
    unit: str
    level: str
    source: str
    points: list = field(default_factory=list)   # [(epoch, value|None)]
    missing: list = field(default_factory=list)  # [(epoch, reason)]

    @property
    def n_points(self) -> int:
        return sum(1 for _, v in self.points if v is not None)


_PALETTE = ("#4c8dff", "#7bd88f", "#ffb454", "#ff7b9c", "#c792ea", "#4dd0e1",
            "#f2f56b", "#9aa3b8")


def _series_colour(name: str) -> str:
    return _PALETTE[sum(ord(c) for c in name) % len(_PALETTE)]


def _fmt_point(value: float) -> str:
    if abs(value) >= 100.0:
        return f"{value:.0f}"
    return f"{value:.4g}"


def _svg_series(series: Series, *, width: int = 470, height: int = 205) -> str:
    """一条序列一张图：只有自己的纵轴刻度与单位，缺测处断线并写"未测"。"""
    pad_l, pad_r, pad_t, pad_b = 62, 76, 26, 32
    xs = [float(e) for e, _ in series.points] or [0.0]
    x0, x1 = min(xs), max(xs)
    if x0 == x1:
        x0, x1 = x0 - 0.5, x1 + 0.5
    vals = [float(v) for _, v in series.points if v is not None]
    if vals:
        lo, hi = min(vals), max(vals)
        if lo == hi:
            span = max(abs(lo) * 0.1, 1e-6)
            lo, hi = lo - span, hi + span
        span = hi - lo
        lo, hi = lo - span * 0.14, hi + span * 0.14
    else:
        lo, hi = 0.0, 1.0
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    def sx(x: float) -> float:
        return pad_l + (x - x0) / (x1 - x0) * plot_w

    def sy(y: float) -> float:
        return height - pad_b - (y - lo) / (hi - lo) * plot_h

    colour = _series_colour(series.name + series.metric)
    out = [f'<svg viewBox="0 0 {width} {height}" width="{width}" '
           f'height="{height}" role="img">',
           f"<title>{_esc(series.name)} · {_esc(series.metric)}</title>",
           f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" '
           f'fill="#0d1117" stroke="#30363d"/>']
    for i in range(5):
        v = lo + (hi - lo) * i / 4.0
        y = sy(v)
        out.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + plot_w}" '
                   f'y2="{y:.1f}" stroke="#1c212c"/>')
        out.append(f'<text x="{pad_l - 5}" y="{y + 3.5:.1f}" class="ax">'
                   f'{_esc(_fmt_point(v))}</text>')
    head = series.unit or series.metric
    out.append(f'<text x="13" y="{pad_t + plot_h / 2:.0f}" class="axlbl" '
               f'transform="rotate(-90 13 {pad_t + plot_h / 2:.0f})">'
               f'{_esc(head)}</text>')
    for e in xs:
        out.append(f'<text x="{sx(e):.1f}" y="{height - pad_b + 14}" '
                   f'class="xtick">{int(e)}</text>')
    out.append(f'<text x="{pad_l + plot_w / 2:.0f}" y="{height - 4}" '
               f'class="xtick">epoch</text>')
    # 缺测留空隙：折线按连续段分开画
    segs: list[list[tuple[float, float]]] = []
    seg: list[tuple[float, float]] = []
    for e, v in series.points:
        if v is None:
            if seg:
                segs.append(seg)
                seg = []
            continue
        seg.append((sx(float(e)), sy(float(v))))
    if seg:
        segs.append(seg)
    for s in segs:
        if len(s) == 1:
            out.append(f'<circle cx="{s[0][0]:.1f}" cy="{s[0][1]:.1f}" r="2.6" '
                       f'fill="{colour}"/>')
        else:
            pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in s)
            out.append(f'<polyline points="{pts}" fill="none" '
                       f'stroke="{colour}" stroke-width="1.8"/>')
    for e, v in series.points:
        x = sx(float(e))
        if v is None:
            out.append(f'<text x="{x:.1f}" y="{height - pad_b - 4}" '
                       f'class="misspt">未测</text>')
            continue
        y = sy(float(v))
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.6" fill="{colour}"/>')
        out.append(f'<text x="{x + 5:.1f}" y="{y - 5:.1f}" class="val">'
                   f'{_esc(_fmt_point(float(v)))}</text>')
    out.append(f'<text x="{pad_l}" y="{pad_t - 8}" class="legend">'
               f'series: {_esc(series.name)} · 指标: {_esc(series.metric)} · '
               f'来源: {_esc(series.source)} · 等级: '
               f'{_esc(LEVEL_LABELS.get(series.level, series.level))}</text>')
    out.append("</svg>")
    return "".join(out)


def _curve_block(series_list: list[Series]) -> str:
    if not series_list:
        return ('<p class="hint">missing data: 没有可画的序列（事件流与 T13 '
                'train_hist.json 都为空）。</p>')
    out = []
    by_metric: dict[str, list[Series]] = {}
    for s in series_list:
        by_metric.setdefault(s.metric, []).append(s)
    for metric in sorted(by_metric):
        group = by_metric[metric]
        out.append(f"<h3>{_esc(metric)} — {_badge(group[0].level)}</h3>")
        out.append("<div>") 
        for s in group:
            fig = (f"<figure>{_svg_series(s)}"
                   f"<figcaption>{_esc(s.name)} · {_esc(metric)} "
                   f"（{len(s.points)} epoch / {s.n_points} 点）</figcaption>"
                   f"</figure>")
            out.append(fig)
        out.append("</div>")
        miss = [(s, e, why) for s in group for e, why in s.missing]
        if miss:
            items = "".join(
                f"<li>epoch {int(e)}: {_esc(metric)} missing — {_esc(why)}"
                f"（{_esc(s.name)}）</li>" for s, e, why in miss)
            out.append(f'<ul class="reasons">{items}</ul>')
    return "".join(out)


# ---------------------------------------------------------------------------
# 读取层：每个输入都返回 {path, readable, error, ...}，读不了只记录原因
# ---------------------------------------------------------------------------
def _load_json(path: Path) -> tuple[object | None, str]:
    p = Path(path)
    try:
        with p.open("r", encoding="utf-8") as fh:
            return json.load(fh), ""
    except FileNotFoundError:
        return None, "file not found"
    except IsADirectoryError:
        return None, "path is a directory, not a JSON file"
    except json.JSONDecodeError as exc:
        return None, f"bad JSON at line {exc.lineno}: {exc.msg}"
    except OSError as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _not_readable(reason: str) -> str:
    return f'<span class="miss">not readable: {_esc(reason)}</span>'


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _ts_epoch(ts: str) -> float | None:
    text = str(ts or "").strip()
    if not text:
        return None
    try:
        from datetime import datetime
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        pass
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return time.mktime(time.strptime(text, fmt))
        except ValueError:
            continue
    return None


def _replay_any(path: Path) -> dict:
    """文件名不是 events.jsonl 时的兜底重放（协议名固定，但 CLI 不该崩）。

    去重键与 ``EventLog.replay`` 相同（``Event.key``），只是少了一层序号排序
    之外的语义；事件数很少时这点重复不值得为它去改库。
    """
    from beamng_autopilot.experiments.events import Event

    events, problems = [], []
    try:
        raw = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {"events": [], "problems": [f"unreadable: {exc}"],
                "n_events": 0, "dropped_duplicates": 0, "by_phase": {},
                "last": None}
    for n, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            blob = json.loads(line)
            events.append(Event(**{k: v for k, v in blob.items()
                                   if k in Event.__dataclass_fields__}))
        except Exception as exc:                      # noqa: BLE001
            problems.append(f"line {n}: unreadable ({exc})")
    ordered = sorted(events, key=lambda e: e.seq)
    seen, kept, dropped = set(), [], 0
    for e in ordered:
        if e.key in seen:
            dropped += 1
            continue
        seen.add(e.key)
        kept.append(e)
    by_phase: dict[str, int] = {}
    for e in kept:
        by_phase[e.phase] = by_phase.get(e.phase, 0) + 1
    return {"events": kept, "problems": problems, "n_events": len(kept),
            "dropped_duplicates": dropped, "by_phase": by_phase,
            "last": kept[-1] if kept else None}


def _events_state(path: Path | None) -> dict:
    state = {"path": str(path) if path else "", "readable": False,
             "error": "", "events": [], "problems": [], "dropped": 0,
             "n_events": 0, "by_phase": {}, "last": None, "phases": []}
    if path is None:
        state["error"] = "no --events path given"
        return state
    p = Path(path)
    if not p.exists():
        state["error"] = f"{p} does not exist"
        return state
    try:
        rep = EventLog(p.parent).replay() if p.name == "events.jsonl" \
            else _replay_any(p)
    except Exception as exc:                          # noqa: BLE001
        state["error"] = f"{type(exc).__name__}: {exc}"
        return state
    state.update(readable=True, events=rep["events"], problems=rep["problems"],
                 dropped=int(rep["dropped_duplicates"]),
                 n_events=int(rep["n_events"]), by_phase=rep["by_phase"],
                 last=rep["last"],
                 phases=sorted({e.phase for e in rep["events"]}))
    return state


def merge_event_streams(run_dir: Path) -> dict:
    """把 `rounds` 运行的多条事件流合并成看板要的形状。

    `rounds` 按设计写"每候选一条流 + run 级一条流"（共用一条流时第一个候选的
    rejected 是终态，第二个候选就进不了 evaluating）。看板此前只读单条流，
    于是要么只看到 queued/auditing/training、要么看不到淘汰理由。这里按
    `Event.key`（phase/status/epoch/step/candidate_id/seed）去重、按 (ts, seq)
    排序——**不能直接拼接**：每条流的 seq 都从 0 开始。
    """
    run_dir = Path(run_dir)
    streams: list[Path] = []
    top = run_dir / "events.jsonl"
    if top.exists():
        streams.append(top)
    streams += sorted(run_dir.glob("candidates/*/events.jsonl"))
    state = {"path": str(run_dir), "readable": False, "error": "",
             "events": [], "problems": [], "dropped": 0, "n_events": 0,
             "by_phase": {}, "last": None, "phases": [],
             "streams": [str(x) for x in streams]}
    if not streams:
        state["error"] = f"{run_dir} 下没有 events.jsonl"
        return state
    events, problems, dropped, seen = [], [], 0, set()
    for path in streams:
        try:
            rep_ = EventLog(path.parent).replay()
        except Exception as exc:                          # noqa: BLE001
            problems.append(f"{path.name}: {type(exc).__name__}: {exc}")
            continue
        for ev in rep_["events"]:
            if ev.key in seen:
                dropped += 1
                continue
            seen.add(ev.key)
            events.append(ev)
        problems += [f"{path.parent.name}/{x}" for x in rep_["problems"]]
        dropped += int(rep_["dropped_duplicates"])
    events.sort(key=lambda e: (str(e.ts), int(e.seq)))
    by_phase: dict = {}
    for ev in events:
        by_phase[ev.phase] = by_phase.get(ev.phase, 0) + 1
    state.update(readable=True, events=events, problems=problems,
                 dropped=dropped, n_events=len(events), by_phase=by_phase,
                 last=events[-1] if events else None,
                 phases=sorted({e.phase for e in events}))
    return state


def _event_series(state: dict) -> list[Series]:
    """训练事件 → 曲线序列；level 只按指标名前缀分（train*/val*）。"""
    by_seed_metric: dict[tuple, dict] = {}
    for ev in state.get("events") or []:
        if ev.epoch is None:
            continue
        seed = getattr(ev, "seed", None)
        for name, rec in (ev.metrics or {}).items():
            level = LEVEL_DEV if str(name).startswith("val") else LEVEL_TRAIN
            key = (seed, str(name), level)
            slot = by_seed_metric.setdefault(key, {"points": {}, "missing": {}})
            epoch = int(ev.epoch)
            if isinstance(rec, dict) and rec.get("value") is not None \
                    and not rec.get("missing"):
                slot["points"][epoch] = float(rec["value"])
            else:
                why = ""
                if isinstance(rec, dict):
                    why = str(rec.get("missing") or "")
                why = why or "the event carries no value for this epoch"
                slot["missing"][epoch] = why
    series = []
    for (seed, metric, level), slot in sorted(by_seed_metric.items(),
                                              key=lambda kv: (kv[0][1], kv[0][0])):
        epochs = sorted(set(slot["points"]) | set(slot["missing"]))
        series.append(Series(
            name=f"seed {seed}", metric=metric,
            unit="", level=level, source="events.jsonl",
            points=[(e, slot["points"].get(e)) for e in epochs],
            missing=[(e, slot["missing"][e]) for e in epochs
                     if e in slot["missing"]]))
    return series


def _t13_state(arg: str | None, hints: list[Path], *,
               recursive: bool = False) -> dict:
    """导入 T13 六个 ``train_hist.json``（方案验收案例 1 的历史样例）。"""
    state = {"root": "", "readable": False, "error": "",
             "series": [], "runs": [], "loaded": 0}
    if arg is None:
        return state
    roots: list[Path] = []
    if arg:
        roots.append(Path(arg))
    for hint in hints:
        if hint:
            roots.append(Path(hint))
    roots.append(T13_DEFAULT_ROOT)
    found: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if root.is_file() and root.name == "train_hist.json":
            found = [root]
        else:
            # 优先六个 arm*_seed* 目录；没有才退回任意子目录，避免把
            # smoke/ 这类中间产物当成 T13 的历史 run。
            if recursive:
                # rounds 运行的历史在 baseline/seed*/ 与 round0/seed*/ 里
                found = sorted(root.rglob("train_hist.json"))
            else:
                found = sorted(root.glob("arm*_seed*/train_hist.json")) or \
                    sorted(root.glob("*/train_hist.json"))
        if found:
            state["root"] = str(root)
            break
    if not found:
        state["error"] = ("no train_hist.json found under "
                          + ", ".join(str(r) for r in roots))
        return state
    state["readable"] = True
    for path in found:
        blob, err = _load_json(path)
        if err:
            state["error"] = f"{path}: {err}"
            continue
        if not isinstance(blob, dict):
            state["error"] = f"{path}: train_hist.json is not an object"
            continue
        tag = path.parent.name
        epochs = blob.get("epoch") or []
        state["runs"].append({"run": tag, "path": str(path),
                              "epochs": len(epochs)})
        state["loaded"] += 1
        for metric, values in blob.items():
            if metric in ("epoch", "lr") or not isinstance(values, list):
                continue
            level = LEVEL_DEV if str(metric).startswith("val") else LEVEL_TRAIN
            pts, miss = [], []
            for i, epoch in enumerate(epochs):
                v = values[i] if i < len(values) else None
                if v is None:
                    pts.append((float(epoch), None))
                    miss.append((float(epoch),
                                 "train_hist.json has no value for this epoch"))
                else:
                    pts.append((float(epoch), float(v)))
            if not pts:
                continue
            state["series"].append(Series(
                name=str(tag), metric=str(metric), unit="", level=level,
                source=Path(path).parent.name + "/train_hist.json",
                points=pts, missing=miss))
    return state


def _manifest_state(path: Path | None) -> dict:
    state = {"path": str(path) if path else "", "readable": False, "error": "",
             "records": [], "audit": {}, "dataset_id": "", "created": "",
             "final_frozen": None, "notes": []}
    if path is None:
        state["error"] = "no --manifest path given"
        return state
    blob, err = _load_json(path)
    if err:
        state["error"] = err
        return state
    if not isinstance(blob, dict):
        state["error"] = "manifest is not a JSON object"
        return state
    state.update(readable=True, dataset_id=str(blob.get("dataset_id") or ""),
                 created=str(blob.get("created") or ""),
                 final_frozen=blob.get("final_frozen"),
                 notes=list(blob.get("notes") or []))
    raw_records = blob.get("records") or []
    records: list[dict] = []
    # 先用库里的 DatasetManifest 读一遍（拿 audit() 的 checked 语义）；数值
    # 一律从 raw dict 取，额外键（例如逐帧任务结果）不会把读取弄崩。
    try:
        from beamng_autopilot.experiments.manifest import DatasetManifest
        mf = DatasetManifest.load(path)
        state["audit"] = mf.audit()
        raw_records = [asdict(r) for r in mf.records]
    except Exception as exc:                          # noqa: BLE001
        state["notes"].append(f"DatasetManifest.load 失败，按原始 JSON 读: {exc}")
    for rec in raw_records:
        if isinstance(rec, dict):
            records.append(rec)
    state["records"] = records
    if not state["audit"]:
        state["audit"] = _leak_audit(records)
    return state


def _leak_audit(records: list[dict]) -> dict:
    """原始记录里的泄漏审计（与 DatasetManifest.audit 的 checked 语义一致）。"""
    live = [r for r in records if not str(r.get("reject_reason") or "")]
    group_sides: dict[str, set] = {}
    content_sides: dict[str, set] = {}
    exp_sides: dict[str, set] = {}
    for r in live:
        split = str(r.get("split") or "none")
        group_sides.setdefault(str(r.get("group") or ""), set()).add(split)
        content_sides.setdefault(str(r.get("content_sha16") or ""),
                                 set()).add(split)
        if r.get("exposure") is not None:
            exp_sides.setdefault(f"{r.get('group')}#e{r.get('exposure')}",
                                 set()).add(split)
    def row(d: dict) -> dict:
        hits = {k: sorted(v) for k, v in d.items() if len(v) > 1}
        return {"n": len(hits), "detail": list(hits)[:8]}
    n_by_split: dict[str, int] = {}
    for r in records:
        key = str(r.get("split") or "none")
        n_by_split[key] = n_by_split.get(key, 0) + 1
    reasons: dict[str, int] = {}
    for r in records:
        why = str(r.get("reject_reason") or "")
        if why:
            reasons[why[:70]] = reasons.get(why[:70], 0) + 1
    # 0 帧不是"查过没问题"：没有任何记录时审计只能是未检查
    audited = bool(records)
    return {
        "n_records": len(records), "n_rejected": len(records) - len(live),
        "reject_reasons": reasons, "n_by_split": n_by_split,
        "group_overlap": {**row(group_sides), "checked": audited},
        "content_overlap": {**row(content_sides), "checked": audited},
        "exposure_overlap": {
            **row(exp_sides),
            "checked": audited and any(r.get("exposure") is not None
                                       for r in records)},
    }


def _eval_state(path: Path | None) -> dict:
    state = {"path": str(path) if path else "", "readable": False, "error": "",
             "splits": {}}
    if path is None:
        state["error"] = "no --eval path given"
        return state
    blob, err = _load_json(path)
    if err:
        state["error"] = err
        return state
    if not isinstance(blob, dict):
        state["error"] = "eval matrix is not a JSON object"
        return state
    for split, models in blob.items():
        if not isinstance(models, dict):
            continue
        rows = {}
        for name, entry in models.items():
            if isinstance(entry, dict):
                rows[str(name)] = entry
        state["splits"][str(split)] = rows
    if not state["splits"]:
        state["error"] = "eval matrix has no split sections"
        return state
    state["readable"] = True
    return state


def _ident_tag(stem: str) -> tuple[str, int | None, str]:
    """``ident_t13_testA2_armA42`` → ("armA42", 42, "t13_testA2")。"""
    parts = stem.split("_")
    tag = parts[-1] if parts else stem
    m = ARM_RE.match(tag)
    if m:
        return tag, int(m.group(2)), tag
    return tag, None, tag


def _ident_state(path: Path | None) -> dict:
    state = {"path": str(path) if path else "", "readable": False, "error": "",
             "files": [], "models": {}}
    if path is None:
        state["error"] = "no --ident path given"
        return state
    p = Path(path)
    if not p.exists():
        state["error"] = f"{p} does not exist"
        return state
    files = sorted(p.glob("ident_*.json")) if p.is_dir() else [p]
    if not files:
        state["error"] = f"no ident_*.json under {p}"
        return state
    state["readable"] = True
    for f in files:
        blob, err = _load_json(f)
        if err:
            state["error"] = f"{f.name}: {err}"
            continue
        if not isinstance(blob, dict):
            state["error"] = f"{f.name}: not an object"
            continue
        summary = blob.get("summary") or {}
        if not isinstance(summary, dict):
            summary = {}
        tag, seed, _ = _ident_tag(f.stem)
        entry = state["models"].setdefault(tag, {"tag": tag, "seed": seed,
                                                 "files": [], "summary": {}})
        entry["files"].append(f.name)
        entry["summary"] = summary
        entry["view"] = blob.get("view")
    return state


def _task_state(path: Path | None) -> dict:
    state = {"path": str(path) if path else "", "readable": False, "error": "",
             "tasks": []}
    if path is None:
        state["error"] = "no --tasks path given"
        return state
    p = Path(path)
    if not p.exists():
        state["error"] = f"{p} does not exist"
        return state
    files = sorted(p.glob("*.json")) if p.is_dir() else [p]
    if not files:
        state["error"] = f"no *.json under {p}"
        return state
    state["readable"] = True
    for f in files:
        blob, err = _load_json(f)
        if err:
            state["tasks"].append({"file": f.name, "error": err})
            continue
        if not isinstance(blob, dict):
            state["tasks"].append({"file": f.name,
                                   "error": "not a JSON object"})
            continue
        task = dict(blob)
        task["file"] = f.name
        state["tasks"].append(task)
    return state


def _probe_state(path: Path | None, out_dir: Path) -> dict:
    state = {"path": str(path) if path else "", "readable": False, "error": "",
             "probes": []}
    if path is None:
        state["error"] = "no --probes path given"
        return state
    p = Path(path)
    if not p.exists():
        state["error"] = f"{p} does not exist"
        return state
    pngs = sorted(p.glob("**/*.png")) if p.is_dir() else [p]
    if not pngs:
        state["error"] = f"no *.png under {p}"
        return state
    state["readable"] = True
    for png in pngs:
        record = {"png": str(png), "sidecar": "", "sidecar_error": "",
                  "fields": {}, "rel": None, "rel_error": "", "exists": True}
        side = png.with_suffix(".json")
        if not side.exists():
            alt = png.with_name(png.stem + ".probe.json")
            side = alt if alt.exists() else None
        if side is None:
            record["sidecar_error"] = "no sidecar json next to the image"
        else:
            record["sidecar"] = str(side)
            blob, err = _load_json(side)
            if err:
                record["sidecar_error"] = err
            elif isinstance(blob, dict):
                record["fields"] = blob
            else:
                record["sidecar_error"] = "sidecar is not a JSON object"
        try:
            rel = os.path.relpath(Path(png).resolve(), Path(out_dir).resolve())
        except ValueError as exc:                     # Windows 跨盘符
            record["rel_error"] = (f"cannot be referenced relatively: {exc}")
        else:
            record["rel"] = rel.replace("\\", "/")
            record["exists"] = (Path(out_dir).resolve() / rel).exists()
        state["probes"].append(record)
    return state


# ---------------------------------------------------------------------------
# 视图 1：运行总览
# ---------------------------------------------------------------------------
def _run_label(ctx: dict) -> str:
    last = (ctx["events"].get("last") if ctx["events"].get("readable")
            else None)
    if last is None:
        return "no run"
    return f'{last.run_id} / {last.candidate_id or "candidate: missing data"}'


def _overview_view(ctx: dict) -> str:
    ev = ctx["events"]
    rows = []
    if not ev.get("readable"):
        rows.append(('<tr><th>事件流</th><td>'
                     + _not_readable(ev.get("error") or "unknown") + "</td></tr>"))
        return ('<section id="overview"><h2>运行总览</h2>'
                '<table>' + "".join(rows) + "</table>"
                '<p class="hint">没有事件流时其余视图只能报"未测"，不会用 0 代替。'
                "</p></section>")
    last = ev["last"]
    events = ev["events"]
    first_ts = _ts_epoch(events[0].ts) if events else None
    last_ts = _ts_epoch(events[-1].ts) if events else None
    elapsed = (None if first_ts is None or last_ts is None
               else max(0.0, last_ts - first_ts))
    total_epochs = None
    for key in ("total_epochs", "epochs_total"):
        rec = (last.metrics or {}).get(key)
        if isinstance(rec, dict) and rec.get("value") is not None:
            total_epochs = float(rec["value"])
            break
        if isinstance(rec, (int, float)):
            total_epochs = float(rec)
            break
    next_step = TRANSITIONS.get(last.phase, ())

    def kv(label: str, value: str, level: str | None = None) -> str:
        badge = _badge(level) if level else ""
        return f"<tr><th>{_esc(label)}</th><td>{value}{badge}</td></tr>"

    rows.append(kv("run_id", f'<span class="mono">{_esc(last.run_id)}</span>'))
    rows.append(kv("candidate_id",
                   (f'<span class="mono">{_esc(last.candidate_id)}</span>'
                    if last.candidate_id else _not_readable(
                        "candidate_id 为空（不编候选身份）"))))
    rows.append(kv("phase / status",
                   f'<span class="mono">{_esc(last.phase)} / '
                   f'{_esc(last.status)}</span>'))
    rows.append(kv("dataset_id",
                   f'<span class="mono">{_esc(last.dataset_id or "")}</span>'
                   if last.dataset_id else
                   _not_readable("dataset_id 为空")))
    rows.append(kv("config_hash",
                   f'<span class="mono">{_esc(last.config_hash or "")}</span>'
                   if last.config_hash else
                   _not_readable("config_hash 为空")))
    rows.append(kv("seed", f'<span class="mono">{_esc(last.seed)}</span>',
                   LEVEL_TRAIN))
    rows.append(kv("epoch / 总轮数",
                   (f'<span class="mono">{_esc(last.epoch)}</span> / '
                    + (f'<span class="mono">{_esc(total_epochs)}</span>'
                       if total_epochs is not None else
                       '<span class="miss">未测（UNKNOWN: 事件流没有 '
                       'total_epochs）</span>'))
                   + _badge(LEVEL_TRAIN)))
    rows.append(kv("step", (f'<span class="mono">{_esc(last.step)}</span>'
                            + _badge(LEVEL_TRAIN))
                   if last.step is not None else
                   _unknown("事件里没有 step", level=LEVEL_TRAIN)))
    rows.append(kv("耗时（首→末事件 ts）",
                   (f'<span class="mono">{elapsed:.1f} s</span>'
                    + _badge(LEVEL_TRAIN)) if elapsed is not None else
                   _unknown("事件 ts 无法解析", level=LEVEL_TRAIN)))
    rows.append(kv("事件数 / 阶段",
                   f'<span class="mono">{ev["n_events"]}</span> '
                   f'{_esc(", ".join(ev["phases"]))}',
                   LEVEL_TRAIN))
    # 重启后的重复点必须写出来：否则曲线会在重启处画两次、出现假跳变
    rows.append(kv("重启重复点",
                   f'<span class="mono">{ev["dropped"]}</span>'
                   '<span class="hint">'
                   '（EventLog.replay()["dropped_duplicates"]：重启后重复写入'
                   "的点按 (phase,status,epoch,step) 去重丢掉的条数；"
                   "曲线因此只画一次，不会在重启处出现假跳变）</span>"))
    note = last.note or ""
    rows.append(kv("下一步 / 停止原因",
                   (f'{_esc(note)}' if note else
                    _unknown("最后一条事件没有 note（无停止/失败原因）"))
                   + (f'<div class="hint">状态机允许的下一步: '
                      f'{_esc(" | ".join(next_step))}</div>'
                      if next_step else
                      '<div class="hint">终态（无下一步）</div>')))
    rows.append(kv("git_commit",
                   f'<span class="mono">{_esc(last.git_commit)}</span>'
                   if last.git_commit else
                   _not_readable("事件没有 git_commit")))
    if ev["problems"]:
        items = "".join(f"<li>{_esc(p)}</li>" for p in ev["problems"])
        rows.append(kv("读取问题", f'<ul class="reasons">{items}</ul>'))
    else:
        rows.append(kv("读取问题",
                       '<span class="ok">无（半写行与坏 JSON 都是 0 条）'
                       "</span>"))
    return ('<section id="overview"><h2>运行总览</h2><table>'
            + "".join(rows) + "</table></section>")


# ---------------------------------------------------------------------------
# 视图 2：学习曲线
# ---------------------------------------------------------------------------
def _curves_view(ctx: dict) -> str:
    series = list(_event_series(ctx["events"])) \
        + list(ctx["t13"].get("series") or [])
    head = ['<section id="curves"><h2>学习曲线</h2>',
            '<p class="hint">每条序列一张图、各自标注纵轴与单位；缺测点断开并在'
            '下方列表里写明原因。训练 loss 与开发集 IoU 只说明优化过程，'
            '**不是安全成绩**：本页的最终结论只能来自冻结确认与 Tech 闭环'
            '两节。</p>']
    if ctx["events"].get("readable") and ctx["events"]["n_events"]:
        head.append(f'<p class="hint">事件流: {_esc(ctx["events"]["path"])}'
                    f'（{ctx["events"]["n_events"]} 条，去重丢弃 '
                    f'{ctx["events"]["dropped"]} 条）</p>')
    if ctx["t13"].get("readable"):
        used = ", ".join(f'{r["run"]}（{r["epochs"]} epoch）'
                         for r in ctx["t13"].get("runs") or [])
        head.append(f'<p class="hint">T13 导入: {_esc(ctx["t13"].get("root"))}'
                    f" → {_esc(used)}</p>")
    elif ctx["t13"].get("error"):
        head.append(f'<p class="hint">{_not_readable(ctx["t13"]["error"])}</p>')
    head.append(_curve_block(series))
    head.append("</section>")
    return "".join(head)


# ---------------------------------------------------------------------------
# 视图 3：固定探针图像
# ---------------------------------------------------------------------------
PROBE_FIELDS = ("frame_sha16", "label_source", "model_sha16", "postprocess",
                "created")


def _probes_view(ctx: dict) -> str:
    state = ctx["probes"]
    out = ['<section id="probes"><h2>固定探针图像</h2>',
           '<p class="hint">只有 sidecar（帧哈希/权重哈希/标签来源/后处理/'
           '生成时间）齐全、且图片相对本 HTML 真实存在时才渲染图片；否则只给'
           '文字，避免把一张没有版本锁定的图当成证据。</p>']
    if not state.get("readable"):
        out.append(f'<p>{_not_readable(state.get("error") or "unknown")}</p>')
        out.append('<p class="hint">missing data: 本轮没有固定探针目录，'
                   '该视图不展示任何图片。</p></section>')
        return "".join(out)
    rows = ['<table><tr><th>探针</th><th>sidecar 校验</th><th>frame_sha16</th>'
            "<th>label_source</th><th>model_sha16</th><th>postprocess</th>"
            "<th>created</th><th>渲染</th></tr>"]
    images = []
    for rec in state["probes"]:
        fields = rec["fields"]
        missing_fields = [f for f in PROBE_FIELDS
                          if not str(fields.get(f) or "").strip()]
        cells = "".join(
            f'<td class="mono">{_esc(fields.get(f, ""))}</td>'
            if str(fields.get(f) or "").strip() else
            '<td><span class="miss">未测</span></td>' for f in PROBE_FIELDS)
        if rec["sidecar_error"]:
            verify = (f'<span class="miss">无法验证（{_esc(rec["sidecar_error"])}'
                      "）：不作为证据展示</span>")
        elif missing_fields:
            verify = ('<span class="miss">无法验证（sidecar 缺字段: '
                      f'{_esc(", ".join(missing_fields))}）</span>')
        else:
            verify = '<span class="ok">sidecar 齐全</span>'
        if rec["rel_error"]:
            render = (f'<span class="miss">missing: {_esc(rec["png"])}'
                      f'（{_esc(rec["rel_error"])}）</span>')
        elif not rec["exists"]:
            render = f'<span class="miss">missing: {_esc(rec["png"])}</span>'
        elif rec["sidecar_error"] or missing_fields:
            render = ('<span class="miss">不渲染（无法验证）</span>')
        else:
            render = f'<span class="mono">{_esc(rec["rel"])}</span>'
            images.append(
                f'<figure><img class="probe" src="{_esc(rec["rel"])}" '
                f'alt="{_esc(Path(rec["png"]).name)}">'
                f'<figcaption>{_esc(Path(rec["png"]).name)} · '
                f'frame {_esc(fields.get("frame_sha16"))}</figcaption></figure>')
        rows.append(f'<tr><td class="mono">{_esc(rec["png"])}</td>'
                    f'<td class="mono">{_esc(rec["sidecar"] or "")}'
                    f"{verify}</td>"
                    f"{cells}<td>{render}</td></tr>")
    rows.append("</table>")
    out.append("".join(rows))
    if images:
        out.append("<h3>已锁定的探针图像</h3>" + "".join(images))
    else:
        out.append('<p class="hint">missing data: 没有可渲染且可验证的探针图像。'
                   "</p>")
    out.append("</section>")
    return "".join(out)


# ---------------------------------------------------------------------------
# 视图 4：候选任务结果
# ---------------------------------------------------------------------------
TASK_ROWS = ("引擎判分（ground truth）", "路面判分（surface）",
             "交通/人物数量", "危险事件数量", "score delta")


def _task_rows(ctx: dict) -> list[dict]:
    """五个字段按固定顺序；缺数据只写 missing data，不编候选身份。"""
    tasks = [t for t in (ctx["tasks"].get("tasks") or []) if not t.get("error")]
    ident = ctx["ident"]
    rows = []
    engine_parts = []
    for t in tasks:
        grader = t.get("engine_grader")
        frame = t.get("frame") or t.get("frame_path") or ""
        if isinstance(grader, dict):
            engine_parts.append(
                f'{_esc(t.get("file"))}: grader: engine'
                + (f' · frame {_esc(frame)}' if frame else
                   ' · <span class="miss">未给出真实帧引用</span>')
                + (f' · gt_line_px {_esc(grader.get("gt_line_px"))}'
                   if grader.get("gt_line_px") is not None else "")
                + (f' · engine_lines_total '
                   f'{_esc(grader.get("engine_lines_total"))}'
                   if grader.get("engine_lines_total") is not None else ""))
    if not engine_parts and ident.get("readable"):
        # 真实 T13 相关帧：ident 探针的引擎真值（grader: engine）
        for tag, entry in sorted(ident["models"].items()):
            s = entry.get("summary") or {}
            if s.get("engine_lines_total") is None:
                continue
            engine_parts.append(
                f'{_esc(entry.get("files", [""])[0])}: grader: engine（真值）'
                f' · engine_lines_total {_esc(s.get("engine_lines_total"))}'
                f' · frames_with_engine_line '
                f'{_esc(s.get("frames_with_engine_line"))}'
                f' · candidates_total {_esc(s.get("candidates_total"))}')
    rows.append({
        "label": TASK_ROWS[0],
        "level": LEVEL_FINAL if engine_parts else LEVEL_UNMEASURED,
        "html": ("<br>".join(engine_parts) if engine_parts else
                 '<span class="miss">missing data: 没有引擎真值判分记录'
                 "（ident 探针与 --tasks 都没有）</span>")})
    surf = []
    for t in tasks:
        sg = t.get("surface_grader")
        if isinstance(sg, dict) and str(sg.get("surface") or ""):
            surf.append(f'{_esc(t.get("file"))}: surface: '
                        f'{_esc(sg.get("surface"))}'
                        + (f'（paved_frac {_esc(sg.get("paved_frac"))}）'
                           if sg.get("paved_frac") is not None else ""))
    rows.append({
        "label": TASK_ROWS[1],
        "level": LEVEL_FINAL if surf else LEVEL_UNMEASURED,
        "html": ("<br>".join(surf) if surf else
                 '<span class="miss">missing data: 本轮没有路面门 '
                 "(surface: asphalt|dirt) 记录；无 Tech 闭环时该项为未测"
                 "</span>")})
    people = []
    for t in tasks:
        tp = t.get("traffic_persona")
        if isinstance(tp, dict):
            people.append(f'{_esc(t.get("file"))}: personas '
                          f'{_esc(tp.get("personas"))} / vehicles '
                          f'{_esc(tp.get("vehicles"))}')
    rows.append({
        "label": TASK_ROWS[2],
        "level": LEVEL_FINAL if people else LEVEL_UNMEASURED,
        "html": ("<br>".join(people) if people else
                 '<span class="miss">missing data: 没有交通/人物计数记录'
                 "</span>")})
    hazards = []
    for t in tasks:
        hz = t.get("hazards")
        if isinstance(hz, dict) and hz.get("hazards") is not None:
            hazards.append(f'{_esc(t.get("file"))}: hazards '
                           f'{_esc(hz.get("hazards"))}')
    rows.append({
        "label": TASK_ROWS[3],
        "level": LEVEL_FINAL if hazards else LEVEL_UNMEASURED,
        "html": ("<br>".join(hazards) if hazards else
                 '<span class="miss">missing data: 没有危险事件计数记录'
                 "</span>")})
    deltas = []
    for t in tasks:
        sd = t.get("score_delta")
        if isinstance(sd, dict):
            ev = _metric_evidence(sd, "score_delta", LEVEL_FINAL)
            deltas.append(f'{_esc(t.get("file"))}: score delta '
                          + ev.cell())
        elif isinstance(sd, (int, float)):
            deltas.append(f'{_esc(t.get("file"))}: score delta '
                          + _metric_evidence(sd, "score_delta",
                                             LEVEL_FINAL).cell())
    rows.append({
        "label": TASK_ROWS[4],
        "level": LEVEL_FINAL if deltas else LEVEL_UNMEASURED,
        "html": ("<br>".join(deltas) if deltas else
                 '<span class="miss">missing data: 没有 score delta 记录'
                 "</span>")})
    return rows


def _cases_html(cases: list[dict]) -> str:
    if not cases:
        return ('<p class="hint">missing data: 没有确定性样例'
                "（--tasks 未给出 case 字段）。</p>")
    rows = ['<table><tr><th>样例</th><th>证据等级</th><th>真值有效区</th>'
            "<th>FP / FN</th><th>指标</th><th>判定 / 淘汰理由</th></tr>"]
    for case in cases:
        level = SPLIT_LEVEL.get(str(case.get("level") or "dev"), LEVEL_DEV)
        quality = case.get("quality") or {}
        cells = []
        for name in ("road", "paint", "pavement"):
            q = quality.get(name) or {}
            if not isinstance(q, dict) or not q:
                cells.append(f'<div>{_esc(name)}: '
                             + _unknown("样例没有该通道的审计")
                             + "</div>")
                continue
            valid = bool(q.get("valid"))
            px = q.get("pixels")
            if valid:
                cells.append(f'<div class="ok">{_esc(name)}: 有效'
                             + (f"（{_esc(px)} px）</div>" if px is not None
                                else "</div>"))
            else:
                cells.append(f'<div class="miss">{_esc(name)}: 不可用'
                             + (f"（{_esc(px)} px）" if px is not None else "")
                             + f' — {_esc(q.get("reason") or "no reason")}'
                             "</div>")
        fp = case.get("metrics", {}).get("offroad_false_line_px") \
            if isinstance(case.get("metrics"), dict) else None
        fn = case.get("metrics", {}).get("missed_true_line_px") \
            if isinstance(case.get("metrics"), dict) else None
        fp_cell = _metric_evidence(fp, "offroad_false_line_px", level).cell() \
            if fp is not None else _unknown("样例没有 FP 像素记录", level=level)
        fn_cell = _metric_evidence(fn, "missed_true_line_px", level).cell() \
            if fn is not None else _unknown("样例没有 FN 像素记录", level=level)
        metrics = case.get("metrics") or {}
        metric_cells = []
        if isinstance(metrics, dict):
            for name in sorted(metrics):
                metric_cells.append(
                    f'<div>{_esc(name)}: '
                    + _metric_evidence(metrics[name], name, level).cell()
                    + "</div>")
        if not metric_cells:
            metric_cells.append(_unknown("样例没有指标记录", level=level))
        reasons = case.get("reasons") or []
        if not isinstance(reasons, list):
            reasons = [str(reasons)]
        why = "".join(f"<li>{_esc(r)}</li>" for r in reasons) or \
            '<li><span class="miss">未给出淘汰理由</span></li>'
        verdict = case.get("decision") or "missing data"
        rows.append(
            f'<tr><td>{_esc(case.get("label") or case.get("case_id") or "case")}'
            f'<div class="hint mono">{_esc(case.get("frame_sha16") or "")}</div>'
            f'</td><td>{_badge(level)}</td>'
            f'<td>{"".join(cells)}</td>'
            f"<td>FP: {fp_cell}<br>FN: {fn_cell}</td>"
            f'<td>{"".join(metric_cells)}</td>'
            f'<td><span class="mono">{_esc(verdict)}</span>'
            f'<ul class="reasons">{why}</ul></td></tr>')
    rows.append("</table>")
    return "".join(rows)


def _tasks_view(ctx: dict) -> str:
    state = ctx["tasks"]
    out = ['<section id="tasks"><h2>候选任务结果</h2>',
           '<p class="hint">顺序固定：引擎判分 → 路面判分 → 交通/人物 → '
           "危险事件 → score delta。没有数据的分项渲染 missing data，"
           "不编候选身份；文件读不了只写 not readable，不中断渲染。</p>"]
    rows = _task_rows(ctx)
    table = ['<table><tr><th>分项</th><th>结果</th></tr>']
    for r in rows:
        table.append(f'<tr><td>{_esc(r["label"])}{_badge(r["level"])}</td>'
                     f'<td>{r["html"]}</td></tr>')
    table.append("</table>")
    out.append("".join(table))
    if not state.get("readable"):
        out.append(f'<p>{_not_readable(state.get("error") or "unknown")}</p>')
    for t in state.get("tasks") or []:
        if t.get("error"):
            out.append(f'<p>{_esc(t.get("file"))}: '
                       f"{_not_readable(t['error'])}</p>")
    cases = [t["case"] for t in (state.get("tasks") or [])
             if isinstance(t.get("case"), dict)]
    out.append("<h3>确定性样例（真值有效区 / FP-FN / UNKNOWN / 淘汰理由）</h3>")
    out.append(_cases_html(cases))
    out.append("</section>")
    return "".join(out)


# ---------------------------------------------------------------------------
# 视图 5：实验对比（成对 seed + 晋级判定）
# ---------------------------------------------------------------------------
EVAL_METRICS = (
    # (指标名, 单位, 越低越好, 分子键, 分母键)
    ("line_recall", "ratio", False, ("tp_px",), ("tp_px", "fn_px")),
    ("line_precision", "ratio", False, ("tp_px",), ("tp_px", "fp_px")),
    ("line_iou", "ratio", False, ("tp_px",), ("tp_px", "fp_px", "fn_px")),
    ("offroad_false_line_px", "px", True, None, None),
    ("offroad_false_ratio", "ratio", True, ("offroad_false_line_px",),
     ("pred_line_px",)),
    ("inference_ms_p50", "ms", True, None, None),
    ("inference_ms_p95", "ms", True, None, None),
)

#: 阈值里用的主指标名 → eval 矩阵里的键（offroad_false_frac_of_pred）。
GATE_KEY = {"offroad_false_ratio": "offroad_false_frac_of_pred"}


def _model_tag(name: str) -> dict:
    raw = str(name).replace("\\", "/")
    ckpt = ""
    if "/" in raw:
        head, _, tail = raw.rpartition("/")
        raw, ckpt = head, tail
    tag = raw.split("(", 1)[0]
    m = ARM_RE.match(tag)
    return {"raw": str(name), "tag": tag, "ckpt": ckpt,
            "arm": (m.group(1) if m else ""),
            "seed": (int(m.group(2)) if m else None)}


def _arm_table(split_rows: dict) -> dict:
    """{arm: {seed: {ckpt: entry}}}。"""
    out: dict[str, dict] = {}
    for name, entry in (split_rows or {}).items():
        info = _model_tag(name)
        if not info["arm"] or info["seed"] is None:
            continue
        out.setdefault(info["arm"], {}).setdefault(info["seed"], {})[
            info["ckpt"] or "-"] = entry
    return out


def _common_seeds(a: dict, b: dict, ckpt: str) -> list[int]:
    return sorted(s for s in a if s in b and ckpt in a[s] and ckpt in b[s])


def _pick_pair(split_rows: dict, champion: str | None) -> tuple[str, str, str]:
    """champion/candidate 由"共同 seed 最多的两臂"推定；并列时按臂名排序。"""
    arms = _arm_table(split_rows)
    names = sorted(arms)
    if len(names) < 2:
        return (champion or (names[0] if names else ""), "", "best.pt")
    if champion and champion in arms:
        others = [n for n in names if n != champion]
        best = max(others, key=lambda n: (len(_common_seeds(
            arms[champion], arms[n], _best_ckpt(arms, champion, n))), n))
        return champion, best, _best_ckpt(arms, champion, best)
    best_pair, best_n, best_ckpt = (names[0], names[1], "best.pt"), -1, "best.pt"
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ck = _best_ckpt(arms, a, b)
            n = len(_common_seeds(arms[a], arms[b], ck))
            if n > best_n:
                best_pair, best_n, best_ckpt = (a, b, ck), n, ck
    return best_pair[0], best_pair[1], best_ckpt


def _best_ckpt(arms: dict, a: str, b: str) -> str:
    for ck in ("best.pt", "checkpoint_last.pt", "-"):
        seeds = _common_seeds(arms.get(a, {}), arms.get(b, {}), ck)
        if seeds:
            return ck
    return "best.pt"


def _paired_summary(champion_vals: list, candidate_vals: list,
                    lower_is_better: bool) -> dict:
    from beamng_autopilot.experiments.gates import PairedResult
    return PairedResult(metric="", champion=list(champion_vals),
                        candidate=list(candidate_vals),
                        lower_is_better=bool(lower_is_better)).summary()


def _promotion(ctx: dict) -> dict:
    """成对 seed 比较 + 硬门槛 + 晋级判定（HTML 与证据包共用同一份计算）。

    ``measured`` 取候选臂的逐 seed 均值；硬门槛逐项检查，缺测不算通过。
    """
    from beamng_autopilot.experiments.gates import (
        Thresholds, decide, threshold_violations)
    ev = ctx["eval"]
    data = {"readable": bool(ev.get("readable")),
            "error": ev.get("error", ""), "split": "", "level": LEVEL_DEV,
            "champion": "", "candidate": "", "checkpoint": "", "seeds": [],
            "rows": [], "pairings": {}, "measured": {}, "missing": [],
            "violations": [], "hards": [], "decision": {},
            "thresholds": Thresholds()}
    if not data["readable"]:
        return data
    split = "dev" if ev["splits"].get("dev") else sorted(ev["splits"])[0]
    rows = ev["splits"][split]
    champion, candidate, ckpt = _pick_pair(rows, ctx.get("champion"))
    data.update(split=split, level=SPLIT_LEVEL.get(split, LEVEL_DEV),
                champion=champion, candidate=candidate, checkpoint=ckpt)
    if not candidate:
        return data
    arms = _arm_table(rows)
    seeds = _common_seeds(arms[champion], arms[candidate], ckpt)
    data["seeds"] = seeds
    for metric, unit, lower, num_keys, den_keys in EVAL_METRICS:
        key = GATE_KEY.get(metric, metric)
        pairs = []
        for seed in seeds:
            b = arms[champion][seed].get(ckpt, {}).get(key)
            c = arms[candidate][seed].get(ckpt, {}).get(key)
            if b is not None and c is not None:
                pairs.append((seed, float(b), float(c)))
        num = den = None
        if num_keys and den_keys:
            num = int(sum(float(arms[candidate][s][ckpt].get(k, 0) or 0)
                          for s in seeds for k in num_keys))
            den = int(sum(float(arms[candidate][s][ckpt].get(k, 0) or 0)
                          for s in seeds for k in den_keys))
        if not pairs:
            data["missing"].append(metric)
            data["rows"].append({"metric": metric, "unit": unit,
                                 "lower": lower, "numerator": num,
                                 "denominator": den, "pairs": [],
                                 "summary": {"metric": metric, "n": 0,
                                             "missing": "no paired seeds "
                                                        "measured"}})
            continue
        summary = _paired_summary([b for _, b, _ in pairs],
                                  [c for _, _, c in pairs], lower)
        data["pairings"][metric] = summary
        data["measured"][metric] = sum(c for _, _, c in pairs) / len(pairs)
        data["rows"].append({"metric": metric, "unit": unit, "lower": lower,
                             "numerator": num, "denominator": den,
                             "pairs": pairs, "summary": summary})
    thr = data["thresholds"]
    violations = threshold_violations(data["measured"], thr)
    data["violations"] = violations
    data["hards"] = [v for v in violations if "UNKNOWN" not in v]
    unknowns = [v for v in violations if "UNKNOWN" in v]
    missing = list(data["missing"]) + [v.split(":")[0] for v in unknowns]
    kwargs = {"pairings": data["pairings"], "thresholds": thr,
              "missing_metrics": sorted(set(missing)) or None}
    if data["hards"]:
        kwargs["hard_gate_violations"] = data["hards"]
    decision = decide(**kwargs)
    if data["hards"]:
        decision = dict(decision,
                        reasons=list(data["hards"])
                        + list(decision.get("reasons") or []))
    data["decision"] = decision
    return data


def _decisions_state(path: Path | None) -> dict:
    """读运行目录里的 `decision_*.json`（成对比较 + 硬门 + 判定理由）。

    为什么单独读它：`rounds` 的判定不在评估矩阵里，而看板的"实验对比"节
    只吃评估矩阵——于是自治循环的"淘汰理由/成对差值/硬门 UNKNOWN"在页面上
    根本看不到。这里把它作为独立证据源喂进来（缺文件就报未测，不猜）。
    """
    state = {"path": str(path) if path else "", "readable": False, "error": "",
             "items": []}
    if path is None:
        state["error"] = "no --decisions/--run-dir given"
        return state
    root = Path(path)
    files = sorted(root.glob("decision_*.json"))
    if not files:
        state["error"] = f"{root} 下没有 decision_*.json"
        return state
    for fp in files:
        try:
            blob = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as exc:                          # noqa: BLE001
            state["items"].append({"file": fp.name, "error":
                                   f"{type(exc).__name__}: {exc}"})
            continue
        pairs = blob.get("pairings") or {}
        hard = blob.get("hard_gate") or {}
        dec = blob.get("decision") or {}
        state["items"].append({
            "file": fp.name,
            "candidate_id": blob.get("candidate_id") or blob.get("candidate") or "",
            "metric": ", ".join(sorted(pairs)) or "",
            "pairings": pairs,
            "hard_gate": hard,
            "hard_unknown": sorted(k for k, v in hard.items() if v is None),
            "decision": dec.get("decision") or "",
            "reasons": list(dec.get("reasons") or []),
            "steps_by_arm": blob.get("steps_by_arm"),
            "max_train_frames": blob.get("max_train_frames"),
            "equal_steps": blob.get("equal_steps_requested"),
            "factor": blob.get("factor"),
            "worst_frames_by_seed": blob.get("worst_frames_by_seed") or {},
            "trivial_all_road": blob.get("road_iou_trivial_all_road"),
            "eval_checkpoint": blob.get("eval_checkpoint"),
            "epochs": blob.get("epochs"),
        })
    state["readable"] = bool([i for i in state["items"] if not i.get("error")])
    return state


def _decisions_view(ctx: dict) -> str:
    """决策与成对比较：逐 seed 值、差值、ci95、硬门 UNKNOWN、淘汰理由。"""
    st = ctx.get("decisions") or {}
    out = ['<section id="decisions"><h2>判定与成对比较（决策文件）</h2>',
           '<p class="hint">数据来自运行目录的 <code>decision_*.json</code>：'
           "逐 seed 实测值、配对差、置信区间、硬门输入与判定理由。"
           "硬门里为 null 的项＝未测（渲染成「未测」，不写 0）。</p>"]
    if not st.get("readable"):
        out.append(f'<p class="hint">{_not_readable(st.get("error") or "无决策文件")}'
                   "</p></section>")
        return "".join(out)
    for it in st["items"]:
        if it.get("error"):
            out.append(f'<p class="hint">{_esc(it["file"])}: '
                       f'{_esc(it["error"])}</p>')
            continue
        out.append(f'<h3>{_esc(it["candidate_id"] or it["file"])}'
                   f' → {_esc(it["decision"] or "未写")}</h3>')
        if it.get("factor"):
            out.append(f'<p class="hint">因子: <code>{_esc(json.dumps(it["factor"], ensure_ascii=False))}'
                       "</code>"
                       + (f' · 等步数对照: 已请求 · 训练帧上限 '
                          f'{_esc(str(it["max_train_frames"]))}'
                          if it.get("equal_steps") else "")
                       + "</p>")
        rows = ['<table><tr><th>指标</th><th>champion 每 seed</th>'
                "<th>candidate 每 seed</th><th>deltas</th>"
                "<th>mean delta</th><th>ci95 半宽</th><th>verdict</th></tr>"]
        for name, sp in sorted((it.get("pairings") or {}).items()):
            rows.append(
                f"<tr><td>{_esc(name)}</td>"
                f'<td>{_esc(_fmt_list(sp.get("champion")))}</td>'
                f'<td>{_esc(_fmt_list(sp.get("candidate")))}</td>'
                f'<td>{_esc(_fmt_list(sp.get("deltas")))}</td>'
                f'<td>{_esc(_fmt_num(sp.get("mean_delta")))}</td>'
                f'<td>{_esc(_fmt_num(sp.get("ci95_halfwidth")))}</td>'
                f'<td>{_esc(str(sp.get("verdict") or ""))}</td></tr>')
        rows.append("</table>")
        out.append("".join(rows))
        if it.get("reasons"):
            out.append("<p class=\"hint\">判定理由:</p><ul>"
                       + "".join(f"<li>{_esc(r)}</li>" for r in it["reasons"])
                       + "</ul>")
        if it.get("trivial_all_road") is not None:
            out.append('<p class="hint">评测口径: checkpoint='
                       f'{_esc(it.get("eval_checkpoint"))} · epochs='
                       f'{_esc(str(it.get("epochs")))} · <b>平凡基线</b>'
                       '（把全部像素预测成路面）= '
                       f'{_esc(_fmt_num(it.get("trivial_all_road")))}'
                       '——读数必须与它比，否则 0.42 会被读成"学会了路面"。</p>')
        worst = it.get("worst_frames_by_seed") or {}
        if worst:
            out.append('<p class="hint">最差帧（点进具体帧：判定→帧路径→IoU→'
                       '该帧路面真值像素）:</p><ul class="mono">')
            for seed, frames in sorted(worst.items(),
                                       key=lambda kv: int(kv[0])):
                cells = "; ".join(
                    f"{_esc(f.get('frame'))} (iou={_esc(_fmt_num(f.get('iou')))}, "
                    f"gt={_esc(str(f.get('gt_px')))})" for f in frames)
                out.append(f"<li>seed {_esc(seed)}: {cells}</li>")
            out.append("</ul>")
        if it.get("hard_unknown"):
            out.append('<p class="hint">硬门未测（UNKNOWN）: '
                       f'{_esc(", ".join(it["hard_unknown"]))}'
                       "——这些项没有测量，既不算通过也不算违反。</p>")
            run_dir = Path(st.get("path") or "")
        if run_dir.exists():
            shots = (sorted(run_dir.glob("review_overlays/*.png"))[:6]
                     + sorted(run_dir.glob("probes/**/*.png"))[:6])
            if shots:
                out.append('<p class="hint">可打开的复核图（相对运行目录）: '
                           + ", ".join(f'<code>{_esc(str(p.relative_to(run_dir)))}</code>'
                                       for p in shots) + "</p>")
        out.append("</section>")
    return "".join(out)


def _fmt_list(v) -> str:
    if not v:
        return "—"
    return "[" + ", ".join(_fmt_num(x) for x in v) + "]"


def _fmt_num(v) -> str:
    if v is None:
        return "未测"
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return str(v)


def _compare_view(ctx: dict) -> str:
    ev = ctx["eval"]
    out = ['<section id="compare"><h2>实验对比</h2>',
           '<p class="hint">成对 seed 比较：开发集口径用于搜索与晋级判定，'
           "冻结集只在配方冻结后做一次确认（本页与下一节分开显示，"
           "不把两者混算）。</p>"]
    if not ev.get("readable"):
        out.append(f'<p>{_not_readable(ev.get("error") or "unknown")}</p>')
        out.append('<p class="hint">missing data: 没有评估矩阵时，配对、'
                   "置信区间与晋级判定都只能报未测。</p></section>")
        return "".join(out)
    promo = _promotion(ctx)
    level = promo["level"]
    if not promo["candidate"]:
        out.append('<p class="hint">missing data: 评估矩阵里没有两个可配对的臂'
                   "（例如只有生产模型与一个候选），成对差值未测。</p>")
        out.append(_compare_ident(ctx))
        out.append("</section>")
        return "".join(out)
    out.append(f'<p class="hint">决策口径: {_esc(promo["split"])}'
               f'（{_esc(SPLIT_LABEL.get(promo["split"], promo["split"]))}）'
               f'{_badge(level)} · champion = {_esc(promo["champion"])}'
               f' · candidate = {_esc(promo["candidate"])}'
               f' · checkpoint = {_esc(promo["checkpoint"])} · 配对 seed: '
               f'{_esc(promo["seeds"]) if promo["seeds"] else "无"}</p>')
    if not promo["seeds"]:
        out.append('<p class="hint">missing data: 两个臂没有共同 seed，'
                   "成对差值未测（不取单 seed 晋级）。</p>")
    table = ['<table><tr><th>指标</th><th>分子/分母</th><th>champion 每 seed</th>'
             "<th>candidate 每 seed</th><th>mean delta</th><th>ci95 半宽</th>"
             "<th>verdict</th></tr>"]
    for row in promo["rows"]:
        metric = row["metric"]
        if not row["pairs"]:
            table.append(f'<tr><td>{_esc(metric)}{_badge(level)}</td>'
                         f"<td>{_unknown('该口径下没有成对测量', level=level)}"
                         '</td><td colspan="5">missing data</td></tr>')
            continue
        summary = row["summary"]
        frac = (f'{row["numerator"]}/{row["denominator"]}'
                if row["numerator"] is not None
                else '<span class="hint">n/a（计数/时延指标，无分子分母）</span>')
        verdict = summary.get("verdict") or "missing data"
        half = summary.get("ci95_halfwidth")
        per_champion = " · ".join(
            f"seed{s}={_fmt_point(b)}" for s, b, _ in row["pairs"])
        per_candidate = " · ".join(
            f"seed{s}={_fmt_point(c)}" for s, _, c in row["pairs"])
        table.append(
            f'<tr><td>{_esc(metric)}{_badge(level)}</td>'
            f'<td class="mono">{frac}</td>'
            f'<td class="mono">{_esc(per_champion)}</td>'
            f'<td class="mono">{_esc(per_candidate)}</td>'
            f'<td class="mono">{summary.get("mean_delta")}</td>'
            f'<td class="mono">{half if half is not None else "未测（单 seed 无区间）"}</td>'
            f'<td class="mono">{_esc(verdict)} <span class="hint">'
            f'{_esc(_verdict_gloss(verdict))}</span></td></tr>')
    table.append("</table>")
    out.append("".join(table))
    thr = promo["thresholds"]
    decision = promo["decision"]
    out.append(f'<h3>晋级判定: <span class="mono">'
               f'{_esc(decision.get("decision"))}</span>{_badge(level)}</h3>')
    out.append('<p class="hint">冻结协议 config_hash '
               f'<span class="mono">{_esc(thr.config_hash)}</span> · '
               f'min_seeds {thr.min_seeds} · '
               'champion/candidate 共用同一开发集与同一生产后处理；'
               "冻结集不参与搜索排序。判定只认测量到的门槛项: "
               f'{_esc(sorted(promo["measured"]))}</p>')
    reasons = decision.get("reasons") or []
    items = "".join(f"<li>{_esc(r)}</li>" for r in reasons) or \
        "<li>missing data: 判定没有给出理由</li>"
    out.append(f'<ul class="reasons">{items}</ul>')
    if promo["violations"]:
        viol = "".join(f"<li>{_esc(v)}</li>" for v in promo["violations"])
        out.append(f'<h3>硬门槛逐项</h3><ul class="reasons">{viol}</ul>')
    if decision.get("note"):
        out.append(f'<p class="hint">{_esc(decision["note"])}</p>')
    out.append(_compare_ident(ctx))
    out.append("</section>")
    return "".join(out)


def _verdict_gloss(verdict: str) -> str:
    return {"candidate_better": "候选更好", "champion_better": "champion 更好",
            "inconclusive": "置信区间跨 0，无法分辨"}.get(verdict, "")


def _compare_ident(ctx: dict) -> str:
    """候选层（ident 探针）的成对比较：分子/分母与每 seed 值。"""
    ident = ctx["ident"]
    out = ["<h3>候选层（ident 探针）成对比较</h3>"]
    if not ident.get("readable"):
        out.append(f'<p>{_not_readable(ident.get("error") or "unknown")}</p>')
        out.append('<p class="hint">missing data: 没有 ident 探针时候选身份率'
                   "为未测（不写 0）。</p>")
        return "".join(out)
    arms: dict[str, dict] = {}
    for tag, entry in sorted(ident["models"].items()):
        seed = entry.get("seed")
        if seed is None:
            continue
        arm = ARM_RE.match(tag)
        if not arm:
            continue
        arms.setdefault(arm.group(1), {})[seed] = entry.get("summary") or {}
    out.append('<p class="hint">ident 跑在冻结集路段上（t13_testA2/t13_testB），'
               "口径是候选层 image-plane / 投影，与像素层不混算。</p>")
    table = ['<table><tr><th>指标</th><th>模型</th><th>每 seed 值（分子/分母）'
             "</th></tr>"]
    rows: dict[str, dict[str, list]] = {}
    for label, field_key, num_key, den_key in (
            ("candidate_identity_rate", "match_rate", "candidates_matched",
             "candidates_total"),
            ("role_agreement_rate", "role_agreement_rate", "roles_agreeing",
             "candidates_total"),
            ("offroad_candidate_ratio", None, "candidates_off_road",
             "candidates_total"),
            ("candidate_paint_recall_p50", "candidate_paint_recall_p50",
             None, None)):
        for tag, entry in sorted(ident["models"].items()):
            s = entry.get("summary") or {}
            if not s:
                continue
            if field_key is None:
                num, den = s.get(num_key), s.get(den_key)
                val = (None if num is None or den in (None, 0)
                       else float(num) / float(den))
            else:
                val = s.get(field_key)
                num, den = s.get(num_key), s.get(den_key)
            ev = Evidence(name=label, level=LEVEL_FINAL,
                          value=(None if val is None else float(val)),
                          unit="ratio", numerator=num, denominator=den,
                          missing=("" if val is not None else
                                   "ident summary 没有这一列"))
            rows.setdefault(label, {}).setdefault(tag, []).append(ev)
    for label in sorted(rows):
        for tag in sorted(rows[label]):
            for ev in rows[label][tag]:
                table.append(f'<tr><td>{_esc(label)}{_badge(LEVEL_FINAL)}</td>'
                             f'<td class="mono">{_esc(tag)}</td>'
                             f"<td>{ev.cell()}</td></tr>")
    table.append("</table>")
    out.append("".join(table))
    return "".join(out)


# ---------------------------------------------------------------------------
# 视图 6（含冻结确认）：最终集一次确认
# ---------------------------------------------------------------------------
def _final_view(ctx: dict) -> str:
    ev = ctx["eval"]
    out = ['<section id="final"><h2>最终集一次确认</h2>',
           '<p class="hint">冻结配方在独立最终集上只做一次确认：这里的结果'
           "不参与搜索排序，也不因失败而回头调参（方案 §2/§6）。</p>"]
    if not ev.get("readable") or not ev["splits"].get("frozen"):
        reason = ev.get("error") if not ev.get("readable") else \
            "评估矩阵里没有 frozen 分节"
        out.append(f'<p>{_not_readable(reason or "unknown")}</p>')
        out.append('<p class="hint">missing data: 最终集结果未测。</p>')
        if ctx["t13"].get("readable"):
            out.append(_t13_runs_table(ctx))
        out.append("</section>")
        return "".join(out)
    rows = ev["splits"]["frozen"]
    table = ['<table><tr><th>模型</th><th>帧数</th><th>line precision</th>'
             "<th>line recall</th><th>line IoU</th><th>路外假线 px</th>"
             "<th>p50</th><th>p95</th></tr>"]
    for name, entry in rows.items():
        mn = entry.get("model_name") or name
        def cell(key: str, unit: str = "ratio", num=None, den=None) -> str:
            v = entry.get(key)
            return Evidence(name=key, level=LEVEL_FINAL,
                            value=(None if v is None else float(v)), unit=unit,
                            numerator=num, denominator=den,
                            missing=("" if v is not None else
                                     "eval matrix 没有这一列")).cell()
        table.append(
            f'<tr><td class="mono">{_esc(mn)}</td>'
            f'<td class="num">{_esc(entry.get("n_frames", "未测"))}</td>'
            f'<td>{cell("line_precision")}</td><td>{cell("line_recall")}</td>'
            f'<td>{cell("line_iou")}</td>'
            f'<td>{cell("offroad_false_line_px", "px")}</td>'
            f'<td>{cell("inference_ms_p50", "ms")}</td>'
            f'<td>{cell("inference_ms_p95", "ms")}</td></tr>')
    table.append("</table>")
    out.append("".join(table))
    out.append("<h3>冻结集上的硬门槛（与晋级判定分开显示）</h3>")
    from beamng_autopilot.experiments.gates import (
        Thresholds, threshold_violations)
    thr = Thresholds()
    champion, candidate, ckpt = _pick_pair(rows, ctx.get("champion"))
    arms = _arm_table(rows)
    seeds = _common_seeds(arms.get(champion, {}), arms.get(candidate, {}), ckpt)
    measured = {}
    for seed in seeds:
        entry = arms[candidate][seed].get(ckpt, {})
        for key in ("line_recall", "line_precision"):
            if entry.get(key) is not None:
                measured.setdefault(key, []).append(float(entry[key]))
        if entry.get("offroad_false_frac_of_pred") is not None:
            measured.setdefault("offroad_false_ratio", []).append(
                float(entry["offroad_false_frac_of_pred"]))
        if entry.get("inference_ms_p95") is not None:
            measured.setdefault("inference_ms_p95", []).append(
                float(entry["inference_ms_p95"]))
    # 硬门槛逐 seed 都要过：取最差的那个 seed 值判定（越大越好的取 min，
    # 越小越好的取 max），不给"只报最好 seed"留口子。
    higher_is_better = {"line_recall", "line_precision",
                        "candidate_identity_rate"}
    measured = {k: (min(v) if k in higher_is_better else max(v))
                for k, v in measured.items()}
    ident_used = ""
    for tag, entry in (ctx["ident"].get("models") or {}).items():
        s = entry.get("summary") or {}
        if s.get("match_rate") is not None:
            measured["candidate_identity_rate"] = float(s["match_rate"])
            ident_used = tag
            break
    out.append(f'<p class="hint">判定对象: {_esc(candidate or "missing data")}'
               f'（{_esc(ckpt)}，seed {_esc(seeds) if seeds else "缺"}）'
               + (f' · candidate_identity_rate 来自 ident {_esc(ident_used)}'
                  if ident_used else
                  ' · candidate_identity_rate: 未测（没有 ident 探针）')
               + "</p>")
    violations = threshold_violations(measured, thr)
    if violations:
        items = "".join(f"<li>{_esc(v)}</li>" for v in violations)
        out.append(f'<ul class="reasons">{items}</ul>')
        out.append('<p class="hint">缺测不算通过（方案 §5）：'
                   "UNKNOWN 与超限一样要写出来。</p>")
    else:
        out.append('<p class="hint">全过（每个硬门槛都有测量且达标）。</p>')
    out.append('<p class="hint">注意：冻结集只用于一次确认；逐行数值'
               "不是成对 seed 结论，也不因一次失败回头调参。</p>")
    if ctx["t13"].get("readable"):
        out.append(_t13_runs_table(ctx))
    out.append("</section>")
    return "".join(out)


def _t13_runs_table(ctx: dict) -> str:
    runs = ctx["t13"].get("runs") or []
    if not runs:
        return ""
    items = "".join(f"<li>{_esc(r['run'])}: {_esc(r['path'])}"
                    f"（{_esc(r['epochs'])} epoch）</li>" for r in runs)
    return (f'<h3>T13 历史导入（{len(runs)} 个 train_hist.json）</h3>'
            f'<ul class="reasons">{items}</ul>'
            '<p class="hint">这些是训练历史（训练中/开发集），'
            "本节的最终集数字才是冻结确认；两者不可互相替代。</p>")


# ---------------------------------------------------------------------------
# 视图 7：数据与标签
# ---------------------------------------------------------------------------
def _coverage_row(records: list[dict], split: str) -> dict:
    rows = [r for r in records
            if str(r.get("split") or "none") == split
            and not str(r.get("reject_reason") or "")]
    out = {"n_frames": len(rows)}
    for name in ("road", "paint", "pavement"):
        valid = sum(1 for r in rows
                    if bool(((r.get("quality") or {}).get(name) or {})
                            .get("valid")))
        px = sum(int(((r.get("quality") or {}).get(name) or {})
                     .get("pixels") or 0) for r in rows)
        out[f"{name}_valid_frames"] = valid
        out[f"{name}_px"] = px
    out["paint_valid_frac"] = (None if not rows
                               else out["paint_valid_frames"] / len(rows))
    # 方案点名：有 Tech annotation 但无可靠标线真值要单独计数
    out["tech_annotation_no_paint_truth"] = sum(
        1 for r in rows
        if not bool(((r.get("quality") or {}).get("paint") or {})
                    .get("valid"))
        and (int(r.get("line_px") or 0) > 0
             or str(((r.get("quality") or {}).get("paint") or {})
                    .get("reason") or "").find("engine annotation") >= 0))
    return out


def _data_view(ctx: dict) -> str:
    mf = ctx["manifest"]
    out = ['<section id="data"><h2>数据与标签</h2>']
    if not mf.get("readable"):
        out.append(f'<p>{_not_readable(mf.get("error") or "unknown")}</p>')
        out.append('<p class="hint">missing data: 没有数据清单时帧数、像素覆盖、'
                   "被拒帧与泄漏审计全部为未测（不写 0）。</p></section>")
        return "".join(out)
    records = mf["records"]
    if not records:
        # 0 帧不是成功（manifest.build 的原话）：不许渲染成一排 0
        out.append('<p class="hint">missing data: 清单里没有 records'
                   "（0 帧既不是成功也不是安全通过）。</p>")
        out.append("<table><tr><th>项</th><th>状态</th></tr>")
        for label in ("每划分帧数", "road/paint/pavement 像素覆盖",
                      "有 Tech annotation 但无可靠标线真值", "被隔离帧",
                      "泄漏审计（组/内容/曝光）"):
            out.append(f'<tr><td>{_esc(label)}</td><td>'
                       + _unknown("清单为空，无法统计", level=LEVEL_TRAIN)
                       + "</td></tr>")
        out.append("</table></section>")
        return "".join(out)
    audit = mf["audit"] or {}
    n_by_split = audit.get("n_by_split") or {}
    if not n_by_split:
        for r in records:
            key = str(r.get("split") or "none")
            n_by_split[key] = n_by_split.get(key, 0) + 1
    out.append('<p class="hint">dataset_id <span class="mono">'
               f'{_esc(mf.get("dataset_id"))}</span> · created '
               f'{_esc(mf.get("created"))} · final_frozen '
               + ("是" if mf.get("final_frozen") else
                  ("否" if mf.get("final_frozen") is not None else "未测"))
               + '（数据集层面的计数按"数据准备阶段"标训练中，'
               "final 划分的帧单独标最终集一次确认）</p>")
    table = ['<table><tr><th>划分</th><th>帧数</th><th>road 有效帧/像素</th>'
             "<th>paint 有效帧/像素</th><th>pavement 有效帧/像素</th>"
             "<th>paint 有效占比</th><th>有 Tech annotation 但无可靠标线真值"
             "</th></tr>"]
    for split in sorted(n_by_split):
        cov = _coverage_row(records, split)
        split_level = SPLIT_LEVEL.get(
            {"train": "train", "dev": "dev", "final": "final"}.get(split, ""),
            LEVEL_TRAIN)
        frac = cov["paint_valid_frac"]
        frac_cell = (f'<span class="mono">{frac * 100:.1f}%</span>'
                     f'<span class="hint">（{cov["paint_valid_frames"]}/'
                     f'{cov["n_frames"]}）</span>{_badge(split_level)}'
                     if frac is not None else
                     _unknown("该划分没有帧", level=split_level))
        table.append(
            f'<tr><td>{_esc(SPLIT_LABEL.get(split, split))}'
            f'{_badge(split_level)}</td>'
            f'<td class="num">{cov["n_frames"]}</td>'
            f'<td class="num">{cov["road_valid_frames"]} / {cov["road_px"]}</td>'
            f'<td class="num">{cov["paint_valid_frames"]} / '
            f'{cov["paint_px"]}</td>'
            f'<td class="num">{cov["pavement_valid_frames"]} / '
            f'{cov["pavement_px"]}</td>'
            f"<td>{frac_cell}</td>"
            f'<td class="num">{cov["tech_annotation_no_paint_truth"]}'
            '<span class="hint">（有 Tech annotation 但无可靠标线真值）</span>'
            "</td></tr>")
    table.append("</table>")
    out.append("".join(table))
    rejected = [r for r in records if str(r.get("reject_reason") or "")]
    out.append(f'<h3>被隔离帧: {len(rejected)}{_badge(LEVEL_TRAIN)}</h3>'
               '<p class="hint">被隔离帧不进入任何划分，'
               "是数据准备阶段的计数。</p>")
    if rejected:
        rows = ['<table><tr><th>帧</th><th>run</th><th>原因</th></tr>']
        for r in rejected[:50]:
            rows.append(f'<tr><td class="mono">{_esc(r.get("path"))}</td>'
                        f'<td class="mono">{_esc(r.get("run"))}</td>'
                        f'<td>{_esc(r.get("reject_reason"))}</td></tr>')
        rows.append("</table>")
        if len(rejected) > 50:
            rows.append(f'<p class="hint">只列前 50 条，共 {len(rejected)} 条。</p>')
        out.append("".join(rows))
    else:
        out.append('<p class="hint">没有带 reject_reason 的帧。</p>')
    out.append("<h3>泄漏审计（checked 为假写“未检查”，不写 0）</h3>")
    arow = ['<table><tr><th>检查</th><th>结论</th><th>明细</th></tr>']
    for key, label in (("group_overlap", "组重叠（整组隔离）"),
                       ("content_overlap", "内容/字节复制重叠"),
                       ("exposure_overlap", "同曝光跨视角重叠")):
        entry = audit.get(key) or {}
        checked = bool(entry.get("checked"))
        if not checked:
            verdict = ('<span class="unknown">未检查</span>'
                       '<span class="hint">（本轮数据没有可用于该检查的字段，'
                       "所以不是 0，而是未测）</span>")
        else:
            n = entry.get("n")
            verdict = (f'<span class="mono">{_esc(n)}</span> 组'
                       if n is not None else
                       _unknown("审计条目没有 n"))
        detail = entry.get("detail") or []
        arow.append(f'<tr><td>{_esc(label)}{_badge(LEVEL_TRAIN)}</td>'
                    f"<td>{verdict}</td>"
                    f'<td class="mono">{_esc(", ".join(map(str, detail[:4])))}'
                    "</td></tr>")
    arow.append("</table>")
    out.append("".join(arow))
    if mf.get("notes"):
        items = "".join(f"<li>{_esc(n)}</li>" for n in mf["notes"])
        out.append(f'<h3>清单备注</h3><ul class="reasons">{items}</ul>')
    out.append("</section>")
    return "".join(out)


# ---------------------------------------------------------------------------
# 视图 8：资源与闭环
# ---------------------------------------------------------------------------
def _closed_loop_state(ctx: dict) -> dict:
    """闭环字段只有事件流里真出现对应指标才算"测到"；否则写固定原因。

    返回 ``{字段: Evidence}``，页面渲染 ``.cell()``、证据包导出
    ``.as_dict()``——两者不能各算一遍，否则证据包和页面会不一致。
    """
    found: dict[str, Evidence] = {}
    for ev in ctx["events"].get("events") or []:
        for name, rec in (ev.metrics or {}).items():
            for label, keys in CLOSED_LOOP_FIELDS:
                if name in keys:
                    found[label] = _metric_evidence(rec, name, LEVEL_FINAL)
    return found


def _resources_view(ctx: dict) -> str:
    out = ['<section id="resources"><h2>资源与闭环</h2>']
    ev = ctx["eval"]
    out.append("<h3>推理时延（离线，逐帧）</h3>")
    if ev.get("readable"):
        rows = ['<table><tr><th>划分/模型</th><th>p50</th><th>p95</th>'
                "<th>帧数</th></tr>"]
        for split in sorted(ev["splits"]):
            level = SPLIT_LEVEL.get(split, LEVEL_DEV)
            for name, entry in ev["splits"][split].items():
                def cell(key: str) -> str:
                    v = entry.get(key)
                    return Evidence(name=key, level=level,
                                    value=(None if v is None else float(v)),
                                    unit="ms",
                                    missing=("" if v is not None else
                                             "eval matrix 没有这一列")).cell()
                rows.append(
                    f'<td class="mono">{_esc(split)} / {_esc(name)}'
                    f'{_badge(level)}</td><td>{cell("inference_ms_p50")}</td>'
                    f'<td>{cell("inference_ms_p95")}</td>'
                    f'<td class="num">{_esc(entry.get("n_frames", "未测"))}'
                    "</td></tr>")
        rows.append("</table>")
        out.append("".join(rows))
        out.append('<p class="hint">离线 p95 ≠ Tech 闭环 p95：完整驾驶 tick 的'
                   "deadline 需要实车 run 才能测（方案 §5 性能层）。</p>")
    else:
        out.append(f'<p>{_not_readable(ev.get("error") or "unknown")}</p>')
        out.append('<p class="hint">missing data: 没有评估矩阵，推理 p50/p95 '
                   "未测。</p>")
    out.append("<h3>训练资源（事件流）</h3>")
    res_rows = []
    for evt in ctx["events"].get("events") or []:
        for name, rec in (evt.metrics or {}).items():
            if any(k in str(name) for k in ("gpu", "vram", "step_s", "s_per",
                                             "epoch_s", "minutes", "mem")):
                res_rows.append((f"epoch {evt.epoch} · {name}",
                                 _metric_evidence(rec, str(name),
                                                  LEVEL_TRAIN)))
    if res_rows:
        items = "".join(f'<li>{_esc(label)}: {item.cell()}</li>'
                        for label, item in res_rows[:20])
        out.append(f'<ul class="reasons">{items}</ul>')
    else:
        out.append("<p>" + _unknown("事件流里没有资源指标（GPU 显存/训练速度/"
                                    "每轮耗时）", level=LEVEL_TRAIN) + "</p>")
    out.append("<h3>闭环字段</h3>")
    found = _closed_loop_state(ctx)
    if found:
        rows = ['<table><tr><th>字段</th><th>实测</th></tr>']
        for label, _ in CLOSED_LOOP_FIELDS:
            cell = found[label].cell() if label in found else _unknown(
                CLOSED_LOOP_REASON)
            rows.append(f'<tr><td>{_esc(label)}</td><td>{cell}</td></tr>')
        rows.append("</table>")
        out.append("".join(rows))
    else:
        rows = ['<table><tr><th>字段</th><th>状态</th></tr>']
        for label, _ in CLOSED_LOOP_FIELDS:
            rows.append(f'<tr><td>{_esc(label)}</td>'
                        f'<td>{_unknown(CLOSED_LOOP_REASON)}</td></tr>')
        rows.append("</table>")
        out.append("".join(rows))
        out.append('<p class="hint">本 run 没有 Tech 闭环事件：碰撞/压线/出铺装/'
                   "停车/deadline/新源消费一律写未测，不写 0 事故，"
                   '也不写"安全通过"。</p>')
    out.append("</section>")
    return "".join(out)


# ---------------------------------------------------------------------------
# 上下文与证据包
# ---------------------------------------------------------------------------
def build_context(args: argparse.Namespace) -> dict:
    out_path = Path(args.out)
    out_dir = out_path.parent if str(out_path.parent) else Path(".")
    run_dir = Path(args.run_dir) if getattr(args, "run_dir", None) else None
    events_path = Path(args.events) if getattr(args, "events", None) else None
    if run_dir is not None and events_path is None:
        events_path = (run_dir / "events.jsonl"
                       if (run_dir / "events.jsonl").exists()
                       else (sorted(run_dir.glob("candidates/*/events.jsonl"))
                             or [None])[0])
    eval_path = Path(args.eval) if getattr(args, "eval", None) else None
    hints = [p.parent for p in (events_path, eval_path) if p is not None]
    ctx = {
        "generated_at": _utc_now(),
        "out_path": out_path,
        "champion": getattr(args, "champion", None),
        "events": (merge_event_streams(run_dir) if run_dir is not None
                   else _events_state(events_path)),
        "decisions": _decisions_state(
            run_dir if run_dir is not None
            else (Path(args.decisions) if getattr(args, "decisions", None)
                  else None)),
        "manifest": _manifest_state(
            Path(args.manifest) if getattr(args, "manifest", None) else None),
        "eval": _eval_state(eval_path),
        "ident": _ident_state(
            Path(args.ident) if getattr(args, "ident", None) else None),
        "t13": _t13_state(
            getattr(args, "t13_import", None) or (str(run_dir) if run_dir else None),
            hints, recursive=run_dir is not None),
        "probes": _probe_state(
            Path(args.probes) if getattr(args, "probes", None) else None,
            out_dir),
        "tasks": _task_state(
            Path(args.tasks) if getattr(args, "tasks", None) else None),
    }
    return ctx


def render_html(ctx: dict) -> str:
    legend = " ".join(f"{_badge(l)}" for l in LEVELS)
    parts = [
        "<!DOCTYPE html>",
        '<html lang="zh-CN">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>M5/T14 只读学习看板 — {_esc(_run_label(ctx))}</title>",
        f"<style>{CSS}</style>",
        "</head>",
        "<body>",
        "<header>",
        "<h1>M5/T14 只读学习看板</h1>",
        f'<p class="sub">生成时间（UTC）: {_esc(ctx["generated_at"])} · '
        f"证据等级: {legend}</p>",
        '<p class="sub">只读：不训练、不连接游戏、不调用 ControlBridge、'
        "不写候选目录。缺测与 UNKNOWN 一律渲染成文字，从不当作 0。</p>",
        "</header>",
        _overview_view(ctx),
        _curves_view(ctx),
        _probes_view(ctx),
        _tasks_view(ctx),
        _compare_view(ctx),
        _decisions_view(ctx),
        _final_view(ctx),
        _data_view(ctx),
        _resources_view(ctx),
        "<footer><p>本页由 scripts/m5_seg_dashboard.py 生成；关闭它不影响"
        "训练。缺少测量时本页只写未测，不写安全通过。</p></footer>",
        "</body>",
        "</html>",
    ]
    return "\n".join(parts) + "\n"


def build_pack(ctx: dict) -> dict:
    """证据包：只导出实测值 + 八项报告字段；未测字段带 status=未测 与原因。"""
    ev, mf = ctx["eval"], ctx["manifest"]
    events = ctx["events"]
    last = events.get("last") if events.get("readable") else None
    measured: list[dict] = []
    unmeasured: list[dict] = []
    for split in sorted((ev.get("splits") or {})):
        level = SPLIT_LEVEL.get(split, LEVEL_DEV)
        for name, entry in ev["splits"][split].items():
            for key, unit in (("line_precision", "ratio"),
                              ("line_recall", "ratio"), ("line_iou", "ratio"),
                              ("offroad_false_line_px", "px"),
                              ("inference_ms_p50", "ms"),
                              ("inference_ms_p95", "ms")):
                v = entry.get(key)
                item = {"metric": key, "split": split,
                        "model": entry.get("model_name") or name,
                        "level": LEVEL_LABELS[level]}
                if v is None:
                    item.update(status="未测", reason="eval matrix 没有这一列")
                    unmeasured.append(item)
                else:
                    item["value"] = float(v)
                    item["unit"] = unit
                    measured.append(item)
    for tag, entry in ((ctx["ident"].get("models") or {}) if
                       ctx["ident"].get("readable") else {}).items():
        s = entry.get("summary") or {}
        for key in ("match_rate", "role_agreement_rate",
                    "candidate_paint_recall_p50"):
            item = {"metric": key, "model": tag, "level": "最终集一次确认",
                    "scope": "ident 候选层（与像素层口径不同）"}
            if s.get(key) is None:
                item.update(status="未测", reason="ident summary 没有这一列")
                unmeasured.append(item)
            else:
                item.update(value=float(s[key]), unit="ratio")
                measured.append(item)
    closed = _closed_loop_state(ctx)
    promo = _promotion(ctx)
    # 成对结果也是测量：只有真配到 seed 才导出数值与 verdict
    for row in promo["rows"]:
        item = {"metric": f'paired:{row["metric"]}', "split": promo["split"],
                "champion": promo["champion"], "candidate": promo["candidate"],
                "checkpoint": promo["checkpoint"],
                "level": LEVEL_LABELS.get(promo["level"], promo["level"])}
        if not row["pairs"]:
            item.update(status="未测", reason="no paired seeds measured")
            unmeasured.append(item)
            continue
        summary = row["summary"]
        item.update(
            n=summary.get("n"), mean_delta=summary.get("mean_delta"),
            ci95_halfwidth=summary.get("ci95_halfwidth"),
            verdict=summary.get("verdict"),
            champion_seeds={str(s): b for s, b, _ in row["pairs"]},
            candidate_seeds={str(s): c for s, _, c in row["pairs"]})
        if row["numerator"] is not None:
            item["numerator"] = row["numerator"]
            item["denominator"] = row["denominator"]
        measured.append(item)
    passed = sorted({m["metric"] for m in measured if m.get("unit") == "ratio"})
    report = {
        "commit/config/run": {
            "run_id": (last.run_id if last else None) or
            {"status": "未测", "reason": "没有事件流"},
            "candidate_id": (last.candidate_id if last else None) or
            {"status": "未测", "reason": "事件流没有 candidate_id"},
            "dataset_id": (last.dataset_id if last else None) or
            {"status": "未测", "reason": "事件流没有 dataset_id"},
            "config_hash": (last.config_hash if last else None) or
            {"status": "未测", "reason": "事件流没有 config_hash"},
            "git_commit": (last.git_commit if last else None) or
            {"status": "未测", "reason": "事件流没有 git_commit"},
            "events": events.get("path") or
            {"status": "未测", "reason": "没有事件流"},
            "n_events": (events.get("n_events") if events.get("readable")
                         else {"status": "未测", "reason": "没有事件流"}),
            "dropped_duplicates": (events.get("dropped")
                                   if events.get("readable") else
                                   {"status": "未测", "reason": "没有事件流"}),
            "control_authority": {
                "status": "未测",
                "reason": "本看板只读事件与产物，没有控制权证据（无 Tech run）"},
        },
        "coverage/UNKNOWN": {
            "dataset_id": mf.get("dataset_id") or
            {"status": "未测", "reason": "没有 manifest"},
            "n_by_split": ((mf.get("audit") or {}).get("n_by_split")
                           if mf.get("readable") else None) or
            {"status": "未测", "reason": "没有 manifest 或清单为空"},
            "n_rejected": ((mf.get("audit") or {}).get("n_rejected")
                           if mf.get("readable") and mf.get("records") else
                           {"status": "未测",
                            "reason": "没有 manifest 或清单为空"}),
            "paint_truth_unusable_frames": (
                sum(_coverage_row(mf["records"], s)
                    ["tech_annotation_no_paint_truth"]
                    for s in sorted({str(r.get("split") or "none")
                                     for r in mf["records"]}))
                if mf.get("readable") and mf.get("records") else
                {"status": "未测", "reason": "没有 manifest 或清单为空"}),
            "note": "有 Tech annotation 但无可靠标线真值的帧单独计数",
        },
        "refresh/new-source consumption": {
            "status": "未测", "reason": CLOSED_LOOP_REASON},
        "geometry": {
            "status": "未测",
            "reason": "no independent geometry labels in this run"},
        "surface gate": {
            "status": "未测", "reason": CLOSED_LOOP_REASON},
        "performance/deadline": {
            "full_tick_deadline": {
                "status": "未测",
                "reason": "offline p50/p95 only; the full drive tick needs a "
                          "Tech run"},
            "offline_inference_ms": ([
                m for m in measured
                if str(m.get("metric", "")).startswith("inference_ms")]
                or {"status": "未测", "reason": "没有评估矩阵时没有离线时延"}),
        },
        "closed loop": ({"status": "未测", "reason": CLOSED_LOOP_REASON}
                        if not closed else
                        {"status": "部分测得",
                         "fields": {k: v.as_dict()
                                    for k, v in closed.items()}}),
        "pass/fail/untested": {
            "decision": promo.get("decision") or
            {"status": "未测", "reason": "没有可配对的评估矩阵"},
            "gate_violations": promo.get("violations") or
            {"status": "未测", "reason": "没有可检查的硬门槛测量"},
            "passed": passed or {"status": "未测", "reason": "没有可用测量"},
            "failed": ({"status": "未测", "reason": "本轮没有可判定的失败项"
                                                  "（或判定需要闭环）"}
                       if not promo.get("hards") else promo["hards"]),
            "untested": sorted({u["metric"] for u in unmeasured}),
            "untested_note": ("没有缺测项" if not unmeasured else
                              "缺测项只列名；数值与原因见 unmeasured 列表"),
        },
    }
    return {
        "schema": 1,
        "generated_at": ctx["generated_at"],
        "levels": LEVEL_LABELS,
        "measured": measured,
        "unmeasured": unmeasured,
        "report": report,
        "notes": [
            "本包只导出实际存在的测量；未测项带 status=未测 与原因，"
            "不给数字。",
            "闭环/refresh/geometry/surface gate 在没有 Tech run 时一律未测，"
            "不能读成 0 事故或安全通过。",
        ],
    }


def write_pack(ctx: dict, pack_path: Path, html_text: str) -> str:
    pack = build_pack(ctx)
    text = json.dumps(pack, ensure_ascii=False, indent=1)
    pack_path.parent.mkdir(parents=True, exist_ok=True)
    if str(pack_path).lower().endswith(".json"):
        pack_path.write_text(text, encoding="utf-8")
        return str(pack_path)
    with zipfile.ZipFile(pack_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("evidence.json", text)
        zf.writestr(ctx["out_path"].name, html_text)
    return str(pack_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", required=True,
                        help="生成的 HTML（自包含，内联 CSS，无外部资源）")
    parser.add_argument("--decisions", default=None,
                        help="decision_*.json 所在目录（成对比较/硬门/"
                             "淘汰理由；--run-dir 会自动带上它）")
    parser.add_argument("--run-dir", default=None,
                        help="一次喂完整运行目录：自动合并 run 级与 "
                             "candidates/*/events.jsonl，并递归发现本 run 的 "
                             "train_hist.json（rounds 运行推荐这样用）")
    parser.add_argument("--events", default=None,
                        help="logs/experiments/<run_id>/events.jsonl")
    parser.add_argument("--manifest", default=None,
                        help="DatasetManifest JSON（dataset_id/records/…）")
    parser.add_argument("--eval", default=None,
                        help="评估矩阵 JSON（frozen/dev 两节）")
    parser.add_argument("--ident", default=None,
                        help="ident_*.json 所在目录（或单个文件）")
    parser.add_argument("--t13-import", nargs="?", const="", default=None,
                        metavar="DIR",
                        help="导入 T13 六个 train_hist.json（缺省用 eval/"
                             "events 同级目录与 logs/m5_seg/"
                             "seg_t13_data_20260924）")
    parser.add_argument("--probes", default=None,
                        help="固定探针目录（*.png + sidecar *.json）")
    parser.add_argument("--tasks", default=None,
                        help="候选任务结果 JSON 文件或目录（含 case 样例）")
    parser.add_argument("--champion", default=None,
                        help="成对比较的 champion 臂名（默认按共同 seed 推定）")
    parser.add_argument("--pack", default=None,
                        help="额外写证据包：以 .json 结尾写 JSON，否则写 zip")


def _render_once(args: argparse.Namespace) -> int:
    ctx = build_context(args)
    html_text = render_html(ctx)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_text, encoding="utf-8")
    n_series = len(list(_event_series(ctx["events"]))) + \
        len(ctx["t13"].get("series") or [])
    print(f"[dashboard] {time.strftime('%Y-%m-%d %H:%M:%S')} wrote {out_path} "
          f"({len(html_text)} bytes, {html_text.count(chr(60) + 'section ')} sections, {n_series} curve series)")
    if not ctx["events"].get("readable"):
        print(f"[dashboard] events: 未测 — {ctx['events'].get('error')}")
    else:
        print(f"[dashboard] events: {ctx['events']['n_events']} kept, "
              f"{ctx['events']['dropped']} duplicate point(s) dropped, "
              f"phase={ctx['events']['last'].phase}/"
              f"{ctx['events']['last'].status}")
    if getattr(args, "pack", None):
        written = write_pack(ctx, Path(args.pack), html_text)
        print(f"[dashboard] evidence pack: {written}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="M5/T14 只读学习看板（不训练、不接触游戏、不写候选目录）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    render = sub.add_parser("render", help="渲染一次到 --out")
    _add_common(render)
    watch = sub.add_parser("watch", help="按 --every 秒反复渲染；Ctrl-C 退出")
    _add_common(watch)
    watch.add_argument("--every", type=float, default=15.0,
                       help="重渲染间隔秒数（默认 15）")
    watch.add_argument("--once", action="store_true",
                       help="只渲染一次后退出（用于验证打开看板不影响别的）")
    args = ap.parse_args(argv)
    if args.cmd == "render":
        return _render_once(args)
    try:
        while True:
            _render_once(args)
            if args.once:
                return 0
            time.sleep(max(1.0, float(args.every)))
    except KeyboardInterrupt:
        # 看板是只读旁路：Ctrl-C 只结束自己，不动训练、不留下状态。
        print("\n[dashboard] watch stopped (training untouched)")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
