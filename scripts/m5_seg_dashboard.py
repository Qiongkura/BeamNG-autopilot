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
:root { color-scheme: light;
  --bg:#f5f6f8; --card:#ffffff; --ink:#1f2430; --muted:#6b7280;
  --line:#e3e6ec; --accent:#2f6df6; --raw:#b8c2d6;
  --ok:#2e9e6b; --warn:#c2410c; --bad:#b91c1c; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink);
  font: 13px/1.5 "Segoe UI", "Microsoft YaHei", system-ui, sans-serif; }
header { position: sticky; top: 0; z-index: 5; background: var(--card);
  border-bottom: 1px solid var(--line); padding: 10px 16px; }
h1 { font-size: 16px; margin: 0 0 2px; }
p.sub { color: var(--muted); font-size: 12px; margin: 2px 0; }
.statusbar { display: flex; flex-wrap: wrap; gap: 6px 22px;
  align-items: baseline; margin-top: 6px; }
.item { white-space: nowrap; }
.item .k { color: var(--muted); margin-right: 6px; }
.item .v { font-variant-numeric: tabular-nums; font-weight: 600; }
.pill { padding: 1px 9px; border-radius: 10px; font-size: 12px;
  font-weight: 600; background: #eef1f6; color: #374151; }
.pill.running { background: #e6f0ff; color: #1d4ed8; }
.pill.completed { background: #e7f7ee; color: #1d7a4d; }
.pill.failed, .pill.rejected { background: #fdeaea; color: #b91c1c; }
.pill.paused, .pill.needs_evidence, .pill.needs_review {
  background: #f4efe3; color: #8a6d1f; }
.pill.shadow_candidate { background: #e6f0ff; color: #1d4ed8; }
.banner { margin: 8px 16px 0; padding: 8px 12px; border-radius: 8px;
  font-size: 13px; }
.banner.demo { background: #fff7e6; border: 1px solid #f3d9a4; color: #8a6d1f; }
.banner.err { background: #fdeaea; border: 1px solid #f0b4b4; color: #8c1c1c; }
.banner.note { background: #eef4ff; border: 1px solid #cfe0ff; color: #26406f; }
.toolbar { display: flex; flex-wrap: wrap; gap: 10px 18px;
  align-items: center; padding: 8px 16px 0; color: var(--muted); }
.toolbar a, .toolbar .btn { padding: 3px 10px; border: 1px solid var(--line);
  background: #fff; border-radius: 6px; color: var(--ink);
  text-decoration: none; font-size: 12px; }
.toolbar a.on { background: #e6f0ff; border-color: #9dbdf7; color: #1d4ed8; }
main { display: grid; grid-template-columns: 1fr 1fr; gap: 12px;
  padding: 12px 16px 28px; }
@media (max-width: 1100px) { main { grid-template-columns: 1fr; } }
.card { background: var(--card); border: 1px solid var(--line);
  border-radius: 10px; padding: 8px 10px 6px; min-width: 0; }
.card.wide { grid-column: 1 / -1; }
.card h3 { margin: 0 0 2px; font-size: 13px; font-weight: 600; }
.card .sub { color: var(--muted); font-size: 12px; margin-bottom: 4px; }
.cv { position: relative; width: 100%; min-height: 150px; }
.cv svg { display: block; border: 0; }
.cv svg.fluid { width: 100%; height: auto; }   /* 只有"整卡大图"才铺满 */
svg.spark { width: 150px; height: 34px; }      /* 汇总页的迷你曲线保持固定尺寸 */
.stats { color: var(--muted); font-size: 12px; padding-top: 2px; }
table { border-collapse: collapse; width: 100%; margin: 2px 0 2px; }
th, td { border-bottom: 1px solid var(--line); padding: 3px 6px;
  text-align: left; vertical-align: top; font-size: 12px; }
th { background: #fafbfd; color: #4b5563; font-weight: 600; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.lvl, .badge { display: inline-block; padding: 0 6px; border-radius: 3px;
  font-size: 11px; border: 1px solid var(--line); color: #4b5563;
  background: #fafbfd; white-space: nowrap; }
.lvl.training { background: #eef4ff; border-color: #c7d9fb; color: #2b56b8; }
.lvl.dev { background: #f2f9f0; border-color: #cfe6c6; color: #2e6b3a; }
.lvl.final { background: #fdf5ef; border-color: #f0d8c4; color: #9a5216; }
.lvl.unmeasured { background: #fdf3f3; border-color: #f0c2c2; color: var(--bad); }
.miss { color: var(--bad); }
.unknown { color: var(--warn); }
.ok { color: var(--ok); }
.hint { color: var(--muted); font-size: 12px; }
.mono { font-family: Consolas, "Cascadia Mono", monospace; font-size: 12px; }
svg { background: transparent; }
svg text.ax { fill: var(--muted); font-size: 10px; text-anchor: end; }
svg text.axlbl { fill: #4b5563; font-size: 10px; text-anchor: middle; }
svg text.legend { fill: #4b5563; font-size: 11px; }
svg text.val { fill: var(--ink); font-size: 10px; }
svg text.misspt { fill: var(--bad); font-size: 9px; text-anchor: middle; }
svg text.xtick { fill: var(--muted); font-size: 10px; text-anchor: middle; }
ul.reasons { margin: 4px 0 4px 18px; padding: 0; }
img.probe { max-width: 420px; border: 1px solid var(--line); border-radius: 4px; }
figure { display: inline-block; margin: 8px 14px 8px 0; vertical-align: top; }
figcaption { color: var(--muted); font-size: 12px; margin-top: 2px; }
footer { margin: 0 16px 24px; color: var(--muted); font-size: 12px; }
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


_PALETTE = ("#2f6df6", "#2e9e6b", "#c2410c", "#7c3aed", "#0891b2",
            "#b45309", "#b91c1c", "#4b5563")


def _series_colour(name: str) -> str:
    return _PALETTE[sum(ord(c) for c in name) % len(_PALETTE)]


def _svg_multi(sers: list, *, width: int = 470, height: int = 205,
               fluid: bool = True) -> str:
    """把同一个指称的多条曲线（不同 seed/臂）画在**一张**图里。

    为什么：用户嫌"分出这么多曲线"——300 张单曲线图里绝大多数可以合并
    （同一场实验的不同 seed 本来就是同一条曲线的重复测量）。图例给出每条线的
    末值，方便直接比。
    """
    sers = [x for x in (sers or []) if x is not None]
    if not sers:
        return '<span class="miss">未测</span>'
    xs, ys = [], []
    for se in sers:
        for x, v in se.points:
            if v is not None:
                xs.append(float(x))
                ys.append(float(v))
    if not xs:
        return '<span class="miss">未测</span>'
    x0, x1 = min(xs), max(xs)
    if x0 == x1:
        x0, x1 = x0 - 0.5, x1 + 0.5
    lo, hi = min(ys), max(ys)
    if lo == hi:
        span = max(abs(lo) * 0.1, 1e-6)
        lo, hi = lo - span, hi + span
    span = hi - lo
    lo, hi = lo - span * 0.12, hi + span * 0.12
    pad_l, pad_r, pad_t, pad_b = 58, 96, 22, 30
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b

    def sx(x):
        return pad_l + (x - x0) / (x1 - x0) * plot_w

    def sy(v):
        return height - pad_b - (v - lo) / (hi - lo) * plot_h

    size_attr = ('width="100%" height="auto"' if fluid
                 else f'width="{width}" height="{height}"')
    out = [f'<svg class="{"fluid" if fluid else ""}" viewBox="0 0 {width} '
           f'{height}" {size_attr} role="img">',
           f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" '
           f'fill="#fbfcfe" stroke="#e3e6ec"/>']
    for i in range(5):
        v = lo + (hi - lo) * i / 4.0
        y = sy(v)
        out.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + plot_w}" '
                   f'y2="{y:.1f}" stroke="#eef1f5"/>')
        out.append(f'<text x="{pad_l - 5}" y="{y + 3.5:.1f}" class="ax">'
                   f'{_esc(_fmt_point(v))}</text>')
    for e in range(int(x0), int(x1) + 1):
        if (x1 - x0) <= 14 or e % max(1, int((x1 - x0) / 8)) == 0:
            out.append(f'<text x="{sx(e):.1f}" y="{height - pad_b + 13}" '
                       f'class="xtick">{e}</text>')
    legend_y = pad_t + 8
    for i, se in enumerate(sers):
        colour = _PALETTE[i % len(_PALETTE)]
        segs, seg = [], []
        for x, v in se.points:
            if v is None:
                if seg:
                    segs.append(seg)
                    seg = []
                continue
            seg.append((float(x), float(v)))
        if seg:
            segs.append(seg)
        for sg in segs:
            d = " ".join(("M" if k == 0 else "L") + f"{sx(x):.1f} {sy(v):.1f}"
                         for k, (x, v) in enumerate(sg))
            out.append(f'<path d="{d}" fill="none" stroke="{colour}" '
                       f'stroke-width="1.6"/>')
        last = [v for _x, v in se.points if v is not None]
        tag = se.name.split("/")[-1] or se.name
        out.append(f'<text x="{pad_l + plot_w + 6}" y="{legend_y:.0f}" '
                   f'class="legend" fill="{colour}">{_esc(tag)} '
                   f'{_esc(_fmt_point(last[-1]) if last else "未测")}</text>')
        legend_y += 13
    out.append("</svg>")
    return "".join(out)


def _sparkline(points, *, w: int = 130, h: int = 30, colour: str = "#2f6df6",
               invert: bool = False) -> str:
    """迷你曲线（无坐标轴），嵌在表格单元格里——一页能看很多条。

    缺测处断线（与主图同一纪律：不跨过没测的点连线）。
    """
    vals = [(float(x), float(v)) for x, v in (points or []) if v is not None]
    if not vals:
        return '<span class="miss">未测</span>'
    xs = [x for x, _ in vals]
    x0, x1 = (min(xs), max(xs)) if len(xs) > 1 else (xs[0] - 1, xs[0] + 1)
    lo, hi = min(v for _, v in vals), max(v for _, v in vals)
    if lo == hi:
        span = max(abs(lo) * 0.1, 1e-6)
        lo, hi = lo - span, hi + span
    pad = 2.0

    def sx(x):
        return pad + (x - x0) / (x1 - x0 or 1) * (w - 2 * pad)

    def sy(v):
        t = (v - lo) / (hi - lo or 1)
        if invert:
            t = 1 - t
        return h - pad - t * (h - 2 * pad)

    segs, seg, last_x = [], [], None
    for x, v in (points or []):
        if v is None:
            if seg:
                segs.append(seg)
                seg = []
            last_x = None
            continue
        if last_x is not None and abs(float(x) - last_x) > 1.5 and seg:
            segs.append(seg)
            seg = []
        seg.append((float(x), float(v)))
        last_x = float(x)
    if seg:
        segs.append(seg)
    paths = []
    for sg in segs:
        d = " ".join(("M" if i == 0 else "L") + f"{sx(x):.1f} {sy(v):.1f}"
                     for i, (x, v) in enumerate(sg))
        paths.append(f'<path d="{d}" fill="none" stroke="{colour}" '
                     f'stroke-width="1.4"/>')
    dot = (f'<circle cx="{sx(vals[-1][0]):.1f}" cy="{sy(vals[-1][1]):.1f}" '
           f'r="1.9" fill="{colour}"/>')
    return (f'<svg class="spark" viewBox="0 0 {w} {h}" width="{w}" '
            f'height="{h}" role="img">{"".join(paths)}{dot}</svg>')


def _fmt_point(value: float) -> str:
    if abs(value) >= 100.0:
        return f"{value:.0f}"
    return f"{value:.4g}"


def _svg_series(series: Series, *, width: int = 470, height: int = 205,
                fluid: bool = False) -> str:
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
    size_attr = ('width="100%" height="auto" style="max-height:230px"'
                 if fluid else f'width="{width}" height="{height}"')
    out = [f'<svg class="{"fluid" if fluid else ""}" '
           f'viewBox="0 0 {width} {height}" {size_attr} role="img">',
           f"<title>{_esc(series.name)} · {_esc(series.metric)}</title>",
           f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" '
           f'fill="#fbfcfe" stroke="#e3e6ec"/>']
    for i in range(5):
        v = lo + (hi - lo) * i / 4.0
        y = sy(v)
        out.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + plot_w}" '
                   f'y2="{y:.1f}" stroke="#eef1f5"/>')
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
        # tag 必须**路径唯一**：baseline/seed42 与 round0/seed42 同名会撞在一起
        # （实测：10 份历史塌成 5 个 seed × 4 指标，卡片重复一遍且臂名丢失）
        try:
            tag = str(path.parent.relative_to(state["root"] or path.parent)
                      ).replace("\\", "/")
        except Exception:                                 # noqa: BLE001
            tag = path.parent.name
        epochs = blob.get("epoch") or []
        state["runs"].append({"run": tag, "path": str(path),
                              "epochs": len(epochs)})
        state["loaded"] += 1
        for metric, values in blob.items():
            # epoch/lr 是横轴与调度量；line_ignored_frames 是**累计计数器**——
            # 画成曲线看起来像"某个指标在涨"（实测在页面上就是一条 0→360 的线），
            # 它已经在"训练超参"节里作为数据因子显示。
            if metric in ("epoch", "lr", "line_ignored_frames")                     or not isinstance(values, list):
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
    """标题/页眉用的运行标签：优先事件流，其次 --run-dir 的目录名。

    实测：只给 `--run-dir` 时（没有 events.jsonl 的单臂训练）标题会写成
    "no run"——浏览器的标签页上就挂着这四个字，等于把"这是哪一轮"丢了。
    """
    last = (ctx["events"].get("last") if ctx["events"].get("readable")
            else None)
    if last is not None:
        return f'{last.run_id} / {last.candidate_id or "candidate"}'
    root = _run_root(ctx)
    if root is not None:
        return root.name
    return "no run"


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
            "<th>FP / FN</th><th>指标<br><span style=font-weight:400;font-size:11px;color:#6b7280>成对比较用的主指标（当前 road_iou；标线指标缺真值时为 UNKNOWN）</span></th><th>判定 / 淘汰理由</th></tr>"]
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
             "items": [], "gpu_minutes": {}}
    if path is None:
        state["error"] = "no --decisions/--run-dir given"
        return state
    root = Path(path)
    ledger = {}
    lp = root / "gpu_minutes.json"
    if lp.exists():
        try:
            ledger = json.loads(lp.read_text(encoding="utf-8"))
        except Exception:                                 # noqa: BLE001
            ledger = {}
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
        # v5 计数契约（方案 v2 §S3.7）：旧判定没有整数计数就**不能**按新分母重判，
        # 页面对新字段只能写"旧协议未记录"（见 gates.legacy_replay_note）。
        # 判定原样留在 item 里，七个 v5 面板直读它、不重算。
        from beamng_autopilot.experiments.gates import legacy_replay_note
        legacy_note = legacy_replay_note(blob) or ""
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
            "n_train_frames_by_arm": blob.get("n_train_frames_by_arm") or {},
            "max_train_frames": blob.get("max_train_frames"),
            "equal_steps": blob.get("equal_steps_requested"),
            "factor": blob.get("factor"),
            "worst_frames_by_seed": blob.get("worst_frames_by_seed") or {},
            "trivial_all_road": blob.get("road_iou_trivial_all_road"),
            "eval_checkpoint": blob.get("eval_checkpoint"),
            "epochs": blob.get("epochs"),
            # 逐 seed / 逐场景的**原始测量**（方案 §10.2/A7：坏 seed 或坏场景
            # 不能被池化均值藏掉，所以页面上要能直接看到）
            "hard_by_seed": blob.get("hard_by_seed") or {},
            "per_scene": blob.get("per_scene") or {},
            "scene_candidates": blob.get("scene_candidates") or {},
            "missing_metrics": list(blob.get("missing_metrics") or []),
            # S4/T16 七面板直读：判定原样保留 + 旧协议标记（新计数缺失时
            # 页面写"旧协议未记录"，绝不补 0）。
            "blob": blob,
            "legacy_note": legacy_note,
        })
    state["gpu_minutes"] = ledger
    state["readable"] = bool([i for i in state["items"] if not i.get("error")])
    return state


#: 事件 (phase, status) -> 时间线阶段（方案 §11：采集 → 审计 → 标注 →
#: 训练 → 评价 → 判定 → 下一轮）。
STAGE_OF_EVENT = {
    ("auditing", "collecting"): "采集",
    ("auditing", "collect_ok"): "审计",
    ("auditing", "collect_rejected"): "审计",
    ("auditing", "collect_blocked"): "审计",
    ("auditing", "collect_failed"): "审计",
    ("auditing", "resource_blocked"): "资源门",
    ("auditing", "resource_checked"): "资源门",
    ("auditing", "stopped_before_start"): "停止",
    ("auditing", "lease_blocked"): "资源门",
    ("auditing", "no_data_configured"): "停止",
    ("needs_review", "collect_rejected"): "标注/复核",
    ("needs_review", "labels_missing"): "标注/复核",
    ("training", "running"): "训练",
    ("training", "train_error"): "训练",
    ("evaluating", "running"): "评价",
    ("evaluating", "evaluated"): "评价",
    ("rejected", "decided"): "判定",
    ("shadow_candidate", "decided"): "判定",
    ("needs_evidence", "decided"): "判定",
    ("paused", "paused"): "暂停",
    ("failed", "failed"): "失败",
}


def _timeline_state(run_dir) -> dict:
    """全流程时间线（方案 §11）：从既有产物重建，不新造数据库。

    验收动作是「**从一个候选反查它来自哪次采集和哪份标签**」，所以除事件流外，
    还要把 ``collect_*.json``（采集与身份审计）、``selection_collect_*.json``
    （选样）、``review_queue_*.json``（复核队列）、
    ``proposals_collect_*.json``（数据因子提议）与 ``decision_*.json``（判定）
    串起来：候选 -> 数据目录 -> 采集记录 -> 标签来源。读不到就写"没有记录/
    未测"，不猜、不补 0。
    """
    st = {"readable": False, "error": "", "candidates": [],
          "n_collections": 0, "n_review_queues": 0, "n_selections": 0}
    if run_dir is None:
        st["error"] = "没有 run 目录"
        return st
    root = Path(run_dir)
    if not root.is_dir():
        st["error"] = f"{root} 不存在"
        return st

    def _loads(pattern):
        out = []
        for fp in sorted(root.glob(pattern)):
            try:
                out.append((fp, json.loads(fp.read_text(encoding="utf-8"))))
            except Exception as exc:                       # noqa: BLE001
                out.append((fp, {"_error": f"{type(exc).__name__}: {exc}"}))
        return out

    collects = _loads("collect_*.json")
    selections = _loads("selection_collect_*.json")
    queues = _loads("review_queue_*.json")
    decisions = _loads("decision_*.json")
    st["n_collections"] = len(collects)
    st["n_selections"] = len(selections)
    st["n_review_queues"] = len(queues)
    if not any((collects, selections, queues, decisions)):
        st["error"] = "run 目录里没有采集/选样/判定产物"
        return st

    coll_by_dir = {}
    for fp, blob in collects:
        if blob.get("_error"):
            continue
        d = str(blob.get("out_dir") or "")
        if d:
            coll_by_dir[d] = {"file": fp.name, "stamp": blob.get("stamp"),
                              "map_name": blob.get("map_name"),
                              "source_id": blob.get("source_id"),
                              "frames_total": blob.get("frames_total"),
                              "ok": blob.get("ok"),
                              "group_spread": blob.get("group_spread"),
                              "selection": blob.get("selection")}

    def _norm(x) -> str:
        return str(x).replace(chr(92), "/")

    def _collection_of(run_paths):
        """候选的数据目录 -> 它来自哪次采集（按目录前缀匹配，不模糊猜）。

        同一采集的多个视角目录要合并成一行：否则一次四视角采集会在页面上
        显示成"来自采集"四遍，看起来像四次采集。
        """
        hits: dict = {}
        # 先按路径去重：同一个目录可能同时出现在 candidate_runs 与
        # baseline_runs 里（基线臂=候选臂的起点），不该被数成两个目录。
        for rp in sorted({str(x) for x in (run_paths or [])}):
            rpn = _norm(rp)
            for d, rec in coll_by_dir.items():
                dn = _norm(d)
                if rpn == dn or rpn.startswith(dn + "/"):
                    key = (str(rec.get("stamp")), str(rec.get("source_id")))
                    if key in hits:
                        hits[key]["n_dirs"] += 1
                        hits[key]["matched_runs"].append(str(rp))
                    else:
                        hits[key] = {**rec, "n_dirs": 1,
                                     "matched_runs": [str(rp)]}
                    break
        return list(hits.values())

    for fp, blob in decisions:
        if blob.get("_error"):
            continue
        runs = list(blob.get("candidate_runs") or []) + \
            list(blob.get("baseline_runs") or [])
        res = blob.get("paint_source_resolution") or {}
        st["candidates"].append({
            "file": fp.name,
            "candidate_id": blob.get("candidate_id"),
            "factor": blob.get("factor"),
            "data_runs": [str(r) for r in runs],
            "collections": _collection_of(runs),
            "label": {
                "declared": blob.get("paint_sources"),
                "research_only": blob.get("research_only"),
                "why": res.get("reasons"),
            },
            "protocol_hash": (blob.get("protocol") or {}).get("hash"),
            "decision": (blob.get("decision") or {}).get("decision"),
            "reasons": list((blob.get("decision") or {}).get("reasons") or []),
            "skipped_factors": blob.get("skipped_factors"),
            "data_factor_note": blob.get("data_factor_note"),
        })
    st["readable"] = True
    return st


def _timeline_view(ctx: dict) -> str:
    """全流程时间线 + 候选反查（方案 §11 的验收动作）。"""
    tl = ctx.get("timeline") or {}
    ev = ctx.get("events") or {}
    out = ['<section id="timeline"><h2>全流程时间线</h2>',
           '<p class="hint">采集 → 审计 → 标注/复核 → 训练 → 评价 → 判定 → '
           '下一轮。时间来自事件流；ID 与帧数来自采集/选样/判定产物；'
           '读不到就写"没有记录"，不补 0。</p>']
    rows = ['<table><tr><th>时间</th><th>阶段</th><th>状态</th>'
            "<th>候选/数据集</th><th>说明</th></tr>"]
    n_rows = 0
    for e in (ev.get("events") or []):
        stage = STAGE_OF_EVENT.get((e.phase, e.status))
        if not stage:
            continue
        rows.append(
            f'<tr><td class="mono">{_esc(e.ts)}</td><td>{_esc(stage)}</td>'
            f'<td class="mono">{_esc(e.phase)}/{_esc(e.status)}</td>'
            f'<td class="mono">{_esc(e.candidate_id or "")}'
            f'{(" · " + _esc(e.dataset_id)) if e.dataset_id else ""}</td>'
            f'<td>{_esc(e.note or "")}</td></tr>')
        n_rows += 1
    if n_rows:
        rows.append("</table>")
        out.append("".join(rows))
    else:
        out.append("<p>" + _unknown("事件流里没有可识别的阶段事件"
                                    "（采集/审计/训练/评价/判定/暂停）",
                                    level=LEVEL_TRAIN) + "</p>")
    out.append("<h3>候选反查：它来自哪次采集、哪份标签</h3>")
    if not tl.get("readable"):
        out.append("<p>" + _unknown(
            f"反查不可用：{tl.get('error') or '没有产物'}", level=LEVEL_TRAIN)
            + "</p></section>")
        return "".join(out)
    if not tl.get("candidates"):
        out.append("<p>" + _unknown("还没有判定文件，无法反查候选",
                                    level=LEVEL_TRAIN) + "</p></section>")
        return "".join(out)
    for c in tl["candidates"]:
        out.append(f'<h4 class="mono">{_esc(c.get("candidate_id") or "?")}</h4>')
        rows2 = ['<table><tr><th>项</th><th>内容</th></tr>']
        rows2.append('<tr><td>因子</td><td class="mono">'
                     + _esc(json.dumps(c.get("factor"), ensure_ascii=False))
                     + "</td></tr>")
        if c.get("collections"):
            for col in c["collections"]:
                sp = col.get("group_spread") or []
                cov = ""
                if sp:
                    cov = (f' · 覆盖比 {_esc(sp[0].get("coverage_ratio"))}'
                           f'（{_esc(sp[0].get("extent_m"))} m）')
                rows2.append(
                    f'<tr><td>来自采集</td><td class="mono">'
                    f'{_esc(col.get("stamp"))} · {_esc(col.get("map_name"))}/'
                    f'{_esc(col.get("source_id"))} · '
                    f'{_esc(col.get("frames_total"))} 帧 · 身份审计'
                    f'{"通过" if col.get("ok") else "未通过"} · '
                    f'{_esc(col.get("n_dirs"))} 个目录{cov}</td></tr>')
        else:
            rows2.append('<tr><td>来自采集</td><td>'
                         + _unknown("这些数据目录没有对应的采集记录"
                                    "（可能是早期数据或手工目录）",
                                    level=LEVEL_TRAIN) + "</td></tr>")
        lab = c.get("label") or {}
        lab_txt = _esc(json.dumps(lab.get("declared"), ensure_ascii=False))
        if lab.get("research_only"):
            lab_txt += ' <span class="badge">research_only：不晋级</span>'
        rows2.append(f'<tr><td>标签来源</td><td class="mono">{lab_txt}</td></tr>')
        rows2.append('<tr><td>协议哈希</td><td class="mono">'
                     + _esc(c.get("protocol_hash") or "未记录") + "</td></tr>")
        rows2.append(f'<tr><td>判定</td><td><b>{_esc(c.get("decision") or "未判定")}'
                     "</b>"
                     + ("".join(f"<div>{_esc(r)}</div>"
                                for r in (c.get("reasons") or []))
                        or '<div class="hint">没有理由记录</div>')
                     + "</td></tr>")
        if c.get("skipped_factors"):
            rows2.append('<tr><td>未采用的因子</td><td class="mono">'
                         + _esc(json.dumps(c["skipped_factors"],
                                           ensure_ascii=False))
                         + "</td></tr>")
        if c.get("data_factor_note"):
            rows2.append('<tr><td>下一轮输入</td><td>'
                         + _esc(c["data_factor_note"]) + "</td></tr>")
        rows2.append("</table>")
        out.append("".join(rows2))
    out.append(f'<p class="hint">产物计数：采集记录 {tl.get("n_collections")} · '
               f'选样 {tl.get("n_selections")} · 复核队列 '
               f'{tl.get("n_review_queues")} · 判定 '
               f'{len(tl.get("candidates") or [])}</p>')
    out.append("</section>")
    return "".join(out)


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
        rows = ['<table><tr><th>指标<br><span style=font-weight:400;font-size:11px;color:#6b7280>成对比较用的主指标（当前 road_iou；标线指标缺真值时为 UNKNOWN）</span></th><th>champion 每 seed</th>'
                "<th>candidate 每 seed</th><th>deltas</th>"
                "<th>mean delta</th><th>ci95 半宽</th><th>verdict<br><span style=font-weight:400;font-size:11px;color:#6b7280>candidate_better / champion_better / inconclusive</span></th></tr>"]
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
        _ntf = it.get("n_train_frames_by_arm") or {}
        if _ntf:
            out.append('<p class="hint">两臂**实际入训样本数**（来自 checkpoint 的 '
                       "train_args；等步数对照的另一半证据）:</p>")
            rows_n = ['<table><tr><th>seed</th><th>基线臂</th><th>候选臂</th></tr>']
            _seeds = sorted(set(_ntf.get("baseline") or {})
                            | set(_ntf.get("candidate") or {}),
                            key=lambda x: str(x))
            for sd in _seeds:
                b = (_ntf.get("baseline") or {}).get(sd)
                c = (_ntf.get("candidate") or {}).get(sd)
                def _n(v):
                    return (Evidence(name=sd, level=LEVEL_TRAIN,
                                     value=(None if v is None else float(v)),
                                     unit="count",
                                     missing=("" if v is not None
                                              else "checkpoint 里没有 n_train")
                                     ).cell())
                rows_n.append(f'<tr><td class="mono">seed {_esc(sd)}</td>'
                              f"<td>{_n(b)}</td><td>{_n(c)}</td></tr>")
            rows_n.append("</table>")
            out.append("".join(rows_n))
        _hbs = it.get("hard_by_seed") or {}
        if _hbs:
            out.append('<p class="hint">逐 seed 硬门输入（每个 seed 的 checkpoint '
                       "必须**自己**满足门槛；某一行明显差就是那个 seed 不合格，"
                       "池化均值会把它藏掉）:</p>")
            _keys = ("candidate_identity_rate", "line_recall", "line_precision",
                     "offroad_false_ratio", "inference_ms_p95")
            rows_seed = ['<table><tr><th>seed</th>'
                         + "".join(f"<th>{_esc(k)}</th>" for k in _keys)
                         + "</tr>"]
            for seed in sorted(_hbs, key=lambda x: str(x)):
                cells = []
                for k in _keys:
                    v = (_hbs.get(seed) or {}).get(k)
                    cells.append("<td>" + (Evidence(
                        name=k, level=LEVEL_TRAIN, value=(None if v is None
                                                          else float(v)),
                        unit=("ms" if k == "inference_ms_p95" else "ratio"),
                        missing=("" if v is not None else "该 seed 未测")
                    ).cell()) + "</td>")
                rows_seed.append(f'<tr><td class="mono">seed {_esc(seed)}</td>'
                                 + "".join(cells) + "</tr>")
            rows_seed.append("</table>")
            out.append("".join(rows_seed))
        _ps = it.get("per_scene") or {}
        if _ps:
            _sc = it.get("scene_candidates") or {}
            out.append('<p class="hint">逐场景（跨 seed 取**最差**；场景用 '
                       "map/source_id 分组，不能只报池化值）:</p>")
            rows_sc = ['<table><tr><th>场景</th><th>line_recall</th>'
                       "<th>line_precision</th><th>offroad_false_ratio</th>"
                       "<th>inference_ms_p95</th><th>候选覆盖率</th>"
                       "<th>左右角色</th></tr>"]
            for g in sorted(_ps):
                m = _ps.get(g) or {}
                cand = _sc.get(g) or {}
                def _cell(v, unit="ratio", miss="该场景未测"):
                    return Evidence(name=g, level=LEVEL_TRAIN,
                                    value=(None if v is None else float(v)),
                                    unit=unit,
                                    missing=("" if v is not None else miss)).cell()
                rows_sc.append(
                    f'<tr><td class="mono">{_esc(g)}</td>'
                    f'<td>{_cell(m.get("line_recall"))}</td>'
                    f'<td>{_cell(m.get("line_precision"))}</td>'
                    f'<td>{_cell(m.get("offroad_false_ratio"))}</td>'
                    f'<td>{_cell(m.get("inference_ms_p95"), "ms")}</td>'
                    f'<td>{_cell(cand.get("candidate_reference_coverage"))}</td>'
                    f'<td>{_cell(cand.get("left_right_role_agreement"))}</td>'
                    "</tr>")
            rows_sc.append("</table>")
            out.append("".join(rows_sc))
        if it.get("missing_metrics"):
            out.append('<p class="hint">缺测清单（进 needs_evidence 的通道）: '
                       + _esc("; ".join(map(str, it["missing_metrics"])))
                       + "</p>")
        if it.get("hard_unknown"):
            out.append('<p class="hint">硬门未测（UNKNOWN）: '
                       f'{_esc(", ".join(it["hard_unknown"]))}'
                       "——这些项没有测量，既不算通过也不算违反。</p>")
        # 账本与"有没有 UNKNOWN 硬门项"无关：原来它缩进在 if 里面，于是
        # 硬门全部测到的判定文件一渲染就 UnboundLocalError（实测：真 run 的
        # 判定文件 hard_unknown 为空，看板直接崩）。
        ledger = st.get("gpu_minutes") or {}
        if ledger:
            days = ", ".join(f"{d}: {v.get('minutes')} min"
                             for d, v in sorted(ledger.items()))
            out.append(f'<p class="hint">GPU 墙钟账本（跨进程累计，供每日上限）: '
                       f'{_esc(days)}</p>')
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


# ---------------------------------------------------------------------------
# 视图：方案 v2 §S4 / T16 七面板（直读判定 JSON，不重算指标）
# ---------------------------------------------------------------------------
#: T16 的七个面板（顺序固定；标题就是验收时检查的标签）。
V5_PANELS = ("本轮身份", "数据入口", "学习过程", "比较结果", "无线诊断",
             "错误定位", "资源与判定")

#: 两种"没有数"必须分开写：新判定缺字段 = 无数据；v5 计数契约之前的旧判定
#: = 旧协议未记录（方案 v2 §S3.7：缺新计数字段时说明"无法用新口径重判"，
#: 不能给旧记录补 0）。
NO_DATA = "无数据"
LEGACY_RECORD = "旧协议未记录"

#: 适用性状态 -> 人读标签（四种状态的取值来自 protocol.APPLICABILITY）。
#: 标签里**保留协议原文取值**（measured/not_applicable/unknown/
#: unverified_labels）：页面要让"不适用"与"未测"一眼分开，而不是把两者都
#: 说成"没测到"——not_applicable 既不算通过也不算缺测（方案 v2 §3.4）。
APPLICABILITY_LABELS = {
    "measured": "measured（实测：有真值且有可判候选）",
    "not_applicable": "not_applicable（不适用：确认真无线，不进入有线门，也不算通过）",
    "unknown": "unknown（未测：有真值但没有可判分母，不是 0 分）",
    "unverified_labels": "unverified_labels（标签未验证：仅诊断，不得当结论）",
}


def _src_note(*fields) -> str:
    """可追溯：这一格读的是哪个 JSON 字段（T16 硬要求）。"""
    text = " / ".join(str(f) for f in fields if str(f))
    return f'<span class="hint">来源: {_esc(text)}</span>' if text else ""


def _v5_no_data(reason: str, *, legacy: bool = False, source: str = "") -> str:
    label = LEGACY_RECORD if legacy else NO_DATA
    out = f'<span class="miss">{label}：{_esc(reason or "字段没有记录")}</span>'
    if source:
        out += " " + _src_note(source)
    return out


def _v5_num(value, *, unit: str = "", numerator=None, denominator=None,
            level: str = LEVEL_TRAIN, legacy: bool = False, reason: str = "",
            source: str = "", digits: int = 4) -> str:
    """JSON 里真实存在的数值才渲染；None/缺失 -> 只写原因（0 就是实测 0）。"""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return _v5_no_data(reason, legacy=legacy, source=source)
    out = Evidence(name="", level=level, value=float(value), unit=unit,
                   numerator=numerator, denominator=denominator,
                   digits=digits).cell()
    if source:
        out += " " + _src_note(source)
    return out


def _v5_pct(value, *, numerator=None, denominator=None, level: str = LEVEL_TRAIN,
            legacy: bool = False, reason: str = "", source: str = "") -> str:
    """比率单元格：带分子/分母；小于 1% 的比率多给有效位。

    为什么不用 Evidence 的固定一位小数：``0.0003`` 会被印成 ``0.0%``，
    等于把"测到了很小的假线率"说成"没测到"（与"不写假 0"同一条纪律）。
    """
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return _v5_no_data(reason, legacy=legacy, source=source)
    v = float(value)
    text = f"{v * 100:.2f}%" if abs(v) >= 0.01 else f"{v * 100:.4g}%"
    out = f'<span class="mono">{text}</span>'
    if numerator is not None and denominator is not None:
        out += (f' <span class="hint">（{int(numerator)}/{int(denominator)}）'
                "</span>")
    out += _badge(level)
    if source:
        out += " " + _src_note(source)
    return out


def _v5_int(value, *, level: str = LEVEL_TRAIN, legacy: bool = False,
            reason: str = "", source: str = "") -> str:
    if value is None:
        return _v5_no_data(reason, legacy=legacy, source=source)
    out = f'<span class="mono">{int(value)}</span>{_badge(level)}'
    if source:
        out += " " + _src_note(source)
    return out


def _v5_pick(blob: dict, *keys):
    """``blob`` 里第一个存在的字段 -> ``(value, key)``；都没有 -> ``(None, "")``。"""
    if not isinstance(blob, dict):
        return None, ""
    for k in keys:
        if k in blob:
            return blob[k], k
    return None, ""


def _v5_row(label: str, value: str, source: str = "") -> str:
    return (f'<tr><td>{_esc(label)}</td><td>{value}</td>'
            f'<td class="mono">{_src_note(source) if source else ""}</td></tr>')


def _v5_rows_table(rows: list, *, headers=("项", "值", "来源")) -> str:
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = "".join(rows)
    return f"<table><tr>{head}</tr>{body}</table>"


def _v5_kv_table(pairs: list) -> str:
    """``[(label, value_html, source)]`` -> 三列表（来源列钉住可追溯性）。"""
    if not pairs:
        return ""
    rows = "".join(_v5_row(str(a), b, c) for a, b, c in pairs)
    return ('<table><tr><th>项</th><th>值</th><th>来源字段</th></tr>'
            f"{rows}</table>")


def _v5_rounds_dataset(ctx: dict) -> dict:
    """运行目录里的 ``rounds_dataset.json``（数据准入门产物，只读）。

    七个面板里"去重/冲突/已见组/空间 UNKNOWN"的权威来源是它（或判定文件里
    同名字段）。读不到只记原因，不猜、不补 0。
    """
    root = _run_root(ctx)
    if root is None:
        return {"readable": False, "error": "没有 run 目录"}
    path = root / "rounds_dataset.json"
    if not path.is_file():
        return {"readable": False, "error": f"{path.name} 不存在"}
    blob, err = _load_json(path)
    if err:
        return {"readable": False, "error": err}
    if not isinstance(blob, dict):
        return {"readable": False, "error": "not a JSON object"}
    return {"readable": True, "error": "", "path": str(path), "blob": blob}


def _v5_frame_ref(frame, out_dir) -> str:
    """原帧引用：能相对 HTML 打开才给链接，否则只给路径（不假装能点）。

    与探针图同一条纪律：页面只引用**相对本文件真实存在**的文件。
    """
    raw = str(frame or "")
    if not raw:
        return _v5_no_data("该条目没有帧路径，无法追溯原帧")
    p = Path(raw)
    if not p.exists():
        return (f'<span class="miss">missing: {_esc(raw)}</span>'
                '<span class="hint">（文件不存在，无法点击定位；路径来自判定'
                "文件，不在页面上造假链接）</span>")
    try:
        rel = os.path.relpath(p.resolve(), Path(out_dir).resolve())
    except ValueError as exc:                          # Windows 跨盘符
        return (f'<span class="mono">{_esc(raw)}</span>'
                f'<span class="hint">（无法相对链接：{_esc(exc)}；按原路径可追溯）'
                "</span>")
    if rel.startswith(".."):
        return (f'<span class="mono">{_esc(raw)}</span>'
                '<span class="hint">（文件在页面目录之外：不做相对链接，'
                "按原路径可追溯）</span>")
    rel = rel.replace("\\", "/")
    return (f'<a href="{_esc(rel)}">{_esc(raw)}</a>'
            f'<span class="hint">（点击打开原帧: {_esc(rel)}）</span>')


def _v5_model_rows(ctx: dict, blob: dict, fname: str) -> list:
    """模型完整路径 + 哈希：优先判定文件，其次评估矩阵（都标来源）。"""
    rows = []
    models = blob.get("models")
    if isinstance(models, dict):
        for name in sorted(models, key=str)[:8]:
            m = models[name] if isinstance(models[name], dict) else {}
            rows.append((str(name), m))
            rows[-1][1].setdefault("_src", f"{fname}: models.{name}")
    elif isinstance(models, list):
        for m in models[:8]:
            if isinstance(m, dict):
                rows.append((str(m.get("name") or m.get("tag") or "model"), m))
                rows[-1][1].setdefault("_src", f"{fname}: models[]")
    if not rows:
        path, pkey = _v5_pick(blob, "model_path", "model_full_path")
        sha, skey = _v5_pick(blob, "model_sha16", "sha16", "sha256_16")
        if path is not None or sha is not None:
            rows.append(("model", {"path": path, "sha16": sha,
                                   "_src": f"{fname}: {pkey or skey}"}))
    if not rows:
        ev = ctx.get("eval") or {}
        for split in sorted(ev.get("splits") or {}):
            for name, entry in sorted((ev["splits"][split] or {}).items()):
                if not isinstance(entry, dict):
                    continue
                p = entry.get("model") or entry.get("model_path")
                s = (entry.get("sha256_16") or entry.get("model_sha16")
                     or entry.get("sha16"))
                if p is None and s is None:
                    continue
                rows.append((f"{split}/{name}",
                             {"path": p, "sha16": s,
                              "_src": f"eval matrix: {split}.{name}"
                                      ".model/.sha256_16"}))
                if len(rows) >= 8:
                    break
    return rows


def _v5_identity_block(ctx: dict, it: dict) -> str:
    """面板 1：本轮身份（commit/dirty、run、dataset、协议/阈值、模型、设备）。"""
    blob = it.get("blob") or {}
    fname = str(it.get("file") or "decision_*.json")
    legacy = bool(it.get("legacy_note"))
    ev = ctx.get("events") or {}
    last = ev.get("last") if ev.get("readable") else None
    mf = ctx.get("manifest") or {}
    root = _run_root(ctx)
    out = ["<h3>本轮身份</h3>"]
    pairs = []
    run_name = root.name if root is not None else "未指定"
    cand = str(it.get("candidate_id") or "")
    pairs.append(("运行 / 候选",
                  f'<span class="mono">{_esc(run_name)}</span> · '
                  + (f'<span class="mono">{_esc(cand)}</span>' if cand else
                     _v5_no_data("判定文件没有 candidate_id"))
                  + _badge(LEVEL_TRAIN),
                  f"--run-dir / {fname}: candidate_id"))
    # commit / dirty：判定文件优先，事件流兜底；"dirty 未记录"不等于干净
    commit, ckey = _v5_pick(blob, "git_commit", "commit")
    csrc = f"{fname}: {ckey}"
    if commit is None and last is not None and last.git_commit:
        commit, csrc = last.git_commit, "events.jsonl: git_commit"
    dirty, dkey = _v5_pick(blob, "git_dirty", "git_dirty_paths", "dirty",
                           "dirty_count")
    dsrc = f"{fname}: {dkey}"
    if dirty is None:
        dirty_txt = _v5_no_data("判定文件与事件流都没有工作树状态："
                                "不能声称 commit 对应可复现代码", source=dsrc)
    elif isinstance(dirty, (list, tuple)):
        dirty_txt = (f'<span class="mono">{len(dirty)}</span> 个未提交路径'
                     + (f'<div class="hint mono">{_esc(", ".join(map(str, dirty[:6])))}'
                        "</div>" if dirty else "")
                     + _badge(LEVEL_TRAIN))
    elif isinstance(dirty, bool):
        dirty_txt = (f'<span class="mono">{"dirty" if dirty else "clean"}'
                     f'</span>{_badge(LEVEL_TRAIN)}')
    else:
        dirty_txt = (f'<span class="mono">{_esc(dirty)}</span>'
                     f'{_badge(LEVEL_TRAIN)}')
    pairs.append(("commit", (f'<span class="mono">{_esc(commit)}</span>'
                             f"{_badge(LEVEL_TRAIN)}") if commit else
                 _v5_no_data("没有 git_commit 记录", source=csrc), csrc))
    pairs.append(("工作树 / dirty", dirty_txt, dsrc))
    ds, k = _v5_pick(blob, "dataset_id")
    dsrc2 = f"{fname}: {k}"
    if ds is None and mf.get("readable") and mf.get("dataset_id"):
        ds, dsrc2 = mf["dataset_id"], "manifest: dataset_id"
    if ds is None and last is not None and last.dataset_id:
        ds, dsrc2 = last.dataset_id, "events.jsonl: dataset_id"
    pairs.append(("数据集 dataset_id",
                  f'<span class="mono">{_esc(str(ds)[:24])}</span>'
                  f'{_badge(LEVEL_TRAIN)}' if ds else
                  _v5_no_data("没有 dataset_id 记录", source=dsrc2), dsrc2))
    proto = blob.get("protocol") if isinstance(blob.get("protocol"), dict) \
        else None
    if proto and proto.get("version"):
        ptxt = (f'<span class="mono">{_esc(proto.get("version"))}</span>'
                + (f' · hash <span class="mono">{_esc(proto.get("hash"))}</span>'
                   if proto.get("hash") else ""))
        pairs.append(("协议版本 / 哈希", ptxt, f"{fname}: protocol.version/hash"))
    else:
        pairs.append(("协议版本 / 哈希",
                      _v5_no_data("判定文件没有协议快照版本（v5 之前的记录）",
                                  legacy=True, source=f"{fname}: protocol"),
                      f"{fname}: protocol"))
    thr = blob.get("thresholds") if isinstance(blob.get("thresholds"), dict) \
        else {}
    thr_v = proto.get("thresholds") if proto else None
    if thr or thr_v:
        ttxt = (f'<span class="mono">{_esc(str(thr.get("config_hash") or ""))}'
                "</span>")
        if thr.get("source"):
            ttxt += f' · <span class="hint">{_esc(str(thr.get("source")))}</span>'
        if isinstance(thr_v, dict):
            ttxt += (f'<div class="hint">协议里的门槛值: '
                     f'{_esc(json.dumps(thr_v, ensure_ascii=False, sort_keys=True))}'
                     "</div>")
        pairs.append(("阈值版本 config_hash", ttxt,
                      f"{fname}: thresholds.config_hash / protocol.thresholds"))
    else:
        pairs.append(("阈值版本 config_hash",
                      _v5_no_data("判定文件没有阈值版本", legacy=legacy,
                                  source=f"{fname}: thresholds"),
                      f"{fname}: thresholds"))
    mrows = _v5_model_rows(ctx, blob, fname)
    if mrows:
        for name, m in mrows[:6]:
            path = m.get("path") or m.get("model") or m.get("checkpoint")
            sha = (m.get("sha16") or m.get("model_sha16")
                   or m.get("sha256_16"))
            mtxt = (f'<span class="mono">{_esc(str(path or "路径未记录"))}</span>'
                    + (f' · sha16 <span class="mono">{_esc(str(sha))}</span>'
                       if sha else
                       ' · ' + _v5_no_data("该条目没有权重哈希")))
            pairs.append((f"模型 {name}", mtxt, str(m.get("_src") or fname)))
    else:
        pairs.append(("模型完整路径 / 哈希",
                      _v5_no_data("判定文件与评估矩阵都没有模型路径/哈希",
                                  source=f"{fname}: models / eval matrix"),
                      f"{fname}: models / eval matrix"))
    dev, dev_key = _v5_pick(blob, "device")
    dev_src = f"{fname}: {dev_key}"
    if dev is None:
        env = blob.get("env") if isinstance(blob.get("env"), dict) else {}
        if env.get("device"):
            dev, dev_src = env["device"], f"{fname}: env.device"
    if dev is None:
        ta = (blob.get("train_args")
              if isinstance(blob.get("train_args"), dict) else {})
        if ta.get("device_name") or ta.get("device"):
            dev = ta.get("device_name") or ta.get("device")
            dev_src = f"{fname}: train_args.device"
    if dev is None and root is not None:
        for r in _ckpt_params(root, limit=4):
            env = r.get("env") or {}
            if env.get("device"):
                dev = env["device"]
                dev_src = f"checkpoint {r.get('arm')}/{r.get('seed')}: env.device"
                break
    pairs.append(("实际设备",
                  f'<span class="mono">{_esc(str(dev))}</span>'
                  f'{_badge(LEVEL_TRAIN)}' if dev is not None else
                  _v5_no_data("判定/checkpoint 都没有实际运行设备记录",
                              source=dev_src), dev_src))
    out.append(_v5_kv_table(pairs))
    if legacy:
        out.append('<p class="hint">' + _esc(str(it.get("legacy_note"))) + "</p>")
    return "".join(out)


def _v5_data_block(ctx: dict, it: dict, rounds: dict) -> str:
    """面板 2：数据入口（生成/复核/入训/评价、去重、冲突、rank、已见组、空间）。"""
    blob = it.get("blob") or {}
    fname = str(it.get("file") or "decision_*.json")
    legacy = bool(it.get("legacy_note"))
    mf = ctx.get("manifest") or {}
    out = ["<h3>数据入口</h3>"]
    rows = []
    records = mf.get("records") if mf.get("readable") else None
    dc = blob.get("data_counts") if isinstance(blob.get("data_counts"),
                                               dict) else {}

    def _four(label, value, src, reason="字段没有记录"):
        if value is None:
            cell = _v5_no_data(reason, legacy=legacy, source=src)
        else:
            cell = _v5_int(value, source=src)
        rows.append(_v5_row(label, cell, src))

    _four("生成帧（data_counts.generated）", dc.get("generated"),
          f"{fname}: data_counts.generated", reason="判定文件没有四个数据计数"
          "（v5 之前不记录）")
    _four("复核帧（data_counts.reviewed）", dc.get("reviewed"),
          f"{fname}: data_counts.reviewed", reason="判定文件没有四个数据计数"
          "（v5 之前不记录）")
    if not dc and records is not None:
        n_verified = sum(
            1 for r in records
            if str(((r.get("quality") or {}).get("paint") or {})
                   .get("rank") or "") == "verified")
        rows.append(_v5_row(
            "生成帧（manifest 记录）", _v5_int(len(records)),
            "manifest: records"))
        rows.append(_v5_row(
            "复核帧（paint rank = verified）", _v5_int(n_verified),
            "manifest: records[].quality.paint.rank"))
        if not n_verified:
            rows.append(_v5_row(
                "复核提示",
                '<span class="hint">0 个 verified 档帧：生成帧数不得当人工真值'
                "数（本页按档位分开）</span>",
                "manifest: records[].quality.paint.rank"))
    else:
        rows.append(_v5_row(
            "生成帧（manifest 记录）",
            _v5_no_data("没有 --manifest，无法统计清单记录") if records is None
            else _v5_int(len(records)), "manifest: records"))
    ntf = blob.get("n_train_frames_by_arm")
    if isinstance(ntf, dict) and ntf:
        parts = []
        for arm in ("baseline", "candidate"):
            per = ntf.get(arm) or {}
            if isinstance(per, dict) and per:
                parts.append(arm + " " + "、".join(
                    f"seed {k}: {_v5_int(v)}" for k, v in sorted(
                        per.items(), key=lambda kv: str(kv[0]))))
        rows.append(_v5_row(
            "入训帧（实测，来自 checkpoint train_args）",
            "<br>".join(parts) if parts else _v5_no_data("两臂都没有记录"),
            f"{fname}: n_train_frames_by_arm"))
    else:
        rows.append(_v5_row("入训帧（实测）",
                            _v5_no_data("判定文件没有 n_train_frames_by_arm",
                                        legacy=legacy),
                            f"{fname}: n_train_frames_by_arm"))
    ev = ctx.get("eval") or {}
    if ev.get("readable"):
        parts = []
        for split in sorted(ev.get("splits") or {}):
            for name, entry in sorted((ev["splits"][split] or {}).items()):
                if isinstance(entry, dict) and entry.get("n_frames") is not None:
                    parts.append(f"{split}/{name}: {int(entry['n_frames'])} 帧")
                elif isinstance(entry, dict):
                    parts.append(f"{split}/{name}: " + _v5_no_data(
                        "评估矩阵条目没有 n_frames"))
        rows.append(_v5_row("评价帧（评估矩阵实测）",
                            "<br>".join(parts[:8]) if parts else
                            _v5_no_data("评估矩阵没有 n_frames"),
                            "eval matrix: <split>.<model>.n_frames"))
    else:
        rows.append(_v5_row("评价帧（评估矩阵实测）",
                            _v5_no_data(ev.get("error") or "没有评估矩阵"),
                            "eval matrix: <split>.<model>.n_frames"))
    # 去重：判定字段优先；否则 manifest 的字节复制隔离；否则准入门产物
    rows.append(_v5_row("去重（字节复制隔离）",
                        _v5_dedupe_cell(blob, fname, mf, rounds, legacy),
                        f"{fname}: dedupe / manifest: content_overlap / "
                        "rounds_dataset.json: rejected[]"))
    # 冲突：判定字段优先；否则组重叠（训练/开发同组）
    rows.append(_v5_row("冲突（组重叠/标签冲突）",
                        _v5_conflict_cell(blob, fname, mf, rounds, legacy),
                        f"{fname}: conflicts / manifest: group_overlap / "
                        "rounds_dataset.json: group_overlap"))
    # 来源 rank（凡能进晋级参考的都要 verified；research_only 明确标出）
    ranks = {}
    if records is not None:
        for r in records:
            rank = str(((r.get("quality") or {}).get("paint") or {})
                       .get("rank") or "absent")
            ranks[rank] = ranks.get(rank, 0) + 1
    ptxt = []
    if ranks:
        ptxt.append("、".join(f"{RANK_LABEL.get(k, k)}: {v}"
                              for k, v in sorted(ranks.items())))
    if blob.get("paint_sources") is not None:
        ptxt.append("判定声明 paint_sources: "
                    + _esc(json.dumps(blob.get("paint_sources"),
                                      ensure_ascii=False)))
    if blob.get("research_only"):
        ptxt.append('<span class="badge">research_only：不晋级</span>')
    rows.append(_v5_row("来源 rank / 资格",
                        "<br>".join(ptxt) if ptxt else
                        _v5_no_data("没有 manifest 档位，也没有 paint_sources",
                                    legacy=legacy),
                        "manifest: records[].quality.paint.rank / "
                        f"{fname}: paint_sources, research_only"))
    # 已见组：判定 lineage 字段优先；否则准入门产物的 train_groups
    seen_txt, seen_src = _v5_seen_cell(blob, fname, rounds)
    rows.append(_v5_row("已见组（训练侧组）", seen_txt, seen_src))
    # 空间 UNKNOWN：没有位置的帧不能默认"隔离通过"
    sp_txt, sp_src = _v5_spatial_cell(blob, fname, rounds)
    rows.append(_v5_row("空间 UNKNOWN（没有位置的帧）", sp_txt, sp_src))
    out.append(_v5_rows_table(rows))
    return "".join(out)


def _v5_dedupe_cell(blob: dict, fname: str, mf: dict, rounds: dict,
                    legacy: bool) -> str:
    node = blob.get("dedupe")
    if isinstance(node, dict):
        checked = node.get("checked")
        removed = node.get("removed", node.get("n_removed"))
        groups = node.get("groups", node.get("n_duplicate_groups"))
        if removed is not None or groups is not None:
            return (f'删除 {_v5_int(removed)} 帧 / 重复组 '
                    f'{_v5_int(groups)}（来源 {_esc(fname)}: dedupe）'
                    if removed is not None and groups is not None else
                    f'删除 {_v5_int(removed)} 帧 / 重复组 '
                    f'{_v5_int(groups)}')
        if checked is False:
            return ('<span class="unknown">未检查</span>'
                    '<span class="hint">（判定文件写 checked:false，'
                    "不是 0 个重复）</span>")
    records = mf.get("records") if mf.get("readable") else None
    if records:
        n_dup = sum(1 for r in records
                    if "byte-identical" in str(r.get("reject_reason") or ""))
        note = ("0（清单里没有字节复制帧：这是查过的 0，不是未测）"
                if n_dup == 0 else f"去重隔离 {n_dup} 帧（byte-identical）")
        return f'<span class="mono">{_esc(note)}</span>'
    rb = rounds.get("blob") if rounds.get("readable") else None
    rej = rb.get("rejected") if isinstance(rb, dict) else None
    if isinstance(rej, list):
        n_dup = sum(1 for r in rej
                    if "byte-identical" in str((r or {}).get("reason") or ""))
        return (f'<span class="mono">去重隔离 {n_dup} 帧（byte-identical）'
                f"</span><span class=\"hint\">（准入门 rejected 列表，"
                f"共 {len(rej)} 条被拒）</span>")
    return _v5_no_data("没有 dedupe 字段，也没有清单/准入门记录",
                       legacy=legacy)


def _v5_conflict_cell(blob: dict, fname: str, mf: dict, rounds: dict,
                      legacy: bool) -> str:
    node = blob.get("conflicts")
    if isinstance(node, dict):
        checked = node.get("checked")
        n = node.get("n", node.get("count"))
        if n is not None:
            return f'<span class="mono">{_v5_int(n)}</span>'
        if checked is False:
            return ('<span class="unknown">未检查</span>'
                    '<span class="hint">（判定文件写 checked:false）</span>')
    elif isinstance(node, list):
        return (f'<span class="mono">{len(node)}</span> 条'
                + (f'<div class="hint mono">{_esc(str(node[:3]))}</div>'
                   if node else
                   '<span class="hint">（列表为空＝查过没有冲突）</span>'))
    audit = (mf.get("audit") or {}) if mf.get("readable") else {}
    go = audit.get("group_overlap") or {}
    if go.get("checked") and go.get("n") is not None:
        return (f'组重叠 <span class="mono">{_v5_int(go.get("n"))}</span> 组'
                + (f'<div class="hint mono">{_esc(str(go.get("detail") or []))}'
                   "</div>" if go.get("n") else ""))
    rb = rounds.get("blob") if rounds.get("readable") else None
    ov = rb.get("group_overlap") if isinstance(rb, dict) else None
    if isinstance(ov, list):
        return (f'组重叠 <span class="mono">{len(ov)}</span> 组'
                + (f'<div class="hint mono">{_esc(", ".join(map(str, ov[:4])))}'
                   "</div>" if ov else
                   '<span class="hint">（空列表＝准入门查过没有组重叠）</span>'))
    return _v5_no_data("没有 conflicts 字段，也没有组重叠审计", legacy=legacy)


def _v5_seen_cell(blob: dict, fname: str, rounds: dict) -> tuple:
    lin = blob.get("lineage")
    if isinstance(lin, dict):
        seen = lin.get("seen_groups") or lin.get("seen_in") or []
        unknown = lin.get("unknown_models") or lin.get("lineage_unknown") or []
        txt = ("、".join(map(str, seen)) if seen else
               "没有命中已见组（" + str(lin.get("n_checked", "n 未记录"))
               + " 项已查）")
        if unknown:
            txt += ('<div class="hint">lineage UNKNOWN（checkpoint 读不到 '
                    "runs）: " + _esc("、".join(map(str, unknown))) + "</div>")
        return txt, f"{fname}: lineage.seen_groups/unknown_models"
    rb = rounds.get("blob") if rounds.get("readable") else None
    tg = rb.get("train_groups") if isinstance(rb, dict) else None
    if isinstance(tg, list):
        return ("、".join(map(str, tg)) if tg else
                "0 个训练组（准入门记录为空列表）",
                "rounds_dataset.json: train_groups")
    return (_v5_no_data("没有 lineage 字段，也没有准入门 train_groups",
                        legacy=not bool(rounds.get("readable"))),
            f"{fname}: lineage / rounds_dataset.json: train_groups")


def _v5_spatial_cell(blob: dict, fname: str, rounds: dict) -> tuple:
    from beamng_autopilot.experiments.protocol import SPATIAL_BUFFER_M
    for node, src in ((blob.get("spatial"), f"{fname}: spatial"),
                      ((rounds.get("blob") or {}).get("spatial")
                       if rounds.get("readable") else None,
                       "rounds_dataset.json: spatial")):
        if not isinstance(node, dict):
            continue
        missing = node.get("n_missing_position")
        if missing is None:
            continue
        txt = (f'<span class="mono">{int(missing)}</span> 帧没有位置'
               + ("（0＝所有帧都有位置，距离判定完整）" if int(missing) == 0
                  else '<span class="hint">（这些帧无法判空间距离，不是通过）'
                       "</span>"))
        if node.get("min_distance_m") is not None:
            txt += (f' · 最小距离 <span class="mono">'
                    f'{_esc(node.get("min_distance_m"))} m</span>(缓冲 '
                    f'{_esc(node.get("buffer_m") or SPATIAL_BUFFER_M)} m)')
        if node.get("n_pairs_checked") is not None:
            txt += f' · 已查 {_esc(node.get("n_pairs_checked"))} 对'
        if node.get("violations"):
            txt += (f' · <span class="miss">违规 '
                    f'{len(node.get("violations"))} 条</span>')
        return txt, src
    return (_v5_no_data("判定与准入门产物都没有空间隔离计数",
                        legacy=not bool(rounds.get("readable"))),
            f"{fname}: spatial / rounds_dataset.json: spatial")


def _v5_learning_block(ctx: dict, it: dict) -> str:
    """面板 3：学习过程（step/epoch、train/dev loss、样本步数、配方、checkpoint）。"""
    blob = it.get("blob") or {}
    fname = str(it.get("file") or "decision_*.json")
    legacy = bool(it.get("legacy_note"))
    ev = ctx.get("events") or {}
    last = ev.get("last") if ev.get("readable") else None
    t13 = ctx.get("t13") or {}
    root = _run_root(ctx)
    out = ["<h3>学习过程</h3>"]
    pairs = []
    ep, ekey = _v5_pick(blob, "candidate_epochs", "epochs")
    if ep is None and last is not None:
        ep, ekey = last.epoch, "events.jsonl: epoch"
    pairs.append(("step / epoch",
                  (f'<span class="mono">epoch {_esc(ep)}</span> · step '
                   f'<span class="mono">{_esc(last.step if last else "未记录")}'
                   "</span>") + _badge(LEVEL_TRAIN)
                  if ep is not None else
                  _v5_no_data("判定文件与事件流都没有 epoch/step 记录",
                              source=f"{fname}: epochs"),
                  f"{fname}: {ekey} / events.jsonl: epoch,step"))
    dyn = ctx.get("dyn") or []
    if dyn:
        parts = []
        for d in dyn[:4]:
            if d.get("error"):
                parts.append(f'{d.get("run")}: ' + _v5_no_data(str(d["error"])))
                continue
            parts.append(
                f'{_esc(d.get("run"))}: 实测步数 '
                f'<span class="mono">{_esc(d.get("n_steps"))}</span>'
                f'/{_esc(d.get("total_steps") or "?")}'
                + (f" · lr {_esc(d.get('lr_first'))}→{_esc(d.get('lr_last'))}"
                   if d.get("lr_first") is not None else "")
                + (f" · step_s p50 {_esc(d.get('step_s_p50'))}"
                   if d.get("step_s_p50") is not None else ""))
        pairs.append(("实际步数（逐 step 指标）", "<br>".join(parts),
                      "metrics.jsonl: kind=train, total_steps(task)"))
    else:
        pairs.append(("实际步数（逐 step 指标）",
                      _v5_no_data("没有 metrics.jsonl（逐 step 记录）"),
                      "metrics.jsonl: kind=train"))
    # train vs dev loss：只取 train_hist.json 里**记录过**的点，末值+epoch
    series = [se for se in (t13.get("series") or [])
              if se.metric == "train_loss" or str(se.metric).startswith("val")]
    if series:
        shown = []
        for se in series[:8]:
            got = [(x, v) for x, v in se.points if v is not None]
            if not got:
                shown.append(f'{_esc(se.name)} · {_esc(se.metric)}: '
                             + _v5_no_data("该序列没有测到的点"))
                continue
            x, v = got[-1]
            shown.append(
                f'{_esc(se.name)} · {_esc(se.metric)} = '
                + _v5_num(v, level=se.level,
                          source=f"{se.source}: {se.metric}[epoch {int(x)}]"))
        pairs.append(("train vs dev loss（记录末值）", "<br>".join(shown),
                      "train_hist.json: train_loss / val_*"))
    else:
        pairs.append(("train vs dev loss（记录末值）",
                      _v5_no_data("没有训练历史（train_hist.json）"),
                      "train_hist.json: train_loss / val_*"))
    ntf = blob.get("n_train_frames_by_arm")
    steps = blob.get("steps_by_arm")
    if isinstance(ntf, dict) or isinstance(steps, dict):
        parts = []
        for arm in ("baseline", "candidate"):
            n_per = (ntf or {}).get(arm) or {}
            s_per = (steps or {}).get(arm) or {}
            if not isinstance(n_per, dict) and not isinstance(s_per, dict):
                continue
            seeds_ = sorted({*n_per, *s_per}, key=lambda x: str(x))
            cells = []
            for sd in seeds_:
                nv, sv = n_per.get(sd), s_per.get(sd)
                cells.append(
                    f"seed {sd}: 样本 "
                    + (_v5_int(nv) if nv is not None else
                       _v5_no_data("该 seed 没有 n_train"))
                    + " · 步数 "
                    + (_v5_int(sv) if sv is not None else
                       _v5_no_data("该 seed 没有步数记录")))
            parts.append(f"{arm}: " + ("；".join(cells) if cells else
                                       _v5_no_data("没有该臂记录")))
        pairs.append(("实际样本与步数（两臂）", "<br>".join(parts),
                      f"{fname}: n_train_frames_by_arm / steps_by_arm"))
    else:
        pairs.append(("实际样本与步数（两臂）",
                      _v5_no_data("判定文件没有 n_train_frames_by_arm / "
                                  "steps_by_arm", legacy=legacy),
                      f"{fname}: n_train_frames_by_arm / steps_by_arm"))
    # 配方激活情况：分"生效/未生效/未采用"三态，命令行列了不等于生效
    act = []
    if "factor" in blob:
        act.append("因子: " + _esc(json.dumps(blob.get("factor"),
                                              ensure_ascii=False)))
    flags = blob.get("applied_flags")
    if isinstance(flags, list):
        act.append("生效标记: " + (_esc("、".join(map(str, flags)))
                                 if flags else "无（applied_flags 为空）"))
    elif flags is not None:
        act.append("生效标记: " + _esc(str(flags)))
    if blob.get("skipped_factors") is not None:
        act.append("未采用: " + _esc(json.dumps(blob.get("skipped_factors"),
                                                ensure_ascii=False)))
    if blob.get("data_factor_note") is not None:
        act.append("下轮输入: " + _esc(json.dumps(
            blob.get("data_factor_note"), ensure_ascii=False)))
    pairs.append(("配方激活情况",
                  "<br>".join(act) if act else
                  _v5_no_data("判定文件没有 factor/applied_flags",
                              legacy=legacy,
                              source=f"{fname}: factor, applied_flags"),
                  f"{fname}: factor / applied_flags / skipped_factors / "
                  "data_factor_note"))
    # checkpoint 变化：只做文件系统证据（不改权重、不重算）
    ck_txt = []
    if root is not None:
        cands = (sorted(root.glob("*/seed*/checkpoint_last.pt"))
                 + sorted(root.glob("seed*/checkpoint_last.pt"))
                 + sorted(root.glob("checkpoint_last.pt")))
        for ck in cands[:8]:
            try:
                stt = ck.stat()
            except OSError as exc:
                ck_txt.append(f'{_esc(ck.name)}: '
                              + _v5_no_data(f"stat 失败 {exc}"))
                continue
            mtime = time.strftime("%Y-%m-%d %H:%M:%S",
                                  time.localtime(stt.st_mtime))
            ck_txt.append(
                f'<span class="mono">{_esc(str(ck.relative_to(root)))}</span>'
                f' · {stt.st_size} B · mtime {_esc(mtime)}')
        if not cands:
            ck_txt.append(_v5_no_data("运行目录里没有 checkpoint_last.pt"))
    else:
        ck_txt.append(_v5_no_data("没有 run 目录，无法列出 checkpoint"))
    ck_txt.append("评测口径: checkpoint = "
                  + _esc(str(blob.get("eval_checkpoint") or "未记录"))
                  + "（判定文件 eval_checkpoint）")
    pairs.append(("checkpoint 变化", "<br>".join(ck_txt),
                  f"文件系统: <run>/*/seed*/checkpoint_last.pt / "
                  f"{fname}: eval_checkpoint"))
    out.append(_v5_kv_table(pairs))
    return "".join(out)


def _v5_compare_block(ctx: dict, it: dict) -> str:
    """面板 4：比较结果（同 seed 两臂、micro/macro/最差场景、C/R/M/L/A）。"""
    from beamng_autopilot.experiments import candidate_metrics as cm
    blob = it.get("blob") or {}
    fname = str(it.get("file") or "decision_*.json")
    legacy = bool(it.get("legacy_note"))
    out = ["<h3>比较结果</h3>"]
    # ---- 同 seed 两臂：champ_by_seed/cand_by_seed 直读；否则 pairings ----
    champ_by = blob.get("champ_by_seed") if isinstance(
        blob.get("champ_by_seed"), dict) else {}
    cand_by = blob.get("cand_by_seed") if isinstance(
        blob.get("cand_by_seed"), dict) else {}
    shown = False
    # 形状一：{seed: 标量}（单主指标，road-only 的判定就是这样写的）
    seeds = sorted(set(champ_by) | set(cand_by), key=lambda x: str(x))
    if seeds and all(isinstance(champ_by.get(s, cand_by.get(s)), (int, float))
                     or champ_by.get(s) is None or cand_by.get(s) is None
                     for s in seeds):
        metric = str(blob.get("metric") or "主指标")
        rows = []
        for s in seeds:
            bv, cv = champ_by.get(s), cand_by.get(s)
            delta = (float(cv) - float(bv)
                     if isinstance(bv, (int, float))
                     and isinstance(cv, (int, float)) else None)
            rows.append(
                f'<tr><td class="mono">seed {_esc(s)}</td>'
                f'<td>{_v5_num(bv, level=LEVEL_DEV, reason="该 seed 未记录", source=f"{fname}: champ_by_seed.{s}")}</td>'
                f'<td>{_v5_num(cv, level=LEVEL_DEV, reason="该 seed 未记录", source=f"{fname}: cand_by_seed.{s}")}</td>'
                f'<td>{_v5_num(delta, level=LEVEL_DEV, reason="缺一侧，无法算差值", source=f"{fname}: cand-champ")}</td></tr>')
        out.append(f'<h4>同 seed 两臂 · {_esc(metric)}</h4>')
        out.append('<table><tr><th>seed</th><th>champion</th><th>candidate'
                   "</th><th>Δ（candidate−champion）</th></tr>"
                   + "".join(rows) + "</table>")
        shown = True
    # 形状二：{seed: {指标: 值}}
    if not shown:
        for seed in sorted(champ_by, key=lambda x: str(x)):
            c = champ_by.get(seed) or {}
            d = cand_by.get(seed) or {}
            if not isinstance(c, dict) or not isinstance(d, dict):
                continue
            keys = sorted({k for k in (*c, *d)
                           if isinstance(c.get(k, d.get(k)), (int, float))
                           or isinstance(d.get(k, c.get(k)), (int, float))})
            if not keys:
                continue
            rows = []
            for k in keys[:12]:
                bv, cv = c.get(k), d.get(k)
                delta = (float(cv) - float(bv)
                         if isinstance(bv, (int, float))
                         and isinstance(cv, (int, float)) else None)
                rows.append(
                    f'<tr><td class="mono">{_esc(k)}</td>'
                    f'<td>{_v5_num(bv, level=LEVEL_DEV, reason="该 seed 未记录", source=f"{fname}: champ_by_seed.{seed}.{k}")}</td>'
                    f'<td>{_v5_num(cv, level=LEVEL_DEV, reason="该 seed 未记录", source=f"{fname}: cand_by_seed.{seed}.{k}")}</td>'
                    f'<td>{_v5_num(delta, level=LEVEL_DEV, reason="缺一侧无法算差值", source=f"{fname}: cand-champ")}</td></tr>')
            out.append(f'<h4>同 seed 两臂 · seed {_esc(seed)}</h4>')
            out.append('<table><tr><th>指标</th><th>champion</th>'
                       "<th>candidate</th><th>Δ（candidate−champion）</th>"
                       "</tr>" + "".join(rows) + "</table>")
            shown = True
    if not shown:
        pairs = blob.get("pairings") if isinstance(blob.get("pairings"),
                                                   dict) else {}
        if pairs:
            rows = []
            for name in sorted(pairs):
                sp = pairs[name] if isinstance(pairs[name], dict) else {}
                rows.append(
                    f'<tr><td>{_esc(name)}</td>'
                    f'<td class="mono">{_esc(_fmt_list(sp.get("champion")))}</td>'
                    f'<td class="mono">{_esc(_fmt_list(sp.get("candidate")))}</td>'
                    f'<td class="mono">{_esc(_fmt_num(sp.get("mean_delta")))}</td>'
                    f'<td>{_esc(str(sp.get("verdict") or "未测"))}</td></tr>')
            out.append('<table><tr><th>指标</th><th>champion 每 seed</th>'
                       "<th>candidate 每 seed</th><th>mean Δ</th><th>verdict"
                       "</th></tr>" + "".join(rows) + "</table>")
            out.append('<p class="hint">来源: ' + _esc(fname)
                       + ": pairings.<metric>.champion/candidate/mean_delta"
                       "</p>")
        else:
            out.append("<p>" + _v5_no_data(
                "判定文件没有 champ_by_seed/cand_by_seed，也没有 pairings",
                legacy=legacy, source=f"{fname}: pairings") + "</p>")
    # ---- C/R/M/L/A 计数（micro 整数总计 + 分母；比率只在有记录时印） ----
    counts = blob.get("counts")
    if isinstance(counts, dict) and any(counts.get(k) is not None
                                        for k in cm.COUNTERS):
        from beamng_autopilot.experiments.protocol import CANDIDATE_COUNTING
        meanings = CANDIDATE_COUNTING.get("counters") or {}
        out.append("<h4>C/R/M/L/A 计数（micro 整数总计，先累加整数再算比率）"
                   "</h4>")
        rows = []
        for k in cm.COUNTERS:
            v = counts.get(k)
            meaning = meanings.get(k, "")
            rows.append(
                f'<tr><td class="mono">{_esc(k)}</td>'
                f'<td>{_v5_int(v, reason="该计数未记录", source=f"{fname}: counts.{k}")}</td>'
                f'<td class="hint">{_esc(meaning)}</td></tr>')
        out.append('<table><tr><th>计数</th><th>值</th><th>含义</th></tr>'
                   + "".join(rows) + "</table>")
        ratios = blob.get("ratios") if isinstance(blob.get("ratios"), dict) \
            else None
        if ratios is None:
            ratios = cm.ratios(counts)   # 计数契约的唯一实现（不是第二套口径）
            rsrc = f"{fname}: counts（按 candidate_metrics.ratios 口径）"
        else:
            rsrc = f"{fname}: ratios"
        rrows = []
        for name, (num, den) in cm.RATIO_SPECS.items():
            n = ratios.get(f"{name}_numerator", counts.get(num))
            dd = ratios.get(f"{name}_denominator", counts.get(den))
            v = ratios.get(name)
            if v is None:
                cell = _v5_no_data(
                    str(ratios.get(f"{name}_missing") or
                        (f"分母 {den}=0：没有可判的样本，不是 0 分")),
                    source=rsrc)
            else:
                cell = _v5_pct(v, numerator=n, denominator=dd,
                               level=LEVEL_DEV, source=rsrc)
            rrows.append(f'<tr><td>{_esc(name)}</td><td>{cell}</td>'
                         f'<td class="hint">{_esc(num)}/{_esc(den)}</td></tr>')
        out.append('<table><tr><th>比率</th><th>值（分子/分母）</th>'
                   "<th>定义</th></tr>" + "".join(rrows) + "</table>")
        out.append('<p class="hint">micro = 场景计数的整数汇总（本轮主口径）；'
                   "每场景一个单位的 macro 只能与单位一起读"
                   "（candidate_metrics.micro_macro）。</p>")
    else:
        out.append("<p>" + _v5_no_data(
            "判定文件没有 v5 整数计数 counts（旧协议记录不能按新分母重判，"
            "也不补 0）", legacy=legacy, source=f"{fname}: counts") + "</p>")
    # ---- micro/macro：判定字段优先；否则用计数契约的聚合函数 ----
    micro, macro = blob.get("micro"), blob.get("macro")
    if isinstance(micro, dict) or isinstance(macro, dict):
        mm = {"micro": micro or {}, "macro": macro or {}}
        mm_src = f"{fname}: micro / macro"
    elif isinstance(blob.get("scene_counts"), dict) and blob.get("scene_counts"):
        mm = cm.micro_macro(blob["scene_counts"])
        mm_src = f"{fname}: scene_counts（candidate_metrics.micro_macro）"
    else:
        mm = None
        mm_src = f"{fname}: scene_counts"
    if mm is None:
        out.append("<p>" + _v5_no_data("判定文件没有 micro/macro 或 scene_counts",
                                       legacy=legacy, source=mm_src) + "</p>")
    else:
        rows = []
        for name, (num, den) in cm.RATIO_SPECS.items():
            mv = (mm.get("micro") or {}).get(name)
            rows.append(
                f'<tr><td>micro {_esc(name)}</td>'
                + "<td>" + (_v5_pct(mv, level=LEVEL_DEV,
                                    numerator=(mm.get("micro") or {}).get(
                                        f"{name}_numerator"),
                                    denominator=(mm.get("micro") or {}).get(
                                        f"{name}_denominator"),
                                    reason="micro 分母为 0，无法给比率",
                                    source=mm_src)
                           if mv is not None else
                           _v5_no_data("micro 没有该比率（分母为 0 或未记录）",
                                       source=mm_src)) + "</td></tr>")
            ma = (mm.get("macro") or {}).get(name)
            units = (mm.get("macro") or {}).get(f"{name}_units")
            rows.append(
                f'<tr><td>macro {_esc(name)}<span class="hint">（每场景一个'
                "单位）</span></td>"
                + "<td>" + (_v5_pct(ma, level=LEVEL_DEV,
                                    reason="没有可平均的场景",
                                    source=mm_src)
                            + (f' <span class="hint">n_units='
                               f"{_esc(units)}</span>"
                               if units is not None else "")
                            if ma is not None else
                            _v5_no_data("macro 没有该比率（没有任何场景有分母）",
                                        source=mm_src)) + "</td></tr>")
        out.append(_v5_rows_table(rows, headers=("聚合", "值")))
        out.append(f'<p class="hint">{_esc(str(mm.get("macro_note") or ""))}'
                   f' {_src_note(mm_src)}</p>')
    # ---- 最差场景：坏场景不能被池化均值藏掉 ----
    worst = blob.get("worst_scene")
    if isinstance(worst, dict) and (worst.get("scene") or worst.get("name")):
        out.append("<h4>最差场景（判定文件记录）</h4>")
        out.append(f'<p><span class="mono">{_esc(worst.get("scene") or worst.get("name"))}'
                   f'</span> · ' + _esc(json.dumps(worst, ensure_ascii=False))
                   + f' {_src_note(f"{fname}: worst_scene")}</p>')
    else:
        per_scene = blob.get("per_scene") if isinstance(blob.get("per_scene"),
                                                       dict) else {}
        got = [(g, (m or {}).get("line_recall")) for g, m in per_scene.items()
               if isinstance(m, dict) and m.get("line_recall") is not None]
        if got:
            g, v = min(got, key=lambda kv: float(kv[1]))
            out.append("<h4>最差场景（按已记录的 line_recall 取最小）</h4>")
            out.append(
                '<p><span class="mono">' + _esc(str(g)) + "</span> · "
                + _v5_pct(v, level=LEVEL_DEV,
                          source=f"{fname}: per_scene.{g}.line_recall")
                + '<span class="hint">（规则：只在有 line_recall 记录的场景里'
                  "取最小；未测/不适用场景不参与，也不会被当成 0。若某场景"
                  "其实未测但旧记录写了 0，那属于旧记录问题，本页不替它改写）"
                  "</span></p>")
        else:
            out.append("<p>" + _v5_no_data(
                "没有 worst_scene，也没有任何场景记录了 line_recall",
                legacy=legacy, source=f"{fname}: per_scene") + "</p>")
    # ---- 逐场景适用性：UNKNOWN 与 not_applicable 必须分开 ----
    app = blob.get("applicability")
    if not isinstance(app, dict):
        app = blob.get("scene_applicability")
    scene_counts = blob.get("scene_counts") if isinstance(
        blob.get("scene_counts"), dict) else {}
    merged = {}
    if isinstance(app, dict):
        merged.update(app)
    for g, c in scene_counts.items():
        if isinstance(c, dict) and (c.get("applicability") or c.get("status")):
            merged.setdefault(g, c.get("applicability") or c.get("status"))
    if merged or scene_counts:
        out.append("<h4>逐场景适用性（未测与不适用分开显示）</h4>")
        rows = []
        for g in sorted(set(merged) | set(scene_counts), key=str):
            st = merged.get(g)
            why = ""
            if isinstance(st, dict):
                why = str(st.get("why") or st.get("reason") or "")
                st = st.get("status") or st.get("applicability")
            st = str(st or "")
            if not st:
                cell = _v5_no_data("该场景没有适用性记录（不是实测通过）",
                                   source=f"{fname}: applicability.{g}")
            else:
                label = APPLICABILITY_LABELS.get(st, st)
                cls = {"measured": "ok", "unknown": "miss",
                       "not_applicable": "hint",
                       "unverified_labels": "unknown"}.get(st, "")
                cell = (f'<span class="{cls}">{_esc(label)}</span>'
                        + (f'<div class="hint">{_esc(why)}</div>' if why
                           else ""))
            cc = scene_counts.get(g) if isinstance(scene_counts.get(g),
                                                   dict) else {}
            counts_txt = "、".join(
                f"{k}={_esc(cc.get(k))}" for k in cm.COUNTERS
                if cc.get(k) is not None) or _v5_no_data("该场景没有计数")
            rows.append(f'<tr><td class="mono">{_esc(g)}</td><td>{cell}</td>'
                        f'<td class="mono">{counts_txt}</td></tr>')
        out.append('<table><tr><th>场景</th><th>适用性</th>'
                   "<th>计数（C/R/M/L/A…）</th></tr>" + "".join(rows)
                   + "</table>")
        out.append(_src_note(f"{fname}: applicability / scene_counts",
                             "protocol.APPLICABILITY 定义 未测≠不适用≠通过"))
    # ---- 逐场景候选口径：判定文件记录值（None -> 未测，不补 0） ----
    sc = blob.get("scene_candidates")
    if isinstance(sc, dict) and sc:
        out.append("<h4>逐场景候选口径（判定文件记录值）</h4>")
        rows = []
        for g in sorted(sc, key=str):
            m = sc[g] if isinstance(sc[g], dict) else {}

            def _cell(key: str) -> str:
                if m.get(key) is None:
                    return _v5_no_data(
                        "该场景该指标未记录（UNKNOWN），不是 0",
                        source=f"{fname}: scene_candidates.{g}.{key}")
                return _v5_pct(
                    m.get(key), level=LEVEL_DEV,
                    numerator=m.get(f"{key}_numerator"),
                    denominator=m.get(f"{key}_denominator"),
                    source=f"{fname}: scene_candidates.{g}.{key}")

            rows.append(
                f'<tr><td class="mono">{_esc(g)}</td>'
                f'<td>{_cell("candidate_reference_coverage")}</td>'
                f'<td>{_cell("candidate_identity_rate")}</td>'
                f'<td>{_cell("left_right_role_agreement")}</td>'
                f'<td>{_v5_int(m.get("n_candidates"), reason="该场景没有候选数", source=f"{fname}: scene_candidates.{g}.n_candidates")}</td></tr>')
        out.append('<table><tr><th>场景</th><th>候选覆盖率 R/C</th>'
                   "<th>身份率 M/R</th><th>左右角色 A/L</th><th>候选数 C</th>"
                   "</tr>" + "".join(rows) + "</table>")
        out.append(_src_note(f"{fname}: scene_candidates.*"))
    return "".join(out)


def _v5_negative_block(ctx: dict, it: dict) -> str:
    """面板 5：无线诊断（合格/排除帧、严格假线率；诊断，不直接晋级）。"""
    from beamng_autopilot.experiments.protocol import NEGATIVE_DIAGNOSTIC
    blob = it.get("blob") or {}
    fname = str(it.get("file") or "decision_*.json")
    legacy = bool(it.get("legacy_note"))
    out = ["<h3>无线诊断</h3>",
           '<p class="hint">严格假线帧率/像素占比是 <b>诊断，不直接晋级</b>：'
           + _esc(str(NEGATIVE_DIAGNOSTIC.get("gate") or "no threshold")) +
           "；与 offroad_false_ratio 的分母不同（"
           + _esc(str(NEGATIVE_DIAGNOSTIC.get("not_comparable_with") or ""))
           + "）。</p>"]
    entries = []
    nl = blob.get("negative_line")
    if isinstance(nl, dict):
        entries.append((f"{fname}: negative_line", nl))
    ev = ctx.get("eval") or {}
    for split in sorted(ev.get("splits") or {}):
        for name, entry in sorted((ev["splits"][split] or {}).items()):
            n2 = (entry or {}).get("negative_line") if isinstance(entry, dict) \
                else None
            if isinstance(n2, dict):
                entries.append((f"eval matrix: {split}.{name}.negative_line",
                                n2))
    if not entries:
        out.append("<p>" + _v5_no_data(
            "判定文件没有 negative_line 字段，评估矩阵也没有",
            legacy=legacy, source=f"{fname}: negative_line") + "</p>")
        return "".join(out)
    rows = []
    for src, n in entries[:8]:
        status = str(n.get("status") or "")
        elig = n.get("eligible_frames")
        excluded = n.get("excluded_frames")
        # negative_scenes.negative_line_summary 不直接给 excluded_frames：
        # 被排除帧 = frames - eligible_frames（= 正例 + 含未知像素 + 空帧）。
        # 缺任一计数就写无数据，**不补 0**（"一个被排除帧都没有"必须是算出来的）。
        if excluded is None and isinstance(n.get("frames"), (int, float)) \
                and isinstance(elig, (int, float)):
            excluded = int(n["frames"]) - int(elig)
        exc_parts = []
        for k, label in (("unverified_frames", "档位不可信"),
                         ("positive_frames", "正例（有线真值）"),
                         ("unknown_frames", "含未知像素"),
                         ("empty_frames", "空帧")):
            if n.get(k) is not None:
                exc_parts.append(f"{label} {int(n[k])}")
        rows.append(
            f'<tr><td class="mono">{_esc(src)}</td>'
            f'<td>{"实测" if status == "measured" else _esc(status or "未标状态")}</td>'
            + "<td>" + (_v5_int(elig, reason="没有 eligible_frames",
                                source=src)
                        if elig is not None else
                        _v5_no_data("没有合格帧计数", source=src)) + "</td>"
            + "<td>" + (
                (f'<span class="mono">{int(excluded)}</span>'
                 + (f'<div class="hint">{_esc("、".join(exc_parts))}</div>'
                    if exc_parts else ""))
                if excluded is not None else
                (_v5_no_data("没有排除帧计数", source=src))) + "</td>"
            + "<td>" + _v5_pct(
                n.get("false_positive_frame_rate"), numerator=n.get(
                    "false_positive_frames"), denominator=elig,
                level=LEVEL_DEV,
                reason=str(n.get("missing_reason") or
                           "没有合格负例帧，帧率为 UNKNOWN"),
                source=src) + "</td>"
            + "<td>" + _v5_pct(
                n.get("false_positive_pixel_fraction"),
                numerator=n.get("false_positive_px"),
                denominator=n.get("eligible_px"), level=LEVEL_DEV,
                reason=str(n.get("missing_reason") or
                           "没有合格负例像素，像素占比为 UNKNOWN"),
                source=src) + "</td></tr>")
        if n.get("excluded_reason"):
            rows.append(f'<tr><td class="hint" colspan="6">'
                        f'{_esc(str(n.get("excluded_reason")))}</td></tr>')
    out.append('<table><tr><th>来源</th><th>状态</th><th>合格帧</th>'
               "<th>被排除帧</th><th>严格假线帧率<br><span style=\""
               'font-weight:400;font-size:11px;color:#6b7280">'
               "有 ≥1 假线像素的帧 / 合格帧</span></th>"
               "<th>假线像素占比<br><span style=\"font-weight:400;"
               'font-size:11px;color:#6b7280">'
               "假线像素 / 合格帧总像素</span></th></tr>"
               + "".join(rows) + "</table>")
    out.append('<p class="hint">诊断口径来自 protocol.NEGATIVE_DIAGNOSTIC'
               "（本轮不设阈值；没有独立标定前不得当晋级门）。</p>")
    return "".join(out)


def _v5_error_block(ctx: dict, it: dict) -> str:
    """面板 6：错误定位（同帧真值/生产/候选叠图、掩码/候选/投影/角色、原帧）。"""
    blob = it.get("blob") or {}
    fname = str(it.get("file") or "decision_*.json")
    legacy = bool(it.get("legacy_note"))
    out = ["<h3>错误定位</h3>",
           '<p class="hint">本页**不自己画叠图**：只有判定/评估产物给出帧路径与'
           "叠图路径时才显示，并能点回原帧（相对本 HTML 真实存在的文件）。"
           "掩码/候选/投影/角色只渲染记录里有的字段。</p>"]
    out_dir = Path(str(ctx.get("out_path") or ".")).parent
    rows = []
    entries = []
    worst = blob.get("worst_frames_by_seed")
    if isinstance(worst, dict):
        for seed in sorted(worst, key=lambda x: str(x)):
            for f in (worst.get(seed) or [])[:5]:
                if isinstance(f, dict):
                    entries.append((f"seed {seed}",
                                    f"{fname}: worst_frames_by_seed.{seed}[]",
                                    f))
    mc = blob.get("mask_compare")
    if isinstance(mc, dict):
        for f in (mc.get("worst") or [])[:5]:
            if isinstance(f, dict):
                entries.append(("mask_compare",
                                f"{fname}: mask_compare.worst[]", f))
    ev = ctx.get("eval") or {}
    for split in sorted(ev.get("splits") or {}):
        for name, entry in sorted((ev["splits"][split] or {}).items()):
            mc2 = (entry or {}).get("mask_compare") if isinstance(entry, dict) \
                else None
            if isinstance(mc2, dict):
                for f in (mc2.get("worst") or [])[:3]:
                    if isinstance(f, dict):
                        entries.append(
                            (f"eval {split}/{name}",
                             f"eval matrix: {split}.{name}.mask_compare.worst[]",
                             f))
    for scope, src, f in entries[:12]:
        frame = f.get("frame") or f.get("path")
        iou = f.get("iou", f.get("road_iou"))
        iou_cell = _v5_num(iou, unit="", level=LEVEL_DEV, digits=4,
                           reason="该帧没有 IoU 记录", source=src + ".iou")
        rows.append(
            f'<tr><td class="mono">{_esc(scope)}</td>'
            f'<td>{_v5_frame_ref(frame, out_dir)}</td>'
            f'<td>{iou_cell}</td>'
            f'<td>'
            + _v5_int(f.get("gt_px"), reason="没有真值像素记录",
                      source=src + ".gt_px") + "</td>"
            + "<td>" + _v5_field(f, ("mask", "mask_px", "truth_mask"),
                                 src, "掩码") + "</td>"
            + "<td>" + _v5_field(f, ("candidate", "candidates", "candidate_n"),
                                 src, "候选") + "</td>"
            + "<td>" + _v5_field(f, ("projection", "projected", "match"),
                                 src, "投影") + "</td>"
            + "<td>" + _v5_field(f, ("role", "roles", "side"), src, "角色")
            + "</td></tr>")
    if rows:
        out.append('<table><tr><th>范围</th><th>原帧（点击追溯）</th>'
                   "<th>IoU</th><th>真值像素</th><th>掩码</th><th>候选</th>"
                   "<th>投影</th><th>角色</th></tr>" + "".join(rows)
                   + "</table>")
        if blob.get("overlays"):
            ol = blob["overlays"] if isinstance(blob["overlays"], dict) else {}
            items = []
            for frame, kinds in list(ol.items())[:6]:
                if not isinstance(kinds, dict):
                    continue
                links = []
                for kind in ("truth", "production", "candidate"):
                    if kinds.get(kind):
                        links.append(f"{_esc(kind)}: "
                                     + _v5_frame_ref(kinds[kind], out_dir))
                items.append(f'<li><span class="mono">{_esc(frame)}</span>: '
                             + " · ".join(links) + "</li>")
            out.append("<h4>同帧真值/生产/候选叠图</h4><ul class=\"reasons\">"
                       + "".join(items) + "</ul>"
                       + _src_note(f"{fname}: overlays"))
        else:
            out.append("<p>" + _v5_no_data(
                "判定文件没有 overlays（同帧真值/生产/候选叠图路径）",
                legacy=legacy, source=f"{fname}: overlays") + "</p>")
    else:
        out.append("<p>" + _v5_no_data(
            "判定/评估产物都没有 worst_frames_by_seed 或 mask_compare 记录",
            legacy=legacy,
            source=f"{fname}: worst_frames_by_seed / mask_compare") + "</p>")
    # 运行目录里的复核图（文件系统证据，不伪造）
    root = _run_root(ctx)
    if root is not None:
        shots = sorted(root.glob("review_overlays/*.png"))[:6]
        if shots:
            out.append('<p class="hint">运行目录复核图: '
                       + "、".join(_v5_frame_ref(p, out_dir) for p in shots)
                       + f' {_src_note("文件系统: review_overlays/*.png")}</p>')
    return "".join(out)


def _v5_field(node: dict, keys: tuple, src: str, label: str) -> str:
    """错误定位行里的可选字段：记录里有才渲染，没有就写无数据（不编）。"""
    for k in keys:
        if k in node and node.get(k) is not None:
            return (f'<span class="mono">{_esc(node.get(k))}</span>'
                    + _src_note(f"{src}.{k}"))
    return _v5_no_data(f"记录里没有{label}字段")


def _v5_resources_block(ctx: dict, it: dict) -> str:
    """面板 7：资源与判定（GPU/CPU/内存/磁盘、阶段耗时、暂停原因、硬门/研究）。"""
    blob = it.get("blob") or {}
    fname = str(it.get("file") or "decision_*.json")
    # 计时重复/计时可信度是 v5 之后才落盘的字段：旧判定写"旧协议未记录"。
    legacy = bool(it.get("legacy_note"))
    ev = ctx.get("events") or {}
    out = ["<h3>资源与判定</h3>"]
    pairs = []
    res = blob.get("resources") if isinstance(blob.get("resources"), dict) \
        else None
    if res:
        txt = " · ".join(
            f'{_esc(k)}: <span class="mono">{_esc(v)}</span>'
            for k, v in sorted(res.items()) if v is not None)
        pairs.append(("GPU/CPU/内存/磁盘", txt or _v5_no_data("resources 为空"),
                      f"{fname}: resources"))
    else:
        found = []
        for e in (ev.get("events") or []):
            for name, rec in (e.metrics or {}).items():
                if any(k in str(name) for k in ("gpu", "vram", "mem", "disk",
                                                "cpu")):
                    found.append((f"epoch {e.epoch} · {name}",
                                  _metric_evidence(rec, str(name),
                                                   LEVEL_TRAIN)))
        if found:
            pairs.append(("GPU/CPU/内存/磁盘（事件流指标）",
                          "<br>".join(f"{_esc(a)}: {b.cell()}"
                                      for a, b in found[:12]),
                          "events.jsonl: metrics.*"))
        else:
            pairs.append(("GPU/CPU/内存/磁盘",
                          _v5_no_data("判定文件没有 resources，事件流也没有"
                                      "资源指标"),
                          f"{fname}: resources / events.jsonl: metrics"))
    # 阶段耗时
    sd = blob.get("stage_durations")
    if isinstance(sd, dict) and sd:
        txt = " · ".join(f'{_esc(k)}: <span class="mono">{_esc(v)}</span> s'
                         for k, v in sorted(sd.items()))
        pairs.append(("阶段耗时", txt, f"{fname}: stage_durations"))
    else:
        # 训练步骤轴与硬件时间轴分开：这里只给事件流各阶段的墙钟跨度
        by_phase = {}
        for e in (ev.get("events") or []):
            t = _ts_epoch(e.ts)
            if t is None:
                continue
            slot = by_phase.setdefault(e.phase, [t, t])
            slot[0], slot[1] = min(slot[0], t), max(slot[1], t)
        if by_phase:
            txt = " · ".join(
                f'{_esc(k)}: <span class="mono">{max(0.0, v[1] - v[0]):.1f}'
                "</span> s" for k, v in sorted(by_phase.items()))
            pairs.append(("阶段耗时（事件 ts 跨度，派生）", txt,
                          "events.jsonl: ts 各阶段 min→max"))
        else:
            pairs.append(("阶段耗时",
                          _v5_no_data("判定文件没有 stage_durations，"
                                      "事件流也没有可解析 ts"),
                          f"{fname}: stage_durations"))
    # 计时重复与可信度（方案 §S4 资源与判定）：重复测量两次是"这个 p95 值不值
    # 得引用"的证据；timing_suspect 非空时本轮 p95 已被记 UNKNOWN，页面必须跟着
    # 说"不可引用"，不能只显示一个漂亮的毫秒数。
    reps = blob.get("timing_repeats")
    if isinstance(reps, list) and reps:
        parts = []
        for r in reps[:8]:
            if not isinstance(r, dict):
                continue
            parts.append(
                f'seed {_esc(r.get("seed"))}: p50 '
                f'{_esc(_fmt_list(r.get("p50")))} · p95 '
                f'{_esc(_fmt_list(r.get("p95")))}'
                + (f' · 采用 <span class="mono">{_esc(r.get("used"))}</span> ms'
                   if r.get("used") is not None else
                   ' · ' + _v5_no_data("该 seed 没有采用值")))
        pairs.append(("计时重复（p50/p95 各测两次）", "<br>".join(parts),
                      f"{fname}: timing_repeats"))
    else:
        pairs.append(("计时重复（p50/p95 各测两次）",
                      _v5_no_data("判定文件没有 timing_repeats"
                                  "（不确定的计时不能当结论）", legacy=legacy,
                                  source=f"{fname}: timing_repeats"),
                      f"{fname}: timing_repeats"))
    suspect = blob.get("timing_suspect")
    if isinstance(suspect, list) and suspect:
        pairs.append((
            "计时可信度",
            '<span class="miss">本轮计时不可引用：inference_ms_p95 已记 '
            f'UNKNOWN（{len(suspect)} 条原因）</span><br>'
            + "<br>".join(_esc(x) for x in suspect[:6]),
            f"{fname}: timing_suspect"))
    elif isinstance(suspect, list):
        pairs.append(("计时可信度",
                      '<span class="ok">timing_suspect 为空：'
                      "计时前提满足（没有游戏在跑）</span>",
                      f"{fname}: timing_suspect"))
    else:
        pairs.append(("计时可信度",
                      _v5_no_data("判定文件没有 timing_suspect 字段",
                                  legacy=legacy, source=f"{fname}: timing_suspect"),
                      f"{fname}: timing_suspect"))
    # 暂停/失败原因
    stop = []
    if ev.get("readable"):
        for e in (ev.get("events") or []):
            if e.phase in ("paused", "failed") or e.status in ("failed",
                                                               "paused"):
                stop.append(f'{_esc(e.ts)} {_esc(e.phase)}/{_esc(e.status)}: '
                            f'{_esc(e.note or "未写原因")}')
    elif blob.get("pause_reasons") or blob.get("failure_reasons"):
        stop.append(_esc(json.dumps(
            blob.get("pause_reasons") or blob.get("failure_reasons"),
            ensure_ascii=False)))
    if stop:
        pairs.append(("暂停/失败原因", "<br>".join(stop),
                      "events.jsonl: phase/status/note"))
    elif ev.get("readable"):
        pairs.append(("暂停/失败原因",
                      '<span class="ok">事件流里没有 paused/failed 事件'
                      "（这是查过的 0）</span>",
                      "events.jsonl: phase/status"))
    else:
        pairs.append(("暂停/失败原因",
                      _v5_no_data("没有事件流，暂停/失败原因未记录"),
                      "events.jsonl: phase/status/note"))
    out.append(_v5_kv_table(pairs))
    # 硬门 vs 研究结论：两条通道分开显示
    out.append("<h4>硬门（晋级门，缺测不算通过）</h4>")
    hg = blob.get("hard_gate") if isinstance(blob.get("hard_gate"), dict) \
        else {}
    viol = blob.get("hard_gate_violations") or []
    hrows = []
    for k in sorted(hg):
        v = hg.get(k)
        if v is None:
            cell = _v5_no_data("硬门输入未测（UNKNOWN），既不算通过也不算违反",
                               source=f"{fname}: hard_gate.{k}")
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            cell = _v5_num(v, unit=("ms" if "ms" in str(k) else "ratio"),
                           level=LEVEL_DEV, reason="硬门输入未测（UNKNOWN）",
                           source=f"{fname}: hard_gate.{k}")
        else:
            # note 这类说明文字原样显示（不是指标，不能当数）
            cell = (f'<span class="mono">{_esc(v)}</span>'
                    + _src_note(f"{fname}: hard_gate.{k}"))
        hrows.append(f'<tr><td class="mono">{_esc(k)}</td><td>{cell}</td></tr>')
    for v in viol:
        hrows.append('<tr><td class="miss">违反</td><td class="miss">'
                     f'{_esc(v)}</td></tr>')
    miss = [m for m in (blob.get("missing_metrics") or [])
            if "UNKNOWN" in str(m)]
    for m in miss:
        hrows.append(f'<tr><td class="unknown">缺测</td>'
                     f'<td class="unknown">{_esc(m)}</td></tr>')
    if hrows:
        out.append('<table><tr><th>硬门项</th><th>值 / 结论</th></tr>'
                   + "".join(hrows) + "</table>")
    else:
        out.append("<p>" + _v5_no_data("判定文件没有 hard_gate 记录") + "</p>")
    # 逐 seed 硬门输入：坏 seed 不能被跨 seed 均值藏掉（方案 §10.2）
    hbs = blob.get("hard_by_seed")
    if isinstance(hbs, dict) and hbs:
        from beamng_autopilot.experiments.protocol import METRIC_DEFINITIONS
        keys = sorted({k for sd in hbs.values() if isinstance(sd, dict)
                       for k in sd})
        rows = []
        for sd in sorted(hbs, key=str):
            entry = hbs[sd] if isinstance(hbs[sd], dict) else {}
            cells = []
            for k in keys[:8]:
                unit = str((METRIC_DEFINITIONS.get(k) or {}).get("unit") or "")
                if entry.get(k) is None:
                    cells.append("<td>" + _v5_no_data(
                        "该 seed 未测（不是 0）",
                        source=f"{fname}: hard_by_seed.{sd}.{k}") + "</td>")
                elif unit == "ratio":
                    cells.append("<td>" + _v5_pct(
                        entry.get(k), level=LEVEL_DEV,
                        source=f"{fname}: hard_by_seed.{sd}.{k}") + "</td>")
                else:
                    cells.append("<td>" + _v5_num(
                        entry.get(k), unit=("ms" if unit == "ms" else ""),
                        level=LEVEL_DEV,
                        source=f"{fname}: hard_by_seed.{sd}.{k}") + "</td>")
            rows.append(f'<tr><td class="mono">seed {_esc(sd)}</td>'
                        + "".join(cells) + "</tr>")
        out.append('<h5>逐 seed 硬门输入（每个 seed 自己过门）</h5>'
                   '<table><tr><th>seed</th>'
                   + "".join(f"<th>{_esc(k)}</th>" for k in keys[:8])
                   + "</tr>" + "".join(rows) + "</table>")
        out.append(_src_note(f"{fname}: hard_by_seed.<seed>.<metric>"))
    out.append("<h4>研究结论（不直接晋级）</h4>")
    dec = blob.get("decision") if isinstance(blob.get("decision"), dict) else {}
    rrows = []
    rrows.append(f'<tr><td>判定</td><td><span class="pill '
                 f'{_esc(str(dec.get("decision") or ""))}">'
                 f'{_esc(str(dec.get("decision") or "未写"))}</span></td></tr>')
    for r in (dec.get("reasons") or []):
        rrows.append(f'<tr><td>理由</td><td>{_esc(r)}</td></tr>')
    if blob.get("research_only") is not None:
        rrows.append(f'<tr><td>research_only</td><td>'
                     f'{_esc(str(blob.get("research_only")))}</td></tr>')
    if blob.get("r2_confirmed") is not None:
        rrows.append(f'<tr><td>最终集 R2 确认</td><td>'
                     f'{_esc(str(blob.get("r2_confirmed")))}'
                     '<span class="hint">（研究结论不等于 R2 已确认）</span>'
                     "</td></tr>")
    out.append('<table><tr><th>项</th><th>内容</th></tr>' + "".join(rrows)
               + "</table>")
    out.append(_src_note(f"{fname}: hard_gate, hard_gate_violations, "
                         "missing_metrics, decision, research_only, "
                         "r2_confirmed"))
    return "".join(out)


def _v5_view(ctx: dict) -> str:
    """方案 v2 §S4 / T16：七个面板（只读判定 JSON 与既有产物）。

    纪律：不重算指标、不另建驾驶控制协议；每个值都标来源字段；缺字段写
    「无数据」，v5 计数契约之前的旧判定写「旧协议未记录」，绝不补 0；
    坏 seed/坏场景、UNKNOWN 与 not_applicable 都必须看得见。
    """
    out = ['<section id="v5"><h2>方案 v2 判定看板（T16 七面板）</h2>',
           '<p class="hint">七个面板直读判定与既有只读产物（decision_*.json、'
           "评估矩阵、准入门产物、事件流、checkpoint 元数据）：本页不重算指标、"
           "不另建驾驶控制协议。缺字段写「无数据」，v5 计数契约之前的记录写"
           "「旧协议未记录」，不补 0。</p>"]
    st = ctx.get("decisions") or {}
    items = [i for i in (st.get("items") or []) if not i.get("error")]
    if not st.get("readable") or not items:
        reason = st.get("error") or "没有 decision_*.json"
        out.append("<p>" + _v5_no_data(f"判定文件不可读：{reason}") + "</p>")
        for panel in V5_PANELS:
            out.append(f'<h3>{_esc(panel)}</h3><p>'
                       + _v5_no_data("没有判定文件，本面板不展示任何数字")
                       + "</p>")
        out.append("</section>")
        return "".join(out)
    rounds = _v5_rounds_dataset(ctx)
    for it in items:
        out.append('<p class="mono">判定文件: '
                   f'<b>{_esc(str(it.get("file") or ""))}</b> · 候选 '
                   f'{_esc(str(it.get("candidate_id") or "未写"))}'
                   + (f' · <span class="miss">{LEGACY_RECORD}</span>'
                      '<span class="hint">：本文件没有 v5 整数计数，'
                      "新口径字段无法重判</span>" if it.get("legacy_note")
                      else "")
                   + "</p>")
        if rounds.get("readable"):
            out.append(f'<p class="hint">数据准入门产物: '
                       f'{_esc(str(rounds.get("path")))}</p>')
        out.append(_v5_identity_block(ctx, it))
        out.append(_v5_data_block(ctx, it, rounds))
        out.append(_v5_learning_block(ctx, it))
        out.append(_v5_compare_block(ctx, it))
        out.append(_v5_negative_block(ctx, it))
        out.append(_v5_error_block(ctx, it))
        out.append(_v5_resources_block(ctx, it))
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


def _run_root(ctx: dict):
    """本次渲染对应的运行目录（来自 --run-dir；没有就返回 None）。"""
    st = ctx.get("decisions") or {}
    p = Path(str(st.get("path") or ""))
    return p if str(p) and p.exists() else None


def _ckpt_params(run_root: Path, *, limit: int = 8) -> list:
    """逐 (arm, seed) 读 checkpoint 里的训练超参（只读、weights_only）。"""
    rows = []
    if run_root is None:
        return rows
    try:
        import torch
    except Exception:                                     # noqa: BLE001
        return rows
    cks = sorted(run_root.glob("*/seed*/checkpoint_last.pt"))
    if not cks:
        cks = sorted(run_root.glob("seed*/checkpoint_last.pt"))
    if not cks:
        # 单臂、直接写在运行目录根下的 checkpoint（实测有这种布局）
        cks = sorted(run_root.glob("checkpoint_last.pt"))
    for ck in cks[:limit]:
        try:
            blob = torch.load(str(ck), map_location="cpu", weights_only=True)
        except Exception as exc:                          # noqa: BLE001
            rows.append({"arm": ck.parent.parent.name or ck.parent.name,
                         "seed": ck.parent.name, "error":
                         f"{type(exc).__name__}: {exc}"})
            continue
        ta = dict(blob.get("train_args") or {})
        rows.append({
            "arm": ck.parent.parent.name if ck.parent.parent != run_root
                   else run_root.name,
            "seed": ck.parent.name, "train_args": ta,
            "dataset_id": blob.get("dataset_id"),
            "git_commit": str(blob.get("git_commit") or "")[:8],
            "env": dict(blob.get("env") or {}),
            # 只是"有没有"，不能 bool() 一个含张量的结构（实测踩到
            # RuntimeError: Boolean value of Tensor with more than one value）
            "has_rng": (blob.get("rng_state") is not None
                        or blob.get("torch_rng") is not None),
        })
    return rows


def _train_dynamics(run_root: Path, *, limit: int = 8) -> list:
    """逐 run 读 training metrics：lr 轨迹、梯度范数、吞吐（只读 jsonl）。"""
    out = []
    if run_root is None:
        return out
    cands = sorted((run_root.parent).glob(f"{run_root.name}-*/metrics.jsonl"))
    if not cands:
        cands = sorted(run_root.glob("*/metrics.jsonl"))
    for fp in cands[:limit]:
        steps, hdr = [], {}
        try:
            for line in fp.read_text(encoding="utf-8",
                                     errors="replace").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("kind") == "task":
                    # 一个 run 可能有多条 task 记录（启动/收尾）：合并，非空优先
                    for k, v in rec.items():
                        if v is not None and (k not in hdr or hdr[k] is None):
                            hdr[k] = v
                elif rec.get("kind") == "train":
                    steps.append(rec)
        except Exception as exc:                          # noqa: BLE001
            out.append({"run": fp.parent.name,
                        "error": f"{type(exc).__name__}: {exc}"})
            continue
        lrs = [s_["lr"] for s_ in steps if s_.get("lr") is not None]
        gns = [s_["grad_norm"] for s_ in steps
               if s_.get("grad_norm") is not None]
        sts = [s_["step_s"] for s_ in steps if s_.get("step_s") is not None]
        def _pct(vals, q):
            if not vals:
                return None
            v = sorted(float(x) for x in vals)
            i = min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))
            return round(v[i], 4)
        out.append({
            "run": fp.parent.name, "n_steps": len(steps),
            "lr_first": (round(lrs[0], 6) if lrs else None),
            "lr_last": (round(lrs[-1], 6) if lrs else None),
            "grad_p50": _pct(gns, 0.5), "grad_p95": _pct(gns, 0.95),
            "grad_max": (round(max(gns), 4) if gns else None),
            "step_s_p50": _pct(sts, 0.5),
            "steps_per_s": (None if not sts or _pct(sts, 0.5) in (None, 0)
                            else round(1.0 / _pct(sts, 0.5), 3)),
            "batch": hdr.get("batch"), "epochs": hdr.get("epochs"),
            "seed": hdr.get("seed"), "total_steps": hdr.get("total_steps"),
            "amp": hdr.get("amp"),
        })
    return out


def _val_or_missing(v) -> str:
    """把值渲染成字符串；**0 / 0.0 是真实值**，只有 None 与空串才是未测。

    实测踩到：`eta_min=0.0`、`weight_decay=0.0` 被 `or "未测"` 吃成"未测"，
    等于把"余弦退火到 0"这个关键事实从看板上抹掉。
    """
    if v is None or (isinstance(v, str) and not v.strip()):
        return "未测"
    return str(v)


def _dl_params_view(ctx: dict) -> str:
    """训练超参与训练动力学：迭代深度学习要看的那组数。"""
    root = _run_root(ctx)
    out = ['<section id="dlparams"><h2>训练超参与训练动力学</h2>',
           '<p class="hint">每个 (臂, seed) 一行，直接读 checkpoint 的 train_args '
           '与训练的 metrics.jsonl。缺项写「未测」，不猜；老 checkpoint 没有这些字段'
           '时同样写未测（字段是 2026-09-25 起才补记的）。</p>']
    rows = _ckpt_params(root)
    n_all = 0
    if root is not None:
        n_all = len(sorted(root.glob("*/seed*/checkpoint_last.pt"))
                    or sorted(root.glob("seed*/checkpoint_last.pt"))
                    or sorted(root.glob("checkpoint_last.pt")))
    if rows and n_all > len(rows):
        out.append(f'<p class="hint">只显示前 {len(rows)}/{n_all} 个 checkpoint'
                   '（避免一次读太多权重文件）。</p>')
    if not rows:
        out.append('<p class="hint">missing data: 没有可读的 checkpoint'
                   '（用 --run-dir 指定运行目录）。</p>')
    else:
        out.append("<table><tr><th>臂/seed</th><th>arch / 参数量</th>"
                   "<th>输入</th><th>batch × epochs<br><span style=font-weight:400;font-size:11px;color:#6b7280>批大小 × 训练轮数</span></th>"
                   "<th>lr 初值 → 调度器</th><th>优化器<br><span style=font-weight:400;font-size:11px;color:#6b7280>优化器与其超参（Adam: betas / weight decay）</span></th>"
                   "<th>AMP / 确定性</th><th>类别权重（bg/road/line）</th>"
                   "<th>步/epoch</th><th>n_train/n_val</th>"
                   "<th>数据因子<br><span style=font-weight:400;font-size:11px;color:#6b7280>这一轮改动的是哪一项数据/标签口径（line 屏蔽 / 截帧 / 采样权重…）</span></th><th>数据版本 / commit</th>"
                   "<th>设备 / 环境</th></tr>")
        for r in rows:
            head = f'<td>{_esc(r["arm"])}/{_esc(r["seed"])}</td>'
            if r.get("error"):
                out.append(f'<tr>{head}<td colspan="12" class="miss">'
                           f'未测：{_esc(r["error"])}</td></tr>')
                continue
            ta = dict(r["train_args"])
            sch = dict(ta.get("scheduler") or {})
            opt = dict(ta.get("optimizer") or {})
            env = dict(r.get("env") or {})
            cw = ta.get("class_weights")
            factor = []
            if ta.get("ignore_line_class"):
                factor.append("line 通道屏蔽(paint_source="
                              f"{ta.get('paint_source')}, "
                              f"{ta.get('line_ignored_frames')} 帧)")
            if ta.get("max_train_frames"):
                factor.append(f"截帧 {ta['max_train_frames']}")
            if ta.get("run_weights"):
                factor.append(f"run 权重 {ta['run_weights']}")
            if not factor:
                factor.append("无（等权、全通道）")
            cells = [
                f'<td>{_esc(str(ta.get("arch") or "未测"))} / '
                f'{_esc(str(ta.get("n_params") or "未测"))}</td>',
                f'<td>{_esc(str(ta.get("input_size") or "未测"))}</td>',
                f'<td>{_esc(str(ta.get("batch") or "未测"))} × '
                f'{_esc(str(ta.get("epochs") or "未测"))}</td>',
                f'<td>{_esc(str(ta.get("lr") or "未测"))} → '
                f'{_esc(str(sch.get("name") or "未测"))}'
                f'(T_max={_esc(_val_or_missing(sch.get("T_max")))}, '
                f'eta_min={_esc(_val_or_missing(sch.get("eta_min")))})</td>',
                f'<td>{_esc(str(opt.get("name") or "未测"))} '
                f'betas={_esc(str(opt.get("betas") or "未测"))} '
                f'wd={_esc(_val_or_missing(opt.get("weight_decay")))}</td>',
                f'<td>amp={_esc(str(ta.get("amp")))} / '
                f'det={_esc(str(ta.get("deterministic")))}</td>',
                f'<td>{_esc(str(cw) if cw else "未测")}</td>',
                f'<td>{_esc(str(ta.get("steps_per_epoch") or "未测"))}</td>',
                f'<td>{_esc(str(ta.get("n_train") or "未测"))}/'
                f'{_esc(str(ta.get("n_val") or "未测"))}</td>',
                f'<td>{_esc("；".join(factor))}</td>',
                f'<td>{_esc(str(r.get("dataset_id") or "未测"))} / '
                f'{_esc(str(r.get("git_commit") or "未测"))}</td>',
                f'<td>{_esc(str(env.get("device") or ta.get("device_name") or "未测"))}'
                f' · torch {_esc(str(env.get("torch") or "未测"))}'
                f' · cuda {_esc(str(env.get("cuda") or "未测"))}</td>',
            ]
            out.append("<tr>" + head + "".join(cells) + "</tr>")
        out.append("</table>")
        out.append('<p class="hint">类别权重来自 median-frequency balancing，'
                   'line 类再乘 --line-weight；权重越小说明该类在训练集里越常见。</p>')
    dyn = _train_dynamics(root)
    if dyn:
        out.append("<h3>训练动力学（逐 step 记录）</h3>"
                   "<table><tr><th>run</th><th>步数<br><span style=font-weight:400;font-size:11px;color:#6b7280>本 run 实际记录的优化步数 / 计划步数</span></th>"
                   "<th>lr 首 → 末</th><th>|grad| p50 / p95 / max</th>"
                   "<th>step 耗时 p50</th><th>steps/s</th>"
                   "<th>batch × epochs × seed</th></tr>")
        for d in dyn:
            if d.get("error"):
                out.append(f'<tr><td>{_esc(d["run"])}</td>'
                           f'<td colspan="6" class="miss">未测：'
                           f'{_esc(d["error"])}</td></tr>')
                continue
            cells = (
                f'<td class="mono">{_esc(d["run"])}</td>'
                f'<td>{_esc(str(d["n_steps"]))}/'
                f'{_esc(str(d.get("total_steps") or "?"))}</td>'
                f'<td>{_esc(str(d["lr_first"]))} → {_esc(str(d["lr_last"]))}</td>'
                f'<td>{_esc(str(d["grad_p50"]))} / {_esc(str(d["grad_p95"]))} / '
                f'{_esc(str(d["grad_max"]))}</td>'
                f'<td>{_esc(str(d["step_s_p50"]))} s</td>'
                f'<td>{_esc(str(d["steps_per_s"]))}</td>'
                f'<td>{_esc(str(d.get("batch")))} × {_esc(str(d.get("epochs")))}'
                f' × {_esc(str(d.get("seed")))}</td>')
            out.append("<tr>" + cells + "</tr>")
        out.append("</table>")
        out.append('<p class="hint">|grad| 是 unscale 之后算的（AMP 下直接统计会随 '
                   'scaler 漂移）；NaN/Inf 一旦出现训练立即失败并留 failed 事件。</p>')
    out.append("</section>")
    return "".join(out)


def _status_items(ctx: dict) -> str:
    """header 里的状态栏：与监控页同款 .item(.k/.v) + 状态 Pill。"""
    root = _run_root(ctx)
    ev = ctx.get("events") or {}
    last = ev.get("last") if ev.get("readable") else None
    dec = (ctx.get("decisions") or {})
    mf = ctx.get("manifest") or {}
    _rn = root.name if root else "未指定"
    items = [('<span class="item"><span class="k">运行</span>'
              f'<span class="v">{_esc(_rn)}'
              + (f"（{_esc(_run_label_cn(_rn))}）" if root is not None else "")
              + "</span></span>")]
    if last is not None:
        items.append('<span class="item"><span class="k">阶段</span>'
                     f'<span class="pill {_esc(last.phase)}">'
                     f'{_esc(last.phase)}/{_esc(last.status)}</span></span>')
    if mf.get("readable"):
        items.append('<span class="item"><span class="k">数据集</span>'
                     f'<span class="v">{_esc(str(mf.get("dataset_id") or "未测")[:16])}'
                     "</span></span>")
    first = ((dec.get("items") or [{}])[0] if dec.get("items") else {})
    if first.get("decision"):
        items.append('<span class="item"><span class="k">判定</span>'
                     f'<span class="pill {_esc(str(first["decision"]))}">'
                     f'{_esc(str(first["decision"]))}</span></span>')
    led = (dec.get("gpu_minutes") or {})
    if led:
        today = sorted(led)[-1]
        items.append('<span class="item"><span class="k">今日 GPU</span>'
                     f'<span class="v">{_esc(str(led[today].get("minutes")))} min'
                     "</span></span>")
    items.append('<span class="item"><span class="k">生成</span>'
                 f'<span class="v">{_esc(ctx.get("generated_at", ""))}</span></span>')
    return '<div class="statusbar">' + "".join(items) + "</div>"


def _toolbar(ctx: dict) -> str:
    return ('<div class="toolbar">'
            '<a class="btn on" href="#">一页卡片</a>'
            '<span>表格可横向滚动；缺测写「未测」，不画 0</span></div>')


def _statusbar(ctx: dict) -> str:
    """顶部状态栏：与监控页同一套观感——一眼看到"这是哪一轮、到哪了、判定如何"。"""
    root = _run_root(ctx)
    ev = ctx.get("events") or {}
    last = ev.get("last") if ev.get("readable") else None
    dec = (ctx.get("decisions") or {})
    items = [("运行", root.name if root else "未指定")]
    if last is not None:
        items.append(("阶段", f"{last.phase}/{last.status}"))
    mf = ctx.get("manifest") or {}
    if mf.get("readable"):
        items.append(("数据集", str(mf.get("dataset_id") or "未测")[:16]))
    led = (dec.get("gpu_minutes") or {})
    if led:
        today = sorted(led)[-1]
        items.append(("今日 GPU", f"{led[today].get('minutes')} min"))
    first = (dec.get("items") or [{}])[0] if dec.get("items") else {}
    if first.get("decision"):
        items.append(("判定", str(first["decision"])))
    items.append(("生成", ctx.get("generated_at", "")))
    cells = []
    for i, (k, v) in enumerate(items):
        if i:
            cells.append('<span class="sep">|</span>')
        cls = ""
        if k == "判定":
            cls = (" badge ok" if v in ("shadow_candidate", "approved_for_review")
                   else " badge bad" if v in ("rejected", "failed")
                   else " badge warn")
        cells.append(f'<span class="k">{_esc(k)}</span>'
                     f'<span class="v{" badge" + cls.split("badge")[1] if cls else ""}">{_esc(str(v))}</span>')
    return '<div class="statusbar">' + " ".join(cells) + "</div>"


def _series_stats(se) -> str:
    """卡片下方的统计行：当前 / 均值 / 峰值 / 中位 / n（与监控页同格式）。"""
    got = [float(v) for _x, v in (se.points if se else []) if v is not None]
    if not got:
        return '<span class="miss">未测</span>'
    cur = got[-1]
    mean = sum(got) / len(got)
    vmax = max(got)
    med = sorted(got)[len(got) // 2]
    def f(v):
        return f"{v:.4g}"
    return (f"当前 {f(cur)} · 均值 {f(mean)} · 峰值 {f(vmax)} · 中位 {f(med)}"
            f" · n={len(got)}"
            "<span style=color:#6b7280;font-size:11px>"
            "（当前=最后一个 epoch；均值/峰值/中位=整段；n=参与统计的 epoch 数）"
            "</span>")


def _card(title: str, sub: str, body: str, *, stats: str = "",
          wide: bool = False) -> str:
    cls = "card wide" if wide else "card"
    return (f'<div class="{cls}"><h3>{_esc(title)}</h3>'
            f'<div class="sub">{_esc(sub)}</div>'
            f'<div class="cv">{body}</div>'
            + (f'<div class="stats">{stats}</div>' if stats else "")
            + "</div>")


def _cards_view(ctx: dict) -> str:
    """main 里的两列卡片网格：每 (seed, 指标) 一张图卡 + 若干表卡。"""
    t13 = ctx.get("t13") or {}
    series = list(t13.get("series") or [])
    cards: list[str] = []
    if t13.get("readable") and series:
        # 先按"指标"分组再按臂/seed 交织，保证两种臂、多个 seed 都能露脸
        # （按名字排序会把 8 张卡全给了 baseline/*）
        want = ("train_loss", "val_miou", "val_line_iou", "val_acc")
        names = sorted({se.name for se in series})
        ordered: list = []
        for metric in want:
            key = "val_miou" if metric == "val_miou" else metric
            for name in names:
                se = next((x for x in series
                           if x.name == name and x.metric == key), None)
                if se is not None:
                    ordered.append(se)
        picked = [se for se in series if se.metric in want]
        shown = 0
        if True:
            for se in ordered:
                if shown >= 8:
                    break
                unit = {"train_loss": "loss 值", "val_miou": "mIoU",
                        "val_line_iou": "line IoU", "val_acc": "acc"}.get(
                            se.metric, se.metric)
                cards.append(_card(
                    f"{se.metric} · {se.name}",
                    f"单位：{unit}　横轴：epoch（{'开发集' if se.metric.startswith('val') else '训练'}）",
                    _svg_series(se, fluid=True),
                    stats=_series_stats(se)))
                shown += 1
        if shown < len(picked):
            cards.append(_card(
                "更多曲线未显示",
                f"共 {len(picked)} 条，本页最多 8 条",
                '<div class="hint">用 --view full 看全部序列（每个指标一张大图）。</div>'))
    else:
        cards.append(_card(
            "训练曲线", "没有训练历史（train_hist.json）",
            f'<div class="hint">missing data：{_esc(str(t13.get("error") or "无"))}'
            "。缺训练历史时不画任何数。</div>"))
    # 表类卡片：复用既有分区的表格（它们本身就是 table 结构）
    for sec in (_decisions_view(ctx), _v5_view(ctx), _dl_params_view(ctx),
                _data_view(ctx), _resources_view(ctx), _final_view(ctx),
                _paths_view(ctx)):
        cards.append(_card_wide_from_section(sec))
    return "<main>" + "".join(cards) + "</main>"


#: 运行 ID 前缀 → 人读标签（显示用；来源是 ID 本身，不是重新定义事实）。
_RUN_LABELS = (
    ("entry_accept", "后台入口验收（多轮/淘汰/中断恢复）"),
    ("plateau_arm", "平台期两臂对比（主结论）"),
    ("plateau_meta", "单臂训练（看板字段验证，含逐 step 指标）"),
    ("plateau_live", "单臂训练（给实时监控页看的）"),
    ("dlparams_probe", "超参字段探针（2 epoch 小实验）"),
    ("monitor_e2e", "监控链路自测"),
    ("monitor_check", "监控链路自测"),
    ("roads3", "等步数/早停对照（road-only）"),
    ("realgate", "数据准入门实测（未训练）"),
)


def _run_label_cn(name: str) -> str:
    for key, label in _RUN_LABELS:
        if key in name:
            return label
    return "实验运行"


def _runs_card(ctx: dict) -> str:
    """本次运行 + 历史运行：一个入口页，运行 ID 不再让人猜。"""
    root = _run_root(ctx)
    out_path = Path(str(ctx.get("out_path") or ""))
    here = root.name if root is not None else "未指定"
    rows = [f'<tr><td class="mono">{_esc(here)}</td>'
            f'<td>{_esc(_run_label_cn(here))}</td><td class="ok">本页</td></tr>']
    siblings = []
    if out_path.parent.exists():
        for hp in sorted(out_path.parent.glob("*.html")):
            if hp.name in ("index.html", out_path.name):
                continue
            siblings.append(hp)
    for hp in siblings[:12]:
        rows.append(f'<tr><td class="mono">{_esc(hp.stem)}</td>'
                    f'<td>{_esc(_run_label_cn(hp.stem))}</td>'
                    f'<td><a href="{_esc(hp.name)}">打开</a></td></tr>')
    body = ("<table><tr><th>运行 ID</th><th>是什么</th><th>链接</th></tr>"
            + "".join(rows) + "</table>"
            '<p class="hint">“本页”= 当前这一份；其余是历史运行，点开是同目录的'
            '另一份 HTML。运行 ID 是执行时的目录名，这里只是给它配了人读标签。</p>')
    return ('<div class="card wide"><h3>本次运行与历史运行</h3>'
            '<div class="sub">一个入口页：默认显示最新的那次实验</div>'
            f'<div>{body}</div></div>')


def _card_wide_from_section(html: str) -> str:
    """把既有 `<section><h2>标题</h2>…</section>` 转成一张宽卡（标题/副标题/内容）。"""
    import re as _re
    m = _re.match(r'<section[^>]*><h2>(.*?)</h2>(.*?)</section>$', html, _re.S)
    if not m:
        return f'<div class="card wide">{html}</div>'
    title, body = m.group(1), m.group(2)
    sub = ""
    m2 = _re.match(r'\s*<p class="hint">(.*?)</p>(.*)$', body, _re.S)
    if m2:
        sub, body = m2.group(1), m2.group(2)
    return (f'<div class="card wide"><h3>{title}</h3>'
            + (f'<div class="sub">{sub}</div>' if sub else "")
            + f'<div>{body}</div></div>')


#: 汇总页的读取上限（诚实注明截断，不做无界扫描）。
ALL_MAX_CKPTS = 24
ALL_MAX_OVERLAYS = 12


def _scan_all_runs(runs_root: Path, *, include_all: bool = False) -> dict:
    """有界扫描运行目录：历史/指标/判定/数据版本/账本/checkpoint。"""
    all_ck: list = []
    out = {"root": str(runs_root), "runs": [], "skipped_runs": [],
           "hist": [], "metrics": [], "decisions": [], "datasets": [],
           "gpu": [], "ckpts": [], "n_ckpt_total": 0,
           "collects": [], "include_all": bool(include_all)}
    if not runs_root.exists():
        return out
    for rd in sorted(x for x in runs_root.iterdir() if x.is_dir()):
        hists = sorted(rd.glob("**/train_hist.json"))
        mets = sorted(rd.glob("metrics.jsonl")) + sorted(
            runs_root.glob(rd.name + "-*/metrics.jsonl"))
        decs = sorted(rd.glob("**/decision_*.json"))
        ds = sorted(rd.glob("**/rounds_dataset.json"))
        gpu = sorted(rd.glob("**/gpu_minutes.json"))
        cols = sorted(rd.glob("collect_*.json"))
        cks = sorted(rd.glob("*/seed*/checkpoint_last.pt")) + sorted(
            rd.glob("seed*/checkpoint_last.pt"))
        if not (hists or mets or decs or ds or cks or cols):
            continue
        # 默认只保留"真产出过结论"的运行：有判定或有逐 step 指标。
        # 纯历史训练目录（T13 三臂、t14_rounds1-5、det*_steps、gpu_tol…）默认不进，
        # 否则 150 份历史会被摊成 300 张图（用户实测反馈："为什么这么多曲线"）。
        if not include_all and not (decs or mets or cols):
            out["skipped_runs"].append(rd.name)
            continue
        out["runs"].append(rd.name)
        for h in hists:
            out["hist"].append({"run": rd.name, "path": h,
                                "tag": _rel_tag(h.parent, rd)})
        out["metrics"] += [{"run": rd.name, "path": m} for m in mets]
        out["decisions"] += [{"run": rd.name, "path": d} for d in decs]
        out["datasets"] += [{"run": rd.name, "path": d} for d in ds]
        out["gpu"] += [{"run": rd.name, "path": g} for g in gpu]
        out["collects"] += [{"run": rd.name, "path": c} for c in cols]
        out["n_ckpt_total"] += len(cks)
        all_ck += [(c, rd.name) for c in cks]
    # 超参表读**最新**的 checkpoint，不按目录名字母序：实测踩到——按字母序时
    # "t14_3h_*" 这类老 run 占满名额，字段最全的新 run 一个都没读到，整表看起来
    # 全是"未测"（旧 checkpoint 里根本没有那些字段）。
    all_ck.sort(key=lambda it: it[0].stat().st_mtime, reverse=True)
    for c, run in all_ck[:max(0, ALL_MAX_CKPTS)]:
        out["ckpts"].append({"run": run, "path": c})
    return out


def _rel_tag(d: Path, run_dir: Path) -> str:
    try:
        return str(d.relative_to(run_dir)).replace("\\", "/")
    except Exception:                                     # noqa: BLE001
        return d.name


def _hist_series(path: Path, run: str, tag: str) -> list:
    """读一份 train_hist.json → 系列（与单运行视图同一口径）。"""
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except Exception:                                     # noqa: BLE001
        return []
    epochs = blob.get("epoch") or []
    out = []
    for metric, values in sorted(blob.items()):
        if metric in ("epoch", "lr", "line_ignored_frames") \
                or not isinstance(values, list):
            continue
        pts = [(float(e), (None if i >= len(values) or values[i] is None
                           else float(values[i])))
               for i, e in enumerate(epochs)]
        if not pts:
            continue
        out.append(Series(name=f"{run}/{tag}", metric=str(metric), unit="",
                          level=(LEVEL_DEV if str(metric).startswith("val")
                                 else LEVEL_TRAIN),
                          source=f"{run}/{tag}/train_hist.json", points=pts,
                          missing=[]))
    return out


def _metrics_summary(path: Path) -> dict:
    """读一份 metrics.jsonl → lr/梯度/吞吐摘要（没有的项留 None=未测）。"""
    steps, hdr = [], {}
    try:
        for line in path.read_text(encoding="utf-8",
                                   errors="replace").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("kind") == "task":
                for k, v in rec.items():
                    if v is not None and (k not in hdr or hdr[k] is None):
                        hdr[k] = v
            elif rec.get("kind") == "train":
                steps.append(rec)
    except Exception:                                     # noqa: BLE001
        return {}

    def _pct(vals, q):
        v = sorted(float(x) for x in vals if x is not None)
        if not v:
            return None
        i = min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))
        return round(v[i], 5)
    lrs = [x.get("lr") for x in steps]
    gns = [x.get("grad_norm") for x in steps]
    sts = [x.get("step_s") for x in steps]
    p50 = _pct(sts, 0.5)
    return {"n_steps": len(steps), "total_steps": hdr.get("total_steps"),
            "lr_first": _pct(lrs, 0.0), "lr_last": _pct(lrs, 1.0),
            "grad_p50": _pct(gns, 0.5), "grad_p95": _pct(gns, 0.95),
            "step_s_p50": p50,
            "steps_per_s": (None if not p50 else round(1.0 / p50, 3)),
            "seed": hdr.get("seed"), "batch": hdr.get("batch"),
            "epochs": hdr.get("epochs")}


def _all_ctx(runs_root: Path, *, include_all: bool = False) -> dict:
    scan = _scan_all_runs(runs_root, include_all=include_all)
    series = []
    for h in scan["hist"]:
        series += _hist_series(h["path"], h["run"], h["tag"])
    # 指标总表：每 (run, tag) 一行，末值取自曲线；lr/梯度取自 metrics.jsonl
    rows: dict = {}
    for se in series:
        key = se.name
        row = rows.setdefault(key, {"run": se.name.split("/")[0], "name": se.name})
        got = [(x, v) for x, v in se.points if v is not None]
        if se.metric == "train_loss" and got:
            row["loss_first"], row["loss_last"] = got[0][1], got[-1][1]
        if se.metric.startswith("val") and got:
            if row.get("val_last") is None or se.metric == "val_miou":
                row["val_metric"] = se.metric
                row["val_first"], row["val_last"] = got[0][1], got[-1][1]
    dyn = {}
    for m in scan["metrics"]:
        s_ = _metrics_summary(m["path"])
        if s_:
            dyn[m["run"]] = s_
            # 也挂到同名 run/tag（rounds 的每臂每 seed 指标目录名 = <run>-<arm>-s<seed>）
            for key, row in rows.items():
                if key.startswith(m["run"] + "-") or key.startswith(m["run"] + "/"):
                    row["dyn"] = s_
            rows.setdefault(m["run"], {"run": m["run"], "name": m["run"]})
            rows[m["run"]]["dyn"] = s_
    cols, col_stat = [], {"n": 0, "ok": 0, "rejected": 0, "frames": 0,
                          "paint_frames": 0, "gpu_minutes": 0.0}
    for c in scan["collects"]:
        try:
            b = json.loads(c["path"].read_text(encoding="utf-8"))
        except Exception:                                 # noqa: BLE001
            continue
        b["_run"], b["_path"] = c["run"], str(c["path"])
        cols.append(b)
        col_stat["n"] += 1
        col_stat["ok" if b.get("ok") else "rejected"] += 1
        col_stat["frames"] += int(b.get("frames_total") or 0)
        col_stat["paint_frames"] += sum(
            int(v) for v in (b.get("paint_frames_by_role") or {}).values())
        col_stat["gpu_minutes"] += float(b.get("gpu_minutes") or 0.0)
    return {"scan": scan, "series": series, "rows": rows, "dyn": dyn,
            "collects": cols, "collect_stats": col_stat}


def _all_metrics_table(ctx: dict) -> str:
    a = ctx.get("all") or {}
    rows = a.get("rows") or {}
    if not rows:
        return '<p class="hint">missing data: 没有训练历史。</p>'
    out = ["<table><tr><th>运行 / 臂-seed<br><span style=font-weight:400;font-size:11px;color:#6b7280>运行目录 / 实验臂 + 随机种子；同一 run 的每个臂每个 seed 一行</span></th><th>train loss 首→末<br><span style=font-weight:400;font-size:11px;color:#6b7280>训练集损失，越低越好；首末=第 0 轮与最后一轮</span></th>"
           "<th>开发指标 首→末<br><span style=font-weight:400;font-size:11px;color:#6b7280>验证集指标，越高越好；取该 run 可用的 val_miou/val_line_iou/val_acc</span></th><th>lr 首→末<br><span style=font-weight:400;font-size:11px;color:#6b7280>学习率；余弦退火按 --epochs 收到接近 0（T_max=epochs）</span></th><th>|grad| p50/p95<br><span style=font-weight:400;font-size:11px;color:#6b7280>梯度 L2 范数（AMP 下 unscale 后统计），衡量每步更新幅度</span></th>"
           "<th>步/s<br><span style=font-weight:400;font-size:11px;color:#6b7280>每秒优化步数（吞吐），由逐 step 耗时中位数换算</span></th><th>步数<br><span style=font-weight:400;font-size:11px;color:#6b7280>本 run 实际记录的优化步数 / 计划步数</span></th></tr>"]
    for key in sorted(rows):
        r = rows[key]
        d = r.get("dyn") or {}

        def f(v):
            return "未测" if v is None else _fmt_point(float(v))
        loss = (f"{f(r.get('loss_first'))} → {f(r.get('loss_last'))}"
                if r.get("loss_first") is not None else "未测")
        val = (f"{r.get('val_metric') or 'val'} {f(r.get('val_first'))} → "
               f"{f(r.get('val_last'))}" if r.get("val_last") is not None
               else "未测")
        lr = (f"{d.get('lr_first')} → {d.get('lr_last')}"
              if d.get("lr_first") is not None else "未测")
        gn = (f"{f(d.get('grad_p50'))} / {f(d.get('grad_p95'))}"
              if d.get("grad_p50") is not None else "未测")
        out.append(f'<tr><td class="mono">{_esc(key)}</td><td class="mono">{loss}</td>'
                   f'<td class="mono">{val}</td><td class="mono">{lr}</td>'
                   f'<td class="mono">{gn}</td>'
                   f'<td class="mono">{d.get("steps_per_s") or "未测"}</td>'
                   f'<td class="mono">{d.get("n_steps") or "未测"}</td></tr>')
    out.append("</table>")
    return "".join(out)


def _all_decisions_table(ctx: dict) -> str:
    a = ctx.get("all") or {}
    decs = (a.get("scan") or {}).get("decisions") or []
    if not decs:
        return '<p class="hint">missing data: 没有判定文件。</p>'
    out = ["<table><tr><th>运行<br><span style=font-weight:400;font-size:11px;color:#6b7280>运行目录名：一次后台入口 / 一轮实验的产物都在这个目录里</span></th><th>候选<br><span style=font-weight:400;font-size:11px;color:#6b7280>候选 ID：提议的因子 + 轮次</span></th><th>指标<br><span style=font-weight:400;font-size:11px;color:#6b7280>成对比较用的主指标（当前 road_iou；标线指标缺真值时为 UNKNOWN）</span></th><th>逐 seed（champion / candidate）<br><span style=font-weight:400;font-size:11px;color:#6b7280>基线与候选在同一 seed 下的值，逐对比较</span></th>"
           "<th>均值 / ci95<br><span style=font-weight:400;font-size:11px;color:#6b7280>配对差均值 / 95% 置信半宽；区间跨 0 = 不可判定（不是没效果）</span></th><th>verdict<br><span style=font-weight:400;font-size:11px;color:#6b7280>candidate_better / champion_better / inconclusive</span></th><th>判定<br><span style=font-weight:400;font-size:11px;color:#6b7280>晋级结论：rejected / needs_evidence / shadow_candidate</span></th></tr>"]
    for d in decs:
        try:
            b = json.loads(d["path"].read_text(encoding="utf-8"))
        except Exception:                                 # noqa: BLE001
            continue
        pairs = b.get("pairings") or {}
        dec = (b.get("decision") or {}).get("decision") or ""
        if not pairs:
            continue
        for name, sp in sorted(pairs.items()):
            out.append(
                f'<tr><td class="mono">{_esc(d["run"])}</td>'
                f'<td class="mono">{_esc(str(b.get("candidate_id") or b.get("candidate") or ""))}</td>'
                f'<td>{_esc(name)}</td>'
                f'<td class="mono">{_esc(_fmt_list(sp.get("champion")))} / '
                f'{_esc(_fmt_list(sp.get("candidate")))}</td>'
                f'<td class="mono">{_esc(_fmt_num(sp.get("mean_delta")))} / '
                f'{_esc(_fmt_num(sp.get("ci95_halfwidth")))}</td>'
                f'<td>{_esc(str(sp.get("verdict") or ""))}</td>'
                f'<td><span class="pill {_esc(dec)}">{_esc(dec)}</span></td></tr>')
    out.append("</table>")
    return "".join(out)


def _all_datasets_table(ctx: dict) -> str:
    a = ctx.get("all") or {}
    items = (a.get("scan") or {}).get("datasets") or []
    if not items:
        return '<p class="hint">missing data: 没有数据版本文件。</p>'
    out = ["<table><tr><th>运行<br><span style=font-weight:400;font-size:11px;color:#6b7280>运行目录名：一次后台入口 / 一轮实验的产物都在这个目录里</span></th><th>dataset_id<br><span style=font-weight:400;font-size:11px;color:#6b7280>帧内容+标签+划分的内容哈希；数据动一点就变</span></th><th>train 组<br><span style=font-weight:400;font-size:11px;color:#6b7280>训练用的采集组（map/source_id），整组隔离</span></th>"
           "<th>dev 组<br><span style=font-weight:400;font-size:11px;color:#6b7280>开发集组；与训练组不得重叠</span></th><th>trainable<br><span style=font-weight:400;font-size:11px;color:#6b7280>有可用真值的帧数（路面或标线任一可用）</span></th><th>paint_ok<br><span style=font-weight:400;font-size:11px;color:#6b7280>漆线真值可用的帧数；0 = 标线类指标一律未测，不得当 0 分</span></th>"
           "<th>被拒帧<br><span style=font-weight:400;font-size:11px;color:#6b7280>被准入门隔离的帧数（身份缺失/内容重复/泄漏）</span></th></tr>"]
    for it in items:
        try:
            b = json.loads(it["path"].read_text(encoding="utf-8"))
        except Exception:                                 # noqa: BLE001
            continue
        # 两种文件形状：DatasetManifest（coverage 按 split 嵌套 + groups）
        # 与 `rounds` 的 rounds_dataset.json（coverage 就是 train 那段 + train_groups）
        cov_raw = b.get("coverage") or {}
        cov = cov_raw.get("train") if isinstance(cov_raw.get("train"), dict)             else cov_raw
        groups = b.get("groups") or {}
        train_g = ([g for g, v in groups.items() if v == "train"]
                   or list(b.get("train_groups") or []))
        dev_g = ([g for g, v in groups.items() if v == "dev"]
                 or list(b.get("dev_groups") or []))
        out.append(
            f'<tr><td class="mono">{_esc(it["run"])}</td>'
            f'<td class="mono">{_esc(str(b.get("dataset_id") or "")[:16])}</td>'
            f'<td class="mono">{_esc(", ".join(train_g)[:60] or "未测")}</td>'
            f'<td class="mono">{_esc(", ".join(dev_g)[:60] or "未测")}</td>'
            f'<td class="mono">{_esc(str(cov.get("trainable_frames", "未测")))}</td>'
            f'<td class="mono">{_esc(str(cov.get("paint_valid_frames", "未测")))}</td>'
            f'<td class="mono">{_esc(str(len(b.get("rejected") or [])))}</td></tr>')
    out.append("</table>")
    return "".join(out)


def _all_params_table(ctx: dict) -> str:
    a = ctx.get("all") or {}
    cks = (a.get("scan") or {}).get("ckpts") or []
    total = (a.get("scan") or {}).get("n_ckpt_total") or 0
    if not cks:
        return '<p class="hint">missing data: 没有 checkpoint。</p>'
    out = ["<table><tr><th>运行 / 臂-seed<br><span style=font-weight:400;font-size:11px;color:#6b7280>运行目录 / 实验臂 + 随机种子；同一 run 的每个臂每个 seed 一行</span></th><th>arch / 参数<br><span style=font-weight:400;font-size:11px;color:#6b7280>模型结构 / 可训练参数量</span></th><th>batch × epochs<br><span style=font-weight:400;font-size:11px;color:#6b7280>批大小 × 训练轮数</span></th>"
           "<th>lr → 调度器<br><span style=font-weight:400;font-size:11px;color:#6b7280>初值学习率 → 调度器(T_max)；T_max 跟着 --epochs 走，改 epochs 等于换计划</span></th><th>优化器<br><span style=font-weight:400;font-size:11px;color:#6b7280>优化器与其超参（Adam: betas / weight decay）</span></th><th>类别权重<br><span style=font-weight:400;font-size:11px;color:#6b7280>median-frequency 平衡后的 bg/road/line 权重；line 再乘 --line-weight</span></th>"
           "<th>数据因子<br><span style=font-weight:400;font-size:11px;color:#6b7280>这一轮改动的是哪一项数据/标签口径（line 屏蔽 / 截帧 / 采样权重…）</span></th><th>数据集 / commit<br><span style=font-weight:400;font-size:11px;color:#6b7280>数据版本（dataset_id）与代码提交</span></th></tr>"]
    for c in cks:
        try:
            rows = _ckpt_params(c["path"].parent, limit=1)
            r0 = rows[0] if rows else {}
        except Exception:                                 # noqa: BLE001
            r0 = {}
        if not r0 or r0.get("error"):
            continue
        ta = r0["train_args"]
        sch, opt = dict(ta.get("scheduler") or {}), dict(ta.get("optimizer") or {})
        factors = [x for x in (
            ("line 屏蔽" if ta.get("ignore_line_class") else ""),
            (f"截帧 {ta['max_train_frames']}" if ta.get("max_train_frames") else ""),
            (f"run 权重 {ta['run_weights']}" if ta.get("run_weights") else "")) if x]
        out.append(
            f'<tr><td class="mono">{_esc(c["run"])}/{_esc(_rel_tag(c["path"].parent, c["path"].parents[1]))}</td>'
            f'<td class="mono">{_esc(_val_or_missing(ta.get("arch")))} / '
            f'{_esc(_val_or_missing(ta.get("n_params")))}</td>'
            f'<td class="mono">{_esc(_val_or_missing(ta.get("batch")))} × '
            f'{_esc(_val_or_missing(ta.get("epochs")))}</td>'
            f'<td class="mono">{_esc(_val_or_missing(ta.get("lr")))} → '
            f'{_esc(str(sch.get("name") or "未测"))}'
            f'(T_max={_esc(_val_or_missing(sch.get("T_max")))})</td>'
            f'<td class="mono">{_esc(str(opt.get("name") or "未测"))} '
            f'betas={_esc(str(opt.get("betas") or "未测"))}</td>'
            f'<td class="mono">{_esc(str(ta.get("class_weights") or "未测"))}</td>'
            f'<td class="mono">{_esc("；".join(factors) or "无")}</td>'
            f'<td class="mono">{_esc(str(r0.get("dataset_id") or "未测"))} / '
            f'{_esc(str(r0.get("git_commit") or "未测"))}</td></tr>')
    out.append("</table>")
    if total > len(cks):
        out.append(f'<p class="hint">只读前 {len(cks)}/{total} 个 checkpoint'
                   "（读权重有开销）；要全部请按运行逐个 --run-dir 渲染。</p>")
    return "".join(out)


def _all_collect_table(ctx: dict) -> str:
    """无人值守采集：每次采集一行。**身份审计结果必须看得见**（拒收的采集不产出候选）。"""
    a = ctx.get("all") or {}
    cols = a.get("collects") or []
    if not cols:
        return ('<p class="hint">未测：没有采集记录（`collect_*.json` 只在后台入口'
                '真的启动过采集时写出）。</p>')
    span = "font-weight:400;font-size:11px;color:#6b7280"
    out = ["<table><tr>",
           "<th>运行<br><span style=font-weight:400;font-size:11px;color:#6b7280>运行目录名：一次后台入口 / 一轮实验的产物都在这个目录里</span></th>",
           "<th>时间戳<br><span style=" + span + ">采集开始时刻（本地）</span></th>",
           "<th>地图 / source_id<br><span style=" + span + ">身份读自运行中的会话，"
           "不是命令行参数</span></th>",
           "<th>帧数<br><span style=" + span + ">按视角分；总帧=各视角之和</span></th>",
           "<th>身份审计<br><span style=" + span + ">ok=能进训练；拒收的采集不产出候选"
           "</span></th>",
           "<th>漆线帧<br><span style=" + span + ">line_pixels&gt;0 的帧数：引擎不给 "
           "line 类，这些是「看得见漆线」的帧</span></th>",
           "<th>复核队列<br><span style=" + span + ">按漆线像素排序的待人工修订帧清单"
           "</span></th>",
           "<th>解释器<br><span style=" + span + ">采集用的解释器：没有 beamngpy 会白起一局"
           "</span></th>",
           "<th>GPU 分钟<br><span style=" + span + ">本次采集墙钟；游戏也是 GPU 负载，"
           "记进每日上限</span></th>",
           "<th>rc<br><span style=" + span + ">0=通过；6=前置检查拦下（没启动）；"
           "7=采集失败或审计拒收</span></th>",
           "</tr>"]
    for b in sorted(cols, key=lambda x: str(x.get("stamp") or ""), reverse=True):
        roles = b.get("roles") or {}
        roles_txt = "　".join(f"{k}:{v}" for k, v in sorted(roles.items())) \
            or "未测"
        paint = b.get("paint_frames_by_role") or {}
        if paint:
            paint_txt = "　".join(f"{k}:{v}"
                                 for k, v in sorted(paint.items()))
        elif int(b.get("frames_total") or 0) <= 0:
            paint_txt = "未测（一个帧都没有，不是 0）"
        else:
            paint_txt = "0（有帧但都没有漆线像素）"
        rq = b.get("review_queue")
        rq_txt = ("有" if (rq and Path(rq).exists()) else "未测")
        ident = (f"{b.get('map_name') or '未测'}<br>"
                 f"<span class=mono>{_esc(str(b.get('source_id') or '未测'))}</span>")
        if b.get("ok"):
            ok_txt = f'<span class="pill ok">ok</span>'
        else:
            why = (b.get("reasons") or ["未写原因"])[0]
            ok_txt = (f'<span class="pill bad">拒收</span><br>'
                      f'<span style="{span}">{_esc(str(why)[:90])}</span>')
        py_ = str(b.get("collector_python") or "未测")
        src = str(b.get("collector_python_source") or "")
        gpu = b.get("gpu_minutes")
        gpu_txt = (f"{float(gpu):.2f}" if gpu is not None else "未测")
        out.append(
            f'<tr><td class="mono">{_esc(str(b.get("_run") or ""))}</td>'
            f'<td class="mono">{_esc(str(b.get("stamp") or ""))}</td>'
            f'<td>{ident}</td><td class="mono">{_esc(roles_txt)}</td>'
            f'<td>{ok_txt}</td><td class="mono">{_esc(paint_txt)}</td>'
            f'<td>{rq_txt}</td>'
            f'<td class="mono">{_esc(Path(py_).name)}{_esc("（" + src + "）" if src else "")}</td>'
            f'<td class="mono">{gpu_txt}</td>'
            f'<td class="mono">{_esc(str(b.get("rc")))}</td></tr>')
    out.append("</table>")
    return "".join(out)


def _all_gpu_table(ctx: dict) -> str:
    a = ctx.get("all") or {}
    items = (a.get("scan") or {}).get("gpu") or []
    if not items:
        return '<p class="hint">未测：没有 GPU 账本（只有后台入口运行会写）。</p>'
    out = ["<table><tr><th>运行<br><span style=font-weight:400;font-size:11px;color:#6b7280>运行目录名：一次后台入口 / 一轮实验的产物都在这个目录里</span></th><th>日期<br><span style=font-weight:400;font-size:11px;color:#6b7280>本地日期</span></th><th>分钟<br><span style=font-weight:400;font-size:11px;color:#6b7280>该 run 当日累计 GPU 墙钟（跨进程累计，供每日上限）</span></th></tr>"]
    for it in items:
        try:
            b = json.loads(it["path"].read_text(encoding="utf-8"))
        except Exception:                                 # noqa: BLE001
            continue
        for day, slot in sorted(b.items()):
            out.append(f'<tr><td class="mono">{_esc(it["run"])}</td>'
                       f'<td class="mono">{_esc(day)}</td>'
                       f'<td class="mono">{_esc(str((slot or {{}}).get("minutes")))}</td></tr>')
    out.append("</table>")
    return "".join(out)


def _all_cards(ctx: dict) -> str:
    """一页全汇总：每 (运行, 指标) 一张多 seed 图 + 各总表。"""
    a = ctx.get("all") or {}
    scan = a.get("scan") or {}
    series = a.get("series") or []
    order = ("train_loss", "val_miou", "val_line_iou", "val_acc")
    # 按 (运行, 指标) 归并：同一场的所有 seed/臂画在一张图里
    grouped: dict = {}
    for se in series:
        if se.metric not in order:
            continue
        run = se.name.split("/")[0]
        grouped.setdefault((run, se.metric), []).append(se)
    cards = []
    for run in sorted({k[0] for k in grouped}):
        # 每场只画两条：train_loss + 该场**最好的那个**开发指标
        # （其余指标的数字在"指标总表"里，不必再各占一张图——用户反馈曲线太多）
        picks = ["train_loss"]
        for m in ("val_miou", "val_line_iou", "val_acc"):
            if grouped.get((run, m)):
                picks.append(m)
                break
        for metric in picks:
            sels = grouped.get((run, metric))
            if not sels:
                continue
            sels = sorted(sels, key=lambda x: x.name)
            n_pts = sum(1 for x in sels for _p, v in x.points if v is not None)
            if not n_pts:
                continue
            cards.append(_card(
                f"{run} · {metric}",
                f"单位：{metric}（{_CURVE_MEAN.get(metric, '')}）　"
                f"横轴：epoch　线数：{len(sels)}（同场不同 seed/臂画在一张图里）",
                _svg_multi(sels), stats=_series_stats(sels[0])))
    tables = [
        ("所有运行的指标总表", "每 运行/臂-seed 一行；lr/梯度/吞吐来自逐 step 指标",
         _all_metrics_table(ctx)),
        ("所有判定（成对比较）", "来自各运行的 decision_*.json", _all_decisions_table(ctx)),
        ("所有数据版本", "来自各运行的 rounds_dataset.json", _all_datasets_table(ctx)),
        ("所有训练超参", f"读最新 {len(scan.get('ckpts') or [])}/"
         f"{scan.get('n_ckpt_total') or 0} 个 checkpoint", _all_params_table(ctx)),
        ("无人值守采集", "每次采集一行：身份审计 + 复核队列 + GPU 分钟 + 解释器",
         _all_collect_table(ctx)),
        ("GPU 账本", "跨进程累计（含采集期间的游戏时间）", _all_gpu_table(ctx)),
    ]
    for title, sub, body in tables:
        cards.append(f'<div class="card wide"><h3>{_esc(title)}</h3>'
                     f'<div class="sub">{_esc(sub)}</div>'
                     f'<div>{body}</div></div>')
    return ("<main>" + "".join(cards) + "</main>")


#: 曲线指标与统计行的含义（写在副标题/统计行里）
_CURVE_MEAN = {"train_loss": "训练集损失，越低越好",
               "val_miou": "验证集 mIoU，越高越好",
               "val_line_iou": "验证集标线 IoU（缺可信真值时为未测）",
               "val_acc": "验证集像素准确率，越高越好"}


def _all_status_items(ctx: dict) -> str:
    a = ctx.get("all") or {}
    scan = a.get("scan") or {}
    cs = a.get("collect_stats") or {}
    items = [
        ('<span class="item"><span class="k">范围</span>'
         f'<span class="v">{_esc(str(scan.get("root") or ""))}</span></span>'),
        ('<span class="item"><span class="k">运行</span>'
         f'<span class="v">{len(scan.get("runs") or [])}</span></span>'),
        ('<span class="item"><span class="k">曲线</span>'
         f'<span class="v">{len(a.get("series") or [])}</span></span>'),
        ('<span class="item"><span class="k">采集</span>'
         f'<span class="v">{cs.get("ok", 0)}/{cs.get("n", 0)}</span></span>'
         f'<span class="item"><span class="k">采集帧</span>'
         f'<span class="v">{cs.get("frames", 0)}</span></span>'),
        ('<span class="item"><span class="k">判定</span>'
         f'<span class="v">{len(scan.get("decisions") or [])}</span></span>'),
        ('<span class="item"><span class="k">数据版本</span>'
         f'<span class="v">{len(scan.get("datasets") or [])}</span></span>'),
        ('<span class="item"><span class="k">checkpoint</span>'
         f'<span class="v">{scan.get("n_ckpt_total") or 0}</span></span>'),
        ('<span class="item"><span class="k">生成</span>'
         f'<span class="v">{_esc(ctx.get("generated_at", ""))}</span></span>'),
    ]
    return '<div class="statusbar">' + "".join(items) + "</div>"


def _compact_rows(ctx: dict) -> list:
    """每 (臂, seed) 一行：曲线首末值 + 训练动力学（按 run 名匹配）。"""
    series = list((ctx.get("t13") or {}).get("series") or [])
    by_key = {(se.name, se.metric): se for se in series}
    names = sorted({se.name for se in series})
    dyn = {d.get("run"): d for d in (ctx.get("dyn") or [])}
    root = _run_root(ctx)
    rows = []
    for name in names:

        def _ends(se):
            if se is None:
                return None, None
            got = [(x, v) for x, v in se.points if v is not None]
            return (got[0][1], got[-1][1]) if got else (None, None)

        loss = by_key.get((name, "train_loss"))
        val = next((by_key.get((name, m)) for m in
                    ("val_miou", "val_line_iou", "val_acc")
                    if by_key.get((name, m))), None)
        l0, l1 = _ends(loss)
        v0, v1 = _ends(val)
        d = (dyn.get(f"{root.name}-{name}") if root is not None else None) \
            or dyn.get(name) or {}
        rows.append({
            "name": name, "loss_series": loss, "val_series": val,
            "loss_first": l0, "loss_last": l1,
            "val_metric": (val.metric if val else ""),
            "val_first": v0, "val_last": v1,
            "lr_first": d.get("lr_first"), "lr_last": d.get("lr_last"),
            "grad_p50": d.get("grad_p50"), "grad_p95": d.get("grad_p95"),
            "steps_per_s": d.get("steps_per_s"), "n_steps": d.get("n_steps"),
            "total_steps": d.get("total_steps"),
        })
    return rows


def _paths_view(ctx: dict) -> str:
    """探针图与可打开的复核图：紧凑视图里只列路径，不铺大图。

    （大图留给 `--view full`；这里保证"存在什么、在哪"不被静默丢掉。）
    """
    st = ctx.get("probes") or {}
    rows = []
    if st.get("readable"):
        for rec in st.get("probes", [])[:12]:
            rows.append((rec.get("image") or rec.get("name") or "?",
                         "sidecar 齐全" if not rec.get("sidecar_error")
                         else f"无法验证：{rec['sidecar_error']}"))
    root = _run_root(ctx)
    extra = []
    if root is not None:
        extra = [str(p.relative_to(root)) for p in
                 sorted(root.glob("review_overlays/*.png"))[:6]]
    out = ['<section><h2>图像与可复核帧</h2>']
    if rows:
        out.append("<table><tr><th>探针图</th><th>sidecar</th></tr>"
                   + "".join(f'<tr><td class="mono">{_esc(a)}</td><td>{_esc(b)}</td></tr>'
                             for a, b in rows) + "</table>")
    else:
        out.append('<p class="hint">探针图：未测（没有 --probes 目录或目录为空）。'
                   '用 --view full 看图像本身。</p>')
    if extra:
        out.append('<p class="hint">可打开的复核图（相对运行目录）：'
                   + "、".join(f'<code>{_esc(x)}</code>' for x in extra) + "</p>")
    out.append("</section>")
    return "".join(out)


def _compact_view(ctx: dict) -> str:
    """一页看完：每 seed 一行（迷你曲线）+ 判定表 + 摘要 + 事件流。"""
    t13 = ctx.get("t13") or {}
    if not t13.get("readable"):
        return ('<section><h2>一页总览</h2><p class="hint">missing data: '
                + _esc(str(t13.get("error") or "没有训练历史"))
                + "。缺训练历史时不画任何数，只说明缺什么。</p></section>")
    rows = _compact_rows(ctx)
    out = ['<section><h2>一页总览</h2>']
    if rows:
        out.append("<table><tr><th>臂 / seed</th>"
                   "<th>train loss（迷你曲线 · 首→末）</th>"
                   "<th>开发指标（迷你曲线 · 首→末）</th>"
                   "<th>lr 首→末<br><span style=font-weight:400;font-size:11px;color:#6b7280>学习率；余弦退火按 --epochs 收到接近 0（T_max=epochs）</span></th><th>|grad| p50/p95<br><span style=font-weight:400;font-size:11px;color:#6b7280>梯度 L2 范数（AMP 下 unscale 后统计），衡量每步更新幅度</span></th><th>步/s<br><span style=font-weight:400;font-size:11px;color:#6b7280>每秒优化步数（吞吐），由逐 step 耗时中位数换算</span></th>"
                   "<th>步数<br><span style=font-weight:400;font-size:11px;color:#6b7280>本 run 实际记录的优化步数 / 计划步数</span></th></tr>")
        for r in rows:
            loss_spark = _sparkline(r["loss_series"].points
                                    if r["loss_series"] else None)
            val_spark = _sparkline(r["val_series"].points
                                   if r["val_series"] else None,
                                   colour="#2e9e6b")
            out.append(
                f'<tr><td class="mono">{_esc(r["name"])}</td>'
                f'<td>{loss_spark} <span class="mono">'
                f'{_esc(_fmt_point(r["loss_first"]) if r["loss_first"] is not None else "未测")}'
                f' → {_esc(_fmt_point(r["loss_last"]) if r["loss_last"] is not None else "未测")}'
                "</span></td>"
                f'<td>{val_spark} <span class="mono">{_esc(r["val_metric"])} '
                f'{_esc(_fmt_point(r["val_first"]) if r["val_first"] is not None else "未测")}'
                f' → {_esc(_fmt_point(r["val_last"]) if r["val_last"] is not None else "未测")}'
                "</span></td>"
                f'<td class="mono">{_esc(_val_or_missing(r["lr_first"]))} → '
                f'{_esc(_val_or_missing(r["lr_last"]))}</td>'
                f'<td class="mono">'
                f'{_esc(_fmt_point(r["grad_p50"]) if r["grad_p50"] is not None else "未测")}'
                f' / {_esc(_fmt_point(r["grad_p95"]) if r["grad_p95"] is not None else "未测")}</td>'
                f'<td class="mono">{_esc(_val_or_missing(r["steps_per_s"]))}</td>'
                f'<td class="mono">{_esc(_val_or_missing(r["n_steps"]))}'
                + (f'/{_esc(str(r["total_steps"]))}' if r.get("total_steps")
                   else "") + "</td></tr>")
        out.append("</table>")
        out.append('<p class="hint">迷你曲线：蓝=train loss（越低越好）、'
                   '绿=开发指标（越高越好），缺测处断线。lr/梯度/吞吐来自逐 step '
                   '指标文件，没有就是"未测"（不画 0）。</p>')
    out.append('<p class="hint">判定、超参、数据、资源与图像分别在下面各节；'
               '这一页按顺序往下滚就能看全（隐藏项在 --view full 里有大图与逐帧细节）。'
               '</p>')
    out.append("</section>")
    return "".join(out)


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
    table = ['<table><tr><th>指标<br><span style=font-weight:400;font-size:11px;color:#6b7280>成对比较用的主指标（当前 road_iou；标线指标缺真值时为 UNKNOWN）</span></th><th>分子/分母</th><th>champion 每 seed</th>'
             "<th>candidate 每 seed</th><th>mean delta</th><th>ci95 半宽</th>"
             "<th>verdict<br><span style=font-weight:400;font-size:11px;color:#6b7280>candidate_better / champion_better / inconclusive</span></th></tr>"]
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
    table = ['<table><tr><th>指标<br><span style=font-weight:400;font-size:11px;color:#6b7280>成对比较用的主指标（当前 road_iou；标线指标缺真值时为 UNKNOWN）</span></th><th>模型</th><th>每 seed 值（分子/分母）'
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


#: 档位 -> 人读名字（看板与报告共用同一套说法）
RANK_LABEL = {
    "verified": "人工/独立验证真值",
    "agent": "agent 逐帧核对（研究可用）",
    "pseudo": "弱监督（引擎部分标注）",
    "unreliable": "不可靠（引擎默认）",
    "absent": "无来源/未复核",
}


def _label_source_view(records: list, ctx: dict) -> str:
    """按**标签档位**分别计数（方案 §6.3 验收：四个计数分别可查）。

    方案原话：「640 生成帧、实际训练帧、有效评价帧、人工确认帧四个计数分别可查」。
    最容易犯的错是把"生成 640 帧"显示成"640 帧人工真值"——所以这里把
    **生成帧**与**人工确认帧**分开写，并把"没测到"写"未测"而不是 0。
    """
    by_rank: dict = {}
    by_view: dict = {}
    for r in records:
        q = ((r.get("quality") or {}).get("paint") or {})
        rank = str(q.get("rank") or "")
        key = rank or "unknown"
        by_rank[key] = by_rank.get(key, 0) + 1
        v = str(r.get("view") or "")
        if v:
            by_view[v] = by_view.get(v, 0) + 1
    n_all = len(records)
    n_verified = by_rank.get("verified", 0)
    ev = ctx.get("eval") or {}
    n_eval = 0
    eval_measured = False
    if ev.get("readable"):
        for split in (ev.get("splits") or {}).values():
            for entry in split.values():
                n = entry.get("n_frames")
                if n:
                    n_eval += int(n)
                    eval_measured = True
    out = ["<h3>标签档位与复核覆盖（生成 ≠ 人工真值）</h3>"]
    rows = ['<table><tr><th>计数</th><th>值</th><th>说明</th></tr>']
    rows.append(f'<tr><td>生成帧（清单里全部记录）</td>'
                f'<td class="num">{n_all}</td>'
                f'<td class="hint">采集/标注产出的帧，不代表任何真值等级</td></tr>')
    rows.append(f'<tr><td>人工确认帧（verified 档）</td>'
                f'<td class="num">{n_verified}</td>'
                f'<td class="hint">只有这一档能当晋级评价参考；'
                f'{n_all - n_verified} 帧不属于此档</td></tr>')
    rows.append('<tr><td>有效评价帧（评估矩阵实测）</td>'
                f'<td class="num">{"未测" if not eval_measured else n_eval}</td>'
                '<td class="hint">来自评估矩阵的 n_frames；'
                '没有评估矩阵就是未测</td></tr>')
    rows.append('</table>')
    out.append("".join(rows))
    if n_all and n_verified < n_all:
        out.append('<p class="hint">提示：把生成帧数当作人工真值数是常见的过度声明；'
                   "本页按档位分开计数。</p>")
    rows2 = ['<table><tr><th>档位</th><th>帧数</th></tr>']
    for rank in ("verified", "agent", "pseudo", "unreliable", "absent",
                 "unknown"):
        if rank in by_rank:
            rows2.append(f'<tr><td>{_esc(RANK_LABEL.get(rank, rank))}'
                         f'{_badge(LEVEL_TRAIN)}</td>'
                         f'<td class="num">{by_rank[rank]}</td></tr>')
    rows2.append("</table>")
    out.append("".join(rows2))
    if by_view:
        rows3 = ['<table><tr><th>相机</th><th>帧数</th></tr>']
        for v in sorted(by_view):
            rows3.append(f'<tr><td class="mono">{_esc(v)}</td>'
                         f'<td class="num">{by_view[v]}</td></tr>')
        rows3.append("</table>")
        out.append("<h4>相机覆盖</h4>" + "".join(rows3))
    return "".join(out)


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
    out.append(_label_source_view(records, ctx))
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


def _guard_state(run_dir) -> dict:
    """运行保护观测：入口每次启动写的 `resource_state.json`（方案 §8.2/W6）。

    **未接入就是未接入**：用户活动信号缺失时 ``user_active=None``、
    ``source="not connected"``，页面必须显示"未接入"，不能显示成"用户不在"。
    """
    if run_dir is None:
        return {"readable": False, "error": "没有 run 目录"}
    p = Path(run_dir) / "resource_state.json"
    if not p.is_file():
        return {"readable": False, "error": "还没有资源观测（入口未跑过）"}
    try:
        return {"readable": True, "state": json.loads(p.read_text(
            encoding="utf-8"))}
    except Exception as e:                                 # noqa: BLE001
        return {"readable": False, "error": f"读不了：{e}"}


def _guard_view(ctx: dict) -> str:
    """资源门 / 墙钟 / 用户活动信号——"未测不显示零"同样适用。"""
    g = ctx.get("guard") or {}
    out = ["<h3>运行保护（资源门 / 墙钟 / 用户活动）</h3>"]
    if not g.get("readable"):
        out.append("<p>" + _unknown(f"资源观测不可读：{g.get('error')}",
                                    level=LEVEL_TRAIN) + "</p>")
        return "".join(out)
    st = g["state"]
    src = str(st.get("user_activity_source") or "not connected")
    idle = st.get("user_idle_s")
    active = st.get("user_active")
    if src in ("", "not connected", "unknown") or active is None:
        act = _unknown("未接入：无法按用户活动暂停（不能声称已实现随用随停）",
                       level=LEVEL_TRAIN)
    else:
        act = Evidence(name="user_active", level=LEVEL_TRAIN,
                       value=1.0 if active else 0.0, unit="bool").cell()
    wall = st.get("wall_minutes")
    limit = st.get("max_wall_minutes")
    wall_txt = ("未测" if wall is None else f"{float(wall):.1f} min")
    limit_txt = ("不限制" if not limit or float(limit) <= 0
                 else f"{float(limit):.0f} min")
    rows = [
        ("用户活动信号", act),
        ("信号来源", _esc(src)),
        ("原始空闲秒数", ("未测" if idle is None else f"{float(idle):.1f} s")),
        ("本次连续运行", f"{wall_txt} / 上限 {limit_txt}"),
        ("资源门", ("通过" if st.get("allowed") else
                    "阻止：" + "；".join(st.get("reasons") or []))),
        ("观测时间", _esc(str(st.get("ts") or ""))),
    ]
    out.append("<table><tr><th>项</th><th>值</th></tr>")
    for k, v in rows:
        out.append(f"<tr><td>{_esc(k)}</td><td>{v}</td></tr>")
    out.append("</table>")
    for w in (st.get("warnings") or [])[:4]:
        out.append(f'<p class="hint">{_esc(w)}</p>')
    return "".join(out)


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
    out.append(_guard_view(ctx))
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
        "refresh_sec": float(getattr(args, "refresh_sec", 0) or 0),
        "compact": (getattr(args, "view", "compact") or "compact") == "compact",
        "dyn": _train_dynamics(Path(args.run_dir)
                              if getattr(args, "run_dir", None) else None),
        "all": (_all_ctx(Path(getattr(args, "runs_root", "logs/experiments")),
                         include_all=bool(getattr(args, "include_all", False)))
                if getattr(args, "all_runs", False) else None),
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
        "guard": _guard_state(run_dir),
        "timeline": _timeline_state(run_dir),
    }
    return ctx


def render_html(ctx: dict) -> str:
    legend = " ".join(f"{_badge(l)}" for l in LEVELS)
    all_mode = bool(ctx.get("all"))
    compact = bool(ctx.get("compact"))
    body_views = ([_all_cards(ctx)] if all_mode else
                  [_cards_view(ctx)] if compact else [
        _overview_view(ctx),
        _curves_view(ctx),
        _dl_params_view(ctx),
        _probes_view(ctx),
        _tasks_view(ctx),
        _compare_view(ctx),
        _decisions_view(ctx),
        _v5_view(ctx),
        _timeline_view(ctx),
        _final_view(ctx),
        _data_view(ctx),
        _resources_view(ctx),
    ])
    parts = [
        "<!DOCTYPE html>",
        '<html lang="zh-CN">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>M5/T14 只读学习看板 — {_esc(_run_label(ctx))}</title>",
        (f'<meta http-equiv="refresh" content="{int(ctx["refresh_sec"])}">'
         if int(ctx.get("refresh_sec") or 0) > 0 else ""),
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
        (_all_status_items(ctx) + _toolbar(ctx)) if all_mode else
        ((_status_items(ctx) + _toolbar(ctx)) if compact else _statusbar(ctx)),
        *body_views,
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
    render.add_argument("--all-runs", action="store_true",
                        help="把所有运行的数据汇总到一个网页（有界扫描）")
    render.add_argument("--runs-root", default="logs/experiments",
                        help="--all-runs 的扫描根（默认 logs/experiments）")
    render.add_argument("--include-all", action="store_true",
                        help="连纯历史训练目录也画（默认只画有判定/有指标的运行）")
    render.add_argument("--view", choices=("compact", "full"), default="compact",
                        help="compact=一页总览（默认）；full=原来 10 个分区的文档视图")
    render.add_argument("--refresh-sec", type=float, default=0.0,
                        help="给页面加自动刷新（秒）；默认 0=静态快照")
    _add_common(render)
    watch = sub.add_parser("watch", help="按 --every 秒反复渲染；Ctrl-C 退出")
    _add_common(watch)
    watch.add_argument("--all-runs", action="store_true",
                       help="把所有运行的数据汇总到一个网页（有界扫描）")
    watch.add_argument("--runs-root", default="logs/experiments")
    watch.add_argument("--include-all", action="store_true")
    watch.add_argument("--view", choices=("compact", "full"), default="compact",
                       help="compact=一页总览（默认）；full=10 分区文档视图")
    watch.add_argument("--refresh-sec", type=float, default=10.0,
                       help="页面自带自动刷新间隔（秒）；0=不刷新")
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