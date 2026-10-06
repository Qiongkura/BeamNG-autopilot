"""图内文字排布自检：找出"字挤在一起"的图。

渲染每张图（英文图与中文图各一遍），取所有可见 Text 对象的包围盒，报出问题：

* **重叠**：两个文本框重叠面积超过较小框的 ``--min-frac``（默认 0.30），
  且较小框本身大于 ``--min-px``（默认 60 px²，滤掉刻度的小噪点）；
（不做"越界"检查：savefig 用 ``bbox_inches="tight"``，画布外的文字会被自动纳入
导出范围，按预存画布尺寸判断只会产生假阳性。）

用法：
    .venv\\Scripts\\python.exe scripts\\m5_figures_qa.py             # 英文图 + 中文图
    .venv\\Scripts\\python.exe scripts\\m5_figures_qa.py --lang zh   # 只查中文图
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402

_orig_savefig = Figure.savefig
_issues: list[tuple[str, str]] = []
MIN_FRAC = 0.30
MIN_PX = 60.0


def _texts(fig):
    out = []
    for ax in ([fig] if hasattr(fig, "get_xticklabels") and not hasattr(fig, "get_axes")
               else fig.get_axes()):
        out.append(ax.title)
        out.append(ax.xaxis.label)
        out.append(ax.yaxis.label)
        out.extend(ax.texts)
        out.extend(ax.get_xticklabels())
        out.extend(ax.get_yticklabels())
        leg = ax.get_legend()
        if leg is not None:
            out.extend(leg.get_texts())
            if leg.get_title() is not None:
                out.append(leg.get_title())
    if not hasattr(fig, "get_axes"):
        return out
    out.extend(fig.texts)
    if fig._suptitle is not None:
        out.append(fig._suptitle)
    return out


def _off_view(ax, t) -> bool:
    """刻度位置在坐标轴视窗之外的标签不会绘制，但 Text 对象仍存在。"""
    for axis in (ax.xaxis, ax.yaxis):
        ticks = list(axis.get_major_ticks()) + list(axis.get_minor_ticks())
        for tick in ticks:
            if tick.label1 is t or tick.label2 is t:
                try:
                    loc = tick.get_loc()
                except Exception:
                    return False
                lo, hi = sorted(axis.get_view_interval())
                return not (lo <= loc <= hi)
    return False


def _check(fig: Figure, name: str) -> None:
    try:
        fig.canvas.draw()
    except Exception as exc:                      # 画不出来也要报，不静默
        _issues.append((name, f"draw failed: {type(exc).__name__}: {exc}"))
        return
    r = fig.canvas.get_renderer()
    boxes = []
    seen, keys = set(), set()
    for t in _texts(fig):
        if id(t) in seen:
            continue
        seen.add(id(t))
        s = (t.get_text() or "").strip()
        if not s or not t.get_visible():
            continue
        try:
            bb = t.get_window_extent(renderer=r)
        except Exception:
            continue
        if bb.width <= 0 or bb.height <= 0:
            continue
        fw, fh = fig.canvas.get_width_height()
        if not (bb.x0 < fw and bb.x1 > 0 and bb.y0 < fh and bb.y1 > 0):
            continue                       # 画布外：tight bbox 会纳入，但不可比
        owner = next((ax for ax in fig.get_axes() if t in _texts(ax)), None)
        if owner is not None and _off_view(owner, t):
            continue                       # 视窗外刻度：不绘制
        key = (s, round((bb.x0 + bb.x1) / 8), round((bb.y0 + bb.y1) / 8))
        # 同串且中心相差 <4 px 的算一条（叠在同一处的重复标签在图上只看得见一个）
        if key in keys:
            continue
        keys.add(key)
        boxes.append((s, bb))
    for i in range(len(boxes)):
        s1, b1 = boxes[i]
        for j in range(i + 1, len(boxes)):
            s2, b2 = boxes[j]
            ow = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
            oh = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
            if ow <= 0 or oh <= 0:
                continue
            small = min(b1.width * b1.height, b2.width * b2.height)
            if small < MIN_PX:
                continue
            frac = (ow * oh) / small
            if frac >= MIN_FRAC:
                _issues.append((name, f"重叠 {frac:.0%}: {s1[:32]!r} × {s2[:32]!r}"))


def _run(lang: str, tmp: Path) -> None:
    import m5_paper_figures
    import m5_paper_figures_ext
    import m5_paper_figures_extra

    mods = [m5_paper_figures, m5_paper_figures_ext, m5_paper_figures_extra]

    if lang == "zh":
        import m5_figures_i18n as i18n
        import matplotlib.pyplot as plt
        from matplotlib import font_manager

        have = {f.name for f in font_manager.fontManager.ttflist}
        pick = next((f for f in i18n.FONTS if f in have), None)
        plt.rcParams.update({"font.sans-serif": [pick, *i18n.FONTS],
                             "axes.unicode_minus": False, "font.family": "sans-serif"})

        def hook(self, fname, *a, **kw):
            i18n._walk(self)
            _check(self, f"{lang}:{Path(fname).stem}")
            return _orig_savefig(self, fname, *a, **kw)
    else:
        def hook(self, fname, *a, **kw):
            _check(self, f"{lang}:{Path(fname).stem}")
            return _orig_savefig(self, fname, *a, **kw)

    Figure.savefig = hook
    argv = sys.argv[:]                            # 图脚本 main() 也解析 argv
    sys.argv = [argv[0]]
    try:
        for m in mods:
            if hasattr(m, "OUT"):
                m.OUT = tmp
            m.main()
    finally:
        sys.argv = argv


def main() -> int:
    global MIN_FRAC, MIN_PX
    ap = argparse.ArgumentParser(description="图内文字排布自检")
    ap.add_argument("--lang", default="both", choices=["en", "zh", "both"])
    ap.add_argument("--min-frac", type=float, default=0.30)
    ap.add_argument("--min-px", type=float, default=60.0)
    a = ap.parse_args()
    MIN_FRAC, MIN_PX = a.min_frac, a.min_px
    tmp = ROOT / "logs" / "paper_figures_qa_tmp"
    for lang in (["en", "zh"] if a.lang == "both" else [a.lang]):
        _run(lang, tmp)
    if tmp.exists():
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n=== 图内文字自检：{len(_issues)} 条 ===")
    for name, msg in _issues:
        print(f"  [{name}] {msg}")
    return 1 if _issues else 0


if __name__ == "__main__":
    sys.exit(main())
