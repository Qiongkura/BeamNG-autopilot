"""中文版论文图：复用三个英文图脚本，出图前把图内文字换成中文。

不修改英文图脚本，也不动任何数据：只在 ``Figure.savefig`` 上挂钩子，
把该图里所有 Text 对象（标题/轴标签/刻度标签/图例/注释）按词典换成中文，
并把输出目录换成 ``logs/paper_figures_zh/``。

    .venv\\Scripts\\python.exe scripts\\m5_figures_i18n.py --collect   # 只打印图内文字（建词典用）
    .venv\\Scripts\\python.exe scripts\\m5_figures_i18n.py             # 出中文图
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.ticker import FixedFormatter  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

MODULES = ("m5_paper_figures", "m5_paper_figures_ext", "m5_paper_figures_extra")
ZH_DIR = ROOT / "logs" / "paper_figures_zh"
FONTS = ["Microsoft YaHei", "SimHei", "SimSun"]

from m5_figures_zh_dict import IDENT, PHRASES, TRANS, _IDENT_RE  # noqa: E402

_seen: set[str] = set()
_missing: set[str] = set()
_collect = False


def _is_cjk(s: str) -> bool:
    """汉字、CJK 标点与全角形式都算"已是中文"（避免二次翻译）。"""
    return any("一" <= c <= "鿿" or "　" <= c <= "〿"
               or "＀" <= c <= "￯" for c in s)


def tr(s: str) -> str:
    if not isinstance(s, str) or not s or _is_cjk(s):
        return s            # 已经是中文（同一张图被保存两次时幂等）
    if s in TRANS:
        return TRANS[s]
    out = s
    for a, b in PHRASES:
        out = out.replace(a, b)
    if _collect:
        if any(c.isalpha() for c in s):
            _seen.add(s)
    elif out == s and any(c.isalpha() for c in s) and s not in IDENT             and not _IDENT_RE.match(s):
        _missing.add(s)
    return out


def _walk(fig: Figure) -> None:
    def do_texts(texts):
        for t in texts:
            t.set_text(tr(t.get_text()))

    def fix_ticks(ax):
        # 分类刻度在绘制时会被重建：直接改 Text 会被覆盖，必须换成固定格式器
        for axis in (ax.xaxis, ax.yaxis):
            labs = [t.get_text() for t in axis.get_majorticklabels()]
            new_labs = [tr(lab) for lab in labs]
            if new_labs != labs:
                axis.set_major_formatter(FixedFormatter(new_labs))

    for ax in fig.get_axes():
        fix_ticks(ax)
        ax.set_title(tr(ax.get_title()))
        ax.set_xlabel(tr(ax.get_xlabel()))
        ax.set_ylabel(tr(ax.get_ylabel()))
        do_texts(ax.texts)
        do_texts(ax.get_xticklabels())
        do_texts(ax.get_yticklabels())
        leg = ax.get_legend()
        if leg is not None:
            do_texts(leg.get_texts())
            if leg.get_title() is not None:
                leg.get_title().set_text(tr(leg.get_title().get_text()))
    do_texts(fig.texts)
    if fig._suptitle is not None:
        fig._suptitle.set_text(tr(fig._suptitle.get_text()))


_orig_savefig = Figure.savefig


def _savefig(self, fname, *a, **kw):
    _walk(self)
    if _collect:
        return None
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    return _orig_savefig(self, ZH_DIR / Path(fname).name, *a, **kw)


def _run_modules() -> None:
    for name in MODULES:
        mod = importlib.import_module(name)
        if hasattr(mod, "OUT"):
            mod.OUT = ZH_DIR
        if hasattr(mod, "main"):
            mod.main()


def main() -> int:
    global _collect
    ap = argparse.ArgumentParser(description="中文版论文图（图内文字中文化）")
    ap.add_argument("--collect", action="store_true", help="只收集图内文字，不出图")
    a = ap.parse_args()
    _collect = a.collect

    from matplotlib import font_manager

    have = {f.name for f in font_manager.fontManager.ttflist}
    pick = next((f for f in FONTS if f in have), None)
    if pick is None:
        print("[zh-figs] 未找到中文字体，候选：",
              sorted(n for n in have if "YaHei" in n or "Sim" in n))
        return 2
    plt.rcParams.update({"font.sans-serif": [pick, *FONTS], "axes.unicode_minus": False,
                         "font.family": "sans-serif"})
    print(f"[zh-figs] 字体 {pick}")

    Figure.savefig = _savefig
    argv = sys.argv[:]              # 图脚本的 main() 也解析 argv，先摘掉本脚本的参数
    sys.argv = [argv[0]]
    try:
        _run_modules()
    finally:
        sys.argv = argv

    if _collect:
        print(f"\n=== 图内文字 {len(_seen)} 条 ===")
        for s in sorted(_seen):
            print(repr(s))
    else:
        n = len(list(ZH_DIR.glob("*.png")))
        print(f"[zh-figs] {n} 张 -> {ZH_DIR}")
        if _missing:
            print(f"[zh-figs] 未翻译 {len(_missing)} 条：")
            for s in sorted(_missing):
                print("   ", repr(s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
