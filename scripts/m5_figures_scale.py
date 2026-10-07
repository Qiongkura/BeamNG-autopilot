"""按比例放大图内字号（供跨栏组合图使用）。

为什么：组合图里每格显示宽度 ≈ 单栏宽（约 3.4 in），而图脚本的字号是按 6–8 in 的
figsize 定的（6.5–9 pt）——缩到 3.4 in 后实际只有 ~3.3 pt，读者必须放大才看得清
（审查 P2 #10）。这里把**所有** Text 的字号统一乘一个系数（不逐个改脚本），
输出到 `logs/paper_figures_big/`（英文）与 `logs/paper_figures_big_zh/`（中文），
再由 `m5_paper_composites.py` 拼图。

用法::

    .venv\\Scripts\\python.exe scripts\\m5_figures_scale.py --scale 1.6
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
from matplotlib.text import Text  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

MODULES = ("m5_paper_figures", "m5_paper_figures_ext", "m5_paper_figures_extra")
SCALE = 1.6
_orig_set_fontsize = Text.set_fontsize
_orig_savefig = Figure.savefig
OUT: Path | None = None
ZH = False


def _set_fontsize(self, size):
    try:
        return _orig_set_fontsize(self, float(size) * SCALE)
    except (TypeError, ValueError):
        return _orig_set_fontsize(self, size)


def _savefig(self, fname, *a, **kw):
    assert OUT is not None
    return _orig_savefig(self, OUT / Path(fname).name, *a, **kw)


def main() -> int:
    global SCALE, OUT, ZH
    ap = argparse.ArgumentParser(description="放大图内字号（组合图用）")
    ap.add_argument("--scale", type=float, default=1.6)
    ap.add_argument("--zh", action="store_true", help="同时出中文版（走 i18n 词典）")
    a = ap.parse_args()
    SCALE, ZH = a.scale, a.zh
    if ZH:
        import m5_figures_i18n as i18n
        from matplotlib import font_manager
        have = {f.name for f in font_manager.fontManager.ttflist}
        pick = next((f for f in i18n.FONTS if f in have), None)
        plt.rcParams.update({"font.sans-serif": [pick, *i18n.FONTS],
                             "axes.unicode_minus": False, "font.family": "sans-serif"})
        OUT = ROOT / "logs" / "paper_figures_big_zh"
        OUT.mkdir(parents=True, exist_ok=True)
        Text.set_fontsize = _set_fontsize
        Figure.savefig = _savefig

        def hook(self, fname, *a2, **kw):
            i18n._walk(self)
            return _savefig(self, fname, *a2, **kw)
        Figure.savefig = hook
    else:
        OUT = ROOT / "logs" / "paper_figures_big"
        OUT.mkdir(parents=True, exist_ok=True)
        Text.set_fontsize = _set_fontsize
        Figure.savefig = _savefig

    # rcParams 里的基准字号也要放大（脚本里 set_fontsize 之外的默认文本）
    plt.rcParams.update({k: v * SCALE for k, v in plt.rcParams.items()
                         if k.endswith("size") and isinstance(v, (int, float))})
    argv = sys.argv[:]
    sys.argv = [argv[0]]
    try:
        for name in MODULES:
            mod = importlib.import_module(name)
            if hasattr(mod, "OUT"):
                mod.OUT = OUT
            mod.main()
    finally:
        sys.argv = argv
    n = len(list(OUT.glob("*.png")))
    print(f"[scale] 字号 ×{SCALE}：{n} 张 -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
