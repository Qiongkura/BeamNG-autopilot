"""构建投稿源包：docs/paper/submission/（四份 .tex + 两套图 + README）。

不做的事：不编译 LaTeX（本机无引擎）、不改论文内容。
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"I:\projects\beamng-autopilot")
PAPER = ROOT / "docs" / "paper"
SUB = PAPER / "submission"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
BUILD = ROOT / "logs" / "paper_build"


def run(cmd: list[str]) -> None:
    r = subprocess.run([str(c) for c in cmd], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"命令失败 {cmd}: {r.stdout}\n{r.stderr}")


def main() -> int:
    SUB.mkdir(parents=True, exist_ok=True)
    (SUB / "figures").mkdir(exist_ok=True)
    (SUB / "figures_zh").mkdir(exist_ok=True)

    # 1) 四份 .tex：中英两套图分目录（英文 figures/，中文 figures_zh/）
    run([PY, "docs/paper/build_paper.py", "--md", "docs/paper/paper.md",
         "--tex", str(SUB / "main.tex"), "--html", str(BUILD / "submission_en.html"),
         "--figs", str(SUB / "figures"), "--img-dir", "figures",
         "--figs-src", str(ROOT / "logs" / "paper_figures")])
    run([PY, "docs/paper/build_paper.py", "--md", "docs/paper/paper_zh.md",
         "--tex", str(SUB / "main_zh.tex"), "--html", str(BUILD / "submission_zh.html"),
         "--lang", "zh", "--figs", str(SUB / "figures_zh"), "--img-dir", "figures_zh",
         "--figs-src", str(ROOT / "logs" / "paper_figures_zh")])
    run([PY, "docs/paper/build_paper.py", "--md", "docs/paper/supplement.md",
         "--tex", str(SUB / "supplement.tex"), "--html", str(BUILD / "submission_sup.html"),
         "--fig-prefix", "S", "--figs", str(SUB / "figures"), "--img-dir", "figures",
         "--figs-src", str(ROOT / "logs" / "paper_figures")])
    run([PY, "docs/paper/build_paper.py", "--md", "docs/paper/supplement_zh.md",
         "--tex", str(SUB / "supplement_zh.tex"), "--html", str(BUILD / "submission_sup_zh.html"),
         "--lang", "zh", "--fig-prefix", "S", "--figs", str(SUB / "figures_zh"),
         "--img-dir", "figures_zh", "--figs-src", str(ROOT / "logs" / "paper_figures_zh")])

    # 2) 参考文献源（供期刊用 bibtex 时参考；.tex 自带 thebibliography，不需要它）
    shutil.copy2(PAPER / "references.bib", SUB / "references.bib")
    shutil.copy2(PAPER / "CLAIMS_20261007.md", SUB / "CLAIMS_20261007.md")

    # 3) README
    (SUB / "README.md").write_text(
        "# 投稿源包（T16）\n\n"
        "本目录由 `scripts/m5_build_submission.py` 生成，内容与 `docs/paper/` 下的成品 PDF 同源。\n\n"
        "| 文件 | 说明 |\n|---|---|\n"
        "| `main.tex` | 英文正文（IEEEtran journal 类；自带 `thebibliography`，**单次 pdflatex 即可**，无需 bibtex） |\n"
        "| `main_zh.tex` | 中文正文（同一类 + `ctex`；中文字体用 `fontset=windows`） |\n"
        "| `supplement.tex` / `supplement_zh.tex` | 补充材料（图号 S1–S16） |\n"
        "| `figures/` | 英文图（26 张，300 dpi） |\n"
        "| `figures_zh/` | 中文图（同数据、图内文字中文） |\n"
        "| `references.bib` | 题录源（.tex 已内嵌参考文献，仅在改用 bibtex 时需要） |\n"
        "| `CLAIMS_20261007.md` | 主张→数值→来源→配置的冻结清单 |\n\n"
        "## 编译\n\n"
        "```bash\n"
        "pdflatex main.tex          # 英文正文；两次可稳定交叉引用（本包未用 \\ref 计数）\n"
        "xelatex main_zh.tex        # 中文正文（ctex 建议 xelatex）\n"
        "pdflatex supplement.tex\n"
        "xelatex supplement_zh.tex\n"
        "```\n\n"
        "图路径为相对路径（`figures/xxx.png`），因此请在**本目录内**编译，或把整个目录拷到别处。\n\n"
        "## 已验证 / 未验证\n\n"
        "- **已验证（静态检查）**：环境配平、`\\includegraphics` 指向的文件存在、`\\cite` 与 `\\bibitem` 闭环、\n"
        "  无未替换的模板残留（`{@fig:`、`[@`）、花括号配平、摘要环境唯一。检查器：`scripts/m5_paper_texcheck.py`。\n"
        "- **未验证**：本机**没有安装 LaTeX 引擎**（无 pdflatex/xelatex/tectonic），因此**未编译**，\n"
        "  不声称编译通过。交付 PDF（`docs/paper/T16_*.pdf`）由同一 Markdown 源经 HTML→Edge 打印生成，\n"
        "  不是 LaTeX 产物；两者内容同源但排版路径不同。\n",
        encoding="utf-8")

    print("[submission] 包已生成:", SUB)
    return 0


if __name__ == "__main__":
    sys.exit(main())
